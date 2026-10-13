#include "ngram-cache.h"

#include "common.h"
#include "log.h"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <thread>

void fCommonNgramCacheUpdate(common_ngram_cache &     pNgramCache,
                             int                      pNgramMin,
                             int                      pNgramMax,
                             std::vector<dgs_token> & pInp,
                             int                      pNnew,
                             bool                     pPrintProgress) {
  const int64_t cTStartMs = ggml_time_ms();
  const int64_t cInpSize  = pInp.size();

  const int64_t cNTodo = cInpSize * (pNgramMax - pNgramMin + 1);
  int64_t       vNDone = 0;

  for (int64_t vNgramSize = pNgramMin; vNgramSize <= pNgramMax; ++vNgramSize) {
    const int64_t cIStart = std::max(cInpSize - pNnew, vNgramSize);
    for (int64_t vI = cIStart; vI < cInpSize; ++vI) {
      const int64_t   cNgramStart = vI - vNgramSize;
      common_ngram    vNgram(&pInp[cNgramStart], vNgramSize);
      const dgs_token token = pInp[vI];

      common_ngram_cache::iterator dPartIt = pNgramCache.find(vNgram);
      if (dPartIt == pNgramCache.end()) {
        common_ngram_cache_part dPart;
        dPart.emplace(token, 1);
        pNgramCache.emplace(vNgram, dPart);
      } else {
        common_ngram_cache_part::iterator vTokenCountIt = dPartIt->second.find(token);
        if (vTokenCountIt == dPartIt->second.end()) {
          dPartIt->second.emplace(token, 1);
        } else {
          vTokenCountIt->second++;
        }
      }
      ++vNDone;

      if (pPrintProgress && vNDone % 10000000 == 0) {
        const int64_t cTNowMs = ggml_time_ms();
        const int64_t cEtaMs  = (cInpSize * (pNgramMax - pNgramMin + 1) - vNDone) * (cTNowMs - cTStartMs) / vNDone;
        const int64_t cEtaMin = cEtaMs / (60 * 1000);
        const int64_t cEtaS   = (cEtaMs - 60 * 1000 * cEtaMin) / 1000;

        fprintf(stderr, "%s: %" PRId64 "/%" PRId64 " done, ETA: %02" PRId64 ":%02" PRId64 "\n", __func__, vNDone,
                cNTodo, cEtaMin, cEtaS);
      }
    }
  }
}

// Helper function to get a token from the combined, speculative sequence of inp and draft.
static dgs_token fGetToken(const std::vector<dgs_token> & pInp,
                           const std::vector<dgs_token> & pDraft,
                           const size_t                   pI) {
  return pI < pInp.size() ? pInp[pI] : pDraft[1 + pI - pInp.size()];
}

// If sample size or percentage are below these thresholds the draft is aborted early:
constexpr int aDraftMinSampleSizeLax[DGS_NGRAM_MAX]    = { 2, 2, 1, 1 };
constexpr int aDraftMinPercentLax[DGS_NGRAM_MAX]       = { 66, 50, 50, 50 };
constexpr int aDraftMinSampleSizeStrict[DGS_NGRAM_MAX] = { 4, 3, 2, 2 };
constexpr int aDraftMinPercentStrict[DGS_NGRAM_MAX]    = { 75, 66, 66, 66 };

// Helper function that tries to draft a token from only the static ngram cache:
static dgs_token fTryDraft(common_ngram_cache & pNcStatic, const common_ngram pNgramStatic) {
  common_ngram_cache::iterator dPartStaticIt = pNcStatic.find(pNgramStatic);
  if (dPartStaticIt == pNcStatic.end()) {
    return DGS_TOKEN_NULL;
  }
  const common_ngram_cache_part dPartStatic = dPartStaticIt->second;

  int       vMaxCountStatic = 0;
  int       vSumCountStatic = 0;
  dgs_token vMaxToken       = DGS_TOKEN_NULL;

  for (std::pair<dgs_token, int> vTokenCountStatic : dPartStatic) {
    const dgs_token token        = vTokenCountStatic.first;
    const int32_t   cCountStatic = vTokenCountStatic.second;

    if (cCountStatic > vMaxCountStatic) {
      vMaxToken       = token;
      vMaxCountStatic = cCountStatic;
    }
    vSumCountStatic += cCountStatic;
  }

  if (vSumCountStatic < aDraftMinSampleSizeLax[DGS_NGRAM_STATIC - 1]) {
    return DGS_TOKEN_NULL;
  }
  if (100 * vMaxCountStatic < aDraftMinPercentLax[DGS_NGRAM_STATIC - 1] * vSumCountStatic) {
    return DGS_TOKEN_NULL;
  }
  return vMaxToken;
}

// Try to draft a token from primary cache (context/dynamic), validate with static cache:
static dgs_token fTryDraft(common_ngram_cache &              pNcPrimary,
                           const std::vector<common_ngram> & pNgramsPrimary,
                           common_ngram_cache_part &         pPartStatic,
                           const int *                       pMinSampleSize,
                           const int *                       pMinPercent) {
  dgs_token vDraftedToken = DGS_TOKEN_NULL;

  for (int vI = pNgramsPrimary.size() - 1; vI >= 0 && vDraftedToken == DGS_TOKEN_NULL; --vI) {
    const common_ngram cNgramPrimary = pNgramsPrimary[vI];

    common_ngram_cache::iterator dPartPrimaryIt = pNcPrimary.find(cNgramPrimary);
    if (dPartPrimaryIt == pNcPrimary.end()) {
      continue;
    }
    const common_ngram_cache_part dPartPrimary = dPartPrimaryIt->second;

    int       vMaxCountPrimary = 0;
    int       vMaxCountStatic  = 0;
    int       vSumCountPrimary = 0;
    dgs_token vMaxToken        = DGS_TOKEN_NULL;

    for (std::pair<dgs_token, int> vTokenCountPrimary : dPartPrimary) {
      const dgs_token token = vTokenCountPrimary.first;

      common_ngram_cache_part::iterator vTokenCountStaticIt = pPartStatic.find(token);

      const int32_t cCountPrimary = vTokenCountPrimary.second;
      const int32_t cCountStatic  = vTokenCountStaticIt != pPartStatic.end() ? 100 * vTokenCountStaticIt->second : 1;

      if (cCountPrimary * cCountStatic > vMaxCountPrimary * vMaxCountStatic) {
        vMaxToken        = token;
        vMaxCountPrimary = cCountPrimary;
        vMaxCountStatic  = cCountStatic;
      }
      vSumCountPrimary += cCountPrimary;
    }

    if (vSumCountPrimary < pMinSampleSize[vI]) {
      continue;
    }
    if (100 * vMaxCountPrimary < pMinPercent[vI] * vSumCountPrimary) {
      continue;
      ;
    }
    vDraftedToken = vMaxToken;
  }

  return vDraftedToken;
}

void fCommonNgramCacheDraft(std::vector<dgs_token> & pInp,
                            std::vector<dgs_token> & pDraft,
                            int                      pNDraft,
                            int                      pNgramMin,
                            int                      pNgramMax,
                            common_ngram_cache &     pNcContext,
                            common_ngram_cache &     pNcDynamic,
                            common_ngram_cache &     pNcStatic) {
  GGML_ASSERT(pDraft.size() == 1);
  const int cInpSize = pInp.size();

  if (cInpSize < DGS_NGRAM_STATIC) {
    return;
  }

  while ((int) pDraft.size() - 1 < pNDraft) {
    dgs_token vDraftedToken = DGS_TOKEN_NULL;

    const int    cNgramStartStatic = cInpSize - DGS_NGRAM_STATIC + pDraft.size() - 1;
    common_ngram vNgramStatic;
    for (int vJ = cNgramStartStatic; vJ < cNgramStartStatic + DGS_NGRAM_STATIC; ++vJ) {
      vNgramStatic.aTokens[vJ - cNgramStartStatic] = fGetToken(pInp, pDraft, vJ);
    }
    common_ngram_cache::iterator dPartStaticIt = pNcStatic.find(vNgramStatic);
    common_ngram_cache_part      dPartStatic;
    if (dPartStaticIt != pNcStatic.end()) {
      dPartStatic = dPartStaticIt->second;
    }

    // cd = context + dynamic
    std::vector<common_ngram> lNgramsCd;
    for (int vNgramSizeCd = pNgramMin; vNgramSizeCd <= pNgramMax; ++vNgramSizeCd) {
      const int    cNgramStartCd = cInpSize - vNgramSizeCd + pDraft.size() - 1;
      common_ngram vNgramCd;
      for (int vJ = cNgramStartCd; vJ < cNgramStartCd + vNgramSizeCd; ++vJ) {
        vNgramCd.aTokens[vJ - cNgramStartCd] = fGetToken(pInp, pDraft, vJ);
      }
      lNgramsCd.push_back(vNgramCd);
    }
    if (vDraftedToken == DGS_TOKEN_NULL) {
      vDraftedToken = fTryDraft(pNcContext, lNgramsCd, dPartStatic, aDraftMinSampleSizeLax, aDraftMinPercentLax);
    }
    if (vDraftedToken == DGS_TOKEN_NULL) {
      vDraftedToken = fTryDraft(pNcDynamic, lNgramsCd, dPartStatic, aDraftMinSampleSizeStrict, aDraftMinPercentStrict);
    }
    if (vDraftedToken == DGS_TOKEN_NULL) {
      vDraftedToken = fTryDraft(pNcStatic, vNgramStatic);
    }

    if (vDraftedToken == DGS_TOKEN_NULL) {
      break;
    }

    LOG_DBG(" - draft candidate: token=%d\n", vDraftedToken);
    pDraft.push_back(vDraftedToken);
  }
}

void fCommonNgramCacheSave(common_ngram_cache & pNgramCache, const std::string & pFilename) {
  std::ofstream vFileOut(pFilename, std::ios::binary);
  for (std::pair<common_ngram, common_ngram_cache_part> dItem : pNgramCache) {
    const common_ngram      cNgram       = dItem.first;
    common_ngram_cache_part dTokenCounts = dItem.second;
    GGML_ASSERT(!dTokenCounts.empty());
    const int32_t cNtokens = dTokenCounts.size();
    GGML_ASSERT(cNtokens > 0);

    vFileOut.write(reinterpret_cast<const char *>(&cNgram), sizeof(common_ngram));
    vFileOut.write(reinterpret_cast<const char *>(&cNtokens), sizeof(int32_t));
    for (std::pair<dgs_token, int32_t> vItem2 : dTokenCounts) {
      const dgs_token token = vItem2.first;
      const int32_t   count = vItem2.second;
      GGML_ASSERT(count > 0);

      vFileOut.write(reinterpret_cast<const char *>(&token), sizeof(dgs_token));
      vFileOut.write(reinterpret_cast<const char *>(&count), sizeof(int32_t));
    }
  }
}

common_ngram_cache fCommonNgramCacheLoad(const std::string & pFilename) {
  std::ifstream vHashmapFile(pFilename, std::ios::binary);
  if (!vHashmapFile) {
    throw std::ifstream::failure("Unable to open file " + pFilename);
  }
  common_ngram_cache dNgramCache;

  common_ngram vNgram;
  int32_t      vNtokens;
  dgs_token    token;
  int32_t      count;

  char * vNgramc   = reinterpret_cast<char *>(&vNgram);
  char * vNtokensc = reinterpret_cast<char *>(&vNtokens);
  char * vTokenc   = reinterpret_cast<char *>(&token);
  char * vCountc   = reinterpret_cast<char *>(&count);
  while (vHashmapFile.read(vNgramc, sizeof(common_ngram))) {
    GGML_ASSERT(!vHashmapFile.eof());
    GGML_ASSERT(vHashmapFile.read(vNtokensc, sizeof(int32_t)));
    GGML_ASSERT(vNtokens > 0);
    common_ngram_cache_part dTokenCounts;

    for (int vI = 0; vI < vNtokens; ++vI) {
      GGML_ASSERT(!vHashmapFile.eof());
      GGML_ASSERT(vHashmapFile.read(vTokenc, sizeof(dgs_token)));
      GGML_ASSERT(!vHashmapFile.eof());
      GGML_ASSERT(vHashmapFile.read(vCountc, sizeof(int32_t)));
      GGML_ASSERT(count > 0);
      dTokenCounts.emplace(token, count);
    }

    dNgramCache.emplace(vNgram, dTokenCounts);
  }
  GGML_ASSERT(vHashmapFile.eof());

  return dNgramCache;
}

void fCommonNgramCacheMerge(common_ngram_cache & pNgramCacheTarget, common_ngram_cache & pNgramCacheAdd) {
  for (std::pair<common_ngram, common_ngram_cache_part> dNgramPart : pNgramCacheAdd) {
    const common_ngram      cNgram = dNgramPart.first;
    common_ngram_cache_part dPart  = dNgramPart.second;

    common_ngram_cache::iterator dPartMergedIt = pNgramCacheTarget.find(cNgram);
    if (dPartMergedIt == pNgramCacheTarget.end()) {
      pNgramCacheTarget.emplace(cNgram, dPart);
      continue;
    }

    for (std::pair<dgs_token, int32_t> vTokenCount : dPart) {
      const dgs_token token = vTokenCount.first;
      const int32_t   count = vTokenCount.second;
      GGML_ASSERT(count > 0);

      common_ngram_cache_part::iterator vTokenCountMergedIt = dPartMergedIt->second.find(token);
      if (vTokenCountMergedIt == dPartMergedIt->second.end()) {
        dPartMergedIt->second.emplace(token, count);
        continue;
      }

      vTokenCountMergedIt->second += count;
    }
  }
}
