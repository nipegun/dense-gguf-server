#include "sampling.h"

#include "common.h"
#include "fit.h"
#include "ggml.h"
#include "log.h"
#include "reasoning-budget.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstring>
#include <random>
#include <unordered_map>
#include <vector>

// the ring buffer works similarly to std::deque, but with a fixed capacity
// TODO: deduplicate with dgs-impl.h
template <typename T> struct ring_buffer {
  ring_buffer(size_t pCap) : vCapacity(pCap), data(pCap) {}

  T & front() {
    if (vSz == 0) {
      throw std::runtime_error("ring buffer is empty");
    }
    return data[vFirst];
  }

  const T & front() const {
    if (vSz == 0) {
      throw std::runtime_error("ring buffer is empty");
    }
    return data[vFirst];
  }

  T & back() {
    if (vSz == 0) {
      throw std::runtime_error("ring buffer is empty");
    }
    return data[vPos];
  }

  const T & back() const {
    if (vSz == 0) {
      throw std::runtime_error("ring buffer is empty");
    }
    return data[vPos];
  }

  void push_back(const T & value) {
    if (vSz == vCapacity) {
      // advance the start when buffer is full
      vFirst = (vFirst + 1) % vCapacity;
    } else {
      vSz++;
    }
    data[vPos] = value;
    vPos       = (vPos + 1) % vCapacity;
  }

  T mPopFront() {
    if (vSz == 0) {
      throw std::runtime_error("ring buffer is empty");
    }
    T value = data[vFirst];
    vFirst  = (vFirst + 1) % vCapacity;
    vSz--;
    return value;
  }

  const T & mRat(size_t pI) const {
    if (pI >= vSz) {
      throw std::runtime_error("ring buffer: index out of bounds");
    }
    return data[(vFirst + vSz - pI - 1) % vCapacity];
  }

  std::vector<T> mToVector() const {
    std::vector<T> lResult;
    lResult.reserve(vSz);
    for (size_t vI = 0; vI < vSz; vI++) {
      lResult.push_back(data[(vFirst + vI) % vCapacity]);
    }
    return lResult;
  }

  void clear() {
    // here only reset the status of the buffer
    vSz    = 0;
    vFirst = 0;
    vPos   = 0;
  }

  bool empty() const { return vSz == 0; }

  size_t size() const { return vSz; }

  size_t         vCapacity = 0;
  size_t         vSz       = 0;
  size_t         vFirst    = 0;
  size_t         vPos      = 0;
  std::vector<T> data;
};

struct common_sampler {
  common_params_sampling vParams;

  struct dgs_sampler * vGrmr;
  struct dgs_sampler * vRbudget;
  struct dgs_sampler * vChain;

  ring_buffer<dgs_token> vPrev;

  std::vector<dgs_token_data> lCur;

  dgs_token_data_array vCurP;

  // for rejection sampling; independent of the draft, or the target distribution is not preserved
  std::mt19937 vRng;

  void reset() {
    vPrev.clear();

    fDgsSamplerReset(vChain);
  }

  void mSetLogits(struct dgs_context * pCtx, int pIdx) {
    const float *     vSampledProbs  = fDgsGetSampledProbsIth(pCtx, pIdx);
    const float *     vSampledLogits = fDgsGetSampledLogitsIth(pCtx, pIdx);
    const dgs_token * vSampledIds    = fDgsGetSampledCandidatesIth(pCtx, pIdx);

    const dgs_model * vModel = fDgsGetModel(pCtx);
    const dgs_vocab * vVocab = fDgsModelGetVocab(vModel);

    const int cNVocab = fDgsVocabNTokens(vVocab);

    if (vSampledProbs) {
      const uint32_t cSampledProbsCount = fDgsGetSampledProbsCountIth(pCtx, pIdx);
      lCur.resize(cSampledProbsCount);
      for (uint32_t vI = 0; vI < cSampledProbsCount; ++vI) {
        lCur[vI] = dgs_token_data{ vSampledIds[vI], vSampledLogits[vI], vSampledProbs[vI] };
      }
    } else if (vSampledLogits) {
      const uint32_t cSampledLogitsCount = fDgsGetSampledLogitsCountIth(pCtx, pIdx);
      lCur.resize(cSampledLogitsCount);
      for (uint32_t vI = 0; vI < cSampledLogitsCount; vI++) {
        lCur[vI] = dgs_token_data{ vSampledIds[vI], vSampledLogits[vI], 0.0f };
      }
    } else {
      const auto * vLogits = fDgsGetLogitsIth(pCtx, pIdx);
      GGML_ASSERT(vLogits != nullptr);
      lCur.resize(cNVocab);
      for (dgs_token vTokenId = 0; vTokenId < cNVocab; vTokenId++) {
        lCur[vTokenId] = dgs_token_data{ vTokenId, vLogits[vTokenId], 0.0f };
      }
    }

    vCurP = { lCur.data(), lCur.size(), -1, false };
  }

  common_time_meas mTm() { return common_time_meas(vTTotalUs, vParams.vNoPerf); }

  mutable int64_t vTTotalUs = 0;
};

std::string common_params_sampling::mPrint() const {
  char aResult[1024];

  snprintf(aResult, sizeof(aResult),
           "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
           "\tdry_multiplier = %.3f, dry_base = %.3f, dry_allowed_length = %d, dry_penalty_last_n = %d\n"
           "\ttop_k = %d, top_p = %.3f, min_p = %.3f, xtc_probability = %.3f, xtc_threshold = %.3f, typical_p = %.3f, "
           "top_n_sigma = %.3f, temp = %.3f\n"
           "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f, adaptive_target = %.3f, adaptive_decay = %.3f",
           vPenaltyLastN, vPenaltyRepeat, vPenaltyFreq, vPenaltyPresent, vDryMultiplier, vDryBase, vDryAllowedLength,
           vDryPenaltyLastN, vTopK, vTopP, vMinP, vXtcProbability, vXtcThreshold, vTypP, vTopNSigma, vTemp, vMirostat,
           vMirostatEta, vMirostatTau, vAdaptiveTarget, vAdaptiveDecay);

  return std::string(aResult);
}

struct common_sampler * fCommonSamplerInit(const struct dgs_model * pModel, struct common_params_sampling & pParams) {
  if (!std::isfinite(pParams.vPenaltyRepeat) || pParams.vPenaltyRepeat <= 0.0f ||
      !std::isfinite(1.0f / pParams.vPenaltyRepeat)) {
    throw std::invalid_argument("penalty_repeat must be finite and greater than 0");
  }
  if (!std::isfinite(pParams.vPenaltyFreq)) {
    throw std::invalid_argument("penalty_freq must be finite");
  }
  if (!std::isfinite(pParams.vPenaltyPresent)) {
    throw std::invalid_argument("penalty_present must be finite");
  }
  const dgs_vocab *        vVocab   = fDgsModelGetVocab(pModel);
  dgs_sampler_chain_params vLparams = fDgsSamplerChainDefaultParams();

  vLparams.vNoPerf = pParams.vNoPerf;

  dgs_sampler * vGrmr    = nullptr;
  dgs_sampler * vRbudget = nullptr;
  dgs_sampler * vChain   = fDgsSamplerChainInit(vLparams);

  std::vector<dgs_sampler *> lSamplers;

  const std::string & vGrammarStr = fCommonGrammarValue(pParams.vGrammar);
  if (vGrammarStr.compare(0, 11, "%llguidance") == 0) {
#ifdef DGS_USE_LLGUIDANCE
    grmr = fDgsSamplerInitLlg(vocab, "lark", grammar_str.c_str());
#else
    throw std::runtime_error("failed to parse grammar: llguidance is not enabled");
#endif  // DGS_USE_LLGUIDANCE
  } else {
    std::vector<std::string> lTriggerPatterns;
    std::vector<dgs_token>   lTriggerTokens;
    for (const auto & vTrigger : pParams.lGrammarTriggers) {
      switch (vTrigger.type) {
        case cCommonGrammarTriggerTypeWord:
          {
            const auto & vWord = vTrigger.value;
            lTriggerPatterns.push_back(fRegexEscape(vWord));
            break;
          }
        case cCommonGrammarTriggerTypePattern:
          {
            lTriggerPatterns.push_back(vTrigger.value);
            break;
          }
        case cCommonGrammarTriggerTypePatternFull:
          {
            const auto & vPattern  = vTrigger.value;
            std::string  vAnchored = "^$";
            if (!vPattern.empty()) {
              vAnchored = (vPattern.front() != '^' ? "^" : "") + vPattern + (vPattern.back() != '$' ? "$" : "");
            }
            lTriggerPatterns.push_back(vAnchored);
            break;
          }
        case cCommonGrammarTriggerTypeToken:
          {
            const auto token = vTrigger.token;
            lTriggerTokens.push_back(token);
            break;
          }
        default:
          GGML_ASSERT(false && "unknown trigger type");
      }
    }

    std::vector<const char *> lTriggerPatternsC;
    lTriggerPatternsC.reserve(lTriggerPatterns.size());
    for (const auto & vRegex : lTriggerPatterns) {
      lTriggerPatternsC.push_back(vRegex.c_str());
    }

    if (!vGrammarStr.empty()) {
      if (pParams.vGrammarLazy) {
        vGrmr =
          fDgsSamplerInitGrammarLazyPatterns(vVocab, vGrammarStr.c_str(), "root", lTriggerPatternsC.data(),
                                             lTriggerPatternsC.size(), lTriggerTokens.data(), lTriggerTokens.size());
      } else {
        vGrmr = fDgsSamplerInitGrammar(vVocab, vGrammarStr.c_str(), "root");
      }
    }
  }
  if (!vGrmr && !vGrammarStr.empty()) {
    throw std::runtime_error("failed to parse grammar");
  }

  // Compute prefill tokens from the generation prompt
  std::vector<dgs_token> lPrefillTokens;
  if (!pParams.vGenerationPrompt.empty()) {
    GGML_ASSERT(vVocab != nullptr);
    auto lTokens = fCommonTokenize(vVocab, pParams.vGenerationPrompt, false, true);
    for (size_t vI = 0; vI < lTokens.size(); vI++) {
      std::string vPiece = fCommonTokenToPiece(vVocab, lTokens[vI], true);
      if (vI == 0 && std::isspace(vPiece[0]) && !std::isspace(pParams.vGenerationPrompt[0])) {
        // Some tokenizers will add a space before the first special token, need to exclude
        continue;
      }
      LOG_DBG("%s: prefill token: %d = %s\n", __func__, lTokens[vI], vPiece.c_str());
      lPrefillTokens.push_back(lTokens[vI]);
    }
  }

  // Feed generation prompt tokens to the grammar sampler so it advances past
  // tokens the template already placed in the prompt.
  // Only applies to output-format and tool-call grammars; user-supplied grammars must not be prefilled.
  if (vGrmr && !pParams.vGrammarLazy && fCommonGrammarNeedsPrefill(pParams.vGrammar)) {
    try {
      for (const auto & token : lPrefillTokens) {
        fDgsSamplerAccept(vGrmr, token);
        LOG_DBG("%s: grammar accepted prefill token (%d)\n", __func__, token);
      }
    } catch (std::exception & vE) {
      LOG_ERR("%s: error initializing grammar sampler for grammar:\n%s\n\nGeneration prompt:\n'%s'\n", __func__,
              fCommonGrammarValue(pParams.vGrammar).c_str(), pParams.vGenerationPrompt.c_str());
      throw vE;
    }
  }

  // reasoning budget sampler (skip when budget is unlimited unless a lazy grammar is active, which needs rbudget for thinking-block suppression)
  if (!pParams.lReasoningBudgetStart.empty() && !pParams.lReasoningBudgetEnd.empty() &&
      (pParams.vGrammarLazy || pParams.vReasoningBudgetTokens >= 0 || pParams.vReasoningControl)) {
    vRbudget = fCommonReasoningBudgetInit(
      vVocab, { pParams.lReasoningBudgetStart }, pParams.lReasoningBudgetEnd, pParams.lReasoningBudgetForced,
      pParams.vReasoningBudgetTokens < 0 ? INT_MAX : pParams.vReasoningBudgetTokens);

    for (const auto & token : lPrefillTokens) {
      fDgsSamplerAccept(vRbudget, token);
      LOG_DBG("%s: reasoning-budget accepted prefill token (%d)\n", __func__, token);
    }
  }

  // logit bias: user biases + model suppress tokens (-INFINITY)
  {
    std::vector<dgs_logit_bias> lMerged = pParams.lLogitBias;

    int32_t           vNSuppress = 0;
    const dgs_token * vSuppress  = fDgsVocabGetSuppressTokens(vVocab, &vNSuppress);
    for (int32_t vI = 0; vI < vNSuppress; ++vI) {
      lMerged.push_back({ vSuppress[vI], -INFINITY });
    }

    if (!lMerged.empty()) {
      lSamplers.push_back(fDgsSamplerInitLogitBias(fDgsVocabNTokens(vVocab), lMerged.size(), lMerged.data()));
    }
  }

  if (pParams.vMirostat == 0) {
    bool vUseAdaptiveP = false;  // see below

    for (const auto & vCnstr : pParams.lSamplers) {
      switch (vCnstr) {
        case cCommonSamplerTypeDry:
          {
            std::vector<const char *> lCBreakers;
            lCBreakers.reserve(pParams.lDrySequenceBreakers.size());
            for (const auto & str : pParams.lDrySequenceBreakers) {
              lCBreakers.push_back(str.c_str());
            }
            lSamplers.push_back(fDgsSamplerInitDry(vVocab, pParams.vDryMultiplier, pParams.vDryBase,
                                                   pParams.vDryAllowedLength, pParams.vDryPenaltyLastN,
                                                   lCBreakers.data(), lCBreakers.size()));
          }
          break;
        case cCommonSamplerTypeTopK:
          lSamplers.push_back(fDgsSamplerInitTopK(pParams.vTopK));
          break;
        case cCommonSamplerTypeTopP:
          lSamplers.push_back(fDgsSamplerInitTopP(pParams.vTopP, pParams.vMinKeep));
          break;
        case cCommonSamplerTypeTopNSigma:
          lSamplers.push_back(fDgsSamplerInitTopNSigma(pParams.vTopNSigma));
          break;
        case cCommonSamplerTypeMinP:
          lSamplers.push_back(fDgsSamplerInitMinP(pParams.vMinP, pParams.vMinKeep));
          break;
        case cCommonSamplerTypeXtc:
          lSamplers.push_back(
            fDgsSamplerInitXtc(pParams.vXtcProbability, pParams.vXtcThreshold, pParams.vMinKeep, pParams.vSeed));
          break;
        case cCommonSamplerTypeTypicalP:
          lSamplers.push_back(fDgsSamplerInitTypical(pParams.vTypP, pParams.vMinKeep));
          break;
        case cCommonSamplerTypeTemperature:
          lSamplers.push_back(fDgsSamplerInitTempExt(pParams.vTemp, pParams.vDynatempRange, pParams.vDynatempExponent));
          break;
        case cCommonSamplerTypeInfill:
          lSamplers.push_back(fDgsSamplerInitInfill(vVocab));
          break;
        case cCommonSamplerTypePenalties:
          lSamplers.push_back(fDgsSamplerInitPenalties(fDgsVocabNTokens(vVocab), pParams.vPenaltyLastN,
                                                       pParams.vPenaltyRepeat, pParams.vPenaltyFreq,
                                                       pParams.vPenaltyPresent));
          break;
        case cCommonSamplerTypeAdaptiveP:
          // the `adaptive-p` sampler is like `dist` and `mirostat` in that it selects
          // a single token, so we will add `dist` at the end of the chain by default,
          // unless the user specifically included `adaptive-p`. we set this flag here
          // so we know to add the sampler at the very end.
          vUseAdaptiveP = true;
          break;
        default:
          GGML_ASSERT(false && "unknown sampler type");
      }
    }
    if (vUseAdaptiveP) {
      // only if user explicitly included adaptive-p sampler
      lSamplers.push_back(fDgsSamplerInitAdaptiveP(pParams.vAdaptiveTarget, pParams.vAdaptiveDecay, pParams.vSeed));
    } else {
      // Keep distribution sampling when callers request probabilities.
      const bool cGreedy = pParams.vNProbs == 0 && !pParams.lSamplers.empty() &&
                           ((pParams.lSamplers.back() == cCommonSamplerTypeTemperature && pParams.vTemp == 0.0f &&
                             pParams.vDynatempRange == 0.0f) ||
                            (pParams.lSamplers.back() == cCommonSamplerTypeTopK && pParams.vTopK == 1));
      lSamplers.push_back(cGreedy ? fDgsSamplerInitGreedy() : fDgsSamplerInitDist(pParams.vSeed));
    }
  } else if (pParams.vMirostat == 1) {
    lSamplers.push_back(fDgsSamplerInitTemp(pParams.vTemp));
    lSamplers.push_back(fDgsSamplerInitMirostat(fDgsVocabNTokens(vVocab), pParams.vSeed, pParams.vMirostatTau,
                                                pParams.vMirostatEta, 100));
  } else if (pParams.vMirostat == 2) {
    lSamplers.push_back(fDgsSamplerInitTemp(pParams.vTemp));
    lSamplers.push_back(fDgsSamplerInitMirostatV2(pParams.vSeed, pParams.vMirostatTau, pParams.vMirostatEta));
  } else {
    GGML_ASSERT(false && "unknown mirostat version");
  }

  for (auto * vSmpl : lSamplers) {
    fDgsSamplerChainAdd(vChain, vSmpl);
  }

  if (vGrmr && pParams.vBackendSampling) {
    LOG_WRN("%s: backend sampling is not compatible with grammar, disabling\n", __func__);

    pParams.vBackendSampling = false;
  }

  if (vRbudget && pParams.vBackendSampling) {
    LOG_WRN("%s: backend sampling is not compatible with reasoning budget, disabling\n", __func__);

    pParams.vBackendSampling = false;
  }

  auto * vResult = new common_sampler{
    /* .params  = */ pParams,
    /* .grmr    = */ vGrmr,
    /* .rbudget = */ vRbudget,
    /* .chain   = */ vChain,
    /* .prev    = */ ring_buffer<dgs_token>(std::max(32, pParams.vNPrev)),
    /* .cur     = */ {},
    /* .cur_p   = */ {},
    // mix it, the chain and the draft are seeded from this one too
    /* .rng     = */ std::mt19937(fDgsSamplerGetSeed(vChain) ^ 0x9e3779b9u),
  };

  return vResult;
}

void fCommonSamplerFree(struct common_sampler * pGsmpl) {
  if (!pGsmpl) {
    return;
  }

  fDgsSamplerFree(pGsmpl->vGrmr);
  fDgsSamplerFree(pGsmpl->vRbudget);
  fDgsSamplerFree(pGsmpl->vChain);

  delete pGsmpl;
}

static bool fGrammarShouldApply(struct common_sampler * pGsmpl) {
  if (!pGsmpl->vGrmr) {
    return false;
  }
  if (!pGsmpl->vRbudget) {
    return true;
  }
  if (pGsmpl->vParams.vGrammarLazy) {
    // if grammar is lazy, only apply when reasoning budget is not active
    const auto cState = fCommonReasoningBudgetGetState(pGsmpl->vRbudget);
    return cState == cReasoningBudgetIdle || cState == cReasoningBudgetDone;
  }
  return true;
}

void fCommonSamplerAccept(struct common_sampler * pGsmpl, dgs_token token, bool pIsGenerated) {
  if (!pGsmpl) {
    return;
  }

  const auto cTm = pGsmpl->mTm();

  // grammar_should_apply() checks the reasoning budget state, so calculate this before we accept
  const auto cAcceptGrammar = pIsGenerated && fGrammarShouldApply(pGsmpl);

  if (pGsmpl->vRbudget && pIsGenerated) {
    fDgsSamplerAccept(pGsmpl->vRbudget, token);

    // if done, replay end sequence which may contain a grammar trigger
    const bool cIsDone = fCommonReasoningBudgetGetState(pGsmpl->vRbudget) == cReasoningBudgetDone;
    if (pGsmpl->vGrmr && !cAcceptGrammar && cIsDone) {
      const dgs_tokens * lEndSeq = fCommonReasoningBudgetGetEndMatch(pGsmpl->vRbudget);
      if (lEndSeq) {
        for (const dgs_token cEndToken : *lEndSeq) {
          fDgsSamplerAccept(pGsmpl->vGrmr, cEndToken);
        }
      }
    }
  }

  if (pGsmpl->vGrmr && cAcceptGrammar) {
    fDgsSamplerAccept(pGsmpl->vGrmr, token);
  }

  fDgsSamplerAccept(pGsmpl->vChain, token);

  pGsmpl->vPrev.push_back(token);
}

void fCommonSamplerReset(struct common_sampler * pGsmpl) {
  if (!pGsmpl) {
    return;
  }

  pGsmpl->reset();
}

struct common_sampler * fCommonSamplerClone(common_sampler * pGsmpl) {
  return new common_sampler{
    /* .params  = */ pGsmpl->vParams,
    /* .grmr    = */ fDgsSamplerClone(pGsmpl->vGrmr),
    /* .rbudget = */ fDgsSamplerClone(pGsmpl->vRbudget),
    /* .chain   = */ fDgsSamplerClone(pGsmpl->vChain),
    /* .prev    = */ pGsmpl->vPrev,
    /* .cur     = */ pGsmpl->lCur,
    /* .cur_p   = */ pGsmpl->vCurP,
    /* .rng     = */ pGsmpl->vRng,
  };
}

void fCommonSamplerCopy(const common_sampler * pSrc, common_sampler * pDst) {
  if (!pSrc || !pDst || pSrc == pDst) {
    return;
  }

  GGML_ASSERT((pSrc->vGrmr == nullptr) == (pDst->vGrmr == nullptr));
  GGML_ASSERT((pSrc->vRbudget == nullptr) == (pDst->vRbudget == nullptr));

  fDgsSamplerCopy(pSrc->vGrmr, pDst->vGrmr);
  fDgsSamplerCopy(pSrc->vRbudget, pDst->vRbudget);
  fDgsSamplerCopy(pSrc->vChain, pDst->vChain);

  pDst->vParams    = pSrc->vParams;
  pDst->vPrev      = pSrc->vPrev;
  pDst->lCur       = pSrc->lCur;
  pDst->vCurP      = pSrc->vCurP;
  pDst->vCurP.data = pSrc->vCurP.data ? pDst->lCur.data() : nullptr;  // re-point to dst's buffer
  pDst->vRng       = pSrc->vRng;
  pDst->vTTotalUs  = pSrc->vTTotalUs;
}

void fCommonPerfPrint(const struct dgs_context * pCtx, const struct common_sampler * pGsmpl) {
  // TODO: measure grammar performance

  const double cTSamplingMs = pGsmpl ? 1e-3 * pGsmpl->vTTotalUs : 0;

  dgs_perf_sampler_data vDataSmpl;
  dgs_perf_context_data vDataCtx;

  memset(&vDataSmpl, 0, sizeof(vDataSmpl));
  memset(&vDataCtx, 0, sizeof(vDataCtx));

  if (pGsmpl) {
    auto & data = vDataSmpl;

    data = fDgsPerfSampler(pGsmpl->vChain);

    // note: the sampling time includes the samplers time + extra time spent in common/sampling
    LOG_INF("%s:    sampling time = %10.2f ms\n", __func__, cTSamplingMs);
    LOG_INF("%s:    samplers time = %10.2f ms / %5d tokens\n", __func__, data.vTSampleMs, data.vNSample);
  }

  if (pCtx) {
    auto & data = vDataCtx;

    data = fDgsPerfContext(pCtx);

    const double cTEndMs = 1e-3 * ggml_time_us();

    const double cTTotalMs = cTEndMs - data.vTStartMs;
    const double cTUnaccMs = cTTotalMs - (cTSamplingMs + data.vTPEvalMs + data.vTEvalMs);
    const double cTUnaccPc = 100.0 * cTUnaccMs / cTTotalMs;

    LOG_INF("%s:        load time = %10.2f ms\n", __func__, data.vTLoadMs);
    LOG_INF("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n", __func__,
            data.vTPEvalMs, data.vNPEval, data.vTPEvalMs / data.vNPEval, 1e3 / data.vTPEvalMs * data.vNPEval);
    LOG_INF("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n", __func__,
            data.vTEvalMs, data.vNEval, data.vTEvalMs / data.vNEval, 1e3 / data.vTEvalMs * data.vNEval);
    LOG_INF("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (cTEndMs - data.vTStartMs),
            (data.vNPEval + data.vNEval));
    LOG_INF("%s: unaccounted time = %10.2f ms / %5.1f %%      (total - sampling - prompt eval - eval) / (total)\n",
            __func__, cTUnaccMs, cTUnaccPc);
    LOG_INF("%s:    graphs reused = %10d\n", __func__, data.vNReused);
    LOG_INF("%s:   graph reallocs = %10d    / %5d        (same size) / (changed size)\n", __func__, data.vNReallocsSame,
            data.vNReallocsDiff);

    fCommonMemoryBreakdownPrint(pCtx);
  }
}

struct dgs_sampler * fCommonSamplerGet(const struct common_sampler * pGsmpl) {
  if (!pGsmpl) {
    return nullptr;
  }

  return pGsmpl->vChain;
}

dgs_token fCommonSamplerSample(struct common_sampler * pGsmpl,
                               struct dgs_context *    pCtx,
                               int                     pIdx,
                               bool                    pGrammarFirst) {
  fDgsSynchronize(pCtx);

  // start measuring sampling time after the dgs_context synchronization in order to not measure any ongoing async operations
  const auto cTm = pGsmpl->mTm();

  dgs_token vId = DGS_TOKEN_NULL;

  auto & vGrmr    = pGsmpl->vGrmr;
  auto & vRbudget = pGsmpl->vRbudget;
  auto & vChain   = pGsmpl->vChain;
  auto & vCurP    = pGsmpl->vCurP;  // initialized by set_logits

  pGsmpl->mSetLogits(pCtx, pIdx);

  // Check if a backend sampler has already sampled a token in which case we
  // return that token id directly.
  {
    vId = fDgsGetSampledTokenIth(pCtx, pIdx);

    if (vId != DGS_TOKEN_NULL) {
      LOG_DBG("%s: Backend sampler selected token: '%d'. Will not run any CPU samplers\n", __func__, vId);

      GGML_ASSERT(!pGsmpl->vGrmr && "using grammar in combination with backend sampling is not supported");
      GGML_ASSERT(!pGsmpl->vRbudget && "using reasoning budget in combination with backend sampling is not supported");

      for (size_t vI = 0; vI < vCurP.size; ++vI) {
        if (vCurP.data[vI].vId == vId) {
          vCurP.vSelected = vI;
          break;
        }
      }

      return vId;
    }
  }

  // apply reasoning budget first
  fDgsSamplerApply(vRbudget, &vCurP);

  if (pGrammarFirst && fGrammarShouldApply(pGsmpl)) {
    fDgsSamplerApply(vGrmr, &vCurP);
  }

  fDgsSamplerApply(vChain, &vCurP);

  vId = vCurP.data[vCurP.vSelected].vId;

  if (pGrammarFirst || !fGrammarShouldApply(pGsmpl)) {
    return vId;
  }

  // check if it the sampled token fits the grammar (grammar-based rejection sampling)
  {
    dgs_token_data       vSingleTokenData      = { vId, 1.0f, 0.0f };
    dgs_token_data_array vSingleTokenDataArray = { &vSingleTokenData, 1, -1, false };

    fDgsSamplerApply(vGrmr, &vSingleTokenDataArray);

    const bool cIsValid = vSingleTokenDataArray.data[0].vLogit != -INFINITY;
    if (cIsValid) {
      return vId;
    }
  }

  // resampling:
  // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
  pGsmpl->mSetLogits(pCtx, pIdx);

  fDgsSamplerApply(vRbudget, &vCurP);

  if (fGrammarShouldApply(pGsmpl)) {
    fDgsSamplerApply(vGrmr, &vCurP);
  }

  fDgsSamplerApply(vChain, &vCurP);

  GGML_ASSERT(vCurP.vSelected != -1 && "no selected token during sampling - check your sampling configuration");

  vId = vCurP.data[vCurP.vSelected].vId;

  return vId;
}

std::vector<dgs_token> fCommonSamplerSampleAndAcceptN(struct common_sampler *  pGsmpl,
                                                      struct dgs_context *     pCtx,
                                                      const std::vector<int> & pIdxs,
                                                      const dgs_tokens &       pDraft,
                                                      bool                     pGrammarFirst) {
  GGML_ASSERT(pIdxs.size() == pDraft.size() + 1 && "idxs.size() must be draft.size() + 1");

  std::vector<dgs_token> lResult;
  lResult.reserve(pIdxs.size());

  const dgs_vocab * vVocab = fDgsModelGetVocab(fDgsGetModel(pCtx));

  size_t vI = 0;
  for (; vI < pDraft.size(); vI++) {
    const dgs_token cId = fCommonSamplerSample(pGsmpl, pCtx, pIdxs[vI], pGrammarFirst);

    fCommonSamplerAccept(pGsmpl, cId, true);

    lResult.push_back(cId);

    // do not accept draft tokens after an EOG - they are not output but would stay in the context
    // on replay the last token is from the target and can be EOG, so a trailing EOG is still accepted
    if (pDraft[vI] != cId || (fDgsVocabIsEog(vVocab, cId) && vI + 1 < pDraft.size())) {
      break;
    }
  }

  if (vI == pDraft.size()) {
    const dgs_token cId = fCommonSamplerSample(pGsmpl, pCtx, pIdxs[vI], pGrammarFirst);

    fCommonSamplerAccept(pGsmpl, cId, true);

    lResult.push_back(cId);
  }

  return lResult;
}

static float fProbOf(const dgs_token_data * data, size_t n, dgs_token pId) {
  for (size_t vK = 0; vK < n; ++vK) {
    if (data[vK].vId == pId) {
      return data[vK].vP;
    }
  }
  return 0.0f;
}

// Accept a drafted token with probability min(1, p/q), else draw from norm(max(0, p - q)).
// Preserves the target distribution exactly, and accepts more often than matching does when the
// draft samples instead of taking its argmax.
std::vector<dgs_token> fCommonSamplerSampleAndAcceptNRejection(struct common_sampler *                          pGsmpl,
                                                               struct dgs_context *                             pCtx,
                                                               const std::vector<int> &                         pIdxs,
                                                               const dgs_tokens &                               pDraft,
                                                               const std::vector<std::vector<dgs_token_data>> & pDraftQ,
                                                               bool pGrammarFirst) {
  GGML_ASSERT(pIdxs.size() == pDraft.size() + 1 && "idxs.size() must be draft.size() + 1");
  GGML_ASSERT(pDraftQ.size() == pDraft.size() && "draft_q must have one entry per draft token");

  std::vector<dgs_token> lResult;
  lResult.reserve(pIdxs.size());

  // draws come from the sampler's own stream, so they stay independent of what was drafted
  std::uniform_real_distribution<float> vUni(0.0f, 1.0f);

  std::vector<dgs_token_data> lResidual;

  std::vector<dgs_token_data> lCand;  // candidate array masked by the grammar, if there is one

  size_t vI = 0;
  for (; vI < pDraft.size(); vI++) {
    // leaves the target distribution in the candidate array
    const dgs_token cIdTgt = fCommonSamplerSample(pGsmpl, pCtx, pIdxs[vI], pGrammarFirst);

    const auto * vCurP = fCommonSamplerGetCandidates(pGsmpl, true);
    const auto & lQ    = pDraftQ[vI];

    const bool cMasked = !pGrammarFirst && fGrammarShouldApply(pGsmpl);
    if (cMasked) {
      lCand.assign(vCurP->data, vCurP->data + vCurP->size);
      dgs_token_data_array vArr = { lCand.data(), lCand.size(), -1, false };
      fDgsSamplerApply(pGsmpl->vGrmr, &vArr);
    }

    // a candidate the grammar rejects carries no probability, whatever the target thinks
    auto vPRaw = [&](size_t pK) {
      return cMasked && lCand[pK].vLogit == -INFINITY ? 0.0f : vCurP->data[pK].vP;
    };

    // masking drops probability mass, so rescale what is left or the residual is over-weighted
    float vPSum = 0.0f;
    if (cMasked) {
      for (size_t vK = 0; vK < vCurP->size; ++vK) {
        vPSum += vPRaw(vK);
      }
    }

    const float cPNorm = cMasked && vPSum > 0.0f ? 1.0f / vPSum : 1.0f;

    auto vPOf = [&](size_t pK) {
      return vPRaw(pK) * cPNorm;
    };

    // q_x is never 0 for a token the draft produced, but guard the divide
    const float cQX = fProbOf(lQ.data(), lQ.size(), pDraft[vI]);

    float vPX = 0.0f;
    for (size_t vK = 0; vK < vCurP->size; ++vK) {
      if (vCurP->data[vK].vId == pDraft[vI]) {
        vPX = vPOf(vK);
        break;
      }
    }

    if (cQX > 0.0f && (vPX >= cQX || vUni(pGsmpl->vRng) < vPX / cQX)) {
      fCommonSamplerAccept(pGsmpl, pDraft[vI], true);
      lResult.push_back(pDraft[vI]);
      continue;
    }

    // rejected: tokens outside q's support keep all of p
    lResidual.clear();
    float vSum = 0.0f;
    for (size_t vK = 0; vK < vCurP->size; ++vK) {
      const float cR = vPOf(vK) - fProbOf(lQ.data(), lQ.size(), vCurP->data[vK].vId);
      if (cR > 0.0f) {
        lResidual.push_back({ vCurP->data[vK].vId, 0.0f, cR });
        vSum += cR;
      }
    }

    dgs_token vId = cIdTgt;
    if (vSum > 0.0f) {
      float vU = vUni(pGsmpl->vRng) * vSum;
      vId      = lResidual.back().vId;
      for (const auto & vE : lResidual) {
        vU -= vE.vP;
        if (vU <= 0.0f) {
          vId = vE.vId;
          break;
        }
      }
    }

    fCommonSamplerAccept(pGsmpl, vId, true);
    lResult.push_back(vId);

    break;
  }

  if (vI == pDraft.size()) {
    const dgs_token cId = fCommonSamplerSample(pGsmpl, pCtx, pIdxs[vI], pGrammarFirst);

    fCommonSamplerAccept(pGsmpl, cId, true);

    lResult.push_back(cId);
  }

  return lResult;
}

std::vector<dgs_token> fCommonSamplerSampleAndAcceptN(struct common_sampler * pGsmpl,
                                                      struct dgs_context *    pCtx,
                                                      const dgs_tokens &      pDraft,
                                                      bool                    pGrammarFirst) {
  std::vector<int> lIdxs(pDraft.size() + 1);
  for (size_t vI = 0; vI < lIdxs.size(); ++vI) {
    lIdxs[vI] = vI;
  }

  return fCommonSamplerSampleAndAcceptN(pGsmpl, pCtx, lIdxs, pDraft, pGrammarFirst);
}

uint32_t fCommonSamplerGetSeed(const struct common_sampler * pGsmpl) {
  return fDgsSamplerGetSeed(pGsmpl->vChain);
}

bool fCommonSamplerReasoningBudgetForce(struct common_sampler * pGsmpl) {
  if (!pGsmpl) {
    return false;
  }

  return fCommonReasoningBudgetForce(pGsmpl->vRbudget);
}

// helpers

dgs_token_data_array * fCommonSamplerGetCandidates(struct common_sampler * pGsmpl, bool pDoSort) {
  const auto cTm = pGsmpl->mTm();

  auto * vRes = &pGsmpl->vCurP;

  if (pDoSort && !vRes->vSorted) {
    // remember the selected token before sorting
    const dgs_token cId = vRes->data[vRes->vSelected].vId;

    std::sort(vRes->data, vRes->data + vRes->size,
              [](const dgs_token_data & pA, const dgs_token_data & pB) { return pA.vP > pB.vP; });

    // restore the selected token after sorting
    for (size_t vI = 0; vI < vRes->size; ++vI) {
      if (vRes->data[vI].vId == cId) {
        vRes->vSelected = vI;
        break;
      }
    }

    vRes->vSorted = true;
  }

  return vRes;
}

dgs_token fCommonSamplerLast(const struct common_sampler * pGsmpl) {
  return pGsmpl->vPrev.mRat(0);
}

std::string fCommonSamplerPrint(const struct common_sampler * pGsmpl) {
  std::string vResult = "logits ";

  for (int vI = 0; vI < fDgsSamplerChainN(pGsmpl->vChain); vI++) {
    const auto * vSmpl = fDgsSamplerChainGet(pGsmpl->vChain, vI);
    vResult += std::string("-> ");
    vResult += std::string(fDgsSamplerName(vSmpl)) + " ";
  }

  return vResult;
}

std::string fCommonSamplerPrevStr(common_sampler * pGsmpl, dgs_context * pCtxMain, int n) {
  n = std::min(n, (int) pGsmpl->vPrev.size());

  if (n <= 0) {
    return "";
  }

  std::string vResult;
  vResult.reserve(8 * n);  // 8 is the average length of a token [citation needed], TODO: compute this from the vocab

  for (int vI = n - 1; vI >= 0; vI--) {
    const dgs_token cId = pGsmpl->vPrev.mRat(vI);

    GGML_ASSERT(cId != DGS_TOKEN_NULL && "null token in the sampling history - should not happen");

    vResult += fCommonTokenToPiece(pCtxMain, cId);
  }

  return vResult;
}

char fCommonSamplerTypeToChr(enum common_sampler_type pCnstr) {
  switch (pCnstr) {
    case cCommonSamplerTypeDry:
      return 'd';
    case cCommonSamplerTypeTopK:
      return 'k';
    case cCommonSamplerTypeTypicalP:
      return 'y';
    case cCommonSamplerTypeTopP:
      return 'p';
    case cCommonSamplerTypeTopNSigma:
      return 's';
    case cCommonSamplerTypeMinP:
      return 'm';
    case cCommonSamplerTypeTemperature:
      return 't';
    case cCommonSamplerTypeXtc:
      return 'x';
    case cCommonSamplerTypeInfill:
      return 'i';
    case cCommonSamplerTypePenalties:
      return 'e';
    case cCommonSamplerTypeAdaptiveP:
      return 'a';
    default:
      return '?';
  }
}

std::string fCommonSamplerTypeToStr(enum common_sampler_type pCnstr) {
  switch (pCnstr) {
    case cCommonSamplerTypeDry:
      return "dry";
    case cCommonSamplerTypeTopK:
      return "top_k";
    case cCommonSamplerTypeTypicalP:
      return "typ_p";
    case cCommonSamplerTypeTopP:
      return "top_p";
    case cCommonSamplerTypeTopNSigma:
      return "top_n_sigma";
    case cCommonSamplerTypeMinP:
      return "min_p";
    case cCommonSamplerTypeTemperature:
      return "temperature";
    case cCommonSamplerTypeXtc:
      return "xtc";
    case cCommonSamplerTypeInfill:
      return "infill";
    case cCommonSamplerTypePenalties:
      return "penalties";
    case cCommonSamplerTypeAdaptiveP:
      return "adaptive_p";
    default:
      return "";
  }
}

std::vector<common_sampler_type> fCommonSamplerTypesFromNames(const std::vector<std::string> & pNames) {
  // sampler names can be written multiple ways; generate aliases from canonical names
  static const auto dSamplerNameMap = [] {
    // canonical sampler name mapping
    std::unordered_map<std::string, common_sampler_type> dCanonicalNameMap{
      { "dry",         cCommonSamplerTypeDry         },
      { "top_k",       cCommonSamplerTypeTopK        },
      { "top_p",       cCommonSamplerTypeTopP        },
      { "top_n_sigma", cCommonSamplerTypeTopNSigma   },
      { "typ_p",       cCommonSamplerTypeTypicalP    },
      { "min_p",       cCommonSamplerTypeMinP        },
      { "temperature", cCommonSamplerTypeTemperature },
      { "xtc",         cCommonSamplerTypeXtc         },
      { "infill",      cCommonSamplerTypeInfill      },
      { "penalties",   cCommonSamplerTypePenalties   },
      { "adaptive_p",  cCommonSamplerTypeAdaptiveP   }
    };
    std::unordered_map<std::string, common_sampler_type> dAliasNameMap;
    for (const auto & vEntry : dCanonicalNameMap) {
      const std::string & vCanonical = vEntry.first;
      if (vCanonical.find('_') == std::string::npos) {
        continue;
      }
      // kebab-case: "top-k", "min-p", etc.
      {
        std::string vKebabCase = vCanonical;
        std::replace(vKebabCase.begin(), vKebabCase.end(), '_', '-');
        dAliasNameMap.insert({ vKebabCase, vEntry.second });
      }
      // no dash: "topk", "minp", etc.
      {
        std::string vNoDash = vCanonical;
        vNoDash.erase(std::remove(vNoDash.begin(), vNoDash.end(), '_'), vNoDash.end());
        dAliasNameMap.insert({ vNoDash, vEntry.second });
      }
    }
    // misc. aliases
    dAliasNameMap.insert({ "nucleus", cCommonSamplerTypeTopP });
    dAliasNameMap.insert({ "temp", cCommonSamplerTypeTemperature });
    dAliasNameMap.insert({ "typ", cCommonSamplerTypeTypicalP });
    // include aliases + canonical names in the complete mapping
    dAliasNameMap.merge(dCanonicalNameMap);
    return dAliasNameMap;
  }();

  std::vector<common_sampler_type> lSamplers;
  lSamplers.reserve(pNames.size());

  for (const auto & name : pNames) {
    std::string vNameLower = name;
    std::transform(vNameLower.begin(), vNameLower.end(), vNameLower.begin(), ::tolower);
    auto vSampler = dSamplerNameMap.find(vNameLower);
    if (vSampler != dSamplerNameMap.end()) {
      lSamplers.push_back(vSampler->second);
      continue;
    }
    LOG_WRN("%s: unable to match sampler by name '%s'\n", __func__, vNameLower.c_str());
  }

  return lSamplers;
}

std::vector<common_sampler_type> fCommonSamplerTypesFromChars(const std::string & pChars) {
  std::unordered_map<char, common_sampler_type> dSamplerNameMap = {
    { fCommonSamplerTypeToChr(cCommonSamplerTypeDry),         cCommonSamplerTypeDry         },
    { fCommonSamplerTypeToChr(cCommonSamplerTypeTopK),        cCommonSamplerTypeTopK        },
    { fCommonSamplerTypeToChr(cCommonSamplerTypeTypicalP),    cCommonSamplerTypeTypicalP    },
    { fCommonSamplerTypeToChr(cCommonSamplerTypeTopP),        cCommonSamplerTypeTopP        },
    { fCommonSamplerTypeToChr(cCommonSamplerTypeTopNSigma),   cCommonSamplerTypeTopNSigma   },
    { fCommonSamplerTypeToChr(cCommonSamplerTypeMinP),        cCommonSamplerTypeMinP        },
    { fCommonSamplerTypeToChr(cCommonSamplerTypeTemperature), cCommonSamplerTypeTemperature },
    { fCommonSamplerTypeToChr(cCommonSamplerTypeXtc),         cCommonSamplerTypeXtc         },
    { fCommonSamplerTypeToChr(cCommonSamplerTypeInfill),      cCommonSamplerTypeInfill      },
    { fCommonSamplerTypeToChr(cCommonSamplerTypePenalties),   cCommonSamplerTypePenalties   },
    { fCommonSamplerTypeToChr(cCommonSamplerTypeAdaptiveP),   cCommonSamplerTypeAdaptiveP   },
  };

  std::vector<common_sampler_type> lSamplers;
  lSamplers.reserve(pChars.size());

  for (const auto & vC : pChars) {
    const auto cSampler = dSamplerNameMap.find(vC);
    if (cSampler != dSamplerNameMap.end()) {
      lSamplers.push_back(cSampler->second);
    } else {
      LOG_WRN("%s: unable to match sampler by char '%c'\n", __func__, vC);
    }
  }

  return lSamplers;
}

void fCommonSamplingAddPreservedTokens(common_params_sampling &         pSampling,
                                       const dgs_vocab *                pVocab,
                                       const std::vector<std::string> & pTokens) {
  GGML_ASSERT(pVocab != nullptr);
  for (const auto & vT : pTokens) {
    auto lIds = fCommonTokenize(pVocab, vT, false, true);
    if (lIds.size() == 1) {
      pSampling.sPreservedTokens.insert(lIds[0]);
    }
  }
}

void fCommonSamplingAddGrammarTriggers(common_params_sampling &            pSampling,
                                       const dgs_vocab *                   pVocab,
                                       std::vector<common_grammar_trigger> pTriggers) {
  GGML_ASSERT(pVocab != nullptr);
  for (auto & vTrigger : pTriggers) {
    if (vTrigger.type == cCommonGrammarTriggerTypeWord) {
      const auto & vWord = vTrigger.value;
      auto         lIds  = fCommonTokenize(pVocab, vWord, false, true);
      if (lIds.size() == 1) {
        auto token = lIds[0];
        if (std::find(pSampling.sPreservedTokens.begin(), pSampling.sPreservedTokens.end(), (dgs_token) token) ==
            pSampling.sPreservedTokens.end()) {
          throw std::runtime_error("Grammar trigger word should be marked as preserved token: " + vWord);
        }
        common_grammar_trigger vTokenTrigger;
        vTokenTrigger.type  = cCommonGrammarTriggerTypeToken;
        vTokenTrigger.value = vWord;
        vTokenTrigger.token = token;
        pSampling.lGrammarTriggers.push_back(std::move(vTokenTrigger));
      } else {
        pSampling.lGrammarTriggers.push_back({ cCommonGrammarTriggerTypeWord, vWord });
      }
    } else {
      pSampling.lGrammarTriggers.push_back(std::move(vTrigger));
    }
  }
  if (pSampling.vGrammarLazy && pSampling.lGrammarTriggers.empty()) {
    throw std::runtime_error("Error: no triggers set for lazy grammar!");
  }
}
