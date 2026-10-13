#include "speculative.h"

#include "../src/dgs-ext.h"  // staging API: dgs_set_embeddings_nextn / dgs_get_embeddings_nextn_ith (used by MTP)
#include "common.h"
#include "dgs.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <map>

#define SPC_DBG(fmt, ...) LOG_DBG("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_TRC(fmt, ...) LOG_TRC("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_INF(fmt, ...) LOG_INF("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_WRN(fmt, ...) LOG_WRN("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_ERR(fmt, ...) LOG_ERR("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_CNT(fmt, ...) LOG_CNT("" fmt, __VA_ARGS__)

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

// Rebuild seq_id's draft sampler at the target's temperature: rejection weighs q against p, so
// both have to sample alike. Only temp and seed carry over; the draft keeps its own top_k.
static void fSpecRetune(std::vector<common_sampler_ptr> &     pSmpls,
                        std::vector<common_params_sampling> & pCfg,
                        const dgs_model *                     pModel,
                        dgs_seq_id                            pSeqId,
                        float                                 pTemp,
                        uint32_t                              pSeed) {
  if (pCfg.size() != pSmpls.size()) {
    const size_t cNOld = pCfg.size();
    pCfg.resize(pSmpls.size());

    // the initial sampler has no temperature, so no request may match the cache and skip a rebuild
    for (size_t vI = cNOld; vI < pCfg.size(); ++vI) {
      pCfg[vI].vTemp = NAN;
    }
  }

  auto & vCur = pCfg[pSeqId];

  if (vCur.vTemp == pTemp && vCur.vSeed == pSeed) {
    return;
  }

  vCur.vTemp = pTemp;
  vCur.vSeed = pSeed;

  common_params_sampling vSparams;
  vSparams.vNoPerf   = false;
  vSparams.vTopK     = 10;
  vSparams.vTemp     = vCur.vTemp;
  // must be explicit, the default reseeds at random; mixed so it differs from the target's
  vSparams.vSeed     = vCur.vSeed == DGS_DEFAULT_SEED ? vCur.vSeed : vCur.vSeed ^ 0x85ebca6bu;
  vSparams.lSamplers = { cCommonSamplerTypeTopK, cCommonSamplerTypeTemperature };

  pSmpls[pSeqId].reset(fCommonSamplerInit(pModel, vSparams));
}

const std::map<std::string, common_speculative_type> dCommonSpeculativeTypeFromNameMap = {
  { "none",          cCommonSpeculativeTypeNone        },
  { "draft-simple",  cCommonSpeculativeTypeDraftSimple },
  { "draft-eagle3",  cCommonSpeculativeTypeDraftEagle3 },
  { "draft-mtp",     cCommonSpeculativeTypeDraftMtp    },
  { "draft-dflash",  cCommonSpeculativeTypeDraftDflash },
  { "draft-dspark",  cCommonSpeculativeTypeDraftDspark },
  { "ngram-simple",  cCommonSpeculativeTypeNgramSimple },
  { "ngram-map-k",   cCommonSpeculativeTypeNgramMapK   },
  { "ngram-map-k4v", cCommonSpeculativeTypeNgramMapK4v },
  { "ngram-mod",     cCommonSpeculativeTypeNgramMod    },
  { "ngram-cache",   cCommonSpeculativeTypeNgramCache  }
};

static std::string fCommonSpeculativeGetDevicesStr(const std::vector<ggml_backend_dev_t> & pDevices) {
  std::string vResult;
  for (size_t vI = 0; vI < pDevices.size(); vI++) {
    if (pDevices[vI] == nullptr) {
      continue;
    }
    if (!vResult.empty()) {
      vResult += ", ";
    }
    vResult += ggml_backend_dev_name(pDevices[vI]);
  }
  return vResult.empty() ? "default" : vResult;
}

struct common_speculative_config {
  common_speculative_type   type;
  common_params_speculative vParams;

  common_speculative_config(common_speculative_type           pT,
                            const common_params_speculative & pP = common_params_speculative{}) :
    type(pT),
    vParams(pP) {}
};

bool fCommonSpeculativeAreCompatible(const dgs_model * pModelTgt, const dgs_model * pModelDft) {
  const dgs_vocab * vVocabTgt = fDgsModelGetVocab(pModelTgt);
  const dgs_vocab * vVocabDft = fDgsModelGetVocab(pModelDft);

  const auto cVocabTypeTgt = fDgsVocabType(vVocabTgt);
  SPC_DBG("vocab_type tgt: %d\n", cVocabTypeTgt);

  const auto cVocabTypeDft = fDgsVocabType(vVocabDft);
  SPC_DBG("vocab_type dft: %d\n", cVocabTypeDft);

  if (cVocabTypeTgt != cVocabTypeDft) {
    SPC_WRN(
      "draft model vocab type must match target model to use speculation but "
      "vocab_type_dft = %d while vocab_type_tgt = %d\n",
      cVocabTypeDft, cVocabTypeTgt);
    return false;
  }

  if (fDgsVocabGetAddBos(vVocabTgt) != fDgsVocabGetAddBos(vVocabDft) ||
      (fDgsVocabGetAddBos(vVocabTgt) && fDgsVocabBos(vVocabTgt) != fDgsVocabBos(vVocabDft))) {
    SPC_WRN("draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
            fDgsVocabGetAddBos(vVocabTgt), fDgsVocabGetAddBos(vVocabDft), fDgsVocabBos(vVocabTgt),
            fDgsVocabBos(vVocabDft));
    return false;
  }

  if (fDgsVocabGetAddEos(vVocabTgt) != fDgsVocabGetAddEos(vVocabDft) ||
      (fDgsVocabGetAddEos(vVocabTgt) && fDgsVocabEos(vVocabTgt) != fDgsVocabEos(vVocabDft))) {
    SPC_WRN("draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
            fDgsVocabGetAddEos(vVocabTgt), fDgsVocabGetAddEos(vVocabDft), fDgsVocabEos(vVocabTgt),
            fDgsVocabEos(vVocabDft));
    return false;
  }

  {
    const int cNVocabTgt = fDgsVocabNTokens(vVocabTgt);
    const int cNVocabDft = fDgsVocabNTokens(vVocabDft);
    const int cVocabDiff = cNVocabTgt > cNVocabDft ? cNVocabTgt - cNVocabDft : cNVocabDft - cNVocabTgt;

    if (cVocabDiff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
      SPC_DBG(
        "draft model vocab must closely match target model to use speculation but "
        "target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
        cNVocabTgt, fDgsVocabNTokens(vVocabDft), cVocabDiff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
      return false;
    }

    for (int vI = SPEC_VOCAB_CHECK_START_TOKEN_ID; vI < std::min(cNVocabTgt, cNVocabDft); ++vI) {
      const char * vTokenTextTgt = fDgsVocabGetText(vVocabTgt, vI);
      const char * vTokenTextDft = fDgsVocabGetText(vVocabDft, vI);

      if (std::strcmp(vTokenTextTgt, vTokenTextDft) != 0) {
        SPC_DBG(
          "draft model vocab must match target model to use speculation but "
          "token %d content differs - target '%s', draft '%s'\n",
          vI, fCommonTokenToPiece(vVocabTgt, vI).c_str(), fCommonTokenToPiece(vVocabDft, vI).c_str());
        return false;
      }
    }
  }

  return true;
}

using common_speculative_draft_params_vec = std::vector<common_speculative_draft_params>;

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_impl
struct common_speculative_impl {
  const common_speculative_type type;

  uint32_t vNSeq;
  int32_t  vNMax;                         // maximum draft length after implementation-specific limits

  size_t vNCallBegin  = 0;                // number of times this implementation was called for refresh.
  size_t vNCallDraft  = 0;                // number of times this implementation was called for generation.
  size_t vNCallAccept = 0;                // number of times this implementation was called for accumulation.

  size_t vNGenDrafts = 0;                 // number of times a draft or part was generated by this implementation.
  size_t vNAccDrafts = 0;                 // number of times a draft or part was accepted by the target model.
  size_t vNGenTokens = 0;                 // number of tokens generated by this implementation.
  size_t vNAccTokens = 0;                 // number of tokens accepted by the target model.

  std::vector<size_t> lNAccTokensPerPos;  // number of tokens accepted per draft position.

  // TODO: track performance of most recent calls
  const bool cGenPerf = true;  // whether to generate performance stats.

  int64_t vTBeginUs  = 0;      // total time spent in refresh of this implementation in microseconds.
  int64_t vTDraftUs  = 0;      // total time spent in generating drafts in this implementation in microseconds.
  int64_t vTAcceptUs = 0;      // total time spent in accumulation of this implementation in microseconds.

  common_speculative_impl(common_speculative_type type, uint32_t pNSeq, int32_t pNMax) :
    type(type),
    vNSeq(pNSeq),
    vNMax(pNMax) {}

  virtual ~common_speculative_impl() = default;

  virtual void begin(dgs_seq_id pSeqId, const dgs_tokens & pPrompt) = 0;

  virtual bool mProcess(const common_batch & pBatch) = 0;

  virtual void mDraft(common_speculative_draft_params_vec & pDparams) = 0;

  virtual void accept(dgs_seq_id pSeqId, uint16_t pNAccepted, bool pIsOther) = 0;

  // (optional) serialize/restore per-seq internal state (e.g. eagle3's deferred boundary).
  virtual bool mGetState(dgs_seq_id /*seq_id*/, std::vector<uint8_t> & /*data*/) const { return false; }

  virtual void mSetState(dgs_seq_id /*seq_id*/, const std::vector<uint8_t> & /*data*/) {}
};

struct common_speculative_impl_draft_simple : public common_speculative_impl {
  common_params_speculative_draft vParams;

  common_batch vBatch;

  // zero row at the draft input width, stands in for target embeddings the draft cannot read
  std::vector<float> lZeros;
  bool               vZerosWarned = false;  // the substitution is reported once

  std::vector<common_sampler_ptr> lSmpls;

  std::vector<common_params_sampling> lSmplsCfg;

  common_speculative_impl_draft_simple(const common_params_speculative & pParams, uint32_t pNSeq) :
    common_speculative_impl(cCommonSpeculativeTypeDraftSimple, pNSeq, pParams.vDraft.vNMax),
    vParams(pParams.vDraft) {
    auto * vCtxDft = this->vParams.vCtxDft;
    auto * vCtxTgt = this->vParams.vCtxTgt;

    if (!vCtxDft) {
      throw std::runtime_error("draft-simple requires a draft context");
    }

    lZeros.assign(fDgsModelNEmbdInp(fDgsGetModel(vCtxDft)), 0.0f);

    SPC_TRC("%s", "adding speculative implementation 'draft-simple'\n");
    SPC_TRC("- n_max=%d, n_min=%d, p_min=%f\n", this->vParams.vNMax, this->vParams.vNMin, this->vParams.vPMin);
    SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
            this->vParams.vNGpuLayers, ggml_type_name(this->vParams.vCacheTypeK),
            ggml_type_name(this->vParams.vCacheTypeV), vCtxTgt ? "yes" : "no", vCtxDft ? "yes" : "no",
            fCommonSpeculativeGetDevicesStr(this->vParams.lDevices).c_str());

    vBatch = common_batch(vCtxDft);

    // TODO: optimize or pass from outside?
    // {
    //     common_params_sampling params;
    //     params.no_perf = false;
    //
    //     params.top_k = 40;
    //     params.top_p = 0.9;
    //
    //     params.samplers = {
    //         COMMON_SAMPLER_TYPE_TOP_K,
    //         COMMON_SAMPLER_TYPE_TOP_P,
    //         COMMON_SAMPLER_TYPE_INFILL,
    //     };
    //
    //     result->smpl = common_sampler_init(dgs_get_model(ctx_dft), params);
    // }

    lSmpls.resize(pNSeq);
    for (auto & vSmpl : lSmpls) {
      common_params_sampling vParams;
      vParams.vNoPerf = false;
      vParams.vTopK   = 10;
      vParams.lSamplers.assign(1, cCommonSamplerTypeTopK);

      vSmpl.reset(fCommonSamplerInit(fDgsGetModel(vCtxDft), vParams));
    }

    const bool cVocabCmpt = fCommonSpeculativeAreCompatible(fDgsGetModel(vCtxTgt), fDgsGetModel(vCtxDft));
    SPC_DBG("vocab_cmpt = %d\n", cVocabCmpt);

    if (!cVocabCmpt) {
      SPC_ERR("%s", "the target and draft vocabs are not compatible\n");

      throw std::runtime_error("draft model vocab type must match target model to use speculation");
    }

    if (pNSeq != fDgsNSeqMax(vCtxDft)) {
      SPC_ERR("n_seq mismatch: %d != %d\n", pNSeq, fDgsNSeqMax(vCtxDft));

      throw std::runtime_error("the draft model number of sequences is incompatible with the speculative n_seq");
    }
  }

  void begin(dgs_seq_id pSeqId, const dgs_tokens & /*prompt*/) override {
    // reset here rather than per round, or two identical requests differ
    fCommonSamplerReset(lSmpls[pSeqId].get());
  }

  bool mProcess(const common_batch & pBatchIn) override {
    auto * vCtxDft = vParams.vCtxDft;

    // copy the entries to a batch owned by the draft context, only the last token is output
    vBatch.clear();
    const int32_t cNTokens = pBatchIn.size();
    for (int32_t vK = 0; vK < cNTokens; ++vK) {
      const auto & vT      = pBatchIn.lTokens[vK];
      const bool   cOutput = vK == cNTokens - 1;
      if (vT.vId != DGS_TOKEN_NULL) {
        const int32_t cIdx = vBatch.mAdd(vT.vId, vT.aPos[0], vT.vSeqId, cOutput);
        if (vT.vEmbd.data) {
          vBatch.mSetEmbd(cIdx, vT.vEmbd);
        }
      } else {
        // mtmd input is projected by the target encoder, a draft with a different width cannot read it
        // it gets zeros instead, keeping its positions contiguous
        const size_t cNEmbd     = vT.vEmbd.vNRows * vT.vEmbd.vNEmbd;
        const bool   cSameWidth = cNEmbd == lZeros.size();
        if (!cSameWidth && !vZerosWarned) {
          SPC_WRN(
            "target embeddings of size %zu do not fit the draft input width %zu, "
            "the draft receives zero rows for them and drafts after multimodal input will be poor\n",
            cNEmbd, lZeros.size());
          vZerosWarned = true;
        }
        const dgs_embd cEmbd = cSameWidth ? vT.vEmbd : dgs_embd{ lZeros.data(), 1, lZeros.size() };
        vBatch.mAddEmbd(cEmbd, vT.aPos.data(), vT.vSeqId, cOutput);
      }
    }

    if (vBatch.size() == 0) {
      return true;
    }

    const int cRet = fDgsProcess(vCtxDft, cDgsProcessTypeDecode, vBatch.get());

    if (cRet != 0) {
      SPC_ERR("failed to decode draft batch, ret = %d\n", cRet);

      return false;
    }

    return true;
  }

  void mDraft(common_speculative_draft_params_vec & pDparams) override {
    auto & vCtxDft = vParams.vCtxDft;

    vBatch.clear();

    // keep track of which sequences are still drafting
    int               vNDrafting = 0;
    std::vector<bool> lDrafting(vNSeq);

    for (dgs_seq_id vSeqId = 0; vSeqId < (dgs_seq_id) vNSeq; ++vSeqId) {
      auto & vDp = pDparams[vSeqId];

      if (!vDp.vDrafting) {
        continue;
      }

      vNDrafting++;
      lDrafting[vSeqId] = true;
      // greedy drafting leaves no candidates behind, so the verifier falls back to sample-and-match
      if (!vParams.vProbabilistic) {
        vDp.lResultQ = nullptr;
      }

      // result_q is only set when the caller wants rejection, so it also gates the retune
      if (vDp.lResultQ) {
        fSpecRetune(lSmpls, lSmplsCfg, fDgsGetModel(vCtxDft), vSeqId, vDp.vTemp, vDp.vSeed);
      }

      // a reset reseeds the chain, which breaks probabilistic drafting
      if (!vDp.lResultQ) {
        fCommonSamplerReset(lSmpls[vSeqId].get());
      }

      vBatch.mAdd(vDp.vIdLast, vDp.vPos0, vSeqId, true);
    }

    int vRet = fDgsProcess(vCtxDft, cDgsProcessTypeDecode, vBatch.get());
    if (vRet != 0) {
      SPC_ERR("dgs_process returned %d\n", vRet);
      return;
    }

    int vI = 0;

    while (vNDrafting > 0) {
      int vIBatch = 0;

      vBatch.clear();

      for (dgs_seq_id vSeqId = 0; vSeqId < (dgs_seq_id) vNSeq; ++vSeqId) {
        if (!lDrafting[vSeqId]) {
          continue;
        }

        auto * vSmpl = lSmpls[vSeqId].get();

        const dgs_token cIdSampled = fCommonSamplerSample(vSmpl, vCtxDft, vIBatch, true);
        ++vIBatch;

        const auto * vCurP = fCommonSamplerGetCandidates(vSmpl, true);

        for (int vK = 0; vK < std::min(3, (int) vCurP->size); ++vK) {
          SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n", vSeqId, vK, vI, vCurP->data[vK].vId,
                  vCurP->data[vK].vP, fCommonTokenToPiece(vCtxDft, vCurP->data[vK].vId).c_str());
        }

        // add drafted token for each sequence
        const dgs_token cId = pDparams.at(vSeqId).lResultQ ? cIdSampled : vCurP->data[0].vId;

        // only collect very high-confidence draft tokens
        if (vCurP->data[0].vP < vParams.vPMin) {
          lDrafting[vSeqId] = false;
          vNDrafting--;

          continue;
        }

        fCommonSamplerAccept(vSmpl, cId, true);

        auto & vDp     = pDparams.at(vSeqId);
        auto & lResult = *vDp.lResult;

        lResult.push_back(cId);

        if (vDp.lResultQ) {
          vDp.lResultQ->emplace_back(vCurP->data, vCurP->data + vCurP->size);
        }

        if ((vParams.vNMax <= (int) lResult.size()) || (vDp.vNMax > 0 && vDp.vNMax <= (int) lResult.size())) {
          lDrafting[vSeqId] = false;
          vNDrafting--;
          continue;
        }

        vBatch.mAdd(cId, vDp.vPos0 + vI + 1, vSeqId, true);
      }

      if (vBatch.size() == 0) {
        break;
      }

      // evaluate the drafted tokens on the draft model
      vRet = fDgsProcess(vCtxDft, cDgsProcessTypeDecode, vBatch.get());
      if (vRet != 0) {
        SPC_ERR("dgs_process[%d] returned %d\n", vI, vRet);
        break;
      }

      ++vI;
    }

    for (auto & vDp : pDparams) {
      if (!vDp.vDrafting) {
        continue;
      }

      if (vDp.lResult->size() < (size_t) vParams.vNMin) {
        vDp.lResult->clear();
      }
    }
  }

  void accept(dgs_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
    // noop
  }
};

// EAGLE3 speculative decoding state
//
// Input of draft decoder: (This is different compared to MTP)
//   At "pos P", the decoder takes input pair (t_{P+1}, g_P), with RoPE at P.
//     - t_{P+1} = token at sequence pos P+1 (the *next* token after P)
//     - g_P     = encoder output = projection of target's extracted hidden states at P
//
// Deferred boundary (MTP doesn't have this issue):
//   Within a single process() call with n_tokens, we can only write decoder KV for
//   training pos 0..n_tokens-2. The last training pos (n_tokens-1) needs t_{n_tokens}
//   which lies *outside* this batch — it is the token target will sample next or the first token from next ubatch.
//   So the last training pos of each process() call is *deferred* to whichever next call has
//   the missing token in hand:
//     - multi-ubatch prefill: the next process()'s first token completes the pair
//                              (handled by the per-seq "cross-ubatch bridge")
//     - single-ubatch prefill / after verify: draft()'s seed step uses "dp.id_last"
//                              (target's freshest sample) to complete the pair
//
// Per-seq carry-over state:
//   pending_g_last    [n_embd_dec]  ┐  the deferred boundary's (g, pos). Set by
//   pending_pos_last  dgs_pos     ┘  process() at end of ubatch (= last row);
//                                       rebased by accept() to first-non-accepted pos.
//   verify_g          [N × n_embd_dec] snapshot of process()'s encoder output;
//   verify_pos_first  dgs_pos         consumed by accept() to recover the right
//   verify_g_rows     int32_t           pending_g_last row for any n_accepted value.
//
// Performance is overall good but there is waste in verify cycle:
//   process() runs encoder + decoder on the *full* verify batch including rows for
//   rejected drafts. The KV at those positions is then dropped.
//
// TODO: Not sure if we need optimization for this waste?
// If so we may need hybrid stash:
//      in verify mode, have process() only stash features and let draft() seed run
//      encoder+decoder on n_accepted+1 rows).
struct common_speculative_impl_draft_eagle3 : public common_speculative_impl {
  common_params_speculative_draft vParams;
  common_batch                    vBatch;     // decoder input, (token, g_embd) pairs
  common_batch                    vBatchEnc;  // encoder input, built from the extracted target features

  std::vector<common_sampler_ptr> lSmpls;

  // backend sampler chain per seq, attached to ctx_dft
  std::vector<dgs_sampler *> lBackendChains;

  int32_t vNEmbdDec  = 0;                      // draft hidden size
  int32_t vNEmbdEnc  = 0;                      // target_layer_ids_n * target_hidden_size
  int32_t vNEmbdTgt  = 0;                      // target model hidden size
  int32_t vNLayerTgt = 0;                      // target model layer count

  const int32_t * vTargetLayerIds  = nullptr;  // model_dft's extract layer indices
  uint32_t        vTargetLayerIdsN = 0;

  // [per-seq] deferred boundary state
  std::vector<std::vector<float>> lPendingGLast;
  std::vector<dgs_pos>            lPendingPosLast;

  // [per-seq] snapshot of the most recent process()'s encoder output
  std::vector<std::vector<float>> lVerifyG;         // [n_seq][n_rows * n_embd_dec]
  std::vector<dgs_pos>            lVerifyPosFirst;  // [n_seq] — pos of verify_g[seq][0]
  std::vector<int32_t>            lVerifyGRows;     // [n_seq] — number of rows

  // scratch buffer for concatenated target features [n_tokens, n_embd_enc]
  std::vector<float> lFeaturesBuf;
  std::vector<float> lGEmbdBuf;

  common_speculative_impl_draft_eagle3(const common_params_speculative & pParams, uint32_t pNSeq) :
    common_speculative_impl(cCommonSpeculativeTypeDraftEagle3, pNSeq, pParams.vDraft.vNMax),
    vParams(pParams.vDraft) {
    SPC_TRC("%s", "adding speculative implementation 'draft-eagle3'\n");
    SPC_TRC("- n_max=%d, n_min=%d, p_min=%f, backend_sampling=%d\n", pParams.vDraft.vNMax, pParams.vDraft.vNMin,
            pParams.vDraft.vPMin, (int) pParams.vDraft.vBackendSampling);

    auto * vCtxTgt = this->vParams.vCtxTgt;
    auto * vCtxDft = this->vParams.vCtxDft;
    GGML_ASSERT(vCtxTgt && vCtxDft && "EAGLE3 requires ctx_tgt and ctx_dft to be set");

    const dgs_model * vModelDft = fDgsGetModel(vCtxDft);
    const dgs_model * vModelTgt = fDgsGetModel(vCtxTgt);

    vTargetLayerIds  = fDgsModelTargetLayerIds(vModelDft);
    vTargetLayerIdsN = fDgsModelTargetLayerIdsN(vModelDft);
    if (vTargetLayerIdsN != 3) {
      throw std::runtime_error("draft model is not eagle3 (expected 3 extract layers, got " +
                               std::to_string(vTargetLayerIdsN) + ")");
    }

    vNEmbdTgt  = fDgsModelNEmbd(vModelTgt);
    vNEmbdDec  = fDgsModelNEmbd(vModelDft);
    vNEmbdEnc  = (int32_t) vTargetLayerIdsN * vNEmbdTgt;
    vNLayerTgt = fDgsModelNLayer(vModelTgt);

    vBatch    = common_batch(vCtxDft);
    vBatchEnc = common_batch(vCtxDft);

    lSmpls.resize(pNSeq);
    for (auto & vS : lSmpls) {
      common_params_sampling vSparams;
      vSparams.vNoPerf   = false;
      vSparams.vTopK     = 10;
      vSparams.lSamplers = { cCommonSamplerTypeTopK };
      vS.reset(fCommonSamplerInit(fDgsGetModel(vCtxDft), vSparams));
    }

    // offload draft sampling to the backend
    lBackendChains.assign(pNSeq, nullptr);
    if (this->vParams.vBackendSampling) {
      for (dgs_seq_id vSeqId = 0; vSeqId < (dgs_seq_id) pNSeq; ++vSeqId) {
        dgs_sampler * vChain = fDgsSamplerChainInit(fDgsSamplerChainDefaultParams());
        fDgsSamplerChainAdd(vChain, fDgsSamplerInitTopK(10));

        if (!fDgsSetSampler(vCtxDft, vSeqId, vChain)) {
          SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) vSeqId);
          fDgsSamplerFree(vChain);
          vChain = nullptr;
        }
        lBackendChains[vSeqId] = vChain;
      }
    }

    // turn on extraction of the target layers' hidden states
    for (uint32_t vK = 0; vK < vTargetLayerIdsN; ++vK) {
      if (vTargetLayerIds[vK] < vNLayerTgt) {
        fDgsSetEmbeddingsLayerInp(vCtxTgt, (uint32_t) vTargetLayerIds[vK], true);
      } else if (vTargetLayerIds[vK] == vNLayerTgt) {
        fDgsSetEmbeddingsNextn(vCtxTgt, true, /*masked*/ false);
      } else {
        GGML_ABORT("EAGLE3: target layer id %d exceeds target n_layer %d", vTargetLayerIds[vK], vNLayerTgt);
      }
    }

    // turn on extraction of the draft model's pre-norm hidden state
    // (used both for the encoder output g_embd and the decoder pre-norm output).
    fDgsSetEmbeddingsNextn(vCtxDft, true, /*masked*/ true);

    lPendingGLast.assign(pNSeq, std::vector<float>(vNEmbdDec, 0.0f));
    lPendingPosLast.assign(pNSeq, -1);

    lVerifyG.assign(pNSeq, std::vector<float>());
    lVerifyPosFirst.assign(pNSeq, -1);
    lVerifyGRows.assign(pNSeq, 0);
  }

  ~common_speculative_impl_draft_eagle3() override {
    auto * vCtxDft = this->vParams.vCtxDft;
    for (dgs_seq_id vSeqId = 0; vSeqId < (dgs_seq_id) lBackendChains.size(); ++vSeqId) {
      if (lBackendChains[vSeqId] == nullptr) {
        continue;
      }
      if (vCtxDft) {
        fDgsSetSampler(vCtxDft, vSeqId, nullptr);
      }
      fDgsSamplerFree(lBackendChains[vSeqId]);
    }
    lBackendChains.clear();
  }

  void begin(dgs_seq_id pSeqId, const dgs_tokens & pPrompt) override {
    const int32_t cN = (int32_t) pPrompt.size();
    if (cN <= 0) {
      return;
    }
    // expected state after prefill: ctx_dft has pos 0..N-2 (last position is deferred to
    // draft()'s seed step). Warn only if more than one position is missing.
    auto *        vCtxDft = this->vParams.vCtxDft;
    const dgs_pos cPosMax = fDgsMemorySeqPosMax(fDgsGetMemory(vCtxDft), pSeqId);
    if (cPosMax < cN - 2) {
      SPC_WRN(
        "ctx_dft pos_max=%d < N-2=%d — process() did not run on every prefill ubatch. "
        "Drafts may degrade.\n",
        (int) cPosMax, cN - 2);
    }
  }

  bool mProcess(const common_batch & pBatchIn) override {
    if (pBatchIn.size() <= 0) {
      return true;
    }

    if (!pBatchIn.mHasToken() || pBatchIn.mHasEmbd()) {
      return true;
    }

    const int32_t cNTokens = pBatchIn.size();

    // i_batch_beg[seq] / i_batch_end[seq]: inclusive batch indices of this seq's
    // first/last token in batch_in. Assumes per-seq tokens are contiguous within
    // the ubatch (server's default ordering).
    std::vector<int32_t> lIBatchBeg(vNSeq, -1);
    std::vector<int32_t> lIBatchEnd(vNSeq, -1);
    for (int vK = 0; vK < cNTokens; ++vK) {
      const dgs_seq_id cSeqId = pBatchIn.lTokens[vK].vSeqId;
      if (cSeqId < 0 || cSeqId >= (dgs_seq_id) vNSeq) {
        continue;
      }
      lIBatchEnd[cSeqId] = vK;
      if (lIBatchBeg[cSeqId] < 0) {
        lIBatchBeg[cSeqId] = vK;
      }
    }

    auto * vCtxTgt = this->vParams.vCtxTgt;
    auto * vCtxDft = this->vParams.vCtxDft;

    // Interleave each extract_layer's hidden state into a contiguous buffer of
    // shape [n_tokens, target_layer_ids_n * n_embd_tgt]. Then run EAGLE3 encoder
    // to get one g_embd row per token.
    lFeaturesBuf.resize((size_t) cNTokens * vNEmbdEnc, 0.0f);

    for (uint32_t vK = 0; vK < vTargetLayerIdsN; ++vK) {
      const float * layer = vTargetLayerIds[vK] < vNLayerTgt ?
                              fDgsGetEmbeddingsLayerInp(vCtxTgt, (uint32_t) vTargetLayerIds[vK]) :
                              fDgsGetEmbeddingsNextn(vCtxTgt);
      if (!layer) {
        GGML_ABORT("EAGLE3: target layer %d input not extracted.", vTargetLayerIds[vK]);
      }
      for (int32_t vI = 0; vI < cNTokens; ++vI) {
        float *       vDst = lFeaturesBuf.data() + (size_t) vI * vNEmbdEnc + vK * (size_t) vNEmbdTgt;
        const float * vSrc = layer + (size_t) vI * vNEmbdTgt;
        std::memcpy(vDst, vSrc, (size_t) vNEmbdTgt * sizeof(float));
      }
    }

    lGEmbdBuf.resize((size_t) cNTokens * vNEmbdDec);

    // dgs_process() requires the full encoder batch to fit in n_ubatch.
    // Allow batch > ubatch: eagle3's per-token encoder can be chunked safely.
    const int32_t cNUbatchDft = (int32_t) fDgsNUbatch(vCtxDft);
    for (int32_t vI = 0; vI < cNTokens; vI += cNUbatchDft) {
      const int32_t cNChunk = std::min(cNUbatchDft, cNTokens - vI);

      // the per-token encoder does not use positions, generate placeholder ones from the memory state
      vBatchEnc.clear();
      dgs_pos vPos = fDgsMemorySeqPosMax(fDgsGetMemory(vCtxDft), 0) + 1;
      for (int32_t vJ = 0; vJ < cNChunk; ++vJ) {
        vBatchEnc.mAddEmbd({ lFeaturesBuf.data() + (size_t) (vI + vJ) * vNEmbdEnc, 1, (size_t) vNEmbdEnc }, &vPos, 0,
                           true);
        vPos++;
      }

      const int32_t cRc = fDgsProcess(vCtxDft, cDgsProcessTypeEncode, vBatchEnc.get());
      if (cRc != 0) {
        SPC_ERR("dgs_process(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n", cRc, (int) cNChunk, (int) vI);
        return false;
      }

      // g_embd has shape [n_chunk, n_embd_dec] in ctx_dft's pre-norm embeddings buffer.
      const float * vGEmbdChunk = fDgsGetEmbeddingsNextn(vCtxDft);
      GGML_ASSERT(vGEmbdChunk && "EAGLE3 encoder produced no output.");
      std::memcpy(lGEmbdBuf.data() + (size_t) vI * vNEmbdDec, vGEmbdChunk,
                  (size_t) cNChunk * vNEmbdDec * sizeof(float));
    }

    const float * vGEmbd = lGEmbdBuf.data();

    const size_t cRowBytes = (size_t) vNEmbdDec * sizeof(float);

    // EAGLE3 decoder input convention: at memory pos P the input pair is
    // (token[P+1], g_embd[P]). This shifts the token index "left by one" relative to g_embd.
    //
    // Per seq, in order:
    //   (a) cross-ubatch bridge — when applicable, write the previously-deferred
    //       pos using this ubatch's first token + pending_g_last.
    //   (b) main write loop — for k in [beg, end-1], write (token[k+1], g_embd[k])
    //       at pos[k]. The last training pos (k=end) is left unwritten = new
    //       deferred boundary, completed by the next process() or draft() call.
    //   (c) refresh deferred state — stash this ubatch's full g_embd into verify_g,
    //       update pending_g_last / pending_pos_last to the last row.
    vBatch.clear();

    for (dgs_seq_id vSeqId = 0; vSeqId < (dgs_seq_id) vNSeq; ++vSeqId) {
      const int32_t cBeg = lIBatchBeg[vSeqId];
      const int32_t end  = lIBatchEnd[vSeqId];
      if (cBeg < 0 || end < 0) {
        continue;
      }

      // cross-ubatch bridge — complete the prior ubatch's deferred boundary.
      // Fires iff all three preconditions hold:
      //   1) pending_pos_last >= 0
      //   2) pending_pos_last + 1 == pos[beg]
      //   3) pending_pos_last > dft_pos_max // TODO: is this check needed?
      const dgs_pos cPendingPos = lPendingPosLast[vSeqId];
      if (cPendingPos >= 0 && cPendingPos + 1 == pBatchIn.lTokens[cBeg].aPos[0]) {
        const dgs_pos cDftPosMax = fDgsMemorySeqPosMax(fDgsGetMemory(vCtxDft), vSeqId);
        if (cPendingPos > cDftPosMax) {
          const int32_t cIdx = vBatch.mAdd(pBatchIn.lTokens[cBeg].vId, cPendingPos, vSeqId, /*output=*/false);
          vBatch.mSetEmbd(cIdx, { lPendingGLast[vSeqId].data(), 1, (size_t) vNEmbdDec });
        }
      }

      for (int32_t vK = cBeg; vK < end; ++vK) {
        const int32_t cIdx =
          vBatch.mAdd(pBatchIn.lTokens[vK + 1].vId, pBatchIn.lTokens[vK].aPos[0], vSeqId, /*output=*/false);
        vBatch.mSetEmbd(cIdx, { vGEmbd + (size_t) vK * vNEmbdDec, 1, (size_t) vNEmbdDec });
      }

      // refresh deferred state
      const int32_t cNRows    = end - cBeg + 1;
      lVerifyPosFirst[vSeqId] = pBatchIn.lTokens[cBeg].aPos[0];
      lPendingPosLast[vSeqId] = pBatchIn.lTokens[end].aPos[0];
      lVerifyGRows[vSeqId]    = cNRows;
      lVerifyG[vSeqId].resize((size_t) cNRows * vNEmbdDec, 0.0f);
      std::memcpy(lVerifyG[vSeqId].data(), vGEmbd + (size_t) cBeg * vNEmbdDec, cRowBytes * cNRows);
      std::memcpy(lPendingGLast[vSeqId].data(), vGEmbd + (size_t) end * vNEmbdDec, cRowBytes);
    }

    if (vBatch.size() > 0) {
      const int32_t cRc = fDgsProcess(vCtxDft, cDgsProcessTypeDecode, vBatch.get());
      if (cRc != 0) {
        SPC_ERR("dgs_process(ctx_dft) failed rc=%d (n_tokens=%d, ubatch_pos[0]=%d)\n", cRc, (int) vBatch.size(),
                (int) pBatchIn.lTokens[0].aPos[0]);
        return false;
      }
    }

    return true;
  }

  void mDraft(common_speculative_draft_params_vec & pDparams) override {
    auto & vCtxDft = vParams.vCtxDft;

    vBatch.clear();

    // keep track of which sequences are still drafting
    int               vNDrafting = 0;
    std::vector<bool> lDrafting(vNSeq);

    // Complete the deferred boundary pair (dp.id_last, pending_g_last) at memory
    // pos pending_pos_last. dp.id_last is target's freshest sample (= corrected
    // token after verify, or first generated token after prefill), matching the
    // EAGLE3 input convention (token[P+1], g_embd[P]) at pos P.
    for (dgs_seq_id vSeqId = 0; vSeqId < (dgs_seq_id) vNSeq; ++vSeqId) {
      auto & vDp = pDparams[vSeqId];

      if (!vDp.vDrafting) {
        continue;
      }
      if (lPendingPosLast[vSeqId] < 0) {
        continue;
      }

      vNDrafting++;
      lDrafting[vSeqId] = true;
      fCommonSamplerReset(lSmpls[vSeqId].get());

      fDgsMemorySeqRm(fDgsGetMemory(vCtxDft), vSeqId, lPendingPosLast[vSeqId], -1);

      const int32_t cIdx = vBatch.mAdd(vDp.vIdLast, lPendingPosLast[vSeqId], vSeqId, true);
      vBatch.mSetEmbd(cIdx, { lPendingGLast[vSeqId].data(), 1, (size_t) vNEmbdDec });
    }

    if (vBatch.size() == 0) {
      return;
    }

    int vRet = fDgsProcess(vCtxDft, cDgsProcessTypeDecode, vBatch.get());
    if (vRet != 0) {
      SPC_ERR("dgs_process returned %d\n", vRet);
      return;
    }

    int vI = 0;

    while (vNDrafting > 0) {
      int vIBatch = 0;

      vBatch.clear();

      for (dgs_seq_id vSeqId = 0; vSeqId < (dgs_seq_id) vNSeq; ++vSeqId) {
        if (!lDrafting[vSeqId]) {
          continue;
        }

        auto * vSmpl = lSmpls[vSeqId].get();

        fCommonSamplerSample(vSmpl, vCtxDft, vIBatch, true);
        // pre-norm hidden state of this position becomes g_embd for the next step
        const float * vPrenorm = fDgsGetEmbeddingsNextnIth(vCtxDft, vIBatch);
        ++vIBatch;

        const auto * vCurP = fCommonSamplerGetCandidates(vSmpl, true);

        for (int vK = 0; vK < std::min(3, (int) vCurP->size); ++vK) {
          SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n", vSeqId, vK, vI, vCurP->data[vK].vId,
                  vCurP->data[vK].vP, fCommonTokenToPiece(vCtxDft, vCurP->data[vK].vId).c_str());
        }

        const dgs_token cId = vCurP->data[0].vId;

        // only collect very high-confidence draft tokens
        // (configurable via --spec-draft-p-min, set to 0.0 to disable early-stop)
        if (vCurP->data[0].vP < vParams.vPMin) {
          lDrafting[vSeqId] = false;
          vNDrafting--;

          continue;
        }

        fCommonSamplerAccept(vSmpl, cId, true);

        auto & vDp     = pDparams.at(vSeqId);
        auto & lResult = *vDp.lResult;

        lResult.push_back(cId);

        if (vParams.vNMax <= (int) lResult.size()) {
          lDrafting[vSeqId] = false;
          vNDrafting--;
          continue;
        }

        const int32_t cIdx = vBatch.mAdd(cId, lPendingPosLast[vSeqId] + (vI + 1), vSeqId, true);
        vBatch.mSetEmbd(cIdx, { vPrenorm, 1, (size_t) vNEmbdDec });
      }

      if (vBatch.size() == 0) {
        break;
      }

      vRet = fDgsProcess(vCtxDft, cDgsProcessTypeDecode, vBatch.get());
      if (vRet != 0) {
        SPC_ERR("dgs_process[%d] returned %d\n", vI, vRet);
        break;
      }

      ++vI;
    }

    for (dgs_seq_id vSeqId = 0; vSeqId < (dgs_seq_id) vNSeq; ++vSeqId) {
      auto & vDp = pDparams[vSeqId];
      if (!vDp.vDrafting) {
        continue;
      }

      if (vDp.lResult->size() < (size_t) vParams.vNMin) {
        vDp.lResult->clear();
      }
    }
  }

  void accept(dgs_seq_id pSeqId, uint16_t pNAccepted, bool /*is_other*/) override {
    if (pSeqId < 0 || pSeqId >= (dgs_seq_id) vNSeq) {
      return;
    }

    const int32_t cNRows = lVerifyGRows[pSeqId];
    if (cNRows <= 0) {
      return;
    }

    const int32_t cIG       = std::min<int32_t>(pNAccepted, cNRows - 1);
    lPendingPosLast[pSeqId] = lVerifyPosFirst[pSeqId] + cIG;
    std::memcpy(lPendingGLast[pSeqId].data(), lVerifyG[pSeqId].data() + (size_t) cIG * vNEmbdDec,
                (size_t) vNEmbdDec * sizeof(float));
  }

  // we only need to stash the deferred boundary's g_embd row for recurrent/hybrid targets:
  // their single-position checkpoints drop it on restore
  bool mNeedBoundaryStash() const {
    const dgs_model * vModelTgt = fDgsGetModel(vParams.vCtxTgt);
    return fDgsModelIsRecurrent(vModelTgt) || fDgsModelIsHybrid(vModelTgt);
  }

  bool mGetState(dgs_seq_id pSeqId, std::vector<uint8_t> & data) const override {
    if (!mNeedBoundaryStash()) {
      return false;
    }
    if (pSeqId < 0 || pSeqId >= (dgs_seq_id) vNSeq || lPendingPosLast[pSeqId] < 0) {
      return false;
    }

    const dgs_pos              cPos = lPendingPosLast[pSeqId];
    const std::vector<float> & lG   = lPendingGLast[pSeqId];

    data.resize(sizeof(dgs_pos) + lG.size() * sizeof(float));
    std::memcpy(data.data(), &cPos, sizeof(dgs_pos));
    std::memcpy(data.data() + sizeof(dgs_pos), lG.data(), lG.size() * sizeof(float));
    return true;
  }

  void mSetState(dgs_seq_id pSeqId, const std::vector<uint8_t> & data) override {
    if (!mNeedBoundaryStash()) {
      return;
    }
    if (pSeqId < 0 || pSeqId >= (dgs_seq_id) vNSeq) {
      return;
    }
    if (data.size() != sizeof(dgs_pos) + (size_t) vNEmbdDec * sizeof(float)) {
      return;
    }

    dgs_pos vPos = -1;
    std::memcpy(&vPos, data.data(), sizeof(dgs_pos));

    lPendingPosLast[pSeqId] = vPos;
    lPendingGLast[pSeqId].resize(vNEmbdDec);
    std::memcpy(lPendingGLast[pSeqId].data(), data.data() + sizeof(dgs_pos), (size_t) vNEmbdDec * sizeof(float));
  }
};

// DFlash: block-diffusion drafting with a draft-side KV cache injection
struct common_speculative_impl_draft_dflash : public common_speculative_impl {
  common_params_speculative_draft vParams;

  common_batch vBatch;              // noise tokens
  common_batch vBatchInject;        // target features for KV cache injection

  std::vector<float> lFeaturesBuf;  // [n_chunk, n_embd_enc] gathered target features

  std::vector<common_sampler_ptr> lSmpls;

  // backend sampler chain per seq, attached to ctx_dft
  std::vector<dgs_sampler *> lBackendChains;

  int32_t vNEmbdDec = 0;  // draft hidden size
  int32_t vNEmbdEnc = 0;  // target_layer_ids_n * target_hidden_size
  int32_t vNEmbdTgt = 0;  // target model hidden size

  int32_t   vBlockSize   = 0;
  dgs_token vMaskTokenId = 0;

  bool    vIsDflash2    = false;
  bool    vIsMrope      = false;
  int32_t vSelectorTopK = 0;

  // draft-dspark: the draft carries a Markov head and uses an anchor-first block layout
  const bool cIsDspark;

  // dspark speculators
  bool vSampleFromAnchor = true;

  // block-internal attention
  bool vCausalAttn = false;

  const int32_t * vTargetLayerIds  = nullptr;  // model_dft's extract layer indices
  uint32_t        vTargetLayerIdsN = 0;

  common_speculative_impl_draft_dflash(const common_params_speculative & pParams,
                                       uint32_t                          pNSeq,
                                       common_speculative_type           type = cCommonSpeculativeTypeDraftDflash) :
    common_speculative_impl(type, pNSeq, pParams.vDraft.vNMax),
    vParams(pParams.vDraft),
    cIsDspark(type == cCommonSpeculativeTypeDraftDspark) {
    auto * vCtxTgt = this->vParams.vCtxTgt;
    auto * vCtxDft = this->vParams.vCtxDft;
    GGML_ASSERT(vCtxTgt && vCtxDft && "DFlash requires ctx_tgt and ctx_dft to be set");

    const dgs_model * vModelDft = fDgsGetModel(vCtxDft);
    const dgs_model * vModelTgt = fDgsGetModel(vCtxTgt);

    vTargetLayerIds  = fDgsModelTargetLayerIds(vModelDft);
    vTargetLayerIdsN = fDgsModelTargetLayerIdsN(vModelDft);
    GGML_ASSERT(vTargetLayerIdsN > 0 && "DFlash model has no target_layer_ids");

    vNEmbdTgt = fDgsModelNEmbd(vModelTgt);
    vNEmbdDec = fDgsModelNEmbd(vModelDft);
    vNEmbdEnc = (int32_t) vTargetLayerIdsN * vNEmbdTgt;

    // read the trained block size from the dflash.block_size metadata key
    vBlockSize = 16;
    {
      char aBuf[32] = {};
      if (fDgsModelMetaValStr(vModelDft, "dflash.block_size", aBuf, sizeof(aBuf)) >= 0) {
        vBlockSize = std::atoi(aBuf);
      }
      if (fDgsModelMetaValStr(vModelDft, "dflash.sample_from_anchor", aBuf, sizeof(aBuf)) >= 0) {
        vSampleFromAnchor = std::strcmp(aBuf, "true") == 0;
      }
      if (fDgsModelMetaValStr(vModelDft, "dflash.attention.causal", aBuf, sizeof(aBuf)) >= 0) {
        vCausalAttn = std::strcmp(aBuf, "true") == 0;
      }
    }

    vSelectorTopK = fDgsModelDflashSelectorTopK(vModelDft);
    vIsDflash2    = vSelectorTopK > 0;
    vMaskTokenId  = fDgsVocabMask(fDgsModelGetVocab(vModelDft));

    if (cIsDspark && this->vParams.vPMin > 0.0f) {
      char       aBuf[16] = {};
      const bool cHasConf = fDgsModelMetaValStr(vModelDft, "dflash.has_confidence_head", aBuf, sizeof(aBuf)) < 0 ||
                            std::strcmp(aBuf, "true") == 0;
      if (!cHasConf) {
        throw std::runtime_error("DSpark draft has no confidence head: please set --spec-draft-p-min 0");
      }
    }

    LOG_INF("%s: adding speculative implementation '%s'\n", __func__, fCommonSpeculativeTypeToStr(type).c_str());
    LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%.2f\n", __func__, this->vParams.vNMax, this->vParams.vNMin,
            this->vParams.vPMin);
    LOG_INF("%s: - block_size=%d, mask_token_id=%d, n_extract=%u, sample_from_anchor=%s\n", __func__, vBlockSize,
            vMaskTokenId, vTargetLayerIdsN, vSampleFromAnchor ? "true" : "false");

    // DFlash input is [id_last, <mask> * (block_size-1)]: in-place denoising yields at most
    // block_size-1 draft tokens, anchor-first DSpark yields a full block_size draft tokens
    const int32_t cNDraftMax = cIsDspark && vSampleFromAnchor ? vBlockSize : vBlockSize - 1;
    if (this->vParams.vNMax > cNDraftMax || this->vParams.vNMin > cNDraftMax) {
      LOG_WRN("%s: requested draft size (n_max=%d, n_min=%d) exceeds the trained block size %d -- clamping to %d\n",
              __func__, this->vParams.vNMax, this->vParams.vNMin, vBlockSize, cNDraftMax);
      this->vParams.vNMax = std::min(this->vParams.vNMax, cNDraftMax);
      this->vParams.vNMin = std::min(this->vParams.vNMin, cNDraftMax);
    }
    this->vNMax = this->vParams.vNMax;

    vBatch       = common_batch(vCtxDft);
    vBatchInject = common_batch(vCtxDft);

    // embd batches on an M-RoPE draft carry 4 position rows per token
    vIsMrope = fDgsModelRopeType(vModelDft) == cDgsRopeTypeMrope;

    lSmpls.resize(pNSeq);
    for (auto & vS : lSmpls) {
      common_params_sampling vSparams;
      vSparams.vNoPerf   = false;
      vSparams.vTopK     = 10;
      vSparams.lSamplers = { cCommonSamplerTypeTopK };
      vS.reset(fCommonSamplerInit(vModelDft, vSparams));
    }

    // offload draft sampling to the backend
    lBackendChains.assign(pNSeq, nullptr);
    if (this->vParams.vBackendSampling && !vIsDflash2) {
      for (dgs_seq_id vSeqId = 0; vSeqId < (dgs_seq_id) pNSeq; ++vSeqId) {
        dgs_sampler * vChain = fDgsSamplerChainInit(fDgsSamplerChainDefaultParams());
        fDgsSamplerChainAdd(vChain, fDgsSamplerInitTopK(10));

        if (!fDgsSetSampler(vCtxDft, vSeqId, vChain)) {
          SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) vSeqId);
          fDgsSamplerFree(vChain);
          vChain = nullptr;
        }
        lBackendChains[vSeqId] = vChain;
      }
    }

    // turn on extraction of the target layers' input embeddings
    for (uint32_t vK = 0; vK < vTargetLayerIdsN; ++vK) {
      fDgsSetEmbeddingsLayerInp(vCtxTgt, (uint32_t) vTargetLayerIds[vK], true);
    }

    // DFlash2 reads its selector lattice from h_nextn and never consumes raw logits.
    fDgsSetEmbeddingsNextn(vCtxDft, true, /*masked*/ !vIsDflash2);
    fDgsSetCausalAttn(vCtxDft, vCausalAttn);  // DFlash needs non-causal attention unless the model says otherwise
  }

  ~common_speculative_impl_draft_dflash() override {
    auto * vCtxDft = this->vParams.vCtxDft;
    for (dgs_seq_id vSeqId = 0; vSeqId < (dgs_seq_id) lBackendChains.size(); ++vSeqId) {
      if (lBackendChains[vSeqId] == nullptr) {
        continue;
      }
      if (vCtxDft) {
        fDgsSetSampler(vCtxDft, vSeqId, nullptr);
      }
      fDgsSamplerFree(lBackendChains[vSeqId]);
    }
    lBackendChains.clear();
  }

  void begin(dgs_seq_id pSeqId, const dgs_tokens & pPrompt) override {
    if (pSeqId < 0 || pSeqId >= (dgs_seq_id) vNSeq) {
      return;
    }

    const int32_t cN = (int32_t) pPrompt.size();
    if (cN <= 0) {
      return;
    }

    const dgs_pos cPosMax = fDgsMemorySeqPosMax(fDgsGetMemory(vParams.vCtxDft), pSeqId);
    if (cPosMax < cN - 1) {
      LOG_WRN(
        "%s: ctx_dft pos_max=%d < N-1=%d - process() did not run on every prefill ubatch. "
        "Drafts may degrade.\n",
        __func__, (int) cPosMax, cN - 1);
    }
  }

  bool mProcess(const common_batch & pBatchIn) override {
    if (pBatchIn.size() <= 0) {
      return true;
    }

    // Target prefill may contain token IDs or multimodal embeddings. Both
    // produce the target-layer features used to seed the draft KV cache, so
    // embeddings are injected too, except the pinned ones skipped below.
    // TODO: revisit after  is merged
    const bool cHasTokens     = pBatchIn.mHasToken();
    const bool cHasEmbeddings = pBatchIn.mHasEmbd();
    if (cHasTokens == cHasEmbeddings) {
      return true;
    }

    const int32_t cNTokens = pBatchIn.size();

    // per-seq inclusive batch range (assumes each seq's tokens are contiguous in the batch)
    std::vector<int32_t> lIBatchBeg(vNSeq, -1);
    std::vector<int32_t> lIBatchEnd(vNSeq, -1);
    for (int32_t vK = 0; vK < cNTokens; ++vK) {
      const dgs_seq_id cSeqId = pBatchIn.lTokens[vK].vSeqId;
      if (cSeqId < 0 || cSeqId >= (dgs_seq_id) vNSeq) {
        continue;
      }
      lIBatchEnd[cSeqId] = vK;
      if (lIBatchBeg[cSeqId] < 0) {
        lIBatchBeg[cSeqId] = vK;
      }
    }

    auto * vCtxTgt = this->vParams.vCtxTgt;
    auto * vCtxDft = this->vParams.vCtxDft;

    const int32_t cNUbatch = (int32_t) fDgsNUbatch(vCtxDft);

    for (dgs_seq_id vSeqId = 0; vSeqId < (dgs_seq_id) vNSeq; ++vSeqId) {
      if (lIBatchBeg[vSeqId] < 0) {
        continue;
      }
      const int32_t cNRows = lIBatchEnd[vSeqId] - lIBatchBeg[vSeqId] + 1;

      // an M-RoPE image pins all its rows to one position, so a windowed draft
      // cache cannot free cells for it - skip it, the draft can jump over the gap
      const bool cPosPinned =
        pBatchIn.lTokens[lIBatchBeg[vSeqId]].aPos[0] == pBatchIn.lTokens[lIBatchEnd[vSeqId]].aPos[0];
      if (cHasEmbeddings && cNRows > 1 && cPosPinned) {
        continue;
      }

      for (int32_t vOffset = 0; vOffset < cNRows; vOffset += cNUbatch) {
        const int32_t cNChunk = std::min(cNUbatch, cNRows - vOffset);

        // gather target features per extract layer; the fused decode encodes and
        // injects them into the K/V cache at the target positions
        lFeaturesBuf.resize((size_t) cNChunk * vNEmbdEnc);
        for (uint32_t vK = 0; vK < vTargetLayerIdsN; ++vK) {
          const float * layer = fDgsGetEmbeddingsLayerInp(vCtxTgt, (uint32_t) vTargetLayerIds[vK]);
          if (!layer) {
            GGML_ABORT("DFlash: target layer %d input not extracted.", vTargetLayerIds[vK]);
          }
          for (int32_t vI = 0; vI < cNChunk; ++vI) {
            float *       vDst = lFeaturesBuf.data() + (size_t) vI * vNEmbdEnc + vK * (size_t) vNEmbdTgt;
            const float * vSrc = layer + (size_t) (lIBatchBeg[vSeqId] + vOffset + vI) * vNEmbdTgt;
            std::memcpy(vDst, vSrc, (size_t) vNEmbdTgt * sizeof(float));
          }
        }

        vBatchInject.clear();
        for (int32_t vI = 0; vI < cNChunk; ++vI) {
          const dgs_pos cP         = pBatchIn.lTokens[lIBatchBeg[vSeqId] + vOffset + vI].aPos[0];
          const dgs_pos aPosArr[4] = { cP, cP, cP, 0 };
          vBatchInject.mAddEmbd({ lFeaturesBuf.data() + (size_t) vI * vNEmbdEnc, 1, (size_t) vNEmbdEnc }, aPosArr,
                                vSeqId, false);
        }
        const int32_t cRc = fDgsProcess(vCtxDft, cDgsProcessTypeDecode, vBatchInject.get());
        if (cRc != 0) {
          LOG_ERR("%s: dgs_process(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n", __func__, cRc, (int) cNChunk,
                  (int) vOffset);
          return false;
        }
      }
    }

    return true;
  }

  void mDraft(common_speculative_draft_params_vec & pDparams) override {
    auto & vCtxDft = vParams.vCtxDft;

    vBatch.clear();

    // build one batch holding every drafting sequence's noise block into a single decode)
    // record where each block starts and its size
    std::vector<int32_t> lIBlockBeg(vNSeq, -1);
    std::vector<int32_t> lNBlock(vNSeq, 0);

    for (dgs_seq_id vSeqId = 0; vSeqId < (dgs_seq_id) vNSeq; ++vSeqId) {
      auto & vDp = pDparams[vSeqId];
      if (!vDp.vDrafting) {
        continue;
      }

      fCommonSamplerReset(lSmpls[vSeqId].get());

      const int32_t n = (int32_t) vDp.vPos0;

      const int32_t cNDraft = vParams.vNMax;

      const int32_t cNBlockTokens = cNDraft + (cIsDspark && vSampleFromAnchor ? 0 : 1);
      lIBlockBeg[vSeqId]          = vBatch.size();
      lNBlock[vSeqId]             = cNBlockTokens;
      for (int32_t vI = 0; vI < cNBlockTokens; ++vI) {
        vBatch.mAdd(vI == 0 ? vDp.vIdLast : vMaskTokenId, n + vI, vSeqId, !vIsDflash2);
      }
    }

    if (vBatch.size() == 0) {
      return;
    }

    // decode all sequence's noise block in a single batch
    int vRet = fDgsProcess(vCtxDft, cDgsProcessTypeDecode, vBatch.get());
    if (vRet != 0) {
      LOG_WRN("%s: dgs_process returned %d\n", __func__, vRet);
      return;
    }

    for (dgs_seq_id vSeqId = 0; vSeqId < (dgs_seq_id) vNSeq; ++vSeqId) {
      if (lIBlockBeg[vSeqId] < 0) {
        continue;
      }
      auto & vDp = pDparams[vSeqId];

      const int32_t cBeg          = lIBlockBeg[vSeqId];
      const int32_t cNBlockTokens = lNBlock[vSeqId];

      auto * vSmpl = lSmpls[vSeqId].get();

      auto & lResult = *vDp.lResult;

      if (vIsDflash2) {
        const float * vLattice = fDgsGetEmbeddingsNextn(vCtxDft);
        GGML_ASSERT(vLattice && "DFlash2 selector produced no lattice");

        int32_t vPredecessor = 0;
        for (int32_t vI = 1; vI < cNBlockTokens; ++vI) {
          const float * vRow    = vLattice + (size_t) (cBeg + vI) * vNEmbdDec;
          const float * vScores = vRow + vSelectorTopK + (size_t) vPredecessor * vSelectorTopK;

          vPredecessor = (int32_t) std::distance(vScores, std::max_element(vScores, vScores + vSelectorTopK));
          if (vParams.vPMin > 0.0f) {
            // softmax(scores) at the argmax, i.e. 1 / sum(exp(s_k - s_max))
            float vSum = 0.0f;
            for (int32_t vK = 0; vK < vSelectorTopK; ++vK) {
              vSum += std::exp(vScores[vK] - vScores[vPredecessor]);
            }
            if (1.0f / vSum < vParams.vPMin) {
              break;
            }
          }
          lResult.push_back((dgs_token) vRow[vPredecessor]);
        }

        if (lResult.size() < (size_t) vParams.vNMin) {
          lResult.clear();
        }
        continue;
      }

      if (cIsDspark) {
        // DSpark: read from the first draft slot, truncate below the confidence threshold
        const float * vConf      = vParams.vPMin > 0.0f ? fDgsGetEmbeddingsNextn(vCtxDft) : nullptr;
        // bonus-anchor drafts read the mask positions only, like DFlash
        const int32_t cIDraftBeg = vSampleFromAnchor ? 0 : 1;
        for (int32_t vI = cIDraftBeg; vI < cNBlockTokens; ++vI) {
          const int32_t cIdx = cBeg + vI;

          if (vConf && vConf[(size_t) cIdx * vNEmbdDec] < vParams.vPMin) {
            break;
          }

          fCommonSamplerSample(vSmpl, vCtxDft, cIdx, true);

          const auto * vCurP = fCommonSamplerGetCandidates(vSmpl, true);

          for (int vK = 0; vK < std::min(3, (int) vCurP->size); ++vK) {
            LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n", vSeqId, vK, vI,
                    vCurP->data[vK].vId, vCurP->data[vK].vP, fCommonTokenToPiece(vCtxDft, vCurP->data[vK].vId).c_str());
          }

          const dgs_token cId = vCurP->data[0].vId;

          fCommonSamplerAccept(vSmpl, cId, true);

          lResult.push_back(cId);
        }
      } else {
        // greedily read the predicted block at this sequence's noise positions 1..n_block_tokens-1
        for (int32_t vI = 1; vI < cNBlockTokens; ++vI) {
          fCommonSamplerSample(vSmpl, vCtxDft, cBeg + vI, true);

          const auto * vCurP = fCommonSamplerGetCandidates(vSmpl, true);

          for (int vK = 0; vK < std::min(3, (int) vCurP->size); ++vK) {
            LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n", vSeqId, vK, vI - 1,
                    vCurP->data[vK].vId, vCurP->data[vK].vP, fCommonTokenToPiece(vCtxDft, vCurP->data[vK].vId).c_str());
          }

          const dgs_token cId = vCurP->data[0].vId;

          if (vCurP->data[0].vP < vParams.vPMin) {
            break;
          }

          fCommonSamplerAccept(vSmpl, cId, true);

          lResult.push_back(cId);
        }
      }

      if (lResult.size() < (size_t) vParams.vNMin) {
        lResult.clear();
      }
    }
  }

  void accept(dgs_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
    // noop
  }
};

struct common_speculative_impl_draft_mtp : public common_speculative_impl {
  common_params_speculative_draft vParams;  // reuses the draft-model params slot (ctx_tgt/ctx_dft)

  common_batch vBatch;

  std::vector<common_sampler_ptr> lSmpls;

  std::vector<common_params_sampling> lSmplsCfg;

  // backend sampler chain per seq, attached to ctx_dft
  std::vector<dgs_sampler *> lBackendChains;

  int32_t vNEmbd = 0;

  // One MTP draft driver, three modes (set once in the ctor):
  //   is_mem_shared (gemma4): shares the target KV, runs all heads in one graph.
  //   chain_heads (step35): n_mtp_layers trained heads, one per draft step.
  //   neither (qwen35 / qwen35moe): a single trained MTP head.
  int32_t vNMtpLayers  = 1;
  bool    vIsMemShared = false;  // gemma4
  bool    vChainHeads  = false;  // derived in the ctor: n_mtp_layers > 1 && !is_mem_shared

  // Per-sequence cross-batch carryover: pair (h_p, x_{p+1}) at MTP pos p+1.
  // The last h-row of one process() call needs the first token of the NEXT
  // call to pair with, so it's stashed here until that next call fires.
  std::vector<std::vector<float>> lPendingH;  // [n_seq][n_embd]

  std::vector<int32_t> lIBatchBeg;
  std::vector<int32_t> lIBatchEnd;

  // Hidden rows from the most recent target verification batch, grouped by seq.
  // Row 0 corresponds to the sampled token, row N to the Nth accepted draft token.
  std::vector<std::vector<float>> lVerifyH;
  std::vector<int32_t>            lVerifyHRows;

  std::vector<int>                lILast;
  std::vector<std::vector<float>> lChainH;

  common_speculative_impl_draft_mtp(const common_params_speculative & pParams, uint32_t pNSeq) :
    common_speculative_impl(cCommonSpeculativeTypeDraftMtp, pNSeq, pParams.vDraft.vNMax),
    vParams(pParams.vDraft) {
    auto * vCtxTgt = this->vParams.vCtxTgt;
    auto * vCtxDft = this->vParams.vCtxDft;
    GGML_ASSERT(vCtxTgt && vCtxDft && "MTP requires ctx_tgt and ctx_dft to be set");

    vNEmbd = fDgsModelNEmbdOut(fDgsGetModel(vCtxDft));
    GGML_ASSERT(vNEmbd == fDgsModelNEmbdOut(fDgsGetModel(vCtxTgt)) &&
                "MTP input row width must match the target h_nextn width");
    vNMtpLayers = std::max(1, (int) fDgsModelNLayerNextn(fDgsGetModel(vCtxDft)));

    SPC_TRC("%s", "adding speculative implementation 'draft-mtp'\n");
    SPC_TRC("- n_max=%d, n_min=%d, p_min=%.2f, n_embd=%d, backend_sampling=%d\n", this->vParams.vNMax,
            this->vParams.vNMin, this->vParams.vPMin, vNEmbd, (int) this->vParams.vBackendSampling);
    SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
            this->vParams.vNGpuLayers, ggml_type_name(this->vParams.vCacheTypeK),
            ggml_type_name(this->vParams.vCacheTypeV), vCtxTgt ? "yes" : "no", vCtxDft ? "yes" : "no",
            fCommonSpeculativeGetDevicesStr(this->vParams.lDevices).c_str());

    vBatch = common_batch(vCtxDft);

    lSmpls.resize(pNSeq);
    for (auto & vS : lSmpls) {
      common_params_sampling vSparams;
      vSparams.vNoPerf   = false;
      vSparams.vTopK     = 10;
      vSparams.lSamplers = { cCommonSamplerTypeTopK };
      vS.reset(fCommonSamplerInit(fDgsGetModel(vCtxDft), vSparams));
    }

    // offload draft sampling to the backend
    lBackendChains.assign(pNSeq, nullptr);
    if (this->vParams.vBackendSampling) {
      for (dgs_seq_id vSeqId = 0; vSeqId < (dgs_seq_id) pNSeq; ++vSeqId) {
        dgs_sampler * vChain = fDgsSamplerChainInit(fDgsSamplerChainDefaultParams());
        fDgsSamplerChainAdd(vChain, fDgsSamplerInitTopK(10));

        if (!fDgsSetSampler(vCtxDft, vSeqId, vChain)) {
          SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) vSeqId);
          fDgsSamplerFree(vChain);
          vChain = nullptr;
        }
        lBackendChains[vSeqId] = vChain;
      }
    }

    fDgsSetEmbeddingsNextn(vCtxTgt, true, /*masked*/ false);
    fDgsSetEmbeddingsNextn(vCtxDft, true, /*masked*/ true);

    vIsMemShared = fDgsGetCtxOther(vCtxDft) == vCtxTgt;
    vChainHeads  = vNMtpLayers > 1 && !vIsMemShared;

    if (vChainHeads) {
      this->vParams.vNMax = std::min(this->vParams.vNMax, vNMtpLayers);

      lChainH.assign(pNSeq, {});
      for (auto & lC : lChainH) {
        lC.reserve((size_t) (this->vParams.vNMax + 1) * vNEmbd);
      }
    }
    this->vNMax = this->vParams.vNMax;

    lPendingH.assign(pNSeq, std::vector<float>(vNEmbd, 0.0f));

    lILast.assign(pNSeq, -1);
    lIBatchBeg.assign(pNSeq, -1);
    lIBatchEnd.assign(pNSeq, -1);

    lVerifyH.assign(pNSeq, {});
    lVerifyHRows.assign(pNSeq, 0);
  }

  ~common_speculative_impl_draft_mtp() override {
    auto * vCtxDft = this->vParams.vCtxDft;
    for (dgs_seq_id vSeqId = 0; vSeqId < (dgs_seq_id) lBackendChains.size(); ++vSeqId) {
      if (lBackendChains[vSeqId] == nullptr) {
        continue;
      }
      if (vCtxDft) {
        fDgsSetSampler(vCtxDft, vSeqId, nullptr);
      }
      fDgsSamplerFree(lBackendChains[vSeqId]);
    }
    lBackendChains.clear();
  }

  void begin(dgs_seq_id pSeqId, const dgs_tokens & pPrompt) override {
    // reset here rather than per round, or two identical requests differ
    fCommonSamplerReset(lSmpls[pSeqId].get());

    const int32_t cN = (int32_t) pPrompt.size();
    if (cN <= 0) {
      return;
    }

    auto *        vCtxDft = this->vParams.vCtxDft;
    const dgs_pos cPosMax = fDgsMemorySeqPosMax(fDgsGetMemory(vCtxDft), pSeqId);

    if (cPosMax < cN - 1 && !vIsMemShared) {
      SPC_WRN(
        "ctx_dft pos_max=%d < N-1=%d - "
        "process() hook may not have run on every prefill ubatch "
        "(need_embd / output flag on every prompt position?). "
        "Drafts may degrade.\n",
        (int) cPosMax, cN - 1);
    }
  }

  bool mProcess(const common_batch & pBatchIn) override {
    if (pBatchIn.size() <= 0) {
      return true;
    }

    if (!pBatchIn.mHasToken() && !pBatchIn.mHasEmbd()) {
      return true;
    }

    const int32_t cNTokens = pBatchIn.size();

    // remember the first and last batch index for each sequence
    std::fill(lIBatchBeg.begin(), lIBatchBeg.end(), -1);
    std::fill(lIBatchEnd.begin(), lIBatchEnd.end(), -1);

    for (int vK = 0; vK < cNTokens; ++vK) {
      for (dgs_seq_id vSeqId = 0; vSeqId < (dgs_seq_id) vNSeq; ++vSeqId) {
        if (pBatchIn.lTokens[vK].vSeqId == vSeqId) {
          lIBatchEnd[vSeqId] = vK;
          if (lIBatchBeg[vSeqId] < 0) {
            lIBatchBeg[vSeqId] = vK;
          }
        }
      }
    }

    auto * vCtxTgt = this->vParams.vCtxTgt;
    auto * vCtxDft = this->vParams.vCtxDft;

    const size_t cRowBytes = (size_t) vNEmbd * sizeof(float);

    // if kv is shared with target (e.g Gemma4), then we can skip this catch-up decode
    if (!vIsMemShared) {
      vBatch.clear();

      // pair each token with the tgt embedding shifted right by one position, and
      // the first token of each sequence with the pending embedding from a previous run
      // assumes that the tokens in the batch are sequential for each sequence
      // i.e. we cannot have seq_id like this: [0, 0, 0, 1, 1, 0, 1, 1]
      //                                                       ^--- this is a problem
      // TODO:this is generally true, but would be nice to assert it
      const float * vHTgt = fDgsGetEmbeddingsNextn(vCtxTgt);

      for (int vK = 0; vK < cNTokens; ++vK) {
        const auto & vT = pBatchIn.lTokens[vK];

        const dgs_seq_id cSeqId = vT.vSeqId;

        // vision tokens carry an embedding instead of an id
        const int32_t cIdx = vT.vId != DGS_TOKEN_NULL ? vBatch.mAdd(vT.vId, vT.aPos[0], cSeqId, false) :
                                                        vBatch.mAddEmbd(vT.vEmbd, vT.aPos.data(), cSeqId, false);

        const float * vHRow = vK == lIBatchBeg[cSeqId] ? lPendingH[cSeqId].data() : vHTgt + (size_t) (vK - 1) * vNEmbd;

        vBatch.mSetEmbdState(cIdx, { vHRow, 1, (size_t) vNEmbd });
      }

      auto * vMemDft = fDgsGetMemory(vCtxDft);

      bool vOk = true;
      for (int vHead = 0; vHead < vNMtpLayers; ++vHead) {
        if (vChainHeads) {
          for (dgs_seq_id vSeqId = 0; vSeqId < (dgs_seq_id) vNSeq; ++vSeqId) {
            if (lIBatchBeg[vSeqId] < 0) {
              continue;
            }
            fDgsMemorySeqRm(vMemDft, vSeqId, pBatchIn.lTokens[lIBatchBeg[vSeqId]].aPos[0], -1);
          }
          fDgsSetNextnLayerOffset(vCtxDft, vHead);
        }

        const int32_t cRc = fDgsProcess(vCtxDft, cDgsProcessTypeDecode, vBatch.get());
        if (cRc != 0) {
          SPC_ERR("dgs_process(ctx_dft) head=%d failed rc=%d (pos=%d)\n", vHead, (int) cRc,
                  (int) pBatchIn.lTokens[0].aPos[0]);
          vOk = false;
          break;
        }
      }

      if (vChainHeads) {
        fDgsSetNextnLayerOffset(vCtxDft, 0);  // restore default for non-draft decodes
      }
      if (!vOk) {
        return false;
      }
    }

    for (dgs_seq_id vSeqId = 0; vSeqId < (dgs_seq_id) vNSeq; ++vSeqId) {
      if (lIBatchEnd[vSeqId] < 0) {
        continue;
      }

      const int32_t cNRows = lIBatchEnd[vSeqId] - lIBatchBeg[vSeqId] + 1;
      lVerifyHRows[vSeqId] = cNRows;
      lVerifyH[vSeqId].resize((size_t) cNRows * vNEmbd);

      for (int32_t vI = 0; vI < cNRows; ++vI) {
        const float * vH = fDgsGetEmbeddingsNextnIth(vCtxTgt, lIBatchBeg[vSeqId] + vI);
        std::memcpy(lVerifyH[vSeqId].data() + (size_t) vI * vNEmbd, vH, cRowBytes);
      }

      std::memcpy(lPendingH[vSeqId].data(), lVerifyH[vSeqId].data() + (size_t) (cNRows - 1) * vNEmbd, cRowBytes);
    }

    return true;
  }

  void mDraft(common_speculative_draft_params_vec & pDparams) override {
    auto & vCtxDft = vParams.vCtxDft;

    vBatch.clear();

    // keep track of which sequences are still drafting
    int               vNDrafting = 0;
    std::vector<bool> lDrafting(vNSeq);

    for (dgs_seq_id vSeqId = 0; vSeqId < (dgs_seq_id) vNSeq; ++vSeqId) {
      auto & vDp = pDparams[vSeqId];

      if (!vDp.vDrafting) {
        continue;
      }

      vNDrafting++;
      lDrafting[vSeqId] = true;
      // greedy drafting leaves no candidates behind, so the verifier falls back to sample-and-match
      if (!vParams.vProbabilistic) {
        vDp.lResultQ = nullptr;
      }

      // result_q is only set when the caller wants rejection, so it also gates the retune
      if (vDp.lResultQ) {
        fSpecRetune(lSmpls, lSmplsCfg, fDgsGetModel(vCtxDft), vSeqId, vDp.vTemp, vDp.vSeed);
      }

      // a reset reseeds the chain, which breaks probabilistic drafting
      if (!vDp.lResultQ) {
        fCommonSamplerReset(lSmpls[vSeqId].get());
      }

      const int32_t cIdx = vBatch.mAdd(vDp.vIdLast, vDp.vPos0, vSeqId, true);
      vBatch.mSetEmbdState(cIdx, { lPendingH[vSeqId].data(), 1, (size_t) vNEmbd });

      lILast[vSeqId] = cIdx;

      if (vChainHeads) {
        lChainH[vSeqId].assign(lPendingH[vSeqId].begin(), lPendingH[vSeqId].end());
      }
    }

    int vI = 0;

    while (vNDrafting > 0) {
      // each step decodes under a different head, i.e. a different decoder layer, and
      // KV is per layer. process() filled this layer's KV only for positions < pos0
      // (prompt + accepted prefix) — nothing in the draft region yet. so reset the
      // draft region (the seq_rm lower bound is pos0, leaving the prompt KV intact)
      // and select head i so it rebuilds its own layer's KV there; decoding just the
      // latest token would leave its attention reading cells only another head wrote.
      if (vChainHeads) {
        auto * vMemDft = fDgsGetMemory(vCtxDft);
        for (dgs_seq_id vSeqId = 0; vSeqId < (dgs_seq_id) vNSeq; ++vSeqId) {
          if (lDrafting[vSeqId]) {
            fDgsMemorySeqRm(vMemDft, vSeqId, pDparams[vSeqId].vPos0, -1);
          }
        }
        fDgsSetNextnLayerOffset(vCtxDft, vI);
      }

      int vRet = fDgsProcess(vCtxDft, cDgsProcessTypeDecode, vBatch.get());
      if (vRet != 0) {
        SPC_ERR("dgs_process[%d] returned %d\n", vI, vRet);
        break;
      }

      // rebuild the batch for the next step: the growing-KV paths re-add only the
      // new token (the KV already holds the prefix), while chained heads re-add the
      // whole prefix at the next head. dropped sequences are simply not re-added.
      vBatch.clear();

      for (dgs_seq_id vSeqId = 0; vSeqId < (dgs_seq_id) vNSeq; ++vSeqId) {
        if (!lDrafting[vSeqId]) {
          continue;
        }

        auto * vSmpl = lSmpls[vSeqId].get();

        const dgs_token cIdSampled = fCommonSamplerSample(vSmpl, vCtxDft, lILast[vSeqId], true);
        const float *   vHRow      = fDgsGetEmbeddingsNextnIth(vCtxDft, lILast[vSeqId]);

        const auto * vCurP = fCommonSamplerGetCandidates(vSmpl, true);

        for (int vK = 0; vK < std::min(3, (int) vCurP->size); ++vK) {
          SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n", vSeqId, vK, vI, vCurP->data[vK].vId,
                  vCurP->data[vK].vP, fCommonTokenToPiece(vCtxDft, vCurP->data[vK].vId).c_str());
        }

        // add drafted token for each sequence
        const dgs_token cId = pDparams.at(vSeqId).lResultQ ? cIdSampled : vCurP->data[0].vId;

        // only collect very high-confidence draft tokens
        if (vCurP->data[0].vP < vParams.vPMin) {
          lDrafting[vSeqId] = false;
          vNDrafting--;

          continue;
        }

        fCommonSamplerAccept(vSmpl, cId, true);

        auto & vDp     = pDparams.at(vSeqId);
        auto & lResult = *vDp.lResult;

        lResult.push_back(cId);

        if (vDp.lResultQ) {
          vDp.lResultQ->emplace_back(vCurP->data, vCurP->data + vCurP->size);
        }

        if (vParams.vNMax <= (int) lResult.size()) {
          lDrafting[vSeqId] = false;
          vNDrafting--;
          continue;
        }

        if (vChainHeads) {
          lChainH[vSeqId].insert(lChainH[vSeqId].end(), vHRow, vHRow + vNEmbd);

          const int cNRows = (int) lResult.size() + 1;  // id_last + tokens drafted so far
          for (int vT = 0; vT < cNRows; ++vT) {
            const dgs_token cTok = (vT == 0) ? vDp.vIdLast : lResult[vT - 1];
            const int32_t   cIdx = vBatch.mAdd(cTok, vDp.vPos0 + vT, vSeqId, vT == cNRows - 1);
            vBatch.mSetEmbdState(cIdx, { lChainH[vSeqId].data() + (size_t) vT * vNEmbd, 1, (size_t) vNEmbd });
            lILast[vSeqId] = cIdx;
          }
        } else if (vIsMemShared) {
          // note: with shared memory (e.g. Gemma4 assistants) we use the same position for all draft tokens
          // ref: https://github.com/huggingface/transformers/blob/effde20942e3f82a1b97449f60b3a48c5ff96145/docs/source/en/model_doc/gemma4_assistant.md?plain=1#L36-L37
          const int32_t cIdx = vBatch.mAdd(cId, vDp.vPos0, vSeqId, true);
          vBatch.mSetEmbdState(cIdx, { vHRow, 1, (size_t) vNEmbd });
          lILast[vSeqId] = cIdx;
        } else {
          const int32_t cIdx = vBatch.mAdd(cId, vDp.vPos0 + vI + 1, vSeqId, true);
          vBatch.mSetEmbdState(cIdx, { vHRow, 1, (size_t) vNEmbd });
          lILast[vSeqId] = cIdx;
        }
      }

      if (vBatch.size() == 0) {
        break;
      }

      ++vI;
    }

    if (vChainHeads) {
      fDgsSetNextnLayerOffset(vCtxDft, 0);  // restore default for non-draft decodes
    }

    for (dgs_seq_id vSeqId = 0; vSeqId < (dgs_seq_id) vNSeq; ++vSeqId) {
      auto & vDp = pDparams[vSeqId];
      if (!vDp.vDrafting) {
        continue;
      }

      if (vDp.lResult->size() < (size_t) vParams.vNMin) {
        vDp.lResult->clear();
      }
    }
  }

  void accept(dgs_seq_id pSeqId, uint16_t pNAccepted, bool /*is_other*/) override {
    if (pSeqId < 0 || pSeqId >= (dgs_seq_id) vNSeq) {
      return;
    }

    const int32_t cNRows = lVerifyHRows[pSeqId];
    if (cNRows <= 0) {
      return;
    }

    const int32_t cIH       = std::min<int32_t>(pNAccepted, cNRows - 1);
    const size_t  cRowBytes = (size_t) vNEmbd * sizeof(float);
    std::memcpy(lPendingH[pSeqId].data(), lVerifyH[pSeqId].data() + (size_t) cIH * vNEmbd, cRowBytes);
  }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_impl_ngram_simple : public common_speculative_impl {
  common_params_speculative_ngram_map vParams;

  // shared across all sequences
  common_ngram_simple_config vConfig;

  common_speculative_impl_ngram_simple(const common_params_speculative & pParams,
                                       uint32_t                          pNSeq,
                                       common_ngram_simple_config        pConfig) :
    common_speculative_impl(cCommonSpeculativeTypeNgramSimple, pNSeq, pParams.vNgramSimple.vSizeM),
    vParams(pParams.vNgramSimple),
    vConfig(pConfig) {
    SPC_TRC("%s", "adding speculative implementation 'ngram-simple'\n");
    SPC_TRC("- size_n=%d, size_m=%d, min_hits=%d\n", this->vParams.vSizeN, this->vParams.vSizeM,
            this->vParams.vMinHits);
  }

  void begin(dgs_seq_id /*seq_id*/, const dgs_tokens & /*prompt*/) override {
    // noop
  }

  bool mProcess(const common_batch & /*batch*/) override {
    // TODO: implement
    return true;
  }

  void mDraft(common_speculative_draft_params_vec & pDparams) override {
    assert(dparams.size() == n_seq);

    for (dgs_seq_id vSeqId = 0; vSeqId < (dgs_seq_id) vNSeq; ++vSeqId) {
      auto & vDp = pDparams[vSeqId];
      if (!vDp.vDrafting) {
        continue;
      }

      *vDp.lResult = fCommonNgramSimpleDraft(vConfig, *vDp.lPrompt, vDp.vIdLast);
    }
  }

  void accept(dgs_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
    // noop
  }
};

struct common_speculative_impl_ngram_map_k : public common_speculative_impl {
  // n_seq configs
  std::vector<common_ngram_map> lConfig;

  common_speculative_impl_ngram_map_k(const common_ngram_map & pConfig, uint32_t pNSeq) :
    common_speculative_impl(pConfig.vKeyOnly ? cCommonSpeculativeTypeNgramMapK : cCommonSpeculativeTypeNgramMapK4v,
                            pNSeq,
                            pConfig.vSizeValue) {
    for (uint32_t vI = 0; vI < pNSeq; vI++) {
      this->lConfig.push_back(pConfig);
    }

    SPC_TRC("adding speculative implementation '%s'\n", fCommonSpeculativeTypeToStr(this->type).c_str());
    SPC_TRC("- size_key=%d, size_value=%d, key_only=%d, min_hits=%d\n", pConfig.vSizeKey, pConfig.vSizeValue,
            pConfig.vKeyOnly, pConfig.vMinHits);
  }

  void begin(dgs_seq_id pSeqId, const dgs_tokens & pPrompt) override {
    GGML_ASSERT(pSeqId < (dgs_seq_id) vNSeq);

    fCommonNgramMapBegin(lConfig[pSeqId], pPrompt);
  }

  bool mProcess(const common_batch & /*batch*/) override {
    // TODO: implement
    return true;
  }

  void mDraft(common_speculative_draft_params_vec & pDparams) override {
    assert(dparams.size() == n_seq);

    for (dgs_seq_id vSeqId = 0; vSeqId < (dgs_seq_id) vNSeq; ++vSeqId) {
      auto & vDp = pDparams[vSeqId];
      if (!vDp.vDrafting) {
        continue;
      }

      fCommonNgramMapDraft(lConfig[vSeqId], *vDp.lPrompt, vDp.vIdLast, *vDp.lResult);
    }
  }

  void accept(dgs_seq_id pSeqId, uint16_t pNAccepted, bool pIsOther) override {
    GGML_ASSERT((pSeqId < (dgs_seq_id) lConfig.size()));

    if (pIsOther) {
      return;
    }

    fCommonNgramMapAccept(lConfig[pSeqId], pNAccepted);
  }
};

struct common_speculative_impl_ngram_mod : public common_speculative_impl {
  common_params_speculative_ngram_mod vParams;

  // shared across all sequences
  common_ngram_mod vMod;

  // enable trace logging if DGS_TRACE is set
  const bool cVerbose;

  struct seq_info {
    // the last position in the prompt that was added to the ngram container
    size_t vILast = 0;

    // length of the last drafted n-gram (number of tokens returned by draft)
    size_t vNDraftLast = 0;

    // consecutive accept rounds with low acceptance fraction (< 0.5)
    int vNLow = 0;
  };

  std::vector<seq_info> lSinfos;

  common_speculative_impl_ngram_mod(const common_params_speculative & pParams, uint32_t pNSeq) :
    common_speculative_impl(cCommonSpeculativeTypeNgramMod, pNSeq, pParams.vNgramMod.vNMax),
    vParams(pParams.vNgramMod),
    vMod(pParams.vNgramMod.vNMatch, 4 * 1024 * 1024),
    cVerbose(std::getenv("DGS_TRACE") != nullptr) {
    static_assert(sizeof(dgs_token) == sizeof(common_ngram_mod::entry_t));

    SPC_TRC("%s", "adding speculative implementation 'ngram-mod'\n");
    SPC_TRC("- n_match=%d, n_max=%d, n_min=%d\n", this->vParams.vNMatch, this->vParams.vNMax, this->vParams.vNMin);
    SPC_TRC("- mod size=%zu (%.3f MB)\n", vMod.size(), (float) (vMod.mSizeBytes()) / 1024 / 1024);

    if (this->vParams.vNMatch < 16) {
      SPC_WRN(
        "ngram_mod n_match=%d is too small - poor quality is possible, "
        "see: doc/MANUAL.md\n",
        this->vParams.vNMatch);
    }

    lSinfos.resize(pNSeq);
  }

  void begin(dgs_seq_id pSeqId, const dgs_tokens & pPrompt) override {
    auto & vSinfo = lSinfos[pSeqId];

    vSinfo.vILast      = 0;
    vSinfo.vNDraftLast = 0;

    const size_t n = vMod.mGetN();
    if (pPrompt.size() < n) {
      return;
    }

    for (size_t vI = 0; vI < pPrompt.size() - n; ++vI) {
      vMod.mAdd(pPrompt.data() + vI);
    }

    vSinfo.vILast = pPrompt.size() - n;

    const double cF = (double) vMod.mGetUsed() / (double) vMod.size();
    SPC_TRC("ngram_mod occupancy = %zu/%zu (%.2f)\n", vMod.mGetUsed(), vMod.size(), cF);

    constexpr double cFThold = 0.25;
    if (cF > cFThold) {
      SPC_WRN("ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", cF, cFThold);

      vMod.reset();
    }
  }

  void mDraftOne(dgs_seq_id pSeqId, common_speculative_draft_params & pDparams) {
    auto & vSinfo  = lSinfos[pSeqId];
    auto & lResult = *pDparams.lResult;

    const auto & lPrompt = *pDparams.lPrompt;

    vSinfo.vNDraftLast = 0;

    const size_t cCurLen = lPrompt.size();
    if (cCurLen < vMod.mGetN()) {
      return;
    }

    const size_t n = vMod.mGetN();

    // add new ngrams in chunks
    if (vSinfo.vILast + 32 < cCurLen) {
      for (size_t vI = vSinfo.vILast; vI < cCurLen - n; ++vI) {
        vMod.mAdd(lPrompt.data() + vI);
      }

      vSinfo.vILast = cCurLen - n;
    }

    lResult.resize(n + vParams.vNMax);
    for (size_t vI = 0; vI < n - 1; ++vI) {
      lResult[vI] = lPrompt.at(cCurLen - n + 1 + vI);
    }
    lResult[n - 1] = pDparams.vIdLast;

    for (int vI = 0; vI < vParams.vNMax; ++vI) {
      const dgs_token token = vMod.get(lResult.data() + vI);
      if (token == common_ngram_mod::cEmpty) {
        if (vI < vParams.vNMin) {
          lResult.clear();
          return;
        }

        lResult.resize(n + vI);
        break;
      }
      lResult[n + vI] = token;
    }

    // only return the m tokens that were drafted
    for (size_t vI = 0; n + vI < lResult.size(); ++vI) {
      lResult[vI] = lResult[n + vI];
    }
    lResult.resize(lResult.size() - n);

    // store length of drafted n-gram for later acceptance analysis
    vSinfo.vNDraftLast = lResult.size();
  }

  bool mProcess(const common_batch & /*batch*/) override {
    // TODO: implement
    return true;
  }

  void mDraft(common_speculative_draft_params_vec & pDparams) override {
    assert(dparams.size() == n_seq);

    for (dgs_seq_id vSeqId = 0; vSeqId < (dgs_seq_id) vNSeq; ++vSeqId) {
      auto & vDp = pDparams[vSeqId];
      if (!vDp.vDrafting) {
        continue;
      }

      mDraftOne(vSeqId, vDp);
    }
  }

  void accept(dgs_seq_id pSeqId, uint16_t pNAccepted, bool pIsOther) override {
    if (pIsOther) {
      return;
    }

    auto & vSinfo = lSinfos[pSeqId];

    // compute acceptance fraction if we have a recorded draft length
    if (vSinfo.vNDraftLast > 0) {
      const double cFAcc = (double) pNAccepted / (double) vSinfo.vNDraftLast;
      if (cFAcc < 0.25) {
        vSinfo.vNLow++;
        if (vSinfo.vNLow >= 5) {
          if (cVerbose) {
            SPC_TRC("low acceptance streak (%d) - resetting ngram_mod\n", vSinfo.vNLow);
          }

          vMod.reset();
          vSinfo.vNLow  = 0;
          vSinfo.vILast = 0;
        }
      } else {
        vSinfo.vNLow = 0;
      }
    }
  }
};

struct common_speculative_impl_ngram_cache : public common_speculative_impl {
  common_params_speculative_ngram_cache vParams;

  uint16_t vNDraft;

  bool vSaveDynamic;
  bool vSaveStatic;

  struct seq_info {
    size_t vCacheSize = 0;  // number of tokens in n-gram cache

    common_ngram_cache dNgramCacheContext;
    common_ngram_cache dNgramCacheDynamic;
    common_ngram_cache dNgramCacheStatic;
  };

  std::vector<seq_info> lSinfos;

  common_speculative_impl_ngram_cache(const common_params_speculative & pParams,
                                      uint32_t                          pNSeq,
                                      uint16_t                          pNDraft,
                                      const std::string &               pPathStatic,
                                      const std::string &               pPathDynamic,
                                      bool                              pSaveDynamic,
                                      bool                              pSaveStatic) :
    common_speculative_impl(cCommonSpeculativeTypeNgramCache, pNSeq, pNDraft),
    vParams(pParams.vNgramCache),
    vNDraft(pNDraft),
    vSaveDynamic(pSaveDynamic),
    vSaveStatic(pSaveStatic) {
    SPC_TRC("%s", "adding speculative implementation 'ngram-cache'\n");
    SPC_TRC("- n_draft=%d, cache_static=%s, cache_dynamic=%s\n", pNDraft,
            pPathStatic.empty() ? "none" : pPathStatic.c_str(), pPathDynamic.empty() ? "none" : pPathDynamic.c_str());

    lSinfos.resize(pNSeq);

    if (!pPathStatic.empty()) {
      try {
        auto dNgramCacheStatic = fCommonNgramCacheLoad(pPathStatic);

        for (auto & vSinfo : lSinfos) {
          vSinfo.dNgramCacheStatic = dNgramCacheStatic;
        }
      } catch (...) {
        SPC_ERR("failed to open static lookup cache: %s", pPathStatic.c_str());
        GGML_ABORT("Couldn't read static lookup cache");
      }
    }

    if (!pPathDynamic.empty()) {
      try {
        auto dNgramCacheDynamic = fCommonNgramCacheLoad(pPathDynamic);

        for (auto & vSinfo : lSinfos) {
          vSinfo.dNgramCacheDynamic = dNgramCacheDynamic;
        }
      } catch (...) {
        SPC_ERR("failed to open dynamic lookup cache: %s", pPathDynamic.c_str());
        GGML_ABORT("Couldn't read dynamic lookup cache");
      }
    }
  }

  void begin(dgs_seq_id /*seq_id*/, const dgs_tokens & /*prompt*/) override {
    // noop
  }

  void mDraftOne(dgs_seq_id pSeqId, common_speculative_draft_params & pDparams) {
    auto & vSinfo  = lSinfos[pSeqId];
    auto & lResult = *pDparams.lResult;

    const auto & lPrompt = *pDparams.lPrompt;

    if (vSinfo.vCacheSize < lPrompt.size() + 1) {
      dgs_tokens lTokensNew;
      lTokensNew.reserve(lPrompt.size() + 1 - vSinfo.vCacheSize);
      for (size_t vJ = vSinfo.vCacheSize; vJ < lPrompt.size(); ++vJ) {
        lTokensNew.push_back(lPrompt[vJ]);
      }
      lTokensNew.push_back(pDparams.vIdLast);  // add the last token

      // Update context ngram cache with new dparams.prompt:
      fCommonNgramCacheUpdate(vSinfo.dNgramCacheContext, DGS_NGRAM_MIN, DGS_NGRAM_MAX, lTokensNew, lTokensNew.size(),
                              false);
      vSinfo.vCacheSize = lPrompt.size() + 1;
    }

    dgs_tokens lInp;
    lInp.reserve(lPrompt.size() + 1);
    for (size_t vJ = 0; vJ < lPrompt.size(); ++vJ) {
      lInp.push_back(lPrompt[vJ]);
    }
    lInp.push_back(pDparams.vIdLast);

    lResult.push_back(pDparams.vIdLast);

    fCommonNgramCacheDraft(lInp, lResult, vNDraft, DGS_NGRAM_MIN, DGS_NGRAM_MAX, vSinfo.dNgramCacheContext,
                           vSinfo.dNgramCacheDynamic, vSinfo.dNgramCacheStatic);

    if (lResult.size() > 0) {
      // delete first token in result (which is the id_last token)
      lResult.erase(lResult.begin());
    }
  }

  bool mProcess(const common_batch & /*batch*/) override {
    // TODO: implement
    return true;
  }

  void mDraft(common_speculative_draft_params_vec & pDparams) override {
    assert(dparams.size() == n_seq);

    for (dgs_seq_id vSeqId = 0; vSeqId < (dgs_seq_id) vNSeq; ++vSeqId) {
      auto & vDp = pDparams[vSeqId];
      if (!vDp.vDrafting) {
        continue;
      }

      mDraftOne(vSeqId, vDp);
    }
  }

  void accept(dgs_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
    // noop
  }
};

struct common_speculative {
  common_speculative_draft_params_vec lDparams;

  // list of implementations to use and their states
  std::vector<std::unique_ptr<common_speculative_impl>> lImpls;

  // which implementaion was used for a given seq_id
  std::vector<common_speculative_impl *> lImplLast;

  std::vector<double> lSynthProbs;
};

static common_ngram_map fGetCommonNgramMap(common_speculative_type                     type,
                                           const common_params_speculative_ngram_map & pConfig) {
  uint16_t vSizeKey   = pConfig.vSizeN;
  uint16_t vSizeValue = pConfig.vSizeM;
  bool     vKeyOnly   = type == cCommonSpeculativeTypeNgramMapK;
  uint16_t vMinHits   = pConfig.vMinHits;

  return common_ngram_map(vSizeKey, vSizeValue, vKeyOnly, vMinHits);
}

static common_speculative_impl_ngram_cache fCreateStateNgramCache(const common_speculative_config & pConfig,
                                                                  uint32_t                          pNSeq,
                                                                  const std::string &               pPathStatic,
                                                                  const std::string &               pPathDynamic) {
  uint16_t vNDraft = 8;  // TODO get from config?

  // TODO bool param in common/common.h to set save_static/save_dynamic?
  bool vSaveStatic  = false;
  bool vSaveDynamic = false;

  common_speculative_impl_ngram_cache vState(pConfig.vParams, pNSeq, vNDraft, pPathStatic, pPathDynamic, vSaveStatic,
                                             vSaveDynamic);

  return vState;
}

std::string fCommonSpeculativeTypeNameStr(const std::vector<common_speculative_type> & pTypes) {
  std::string vResult;

  for (size_t vI = 0; vI < pTypes.size(); vI++) {
    if (vI > 0) {
      vResult += ",";
    }
    vResult += fCommonSpeculativeTypeToStr(pTypes[vI]);
  }
  return vResult;
}

const char * fCommonSpeculativeAllTypesStr() {
  static std::string vAllTypesStr = []() {
    std::vector<common_speculative_type> lTypes;
    lTypes.reserve(cCommonSpeculativeTypeCount);
    for (int vI = 0; vI < cCommonSpeculativeTypeCount; vI++) {
      lTypes.push_back((common_speculative_type) vI);
    }
    return fCommonSpeculativeTypeNameStr(lTypes);
  }();
  return vAllTypesStr.c_str();
}

std::string fCommonSpeculativeTypeToStr(common_speculative_type type) {
  switch (type) {
    case cCommonSpeculativeTypeNone:
      return "none";
    case cCommonSpeculativeTypeDraftSimple:
      return "draft-simple";
    case cCommonSpeculativeTypeDraftEagle3:
      return "draft-eagle3";
    case cCommonSpeculativeTypeDraftMtp:
      return "draft-mtp";
    case cCommonSpeculativeTypeDraftDflash:
      return "draft-dflash";
    case cCommonSpeculativeTypeDraftDspark:
      return "draft-dspark";
    case cCommonSpeculativeTypeNgramSimple:
      return "ngram-simple";
    case cCommonSpeculativeTypeNgramMapK:
      return "ngram-map-k";
    case cCommonSpeculativeTypeNgramMapK4v:
      return "ngram-map-k4v";
    case cCommonSpeculativeTypeNgramMod:
      return "ngram-mod";
    case cCommonSpeculativeTypeNgramCache:
      return "ngram-cache";
    default:
      return "unknown";
  }
}

std::vector<common_speculative_type> fCommonSpeculativeTypesFromNames(const std::vector<std::string> & pNames) {
  std::vector<common_speculative_type> lTypes;
  lTypes.reserve(pNames.size());

  for (const auto & name : pNames) {
    auto type = dCommonSpeculativeTypeFromNameMap.find(name);
    if (type != dCommonSpeculativeTypeFromNameMap.end()) {
      if (type->second == cCommonSpeculativeTypeNone) {
        return std::vector<common_speculative_type>{ cCommonSpeculativeTypeNone };
      }
      lTypes.push_back(type->second);
      continue;
    }
    throw std::invalid_argument("unknown speculative type: " + name);
  }

  return lTypes;
}

common_speculative_type fCommonSpeculativeTypeFromName(const std::string & name) {
  const auto cIt = dCommonSpeculativeTypeFromNameMap.find(name);
  if (cIt == dCommonSpeculativeTypeFromNameMap.end()) {
    return cCommonSpeculativeTypeCount;
  }
  return cIt->second;
}

std::vector<common_speculative_type> fCommonSpeculativeTypesFromGguf(const std::string & pPath) {
  struct gguf_init_params vGgufParams = {
    /* .no_alloc = */ true,
    /* .ctx      = */ nullptr,
  };

  gguf_context_ptr vGgufCtx(gguf_init_from_file(pPath.c_str(), vGgufParams));
  if (!vGgufCtx) {
    return {};
  }

  const int64_t cArchId = gguf_find_key(vGgufCtx.get(), "general.architecture");
  if (cArchId < 0 || gguf_get_kv_type(vGgufCtx.get(), cArchId) != GGUF_TYPE_STRING) {
    return {};
  }

  const std::string cArch = gguf_get_val_str(vGgufCtx.get(), cArchId);
  if (cArch != "dflash") {
    const uint32_t cBlockCount =
      gguf_get_val_u32(vGgufCtx.get(), gguf_find_key(vGgufCtx.get(), (cArch + ".block_count").c_str()));

    if (gguf_find_tensor(vGgufCtx.get(),
                         ("blk." + std::to_string(cBlockCount - 1) + ".nextn.eh_proj.weight").c_str()) >= 0) {
      return { cCommonSpeculativeTypeDraftMtp };
    }

    return {};
  }

  // the Markov head distinguishes draft-dspark from draft-dflash
  const auto type = gguf_find_tensor(vGgufCtx.get(), "markov_w1.weight") >= 0 ? cCommonSpeculativeTypeDraftDspark :
                                                                                cCommonSpeculativeTypeDraftDflash;

  SPC_INF("auto-detected speculative type '%s' from the draft model metadata\n",
          fCommonSpeculativeTypeToStr(type).c_str());

  return { type };
}

static uint32_t fCommonGetEnabledSpeculativeConfigs(const std::vector<common_speculative_type> & pConfigs) {
  uint32_t vResult = 0;
  for (size_t vI = 0; vI < pConfigs.size(); vI++) {
    vResult |= (1u << pConfigs[vI]);
  }
  return vResult;
}

int32_t fCommonSpeculativeNMax(const common_params_speculative * pSpec) {
  int32_t vNMax = 0;

  for (const auto type : pSpec->lTypes) {
    switch (type) {
      case cCommonSpeculativeTypeDraftSimple:
      case cCommonSpeculativeTypeDraftEagle3:
      case cCommonSpeculativeTypeDraftMtp:
      case cCommonSpeculativeTypeDraftDflash:
      case cCommonSpeculativeTypeDraftDspark:
        vNMax = std::max(vNMax, std::max(0, pSpec->vDraft.vNMax));
        break;
      case cCommonSpeculativeTypeNgramSimple:
        vNMax = std::max(vNMax, (int32_t) pSpec->vNgramSimple.vSizeM);
        break;
      case cCommonSpeculativeTypeNgramMapK:
        vNMax = std::max(vNMax, (int32_t) pSpec->vNgramMapK.vSizeM);
        break;
      case cCommonSpeculativeTypeNgramMapK4v:
        vNMax = std::max(vNMax, (int32_t) pSpec->vNgramMapK4v.vSizeM);
        break;
      case cCommonSpeculativeTypeNgramMod:
        vNMax = std::max(vNMax, std::max(0, pSpec->vNgramMod.vNMax));
        break;
      case cCommonSpeculativeTypeNgramCache:
        vNMax = std::max(vNMax, (int32_t) 8);
        break;
      case cCommonSpeculativeTypeNone:
      case cCommonSpeculativeTypeCount:
        break;
    }
  }

  return vNMax;
}

int32_t fCommonSpeculativeNMax(const common_speculative * pSpec) {
  int32_t vNMax = 0;

  if (pSpec == nullptr) {
    return vNMax;
  }

  for (const auto & impl : pSpec->lImpls) {
    vNMax = std::max(vNMax, std::max(0, impl->vNMax));
  }

  return vNMax;
}

std::vector<double> fCommonSpeculativeSynthRatesResolve(const common_params_speculative * pSpec, int32_t pNMax) {
  const bool cHasLength = pSpec->vSynthLen != -1.0;
  const bool cHasRates  = !pSpec->lSynthRates.empty();

  if (!cHasLength && !cHasRates) {
    return {};
  }
  if (cHasLength && cHasRates) {
    throw std::invalid_argument("synthetic acceptance length and rates are mutually exclusive");
  }

  if (pNMax <= 0) {
    throw std::invalid_argument("synthetic acceptance requires at least one speculative token");
  }

  if (cHasRates) {
    const auto & lRates = pSpec->lSynthRates;
    if (lRates.size() != (size_t) pNMax) {
      throw std::invalid_argument(
        fStringFormat("synthetic acceptance rates must contain %d values, got %zu", pNMax, lRates.size()));
    }

    for (size_t vI = 0; vI < lRates.size(); ++vI) {
      if (!std::isfinite(lRates[vI]) || lRates[vI] < 0.0 || lRates[vI] > 1.0) {
        throw std::invalid_argument("synthetic acceptance rates must be finite and within [0, 1]");
      }
      if (vI > 0 && lRates[vI] > lRates[vI - 1]) {
        throw std::invalid_argument("synthetic acceptance rates must be monotonically non-increasing");
      }
    }

    return lRates;
  }

  const double length     = pSpec->vSynthLen;
  const double cLengthMax = (double) pNMax + 1.0;
  if (!std::isfinite(length) || length < 1.0 || length > cLengthMax) {
    throw std::invalid_argument(
      fStringFormat("synthetic acceptance length must be finite and within [1, %.0f]", cLengthMax));
  }

  double vP = 0.0;
  if (length == cLengthMax) {
    vP = 1.0;
  } else if (length > 1.0) {
    double vPMin = 0.0;
    double vPMax = 1.0;
    for (int vI = 0; vI < 32; ++vI) {
      const double cPMid = 0.5 * (vPMin + vPMax);
      double       vSum  = 0.0;
      double       vTerm = cPMid;
      for (int32_t vJ = 0; vJ < pNMax; ++vJ) {
        vSum += vTerm;
        vTerm *= cPMid;
      }

      if (vSum < length - 1.0) {
        vPMin = cPMid;
      } else {
        vPMax = cPMid;
      }
    }
    vP = 0.5 * (vPMin + vPMax);
  }

  std::vector<double> lRates;
  lRates.reserve(pNMax);
  double vRate = vP;
  for (int32_t vI = 0; vI < pNMax; ++vI) {
    lRates.push_back(vRate);
    vRate *= vP;
  }

  return lRates;
}

const std::vector<double> & fCommonSpeculativeGetSynthProbs(const common_speculative * pSpec) {
  GGML_ASSERT(pSpec);
  return pSpec->lSynthProbs;
}

common_params fCommonBaseParamsToSpeculative(const common_params & pParams) {
  const bool cHasDraft = pParams.vSpeculative.mHasDft();

  const auto &  vParamsSpec = pParams.vSpeculative.vDraft;
  common_params vResult     = pParams;

  vResult.vEmbedding   = false;
  vResult.vPoolingType = cDgsPoolingTypeUnspecified;

  if (cHasDraft) {
    // default to global devices value
    if (!vParamsSpec.lDevices.empty()) {
      vResult.lDevices = vParamsSpec.lDevices;
    }
    vResult.vModel               = vParamsSpec.vMparams;
    vResult.vNGpuLayers          = vParamsSpec.vNGpuLayers;
    vResult.lTensorBuftOverrides = vParamsSpec.lTensorBuftOverrides;

    // a draft pinned to a single device doesn't need the meta wrapper an inherited -sm tensor would give it
    // (the device list is null-terminated, so a single device means size 2)
    const size_t cNDevs = std::count_if(vParamsSpec.lDevices.begin(), vParamsSpec.lDevices.end(),
                                        [](ggml_backend_dev_t pD) { return pD != nullptr; });
    if (cNDevs == 1) {
      vResult.vSplitMode = cDgsSplitModeLayer;
    }

    if (vParamsSpec.vCpuparams.vNThreads > 0) {
      vResult.vCpuparams.vNThreads      = vParamsSpec.vCpuparams.vNThreads;
      vResult.vCpuparamsBatch.vNThreads = vParamsSpec.vCpuparamsBatch.vNThreads;
    }
  }

  vResult.vCacheTypeK        = vParamsSpec.vCacheTypeK;
  vResult.vCacheTypeV        = vParamsSpec.vCacheTypeV;
  vResult.vNOutputsMax       = pParams.vNParallel;
  vResult.vNOutputsMaxPerSeq = 1;

  // dflash/dspark decode the whole noise block in a single pass and sample every block position on the backend
  // TODO: refactor such properties to be announced by the speculative types
  //       something like `struct common_speculative_type_props common_speculative_type_get_props(...);`
  const bool cHasBlockDraft =
    std::any_of(pParams.vSpeculative.lTypes.begin(), pParams.vSpeculative.lTypes.end(), [](common_speculative_type pT) {
      return pT == cCommonSpeculativeTypeDraftDflash || pT == cCommonSpeculativeTypeDraftDspark;
    });
  if (cHasBlockDraft) {
    // per-seq output positions: DFlash decodes anchor + n_max masks (n_max + 1); DSpark n_max -> +1 covers both
    const int32_t cPerSeq = std::max(1, vParamsSpec.vNMax + 1);
    vResult.vNOutputsMax  = pParams.vNParallel * cPerSeq;
    if (vParamsSpec.vBackendSampling) {
      vResult.vNOutputsMaxPerSeq = cPerSeq;
    }
  }

  return vResult;
}

struct common_speculative_init_result::impl {
  impl()  = default;
  ~impl() = default;

  // note: the order in which model, context, etc. are declared matters because their destructors will be called bottom-to-top
  dgs_model_ptr   vModel;
  dgs_context_ptr vContext;
};

common_speculative_init_result::common_speculative_init_result(common_params & pParams,
                                                               dgs_model *     pModelTgt,
                                                               dgs_context *   pCtxTgt) :
  vPimpl(new impl{}) {
  const bool cHasDraft = pParams.vSpeculative.mHasDft();
  const bool cSpecMtp  = std::find(pParams.vSpeculative.lTypes.begin(), pParams.vSpeculative.lTypes.end(),
                                   cCommonSpeculativeTypeDraftMtp) != pParams.vSpeculative.lTypes.end();

  auto vMparams = fCommonModelParamsToDgs(pParams);
  auto vCparams = fCommonContextParamsToDgs(pParams);

  if (cSpecMtp) {
    vCparams.vCtxType = cDgsContextTypeMtp;
  }

  // the draft context holds as many tokens per sequence as the target context
  vCparams.vNCtx = fDgsNCtx(pCtxTgt);

  // note: for small models maybe we can set this to the maximum possible draft from all speculative types
  //       the extra memory for small models is likely negligible?
  vCparams.vNRsSeq   = 0;
  vCparams.vCtxOther = pCtxTgt;

  std::string vModelPath;
  if (cHasDraft) {
    vModelPath = pParams.vSpeculative.vDraft.vMparams.vPath;
    LOG_INF("%s: loading draft model '%s'\n", __func__, vModelPath.c_str());

    dgs_model * vModelDft = fDgsModelLoadFromFile(vModelPath.c_str(), vMparams);
    if (vModelDft == NULL) {
      LOG_ERR("%s: failed to load draft model, '%s'\n", __func__, vModelPath.c_str());
      return;
    }

    vPimpl->vModel.reset(vModelDft);

    dgs_context * vCtxDft = fDgsInitFromModel(vModelDft, vCparams);
    if (vCtxDft == nullptr) {
      LOG_ERR("%s: failed to create MTP context\n", __func__);
      return;
    }

    vPimpl->vContext.reset(vCtxDft);
  } else if (cSpecMtp) {
    vModelPath = pParams.vModel.vPath;

    LOG_INF("%s: creating MTP draft context against the target model '%s'\n", __func__, vModelPath.c_str());

    dgs_context * vCtxDft = fDgsInitFromModel(pModelTgt, vCparams);
    if (vCtxDft == nullptr) {
      LOG_ERR("%s: failed to create MTP context\n", __func__);
      return;
    }

    vPimpl->vContext.reset(vCtxDft);
  }
}

common_speculative_init_result::~common_speculative_init_result() = default;

dgs_model * common_speculative_init_result::mModel() {
  return vPimpl->vModel.get();
}

dgs_context * common_speculative_init_result::mContext() {
  return vPimpl->vContext.get();
}

common_speculative_init_result_ptr fCommonSpeculativeInitFromParams(common_params & pParams,
                                                                    dgs_model *     pModelTgt,
                                                                    dgs_context *   pCtxTgt) {
  return std::make_unique<common_speculative_init_result>(pParams, pModelTgt, pCtxTgt);
}

common_speculative_output_limits fCommonSpeculativeGetOutputLimits(int32_t pNBatch,
                                                                   int32_t pNParallel,
                                                                   int32_t pNDraft) {
  const int64_t cPerSeq = 1 + (int64_t) std::max(0, pNDraft);
  const int64_t cTotal  = (int64_t) pNParallel * cPerSeq;

  return {
    /* .total   = */ (int32_t) std::min<int64_t>(pNBatch, cTotal),
    /* .per_seq = */ (int32_t) std::min<int64_t>(pNBatch, cPerSeq),
  };
}

// initialization of the speculative decoding system
//
common_speculative * fCommonSpeculativeInit(common_params_speculative & pParams, uint32_t pNSeq) {
  // Compute the implementations to use based on the config and their order of preference
  std::vector<common_speculative_config> lConfigs = {};  // list of speculative configs to try
  {
    uint32_t vEnabledConfigs = fCommonGetEnabledSpeculativeConfigs(pParams.lTypes);

    auto vAddConfigIfEnabled = [&](common_speculative_type type, bool pAvailable = true) {
      if (pAvailable && (vEnabledConfigs & (1u << type))) {
        lConfigs.emplace_back(type, pParams);
      }
    };

    // when adding a new type - update here the logic above
    static_assert(cCommonSpeculativeTypeCount == 11);

    // this list here defines the priority of the speculators
    // the one with highest priority are listed first
    vAddConfigIfEnabled(cCommonSpeculativeTypeNgramSimple);
    vAddConfigIfEnabled(cCommonSpeculativeTypeNgramMapK);
    vAddConfigIfEnabled(cCommonSpeculativeTypeNgramMapK4v);
    vAddConfigIfEnabled(cCommonSpeculativeTypeNgramMod);
    vAddConfigIfEnabled(cCommonSpeculativeTypeNgramCache);

    vAddConfigIfEnabled(cCommonSpeculativeTypeDraftSimple);
    vAddConfigIfEnabled(cCommonSpeculativeTypeDraftEagle3, pParams.vDraft.vCtxDft != nullptr);
    vAddConfigIfEnabled(cCommonSpeculativeTypeDraftMtp, pParams.vDraft.vCtxDft != nullptr);
    vAddConfigIfEnabled(cCommonSpeculativeTypeDraftDflash, pParams.vDraft.vCtxDft != nullptr);
    vAddConfigIfEnabled(cCommonSpeculativeTypeDraftDspark, pParams.vDraft.vCtxDft != nullptr);
  }

  std::vector<std::unique_ptr<common_speculative_impl>> lImpls = {};

  for (const common_speculative_config & vConfig : lConfigs) {
    switch (vConfig.type) {
      case cCommonSpeculativeTypeNone:
        break;
      case cCommonSpeculativeTypeDraftSimple:
        {
          lImpls.push_back(std::make_unique<common_speculative_impl_draft_simple>(vConfig.vParams, pNSeq));
          break;
        }
      case cCommonSpeculativeTypeDraftEagle3:
        {
          lImpls.push_back(std::make_unique<common_speculative_impl_draft_eagle3>(vConfig.vParams, pNSeq));
          break;
        }
      case cCommonSpeculativeTypeDraftMtp:
        {
          lImpls.push_back(std::make_unique<common_speculative_impl_draft_mtp>(vConfig.vParams, pNSeq));
          break;
        }
      case cCommonSpeculativeTypeDraftDflash:
        {
          lImpls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(vConfig.vParams, pNSeq));
          break;
        }
      case cCommonSpeculativeTypeDraftDspark:
        {
          lImpls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(vConfig.vParams, pNSeq,
                                                                                  cCommonSpeculativeTypeDraftDspark));
          break;
        }
      case cCommonSpeculativeTypeNgramSimple:
        {
          common_ngram_map vNgramMap = fGetCommonNgramMap(vConfig.type, vConfig.vParams.vNgramSimple);

          uint16_t vNgramSizeKey   = vNgramMap.vSizeKey;
          uint16_t vMgramSizeValue = vNgramMap.vSizeValue;

          auto vConfigSimple = common_ngram_simple_config{ /* .size_ngram = */ vNgramSizeKey,
                                                           /* .size_mgram = */ vMgramSizeValue };
          auto vState        = std::make_unique<common_speculative_impl_ngram_simple>(
            /* .params = */ vConfig.vParams,
            /* .n_seq  = */ pNSeq,
            /* .state  = */ vConfigSimple);
          lImpls.push_back(std::move(vState));
          break;
        }
      case cCommonSpeculativeTypeNgramMapK:
        {
          lImpls.push_back(std::make_unique<common_speculative_impl_ngram_map_k>(
            fGetCommonNgramMap(vConfig.type, vConfig.vParams.vNgramMapK), pNSeq));
          break;
        }
      case cCommonSpeculativeTypeNgramMapK4v:
        {
          lImpls.push_back(std::make_unique<common_speculative_impl_ngram_map_k>(
            fGetCommonNgramMap(vConfig.type, vConfig.vParams.vNgramMapK4v), pNSeq));
          break;
        }
      case cCommonSpeculativeTypeNgramMod:
        {
          lImpls.push_back(std::make_unique<common_speculative_impl_ngram_mod>(vConfig.vParams, pNSeq));
          break;
        }
      case cCommonSpeculativeTypeNgramCache:
        {
          auto vState = fCreateStateNgramCache(vConfig, pNSeq, pParams.vNgramCache.vLookupCacheStatic,
                                               pParams.vNgramCache.vLookupCacheDynamic);
          lImpls.push_back(std::make_unique<common_speculative_impl_ngram_cache>(vState));
          break;
        }
      default:
        break;
    }
  }

  if (lImpls.empty()) {
    SPC_TRC("%s", "no implementations specified for speculative decoding\n");
    return nullptr;
  }

  common_speculative_ptr vResult(new common_speculative{
    /* .dparams     = */ common_speculative_draft_params_vec(pNSeq),
    /* .impls       = */ std::move(lImpls),
    /* .impl_last   = */ std::vector<common_speculative_impl *>(pNSeq, nullptr),
    /* .synth_probs = */ {},
  });

  const int32_t cNMaxConfigured = fCommonSpeculativeNMax(&pParams);
  const int32_t cNMaxEffective  = fCommonSpeculativeNMax(vResult.get());
  const auto    lRates          = fCommonSpeculativeSynthRatesResolve(&pParams, cNMaxEffective);

  std::vector<std::string> lRatesStr;
  lRatesStr.reserve(lRates.size());
  vResult->lSynthProbs.reserve(lRates.size());
  double vRatePrev         = 1.0;
  double vAcceptanceLength = 1.0;
  for (const double cRate : lRates) {
    vResult->lSynthProbs.push_back(vRatePrev > 0.0 ? cRate / vRatePrev : 0.0);
    lRatesStr.push_back(fStringFormat("%.6g", cRate));
    vRatePrev = cRate;
    vAcceptanceLength += cRate;
  }
  if (!vResult->lSynthProbs.empty()) {
    SPC_WRN("%s", "synthetic speculative acceptance is enabled for benchmarking; generated output is not valid\n");
    if (cNMaxEffective != cNMaxConfigured) {
      SPC_WRN(
        "synthetic acceptance draft limit was reduced from %d to %d by the initialized speculative implementations\n",
        cNMaxConfigured, cNMaxEffective);
    }
    SPC_INF("synthetic acceptance: n_max = %zu, mean length = %.6f, rates = [%s]\n", lRates.size(), vAcceptanceLength,
            string_join(lRatesStr, ", ").c_str());
  }

  return vResult.release();
}

void fCommonSpeculativeFree(common_speculative * pSpec) {
  if (pSpec == nullptr) {
    return;
  }

  delete pSpec;
}

common_speculative_draft_params & fCommonSpeculativeGetDraftParams(common_speculative * pSpec, dgs_seq_id pSeqId) {
  GGML_ASSERT(pSpec);
  GGML_ASSERT(pSeqId < (dgs_seq_id) pSpec->lDparams.size());

  return pSpec->lDparams[pSeqId];
}

void fCommonSpeculativeBegin(common_speculative * pSpec, dgs_seq_id pSeqId, const dgs_tokens & pPrompt) {
  if (pSpec == nullptr) {
    return;
  }

  for (auto & impl : pSpec->lImpls) {
    common_time_meas vTm(impl->vTBeginUs, !impl->cGenPerf);
    impl->begin(pSeqId, pPrompt);
    impl->vNCallBegin++;
  }
}

bool fCommonSpeculativeProcess(common_speculative * pSpec, const common_batch & pBatch) {
  bool vResult = true;

  if (pSpec == nullptr) {
    return vResult;
  }

  for (auto & impl : pSpec->lImpls) {
    vResult = vResult && impl->mProcess(pBatch);
  }

  return vResult;
}

void fCommonSpeculativeDraft(common_speculative * pSpec) {
  if (pSpec == nullptr) {
    return;
  }

  auto & lDparams = pSpec->lDparams;

  {
    int vNDrafting = 0;

    for (auto & vDp : lDparams) {
      GGML_ASSERT(!vDp.vDrafting || vDp.lResult->empty());

      if (vDp.vDrafting) {
        vNDrafting++;
      }
    }

    if (vNDrafting == 0) {
      return;
    }
  }

  for (auto & impl : pSpec->lImpls) {
    {
      common_time_meas vTm(impl->vTDraftUs, !impl->cGenPerf);
      impl->mDraft(lDparams);
      impl->vNCallDraft++;
    }

    int vNDrafting = 0;

    for (dgs_seq_id vSeqId = 0; vSeqId < (dgs_seq_id) lDparams.size(); ++vSeqId) {
      auto & vDp = lDparams[vSeqId];

      if (!vDp.vDrafting) {
        continue;
      }

      auto & lResult = *vDp.lResult;

      // a new draft has been sampled
      if (vDp.vDrafting && !lResult.empty()) {
        vDp.vDrafting = false;

        if (vDp.vNMax > 0) {
          if (!lResult.empty() && (int) lResult.size() > vDp.vNMax) {
            SPC_DBG("truncating draft to %d tokens\n", vDp.vNMax);
            lResult.resize(vDp.vNMax);

            // trim the candidates only if the drafter produced them (n-gram drafters do not)
            if (vDp.lResultQ && !vDp.lResultQ->empty()) {
              vDp.lResultQ->resize(vDp.vNMax);
            }
          }
        }

        if (!lResult.empty()) {
          SPC_DBG("called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n",
                  fCommonSpeculativeTypeToStr(impl.get()->type).c_str(), vDp.lPrompt->size(), impl.get()->vNCallDraft,
                  lResult.size());

          // remember which implementation was used
          pSpec->lImplLast[vSeqId] = impl.get();

          impl->vNGenDrafts++;
          impl->vNGenTokens += lResult.size();
        }
      }

      if (vDp.vDrafting) {
        vNDrafting++;
      }
    }

    if (vNDrafting == 0) {
      break;
    }
  }

  // these sequences failed to generate a draft
  for (dgs_seq_id vSeqId = 0; vSeqId < (dgs_seq_id) lDparams.size(); ++vSeqId) {
    auto & vDp = lDparams[vSeqId];

    if (vDp.vDrafting) {
      vDp.vDrafting = false;
    }
  }
}

void fCommonSpeculativeAccept(common_speculative * pSpec, dgs_seq_id pSeqId, uint16_t pNAccepted) {
  common_speculative_impl * impl = pSpec->lImplLast[pSeqId];

  if (impl == nullptr) {
    GGML_ASSERT(pNAccepted == 0);
    return;
  }

  {
    common_time_meas vTm(impl->vTAcceptUs, !impl->cGenPerf);

    if (impl->lNAccTokensPerPos.size() < pNAccepted) {
      impl->lNAccTokensPerPos.resize(pNAccepted, 0);
    }

    for (size_t vI = 0; vI < pNAccepted; ++vI) {
      impl->lNAccTokensPerPos[vI]++;
    }

    if (pNAccepted > 0) {
      impl->vNAccDrafts++;
      impl->vNAccTokens += pNAccepted;
    }

    impl->accept(pSeqId, pNAccepted, false);
    impl->vNCallAccept++;
  }

  // accept with the rest of the implementations, using is_other == true
  for (auto & vImplOther : pSpec->lImpls) {
    if (vImplOther.get() != impl) {
      vImplOther->accept(pSeqId, pNAccepted, true);
    }
  }
}

// TODO: support the case of more than one speculative implementations having a state
bool fCommonSpeculativeGetState(common_speculative * pSpec, dgs_seq_id pSeqId, std::vector<uint8_t> & data) {
  if (pSpec == nullptr) {
    return false;
  }

  for (auto & impl : pSpec->lImpls) {
    if (impl->mGetState(pSeqId, data)) {
      return true;
    }
  }

  return false;
}

void fCommonSpeculativeSetState(common_speculative * pSpec, dgs_seq_id pSeqId, const std::vector<uint8_t> & data) {
  if (pSpec == nullptr) {
    return;
  }

  for (auto & impl : pSpec->lImpls) {
    impl->mSetState(pSeqId, data);
  }
}

void fCommonSpeculativePrintStats(const common_speculative * pSpec) {
  if (pSpec == nullptr) {
    return;
  }

  for (const auto & impl : pSpec->lImpls) {
    std::string vStrPerf;
    if (impl->cGenPerf) {
      std::ostringstream vOss;
      vOss << std::fixed << std::setprecision(3) << impl->vTBeginUs / 1000.0 << ", ";
      vOss << std::fixed << std::setprecision(3) << impl->vTDraftUs / 1000.0 << ", ";
      vOss << std::fixed << std::setprecision(3) << impl->vTAcceptUs / 1000.0;
      vStrPerf = ", dur(b,g,a) = " + vOss.str() + " ms";
    } else {
      vStrPerf = "";
    }

    std::string vStrStats;
    if (impl->vNCallAccept > 0) {
      const double       cMean = 1.0 + (double) impl->vNAccTokens / (double) impl->vNCallAccept;
      std::ostringstream vTmp;
      vTmp << std::fixed << std::setprecision(3);
      for (size_t vI = 0; vI < impl->lNAccTokensPerPos.size(); ++vI) {
        if (vI > 0) {
          vTmp << ", ";
        }
        vTmp << (double) impl->lNAccTokensPerPos[vI] / (double) impl->vNCallAccept;
      }
      std::ostringstream vOss;
      vOss << std::fixed << std::setprecision(2) << cMean;
      vStrStats = ", #mean acc len = " + vOss.str() + ", #acc rate/pos = (" + vTmp.str() + ")";
    }

    SPC_TRC(
      "statistics %16s: #calls(b,g,a) = %4zu %6zu %6zu, #gen drafts = %6zu, #acc drafts = %5zu, #gen tokens = %6zu, "
      "#acc tokens = %5zu%s%s\n",
      fCommonSpeculativeTypeToStr(impl->type).c_str(), impl->vNCallBegin, impl->vNCallDraft, impl->vNCallAccept,
      impl->vNGenDrafts, impl->vNAccDrafts, impl->vNGenTokens, impl->vNAccTokens, vStrStats.c_str(), vStrPerf.c_str());
  }
}
