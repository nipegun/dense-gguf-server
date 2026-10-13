#include "server-context.h"

#include "../../src/dgs-ext.h"
#include "build-info.h"
#include "common.h"
#include "dgs.h"
#include "fit.h"
#include "log.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "sampling.h"
#include "server-chat.h"
#include "server-common.h"
#include "server-decision.h"
#include "server-http.h"
#include "server-monitor.h"
#include "server-queue.h"
#include "server-schema.h"
#include "server-stream.h"
#include "server-task.h"
#include "speculative.h"

#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <utility>

// fix problem with std::min and std::max
#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#endif

constexpr int cHttpPollingSeconds = 1;

static common_speculative_output_limits fServerOutputLimits(const common_params & pParams) {
  if (pParams.vEmbedding ||
      (pParams.vPoolingType != cDgsPoolingTypeUnspecified && pParams.vPoolingType != cDgsPoolingTypeNone)) {
    return { pParams.vNBatch, 1 };
  }

  auto vResult = fCommonSpeculativeGetOutputLimits(pParams.vNBatch, pParams.vNParallel,
                                                   fCommonSpeculativeNMax(&pParams.vSpeculative));

  vResult.vTotal  = std::max<int32_t>(1, vResult.vTotal);
  vResult.vPerSeq = std::max<int32_t>(1, vResult.vPerSeq);
  return vResult;
}

// a checkpoint restore dropped tokens the target had accepted - re-accept them rather than verify again
static std::vector<dgs_token> fServerAcceptReplay(common_sampler *             pSmpl,
                                                  dgs_context *                pCtx,
                                                  const std::vector<int32_t> & pIdxs,
                                                  const dgs_tokens &           pDraft) {
  GGML_ASSERT(pIdxs.size() == pDraft.size() + 1);

  std::vector<dgs_token> lResult;
  lResult.reserve(pIdxs.size());

  for (size_t vI = 0; vI < pDraft.size(); ++vI) {
    // the token is discarded - the call is what advances the sampler over this position
    fCommonSamplerSample(pSmpl, pCtx, pIdxs[vI]);
    fCommonSamplerAccept(pSmpl, pDraft[vI], true);
    lResult.push_back(pDraft[vI]);
  }

  const dgs_token cId = fCommonSamplerSample(pSmpl, pCtx, pIdxs[pDraft.size()]);
  fCommonSamplerAccept(pSmpl, cId, true);
  lResult.push_back(cId);

  return lResult;
}

// synthetic draft verification for benchmarking - accept draft tokens at random instead of by match with the target
// on replay the draft was already accepted before a context checkpoint restore, so repeat the same decisions
static std::vector<dgs_token> fServerSampleAndAcceptSynth(common_sampler *             pSmpl,
                                                          dgs_context *                pCtx,
                                                          const std::vector<int32_t> & pIdxs,
                                                          const dgs_tokens &           pDraft,
                                                          const std::vector<double> &  pSynthProbs,
                                                          std::mt19937 &               pRng,
                                                          bool                         pIsReplay) {
  GGML_ASSERT(pIdxs.size() == pDraft.size() + 1);
  GGML_ASSERT(pSynthProbs.size() >= pDraft.size());

  std::vector<dgs_token> lResult;
  lResult.reserve(pIdxs.size());

  const dgs_vocab *                      vVocab = fDgsModelGetVocab(fDgsGetModel(pCtx));
  std::uniform_real_distribution<double> vDist(0.0, 1.0);
  for (size_t vI = 0; vI < pDraft.size(); ++vI) {
    const dgs_token cId    = fCommonSamplerSample(pSmpl, pCtx, pIdxs[vI]);
    const bool      accept = pIsReplay || vDist(pRng) < pSynthProbs[vI];
    // do not accept a drafted EOG token - it would end the generation early
    // on replay the last token is from the target and can be EOG, so skip this check
    if (accept && (pIsReplay || !fDgsVocabIsEog(vVocab, pDraft[vI]))) {
      // synthetic draft tokens do not advance grammar or reasoning state
      // the last replay token is from the target and must advance both
      const bool cIsReplayTarget = pIsReplay && vI + 1 == pDraft.size();
      fCommonSamplerAccept(pSmpl, pDraft[vI], cIsReplayTarget);
      lResult.push_back(pDraft[vI]);
      continue;
    }

    fCommonSamplerAccept(pSmpl, cId, true);
    lResult.push_back(cId);
    return lResult;
  }

  const dgs_token cId = fCommonSamplerSample(pSmpl, pCtx, pIdxs[pDraft.size()]);
  fCommonSamplerAccept(pSmpl, cId, true);
  lResult.push_back(cId);

  return lResult;
}

// state diagram: 
enum slot_state {
  cSlotStateIdle,
  cSlotStateWaitOther,  // after assigning a task, but waiting for parent slot to process prompt
  cSlotStateStarted,    // after assigning a task and about to process prompt
  cSlotStateProcessingPrompt,
  cSlotStateDonePrompt,
  cSlotStateGenerating,
};

struct server_slot;  // forward declaration

struct server_batch {
  common_batch vView;  // the rendered sub-batch [off, off + n_tokens), see render()

  struct token {
    int32_t                                  vIdSlot;
    dgs_token                                token;
    std::array<dgs_pos, GGML_MROPE_SECTIONS> aPos;                 // only pos[0] is used for text tokens
    bool                                     vOutput;
    bool                                     vIsPrompt;            // for stats tracking
    int32_t                                  vDecisionOrder = 0;
    int32_t                                  vIEmbd         = -1;  // row in embd, -1 if this is a token
  };

  std::vector<token> lTokens;
  int32_t            vNTokensAlloc = 0;
  int32_t            vNEmbd        = 0;

  // track if given slot can be batched with slots already in the batch
  server_slot * vSlotBatched = nullptr;

  // embedding entries, they can be mixed with tokens if the context supports it
  std::vector<float> lEmbd;  // n_embd per row

  float  vAloraScale      = -1.0f;
  size_t vAloraDisabledId = 0;

  void init(dgs_context * pCtx, int32_t pNTokensAlloc, int32_t pNEmbd) {
    this->vNTokensAlloc = pNTokensAlloc;
    this->vNEmbd        = pNEmbd;
    vView               = common_batch(pCtx);
    lTokens.reserve(pNTokensAlloc);
  }

  bool mAdd(int32_t pIdSlot, dgs_token token, dgs_pos pPos, bool pOutput, bool pIsPrompt) {
    if ((int32_t) lTokens.size() >= vNTokensAlloc) {
      return false;
    }
    lTokens.push_back({
      pIdSlot, token, { pPos, 0, 0, 0 },
        pOutput, pIsPrompt
    });
    return true;
  }

  // embd_in has n_embd values, pos has GGML_MROPE_SECTIONS values
  bool mAddEmbd(int32_t pIdSlot, const float * pEmbdIn, const dgs_pos * pPos, bool pOutput, bool pIsPrompt) {
    if ((int32_t) lTokens.size() >= vNTokensAlloc) {
      return false;
    }
    lTokens.push_back({
      pIdSlot, DGS_TOKEN_NULL, { pPos[0], pPos[1], pPos[2], pPos[3] },
        pOutput, pIsPrompt
    });
    lTokens.back().vIEmbd = (int32_t) (lEmbd.size() / vNEmbd);
    lEmbd.insert(lEmbd.end(), pEmbdIn, pEmbdIn + vNEmbd);
    return true;
  }

  bool mHasEmbd() const { return !lEmbd.empty(); }

  void clear() {
    lTokens.clear();
    lEmbd.clear();
    vView.clear();
    vSlotBatched     = nullptr;
    vAloraScale      = -1.0f;
    vAloraDisabledId = 0;
  }

  int32_t size() const { return (int32_t) lTokens.size(); }

  // remove the entries after the first n
  void mTruncate(int32_t n) {
    GGML_ASSERT(n >= 0 && n <= size());
    for (int32_t vI = n; vI < size(); vI++) {
      if (lTokens[vI].vIEmbd >= 0) {
        lEmbd.resize((size_t) lTokens[vI].vIEmbd * vNEmbd);
        break;
      }
    }
    lTokens.resize(n);
  }

  void mSetOutput(int32_t pIdx, bool pOutput) {
    GGML_ASSERT(pIdx >= 0 && pIdx < (int32_t) lTokens.size());
    lTokens[pIdx].vOutput = pOutput;
  }

  void mSetDecisionOrder(int32_t pIdx, int32_t pOrder) {
    GGML_ASSERT(pIdx >= 0 && pIdx < (int32_t) lTokens.size());
    lTokens[pIdx].vDecisionOrder = pOrder;
  }

  // render the sub-batch [off, off + n_tokens) into view, index i in view is index off + i here
  void mRender(int32_t pOff, int32_t pNTokens) {
    GGML_ASSERT(pOff >= 0 && pOff < size());
    GGML_ASSERT(pNTokens > 0 && pOff + pNTokens <= size());

    vView.clear();
    for (int32_t vI = pOff; vI < pOff + pNTokens; vI++) {
      const auto & vT = lTokens[vI];
      if (vT.vIEmbd >= 0) {
        vView.mAddEmbd({ lEmbd.data() + (size_t) vT.vIEmbd * vNEmbd, 1, (size_t) vNEmbd }, vT.aPos.data(), vT.vIdSlot,
                       vT.vOutput);
      } else {
        vView.mAdd(vT.token, vT.aPos[0], vT.vIdSlot, vT.vOutput);
      }
      vView.lTokens.back().vDecisionOrder = vT.vDecisionOrder;
    }
  }
};

struct server_slot {
  int vId;

  dgs_context * vCtxTgt = nullptr;
  dgs_context * vCtxDft = nullptr;

  common_memory vMem;

  // multimodal
  mtmd_context *  vMctx   = nullptr;
  mtmd::batch_ptr vMbatch = nullptr;

  // speculative decoding
  common_speculative * vSpec;

  dgs_tokens lSpecDraft;

  // draft candidates per token in spec_draft; only draft-simple and draft-mtp fill it
  std::vector<std::vector<dgs_token_data>> lSpecDraftQ;
  dgs_tokens                               lSpecPrompt;
  std::vector<int32_t>                     lSpecIBatch;
  common_prompt_checkpoint                 vSpecCkpt;
  bool                                     vSpecIsReplay = false;
  std::mt19937                             vSpecSynthRng;

  // TODO: move members that belong to the task (such as `generated`, `has_new_line`) to task_results_state
  std::unique_ptr<const server_task> vTask;
  std::unique_ptr<const server_task> vTaskPrev;  // used for debugging

  // used to determine the slot that has been used the longest
  int64_t vTLastUsed = -1;

  // generation props
  int32_t vNCtx   = 0;  // context size per slot
  int32_t vNKeep  = 0;
  int32_t vIBatch = -1;

  // effective generation limit for the current task, -1 means unlimited
  int32_t vNPredictMax = -1;

  size_t vLastNlPos = 0;

  common_chat_input vGenerated;
  std::string       vDebugGeneratedText;
  dgs_tokens        lGeneratedTokens;
  size_t            vNSentText = 0;  // number of sent text character (i.e. handle partial UTF-8 on streaming)

  std::vector<completion_token_output> lGeneratedTokenProbs;

  bool vHasNextToken = true;
  bool vHasNewLine   = false;
  bool vTruncated    = false;

  stop_type vStop;

  std::string vStoppingWord;

  // state
  slot_state vState = cSlotStateIdle;

  server_prompt vPrompt;

  bool mPromptSave(server_prompt_cache & pPromptCache) const {
    if (vPrompt.vTokens.size() == 0) {
      return false;
    }

    const size_t cCurSizeTgt = fDgsStateSeqGetSizeExt(vCtxTgt, vId, DGS_STATE_SEQ_FLAGS_NONE);
    const size_t cCurSizeDft = vCtxDft ? fDgsStateSeqGetSizeExt(vCtxDft, vId, DGS_STATE_SEQ_FLAGS_NONE) : 0;

    const size_t cCurSize = cCurSizeTgt + cCurSizeDft;

    SRV_TRC(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)\n",
            (int) vPrompt.vTokens.size(), cCurSize / (1024.0 * 1024.0), cCurSizeDft / (1024.0 * 1024.0));

    auto * vCur = pPromptCache.mAlloc(vPrompt, cCurSizeTgt, cCurSizeDft);
    if (vCur == nullptr) {
      return false;
    }

    fDgsStateSeqGetDataExt(vCtxTgt, vCur->data.main.data(), cCurSizeTgt, vId, DGS_STATE_SEQ_FLAGS_NONE);
    if (vCtxDft) {
      fDgsStateSeqGetDataExt(vCtxDft, vCur->data.lDrft.data(), cCurSizeDft, vId, DGS_STATE_SEQ_FLAGS_NONE);
    }

    return true;
  }

  bool mPromptLoad(server_prompt_cache & pPromptCache, const server_tokens & pTokens) {
    bool vRes = pPromptCache.mLoad(vPrompt, pTokens, vCtxTgt, vCtxDft, vId);
    if (!vRes) {
      SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
    }

    return vRes;
  }

  void mPromptClear() {
    SLT_TRC(*this, "clearing prompt with %zu tokens\n", vPrompt.vTokens.size());

    vMem.mSeqRm(vId, -1, -1);

    vPrompt.clear();
  }

  std::vector<common_adapter_lora_info> lora;
  int32_t                               vAloraInvocationStart = -1;

  // sampling
  json vJsonSchema;

  common_sampler_ptr vSmpl;

  dgs_token vSampled;  // in speculative mode, this is the last accepted token

  // for TTS models, this is the embd generated from prev step, decode this to generate next hidden state
  // corresponding to one token position (size = n_embd)
  std::vector<float> lInpEmbd;

  server_slot_stats vStats;

  // accepted tokens per draft position
  // not in server_slot_stats to avoid copying to every task result
  std::vector<uint64_t> lNAcceptedPerPos;

  std::function<void(int /* id_slot */)>   vCallbackOnRelease;
  std::function<void(const server_slot &)> vCallbackOnReset;  // called before reset()

  // this is for printing timings with slot progress, not part of metrics
  int64_t vTPrintLast = 0;
  int32_t vNGenLast   = 0;

  void reset() {
    SLT_DBG(*this, "%s", "\n");

    vSpecIsReplay = false;

    vLastNlPos    = 0;
    vGenerated    = {};
    vHasNewLine   = false;
    vTruncated    = false;
    vStop         = cStopTypeNone;
    vStoppingWord = "";
    vNSentText    = 0;

    if (mCanSpeculate()) {
      lSpecDraft.clear();
      lSpecIBatch.clear();
      vSpecCkpt.clear();
    }
    lGeneratedTokens.clear();
    lGeneratedTokenProbs.clear();
    vJsonSchema = json();

    vTaskPrev = std::move(vTask);
    vTask.reset();

    // note: callback_on_reset() must have run before this, see release()
    vStats = {};
    lNAcceptedPerPos.clear();

    vNPredictMax = -1;

    fDgsSetSampler(vCtxTgt, vId, nullptr);

    // clear alora start
    vAloraInvocationStart = -1;

    // clear multimodal state
    vMbatch.reset();
  }

  void mInitSampler() const {
    fCommonSamplerReset(vSmpl.get());

    if (!vTask->mNeedSampling()) {
      return;
    }

    const int64_t cTStart = ggml_time_us();

    int vNText = 0;

    for (int vI = 0; vI < (int) vPrompt.vTokens.size(); vI++) {
      const dgs_token cId = vPrompt.vTokens[vI];

      if (cId != DGS_TOKEN_NULL) {
        fCommonSamplerAccept(vSmpl.get(), cId, false);
        vNText++;
      }
    }

    SLT_TRC(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n", (ggml_time_us() - cTStart) / 1000.0,
            vNText, (int) vPrompt.vTokens.size());
  }

  bool mNeedEmbd() const {
    GGML_ASSERT(vTask);
    return vTask->mNeedEmbd();
  }

  bool mCanSplit() const {
    GGML_ASSERT(vTask);
    // a decision over a non-causal context reads the whole prompt in one batch
    if (vTask->type == cServerTaskTypeDecision && !fDgsGetCausalAttn(vCtxTgt)) {
      return false;
    }
    // MTP supports splitting - uses task->need_embd() not need_embd()
    if (!vTask->mNeedEmbd()) {
      return true;
    }
    // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
    if (!fDgsGetMemory(vCtxTgt)) {
      return false;
    }
    // context can be chunked/split if the pooling type is LAST
    const auto cPooling = fDgsPoolingType(vCtxTgt);
    if (cPooling == cDgsPoolingTypeLast) {
      return true;
    }
    // causal rerankers read the last token and have a KV cache, so they can also be chunked/split.
    if (cPooling == cDgsPoolingTypeRank && fDgsGetCausalAttn(vCtxTgt)) {
      return true;
    }
    // a decision task reads its outputs from the last batch
    if (vTask->type == cServerTaskTypeDecision) {
      return true;
    }
    return false;
  }

  bool mCanBatchWith(server_slot & pOtherSlot) const {
    GGML_ASSERT(vTask);

    // a joint decision head reads the whole batch
    if (!vTask->decision.lOrder.empty() || !pOtherSlot.vTask->decision.lOrder.empty()) {
      return false;
    }

    return vTask->type == pOtherSlot.vTask->type && lInpEmbd.size() == pOtherSlot.lInpEmbd.size() &&
           fAreLoraEqual(lora, pOtherSlot.lora);
  }

  // returns -1 if the generation is limitless
  int32_t mNRemaining() const { return vNPredictMax == -1 ? -1 : vNPredictMax - (int32_t) vStats.vNGen; }

  bool mHasBudget() const { return vNPredictMax == -1 || mNRemaining() > 0; }

  bool mIsProcessing() const { return vState != cSlotStateIdle; }

  bool mCanSpeculate() const { return !!vSpec; }

  // at temp 0 both p and q are point masses, so rejection is the same as sample-and-match
  bool mUseSpecRejection() const { return vTask && vTask->vParams.vSampling.vTemp > 0.0f; }

  void mAddToken(const completion_token_output & token) {
    if (!mIsProcessing()) {
      SLT_WRN(*this, "%s", "slot is not processing\n");
      return;
    }

    lGeneratedTokenProbs.push_back(token);
  }

  int mGetNDraftMax() const {
    GGML_ASSERT(vTask);

    if (!mCanSpeculate()) {
      return 0;
    }

    // determine the max draft that fits the current slot state
    // note: slot.prompt is not yet expanded with the `id` token sampled above
    //       also, need to leave space for 1 extra token to allow context shifts
    int vNDraftMax = vNCtx - vPrompt.mNTokens() - 2;

    if (mNRemaining() > 0) {
      vNDraftMax = std::min(vNDraftMax, mNRemaining() - 1);
    }

    SLT_DBG(*this, "max possible draft: %d\n", vNDraftMax);

    return vNDraftMax;
  }

  // add sampled token of this slot to the batch, optionally add the speculative draft tokens if any
  void mHandleLastSampledToken(server_batch & pBatch) {
    bool vAddOk = true;
    if (lSpecDraft.empty()) {
      // no speculative decoding
      vIBatch = pBatch.size();

      if (!lInpEmbd.empty()) {
        // text embeddings broadcast the same position across the M-RoPE sections
        const dgs_pos cP                        = vPrompt.vTokens.mPosNext();
        const dgs_pos aPos[GGML_MROPE_SECTIONS] = { cP, cP, cP, 0 };
        GGML_ASSERT((int32_t) lInpEmbd.size() == pBatch.vNEmbd);
        vAddOk &= pBatch.mAddEmbd(vId, lInpEmbd.data(), aPos, true, false);
      } else {
        vAddOk &= pBatch.mAdd(vId, vSampled, vPrompt.vTokens.mPosNext(), true, false);
      }

      SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n", vSampled, vNCtx,
              vPrompt.mNTokens(), vTruncated);
    } else {
      SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n", vSampled, vPrompt.vTokens.size(),
              lSpecDraft.size(), vPrompt.vTokens.mPosNext());

      GGML_ASSERT(lSpecIBatch.empty());

      lSpecIBatch.push_back(pBatch.size());
      for (size_t vI = 0; vI < lSpecDraft.size(); vI++) {
        lSpecIBatch.push_back(pBatch.size() + vI + 1);
      }

      auto vPos0 = vPrompt.vTokens.mPosNext();

      vAddOk &= pBatch.mAdd(vId, vSampled, vPos0++, true, false);
      for (auto token : lSpecDraft) {
        vAddOk &= pBatch.mAdd(this->vId, token, vPos0++, true, false);
      }
    }

    GGML_ASSERT(vAddOk && "batch must be large enough to hold the sampled and draft tokens");

    vPrompt.vTokens.push_back(vSampled);
    vPrompt.vTokens.insert(lSpecDraft);
  }

  void mRelease() {
    if (mIsProcessing()) {
      GGML_ASSERT(vTask);

      SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", vPrompt.mNTokens(), vTruncated);

      vTLastUsed = ggml_time_us();

      vState = cSlotStateIdle;

      // do not keep context of the child slots - the parent's context is enough
      if (vTask->mIsChild()) {
        mPromptClear();
      }

      vCallbackOnReset(*this);

      reset();

      vCallbackOnRelease(vId);
    }
  }

  size_t mFindStoppingStrings(const std::string & pText, const size_t pLastTokenSize, bool pIsFullStop) {
    GGML_ASSERT(vTask);

    size_t vStopPos = std::string::npos;

    for (const std::string & vWord : vTask->vParams.lAntiprompt) {
      size_t vPos;

      if (pIsFullStop) {
        const size_t cTmp     = vWord.size() + pLastTokenSize;
        const size_t cFromPos = pText.size() > cTmp ? pText.size() - cTmp : 0;

        vPos = pText.find(vWord, cFromPos);
      } else {
        // otherwise, partial stop
        vPos = fStringFindPartialStop(pText, vWord);
      }

      if (vPos != std::string::npos && (vStopPos == std::string::npos || vPos < vStopPos)) {
        if (pIsFullStop) {
          vStop         = cStopTypeWord;
          vStoppingWord = vWord;
          vHasNextToken = false;
        }
        vStopPos = vPos;
      }
    }

    return vStopPos;
  }

  void mPrintTimingsTg() {
    if (vStats.vNGen < 100) {
      return;
    }

    const int64_t cTNow = ggml_time_us();

    if (cTNow - vTPrintLast < 3 * 1000 * 1000) {
      return;
    }

    const double cNGenSecond    = vStats.mNGenTps();
    const double cNGenSecondWin = 1e6 / (cTNow - vTPrintLast) * (vStats.vNGen - vNGenLast);

    vTPrintLast = cTNow;
    vNGenLast   = vStats.vNGen;

    SLT_INF(*this, "n_gen = %6d, tg = %6.2f t/s, tg_3s = %6.2f t/s\n", (int) vStats.vNGen, cNGenSecond, cNGenSecondWin);
  }

  void mPrintTimingsPp() const {
    const double cTPromptTotal = vStats.mTPromptMs();

    if (cTPromptTotal < 3000.0) {
      return;
    }

    const double cNPromptSecond = vStats.mNPromptTps();
    const double cFProgress     = vTask->mNTokens() > 0 ? (double) vPrompt.mNTokens() / vTask->mNTokens() : 0.0;

    SLT_INF(*this, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
            (int) vStats.vNPromptProcessed, cFProgress, cTPromptTotal / 1e3, cNPromptSecond);
  }

  void mPrintTimings() const {
    const double cTPromptTotal = vStats.mTPromptMs();
    const double cTGenTotal    = vStats.mTGenMs();

    const double cTPrompt       = vStats.mTPromptPerTokenMs();
    const double cNPromptSecond = vStats.mNPromptTps();

    const double cTGen       = vStats.mTGenPerTokenMs();
    const double cNGenSecond = vStats.mNGenTps();

    SLT_INF(*this, "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            cTPromptTotal, (int) vStats.vNPromptProcessed, cTPrompt, cNPromptSecond);

    SLT_INF(*this, "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            cTGenTotal, (int) vStats.vNGen, cTGen, cNGenSecond);

    SLT_INF(*this, "      total time = %10.2f ms / %5d tokens\n", cTPromptTotal + cTGenTotal,
            (int) (vStats.vNPromptProcessed + vStats.vNGen));

    SLT_INF(*this, "   graphs reused = %10d\n", fDgsPerfContext(vCtxTgt).vNReused);

    SLT_INF(*this, "  graph reallocs = %10d    / %5d        (same size) / (changed size)\n",
            fDgsPerfContext(vCtxTgt).vNReallocsSame, fDgsPerfContext(vCtxTgt).vNReallocsDiff);

    const int32_t cNDraftTotal      = vStats.vNDraftTokens;
    const int32_t cNDraftAccepted   = vStats.vNDraftAccepted;
    const int32_t cNDraftVerifSteps = vStats.vNDraftVerifSteps;

    if (cNDraftTotal > 0) {
      const float  cDraftRatio = (float) cNDraftAccepted / cNDraftTotal;
      const double cMeanAccLen =
        cNDraftVerifSteps > 0 ? 1.0 + (double) cNDraftAccepted / (double) cNDraftVerifSteps : 1.0;

      std::string vAcceptanceRatesPerPos;
      if (cNDraftVerifSteps > 0) {
        for (size_t vI = 0; vI < lNAcceptedPerPos.size(); ++vI) {
          if (vI > 0) {
            vAcceptanceRatesPerPos += ", ";
          }
          vAcceptanceRatesPerPos += fStringFormat("%.3f", (double) lNAcceptedPerPos[vI] / (double) cNDraftVerifSteps);
        }
      }

      SLT_INF(*this, "draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f\n", cDraftRatio,
              cNDraftAccepted, cNDraftTotal, cMeanAccLen);
      SLT_TRC(*this, "     acc per pos = (%s)\n", vAcceptanceRatesPerPos.c_str());
    }

    fCommonSpeculativePrintStats(vSpec);
  }

  json to_json(bool pOnlyMetrics = false) const {
    json vRes;

    vRes = {
      { "id",            vId             },
      { "n_ctx",         vNCtx           },
      { "speculative",   mCanSpeculate() },
      { "is_processing", mIsProcessing() },
    };

    const auto & vPtask = vTask ? vTask : vTaskPrev;

    if (vPtask) {
      vRes["id_task"]                   = vPtask->vId;
      vRes["n_prompt_tokens"]           = (int32_t) vPrompt.vTokens.size();
      vRes["n_prompt_tokens_processed"] = vStats.vNPromptProcessed;
      vRes["n_prompt_tokens_cache"]     = vStats.vNPromptCached;
      vRes["params"]                    = vPtask->vParams.to_json(pOnlyMetrics);
      vRes["next_token"]                = json::mArray({
        {
         { "has_next_token", vHasNextToken },
         { "has_new_line", vHasNewLine },
         { "n_remain", mNRemaining() },
         { "n_decoded", vStats.vNGen },
         }
      });

      if (!pOnlyMetrics) {
        vRes["prompt"]    = vPtask->vTokens.mDetokenize(vCtxTgt, true);
        vRes["generated"] = vGenerated.empty() ? vDebugGeneratedText : vGenerated.vText;
      }
    }

    return vRes;
  }

  // the other slot continues from the tokens processed so far
  void mCopyPromptTo(server_slot & pOther) const {
    vMem.mSeqRm(pOther.vId, -1, -1);
    vMem.mSeqCp(vId, pOther.vId, -1, -1);

    pOther.vPrompt = vPrompt.clone();
  }

  void mCopyStateTo(server_slot & pOther) const {
    GGML_ASSERT(vState == cSlotStateDonePrompt);

    vMem.mSeqRm(pOther.vId, -1, -1);
    vMem.mSeqCp(vId, pOther.vId, -1, -1);

    pOther.vIBatch = vIBatch;

    pOther.vStats = vStats;

    pOther.vPrompt = vPrompt.clone();
    pOther.mInitSampler();
  }
};

// encode the mtmd chunk at idx, batched with as many of the next chunks as possible
// returns 0 on success
static int fEncodeMtmdChunk(const server_slot & pSlot, mtmd::batch_ptr & pMbatch, size_t pIdx) {
  const auto & vMctx        = pSlot.vMctx;
  const auto & vInputTokens = pSlot.vTask->vTokens;
  const auto & vChunk       = vInputTokens.mFindChunk(pIdx);

  pMbatch.reset(fMtmdBatchInit(vMctx));
  int32_t vRes = fMtmdBatchAddChunk(pMbatch.get(), vChunk.get());
  GGML_ASSERT(vRes == 0);  // we should never have an empty batch

  // try batching as much as possible
  int    vNAdded = 1;
  size_t vIdxCur = pIdx;
  while (vRes == 0) {
    auto [next_chunk, next_idx] = vInputTokens.mFindNextMediaChunk(vIdxCur);
    if (next_chunk == nullptr) {
      break;
    }
    vRes = fMtmdBatchAddChunk(pMbatch.get(), next_chunk->get());
    vNAdded += (vRes == 0 ? 1 : 0);
    vIdxCur = next_idx;
    SLT_DBG(pSlot, "try adding media chunk idx = %zu to batch, res = %d\n", next_idx, vRes);
    // if res != 0, batch is full or chunk is not compatible -> this loop breaks
  }

  // TODO @ngxson : move this log line to debug when it become more stable
  SLT_TRC(pSlot, "encoding mtmd batch from idx = %zu, n_chunks = %d\n", pIdx, vNAdded);

  vRes = fMtmdBatchEncode(pMbatch.get());
  if (vRes != 0) {
    SLT_ERR(pSlot, "failed to encode mtmd batch for chunk idx = %zu, res = %d\n", pIdx, vRes);
    return -1;
  }

  return 0;
}

// returns 0 on success
// caller need to update prompt.tokens after a successful call to keep track of the processing progress
// note: this is not a member of server_slot because we want to run it inside yield_to_queue
//       slot is passed as const to avoid accidental modification of the slot state
//       some pointers are allowed to be used, they are not used by to_json()
static int fProcessMtmdChunk(const server_slot & pSlot, mtmd::batch_ptr & pMbatch, size_t pIdx, size_t & pNTokensOut) {
  GGML_ASSERT(pSlot.vMctx);
  const auto & vMctx        = pSlot.vMctx;
  const auto & vInputTokens = pSlot.vTask->vTokens;
  const auto & vChunk       = vInputTokens.mFindChunk(pIdx);
  int32_t      vRes         = 0;

  auto vTryDecode = [&]() -> int32_t {
    if (pMbatch) {
      float * vEmbd = fMtmdBatchGetOutputEmbd(pMbatch.get(), vChunk.get());
      if (vEmbd) {
        struct cb_data_t {
          common_speculative * vSpec;
          dgs_context *        vCtx;
        } vCbData = { pSlot.vSpec, pSlot.vCtxTgt };

        static auto vCb = [](const mtmd_helper_embd_batch * pB, void * pUserData) {
          const auto * data = static_cast<cb_data_t *>(pUserData);

          common_batch vBatch(data->vCtx);
          for (int32_t vI = 0; vI < pB->vNTokens; ++vI) {
            dgs_pos aPos[GGML_MROPE_SECTIONS] = { 0, 0, 0, 0 };
            for (int32_t vJ = 0; vJ < pB->vNPos; ++vJ) {
              aPos[vJ] = pB->vPos[vJ * pB->vNTokens + vI];
            }
            vBatch.mAddEmbd({ pB->vEmbd + (size_t) vI * pB->vNEmbd, 1, (size_t) pB->vNEmbd }, aPos, pB->vSeqId, false);
          }

          return fCommonSpeculativeProcess(data->vSpec, vBatch) ? 0 : 1;
        };

        dgs_pos vNewNPast;  // unused for now
        vRes = fMtmdHelperDecodeImageChunk(vMctx, pSlot.vCtxTgt, vChunk.get(), vEmbd, pSlot.vPrompt.vTokens.mPosNext(),
                                           pSlot.vId, fDgsNBatch(pSlot.vCtxTgt), &vNewNPast, vCb, &vCbData);
        if (vRes != 0) {
          SLT_ERR(pSlot, "failed to decode mtmd chunk, idx = %zu, res = %d\n", pIdx, vRes);
          return -1;
        }
        pNTokensOut = fMtmdInputChunkGetNTokens(vChunk.get());
        return 0;  // success
      }
    }
    return 1;  // (non-error) need to create & encode batch
  };

  // if the batch is already exist, try searching & encode
  vRes = vTryDecode();
  if (vRes == 0) {
    return 0;
  }
  if (vRes < 0) {
    // fatal error
    return vRes;
  }

  // otherwise, the batch is either uninitialized or is used up
  // we need to create & encode a new batch
  if (fEncodeMtmdChunk(pSlot, pMbatch, pIdx) != 0) {
    return -1;
  }

  return vTryDecode();
}

// add an encoded mtmd chunk to the batch of the text tokens, instead of decoding it on its own like process_mtmd_chunk()
// embd is the output of the encoder for this chunk
// returns false if the batch is full
static bool fAddMtmdChunk(const server_slot &      pSlot,
                          const mtmd_input_chunk * pChunk,
                          const float *            pEmbd,
                          server_batch &           pBatch) {
  // positions are the ones of mtmd_helper_decode_image_chunk()
  const auto *  vImage   = fMtmdInputChunkGetTokensImage(pChunk);
  const bool    cIsMrope = fMtmdDecodeUseMrope(pSlot.vMctx);
  const size_t  cNTokens = fMtmdInputChunkGetNTokens(pChunk);
  const size_t  cNEmbd   = pBatch.vNEmbd;
  const dgs_pos cPos0    = pSlot.vPrompt.vTokens.mPosNext();

  for (size_t vI = 0; vI < cNTokens; vI++) {
    const dgs_pos cP                        = cPos0 + (dgs_pos) vI;
    dgs_pos       aPos[GGML_MROPE_SECTIONS] = { cP, cP, cP, cP };
    if (cIsMrope && vImage) {
      const mtmd_decoder_pos cRel = fMtmdImageTokensGetDecoderPos(vImage, cPos0, vI);
      aPos[0]                     = cRel.vT;
      aPos[1]                     = cRel.vY;
      aPos[2]                     = cRel.vX;
      aPos[3]                     = cRel.vZ;
    }
    if (!pBatch.mAddEmbd(pSlot.vId, pEmbd + vI * cNEmbd, aPos, pSlot.mNeedEmbd(), /* is_prompt */ true)) {
      return false;
    }
  }
  return true;
}

//
// server_context_impl (private implementation)
//

struct server_context_impl {
  friend struct server_context;

public:
  // only use these pointers outside of this class:
  //  - when not in sleeping state
  //  - and, with thread-safe APIs (e.g., tokenizer calls)
  dgs_model * vModelTgt = nullptr;

  mtmd_context *       vMctx    = nullptr;
  // note: video_params.ffmpeg_bin_dir points into params_base, which outlives this struct
  mtmd_helper_init_opt vInitOpt = fMtmdHelperInitOptDefault();
  const dgs_vocab *    vVocab   = nullptr;

  server_decision_context decision;

  server_queue    vQueueTasks;
  server_response vQueueResults;

  // note: chat_params must not be refreshed upon existing sleeping state
  server_chat_params vChatParams;

  server_state_callback_t vCallbackState = [](server_state, json) -> void {
  };

  server_context_impl() { fMtmdHelperLogSet(fCommonLogDefaultCallback, nullptr); }

  json mGetMonitorSnapshot() const {
    std::lock_guard<std::mutex> vLock(vMonitorMutex);
    return dMonitorSnapshot;
  }

  ~server_context_impl() {
    if (!vSleeping) {
      // destroy() is already called when entering sleeping state
      // we don't call it again here to avoid double free
      mDestroy();
    }
  }

  server_metrics mGetMetrics() const { return vMetrics; }

  void mResetMetricsBucket() { vMetrics.mResetBucket(); }

private:
  // note: accessing these fields outside of this class is not thread-safe
  // use server_context methods instead

  common_params vParamsBase;

  // note: keep these alive - they determine the lifetime of the model, context, etc.
  common_init_result_ptr vDgsInit;

  dgs_context * vCtxTgt = nullptr;

  server_batch vBatch;

  dgs_model *   vModelDft = nullptr;
  dgs_context * vCtxDft   = nullptr;

  common_speculative_init_result_ptr vSpecInit;

  common_context_seq_rm_type vCtxTgtSeqRmType = cCommonContextSeqRmTypeNo;
  common_context_seq_rm_type vCtxDftSeqRmType = cCommonContextSeqRmTypeNo;

  common_speculative_ptr vSpec;

  bool vAddBosToken = true;

  int32_t vNCtx;  // total context for all clients / slots

  // set to dgs_model_n_swa(model)
  // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
  int32_t vNSwa;

  // slots / clients
  std::vector<server_slot> lSlots;

  int vTrace      = 0;  // env: DGS_TRACE
  int vSlotsDebug = 0;  // env: DGS_SERVER_SLOTS_DEBUG
  int vSlotsNDiff = 0;  // env: DGS_SERVER_SLOTS_N_DIFF

  int vNEmptyConsecutive = 0;

  std::unique_ptr<server_prompt_cache> vPromptCache;

  server_metrics vMetrics;

  // queued prompt stats - dgs_decode() is async, so the timing is only valid after a sync
  // note: kept out of server_metrics, which is copied as-is into the task result
  int64_t  vTDecodeStart  = 0;  // start of the last submitted decode
  int64_t  vTPromptStart  = 0;  // start of the oldest queued prompt decode
  uint64_t vNPromptQueued = 0;

  json vJsonUiSettings = json::mObject();

  // Necessary similarity of prompt for slot selection
  float vSlotPromptSimilarity = 0.0f;

  std::string           vModelName;     // name of the loaded model, to be used by API
  std::set<std::string> sModelAliases;  // additional names for the model
  std::set<std::string> sModelTags;     // informational tags

  bool vSleeping = false;

  int64_t vTLastLoadProgressMs = 0;

  // memory monitoring: the snapshot is built on the main loop and copied by the HTTP threads
  mutable std::mutex vMonitorMutex;
  json               dMonitorSnapshot = json{
                  { "state", "loading" }
  };
  int64_t                 vMonitorRefreshMs = 0;
  dgs_model_mtp_breakdown dMonitorWeights;          // MTP weights of model_tgt, fixed after loading
  bool                    vMonitorSpecMtp = false;  // ctx_dft is an MTP context (its own weights are MTP weights)

  void mPublishMonitorSnapshot(json pDSnapshot) {
    std::lock_guard<std::mutex> vLock(vMonitorMutex);
    dMonitorSnapshot = std::move(pDSnapshot);
  }

  // Memory of the loaded contexts by purpose, split between the GPUs and the RAM. Only the main
  // loop may call it: it reads the contexts and the slots.
  json mBuildMonitorSnapshot() const {
    std::map<ggml_backend_buffer_type_t, std::map<std::string, size_t>> dSegments;
    const auto fAdd = [&dSegments](ggml_backend_buffer_type_t pType, const char * pKey, size_t pBytes) {
      if (pType != nullptr && pBytes > 0) {
        dSegments[pType][pKey] += pBytes;
      }
    };

    for (const auto & [vType, vData] : fDgsGetMemoryBreakdown(vCtxTgt)) {
      size_t     vMtp   = 0;
      const auto vSplit = dMonitorWeights.find(vType);
      if (vSplit != dMonitorWeights.end()) {
        vMtp = std::min(vSplit->second, vData.vModel);
      }
      fAdd(vType, "mtp_weights", vMtp);
      fAdd(vType, "dense_weights", vData.vModel - vMtp);
      fAdd(vType, "kv_cache", vData.vContext);
      fAdd(vType, "compute", vData.vCompute);
    }

    if (vCtxDft != nullptr) {
      // an MTP context on the target model shares its weights; a separate GGUF brings its own
      const bool vOwnWeights = fDgsGetModel(vCtxDft) != vModelTgt;
      for (const auto & [vType, vData] : fDgsGetMemoryBreakdown(vCtxDft)) {
        if (vOwnWeights) {
          fAdd(vType, vMonitorSpecMtp ? "mtp_weights" : "draft_weights", vData.vModel);
        }
        fAdd(vType, "draft_kv_cache", vData.vContext);
        fAdd(vType, "compute", vData.vCompute);
      }
    }

    // devices of the models first, in their order, then any other device that holds a buffer
    std::vector<ggml_backend_dev_t> lDevices;
    const auto                      fAddDevice = [&lDevices](ggml_backend_dev_t pDevice) {
      if (pDevice != nullptr && std::find(lDevices.begin(), lDevices.end(), pDevice) == lDevices.end()) {
        lDevices.push_back(pDevice);
      }
    };
    for (int vIndex = 0; vIndex < fDgsModelNDevices(vModelTgt); ++vIndex) {
      fAddDevice(fDgsModelGetDevice(vModelTgt, vIndex));
    }
    if (vModelDft != nullptr) {
      for (int vIndex = 0; vIndex < fDgsModelNDevices(vModelDft); ++vIndex) {
        fAddDevice(fDgsModelGetDevice(vModelDft, vIndex));
      }
    }

    json                               dHostSegments = json::mObject();
    std::map<ggml_backend_dev_t, json> dDeviceSegments;
    for (const auto & [vType, dKeys] : dSegments) {
      ggml_backend_dev_t               vDevice = ggml_backend_buft_get_device(vType);
      const enum ggml_backend_dev_type vDeviceType =
        vDevice != nullptr ? ggml_backend_dev_type(vDevice) : GGML_BACKEND_DEVICE_TYPE_CPU;
      // pinned host buffers of a GPU are RAM too
      const bool vHost = ggml_backend_buft_is_host(vType) ||
                         (vDeviceType != GGML_BACKEND_DEVICE_TYPE_GPU && vDeviceType != GGML_BACKEND_DEVICE_TYPE_IGPU);
      if (!vHost) {
        fAddDevice(vDevice);
      }
      json & dTarget = vHost ? dHostSegments : dDeviceSegments[vDevice];
      if (!dTarget.is_object()) {
        dTarget = json::mObject();
      }
      for (const auto & [vKey, vBytes] : dKeys) {
        dTarget[vKey] = dTarget.value(vKey, (uint64_t) 0) + (uint64_t) vBytes;
      }
    }

    // saved prompts (--cache-ram) and the context checkpoints of the slots live in RAM
    size_t vPromptBytes = vPromptCache ? vPromptCache->size() : 0;
    for (const server_slot & vSlot : lSlots) {
      for (const auto & vCheckpoint : vSlot.vPrompt.lCheckpoints) {
        vPromptBytes += vCheckpoint.size();
      }
    }
    if (vPromptBytes > 0) {
      dHostSegments["prompt_cache"] = dHostSegments.value("prompt_cache", (uint64_t) 0) + (uint64_t) vPromptBytes;
    }

    // where the kernel driver really keeps this process's GPU memory (VRAM or system memory through GTT)
    const json dResidency = fMonitorDrmResidency();

    json aDevices = json::mArray();
    for (ggml_backend_dev_t vDevice : lDevices) {
      const enum ggml_backend_dev_type vDeviceType = ggml_backend_dev_type(vDevice);
      if (vDeviceType != GGML_BACKEND_DEVICE_TYPE_GPU && vDeviceType != GGML_BACKEND_DEVICE_TYPE_IGPU) {
        continue;
      }
      ggml_backend_dev_props vProps;
      ggml_backend_dev_get_props(vDevice, &vProps);
      const auto        vSegments = dDeviceSegments.find(vDevice);
      const std::string vId       = vProps.device_id != nullptr ? vProps.device_id : ggml_backend_dev_name(vDevice);
      json              dDevice   = {
        { "id",          vId                                                                      },
        { "name",        ggml_backend_dev_name(vDevice)                                           },
        { "description", ggml_backend_dev_description(vDevice)                                    },
        { "total_bytes", (uint64_t) vProps.memory_total                                           },
        { "free_bytes",  (uint64_t) vProps.memory_free                                            },
        { "segments",    vSegments != dDeviceSegments.end() ? vSegments->second : json::mObject() },
      };
      const std::string vKey = fMonitorDeviceKey(vId);
      if (dResidency.contains(vKey)) {
        dDevice["residency"] = dResidency.at(vKey);
      }
      aDevices.push_back(std::move(dDevice));
    }

    return json{
      { "state",     "ready"                           },
      { "model",     vModelName                        },
      { "rss_bytes", (uint64_t) fMonitorProcessRss()   },
      { "devices",   aDevices                          },
      { "host",      { { "segments", dHostSegments } } },
    };
  }

  // Rebuilds the snapshot at most once per second while the slots work, and always when they are
  // idle (the end of a request saves its prompt and may leave the last second unpublished).
  void mRefreshMonitorSnapshot(bool pForce) {
    if (vCtxTgt == nullptr) {
      return;
    }
    const int64_t vNow = ggml_time_ms();
    if (!pForce && vNow - vMonitorRefreshMs < 1000) {
      return;
    }
    vMonitorRefreshMs = vNow;
    try {
      json dSnapshot = mBuildMonitorSnapshot();
      mPublishMonitorSnapshot(std::move(dSnapshot));
    } catch (const std::exception & vE) {
      SRV_WRN("failed to update the memory monitor: %s\n", vE.what());
    }
  }

  void mDestroy() {
    vSpec.reset();
    vSpecInit.reset();

    vCtxDft   = nullptr;
    vModelDft = nullptr;

    vDgsInit.reset();

    vCtxTgt   = nullptr;
    vModelTgt = nullptr;

    fMtmdFree(vMctx);
    vMctx = nullptr;
  }

  void mHandleSleepingState(bool pNewState) {
    GGML_ASSERT(vSleeping != pNewState);
    if (pNewState) {
      if (vCallbackState) {
        vCallbackState(cServerStateSleeping, {});
        // note: for sleeping == false, event is emitted by load_model()
      }
      SRV_INF("%s", "server is entering sleeping state\n");
      mDestroy();
      mPublishMonitorSnapshot(json{
        { "state", "sleeping" },
        { "model", vModelName }
      });
    } else {
      SRV_INF("%s", "server is exiting sleeping state\n");
      if (!mLoadModel(vParamsBase)) {
        GGML_ABORT("failed to reload model after sleeping");
      }
    }
    vSleeping = pNewState;
  }

  struct load_progress_data {
    server_context_impl *    vCtx;
    std::string              stage;
    std::vector<std::string> lStages;
    int64_t                  vTLastLoadProgressMs = 0;

    load_progress_data(server_context_impl * pCtx, const std::string & stage) : vCtx(pCtx), stage(stage) {}
  };

  static bool mLoadProgressCallback(float pProgress, void * pUserData) {
    auto * vD = static_cast<load_progress_data *>(pUserData);
    GGML_ASSERT(vD);
    // always emit the first and final sample; throttle the rest to one per 200ms
    {
      auto &        vTLast     = vD->vTLastLoadProgressMs;
      const int64_t cTNow      = ggml_time_ms();
      const bool    cFirst     = vTLast == 0;
      const bool    cDone      = pProgress >= 1.0f;
      const bool    cThrottled = !cFirst && !cDone && (cTNow - vTLast) < 200;
      if (cThrottled) {
        return true;
      }
      vTLast = cTNow;
    }
    if (vD->vCtx->vCallbackState) {
      vD->vCtx->vCallbackState(cServerStateLoading, {
                                                      { "stages",  vD->lStages },
                                                      { "current", vD->stage   },
                                                      { "value",   pProgress   },
      });
    }
    return true;
  }

  // load the model and initialize dgs_context
  // this may also be called to resume from sleeping state
  bool mLoadModel(common_params & pParams) {
    load_progress_data vLoadProgressText(this, "text_model");
    load_progress_data vLoadProgressMmproj(this, "mmproj_model");
    load_progress_data vLoadProgressSpec(this, "spec_model");

    const bool cIsResume = vSleeping;

    vParamsBase                    = pParams;
    const auto cOutputLimits       = fServerOutputLimits(vParamsBase);
    vParamsBase.vNOutputsMax       = cOutputLimits.vTotal;
    vParamsBase.vNOutputsMaxPerSeq = cOutputLimits.vPerSeq;

    const bool cHasMmproj = !pParams.vMmproj.vPath.empty();
    const bool cHasDraft  = pParams.vSpeculative.mHasDft();
    const bool cSpecMtp   = std::find(vParamsBase.vSpeculative.lTypes.begin(), vParamsBase.vSpeculative.lTypes.end(),
                                      cCommonSpeculativeTypeDraftMtp) != vParamsBase.vSpeculative.lTypes.end();
    const bool cHasSpec   = cHasDraft || cSpecMtp;

    if (vCallbackState) {
      std::vector<std::string> lStages = { "text_model" };
      if (cHasSpec) {
        lStages.push_back("spec_model");
      }
      if (cHasMmproj) {
        lStages.push_back("mmproj_model");
      }
      vLoadProgressText.lStages   = lStages;
      vLoadProgressMmproj.lStages = lStages;
      vLoadProgressSpec.lStages   = lStages;

      // trigger 0% progress
      mLoadProgressCallback(0.0f, &vLoadProgressText);
    }

    SRV_INF("loading model '%s'\n", pParams.vModel.mGetName().c_str());
    SRV_TRC("local path '%s'\n", pParams.vModel.vPath.c_str());

    std::string &       vMmprojPath = vParamsBase.vMmproj.vPath;
    mtmd_context_params vMparams    = fMtmdContextParamsDefault();
    if (cHasMmproj) {
      vMparams.vUseGpu                   = vParamsBase.vMmprojUseGpu;
      vMparams.vDevice                   = vParamsBase.vMmprojDevice;
      vMparams.vPrintTimings             = false;
      vMparams.vNThreads                 = vParamsBase.vCpuparams.vNThreads;
      vMparams.vFlashAttnType            = vParamsBase.vFlashAttnType;
      vMparams.vWarmup                   = vParamsBase.vWarmup;
      vMparams.vImageMinTokens           = vParamsBase.vImageMinTokens;
      vMparams.vImageMaxTokens           = vParamsBase.vImageMaxTokens;
      vMparams.vBatchMaxTokens           = vParamsBase.vMtmdBatchMaxTokens;
      vMparams.vMediaMarker              = fGetMediaMarker();
      // progress callback
      vMparams.vProgressCallback         = mLoadProgressCallback;
      vMparams.vProgressCallbackUserData = &vLoadProgressMmproj;
    }

    // get the memory usage of mmproj, also used to check image_max_tokens against n_ubatch
    mtmd_memory_usage vMmprojUsage    = { {}, -1, false };
    int64_t           vMmprojUsageTUs = 0;
    if (cHasMmproj) {
      int64_t vTStart = ggml_time_us();
      vMmprojUsage    = fMtmdGetMemoryUsage(vMmprojPath.c_str(), vMparams);
      vMmprojUsageTUs = ggml_time_us() - vTStart;
    }

    // optionally fit mmproj memory usage
    if (cHasMmproj && vParamsBase.vFitParams) {
      const auto &  dMmprojMem = vMmprojUsage.dBackendMemUsage;
      const int64_t cTElapsed  = vMmprojUsageTUs;
      if (!dMmprojMem.empty()) {
        size_t vTotal = 0;
        for (auto & [dev, size] : dMmprojMem) {
          vTotal += size;
        }
        SRV_TRC("[mtmd] estimated worst-case memory usage of mmproj is %.2f MiB (took %.2f ms)\n",
                vTotal / (1024.0 * 1024.0), cTElapsed / 1000.0);
        GGML_ASSERT(!vParamsBase.lFitParamsTarget.empty());
        for (auto & [dev, size] : dMmprojMem) {
          for (size_t vI = 0; vI < ggml_backend_dev_count(); vI++) {
            if (ggml_backend_dev_get(vI) == dev) {
              if (vI < vParamsBase.lFitParamsTarget.size()) {
                SRV_DBG("[mtmd] adding %.2f MiB to fit_params_target for device %s\n", size / (1024.0 * 1024.0),
                        ggml_backend_dev_name(dev));
                vParamsBase.lFitParamsTarget[vI] += size;
              }
              break;
            }
          }
        }
      } else {
        SRV_ERR("%s", "[mtmd] failed to get memory usage of mmproj\n");
      }
    }

    // note: the draft / MTP context is fitted together with the target model, see common_fit_extra_model

    // attach a progress callback
    {
      vParamsBase.vLoadProgressCallback         = mLoadProgressCallback;
      vParamsBase.vLoadProgressCallbackUserData = &vLoadProgressText;
    }

    vDgsInit = fCommonInitFromParams(vParamsBase);

    vModelTgt = vDgsInit->mModel();
    vCtxTgt   = vDgsInit->mContext();

    if (vModelTgt == nullptr) {
      SRV_ERR("failed to load model, '%s'\n", vParamsBase.vModel.vPath.c_str());
      return false;
    }

    if (vCtxTgt == nullptr) {
      SRV_ERR("failed to create_context with model '%s'\n", vParamsBase.vModel.vPath.c_str());
      return false;
    }

    vVocab = fDgsModelGetVocab(vModelTgt);

    try {
      decision.init(vModelTgt);
    } catch (const std::exception & vE) {
      SRV_ERR("failed to init decision model: %s\n", vE.what());
      return false;
    }

    vNCtx = fDgsNCtx(vCtxTgt);

    vAddBosToken = fDgsVocabGetAddBos(vVocab);

    if (cHasSpec) {
      // spec_mtp doesn't use load a model internally, so we report 0.0 and 1.0 manually
      mLoadProgressCallback(0.0f, &vLoadProgressSpec);
      vLoadProgressSpec.vTLastLoadProgressMs = 0;  // reset so internal cbs aren't delayed

      {
        common_params vParamsDft = fCommonBaseParamsToSpeculative(vParamsBase);

        // progress callback
        vParamsDft.vLoadProgressCallback         = mLoadProgressCallback;
        vParamsDft.vLoadProgressCallbackUserData = &vLoadProgressSpec;

        vSpecInit = fCommonSpeculativeInitFromParams(vParamsDft, vModelTgt, vCtxTgt);
        vModelDft = vSpecInit->mModel();
        vCtxDft   = vSpecInit->mContext();

        if (cHasDraft && vModelDft == nullptr) {
          SRV_ERR("failed to load draft model, '%s'\n", vParamsDft.vModel.vPath.c_str());
          return false;
        }

        if (vCtxDft == nullptr) {
          SRV_ERR("%s", "failed to create MTP context\n");
          return false;
        }

        vParamsBase.vSpeculative.vDraft.vCtxTgt = vCtxTgt;
        vParamsBase.vSpeculative.vDraft.vCtxDft = vCtxDft;
      }

      mLoadProgressCallback(1.0f, &vLoadProgressSpec);
    }

    if (cHasMmproj) {
      if (vCallbackState) {
        vCallbackState(cServerStateLoading, {
                                              { "stage", "mmproj_model" }
        });
      }

      if (!cIsResume) {
        fMtmdHelperLogSet(fCommonLogDefaultCallback, nullptr);
      }

      // the image must fit in one ubatch if the model needs non-causal attention on it
      // a non-causal model also has the text of the prompt in that ubatch, leave half of it to the text
      {
        const int  cNUbatch = fDgsNUbatch(vCtxTgt);
        const bool cIsMixed = mUseMixedBatch();
        if (cIsMixed || vMmprojUsage.vUseNonCausal) {
          const int cNMax = cIsMixed ? cNUbatch / 2 : cNUbatch;
          if (vMmprojUsage.vImageMaxTokens > cNMax) {
            SRV_WRN(
              "cap image_max_tokens (original=%d) to %d (n_ubatch = %d) because %s\n", vMmprojUsage.vImageMaxTokens,
              cNMax, cNUbatch,
              cIsMixed ? "the model processes the prompt in one ubatch" : "model needs non-causal attention on image");
            SRV_WRN("%s\n", "increase n_ubatch (-ub) to increase vision token budget");
            vMparams.vImageMaxTokens = cNMax;
            vMparams.vImageMinTokens = std::min(vMparams.vImageMinTokens, cNMax);
          }
        }
      }

      vMctx = fMtmdInitFromFile(vMmprojPath.c_str(), vModelTgt, vMparams);
      if (vMctx == nullptr) {
        SRV_ERR("failed to load multimodal model, '%s'\n", vMmprojPath.c_str());
        return false;
      }
      SRV_INF("loaded multimodal model, '%s'\n", vMmprojPath.c_str());

      vInitOpt.vVideoParams.vFpsTarget           = vParamsBase.vVideoFps;
      vInitOpt.vVideoParams.vTimestampIntervalMs = vParamsBase.vVideoTimestampIntervalMs;
      vInitOpt.vVideoParams.vFfmpegBinDir =
        vParamsBase.vVideoFfmpegBinDir.empty() ? nullptr : vParamsBase.vVideoFfmpegBinDir.c_str();

      if (vParamsBase.vCtxShift) {
        vParamsBase.vCtxShift = false;
        SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
      }

      if (vParamsBase.vNCacheReuse) {
        vParamsBase.vNCacheReuse = 0;
        SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
      }
    }

    if (!fDgsMemoryCanShift(fDgsGetMemory(vCtxTgt))) {
      if (vParamsBase.vCtxShift) {
        vParamsBase.vCtxShift = false;
        SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
      }

      if (vParamsBase.vNCacheReuse) {
        vParamsBase.vNCacheReuse = 0;
        SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
      }
    }

    if (fDgsModelNSwa(vModelTgt) == 0) {
      if (vParamsBase.vSwaFull) {
        vParamsBase.vSwaFull = false;
        SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
      }
    }

    vNSwa = vParamsBase.vSwaFull ? 0 : fDgsModelNSwa(vModelTgt);

    // Necessary similarity of prompt for slot selection
    vSlotPromptSimilarity = vParamsBase.vSlotPromptSimilarity;

    const int cNCtxTrain = fDgsModelNCtxTrain(vModelTgt);

    {
      // note: the capping itself is done in n_ctx_slot(), here we only report it
      const int cNCtxSeq = fDgsNCtxSeq(vCtxTgt);

      if (vParamsBase.vKvUnifiedPerSlot > 0) {
        if (cNCtxSeq > vParamsBase.vKvUnifiedPerSlot) {
          SRV_INF("capping per-slot context (%d) to --kv-unified-per-slot (%d)\n", cNCtxSeq,
                  vParamsBase.vKvUnifiedPerSlot);
        } else if (vParamsBase.vKvUnifiedPerSlot > cNCtxSeq) {
          // cap is above the per-slot pool capacity, so it can never bind
          SRV_WRN(
            "--kv-unified-per-slot (%d) exceeds the per-slot pool capacity (%d) - cap has no effect, "
            "slots are limited to %d (raise the KV pool with -c, or unset -c to size it to "
            "n_parallel * kv_unified_per_slot)\n",
            vParamsBase.vKvUnifiedPerSlot, cNCtxSeq, cNCtxSeq);
        }
      }

      const int cNCtxCapped =
        vParamsBase.vKvUnifiedPerSlot > 0 ? std::min(cNCtxSeq, vParamsBase.vKvUnifiedPerSlot) : cNCtxSeq;

      if (cNCtxCapped > cNCtxTrain) {
        SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n", cNCtxCapped,
                cNCtxTrain);
      }
    }

    lSlots.clear();

    vCtxTgtSeqRmType = fCommonContextCanSeqRm(vCtxTgt);
    if (vCtxTgtSeqRmType == cCommonContextSeqRmTypeNo) {
      SRV_WRN("%s", "speculative decoding not supported by this context\n");
    }

    if (vCtxTgtSeqRmType == cCommonContextSeqRmTypeFull) {
      SRV_TRC("%s", "speculative decoding will use checkpoints\n");
    }

    // setup slots
    SRV_INF("initializing, n_slots = %d, n_ctx_slot = %d, kv_unified = '%s'\n", vParamsBase.vNParallel, mNCtxSlot(),
            vParamsBase.vKvUnified ? "true" : "false");

    // initialize slots
    for (int vI = 0; vI < vParamsBase.vNParallel; vI++) {
      lSlots.emplace_back();
    }

    // try speculative decoding
    if (vCtxTgtSeqRmType != cCommonContextSeqRmTypeNo) {
      try {
        vSpec.reset(fCommonSpeculativeInit(vParamsBase.vSpeculative, vParamsBase.vNParallel));
      } catch (const std::exception & vE) {
        SRV_ERR("failed to initialize speculative decoding context: %s\n", vE.what());
        if (vParamsBase.vSpeculative.mHasSynth()) {
          return false;
        }
      }
    }

    if (vCtxDft) {
      vCtxDftSeqRmType = fCommonContextCanSeqRm(vCtxDft);
    }

    if (vSpec) {
      SRV_TRC("%s", "speculative decoding context initialized\n");
    } else {
      vSpecInit.reset();
      vCtxDft   = nullptr;
      vModelDft = nullptr;
    }

    if (!vSpec && vParamsBase.vSpeculative.mHasSynth()) {
      SRV_ERR("%s", "synthetic acceptance requires an initialized speculative decoding context\n");
      return false;
    }

    for (int vI = 0; vI < vParamsBase.vNParallel; vI++) {
      server_slot & vSlot = lSlots[vI];

      vSlot.vId     = vI;
      vSlot.vCtxTgt = vCtxTgt;
      vSlot.vCtxDft = vCtxDft;
      vSlot.vMem.init(vCtxTgt, vCtxDft);
      vSlot.vSpec = vSpec.get();
      vSlot.vNCtx = mNCtxSlot();

      vSlot.vMctx                    = vMctx;
      vSlot.vPrompt.vTokens.has_mtmd = vMctx != nullptr;

      SLT_TRC(vSlot, "new slot, n_ctx = %d\n", vSlot.vNCtx);

      vSlot.vCallbackOnRelease = [this](int pIdSlot) {
        vQueueTasks.mPopDeferredTask(pIdSlot);
      };

      vSlot.vCallbackOnReset = [this](const server_slot & pSlot) {
        // flush the generated token stats before reset()
        if (pSlot.vStats.vNGen > 0) {
          mMetricsOnPrediction(pSlot);
        }
      };

      vSlot.reset();
    }

    {
      const char * vDgsTrace = getenv("DGS_TRACE");
      vTrace                 = vDgsTrace ? atoi(vDgsTrace) : 0;

      if (vTrace) {
        SRV_WRN("DGS_TRACE = %d\n", vTrace);
      }
    }

    {
      const char * vDgsServerSlotsDebug = getenv("DGS_SERVER_SLOTS_DEBUG");
      vSlotsDebug                       = vDgsServerSlotsDebug ? atoi(vDgsServerSlotsDebug) : 0;

      if (vSlotsDebug) {
        SRV_WRN("DGS_SERVER_SLOTS_DEBUG = %d\n", vSlotsDebug);
      }
    }

    {
      const char * vDgsServerSlotsNDiff = getenv("DGS_SERVER_SLOTS_N_DIFF");
      vSlotsNDiff                       = vDgsServerSlotsNDiff ? atoi(vDgsServerSlotsNDiff) : 0;

      if (vSlotsNDiff) {
        SRV_WRN("DGS_SERVER_SLOTS_N_DIFF = %d\n", vSlotsNDiff);
      }
    }

    // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
    // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
    {
      const int32_t cNBatch = fDgsNBatch(vCtxTgt);
      const int32_t cNEmbd  = fDgsModelNEmbdInp(vModelTgt);
      vBatch.init(vCtxTgt, std::max(cNBatch, vParamsBase.vNParallel), cNEmbd);
    }

    if (vParamsBase.vCacheRamMib != 0) {
      if (vParamsBase.vCacheRamMib < 0) {
        SRV_TRC("prompt cache is enabled, size limit: %s\n", "no limit");
      } else {
        SRV_TRC("prompt cache is enabled, size limit: %d MiB\n", vParamsBase.vCacheRamMib);
      }
      SRV_TRC("%s", "use `--cache-ram 0` to disable the prompt cache\n");

      vPromptCache = std::make_unique<server_prompt_cache>(vParamsBase.vCacheRamMib, vNCtx);
    } else {
      SRV_TRC("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
    }
    SRV_TRC("%s", "for more info see doc/MANUAL.md\n");

    if (vParamsBase.vNCtxCheckpoints > 0) {
      SRV_TRC("context checkpoints enabled, max = %d, min spacing = %d\n", vParamsBase.vNCtxCheckpoints,
              vParamsBase.vCheckpointMinStep);
    } else {
      SRV_TRC("%s", "context checkpoints disabled\n");
    }

    if (!vParamsBase.vModel.mGetName().empty()) {
      vModelName = vParamsBase.vModel.mGetName();
    } else {
      // fallback: derive model name from file name
      auto vModelPath = std::filesystem::path(vParamsBase.vModel.vPath);
      vModelName      = vModelPath.filename().string();
    }

    sModelAliases = vParamsBase.sModelAlias;
    sModelTags    = vParamsBase.sModelTags;

    dMonitorWeights         = fDgsModelGetMtpBreakdown(vModelTgt);
    const auto & lSpecTypes = vParamsBase.vSpeculative.lTypes;
    vMonitorSpecMtp =
      std::find(lSpecTypes.begin(), lSpecTypes.end(), cCommonSpeculativeTypeDraftMtp) != lSpecTypes.end();
    mRefreshMonitorSnapshot(true);

    // propagate new defaults back to caller
    pParams = vParamsBase;

    if (!cIsResume) {
      return init();
    }

    if (vCallbackState) {
      vCallbackState(cServerStateReady, {});
    }

    return true;
  }

  // unlike load_model(), this is only called once during initialization
  bool init() {
    GGML_ASSERT(vCtxTgt != nullptr);
    GGML_ASSERT(vModelTgt != nullptr);

    GGML_ASSERT(!vSleeping);

    // wiring up server queues
    vQueueTasks.mOnNewTask(
      [this](server_task && pTask, bool pIsYielding) { return mProcessSingleTask(std::move(pTask), pIsYielding); });
    vQueueTasks.mOnUpdateSlots([this]() {
      mUpdateSlots();
      mRefreshMonitorSnapshot(
        std::none_of(lSlots.begin(), lSlots.end(), [](const server_slot & pSlot) { return pSlot.mIsProcessing(); }));
    });
    vQueueTasks.mOnSleepingState([this](bool pSleeping) { mHandleSleepingState(pSleeping); });

    vMetrics.init();

    if (vParamsBase.vCacheIdleSlots) {
      if (vParamsBase.vCacheRamMib == 0) {
        SRV_WRN("%s", "--cache-idle-slots requires --cache-ram, disabling\n");
        vParamsBase.vCacheIdleSlots = false;
      } else {
        if (vParamsBase.vKvUnified) {
          SRV_TRC("%s", "idle slots will be saved to prompt cache and cleared upon starting a new task\n");
        } else {
          // without a unified KV cache, clearing a slot frees no reusable room, so we only
          // publish a RAM-cache copy of idle slots (their KV stays in VRAM) [TAG_IDLE_SLOT_CLEAR]
          SRV_TRC("%s", "idle slots will be saved to prompt cache upon starting a new task\n");
        }
        SRV_DBG("%s", "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__\n");
      }
    }

    {
      const std::string & vCfg = vParamsBase.vUiConfigJson;
      if (!vCfg.empty()) {
        try {
          json vJsonSettings = json::mParse(vCfg);
          vJsonUiSettings    = vJsonSettings;
        } catch (const std::exception & vE) {
          SRV_ERR("%s: failed to parse UI config: %s\n", __func__, vE.what());
          return false;
        }
      }
    }

    // populate chat template params
    {
      common_chat_templates_ptr vChatTemplates;
      bool                      vEnableThinking = false;

      try {
        vChatTemplates = fCommonChatTemplatesInit(vModelTgt, vParamsBase.vChatTemplate);

        SRV_TRC(
          "%s: chat template, example_format: '%s'\n", __func__,
          fCommonChatFormatExample(vChatTemplates.get(), vParamsBase.vUseJinja, vParamsBase.dDefaultTemplateKwargs)
            .c_str());

        // thinking is enabled if:
        // 1. It's not explicitly disabled via --reasoning off
        // 2. The chat template supports it
        const bool cTemplateSupportsThinking =
          vParamsBase.vUseJinja && fCommonChatTemplatesSupportEnableThinking(vChatTemplates.get());
        vEnableThinking = vParamsBase.vEnableReasoning != 0 && cTemplateSupportsThinking;
        SRV_TRC("%s: chat template, thinking = %d\n", __func__, vEnableThinking);
      } catch (const std::exception & vE) {
        SRV_ERR("%s: chat template parsing error: %s\n", __func__, vE.what());
        SRV_ERR("%s: provide a complete Qwen Jinja template via --chat-template-file\n", __func__);
        return false;
      }

      // IMPORTANT: chat_params is reused across sleeping / resuming states,
      //            never store dgs_context/dgs_model pointers in chat_params,
      //            as they may be invalidated after sleeping
      vChatParams = { /* use_jinja             */ vParamsBase.vUseJinja,
                      /* prefill_assistant     */ vParamsBase.vPrefillAssistant,
                      /* reasoning_format      */ vParamsBase.vReasoningFormat,
                      /* chat_template_kwargs  */ vParamsBase.dDefaultTemplateKwargs,
                      /* tmpls                 */ std::move(vChatTemplates),
                      /* allow_image           */ vMctx ? fMtmdSupportVision(vMctx) : false,
                      /* allow_audio           */ vMctx ? fMtmdSupportAudio(vMctx) : false,
                      /* allow_video           */ vMctx ? fMtmdHelperSupportVideo(vMctx) : false,
                      /* enable_thinking       */ vEnableThinking,
                      /* reasoning_budget      */ vParamsBase.vSampling.vReasoningBudgetTokens,
                      /* reasoning_budget_msg  */ vParamsBase.vSampling.vReasoningBudgetMessage,
                      /* media_path            */ vParamsBase.vMediaPath,
                      /* force_pure_content    */ vParamsBase.vForcePureContentParser };

      {
        auto caps       = fCommonChatTemplatesGetCaps(vChatParams.vTmpls.get());
        auto vIt        = vParamsBase.dDefaultTemplateKwargs.find("preserve_reasoning");
        bool vSupported = caps.at("supports_preserve_reasoning");
        bool vSpecified = vParamsBase.vPreserveReasoningSpecified;
        // note: the kwarg is enabled by default if not specified explicitly, so check the value
        bool vEnabled   = vIt != vParamsBase.dDefaultTemplateKwargs.end() && vIt->second == "true";
        if (vSupported) {
          SRV_TRC("preserve_reasoning kwarg: %s\n",
                  vIt == vParamsBase.dDefaultTemplateKwargs.end() ? "unset (template default)" : vIt->second.c_str());
        } else {
          SRV_TRC("%s", "preserve_reasoning kwarg: not supported by template\n");
        }
        if (vSupported && !vSpecified) {
          SRV_WRN("%s",
                  "chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable "
                  "via --no-reasoning-preserve)\n");
        }
        if (vSupported && !vEnabled) {
          SRV_INF("%s", "chat template supports preserving reasoning, consider enabling it via --reasoning-preserve\n");
        }
        if (!vSupported && vSpecified && vEnabled) {
          SRV_WRN("%s", "chat template does NOT support preserving reasoning, --reasoning-preserve has no effect\n");
        }
      }
    }

    return true;
  }

  server_slot * mGetSlotById(int pIdSlot) {
    // note: allow id_slot to be out of bounds (wrap around)
    pIdSlot = pIdSlot % lSlots.size();

    for (server_slot & vSlot : lSlots) {
      if (vSlot.vId == pIdSlot) {
        return &vSlot;
      }
    }

    return nullptr;
  }

  server_slot * mGetSlotByCmplId(const std::string & pCmplId) {
    if (pCmplId.empty()) {
      return nullptr;
    }

    for (server_slot & vSlot : lSlots) {
      if (vSlot.mIsProcessing() && vSlot.vTask && vSlot.vTask->vParams.vOaicompatCmplId == pCmplId) {
        return &vSlot;
      }
    }

    return nullptr;
  }

  server_slot * mGetAvailableSlot(const server_task & pTask) {
    server_slot * vRet = nullptr;

    bool vUpdateCache = false;

    // if a specific slot is requested, use it (still goes through cache update logic below)
    if (pTask.vIdSlot != -1) {
      vRet = mGetSlotById(pTask.vIdSlot);
      if (vRet) {
        // a busy slot is returned untouched, the caller defers the task
        if (vRet->mIsProcessing()) {
          return vRet;
        }

        SLT_INF(*vRet, "selected slot by id (%d)\n", pTask.vIdSlot);
      }
    }

    // find the slot that has at least n% prompt similarity
    if (vSlotPromptSimilarity != 0.0f) {
      float vFSimBest = 0;

      for (server_slot & vSlot : lSlots) {
        if (pTask.vIdSlot != -1 && vSlot.vId != pTask.vIdSlot) {
          continue;
        }

        // skip the slot if it is not available
        if (vSlot.mIsProcessing()) {
          SLT_TRC(vSlot, " - skipping, is_processing = %d\n", vSlot.mIsProcessing());
          continue;
        }

        const auto & vTokens = vSlot.vPrompt.vTokens;

        // skip the slot if it does not contains cached tokens
        if (vTokens.empty()) {
          SLT_TRC(vSlot, "%s", " - skipping, slot is empty\n");
          continue;
        }

        // fraction of the Longest Common Prefix length with respect to the input prompt length
        const size_t cLcpLen  = vTokens.mGetCommonPrefix(pTask.vTokens);
        const float  cFSimCur = float(cLcpLen) / pTask.vTokens.size();

        SLT_TRC(vSlot, " - checking sim = %.3f (%zu/%zu) > %.3f\n", cFSimCur, cLcpLen, pTask.vTokens.size(),
                vSlotPromptSimilarity);

        // select the current slot if the criteria match
        if (cFSimCur > vFSimBest && cFSimCur > vSlotPromptSimilarity) {
          vFSimBest = cFSimCur;

          vRet = &vSlot;
        }
      }

      if (vRet != nullptr) {
        const float cFKeep = (vFSimBest * pTask.vTokens.size()) / vRet->vPrompt.vTokens.size();

        if (pTask.vIdSlot == -1) {
          SLT_INF(*vRet, "selected slot by LCP similarity, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                  vFSimBest, vSlotPromptSimilarity, cFKeep);
        }

        // if we are about to lose a large portion of the existing context - save it in the prompt cache
        if (cFKeep < 0.5f) {
          vUpdateCache = true;
        }
      }
    }

    // find the slot that has been least recently used
    if (vRet == nullptr) {
      int64_t vTLast = -1;

      for (server_slot & vSlot : lSlots) {
        // skip the slot if it is not available
        if (vSlot.mIsProcessing()) {
          continue;
        }

        // select the current slot if the criteria match
        if (!vRet || vSlot.vTLastUsed <= vTLast) {
          vTLast = vSlot.vTLastUsed;
          vRet   = &vSlot;
        }
      }

      if (vRet != nullptr) {
        SLT_INF(*vRet, "selected slot by LRU, t_last = %" PRId64 "\n", vTLast);

        vUpdateCache = true;
      }
    }

    if (vRet) {
      vUpdateCache = vUpdateCache && vPromptCache;

      // cache prompts only for completion tasks
      vUpdateCache = vUpdateCache && pTask.type == cServerTaskTypeCompletion;

      if (vUpdateCache) {
        SRV_TRC("%s", "updating prompt cache\n");

        const int64_t cTStart = ggml_time_us();

        vRet->mPromptSave(*vPromptCache);

        if (!vRet->mPromptLoad(*vPromptCache, pTask.vTokens)) {
          vRet->mPromptClear();
        }

        vPromptCache->mUpdate();

        SRV_TRC("prompt cache update took %.2f ms\n", (ggml_time_us() - cTStart) / 1000.0);
      }
    }

    return vRet;
  }

  // return true if at least one slot has been cleared
  // TODO: improve logic
  //       - smarter decision which slot to clear (LRU or longest prompt?)
  //       - move slot to level 2 cache instead of removing?
  //       - instead of purging, try to store and resume later?
  bool mTryClearIdleSlots() {
    bool vRes = false;

    if (!vParamsBase.vKvUnified) {
      return vRes;
    }

    for (auto & vSlot : lSlots) {
      if (vSlot.mIsProcessing()) {
        continue;
      }

      if (vSlot.vPrompt.mNTokens() > 0) {
        SRV_WRN("purging slot %d with %zu tokens\n", vSlot.vId, vSlot.vPrompt.vTokens.size());

        vSlot.mPromptClear();

        vRes = true;

        // clear slots one by one
        break;
      }
    }

    return vRes;
  }

  std::vector<common_adapter_lora_info> mConstructLoraList(const std::map<int, float> & pConfig) const {
    std::vector<common_adapter_lora_info> lOutput = vParamsBase.lLoraAdapters;  // copy
    for (size_t vI = 0; vI < lOutput.size(); ++vI) {
      auto vIt = pConfig.find(vI);
      if (vIt != pConfig.end()) {
        lOutput[vI].vScale = vIt->second;
      } else {
        lOutput[vI].vScale = 0.0f;
      }
    }
    return lOutput;
  }

  bool mLaunchSlotWithTask(server_slot & pSlot, server_task && pTask) {
    // process per-request lora adapters
    if (!pTask.vParams.lora.empty()) {
      auto lTaskLoras = mConstructLoraList(pTask.vParams.lora);
      if (!fAreLoraEqual(lTaskLoras, pSlot.lora)) {
        // if lora has changed, check to see if the cache should be cleared
        if (fLoraShouldClearCache(pSlot.lora, lTaskLoras)) {
          SLT_TRC(pSlot, "clearing cache for lora change. %zu loras -> %zu loras\n", pSlot.lora.size(),
                  pTask.vParams.lora.size());
          pSlot.vPrompt.clear();
        } else {
          SLT_TRC(pSlot, "keeping cache for alora. %zu target loras\n", lTaskLoras.size());
        }
        pSlot.lora = lTaskLoras;
      }
    } else {
      pSlot.lora = vParamsBase.lLoraAdapters;
    }

    // if using alora, make sure it's only a single one requested and active
    size_t vAloraInvocationStart = pTask.vTokens.size();
    if (fLoraAllAlora(pSlot.lora)) {
      const auto & lEnabledIds = fLoraGetEnabledIds(pSlot.lora);
      // TODO: This will error out if a user requests two aloras, but only
      // provides the activation string for one. We could, instead search
      // for all requested alora activation strings and then either keep
      // only the last one, or reject if multiple are found.
      if (lEnabledIds.size() != 1) {
        mSendError(pTask, "Cannot run multiple aLoRAs in a single request", cErrorTypeInvalidRequest);
        return false;
      }
      const auto & lora = pSlot.lora[lEnabledIds[0]].vPtr;

      // get the pointer and count for the invocation tokens
      const uint64_t    cNInvocationTokens = fDgsAdapterGetAloraNInvocationTokens(lora);
      const dgs_token * vInvocationTokens  = fDgsAdapterGetAloraInvocationTokens(lora);

      // scan backwards through the prompt tokens to find the last
      // occurrence of the invocation sequence
      int vMatchIdx = static_cast<int>(cNInvocationTokens) - 1;
      for (int vI = pTask.vTokens.size() - 1; vI >= 0; --vI) {
        // the token in this position matches the next token to find in
        // the invocation sequence
        if (pTask.vTokens[vI] == vInvocationTokens[vMatchIdx]) {
          // if it's a full match, we've found the start
          if (vMatchIdx == 0) {
            vAloraInvocationStart = vI;
            break;
          }
          // otherwise, check the next token in the sequence
          --vMatchIdx;
        } else {
          // no match in this position, so start looking over again
          vMatchIdx = static_cast<int>(cNInvocationTokens) - 1;
        }
      }

      // if the activation string is not found, disable the alora
      if (vAloraInvocationStart == pTask.vTokens.size()) {
        SLT_DBG(pSlot, "alora %zu requested, but not found. deactivating\n", lEnabledIds[0]);
        pSlot.lora[lEnabledIds[0]].vScale = 0.0f;
      } else {
        SLT_DBG(pSlot, "alora %zu activated starting at %zu\n", lEnabledIds[0], vAloraInvocationStart);
        pSlot.vAloraInvocationStart = vAloraInvocationStart;
      }
    }

    if (!pTask.vTokens.mValidate(vCtxTgt)) {
      mSendError(pTask, "Prompt contains invalid tokens", cErrorTypeInvalidRequest);
      return false;
    }

    SLT_DBG(pSlot, "launching slot : %s\n", fSafeJsonToStr(pSlot.to_json()).c_str());

    // initialize samplers
    if (pTask.mNeedSampling()) {
      try {
        pSlot.vSmpl.reset(fCommonSamplerInit(vModelTgt, pTask.vParams.vSampling));
      } catch (std::exception & vE) {
        std::string vErrMsg = std::string("Failed to initialize samplers: ") + vE.what();
        mSendError(pTask, vErrMsg, cErrorTypeInvalidRequest);
        return false;
      }

      const bool cNeedPreSampleLogits = pTask.vParams.vSampling.vNProbs > 0 && !pTask.vParams.vPostSamplingProbs;

      bool vUseBackendSampling = pTask.vParams.vSampling.vBackendSampling;

      // TODO: getting pre sampling logits is not yet supported with backend sampling
      vUseBackendSampling &= !cNeedPreSampleLogits;

      // TODO: tmp until backend sampling is fully implemented
      if (vUseBackendSampling) {
        fDgsSetSampler(vCtxTgt, pSlot.vId, fCommonSamplerGet(pSlot.vSmpl.get()));
      } else {
        fDgsSetSampler(vCtxTgt, pSlot.vId, nullptr);
      }

      SLT_TRC(pSlot, "sampler chain: %s\n", fCommonSamplerPrint(pSlot.vSmpl.get()).c_str());
      SLT_TRC(pSlot, "sampler params: \n%s\n", pTask.vParams.vSampling.mPrint().c_str());

      if (vSpec && !fCommonSpeculativeGetSynthProbs(vSpec.get()).empty()) {
        const uint32_t cSeed =
          pTask.vParams.vSampling.vSeed == DGS_DEFAULT_SEED ? std::random_device{}() : pTask.vParams.vSampling.vSeed;
        pSlot.vSpecSynthRng.seed(cSeed);
      }
    } else {
      pSlot.vSmpl.reset();
    }

    // the per-request limit takes priority over the global one
    pSlot.vNPredictMax = pTask.vParams.vNPredict != -1 ? pTask.vParams.vNPredict : vParamsBase.vNPredict;

    pSlot.vTask = std::make_unique<const server_task>(std::move(pTask));

    pSlot.vState = pSlot.vTask->mIsChild() ? cSlotStateWaitOther  // wait for the parent to process prompt
                                             :
                                             cSlotStateStarted;

    // reset server kill-switch counter
    vNEmptyConsecutive = 0;

    SLT_INF(pSlot, "processing task, is_child = %d\n", pSlot.vTask->mIsChild());
    return true;
  }

  bool mProcessToken(completion_token_output & pResult, server_slot & pSlot) {
    // remember which tokens were sampled - used for repetition penalties during sampling
    const std::string cTokenStr = pResult.vTextToSend;
    pSlot.vSampled              = pResult.vTok;

    pSlot.vGenerated.mAppend(cTokenStr, pResult.vTok);
    if (pSlot.vTask->vParams.vReturnTokens) {
      pSlot.lGeneratedTokens.push_back(pResult.vTok);
    }
    pSlot.vHasNextToken = true;

    // check if there is incomplete UTF-8 character at the end
    bool vIncomplete = fValidateUtf8(pSlot.vGenerated.vText) < pSlot.vGenerated.vText.size();

    // search stop word and delete it
    if (!vIncomplete) {
      size_t vPos = std::min(pSlot.vNSentText, pSlot.vGenerated.vText.size());

      const std::string cStrTest  = pSlot.vGenerated.vText.substr(vPos);
      bool              vSendText = true;

      size_t vStopPos = pSlot.mFindStoppingStrings(cStrTest, cTokenStr.size(), true);
      if (vStopPos != std::string::npos) {
        pSlot.vGenerated.mTruncate(vPos + vStopPos);
        vPos = std::min(pSlot.vNSentText, pSlot.vGenerated.vText.size());
      } else if (pSlot.vHasNextToken && !fDgsVocabIsEog(vVocab, pResult.vTok)) {
        vStopPos  = pSlot.mFindStoppingStrings(cStrTest, cTokenStr.size(), false);
        vSendText = vStopPos == std::string::npos;
      }

      // check if there is any token to predict
      if (vSendText) {
        // no send the stop word in the response
        pResult.vTextToSend = pSlot.vGenerated.vText.substr(vPos, std::string::npos);
        pSlot.vNSentText += pResult.vTextToSend.size();
        // add the token to slot queue and cache
      } else {
        pResult.vTextToSend = "";
      }

      pSlot.mAddToken(pResult);
      if (pSlot.vTask->vParams.stream) {
        mSendPartialResponse(pSlot, pResult, false);
      }
    }

    if (vIncomplete) {
      pSlot.vHasNextToken = true;
    }

    // if context shifting is disabled, make sure that we don't run out of context
    if (!vParamsBase.vCtxShift && pSlot.vPrompt.mNTokens() + 1 >= pSlot.vNCtx) {
      pSlot.vTruncated    = true;
      pSlot.vStop         = cStopTypeLimit;
      pSlot.vHasNextToken = false;

      SLT_DBG(pSlot,
              "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_gen = %d, "
              "n_ctx = %d\n",
              pSlot.vPrompt.mNTokens(), pSlot.vTask->mNTokens(), (int) pSlot.vStats.vNGen, pSlot.vNCtx);
    }

    // check the limits
    if (pSlot.vStats.vNGen > 0 && pSlot.vHasNextToken && !pSlot.mHasBudget()) {
      pSlot.vStop         = cStopTypeLimit;
      pSlot.vHasNextToken = false;

      SLT_DBG(pSlot, "stopped by limit, n_gen = %d, n_predict = %d\n", (int) pSlot.vStats.vNGen,
              pSlot.vTask->vParams.vNPredict);
    }

    if (pSlot.vHasNewLine) {
      // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
      if (pSlot.vTask->vParams.vNIndent > 0) {
        // check the current indentation
        // TODO: improve by not doing it more than once for each new line
        if (pSlot.vLastNlPos > 0) {
          size_t vPos = pSlot.vLastNlPos;

          int vNIndent = 0;
          while (vPos < pSlot.vGenerated.vText.size() &&
                 (pSlot.vGenerated.vText[vPos] == ' ' || pSlot.vGenerated.vText[vPos] == '\t')) {
            vNIndent++;
            vPos++;
          }

          if (vPos < pSlot.vGenerated.vText.size() && vNIndent < pSlot.vTask->vParams.vNIndent) {
            pSlot.vStop         = cStopTypeLimit;
            pSlot.vHasNextToken = false;

            // cut the last line
            pSlot.vGenerated.mTruncate(vPos);

            SLT_DBG(pSlot, "stopped by indentation limit, n_gen = %d, n_indent = %d\n", (int) pSlot.vStats.vNGen,
                    vNIndent);
          }
        }

        // find the next new line
        {
          const size_t cPos = pSlot.vGenerated.vText.find('\n', pSlot.vLastNlPos);

          if (cPos != std::string::npos) {
            pSlot.vLastNlPos = cPos + 1;
          }
        }
      }
    }

    // check if there is a new line in the generated text
    if (pResult.vTextToSend.find('\n') != std::string::npos) {
      pSlot.vHasNewLine = true;

      // if we have seen a new line, we stop after a certain time limit, but only upon another new line
      if (pSlot.vTask->vParams.vTMaxPredictMs > 0 && pSlot.vStats.mTGenMs() > pSlot.vTask->vParams.vTMaxPredictMs) {
        pSlot.vStop         = cStopTypeLimit;
        pSlot.vHasNextToken = false;

        SLT_DBG(pSlot, "stopped by time limit, n_gen = %d, t_max_predict_ms = %d ms\n", (int) pSlot.vStats.vNGen,
                (int) pSlot.vTask->vParams.vTMaxPredictMs);
      }
    }

    if (fDgsVocabIsEog(vVocab, pResult.vTok)) {
      pSlot.vStop         = cStopTypeEos;
      pSlot.vHasNextToken = false;

      SLT_DBG(pSlot, "%s", "stopped by EOS\n");
    }

    SLT_DBG(pSlot, "n_gen = %d, n_remaining = %d, next token: %5d '%s'\n", (int) pSlot.vStats.vNGen,
            pSlot.mNRemaining(), pResult.vTok, cTokenStr.c_str());

    return pSlot.vHasNextToken;  // continue
  }

  void mPopulateTokenProbs(const server_slot &       pSlot,
                           completion_token_output & pResult,
                           bool                      pPostSampling,
                           bool                      pSpecial,
                           int                       pIdx) const {
    const size_t cNProbsRequest = pSlot.vTask->vParams.vSampling.vNProbs;

    if (pPostSampling) {
      const auto * vCurP     = fCommonSamplerGetCandidates(pSlot.vSmpl.get(), true);
      const size_t cMaxProbs = vCurP->size;
      const size_t cNProbs   = std::min(cMaxProbs, cNProbsRequest);

      // set probability for sampled token
      for (size_t vI = 0; vI < cMaxProbs; vI++) {
        if (vCurP->data[vI].vId == pResult.vTok) {
          pResult.vProb = vCurP->data[vI].vP;
          break;
        }
      }

      // set probability for top n_probs tokens
      pResult.lProbs.reserve(cNProbs);
      for (size_t vI = 0; vI < cNProbs; vI++) {
        // Some samplers do return 0.0 probabilities, others don't.
        // Filter 0.0 probailities, to ensure the behavior is consistent.
        if (vCurP->data[vI].vP == 0.0) {
          break;
        }

        pResult.lProbs.push_back(
          { vCurP->data[vI].vId, fCommonTokenToPiece(vCtxTgt, vCurP->data[vI].vId, pSpecial), vCurP->data[vI].vP });
      }
    } else {
      std::vector<dgs_token_data> lCur      = fGetTokenProbabilities(vCtxTgt, pIdx, cNProbsRequest);
      const size_t                cMaxProbs = lCur.size();
      const size_t                cNProbs   = std::min(cMaxProbs, cNProbsRequest);

      // set probability for sampled token
      for (size_t vI = 0; vI < cMaxProbs; vI++) {
        // set probability for sampled token
        if (lCur[vI].vId == pResult.vTok) {
          pResult.vProb = lCur[vI].vP;
          break;
        }
      }

      // set probability for top n_probs tokens
      pResult.lProbs.reserve(cNProbs);
      for (size_t vI = 0; vI < cNProbs; vI++) {
        pResult.lProbs.push_back({ lCur[vI].vId, fCommonTokenToPiece(vCtxTgt, lCur[vI].vId, pSpecial), lCur[vI].vP });
      }
    }
  }

  void mSendError(const server_task &   pTask,
                  const std::string &   pError,
                  const enum error_type type = cErrorTypeServer) {
    mSendError(pTask.vId, pError, type);
  }

  void mSendError(const server_slot &   pSlot,
                  const std::string &   pError,
                  const enum error_type type = cErrorTypeServer) {
    mSendError(pSlot.vTask->vId, pError, type, pSlot.vTask->mNTokens(), pSlot.vNCtx);
  }

  void mSendError(const int             pIdTask,
                  const std::string &   pError,
                  const enum error_type type           = cErrorTypeServer,
                  const int32_t         pNPromptTokens = 0,
                  const int32_t         pNCtx          = 0) {
    SRV_ERR("task id = %d, error: %s\n", pIdTask, pError.c_str());

    if (type == cErrorTypeExceedContextSize) {
      GGML_ASSERT(pNCtx > 0 && pNPromptTokens > 0);
    }

    auto vRes            = std::make_unique<server_task_result_error>();
    vRes->vId            = pIdTask;
    vRes->vErrType       = type;
    vRes->vErrMsg        = pError;
    vRes->vNPromptTokens = pNPromptTokens;
    vRes->vNCtx          = pNCtx;

    vQueueResults.mSend(std::move(vRes));
  }

  void mSendPartialResponse(server_slot &                   pSlot,
                            const completion_token_output & pTkn,
                            bool                            pIsProgress,
                            bool                            pIsBegin = false) {
    auto vRes = std::make_unique<server_task_result_cmpl_partial>();

    vRes->vId    = pSlot.vTask->vId;
    vRes->vIndex = pSlot.vTask->vIndex;

    if (pIsProgress) {
      vRes->vIsProgress          = true;
      vRes->vProgress.vTotal     = pSlot.vTask->mNTokens();
      vRes->vProgress.vCache     = pSlot.vStats.vNPromptCached;
      vRes->vProgress.vProcessed = pSlot.vPrompt.vTokens.size();
      vRes->vProgress.vTimeMs    = pSlot.vStats.mTElapsedUs() / 1000;
    }
    if (pIsBegin) {
      vRes->vIsBegin = true;
    } else {
      vRes->vContent = pSlot.vGenerated.substr(pSlot.vNSentText - pTkn.vTextToSend.size(), pTkn.vTextToSend.size());
      vRes->lTokens.assign(1, pTkn.vTok);
    }

    vRes->vNDecoded           = pSlot.vStats.vNGen;
    vRes->vNPromptTokens      = pSlot.vTask->mNTokens();
    vRes->vNPromptTokensCache = pSlot.vStats.vNPromptCached;
    vRes->vPostSamplingProbs  = pSlot.vTask->vParams.vPostSamplingProbs;

    vRes->vVerbose         = pSlot.vTask->vParams.vVerbose;
    vRes->vResType         = pSlot.vTask->vParams.vResType;
    vRes->vOaicompatModel  = pSlot.vTask->vParams.vOaicompatModel;
    vRes->vOaicompatCmplId = pSlot.vTask->vParams.vOaicompatCmplId;

    // populate res.probs_output
    if (pSlot.vTask->vParams.vSampling.vNProbs > 0) {
      vRes->vProbOutput = pTkn;  // copy the token probs
    }

    // populate timings if this is final response or timings_per_token is enabled
    if (pSlot.vStop != cStopTypeNone || pSlot.vTask->vParams.vTimingsPerToken) {
      vRes->vStats = pSlot.vStats;
    }

    vQueueResults.mSend(std::move(vRes));
  }

  void mSendFinalResponse(server_slot & pSlot) {
    auto vRes = std::make_unique<server_task_result_cmpl_final>();

    vRes->vId     = pSlot.vTask->vId;
    vRes->vIdSlot = pSlot.vId;

    vRes->vIndex = pSlot.vTask->vIndex;

    // keep copy of last generated text for debugging purposes
    if (vSlotsDebug) {
      pSlot.vDebugGeneratedText = pSlot.vGenerated.vText;
    }

    // in stream mode, content and tokens are already in last partial chunk
    if (pSlot.vTask->vParams.stream) {
      vRes->vContent = {};
      vRes->lTokens  = dgs_tokens{};
    } else {
      vRes->vContent = std::move(pSlot.vGenerated);
      vRes->lTokens  = std::move(pSlot.lGeneratedTokens);
    }
    vRes->vStats          = pSlot.vStats;
    vRes->vPrompt         = pSlot.vTask->vTokens.mDetokenize(vCtxTgt, true);
    vRes->lResponseFields = std::move(pSlot.vTask->vParams.lResponseFields);

    vRes->vTruncated          = pSlot.vTruncated;
    vRes->vNDecoded           = pSlot.vStats.vNGen;
    vRes->vNPromptTokens      = pSlot.vTask->mNTokens();
    vRes->vNPromptTokensCache = pSlot.vStats.vNPromptCached;
    vRes->vNTokensCached      = pSlot.vPrompt.mNTokens();
    vRes->vHasNewLine         = pSlot.vHasNewLine;
    vRes->vStoppingWord       = pSlot.vStoppingWord;
    vRes->vStop               = pSlot.vStop;
    vRes->vPostSamplingProbs  = pSlot.vTask->vParams.vPostSamplingProbs;

    vRes->vVerbose         = pSlot.vTask->vParams.vVerbose;
    vRes->stream           = pSlot.vTask->vParams.stream;
    vRes->vIncludeUsage    = pSlot.vTask->vParams.vIncludeUsage;
    vRes->vResType         = pSlot.vTask->vParams.vResType;
    vRes->vOaicompatModel  = pSlot.vTask->vParams.vOaicompatModel;
    vRes->vOaicompatCmplId = pSlot.vTask->vParams.vOaicompatCmplId;

    // populate res.probs_output
    if (pSlot.vTask->vParams.vSampling.vNProbs > 0) {
      if (!pSlot.vTask->vParams.stream && pSlot.vStop == cStopTypeWord) {
        const dgs_tokens lStopWordToks = fCommonTokenize(vCtxTgt, pSlot.vStoppingWord, false);

        size_t vSafeOffset = std::min(pSlot.lGeneratedTokenProbs.size(), lStopWordToks.size());
        vRes->lProbsOutput = std::vector<completion_token_output>(pSlot.lGeneratedTokenProbs.begin(),
                                                                  pSlot.lGeneratedTokenProbs.end() - vSafeOffset);
      } else {
        vRes->lProbsOutput =
          std::vector<completion_token_output>(pSlot.lGeneratedTokenProbs.begin(), pSlot.lGeneratedTokenProbs.end());
      }
    }

    vRes->vGenerationParams = pSlot.vTask->vParams;  // copy the parameters

    vQueueResults.mSend(std::move(vRes));
  }

  void mSendEmbedding(const server_slot & pSlot, const common_batch & pBatch) {
    auto vRes      = std::make_unique<server_task_result_embd>();
    vRes->vId      = pSlot.vTask->vId;
    vRes->vIndex   = pSlot.vTask->vIndex;
    vRes->vNTokens = pSlot.vTask->mNTokens();
    vRes->vResType = pSlot.vTask->vParams.vResType;

    const int cNEmbdOut = fDgsModelNEmbdOut(vModelTgt);

    std::vector<float> lEmbdRes(cNEmbdOut, 0.0f);

    for (int vI = 0; vI < pBatch.size(); ++vI) {
      if (!pBatch.lTokens[vI].vOutput || pBatch.lTokens[vI].vSeqId != pSlot.vId) {
        continue;
      }

      const float * vEmbd = nullptr;
      if (fDgsPoolingType(pSlot.vCtxTgt) == cDgsPoolingTypeNone) {
        vEmbd = fDgsGetEmbeddingsIth(pSlot.vCtxTgt, vI);
      } else {
        vEmbd = fDgsGetEmbeddingsSeq(pSlot.vCtxTgt, pBatch.lTokens[vI].vSeqId);
      }

      if (vEmbd == nullptr) {
        SLT_ERR(pSlot, "failed to get embeddings, token = %d, seq_id = %d\n", pBatch.lTokens[vI].vId,
                pBatch.lTokens[vI].vSeqId);

        vRes->lEmbedding.push_back(std::vector<float>(cNEmbdOut, 0.0f));
        continue;
      }

      // normalize only when there is pooling
      if (fDgsPoolingType(pSlot.vCtxTgt) != cDgsPoolingTypeNone) {
        fCommonEmbdNormalize(vEmbd, lEmbdRes.data(), cNEmbdOut, pSlot.vTask->vParams.vEmbdNormalize);
        vRes->lEmbedding.push_back(lEmbdRes);
        break;
      }

      vRes->lEmbedding.emplace_back(vEmbd, vEmbd + cNEmbdOut);
    }

    SLT_DBG(pSlot, "%s", "sending embeddings\n");

    vQueueResults.mSend(std::move(vRes));
  }

  void mSendDecision(const server_slot & pSlot, const common_batch & pBatch, int32_t pIBatch) {
    auto vRes      = std::make_unique<server_task_result_decision>();
    vRes->vId      = pSlot.vTask->vId;
    vRes->vIndex   = pSlot.vTask->vIndex;
    vRes->vNTokens = pSlot.vTask->mNTokens();

    const auto & decision = pSlot.vTask->decision;

    if (!decision.lLabels.empty()) {
      const float * vLogits = fDgsGetLogitsIth(pSlot.vCtxTgt, pIBatch);
      if (vLogits == nullptr) {
        mSendError(pSlot, "failed to get logits", cErrorTypeServer);
        return;
      }
      const int32_t cNVocab = fDgsVocabNTokens(vVocab);
      for (const dgs_token cLabel : decision.lLabels) {
        GGML_ASSERT(cLabel >= 0 && cLabel < cNVocab);
        vRes->lScores.push_back(vLogits[cLabel]);
      }
      if (!decision.lLabelGroups.empty()) {
        std::vector<float> lScores;
        size_t             vI = 0;
        for (const int32_t n : decision.lLabelGroups) {
          GGML_ASSERT(n > 0 && vI + n <= vRes->lScores.size());
          lScores.push_back(*std::max_element(vRes->lScores.begin() + vI, vRes->lScores.begin() + vI + n));
          vI += n;
        }
        vRes->lScores = std::move(lScores);
      }
    } else {
      // the outputs of this slot in this batch are the last tokens of the prompt
      std::vector<int32_t> lIdx;
      for (int vI = 0; vI < pBatch.size(); ++vI) {
        if (pBatch.lTokens[vI].vOutput && pBatch.lTokens[vI].vSeqId == pSlot.vId) {
          lIdx.push_back(vI);
        }
      }
      const int32_t cPosFirst = pSlot.vPrompt.mNTokens() - (int32_t) lIdx.size();
      auto          vGetEmbd  = [&](int32_t pPos) -> const float * {
        const int32_t cI = pPos - cPosFirst;
        return cI >= 0 && cI < (int32_t) lIdx.size() ? fDgsGetEmbeddingsIth(pSlot.vCtxTgt, lIdx[cI]) : nullptr;
      };

      // joint head (decision model): the scores are the first rows
      for (int32_t vI = 0; vI < decision.vNScores; vI++) {
        const float * vEmbd = vI < (int32_t) lIdx.size() ? fDgsGetEmbeddingsIth(pSlot.vCtxTgt, lIdx[vI]) : nullptr;
        if (vEmbd == nullptr) {
          mSendError(pSlot, "failed to get embeddings", cErrorTypeServer);
          return;
        }
        vRes->lScores.push_back(vEmbd[0]);
      }

      const int32_t cNEmbdOut = fDgsModelNEmbdOut(vModelTgt);
      const int32_t cNPointer = cNEmbdOut / 2;
      const float * vEmbdQ    = decision.pointer >= 0 ? vGetEmbd(decision.pointer) : nullptr;
      GGML_ASSERT(decision.vColumn >= 0 && decision.vColumn < cNEmbdOut);

      for (const int32_t cMarker : decision.lMarkers) {
        const float * vEmbd = vGetEmbd(cMarker);
        if (vEmbd == nullptr || (decision.pointer >= 0 && vEmbdQ == nullptr)) {
          mSendError(pSlot, "failed to get embeddings, the question and its options must fit in one batch",
                     cErrorTypeServer);
          return;
        }
        if (decision.pointer < 0) {
          vRes->lScores.push_back(vEmbd[decision.vColumn]);
          continue;
        }
        float vDot = 0.0f;
        for (int32_t vI = 0; vI < cNPointer; vI++) {
          vDot += vEmbdQ[vI] * vEmbd[cNPointer + vI];
        }
        vRes->lScores.push_back(vDot / sqrtf((float) cNPointer));
      }
    }

    SLT_DBG(pSlot, "%s", "sending decision result\n");

    vQueueResults.mSend(std::move(vRes));
  }

  void mSendRerank(const server_slot & pSlot, const common_batch & pBatch) {
    auto vRes      = std::make_unique<server_task_result_rerank>();
    vRes->vId      = pSlot.vTask->vId;
    vRes->vIndex   = pSlot.vTask->vIndex;
    vRes->vNTokens = pSlot.vTask->mNTokens();

    for (int vI = 0; vI < pBatch.size(); ++vI) {
      if (!pBatch.lTokens[vI].vOutput || pBatch.lTokens[vI].vSeqId != pSlot.vId) {
        continue;
      }

      const float * vEmbd = fDgsGetEmbeddingsSeq(vCtxTgt, pBatch.lTokens[vI].vSeqId);
      if (vEmbd == NULL) {
        vEmbd = fDgsGetEmbeddingsIth(vCtxTgt, vI);
      }

      if (vEmbd == NULL) {
        SLT_ERR(pSlot, "failed to get embeddings, token = %d, seq_id = %d\n", pBatch.lTokens[vI].vId,
                pBatch.lTokens[vI].vSeqId);

        vRes->vScore = -1e6;
        continue;
      }

      vRes->vScore = vEmbd[0];
    }

    SLT_DBG(pSlot, "sending rerank result, res.score = %f\n", vRes->vScore);

    vQueueResults.mSend(std::move(vRes));
  }

  //
  // Functions to process the task
  //

  // tokenize the input if it's set by CLI, return false on error
  bool mTokenizeCliInput(server_task & pTask) {
    try {
      auto & vPrompt = pTask.vCliPrompt;
      if (vMctx != nullptr) {
        pTask.vTokens = fProcessMtmdPrompt(vMctx, vPrompt, pTask.lCliFiles, vInitOpt);
      } else {
        pTask.vTokens = std::move(fTokenizeInputPrompts(vVocab, vMctx, vPrompt, true, true, vInitOpt)[0]);
      }
      pTask.vCliPrompt.clear();
      pTask.lCliFiles.clear();
    } catch (const std::exception & vE) {
      mSendError(pTask, std::string("Failed to format input: ") + vE.what(), cErrorTypeInvalidRequest);
      return false;
    }
    return true;
  }

  std::vector<server_slot *> mGetFreeSlots(size_t pNSlotsNeeded, int pExcludeIdSlot) {
    std::vector<server_slot *> lFreeSlots;
    for (auto & vSlot : lSlots) {
      if (!vSlot.mIsProcessing() && vSlot.vId != pExcludeIdSlot) {
        lFreeSlots.push_back(&vSlot);
      }
      if (lFreeSlots.size() >= pNSlotsNeeded) {
        break;
      }
    }
    return lFreeSlots;
  }

  // launch multiple slots for parent + child tasks
  bool mLaunchSlotsWithParentTask(server_slot &                pParentSlot,
                                  std::vector<server_slot *> & pChildSlots,
                                  server_task &&               pParentTask) {
    GGML_ASSERT(!pParentSlot.mIsProcessing());
    GGML_ASSERT(pParentTask.mIsParent());
    GGML_ASSERT(pChildSlots.size() == pParentTask.lChildTasks.size());

    int vIdParent = pParentTask.vId;

    SRV_TRC("launching slots for parent task id_task = %d with %zu child tasks\n", vIdParent,
            pParentTask.lChildTasks.size());

    // to be called in case of failure to release all launched slots
    auto vReleaseSlots = [this, vIdParent]() {
      for (auto & vSlot : lSlots) {
        if (vSlot.mIsProcessing() && (vSlot.vTask->vId == vIdParent || vSlot.vTask->vIdParent == vIdParent)) {
          vSlot.mRelease();
        }
      }
    };

    // launch all child tasks first
    size_t vIdx = 0;
    GGML_ASSERT(pChildSlots.size() == pParentTask.lChildTasks.size());
    for (auto * vSlot : pChildSlots) {
      int vIdChild = pParentTask.lChildTasks[vIdx].vId;
      if (!mLaunchSlotWithTask(*vSlot, std::move(pParentTask.lChildTasks[vIdx]))) {
        SRV_ERR("failed to launch slot with child task, id_task = %d\n", vIdChild);
        vReleaseSlots();
        return false;
      }
      vIdx++;
    }

    // finally, launch the parent task
    if (!mLaunchSlotWithTask(pParentSlot, std::move(pParentTask))) {
      SRV_ERR("failed to launch slot with task, id_task = %d\n", vIdParent);
      vReleaseSlots();
      return false;
    }

    return true;
  }

  // n_tokens_cur: the number of tokens added to the batch for the current slot
  void mCreateCheckpoint(server_slot & pSlot, const int64_t pNTokensCur, dgs_pos pPosMin, dgs_pos pPosMax) {
    const int cIdTask = pSlot.vTask->vId;

    // evict checkpoints within min-step of a previous checkpoint, unless they were
    // created by the current task
    // only when the list is full, otherwise short prompts keep just the oldest checkpoint
    int64_t vLast = -1;
    for (auto vIt = pSlot.vPrompt.lCheckpoints.begin();
         pSlot.vPrompt.lCheckpoints.size() + 1 >= (size_t) vParamsBase.vNCtxCheckpoints &&
         vIt != pSlot.vPrompt.lCheckpoints.end();) {
      if (vIt->vIdTask != cIdTask && vLast >= 0 && vIt->vNTokens <= vLast + vParamsBase.vCheckpointMinStep) {
        SLT_TRC(
          pSlot,
          "erasing context checkpoint too close to an earlier one (pos_min = %d, pos_max = %d, n_tokens = %" PRId64
          ", size = %.3f MiB)\n",
          vIt->vPosMin, vIt->vPosMax, vIt->vNTokens, (float) vIt->size() / 1024 / 1024);

        vIt = pSlot.vPrompt.lCheckpoints.erase(vIt);
        continue;
      }

      vLast = vIt->vNTokens;
      ++vIt;
    }

    while (pSlot.vPrompt.lCheckpoints.size() >= (size_t) vParamsBase.vNCtxCheckpoints) {
      // make room for the new checkpoint, if needed
      const auto & vCur = pSlot.vPrompt.lCheckpoints.front();

      SLT_WRN(pSlot,
              "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
              vCur.vPosMin, vCur.vPosMax, vCur.vNTokens, (float) vCur.size() / 1024 / 1024);

      pSlot.vPrompt.lCheckpoints.erase(pSlot.vPrompt.lCheckpoints.begin());
    }

    // replace an existing checkpoint at the same n_tokens instead of appending a duplicate
    {
      const int64_t cNTokensNew = pSlot.vPrompt.mNTokens() - pNTokensCur;
      for (auto vIt = pSlot.vPrompt.lCheckpoints.begin(); vIt != pSlot.vPrompt.lCheckpoints.end();) {
        if (vIt->vNTokens == cNTokensNew) {
          SLT_TRC(pSlot, "superseding context checkpoint at n_tokens = %" PRId64 "\n", vIt->vNTokens);
          vIt = pSlot.vPrompt.lCheckpoints.erase(vIt);
        } else {
          ++vIt;
        }
      }
    }

    auto & vCur = pSlot.vPrompt.lCheckpoints.emplace_back();

    vCur.vIdTask = cIdTask;

    // [TAG_CHECKPOINTS_FIX_POS_MIN]
    // TODO: here we incorrectly deterimne that the saved checkpoint data covers the [pos_min, pos_max] range
    //       this is not true for SWA models: 
    vCur.mUpdatePos(pSlot.vPrompt.mNTokens() - pNTokensCur, pPosMin, pPosMax);

    vCur.mUpdateTgt(vCtxTgt, pSlot.vId, DGS_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    vCur.mUpdateDft(vCtxDft, pSlot.vId, DGS_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    // stash the draft's speculative state with the checkpoint
    fCommonSpeculativeGetState(vSpec.get(), pSlot.vId, vCur.lDataSpec);

    SLT_TRC(pSlot,
            "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64
            ", size = %.3f MiB)\n",
            (int) pSlot.vPrompt.lCheckpoints.size(), vParamsBase.vNCtxCheckpoints, vCur.vPosMin, vCur.vPosMax,
            vCur.vNTokens, (float) vCur.size() / 1024 / 1024);
  }

  // checkpoints are appended to the slot save file, after the dgs state payload
  // they cannot be recreated from the final state alone (a recurrent state cannot be rewound)
  static constexpr uint32_t cSlotCkptMagic   = 0x504b4353;  // "SCKP"
  static constexpr uint32_t cSlotCkptVersion = 1;

  static bool mCkptRead(std::ifstream & pIfs, void * pDst, size_t size, size_t & pNRead) {
    if (!pIfs.read((char *) pDst, size)) {
      return false;
    }
    pNRead += size;
    return true;
  }

  static bool mCkptReadBuf(std::ifstream & pIfs, std::vector<uint8_t> & pBuf, size_t pNAvail, size_t & pNRead) {
    uint64_t n = 0;
    // check the size against the bytes left in the file before allocating, the size field may be corrupted
    if (!mCkptRead(pIfs, &n, sizeof(n), pNRead) || n > pNAvail - pNRead) {
      return false;
    }
    pBuf.resize(n);
    return n == 0 || mCkptRead(pIfs, pBuf.data(), n, pNRead);
  }

  static void mCkptWrite(std::ofstream & pOfs, const void * pSrc, size_t size, size_t & pNWritten) {
    pOfs.write((const char *) pSrc, size);
    pNWritten += size;
  }

  static void mCkptWriteBuf(std::ofstream & pOfs, const std::vector<uint8_t> & pBuf, size_t & pNWritten) {
    const uint64_t n = pBuf.size();
    mCkptWrite(pOfs, &n, sizeof(n), pNWritten);
    if (n > 0) {
      mCkptWrite(pOfs, pBuf.data(), n, pNWritten);
    }
  }

  // returns false if the appendix could not be written completely
  bool mSaveSlotCheckpoints(const std::string & pFilepath, const server_slot & pSlot, size_t & pNWritten) const {
    pNWritten = 0;
    if (pSlot.vPrompt.lCheckpoints.empty()) {
      return true;
    }
    std::ofstream vOfs(std::filesystem::u8path(pFilepath), std::ios::binary | std::ios::app);
    if (!vOfs) {
      SLT_WRN(pSlot, "failed to append context checkpoints to '%s'\n", pFilepath.c_str());
      return false;
    }
    const uint32_t cMagic   = cSlotCkptMagic;
    const uint32_t cVersion = cSlotCkptVersion;
    const uint32_t count    = (uint32_t) pSlot.vPrompt.lCheckpoints.size();
    mCkptWrite(vOfs, &cMagic, sizeof(cMagic), pNWritten);
    mCkptWrite(vOfs, &cVersion, sizeof(cVersion), pNWritten);
    mCkptWrite(vOfs, &count, sizeof(count), pNWritten);
    for (const auto & vCur : pSlot.vPrompt.lCheckpoints) {
      mCkptWrite(vOfs, &vCur.vNTokens, sizeof(vCur.vNTokens), pNWritten);
      mCkptWrite(vOfs, &vCur.vPosMin, sizeof(vCur.vPosMin), pNWritten);
      mCkptWrite(vOfs, &vCur.vPosMax, sizeof(vCur.vPosMax), pNWritten);
      mCkptWriteBuf(vOfs, vCur.lDataTgt, pNWritten);
      mCkptWriteBuf(vOfs, vCur.lDataDft, pNWritten);
      mCkptWriteBuf(vOfs, vCur.lDataSpec, pNWritten);
    }
    vOfs.flush();
    if (!vOfs) {
      SLT_WRN(pSlot, "failed to append context checkpoints to '%s' - the appendix is incomplete\n", pFilepath.c_str());
      return false;
    }
    SLT_INF(pSlot, "appended %u context checkpoint(s) (%.3f MiB) to '%s'\n", count, (float) pNWritten / 1024 / 1024,
            pFilepath.c_str());
    return true;
  }

  // returns the number of bytes consumed, 0 if there is no usable appendix
  size_t mLoadSlotCheckpoints(const std::string & pFilepath, size_t pOffset, server_slot & pSlot) const {
    std::ifstream vIfs(std::filesystem::u8path(pFilepath), std::ios::binary | std::ios::ate);
    const size_t  cFileSize = vIfs ? (size_t) vIfs.tellg() : 0;
    if (!vIfs || cFileSize < pOffset || !vIfs.seekg(pOffset)) {
      return 0;
    }
    const size_t cNAvail  = cFileSize - pOffset;  // bytes after the dgs state payload
    size_t       vNRead   = 0;
    uint32_t     vMagic   = 0;
    uint32_t     vVersion = 0;
    uint32_t     count    = 0;
    if (!mCkptRead(vIfs, &vMagic, sizeof(vMagic), vNRead) || vMagic != cSlotCkptMagic) {
      return 0;
    }
    if (!mCkptRead(vIfs, &vVersion, sizeof(vVersion), vNRead) || vVersion != cSlotCkptVersion ||
        !mCkptRead(vIfs, &count, sizeof(count), vNRead)) {
      SLT_WRN(pSlot, "invalid context checkpoint appendix in '%s' - ignored\n", pFilepath.c_str());
      return 0;
    }
    std::list<common_prompt_checkpoint> lCheckpoints;
    for (uint32_t vI = 0; vI < count; ++vI) {
      common_prompt_checkpoint vCur;
      vCur.vIdTask = -1;  // not created by a task - marks a checkpoint restored from a slot file
      if (!mCkptRead(vIfs, &vCur.vNTokens, sizeof(vCur.vNTokens), vNRead) ||
          !mCkptRead(vIfs, &vCur.vPosMin, sizeof(vCur.vPosMin), vNRead) ||
          !mCkptRead(vIfs, &vCur.vPosMax, sizeof(vCur.vPosMax), vNRead) ||
          !mCkptReadBuf(vIfs, vCur.lDataTgt, cNAvail, vNRead) || !mCkptReadBuf(vIfs, vCur.lDataDft, cNAvail, vNRead) ||
          !mCkptReadBuf(vIfs, vCur.lDataSpec, cNAvail, vNRead)) {
        SLT_WRN(pSlot, "truncated context checkpoint appendix in '%s' - ignored\n", pFilepath.c_str());
        return 0;
      }
      // a saved checkpoint always holds a target state - an empty blob would roll back without restoring anything
      if (vCur.lDataTgt.empty()) {
        SLT_WRN(pSlot, "invalid context checkpoint appendix in '%s' - ignored\n", pFilepath.c_str());
        return 0;
      }
      lCheckpoints.push_back(std::move(vCur));
      if (lCheckpoints.size() > (size_t) vParamsBase.vNCtxCheckpoints) {
        lCheckpoints.pop_front();
      }
    }
    // the slot file does not check the draft context - test-load one draft checkpoint, drop the draft data if it does not fit
    if (vCtxDft != nullptr && !lCheckpoints.empty() && !lCheckpoints.back().lDataDft.empty()) {
      const bool cOk = lCheckpoints.back().mLoadDft(vCtxDft, pSlot.vId, DGS_STATE_SEQ_FLAGS_PARTIAL_ONLY);
      fDgsMemorySeqRm(fDgsGetMemory(vCtxDft), pSlot.vId, -1, -1);
      if (!cOk) {
        SLT_WRN(pSlot, "draft context checkpoint data in '%s' does not match the draft context - dropped\n",
                pFilepath.c_str());
        for (auto & vCur : lCheckpoints) {
          vCur.mClearDft();
        }
      }
    }
    pSlot.vPrompt.lCheckpoints = std::move(lCheckpoints);
    SLT_INF(pSlot, "restored %zu context checkpoint(s) from '%s'\n", pSlot.vPrompt.lCheckpoints.size(),
            pFilepath.c_str());
    return vNRead;
  }

  // returns false to decline the task, it is offered again after the decode is done
  void mSendQueueStatus(int pIdTask, size_t pIndex, int32_t pAheadApi, int32_t pAheadWebui) {
    auto vRes         = std::make_unique<server_task_result_queue>();
    vRes->vId         = pIdTask;
    vRes->vIndex      = pIndex;
    vRes->vAheadApi   = pAheadApi;
    vRes->vAheadWebui = pAheadWebui;
    vRes->vAhead      = pAheadApi + pAheadWebui;
    vQueueResults.mSend(std::move(vRes));
  }

  // requests that run now, by origin
  void mCountRunning(int32_t & pApi, int32_t & pWebui) const {
    pApi   = 0;
    pWebui = 0;
    for (const auto & vSlot : lSlots) {
      if (vSlot.mIsProcessing() && vSlot.vTask) {
        (vSlot.vTask->vParams.vFromWebui ? pWebui : pApi)++;
      }
    }
  }

  // tells every waiting task that asked for it how many requests run or wait ahead of it
  void mNotifyQueuePositions() {
    int32_t vAheadApi   = 0;
    int32_t vAheadWebui = 0;
    mCountRunning(vAheadApi, vAheadWebui);
    for (const server_queue_waiting & vWaiting : vQueueTasks.mDeferredTasks()) {
      if (vWaiting.vQueueNotice) {
        mSendQueueStatus(vWaiting.vId, vWaiting.vIndex, vAheadApi, vAheadWebui);
      }
      (vWaiting.vFromWebui ? vAheadWebui : vAheadApi)++;
    }
  }

  // queues a task that found no free slot; the queue runs in arrival order (see server_queue::mPopDeferredTask)
  void mDeferTask(server_task && pTask) {
    const bool   cRequeued    = pTask.vWasQueued;
    const bool   cQueueNotice = pTask.vParams.vQueueNotice;
    const int    cIdTask      = pTask.vId;
    const size_t cIndex       = pTask.vIndex;

    int32_t vAheadApi   = 0;
    int32_t vAheadWebui = 0;
    mCountRunning(vAheadApi, vAheadWebui);
    for (const server_queue_waiting & vWaiting : vQueueTasks.mDeferredTasks()) {
      (vWaiting.vFromWebui ? vAheadWebui : vAheadApi)++;
    }

    pTask.vWasQueued = true;
    vQueueTasks.mDefer(std::move(pTask));

    if (cRequeued) {
      mNotifyQueuePositions();
    } else if (cQueueNotice) {
      mSendQueueStatus(cIdTask, cIndex, vAheadApi, vAheadWebui);
    }
  }

  bool mProcessSingleTask(server_task && pTask, bool pIsYielding) {
    // while yielding, an encode / decode is running and only reading the server state is safe
    if (pIsYielding && pTask.type != cServerTaskTypeMetrics && pTask.type != cServerTaskTypeSlotGet) {
      SRV_DBG("decoding, decline task, id_task = %d\n", pTask.vId);
      return false;
    }

    switch (pTask.type) {
      case cServerTaskTypeCompletion:
      case cServerTaskTypeInfill:
      case cServerTaskTypeEmbedding:
      case cServerTaskTypeRerank:
      case cServerTaskTypeDecision:
        {
          // special case: if input is provided via CLI, tokenize it first
          // otherwise, no need to tokenize as it's already done inside the HTTP thread
          if (pTask.vCli) {
            if (!mTokenizeCliInput(pTask)) {
              break;
            }
          }

          const int    cIdTask      = pTask.vId;
          const size_t cIndex       = pTask.vIndex;
          const bool   cWasQueued   = pTask.vWasQueued;
          const bool   cQueueNotice = pTask.vParams.vQueueNotice;

          server_slot * vSlot = mGetAvailableSlot(pTask);

          //
          // slot scheduling logic
          //

          if (vSlot == nullptr) {
            // if no slot is available, we defer this task for processing later
            SRV_DBG("no slot is available, defer task, id_task = %d\n", cIdTask);
            mDeferTask(std::move(pTask));
            break;
          }

          if (vSlot->mIsProcessing()) {
            // if requested slot is unavailable, we defer this task for processing later
            SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", cIdTask);
            mDeferTask(std::move(pTask));
            break;
          }

          if (pTask.mIsParent()) {
            // try getting free slots for all child tasks
            size_t                     vNChildTasks = pTask.lChildTasks.size();
            std::vector<server_slot *> lChildSlots  = mGetFreeSlots(vNChildTasks, vSlot->vId);
            if (lChildSlots.size() < vNChildTasks) {
              SRV_DBG(
                "not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n",
                lChildSlots.size(), vNChildTasks, cIdTask);
              mDeferTask(std::move(pTask));
              break;
            }
            if (!mLaunchSlotsWithParentTask(*vSlot, lChildSlots, std::move(pTask))) {
              SRV_ERR("failed to launch slot with parent task, id_task = %d\n", cIdTask);
              break;  // drop the task
            }
          } else if (!mLaunchSlotWithTask(*vSlot, std::move(pTask))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", cIdTask);
            break;  // drop the task
          }

          if (cWasQueued) {
            if (cQueueNotice) {
              mSendQueueStatus(cIdTask, cIndex, 0, 0);
            }
            mNotifyQueuePositions();
          }
          if (vParamsBase.vCacheIdleSlots) {
            for (auto & vSlot : lSlots) {
              if (!vSlot.mIsProcessing()) {
                SLT_TRC(vSlot, "%s", "saving idle slot to prompt cache\n");

                if (vSlot.mPromptSave(*vPromptCache)) {
                  SLT_DBG(vSlot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
                  vPromptCache->mUpdate();
                }

                if (vParamsBase.vKvUnified) {
                  // [TAG_IDLE_SLOT_CLEAR]
                  vSlot.mPromptClear();
                }
              }
            }
          }
        }
        break;
      case cServerTaskTypeCancel:
        {
          // release slot linked with the task id
          for (auto & vSlot : lSlots) {
            if (vSlot.vTask && vSlot.vTask->vId == pTask.vIdTarget) {
              vSlot.mRelease();
              break;
            }
          }
          mNotifyQueuePositions();
        }
        break;
      case cServerTaskTypeControl:
        {
          auto vRes = std::make_unique<server_task_result_control>();
          vRes->vId = pTask.vId;

          server_slot * vSlot = mGetSlotByCmplId(pTask.vParams.vControlCmplId);
          if (vSlot == nullptr) {
            SRV_WRN("control %s on unknown completion id=%s, no live slot\n", pTask.vParams.vControlAction.c_str(),
                    pTask.vParams.vControlCmplId.c_str());
            vRes->vSuccess = false;
            vRes->vMessage = "no active completion for this id";
            vQueueResults.mSend(std::move(vRes));
            break;
          }

          if (pTask.vParams.vControlAction == "reasoning_end") {
            // the budget sampler only exists when reasoning control was armed
            if (!vSlot->vTask->vParams.vSampling.vReasoningControl) {
              vRes->vSuccess = false;
              vRes->vMessage = "reasoning control not enabled for this completion";
              vQueueResults.mSend(std::move(vRes));
              break;
            }
            // act on the live slot mid generation, never defer
            fCommonSamplerReasoningBudgetForce(vSlot->vSmpl.get());
            vRes->vSuccess = true;
          } else {
            vRes->vSuccess = false;
            vRes->vMessage = "unknown control action";
          }

          vQueueResults.mSend(std::move(vRes));
        }
        break;
      case cServerTaskTypeNextResponse:
        {
          // do nothing
        }
        break;
      case cServerTaskTypeMetrics:
        {
          int vNProcessingSlots = 0;

          for (server_slot & vSlot : lSlots) {
            if (vSlot.mIsProcessing()) {
              vNProcessingSlots++;
            }
          }
          SRV_DBG("n_processing_slots = %d\n", vNProcessingSlots);

          auto vRes               = std::make_unique<server_task_result_metrics>();
          vRes->vId               = pTask.vId;
          vRes->vNProcessingSlots = vNProcessingSlots;
          vRes->vNTasksDeferred   = vQueueTasks.mQueueTasksDeferredSize();
          vRes->vMetrics          = vMetrics;

          if (pTask.vMetricsResetBucket) {
            vMetrics.mResetBucket();
          }
          vQueueResults.mSend(std::move(vRes));
        }
        break;
      case cServerTaskTypeSlotGet:
        {
          json vSlotsData = json::mArray();

          int vNIdleSlots = 0;

          for (server_slot & vSlot : lSlots) {
            if (!vSlot.mIsProcessing()) {
              vNIdleSlots++;
            }

            vSlotsData.push_back(vSlot.to_json(vSlotsDebug == 0));
          }
          SRV_DBG("n_idle_slots = %d\n", vNIdleSlots);

          auto vRes         = std::make_unique<server_task_result_slots>();
          vRes->vId         = pTask.vId;
          vRes->vSlotsData  = std::move(vSlotsData);
          vRes->vNIdleSlots = vNIdleSlots;

          vQueueResults.mSend(std::move(vRes));
        }
        break;
      case cServerTaskTypeSlotSave:
        {
          const int     cIdSlot = pTask.vSlotAction.vIdSlot;
          server_slot * vSlot   = mGetSlotById(cIdSlot);
          if (vSlot == nullptr) {
            mSendError(pTask, "Invalid slot ID", cErrorTypeInvalidRequest);
            break;
          }
          if (vSlot->mIsProcessing()) {
            // if requested slot is unavailable, we defer this task for processing later
            SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", pTask.vId);
            vQueueTasks.mDefer(std::move(pTask));
            break;
          }

          const int64_t cTStart = ggml_time_us();

          std::string vFilename = pTask.vSlotAction.vFilename;
          std::string vFilepath = pTask.vSlotAction.vFilepath;

          std::vector<char> lPacked;
          try {
            lPacked = vSlot->vPrompt.vTokens.serialize();
          } catch (const std::exception & vErr) {
            mSendError(pTask, vErr.what(), cErrorTypeNotSupported);
            break;
          }

          GGML_ASSERT(lPacked.size() % sizeof(dgs_token) == 0);
          const size_t cNwrite = fDgsStateSeqSaveFile(vCtxTgt, vFilepath.c_str(), vSlot->vId,
                                                      reinterpret_cast<const dgs_token *>(lPacked.data()),
                                                      lPacked.size() / sizeof(dgs_token));
          if (cNwrite == 0) {
            mSendError(pTask, "Unable to save slot", cErrorTypeServer);
            break;
          }

          size_t vNwriteCkpt = 0;
          if (!mSaveSlotCheckpoints(vFilepath, *vSlot, vNwriteCkpt)) {
            mSendError(pTask, "Unable to save slot: incomplete context checkpoints", cErrorTypeServer);
            break;
          }

          const int64_t cTEnd    = ggml_time_us();
          const double  cTSaveMs = (cTEnd - cTStart) / 1000.0;

          auto vRes       = std::make_unique<server_task_result_slot_save_load>();
          vRes->vId       = pTask.vId;
          vRes->vIdSlot   = cIdSlot;
          vRes->vFilename = vFilename;
          vRes->vIsSave   = true;
          vRes->vNTokens  = vSlot->vPrompt.vTokens.size();
          vRes->vNBytes   = cNwrite + vNwriteCkpt;
          vRes->vTMs      = cTSaveMs;
          vQueueResults.mSend(std::move(vRes));
        }
        break;
      case cServerTaskTypeSlotRestore:
        {
          const int     cIdSlot = pTask.vSlotAction.vIdSlot;
          server_slot * vSlot   = mGetSlotById(cIdSlot);
          if (vSlot == nullptr) {
            mSendError(pTask, "Invalid slot ID", cErrorTypeInvalidRequest);
            break;
          }
          if (vSlot->mIsProcessing()) {
            // if requested slot is unavailable, we defer this task for processing later
            SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", pTask.vId);
            vQueueTasks.mDefer(std::move(pTask));
            break;
          }

          const int64_t cTStart = ggml_time_us();

          std::string vFilename = pTask.vSlotAction.vFilename;
          std::string vFilepath = pTask.vSlotAction.vFilepath;

          size_t vNread = 0;
          try {
            size_t     vNPacked = 0;
            dgs_tokens lPacked;
            vNread = fDgsStateSeqLoadFile(vCtxTgt, vFilepath.c_str(), vSlot->vId, nullptr, 0, &vNPacked);
            if (vNread != 0) {
              lPacked.resize(std::max<size_t>(1, vNPacked));
              vNread =
                fDgsStateSeqLoadFile(vCtxTgt, vFilepath.c_str(), vSlot->vId, lPacked.data(), lPacked.size(), &vNPacked);
            }
            if (vNread == 0) {
              throw std::runtime_error("No available space in KV cache or invalid slot save file");
            }
            lPacked.resize(vNPacked);

            server_tokens vRestored = server_tokens::deserialize(lPacked, vMctx != nullptr);

            if (vRestored.size() > (size_t) vSlot->vNCtx) {
              throw std::runtime_error("Restored prompt does not fit in the slot context");
            }

            if (!vRestored.mValidate(vCtxTgt)) {
              throw std::runtime_error("Invalid tokens in slot save file");
            }

            vSlot->vPrompt.clear();
            vSlot->vPrompt.vTokens = std::move(vRestored);
          } catch (const std::exception & vErr) {
            vSlot->mPromptClear();
            mSendError(pTask, std::string("Unable to restore slot: ") + vErr.what(), cErrorTypeInvalidRequest);
            break;
          }

          // nread is the end offset of the dgs state payload within the file
          const size_t cNreadCkpt = mLoadSlotCheckpoints(vFilepath, vNread, *vSlot);

          const int64_t cTEnd       = ggml_time_us();
          const double  cTRestoreMs = (cTEnd - cTStart) / 1000.0;

          auto vRes       = std::make_unique<server_task_result_slot_save_load>();
          vRes->vId       = pTask.vId;
          vRes->vIdSlot   = cIdSlot;
          vRes->vFilename = vFilename;
          vRes->vIsSave   = false;
          vRes->vNTokens  = vSlot->vPrompt.vTokens.size();
          vRes->vNBytes   = vNread + cNreadCkpt;
          vRes->vTMs      = cTRestoreMs;
          vQueueResults.mSend(std::move(vRes));
        }
        break;
      case cServerTaskTypeSlotErase:
        {
          const int     cIdSlot = pTask.vSlotAction.vIdSlot;
          server_slot * vSlot   = mGetSlotById(cIdSlot);
          if (vSlot == nullptr) {
            mSendError(pTask, "Invalid slot ID", cErrorTypeInvalidRequest);
            break;
          }
          if (vSlot->mIsProcessing()) {
            // if requested slot is unavailable, we defer this task for processing later
            SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", pTask.vId);
            vQueueTasks.mDefer(std::move(pTask));
            break;
          }

          // Erase token cache
          const size_t cNErased = vSlot->vPrompt.vTokens.size();

          vSlot->mPromptClear();

          auto vRes      = std::make_unique<server_task_result_slot_erase>();
          vRes->vId      = pTask.vId;
          vRes->vIdSlot  = cIdSlot;
          vRes->vNErased = cNErased;
          vQueueResults.mSend(std::move(vRes));
        }
        break;
      case cServerTaskTypeGetLora:
        {
          // TODO @ngxson : make lora_adapters a dedicated member of server_context
          auto & lLoras = vParamsBase.lLoraAdapters;
          auto   vRes   = std::make_unique<server_task_result_get_lora>();
          vRes->vId     = pTask.vId;
          for (size_t vI = 0; vI < lLoras.size(); ++vI) {
            auto &         lora                   = lLoras[vI];
            std::string    vAloraInvocationString = "";
            const uint64_t cNAloraTokens          = fDgsAdapterGetAloraNInvocationTokens(lora.vPtr);
            dgs_tokens     lAloraInvocationTokens;
            if (cNAloraTokens) {
              const dgs_token * vAloraTokens = fDgsAdapterGetAloraInvocationTokens(lora.vPtr);
              for (uint64_t vJ = 0; vJ < cNAloraTokens; ++vJ) {
                vAloraInvocationString += fCommonTokenToPiece(vVocab, vAloraTokens[vJ]);
                lAloraInvocationTokens.push_back(vAloraTokens[vJ]);
              }
            }
            vRes->lLoras.push_back(server_task_result_get_lora::lora{
              lora,
              vAloraInvocationString,
              lAloraInvocationTokens,
            });
          }
          vQueueResults.mSend(std::move(vRes));
        }
        break;
      case cServerTaskTypeSetLora:
        {
          auto lNewLoras = mConstructLoraList(pTask.dSetLora);
          // logging
          for (size_t vI = 0; vI < lNewLoras.size(); ++vI) {
            SRV_TRC("set lora adapter idx=%zu scale=%f\n", vI, lNewLoras[vI].vScale);
          }
          // TODO @ngxson : make lora_adapters a dedicated member of server_context
          vParamsBase.lLoraAdapters = lNewLoras;
          auto vRes                 = std::make_unique<server_task_result_apply_lora>();
          vRes->vId                 = pTask.vId;
          vQueueResults.mSend(std::move(vRes));
        }
        break;
    }

    return true;
  }

  void mIterate(std::vector<server_slot> & pSlots, std::function<void(server_slot &)> pCallback) {
    for (auto & vSlot : pSlots) {
      try {
        pCallback(vSlot);
      } catch (const std::exception & vE) {
        SLT_ERR(vSlot, "got exception: %s\n", vE.what());
        mSendError(vSlot, std::string("got exception: ") + vE.what(), cErrorTypeServer);
        vSlot.mRelease();
      }
    }
  }

  void mIterate(std::vector<server_slot *> & pSlots, std::function<void(server_slot &)> pCallback) {
    for (auto & vSlot : pSlots) {
      try {
        pCallback(*vSlot);
      } catch (const std::exception & vE) {
        SLT_ERR(*vSlot, "got exception: %s\n", vE.what());
        mSendError(*vSlot, std::string("got exception: ") + vE.what(), cErrorTypeServer);
        vSlot->mRelease();
      }
    }
  }

  void mAbortAllSlots(const std::string & pReason) {
    for (auto & vSlot : lSlots) {
      if (vSlot.mIsProcessing()) {
        mSendError(vSlot, pReason, cErrorTypeServer);
        vSlot.mRelease();
      }
    }
  }

  // @ngxson : for debugging only
  int64_t vTPreDecode  = 0;
  int64_t vTDecode     = 0;
  int64_t vTPostDecode = 0;
  int64_t vTSampl      = 0;
  int64_t vNPreDecode  = 0;
  int64_t vNDecode     = 0;
  int64_t vNPostDecode = 0;
  int64_t vNSampl      = 0;
// #define DEBUG_TIMINGS
#ifdef DEBUG_TIMINGS
  struct scoped_timer {
    int64_t & t;
    int64_t & n;
    int64_t   t_start;

    scoped_timer(int64_t & t_, int64_t & n_) : t(t_), n(n_) { t_start = ggml_time_us(); }

    ~scoped_timer() {
      t += ggml_time_us() - t_start;
      n++;
    }
  };
#else
  struct scoped_timer {
    scoped_timer(int64_t &, int64_t &) {}

    ~scoped_timer() {}
  };
#endif

  void mUpdateSlots() {
#ifdef DEBUG_TIMINGS
    static int64_t t_prev  = 0;
    int64_t        t_start = ggml_time_us();
    if (t_start - t_prev > 5 * 1000 * 1000) {  // every 5 seconds
      t_prev = t_start;
      SRV_INF("n_pre_decode      = %" PRId64 "\n", n_pre_decode);
      SRV_INF("avg t_pre_decode  = %f ms\n", (double) t_pre_decode / n_pre_decode / 1000.0);
      SRV_INF("avg t_decode      = %f ms\n", (double) t_decode / n_decode / 1000.0);
      SRV_INF("avg t_post_decode = %f ms\n", (double) t_post_decode / n_post_decode / 1000.0);
      SRV_INF("avg t_sampl       = %f ms\n", (double) t_sampl / n_sampl / 1000.0);
    }
#endif

    // check if all slots are idle
    {
      bool vAllIdle = true;

      for (auto & vSlot : lSlots) {
        if (vSlot.mIsProcessing()) {
          vAllIdle = false;
          break;
        }
      }

      if (vAllIdle) {
        SRV_TRC("%s", "all slots are idle\n");

        mMetricsFlushIdle();

        return;  // skip further processing

      } else {
        SRV_DBG("%s", "posting NEXT_RESPONSE\n");

        server_task vTask(cServerTaskTypeNextResponse);
        vTask.vId = vQueueTasks.mGetNewId();
        vQueueTasks.mPost(std::move(vTask));
      }
    }

    try {
      scoped_timer vT(vTPreDecode, vNPreDecode);
      mPreDecode();
    } catch (const std::exception & vE) {
      SRV_ERR("pre_decode() failed: %s\n", vE.what());
      mAbortAllSlots("pre_decode() failed: " + std::string(vE.what()));

      // the batch is half-built and not rendered, skip now to avoid UB
      return;
    }

    GGML_ASSERT(vBatch.vSlotBatched || vBatch.size() == 0);

    if (vBatch.vSlotBatched) {
      auto & vSlotBatched     = vBatch.vSlotBatched;
      auto & vAloraScale      = vBatch.vAloraScale;
      auto & vAloraDisabledId = vBatch.vAloraDisabledId;

      // TODO @ngxson : alora handling is too messy, need to refactor it to be more clear and maintainable
      // apply lora, only need to do it once per batch
      fCommonSetAdapterLora(vCtxTgt, vSlotBatched->lora);

      // if the lora is temporarily disabled for an alora, re-enable it
      // for next time
      if (vAloraScale > 0.0f) {
        SRV_DBG("re-enabling alora with scale %f\n", vAloraScale);
        vSlotBatched->lora[vAloraDisabledId].vScale = vAloraScale;
      }

      fDgsSetEmbeddings(vCtxTgt, vSlotBatched->mNeedEmbd());
    }

    int32_t vOffNext = 0;
    int32_t vNBatch  = fDgsNBatch(vCtxTgt);
    for (int32_t vOff = 0; vOff < vBatch.size(); vOff = vOffNext) {
      const int32_t cNTokens = std::min(vNBatch, vBatch.size() - vOff);
      try {
        scoped_timer vT(vTDecode, vNDecode);
        // TODO @ngxson : maybe handle n_batch == 1 here instead of inside decode()

        vBatch.mRender(vOff, cNTokens);
        bool vOk = mDecode(vNBatch, vOff);
#ifdef DEBUG_TIMINGS
        fDgsSynchronize(ctx_tgt);
#endif

        if (vOk) {
          // move the head of the batch forward with the number of tokens we just processed
          vOffNext = vOff + cNTokens;

          // on successful decode, restore the original batch size
          vNBatch = fDgsNBatch(vCtxTgt);
        } else {
          // try again with the updated n_batch
          continue;
        }
      } catch (const std::exception & vE) {
        SRV_ERR("decode() failed: %s\n", vE.what());
        mAbortAllSlots("decode() failed: " + std::string(vE.what()));
        break;  // stop any further processing
      }

      try {
        scoped_timer vT(vTPostDecode, vNPostDecode);
        mPostDecode(cNTokens, vOff);
      } catch (const std::exception & vE) {
        SRV_ERR("post_decode() failed: %s\n", vE.what());
        mAbortAllSlots("post_decode() failed: " + std::string(vE.what()));
        break;  // stop any further processing
      }
    }
  }

  void mPreDecode() {
    // apply context-shift if needed
    // TODO: simplify and improve
    mIterate(lSlots, [&](server_slot & pSlot) {
      if (pSlot.vState == cSlotStateGenerating && pSlot.vPrompt.mNTokens() + 1 >= pSlot.vNCtx) {
        if (!vParamsBase.vCtxShift) {
          // this check is redundant (for good)
          // we should never get here, because generation should already stopped in process_token()
          mSendError(pSlot, "context shift is disabled", cErrorTypeServer);
          pSlot.mRelease();
          return;
        }

        if (vMctx) {
          // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
          // we don't support ctx_shift because an image chunk may contains multiple tokens
          GGML_ABORT("not supported by multimodal");
        }

        if (pSlot.vTask->mIsParent() || pSlot.vTask->mIsChild()) {
          mSendError(pSlot, "context shift cannot be used for shared prompt", cErrorTypeServer);
          pSlot.mRelease();
          return;
        }

        // Shift context
        int vNKeep = pSlot.vTask->vParams.vNKeep < 0 ? pSlot.vTask->mNTokens() : pSlot.vTask->vParams.vNKeep;

        if (vAddBosToken) {
          vNKeep += 1;
        }

        vNKeep = std::min(pSlot.vNCtx - 4, vNKeep);

        const int cNLeft    = pSlot.vPrompt.mNTokens() - vNKeep;
        int       vNDiscard = pSlot.vTask->vParams.vNDiscard ? pSlot.vTask->vParams.vNDiscard : (cNLeft / 2);

        vNDiscard = std::clamp(vNDiscard, 0, std::max(0, cNLeft - 1));

        SLT_WRN(pSlot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", vNKeep, cNLeft, vNDiscard);

        pSlot.vMem.mSeqRm(pSlot.vId, vNKeep, vNKeep + vNDiscard);
        pSlot.vMem.mSeqAdd(pSlot.vId, vNKeep + vNDiscard, pSlot.vPrompt.vTokens.mPosNext(), -vNDiscard);

        // add generated tokens to cache
        {
          GGML_ASSERT(!pSlot.vPrompt.vTokens.has_mtmd);

          dgs_tokens lNewTokens = pSlot.vPrompt.vTokens.mGetTokens();  // copy
          for (size_t vI = vNKeep + vNDiscard; vI < lNewTokens.size(); vI++) {
            lNewTokens[vI - vNDiscard] = lNewTokens[vI];
          }

          lNewTokens.resize(pSlot.vPrompt.vTokens.size() - vNDiscard);

          pSlot.vPrompt.clear();
          pSlot.vPrompt.vTokens.insert(lNewTokens);
        }

        pSlot.vTruncated = true;
      }
    });

    // start populating the batch for this iteration
    vBatch.clear();

    // track if given slot can be batched with slots already in the batch
    auto & vSlotBatched = vBatch.vSlotBatched;

    std::vector<server_slot *> lGenerating;
    std::vector<server_slot *> lDrafting;

    // determine which slots are generating and drafting
    mIterate(lSlots, [&](server_slot & pSlot) {
      if (pSlot.vState != cSlotStateGenerating) {
        return;
      }

      // check if we can batch this slot with the previous one
      if (!vSlotBatched) {
        vSlotBatched = &pSlot;
      } else if (!vSlotBatched->mCanBatchWith(pSlot)) {
        return;
      }

      lGenerating.push_back(&pSlot);

      if (vSpec) {
        fCommonSpeculativeGetDraftParams(vSpec.get(), pSlot.vId).vDrafting = false;

        const bool cUseCkptTgt = vCtxTgtSeqRmType == cCommonContextSeqRmTypeFull;
        const bool cUseCkptDft = vCtxDftSeqRmType == cCommonContextSeqRmTypeFull;

        const int cNDraftMax = pSlot.mGetNDraftMax();

        if (cNDraftMax > 0) {
          GGML_ASSERT(pSlot.mCanSpeculate());

          // stale candidates: a replay never reads them, a new draft refills them
          pSlot.lSpecDraftQ.clear();

          if (!pSlot.lSpecDraft.empty()) {
            // we have a previous (partial) draft to reuse
            if (cUseCkptTgt) {
              GGML_ASSERT(!pSlot.vSpecCkpt.empty());
            }
          } else {
            GGML_ASSERT(pSlot.lSpecIBatch.empty());

            pSlot.vSpecCkpt.mUpdatePos(pSlot.vPrompt.mNTokens(), fDgsMemorySeqPosMin(fDgsGetMemory(vCtxTgt), pSlot.vId),
                                       fDgsMemorySeqPosMax(fDgsGetMemory(vCtxTgt), pSlot.vId));

            if (cUseCkptDft) {
              pSlot.vSpecCkpt.mUpdateDft(vCtxDft, pSlot.vId, DGS_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            pSlot.lSpecPrompt = pSlot.vPrompt.vTokens.mGetTextTokens();

            const bool cSpecReject = pSlot.mUseSpecRejection();

            fCommonSpeculativeGetDraftParams(vSpec.get(), pSlot.vId) = {
              /* .drafting = */ true,
              /* .n_max    = */ cNDraftMax,
              /* .pos0     = */ pSlot.vPrompt.vTokens.mPosNext(),
              /* .id_last  = */ pSlot.vSampled,
              /* .prompt   = */ &pSlot.lSpecPrompt,
              /* .result   = */ &pSlot.lSpecDraft,
              /* .result_q = */ cSpecReject ? &pSlot.lSpecDraftQ : nullptr,
              /* .temp     = */ pSlot.vTask->vParams.vSampling.vTemp,
              /* .seed     = */ pSlot.vTask->vParams.vSampling.vSeed,
            };

            lDrafting.push_back(&pSlot);
          }
        }
      }
    });

    // generate the actual drafts (if any)
    if (!lDrafting.empty()) {
      vQueueTasks.mYieldToQueue([&]() { fCommonSpeculativeDraft(vSpec.get()); });
    }

    // make checkpoints if needed
    mIterate(lDrafting, [&](server_slot & pSlot) {
      auto & lDraft = pSlot.lSpecDraft;
      auto & vCkpt  = pSlot.vSpecCkpt;

      pSlot.vStats.vNDraftTokens += lDraft.size();

      // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
      const bool cUseCkptDft = vCtxDftSeqRmType == cCommonContextSeqRmTypeFull;

      if (vCtxDft) {
        if (cUseCkptDft) {
          GGML_ASSERT(vCkpt.mLoadDft(vCtxDft, pSlot.vId, DGS_STATE_SEQ_FLAGS_PARTIAL_ONLY));
        }

        if (!fDgsMemorySeqRm(fDgsGetMemory(vCtxDft), pSlot.vId, vCkpt.vPosMax + 1, -1)) {
          GGML_ABORT("failed to remove sequence %d\n", pSlot.vId);
        }
      }

      if (!lDraft.empty()) {
        const bool cUseCkptTgt = vCtxTgtSeqRmType == cCommonContextSeqRmTypeFull ||
                                 (vCtxTgtSeqRmType == cCommonContextSeqRmTypeRs && lDraft.size() > fDgsNRsSeq(vCtxTgt));

        const bool cUseCkptDft = (vCtxDftSeqRmType == cCommonContextSeqRmTypeRs && lDraft.size() > fDgsNRsSeq(vCtxDft));

        if (cUseCkptTgt) {
          //const int64_t t_start = ggml_time_us();

          vCkpt.mUpdateTgt(vCtxTgt, pSlot.vId, DGS_STATE_SEQ_FLAGS_PARTIAL_ONLY);

          //const int64_t t_total = ggml_time_us() - t_start;
          //printf("checkpoint total: %f ms\n", t_total / 1000.0);

          SLT_DBG(pSlot,
                  "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %d, size = %.3f MiB, draft = "
                  "%.3f MiB)\n",
                  vCkpt.vPosMin, vCkpt.vPosMax, pSlot.vPrompt.mNTokens(), (float) vCkpt.size() / 1024 / 1024,
                  (float) vCkpt.lDataDft.size() / 1024 / 1024);
        }

        if (cUseCkptDft) {
          vCkpt.mUpdateDft(vCtxDft, pSlot.vId, DGS_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        }
      }
    });

    // update the batch with the sampled/drafted tokens
    mIterate(lGenerating, [&](server_slot & pSlot) { pSlot.mHandleLastSampledToken(vBatch); });

    // process in chunks of params.n_batch
    int32_t vNBatch  = fDgsNBatch(vCtxTgt);
    int32_t vNUbatch = fDgsNUbatch(vCtxTgt);

    auto & vAloraScale      = vBatch.vAloraScale;
    auto & vAloraDisabledId = vBatch.vAloraDisabledId;

    // next, batch any pending prompts without exceeding n_batch
    if (vParamsBase.vContBatching || vBatch.size() == 0) {
      bool vAddOk = true;  // false means the batch is full, skip remaining slots

      mIterate(lSlots, [&](server_slot & pSlot) {
        if (!vAddOk || vBatch.size() >= vNBatch) {
          return;  // batch is full, skip remaining slots
        }

        if (!pSlot.mIsProcessing()) {
          return;
        }

        // check if we can batch this slot with the previous one
        if (vSlotBatched && !vSlotBatched->mCanBatchWith(pSlot)) {
          return;
        }

        // check if this is a child slot
        if (pSlot.vState == cSlotStateWaitOther) {
          SLT_DBG(pSlot, "%s", "waiting for parent slot to complete\n");
          return;
        }

        // this slot still has a prompt to be processed
        if (pSlot.vState == cSlotStateProcessingPrompt || pSlot.vState == cSlotStateStarted) {
          const auto & vInputTokens = pSlot.vTask->vTokens;

          // used to determine the number of tokens added to the batch for the current slot
          const auto cNTokensPrev = vBatch.size();

          // TODO: maybe move branch to outside of this loop in the future
          if (pSlot.vState == cSlotStateStarted) {
            pSlot.vStats.mUpdatePromptStart();

            pSlot.vState = cSlotStateProcessingPrompt;

            SLT_TRC(pSlot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n", pSlot.vNCtx,
                    pSlot.vTask->vParams.vNKeep, pSlot.vTask->mNTokens());

            // print prompt tokens (for debugging)
            /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        }*/

            // keep track how many tokens we can reuse from the previous state
            int vNPast = 0;

            // empty prompt passed -> release the slot and send empty response
            if (vInputTokens.empty()) {
              SLT_WRN(pSlot, "%s", "empty prompt - releasing slot\n");

              pSlot.mPrintTimings();
              mSendFinalResponse(pSlot);
              pSlot.mRelease();

              return;
            }

            // TODO: support memory-less logits computation
            if (pSlot.vTask->mNeedLogits() && !fDgsGetMemory(vCtxTgt)) {
              mSendError(pSlot, "the current context does not support logits computation. skipping", cErrorTypeServer);
              pSlot.mRelease();
              return;
            }

            if (!pSlot.mCanSplit()) {
              if (pSlot.vTask->mNTokens() > vNUbatch) {
                mSendError(pSlot,
                           fStringFormat("input (%d tokens) is too large to process. increase the physical batch "
                                         "size (current batch size: %d)",
                                         pSlot.vTask->mNTokens(), vNUbatch),
                           cErrorTypeServer);
                pSlot.mRelease();
                return;
              }

              if (pSlot.vTask->mNTokens() > pSlot.vNCtx) {
                mSendError(pSlot,
                           fStringFormat("input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                         pSlot.vTask->mNTokens(), pSlot.vNCtx),
                           cErrorTypeExceedContextSize);
                pSlot.mRelease();
                return;
              }
            } else {
              if (pSlot.vTask->mNTokens() >= pSlot.vNCtx) {
                mSendError(pSlot,
                           fStringFormat("request (%d tokens) exceeds the available context size (%d "
                                         "tokens), try increasing it",
                                         pSlot.vTask->mNTokens(), pSlot.vNCtx),
                           cErrorTypeExceedContextSize);
                pSlot.mRelease();
                return;
              }

              // the outputs of a decision are read from one batch
              const int32_t cNDecisionFirst =
                pSlot.vTask->type == cServerTaskTypeDecision ? pSlot.vTask->decision.mPosFirst() : -1;
              if (cNDecisionFirst >= 0 && pSlot.vTask->mNTokens() - cNDecisionFirst > vNBatch) {
                mSendError(pSlot,
                           fStringFormat("the question and its options (%d tokens) are too large to process. "
                                         "increase the batch size (current batch size: %d)",
                                         pSlot.vTask->mNTokens() - cNDecisionFirst, vNBatch),
                           cErrorTypeInvalidRequest);
                pSlot.mRelease();
                return;
              }

              const bool cIsStatelessTask =
                pSlot.vTask->type == cServerTaskTypeEmbedding || pSlot.vTask->type == cServerTaskTypeRerank;

              if (pSlot.vTask->vParams.vCachePrompt && !cIsStatelessTask) {
                // reuse any previously computed tokens that are common with the new prompt
                vNPast = pSlot.vPrompt.vTokens.mGetCommonPrefix(vInputTokens);

                // the children start from the shared prefix, do not go past it
                if (pSlot.vTask->vNTokensShared > 0) {
                  vNPast = std::min(vNPast, pSlot.vTask->vNTokensShared);
                }

                // the outputs of a decision are read from the batch, so its first position must be evaluated
                if (cNDecisionFirst >= 0) {
                  vNPast = std::min(vNPast, cNDecisionFirst);
                }

                // if there is an alora invoked, don't cache after the invocation start
                if (pSlot.vAloraInvocationStart > 0) {
                  SLT_DBG(pSlot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n",
                          vNPast, pSlot.vAloraInvocationStart);
                  vNPast = std::min(vNPast, pSlot.vAloraInvocationStart - 1);
                }

                const auto cNCacheReuse = pSlot.vTask->vParams.vNCacheReuse;

                const bool cCanCacheReuse =
                  fDgsMemoryCanShift(fDgsGetMemory(vCtxTgt)) && !pSlot.vPrompt.vTokens.has_mtmd;

                if (!cCanCacheReuse && cNCacheReuse > 0) {
                  SLT_WRN(pSlot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", cNCacheReuse);
                }

                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                if (cCanCacheReuse && cNCacheReuse > 0) {
                  GGML_ASSERT(!pSlot.vPrompt.vTokens.has_mtmd);

                  size_t vHeadC = vNPast;  // cache
                  size_t vHeadP = vNPast;  // current prompt

                  if (vMctx) {
                    // we should never reach this
                    GGML_ABORT("not supported by multimodal");
                  }

                  SLT_DBG(pSlot, "trying to reuse chunks with size > %d, n_past = %d\n", cNCacheReuse, vNPast);

                  while (vHeadC < pSlot.vPrompt.vTokens.size() && vHeadP < vInputTokens.size()) {
                    size_t vNMatch = 0;
                    while (vHeadC + vNMatch < pSlot.vPrompt.vTokens.size() && vHeadP + vNMatch < vInputTokens.size() &&
                           pSlot.vPrompt.vTokens[vHeadC + vNMatch] == vInputTokens[vHeadP + vNMatch]) {
                      vNMatch++;
                    }

                    if (vNMatch >= (size_t) cNCacheReuse) {
                      SLT_TRC(pSlot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n",
                              vNMatch, vHeadC, vHeadC + vNMatch, vHeadP, vHeadP + vNMatch);
                      //for (size_t i = head_p; i < head_p + n_match; i++) {
                      //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx_tgt, prompt_tokens[i]).c_str());
                      //}

                      const int64_t cKvShift = (int64_t) vHeadP - (int64_t) vHeadC;

                      pSlot.vMem.mSeqRm(pSlot.vId, vHeadP, vHeadC);
                      pSlot.vMem.mSeqAdd(pSlot.vId, vHeadC, vHeadC + vNMatch, cKvShift);

                      for (size_t vI = 0; vI < vNMatch; vI++) {
                        pSlot.vPrompt.vTokens.mSetToken(vHeadP + vI, pSlot.vPrompt.vTokens[vHeadC + vI]);
                        vNPast++;
                      }

                      vHeadC += vNMatch;
                      vHeadP += vNMatch;
                    } else {
                      vHeadC += 1;
                    }
                  }

                  SLT_DBG(pSlot, "after context reuse, new n_past = %d\n", vNPast);
                }
              } else {
                // if we don't cache the prompt, we have to remove all previous tokens
                vNPast = 0;
              }

              dgs_pos vPosNext = pSlot.vPrompt.vTokens.mPosNext(vNPast);

              const bool cHasNewTokens = (vNPast < pSlot.vTask->mNTokens());

              // the largest pos_min required for a checkpoint to be useful
              const auto cPosMinThold = std::max(0, vPosNext - vNSwa - (cHasNewTokens ? 0 : 1));

              if (vNPast > 0 && vNPast <= pSlot.vPrompt.mNTokens()) {
                const auto cPosMin = fDgsMemorySeqPosMin(fDgsGetMemory(vCtxTgt), pSlot.vId);
                if (cPosMin == -1) {
                  SLT_ERR(pSlot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", vNPast,
                          (int) pSlot.vPrompt.vTokens.size(), pSlot.vId, cPosMin);
                  GGML_ABORT(
                    "pos_min == -1, but n_past > 0 - should not happen: "
                    "doc/MANUAL.md");
                }

                // when the prompt prefix does not match, print the tokens around the mismatch
                // this is useful for debugging prompt caching
                if (vSlotsDebug) {
                  const int cNp0 = std::max<int>(vNPast - vSlotsNDiff, 0);
                  const int cNp1 = std::min<int>(vNPast + vSlotsNDiff + 2,
                                                 std::min(pSlot.vPrompt.vTokens.size(), pSlot.vTask->vTokens.size()));

                  std::stringstream vSs0;
                  std::stringstream vSs1;

                  std::stringstream vSt0;
                  std::stringstream vSt1;

                  vSs0 << "old: ... ";
                  vSs1 << "new: ... ";

                  for (int vI = cNp0; vI < cNp1; vI++) {
                    if (vI == vNPast) {
                      vSs0 << " | ";
                      vSs1 << " | ";
                    }

                    {
                      const auto token  = pSlot.vPrompt.vTokens[vI];
                      const auto cPiece = token != DGS_TOKEN_NULL ? fCommonTokenToPiece(vCtxTgt, token) : "[mtmd]";
                      vSs0 << cPiece;
                      vSt0 << std::setw(8) << token;
                    }

                    {
                      const auto token  = pSlot.vTask->vTokens[vI];
                      const auto cPiece = token != DGS_TOKEN_NULL ? fCommonTokenToPiece(vCtxTgt, token) : "[mtmd]";
                      vSs1 << cPiece;
                      vSt1 << std::setw(8) << token;
                    }
                  }

                  SLT_WRN(pSlot, "%s\n", vSs0.str().c_str());
                  SLT_WRN(pSlot, "%s\n", vSs1.str().c_str());

                  SLT_WRN(pSlot, "%s\n", vSt0.str().c_str());
                  SLT_WRN(pSlot, "%s\n", vSt1.str().c_str());
                }

                if (cPosMin >= cPosMinThold) {
                  // search for a context checkpoint
                  const auto cIt = std::find_if(pSlot.vPrompt.lCheckpoints.rbegin(), pSlot.vPrompt.lCheckpoints.rend(),
                                                [&](const auto & pCur) {
                                                  // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                                  SLT_TRC(pSlot, "checking checkpoint with [%d, %d] against %d...\n",
                                                          pCur.vPosMin, pCur.vPosMax, cPosMinThold);
                                                  // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
                                                  if (pCur.vPosMax > vPosNext) {
                                                    return false;
                                                  }
                                                  return pCur.vPosMin < cPosMinThold || pCur.vPosMin == 0;
                                                });

                  bool vDoReset = cIt == pSlot.vPrompt.lCheckpoints.rend();

                  if (!vDoReset) {
                    // restore the context checkpoint
                    if (!cIt->mLoadTgt(vCtxTgt, pSlot.vId, DGS_STATE_SEQ_FLAGS_PARTIAL_ONLY) ||
                        !cIt->mLoadDft(vCtxDft, pSlot.vId, DGS_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
                      if (cIt->vIdTask != -1) {
                        GGML_ABORT("failed to restore context checkpoint\n");
                      }
                      // restored from a slot file, not guaranteed to load - fall back to full prompt re-processing
                      SLT_WRN(pSlot, "%s", "failed to load context checkpoint restored from a slot file\n");
                      vDoReset = true;
                    }
                  }

                  if (!vDoReset) {
                    // restore the draft's speculative state
                    fCommonSpeculativeSetState(vSpec.get(), pSlot.vId, cIt->lDataSpec);

                    vPosNext = std::min(vPosNext, std::max(cIt->vPosMin + 1, cIt->vPosMax));
                    vNPast   = std::min(pSlot.vPrompt.vTokens.mSizeUpToPos(vPosNext), (size_t) cIt->vNTokens);
                    SLT_TRC(pSlot,
                            "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64
                            ", n_past = %d, size = %.3f MiB)\n",
                            cIt->vPosMin, cIt->vPosMax, cIt->vNTokens, vNPast, (float) cIt->size() / 1024 / 1024);
                  }

                  if (vDoReset) {
                    SLT_TRC(pSlot,
                            "forcing full prompt re-processing due to lack of cache data (likely due to SWA or "
                            "hybrid/recurrent memory, see %s)\n",
                            "doc/MANUAL.md");
                    vPosNext = 0;
                    vNPast   = 0;
                  }
                }
              }

              {
                // erase any checkpoints with pos_max > pos_next
                for (auto vIt = pSlot.vPrompt.lCheckpoints.begin(); vIt != pSlot.vPrompt.lCheckpoints.end();) {
                  const auto & vCur = *vIt;
                  if (vCur.vPosMax > vPosNext) {
                    SLT_TRC(pSlot,
                            "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64
                            ", n_swa = %d, pos_next = %d, size = %.3f MiB)\n",
                            vCur.vPosMin, vCur.vPosMax, vCur.vNTokens, vNSwa, vPosNext,
                            (float) vCur.size() / 1024 / 1024);
                    vIt = pSlot.vPrompt.lCheckpoints.erase(vIt);
                  } else {
                    ++vIt;
                  }
                }
              }
            }

            // [TAG_PROMPT_LOGITS]
            if (vNPast == pSlot.vTask->mNTokens() && vNPast > 0) {
              SLT_WRN(pSlot,
                      "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n",
                      vNPast, pSlot.vTask->mNTokens());
              vNPast--;
              SLT_WRN(pSlot, "n_past was set to %d\n", vNPast);
            }

            pSlot.vStats.vNPromptCached    = vNPast;
            pSlot.vStats.vNPromptProcessed = 0;

            vMetrics.mAddPromptCached(vNPast);

            pSlot.vPrompt.vTokens.mKeepFirst(vNPast);

            // this is to signal the client that the request has started processing
            if (pSlot.vTask->vParams.stream) {
              if (pSlot.vTask->vParams.vReturnProgress) {
                // send initial 0% progress update if needed
                mSendPartialResponse(pSlot, {}, true);
              } else {
                // otherwise, for streaming without progress, signal HTTP to send the headers (i.e. 200 status)
                mSendPartialResponse(pSlot, {}, false, true);
              }
            }
          }  // end of SLOT_STATE_STARTED

          if (!pSlot.mCanSplit()) {
            // cannot fit the prompt in the current batch - will try next iter
            if (vBatch.size() + pSlot.vTask->mNTokens() > vNBatch) {
              return;
            }
          }

          // note: the prompt timing is advanced in post_decode(), so it does not cover
          //       the tokens added to the batch below
          pSlot.mPrintTimingsPp();

          // truncate any tokens that are beyond n_past for this slot
          const dgs_pos cP0 = pSlot.vPrompt.vTokens.mPosNext();

          SLT_TRC(pSlot, "cached n_tokens = %d, memory_seq_rm [%d, end)\n", pSlot.vPrompt.mNTokens(), cP0);

          pSlot.vMem.mSeqRm(pSlot.vId, cP0, -1);

          // shared prompt prefix: once it is processed, the children continue from it with their own prompt
          bool vWaitShared = false;
          if (pSlot.vTask->vNTokensShared > 0) {
            const bool cIsSharedDone = pSlot.vPrompt.mNTokens() == pSlot.vTask->vNTokensShared;
            for (auto & vOther : lSlots) {
              if (vOther.vState != cSlotStateWaitOther || vOther.vTask->vIdParent != pSlot.vTask->vId) {
                continue;
              }
              if (cIsSharedDone) {
                SLT_TRC(pSlot, " - copying shared prompt (%d tokens) to child %d\n", pSlot.vPrompt.mNTokens(),
                        vOther.vId);
                pSlot.mCopyPromptTo(vOther);
                vOther.vState = cSlotStateStarted;
              } else {
                vWaitShared = true;
              }
            }
          }

          // If using an alora, there may be uncached tokens that come
          // before the invocation sequence. When this happens, the
          // tokens before the invocation sequence need to be
          // processed without the adapter in a separate batch, then
          // the adapter needs to be enabled for the remaining tokens.
          if (fLoraAllAlora(pSlot.lora) && pSlot.vAloraInvocationStart - 1 > pSlot.vPrompt.mNTokens()) {
            SLT_DBG(pSlot,
                    "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n",
                    pSlot.vPrompt.mNTokens(), pSlot.vAloraInvocationStart);
            const auto & lEnabledLoras = fLoraGetEnabledIds(pSlot.lora);
            GGML_ASSERT(lEnabledLoras.size() == 1);
            vAloraScale                         = pSlot.lora[lEnabledLoras[0]].vScale;
            pSlot.lora[lEnabledLoras[0]].vScale = 0.0f;
            vAloraDisabledId                    = lEnabledLoras[0];
          }

          bool vDoCheckpoint = vParamsBase.vNCtxCheckpoints > 0;

          // make checkpoints only for completion tasks
          vDoCheckpoint = vDoCheckpoint && pSlot.vTask->type == cServerTaskTypeCompletion;

          // make a checkpoint of the parts of the memory that cannot be rolled back.
          // checkpoints are created only if:
          // - the model does not support partial sequence removal
          // - the model uses SWA (and we are not using `swa_full`)
          // - the model supports partial sequence removal but only up to a fixed bound
          vDoCheckpoint = vDoCheckpoint && (vCtxTgtSeqRmType == cCommonContextSeqRmTypeFull ||
                                            vCtxTgtSeqRmType == cCommonContextSeqRmTypeRs || vNSwa > 0);

          // TODO: do the same for all models, then remove process_mtmd_chunk()
          if (mUseMixedBatch() && !pSlot.mCanSplit() && vInputTokens.has_mtmd) {
            if (!mAddPromptMixed(pSlot)) {
              return;
            }
          }

          bool vHasMtmd = false;

          // check if we should process the mtmd chunk
          while (true) {
            auto vCurTokenIdx = pSlot.vPrompt.mNTokens();
            if (vCurTokenIdx >= pSlot.vTask->mNTokens() ||
                vInputTokens[vCurTokenIdx] != DGS_TOKEN_NULL  // encountered a text token
            ) {
              break;
            }

            // process the mtmd chunk
            // note: it submits its own decode, potentially be async
            //       so the timing is queued and flushed on the next sync
            mMetricsPreDecode();

            // encode on the worker thread, so we can still handle metrics tasks
            size_t  vNTokensOut = 0;
            int32_t vRes        = 0;
            vQueueTasks.mYieldToQueue(
              [&]() { vRes = fProcessMtmdChunk(pSlot, pSlot.vMbatch, vCurTokenIdx, vNTokensOut); });

            if (vRes != 0) {
              SLT_ERR(pSlot, "failed to process mtmd chunk, res = %d\n", vRes);
              mSendError(pSlot, "failed to process mtmd chunk", cErrorTypeServer);
              pSlot.mRelease();
              return;  // the slot is done, skip it entirely
            }

            mMetricsQueuePrompt(vNTokensOut);
            pSlot.vStats.vNPromptProcessed += vNTokensOut;
            pSlot.vStats.mUpdatePromptLast();

            // add the mtmd chunk to cache
            {
              const auto & vChunk = vInputTokens.mFindChunk(vCurTokenIdx);
              // the chunk is already in the KV cache at this point, so we don't need to keep its data around
              pSlot.vPrompt.vTokens.mPushBackPlaceholder(vChunk.get());
            }

            vHasMtmd = true;
          }

          const auto & vSpans       = pSlot.vTask->vParams.vMessageSpans;
          const auto   cLastUserPos = vSpans.mLastUserMessagePos();

          const int32_t cNDecisionFirst =
            pSlot.vTask->type == cServerTaskTypeDecision ? pSlot.vTask->decision.mPosFirst() : -1;

          // add prompt tokens for processing in the current batch
          while (pSlot.vPrompt.mNTokens() < pSlot.vTask->mNTokens() && vBatch.size() < vNBatch) {
            // get next token to process
            dgs_token vCurTok = vInputTokens[pSlot.vPrompt.mNTokens()];
            if (vCurTok == DGS_TOKEN_NULL) {
              break;  // end of text chunk
            }

            // stop at the end of the shared prefix, the children are started from this state
            if (vWaitShared && pSlot.vPrompt.mNTokens() == pSlot.vTask->vNTokensShared) {
              break;
            }

            // the outputs of a decision are read from one batch, do not split them
            if (pSlot.vPrompt.mNTokens() == cNDecisionFirst &&
                vBatch.size() + pSlot.vTask->mNTokens() - cNDecisionFirst > vNBatch) {
              break;
            }

            // if this is an alora request with pre-invocation
            // tokens that are not cached, we need to stop filling
            // this batch at those pre-invocation tokens.
            if (vAloraScale > 0 && pSlot.vPrompt.mNTokens() == pSlot.vAloraInvocationStart - 1) {
              SLT_DBG(pSlot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n",
                      pSlot.vPrompt.mNTokens(), pSlot.vAloraInvocationStart);
              break;
            }

            // embedding requires all tokens in the batch to be output;
            // MTP also wants logits at every prompt position so the
            // streaming hook can mirror t_h_nextn into ctx_dft.
            vAddOk &= vBatch.mAdd(pSlot.vId, vCurTok,
                                  /* pos       = */ pSlot.vPrompt.vTokens.mPosNext(),
                                  /* output    = */ pSlot.mNeedEmbd(),
                                  /* is_prompt = */ true);
            if (!pSlot.vTask->decision.lOrder.empty()) {
              vBatch.mSetDecisionOrder(vBatch.size() - 1, pSlot.vTask->decision.lOrder[pSlot.vPrompt.mNTokens()]);
            }
            pSlot.vPrompt.vTokens.push_back(vCurTok);

            // break at the last user message, or at user messages at least min step past the last checkpoint
            if (vDoCheckpoint && vSpans.mIsUserStart(pSlot.vPrompt.mNTokens())) {
              const auto   cPos         = pSlot.vPrompt.mNTokens();
              const auto & lCheckpoints = pSlot.vPrompt.lCheckpoints;

              if (cPos == cLastUserPos || lCheckpoints.empty() ||
                  cPos > lCheckpoints.back().vNTokens + vParamsBase.vCheckpointMinStep) {
                break;
              }
            }

            // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
            // create checkpoints that many tokens before the end of the prompt:
            //  - 4 + n_ubatch
            //  - 4
            if (vDoCheckpoint) {
              static const int aCheckpointOffsets[] = { 4 + vNUbatch, 4 };

              bool vShouldBreak = false;
              for (int vOffset : aCheckpointOffsets) {
                const int cNLast = std::min(vNBatch, vOffset);
                if (pSlot.vTask->mNTokens() == pSlot.vPrompt.mNTokens() + cNLast) {
                  vShouldBreak = true;
                  break;
                }
              }
              if (vShouldBreak) {
                break;
              }
            }
          }

          // the number of tokens added to the batch for the current slot
          const auto cNTokensCur = vBatch.size() - cNTokensPrev;

          const auto cNTokensStart = pSlot.vPrompt.mNTokens() - cNTokensCur;

          const bool cNearPromptEnd = pSlot.vTask->mNTokens() < pSlot.vPrompt.mNTokens() + vNUbatch;

          const bool cIsUserStart       = vSpans.mIsUserStart(cNTokensStart);
          const bool cIsLastUserMessage = cNTokensStart == cLastUserPos;

          // entire prompt has been processed
          if (pSlot.vPrompt.mNTokens() == pSlot.vTask->mNTokens()) {
            pSlot.vState = cSlotStateDonePrompt;

            GGML_ASSERT(vBatch.size() > 0);

            // extract the logits only for the last token
            vBatch.mSetOutput(vBatch.size() - 1, true);

            pSlot.vStats.vNGen = 0;
            pSlot.vIBatch      = vBatch.size() - 1;

            pSlot.mInitSampler();
          } else {
            // skip ordinary mid-prompt checkpoints, unless the batch starts a user
            // message or we are near the end of the prompt
            if (!cIsUserStart && !cNearPromptEnd) {
              vDoCheckpoint = false;
            }
          }

          const auto cPosMin = fDgsMemorySeqPosMin(fDgsGetMemory(vCtxTgt), pSlot.vId);
          const auto cPosMax = fDgsMemorySeqPosMax(fDgsGetMemory(vCtxTgt), pSlot.vId);

          // nothing to checkpoint yet
          // TODO: is this check needed?
          if (vDoCheckpoint && cPosMin < 0) {
            vDoCheckpoint = false;
          }

          // do not checkpoint after mtmd chunks
          vDoCheckpoint = vDoCheckpoint && !vHasMtmd;

          // no need to create checkpoints that are too close together, unless it's the last user message
          vDoCheckpoint = vDoCheckpoint &&
                          (pSlot.vPrompt.lCheckpoints.empty() || cIsLastUserMessage || cNearPromptEnd ||
                           cNTokensStart > pSlot.vPrompt.lCheckpoints.back().vNTokens + vParamsBase.vCheckpointMinStep);
          SLT_DBG(pSlot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", vDoCheckpoint ? "yes" : "no", cPosMin,
                  cPosMax);

          // note: we create the checkpoint before calling dgs_decode(), so the current batch is not
          //       yet processed and therefore it is not part of the checkpoint.
          if (vDoCheckpoint) {
            mCreateCheckpoint(pSlot, cNTokensCur, cPosMin, cPosMax);
          }
        }

        if (!vSlotBatched) {
          vSlotBatched = &pSlot;
        }
      });
    }
  }

  // TODO @ngxson : maybe remove this once we use "mixed" batch everywhere
  bool mUseMixedBatch() const { return !fDgsGetMemory(vCtxTgt) || !fDgsGetCausalAttn(vCtxTgt); }

  // add the rest of the prompt to the batch, the mtmd chunks are added as embeddings next to the text tokens
  // the caller makes sure that it fits in the batch
  // returns false on error, the slot is then released
  bool mAddPromptMixed(server_slot & pSlot) {
    const auto & vInputTokens = pSlot.vTask->vTokens;
    const auto   cNTokensPrev = vBatch.size();

    while (pSlot.vPrompt.mNTokens() < pSlot.vTask->mNTokens()) {
      const auto      cCurTokenIdx = pSlot.vPrompt.mNTokens();
      const dgs_token cCurTok      = vInputTokens[cCurTokenIdx];

      if (cCurTok != DGS_TOKEN_NULL) {
        const bool cAddOk = vBatch.mAdd(pSlot.vId, cCurTok,
                                        /* pos       = */ pSlot.vPrompt.vTokens.mPosNext(),
                                        /* output    = */ pSlot.mNeedEmbd(),
                                        /* is_prompt = */ true);
        GGML_ASSERT(cAddOk);
        if (!pSlot.vTask->decision.lOrder.empty()) {
          vBatch.mSetDecisionOrder(vBatch.size() - 1, pSlot.vTask->decision.lOrder[cCurTokenIdx]);
        }
        pSlot.vPrompt.vTokens.push_back(cCurTok);
        continue;
      }

      const auto & vChunk = vInputTokens.mFindChunk(cCurTokenIdx);

      float * vEmbd = pSlot.vMbatch ? fMtmdBatchGetOutputEmbd(pSlot.vMbatch.get(), vChunk.get()) : nullptr;
      if (!vEmbd) {
        // encode on the worker thread, so we can still handle metrics tasks
        int32_t vRes = 0;
        vQueueTasks.mYieldToQueue([&]() { vRes = fEncodeMtmdChunk(pSlot, pSlot.vMbatch, cCurTokenIdx); });
        vEmbd = vRes == 0 ? fMtmdBatchGetOutputEmbd(pSlot.vMbatch.get(), vChunk.get()) : nullptr;
      }

      if (!vEmbd || !fAddMtmdChunk(pSlot, vChunk.get(), vEmbd, vBatch)) {
        SLT_ERR(pSlot, "%s", "failed to process mtmd chunk\n");
        // the batch must not keep the entries of a released slot
        vBatch.mTruncate(cNTokensPrev);
        mSendError(pSlot, "failed to process mtmd chunk", cErrorTypeServer);
        pSlot.mRelease();
        return false;
      }

      pSlot.vPrompt.vTokens.mPushBackPlaceholder(vChunk.get());
    }

    return true;
  }

  // returns true = success ; false = retry with smaller batch size
  // throw std::runtime_error on fatal error
  bool mDecode(int32_t & pNBatch, int32_t pOff) {
    SRV_DBG("n_batch (effective) = %d, off = %d\n", pNBatch, pOff);

    mMetricsPreDecode();

    if (vBatch.size() == 0) {
      SRV_WRN("%s", "no tokens to decode\n");

      if (++vNEmptyConsecutive > 3) {
        GGML_ABORT("fatal error - please provide logs and repro in %s\n",
                   "doc/MANUAL.md");
      }

      return true;  // nothing to decode
    } else {
      vNEmptyConsecutive = 0;
    }

    // TODO @ngxson : dft model may have different n_embd than the tgt model, so we check & reject if that's the case
    // this case is not currently used by any models, but may need to be supported in the future
    if (vSpec && vBatch.mHasEmbd()) {
      if (fDgsModelNEmbdInp(vModelDft) != fDgsModelNEmbdInp(vModelTgt)) {
        SRV_ERR("%s", "unsupported batch.has_embd + spec case\n");
        throw std::runtime_error("unsupported batch.has_embd + spec case");
      }
    }

    bool vHasOutput = false;
    for (int vI = pOff; vI < pOff + vBatch.vView.size(); ++vI) {
      vHasOutput |= vBatch.lTokens[vI].vOutput;
    }

    // yield to the queue, so we can still handle metrics tasks while decoding
    // note: the sync is done here too, so that the wait is also covered by the yield
    int vRet = 0;
    vQueueTasks.mYieldToQueue([&]() {
      vRet = fDgsProcess(vCtxTgt, cDgsProcessTypeDecode, vBatch.vView.get());
      if (vRet == 0 && vHasOutput) {
        fDgsSynchronize(vCtxTgt);
      }
    });

    if (vRet != 0) {
      {
        std::string vErr;

        if (pNBatch == 1 && vRet == 1) {
          // TODO: try to terminate only the largest active slot/sequence and continue with the rest
          //       need to remove the tokens from the current batch too
          vErr = "Context size has been exceeded.";
        }

        if (vRet == -1) {
          vErr = "Invalid input batch.";
        }

        if (vRet < -1) {
          // TODO: update slot state based on dgs_memory_seq_pos_min() and dgs_memory_seq_pos_max()
          vErr = "Compute error.";
        }

        // TODO: handle ret == 2 (abort) when we start aborting

        if (!vErr.empty()) {
          SRV_ERR("%s off = %d, n_batch = %d, ret = %d\n", vErr.c_str(), pOff, pNBatch, vRet);

          for (auto & vSlot : lSlots) {
            if (vSlot.mIsProcessing()) {
              mSendError(vSlot, vErr);
              vSlot.mRelease();

              // note: it's complicated to keep track of how much of the current batch has been
              //       processed before the error occurred, so we simply clear the entire context
              vSlot.mPromptClear();
            }
          }

          // stop, do not retry with smaller batch size
          throw std::runtime_error(vErr);
        }
      }

      // retry with half the batch size to try to find a free slot in the KV cache
      if (!mTryClearIdleSlots()) {
        pNBatch /= 2;
      }

      SRV_WRN(
        "failed to find free space in the KV cache, retrying with smaller batch size, off = %d, n_batch = %d, ret = "
        "%d\n",
        pOff, pNBatch, vRet);

      return false;  // retry with the updated n_batch
    } else {
      // success, apply batch metrics
      mMetricsPostDecode(pOff, vBatch.vView.size(), vHasOutput);
    }

    // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
    //       for now, always re-evaluate for simplicity
    if (vSpec) {
      bool vOk = true;
      vQueueTasks.mYieldToQueue([&]() { vOk = fCommonSpeculativeProcess(vSpec.get(), vBatch.vView); });

      if (!vOk) {
        SRV_ERR("%s", "failed to process speculative batch\n");

        // TODO: handle error
        throw std::runtime_error("failed to process speculative batch");
      }
    }

    // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
    for (auto & vSlot : lSlots) {
      if (vSlot.vState == cSlotStateDonePrompt && vSlot.vTask->mIsParent()) {
        std::vector<server_slot *> lChildren;
        for (auto & vOther : lSlots) {
          if (vOther.vState == cSlotStateWaitOther && vSlot.vTask->vId == vOther.vTask->vIdParent) {
            lChildren.push_back(&vOther);
          }
        }

        // all children slots should already launched by launch_slots_with_parent_task()
        // copy state to the child slots
        for (auto & vChild : lChildren) {
          SLT_TRC(vSlot, " - copying state to child %d\n", vChild->vId);

          GGML_ASSERT(vChild->vState == cSlotStateWaitOther);
          // children with their own prompt are started at the end of the shared prefix
          GGML_ASSERT(vSlot.vTask->vNTokensShared == 0);

          vSlot.mCopyStateTo(*vChild);
          vChild->vState = cSlotStateDonePrompt;
        }
      }
    }

    return true;
  }

  void mPostDecode(int32_t pNBatchTokens, int32_t pOff) {
    // for checking if a given batch index is inside the current sub-batch
    auto vIsInsideView = [&](int32_t pIdx) {
      return pIdx >= pOff && pIdx < pOff + pNBatchTokens;
    };

    // TODO @ngxson : it's tricky to make sub-batch compatible with common_sampler_sample_and_accept_n,
    // so for now we will throw an error in this case: 
    mIterate(lSlots, [&](server_slot & pSlot) {
      for (auto & vI : pSlot.lSpecIBatch) {
        if (!vIsInsideView(vI)) {
          throw std::runtime_error(fStringFormat(
            "speculative batch index %d is not inside the current sub-batch [%d, %d)", vI, pOff, pOff + pNBatchTokens));
        }
      }
    });

    auto vAcceptSpecialToken = [&](server_slot & pSlot, dgs_token token) {
      return vParamsBase.vSpecial || pSlot.vTask->vParams.vSampling.sPreservedTokens.find(token) !=
                                       pSlot.vTask->vParams.vSampling.sPreservedTokens.end();
    };

    mIterate(lSlots, [&](server_slot & pSlot) {
      // optionally send prompt processing progress
      if (pSlot.vState == cSlotStateProcessingPrompt || pSlot.vState == cSlotStateDonePrompt) {
        if (pSlot.vTask->vParams.stream && pSlot.vTask->vParams.vReturnProgress) {
          mSendPartialResponse(pSlot, {}, true);
        }
      }

      if (!vIsInsideView(pSlot.vIBatch)) {
        // the required token not in this sub-batch, skip
        return;
      }

      if (pSlot.vState == cSlotStateDonePrompt) {
        if (pSlot.vTask->type == cServerTaskTypeEmbedding) {
          // prompt evaluated for embedding
          mSendEmbedding(pSlot, vBatch.vView);
          pSlot.mRelease();
          pSlot.vIBatch = -1;
          return;
        }

        if (pSlot.vTask->type == cServerTaskTypeRerank) {
          mSendRerank(pSlot, vBatch.vView);
          pSlot.mRelease();
          pSlot.vIBatch = -1;
          return;
        }

        if (pSlot.vTask->type == cServerTaskTypeDecision) {
          mSendDecision(pSlot, vBatch.vView, pSlot.vIBatch - pOff);
          pSlot.mRelease();
          pSlot.vIBatch = -1;
          return;
        }

        GGML_ASSERT(pSlot.vTask->mNeedSampling());

        // prompt evaluated for next-token prediction
        pSlot.vState = cSlotStateGenerating;

        if (pSlot.mCanSpeculate()) {
          fCommonSpeculativeBegin(vSpec.get(), pSlot.vId, pSlot.vPrompt.vTokens.mGetTextTokens());
        }
      } else if (pSlot.vState != cSlotStateGenerating) {
        return;
      }

      if (pSlot.mCanSpeculate() && !pSlot.lSpecDraft.empty()) {
        return;  // sample using speculative decoding
      }

      // shifted according to the current sub-batch
      const int cTokIdx = pSlot.vIBatch - pOff;

      dgs_token vId;
      {
        scoped_timer vTimer(vTSampl, vNSampl);
        vId = fCommonSamplerSample(pSlot.vSmpl.get(), pSlot.vCtxTgt, cTokIdx);
      }

      pSlot.vIBatch = -1;

      fCommonSamplerAccept(pSlot.vSmpl.get(), vId, true);

      // here we have synchronized the dgs_context (due to the sampling above), so we can do time measurement
      const int64_t cTNow = ggml_time_us();

      pSlot.vStats.vNGen += 1;

      if (pSlot.vStats.vNGen == 1) {
        pSlot.vStats.mUpdatePromptLast();
        pSlot.vTPrintLast = cTNow;
        pSlot.vNGenLast   = 0;
      }

      pSlot.vStats.mUpdateGenLast();

      completion_token_output vResult;
      vResult.vTok        = vId;
      vResult.vTextToSend = fCommonTokenToPiece(pSlot.vCtxTgt, vResult.vTok, vAcceptSpecialToken(pSlot, vResult.vTok));
      vResult.vProb       = 1.0f;  // TODO: set it here instead of doing inside populate_token_probs

      if (pSlot.vTask->vParams.vSampling.vNProbs > 0) {
        mPopulateTokenProbs(pSlot, vResult, pSlot.vTask->vParams.vPostSamplingProbs, vParamsBase.vSpecial, cTokIdx);
      }

      if (!mProcessToken(vResult, pSlot)) {
        // release slot because of stop condition
        pSlot.mPrintTimings();
        mSendFinalResponse(pSlot);
        pSlot.mRelease();

        return;
      }

      pSlot.mPrintTimingsTg();
    });

    // speculative decoding - main model sample and accept
    mIterate(lSlots, [&](server_slot & pSlot) {
      if (pSlot.vState != cSlotStateGenerating || !pSlot.mCanSpeculate() || pSlot.lSpecDraft.empty() ||
          pSlot.lSpecIBatch.empty()) {
        return;
      }

      // save the original draft size
      const size_t cNDraft = pSlot.lSpecDraft.size();

      GGML_ASSERT(cNDraft > 0);

      // verify and try to accept the draft
      {
        common_sampler_ptr vSmplSave(fCommonSamplerClone(pSlot.vSmpl.get()));

        GGML_ASSERT(pSlot.lSpecIBatch.size() == cNDraft + 1);
        GGML_ASSERT(pSlot.lSpecDraftQ.empty() || (pSlot.lSpecDraftQ.size() == pSlot.lSpecDraft.size()));
        const auto & lSynthProbs = fCommonSpeculativeGetSynthProbs(vSpec.get());

        // drafters that fill no distribution fall back here
        const bool cUseRejection = pSlot.mUseSpecRejection() && !pSlot.lSpecDraftQ.empty();

        std::vector<dgs_token> lAccepted;
        if (!lSynthProbs.empty()) {
          // synthetic acceptance replaces verification entirely, so it comes first
          lAccepted = fServerSampleAndAcceptSynth(pSlot.vSmpl.get(), pSlot.vCtxTgt, pSlot.lSpecIBatch, pSlot.lSpecDraft,
                                                  lSynthProbs, pSlot.vSpecSynthRng, pSlot.vSpecIsReplay);
        } else if (pSlot.vSpecIsReplay && pSlot.mUseSpecRejection()) {
          lAccepted = fServerAcceptReplay(pSlot.vSmpl.get(), pSlot.vCtxTgt, pSlot.lSpecIBatch, pSlot.lSpecDraft);
        } else if (cUseRejection) {
          lAccepted = fCommonSamplerSampleAndAcceptNRejection(pSlot.vSmpl.get(), pSlot.vCtxTgt, pSlot.lSpecIBatch,
                                                              pSlot.lSpecDraft, pSlot.lSpecDraftQ);
        } else {
          lAccepted =
            fCommonSamplerSampleAndAcceptN(pSlot.vSmpl.get(), pSlot.vCtxTgt, pSlot.lSpecIBatch, pSlot.lSpecDraft);
        }
        pSlot.lSpecIBatch.clear();

        GGML_ASSERT(lAccepted.size() >= 1);

        const uint32_t cNRollback = pSlot.lSpecDraft.size() + 1 - lAccepted.size();

        const bool cUseCkptTgt = vCtxTgtSeqRmType == cCommonContextSeqRmTypeFull ||
                                 (vCtxTgtSeqRmType == cCommonContextSeqRmTypeRs && cNRollback > fDgsNRsSeq(vCtxTgt));

        // check for partial draft acceptance
        if (cNRollback > 0) {
          if (cUseCkptTgt) {
            if (vTrace > 0) {
              SLT_INF(pSlot, "accepted %2zu/%2zu draft tokens (restore checkpoint)\n", lAccepted.size() - 1,
                      pSlot.lSpecDraft.size());
            }

            // partial acceptance is not supported by the context -> truncate the draft and restore the state
            pSlot.vSpecIsReplay = true;
            pSlot.lSpecDraft    = std::move(lAccepted);

            const auto & vCkpt = pSlot.vSpecCkpt;

            SLT_DBG(pSlot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n", vCkpt.vPosMin,
                    vCkpt.vPosMax, vCkpt.size());

            GGML_ASSERT(vCkpt.mLoadTgt(pSlot.vCtxTgt, pSlot.vId, DGS_STATE_SEQ_FLAGS_PARTIAL_ONLY));

            if (pSlot.vCtxDft) {
              GGML_ASSERT(vCkpt.mLoadDft(pSlot.vCtxDft, pSlot.vId, DGS_STATE_SEQ_FLAGS_PARTIAL_ONLY));
            }

            pSlot.vMem.mSeqRm(pSlot.vId, vCkpt.vPosMax + 1, -1);

            pSlot.vPrompt.vTokens.mKeepFirst(vCkpt.vNTokens);
            fCommonSamplerCopy(vSmplSave.get(), pSlot.vSmpl.get());

            return;
          }
        }

        if (vTrace > 0) {
          SLT_INF(pSlot, "accepted %2zu/%2zu draft tokens\n", lAccepted.size() - 1, cNDraft);
        }

        fCommonSpeculativeAccept(vSpec.get(), pSlot.vId, lAccepted.size() - 1);

        pSlot.lSpecDraft = std::move(lAccepted);
      }

      const auto lIds = std::move(pSlot.lSpecDraft);

      size_t vNAccepted = lIds.size() - 1;
      if (pSlot.vSpecIsReplay && vNAccepted > 0) {
        vNAccepted--;
      }
      pSlot.vSpecIsReplay = false;

      pSlot.vStats.mUpdateGenLast();

      // update how many tokens out of those tested were accepted
      pSlot.vStats.vNDraftAccepted += vNAccepted;
      pSlot.vStats.vNDraftVerifSteps += 1;

      auto & lNAcceptedPerPos = pSlot.lNAcceptedPerPos;
      if (lNAcceptedPerPos.empty()) {
        lNAcceptedPerPos.resize(fCommonSpeculativeNMax(vSpec.get()), 0);
      }
      for (size_t vI = 0; vI < vNAccepted && vI < lNAcceptedPerPos.size(); ++vI) {
        lNAcceptedPerPos[vI]++;
      }

      // add accepted tokens to the prompt
      pSlot.vPrompt.vTokens.mKeepFirst(pSlot.vPrompt.mNTokens() - cNDraft);
      pSlot.vPrompt.vTokens.insert({ lIds.begin(), lIds.end() - 1 });

      pSlot.vSampled = lIds.back();  // last accepted token
      SLT_DBG(pSlot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", pSlot.vSampled, lIds.size(),
              cNDraft);

      pSlot.vMem.mSeqRm(pSlot.vId, pSlot.vPrompt.vTokens.mPosNext(), -1);

      for (size_t vI = 0; vI < lIds.size(); ++vI) {
        completion_token_output vResult;

        vResult.vTok = lIds[vI];
        vResult.vTextToSend =
          fCommonTokenToPiece(pSlot.vCtxTgt, vResult.vTok, vAcceptSpecialToken(pSlot, vResult.vTok));
        vResult.vProb = 1.0f;  // set later

        // TODO: set result.probs

        pSlot.vStats.vNGen += 1;

        if (!mProcessToken(vResult, pSlot)) {
          pSlot.mPrintTimings();
          mSendFinalResponse(pSlot);
          pSlot.mRelease();

          return;
        }
      }

      pSlot.mPrintTimingsTg();

      SLT_DBG(pSlot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) vNAccepted, (int) cNDraft,
              pSlot.vPrompt.mNTokens());
    });
  }

  // context size of a single slot, capped by --kv-unified-per-slot and by the training context of the model
  int mNCtxSlot() const {
    int vRes = fDgsNCtxSeq(vCtxTgt);

    if (vParamsBase.vKvUnifiedPerSlot > 0) {
      vRes = std::min(vRes, vParamsBase.vKvUnifiedPerSlot);
    }

    return std::min(vRes, fDgsModelNCtxTrain(vModelTgt));
  }

  server_response_reader mGetResponseReader() {
    return server_response_reader(vQueueTasks, vQueueResults, cHttpPollingSeconds);
  }

  //
  // metrics helpers
  //

  // call before submitting a decode, so that the queued prompt stats can be timed
  void mMetricsPreDecode() { vTDecodeStart = ggml_time_us(); }

  // the batch is submitted, but its compute may not be done yet
  void mMetricsQueuePrompt(uint64_t pNTokens) {
    if (pNTokens == 0) {
      return;
    }
    if (vNPromptQueued == 0) {
      vTPromptStart = vTDecodeStart;
    }
    vNPromptQueued += pNTokens;
  }

  // call only after the context is synchronized, otherwise the time is meaningless
  void mMetricsFlushPrompt() {
    if (vNPromptQueued == 0) {
      return;
    }
    vMetrics.mAddPrompt(vNPromptQueued, ggml_time_us() - vTPromptStart);
    vNPromptQueued = 0;
  }

  // has_output is computed by the caller, which also already synchronized the context if it is set
  void mMetricsPostDecode(int32_t pOff, int32_t pNTokens, bool pHasOutput) {
    vMetrics.vNDecode++;
    for (const auto & vSlot : lSlots) {
      if (vSlot.mIsProcessing()) {
        vMetrics.vNBusySlots++;
      }
      vMetrics.vNTokensMax = std::max(vMetrics.vNTokensMax, (uint64_t) vSlot.vPrompt.mNTokens());
    }

    // apply enqueued prompt tokens stats
    // note: a slot can be released before we get here, which clears its stats
    //       the tokens were still computed, counted in the global metrics, not in slot
    uint64_t vNPromptTokens = 0;

    for (int vI = pOff; vI < pOff + pNTokens; ++vI) {
      const auto & vT = vBatch.lTokens[vI];

      if (!vT.vIsPrompt) {
        continue;  // generated tokens are handled after sampling
      }

      vNPromptTokens++;

      auto & vSlot = lSlots[vT.vIdSlot];
      if (vSlot.vStats.mIsSet()) {
        vSlot.vStats.vNPromptProcessed++;
      }
    }

    mMetricsQueuePrompt(vNPromptTokens);

    if (pHasOutput) {
      // the context is already synchronized, so the timings are correct
      mMetricsFlushPrompt();
    }

    // advance the prompt timing of the slots that had tokens in this batch
    // note: a second pass, it must run after the sync to reflect the compute
    const int64_t cTNow = ggml_time_us();
    for (int vI = pOff; vI < pOff + pNTokens; ++vI) {
      const auto & vT    = vBatch.lTokens[vI];
      auto &       vSlot = lSlots[vT.vIdSlot];
      if (vT.vIsPrompt && vSlot.vStats.mIsSet()) {
        vSlot.vStats.mSetPromptLast(cTNow);
      }
    }
  }

  // flush any queued prompt metrics if all slots are now idle
  void mMetricsFlushIdle() {
    if (vNPromptQueued == 0) {
      return;
    }

    fDgsSynchronize(vCtxTgt);
    mMetricsFlushPrompt();
  }

  void mMetricsOnPrediction(const server_slot & pSlot) {
    const uint64_t cTUs    = pSlot.vStats.mTGenUs();
    const uint64_t n       = pSlot.vStats.vNGen;
    const uint64_t cNSteps = pSlot.vStats.mNGenSteps();

    vMetrics.vPredict.mAdd(n, cNSteps, cTUs);
    vMetrics.vPredictBucket.mAdd(n, cNSteps, cTUs);

    vMetrics.vNDraftTokens += pSlot.vStats.vNDraftTokens;
    vMetrics.vNDraftAccepted += pSlot.vStats.vNDraftAccepted;
    vMetrics.vNDraftVerifSteps += pSlot.vStats.vNDraftVerifSteps;

    auto &       lDst = vMetrics.lNAcceptedPerPos;
    const auto & lSrc = pSlot.lNAcceptedPerPos;

    if (lDst.size() < lSrc.size()) {
      lDst.resize(lSrc.size(), 0);
    }
    for (size_t vI = 0; vI < lSrc.size(); vI++) {
      lDst[vI] += lSrc[vI];
    }
  }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}

server_context::~server_context() = default;

bool server_context::mLoadModel(common_params & pParams) {
  return impl->mLoadModel(pParams);
}

void server_context::mStartLoop() {
  auto & vParams = impl->vParamsBase;
  impl->vQueueTasks.mStartLoop(vParams.vSleepIdleSeconds * 1000);
}

json server_context::mGetMonitorSnapshot() const {
  return impl->mGetMonitorSnapshot();
}

void server_context::mTerminate() {
  impl->vQueueTasks.mTerminate();
}

dgs_context * server_context::mGetDgsContext() const {
  return impl->vCtxTgt;
}

server_response_reader server_context::mGetResponseReader() {
  return impl->mGetResponseReader();
}

server_context_meta server_context::mGetMeta() const {
  auto vBosId       = fDgsVocabBos(impl->vVocab);
  auto vEosId       = fDgsVocabEos(impl->vVocab);
  auto vBosTokenStr = vBosId != DGS_TOKEN_NULL ? fCommonTokenToPiece(impl->vCtxTgt, vBosId, true) : "";
  auto vEosTokenStr = vEosId != DGS_TOKEN_NULL ? fCommonTokenToPiece(impl->vCtxTgt, vEosId, true) : "";

  const char * vFtypeName = fDgsFtypeName(fDgsModelFtype(impl->vModelTgt));

  return server_context_meta{
    /* build_info              */ std::string(fDgsBuildInfo()),
    /* model_name              */ impl->vModelName,
    /* model_aliases           */ impl->sModelAliases,
    /* model_tags              */ impl->sModelTags,
    /* model_path              */ impl->vParamsBase.vModel.vPath,
    /* model_output_modalities */ fServerModelOutputModalities(fCommonGetDecisionType(impl->vModelTgt)),
    /* has_mtmd                */ impl->vMctx != nullptr,
    /* has_inp_image           */ impl->vChatParams.vAllowImage,
    /* has_inp_audio           */ impl->vChatParams.vAllowAudio,
    /* has_inp_video           */ impl->vChatParams.vAllowVideo,
    /* json_ui_settings        */ impl->vJsonUiSettings,
    /* slot_n_ctx              */ impl->mNCtxSlot(),
    /* pooling_type            */ fDgsPoolingType(impl->vCtxTgt),

    /* chat_params             */ impl->vChatParams,
    /* chat_template_caps      */ fCommonChatTemplatesGetCaps(impl->vChatParams.vTmpls.get()),

    /* bos_token_str           */ vBosTokenStr,
    /* eos_token_str           */ vEosTokenStr,
    /* fim_pre_token           */ fDgsVocabFimPre(impl->vVocab),
    /* fim_sub_token           */ fDgsVocabFimSuf(impl->vVocab),
    /* fim_mid_token           */ fDgsVocabFimMid(impl->vVocab),
    /* fim_pad_token           */ fDgsVocabFimPad(impl->vVocab),
    /* fim_rep_token           */ fDgsVocabFimRep(impl->vVocab),
    /* fim_sep_token           */ fDgsVocabFimSep(impl->vVocab),

    /* logit_bias_eog          */ impl->vParamsBase.vSampling.lLogitBiasEog,

    /* model_vocab_type        */ fDgsVocabType(impl->vVocab),
    /* model_vocab_n_tokens    */ fDgsVocabNTokens(impl->vVocab),
    /* model_n_ctx_train       */ fDgsModelNCtxTrain(impl->vModelTgt),
    /* model_n_embd_inp        */ fDgsModelNEmbd(impl->vModelTgt),
    /* model_n_params          */ fDgsModelNParams(impl->vModelTgt),
    /* model_size              */ fDgsModelSize(impl->vModelTgt),
    /* model_ftype             */ vFtypeName,
  };
}

// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_res_spipe {
  server_response_reader vRd;

  server_res_generator(server_queue &    pQueueTasks,
                       server_response & pQueueResults,
                       int               pSleepIdleSeconds,
                       bool              pBypassSleep = false) :
    vRd(pQueueTasks, pQueueResults, cHttpPollingSeconds) {
    // fast path in case sleeping is disabled
    pBypassSleep |= pSleepIdleSeconds < 0;
    if (!pBypassSleep) {
      pQueueTasks.mWaitUntilNoSleep();
    }
  }

  void mOk(const json & pResponseData) {
    vStatus = 200;
    data    = fSafeJsonToStr(pResponseData);
  }

  void mError(const json & pErrorData) {
    vStatus = fJsonValue(pErrorData, "code", 500);
    data    = fSafeJsonToStr({
      { "error", pErrorData }
    });
  }
};

void server_context::mSetStateCallback(server_state_callback_t pCallback) {
  impl->vCallbackState = std::move(pCallback);
}

//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::mHandleCompletionsImpl(const server_http_req &         pReq,
                                                                            server_task_type                type,
                                                                            const json &                    data,
                                                                            const std::vector<raw_buffer> & pFiles,
                                                                            task_response_type              pResType,
                                                                            const common_chat_session & pChatSession) {
  GGML_ASSERT(type == cServerTaskTypeCompletion || type == cServerTaskTypeInfill);

  auto   vRes          = mCreateResponse();
  auto   vCompletionId = fGenChatcmplid();
  auto & vRd           = vRes->vRd;
  auto & vParams       = this->vParams;

  vRes->mSetReq(&pReq);  // will also set spipe if needed

  int32_t vSsePingInterval = vParams.vSsePingInterval;

  try {
    std::vector<server_task> lTasks;

    const auto & vPrompt = data.at("prompt");
    // TODO: this log can become very long, put it behind a flag or think about a more compact format
    //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

    if (!vParams.vPathPromptsLogDir.empty()) {
      const auto cFilePath =
        std::filesystem::path(vParams.vPathPromptsLogDir) / fStringFormat("%012" PRId64 ".txt", ggml_time_ms());
      std::ofstream vF(cFilePath);
      if (vF) {
        vF << (vPrompt.is_string() ? vPrompt.get<std::string>().c_str() : vPrompt.mDump(2).c_str());
      } else {
        SRV_ERR("failed to create %s\n", cFilePath.string().c_str());
      }
    }

    // process prompt
    std::vector<server_tokens> lInputs;

    if (pResType != cTaskResponseTypeNone && vCtxServer.vMctx != nullptr) {
      // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
      lInputs.push_back(fProcessMtmdPrompt(vCtxServer.vMctx, vPrompt.get<std::string>(), pFiles, vCtxServer.vInitOpt));
    } else {
      // Everything else, including multimodal completions.
      lInputs = fTokenizeInputPrompts(vCtxServer.vVocab, vCtxServer.vMctx, vPrompt, true, true, vCtxServer.vInitOpt);
    }

    // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

    for (size_t vI = 0; vI < lInputs.size(); vI++) {
      server_task vTask = server_task(type);

      vTask.vId = vRd.mGetNewId();

      vTask.vTokens = std::move(lInputs[vI]);
      vTask.vParams = server_schema::fEvalDgsCmplSchema(vCtxServer.vVocab, vParams, vMeta->lLogitBiasEog, data);

      vTask.mApplyChatSession(pChatSession);

      vTask.vIdSlot    = fJsonValue(data, "id_slot", -1);
      vSsePingInterval = vTask.vParams.vSsePingInterval;

      // OAI-compat
      vTask.vParams.vResType         = pResType;
      vTask.vParams.vOaicompatCmplId = vCompletionId;
      vTask.vParams.vOaicompatModel  = vMeta->vModelName;

      vTask.vParams.vQueueNotice = vTask.vParams.vFromWebui && vTask.vParams.stream &&
                                   pResType != cTaskResponseTypeAnthropic && pResType != cTaskResponseTypeOaiResp;

      // prepare child tasks
      if (vTask.vParams.vNCmpl > 1) {
        int vNChildren = vTask.vParams.vNCmpl - 1;
        for (int vJ = 0; vJ < vNChildren; vJ++) {
          vTask.mAddChild(vTask.vId, vRd.mGetNewId());
        }
      }

      lTasks.push_back(std::move(vTask));
    }

    vRd.mPostTasks(std::move(lTasks), pChatSession);
  } catch (const std::exception & vE) {
    vRes->mError(fFormatErrorResponse(vE.what(), cErrorTypeInvalidRequest));
    return vRes;
  }

  bool stream = fJsonValue(data, "stream", false);

  if (!stream) {
    // non-stream, wait for the results
    auto vAllResults = vRd.mWaitForAll(pReq.vShouldStop);
    if (vAllResults.vIsTerminated) {
      return vRes;  // connection is closed
    } else if (vAllResults.vError) {
      vRes->mError(vAllResults.vError->to_json());
      return vRes;
    } else {
      json vArr = json::mArray();
      for (auto & vRes : vAllResults.lResults) {
        GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final *>(vRes.get()) != nullptr);
        vArr.push_back(vRes->to_json());
      }
      GGML_ASSERT(!vArr.empty() && "empty results");
      if (vArr.size() == 1) {
        // if single request, return single object instead of array
        vRes->mOk(vArr[0]);
      } else if (pResType == cTaskResponseTypeOaiChat || pResType == cTaskResponseTypeOaiCmpl) {
        // if multiple results in OAI format, we need to re-format them
        json & vChoices = vArr[0]["choices"];
        for (size_t vI = 1; vI < vArr.size(); vI++) {
          vChoices.push_back(std::move(vArr[vI]["choices"][0]));
        }
        vRes->mOk(vArr[0]);
      } else {
        // multi-results, non-OAI compat
        vRes->mOk(vArr);
      }
    }
  } else {
    // in streaming mode, the first error must be treated as non-stream response
    // this is to match the OAI API behavior
    auto vFirstResult = vRd.mNext(pReq.vShouldStop);
    if (vFirstResult == nullptr) {
      GGML_ASSERT(pReq.vShouldStop());
      return vRes;  // connection is closed
    }

    if (vFirstResult->mIsError()) {
      vRes->mError(vFirstResult->to_json());
      return vRes;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_cmpl_partial *>(vFirstResult.get()) != nullptr ||
                dynamic_cast<server_task_result_cmpl_final *>(vFirstResult.get()) != nullptr ||
                dynamic_cast<server_task_result_queue *>(vFirstResult.get()) != nullptr);

    // next responses are streamed
    // to be sent immediately
    json vFirstResultJson = vFirstResult->to_json();
    if (vFirstResultJson == nullptr) {
      vRes->data = "";  // simply send HTTP headers and status code
    } else if (pResType == cTaskResponseTypeAnthropic) {
      vRes->data = fFormatAnthropicSse(vFirstResultJson);
    } else if (pResType == cTaskResponseTypeOaiResp) {
      vRes->data = fFormatOaiRespSse(vFirstResultJson);
    } else {
      vRes->data = fFormatOaiSse(vFirstResultJson);
    }
    vRes->vStatus      = 200;
    vRes->vContentType = "text/event-stream";
    vRes->mSetNext([vResThis = vRes.get(), pResType, vSsePingInterval](std::string & pOutput) -> bool {
      static auto vFormatError = [](task_response_type pResType, const json & pResJson) {
        if (pResType == cTaskResponseTypeAnthropic) {
          return fFormatAnthropicSse({
            { "event", "error"  },
            { "data",  pResJson },
          });
        } else {
          return fFormatOaiSse(json{
            { "error", pResJson }
          });
        }
      };

      auto vEffectiveShouldStop = [&vResThis]() {
        return vResThis->mShouldStop();
      };

      try {
        if (vEffectiveShouldStop()) {
          SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
          return false;  // should_stop condition met
        }

        if (!vResThis->data.empty()) {
          // flush the first chunk
          pOutput = std::move(vResThis->data);
          vResThis->data.clear();
          return true;
        }

        server_response_reader & vRd = vResThis->vRd;

        // check if there is more data
        if (!vRd.mHasNext()) {
          switch (pResType) {
            case cTaskResponseTypeNone:
            case cTaskResponseTypeOaiResp:
            case cTaskResponseTypeAnthropic:
              pOutput = "";
              break;

            default:
              pOutput = "data: [DONE]\n\n";
              break;
          }
          SRV_DBG("%s", "all results received, terminating stream\n");
          return false;  // no more data, terminate
        }

        // receive subsequent results
        bool    vTimeout   = false;
        int64_t vStartTime = ggml_time_ms();
        auto    vResult    = vRd.mNext([&vTimeout, &vStartTime, vSsePingInterval, &vEffectiveShouldStop]() {
          if (vEffectiveShouldStop()) {
            return true;  // should_stop condition met
          } else if (vSsePingInterval > 0 && ggml_time_ms() - vStartTime > (int64_t) vSsePingInterval * 1000) {
            vTimeout = true;
            return true;  // timeout
          }
          return false;
        });

        if (vTimeout) {
          // some clients may time out (e.g. undici) will time out if no data is received for a while, so we need to send a ping to keep the connection alive
          SRV_DBG("%s", "sending SSE ping\n");
          pOutput = ":\n\n";
          return true;
        }

        if (vResult == nullptr) {
          SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
          GGML_ASSERT(vEffectiveShouldStop());
          return false;  // should_stop condition met
        }

        // send the results
        if (vResult->mIsError()) {
          json vResJson = vResult->to_json();
          pOutput       = vFormatError(pResType, vResJson);
          SRV_DBG("%s", "error received during streaming, terminating stream\n");
          return false;  // terminate on error
        } else {
          GGML_ASSERT(dynamic_cast<server_task_result_cmpl_partial *>(vResult.get()) != nullptr ||
                      dynamic_cast<server_task_result_cmpl_final *>(vResult.get()) != nullptr ||
                      dynamic_cast<server_task_result_queue *>(vResult.get()) != nullptr);
          json vResJson = vResult->to_json();
          if (pResType == cTaskResponseTypeAnthropic) {
            pOutput = fFormatAnthropicSse(vResJson);
          } else if (pResType == cTaskResponseTypeOaiResp) {
            pOutput = fFormatOaiRespSse(vResJson);
          } else {
            pOutput = fFormatOaiSse(vResJson);
          }
        }

        // has next data, continue
        return true;

      } catch (const std::exception & vE) {
        json vErrorJson = fFormatErrorResponse(vE.what(), cErrorTypeServer);
        pOutput         = vFormatError(pResType, vErrorJson);

        // terminate on exception
        return false;
      }
    });
  }

  return vRes;
}

std::unique_ptr<server_res_generator> server_routes::mCreateResponse(bool pBypassSleep) {
  return std::make_unique<server_res_generator>(vQueueTasks, vQueueResults, vParams.vSleepIdleSeconds, pBypassSleep);
}

server_routes::server_routes(const common_params & pParams, server_context & pCtxServer) :
  vParams(pParams),
  vCtxServer(*pCtxServer.impl),
  vQueueTasks(pCtxServer.impl->vQueueTasks),
  vQueueResults(pCtxServer.impl->vQueueResults) {
  mInitRoutes();

  // note: this must be registered before load_model()
  //       so that on sleep phase, the callback is called before ctx is destroyed
  vQueueTasks.mOnSleepingState([this](bool pIsSleeping) { mUpdateCachedResponses(pIsSleeping); });
}

static json fGetResModelInfo(const server_context_meta & pMeta) {
  // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

  return {
    { "id", pMeta.vModelName },
    { "aliases", pMeta.sModelAliases },
    { "tags", pMeta.sModelTags },
    { "object", "model" },
    { "architecture", fServerModelArchitectureJson(pMeta.vHasInpImage, pMeta.vHasInpAudio, pMeta.vHasInpVideo,
     pMeta.lModelOutputModalities) },
    { "created", std::time(0) },
    { "owned_by", "dgscpp" },
    { "meta",
     {
        { "vocab_type", pMeta.vModelVocabType },
        { "n_vocab", pMeta.vModelVocabNTokens },
        { "n_ctx", pMeta.vSlotNCtx },
        { "n_ctx_train", pMeta.vModelNCtxTrain },
        { "n_embd", pMeta.vModelNEmbdInp },
        { "n_params", pMeta.vModelNParams },
        { "size", pMeta.vModelSize },
        { "ftype", pMeta.vModelFtype },
      } },
  };
}

static json fGetResModels(const server_context_meta & pMeta) {
  // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

  return json{
    { "models", json::mArray({ { { "name", pMeta.vModelName },
                                 { "model", pMeta.vModelName },
                                 { "modified_at", "" },
                                 { "size", "" },
                                 { "digest", "" },  // dummy value, dgs.cpp does not support managing model file's hash
                                 { "type", "model" },
                                 { "description", "" },
                                 { "tags", json::mArray({ "" }) },
                                 { "capabilities", pMeta.vHasMtmd ? json::mArray({ "completion", "multimodal" }) :
                                                                    json::mArray({ "completion" }) },
                                 { "parameters", "" },
                                 { "details",
                                   { { "parent_model", "" },
                                     { "format", "gguf" },
                                     { "family", "" },
                                     { "families", json::mArray({ "" }) },
                                     { "parameter_size", "" },
                                     { "quantization_level", "" } } } } }) },
    { "object", "list"                                                                    },
    { "data",   json::mArray({
                fGetResModelInfo(pMeta),
              })                                                             }
  };
}

static json fGetResProps(const server_context_meta & pMeta, const common_params & pParams, bool pIsSleeping) {
  // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

  task_params vTparams;
  vTparams.vSampling                      = pParams.vSampling;
  json vDefaultGenerationSettingsForProps = json{
    { "params", vTparams.to_json(true) },
    { "n_ctx",  pMeta.vSlotNCtx        },
  };

  std::string vTmplDefault = fCommonChatTemplatesSource(pMeta.vChatParams.vTmpls.get(), "");
  std::string vTmplTools   = fCommonChatTemplatesSource(pMeta.vChatParams.vTmpls.get(), "tool_use");

  json vProps = {
    { "default_generation_settings", vDefaultGenerationSettingsForProps },
    { "total_slots",                 pParams.vNParallel                 },
    { "model_alias",                 pMeta.vModelName                   },
    { "model_ftype",                 pMeta.vModelFtype                  },
    { "model_path",                  pMeta.vModelPath                   },
    { "modalities",
     json{
        { "vision", pMeta.vHasInpImage },
        { "video", pMeta.vHasInpVideo },
        { "audio", pMeta.vHasInpAudio },
      }                                                                 },
    { "media_marker",                fGetMediaMarker()                  },
    { "endpoint_slots",              pParams.vEndpointSlots             },
    { "endpoint_props",              pParams.vEndpointProps             },
    { "endpoint_metrics",            pParams.vEndpointMetrics           },
    { "ui",                          pParams.vUi                        },
    { "ui_settings",                 pMeta.vJsonUiSettings              },
    { "chat_template",               vTmplDefault                       },
    { "chat_template_caps",          pMeta.dChatTemplateCaps            },
    { "bos_token",                   pMeta.vBosTokenStr                 },
    { "eos_token",                   pMeta.vEosTokenStr                 },
    { "build_info",                  pMeta.vBuildInfo                   },
    { "is_sleeping",                 pIsSleeping                        },
    { "shell_mcp_server",            pParams.vShellMcpServer            },
    { "cors_proxy_enabled",          pParams.vUiMcpProxy                },
  };
  if (pParams.vUseJinja) {
    if (!vTmplTools.empty()) {
      vProps["chat_template_tool_use"] = vTmplTools;
    }
  }

  return vProps;
}

json server_routes::mGetModelInfo() const {
  return fGetResModelInfo(*vMeta);
}

void server_routes::mInitRoutes() {
  // IMPORTANT: all lambda functions must start with create_response()
  // this is to ensure that the server_res_generator can handle sleeping case correctly

  this->vGetHealth = [this](const server_http_req &) {
    // error and loading states are handled by middleware
    auto vRes = mCreateResponse(true);

    // this endpoint can be accessed during sleeping
    // the next LOC is to avoid someone accidentally use ctx_server
    bool vCtxServer;  // do NOT delete this line
    GGML_UNUSED(vCtxServer);

    vRes->mOk({
      { "status", "ok" }
    });
    return vRes;
  };

  this->vGetMonitor = [this](const server_http_req &) {
    auto vResult  = std::make_unique<server_http_res>();
    vResult->data = fSafeJsonToStr(fMonitorReport({ vCtxServer.mGetMonitorSnapshot() }));
    return vResult;
  };
  this->vGetMonitorMemory = [this](const server_http_req &) {
    auto vResult  = std::make_unique<server_http_res>();
    vResult->data = fSafeJsonToStr(json{
      { "models", json::mArray({ vCtxServer.mGetMonitorSnapshot() }) }
    });
    return vResult;
  };
  this->vGetMonitorUsage = [](const server_http_req &) {
    auto vResult  = std::make_unique<server_http_res>();
    vResult->data = fSafeJsonToStr(fMonitorUsageJson());
    return vResult;
  };

  this->vGetMetrics = [this](const server_http_req & pReq) {
    auto vRes = mCreateResponse(true);
    if (!vParams.vEndpointMetrics) {
      vRes->mError(fFormatErrorResponse("This server does not support metrics endpoint. Start it with `--metrics`",
                                        cErrorTypeNotSupported));
      return vRes;
    }

    // render response using cached_metrics
    auto vUseCachedMetrics = [&]() {
      std::unique_lock<std::mutex> vLock(vMutexCache);
      vRes->dHeaders["Process-Start-Time-Unix"] = std::to_string(vCachedMetrics.vTStart);
      server_task_result_metrics vTmp;
      vTmp.vMetrics      = vCachedMetrics;
      vRes->vContentType = "text/plain; version=0.0.4";
      vRes->vStatus      = 200;
      vRes->data         = vTmp.mToMetrics();
      // the gauges are averaged over the window between two scrapes
      vCachedMetrics.mResetBucket();
      vShouldResetBuckets = true;
    };

    if (vQueueTasks.mIsSleeping()) {
      vUseCachedMetrics();

    } else {
      // request slots data using task queue
      {
        server_task vTask(cServerTaskTypeMetrics);
        vTask.vId                 = vRes->vRd.mGetNewId();
        // the gauges are averaged over the window between two scrapes
        vTask.vMetricsResetBucket = true;
        vRes->vRd.mPostTask(std::move(vTask), true);  // high-priority task
      }

      // a task posted right before sleeping is never processed, do not wait for it
      auto vResult = vRes->vRd.mNext([&] { return pReq.vShouldStop() || vQueueTasks.mIsSleeping(); });
      if (!vResult) {
        if (!pReq.vShouldStop()) {
          vUseCachedMetrics();
        }
        return vRes;
      }

      if (vResult->mIsError()) {
        vRes->mError(vResult->to_json());
        return vRes;
      }

      auto vResTask = dynamic_cast<server_task_result_metrics *>(vResult.get());
      GGML_ASSERT(vResTask != nullptr);

      vRes->dHeaders["Process-Start-Time-Unix"] = std::to_string(vResTask->vMetrics.vTStart);
      vRes->vContentType                        = "text/plain; version=0.0.4";
      vRes->vStatus                             = 200;
      vRes->data                                = vResTask->mToMetrics();
    }

    return vRes;
  };

  this->vGetSlots = [this](const server_http_req & pReq) {
    auto vRes = mCreateResponse();
    if (!vParams.vEndpointSlots) {
      vRes->mError(fFormatErrorResponse("This server does not support slots endpoint. Start it with `--slots`",
                                        cErrorTypeNotSupported));
      return vRes;
    }

    // request slots data using task queue
    {
      server_task vTask(cServerTaskTypeSlotGet);
      vTask.vId = vRes->vRd.mGetNewId();
      vRes->vRd.mPostTask(std::move(vTask), true);  // high-priority task
    }

    // get the result
    auto vResult = vRes->vRd.mNext(pReq.vShouldStop);
    if (!vResult) {
      // connection was closed
      GGML_ASSERT(pReq.vShouldStop());
      return vRes;
    }

    if (vResult->mIsError()) {
      vRes->mError(vResult->to_json());
      return vRes;
    }

    auto * vResTask = dynamic_cast<server_task_result_slots *>(vResult.get());
    GGML_ASSERT(vResTask != nullptr);

    // optionally return "fail_on_no_slot" error
    if (!pReq.mGetParam("fail_on_no_slot").empty()) {
      if (vResTask->vNIdleSlots == 0) {
        vRes->mError(fFormatErrorResponse("no slot available", cErrorTypeUnavailable));
        return vRes;
      }
    }

    vRes->mOk(vResTask->to_json());
    return vRes;
  };

  this->vPostSlots = [this](const server_http_req & pReq) {
    auto vRes = mCreateResponse();
    if (vParams.vSlotSavePath.empty()) {
      vRes->mError(fFormatErrorResponse("This server does not support slots action. Start it with `--slot-save-path`",
                                        cErrorTypeNotSupported));
      return vRes;
    }

    std::string vIdSlotStr = pReq.mGetParam("id_slot");

    int vIdSlot;
    try {
      vIdSlot = std::stoi(vIdSlotStr);
    } catch (const std::exception &) {
      vRes->mError(fFormatErrorResponse("Invalid slot ID", cErrorTypeInvalidRequest));
      return vRes;
    }

    std::string vAction = pReq.mGetParam("action");

    if (vAction == "save") {
      return mHandleSlotsSave(pReq, vIdSlot);
    }
    if (vAction == "restore") {
      return mHandleSlotsRestore(pReq, vIdSlot);
    }
    if (vAction == "erase") {
      return mHandleSlotsErase(pReq, vIdSlot);
    }

    vRes->mError(fFormatErrorResponse("Invalid action", cErrorTypeInvalidRequest));
    return vRes;
  };

  this->vGetProps = [this](const server_http_req &) {
    auto vRes = mCreateResponse(true);
    // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
    if (vQueueTasks.mIsSleeping()) {
      std::unique_lock<std::mutex> vLock(vMutexCache);
      vRes->mOk(vCachedProps);
    } else {
      vRes->mOk(fGetResProps(*vMeta, vParams, false));
    }
    return vRes;
  };

  this->vPostProps = [this](const server_http_req &) {
    auto vRes = mCreateResponse();
    if (!vParams.vEndpointProps) {
      vRes->mError(fFormatErrorResponse(
        "This server does not support changing global properties. Start it with `--props`", cErrorTypeNotSupported));
      return vRes;
    }
    // update any props here

    vRes->mOk({
      { "success", true }
    });
    return vRes;
  };

  this->vPostInfill = [this](const server_http_req & pReq) {
    auto        vRes = mCreateResponse();
    // check model compatibility
    std::string vErr;
    if (fDgsVocabFimPre(vCtxServer.vVocab) == DGS_TOKEN_NULL) {
      vErr += "prefix token is missing. ";
    }
    if (fDgsVocabFimSuf(vCtxServer.vVocab) == DGS_TOKEN_NULL) {
      vErr += "suffix token is missing. ";
    }
    if (fDgsVocabFimMid(vCtxServer.vVocab) == DGS_TOKEN_NULL) {
      vErr += "middle token is missing. ";
    }
    if (!vErr.empty()) {
      vRes->mError(fFormatErrorResponse(fStringFormat("Infill is not supported by this model: %s", vErr.c_str()),
                                        cErrorTypeNotSupported));
      return vRes;
    }

    // validate input
    json data = json::mParse(pReq.vBody);
    if (data.contains("prompt") && !data.at("prompt").is_string()) {
      // prompt is optional
      vRes->mError(fFormatErrorResponse("\"prompt\" must be a string", cErrorTypeInvalidRequest));
    }

    if (!data.contains("input_prefix")) {
      vRes->mError(fFormatErrorResponse("\"input_prefix\" is required", cErrorTypeInvalidRequest));
    }

    if (!data.contains("input_suffix")) {
      vRes->mError(fFormatErrorResponse("\"input_suffix\" is required", cErrorTypeInvalidRequest));
    }

    if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
      // input_extra is optional
      vRes->mError(fFormatErrorResponse("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}",
                                        cErrorTypeInvalidRequest));
      return vRes;
    }

    json vInputExtra = fJsonValue(data, "input_extra", json::mArray());
    for (const auto & vChunk : vInputExtra) {
      // { "text": string, "filename": string }
      if (!vChunk.contains("text") || !vChunk.at("text").is_string()) {
        vRes->mError(fFormatErrorResponse("extra_context chunk must contain a \"text\" field with a string value",
                                          cErrorTypeInvalidRequest));
        return vRes;
      }
      // filename is optional
      if (vChunk.contains("filename") && !vChunk.at("filename").is_string()) {
        vRes->mError(
          fFormatErrorResponse("extra_context chunk's \"filename\" field must be a string", cErrorTypeInvalidRequest));
        return vRes;
      }
    }
    data["input_extra"] = vInputExtra;  // default to empty array if it's not exist

    std::string                vPrompt = fJsonValue(data, "prompt", std::string());
    std::vector<server_tokens> lTokenizedPrompts =
      fTokenizeInputPrompts(vCtxServer.vVocab, vCtxServer.vMctx, vPrompt, false, true, vCtxServer.vInitOpt);
    SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) lTokenizedPrompts.size());
    data["prompt"] =
      fFormatPromptInfill(vCtxServer.vVocab, data.at("input_prefix"), data.at("input_suffix"), data.at("input_extra"),
                          vParams.vNBatch, vParams.vNPredict, vMeta->vSlotNCtx, vParams.vSpmInfill,
                          lTokenizedPrompts[0].mGetTokens()  // TODO: this could maybe be multimodal.
      );

    std::vector<raw_buffer> lFiles;                        // dummy
    return mHandleCompletionsImpl(pReq, cServerTaskTypeInfill, data, lFiles,
                                  cTaskResponseTypeNone);  // infill is not OAI compatible
  };

  this->vPostCompletions = [this](const server_http_req & pReq) {
    auto                    vRes = mCreateResponse();
    std::vector<raw_buffer> lFiles;  // dummy
    const json              cBody = json::mParse(pReq.vBody);
    return mHandleCompletionsImpl(pReq, cServerTaskTypeCompletion, cBody, lFiles, cTaskResponseTypeNone);
  };

  this->vPostCompletionsOai = [this](const server_http_req & pReq) {
    auto                    vRes = mCreateResponse();
    std::vector<raw_buffer> lFiles;  // dummy
    const json              cBody = json::mParse(pReq.vBody);
    return mHandleCompletionsImpl(pReq, cServerTaskTypeCompletion, cBody, lFiles, cTaskResponseTypeOaiCmpl);
  };

  this->vPostChatCompletions = [this](const server_http_req & pReq) {
    auto                    vRes = mCreateResponse();
    std::vector<raw_buffer> lFiles;
    json                    vBody = json::mParse(pReq.vBody);
    common_chat_session     vSession;
    json vBodyParsed = fOaicompatChatParamsParse(vCtxServer.vVocab, vBody, vMeta->vChatParams, lFiles, vSession);
    return mHandleCompletionsImpl(pReq, cServerTaskTypeCompletion, vBodyParsed, lFiles, cTaskResponseTypeOaiChat,
                                  vSession);
  };

  this->vPostChatCompletionsTok = [this](const server_http_req & pReq) {
    return mHandleCountTokens(pReq, cTaskResponseTypeOaiChat);
  };

  this->vPostControl = [this](const server_http_req & pReq) {
    auto       vRes  = mCreateResponse();
    const json cBody = json::mParse(pReq.vBody);

    const std::string cCmplId = fJsonValue(cBody, "id", std::string());
    const std::string cAction = fJsonValue(cBody, "action", std::string());
    if (cCmplId.empty()) {
      vRes->mError(fFormatErrorResponse("missing completion id", cErrorTypeInvalidRequest));
      return vRes;
    }
    if (cAction != "reasoning_end") {
      vRes->mError(fFormatErrorResponse("unknown control action", cErrorTypeInvalidRequest));
      return vRes;
    }

    auto & vRd = vRes->vRd;
    {
      server_task vTask(cServerTaskTypeControl);
      vTask.vId                    = vRd.mGetNewId();
      vTask.vParams.vControlCmplId = cCmplId;
      vTask.vParams.vControlAction = cAction;
      vRd.mPostTask(std::move(vTask));
    }

    auto vResult = vRd.mNext(pReq.vShouldStop);
    if (!vResult) {
      GGML_ASSERT(pReq.vShouldStop());
      return vRes;
    }
    if (vResult->mIsError()) {
      vRes->mError(vResult->to_json());
      return vRes;
    }
    vRes->mOk(vResult->to_json());
    return vRes;
  };

  this->vPostResponsesOai = [this](const server_http_req & pReq) {
    auto                    vRes = mCreateResponse();
    std::vector<raw_buffer> lFiles;
    json                    vBody = fServerChatConvertResponsesToChatcmpl(json::mParse(pReq.vBody));
    SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
    SRV_DBG("converted request: %s\n", vBody.mDump().c_str());
    common_chat_session vSession;
    json vBodyParsed = fOaicompatChatParamsParse(vCtxServer.vVocab, vBody, vMeta->vChatParams, lFiles, vSession);
    return mHandleCompletionsImpl(pReq, cServerTaskTypeCompletion, vBodyParsed, lFiles, cTaskResponseTypeOaiResp,
                                  vSession);
  };

  this->vPostResponsesTokOai = [this](const server_http_req & pReq) {
    return mHandleCountTokens(pReq, cTaskResponseTypeOaiResp);
  };

  this->vPostTranscriptionsOai = [this](const server_http_req & pReq) {
    auto vRes = mCreateResponse();

    if (!vMeta->vHasMtmd || !vMeta->vChatParams.vAllowAudio) {
      vRes->mError(fFormatErrorResponse("The current model does not support audio input.", cErrorTypeNotSupported));
      return vRes;
    }

    std::vector<raw_buffer> lFiles;
    json                    vBody =
      fConvertTranscriptionsToChatcmpl(json::mParse(pReq.vBody), vMeta->vChatParams.vTmpls.get(), pReq.dFiles, lFiles);
    SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
    SRV_DBG("converted request: %s\n", vBody.mDump().c_str());
    common_chat_session vSession;
    json vBodyParsed = fOaicompatChatParamsParse(vCtxServer.vVocab, vBody, vMeta->vChatParams, lFiles, vSession);
    return mHandleCompletionsImpl(pReq, cServerTaskTypeCompletion, vBodyParsed, lFiles, cTaskResponseTypeOaiAsr,
                                  vSession);
  };

  this->vPostAnthropicMessages = [this](const server_http_req & pReq) {
    auto                    vRes = mCreateResponse();
    std::vector<raw_buffer> lFiles;
    json                    vBody = fServerChatConvertAnthropicToOai(json::mParse(pReq.vBody));
    SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
    SRV_DBG("converted request: %s\n", vBody.mDump().c_str());
    common_chat_session vSession;
    json vBodyParsed = fOaicompatChatParamsParse(vCtxServer.vVocab, vBody, vMeta->vChatParams, lFiles, vSession);
    return mHandleCompletionsImpl(pReq, cServerTaskTypeCompletion, vBodyParsed, lFiles, cTaskResponseTypeAnthropic,
                                  vSession);
  };

  this->vPostAnthropicCountTokens = [this](const server_http_req & pReq) {
    return mHandleCountTokens(pReq, cTaskResponseTypeAnthropic);
  };

  // same with handle_chat_completions, but without inference part
  this->vPostApplyTemplate = [this](const server_http_req & pReq) {
    auto                    vRes = mCreateResponse();
    std::vector<raw_buffer> lFiles;    // dummy, unused
    common_chat_session     vSession;  // dummy, unused
    json                    vBody = json::mParse(pReq.vBody);
    json data = fOaicompatChatParamsParse(vCtxServer.vVocab, vBody, vMeta->vChatParams, lFiles, vSession);
    vRes->mOk({
      { "prompt", std::move(data.at("prompt")) }
    });
    return vRes;
  };

  this->vGetModels = [this](const server_http_req &) {
    auto vRes = mCreateResponse(true);
    // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
    if (vQueueTasks.mIsSleeping()) {
      std::unique_lock<std::mutex> vLock(vMutexCache);
      vRes->mOk(vCachedModels);
    } else {
      vRes->mOk(fGetResModels(*vMeta));
    }
    return vRes;
  };

  this->vPostTokenize = [this](const server_http_req & pReq) {
    auto       vRes            = mCreateResponse();
    const json cBody           = json::mParse(pReq.vBody);
    json       vTokensResponse = json::mArray();
    if (cBody.count("content") != 0) {
      const bool cAddSpecial   = fJsonValue(cBody, "add_special", false);
      const bool cParseSpecial = fJsonValue(cBody, "parse_special", true);
      const bool cWithPieces   = fJsonValue(cBody, "with_pieces", false);

      dgs_tokens lTokens = fTokenizeMixed(vCtxServer.vVocab, cBody.at("content"), cAddSpecial, cParseSpecial);

      if (cWithPieces) {
        for (const auto & token : lTokens) {
          std::string vPiece = fCommonTokenToPiece(vCtxServer.vVocab, token);
          json        vPieceJson;

          // Check if the piece is valid UTF-8
          if (fIsValidUtf8(vPiece)) {
            vPieceJson = vPiece;
          } else {
            // If not valid UTF-8, store as array of byte values
            vPieceJson = json::mArray();
            for (unsigned char vC : vPiece) {
              vPieceJson.push_back(static_cast<int>(vC));
            }
          }

          vTokensResponse.push_back({
            { "id",    token      },
            { "piece", vPieceJson }
          });
        }
      } else {
        vTokensResponse = lTokens;
      }
    }

    vRes->mOk(json{
      { "tokens", std::move(vTokensResponse) }
    });
    return vRes;
  };

  this->vPostDetokenize = [this](const server_http_req & pReq) {
    auto       vRes  = mCreateResponse();
    const json cBody = json::mParse(pReq.vBody);

    std::string vContent;
    if (cBody.count("tokens") != 0) {
      const dgs_tokens lTokens = cBody.at("tokens").get<dgs_tokens>();
      vContent                 = fTokensToStr(vCtxServer.vVocab, lTokens);
    }

    vRes->mOk(json{
      { "content", std::move(vContent) }
    });
    return vRes;
  };

  this->vPostEmbeddings = [this](const server_http_req & pReq) {
    return mHandleEmbeddingsImpl(pReq, cTaskResponseTypeNone);
  };

  this->vPostEmbeddingsOai = [this](const server_http_req & pReq) {
    return mHandleEmbeddingsImpl(pReq, cTaskResponseTypeOaiEmbd);
  };

  this->vPostRerank = [this](const server_http_req & pReq) {
    auto vRes = mCreateResponse();
    if (!vParams.vEmbedding || vParams.vPoolingType != cDgsPoolingTypeRank) {
      vRes->mError(fFormatErrorResponse("This server does not support reranking. Start it with `--reranking`",
                                        cErrorTypeNotSupported));
      return vRes;
    }

    const json cBody = json::mParse(pReq.vBody);

    // if true, use TEI API format, otherwise use Jina API format
    // Jina: https://jina.ai/reranker/
    // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
    bool vIsTeiFormat = cBody.contains("texts");

    json vQuery;
    if (cBody.count("query") == 1) {
      vQuery = cBody.at("query");
      if (!vQuery.is_string()) {
        vRes->mError(fFormatErrorResponse("\"query\" must be a string", cErrorTypeInvalidRequest));
        return vRes;
      }
    } else {
      vRes->mError(fFormatErrorResponse("\"query\" must be provided", cErrorTypeInvalidRequest));
      return vRes;
    }

    std::vector<std::string> lDocuments =
      fJsonValue(cBody, "documents", fJsonValue(cBody, "texts", std::vector<std::string>()));
    if (lDocuments.empty()) {
      vRes->mError(fFormatErrorResponse("\"documents\" must be a non-empty string array", cErrorTypeInvalidRequest));
      return vRes;
    }

    int vTopN = fJsonValue(cBody, "top_n", (int) lDocuments.size());

    // create and queue the task
    json   vResponses = json::mArray();
    auto & vRd        = vRes->vRd;
    {
      std::vector<server_task> lTasks;
      lTasks.reserve(lDocuments.size());
      for (size_t vI = 0; vI < lDocuments.size(); vI++) {
        auto        vTmp  = fFormatPromptRerank(vCtxServer.vModelTgt, vCtxServer.vVocab, vCtxServer.vMctx, vQuery,
                                                lDocuments[vI], vCtxServer.vInitOpt);
        server_task vTask = server_task(cServerTaskTypeRerank);
        vTask.vId         = vRd.mGetNewId();
        vTask.vTokens     = std::move(vTmp);
        lTasks.push_back(std::move(vTask));
      }
      vRd.mPostTasks(std::move(lTasks));
    }

    // wait for the results
    auto vAllResults = vRd.mWaitForAll(pReq.vShouldStop);

    // collect results
    if (vAllResults.vIsTerminated) {
      return vRes;  // connection is closed
    } else if (vAllResults.vError) {
      vRes->mError(vAllResults.vError->to_json());
      return vRes;
    } else {
      for (auto & vRes : vAllResults.lResults) {
        GGML_ASSERT(dynamic_cast<server_task_result_rerank *>(vRes.get()) != nullptr);
        vResponses.push_back(vRes->to_json());
      }
    }

    // write JSON response
    json vRoot = fFormatResponseRerank(cBody, vMeta->vModelName, vResponses, vIsTeiFormat, lDocuments, vTopN);

    vRes->mOk(vRoot);
    return vRes;
  };

  this->vPostSystemone = [this](const server_http_req & pReq) {
    auto         vRes     = mCreateResponse();
    const auto & decision = vCtxServer.decision;
    if (decision.type == cCommonDecisionTypeNone) {
      vRes->mError(fFormatErrorResponse("This model is not a decision model", cErrorTypeNotSupported));
      return vRes;
    }

    const json cBody      = json::mParse(pReq.vBody);
    const auto lQuestions = decision.mParseQuestions(cBody);

    std::vector<raw_buffer> lFiles;
    const json              cState = decision.mParseState(cBody, lFiles);
    if (!lFiles.empty() && (!decision.mCanUseImages() || !vMeta->vHasInpImage)) {
      vRes->mError(
        fFormatErrorResponse("This server does not support image input for decisions. For a model that supports it, "
                             "start it with `--mmproj`",
                             cErrorTypeNotSupported));
      return vRes;
    }

    // one task per variant of each question, or one task for all the questions
    auto & vRd = vRes->vRd;
    {
      std::vector<server_task> lTasks;
      if (decision.mIsJoint()) {
        server_task vTask = server_task(cServerTaskTypeDecision);
        vTask.vId         = vRd.mGetNewId();
        decision.mFillTaskJoint(cState, lQuestions, lFiles, vCtxServer.vMctx, vCtxServer.vInitOpt, vTask);
        lTasks.push_back(std::move(vTask));
      } else {
        for (const auto & vQuestion : lQuestions) {
          for (size_t vVariant = 0; vVariant < decision.mNVariants(vQuestion); vVariant++) {
            server_task vTask = server_task(cServerTaskTypeDecision);
            vTask.vId         = vRd.mGetNewId();
            decision.mFillTask(cState, lQuestions, vQuestion, vVariant, lFiles, vCtxServer.vMctx, vCtxServer.vInitOpt,
                               vTask);
            lTasks.push_back(std::move(vTask));
          }
        }
      }
      if (decision.mCanSharePrompt()) {
        lTasks = fServerDecisionGroupTasks(std::move(lTasks), vParams.vNParallel);
      }
      vRd.mPostTasks(std::move(lTasks));
    }

    auto vAllResults = vRd.mWaitForAll(pReq.vShouldStop);

    if (vAllResults.vIsTerminated) {
      return vRes;  // connection is closed
    } else if (vAllResults.vError) {
      vRes->mError(vAllResults.vError->to_json());
      return vRes;
    }

    json    vAnswers = json::mObject();
    int32_t vNTokens = 0;
    size_t  vIResult = 0;
    size_t  vIScore  = 0;
    for (const auto & vQuestion : lQuestions) {
      std::vector<std::vector<float>> lScores;
      if (decision.mIsJoint()) {
        // one result with the scores of all the questions, in order
        auto * vResult = dynamic_cast<server_task_result_decision *>(vAllResults.lResults[0].get());
        GGML_ASSERT(vResult != nullptr && vIScore + vQuestion.lOptions.size() <= vResult->lScores.size());
        lScores.emplace_back(vResult->lScores.begin() + vIScore,
                             vResult->lScores.begin() + vIScore + vQuestion.lOptions.size());
        vIScore += vQuestion.lOptions.size();
        vNTokens = vResult->vNTokens;
      }
      for (size_t vVariant = 0; !decision.mIsJoint() && vVariant < decision.mNVariants(vQuestion); vVariant++) {
        auto * vResult = dynamic_cast<server_task_result_decision *>(vAllResults.lResults[vIResult++].get());
        GGML_ASSERT(vResult != nullptr);
        lScores.push_back(vResult->lScores);
        vNTokens += vResult->vNTokens;
      }
      vAnswers[vQuestion.vId] = decision.mFormatAnswer(vQuestion, lScores, !lFiles.empty());
    }

    vRes->mOk(json{
      { "model",   vMeta->vModelName },
      { "answers", vAnswers          },
      { "usage",
       {
          { "input_tokens", vNTokens },
          { "output_tokens", 0 },
        }                            },
    });
    return vRes;
  };

  this->vGetLoraAdapters = [this](const server_http_req & pReq) {
    auto vRes = mCreateResponse();

    auto & vRd = vRes->vRd;
    {
      server_task vTask(cServerTaskTypeGetLora);
      vTask.vId = vRd.mGetNewId();
      vRd.mPostTask(std::move(vTask));
    }

    // get the result
    auto vResult = vRd.mNext(pReq.vShouldStop);
    if (!vResult) {
      // connection was closed
      GGML_ASSERT(pReq.vShouldStop());
      return vRes;
    }

    if (vResult->mIsError()) {
      vRes->mError(vResult->to_json());
      return vRes;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_get_lora *>(vResult.get()) != nullptr);
    vRes->mOk(vResult->to_json());
    return vRes;
  };

  this->vPostLoraAdapters = [this](const server_http_req & pReq) {
    auto       vRes  = mCreateResponse();
    const json cBody = json::mParse(pReq.vBody);
    if (!cBody.is_array()) {
      vRes->mError(fFormatErrorResponse("Request body must be an array", cErrorTypeInvalidRequest));
      return vRes;
    }

    auto & vRd = vRes->vRd;
    {
      server_task vTask(cServerTaskTypeSetLora);
      vTask.vId      = vRd.mGetNewId();
      vTask.dSetLora = fParseLoraRequest(cBody);
      vRd.mPostTask(std::move(vTask));
    }

    // get the result
    auto vResult = vRd.mNext(pReq.vShouldStop);
    if (!vResult) {
      // connection was closed
      GGML_ASSERT(pReq.vShouldStop());
      return vRes;
    }

    if (vResult->mIsError()) {
      vRes->mError(vResult->to_json());
      return vRes;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_apply_lora *>(vResult.get()) != nullptr);
    vRes->mOk(vResult->to_json());
    return vRes;
  };
}

std::unique_ptr<server_res_generator> server_routes::mHandleSlotsSave(const server_http_req & pReq, int pIdSlot) {
  auto        vRes         = mCreateResponse();
  const json  cRequestData = json::mParse(pReq.vBody);
  std::string vFilename    = cRequestData.at("filename");
  if (!fFsValidateFilename(vFilename)) {
    vRes->mError(fFormatErrorResponse("Invalid filename", cErrorTypeInvalidRequest));
    return vRes;
  }
  std::string vFilepath = vParams.vSlotSavePath + vFilename;

  auto & vRd = vRes->vRd;
  {
    server_task vTask(cServerTaskTypeSlotSave);
    vTask.vId                   = vRd.mGetNewId();
    vTask.vSlotAction.vIdSlot   = pIdSlot;
    vTask.vSlotAction.vFilename = vFilename;
    vTask.vSlotAction.vFilepath = vFilepath;
    vRd.mPostTask(std::move(vTask));
  }

  auto vResult = vRd.mNext(pReq.vShouldStop);
  if (!vResult) {
    // connection was closed
    GGML_ASSERT(pReq.vShouldStop());
    return vRes;
  }

  if (vResult->mIsError()) {
    vRes->mError(vResult->to_json());
    return vRes;
  }

  vRes->mOk(vResult->to_json());
  return vRes;
}

std::unique_ptr<server_res_generator> server_routes::mHandleSlotsRestore(const server_http_req & pReq, int pIdSlot) {
  auto        vRes         = mCreateResponse();
  const json  cRequestData = json::mParse(pReq.vBody);
  std::string vFilename    = cRequestData.at("filename");
  if (!fFsValidateFilename(vFilename)) {
    vRes->mError(fFormatErrorResponse("Invalid filename", cErrorTypeInvalidRequest));
    return vRes;
  }
  std::string vFilepath = vParams.vSlotSavePath + vFilename;

  auto & vRd = vRes->vRd;
  {
    server_task vTask(cServerTaskTypeSlotRestore);
    vTask.vId                   = vRd.mGetNewId();
    vTask.vSlotAction.vIdSlot   = pIdSlot;
    vTask.vSlotAction.vFilename = vFilename;
    vTask.vSlotAction.vFilepath = vFilepath;
    vRd.mPostTask(std::move(vTask));
  }

  auto vResult = vRd.mNext(pReq.vShouldStop);
  if (!vResult) {
    // connection was closed
    GGML_ASSERT(pReq.vShouldStop());
    return vRes;
  }

  if (vResult->mIsError()) {
    vRes->mError(vResult->to_json());
    return vRes;
  }

  GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load *>(vResult.get()) != nullptr);
  vRes->mOk(vResult->to_json());
  return vRes;
}

std::unique_ptr<server_res_generator> server_routes::mHandleSlotsErase(const server_http_req & pReq, int pIdSlot) {
  auto   vRes = mCreateResponse();
  auto & vRd  = vRes->vRd;
  {
    server_task vTask(cServerTaskTypeSlotErase);
    vTask.vId                 = vRd.mGetNewId();
    vTask.vSlotAction.vIdSlot = pIdSlot;
    vRd.mPostTask(std::move(vTask));
  }

  auto vResult = vRd.mNext(pReq.vShouldStop);
  if (!vResult) {
    // connection was closed
    GGML_ASSERT(pReq.vShouldStop());
    return vRes;
  }

  if (vResult->mIsError()) {
    vRes->mError(vResult->to_json());
    return vRes;
  }

  GGML_ASSERT(dynamic_cast<server_task_result_slot_erase *>(vResult.get()) != nullptr);
  vRes->mOk(vResult->to_json());
  return vRes;
}

std::unique_ptr<server_res_generator> server_routes::mHandleEmbeddingsImpl(const server_http_req & pReq,
                                                                           task_response_type      pResType) {
  auto vRes = mCreateResponse();
  if (!vParams.vEmbedding) {
    vRes->mError(fFormatErrorResponse("This server does not support embeddings. Start it with `--embeddings`",
                                      cErrorTypeNotSupported));
    return vRes;
  }

  if (pResType != cTaskResponseTypeNone && vMeta->vPoolingType == cDgsPoolingTypeNone) {
    vRes->mError(fFormatErrorResponse("Pooling type 'none' is not OAI compatible. Please use a different pooling type",
                                      cErrorTypeInvalidRequest));
    return vRes;
  }

  const json cBody = json::mParse(pReq.vBody);

  // for the shape of input/content, see tokenize_input_prompts()
  json vPrompt;
  if (cBody.count("input") != 0) {
    vPrompt = cBody.at("input");
  } else if (cBody.contains("content")) {
    pResType = cTaskResponseTypeNone;  // "content" field is not OAI compatible
    vPrompt  = cBody.at("content");
  } else {
    vRes->mError(fFormatErrorResponse("\"input\" or \"content\" must be provided", cErrorTypeInvalidRequest));
    return vRes;
  }

  bool vUseBase64 = false;
  if (cBody.count("encoding_format") != 0) {
    const std::string & format = cBody.at("encoding_format");
    if (format == "base64") {
      vUseBase64 = true;
    } else if (format != "float") {
      vRes->mError(fFormatErrorResponse("The format to return the embeddings in. Can be either float or base64",
                                        cErrorTypeInvalidRequest));
      return vRes;
    }
  }

  // same shapes as tokenize_input_prompts(), plus OAI content: { "content": [ { "type": "text"|"image_url"|"input_audio"|"input_video", ... } ] }
  auto vTokenizeEntry = [&](const json & pP) {
    if (pP.is_object() && pP.contains("content")) {
      return fTokenizeOaiContentArray(vCtxServer.vVocab, vCtxServer.vMctx, vMeta->vChatParams, pP.at("content"), true,
                                      true, vCtxServer.vInitOpt);
    }
    return fTokenizeInputSubprompt(vCtxServer.vVocab, vCtxServer.vMctx, pP, true, true, vCtxServer.vInitOpt);
  };

  std::vector<server_tokens> lTokenizedPrompts;
  if (vPrompt.is_array() && !fJsonIsArrayAndContainsNumbers(vPrompt)) {
    for (const auto & vP : vPrompt) {
      lTokenizedPrompts.push_back(vTokenizeEntry(vP));
    }
  } else {
    lTokenizedPrompts.push_back(vTokenizeEntry(vPrompt));
  }
  if (lTokenizedPrompts.empty()) {
    vRes->mError(fFormatErrorResponse("\"input\" must not be empty", cErrorTypeInvalidRequest));
    return vRes;
  }

  for (const auto & vTokens : lTokenizedPrompts) {
    // this check is necessary for models that do not add BOS token to the input
    if (vTokens.empty()) {
      vRes->mError(fFormatErrorResponse("Input content cannot be empty", cErrorTypeInvalidRequest));
      return vRes;
    }
  }

  int vEmbdNormalize = vParams.vEmbdNormalize;
  if (cBody.count("embd_normalize") != 0) {
    vEmbdNormalize = cBody.at("embd_normalize").get<int>();
    if (vMeta->vPoolingType == cDgsPoolingTypeNone) {
      SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", vMeta->vPoolingType);
    }
  }

  // create and queue the task
  json   vResponses = json::mArray();
  auto & vRd        = vRes->vRd;
  {
    std::vector<server_task> lTasks;
    for (size_t vI = 0; vI < lTokenizedPrompts.size(); vI++) {
      server_task vTask = server_task(cServerTaskTypeEmbedding);

      vTask.vId     = vRd.mGetNewId();
      vTask.vTokens = std::move(lTokenizedPrompts[vI]);

      // OAI-compat
      vTask.vParams.vResType       = pResType;
      vTask.vParams.vEmbdNormalize = vEmbdNormalize;

      lTasks.push_back(std::move(vTask));
    }
    vRd.mPostTasks(std::move(lTasks));
  }

  // wait for the results
  auto vAllResults = vRd.mWaitForAll(pReq.vShouldStop);

  // collect results
  if (vAllResults.vIsTerminated) {
    return vRes;  // connection is closed
  } else if (vAllResults.vError) {
    vRes->mError(vAllResults.vError->to_json());
    return vRes;
  } else {
    for (auto & vRes : vAllResults.lResults) {
      GGML_ASSERT(dynamic_cast<server_task_result_embd *>(vRes.get()) != nullptr);
      vResponses.push_back(vRes->to_json());
    }
  }

  // write JSON response
  json vRoot = pResType == cTaskResponseTypeOaiEmbd ?
                 fFormatEmbeddingsResponseOaicompat(cBody, vMeta->vModelName, vResponses, vUseBase64) :
                 json(vResponses);
  vRes->mOk(vRoot);
  return vRes;
}

std::unique_ptr<server_res_generator> server_routes::mHandleCountTokens(const server_http_req & pReq,
                                                                        task_response_type      pResType) {
  auto                    vRes = mCreateResponse();
  std::vector<raw_buffer> lFiles;
  json                    vBody  = json::mParse(pReq.vBody);
  bool                    vIsOai = false;

  switch (pResType) {
    case cTaskResponseTypeOaiChat:
      {
        vIsOai = true;
      }
      break;
    case cTaskResponseTypeOaiResp:
      {
        vIsOai = true;
        vBody  = fServerChatConvertResponsesToChatcmpl(vBody);
      }
      break;
    case cTaskResponseTypeAnthropic:
      {
        vBody = fServerChatConvertAnthropicToOai(vBody);
      }
      break;
    default:
      vRes->mError(fFormatErrorResponse("invalid res_type", cErrorTypeInvalidRequest));
      return vRes;
  }

  common_chat_session vSession;  // dummy, unused
  json vBodyParsed = fOaicompatChatParamsParse(vCtxServer.vVocab, vBody, vMeta->vChatParams, lFiles, vSession);
  json vPrompt     = vBodyParsed.at("prompt");
  // SRV_DBG("prompt = %s\n", prompt.dump().c_str());

  // TODO @ngxson : refactor this code block, move this to server-common and reuse it in other places
  size_t vNTokens;
  if (vCtxServer.vMctx != nullptr) {
    if (!vPrompt.is_string()) {
      throw std::runtime_error("for mtmd, input prompt must be a string.");
    }
    vNTokens =
      fProcessMtmdPrompt(vCtxServer.vMctx, vPrompt.get<std::string>(), lFiles, vCtxServer.vInitOpt, true).size();
  } else {
    vNTokens = fTokenizeMixed(vCtxServer.vVocab, vPrompt, true, true).size();
  }

  json vResponse = {
    { "input_tokens", static_cast<int64_t>(vNTokens) }
  };
  if (vIsOai) {
    vResponse["object"] = "response.input_tokens";
  }
  vRes->mOk(vResponse);
  return vRes;
}

void server_routes::mUpdateCachedResponses(bool pIsSleeping) {
  // caller is task_queue, so ctx_server can be accessed without holding locks
  std::unique_lock<std::mutex> vLock(vMutexCache);

  if (pIsSleeping) {
    vCachedModels  = fGetResModels(*vMeta);
    vCachedProps   = fGetResProps(*vMeta, vParams, true);
    vCachedMetrics = vCtxServer.mGetMetrics();

    vShouldResetBuckets = false;

    SRV_DBG("%s\n", "cached responses updated");

  } else if (vShouldResetBuckets) {
    // a scrape during sleep already reported these buckets
    vCtxServer.mResetMetricsBucket();

    vShouldResetBuckets = false;
  }
}
