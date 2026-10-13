#include "dgs-memory-hybrid.h"

#include "dgs-context.h"
#include "dgs-impl.h"
#include "dgs-model.h"

//
// dgs_memory_hybrid
//

dgs_memory_hybrid::dgs_memory_hybrid(const dgs_model &       pModel,
                                     /* attn */
                                     ggml_type               pTypeK,
                                     ggml_type               pTypeV,
                                     bool                    pVTrans,
                                     uint32_t                pKvSize,
                                     uint32_t                pNPad,
                                     uint32_t                pNSwa,
                                     dgs_swa_type            pSwaType,
                                     /* recurrent */
                                     ggml_type               pTypeR,
                                     ggml_type               pTypeS,
                                     uint32_t                pRsSize,
                                     /* common */
                                     uint32_t                pNSeqMax,
                                     uint32_t                pNRsSeq,
                                     bool                    pOffload,
                                     bool                    pUnified,
                                     /* layer filters */
                                     const layer_filter_cb & pFilterAttn,
                                     const layer_filter_cb & pFilterRecr) :
  vHparams(pModel.hparams),
  cMemAttn(new dgs_kv_cache(pModel,
                            pModel.hparams,
                            pTypeK,
                            pTypeV,
                            pVTrans,
                            pOffload,
                            pUnified,
                            pKvSize,
                            pNSeqMax,
                            pNPad,
                            pNSwa,
                            pSwaType,
                            nullptr,
                            pFilterAttn == nullptr ?
                              [&](int32_t pIl) {
                                return !vHparams.mIsRecr(pIl);
                              } :
                              pFilterAttn,
                            nullptr,
                            nullptr)),
  cMemRecr(new dgs_memory_recurrent(pModel,
                                    pTypeR,
                                    pTypeS,
                                    pOffload,
                                    pRsSize,
                                    pNSeqMax,
                                    pNRsSeq,
                                    pFilterRecr == nullptr ?
                                      [&](int32_t pIl) {
                                        return vHparams.mIsRecr(pIl);
                                      } :
                                      pFilterRecr)) {}

dgs_memory_context_ptr dgs_memory_hybrid::mInitBatch(dgs_batch_allocr & pBalloc, uint32_t pNUbatch, bool pEmbdAll) {
  do {
    pBalloc.mSplitReset();

    // follow the recurrent pattern for creating the ubatch splits
    std::vector<dgs_ubatch> lUbatches;

    while (true) {
      dgs_ubatch vUbatch;

      if (pEmbdAll) {
        // if all tokens are output, split by sequence
        vUbatch = pBalloc.mSplitSeq(pNUbatch);
      } else {
        // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
        const bool cUnified = (cMemAttn->mGetNStream() == 1);

        // [TAG_RECURRENT_ROLLBACK_SPLITS]
        // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
        //   so that the rollback snapshots remain valid
        const uint32_t cNRsSeq = cMemRecr->vNRsSeq;

        vUbatch = pBalloc.mSplitEqual(pNUbatch, !cUnified, cNRsSeq > 0 ? cNRsSeq + 1 : 0);
      }

      if (vUbatch.vNTokens == 0) {
        break;
      }

      lUbatches.push_back(std::move(vUbatch));  // NOLINT
    }

    if (pBalloc.mGetNUsed() < pBalloc.mGetNTokens()) {
      // failed to find a suitable split
      break;
    }

    // prepare the recurrent batches first
    if (!cMemRecr->mPrepare(lUbatches)) {
      // TODO: will the recurrent cache be in an undefined context at this point?
      DGS_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
      return std::make_unique<dgs_memory_hybrid_context>(cDgsMemoryStatusFailedPrepare);
    }

    // prepare the attention cache
    auto lHeadsAttn = cMemAttn->mPrepare(lUbatches);
    if (lHeadsAttn.empty()) {
      DGS_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
      return std::make_unique<dgs_memory_hybrid_context>(cDgsMemoryStatusFailedPrepare);
    }

    return std::make_unique<dgs_memory_hybrid_context>(this, std::move(lHeadsAttn), std::move(lUbatches));
  } while (false);

  return std::make_unique<dgs_memory_hybrid_context>(cDgsMemoryStatusFailedPrepare);
}

dgs_memory_context_ptr dgs_memory_hybrid::mInitFull() {
  return std::make_unique<dgs_memory_hybrid_context>(this);
}

dgs_memory_context_ptr dgs_memory_hybrid::mInitUpdate(dgs_context * pLctx, bool pOptimize) {
  return std::make_unique<dgs_memory_hybrid_context>(this, pLctx, pOptimize);
}

bool dgs_memory_hybrid::mGetCanShift() const {
  // Shifting is trivially supported for recurrent
  return cMemAttn->mGetCanShift();
}

void dgs_memory_hybrid::clear(bool data) {
  cMemAttn->clear(data);
  cMemRecr->clear(data);
}

bool dgs_memory_hybrid::mSeqRm(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1) {
  // Try removing from the recurrent cache first since it may fail. If it does
  // fail, the cache will not have been mutated.
  if (!cMemRecr->mSeqRm(pSeqId, pP0, pP1)) {
    return false;
  }
  return cMemAttn->mSeqRm(pSeqId, pP0, pP1);
}

void dgs_memory_hybrid::mSeqCp(dgs_seq_id pSeqIdSrc, dgs_seq_id pSeqIdDst, dgs_pos pP0, dgs_pos pP1) {
  cMemAttn->mSeqCp(pSeqIdSrc, pSeqIdDst, pP0, pP1);
  cMemRecr->mSeqCp(pSeqIdSrc, pSeqIdDst, pP0, pP1);
}

void dgs_memory_hybrid::mSeqKeep(dgs_seq_id pSeqId) {
  cMemAttn->mSeqKeep(pSeqId);
  cMemRecr->mSeqKeep(pSeqId);
}

void dgs_memory_hybrid::mSeqAdd(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1, dgs_pos pShift) {
  cMemAttn->mSeqAdd(pSeqId, pP0, pP1, pShift);
  cMemRecr->mSeqAdd(pSeqId, pP0, pP1, pShift);
}

void dgs_memory_hybrid::mSeqDiv(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1, int pD) {
  cMemAttn->mSeqDiv(pSeqId, pP0, pP1, pD);
  cMemRecr->mSeqDiv(pSeqId, pP0, pP1, pD);
}

dgs_pos dgs_memory_hybrid::mSeqPosMin(dgs_seq_id pSeqId) const {
  // the min of the total cache is the max of the two caches' min values
  return std::max(cMemAttn->mSeqPosMin(pSeqId), cMemRecr->mSeqPosMin(pSeqId));
}

dgs_pos dgs_memory_hybrid::mSeqPosMax(dgs_seq_id pSeqId) const {
  // the max of the total cache is the min of the two caches' max values
  return std::min(cMemAttn->mSeqPosMax(pSeqId), cMemRecr->mSeqPosMax(pSeqId));
}

std::map<ggml_backend_buffer_type_t, size_t> dgs_memory_hybrid::mMemoryBreakdown() const {
  std::map<ggml_backend_buffer_type_t, size_t> dMb = cMemAttn->mMemoryBreakdown();
  for (const auto & vBuftSize : cMemRecr->mMemoryBreakdown()) {
    dMb[vBuftSize.first] += vBuftSize.second;
  }
  return dMb;
}

void dgs_memory_hybrid::mStateWrite(dgs_io_write_i & pIo, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags) const {
  if ((pFlags & DGS_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
    cMemAttn->mStateWrite(pIo, pSeqId, pFlags);
  }
  cMemRecr->mStateWrite(pIo, pSeqId, pFlags);
}

void dgs_memory_hybrid::mStateRead(dgs_io_read_i & pIo, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags) {
  const bool cReadAttn = (pFlags & DGS_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0;

  if (cReadAttn) {
    cMemAttn->mStateRead(pIo, pSeqId, pFlags);
  }

  try {
    cMemRecr->mStateRead(pIo, pSeqId, pFlags);
  } catch (...) {
    // the attention part is already restored - undo it
    if (cReadAttn) {
      cMemAttn->mStateClear(pSeqId);
    }

    throw;
  }
}

dgs_kv_cache * dgs_memory_hybrid::mGetMemAttn() const {
  return cMemAttn.get();
}

dgs_memory_recurrent * dgs_memory_hybrid::mGetMemRecr() const {
  return cMemRecr.get();
}

dgs_memory_hybrid_context::dgs_memory_hybrid_context(dgs_memory_status pStatus) : cStatus(pStatus) {}

dgs_memory_hybrid_context::dgs_memory_hybrid_context(dgs_memory_hybrid * pMem) :
  cCtxAttn(pMem->mGetMemAttn()->mInitFull()),
  cCtxRecr(pMem->mGetMemRecr()->mInitFull()),
  cStatus(fDgsMemoryStatusCombine(cCtxAttn->mGetStatus(), cCtxRecr->mGetStatus())) {}

dgs_memory_hybrid_context::dgs_memory_hybrid_context(dgs_memory_hybrid * pMem, dgs_context * pLctx, bool pOptimize) :
  cCtxAttn(pMem->mGetMemAttn()->mInitUpdate(pLctx, pOptimize)),
  cCtxRecr(pMem->mGetMemRecr()->mInitUpdate(pLctx, pOptimize)),
  cStatus(fDgsMemoryStatusCombine(cCtxAttn->mGetStatus(), cCtxRecr->mGetStatus())) {}

dgs_memory_hybrid_context::dgs_memory_hybrid_context(dgs_memory_hybrid *     pMem,
                                                     slot_info_vec_t         pSinfosAttn,
                                                     std::vector<dgs_ubatch> pUbatches) :
  lUbatches(std::move(pUbatches)),
  // note: here we copy the ubatches. not sure if this is ideal
  cCtxAttn(new dgs_kv_cache_context(pMem->mGetMemAttn(), std::move(pSinfosAttn), this->lUbatches)),
  cCtxRecr(new dgs_memory_recurrent_context(pMem->mGetMemRecr(), this->lUbatches)),
  cStatus(fDgsMemoryStatusCombine(cCtxAttn->mGetStatus(), cCtxRecr->mGetStatus())) {}

bool dgs_memory_hybrid_context::mNext() {
  assert(status == cDgsMemoryStatusSuccess);

  cCtxAttn->mNext();
  cCtxRecr->mNext();

  if (++vINext >= lUbatches.size()) {
    return false;
  }

  return true;
}

bool dgs_memory_hybrid_context::apply() {
  assert(!fDgsMemoryStatusIsFail(status));

  bool vRes = true;

  vRes = vRes & cCtxAttn->apply();
  vRes = vRes & cCtxRecr->apply();

  return vRes;
}

dgs_memory_status dgs_memory_hybrid_context::mGetStatus() const {
  return cStatus;
}

const dgs_ubatch & dgs_memory_hybrid_context::mGetUbatch() const {
  assert(status == cDgsMemoryStatusSuccess);
  return lUbatches[vINext];
}

const dgs_kv_cache_context * dgs_memory_hybrid_context::mGetAttn() const {
  return static_cast<const dgs_kv_cache_context *>(cCtxAttn.get());
}

const dgs_memory_recurrent_context * dgs_memory_hybrid_context::mGetRecr() const {
  return static_cast<const dgs_memory_recurrent_context *>(cCtxRecr.get());
}
