#include "dgs-memory-hybrid-iswa.h"

#include "dgs-context.h"
#include "dgs-impl.h"
#include "dgs-model.h"

//
// dgs_memory_hybrid_iswa
//

dgs_memory_hybrid_iswa::dgs_memory_hybrid_iswa(const dgs_model &       pModel,
                                               /* attn */
                                               ggml_type               pTypeK,
                                               ggml_type               pTypeV,
                                               bool                    pVTrans,
                                               bool                    pSwaFull,
                                               uint32_t                pKvSize,
                                               uint32_t                pNUbatch,
                                               uint32_t                pNPad,
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
  cMemAttn(new dgs_kv_cache_iswa(pModel,
                                 pTypeK,
                                 pTypeV,
                                 pVTrans,
                                 pOffload,
                                 pSwaFull,
                                 pUnified,
                                 pKvSize,
                                 pNSeqMax,
                                 pNUbatch,
                                 pNPad,
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

dgs_memory_context_ptr dgs_memory_hybrid_iswa::mInitBatch(dgs_batch_allocr & pBalloc,
                                                          uint32_t           pNUbatch,
                                                          bool               pEmbdAll) {
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
        const bool cUnified = (cMemAttn->mGetBase()->mGetNStream() == 1);

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
      return std::make_unique<dgs_memory_hybrid_iswa_context>(cDgsMemoryStatusFailedPrepare);
    }

    // prepare the attention cache (iswa version returns both base and swa slot infos)
    auto lSinfosBase = cMemAttn->mGetBase()->mPrepare(lUbatches);
    if (lSinfosBase.empty()) {
      DGS_LOG_ERROR("%s: failed to prepare attention base ubatches\n", __func__);
      return std::make_unique<dgs_memory_hybrid_iswa_context>(cDgsMemoryStatusFailedPrepare);
    }

    auto lSinfosSwa = cMemAttn->mGetSwa()->mPrepare(lUbatches);
    if (lSinfosSwa.empty()) {
      DGS_LOG_ERROR("%s: failed to prepare attention swa ubatches\n", __func__);
      return std::make_unique<dgs_memory_hybrid_iswa_context>(cDgsMemoryStatusFailedPrepare);
    }

    return std::make_unique<dgs_memory_hybrid_iswa_context>(this, std::move(lSinfosBase), std::move(lSinfosSwa),
                                                            std::move(lUbatches));
  } while (false);

  return std::make_unique<dgs_memory_hybrid_iswa_context>(cDgsMemoryStatusFailedPrepare);
}

dgs_memory_context_ptr dgs_memory_hybrid_iswa::mInitFull() {
  return std::make_unique<dgs_memory_hybrid_iswa_context>(this);
}

dgs_memory_context_ptr dgs_memory_hybrid_iswa::mInitUpdate(dgs_context * pLctx, bool pOptimize) {
  return std::make_unique<dgs_memory_hybrid_iswa_context>(this, pLctx, pOptimize);
}

bool dgs_memory_hybrid_iswa::mGetCanShift() const {
  // Shifting is trivially supported for recurrent
  return cMemAttn->mGetCanShift();
}

void dgs_memory_hybrid_iswa::clear(bool data) {
  cMemAttn->clear(data);
  cMemRecr->clear(data);
}

bool dgs_memory_hybrid_iswa::mSeqRm(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1) {
  // Try removing from the recurrent cache first since it may fail. If it does
  // fail, the cache will not have been mutated.
  if (!cMemRecr->mSeqRm(pSeqId, pP0, pP1)) {
    return false;
  }
  return cMemAttn->mSeqRm(pSeqId, pP0, pP1);
}

void dgs_memory_hybrid_iswa::mSeqCp(dgs_seq_id pSeqIdSrc, dgs_seq_id pSeqIdDst, dgs_pos pP0, dgs_pos pP1) {
  cMemAttn->mSeqCp(pSeqIdSrc, pSeqIdDst, pP0, pP1);
  cMemRecr->mSeqCp(pSeqIdSrc, pSeqIdDst, pP0, pP1);
}

void dgs_memory_hybrid_iswa::mSeqKeep(dgs_seq_id pSeqId) {
  cMemAttn->mSeqKeep(pSeqId);
  cMemRecr->mSeqKeep(pSeqId);
}

void dgs_memory_hybrid_iswa::mSeqAdd(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1, dgs_pos pShift) {
  cMemAttn->mSeqAdd(pSeqId, pP0, pP1, pShift);
  cMemRecr->mSeqAdd(pSeqId, pP0, pP1, pShift);
}

void dgs_memory_hybrid_iswa::mSeqDiv(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1, int pD) {
  cMemAttn->mSeqDiv(pSeqId, pP0, pP1, pD);
  cMemRecr->mSeqDiv(pSeqId, pP0, pP1, pD);
}

dgs_pos dgs_memory_hybrid_iswa::mSeqPosMin(dgs_seq_id pSeqId) const {
  // the min of the total cache is the max of the two caches' min values
  return std::max(cMemAttn->mSeqPosMin(pSeqId), cMemRecr->mSeqPosMin(pSeqId));
}

dgs_pos dgs_memory_hybrid_iswa::mSeqPosMax(dgs_seq_id pSeqId) const {
  // the max of the total cache is the min of the two caches' max values
  return std::min(cMemAttn->mSeqPosMax(pSeqId), cMemRecr->mSeqPosMax(pSeqId));
}

std::map<ggml_backend_buffer_type_t, size_t> dgs_memory_hybrid_iswa::mMemoryBreakdown() const {
  std::map<ggml_backend_buffer_type_t, size_t> dMb = cMemAttn->mMemoryBreakdown();
  for (const auto & vBuftSize : cMemRecr->mMemoryBreakdown()) {
    dMb[vBuftSize.first] += vBuftSize.second;
  }
  return dMb;
}

void dgs_memory_hybrid_iswa::mStateWrite(dgs_io_write_i & pIo, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags) const {
  cMemAttn->mStateWrite(pIo, pSeqId, pFlags);
  cMemRecr->mStateWrite(pIo, pSeqId, pFlags);
}

void dgs_memory_hybrid_iswa::mStateRead(dgs_io_read_i & pIo, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags) {
  cMemAttn->mStateRead(pIo, pSeqId, pFlags);
  cMemRecr->mStateRead(pIo, pSeqId, pFlags);
}

dgs_kv_cache_iswa * dgs_memory_hybrid_iswa::mGetMemAttn() const {
  return cMemAttn.get();
}

dgs_memory_recurrent * dgs_memory_hybrid_iswa::mGetMemRecr() const {
  return cMemRecr.get();
}

//
// dgs_memory_hybrid_iswa_context
//

dgs_memory_hybrid_iswa_context::dgs_memory_hybrid_iswa_context(dgs_memory_status pStatus) : cStatus(pStatus) {}

dgs_memory_hybrid_iswa_context::dgs_memory_hybrid_iswa_context(dgs_memory_hybrid_iswa * pMem) :
  cCtxAttn(pMem->mGetMemAttn()->mInitFull()),
  cCtxRecr(pMem->mGetMemRecr()->mInitFull()),
  cStatus(fDgsMemoryStatusCombine(cCtxAttn->mGetStatus(), cCtxRecr->mGetStatus())) {}

dgs_memory_hybrid_iswa_context::dgs_memory_hybrid_iswa_context(dgs_memory_hybrid_iswa * pMem,
                                                               dgs_context *            pLctx,
                                                               bool                     pOptimize) :
  cCtxAttn(pMem->mGetMemAttn()->mInitUpdate(pLctx, pOptimize)),
  cCtxRecr(pMem->mGetMemRecr()->mInitUpdate(pLctx, pOptimize)),
  cStatus(fDgsMemoryStatusCombine(cCtxAttn->mGetStatus(), cCtxRecr->mGetStatus())) {}

dgs_memory_hybrid_iswa_context::dgs_memory_hybrid_iswa_context(dgs_memory_hybrid_iswa * pMem,
                                                               slot_info_vec_t          pSinfosBase,
                                                               slot_info_vec_t          pSinfosSwa,
                                                               std::vector<dgs_ubatch>  pUbatches) :
  lUbatches(std::move(pUbatches)),
  // note: here we copy the ubatches. not sure if this is ideal
  cCtxAttn(
    new dgs_kv_cache_iswa_context(pMem->mGetMemAttn(), std::move(pSinfosBase), std::move(pSinfosSwa), this->lUbatches)),
  cCtxRecr(new dgs_memory_recurrent_context(pMem->mGetMemRecr(), this->lUbatches)),
  cStatus(fDgsMemoryStatusCombine(cCtxAttn->mGetStatus(), cCtxRecr->mGetStatus())) {}

bool dgs_memory_hybrid_iswa_context::mNext() {
  assert(status == cDgsMemoryStatusSuccess);

  cCtxAttn->mNext();
  cCtxRecr->mNext();

  if (++vINext >= lUbatches.size()) {
    return false;
  }

  return true;
}

bool dgs_memory_hybrid_iswa_context::apply() {
  assert(!fDgsMemoryStatusIsFail(status));

  bool vRes = true;

  vRes = vRes & cCtxAttn->apply();
  vRes = vRes & cCtxRecr->apply();

  return vRes;
}

dgs_memory_status dgs_memory_hybrid_iswa_context::mGetStatus() const {
  return cStatus;
}

const dgs_ubatch & dgs_memory_hybrid_iswa_context::mGetUbatch() const {
  assert(status == cDgsMemoryStatusSuccess);
  return lUbatches[vINext];
}

const dgs_kv_cache_iswa_context * dgs_memory_hybrid_iswa_context::mGetAttn() const {
  return static_cast<const dgs_kv_cache_iswa_context *>(cCtxAttn.get());
}

const dgs_memory_recurrent_context * dgs_memory_hybrid_iswa_context::mGetRecr() const {
  return static_cast<const dgs_memory_recurrent_context *>(cCtxRecr.get());
}
