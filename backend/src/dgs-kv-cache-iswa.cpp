#include "dgs-kv-cache-iswa.h"

#include "dgs-batch.h"
#include "dgs-impl.h"
#include "dgs-model.h"

#include <algorithm>
#include <cassert>

//
// dgs_kv_cache_iswa
//

dgs_kv_cache_iswa::dgs_kv_cache_iswa(const dgs_model &       pModel,
                                     ggml_type               pTypeK,
                                     ggml_type               pTypeV,
                                     bool                    pVTrans,
                                     bool                    pOffload,
                                     bool                    pSwaFull,
                                     bool                    pUnified,
                                     uint32_t                pKvSize,
                                     uint32_t                pNSeqMax,
                                     uint32_t                pNUbatch,
                                     uint32_t                pNPad,
                                     dgs_memory_t            pMemOther,
                                     const layer_filter_cb & pFilter,
                                     const layer_reuse_cb &  pReuse,
                                     const layer_share_cb &  pShare) :
  dgs_kv_cache_iswa(pModel,
                    pModel.hparams,
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
                    pMemOther,
                    pFilter,
                    pReuse,
                    pShare) {}

dgs_kv_cache_iswa::dgs_kv_cache_iswa(const dgs_model &       pModel,
                                     const dgs_hparams &     pHparams,
                                     ggml_type               pTypeK,
                                     ggml_type               pTypeV,
                                     bool                    pVTrans,
                                     bool                    pOffload,
                                     bool                    pSwaFull,
                                     bool                    pUnified,
                                     uint32_t                pKvSize,
                                     uint32_t                pNSeqMax,
                                     uint32_t                pNUbatch,
                                     uint32_t                pNPad,
                                     dgs_memory_t            pMemOther,
                                     const layer_filter_cb & pFilter,
                                     const layer_reuse_cb &  pReuse,
                                     const layer_share_cb &  pShare) :
  cUnified(pUnified) {
  // chain filters
  const layer_filter_cb cFilterBase = [&](int32_t pIl) {
    if (pFilter && !pFilter(pIl)) {
      return false;
    }

    return !pModel.hparams.mIsSwa(pIl);
  };

  const layer_filter_cb cFilterSwa = [&](int32_t pIl) {
    if (pFilter && !pFilter(pIl)) {
      return false;
    }

    return pModel.hparams.mIsSwa(pIl);
  };

  const uint32_t cSizeBase = pKvSize;

  // note: the SWA cache is always padded to 256 for performance
  uint32_t vSizeSwa = GGML_PAD(std::min(cSizeBase, pHparams.vNSwa * (pUnified ? pNSeqMax : 1) + pNUbatch), 256);

  // when using full-size SWA cache, we set the SWA cache size to be equal to the base cache size
  if (pSwaFull) {
    DGS_LOG_WARN("%s: using full-size SWA cache (ref: %s)\n", __func__,
                 "doc/MANUAL.md");

    vSizeSwa = cSizeBase;
  }

  DGS_LOG_INFO("%s: creating non-SWA KV cache, size = %u cells\n", __func__, cSizeBase);

  dgs_memory_t vMemOtherBase = nullptr;
  if (pMemOther) {
    vMemOtherBase = static_cast<dgs_kv_cache_iswa *>(pMemOther)->mGetBase();
  }

  dgs_memory_t vMemOtherSwa = nullptr;
  if (pMemOther) {
    vMemOtherSwa = static_cast<dgs_kv_cache_iswa *>(pMemOther)->mGetSwa();
  }

  vKvBase =
    std::make_unique<dgs_kv_cache>(pModel, pHparams, pTypeK, pTypeV, pVTrans, pOffload, pUnified, cSizeBase, pNSeqMax,
                                   pNPad, 0, cDgsSwaTypeNone, vMemOtherBase, cFilterBase, pReuse, pShare);

  DGS_LOG_INFO("%s: creating     SWA KV cache, size = %u cells\n", __func__, vSizeSwa);

  vKvSwa =
    std::make_unique<dgs_kv_cache>(pModel, pHparams, pTypeK, pTypeV, pVTrans, pOffload, pUnified, vSizeSwa, pNSeqMax,
                                   pNPad, pHparams.vNSwa, pHparams.vSwaType, vMemOtherSwa, cFilterSwa, pReuse, pShare);
}

void dgs_kv_cache_iswa::clear(bool data) {
  vKvBase->clear(data);
  vKvSwa->clear(data);
}

bool dgs_kv_cache_iswa::mSeqRm(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1) {
  bool vRes = true;

  vRes = vRes & vKvBase->mSeqRm(pSeqId, pP0, pP1);
  vRes = vRes & vKvSwa->mSeqRm(pSeqId, pP0, pP1);

  return vRes;
}

void dgs_kv_cache_iswa::mSeqCp(dgs_seq_id pSeqIdSrc, dgs_seq_id pSeqIdDst, dgs_pos pP0, dgs_pos pP1) {
  vKvBase->mSeqCp(pSeqIdSrc, pSeqIdDst, pP0, pP1);
  vKvSwa->mSeqCp(pSeqIdSrc, pSeqIdDst, pP0, pP1);
}

void dgs_kv_cache_iswa::mSeqKeep(dgs_seq_id pSeqId) {
  vKvBase->mSeqKeep(pSeqId);
  vKvSwa->mSeqKeep(pSeqId);
}

void dgs_kv_cache_iswa::mSeqAdd(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1, dgs_pos pShift) {
  vKvBase->mSeqAdd(pSeqId, pP0, pP1, pShift);
  vKvSwa->mSeqAdd(pSeqId, pP0, pP1, pShift);
}

void dgs_kv_cache_iswa::mSeqDiv(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1, int pD) {
  vKvBase->mSeqDiv(pSeqId, pP0, pP1, pD);
  vKvSwa->mSeqDiv(pSeqId, pP0, pP1, pD);
}

dgs_pos dgs_kv_cache_iswa::mSeqPosMin(dgs_seq_id pSeqId) const {
  // the base cache is a superset of the SWA cache, so we can just check the SWA cache
  return vKvSwa->mSeqPosMin(pSeqId);
}

dgs_pos dgs_kv_cache_iswa::mSeqPosMax(dgs_seq_id pSeqId) const {
  return vKvSwa->mSeqPosMax(pSeqId);
}

std::map<ggml_backend_buffer_type_t, size_t> dgs_kv_cache_iswa::mMemoryBreakdown() const {
  std::map<ggml_backend_buffer_type_t, size_t> dMb = vKvBase->mMemoryBreakdown();
  for (const auto & vBuftSize : vKvSwa->mMemoryBreakdown()) {
    dMb[vBuftSize.first] += vBuftSize.second;
  }
  return dMb;
}

dgs_memory_context_ptr dgs_kv_cache_iswa::mInitBatch(dgs_batch_allocr & pBalloc, uint32_t pNUbatch, bool pEmbdAll) {
  // first try simple split
  do {
    if (!cUnified) {
      // requires equal splits, so we skip the simple split
      break;
    }

    pBalloc.mSplitReset();

    std::vector<dgs_ubatch> lUbatches;
    while (true) {
      auto vUbatch = pBalloc.mSplitSimple(pNUbatch);

      if (vUbatch.vNTokens == 0) {
        break;
      }

      lUbatches.push_back(std::move(vUbatch));  // NOLINT
    }

    if (pBalloc.mGetNUsed() < pBalloc.mGetNTokens()) {
      // failed to find a suitable split
      break;
    }

    auto lSinfosBase = vKvBase->mPrepare(lUbatches);
    if (lSinfosBase.empty()) {
      break;
    }

    auto lSinfosSwa = vKvSwa->mPrepare(lUbatches);
    if (lSinfosSwa.empty()) {
      break;
    }

    assert(sinfos_base.size() == sinfos_swa.size());

    return std::make_unique<dgs_kv_cache_iswa_context>(this, std::move(lSinfosBase), std::move(lSinfosSwa),
                                                       std::move(lUbatches));
  } while (false);

  // if it fails, try equal split
  do {
    pBalloc.mSplitReset();

    std::vector<dgs_ubatch> lUbatches;
    while (true) {
      auto vUbatch = pEmbdAll ? pBalloc.mSplitSeq(pNUbatch) : pBalloc.mSplitEqual(pNUbatch, !cUnified, 0);

      if (vUbatch.vNTokens == 0) {
        break;
      }

      lUbatches.push_back(std::move(vUbatch));  // NOLINT
    }

    if (pBalloc.mGetNUsed() < pBalloc.mGetNTokens()) {
      // failed to find a suitable split
      break;
    }

    auto lSinfosBase = vKvBase->mPrepare(lUbatches);
    if (lSinfosBase.empty()) {
      break;
    }

    auto lSinfosSwa = vKvSwa->mPrepare(lUbatches);
    if (lSinfosSwa.empty()) {
      break;
    }

    assert(sinfos_base.size() == sinfos_swa.size());

    return std::make_unique<dgs_kv_cache_iswa_context>(this, std::move(lSinfosBase), std::move(lSinfosSwa),
                                                       std::move(lUbatches));
  } while (false);

  // TODO: if we fail again, we should attempt different splitting strategies
  //       but to do that properly, we first have to refactor the batches to be more flexible

  return std::make_unique<dgs_kv_cache_iswa_context>(cDgsMemoryStatusFailedPrepare);
}

dgs_memory_context_ptr dgs_kv_cache_iswa::mInitFull() {
  return std::make_unique<dgs_kv_cache_iswa_context>(this);
}

dgs_memory_context_ptr dgs_kv_cache_iswa::mInitUpdate(dgs_context * pLctx, bool pOptimize) {
  return std::make_unique<dgs_kv_cache_iswa_context>(this, pLctx, pOptimize);
}

bool dgs_kv_cache_iswa::mGetCanShift() const {
  return vKvBase->mGetCanShift() && vKvSwa->mGetCanShift() && vKvBase->mGetSize() == vKvSwa->mGetSize();
}

void dgs_kv_cache_iswa::mStateWrite(dgs_io_write_i & pIo, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags) const {
  if ((pFlags & DGS_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
    vKvBase->mStateWrite(pIo, pSeqId, pFlags);
  }

  vKvSwa->mStateWrite(pIo, pSeqId, pFlags);
}

void dgs_kv_cache_iswa::mStateRead(dgs_io_read_i & pIo, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags) {
  if ((pFlags & DGS_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
    vKvBase->mStateRead(pIo, pSeqId, pFlags);
  }

  vKvSwa->mStateRead(pIo, pSeqId, pFlags);
}

dgs_kv_cache * dgs_kv_cache_iswa::mGetBase() const {
  return vKvBase.get();
}

dgs_kv_cache * dgs_kv_cache_iswa::mGetSwa() const {
  return vKvSwa.get();
}

//
// dgs_kv_cache_iswa_context
//

dgs_kv_cache_iswa_context::dgs_kv_cache_iswa_context(dgs_memory_status pStatus) : cStatus(pStatus) {}

dgs_kv_cache_iswa_context::dgs_kv_cache_iswa_context(dgs_kv_cache_iswa * pKv) :
  cCtxBase(pKv->mGetBase()->mInitFull()),
  cCtxSwa(pKv->mGetSwa()->mInitFull()),
  cStatus(fDgsMemoryStatusCombine(cCtxBase->mGetStatus(), cCtxSwa->mGetStatus())) {}

dgs_kv_cache_iswa_context::dgs_kv_cache_iswa_context(dgs_kv_cache_iswa * pKv, dgs_context * pLctx, bool pOptimize) :
  cCtxBase(pKv->mGetBase()->mInitUpdate(pLctx, pOptimize)),
  cCtxSwa(pKv->mGetSwa()->mInitUpdate(pLctx, pOptimize)),
  cStatus(fDgsMemoryStatusCombine(cCtxBase->mGetStatus(), cCtxSwa->mGetStatus())) {}

dgs_kv_cache_iswa_context::dgs_kv_cache_iswa_context(dgs_kv_cache_iswa *     pKv,
                                                     slot_info_vec_t         pSinfosBase,
                                                     slot_info_vec_t         pSinfosSwa,
                                                     std::vector<dgs_ubatch> pUbatches) :
  lUbatches(std::move(pUbatches)),
  // note: here we copy the ubatches. not sure if this is ideal
  cCtxBase(new dgs_kv_cache_context(pKv->mGetBase(), std::move(pSinfosBase), this->lUbatches)),
  cCtxSwa(new dgs_kv_cache_context(pKv->mGetSwa(), std::move(pSinfosSwa), this->lUbatches)),
  cStatus(fDgsMemoryStatusCombine(cCtxBase->mGetStatus(), cCtxSwa->mGetStatus())) {}

dgs_kv_cache_iswa_context::~dgs_kv_cache_iswa_context() = default;

bool dgs_kv_cache_iswa_context::mNext() {
  assert(status == cDgsMemoryStatusSuccess);

  cCtxBase->mNext();
  cCtxSwa->mNext();

  if (++vINext >= lUbatches.size()) {
    return false;
  }

  return true;
}

bool dgs_kv_cache_iswa_context::apply() {
  assert(!fDgsMemoryStatusIsFail(status));

  bool vRes = true;

  vRes = vRes & cCtxBase->apply();
  vRes = vRes & cCtxSwa->apply();

  return vRes;
}

dgs_memory_status dgs_kv_cache_iswa_context::mGetStatus() const {
  return cStatus;
}

const dgs_ubatch & dgs_kv_cache_iswa_context::mGetUbatch() const {
  assert(status == cDgsMemoryStatusSuccess);

  return lUbatches[vINext];
}

const dgs_kv_cache_context * dgs_kv_cache_iswa_context::mGetBase() const {
  assert(status == cDgsMemoryStatusSuccess);

  return static_cast<const dgs_kv_cache_context *>(cCtxBase.get());
}

const dgs_kv_cache_context * dgs_kv_cache_iswa_context::mGetSwa() const {
  assert(status == cDgsMemoryStatusSuccess);

  return static_cast<const dgs_kv_cache_context *>(cCtxSwa.get());
}
