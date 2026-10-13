#include "dgs-kv-cache-dsa-iswa.h"

#include "dgs-batch.h"
#include "dgs-impl.h"
#include "dgs-model.h"

#include <algorithm>
#include <cassert>

//
// dgs_kv_cache_dsa_iswa
//

dgs_kv_cache_dsa_iswa::dgs_kv_cache_dsa_iswa(const dgs_model &       pModel,
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
                                             const layer_filter_cb & pFilterMla,
                                             const layer_filter_cb & pFilterLid,
                                             const layer_reuse_cb &  pReuse) :
  cUnified(pUnified) {
  const auto & vHparams = pModel.hparams;

  // chain filters
  const layer_filter_cb cFilterDsa = [&](int32_t pIl) {
    if (pFilterMla && !pFilterMla(pIl)) {
      return false;
    }

    return !vHparams.mIsSwa(pIl);
  };

  const layer_filter_cb cFilterSwa = [&](int32_t pIl) {
    if (pFilterMla && !pFilterMla(pIl)) {
      return false;
    }

    return vHparams.mIsSwa(pIl);
  };

  const uint32_t cSizeDsa = pKvSize;

  // note: the SWA cache is always padded to 256 for performance
  uint32_t vSizeSwa = GGML_PAD(std::min(cSizeDsa, vHparams.vNSwa * (pUnified ? pNSeqMax : 1) + pNUbatch), 256);

  // when using full-size SWA cache, we set the SWA cache size to be equal to the base cache size
  if (pSwaFull) {
    DGS_LOG_WARN("%s: using full-size SWA cache (ref: %s)\n", __func__,
                 "doc/MANUAL.md");

    vSizeSwa = cSizeDsa;
  }

  DGS_LOG_INFO("%s: creating DSA KV cache, size = %u cells\n", __func__, cSizeDsa);

  vKvDsa = std::make_unique<dgs_kv_cache_dsa>(pModel, pTypeK, pTypeV, pVTrans, pOffload, pUnified, cSizeDsa, pNSeqMax,
                                              pNPad, 0, cDgsSwaTypeNone, cFilterDsa, pFilterLid, pReuse);

  DGS_LOG_INFO("%s: creating SWA KV cache, size = %u cells\n", __func__, vSizeSwa);

  vKvSwa =
    std::make_unique<dgs_kv_cache>(pModel, vHparams, pTypeK, pTypeV, pVTrans, pOffload, pUnified, vSizeSwa, pNSeqMax,
                                   pNPad, vHparams.vNSwa, vHparams.vSwaType, nullptr, cFilterSwa, pReuse, nullptr);
}

void dgs_kv_cache_dsa_iswa::clear(bool data) {
  vKvDsa->clear(data);
  vKvSwa->clear(data);
}

bool dgs_kv_cache_dsa_iswa::mSeqRm(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1) {
  bool vRes = true;

  vRes = vRes & vKvDsa->mSeqRm(pSeqId, pP0, pP1);
  vRes = vRes & vKvSwa->mSeqRm(pSeqId, pP0, pP1);

  return vRes;
}

void dgs_kv_cache_dsa_iswa::mSeqCp(dgs_seq_id pSeqIdSrc, dgs_seq_id pSeqIdDst, dgs_pos pP0, dgs_pos pP1) {
  vKvDsa->mSeqCp(pSeqIdSrc, pSeqIdDst, pP0, pP1);
  vKvSwa->mSeqCp(pSeqIdSrc, pSeqIdDst, pP0, pP1);
}

void dgs_kv_cache_dsa_iswa::mSeqKeep(dgs_seq_id pSeqId) {
  vKvDsa->mSeqKeep(pSeqId);
  vKvSwa->mSeqKeep(pSeqId);
}

void dgs_kv_cache_dsa_iswa::mSeqAdd(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1, dgs_pos pShift) {
  vKvDsa->mSeqAdd(pSeqId, pP0, pP1, pShift);
  vKvSwa->mSeqAdd(pSeqId, pP0, pP1, pShift);
}

void dgs_kv_cache_dsa_iswa::mSeqDiv(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1, int pD) {
  vKvDsa->mSeqDiv(pSeqId, pP0, pP1, pD);
  vKvSwa->mSeqDiv(pSeqId, pP0, pP1, pD);
}

dgs_pos dgs_kv_cache_dsa_iswa::mSeqPosMin(dgs_seq_id pSeqId) const {
  // the DSA cache is a superset of the SWA cache, so we can just check the SWA cache
  return vKvSwa->mSeqPosMin(pSeqId);
}

dgs_pos dgs_kv_cache_dsa_iswa::mSeqPosMax(dgs_seq_id pSeqId) const {
  return vKvSwa->mSeqPosMax(pSeqId);
}

std::map<ggml_backend_buffer_type_t, size_t> dgs_kv_cache_dsa_iswa::mMemoryBreakdown() const {
  std::map<ggml_backend_buffer_type_t, size_t> dMb = vKvDsa->mMemoryBreakdown();
  for (const auto & vBuftSize : vKvSwa->mMemoryBreakdown()) {
    dMb[vBuftSize.first] += vBuftSize.second;
  }
  return dMb;
}

dgs_memory_context_ptr dgs_kv_cache_dsa_iswa::mInitBatch(dgs_batch_allocr & pBalloc, uint32_t pNUbatch, bool pEmbdAll) {
  GGML_UNUSED(pEmbdAll);

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

    auto lSinfosMla = vKvDsa->mGetMla()->mPrepare(lUbatches);
    if (lSinfosMla.empty()) {
      break;
    }

    auto lSinfosLid = vKvDsa->mGetLid()->mPrepare(lUbatches);
    if (lSinfosLid.empty()) {
      break;
    }

    auto lSinfosSwa = vKvSwa->mPrepare(lUbatches);
    if (lSinfosSwa.empty()) {
      break;
    }

    assert(sinfos_mla.size() == sinfos_swa.size());

    return std::make_unique<dgs_kv_cache_dsa_iswa_context>(this, std::move(lSinfosMla), std::move(lSinfosLid),
                                                           std::move(lSinfosSwa), std::move(lUbatches));
  } while (false);

  // if it fails, try equal split
  do {
    pBalloc.mSplitReset();

    std::vector<dgs_ubatch> lUbatches;
    while (true) {
      auto vUbatch = pBalloc.mSplitEqual(pNUbatch, !cUnified, 0);

      if (vUbatch.vNTokens == 0) {
        break;
      }

      lUbatches.push_back(std::move(vUbatch));  // NOLINT
    }

    if (pBalloc.mGetNUsed() < pBalloc.mGetNTokens()) {
      // failed to find a suitable split
      break;
    }

    auto lSinfosMla = vKvDsa->mGetMla()->mPrepare(lUbatches);
    if (lSinfosMla.empty()) {
      break;
    }

    auto lSinfosLid = vKvDsa->mGetLid()->mPrepare(lUbatches);
    if (lSinfosLid.empty()) {
      break;
    }

    auto lSinfosSwa = vKvSwa->mPrepare(lUbatches);
    if (lSinfosSwa.empty()) {
      break;
    }

    assert(sinfos_mla.size() == sinfos_swa.size());

    return std::make_unique<dgs_kv_cache_dsa_iswa_context>(this, std::move(lSinfosMla), std::move(lSinfosLid),
                                                           std::move(lSinfosSwa), std::move(lUbatches));
  } while (false);

  return std::make_unique<dgs_kv_cache_dsa_iswa_context>(cDgsMemoryStatusFailedPrepare);
}

dgs_memory_context_ptr dgs_kv_cache_dsa_iswa::mInitFull() {
  return std::make_unique<dgs_kv_cache_dsa_iswa_context>(this);
}

dgs_memory_context_ptr dgs_kv_cache_dsa_iswa::mInitUpdate(dgs_context * pLctx, bool pOptimize) {
  return std::make_unique<dgs_kv_cache_dsa_iswa_context>(this, pLctx, pOptimize);
}

bool dgs_kv_cache_dsa_iswa::mGetCanShift() const {
  return vKvDsa->mGetCanShift() && vKvSwa->mGetCanShift() && vKvDsa->mGetMla()->mGetSize() == vKvSwa->mGetSize();
}

void dgs_kv_cache_dsa_iswa::mStateWrite(dgs_io_write_i & pIo, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags) const {
  if ((pFlags & DGS_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
    vKvDsa->mStateWrite(pIo, pSeqId, pFlags);
  }

  vKvSwa->mStateWrite(pIo, pSeqId, pFlags);
}

void dgs_kv_cache_dsa_iswa::mStateRead(dgs_io_read_i & pIo, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags) {
  if ((pFlags & DGS_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
    vKvDsa->mStateRead(pIo, pSeqId, pFlags);
  }

  vKvSwa->mStateRead(pIo, pSeqId, pFlags);
}

dgs_kv_cache_dsa * dgs_kv_cache_dsa_iswa::mGetDsa() const {
  return vKvDsa.get();
}

dgs_kv_cache * dgs_kv_cache_dsa_iswa::mGetSwa() const {
  return vKvSwa.get();
}

//
// dgs_kv_cache_dsa_iswa_context
//

dgs_kv_cache_dsa_iswa_context::dgs_kv_cache_dsa_iswa_context(dgs_memory_status pStatus) : cStatus(pStatus) {}

dgs_kv_cache_dsa_iswa_context::dgs_kv_cache_dsa_iswa_context(dgs_kv_cache_dsa_iswa * pKv) :
  cCtxDsa(pKv->mGetDsa()->mInitFull()),
  cCtxSwa(pKv->mGetSwa()->mInitFull()),
  cStatus(fDgsMemoryStatusCombine(cCtxDsa->mGetStatus(), cCtxSwa->mGetStatus())) {}

dgs_kv_cache_dsa_iswa_context::dgs_kv_cache_dsa_iswa_context(dgs_kv_cache_dsa_iswa * pKv,
                                                             dgs_context *           pLctx,
                                                             bool                    pOptimize) :
  cCtxDsa(pKv->mGetDsa()->mInitUpdate(pLctx, pOptimize)),
  cCtxSwa(pKv->mGetSwa()->mInitUpdate(pLctx, pOptimize)),
  cStatus(fDgsMemoryStatusCombine(cCtxDsa->mGetStatus(), cCtxSwa->mGetStatus())) {}

dgs_kv_cache_dsa_iswa_context::dgs_kv_cache_dsa_iswa_context(dgs_kv_cache_dsa_iswa * pKv,
                                                             slot_info_vec_t         pSinfosMla,
                                                             slot_info_vec_t         pSinfosLid,
                                                             slot_info_vec_t         pSinfosSwa,
                                                             std::vector<dgs_ubatch> pUbatches) :
  lUbatches(std::move(pUbatches)),
  // note: here we copy the ubatches. not sure if this is ideal
  cCtxDsa(new dgs_kv_cache_dsa_context(pKv->mGetDsa(), std::move(pSinfosMla), std::move(pSinfosLid), this->lUbatches)),
  cCtxSwa(new dgs_kv_cache_context(pKv->mGetSwa(), std::move(pSinfosSwa), this->lUbatches)),
  cStatus(fDgsMemoryStatusCombine(cCtxDsa->mGetStatus(), cCtxSwa->mGetStatus())) {}

dgs_kv_cache_dsa_iswa_context::~dgs_kv_cache_dsa_iswa_context() = default;

bool dgs_kv_cache_dsa_iswa_context::mNext() {
  assert(status == cDgsMemoryStatusSuccess);

  cCtxDsa->mNext();
  cCtxSwa->mNext();

  if (++vINext >= lUbatches.size()) {
    return false;
  }

  return true;
}

bool dgs_kv_cache_dsa_iswa_context::apply() {
  assert(!fDgsMemoryStatusIsFail(status));

  bool vRes = true;

  vRes = vRes & cCtxDsa->apply();
  vRes = vRes & cCtxSwa->apply();

  return vRes;
}

dgs_memory_status dgs_kv_cache_dsa_iswa_context::mGetStatus() const {
  return cStatus;
}

const dgs_ubatch & dgs_kv_cache_dsa_iswa_context::mGetUbatch() const {
  assert(status == cDgsMemoryStatusSuccess);

  return lUbatches[vINext];
}

const dgs_kv_cache_dsa_context * dgs_kv_cache_dsa_iswa_context::mGetDsa() const {
  assert(status == cDgsMemoryStatusSuccess);

  return static_cast<const dgs_kv_cache_dsa_context *>(cCtxDsa.get());
}

const dgs_kv_cache_context * dgs_kv_cache_dsa_iswa_context::mGetSwa() const {
  assert(status == cDgsMemoryStatusSuccess);

  return static_cast<const dgs_kv_cache_context *>(cCtxSwa.get());
}
