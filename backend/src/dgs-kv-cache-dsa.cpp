#include "dgs-kv-cache-dsa.h"

#include "dgs-batch.h"
#include "dgs-impl.h"
#include "dgs-model.h"

#include <algorithm>
#include <cassert>

//
// dgs_kv_cache_dsa
//

dgs_kv_cache_dsa::dgs_kv_cache_dsa(const dgs_model &       pModel,
                                   ggml_type               pTypeK,
                                   ggml_type               pTypeV,
                                   bool                    pVTrans,
                                   bool                    pOffload,
                                   bool                    pUnified,
                                   uint32_t                pKvSize,
                                   uint32_t                pNSeqMax,
                                   uint32_t                pNPad,
                                   uint32_t                pNSwa,
                                   dgs_swa_type            pSwaType,
                                   const layer_filter_cb & pFilterMla,
                                   const layer_filter_cb & pFilterLid,
                                   const layer_reuse_cb &  pReuse) :
  vHparamsLid(pModel.hparams),
  cNStream(pUnified ? 1 : pNSeqMax) {
  DGS_LOG_INFO("%s: creating main KV cache, size = %u cells\n", __func__, pKvSize);

  vKvMla = std::make_unique<dgs_kv_cache>(pModel, pModel.hparams, pTypeK, pTypeV, pVTrans, pOffload, pUnified, pKvSize,
                                          pNSeqMax, pNPad, pNSwa, pSwaType, nullptr, pFilterMla, pReuse, nullptr);

  // we use dgs_kv_cache for caching indexer keys
  // by hand-tweaking some hparams we fool it to create
  // indexer key cache tensors with correct dimensions

  // DSA lightning indexer uses MQA with single key head
  std::fill(vHparamsLid.aNHeadKvArr.begin(), vHparamsLid.aNHeadKvArr.end(), 1);
  vHparamsLid.vNEmbdHeadKFull = pModel.hparams.vIndexerHeadSize;
  vHparamsLid.vRopeType       = cDgsRopeTypeNeox;

  DGS_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, pKvSize);

  vKvLid = std::make_unique<dgs_kv_cache>(pModel, vHparamsLid, pTypeK, pTypeV, pVTrans, pOffload, pUnified, pKvSize,
                                          pNSeqMax, pNPad, pNSwa, pSwaType, nullptr, pFilterLid, pReuse, nullptr);
}

void dgs_kv_cache_dsa::clear(bool data) {
  vKvMla->clear(data);
  vKvLid->clear(data);
}

bool dgs_kv_cache_dsa::mSeqRm(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1) {
  bool vRes = true;

  vRes = vRes & vKvMla->mSeqRm(pSeqId, pP0, pP1);
  vRes = vRes & vKvLid->mSeqRm(pSeqId, pP0, pP1);

  return vRes;
}

void dgs_kv_cache_dsa::mSeqCp(dgs_seq_id pSeqIdSrc, dgs_seq_id pSeqIdDst, dgs_pos pP0, dgs_pos pP1) {
  vKvMla->mSeqCp(pSeqIdSrc, pSeqIdDst, pP0, pP1);
  vKvLid->mSeqCp(pSeqIdSrc, pSeqIdDst, pP0, pP1);
}

void dgs_kv_cache_dsa::mSeqKeep(dgs_seq_id pSeqId) {
  vKvMla->mSeqKeep(pSeqId);
  vKvLid->mSeqKeep(pSeqId);
}

void dgs_kv_cache_dsa::mSeqAdd(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1, dgs_pos pShift) {
  vKvMla->mSeqAdd(pSeqId, pP0, pP1, pShift);
  vKvLid->mSeqAdd(pSeqId, pP0, pP1, pShift);
}

void dgs_kv_cache_dsa::mSeqDiv(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1, int pD) {
  vKvMla->mSeqDiv(pSeqId, pP0, pP1, pD);
  vKvLid->mSeqDiv(pSeqId, pP0, pP1, pD);
}

dgs_pos dgs_kv_cache_dsa::mSeqPosMin(dgs_seq_id pSeqId) const {
  return vKvMla->mSeqPosMin(pSeqId);
}

dgs_pos dgs_kv_cache_dsa::mSeqPosMax(dgs_seq_id pSeqId) const {
  return vKvMla->mSeqPosMax(pSeqId);
}

std::map<ggml_backend_buffer_type_t, size_t> dgs_kv_cache_dsa::mMemoryBreakdown() const {
  std::map<ggml_backend_buffer_type_t, size_t> dMb = vKvMla->mMemoryBreakdown();
  for (const auto & vBuftSize : vKvLid->mMemoryBreakdown()) {
    dMb[vBuftSize.first] += vBuftSize.second;
  }
  return dMb;
}

dgs_memory_context_ptr dgs_kv_cache_dsa::mInitBatch(dgs_batch_allocr & pBalloc, uint32_t pNUbatch, bool pEmbdAll) {
  GGML_UNUSED(pEmbdAll);

  do {
    pBalloc.mSplitReset();

    std::vector<dgs_ubatch> lUbatches;
    while (true) {
      auto vUbatch = cNStream == 1 ? pBalloc.mSplitSimple(pNUbatch) : pBalloc.mSplitEqual(pNUbatch, true, 0);

      if (vUbatch.vNTokens == 0) {
        break;
      }

      lUbatches.push_back(std::move(vUbatch));  // NOLINT
    }

    if (pBalloc.mGetNUsed() < pBalloc.mGetNTokens()) {
      // failed to find a suitable split
      break;
    }

    auto lSinfosMla = vKvMla->mPrepare(lUbatches);
    if (lSinfosMla.empty()) {
      break;
    }

    auto lSinfosLid = vKvLid->mPrepare(lUbatches);
    if (lSinfosLid.empty()) {
      break;
    }

    assert(sinfos_mla.size() == sinfos_lid.size());

    return std::make_unique<dgs_kv_cache_dsa_context>(this, std::move(lSinfosMla), std::move(lSinfosLid),
                                                      std::move(lUbatches));
  } while (false);

  return std::make_unique<dgs_kv_cache_dsa_context>(cDgsMemoryStatusFailedPrepare);
}

dgs_memory_context_ptr dgs_kv_cache_dsa::mInitFull() {
  return std::make_unique<dgs_kv_cache_dsa_context>(this);
}

dgs_memory_context_ptr dgs_kv_cache_dsa::mInitUpdate(dgs_context * pLctx, bool pOptimize) {
  return std::make_unique<dgs_kv_cache_dsa_context>(this, pLctx, pOptimize);
}

bool dgs_kv_cache_dsa::mGetCanShift() const {
  return vKvMla->mGetCanShift() && vKvLid->mGetCanShift() && vKvMla->mGetSize() == vKvLid->mGetSize();
}

void dgs_kv_cache_dsa::mStateWrite(dgs_io_write_i & pIo, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags) const {
  vKvMla->mStateWrite(pIo, pSeqId, pFlags);
  vKvLid->mStateWrite(pIo, pSeqId, pFlags);
}

void dgs_kv_cache_dsa::mStateRead(dgs_io_read_i & pIo, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags) {
  vKvMla->mStateRead(pIo, pSeqId, pFlags);

  try {
    vKvLid->mStateRead(pIo, pSeqId, pFlags);
  } catch (...) {
    // the MLA part is already restored - undo it, so that a failed restore leaves nothing behind
    vKvMla->mStateClear(pSeqId);

    throw;
  }
}

dgs_kv_cache * dgs_kv_cache_dsa::mGetMla() const {
  return vKvMla.get();
}

dgs_kv_cache * dgs_kv_cache_dsa::mGetLid() const {
  return vKvLid.get();
}

//
// dgs_kv_cache_dsa_context
//

dgs_kv_cache_dsa_context::dgs_kv_cache_dsa_context(dgs_memory_status pStatus) : cStatus(pStatus) {}

dgs_kv_cache_dsa_context::dgs_kv_cache_dsa_context(dgs_kv_cache_dsa * pKv) :
  cCtxMla(pKv->mGetMla()->mInitFull()),
  cCtxLid(pKv->mGetLid()->mInitFull()),
  cStatus(fDgsMemoryStatusCombine(cCtxMla->mGetStatus(), cCtxLid->mGetStatus())) {}

dgs_kv_cache_dsa_context::dgs_kv_cache_dsa_context(dgs_kv_cache_dsa * pKv, dgs_context * pLctx, bool pOptimize) :
  cCtxMla(pKv->mGetMla()->mInitUpdate(pLctx, pOptimize)),
  cCtxLid(pKv->mGetLid()->mInitUpdate(pLctx, pOptimize)),
  cStatus(fDgsMemoryStatusCombine(cCtxMla->mGetStatus(), cCtxLid->mGetStatus())) {}

dgs_kv_cache_dsa_context::dgs_kv_cache_dsa_context(dgs_kv_cache_dsa *      pKv,
                                                   slot_info_vec_t         pSinfosMla,
                                                   slot_info_vec_t         pSinfosLid,
                                                   std::vector<dgs_ubatch> pUbatches) :
  lUbatches(std::move(pUbatches)),
  // note: here we copy the ubatches. not sure if this is ideal
  cCtxMla(new dgs_kv_cache_context(pKv->mGetMla(), std::move(pSinfosMla), this->lUbatches)),
  cCtxLid(new dgs_kv_cache_context(pKv->mGetLid(), std::move(pSinfosLid), this->lUbatches)),
  cStatus(fDgsMemoryStatusCombine(cCtxMla->mGetStatus(), cCtxLid->mGetStatus())) {}

dgs_kv_cache_dsa_context::~dgs_kv_cache_dsa_context() = default;

bool dgs_kv_cache_dsa_context::mNext() {
  assert(status == cDgsMemoryStatusSuccess);

  cCtxMla->mNext();
  cCtxLid->mNext();

  if (++vINext >= lUbatches.size()) {
    return false;
  }

  return true;
}

bool dgs_kv_cache_dsa_context::apply() {
  assert(!fDgsMemoryStatusIsFail(status));

  bool vRes = true;

  vRes = vRes & cCtxMla->apply();
  vRes = vRes & cCtxLid->apply();

  return vRes;
}

dgs_memory_status dgs_kv_cache_dsa_context::mGetStatus() const {
  return cStatus;
}

const dgs_ubatch & dgs_kv_cache_dsa_context::mGetUbatch() const {
  assert(status == cDgsMemoryStatusSuccess);

  return lUbatches[vINext];
}

const dgs_kv_cache_context * dgs_kv_cache_dsa_context::mGetMla() const {
  assert(status == cDgsMemoryStatusSuccess);

  return static_cast<const dgs_kv_cache_context *>(cCtxMla.get());
}

const dgs_kv_cache_context * dgs_kv_cache_dsa_context::mGetLid() const {
  assert(status == cDgsMemoryStatusSuccess);

  return static_cast<const dgs_kv_cache_context *>(cCtxLid.get());
}
