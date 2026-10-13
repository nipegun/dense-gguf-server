#include "dgs-kv-cache-msa.h"

#include "dgs-batch.h"
#include "dgs-impl.h"
#include "dgs-model.h"

#include <algorithm>
#include <cassert>
#include <cmath>

// dgs_kv_cache_msa

dgs_kv_cache_msa::dgs_kv_cache_msa(const dgs_model &       pModel,
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
                                   const layer_filter_cb & pFilter,
                                   const layer_filter_cb & pFilterIdx,
                                   const layer_reuse_cb &  pReuse) :
  vHparamsIdx(pModel.hparams),
  cNStream(pUnified ? 1 : pNSeqMax),
  cNSeqMax(pNSeqMax),
  cNPad(pNPad),
  cNSwa(pNSwa),
  cSwaType(pSwaType) {
  DGS_LOG_INFO("%s: creating main KV cache, size = %u cells\n", __func__, pKvSize);

  vKvBase = std::make_unique<dgs_kv_cache>(pModel, pModel.hparams, pTypeK, pTypeV, pVTrans, pOffload, pUnified, pKvSize,
                                           pNSeqMax, pNPad, pNSwa, pSwaType, nullptr, pFilter, pReuse, nullptr);

  // the MSA indexer uses a single key head per layer
  std::fill(vHparamsIdx.aNHeadKvArr.begin(), vHparamsIdx.aNHeadKvArr.end(), 1);
  vHparamsIdx.vNEmbdHeadKFull = pModel.hparams.vIndexerHeadSize;
  // the rope parameters are kept identical to the main cache

  DGS_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, pKvSize);

  vKvIdx = std::make_unique<dgs_kv_cache>(pModel, vHparamsIdx, pTypeK, pTypeV, pVTrans, pOffload, pUnified, pKvSize,
                                          pNSeqMax, pNPad, pNSwa, pSwaType, nullptr, pFilterIdx, pReuse, nullptr);
}

void dgs_kv_cache_msa::clear(bool data) {
  vKvBase->clear(data);
  vKvIdx->clear(data);
}

bool dgs_kv_cache_msa::mSeqRm(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1) {
  bool vRes = true;

  vRes = vRes & vKvBase->mSeqRm(pSeqId, pP0, pP1);
  vRes = vRes & vKvIdx->mSeqRm(pSeqId, pP0, pP1);

  return vRes;
}

void dgs_kv_cache_msa::mSeqCp(dgs_seq_id pSeqIdSrc, dgs_seq_id pSeqIdDst, dgs_pos pP0, dgs_pos pP1) {
  vKvBase->mSeqCp(pSeqIdSrc, pSeqIdDst, pP0, pP1);
  vKvIdx->mSeqCp(pSeqIdSrc, pSeqIdDst, pP0, pP1);
}

void dgs_kv_cache_msa::mSeqKeep(dgs_seq_id pSeqId) {
  vKvBase->mSeqKeep(pSeqId);
  vKvIdx->mSeqKeep(pSeqId);
}

void dgs_kv_cache_msa::mSeqAdd(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1, dgs_pos pShift) {
  vKvBase->mSeqAdd(pSeqId, pP0, pP1, pShift);
  vKvIdx->mSeqAdd(pSeqId, pP0, pP1, pShift);
}

void dgs_kv_cache_msa::mSeqDiv(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1, int pD) {
  vKvBase->mSeqDiv(pSeqId, pP0, pP1, pD);
  vKvIdx->mSeqDiv(pSeqId, pP0, pP1, pD);
}

dgs_pos dgs_kv_cache_msa::mSeqPosMin(dgs_seq_id pSeqId) const {
  return vKvBase->mSeqPosMin(pSeqId);
}

dgs_pos dgs_kv_cache_msa::mSeqPosMax(dgs_seq_id pSeqId) const {
  return vKvBase->mSeqPosMax(pSeqId);
}

std::map<ggml_backend_buffer_type_t, size_t> dgs_kv_cache_msa::mMemoryBreakdown() const {
  std::map<ggml_backend_buffer_type_t, size_t> dMb = vKvBase->mMemoryBreakdown();
  for (const auto & vBuftSize : vKvIdx->mMemoryBreakdown()) {
    dMb[vBuftSize.first] += vBuftSize.second;
  }
  return dMb;
}

dgs_memory_context_ptr dgs_kv_cache_msa::mInitBatch(dgs_batch_allocr & pBalloc, uint32_t pNUbatch, bool pEmbdAll) {
  GGML_UNUSED(pEmbdAll);

  do {
    pBalloc.mSplitReset();

    std::vector<dgs_ubatch> lUbatches;
    while (true) {
      auto vUbatch = cNStream == 1 ? pBalloc.mSplitSimple(pNUbatch) : pBalloc.mSplitEqual(pNUbatch, true, 0);

      if (vUbatch.vNTokens == 0) {
        break;
      }

      lUbatches.push_back(std::move(vUbatch));
    }

    if (pBalloc.mGetNUsed() < pBalloc.mGetNTokens()) {
      // failed to find a suitable split
      break;
    }

    auto lSinfosBase = vKvBase->mPrepare(lUbatches);
    if (lSinfosBase.empty()) {
      break;
    }

    auto lSinfosIdx = vKvIdx->mPrepare(lUbatches);
    if (lSinfosIdx.empty()) {
      break;
    }

    assert(sinfos_base.size() == sinfos_idx.size());

    return std::make_unique<dgs_kv_cache_msa_context>(this, std::move(lSinfosBase), std::move(lSinfosIdx),
                                                      std::move(lUbatches));
  } while (false);

  return std::make_unique<dgs_kv_cache_msa_context>(cDgsMemoryStatusFailedPrepare);
}

dgs_memory_context_ptr dgs_kv_cache_msa::mInitFull() {
  return std::make_unique<dgs_kv_cache_msa_context>(this);
}

dgs_memory_context_ptr dgs_kv_cache_msa::mInitUpdate(dgs_context * pLctx, bool pOptimize) {
  return std::make_unique<dgs_kv_cache_msa_context>(this, pLctx, pOptimize);
}

bool dgs_kv_cache_msa::mGetCanShift() const {
  return vKvBase->mGetCanShift() && vKvIdx->mGetCanShift() && vKvBase->mGetSize() == vKvIdx->mGetSize();
}

void dgs_kv_cache_msa::mStateWrite(dgs_io_write_i & pIo, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags) const {
  vKvBase->mStateWrite(pIo, pSeqId, pFlags);
  vKvIdx->mStateWrite(pIo, pSeqId, pFlags);
}

void dgs_kv_cache_msa::mStateRead(dgs_io_read_i & pIo, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags) {
  vKvBase->mStateRead(pIo, pSeqId, pFlags);
  vKvIdx->mStateRead(pIo, pSeqId, pFlags);
}

dgs_kv_cache * dgs_kv_cache_msa::mGetBase() const {
  return vKvBase.get();
}

dgs_kv_cache * dgs_kv_cache_msa::mGetIdx() const {
  return vKvIdx.get();
}

// dgs_kv_cache_msa_context

dgs_kv_cache_msa_context::dgs_kv_cache_msa_context(dgs_memory_status pStatus) : vKv(nullptr), cStatus(pStatus) {}

dgs_kv_cache_msa_context::dgs_kv_cache_msa_context(dgs_kv_cache_msa * pKv) :
  vKv(pKv),
  cCtxBase(pKv->mGetBase()->mInitFull()),
  cCtxIdx(pKv->mGetIdx()->mInitFull()),
  cStatus(fDgsMemoryStatusCombine(cCtxBase->mGetStatus(), cCtxIdx->mGetStatus())) {}

dgs_kv_cache_msa_context::dgs_kv_cache_msa_context(dgs_kv_cache_msa * pKv, dgs_context * pLctx, bool pOptimize) :
  vKv(pKv),
  cCtxBase(pKv->mGetBase()->mInitUpdate(pLctx, pOptimize)),
  cCtxIdx(pKv->mGetIdx()->mInitUpdate(pLctx, pOptimize)),
  cStatus(fDgsMemoryStatusCombine(cCtxBase->mGetStatus(), cCtxIdx->mGetStatus())) {}

dgs_kv_cache_msa_context::dgs_kv_cache_msa_context(dgs_kv_cache_msa *      pKv,
                                                   slot_info_vec_t         pSinfosBase,
                                                   slot_info_vec_t         pSinfosIdx,
                                                   std::vector<dgs_ubatch> pUbatches) :
  vKv(pKv),
  lUbatches(std::move(pUbatches)),
  // here we copy the ubatches. not sure if this is ideal
  cCtxBase(new dgs_kv_cache_context(pKv->mGetBase(), std::move(pSinfosBase), this->lUbatches)),
  cCtxIdx(new dgs_kv_cache_context(pKv->mGetIdx(), std::move(pSinfosIdx), this->lUbatches)),
  cStatus(fDgsMemoryStatusCombine(cCtxBase->mGetStatus(), cCtxIdx->mGetStatus())) {}

dgs_kv_cache_msa_context::~dgs_kv_cache_msa_context() = default;

bool dgs_kv_cache_msa_context::mNext() {
  assert(status == cDgsMemoryStatusSuccess);

  cCtxBase->mNext();
  cCtxIdx->mNext();

  if (++vINext >= lUbatches.size()) {
    return false;
  }

  return true;
}

bool dgs_kv_cache_msa_context::apply() {
  assert(!fDgsMemoryStatusIsFail(status));

  bool vRes = true;

  vRes = vRes & cCtxBase->apply();
  vRes = vRes & cCtxIdx->apply();

  return vRes;
}

dgs_memory_status dgs_kv_cache_msa_context::mGetStatus() const {
  return cStatus;
}

const dgs_ubatch & dgs_kv_cache_msa_context::mGetUbatch() const {
  assert(status == cDgsMemoryStatusSuccess);

  return lUbatches[vINext];
}

const dgs_kv_cache_context * dgs_kv_cache_msa_context::mGetBase() const {
  assert(status == cDgsMemoryStatusSuccess);

  return static_cast<const dgs_kv_cache_context *>(cCtxBase.get());
}

const dgs_kv_cache_context * dgs_kv_cache_msa_context::mGetIdx() const {
  assert(status == cDgsMemoryStatusSuccess);

  return static_cast<const dgs_kv_cache_context *>(cCtxIdx.get());
}

uint32_t dgs_kv_cache_msa_context::mGetNPos() const {
  // pad the value so that the graph remains constant across batches and can be reused
  const uint32_t cNPadCur = std::max(vKv->mGetNPad(), 256u);

  dgs_pos vPosMax = -1;

  for (dgs_seq_id vSeqId = 0; vSeqId < (dgs_seq_id) vKv->mGetNSeqMax(); ++vSeqId) {
    vPosMax = std::max(vPosMax, vKv->mSeqPosMax(vSeqId));
  }

  return std::max(cNPadCur, GGML_PAD((uint32_t) (vPosMax + 1), cNPadCur));
}

void dgs_kv_cache_msa_context::mSetInputCellPos(ggml_tensor * pDst, const dgs_ubatch * pUbatch, int32_t pDiv) const {
  GGML_ASSERT(ggml_backend_buffer_is_host(pDst->buffer));
  GGML_ASSERT(pDst->type == GGML_TYPE_I32);
  GGML_ASSERT(pDiv > 0);

  const int64_t cNTokens   = pUbatch->vNTokens;
  const int64_t cNKv       = pDst->ne[0];
  const int64_t cNStreamUb = pDst->ne[1];

  GGML_ASSERT(cNTokens % cNStreamUb == 0);
  const int64_t cNTps = cNTokens / cNStreamUb;

  int32_t * data = (int32_t *) pDst->data;

  for (int64_t vS = 0; vS < cNStreamUb; ++vS) {
    const dgs_seq_id cSeqId = pUbatch->vSeqId[vS * cNTps][0];

    const auto & vCells = vKv->mGetBase()->mGetCells(cSeqId);

    for (int64_t vJ = 0; vJ < cNKv; ++vJ) {
      // the value for empty or other-sequence cells is irrelevant as consumers mask them
      data[vS * cNKv + vJ] =
        vCells.mIsEmpty(vJ) || !vCells.mSeqHas(vJ, cSeqId) ? 0 : (int32_t) (vCells.mPosGet(vJ) / pDiv);
    }
  }
}

void dgs_kv_cache_msa_context::mSetInputPosSlot(ggml_tensor * pDst, const dgs_ubatch * pUbatch) const {
  GGML_ASSERT(ggml_backend_buffer_is_host(pDst->buffer));
  GGML_ASSERT(pDst->type == GGML_TYPE_I32 || pDst->type == GGML_TYPE_F32);

  const int64_t cNTokens   = pUbatch->vNTokens;
  const int64_t cNPos      = pDst->ne[0];
  const int64_t cNStreamUb = pDst->ne[1];

  GGML_ASSERT(cNTokens % cNStreamUb == 0);
  const int64_t cNTps = cNTokens / cNStreamUb;

  for (int64_t vS = 0; vS < cNStreamUb; ++vS) {
    const dgs_seq_id cSeqId = pUbatch->vSeqId[vS * cNTps][0];

    const auto & vCells = vKv->mGetBase()->mGetCells(cSeqId);

    std::vector<int32_t> lMap(cNPos, 0);

    for (uint32_t vJ = 0; vJ < vCells.size(); ++vJ) {
      if (vCells.mIsEmpty(vJ) || !vCells.mSeqHas(vJ, cSeqId)) {
        continue;
      }

      const dgs_pos cP0 = vCells.mPosGet(vJ);

      if (cP0 < 0 || cP0 >= cNPos) {
        continue;
      }

      lMap[cP0] = (int32_t) vJ;
    }

    if (pDst->type == GGML_TYPE_I32) {
      int32_t * data = (int32_t *) pDst->data + vS * cNPos;
      std::copy(lMap.begin(), lMap.end(), data);
    } else {
      float * data = (float *) pDst->data + vS * cNPos;
      for (int64_t vP = 0; vP < cNPos; ++vP) {
        data[vP] = (float) lMap[vP];
      }
    }
  }
}

void dgs_kv_cache_msa_context::mSetInputPosMask(ggml_tensor * pDst, const dgs_ubatch * pUbatch) const {
  GGML_ASSERT(ggml_backend_buffer_is_host(pDst->buffer));
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);

  const int64_t cNTokens = pUbatch->vNTokens;
  const int64_t cNPos    = pDst->ne[0];

  GGML_ASSERT(pDst->ne[1] == cNTokens);

  const uint32_t     cNSwa    = vKv->mGetNSwa();
  const dgs_swa_type cSwaType = vKv->mGetSwaType();

  float * data = (float *) pDst->data;

  std::fill(data, data + cNPos * cNTokens, -INFINITY);

  for (int64_t vI = 0; vI < cNTokens; ++vI) {
    const dgs_seq_id cSeqId = pUbatch->vSeqId[vI][0];

    const auto & vCells = vKv->mGetBase()->mGetCells(cSeqId);

    const dgs_pos cP1 = pUbatch->vPos[vI];

    for (uint32_t vJ = 0; vJ < vCells.size(); ++vJ) {
      if (vCells.mIsEmpty(vJ) || !vCells.mSeqHas(vJ, cSeqId)) {
        continue;
      }

      const dgs_pos cP0 = vCells.mPosGet(vJ);

      if (cP0 < 0 || cP0 >= cNPos) {
        continue;
      }

      // causal mask
      if (cP0 > cP1) {
        continue;
      }

      // apply SWA if any
      if (dgs_hparams::mIsMaskedSwa(cNSwa, cSwaType, cP0, cP1)) {
        continue;
      }

      data[vI * cNPos + cP0] = 0.0f;
    }
  }
}
