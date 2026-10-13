#include "dgs-kv-cache.h"

#include "dgs-context.h"
#include "dgs-impl.h"
#include "dgs-io.h"
#include "dgs-model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>

static bool fGgmlIsPowerOf2(int n) {
  return (n & (n - 1)) == 0;
}

// orthonormal Walsh-Hadamard rotation matrix
// note: res^2 == I
static void fGgmlGenHadamard(ggml_tensor * pTensor) {
  assert(tensor->type == GGML_TYPE_F32);

  const int n = pTensor->ne[0];

  assert(ggml_is_power_of_2(n));
  assert(tensor->ne[1] == n);
  assert(tensor->ne[2] == 1);
  assert(tensor->ne[3] == 1);

  std::vector<float> lDataF32;

  float * data = (float *) pTensor->data;

  if (pTensor->type != GGML_TYPE_F32) {
    lDataF32.resize(n * n);
    data = lDataF32.data();
  }

  data[0 * n + 0] = 1.0 / sqrtf(n);

  for (int vS = 1; vS < n; vS *= 2) {
    for (int vI = 0; vI < vS; vI++) {
      for (int vJ = 0; vJ < vS; vJ++) {
        const float cVal = data[vI * n + vJ];

        data[(vI + vS) * n + (vJ)]      = cVal;
        data[(vI) *n + (vJ + vS)]       = cVal;
        data[(vI + vS) * n + (vJ + vS)] = -cVal;
      }
    }
  }

  if (pTensor->type != GGML_TYPE_F32) {
    ggml_quantize_chunk(pTensor->type, data, pTensor->data, 0, 1, n * n, nullptr);
  }
}

//
// dgs_kv_cache
//

dgs_kv_cache::dgs_kv_cache(const dgs_model &       pModel,
                           const dgs_hparams &     pHparams,
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
                           dgs_memory_t            pMemOther,
                           const layer_filter_cb & pFilter,
                           const layer_reuse_cb &  pReuse,
                           const layer_share_cb &  pShare,
                           const char *            pNameTag) :
  vModel(pModel),
  vHparams(pHparams),
  vVTrans(pVTrans),
  cNSeqMax(pNSeqMax),
  cNStream(pUnified ? 1 : pNSeqMax),
  cNPad(pNPad),
  cNSwa(pNSwa),
  cSwaType(pSwaType),
  vOther(static_cast<dgs_kv_cache *>(pMemOther)),
  lVCellsImpl(vOther ? vOther->lVCellsImpl : std::make_shared<dgs_kv_cells_vec>()),
  lVCells(*lVCellsImpl) {
  // shared cells view the source cache's K/V tensors, so the cell count
  // follows the source allocation: a fitted target can be smaller than the
  // draft default and oversized views would overflow the source tensors
  if (vOther) {
    const uint32_t cSizeOther = vOther->mGetSize();
    if (pKvSize != cSizeOther) {
      DGS_LOG_WARN("%s: kv_size = %u overridden to %u to match the shared source cache\n", __func__, pKvSize,
                   cSizeOther);
      pKvSize = cSizeOther;
    }
  }

  GGML_ASSERT(pKvSize % pNPad == 0);

  const uint32_t cNLayer = pHparams.n_layer_all;

  // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
  struct ggml_backend_buft_comparator {
    bool operator()(const ggml_backend_buffer_type_t & pLhs, const ggml_backend_buffer_type_t & pRhs) const {
      return strcmp(ggml_backend_buft_name(pLhs), ggml_backend_buft_name(pRhs)) < 0;
    }
  };

  std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> dCtxMap;

  // create a context for each buffer type
  auto vCtxForBuft = [&](ggml_backend_buffer_type_t pBuft) -> ggml_context * {
    auto vIt = dCtxMap.find(pBuft);
    if (vIt == dCtxMap.end()) {
      ggml_init_params vParams = {
        /*.mem_size   =*/size_t(2u * (1 + cNStream) * cNLayer * ggml_tensor_overhead()),
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
      };

      ggml_context * vCtx = ggml_init(vParams);
      if (!vCtx) {
        return nullptr;
      }

      dCtxMap.emplace(pBuft, vCtx);

      return vCtx;
    }

    return vIt->second.get();
  };

  GGML_ASSERT(cNStream == 1 || cNStream == pNSeqMax);

  lVHeads.resize(cNStream);
  for (uint32_t vS = 0; vS < cNStream; ++vS) {
    lVHeads[vS] = 0;
  }

  lVCells.resize(cNStream);
  for (uint32_t vS = 0; vS < cNStream; ++vS) {
    lVCells[vS].resize(pKvSize);
  }

  // by default, all sequence ids are mapped to the 0th stream
  lSeqToStream.resize(DGS_MAX_SEQ, 0);

  if (cNStream > 1) {
    lSeqToStream.resize(cNStream, 0);
    for (uint32_t vS = 0; vS < cNStream; ++vS) {
      lSeqToStream[vS] = vS;
    }
  }

  // [TAG_V_CACHE_VARIABLE]
  if (pVTrans && pHparams.mIsNEmbdVGqaVariable()) {
    DGS_LOG_WARN(
      "%s: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to %d\n",
      __func__, pHparams.mNEmbdVGqaMax());
  }

  const bool cIsMla = pHparams.mIsMla();

  for (uint32_t vIl = 0; vIl < cNLayer; vIl++) {
    if (!pHparams.mHasKv(vIl)) {
      DGS_LOG_DEBUG("%s: layer %3d: does not have KV cache\n", __func__, vIl);
      continue;
    }

    if (pFilter && !pFilter(vIl)) {
      DGS_LOG_DEBUG("%s: layer %3d: filtered\n", __func__, vIl);
      continue;
    }

    if (pShare && vOther) {
      const int32_t cIlShare = pShare(vIl);

      if (cIlShare >= 0) {
        const auto & vLayerShare = vOther->lLayers[vOther->dMapLayerIds[cIlShare]];

        DGS_LOG_WARN("%s: layer %3d: sharing with layer %d. k = %p, v = %p\n", __func__, vIl, cIlShare,
                     vLayerShare.vK->data, vLayerShare.vV->data);

        dMapLayerIds[vIl] = lLayers.size();

        lLayers.push_back(vLayerShare);
        lLayers.back().vIl = vIl;

        continue;
      }
    }

    if (vNEmbdHeadKAll == 0) {
      vNEmbdHeadKAll = (int32_t) pHparams.n_embd_head_k(vIl);
    } else if (vNEmbdHeadKAll > 0 && vNEmbdHeadKAll != (int32_t) pHparams.n_embd_head_k(vIl)) {
      vNEmbdHeadKAll = -1;
    }

    if (!cIsMla) {
      if (vNEmbdHeadVAll == 0) {
        vNEmbdHeadVAll = (int32_t) pHparams.n_embd_head_v(vIl);
      } else if (vNEmbdHeadVAll > 0 && vNEmbdHeadVAll != (int32_t) pHparams.n_embd_head_v(vIl)) {
        vNEmbdHeadVAll = -1;
      }
    }

    // [TAG_V_CACHE_VARIABLE]
    const uint32_t cNEmbdKGqa = pHparams.n_embd_k_gqa(vIl);
    const uint32_t cNEmbdVGqa = !pVTrans ? pHparams.n_embd_v_gqa(vIl) : pHparams.mNEmbdVGqaMax();

    const char * vDevName = "CPU";

    ggml_backend_buffer_type_t vBuft = ggml_backend_cpu_buffer_type();

    if (pOffload) {
      auto * vDev = pModel.mDevLayer(vIl);
      vBuft       = ggml_backend_dev_buffer_type(vDev);

      vDevName = ggml_backend_dev_name(vDev);
    }

    DGS_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, vIl, vDevName);

    ggml_context * vCtx = vCtxForBuft(vBuft);
    if (!vCtx) {
      throw std::runtime_error("failed to create ggml context for kv cache");
    }

    const bool cHasK = true;
    const bool cHasV = !cIsMla;

    ggml_tensor * vK = cHasK ? ggml_new_tensor_3d(vCtx, pTypeK, cNEmbdKGqa, pKvSize, cNStream) : nullptr;
    ggml_tensor * vV = cHasV ? ggml_new_tensor_3d(vCtx, pTypeV, cNEmbdVGqa, pKvSize, cNStream) : nullptr;

    cHasK && ggml_format_name(vK, "cache_%sk_l%d", pNameTag, vIl);
    cHasV && ggml_format_name(vV, "cache_%sv_l%d", pNameTag, vIl);

    std::vector<ggml_tensor *> lKStream;
    std::vector<ggml_tensor *> lVStream;

    for (uint32_t vS = 0; vS < cNStream; ++vS) {
      lKStream.push_back(cHasK ? ggml_view_2d(vCtx, vK, cNEmbdKGqa, pKvSize, vK->nb[1], vS * vK->nb[2]) : nullptr);
      lVStream.push_back(cHasV ? ggml_view_2d(vCtx, vV, cNEmbdVGqa, pKvSize, vV->nb[1], vS * vV->nb[2]) : nullptr);
    }

    dMapLayerIds[vIl] = lLayers.size();

    lLayers.push_back({
      vIl,
      vK,
      vV,
      lKStream,
      lVStream,
    });
  }

  if (pReuse) {
    DGS_LOG_DEBUG("%s: reusing layers:\n", __func__);

    for (uint32_t vIl = 0; vIl < cNLayer; vIl++) {
      const int32_t cIlReuse = pReuse(vIl);

      if (cIlReuse < 0) {
        DGS_LOG_DEBUG("%s: - layer %3d: no reuse\n", __func__, vIl);
        continue;
      }

      if (pFilter && !pFilter(vIl)) {
        DGS_LOG_DEBUG("%s: - layer %3d: filtered\n", __func__, vIl);
        continue;
      }

      GGML_ASSERT(dMapLayerIds.find(cIlReuse) != dMapLayerIds.end());

      dMapLayerIds[vIl] = dMapLayerIds[cIlReuse];

      DGS_LOG_DEBUG("%s: - layer %3d: reuse layer %d, is_swa = %d\n", __func__, vIl, cIlReuse, pHparams.mIsSwa(vIl));
    }
  }

  // allocate tensors and initialize the buffers to avoid NaNs in the padding
  for (auto & [buft, ctx] : dCtxMap) {
    ggml_backend_buffer_t vBuf;
    if (pHparams.vNoAlloc) {
      vBuf = ggml_backend_buft_alloc_buffer(buft, /*size =*/0);  // dummy buffer
      for (ggml_tensor * vT = ggml_get_first_tensor(ctx.get()); vT != nullptr;
           vT               = ggml_get_next_tensor(ctx.get(), vT)) {
        vT->buffer = vBuf;  // set dummy buffer for KV cache so that the backend scheduler won't try to allocate it
      }
    } else {
      vBuf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);  // real buffer
    }
    if (!vBuf) {
      throw std::runtime_error("failed to allocate buffer for kv cache");
    }

    DGS_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(vBuf),
                 ggml_backend_buffer_get_size(vBuf) / 1024.0 / 1024.0);

    ggml_backend_buffer_clear(vBuf, 0);
    lCtxsBufs.emplace_back(std::move(ctx), vBuf);
  }

  {
    const size_t cMemorySizeK = mSizeKBytes();
    const size_t cMemorySizeV = mSizeVBytes();

    DGS_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), K (%s): %7.2f MiB, V (%s): %7.2f MiB\n",
                 __func__, (float) (cMemorySizeK + cMemorySizeV) / (1024.0f * 1024.0f), pKvSize, (int) lLayers.size(),
                 pNSeqMax, cNStream, ggml_type_name(pTypeK), (float) cMemorySizeK / (1024.0f * 1024.0f),
                 ggml_type_name(pTypeV), (float) cMemorySizeV / (1024.0f * 1024.0f));
  }

  // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
  if (vOther) {
    vNEmbdHeadKAll = vOther->vNEmbdHeadKAll;
    vNEmbdHeadVAll = vOther->vNEmbdHeadVAll;

    vNRotK = vOther->vNRotK;
    vNRotV = vOther->vNRotV;
  } else {
    const char * vDgsAttnRotDisable = getenv("DGS_ATTN_ROT_DISABLE");
    const bool   cAttnRotDisable    = vDgsAttnRotDisable ? atoi(vDgsAttnRotDisable) : false;
    if (cAttnRotDisable) {
      DGS_LOG_WARN("%s: attention rotation force disabled (DGS_ATTN_ROT_DISABLE)\n", __func__);
    }

    bool vAttnRotK =
      !cAttnRotDisable && vNEmbdHeadKAll > 0 && ggml_is_quantized(pTypeK) && pHparams.n_embd_head_k() % 64 == 0;

    // always create Hadamard rotation tensors for DeepSeek lightning indexers
    if ((pModel.vArch == cLlmArchDeepseek32 || pModel.vArch == cLlmArchDeepseek4 || pModel.vArch == cLlmArchGlmDsa ||
         pModel.vArch == cLlmArchDots3note) &&
        pHparams.vNEmbdHeadKFull == pHparams.vIndexerHeadSize) {
      vAttnRotK = true;
    }

    const bool cAttnRotV =
      !cAttnRotDisable && vNEmbdHeadVAll > 0 && ggml_is_quantized(pTypeV) && pHparams.n_embd_head_v() % 64 == 0;

    if (vAttnRotK) {
      vNRotK = 64;

      // TODO: investigate if using the smallest rotation matrix is beneficial also for K (similar as for V)
      while (vNEmbdHeadKAll % (2 * vNRotK) == 0) {
        vNRotK *= 2;
      }
    }
    if (cAttnRotV) {
      vNRotV = 64;
      // using smaller rotation matrices for V seems beneficial
      //while (hparams.n_embd_head_v() % (2*n_rot_v) == 0) {
      //    n_rot_v *= 2;
      //}
    }
  }

  DGS_LOG_INFO("%s: n_rot_k = %u, n_embd_head_k_all = %d\n", __func__, vNRotK, vNEmbdHeadKAll);
  DGS_LOG_INFO("%s: n_rot_v = %u, n_embd_head_k_all = %d\n", __func__, vNRotV, vNEmbdHeadVAll);

  // pre-compute the haramard matrices and keep them in host memory
  // TODO: in the future, we can make copies in the backend buffers to avoid host -> device transfers
  if (vNRotK || vNRotV) {
    for (int64_t n = 64; n <= std::max(vNEmbdHeadKAll, vNEmbdHeadVAll); n *= 2) {
      dAttnRotHadamard[n] = std::vector<float>(n * n);

      ggml_init_params vParams = {
        /* .mem_size   = */ 1 * ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
      };

      ggml_context_ptr vCtx{ ggml_init(vParams) };

      ggml_tensor * vTmp = ggml_new_tensor_2d(vCtx.get(), GGML_TYPE_F32, n, n);
      vTmp->data         = dAttnRotHadamard[n].data();

      fGgmlGenHadamard(vTmp);
    }
  }

  const char * vDgsKvCacheDebug = getenv("DGS_KV_CACHE_DEBUG");
  vDebug                        = vDgsKvCacheDebug ? atoi(vDgsKvCacheDebug) : 0;
}

void dgs_kv_cache::clear(bool data) {
  for (uint32_t vS = 0; vS < cNStream; ++vS) {
    lVCells[vS].reset();
    lVHeads[vS] = 0;
  }

  if (data) {
    for (auto & [_, buf] : lCtxsBufs) {
      ggml_backend_buffer_clear(buf.get(), 0);
    }
  }
}

bool dgs_kv_cache::mSeqRm(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1) {
  // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
  if (vOther) {
    return true;
  }

  // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_DGS_SEQ_ID_NEG]
  GGML_ASSERT(pSeqId == -1 || (pSeqId >= 0 && (size_t) pSeqId < lSeqToStream.size()));

  if (pP0 < 0) {
    pP0 = 0;
  }

  if (pP1 < 0) {
    pP1 = std::numeric_limits<dgs_pos>::max();
  }

  if (pSeqId >= 0) {
    auto & vCells = lVCells[lSeqToStream[pSeqId]];
    auto & vHead  = lVHeads[lSeqToStream[pSeqId]];

    uint32_t vNewHead = vCells.size();

    for (uint32_t vI = 0; vI < vCells.size(); ++vI) {
      if (!vCells.mPosIn(vI, pP0, pP1)) {
        continue;
      }

      if (vCells.mSeqHas(vI, pSeqId) && vCells.mSeqRm(vI, pSeqId)) {
        if (vNewHead == vCells.size()) {
          vNewHead = vI;
        }
      }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (vNewHead != vCells.size() && vNewHead < vHead) {
      vHead = vNewHead;
    }
  } else {
    // match any sequence
    for (uint32_t vS = 0; vS < cNStream; ++vS) {
      auto & vCells = lVCells[vS];
      auto & vHead  = lVHeads[vS];

      uint32_t vNewHead = vCells.size();

      for (uint32_t vI = 0; vI < vCells.size(); ++vI) {
        if (!vCells.mPosIn(vI, pP0, pP1)) {
          continue;
        }

        vCells.mRm(vI);

        if (vNewHead == vCells.size()) {
          vNewHead = vI;
        }
      }

      // If we freed up a slot, set head to it so searching can start there.
      if (vNewHead != vCells.size() && vNewHead < vHead) {
        vHead = vNewHead;
      }
    }
  }

  return true;
}

void dgs_kv_cache::mSeqCp(dgs_seq_id pSeqIdSrc, dgs_seq_id pSeqIdDst, dgs_pos pP0, dgs_pos pP1) {
  // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
  if (vOther) {
    return;
  }

  GGML_ASSERT(pSeqIdSrc >= 0 && (size_t) pSeqIdSrc < lSeqToStream.size());
  GGML_ASSERT(pSeqIdDst >= 0 && (size_t) pSeqIdDst < lSeqToStream.size());

  const auto cS0 = lSeqToStream[pSeqIdSrc];
  const auto cS1 = lSeqToStream[pSeqIdDst];

  if (cS0 == cS1) {
    // since both sequences are in the same stream, no data copy is necessary
    // we just have to update the cells meta data

    auto & vCells = lVCells[cS0];

    if (pSeqIdSrc == pSeqIdDst) {
      return;
    }

    if (pP0 < 0) {
      pP0 = 0;
    }

    if (pP1 < 0) {
      pP1 = std::numeric_limits<dgs_pos>::max();
    }

    for (uint32_t vI = 0; vI < vCells.size(); ++vI) {
      if (!vCells.mPosIn(vI, pP0, pP1)) {
        continue;
      }

      if (vCells.mSeqHas(vI, pSeqIdSrc)) {
        vCells.mSeqAdd(vI, pSeqIdDst);
      }
    }

    return;
  }

  // cross-stream sequence copies require to copy the actual buffer data

  bool vIsFull = true;

  if (pP0 > 0 && pP0 + 1 < (int) mGetSize()) {
    vIsFull = false;
  }

  if (pP1 > 0 && pP1 + 1 < (int) mGetSize()) {
    vIsFull = false;
  }

  GGML_ASSERT(vIsFull && "seq_cp() is only supported for full KV buffers");

  // enqueue the copy operation - the buffer copy will be performed during the next update
  vScInfo.lSsrc.push_back(cS0);
  vScInfo.lSdst.push_back(cS1);

  lVCells[cS1].reset();
  for (uint32_t vI = 0; vI < lVCells[cS0].size(); ++vI) {
    if (lVCells[cS0].mSeqHas(vI, pSeqIdSrc)) {
      dgs_pos vPos   = lVCells[cS0].mPosGet(vI);
      dgs_pos vShift = lVCells[cS0].mGetShift(vI);

      dgs_kv_cell_ext vExt = lVCells[cS0].mExtGet(vI);

      if (vShift != 0) {
        vPos -= vShift;
        assert(pos >= 0);
      }

      lVCells[cS1].mPosSet(vI, vPos);
      lVCells[cS1].mSeqAdd(vI, pSeqIdDst);

      if (vShift != 0) {
        lVCells[cS1].mPosAdd(vI, vShift);
      }

      lVCells[cS1].mExtSet(vI, vExt);
    }
  }

  lVHeads[cS1] = lVHeads[cS0];

  //for (uint32_t s = 0; s < n_stream; ++s) {
  //    DGS_LOG_WARN("%s: seq %d: min = %d, max = %d\n", __func__, s, v_cells[s].seq_pos_min(s), v_cells[s].seq_pos_max(s));
  //}
}

void dgs_kv_cache::mSeqKeep(dgs_seq_id pSeqId) {
  // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
  if (vOther) {
    return;
  }

  GGML_ASSERT(pSeqId >= 0 && (size_t) pSeqId < lSeqToStream.size());

  auto & vCells = lVCells[lSeqToStream[pSeqId]];
  auto & vHead  = lVHeads[lSeqToStream[pSeqId]];

  uint32_t vNewHead = vCells.size();

  for (uint32_t vI = 0; vI < vCells.size(); ++vI) {
    if (vCells.mSeqKeep(vI, pSeqId)) {
      if (vNewHead == vCells.size()) {
        vNewHead = vI;
      }
    }
  }

  // If we freed up a slot, set head to it so searching can start there.
  if (vNewHead != vCells.size() && vNewHead < vHead) {
    vHead = vNewHead;
  }
}

void dgs_kv_cache::mSeqAdd(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1, dgs_pos pShift) {
  // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
  if (vOther) {
    return;
  }

  GGML_ASSERT(pSeqId >= 0 && (size_t) pSeqId < lSeqToStream.size());
  GGML_ASSERT(vHparams.mNPosPerEmbd() == 1 && "seq_add() is only supported for n_pos_per_embd() == 1");

  auto & vCells = lVCells[lSeqToStream[pSeqId]];
  auto & vHead  = lVHeads[lSeqToStream[pSeqId]];

  if (pShift == 0) {
    return;
  }

  uint32_t vNewHead = vCells.size();

  if (pP0 < 0) {
    pP0 = 0;
  }

  if (pP1 < 0) {
    pP1 = std::numeric_limits<dgs_pos>::max();
  }

  // If there is no range then return early to avoid looping over all cells.
  if (pP0 == pP1) {
    return;
  }

  for (uint32_t vI = 0; vI < vCells.size(); ++vI) {
    if (!vCells.mPosIn(vI, pP0, pP1)) {
      continue;
    }

    if (vCells.mSeqHas(vI, pSeqId)) {
      if (vCells.mPosAdd(vI, pShift)) {
        if (vNewHead == vCells.size()) {
          vNewHead = vI;
        }
      }
    }
  }

  // If we freed up a slot, set head to it so searching can start there.
  // Otherwise we just start the next search from the beginning.
  vHead = vNewHead != vCells.size() ? vNewHead : 0;
}

void dgs_kv_cache::mSeqDiv(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1, int pD) {
  // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
  if (vOther) {
    return;
  }

  GGML_ASSERT(pSeqId >= 0 && (size_t) pSeqId < lSeqToStream.size());
  GGML_ASSERT(vHparams.mNPosPerEmbd() == 1 && "seq_div() is only supported for n_pos_per_embd() == 1");

  auto & vCells = lVCells[lSeqToStream[pSeqId]];

  if (pD == 1) {
    return;
  }

  if (pP0 < 0) {
    pP0 = 0;
  }

  if (pP1 < 0) {
    pP1 = std::numeric_limits<dgs_pos>::max();
  }

  // If there is no range then return early to avoid looping over the cache.
  if (pP0 == pP1) {
    return;
  }

  for (uint32_t vI = 0; vI < vCells.size(); ++vI) {
    if (!vCells.mPosIn(vI, pP0, pP1)) {
      continue;
    }

    if (vCells.mSeqHas(vI, pSeqId)) {
      vCells.mPosDiv(vI, pD);
    }
  }
}

dgs_pos dgs_kv_cache::mSeqPosMin(dgs_seq_id pSeqId) const {
  // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
  if (vOther) {
    return vOther->mSeqPosMin(pSeqId);
  }

  GGML_ASSERT(pSeqId >= 0 && (size_t) pSeqId < lSeqToStream.size());

  const auto & vCells = lVCells[lSeqToStream[pSeqId]];

  return vCells.mSeqPosMin(pSeqId);
}

dgs_pos dgs_kv_cache::mSeqPosMax(dgs_seq_id pSeqId) const {
  // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
  if (vOther) {
    return vOther->mSeqPosMax(pSeqId);
  }

  GGML_ASSERT(pSeqId >= 0 && (size_t) pSeqId < lSeqToStream.size());

  const auto & vCells = lVCells[lSeqToStream[pSeqId]];

  return vCells.mSeqPosMax(pSeqId);
}

std::map<ggml_backend_buffer_type_t, size_t> dgs_kv_cache::mMemoryBreakdown() const {
  std::map<ggml_backend_buffer_type_t, size_t> dRet;
  for (const auto & [ctx, buf] : lCtxsBufs) {
    ggml_backend_buffer_type_t vBuft = ggml_backend_buffer_get_type(buf.get());

    if (vHparams.vNoAlloc) {
      GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) == nullptr);
      dRet[vBuft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), vBuft);
    } else {
      // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
      dRet[vBuft] += ggml_backend_buffer_get_size(buf.get());
    }
  }

  return dRet;
}

dgs_memory_context_ptr dgs_kv_cache::mInitBatch(dgs_batch_allocr & pBalloc, uint32_t pNUbatch, bool pEmbdAll) {
  do {
    pBalloc.mSplitReset();

    std::vector<dgs_ubatch> lUbatches;
    while (true) {
      auto vUbatch = cNStream == 1 ? pBalloc.mSplitSimple(pNUbatch) :
                     pEmbdAll      ? pBalloc.mSplitSeq(pNUbatch) :
                                     pBalloc.mSplitEqual(pNUbatch, true, 0);

      if (vUbatch.vNTokens == 0) {
        break;
      }

      lUbatches.push_back(std::move(vUbatch));  // NOLINT
    }

    if (pBalloc.mGetNUsed() < pBalloc.mGetNTokens()) {
      // failed to find a suitable split
      break;
    }

    auto lSinfos = mPrepare(lUbatches);
    if (lSinfos.empty()) {
      break;
    }

    return std::make_unique<dgs_kv_cache_context>(this, std::move(lSinfos), std::move(lUbatches));
  } while (false);

  return std::make_unique<dgs_kv_cache_context>(cDgsMemoryStatusFailedPrepare);
}

dgs_memory_context_ptr dgs_kv_cache::mInitFull() {
  return std::make_unique<dgs_kv_cache_context>(this);
}

dgs_memory_context_ptr dgs_kv_cache::mInitUpdate(dgs_context * pLctx, bool pOptimize) {
  GGML_UNUSED(pOptimize);

  bool vDoShift = mGetHasShift();

  return std::make_unique<dgs_kv_cache_context>(this, pLctx, vDoShift, std::move(vScInfo));
}

dgs_kv_cache::slot_info_vec_t dgs_kv_cache::mPrepare(const std::vector<dgs_ubatch> & pUbatches) {
  dgs_kv_cache::slot_info_vec_t lRes;

  struct state_t {
    slot_info vSinfo;                   // slot info for the ubatch

    std::vector<uint32_t> lVHeadsOld;   // old positions of the heads, before placing the ubatch

    std::vector<dgs_kv_cells> lVCells;  // copy of the old cells, before placing the ubatch
  };

  // remember the old state of the cells so we can restore it in the end
  std::vector<state_t> lStates;

  bool vSuccess = true;

  for (const auto & vUbatch : pUbatches) {
    // only find a suitable slot for the ubatch. don't modify the cells yet
    const auto cSinfoNew = mFindSlot(vUbatch, false);
    if (cSinfoNew.empty()) {
      vSuccess = false;
      break;
    }

    // remember the position that we found
    lRes.push_back(cSinfoNew);

    // store the old state of the cells in the recovery stack
    {
      state_t vState = { cSinfoNew, lVHeads, {} };

      for (uint32_t vS = 0; vS < cSinfoNew.mNStream(); ++vS) {
        auto & vCells = lVCells[cSinfoNew.lStrm[vS]];

        vState.lVCells.push_back(vCells.mCp(cSinfoNew.lIdxs[vS]));
      }

      lStates.push_back(std::move(vState));
    }

    // now emplace the ubatch
    mApplyUbatch(cSinfoNew, vUbatch);
  }

  GGML_ASSERT(!lStates.empty() || !vSuccess);

  // iterate backwards and restore the cells to their original state
  for (auto lIt = lStates.rbegin(); lIt != lStates.rend(); ++lIt) {
    const auto & vSinfo = lIt->vSinfo;

    for (uint32_t vS = 0; vS < vSinfo.mNStream(); ++vS) {
      auto & vCells = lVCells[vSinfo.lStrm[vS]];
      auto & vHead  = lVHeads[vSinfo.lStrm[vS]];

      vCells.set(vSinfo.lIdxs[vS], lIt->lVCells[vS]);
      vHead = lIt->lVHeadsOld[vS];
    }
  }

  if (!vSuccess) {
    return {};
  }

  return lRes;
}

bool dgs_kv_cache::mUpdate(dgs_context * pLctx, bool pDoShift, const stream_copy_info & pScInfo) {
  // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
  if (vOther) {
    return true;
  }

  bool vUpdated = false;

  auto * vSched = pLctx->mGetSched();

  if (!pScInfo.empty()) {
    assert(n_stream > 1 && "stream copy should never happen with a single stream");

    fDgsSynchronize(pLctx);

    const size_t cNCopy = pScInfo.lSsrc.size();

    for (size_t vI = 0; vI < cNCopy; ++vI) {
      const auto cSsrc = pScInfo.lSsrc[vI];
      const auto cSdst = pScInfo.lSdst[vI];

      assert(ssrc < n_stream);
      assert(sdst < n_stream);

      DGS_LOG_DEBUG("%s: copying KV buffer: stream %d to stream %d\n", __func__, cSsrc, cSdst);

      assert(ssrc != sdst);

      for (uint32_t vIl = 0; vIl < lLayers.size(); ++vIl) {
        const auto & layer = lLayers[vIl];

        ggml_backend_tensor_copy(layer.lKStream[cSsrc], layer.lKStream[cSdst]);

        if (layer.lVStream[cSsrc]) {
          ggml_backend_tensor_copy(layer.lVStream[cSsrc], layer.lVStream[cSdst]);
        }
      }
    }
  }

  if (pDoShift) {
    if (!mGetCanShift()) {
      GGML_ABORT("The current KV cache / model configuration does not support K-shift");
    }

    DGS_LOG_DEBUG("%s: applying K-shift\n", __func__);

    // apply K-shift if needed
    if (vHparams.vRopeType != cDgsRopeTypeNone) {
      ggml_backend_sched_reset(vSched);

      auto * vRes = pLctx->mGetGfResReserve();

      vRes->reset();

      auto * vGf = mBuildGraphShift(vRes, pLctx);
      if (!ggml_backend_sched_alloc_graph(vSched, vGf)) {
        DGS_LOG_ERROR("%s: failed to allocate compute graph for K-shift\n", __func__);
        return vUpdated;
      }

      vRes->mSetInputs(nullptr);

      if (pLctx->mGraphCompute(vGf, false) != GGML_STATUS_SUCCESS) {
        DGS_LOG_ERROR("%s: failed to compute K-shift\n", __func__);
        return vUpdated;
      }

      vUpdated = true;
    }

    for (uint32_t vS = 0; vS < cNStream; ++vS) {
      auto & vCells = lVCells[vS];

      vCells.mResetShift();
    }
  }

  return vUpdated;
}

dgs_kv_cache::slot_info dgs_kv_cache::mFindSlot(const dgs_ubatch & pUbatch, bool pCont) const {
  if (vDebug > 0) {
    for (uint32_t vS = 0; vS < pUbatch.vNSeqsUnq; ++vS) {
      const auto     cSeqId    = pUbatch.vSeqIdUnq[vS];
      const auto     cStreamId = lSeqToStream[cSeqId];
      const auto &   vCells    = lVCells[cStreamId];
      const uint32_t cHeadCur  = lVHeads[cStreamId];

      DGS_LOG_DEBUG("%s: stream[%d], n = %5d, used = %5d, head = %5d, size = %5d, n_swa = %5d\n", __func__, cStreamId,
                    vCells.mUsedMaxP1(), vCells.mGetUsed(), cHeadCur, mGetSize(), cNSwa);

      if ((vDebug == 2 && cNSwa > 0) || vDebug > 2) {
        std::string vSs;
        for (uint32_t vI = 0; vI < vCells.size(); ++vI) {
          if (vCells.mIsEmpty(vI)) {
            vSs += '.';
          } else {
            assert(cells.seq_count(i) >= 1);

            if (vCells.mSeqCount(vI) == 1) {
              vSs += std::to_string(vCells.mSeqGet(vI));
            } else {
              vSs += 'M';
            }
          }
          if (vI % 256 == 255) {
            vSs += " *";
            vSs += '\n';
          }
        }
        DGS_LOG_DEBUG("\n%s\n", vSs.c_str());
      }

      if ((vDebug == 2 && cNSwa > 0) || vDebug > 2) {
        std::string vSs;
        for (uint32_t vI = 0; vI < vCells.size(); ++vI) {
          std::string vCur;
          if (vCells.mIsEmpty(vI)) {
            vCur = '.';
          } else {
            vCur = std::to_string(vCells.mPosGet(vI));
          }
          const int n = vCur.size();
          for (int vJ = 0; vJ < 5 - n; ++vJ) {
            vCur += ' ';
          }
          vSs += vCur;
          if (vI % 256 == 255) {
            vSs += " *";
          }
          if (vI % 64 == 63) {
            vSs += '\n';
          }
        }
        DGS_LOG_DEBUG("\n%s\n", vSs.c_str());
      }

      for (int vS = 0; vS < DGS_MAX_SEQ; ++vS) {
        if (vCells.mSeqPosMin(vS) < 0) {
          continue;
        }

        DGS_LOG_DEBUG("%s: stream[%d] min[%d] = %5d, max[%d] = %5d\n", __func__, cStreamId, vS, vCells.mSeqPosMin(vS),
                      vS, vCells.mSeqPosMax(vS));
      }
    }
  }

  uint32_t vNTokens = pUbatch.vNTokens;
  uint32_t vNSeqs   = 1;

  if (cNStream > 1) {
    GGML_ASSERT(vNTokens % pUbatch.vNSeqsUnq == 0);

    vNSeqs   = pUbatch.vNSeqsUnq;
    vNTokens = vNTokens / vNSeqs;
  }

  slot_info vRes = {
    /*.s0   =*/DGS_MAX_SEQ,
    /*.s1   =*/0,
    /*.strm =*/{},
    /*.idxs =*/{},
  };

  vRes.resize(vNSeqs);

  for (uint32_t vS = 0; vS < vNSeqs; ++vS) {
    const auto cSeqId = pUbatch.vSeqIdUnq[vS];

    if (cNStream > 1) {
      GGML_ASSERT(pUbatch.vNSeqId[vS * vNTokens] == 1);
      GGML_ASSERT(pUbatch.vSeqId[vS * vNTokens][0] == cSeqId);
    }

    vRes.vS0 = std::min<uint32_t>(vRes.vS0, lSeqToStream[cSeqId]);
    vRes.vS1 = std::max<uint32_t>(vRes.vS1, lSeqToStream[cSeqId]);

    vRes.lStrm[vS] = lSeqToStream[cSeqId];
    vRes.lIdxs[vS].reserve(vNTokens);

    const auto & vCells = lVCells[lSeqToStream[cSeqId]];

    uint32_t vHeadCur = lVHeads[lSeqToStream[cSeqId]];

    // if we have enough unused cells before the current head ->
    //   better to start searching from the beginning of the cache, hoping to fill it
    if (vHeadCur > vCells.mGetUsed() + 2 * vNTokens) {
      vHeadCur = 0;
    }

    if (vNTokens > vCells.size()) {
      DGS_LOG_ERROR("%s: n_tokens = %d > size = %u\n", __func__, vNTokens, vCells.size());
      return {};
    }

    uint32_t vNTested = 0;

    // for continuous slots, we test that all tokens in the ubatch fit, starting from the current head
    // for non-continuous slots, we test the tokens one by one
    const uint32_t cNTest = pCont ? vNTokens : 1;

    while (true) {
      if (vHeadCur + cNTest > vCells.size()) {
        vNTested += vCells.size() - vHeadCur;
        vHeadCur = 0;
        continue;
      }

      for (uint32_t vI = 0; vI < cNTest; vI++) {
        const auto cIdx = vHeadCur;

        vHeadCur++;
        vNTested++;

        //const dgs_pos    pos    = ubatch.pos[i];
        //const dgs_seq_id seq_id = ubatch.seq_id[i][0];

        // can we use this cell? either:
        //  - the cell is empty
        //  - the cell is occupied only by one sequence:
        //    - (disabled) mask causally, if the sequence is the same as the one we are inserting
        //    - mask SWA, using current max pos for that sequence in the cache
        //                always insert in the cell with minimum pos
        bool vCanUse = vCells.mIsEmpty(cIdx);

        if (!vCanUse && vCells.mSeqCount(cIdx) == 1) {
          const dgs_pos cPosCell = vCells.mPosGet(cIdx);

          // (disabled) causal mask
          // note: it's better to purge any "future" tokens beforehand
          //if (cells.seq_has(idx, seq_id)) {
          //    can_use = pos_cell >= pos;
          //}

          if (!vCanUse) {
            const dgs_seq_id cSeqIdCell = vCells.mSeqGet(cIdx);

            // SWA mask
            if (dgs_hparams::mIsMaskedSwa(cNSwa, cSwaType, cPosCell, vCells.mSeqPosMax(cSeqIdCell) + 1)) {
              vCanUse = true;
            }
          }
        }

        if (vCanUse) {
          vRes.lIdxs[vS].push_back(cIdx);
        } else {
          if (pCont) {
            break;
          }
        }
      }

      if (vRes.lIdxs[vS].size() == vNTokens) {
        break;
      }

      if (pCont) {
        vRes.lIdxs[vS].clear();
      }

      if (vNTested >= vCells.size()) {
        //DGS_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
        return {};
      }
    }

    // we didn't find a suitable slot - return empty result
    if (vRes.lIdxs[vS].size() < vNTokens) {
      return {};
    }
  }

  assert(res.s1 >= res.s0);

  return vRes;
}

void dgs_kv_cache::mApplyUbatch(const slot_info & pSinfo, const dgs_ubatch & pUbatch) {
  // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
  if (vOther) {
    return;
  }

  // keep track of the max sequence position that we would overwrite with this ubatch
  // for non-SWA cache, this would be always empty
  dgs_seq_id aSeqPosMaxRm[DGS_MAX_SEQ];
  for (uint32_t vS = 0; vS < DGS_MAX_SEQ; ++vS) {
    aSeqPosMaxRm[vS] = -1;
  }

  assert(ubatch.n_tokens == sinfo.n_stream() * sinfo.size());

  for (uint32_t vS = 0; vS < pSinfo.mNStream(); ++vS) {
    for (uint32_t vIi = 0; vIi < pSinfo.size(); ++vIi) {
      const uint32_t cI = vS * pSinfo.size() + vIi;

      auto & vCells = lVCells[pSinfo.lStrm[vS]];

      const auto cIdx = pSinfo.lIdxs[vS][vIi];

      if (!vCells.mIsEmpty(cIdx)) {
        assert(cells.seq_count(idx) == 1);

        const dgs_seq_id cSeqId = vCells.mSeqGet(cIdx);
        const dgs_pos    cPos   = vCells.mPosGet(cIdx);

        aSeqPosMaxRm[cSeqId] = std::max(aSeqPosMaxRm[cSeqId], cPos);

        vCells.mRm(cIdx);
      }

      vCells.mPosSet(cIdx, pUbatch.vPos[cI]);

      if (pUbatch.mIsPos2d() || pUbatch.token || vHparams.vPleNHeads > 0) {
        dgs_kv_cell_ext vExt;

        if (pUbatch.mIsPos2d()) {
          vExt.vX = pUbatch.vPos[cI + pUbatch.vNTokens * 2];
          vExt.vY = pUbatch.vPos[cI + pUbatch.vNTokens];
        }

        const bool cIsEmbd = !pUbatch.token || (pUbatch.mIsMixed() && pUbatch.type[cI]);

        if (!cIsEmbd) {
          vExt.vTok = pUbatch.token[cI];
        } else if (vHparams.vPleNHeads > 0) {
          // embd batch (multimodal input) has no token ids, need to pad it with the correct ID for PLE layers
          // TODO @ngxson : check if we can do the same as gemma 3n / gemma 4
          vExt.vTok = vHparams.vPleImageTokenId != 0 ? (dgs_token) vHparams.vPleImageTokenId :
                                                       (dgs_token) vHparams.vPleEosTokenId;
        }

        vCells.mExtSet(cIdx, vExt);
      }

      for (int32_t vS = 0; vS < pUbatch.vNSeqId[cI]; vS++) {
        vCells.mSeqAdd(cIdx, pUbatch.vSeqId[cI][vS]);
      }
    }
  }

  // note: we want to preserve the invariant that all positions between [pos_min, pos_max] for each sequence
  //       will be present in the cache. so we have to purge any position which is less than those we would overwrite
  for (uint32_t vS = 0; vS < DGS_MAX_SEQ; ++vS) {
    if (aSeqPosMaxRm[vS] == -1) {
      continue;
    }

    GGML_ASSERT(vS < lSeqToStream.size());

    auto & vCells = lVCells[lSeqToStream[vS]];

    if (vCells.mSeqPosMin(vS) <= aSeqPosMaxRm[vS]) {
      DGS_LOG_DEBUG("%s: purging positions [%d, %d] of sequence %d from KV cache\n", __func__, vCells.mSeqPosMin(vS),
                    aSeqPosMaxRm[vS], vS);

      mSeqRm(vS, vCells.mSeqPosMin(vS), aSeqPosMaxRm[vS] + 1);
    }
  }

  // move the head at the end of the slot
  for (uint32_t vS = 0; vS < pSinfo.mNStream(); ++vS) {
    auto & vHead = lVHeads[pSinfo.lStrm[vS]];

    vHead = pSinfo.lIdxs[vS].back() + 1;
  }
}

bool dgs_kv_cache::mGetCanShift() const {
  // Step35 uses per-layer RoPE dims; K-shift assumes a single global n_rot.
  if (vModel.vArch == cLlmArchStep35) {
    return false;
  }
  if (vHparams.mNPosPerEmbd() > 1) {
    return false;
  }
  return true;
}

uint32_t dgs_kv_cache::mGetSize() const {
  const auto & vCells = lVCells[lSeqToStream[0]];

  return vCells.size();
}

uint32_t dgs_kv_cache::mGetNSeqMax() const {
  return cNSeqMax;
}

uint32_t dgs_kv_cache::mGetNStream() const {
  return cNStream;
}

bool dgs_kv_cache::mGetHasShift() const {
  bool vResult = false;

  for (uint32_t vS = 0; vS < cNStream; ++vS) {
    vResult |= lVCells[vS].mGetHasShift();
  }

  return vResult;
}

ggml_type dgs_kv_cache::mTypeK() const {
  return lLayers[0].vK->type;
}

ggml_type dgs_kv_cache::mTypeV() const {
  return lLayers[0].vV->type;
}

std::vector<uint32_t> dgs_kv_cache::mGetLayerIds() const {
  std::vector<uint32_t> lRes;
  lRes.reserve(lLayers.size());

  for (const auto & layer : lLayers) {
    lRes.push_back(layer.vIl);
  }

  return lRes;
}

ggml_tensor * dgs_kv_cache::mGetKStorage(int32_t pIl) const {
  const int32_t cIkv = dMapLayerIds.at(pIl);

  return lLayers[cIkv].vK;
}

const dgs_kv_cells & dgs_kv_cache::mGetCells(dgs_seq_id pSeqId) const {
  GGML_ASSERT(pSeqId >= 0 && (size_t) pSeqId < lSeqToStream.size());

  return lVCells[lSeqToStream[pSeqId]];
}

uint32_t dgs_kv_cache::mGetStream(dgs_seq_id pSeqId) const {
  GGML_ASSERT(pSeqId >= 0 && (size_t) pSeqId < lSeqToStream.size());

  return lSeqToStream[pSeqId];
}

uint32_t dgs_kv_cache::mGetNKv(const slot_info & pSinfo) const {
  uint32_t vResult = 0;

  // pad the n_kv value so that the graph remains constant across batches and can be reused
  // note: this also helps some backends with performance (f.ex )
  const uint32_t cNPadCur = std::max(cNPad, 256u);

  for (uint32_t vS = 0; vS < pSinfo.mNStream(); ++vS) {
    const auto & vCells = lVCells[pSinfo.lStrm[vS]];

    vResult = std::max(std::min(vCells.size(), std::max(cNPadCur, GGML_PAD(vCells.mUsedMaxP1(), cNPadCur))), vResult);
  }

  return vResult;
}

ggml_tensor * dgs_kv_cache::mGetK(ggml_context * pCtx, int32_t pIl, uint32_t pNKv, const slot_info & pSinfo) const {
  const int32_t cIkv = dMapLayerIds.at(pIl);

  auto * vK = lLayers[cIkv].vK;

  const uint64_t cKvSize    = mGetSize();
  const uint64_t cNEmbdKGqa = vK->ne[0];

  assert(n_embd_k_gqa == hparams.n_embd_k_gqa(il));

  const uint32_t cNs = pSinfo.vS1 - pSinfo.vS0 + 1;

  return ggml_view_4d(pCtx, vK, vHparams.n_embd_head_k(pIl), vHparams.n_head_kv(pIl), pNKv, cNs,
                      ggml_row_size(vK->type, vHparams.n_embd_head_k(pIl)), ggml_row_size(vK->type, cNEmbdKGqa),
                      ggml_row_size(vK->type, cNEmbdKGqa * cKvSize),
                      ggml_row_size(vK->type, cNEmbdKGqa * cKvSize) * pSinfo.vS0);
}

ggml_tensor * dgs_kv_cache::mGetV(ggml_context * pCtx, int32_t pIl, uint32_t pNKv, const slot_info & pSinfo) const {
  const int32_t cIkv = dMapLayerIds.at(pIl);

  auto * vV = lLayers[cIkv].vV;

  const uint64_t cKvSize    = mGetSize();
  const uint64_t cNEmbdVGqa = vV->ne[0];

  // [TAG_V_CACHE_VARIABLE]
  assert(n_embd_v_gqa >= hparams.n_embd_v_gqa(il));

  const uint32_t cNs = pSinfo.vS1 - pSinfo.vS0 + 1;

  if (!vVTrans) {
    // note: v->nb[1] <= v->nb[2]
    return ggml_view_4d(pCtx, vV, vHparams.n_embd_head_v(pIl), vHparams.n_head_kv(pIl), pNKv, cNs,
                        ggml_row_size(vV->type, vHparams.n_embd_head_v(pIl)),  // v->nb[1]
                        ggml_row_size(vV->type, cNEmbdVGqa),                   // v->nb[2]
                        ggml_row_size(vV->type, cNEmbdVGqa * cKvSize),         // v->nb[3]
                        ggml_row_size(vV->type, cNEmbdVGqa * cKvSize) * pSinfo.vS0);
  }

  // note: v->nb[1] > v->nb[2]
  return ggml_view_4d(pCtx, vV, pNKv, vHparams.n_head_kv(pIl), vHparams.n_embd_head_v(pIl), cNs,
                      ggml_row_size(vV->type, cKvSize * vHparams.n_embd_head_v(pIl)),  // v->nb[1]
                      ggml_row_size(vV->type, cKvSize),                                // v->nb[2]
                      ggml_row_size(vV->type, cKvSize * cNEmbdVGqa),                   // v->nb[3]
                      ggml_row_size(vV->type, cKvSize * cNEmbdVGqa) * pSinfo.vS0);
}

ggml_tensor * dgs_kv_cache::mCpyK(ggml_context *    pCtx,
                                  ggml_tensor *     pKCur,
                                  ggml_tensor *     pKIdxs,
                                  int32_t           pIl,
                                  const slot_info & pSinfo) const {
  GGML_UNUSED(pSinfo);

  const int32_t cIkv = dMapLayerIds.at(pIl);

  ggml_tensor * vK = lLayers[cIkv].vK;

  const int64_t cNEmbdHead = pKCur->ne[0];
  const int64_t cNHead     = pKCur->ne[1];
  const int64_t cNTokens   = pKCur->ne[2];

  const int64_t cNEmbdGqa = cNEmbdHead * cNHead;

  // we can merge dims 0 and 1
  // TODO: add ggml helper function for this?
  GGML_ASSERT(ggml_row_size(pKCur->type, cNEmbdHead) == pKCur->nb[1]);

  pKCur = ggml_view_2d(pCtx, pKCur, cNEmbdGqa, cNTokens, pKCur->nb[2], 0);

  const int64_t cNStream = vK->ne[2];

  if (cNStream > 1) {
    const int64_t cKvSize = mGetSize();

    assert(n_embd_gqa == k->ne[0]);
    assert(kv_size == k->ne[1]);

    // merge the buffer across all streams because the idxs are global
    vK = ggml_reshape_2d(pCtx, vK, cNEmbdGqa, cKvSize * cNStream);
  }

  // store the current K values into the cache
  return ggml_set_rows(pCtx, vK, pKCur, pKIdxs);
}

ggml_tensor * dgs_kv_cache::mCpyV(ggml_context *    pCtx,
                                  ggml_tensor *     pVCur,
                                  ggml_tensor *     pVIdxs,
                                  int32_t           pIl,
                                  const slot_info & pSinfo) const {
  GGML_UNUSED(pSinfo);

  const int32_t cIkv = dMapLayerIds.at(pIl);

  auto * vV = lLayers[cIkv].vV;

  const int64_t cNEmbdHead = pVCur->ne[0];
  const int64_t cNHead     = pVCur->ne[1];
  const int64_t cNTokens   = pVCur->ne[2];

  const int64_t cNEmbdGqa = cNEmbdHead * cNHead;

  // we can merge dims 0 and 1
  GGML_ASSERT(ggml_row_size(pVCur->type, cNEmbdHead) == pVCur->nb[1]);

  const int64_t cNStream = vV->ne[2];

  // take this branch when FA is enabled (the V cache is not transposed)
  if (!vVTrans) {
    pVCur = ggml_view_2d(pCtx, pVCur, cNEmbdGqa, cNTokens, pVCur->nb[2], 0);

    if (cNStream > 1) {
      const int64_t cKvSize = mGetSize();

      assert(n_embd_gqa == v->ne[0]);
      assert(kv_size == v->ne[1]);

      // merge the buffer across all streams because the idxs are global
      vV = ggml_reshape_2d(pCtx, vV, cNEmbdGqa, cKvSize * cNStream);
    }

    return ggml_set_rows(pCtx, vV, pVCur, pVIdxs);
  }

  if (ggml_row_size(pVCur->type, cNEmbdGqa) == pVCur->nb[2]) {
    // we can merge dims 0, 1 and 2
    pVCur = ggml_reshape_2d(pCtx, pVCur, cNEmbdGqa, cNTokens);
  } else {
    // otherwise -> make a copy to get contiguous data
    pVCur = ggml_cont_2d(pCtx, pVCur, cNEmbdGqa, cNTokens);
  }

  // [TAG_V_CACHE_VARIABLE]
  if (cNEmbdGqa < vV->ne[0]) {
    pVCur = ggml_pad(pCtx, pVCur, vV->ne[0] - cNEmbdGqa, 0, 0, 0);
  }

  // in this branch the v_idxs are constructed in such a way that each row is a single head element
  ggml_tensor * vVView = ggml_reshape_2d(pCtx, vV, 1, ggml_nelements(vV));

  pVCur = ggml_reshape_2d(pCtx, pVCur, 1, ggml_nelements(pVCur));

  return ggml_set_rows(pCtx, vVView, pVCur, pVIdxs);
}

ggml_tensor * dgs_kv_cache::mBuildInputKIdxs(ggml_context * pCtx, const dgs_ubatch & pUbatch) const {
  const uint32_t cNTokens = pUbatch.vNTokens;

  ggml_tensor * vKIdxs = ggml_new_tensor_1d(pCtx, GGML_TYPE_I64, cNTokens);

  ggml_set_input(vKIdxs);
  ggml_set_name(vKIdxs, "attn_inp_k_idxs");

  return vKIdxs;
}

ggml_tensor * dgs_kv_cache::mBuildInputVIdxs(ggml_context * pCtx, const dgs_ubatch & pUbatch) const {
  const uint32_t cNTokens = pUbatch.vNTokens;

  ggml_tensor * vVIdxs;

  if (!vVTrans) {
    vVIdxs = ggml_new_tensor_1d(pCtx, GGML_TYPE_I64, cNTokens);
  } else {
    vVIdxs = ggml_new_tensor_1d(pCtx, GGML_TYPE_I64, cNTokens * vHparams.mNEmbdVGqaMax());
  }

  ggml_set_input(vVIdxs);
  ggml_set_name(vVIdxs, "attn_inp_v_idxs");

  return vVIdxs;
}

ggml_tensor * dgs_kv_cache::mBuildInputKRot(ggml_context * pCtx) const {
  ggml_tensor * vRes = nullptr;

  if (vNRotK) {
    vRes = ggml_new_tensor_2d(pCtx, GGML_TYPE_F32, vNRotK, vNRotK);
    ggml_set_input(vRes);
    ggml_set_name(vRes, "attn_inp_k_rot");
  }

  return vRes;
}

ggml_tensor * dgs_kv_cache::mBuildInputVRot(ggml_context * pCtx) const {
  ggml_tensor * vRes = nullptr;

  if (vNRotV) {
    vRes = ggml_new_tensor_2d(pCtx, GGML_TYPE_F32, vNRotV, vNRotV);
    ggml_set_input(vRes);
    ggml_set_name(vRes, "attn_inp_v_rot");
  }

  return vRes;
}

void dgs_kv_cache::mSetInputKIdxs(ggml_tensor * pDst, const dgs_ubatch * pUbatch, const slot_info & pSinfo) const {
  const uint32_t cNTokens = pUbatch->vNTokens;
  GGML_ASSERT(cNTokens == (int64_t) pSinfo.size() * pSinfo.mNStream());

  GGML_ASSERT(ggml_backend_buffer_is_host(pDst->buffer));
  int64_t * data = (int64_t *) pDst->data;

  for (uint32_t vS = 0; vS < pSinfo.mNStream(); ++vS) {
    const int64_t cOffs = pSinfo.lStrm[vS] * mGetSize();

    for (uint32_t vI = 0; vI < pSinfo.size(); ++vI) {
      data[vS * pSinfo.size() + vI] = cOffs + pSinfo.lIdxs[vS][vI];
    }
  }
}

void dgs_kv_cache::mSetInputVIdxs(ggml_tensor * pDst, const dgs_ubatch * pUbatch, const slot_info & pSinfo) const {
  const uint32_t cNTokens = pUbatch->vNTokens;
  GGML_ASSERT(cNTokens == (int64_t) pSinfo.size() * pSinfo.mNStream());

  GGML_ASSERT(ggml_backend_buffer_is_host(pDst->buffer));
  int64_t * data = (int64_t *) pDst->data;

  if (!vVTrans) {
    for (uint32_t vS = 0; vS < pSinfo.mNStream(); ++vS) {
      const int64_t cOffs = pSinfo.lStrm[vS] * mGetSize();

      for (uint32_t vI = 0; vI < pSinfo.size(); ++vI) {
        data[vS * pSinfo.size() + vI] = cOffs + pSinfo.lIdxs[vS][vI];
      }
    }
  } else {
    // note: the V cache is transposed when not using flash attention
    const int64_t cKvSize = mGetSize();

    const int64_t cNEmbdVGqa = vHparams.mNEmbdVGqaMax();

    for (uint32_t vS = 0; vS < pSinfo.mNStream(); ++vS) {
      const int64_t cOffs = pSinfo.lStrm[vS] * cKvSize * cNEmbdVGqa;

      for (uint32_t vI = 0; vI < pSinfo.size(); ++vI) {
        for (uint32_t vJ = 0; vJ < cNEmbdVGqa; ++vJ) {
          data[vS * pSinfo.size() * cNEmbdVGqa + vI * cNEmbdVGqa + vJ] = cOffs + vJ * cKvSize + pSinfo.lIdxs[vS][vI];
        }
      }
    }
  }
}

void dgs_kv_cache::mSetInputKShift(ggml_tensor * pDst) const {
  GGML_ASSERT(ggml_backend_buffer_is_host(pDst->buffer));

  int32_t * data = (int32_t *) pDst->data;

  for (uint32_t vS = 0; vS < cNStream; ++vS) {
    const auto & vCells = lVCells[vS];

    for (uint32_t vI = 0; vI < vCells.size(); ++vI) {
      data[vS * vCells.size() + vI] = vCells.mIsEmpty(vI) ? 0 : vCells.mGetShift(vI);
    }
  }
}

struct args_set_input_kq_mask {
  const dgs_hparams & vHparams;
  const dgs_ubatch *  vUbatch;

  const std::vector<dgs_kv_cells> & lVCells;
  const std::vector<uint32_t> &     lSeqToStream;

  uint32_t     vNSwa;
  dgs_swa_type vSwaType;

  int64_t vNKv;
  int64_t vNStream;
  int64_t vNTps;
};

template <typename T, bool causal, bool swa, bool is_2d, bool alibi>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & pArgs, T * data) {
  //const auto & hparams = args.hparams;
  const auto & vUbatch = pArgs.vUbatch;

  const auto & lVCells      = pArgs.lVCells;
  const auto & lSeqToStream = pArgs.lSeqToStream;

  const uint32_t     cNSwa    = pArgs.vNSwa;
  const dgs_swa_type cSwaType = pArgs.vSwaType;

  const int64_t cNKv     = pArgs.vNKv;
  const int64_t cNStream = pArgs.vNStream;
  const int64_t cNTps    = pArgs.vNTps;

  const T cMaskKeep = fDgsCast<T>(0.0f);
  const T cMaskDrop = fDgsCast<T>(-INFINITY);

  // the min position in the batch for each sequence
  dgs_pos aSeqPosMin[DGS_MAX_SEQ];
  std::fill(aSeqPosMin, aSeqPosMin + DGS_MAX_SEQ, INT32_MAX);

  for (uint32_t vI = 0; vI < vUbatch->vNTokens; ++vI) {
    const dgs_seq_id cSeqId = vUbatch->vSeqId[vI][0];

    aSeqPosMin[cSeqId] = std::min(aSeqPosMin[cSeqId], vUbatch->vPos[vI]);
  }

  for (uint32_t vS = 0; vS < cNStream; ++vS) {
    // bookkeeping of the KQ mask cells that could change for other tokens of the same sequence
    std::unordered_map<dgs_seq_id, uint32_t>              dSeqSrct;
    std::unordered_map<dgs_seq_id, std::vector<uint32_t>> dSeqIdxs;

    for (uint32_t vIi = 0; vIi < cNTps; ++vIi) {
      const uint32_t cI = vS * cNTps + vIi;

      const dgs_seq_id cSeqId = vUbatch->vSeqId[cI][0];

      const auto & vCells = lVCells.at(lSeqToStream[cSeqId]);

      dgs_pos       vP0 = -1;
      const dgs_pos cP1 = vUbatch->vPos[cI];

      // for M-RoPE
      const dgs_pos cP1X = is_2d ? vUbatch->vPos[cI + vUbatch->vNTokens * 2] : 0;
      const dgs_pos cP1Y = is_2d ? vUbatch->vPos[cI + vUbatch->vNTokens] : 0;

      const uint64_t cIdst = cNKv * cI;

      // for tokens of the same sequence, the mask is mostly the same, so we can reuse it
      // the only cells that could change are the ones that are with similar positions as the
      //   ones in the batch (i.e. due to causal masking, SWA, etc.)
      // keep track of those cells and shortcut the loop to save time
      // note: this optimization is not compatible with Alibi position encoding
      bool vPrev = false;

      auto & lIdxs = dSeqIdxs[cSeqId];

      if (!alibi) {
        if (dSeqSrct.find(cSeqId) != dSeqSrct.end()) {
          const uint32_t cSrct = dSeqSrct[cSeqId];

          const uint64_t cIdstPrev = cNKv * cSrct;

          std::copy(data + cIdstPrev, data + cIdstPrev + cNKv, data + cIdst);

          vPrev = true;
        } else {
          lIdxs.clear();
          lIdxs.reserve(vUbatch->vNTokens + cNSwa + 32);

          dSeqSrct[cSeqId] = cI;
        }
      }

      for (uint32_t vJj = 0; vJj < cNKv; ++vJj) {
        uint32_t vJ = vJj;

        // we have an exiting mask for this sequence -> update just seq_idxs
        if (!alibi) {
          if (vPrev) {
            if (vJj >= lIdxs.size()) {
              break;
            }

            vJ = lIdxs[vJj];
          }
        }

        if (vCells.mIsEmpty(vJ)) {
          goto skip;
        }

        // mask the token if not the same sequence
        if (!vCells.mSeqHas(vJ, cSeqId)) {
          goto skip;
        }

        vP0 = vCells.mPosGet(vJ);

        if (!alibi) {
          if (!vPrev) {
            // record all cells for which: p0 >= seq_pos_min[seq_id] - n_swa - 32
            if (vP0 + (int32_t) (cNSwa + 32) >= aSeqPosMin[cSeqId]) {
              lIdxs.push_back(vJ);
            }
          }
        }

        if (causal) {
          // mask future tokens
          if (vP0 > cP1) {
            goto skip;
          }

          // M-RoPE causal mask
          if (is_2d) {
            if (vP0 == cP1) {
              const auto & vP0Ext = vCells.mExtGet(vJ);

              if (vP0Ext.mIs2dGt(cP1X, cP1Y)) {
                goto skip;
              }
            }
          }
        }

        // apply SWA if any
        if (swa) {
          // see dgs_non_causal_type
          const bool cInSpan =
            !causal && pArgs.vHparams.vNonCausalType == cDgsNonCausalTypeSwaFull && vP0 >= aSeqPosMin[cSeqId];
          if (!cInSpan && dgs_hparams::mIsMaskedSwa(cNSwa, cSwaType, vP0, cP1)) {
            goto skip;
          }
        }

        if (alibi) {
          data[cIdst + vJ] = fDgsCast<T>(static_cast<float>(-std::abs(vP0 - cP1)));
        } else {
          data[cIdst + vJ] = cMaskKeep;
        }

        continue;
skip:
        data[cIdst + vJ] = cMaskDrop;
      }
    }
  }
}

template <typename T, bool causal, bool swa, bool is_2d>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & pArgs, T * data) {
  const bool cAlibi = pArgs.vHparams.vUseAlibi;
  if (cAlibi) {
    set_input_kq_mask_impl<T, causal, swa, is_2d, true>(pArgs, data);
  } else {
    set_input_kq_mask_impl<T, causal, swa, is_2d, false>(pArgs, data);
  }
}

template <typename T, bool causal, bool swa>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & pArgs, T * data) {
  const bool cIs2d = pArgs.vUbatch->mIsPos2d();
  if (cIs2d) {
    set_input_kq_mask_impl<T, causal, swa, true>(pArgs, data);
  } else {
    set_input_kq_mask_impl<T, causal, swa, false>(pArgs, data);
  }
}

template <typename T, bool causal> static void set_input_kq_mask_impl(const args_set_input_kq_mask & pArgs, T * data) {
  const bool cSwa = pArgs.vSwaType != cDgsSwaTypeNone;
  if (cSwa) {
    set_input_kq_mask_impl<T, causal, true>(pArgs, data);
  } else {
    set_input_kq_mask_impl<T, causal, false>(pArgs, data);
  }
}

template <typename T>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & pArgs, T * data, bool pCausalAttn) {
  if (pCausalAttn) {
    set_input_kq_mask_impl<T, true>(pArgs, data);
  } else {
    set_input_kq_mask_impl<T, false>(pArgs, data);
  }
}

void dgs_kv_cache::mSetInputKqMask(ggml_tensor * pDst, const dgs_ubatch * pUbatch, bool pCausalAttn) const {
  const uint32_t cNTokens = pUbatch->vNTokens;

  GGML_ASSERT(ggml_backend_buffer_is_host(pDst->buffer));

  const int64_t cNKv     = pDst->ne[0];
  const int64_t cNStream = pDst->ne[3];  // num streams in the current ubatch

  GGML_ASSERT(cNTokens % cNStream == 0);

  // n_tps == n_tokens_per_stream
  const int64_t cNTps = cNTokens / cNStream;

  // see dgs_non_causal_type
  // only the SWA cache (or the SWA layers of a single cache) become non-causal
  if (!pCausalAttn && vHparams.vNonCausalType == cDgsNonCausalTypeSwaOnly) {
    pCausalAttn = cSwaType == cDgsSwaTypeNone;
  }

  //const int64_t t_start = ggml_time_us();

  const args_set_input_kq_mask cArgs = {
    /*.hparams          =*/vHparams,
    /*.ubatch           =*/pUbatch,
    /*.v_cells          =*/lVCells,
    /*.seq_to_stream    =*/lSeqToStream,
    /*.n_swa            =*/cNSwa,
    /*.swa_type         =*/cSwaType,
    /*.n_kv             =*/cNKv,
    /*.n_stream         =*/cNStream,
    /*.n_tps            =*/cNTps,
  };

  if (pDst->type == GGML_TYPE_F16) {
    set_input_kq_mask_impl<ggml_fp16_t>(cArgs, (ggml_fp16_t *) pDst->data, pCausalAttn);
  } else {
    set_input_kq_mask_impl<float>(cArgs, (float *) pDst->data, pCausalAttn);
  }

  //const int64_t t_end = ggml_time_us();

  //DGS_LOG_ERROR("%s: kq mask time: %0.3f ms\n", __func__, (t_end - t_start)/1000.0);
}

void dgs_kv_cache::mSetInputPosBucket(ggml_tensor * pDst, const dgs_ubatch * pUbatch) const {
  const int64_t cNTokens = pUbatch->vNTokens;

  GGML_ASSERT(cNStream == 1 && "TODO: support multiple streams");
  const auto & vCells = lVCells[0];

  GGML_ASSERT(ggml_backend_buffer_is_host(pDst->buffer));
  GGML_ASSERT(!pUbatch->mEqualSeqs());  // TODO: use ubatch->n_seqs instead of failing

  int32_t * data = (int32_t *) pDst->data;

  const int32_t cNKv = pDst->ne[0];

  for (int vH = 0; vH < 1; ++vH) {
    for (int vI = 0; vI < cNTokens; ++vI) {
      for (int vJ = 0; vJ < cNKv; ++vJ) {
        // the position when the cells is empty is irrelevant - it will be masked out later in the attention
        const dgs_pos cP0 = vCells.mIsEmpty(vJ) ? -1 : vCells.mPosGet(vJ);

        data[vH * (cNKv * cNTokens) + vI * cNKv + vJ] =
          fDgsRelativePositionBucket(cP0, pUbatch->vPos[vI], vHparams.vNRelAttnBkts, false);
      }
    }
  }
}

void dgs_kv_cache::mSetInputKRot(ggml_tensor * pDst) const {
  GGML_ASSERT(ggml_backend_buffer_is_host(pDst->buffer));

  const auto cNRot = pDst->ne[0];
  GGML_ASSERT(dAttnRotHadamard.count(pDst->ne[0]));

  memcpy(pDst->data, dAttnRotHadamard.at(cNRot).data(), ggml_nbytes(pDst));
}

void dgs_kv_cache::mSetInputVRot(ggml_tensor * pDst) const {
  GGML_ASSERT(ggml_backend_buffer_is_host(pDst->buffer));

  const auto cNRot = pDst->ne[0];
  GGML_ASSERT(dAttnRotHadamard.count(pDst->ne[0]));

  memcpy(pDst->data, dAttnRotHadamard.at(cNRot).data(), ggml_nbytes(pDst));
}

bool dgs_kv_cache::mHasCellExt() const {
  // M-RoPE needs the 2D position, the PLE n-gram hash needs the token id
  return vHparams.mNPosPerEmbd() > 1 || vHparams.vPleNHeads > 0;
}

void dgs_kv_cache::mGetPrevTokens(const dgs_ubatch & pUbatch, uint32_t n, std::vector<dgs_token> & pRes) const {
  const uint32_t cNTokens = pUbatch.vNTokens;

  pRes.clear();
  pRes.resize(cNTokens * n, DGS_TOKEN_NULL);

  if (n == 0) {
    return;
  }

  // note: apply_ubatch() has already stored the current ubatch, so the cells cover the tokens
  //       of this very ubatch as well, which is what we want
  // the nearest cell at or before a position also resolves M-RoPE gaps, where multiple tokens
  // share the same temporal pos

  // an embd (multimodal) ubatch can repeat one position for a whole image, so positions
  // do not encode the token order; resolve its predecessors by ubatch order instead
  // same for a mixed ubatch
  const bool cByOrder = !pUbatch.token || pUbatch.mIsMixed();

  std::vector<uint32_t>                                 lOrd;  // index among the ubatch tokens of the same seq
  std::unordered_map<dgs_seq_id, std::vector<uint32_t>> dSeqIdx;

  if (cByOrder) {
    lOrd.resize(cNTokens);
    for (uint32_t vI = 0; vI < cNTokens; ++vI) {
      auto & lV = dSeqIdx[pUbatch.vSeqId[vI][0]];
      lOrd[vI]  = lV.size();
      lV.push_back(vI);
    }
  }

  for (uint32_t vI = 0; vI < cNTokens; ++vI) {
    // TODO: a token that belongs to more than one sequence has an ambiguous history.
    //       the n-gram architectures have to reject such batches
    const dgs_seq_id cSeqId = pUbatch.vSeqId[vI][0];

    for (uint32_t vJ = 0; vJ < n; ++vJ) {
      const dgs_pos cD = (dgs_pos) (n - vJ);

      dgs_pos vP;
      if (cByOrder) {
        const auto &  lV = dSeqIdx[cSeqId];
        const int64_t cK = (int64_t) lOrd[vI] - cD;
        // k >= 0: an earlier token of this very ubatch; k < 0: before the chunk
        vP               = cK >= 0 ? pUbatch.vPos[lV[cK]] : pUbatch.vPos[lV[0]] + (dgs_pos) cK;
      } else {
        vP = pUbatch.vPos[vI] - cD;
      }

      if (vP < 0) {
        continue;
      }

      pRes[vI * n + vJ] = lVCells[lSeqToStream[cSeqId]].mSeqPosTokLe(cSeqId, vP);
    }
  }
}

size_t dgs_kv_cache::mTotalSize() const {
  size_t size = 0;

  for (const auto & [_, buf] : lCtxsBufs) {
    size += ggml_backend_buffer_get_size(buf.get());
  }

  return size;
}

size_t dgs_kv_cache::mSizeKBytes() const {
  size_t vSizeKBytes = 0;

  for (const auto & layer : lLayers) {
    vSizeKBytes += ggml_nbytes(layer.vK);
  }

  return vSizeKBytes;
}

size_t dgs_kv_cache::mSizeVBytes() const {
  size_t vSizeVBytes = 0;

  for (const auto & layer : lLayers) {
    vSizeVBytes += layer.vV ? ggml_nbytes(layer.vV) : 0;
  }

  return vSizeVBytes;
}

ggml_tensor * dgs_kv_cache::mBuildRopeShift(const dgs_cparams & pCparams,
                                            ggml_context *      pCtx,
                                            ggml_tensor *       pCur,
                                            ggml_tensor *       pShift,
                                            ggml_tensor *       pRot,
                                            ggml_tensor *       pFactors,
                                            float               pFreqBase,
                                            float               pFreqScale,
                                            uint32_t            pIl) const {
  const auto & vNCtxOrig = pCparams.vNCtxOrigYarn;

  const auto & vYarnExtFactor  = pCparams.vYarnExtFactor;
  const auto & vYarnBetaFast   = pCparams.vYarnBetaFast;
  const auto & vYarnBetaSlow   = pCparams.vYarnBetaSlow;
  const auto & vYarnAttnFactor = pCparams.vYarnAttnFactor;

  const auto &  vNRot     = vHparams.n_rot(pIl);
  const auto &  vRopeType = vHparams.vRopeType == cDgsRopeTypeMrope || vHparams.vRopeType == cDgsRopeTypeImrope
                              // @ngxson : this is a workaround
                              // for M-RoPE, we want to rotate the whole vector when doing KV shift
                              // a normal RoPE should work, we just need to use the correct ordering
                              ?
                              cDgsRopeTypeNeox :
                              vHparams.vRopeType;
  ggml_tensor * vTmp;

  if (ggml_is_quantized(pCur->type)) {
    // dequantize to f32 -> RoPE -> quantize back
    vTmp = ggml_cast(pCtx, pCur, GGML_TYPE_F32);

    // rotate back
    vTmp = fDgsMulMatHadamard(pCtx, vTmp, pRot);

    vTmp = ggml_rope_ext(pCtx, vTmp, pShift, pFactors, vNRot, vRopeType, vNCtxOrig, pFreqBase, pFreqScale,
                         vYarnExtFactor, vYarnAttnFactor, vYarnBetaFast, vYarnBetaSlow);

    // rotate fwd
    vTmp = fDgsMulMatHadamard(pCtx, vTmp, pRot);

    vTmp = ggml_cpy(pCtx, vTmp, pCur);
  } else {
    // we rotate only the first n_rot dimensions
    vTmp = ggml_rope_ext_inplace(pCtx, pCur, pShift, pFactors, vNRot, vRopeType, vNCtxOrig, pFreqBase, pFreqScale,
                                 vYarnExtFactor, vYarnAttnFactor, vYarnBetaFast, vYarnBetaSlow);
  }

  return vTmp;
}

class llm_graph_input_k_shift : public llm_graph_input_i {
public:
  llm_graph_input_k_shift(const dgs_kv_cache * pKvSelf) : vKvSelf(pKvSelf) {}

  virtual ~llm_graph_input_k_shift() = default;

  void mSetInput(const dgs_ubatch * pUbatch) override;

  ggml_tensor * vKShift;  // I32 [kv_size*n_stream]

  // note: assumes k_rot^2 == I
  ggml_tensor * vKRot = nullptr;

  const dgs_kv_cache * vKvSelf;
};

void llm_graph_input_k_shift::mSetInput(const dgs_ubatch * pUbatch) {
  GGML_UNUSED(pUbatch);

  if (vKShift) {
    vKvSelf->mSetInputKShift(vKShift);
  }

  if (vKRot && vKRot->buffer) {
    vKvSelf->mSetInputKRot(vKRot);
  }
}

ggml_cgraph * dgs_kv_cache::mBuildGraphShift(llm_graph_result * pRes, dgs_context * pLctx) const {
  // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
  GGML_ASSERT(!vOther);

  auto * vCtx = pRes->mGetCtx();
  auto * vGf  = pRes->mGetGf();

  auto vInp = std::make_unique<llm_graph_input_k_shift>(this);

  vInp->vKShift = ggml_new_tensor_1d(vCtx, GGML_TYPE_I32, (int64_t) mGetSize() * cNStream);
  ggml_set_input(vInp->vKShift);

  vInp->vKRot = mBuildInputKRot(vCtx);

  const auto & vCparams = pLctx->mGetCparams();

  for (const auto & layer : lLayers) {
    const uint32_t cIl = layer.vIl;

    if (!vHparams.mHasRope(cIl)) {
      continue;
    }

    const int64_t cNHeadKv   = vHparams.n_head_kv(cIl);
    const int64_t cNEmbdKGqa = vHparams.n_embd_k_gqa(cIl);

    const auto cNRot       = vHparams.n_rot(cIl);
    const auto cNEmbdHeadK = vHparams.n_embd_head_k(cIl);
    const auto cNEmbdNope  = vHparams.vNLoraKv > 0 ? cNEmbdHeadK - cNRot : 0;

    const float cFreqBaseL  = vModel.mGetRopeFreqBase(vCparams, cIl);
    const float cFreqScaleL = vModel.mGetRopeFreqScale(vCparams, cIl);

    ggml_tensor * vRopeFactors = vModel.mGetRopeFactors(vCparams, cIl);

    ggml_tensor * vK =
      ggml_view_3d(vCtx, layer.vK, cNRot, cNHeadKv, mGetSize() * cNStream, ggml_row_size(layer.vK->type, cNEmbdHeadK),
                   ggml_row_size(layer.vK->type, cNEmbdKGqa), ggml_row_size(layer.vK->type, cNEmbdNope));

    ggml_tensor * vCur =
      mBuildRopeShift(vCparams, vCtx, vK, vInp->vKShift, vInp->vKRot, vRopeFactors, cFreqBaseL, cFreqScaleL, cIl);

    ggml_build_forward_expand(vGf, vCur);
  }

  pRes->mAddInput(std::move(vInp));

  return vGf;
}

void dgs_kv_cache::mStateWrite(dgs_io_write_i & pIo, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags) const {
  // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
  if (vOther) {
    return;
  }

  GGML_UNUSED(pFlags);

  pIo.write(&cNStream, sizeof(cNStream));

  for (uint32_t vS = 0; vS < cNStream; ++vS) {
    cell_ranges_t vCr{ vS, {} };

    uint32_t vCellCount = 0;

    const auto & vCells = lVCells[vS];

    // Count the number of cells with the specified seq_id
    // Find all the ranges of cells with this seq id (or all, when -1)
    uint32_t vCellRangeBegin = vCells.size();

    for (uint32_t vI = 0; vI < vCells.size(); ++vI) {
      bool vAddCell = true;

      vAddCell = vAddCell && !vCells.mIsEmpty(vI);
      vAddCell = vAddCell && (pSeqId == -1 || vCells.mSeqHas(vI, pSeqId));

      // check the cell is not SWA-masked
      if (vAddCell && pSeqId != -1) {
        const bool cIsMasked =
          dgs_hparams::mIsMaskedSwa(cNSwa, cSwaType, vCells.mPosGet(vI), vCells.mSeqPosMax(pSeqId));

        vAddCell = !cIsMasked;
      }

      if (vAddCell) {
        ++vCellCount;
        if (vCellRangeBegin == vCells.size()) {
          vCellRangeBegin = vI;
        }
      } else {
        if (vCellRangeBegin != vCells.size()) {
          vCr.data.emplace_back(vCellRangeBegin, vI);
          vCellRangeBegin = vCells.size();
        }
      }
    }

    if (vCellRangeBegin != vCells.size()) {
      vCr.data.emplace_back(vCellRangeBegin, vCells.size());
    }

    // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
    uint32_t vCellCountCheck = 0;
    for (const auto & vRange : vCr.data) {
      vCellCountCheck += vRange.second - vRange.first;
    }
    GGML_ASSERT(vCellCount == vCellCountCheck);

    pIo.write(&vCellCount, sizeof(vCellCount));

    // skip empty streams
    if (vCellCount == 0) {
      continue;
    }

    mStateWriteMeta(pIo, vCr, pSeqId);
    mStateWriteData(pIo, vCr);
  }
}

void dgs_kv_cache::mStateRead(dgs_io_read_i & pIo, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags) {
  mStateReadSinfo(pIo, pSeqId, pFlags, nullptr, nullptr);
}

void dgs_kv_cache::mStateReadSinfo(dgs_io_read_i &         pIo,
                                   dgs_seq_id              pSeqId,
                                   dgs_state_seq_flags     pFlags,
                                   slot_info_vec_t *       pSinfosOut,
                                   const slot_info_vec_t * pSinfosIn) {
  // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
  if (vOther) {
    return;
  }

  GGML_UNUSED(pFlags);

  // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_DGS_SEQ_ID_NEG]
  GGML_ASSERT(pSeqId == -1 || (pSeqId >= 0 && (size_t) pSeqId < lSeqToStream.size()));

  if (pSinfosOut) {
    pSinfosOut->assign(cNStream, slot_info{});
  }

  if (pSinfosIn && pSinfosIn->size() != cNStream) {
    throw std::runtime_error("failed to restore kv cache: mirrored slot layout has the wrong stream count");
  }

  uint32_t vNStreamCur;
  pIo.read(&vNStreamCur, sizeof(vNStreamCur));
  if (vNStreamCur != cNStream) {
    throw std::runtime_error("n_stream mismatch");
  }

  // a whole-context restore replaces every stream, so the cache is emptied once here
  // clear() resets all streams at once, so doing it per stream below would keep only the last one
  if (pSeqId == -1) {
    clear(true);
  }

  for (uint32_t vS = 0; vS < cNStream; ++vS) {
    uint32_t vCellCount;
    pIo.read(&vCellCount, sizeof(vCellCount));

    if (vCellCount == 0) {
      // a mirrored cache must be empty here as well, or the two no longer agree cell for cell
      if (pSinfosIn && !(*pSinfosIn)[vS].empty()) {
        throw std::runtime_error("failed to restore kv cache: mirrored cache holds cells this one does not");
      }
      continue;
    }

    const uint32_t cStrm = pSeqId == -1 ? vS : lSeqToStream[pSeqId];

    slot_info vSinfo;

    bool vRes = true;
    vRes      = vRes && mStateReadMeta(pIo, cStrm, vCellCount, vSinfo, pSeqId, pSinfosIn ? &(*pSinfosIn)[vS] : nullptr);

    try {
      vRes = vRes && mStateReadData(pIo, cStrm, vCellCount, vSinfo);
    } catch (...) {
      vRes = false;
    }

    if (!vRes) {
      mStateClear(pSeqId, cStrm, vSinfo);
      throw std::runtime_error("failed to restore kv cache");
    }

    if (pSinfosOut) {
      (*pSinfosOut)[vS] = vSinfo;
    }
  }
}

void dgs_kv_cache::mStateWriteMeta(dgs_io_write_i & pIo, const cell_ranges_t & pCr, dgs_seq_id pSeqId) const {
  const auto & vCells = lVCells[pCr.vStrm];

  for (const auto & vRange : pCr.data) {
    for (uint32_t vI = vRange.first; vI < vRange.second; ++vI) {
      std::vector<dgs_seq_id> lSeqIds;

      for (dgs_seq_id vCur = 0; vCur < (int) cNSeqMax; ++vCur) {
        if (vCur == pSeqId || pSeqId == -1) {
          if (vCells.mSeqHas(vI, vCur)) {
            lSeqIds.push_back(vCur);
          }
        }
      }

      const dgs_pos  cPos    = vCells.mPosGet(vI);
      const uint32_t cNSeqId = lSeqIds.size();

      pIo.write(&cPos, sizeof(cPos));
      pIo.write(&cNSeqId, sizeof(cNSeqId));

      if (mHasCellExt()) {
        const dgs_kv_cell_ext cExt = vCells.mExtGet(vI);
        pIo.write(&cExt, sizeof(cExt));
      }

      for (const auto & vSeqId : lSeqIds) {
        pIo.write(&vSeqId, sizeof(vSeqId));
      }
    }
  }
}

void dgs_kv_cache::mStateWriteData(dgs_io_write_i & pIo, const cell_ranges_t & pCr) const {
  const auto & vCells = lVCells[pCr.vStrm];

  const uint32_t cVTrans = this->vVTrans ? 1 : 0;
  const uint32_t cNLayer = lLayers.size();

  pIo.write(&cVTrans, sizeof(cVTrans));
  pIo.write(&cNLayer, sizeof(cNLayer));
  pIo.write(&vNRotK, sizeof(vNRotK));
  pIo.write(&vNRotV, sizeof(vNRotV));

  // Iterate and write all the keys first, each row is a cell
  // Get whole range at a time
  for (const auto & layer : lLayers) {
    const uint32_t cIl = layer.vIl;

    const uint32_t cNEmbdKGqa = vHparams.n_embd_k_gqa(cIl);

    auto * vK = layer.lKStream[pCr.vStrm];

    // Write key type
    const int32_t cKTypeI = (int32_t) vK->type;
    pIo.write(&cKTypeI, sizeof(cKTypeI));

    // Write row size of key
    const uint64_t cKSizeRow = ggml_row_size(vK->type, cNEmbdKGqa);
    pIo.write(&cKSizeRow, sizeof(cKSizeRow));

    // Read each range of cells of k_size length and write out
    for (const auto & vRange : pCr.data) {
      const size_t cRangeSize = vRange.second - vRange.first;
      const size_t cBufSize   = cRangeSize * cKSizeRow;
      pIo.mWriteTensor(vK, vRange.first * cKSizeRow, cBufSize);
    }
  }

  if (!cVTrans) {
    for (const auto & layer : lLayers) {
      const uint32_t cIl = layer.vIl;

      const uint32_t cNEmbdVGqa = vHparams.n_embd_v_gqa(cIl);

      auto * vV = layer.lVStream[pCr.vStrm];
      if (!vV) {
        continue;
      }

      // Write value type
      const int32_t cVTypeI = (int32_t) vV->type;
      pIo.write(&cVTypeI, sizeof(cVTypeI));

      // Write row size of value
      const uint64_t cVSizeRow = ggml_row_size(vV->type, cNEmbdVGqa);
      pIo.write(&cVSizeRow, sizeof(cVSizeRow));

      // Read each range of cells of v_size length and write out
      for (const auto & vRange : pCr.data) {
        const size_t cRangeSize = vRange.second - vRange.first;
        const size_t cBufSize   = cRangeSize * cVSizeRow;
        pIo.mWriteTensor(vV, vRange.first * cVSizeRow, cBufSize);
      }
    }
  } else {
    // When v is transposed, we also need the element size and get the element ranges from each row
    const uint32_t cKvSize = vCells.size();

    for (const auto & layer : lLayers) {
      const uint32_t cIl = layer.vIl;

      const uint32_t cNEmbdVGqa = vHparams.n_embd_v_gqa(cIl);

      auto * vV = layer.lVStream[pCr.vStrm];
      if (!vV) {
        continue;
      }

      // Write value type
      const int32_t cVTypeI = (int32_t) vV->type;
      pIo.write(&cVTypeI, sizeof(cVTypeI));

      // Write element size
      const uint32_t cVSizeEl = ggml_type_size(vV->type);
      pIo.write(&cVSizeEl, sizeof(cVSizeEl));

      // Write GQA embedding size
      pIo.write(&cNEmbdVGqa, sizeof(cNEmbdVGqa));

      // For each row, we get the element values of each cell
      for (uint32_t vJ = 0; vJ < cNEmbdVGqa; ++vJ) {
        // Read each range of cells of v_size_el length and write out
        for (const auto & vRange : pCr.data) {
          const size_t cRangeSize = vRange.second - vRange.first;
          const size_t cSrcOffset = (vRange.first + vJ * cKvSize) * cVSizeEl;
          const size_t cBufSize   = cRangeSize * cVSizeEl;
          pIo.mWriteTensor(vV, cSrcOffset, cBufSize);
        }
      }
    }
  }
}

bool dgs_kv_cache::mStateReadMeta(dgs_io_read_i &   pIo,
                                  uint32_t          pStrm,
                                  uint32_t          pCellCount,
                                  slot_info &       pSinfo,
                                  dgs_seq_id        pDestSeqId,
                                  const slot_info * pSinfoIn) {
  auto & vCells = lVCells[pStrm];
  auto & vHead  = lVHeads[pStrm];

  if (pDestSeqId != -1) {
    // single sequence
    if (pCellCount > vCells.size()) {
      DGS_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
      return false;
    }

    mSeqRm(pDestSeqId, -1, -1);

    dgs_batch_allocr vBalloc(vHparams.mNPosPerEmbd());

    dgs_ubatch vUbatch = vBalloc.mUbatchReserve(pCellCount, 1);

    vUbatch.vSeqIdUnq[0] = pDestSeqId;

    // the ext as it was saved, to put back after apply_ubatch()
    std::vector<dgs_kv_cell_ext> lExts;
    if (mHasCellExt()) {
      lExts.resize(pCellCount);
    }

    for (uint32_t vI = 0; vI < pCellCount; ++vI) {
      dgs_pos  vPos;
      uint32_t vNSeqId;

      pIo.read(&vPos, sizeof(vPos));
      pIo.read(&vNSeqId, sizeof(vNSeqId));

      if (vNSeqId != 1) {
        DGS_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
        return false;
      }

      if (mHasCellExt()) {
        dgs_kv_cell_ext vExt;
        pIo.read(&vExt, sizeof(vExt));

        if (vHparams.mNPosPerEmbd() > 1) {
          vUbatch.vPos[vI + vUbatch.vNTokens]     = vExt.vY;
          vUbatch.vPos[vI + vUbatch.vNTokens * 2] = vExt.vX;
        }

        // apply_ubatch() below restores ext.tok from the ubatch tokens
        vUbatch.token[vI] = vExt.vTok;

        lExts[vI] = vExt;
      }

      // read the sequence id, but directly discard it - we will use dest_seq_id instead
      {
        dgs_seq_id vSeqId;
        pIo.read(&vSeqId, sizeof(vSeqId));
      }

      vUbatch.vPos[vI]    = vPos;
      vUbatch.vNSeqId[vI] = vNSeqId;
      vUbatch.vSeqId[vI]  = &pDestSeqId;
    }

    if (pSinfoIn) {
      // this cache mirrors another one, so it takes that cache's layout instead of searching for its own cells
      if (pSinfoIn->empty() || pSinfoIn->mNStream() != 1 || pSinfoIn->lIdxs[0].size() != pCellCount) {
        DGS_LOG_ERROR("%s: mirrored slot layout holds %d cells, this cache restores %d\n", __func__,
                      pSinfoIn->empty() ? 0 : (int) pSinfoIn->lIdxs[0].size(), pCellCount);
        return false;
      }

      pSinfo = *pSinfoIn;

      // the layout is cell indices, so it means the same in both caches only while their streams line up
      pSinfo.vS0      = pStrm;
      pSinfo.vS1      = pStrm;
      pSinfo.lStrm[0] = pStrm;

      // seq_rm above freed exactly the cells this sequence held
      // anything else in the way is a cache that had already drifted, which this restore must not hide
      for (uint32_t vI = 0; vI < pCellCount; ++vI) {
        const uint32_t cIdx = pSinfo.lIdxs[0][vI];

        if (cIdx >= vCells.size() || !vCells.mIsEmpty(cIdx)) {
          DGS_LOG_ERROR("%s: cell %u of the mirrored slot layout is not free\n", __func__, cIdx);
          return false;
        }
      }
    } else {
      pSinfo = mFindSlot(vUbatch, false);
      if (pSinfo.empty()) {
        DGS_LOG_ERROR("%s: failed to find %d available cells in kv cache\n", __func__, pCellCount);
        return false;
      }
    }

    // note: apply_ubatch() rebuilds dgs_kv_cell_ext from the ubatch
    //       only ext.tok and the M-RoPE 2D position round-trip through it
    mApplyUbatch(pSinfo, vUbatch);

    // apply_ubatch() takes the 2D position from the ubatch, and that ubatch is built with this
    // cache's own n_pos_per_embd. a cache that does not use M-RoPE itself but mirrors one that
    // does (the qwen4exp QSA indexer) would drop x and y. put the saved ext back instead, which
    // is what the whole-context path below already does.
    for (uint32_t vI = 0; vI < (uint32_t) lExts.size(); ++vI) {
      vCells.mExtSet(pSinfo.lIdxs[0][vI], lExts[vI]);
    }

    DGS_LOG_DEBUG("%s: cell_count = %d, dest_seq_id = %d\n", __func__, pCellCount, pDestSeqId);

    // DEBUG CHECK: verify that all cells were allocated and have correct seq_id and pos values
    GGML_ASSERT(pSinfo.mNStream() == 1);
    GGML_ASSERT(pSinfo.lIdxs[0].size() == pCellCount);
    for (uint32_t vI = 0; vI < pCellCount; ++vI) {
      const uint32_t cIdx = pSinfo.lIdxs[0][vI];
      GGML_ASSERT(vCells.mPosGet(cIdx) == vUbatch.vPos[vI]);
      GGML_ASSERT(vCells.mSeqHas(cIdx, pDestSeqId));
    }
  } else {
    // whole KV cache restore

    if (pCellCount > vCells.size()) {
      DGS_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
      return false;
    }

    // the cells go in from 0, so a mirrored cache lands on the same ones as long as it restores the same count. the layout itself carries no more information here
    if (pSinfoIn && (pSinfoIn->empty() || pSinfoIn->mNStream() != 1 || pSinfoIn->lIdxs[0].size() != pCellCount)) {
      DGS_LOG_ERROR("%s: mirrored slot layout holds %d cells, this cache restores %d\n", __func__,
                    pSinfoIn->empty() ? 0 : (int) pSinfoIn->lIdxs[0].size(), pCellCount);
      return false;
    }

    for (uint32_t vI = 0; vI < pCellCount; ++vI) {
      dgs_pos  vPos;
      uint32_t vNSeqId;

      pIo.read(&vPos, sizeof(vPos));
      pIo.read(&vNSeqId, sizeof(vNSeqId));

      vCells.mPosSet(vI, vPos);

      if (mHasCellExt()) {
        dgs_kv_cell_ext vExt;
        pIo.read(&vExt, sizeof(vExt));
        vCells.mExtSet(vI, vExt);
      }

      for (uint32_t vJ = 0; vJ < vNSeqId; ++vJ) {
        dgs_seq_id vSeqId;
        pIo.read(&vSeqId, sizeof(vSeqId));

        if (vSeqId < 0 || (uint32_t) vSeqId >= cNSeqMax) {
          DGS_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, vSeqId, cNSeqMax);
          return false;
        }

        vCells.mSeqAdd(vI, vSeqId);
      }
    }

    // Create contiguous slot_info for whole cache restore
    pSinfo.vS0 = pStrm;
    pSinfo.vS1 = pStrm;
    pSinfo.resize(1);
    pSinfo.lStrm[0] = pStrm;
    pSinfo.lIdxs[0].resize(pCellCount);
    for (uint32_t vI = 0; vI < pCellCount; ++vI) {
      pSinfo.lIdxs[0][vI] = vI;
    }

    vHead = 0;
  }

  return true;
}

bool dgs_kv_cache::mStateReadData(dgs_io_read_i & pIo, uint32_t pStrm, uint32_t pCellCount, const slot_info & pSinfo) {
  auto & vCells = lVCells[pStrm];

  // batch the scatter reads per contiguous run of destination indices
  // from inclusive, to exclusive - same convention as cell_ranges_t
  // contiguous cells yield a single run covering the whole block
  struct cell_run {
    uint32_t vFrom;
    uint32_t vTo;
  };

  std::vector<cell_run> lRuns;
  if (pCellCount > 0) {
    const auto & lIdxs = pSinfo.lIdxs[0];
    uint32_t     vI0   = 0;
    while (vI0 < pCellCount) {
      uint32_t vI1 = vI0 + 1;
      while (vI1 < pCellCount && lIdxs[vI1] == lIdxs[vI1 - 1] + 1) {
        ++vI1;
      }
      lRuns.push_back({ lIdxs[vI0], lIdxs[vI1 - 1] + 1 });
      vI0 = vI1;
    }
  }

  uint32_t vVTrans;
  uint32_t vNLayer;
  uint32_t vNRotKRef;
  uint32_t vNRotVRef;

  pIo.read(&vVTrans, sizeof(vVTrans));
  pIo.read(&vNLayer, sizeof(vNLayer));
  pIo.read(&vNRotKRef, sizeof(vNRotKRef));
  pIo.read(&vNRotVRef, sizeof(vNRotVRef));

  if (vNLayer != lLayers.size()) {
    DGS_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, vNLayer, (uint32_t) lLayers.size());
    return false;
  }

  if (pCellCount > vCells.size()) {
    DGS_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, pCellCount, vCells.size());
    return false;
  }

  if (this->vVTrans != (bool) vVTrans) {
    DGS_LOG_ERROR("%s: incompatible V transposition\n", __func__);
    return false;
  }

  if (vNRotKRef != vNRotK) {
    DGS_LOG_ERROR("%s: incompatible key rotation (%u instead of %u)\n", __func__, vNRotKRef, vNRotK);
    return false;
  }

  if (vNRotVRef != vNRotV) {
    DGS_LOG_ERROR("%s: incompatible value rotation (%u instead of %u)\n", __func__, vNRotVRef, vNRotV);
    return false;
  }

  // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
  for (const auto & layer : lLayers) {
    const uint32_t cIl = layer.vIl;

    const uint32_t cNEmbdKGqa = vHparams.n_embd_k_gqa(cIl);

    auto * vK = layer.lKStream[pStrm];

    // Read type of key
    int32_t vKTypeIRef;
    pIo.read(&vKTypeIRef, sizeof(vKTypeIRef));
    const int32_t cKTypeI = (int32_t) vK->type;
    if (cKTypeI != vKTypeIRef) {
      DGS_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d)\n", __func__, cKTypeI, vKTypeIRef, cIl);
      return false;
    }

    // Read row size of key
    uint64_t vKSizeRowRef;
    pIo.read(&vKSizeRowRef, sizeof(vKSizeRowRef));
    const size_t cKSizeRow = ggml_row_size(vK->type, cNEmbdKGqa);
    if (cKSizeRow != vKSizeRowRef) {
      DGS_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d)\n", __func__, cKSizeRow, (size_t) vKSizeRowRef,
                    cIl);
      return false;
    }

    for (const auto & vR : lRuns) {
      pIo.mReadTensor(vK, (size_t) vR.vFrom * cKSizeRow, (size_t) (vR.vTo - vR.vFrom) * cKSizeRow);
    }
  }

  if (!this->vVTrans) {
    for (const auto & layer : lLayers) {
      const uint32_t cIl = layer.vIl;

      const uint32_t cNEmbdVGqa = vHparams.n_embd_v_gqa(cIl);

      auto * vV = layer.lVStream[pStrm];
      if (!vV) {
        continue;
      }

      // Read type of value
      int32_t vVTypeIRef;
      pIo.read(&vVTypeIRef, sizeof(vVTypeIRef));
      const int32_t cVTypeI = (int32_t) vV->type;
      if (cVTypeI != vVTypeIRef) {
        DGS_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, cVTypeI, vVTypeIRef, cIl);
        return false;
      }

      // Read row size of value
      uint64_t vVSizeRowRef;
      pIo.read(&vVSizeRowRef, sizeof(vVSizeRowRef));
      const size_t cVSizeRow = ggml_row_size(vV->type, cNEmbdVGqa);
      if (cVSizeRow != vVSizeRowRef) {
        DGS_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, cVSizeRow,
                      (size_t) vVSizeRowRef, cIl);
        return false;
      }

      for (const auto & vR : lRuns) {
        pIo.mReadTensor(vV, (size_t) vR.vFrom * cVSizeRow, (size_t) (vR.vTo - vR.vFrom) * cVSizeRow);
      }
    }
  } else {
    // For each layer, read the values for each cell (transposed)
    for (const auto & layer : lLayers) {
      const uint32_t cIl = layer.vIl;

      const uint32_t cNEmbdVGqa = vHparams.n_embd_v_gqa(cIl);

      auto * vV = layer.lVStream[pStrm];
      if (!vV) {
        continue;
      }

      // Read type of value
      int32_t vVTypeIRef;
      pIo.read(&vVTypeIRef, sizeof(vVTypeIRef));
      const int32_t cVTypeI = (int32_t) vV->type;
      if (cVTypeI != vVTypeIRef) {
        DGS_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, cVTypeI, vVTypeIRef, cIl);
        return false;
      }

      // Read element size of value
      uint32_t vVSizeElRef;
      pIo.read(&vVSizeElRef, sizeof(vVSizeElRef));
      const size_t cVSizeEl = ggml_type_size(vV->type);
      if (cVSizeEl != vVSizeElRef) {
        DGS_LOG_ERROR("%s: mismatched value element size (%zu != %zu, layer %d)\n", __func__, cVSizeEl,
                      (size_t) vVSizeElRef, cIl);
        return false;
      }

      // Read GQA embedding size
      uint32_t vNEmbdVGqaRef;
      pIo.read(&vNEmbdVGqaRef, sizeof(vNEmbdVGqaRef));
      if (cNEmbdVGqa != vNEmbdVGqaRef) {
        DGS_LOG_ERROR("%s: mismatched GQA embedding size (%u != %u, layer %d)\n", __func__, cNEmbdVGqa, vNEmbdVGqaRef,
                      cIl);
        return false;
      }

      for (uint32_t vJ = 0; vJ < cNEmbdVGqa; ++vJ) {
        for (const auto & vR : lRuns) {
          const size_t cDstOffset = ((size_t) vR.vFrom + vJ * vCells.size()) * cVSizeEl;
          pIo.mReadTensor(vV, cDstOffset, (size_t) (vR.vTo - vR.vFrom) * cVSizeEl);
        }
      }
    }
  }

  return true;
}

void dgs_kv_cache::mStateClear(dgs_seq_id pSeqId) {
  if (pSeqId == -1) {
    clear(true);
    return;
  }

  GGML_ASSERT(pSeqId >= 0 && (size_t) pSeqId < lSeqToStream.size());

  const uint32_t cStrm = lSeqToStream[pSeqId];

  const auto & vCells = lVCells[cStrm];

  slot_info vSinfo;
  vSinfo.vS0 = cStrm;
  vSinfo.vS1 = cStrm;
  vSinfo.resize(1);
  vSinfo.lStrm[0] = cStrm;

  // a cell that another sequence still uses keeps its data
  for (uint32_t vI = 0; vI < vCells.size(); ++vI) {
    if (vCells.mSeqHas(vI, pSeqId) && vCells.mSeqCount(vI) == 1) {
      vSinfo.lIdxs[0].push_back(vI);
    }
  }

  mStateClear(pSeqId, cStrm, vSinfo);
}

// the cleared ranges mirror the write pattern of state_read_data() - keep both in sync
void dgs_kv_cache::mStateClear(dgs_seq_id pSeqId, uint32_t pStrm, const slot_info & pSinfo) {
  if (pSeqId == -1) {
    clear(true);
    return;
  }

  mSeqRm(pSeqId, -1, -1);

  // zero the K/V data of the failed restore attempt - the attention can still read the data of free cells
  if (pSinfo.empty() || pSinfo.size() == 0) {
    return;
  }

  const auto & vCells = lVCells[pStrm];

  const uint32_t cCellCount = pSinfo.size();

  const bool cIsContiguous = pSinfo.mIsContiguous();

  for (const auto & layer : lLayers) {
    const uint32_t cIl = layer.vIl;

    const uint32_t cNEmbdKGqa = vHparams.n_embd_k_gqa(cIl);

    auto * vK = layer.lKStream[pStrm];

    const size_t cKSizeRow = ggml_row_size(vK->type, cNEmbdKGqa);

    if (cIsContiguous) {
      fDgsClearTensorData(vK, pSinfo.mHead() * cKSizeRow, cCellCount * cKSizeRow);
    } else {
      for (uint32_t vI = 0; vI < cCellCount; ++vI) {
        fDgsClearTensorData(vK, pSinfo.lIdxs[0][vI] * cKSizeRow, cKSizeRow);
      }
    }
  }

  for (const auto & layer : lLayers) {
    const uint32_t cIl = layer.vIl;

    const uint32_t cNEmbdVGqa = vHparams.n_embd_v_gqa(cIl);

    auto * vV = layer.lVStream[pStrm];
    if (!vV) {
      continue;
    }

    if (!vVTrans) {
      const size_t cVSizeRow = ggml_row_size(vV->type, cNEmbdVGqa);

      if (cIsContiguous) {
        fDgsClearTensorData(vV, pSinfo.mHead() * cVSizeRow, cCellCount * cVSizeRow);
      } else {
        for (uint32_t vI = 0; vI < cCellCount; ++vI) {
          fDgsClearTensorData(vV, pSinfo.lIdxs[0][vI] * cVSizeRow, cVSizeRow);
        }
      }
    } else {
      const size_t cVSizeEl = ggml_type_size(vV->type);

      if (cIsContiguous) {
        const uint32_t cH = pSinfo.mHead();

        for (uint32_t vJ = 0; vJ < cNEmbdVGqa; ++vJ) {
          fDgsClearTensorData(vV, (cH + vJ * vCells.size()) * cVSizeEl, cCellCount * cVSizeEl);
        }
      } else {
        for (uint32_t vJ = 0; vJ < cNEmbdVGqa; ++vJ) {
          for (uint32_t vI = 0; vI < cCellCount; ++vI) {
            fDgsClearTensorData(vV, (pSinfo.lIdxs[0][vI] + vJ * vCells.size()) * cVSizeEl, cVSizeEl);
          }
        }
      }
    }
  }
}

//
// dgs_kv_cache_context
//

dgs_kv_cache_context::dgs_kv_cache_context(dgs_memory_status pStatus) : vStatus(pStatus) {}

dgs_kv_cache_context::dgs_kv_cache_context(dgs_kv_cache * pKv) : vStatus(cDgsMemoryStatusSuccess), vKv(pKv) {
  vNKv = pKv->mGetSize();

  const uint32_t cNStream = pKv->mGetNStream();

  // create a dummy slot info - the actual data is irrelevant. we just need to build the graph
  lSinfos.resize(1);
  lSinfos[0].vS0 = 0;
  lSinfos[0].vS1 = cNStream - 1;
  lSinfos[0].lIdxs.resize(cNStream);
  for (uint32_t vS = 0; vS < cNStream; ++vS) {
    lSinfos[0].lStrm.push_back(vS);
    lSinfos[0].lIdxs[vS].resize(1, 0);
  }
}

dgs_kv_cache_context::dgs_kv_cache_context(dgs_kv_cache *   pKv,
                                           dgs_context *    pLctx,
                                           bool             pDoShift,
                                           stream_copy_info pScInfo) :
  vStatus(cDgsMemoryStatusSuccess),
  vKv(pKv),
  vLctx(pLctx),
  vDoShift(pDoShift),
  vScInfo(std::move(pScInfo)) {
  if (!pDoShift && this->vScInfo.empty()) {
    vStatus = cDgsMemoryStatusNoUpdate;
  }
}

dgs_kv_cache_context::dgs_kv_cache_context(dgs_kv_cache *                pKv,
                                           dgs_kv_cache::slot_info_vec_t pSinfos,
                                           std::vector<dgs_ubatch>       pUbatches) :
  vStatus(cDgsMemoryStatusSuccess),
  vKv(pKv),
  lSinfos(std::move(pSinfos)),
  lUbatches(std::move(pUbatches)) {}

dgs_kv_cache_context::~dgs_kv_cache_context() = default;

bool dgs_kv_cache_context::mNext() {
  assert(status == cDgsMemoryStatusSuccess);

  if (++vICur >= lUbatches.size()) {
    return false;
  }

  return true;
}

bool dgs_kv_cache_context::apply() {
  assert(!fDgsMemoryStatusIsFail(status));

  // no ubatches -> this is a KV cache update
  if (lUbatches.empty()) {
    vKv->mUpdate(vLctx, vDoShift, vScInfo);

    return true;
  }

  vKv->mApplyUbatch(lSinfos[vICur], lUbatches[vICur]);
  vNKv = vKv->mGetNKv(lSinfos[vICur]);

  return true;
}

dgs_memory_status dgs_kv_cache_context::mGetStatus() const {
  return vStatus;
}

const dgs_ubatch & dgs_kv_cache_context::mGetUbatch() const {
  assert(status == cDgsMemoryStatusSuccess);

  return lUbatches[vICur];
}

uint32_t dgs_kv_cache_context::mGetNKv() const {
  return vNKv;
}

ggml_type dgs_kv_cache_context::mTypeK() const {
  return vKv->mTypeK();
}

ggml_type dgs_kv_cache_context::mTypeV() const {
  return vKv->mTypeV();
}

ggml_tensor * dgs_kv_cache_context::mGetK(ggml_context * pCtx, int32_t pIl) const {
  return vKv->mGetK(pCtx, pIl, vNKv, lSinfos[vICur]);
}

ggml_tensor * dgs_kv_cache_context::mGetV(ggml_context * pCtx, int32_t pIl) const {
  return vKv->mGetV(pCtx, pIl, vNKv, lSinfos[vICur]);
}

ggml_tensor * dgs_kv_cache_context::mCpyK(ggml_context * pCtx,
                                          ggml_tensor *  pKCur,
                                          ggml_tensor *  pKIdxs,
                                          int32_t        pIl) const {
  return vKv->mCpyK(pCtx, pKCur, pKIdxs, pIl, lSinfos[vICur]);
}

ggml_tensor * dgs_kv_cache_context::mCpyV(ggml_context * pCtx,
                                          ggml_tensor *  pVCur,
                                          ggml_tensor *  pVIdxs,
                                          int32_t        pIl) const {
  return vKv->mCpyV(pCtx, pVCur, pVIdxs, pIl, lSinfos[vICur]);
}

ggml_tensor * dgs_kv_cache_context::mBuildInputKIdxs(ggml_context * pCtx, const dgs_ubatch & pUbatch) const {
  return vKv->mBuildInputKIdxs(pCtx, pUbatch);
}

ggml_tensor * dgs_kv_cache_context::mBuildInputVIdxs(ggml_context * pCtx, const dgs_ubatch & pUbatch) const {
  return vKv->mBuildInputVIdxs(pCtx, pUbatch);
}

ggml_tensor * dgs_kv_cache_context::mBuildInputKRot(ggml_context * pCtx) const {
  return vKv->mBuildInputKRot(pCtx);
}

ggml_tensor * dgs_kv_cache_context::mBuildInputVRot(ggml_context * pCtx) const {
  return vKv->mBuildInputVRot(pCtx);
}

void dgs_kv_cache_context::mSetInputKShift(ggml_tensor * pDst) const {
  vKv->mSetInputKShift(pDst);
}

void dgs_kv_cache_context::mSetInputKIdxs(ggml_tensor * pDst, const dgs_ubatch * pUbatch) const {
  vKv->mSetInputKIdxs(pDst, pUbatch, lSinfos[vICur]);
}

void dgs_kv_cache_context::mSetInputVIdxs(ggml_tensor * pDst, const dgs_ubatch * pUbatch) const {
  vKv->mSetInputVIdxs(pDst, pUbatch, lSinfos[vICur]);
}

void dgs_kv_cache_context::mSetInputKqMask(ggml_tensor * pDst, const dgs_ubatch * pUbatch, bool pCausalAttn) const {
  vKv->mSetInputKqMask(pDst, pUbatch, pCausalAttn);
}

void dgs_kv_cache_context::mSetInputPosBucket(ggml_tensor * pDst, const dgs_ubatch * pUbatch) const {
  vKv->mSetInputPosBucket(pDst, pUbatch);
}

void dgs_kv_cache_context::mSetInputKRot(ggml_tensor * pDst) const {
  vKv->mSetInputKRot(pDst);
}

void dgs_kv_cache_context::mSetInputVRot(ggml_tensor * pDst) const {
  vKv->mSetInputVRot(pDst);
}

void dgs_kv_cache_context::mGetPrevTokens(const dgs_ubatch & pUbatch, uint32_t n, std::vector<dgs_token> & pRes) const {
  vKv->mGetPrevTokens(pUbatch, n, pRes);
}
