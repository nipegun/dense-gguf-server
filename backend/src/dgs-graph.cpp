#include "dgs-graph.h"

#include "dgs-batch.h"
#include "dgs-cparams.h"
#include "dgs-impl.h"
#include "dgs-kv-cache-dsa-iswa.h"
#include "dgs-kv-cache-dsa.h"
#include "dgs-kv-cache-dsv4.h"
#include "dgs-kv-cache-iswa.h"
#include "dgs-kv-cache-msa.h"
#include "dgs-kv-cache.h"
#include "dgs-memory-hybrid-iswa.h"
#include "dgs-memory-hybrid.h"
#include "dgs-memory-recurrent.h"
#include "dgs-model.h"
#include "dgs-sampler.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_set>

// dedup helpers

static ggml_tensor * fBuildAttnInpKqMask(ggml_context *               pCtx,
                                         const dgs_kv_cache_context * pMctx,
                                         const dgs_ubatch &           pUbatch,
                                         const dgs_cparams &          pCparams) {
  const auto cNKv     = pMctx->mGetNKv();
  const auto cNTokens = pUbatch.vNTokens;
  const auto cNStream = pCparams.vKvUnified ? 1 : pUbatch.vNSeqsUnq;

  // flash attention requires an f16 mask
  const auto type = pCparams.vFlashAttn ? GGML_TYPE_F16 : GGML_TYPE_F32;

  ggml_tensor * vRes = ggml_new_tensor_4d(pCtx, type, cNKv, cNTokens / cNStream, 1, cNStream);
  ggml_set_input(vRes);
  ggml_set_name(vRes, "attn_inp_kq_mask");

  return vRes;
}

static bool fCanReuseKqMask(ggml_tensor *                pKqMask,
                            const dgs_kv_cache_context * pMctx,
                            const dgs_ubatch &           pUbatch,
                            const dgs_cparams &          pCparams) {
  const auto cNKv     = pMctx->mGetNKv();
  const auto cNTokens = pUbatch.vNTokens;
  const auto cNStream = pCparams.vKvUnified ? 1 : pUbatch.vNSeqsUnq;

  bool vRes = true;

  vRes &= (pKqMask->ne[0] == cNKv);
  vRes &= (pKqMask->ne[1] == cNTokens / cNStream);
  vRes &= (pKqMask->ne[2] == 1);
  vRes &= (pKqMask->ne[3] == cNStream);

  return vRes;
}

// impl

void llm_graph_input_embd::mSetInput(const dgs_ubatch * pUbatch) {
  if (pUbatch->token) {
    const int64_t cNTokens = pUbatch->vNTokens;

    ggml_backend_tensor_set(vTokens, pUbatch->token, 0, cNTokens * ggml_element_size(vTokens));
  }

  if (pUbatch->vEmbd && vEmbd && !pUbatch->mIsMixed()) {
    GGML_ASSERT(cNEmbd == vEmbd->ne[0]);

    const int64_t cNTokens = pUbatch->vNTokens;

    ggml_backend_tensor_set(vEmbd, pUbatch->vEmbd, 0, cNTokens * cNEmbd * ggml_element_size(vEmbd));
  }

  if (pUbatch->mIsMixed() && vEmbd) {
    GGML_ASSERT(vMixedTokens && vMixedSlots && vMixedEmbd && "mixed token/embd ubatch is not supported here");

    std::vector<int32_t> lIds;
    std::vector<int64_t> lSlots;
    for (uint32_t vI = 0; vI < pUbatch->vNTokens; ++vI) {
      if (!pUbatch->type[vI]) {
        lIds.push_back(pUbatch->token[vI]);
        lSlots.push_back(vI);
      }
    }
    GGML_ASSERT((int64_t) lIds.size() == vMixedTokens->ne[0]);
    GGML_ASSERT(cNEmbd == vMixedEmbd->ne[0]);

    ggml_backend_tensor_set(vMixedTokens, lIds.data(), 0, ggml_nbytes(vMixedTokens));
    ggml_backend_tensor_set(vMixedSlots, lSlots.data(), 0, ggml_nbytes(vMixedSlots));
    ggml_backend_tensor_set(vMixedEmbd, pUbatch->vEmbd, 0, ggml_nbytes(vMixedEmbd));
  }

  if (vScaleRows) {
    const int64_t cNTokens = pUbatch->vNTokens;

    std::vector<float> data(cNTokens);
    for (int64_t vI = 0; vI < cNTokens; ++vI) {
      const bool cIsEmbd = !pUbatch->token || (pUbatch->mIsMixed() && pUbatch->type[vI]);
      data[vI]           = cIsEmbd ? 1.0f : vScaleTok;
    }
    ggml_backend_tensor_set(vScaleRows, data.data(), 0, ggml_nbytes(vScaleRows));
  }
}

// number of token rows of the mixed path, a non-mixed ubatch is sized for the worst case
static int64_t fLlmGraphNTokRows(const dgs_ubatch & pUbatch) {
  if (!pUbatch.mIsMixed()) {
    return pUbatch.vNTokens;
  }
  int64_t n = 0;
  for (uint32_t vI = 0; vI < pUbatch.vNTokens; ++vI) {
    n += !pUbatch.type[vI];
  }
  return n;
}

bool llm_graph_input_embd::mCanReuse(const llm_graph_params & pParams) {
  bool vRes = true;

  vRes &= (!pParams.vUbatch.token) || (vTokens && vTokens->ne[0] == pParams.vUbatch.vNTokens);
  vRes &= (!pParams.vUbatch.vEmbd) || (vEmbd && vEmbd->ne[1] == pParams.vUbatch.vNTokens);
  vRes &= (!vMixedTokens) || vMixedTokens->ne[0] == fLlmGraphNTokRows(pParams.vUbatch);
  vRes &= (!vMixedEmbd) || vMixedEmbd->ne[1] == pParams.vUbatch.vNTokens;
  vRes &= (!vScaleRows) || vScaleRows->ne[1] == pParams.vUbatch.vNTokens;

  return vRes;
}

void llm_graph_input_embd_h::mSetInput(const dgs_ubatch * pUbatch) {
  ASSERT_EMBD_OR_TOKEN(*pUbatch);

  const int64_t cNTokens = pUbatch->vNTokens;

  if (pUbatch->token) {
    ggml_backend_tensor_set(vTokens, pUbatch->token, 0, cNTokens * ggml_element_size(vTokens));
  } else {
    // note: mtmd embedding input goes through here
    GGML_ASSERT(pUbatch->vEmbd);
    GGML_ASSERT(cNEmbd == vEmbd->ne[0]);

    ggml_backend_tensor_set(vEmbd, pUbatch->vEmbd, 0, cNTokens * cNEmbd * ggml_element_size(vEmbd));
  }

  GGML_ASSERT(pUbatch->vEmbdState && "this graph requires a state embedding, see dgs_batch_ext_set_embd_state()");
  GGML_ASSERT(cNEmbdState == vH->ne[0]);

  ggml_backend_tensor_set(vH, pUbatch->vEmbdState, 0, cNTokens * cNEmbdState * ggml_element_size(vH));
}

bool llm_graph_input_embd_h::mCanReuse(const llm_graph_params & pParams) {
  bool vRes = true;

  vRes &= (!pParams.vUbatch.token) || (vTokens && vTokens->ne[0] == pParams.vUbatch.vNTokens);
  vRes &= (!pParams.vUbatch.vEmbd) || (vEmbd && vEmbd->ne[1] == pParams.vUbatch.vNTokens);
  vRes &= (!pParams.vUbatch.vEmbdState) || (vH && vH->ne[1] == pParams.vUbatch.vNTokens);

  return vRes;
}

void llm_graph_input_pos::mSetInput(const dgs_ubatch * pUbatch) {
  if (pUbatch->vPos && vPos) {
    const int64_t cNTokens = pUbatch->vNTokens;

    const bool cHasEmbd = pUbatch->mIsMixed() || pUbatch->token == nullptr;
    if (cRopeSectionOrder == cDgsRopeSectionOrderTyxz || !cHasEmbd) {
      ggml_backend_tensor_set(vPos, pUbatch->vPos, 0, cNTokens * cNPosPerEmbd * ggml_element_size(vPos));
      return;
    }

    // input is always [t, y, x, z]
    // token entries are expanded by the batch to [p, p, p, 0]
    GGML_ASSERT(cNPosPerEmbd == 4);

    // slot index per section, slots are t = 0, y = 1, x = 2, z = 3
    std::array<int64_t, 4> aSlotOfSection = { 0, 1, 2, 3 };
    switch (cRopeSectionOrder) {
      case cDgsRopeSectionOrderTyxz:
        break;
      case cDgsRopeSectionOrderZyxt:
        aSlotOfSection = { 3, 1, 2, 0 };
        break;
      default:
        GGML_ABORT("unsupported rope section order");
    }

    std::vector<dgs_pos> lPosData(cNTokens * cNPosPerEmbd);
    for (int64_t vI = 0; vI < cNTokens; ++vI) {
      const bool cIsEmbd = pUbatch->mIsMixed() ? pUbatch->type[vI] != 0 : true;
      for (int64_t vS = 0; vS < 4; ++vS) {
        const int64_t cSlot          = cIsEmbd ? aSlotOfSection[vS] : vS;
        lPosData[vS * cNTokens + vI] = pUbatch->vPos[cSlot * cNTokens + vI];
      }
    }
    ggml_backend_tensor_set(vPos, lPosData.data(), 0, lPosData.size() * ggml_element_size(vPos));
  }
}

bool llm_graph_input_pos::mCanReuse(const llm_graph_params & pParams) {
  bool vRes = true;

  vRes &= vPos->ne[0] == pParams.vUbatch.vNTokens * cNPosPerEmbd;

  return vRes;
}

void llm_graph_input_attn_temp::mSetInput(const dgs_ubatch * pUbatch) {
  if (pUbatch->vPos && vAttnScale) {
    const int64_t cNTokens = pUbatch->vNTokens;

    GGML_ASSERT(cFAttnTempScale != 0.0f);
    GGML_ASSERT(cNAttnTempFloorScale != 0);

    std::vector<float> lAttnScaleData(cNTokens, 0.0f);
    for (int vI = 0; vI < cNTokens; ++vI) {
      const float cPos = pUbatch->vPos[vI];
      lAttnScaleData[vI] =
        std::log(std::floor((cPos + cFAttnTempOffset) / cNAttnTempFloorScale) + 1.0) * cFAttnTempScale + 1.0;
    }

    ggml_backend_tensor_set(vAttnScale, lAttnScaleData.data(), 0, cNTokens * ggml_element_size(vAttnScale));
  }
}

void llm_graph_input_pos_bucket::mSetInput(const dgs_ubatch * pUbatch) {
  if (vPosBucket) {
    const int64_t cNTokens = pUbatch->vNTokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(vPosBucket->buffer));
    GGML_ASSERT(!pUbatch->mEqualSeqs());  // TODO: use ubatch->n_seqs instead of failing

    int32_t * data = (int32_t *) vPosBucket->data;

    for (int vJ = 0; vJ < cNTokens; ++vJ) {
      for (int vI = 0; vI < cNTokens; ++vI) {
        data[vJ * cNTokens + vI] =
          fDgsRelativePositionBucket(pUbatch->vPos[vI], pUbatch->vPos[vJ], cHparams.vNRelAttnBkts, true);
      }
    }
  }
}

void llm_graph_input_pos_bucket_kv::mSetInput(const dgs_ubatch * pUbatch) {
  if (vPosBucket) {
    vMctx->mSetInputPosBucket(vPosBucket, pUbatch);
  }
}

void llm_graph_input_out_ids::mSetInput(const dgs_ubatch * pUbatch) {
  GGML_ASSERT(vOutIds);

  const int64_t cNTokens = pUbatch->vNTokens;

  GGML_ASSERT(ggml_backend_buffer_is_host(vOutIds->buffer));
  int32_t * data = (int32_t *) vOutIds->data;

  if (cNOutputs == cNTokens) {
    for (int vI = 0; vI < cNTokens; ++vI) {
      data[vI] = vI;
    }

    return;
  }

  GGML_ASSERT(pUbatch->vOutput);

  int vNOutputs = 0;

  for (int vI = 0; vI < cNTokens; ++vI) {
    if (pUbatch->vOutput[vI]) {
      data[vNOutputs++] = vI;
    }
  }
}

bool llm_graph_input_out_ids::mCanReuse(const llm_graph_params & pParams) {
  bool vRes = true;

  vRes &= cNOutputs == pParams.vNOutputs;

  return vRes;
}

void llm_graph_input_mean::mSetInput(const dgs_ubatch * pUbatch) {
  if (cCparams.vEmbeddings &&
      (cCparams.vPoolingType == cDgsPoolingTypeMean || cCparams.vPoolingType == cDgsPoolingTypeRank)) {
    const int64_t cNTokens    = pUbatch->vNTokens;
    const int64_t cNSeqTokens = pUbatch->vNSeqTokens;
    const int64_t cNSeqsUnq   = pUbatch->vNSeqsUnq;

    GGML_ASSERT(vMean);
    GGML_ASSERT(ggml_backend_buffer_is_host(vMean->buffer));

    float * data = (float *) vMean->data;
    memset(vMean->data, 0, cNTokens * cNSeqsUnq * ggml_element_size(vMean));

    std::vector<uint64_t> lSums(cNSeqsUnq, 0);
    for (int vI = 0; vI < cNTokens; vI += cNSeqTokens) {
      for (int vS = 0; vS < pUbatch->vNSeqId[vI]; ++vS) {
        const dgs_seq_id cSeqId  = pUbatch->vSeqId[vI][vS];
        const int32_t    cSeqIdx = pUbatch->vSeqIdx[cSeqId];

        lSums[cSeqIdx] += pUbatch->vNSeqTokens;
      }
    }

    std::vector<float> lDiv(cNSeqsUnq, 0.0f);
    for (int vS = 0; vS < cNSeqsUnq; ++vS) {
      const uint64_t cSum = lSums[vS];
      if (cSum > 0) {
        lDiv[vS] = 1.0f / float(cSum);
      }
    }

    for (int vI = 0; vI < cNTokens; vI += cNSeqTokens) {
      for (int vS = 0; vS < pUbatch->vNSeqId[vI]; ++vS) {
        const dgs_seq_id cSeqId  = pUbatch->vSeqId[vI][vS];
        const int32_t    cSeqIdx = pUbatch->vSeqIdx[cSeqId];

        for (int vJ = 0; vJ < cNSeqTokens; ++vJ) {
          data[cSeqIdx * cNTokens + vI + vJ] = lDiv[cSeqIdx];
        }
      }
    }
  }
}

void llm_graph_input_cls::mSetInput(const dgs_ubatch * pUbatch) {
  const int64_t cNTokens  = pUbatch->vNTokens;
  const int64_t cNSeqsUnq = pUbatch->vNSeqsUnq;

  if (cCparams.vEmbeddings &&
      (cCparams.vPoolingType == cDgsPoolingTypeCls || cCparams.vPoolingType == cDgsPoolingTypeRank ||
       cCparams.vPoolingType == cDgsPoolingTypeLast)) {
    GGML_ASSERT(vCls);
    GGML_ASSERT(ggml_backend_buffer_is_host(vCls->buffer));

    uint32_t * data = (uint32_t *) vCls->data;
    memset(vCls->data, 0, cNSeqsUnq * ggml_element_size(vCls));

    std::vector<int> lTargetPos(cNSeqsUnq, -1);
    std::vector<int> lTargetRow(cNSeqsUnq, -1);

    const bool cLast = (cCparams.vPoolingType == cDgsPoolingTypeLast ||
                        (cCparams.vPoolingType == cDgsPoolingTypeRank && cCparams.vCausalAttn));

    for (int vI = 0; vI < cNTokens; ++vI) {
      const dgs_pos cPos = pUbatch->vPos[vI];

      for (int vS = 0; vS < pUbatch->vNSeqId[vI]; ++vS) {
        const dgs_seq_id cSeqId  = pUbatch->vSeqId[vI][vS];
        const int32_t    cSeqIdx = pUbatch->vSeqIdx[cSeqId];

        if ((lTargetPos[cSeqIdx] == -1) || (cLast && cPos >= lTargetPos[cSeqIdx]) ||
            (!cLast && cPos < lTargetPos[cSeqIdx])) {
          lTargetPos[cSeqIdx] = cPos;
          lTargetRow[cSeqIdx] = vI;
        }
      }
    }

    for (int vS = 0; vS < cNSeqsUnq; ++vS) {
      if (lTargetRow[vS] >= 0) {
        data[vS] = lTargetRow[vS];
      }
    }
  }
}

void llm_graph_input_rs::mSetInput(const dgs_ubatch * pUbatch) {
  GGML_UNUSED(pUbatch);

  const int64_t cNRs = vMctx->mGetNRs();

  if (vSCopy) {
    GGML_ASSERT(ggml_backend_buffer_is_host(vSCopy->buffer));
    int32_t * data = (int32_t *) vSCopy->data;

    // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
    for (uint32_t vI = 0; vI < cNRs; ++vI) {
      data[vI] = vMctx->mSCopy(vI);
    }
  }
}

bool llm_graph_input_rs::mCanReuse(const llm_graph_params & pParams) {
  const auto * vMctx = static_cast<const dgs_memory_recurrent_context *>(pParams.vMctx);

  this->vMctx = vMctx;

  bool vRes = true;

  vRes &= vSCopy->ne[0] == vMctx->mGetNRs();

  vRes &= vSCopyMain->ne[0] == pParams.vUbatch.vNSeqs;

  vRes &= vHead == vMctx->mGetHead();
  vRes &= vRsZ == vMctx->mGetRsZ();

  return vRes;
}

void llm_graph_input_cross_embd::mSetInput(const dgs_ubatch * pUbatch) {
  GGML_UNUSED(pUbatch);

  if (vCrossEmbd && !vCross->lVEmbd.empty()) {
    assert(cross_embd->type == GGML_TYPE_F32);

    ggml_backend_tensor_set(vCrossEmbd, vCross->lVEmbd.data(), 0, ggml_nbytes(vCrossEmbd));
  }
}

template <typename T>
static void print_mask(const T * data, int64_t pNTokens, int64_t pNKv, int64_t pNSwa, dgs_swa_type pSwaType) {
  DGS_LOG_DEBUG("%s: === Attention mask ===\n", __func__);
  const char * vSwaTypeStr = "unknown";

  switch (pSwaType) {
    case cDgsSwaTypeNone:
      vSwaTypeStr = "DGS_SWA_TYPE_NONE";
      break;
    case cDgsSwaTypeStandard:
      vSwaTypeStr = "DGS_SWA_TYPE_STANDARD";
      break;
    case cDgsSwaTypeChunked:
      vSwaTypeStr = "DGS_SWA_TYPE_CHUNKED";
      break;
    case cDgsSwaTypeSymmetric:
      vSwaTypeStr = "DGS_SWA_TYPE_SYMMETRIC";
      break;
  };

  DGS_LOG_DEBUG("%s: n_swa : %d, n_kv: %d, swa_type: %s\n", __func__, (int) pNSwa, (int) pNKv, vSwaTypeStr);
  DGS_LOG_DEBUG("%s: '0' = can attend, '∞' = masked\n", __func__);
  DGS_LOG_DEBUG("%s: Rows = query tokens, Columns = key/value tokens\n\n", __func__);

  DGS_LOG_DEBUG("    ");
  for (int vJ = 0; vJ < std::min((int64_t) 20, pNKv); ++vJ) {
    DGS_LOG_DEBUG("%2d", vJ);
  }
  DGS_LOG_DEBUG("\n");

  for (int vI = 0; vI < std::min((int64_t) 20, pNTokens); ++vI) {
    DGS_LOG_DEBUG(" %2d ", vI);
    for (int vJ = 0; vJ < std::min((int64_t) 20, pNKv); ++vJ) {
      float vVal = fDgsCast<float>(data[vI * pNKv + vJ]);
      if (vVal == -INFINITY) {
        DGS_LOG_DEBUG(" ∞");
      } else {
        DGS_LOG_DEBUG(" 0");
      }
    }
    DGS_LOG_DEBUG("\n");
  }
}

void llm_graph_input_attn_no_cache::mSetInput(const dgs_ubatch * pUbatch) {
  const int64_t cNKv     = vSelfKqMask->ne[0];
  const int64_t cNTokens = pUbatch->vNTokens;

  const auto cFillMask = [&](auto * data, int64_t pNe, int pNSwa, dgs_swa_type pSwaType) {
    using T = std::remove_reference_t<decltype(*data)>;
    std::fill(data, data + pNe, fDgsCast<T>(-INFINITY));

    for (int vI1 = 0; vI1 < cNTokens; ++vI1) {
      const dgs_seq_id cS1 = pUbatch->vSeqId[vI1][0];
      const dgs_pos    cP1 = pUbatch->vPos[vI1];

      const uint64_t cIdst = vI1 * cNKv;

      for (int vI0 = 0; vI0 < cNTokens; ++vI0) {
        const dgs_seq_id cS0 = pUbatch->vSeqId[vI0][0];
        const dgs_pos    cP0 = pUbatch->vPos[vI0];

        // mask different sequences
        if (cS0 != cS1) {
          continue;
        }

        // mask future tokens
        if (cCparams.vCausalAttn && cP0 > cP1) {
          continue;
        }

        // apply SWA if any
        if (dgs_hparams::mIsMaskedSwa(pNSwa, pSwaType, cP0, cP1)) {
          continue;
        }

        data[cIdst + vI0] = fDgsCast<T>(cHparams.vUseAlibi ? -std::abs(cP0 - cP1) : 0.0f);
      }
    }

    if (vDebug) {
      print_mask(data, cNTokens, cNKv, pNSwa, pSwaType);
    }
  };

  GGML_ASSERT(vSelfKqMask);
  GGML_ASSERT(ggml_backend_buffer_is_host(vSelfKqMask->buffer));
  if (vSelfKqMask->type == GGML_TYPE_F16) {
    cFillMask((ggml_fp16_t *) vSelfKqMask->data, ggml_nelements(vSelfKqMask), 0, cDgsSwaTypeNone);
  } else {
    cFillMask((float *) vSelfKqMask->data, ggml_nelements(vSelfKqMask), 0, cDgsSwaTypeNone);
  }

  if (cHparams.vSwaType != cDgsSwaTypeNone) {
    GGML_ASSERT(vSelfKqMaskSwa);
    GGML_ASSERT(ggml_backend_buffer_is_host(vSelfKqMaskSwa->buffer));
    if (vSelfKqMaskSwa->type == GGML_TYPE_F16) {
      cFillMask((ggml_fp16_t *) vSelfKqMaskSwa->data, ggml_nelements(vSelfKqMaskSwa), cHparams.vNSwa,
                cHparams.vSwaType);
    } else {
      cFillMask((float *) vSelfKqMaskSwa->data, ggml_nelements(vSelfKqMaskSwa), cHparams.vNSwa, cHparams.vSwaType);
    }
  }
}

void llm_graph_input_attn_kv::mSetInput(const dgs_ubatch * pUbatch) {
  // the idxs are left unallocated when the KV cache is bypassed during training
  if (vSelfKIdxs && vSelfKIdxs->buffer) {
    vMctx->mSetInputKIdxs(vSelfKIdxs, pUbatch);
  }
  if (vSelfVIdxs && vSelfVIdxs->buffer) {
    vMctx->mSetInputVIdxs(vSelfVIdxs, pUbatch);
  }

  // the mask is left unallocated when the graph only stores K/V without attending
  // (e.g. DFlash's KV-injection pass)
  if (vSelfKqMask && vSelfKqMask->buffer) {
    vMctx->mSetInputKqMask(vSelfKqMask, pUbatch, cCparams.vCausalAttn);
  }

  if (vSelfKRot && vSelfKRot->buffer) {
    vMctx->mSetInputKRot(vSelfKRot);
  }

  if (vSelfVRot && vSelfVRot->buffer) {
    vMctx->mSetInputVRot(vSelfVRot);
  }
}

bool llm_graph_input_attn_kv::mCanReuse(const llm_graph_params & pParams) {
  const auto * vMctx = static_cast<const dgs_kv_cache_context *>(pParams.vMctx);

  this->vMctx = vMctx;

  bool vRes = true;

  vRes &= vSelfKIdxs->ne[0] == pParams.vUbatch.vNTokens;
  //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

  vRes &= fCanReuseKqMask(vSelfKqMask, vMctx, pParams.vUbatch, pParams.vCparams);

  return vRes;
}

void llm_graph_input_attn_k::mSetInput(const dgs_ubatch * pUbatch) {
  vMctx->mSetInputKIdxs(vSelfKIdxs, pUbatch);

  vMctx->mSetInputKqMask(vSelfKqMask, pUbatch, cCparams.vCausalAttn);
}

bool llm_graph_input_attn_k::mCanReuse(const llm_graph_params & pParams) {
  vMctx = static_cast<const dgs_kv_cache_context *>(pParams.vMctx);

  return mCanReuseImpl(pParams);
}

bool llm_graph_input_attn_k::mCanReuseImpl(const llm_graph_params & pParams) {
  bool vRes = true;

  vRes &= vSelfKIdxs->ne[0] == pParams.vUbatch.vNTokens;

  vRes &= fCanReuseKqMask(vSelfKqMask, vMctx, pParams.vUbatch, pParams.vCparams);

  return vRes;
}

llm_graph_input_attn_kv_msa::llm_graph_input_attn_kv_msa(const dgs_hparams &              pHparams,
                                                         const dgs_cparams &              pCparams,
                                                         const dgs_kv_cache_msa_context * pMctx) :
  llm_graph_input_attn_kv(pHparams, pCparams, pMctx->mGetBase()),
  vMctxMsa(pMctx) {}

void llm_graph_input_attn_kv_msa::mSetInput(const dgs_ubatch * pUbatch) {
  llm_graph_input_attn_kv::mSetInput(pUbatch);

  if (vSelfKIdxsIdx) {
    vMctxMsa->mGetIdx()->mSetInputKIdxs(vSelfKIdxsIdx, pUbatch);
  }
}

bool llm_graph_input_attn_kv_msa::mCanReuse(const llm_graph_params & pParams) {
  vMctxMsa = static_cast<const dgs_kv_cache_msa_context *>(pParams.vMctx);

  // the parent class operates on the base cache context
  this->vMctx = vMctxMsa->mGetBase();

  bool vRes = true;

  vRes &= vSelfKIdxs->ne[0] == pParams.vUbatch.vNTokens;
  if (vSelfKIdxsIdx) {
    vRes &= vSelfKIdxsIdx->ne[0] == pParams.vUbatch.vNTokens;
  }

  vRes &= fCanReuseKqMask(vSelfKqMask, this->vMctx, pParams.vUbatch, pParams.vCparams);

  return vRes;
}

void llm_graph_input_attn_k_dsa::mSetInput(const dgs_ubatch * pUbatch) {
  vMctx->mGetMla()->mSetInputKIdxs(vSelfKIdxsMla, pUbatch);

  vMctx->mGetMla()->mSetInputKqMask(vSelfKqMaskMla, pUbatch, cCparams.vCausalAttn);

  vMctx->mGetLid()->mSetInputKIdxs(vSelfKIdxsLid, pUbatch);

  vMctx->mGetLid()->mSetInputKqMask(vSelfKqMaskLid, pUbatch, cCparams.vCausalAttn);

  // left unallocated when the indexer does not use the rotation
  if (vSelfKRotLid && vSelfKRotLid->buffer) {
    vMctx->mGetLid()->mSetInputKRot(vSelfKRotLid);
  }
}

bool llm_graph_input_attn_k_dsa::mCanReuse(const llm_graph_params & pParams) {
  vMctx = static_cast<const dgs_kv_cache_dsa_context *>(pParams.vMctx);

  return mCanReuseImpl(pParams);
}

bool llm_graph_input_attn_k_dsa::mCanReuseImpl(const llm_graph_params & pParams) {
  bool vRes = true;

  vRes &= vSelfKIdxsMla->ne[0] == pParams.vUbatch.vNTokens;
  vRes &= vSelfKIdxsLid->ne[0] == pParams.vUbatch.vNTokens;

  vRes &= fCanReuseKqMask(vSelfKqMaskMla, vMctx->mGetMla(), pParams.vUbatch, pParams.vCparams);
  vRes &= fCanReuseKqMask(vSelfKqMaskLid, vMctx->mGetLid(), pParams.vUbatch, pParams.vCparams);

  return vRes;
}

void llm_graph_input_attn_k_dsa_iswa::mSetInput(const dgs_ubatch * pUbatch) {
  vInpDsa->mSetInput(pUbatch);
  vInpSwa->mSetInput(pUbatch);
}

bool llm_graph_input_attn_k_dsa_iswa::mCanReuse(const llm_graph_params & pParams) {
  vMctx = static_cast<const dgs_kv_cache_dsa_iswa_context *>(pParams.vMctx);

  vInpDsa->vMctx = vMctx->mGetDsa();
  vInpSwa->vMctx = vMctx->mGetSwa();

  bool vRes = true;

  vRes &= vInpDsa->mCanReuseImpl(pParams);
  vRes &= vInpSwa->mCanReuseImpl(pParams);

  return vRes;
}

void llm_graph_input_attn_kv_iswa::mSetInput(const dgs_ubatch * pUbatch) {
  // base tensors may not be allocated if there are no non-SWA attention layers
  if (vSelfKIdxs && vSelfKIdxs->buffer) {
    vMctx->mGetBase()->mSetInputKIdxs(vSelfKIdxs, pUbatch);
    if (vSelfVIdxs) {
      vMctx->mGetBase()->mSetInputVIdxs(vSelfVIdxs, pUbatch);
    }
  }

  // the kq mask guards on its own buffer: shared cells leave idxs unbacked while the mask stays live
  if (vSelfKqMask && vSelfKqMask->buffer) {
    vMctx->mGetBase()->mSetInputKqMask(vSelfKqMask, pUbatch, cCparams.vCausalAttn);
  }

  // swa tensors may not be allocated if there are no SWA attention layers
  if (vSelfKIdxsSwa && vSelfKIdxsSwa->buffer) {
    vMctx->mGetSwa()->mSetInputKIdxs(vSelfKIdxsSwa, pUbatch);
    if (vSelfVIdxsSwa) {
      vMctx->mGetSwa()->mSetInputVIdxs(vSelfVIdxsSwa, pUbatch);
    }
  }

  if (vSelfKqMaskSwa && vSelfKqMaskSwa->buffer) {
    vMctx->mGetSwa()->mSetInputKqMask(vSelfKqMaskSwa, pUbatch, cCparams.vCausalAttn);
  }

  if (vSelfKRot && vSelfKRot->buffer) {
    vMctx->mGetBase()->mSetInputKRot(vSelfKRot);
  }

  if (vSelfVRot && vSelfVRot->buffer) {
    vMctx->mGetBase()->mSetInputVRot(vSelfVRot);
  }

  if (vSelfKRotSwa && vSelfKRotSwa->buffer) {
    vMctx->mGetSwa()->mSetInputKRot(vSelfKRotSwa);
  }

  if (vSelfVRotSwa && vSelfVRotSwa->buffer) {
    vMctx->mGetSwa()->mSetInputVRot(vSelfVRotSwa);
  }
}

bool llm_graph_input_attn_kv_iswa::mCanReuse(const llm_graph_params & pParams) {
  const auto * vMctx = static_cast<const dgs_kv_cache_iswa_context *>(pParams.vMctx);

  this->vMctx = vMctx;

  bool vRes = true;

  // base tensors may not be allocated if there are no non-SWA attention layers
  if (vSelfKIdxs && vSelfKIdxs->buffer) {
    vRes &= vSelfKIdxs->ne[0] == pParams.vUbatch.vNTokens;
    //res &= self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
  }

  if (vSelfKqMask && vSelfKqMask->buffer) {
    vRes &= fCanReuseKqMask(vSelfKqMask, vMctx->mGetBase(), pParams.vUbatch, pParams.vCparams);
  }

  // swa tensors may not be allocated if there are no SWA attention layers
  if (vSelfKIdxsSwa && vSelfKIdxsSwa->buffer) {
    vRes &= vSelfKIdxsSwa->ne[0] == pParams.vUbatch.vNTokens;
    //res &= self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
  }

  if (vSelfKqMaskSwa && vSelfKqMaskSwa->buffer) {
    vRes &= fCanReuseKqMask(vSelfKqMaskSwa, vMctx->mGetSwa(), pParams.vUbatch, pParams.vCparams);
  }

  return vRes;
}

void llm_graph_input_attn_k_iswa::mSetInput(const dgs_ubatch * pUbatch) {
  // base tensors may not be allocated if there are no non-SWA attention layers
  if (vSelfKIdxs && vSelfKIdxs->buffer) {
    vMctx->mGetBase()->mSetInputKIdxs(vSelfKIdxs, pUbatch);
  }

  // the kq mask guards on its own buffer: shared cells leave idxs unbacked while the mask stays live
  if (vSelfKqMask && vSelfKqMask->buffer) {
    vMctx->mGetBase()->mSetInputKqMask(vSelfKqMask, pUbatch, cCparams.vCausalAttn);
  }

  // swa tensors may not be allocated if there are no SWA attention layers
  if (vSelfKIdxsSwa && vSelfKIdxsSwa->buffer) {
    vMctx->mGetSwa()->mSetInputKIdxs(vSelfKIdxsSwa, pUbatch);
  }

  if (vSelfKqMaskSwa && vSelfKqMaskSwa->buffer) {
    vMctx->mGetSwa()->mSetInputKqMask(vSelfKqMaskSwa, pUbatch, cCparams.vCausalAttn);
  }

  if (vSelfKRot && vSelfKRot->buffer) {
    vMctx->mGetBase()->mSetInputKRot(vSelfKRot);
  }

  if (vSelfKRotSwa && vSelfKRotSwa->buffer) {
    vMctx->mGetSwa()->mSetInputKRot(vSelfKRotSwa);
  }
}

bool llm_graph_input_attn_k_iswa::mCanReuse(const llm_graph_params & pParams) {
  const auto * vMctx = static_cast<const dgs_kv_cache_iswa_context *>(pParams.vMctx);

  this->vMctx = vMctx;

  bool vRes = true;

  // base tensors may not be allocated if there are no non-SWA attention layers
  if (vSelfKIdxs && vSelfKIdxs->buffer) {
    vRes &= vSelfKIdxs->ne[0] == pParams.vUbatch.vNTokens;
  }

  if (vSelfKqMask && vSelfKqMask->buffer) {
    vRes &= fCanReuseKqMask(vSelfKqMask, vMctx->mGetBase(), pParams.vUbatch, pParams.vCparams);
  }

  // swa tensors may not be allocated if there are no SWA attention layers
  if (vSelfKIdxsSwa && vSelfKIdxsSwa->buffer) {
    vRes &= vSelfKIdxsSwa->ne[0] == pParams.vUbatch.vNTokens;
  }

  if (vSelfKqMaskSwa && vSelfKqMaskSwa->buffer) {
    vRes &= fCanReuseKqMask(vSelfKqMaskSwa, vMctx->mGetSwa(), pParams.vUbatch, pParams.vCparams);
  }

  return vRes;
}

static void fDsv4SetI64(ggml_tensor * pDst, const std::vector<int64_t> & pSrc) {
  if (!pDst || !pDst->buffer) {
    return;
  }

  GGML_ASSERT(pDst->ne[0] == (int64_t) pSrc.size());
  ggml_backend_tensor_set(pDst, pSrc.data(), 0, pSrc.size() * ggml_element_size(pDst));
}

static void fDsv4SetI32(ggml_tensor * pDst, const std::vector<int32_t> & pSrc) {
  if (!pDst || !pDst->buffer) {
    return;
  }

  GGML_ASSERT(pDst->ne[0] == (int64_t) pSrc.size());
  ggml_backend_tensor_set(pDst, pSrc.data(), 0, pSrc.size() * ggml_element_size(pDst));
}

static void fDsv4SetKqMask(ggml_tensor *                                pDst,
                           const dgs_kv_cache_dsv4_context::comp_plan & pPlan,
                           uint32_t                                     pNTokens,
                           int64_t                                      pNStream) {
  if (!pDst || !pDst->buffer) {
    return;
  }

  GGML_ASSERT(pDst->type == GGML_TYPE_F32 || pDst->type == GGML_TYPE_F16);
  GGML_ASSERT(pNStream > 0);
  GGML_ASSERT(pNTokens % pNStream == 0);
  GGML_ASSERT(pDst->ne[0] == pPlan.vNKv);
  GGML_ASSERT(pDst->ne[1] == (int64_t) pNTokens / pNStream);
  GGML_ASSERT(pDst->ne[2] == 1);
  GGML_ASSERT(pDst->ne[3] == pNStream);
  GGML_ASSERT((int64_t) pPlan.lNVisible.size() == (int64_t) pNTokens);
  GGML_ASSERT(ggml_backend_buffer_is_host(pDst->buffer));

  if (pDst->type == GGML_TYPE_F32) {
    float * data = (float *) pDst->data;

    for (int64_t vI = 0; vI < (int64_t) pNTokens; ++vI) {
      const int32_t cNVisible = pPlan.lNVisible[vI];

      for (int64_t vJ = 0; vJ < pDst->ne[0]; ++vJ) {
        data[vI * pDst->ne[0] + vJ] = vJ < cNVisible ? 0.0f : -INFINITY;
      }
    }
  } else if (pDst->type == GGML_TYPE_F16) {
    ggml_fp16_t *     data      = (ggml_fp16_t *) pDst->data;
    const ggml_fp16_t cFp16Ninf = fDgsCast<ggml_fp16_t>(-INFINITY);
    const ggml_fp16_t cFp16Zero = fDgsCast<ggml_fp16_t>(0.0f);

    for (int64_t vI = 0; vI < (int64_t) pNTokens; ++vI) {
      const int32_t cNVisible = pPlan.lNVisible[vI];

      for (int64_t vJ = 0; vJ < pDst->ne[0]; ++vJ) {
        data[vI * pDst->ne[0] + vJ] = vJ < cNVisible ? cFp16Zero : cFp16Ninf;
      }
    }
  }
}

static ggml_tensor * fDsv4BuildRawKqMask(ggml_context *                        pCtx,
                                         const dgs_kv_cache_dsv4_raw_context * pMctx,
                                         const dgs_ubatch &                    pUbatch,
                                         const dgs_cparams &                   pCparams,
                                         int64_t                               pNStream) {
  const auto cNKv     = pMctx->mGetNKv();
  const auto cNTokens = pUbatch.vNTokens;

  GGML_ASSERT(pNStream > 0);
  GGML_ASSERT(cNTokens % pNStream == 0);

  const auto type = pCparams.vFlashAttn ? GGML_TYPE_F16 : GGML_TYPE_F32;

  ggml_tensor * vRes = ggml_new_tensor_4d(pCtx, type, cNKv, cNTokens / pNStream, 1, pNStream);
  ggml_set_input(vRes);
  ggml_set_name(vRes, "attn_inp_kq_mask");

  return vRes;
}

static bool fDsv4CanReuseRawKqMask(ggml_tensor *                         pKqMask,
                                   const dgs_kv_cache_dsv4_raw_context * pMctx,
                                   const dgs_ubatch &                    pUbatch,
                                   int64_t                               pNStream) {
  const auto cNKv     = pMctx->mGetNKv();
  const auto cNTokens = pUbatch.vNTokens;

  GGML_ASSERT(pNStream > 0);

  bool vRes = true;

  vRes &= (pKqMask->ne[0] == cNKv);
  vRes &= (pKqMask->ne[1] == cNTokens / pNStream);
  vRes &= (pKqMask->ne[2] == 1);
  vRes &= (pKqMask->ne[3] == pNStream);

  return vRes;
}

static std::string fDsv4PlanPositions(const std::vector<int32_t> & pValues) {
  std::ostringstream vSs;
  vSs << "[";
  for (size_t vI = 0; vI < pValues.size(); ++vI) {
    if (vI > 0) {
      vSs << ", ";
    }
    vSs << pValues[vI];
  }
  vSs << "]";
  return vSs.str();
}

static bool fDsv4CompressDebug() {
  static const bool cDebug = []() {
    const char * vEnv = getenv("DGS_DSV4_COMPRESS_DEBUG");
    return vEnv && atoi(vEnv) > 0;
  }();

  return cDebug;
}

static void fDsv4SetCompInputs(const llm_graph_input_dsv4::comp_input &     pInp,
                               const dgs_kv_cache_dsv4_context::comp_plan & pPlan,
                               const char *                                 name,
                               bool                                         pDebug,
                               uint32_t                                     pNTokens,
                               int64_t                                      pNStream) {
  fDsv4SetI32(pInp.vStatePos, pPlan.lStatePos);
  fDsv4SetI32(pInp.vStatePersistSrcIdxs, pPlan.lStatePersistSrcIdxs);
  fDsv4SetI32(pInp.vStatePersistDstIdxs, pPlan.lStatePersistDstIdxs);
  fDsv4SetI32(pInp.vStateRestoreSrcIdxs, pPlan.lStateRestoreSrcIdxs);
  fDsv4SetI32(pInp.vStateRestoreDstIdxs, pPlan.lStateRestoreDstIdxs);
  fDsv4SetI32(pInp.vStateSnapshotSrcIdxs, pPlan.lStateSnapshotSrcIdxs);
  fDsv4SetI32(pInp.vStateSnapshotDstIdxs, pPlan.lStateSnapshotDstIdxs);
  fDsv4SetI32(pInp.vStateReadIdxs, pPlan.lStateReadIdxs);
  fDsv4SetI64(pInp.vStateWriteIdxs, pPlan.lStateWriteIdxs);
  fDsv4SetI32(pInp.vStateWritePos, pPlan.lStateWritePos);
  fDsv4SetKqMask(pInp.vKqMask, pPlan, pNTokens, pNStream);

  if (pDebug || fDsv4CompressDebug()) {
    DGS_LOG_INFO("%s: %s n_tokens=%u, n_stream=%d, state_persist_dst=%s, state_write_pos=%s\n", __func__, name,
                 pNTokens, (int) pNStream, fDsv4PlanPositions(pPlan.lStatePersistDstIdxs).c_str(),
                 fDsv4PlanPositions(pPlan.lStateWritePos).c_str());
  }
}

static bool fDsv4CanReuseTensor1d(ggml_tensor * pT, int64_t pNe0) {
  return (pT == nullptr && pNe0 == 0) || (pT != nullptr && pT->ne[0] == pNe0);
}

static bool fDsv4CanReuseKqMask(ggml_tensor *                                pT,
                                const dgs_kv_cache_dsv4_context::comp_plan & pPlan,
                                uint32_t                                     pNTokens,
                                int64_t                                      pNStream) {
  if (pPlan.vNKv == 0) {
    return pT == nullptr;
  }

  GGML_ASSERT(pNStream > 0);

  return pT != nullptr && pT->ne[0] == pPlan.vNKv && pT->ne[1] == (int64_t) pNTokens / pNStream && pT->ne[2] == 1 &&
         pT->ne[3] == pNStream;
}

static bool fDsv4CanReuseCompInput(const llm_graph_input_dsv4::comp_input &     pInp,
                                   const dgs_kv_cache_dsv4_context::comp_plan & pPlan,
                                   uint32_t                                     pNTokens,
                                   int64_t                                      pNStream) {
  bool vRes = true;
  vRes &= fDsv4CanReuseTensor1d(pInp.vStatePos, pPlan.lStatePos.size());
  vRes &= fDsv4CanReuseTensor1d(pInp.vStatePersistSrcIdxs, pPlan.lStatePersistSrcIdxs.size());
  vRes &= fDsv4CanReuseTensor1d(pInp.vStatePersistDstIdxs, pPlan.lStatePersistDstIdxs.size());
  vRes &= fDsv4CanReuseTensor1d(pInp.vStateRestoreSrcIdxs, pPlan.lStateRestoreSrcIdxs.size());
  vRes &= fDsv4CanReuseTensor1d(pInp.vStateRestoreDstIdxs, pPlan.lStateRestoreDstIdxs.size());
  vRes &= fDsv4CanReuseTensor1d(pInp.vStateSnapshotSrcIdxs, pPlan.lStateSnapshotSrcIdxs.size());
  vRes &= fDsv4CanReuseTensor1d(pInp.vStateSnapshotDstIdxs, pPlan.lStateSnapshotDstIdxs.size());
  vRes &= fDsv4CanReuseTensor1d(pInp.vStateReadIdxs, pPlan.lStateReadIdxs.size());
  vRes &= fDsv4CanReuseTensor1d(pInp.vStateWriteIdxs, pPlan.lStateWriteIdxs.size());
  vRes &= fDsv4CanReuseTensor1d(pInp.vStateWritePos, pPlan.lStateWritePos.size());
  vRes &= fDsv4CanReuseKqMask(pInp.vKqMask, pPlan, pNTokens, pNStream);

  return vRes;
}

static ggml_tensor * fDsv4BuildInput1d(ggml_context * pCtx, ggml_type type, int64_t pNe0, const std::string & name) {
  if (pNe0 == 0) {
    return nullptr;
  }

  ggml_tensor * vRes = ggml_new_tensor_1d(pCtx, type, pNe0);
  ggml_set_input(vRes);
  ggml_set_name(vRes, name.c_str());

  return vRes;
}

static void fDsv4BuildCompInputs(ggml_context *                               pCtx,
                                 llm_graph_input_dsv4::comp_input &           pInp,
                                 const dgs_kv_cache_dsv4_context::comp_plan & pPlan,
                                 const char *                                 name,
                                 const dgs_cparams &                          pCparams,
                                 int64_t                                      pNStream) {
  pInp.vStatePos =
    fDsv4BuildInput1d(pCtx, GGML_TYPE_I32, pPlan.lStatePos.size(), std::string("dsv4_") + name + "_state_pos");
  pInp.vStatePersistSrcIdxs  = fDsv4BuildInput1d(pCtx, GGML_TYPE_I32, pPlan.lStatePersistSrcIdxs.size(),
                                                 std::string("dsv4_") + name + "_state_persist_src_idxs");
  pInp.vStatePersistDstIdxs  = fDsv4BuildInput1d(pCtx, GGML_TYPE_I32, pPlan.lStatePersistDstIdxs.size(),
                                                 std::string("dsv4_") + name + "_state_persist_dst_idxs");
  pInp.vStateRestoreSrcIdxs  = fDsv4BuildInput1d(pCtx, GGML_TYPE_I32, pPlan.lStateRestoreSrcIdxs.size(),
                                                 std::string("dsv4_") + name + "_state_restore_src_idxs");
  pInp.vStateRestoreDstIdxs  = fDsv4BuildInput1d(pCtx, GGML_TYPE_I32, pPlan.lStateRestoreDstIdxs.size(),
                                                 std::string("dsv4_") + name + "_state_restore_dst_idxs");
  pInp.vStateSnapshotSrcIdxs = fDsv4BuildInput1d(pCtx, GGML_TYPE_I32, pPlan.lStateSnapshotSrcIdxs.size(),
                                                 std::string("dsv4_") + name + "_state_snapshot_src_idxs");
  pInp.vStateSnapshotDstIdxs = fDsv4BuildInput1d(pCtx, GGML_TYPE_I32, pPlan.lStateSnapshotDstIdxs.size(),
                                                 std::string("dsv4_") + name + "_state_snapshot_dst_idxs");
  pInp.vStateReadIdxs        = fDsv4BuildInput1d(pCtx, GGML_TYPE_I32, pPlan.lStateReadIdxs.size(),
                                                 std::string("dsv4_") + name + "_state_read_idxs");
  pInp.vStateWriteIdxs       = fDsv4BuildInput1d(pCtx, GGML_TYPE_I64, pPlan.lStateWriteIdxs.size(),
                                                 std::string("dsv4_") + name + "_state_write_idxs");
  pInp.vStateWritePos        = fDsv4BuildInput1d(pCtx, GGML_TYPE_I32, pPlan.lStateWritePos.size(),
                                                 std::string("dsv4_") + name + "_state_write_pos");

  if (pPlan.vNKv > 0) {
    const int64_t cNTokens = (int64_t) pPlan.lNVisible.size();

    GGML_ASSERT(pNStream > 0);
    GGML_ASSERT(cNTokens % pNStream == 0);

    pInp.vKqMask = ggml_new_tensor_4d(
      pCtx,
      (strcmp(name, "lid") != 0 && pCparams.vFlashAttn) || (strcmp(name, "lid") == 0 && pCparams.vFusedLid) ?
        GGML_TYPE_F16 :
        GGML_TYPE_F32,
      pPlan.vNKv, cNTokens / pNStream, 1, pNStream);
    ggml_set_input(pInp.vKqMask);
    ggml_set_name(pInp.vKqMask, (std::string("dsv4_") + name + "_kq_mask").c_str());
  }
}

void llm_graph_input_dsv4_raw::mSetInput(const dgs_ubatch * pUbatch) {
  if (vSelfKIdxs && vSelfKIdxs->buffer) {
    vMctx->mSetInputKIdxs(vSelfKIdxs);
  }

  if (vSelfKqMask && vSelfKqMask->buffer) {
    vMctx->mSetInputKqMask(vSelfKqMask, pUbatch, cCparams.vCausalAttn);
  }

  if (vSelfKRot) {
    vMctx->mSetInputKRot(vSelfKRot);
  }
}

void llm_graph_input_dsv4::mSetInput(const dgs_ubatch * pUbatch) {
  const auto &  vPlanCsa = vMctx->mGetCsaPlan(*pUbatch);
  const auto &  vPlanHca = vMctx->mGetHcaPlan(*pUbatch);
  const auto &  vPlanLid = vMctx->mGetLidPlan(*pUbatch);
  const int64_t cNStream = vPlanCsa.vNStream;

  vInpRaw->vMctx = vMctx->mGetRaw();
  vInpRaw->mSetInput(pUbatch);

  fDsv4SetCompInputs(vInpCsa, vPlanCsa, "csa", vDebug > 0, pUbatch->vNTokens, cNStream);
  fDsv4SetCompInputs(vInpHca, vPlanHca, "hca", vDebug > 0, pUbatch->vNTokens, cNStream);
  fDsv4SetCompInputs(vInpLid, vPlanLid, "lid", vDebug > 0, pUbatch->vNTokens, cNStream);

  if (vInpCsa.vKRot && vInpCsa.vKRot->buffer) {
    vMctx->mGetCsa()->mSetInputKRot(vInpCsa.vKRot);
  }

  if (vInpHca.vKRot && vInpHca.vKRot->buffer) {
    vMctx->mGetHca()->mSetInputKRot(vInpHca.vKRot);
  }

  if (vInpLid.vKRot && vInpLid.vKRot->buffer) {
    vMctx->mGetLid()->mSetInputKRot(vInpLid.vKRot);
  }
}

bool llm_graph_input_dsv4::mCanReuse(const llm_graph_params & pParams) {
  const auto * vMctx = static_cast<const dgs_kv_cache_dsv4_context *>(pParams.vMctx);

  this->vMctx    = vMctx;
  vInpRaw->vMctx = vMctx->mGetRaw();

  bool vRes = true;

  const auto &  vPlanCsa = vMctx->mGetCsaPlan(pParams.vUbatch);
  const auto &  vPlanHca = vMctx->mGetHcaPlan(pParams.vUbatch);
  const auto &  vPlanLid = vMctx->mGetLidPlan(pParams.vUbatch);
  const int64_t cNStream = vPlanCsa.vNStream;

  const auto * vRawCtx = vMctx->mGetRaw();
  vInpRaw->vMctx       = vRawCtx;

  if (vInpRaw->vSelfKIdxs && vInpRaw->vSelfKIdxs->buffer) {
    vRes &= vInpRaw->vSelfKIdxs->ne[0] == vRawCtx->mGetNWrite();
  }
  if (vInpRaw->vSelfKqMask && vInpRaw->vSelfKqMask->buffer) {
    vRes &= fDsv4CanReuseRawKqMask(vInpRaw->vSelfKqMask, vRawCtx, pParams.vUbatch, cNStream);
  }

  vRes &= fDsv4CanReuseCompInput(vInpCsa, vPlanCsa, pParams.vUbatch.vNTokens, cNStream);
  vRes &= fDsv4CanReuseCompInput(vInpHca, vPlanHca, pParams.vUbatch.vNTokens, cNStream);
  vRes &= fDsv4CanReuseCompInput(vInpLid, vPlanLid, pParams.vUbatch.vNTokens, cNStream);

  return vRes;
}

void llm_graph_input_attn_cross::mSetInput(const dgs_ubatch * pUbatch) {
  GGML_ASSERT(vCrossKqMask);

  const int64_t cNEnc    = vCrossKqMask->ne[0];
  const int64_t cNTokens = pUbatch->vNTokens;

  GGML_ASSERT(ggml_backend_buffer_is_host(vCrossKqMask->buffer));
  GGML_ASSERT(!pUbatch->mEqualSeqs());  // TODO: use ubatch->n_seqs instead of failing

  const auto cFillMask = [&](auto * data) {
    using T = std::remove_reference_t<decltype(*data)>;
    for (int vI = 0; vI < cNTokens; ++vI) {
      GGML_ASSERT(!vCross->sSeqIdsEnc.empty() && "dgs_encode must be called first");
      for (int vJ = 0; vJ < cNEnc; ++vJ) {
        float vF = -INFINITY;

        for (int vS = 0; vS < pUbatch->vNSeqId[vI]; ++vS) {
          const dgs_seq_id cSeqId = pUbatch->vSeqId[vI][vS];

          if (vCross->sSeqIdsEnc[vJ].find(cSeqId) != vCross->sSeqIdsEnc[vJ].end()) {
            vF = 0.0f;
          }
        }

        data[vI * cNEnc + vJ] = fDgsCast<T>(vF);
      }
    }
  };

  if (vCrossKqMask->type == GGML_TYPE_F16) {
    cFillMask((ggml_fp16_t *) vCrossKqMask->data);
  } else {
    cFillMask((float *) vCrossKqMask->data);
  }
}

void llm_graph_input_mem_hybrid::mSetInput(const dgs_ubatch * pUbatch) {
  vMctx->mGetAttn()->mSetInputKIdxs(vInpAttn->vSelfKIdxs, pUbatch);
  vMctx->mGetAttn()->mSetInputVIdxs(vInpAttn->vSelfVIdxs, pUbatch);

  vMctx->mGetAttn()->mSetInputKqMask(vInpAttn->vSelfKqMask, pUbatch, cCparams.vCausalAttn);

  if (vInpAttn->vSelfKRot) {
    vMctx->mGetAttn()->mSetInputKRot(vInpAttn->vSelfKRot);
  }

  if (vInpAttn->vSelfVRot) {
    vMctx->mGetAttn()->mSetInputVRot(vInpAttn->vSelfVRot);
  }

  const int64_t cNRs = vMctx->mGetRecr()->mGetNRs();

  if (vInpRs->vSCopy) {
    GGML_ASSERT(ggml_backend_buffer_is_host(vInpRs->vSCopy->buffer));
    int32_t * data = (int32_t *) vInpRs->vSCopy->data;

    // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
    for (uint32_t vI = 0; vI < cNRs; ++vI) {
      data[vI] = vMctx->mGetRecr()->mSCopy(vI);
    }
  }
}

bool llm_graph_input_mem_hybrid::mCanReuse(const llm_graph_params & pParams) {
  const auto * vMctx = static_cast<const dgs_memory_hybrid_context *>(pParams.vMctx);

  this->vMctx = vMctx;

  bool vRes = true;

  vRes &= vInpAttn->vSelfKIdxs->ne[0] == pParams.vUbatch.vNTokens;
  //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there

  vRes &= fCanReuseKqMask(vInpAttn->vSelfKqMask, vMctx->mGetAttn(), pParams.vUbatch, pParams.vCparams);

  vRes &= vInpRs->vSCopy->ne[0] == vMctx->mGetRecr()->mGetNRs();

  vRes &= vInpRs->vSCopyMain->ne[0] == pParams.vUbatch.vNSeqs;

  vRes &= vInpRs->vHead == vMctx->mGetRecr()->mGetHead();
  vRes &= vInpRs->vRsZ == vMctx->mGetRecr()->mGetRsZ();

  return vRes;
}

// TODO: Hybrid input classes are a bit redundant.
// Instead of creating a hybrid input, the graph can simply create 2 separate inputs.
// Refactoring is required in the future.
void llm_graph_input_mem_hybrid_k::mSetInput(const dgs_ubatch * pUbatch) {
  vMctx->mGetAttn()->mSetInputKIdxs(vInpAttn->vSelfKIdxs, pUbatch);

  vMctx->mGetAttn()->mSetInputKqMask(vInpAttn->vSelfKqMask, pUbatch, cCparams.vCausalAttn);

  const int64_t cNRs = vMctx->mGetRecr()->mGetNRs();

  if (vInpRs->vSCopy) {
    GGML_ASSERT(ggml_backend_buffer_is_host(vInpRs->vSCopy->buffer));
    int32_t * data = (int32_t *) vInpRs->vSCopy->data;

    // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
    for (uint32_t vI = 0; vI < cNRs; ++vI) {
      data[vI] = vMctx->mGetRecr()->mSCopy(vI);
    }
  }
}

bool llm_graph_input_mem_hybrid_k::mCanReuse(const llm_graph_params & pParams) {
  const auto * vMctx = static_cast<const dgs_memory_hybrid_context *>(pParams.vMctx);

  this->vMctx = vMctx;

  bool vRes = true;

  vRes &= vInpAttn->vSelfKIdxs->ne[0] == pParams.vUbatch.vNTokens;

  vRes &= fCanReuseKqMask(vInpAttn->vSelfKqMask, vMctx->mGetAttn(), pParams.vUbatch, pParams.vCparams);

  vRes &= vInpRs->vSCopy->ne[0] == vMctx->mGetRecr()->mGetNRs();

  vRes &= vInpRs->vSCopyMain->ne[0] == pParams.vUbatch.vNSeqs;

  vRes &= vInpRs->vHead == vMctx->mGetRecr()->mGetHead();
  vRes &= vInpRs->vRsZ == vMctx->mGetRecr()->mGetRsZ();

  return vRes;
}

void llm_graph_input_mem_hybrid_iswa::mSetInput(const dgs_ubatch * pUbatch) {
  const auto * vAttnCtx = vMctx->mGetAttn();

  // base tensors may not be allocated if there are no non-SWA attention layers
  if (vInpAttn->vSelfKIdxs && vInpAttn->vSelfKIdxs->buffer) {
    vAttnCtx->mGetBase()->mSetInputKIdxs(vInpAttn->vSelfKIdxs, pUbatch);
    vAttnCtx->mGetBase()->mSetInputVIdxs(vInpAttn->vSelfVIdxs, pUbatch);
  }

  if (vInpAttn->vSelfKqMask && vInpAttn->vSelfKqMask->buffer) {
    vAttnCtx->mGetBase()->mSetInputKqMask(vInpAttn->vSelfKqMask, pUbatch, cCparams.vCausalAttn);
  }

  // swa tensors may not be allocated if there are no SWA attention layers
  if (vInpAttn->vSelfKIdxsSwa && vInpAttn->vSelfKIdxsSwa->buffer) {
    vAttnCtx->mGetSwa()->mSetInputKIdxs(vInpAttn->vSelfKIdxsSwa, pUbatch);
    vAttnCtx->mGetSwa()->mSetInputVIdxs(vInpAttn->vSelfVIdxsSwa, pUbatch);
  }

  if (vInpAttn->vSelfKqMaskSwa && vInpAttn->vSelfKqMaskSwa->buffer) {
    vAttnCtx->mGetSwa()->mSetInputKqMask(vInpAttn->vSelfKqMaskSwa, pUbatch, cCparams.vCausalAttn);
  }

  if (vInpAttn->vSelfKRot) {
    vAttnCtx->mGetBase()->mSetInputKRot(vInpAttn->vSelfKRot);
  }

  if (vInpAttn->vSelfVRot) {
    vAttnCtx->mGetBase()->mSetInputVRot(vInpAttn->vSelfVRot);
  }

  if (vInpAttn->vSelfKRotSwa) {
    vAttnCtx->mGetSwa()->mSetInputKRot(vInpAttn->vSelfKRotSwa);
  }

  if (vInpAttn->vSelfVRotSwa) {
    vAttnCtx->mGetSwa()->mSetInputVRot(vInpAttn->vSelfVRotSwa);
  }

  const int64_t cNRs = vMctx->mGetRecr()->mGetNRs();

  if (vInpRs->vSCopy) {
    GGML_ASSERT(ggml_backend_buffer_is_host(vInpRs->vSCopy->buffer));
    int32_t * data = (int32_t *) vInpRs->vSCopy->data;

    // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
    for (uint32_t vI = 0; vI < cNRs; ++vI) {
      data[vI] = vMctx->mGetRecr()->mSCopy(vI);
    }
  }
}

bool llm_graph_input_mem_hybrid_iswa::mCanReuse(const llm_graph_params & pParams) {
  const auto * vMctx = static_cast<const dgs_memory_hybrid_iswa_context *>(pParams.vMctx);

  this->vMctx = vMctx;

  bool vRes = true;

  const auto * vAttnCtx = vMctx->mGetAttn();

  // base tensors may not be allocated if there are no non-SWA attention layers
  if (vInpAttn->vSelfKIdxs && vInpAttn->vSelfKIdxs->buffer) {
    vRes &= vInpAttn->vSelfKIdxs->ne[0] == pParams.vUbatch.vNTokens;
    //res &= inp_attn->self_v_idxs->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
  }

  vRes &= fCanReuseKqMask(vInpAttn->vSelfKqMask, vAttnCtx->mGetBase(), pParams.vUbatch, pParams.vCparams);

  // swa tensors may not be allocated if there are no SWA attention layers
  if (vInpAttn->vSelfKIdxsSwa && vInpAttn->vSelfKIdxsSwa->buffer) {
    vRes &= vInpAttn->vSelfKIdxsSwa->ne[0] == pParams.vUbatch.vNTokens;
    //res &= inp_attn->self_v_idxs_swa->ne[0] == params.ubatch.n_tokens; // TODO: need to move this to the unified cache and check there
  }

  vRes &= fCanReuseKqMask(vInpAttn->vSelfKqMaskSwa, vAttnCtx->mGetSwa(), pParams.vUbatch, pParams.vCparams);

  vRes &= vInpRs->vSCopy->ne[0] == vMctx->mGetRecr()->mGetNRs();

  vRes &= vInpRs->vSCopyMain->ne[0] == pParams.vUbatch.vNSeqs;

  vRes &= vInpRs->vHead == vMctx->mGetRecr()->mGetHead();
  vRes &= vInpRs->vRsZ == vMctx->mGetRecr()->mGetRsZ();

  return vRes;
}

void llm_graph_input_sampling::mSetInput(const dgs_ubatch * pUbatch) {
  // set the inputs only for the active samplers in the current ubatch
  std::unordered_set<dgs_seq_id> sActiveSamplers;
  for (uint32_t vI = 0; vI < pUbatch->vNTokens; vI++) {
    if (pUbatch->vOutput[vI]) {
      dgs_seq_id vSeqId = pUbatch->vSeqId[vI][0];
      sActiveSamplers.insert(vSeqId);
    }
  }

  for (auto vSeqId : sActiveSamplers) {
    if (dSamplers.find(vSeqId) == dSamplers.end()) {
      continue;
    }

    auto & vSampler = dSamplers[vSeqId];

    if (vSampler->vIface->vBackendSetInput) {
      vSampler->vIface->vBackendSetInput(vSampler);
    }
  }
}

bool llm_graph_input_sampling::mCanReuse(const llm_graph_params & pParams) {
  if (dSamplers.size() != pParams.dSamplers.size()) {
    return false;
  }

  for (const auto & [seq_id, sampler] : pParams.dSamplers) {
    if (dSamplers[seq_id] != sampler) {
      return false;
    }
  }

  return true;
}

//
// llm_graph_result
//

llm_graph_result::llm_graph_result(int64_t pMaxNodes) : vMaxNodes(pMaxNodes) {
  reset();

  const char * vDgsGraphResultDebug = getenv("DGS_GRAPH_RESULT_DEBUG");
  vDebug                            = vDgsGraphResultDebug ? atoi(vDgsGraphResultDebug) : 0;
}

int64_t llm_graph_result::mGetMaxNodes() const {
  return vMaxNodes;
}

void llm_graph_result::reset() {
  vTInpTokens  = nullptr;
  vTInpEmbd    = nullptr;
  vTLogits     = nullptr;
  vTEmbd       = nullptr;
  vTEmbdPooled = nullptr;
  vTHNextn     = nullptr;

  lTLayerInp.resize(DGS_MAX_LAYERS + 1);
  std::fill(lTLayerInp.begin(), lTLayerInp.end(), nullptr);

  lTSampled.clear();
  lTSampledProbs.clear();
  lTSampledLogits.clear();
  lTCandidates.clear();

  vParams = {};

  lInputs.clear();
  lFusedNodes.clear();
  dHdmdInputs.clear();

  lBufComputeMeta.resize(ggml_tensor_overhead() * vMaxNodes + ggml_graph_overhead_custom(vMaxNodes, false));

  ggml_init_params vParams = {
    /*.mem_size   =*/lBufComputeMeta.size(),
    /*.mem_buffer =*/lBufComputeMeta.data(),
    /*.no_alloc   =*/true,
  };

  vCtxCompute.reset(ggml_init(vParams));

  vGf = ggml_new_graph_custom(vCtxCompute.get(), vMaxNodes, false);
}

void llm_graph_result::mSetInputs(const dgs_ubatch * pUbatch) {
  for (auto & vInput : lInputs) {
    vInput->mSetInput(pUbatch);
  }
}

void llm_graph_result::mSetOutputs(const llm_graph_params & pParams) {
  if (vTLogits != nullptr) {
    ggml_set_output(vTLogits);
  }
  if (vTEmbd != nullptr) {
    ggml_set_output(vTEmbd);
  }
  if (vTEmbdPooled != nullptr) {
    ggml_set_output(vTEmbdPooled);
  }
  if (vTHNextn != nullptr) {
    ggml_set_output(vTHNextn);
  }
  {
    const auto & lEmbeddingsLayerInp = pParams.vCparams.lEmbeddingsLayerInp;
    for (size_t vIl = 0; vIl < lEmbeddingsLayerInp.size(); ++vIl) {
      if (lEmbeddingsLayerInp[vIl]) {
        GGML_ASSERT(lTLayerInp[vIl] != nullptr && "layer input tensor is null");
        ggml_set_output(lTLayerInp[vIl]);
      }
    }
  }
  for (auto * vTensor : lTSampled) {
    if (vTensor != nullptr) {
      ggml_set_output(vTensor);
    }
  }
  for (auto * vTensor : lTSampledProbs) {
    if (vTensor != nullptr) {
      ggml_set_output(vTensor);
    }
  }
  for (auto * vTensor : lTSampledLogits) {
    if (vTensor != nullptr) {
      ggml_set_output(vTensor);
    }
  }
  for (auto * vTensor : lTCandidates) {
    if (vTensor != nullptr) {
      ggml_set_output(vTensor);
    }
  }
}

bool llm_graph_result::mCanReuse(const llm_graph_params & pParams) {
  if (!this->vParams.mAllowReuse(pParams)) {
    if (vDebug > 1) {
      DGS_LOG_DEBUG("%s: cannot reuse graph due to incompatible graph parameters\n", __func__);
    }

    return false;
  }

  if (vDebug > 1) {
    DGS_LOG_DEBUG("%s: checking compatibility of %d inputs:\n", __func__, (int) lInputs.size());
  }

  bool vRes = true;

  for (auto & vInput : lInputs) {
    const bool cCur = vInput->mCanReuse(pParams);

    if (vDebug > 1) {
      DGS_LOG_DEBUG("%s: can_reuse = %d\n", "placeholder", cCur);
    }

    vRes = vRes && cCur;
  }

  if (vDebug > 0) {
    DGS_LOG_DEBUG("%s: can reuse graph = %d\n", __func__, vRes);
  }

  return vRes;
}

llm_graph_input_i * llm_graph_result::mAddInput(llm_graph_input_ptr pInput) {
  lInputs.emplace_back(std::move(pInput));
  return lInputs.back().get();
}

void llm_graph_result::mAddFusedNode(llm_graph_fused_node pResult) {
  lFusedNodes.push_back(pResult);
}

ggml_tensor * llm_graph_result::mGetHdmdInput(const ggml_tensor * pCur, const ggml_tensor * pRot) const {
  const auto cIt = dHdmdInputs.find({ pCur, pRot });
  return cIt == dHdmdInputs.end() ? nullptr : cIt->second;
}

void llm_graph_result::mSetHdmdInput(const ggml_tensor * pCur, const ggml_tensor * pRot, ggml_tensor * pRes) {
  dHdmdInputs[{ pCur, pRot }] = pRes;
}

void llm_graph_result::mSetParams(const llm_graph_params & pParams) {
  this->vParams = pParams;
}

//
// llm_graph_context
//

llm_graph_context::llm_graph_context(const llm_graph_params & pParams) :
  cArch(pParams.vArch),
  vHparams(pParams.vHparams),
  vCparams(pParams.vCparams),
  vUbatch(pParams.vUbatch),
  cNEmbd(vHparams.n_embd),
  cNLayer(vHparams.n_layer()),
  cNLayerNextn(vHparams.n_layer_nextn),
  cNRot(vHparams.n_rot()),
  cNCtx(vCparams.vNCtx),
  cNHead(vHparams.n_head()),
  cNHeadKv(vHparams.n_head_kv()),
  cNEmbdHeadK(vHparams.n_embd_head_k()),
  cNEmbdKGqa(vHparams.n_embd_k_gqa()),
  cNEmbdHeadV(vHparams.n_embd_head_v()),
  cNEmbdVGqa(vHparams.n_embd_v_gqa()),
  cNExpert(vHparams.n_expert),
  cNExpertUsed(vCparams.vWarmup ? vHparams.n_expert : vHparams.n_expert_used()),
  cFreqBase(vCparams.vRopeFreqBase),
  cFreqScale(vCparams.vRopeFreqScale),
  cExtFactor(vCparams.vYarnExtFactor),
  cAttnFactor(vCparams.vYarnAttnFactor),
  cBetaFast(vCparams.vYarnBetaFast),
  cBetaSlow(vCparams.vYarnBetaSlow),
  cNormEps(vHparams.vFNormEps),
  cNormRmsEps(vHparams.vFNormRmsEps),
  cNTokens(vUbatch.vNTokens),
  cNOutputs(pParams.vNOutputs),
  cNCtxOrig(vCparams.vNCtxOrigYarn),
  cPoolingType(vCparams.vPoolingType),
  cRopeType(vHparams.vRopeType),
  vSched(pParams.vSched),
  vBackendCpu(pParams.vBackendCpu),
  vCvec(pParams.vCvec),
  dLoras(pParams.dLoras),
  vMctx(pParams.vMctx),
  vCross(pParams.vCross),
  vPrecPolicy(pParams.vPrecPolicy),
  vHdmd(pParams.vHdmd),
  dSamplers(pParams.dSamplers),
  vCbFunc(pParams.vCb),
  vRes(pParams.vRes),
  vCtx0(vRes->mGetCtx()),
  vGf(vRes->mGetGf()) {
  vRes->mSetParams(pParams);
}

void llm_graph_context::mCb(ggml_tensor * pCur, const char * name, int pIl) const {
  if (vCbFunc) {
    vCbFunc(vUbatch, pCur, name, pIl);
  }
}

ggml_tensor * llm_graph_context::mBuildCvec(ggml_tensor * pCur, int pIl) const {
  return vCvec->mApplyTo(vCtx0, pCur, pIl);
}

ggml_tensor * llm_graph_context::mBuildHadamardInput(ggml_tensor * pW, ggml_tensor * pCur) const {
  if (!vHdmd) {
    return pCur;
  }
  const auto cIt = vHdmd->dRot.find(pW);
  if (cIt == vHdmd->dRot.end()) {
    return pCur;
  }
  const auto & vT = cIt->second;
  if (ggml_tensor * vX = vRes->mGetHdmdInput(pCur, vT.vRot)) {
    return vX;
  }
  ggml_tensor * vX = pCur;
  if (vT.vPermRep > 1) {
    // tiled [hd, nk, rep] -> grouped [hd, rep, nk] feature order
    vX                 = ggml_is_contiguous(vX) ? vX : ggml_cont(vCtx0, vX);
    const int64_t cNe1 = vX->ne[1], cNe2 = vX->ne[2], cNe3 = vX->ne[3];
    vX = ggml_reshape_4d(vCtx0, vX, vT.vPermHd, vT.vPermNk, vT.vPermRep, cNe1 * cNe2 * cNe3);
    vX = ggml_cont(vCtx0, ggml_permute(vCtx0, vX, 0, 2, 1, 3));
    vX = ggml_reshape_4d(vCtx0, vX, vT.vPermHd * vT.vPermNk * vT.vPermRep, cNe1, cNe2, cNe3);
  }
  if (vT.vSigns) {
    vX = ggml_mul(vCtx0, vX, vT.vSigns);
  }
  vX = fDgsMulMatHadamard(vCtx0, vX, vT.vRot);
  vRes->mSetHdmdInput(pCur, vT.vRot, vX);
  return vX;
}

ggml_tensor * llm_graph_context::mBuildLoraMm(ggml_tensor * pW, ggml_tensor * pCur, ggml_tensor * pWS) const {
  pCur               = mBuildHadamardInput(pW, pCur);
  ggml_tensor * vRes = ggml_mul_mat(vCtx0, pW, pCur);

  if (vPrecPolicy) {
    vPrecPolicy->apply(vRes);
  }

  if (pW->type == GGML_TYPE_NVFP4) {
    ggml_prec_set_acc(vRes, GGML_PREC_BF16);
  }

  if (pWS) {
    vRes = ggml_mul(vCtx0, vRes, pWS);
  }

  for (const auto & lora : *dLoras) {
    dgs_adapter_lora_weight * vLw = lora.first->mGetWeight(pW);
    if (vLw == nullptr) {
      continue;
    }

    const float cAdapterScale = lora.second;
    const float cScale        = vLw->mGetScale(lora.first->vAlpha, cAdapterScale);

    ggml_tensor * vAbCur = ggml_mul_mat(vCtx0, vLw->vB, ggml_mul_mat(vCtx0, vLw->vA, pCur));

    vAbCur = ggml_scale(vCtx0, vAbCur, cScale);
    vRes   = ggml_add(vCtx0, vRes, vAbCur);
  }

  return vRes;
}

ggml_tensor * llm_graph_context::mBuildNorm(ggml_tensor * pCur,
                                            ggml_tensor * pMw,
                                            ggml_tensor * pMb,
                                            llm_norm_type type,
                                            int           pIl) const {
  switch (type) {
    case cLlmNorm:
      pCur = ggml_norm(vCtx0, pCur, vHparams.vFNormEps);
      break;
    case cLlmNormRms:
      pCur = ggml_rms_norm(vCtx0, pCur, vHparams.vFNormRmsEps);
      break;
    case cLlmNormGroup:
      {
        pCur = ggml_reshape_3d(vCtx0, pCur, pCur->ne[0], 1, pCur->ne[1]);
        pCur = ggml_group_norm(vCtx0, pCur, vHparams.vNNormGroups, vHparams.vFNormGroupEps);
        pCur = ggml_reshape_2d(vCtx0, pCur, pCur->ne[0], pCur->ne[2]);
      }
      break;
  }

  if (pMw || pMb) {
    mCb(pCur, "norm", pIl);
  }

  if (pMw) {
    pCur = ggml_mul(vCtx0, pCur, pMw);
    if (pMb) {
      mCb(pCur, "norm_w", pIl);
    }
  }

  if (pMb) {
    pCur = ggml_add(vCtx0, pCur, pMb);
  }

  return pCur;
}

llm_graph_qkv llm_graph_context::build_qkv(const dgs_layer & layer,
                                           ggml_tensor *     pCur,
                                           int64_t           pNEmbdHead,
                                           int64_t           pNHead,
                                           int64_t           pNHeadKv,
                                           int               pIl) const {
  return build_qkv(layer, pCur, pNEmbdHead, pNHead, pNEmbdHead, pNHeadKv, pNEmbdHead, pNHeadKv, pIl);
}

llm_graph_qkv llm_graph_context::build_qkv(const dgs_layer & layer,
                                           ggml_tensor *     pCur,
                                           int64_t           pNEmbdHeadQ,
                                           int64_t           pNHeadQ,
                                           int64_t           pNEmbdHeadK,
                                           int64_t           pNHeadK,
                                           int64_t           pNEmbdHeadV,
                                           int64_t           pNHeadV,
                                           int               pIl,
                                           bool              pReshape) const {
  const int64_t cNEmbdQ = pNEmbdHeadQ * pNHeadQ;
  const int64_t cNEmbdK = pNEmbdHeadK * pNHeadK;

  ggml_tensor *vQcur, *vKcur, *vVcur;

  if (layer.vWqkv) {
    // fused QKV path
    ggml_tensor * vQkv = mBuildLoraMm(layer.vWqkv, pCur, layer.vWqkvS);
    mCb(vQkv, "wqkv", pIl);
    if (layer.vWqkvB) {
      vQkv = ggml_add(vCtx0, vQkv, layer.vWqkvB);
      mCb(vQkv, "wqkv_b", pIl);
    } else if (layer.vWqB && layer.vWkB && layer.vWvB) {
      // Current fused GGUF projections may use separate Q/K/V biases.
      ggml_tensor * vQkvB = ggml_concat(vCtx0, ggml_concat(vCtx0, layer.vWqB, layer.vWkB, 0), layer.vWvB, 0);
      vQkv                = ggml_add(vCtx0, vQkv, vQkvB);
      mCb(vQkv, "wqkv_b", pIl);
    }
    if (pReshape && vHparams.vFClampKqv > 0.0f) {
      vQkv = ggml_clamp(vCtx0, vQkv, -vHparams.vFClampKqv, vHparams.vFClampKqv);
      mCb(vQkv, "wqkv_clamped", pIl);
    }
    if (pReshape) {
      vQcur = ggml_view_3d(vCtx0, vQkv, pNEmbdHeadQ, pNHeadQ, cNTokens, ggml_row_size(vQkv->type, pNEmbdHeadQ),
                           vQkv->nb[1], 0);
      vKcur = ggml_view_3d(vCtx0, vQkv, pNEmbdHeadK, pNHeadK, cNTokens, ggml_row_size(vQkv->type, pNEmbdHeadK),
                           vQkv->nb[1], ggml_row_size(vQkv->type, cNEmbdQ));
      vVcur = ggml_view_3d(vCtx0, vQkv, pNEmbdHeadV, pNHeadV, cNTokens, ggml_row_size(vQkv->type, pNEmbdHeadV),
                           vQkv->nb[1], ggml_row_size(vQkv->type, cNEmbdQ + cNEmbdK));
    } else {
      vQcur = ggml_view_2d(vCtx0, vQkv, cNEmbdQ, cNTokens, vQkv->nb[1], 0);
      vKcur = ggml_view_2d(vCtx0, vQkv, cNEmbdK, cNTokens, vQkv->nb[1], ggml_row_size(vQkv->type, cNEmbdQ));
      vVcur = ggml_view_2d(vCtx0, vQkv, pNEmbdHeadV * pNHeadV, cNTokens, vQkv->nb[1],
                           ggml_row_size(vQkv->type, cNEmbdQ + cNEmbdK));
    }
    if (!pReshape) {
      vQcur = ggml_cont(vCtx0, vQcur);
      vKcur = ggml_cont(vCtx0, vKcur);
      vVcur = ggml_cont(vCtx0, vVcur);
    }
  } else {
    // separate Q/K/V path
    vQcur = mBuildLoraMm(layer.vWq, pCur, layer.vWqS);
    if (pReshape) {
      mCb(vQcur, "Qcur", pIl);
    }
    if (layer.vWqB) {
      vQcur = ggml_add(vCtx0, vQcur, layer.vWqB);
      if (pReshape) {
        mCb(vQcur, "Qcur", pIl);
      }
    }
    if (pReshape && vHparams.vFClampKqv > 0.0f) {
      vQcur = ggml_clamp(vCtx0, vQcur, -vHparams.vFClampKqv, vHparams.vFClampKqv);
      mCb(vQcur, "Qcur_clamped", pIl);
    }
    vKcur = mBuildLoraMm(layer.vWk, pCur, layer.vWkS);
    if (pReshape) {
      mCb(vKcur, "Kcur", pIl);
    }
    if (layer.vWkB) {
      vKcur = ggml_add(vCtx0, vKcur, layer.vWkB);
      if (pReshape) {
        mCb(vKcur, "Kcur", pIl);
      }
    }
    if (pReshape && vHparams.vFClampKqv > 0.0f) {
      vKcur = ggml_clamp(vCtx0, vKcur, -vHparams.vFClampKqv, vHparams.vFClampKqv);
      mCb(vKcur, "Kcur_clamped", pIl);
    }
    vVcur = mBuildLoraMm(layer.vWv, pCur, layer.vWvS);
    if (pReshape) {
      mCb(vVcur, "Vcur", pIl);
    }
    if (layer.vWvB) {
      vVcur = ggml_add(vCtx0, vVcur, layer.vWvB);
      if (pReshape) {
        mCb(vVcur, "Vcur", pIl);
      }
    }
    if (pReshape && vHparams.vFClampKqv > 0.0f) {
      vVcur = ggml_clamp(vCtx0, vVcur, -vHparams.vFClampKqv, vHparams.vFClampKqv);
      mCb(vVcur, "Vcur_clamped", pIl);
    }
    if (pReshape) {
      vQcur = ggml_reshape_3d(vCtx0, vQcur, pNEmbdHeadQ, pNHeadQ, cNTokens);
      vKcur = ggml_reshape_3d(vCtx0, vKcur, pNEmbdHeadK, pNHeadK, cNTokens);
      vVcur = ggml_reshape_3d(vCtx0, vVcur, pNEmbdHeadV, pNHeadV, cNTokens);
    }
  }

  if (pReshape) {
    mCb(vQcur, "Qcur", pIl);
    mCb(vKcur, "Kcur", pIl);
    mCb(vVcur, "Vcur", pIl);
  }

  return { vQcur, vKcur, vVcur };
}

ggml_tensor * llm_graph_context::mBuildFfn(ggml_tensor *     pCur,
                                           ggml_tensor *     pUp,
                                           ggml_tensor *     pUpB,
                                           ggml_tensor *     pUpS,
                                           ggml_tensor *     pGate,
                                           ggml_tensor *     pGateB,
                                           ggml_tensor *     pGateS,
                                           ggml_tensor *     pDown,
                                           ggml_tensor *     pDownB,
                                           ggml_tensor *     pDownS,
                                           ggml_tensor *     pActScales,
                                           llm_ffn_op_type   pTypeOp,
                                           llm_ffn_gate_type pTypeGate,
                                           int               pIl) const {
  // NVFP4 support is currently restricted to
  // 1) LORA absence (*_s would be applied after LORA residual, which is incorrect)
  // 2) bias absense (*_s would be applied after bias addition, which is incorrect)
  // TODO: disambiguate LLM-architectural scales (which use *_s) from NVFP4 scale_2 (which also uses *_s currently)
  auto vHasLora = [this](ggml_tensor * pW) {
    if (!pW) {
      return false;
    }
    for (const auto & lora : *dLoras) {
      if (lora.first->mGetWeight(pW) != nullptr) {
        return true;
      }
    }
    return false;
  };

  GGML_ASSERT(!pUpS || !pUpB || !pUp || pUp->type != GGML_TYPE_NVFP4);
  GGML_ASSERT(!pGateS || !pGateB || !pGate || pGate->type != GGML_TYPE_NVFP4);
  GGML_ASSERT(!pDownS || !pDownB || !pDown || pDown->type != GGML_TYPE_NVFP4);
  GGML_ASSERT(!pUpS || !pUp || pUp->type != GGML_TYPE_NVFP4 || !vHasLora(pUp));
  GGML_ASSERT(!pGateS || !pGate || pGate->type != GGML_TYPE_NVFP4 || !vHasLora(pGate));
  GGML_ASSERT(!pDownS || !pDown || pDown->type != GGML_TYPE_NVFP4 || !vHasLora(pDown));

  ggml_tensor * vTmp = pUp ? mBuildLoraMm(pUp, pCur) : pCur;
  mCb(vTmp, "ffn_up", pIl);

  if (pUpB) {
    vTmp = ggml_add(vCtx0, vTmp, pUpB);
    mCb(vTmp, "ffn_up_b", pIl);
  }

  if (pUpS) {
    vTmp = ggml_mul(vCtx0, vTmp, pUpS);
    mCb(vTmp, "ffn_up_s", pIl);
  }

  if (pGate) {
    switch (pTypeGate) {
      case cLlmFfnSeq:
        {
          pCur = mBuildLoraMm(pGate, vTmp);
          mCb(pCur, "ffn_gate", pIl);
        }
        break;
      case cLlmFfnPar:
        {
          pCur = mBuildLoraMm(pGate, pCur);
          mCb(pCur, "ffn_gate", pIl);
        }
        break;
    }

    if (pGateB) {
      pCur = ggml_add(vCtx0, pCur, pGateB);
      mCb(pCur, "ffn_gate_b", pIl);
    }

    if (pGateS) {
      pCur = ggml_mul(vCtx0, pCur, pGateS);
      mCb(pCur, "ffn_gate_s", pIl);
    }

  } else {
    pCur = vTmp;
  }

  switch (pTypeOp) {
    case cLlmFfnSilu:
      if (pGate && pTypeGate == cLlmFfnPar) {
        if (pIl >= 0) {
          const float     cLimit = vHparams.aSwigluClampShexp[pIl];
          constexpr float cEps   = 1e-6f;
          if (cLimit > cEps) {
            if (cArch == cLlmArchDeepseek4 || cArch == cLlmArchGlm5Next ||
                (cArch == cLlmArchDflash && vHparams.vDsv4HcMult > 0)) {
              pCur = ggml_swiglu_clamp(vCtx0, pCur, vTmp, cLimit);
            } else {
              vTmp = ggml_clamp(vCtx0, vTmp, -cLimit, cLimit);
              mCb(vTmp, "ffn_up_clamped", pIl);
              ggml_tensor * vGateAct = ggml_silu(vCtx0, pCur);
              mCb(vGateAct, "ffn_silu", pIl);
              vGateAct = ggml_clamp(vCtx0, vGateAct, -INFINITY, cLimit);
              mCb(vGateAct, "ffn_silu_clamped", pIl);
              pCur = ggml_mul(vCtx0, vGateAct, vTmp);
            }
            mCb(pCur, "ffn_swiglu_limited", pIl);
            pTypeGate = cLlmFfnSeq;
            break;
          }
        }

        pCur = ggml_swiglu_split(vCtx0, pCur, vTmp);
        mCb(pCur, "ffn_swiglu", pIl);
        pTypeGate = cLlmFfnSeq;
      } else {
        pCur = ggml_silu(vCtx0, pCur);
        mCb(pCur, "ffn_silu", pIl);
      }
      break;
    case cLlmFfnGelu:
      if (pGate && pTypeGate == cLlmFfnPar) {
        pCur = ggml_geglu_split(vCtx0, pCur, vTmp);
        mCb(pCur, "ffn_geglu", pIl);
        pTypeGate = cLlmFfnSeq;
      } else {
        pCur = ggml_gelu(vCtx0, pCur);
        mCb(pCur, "ffn_gelu", pIl);
        if (pActScales != NULL) {
          pCur = ggml_div(vCtx0, pCur, pActScales);
          mCb(pCur, "ffn_act", pIl);
        }
      }
      break;
    case cLlmFfnRelu:
      if (pGate && pTypeGate == cLlmFfnPar) {
        pCur = ggml_reglu_split(vCtx0, pCur, vTmp);
        mCb(pCur, "ffn_reglu", pIl);
        pTypeGate = cLlmFfnSeq;
      } else {
        pCur = ggml_relu(vCtx0, pCur);
        mCb(pCur, "ffn_relu", pIl);
      }
      break;
    case cLlmFfnReluSqr:
      {
        pCur = ggml_relu(vCtx0, pCur);
        mCb(pCur, "ffn_relu", pIl);

        pCur = ggml_sqr(vCtx0, pCur);
        mCb(pCur, "ffn_sqr(relu)", pIl);
      }
      break;
    case cLlmFfnSwiglu:
      {
        pCur = ggml_swiglu(vCtx0, pCur);
        mCb(pCur, "ffn_swiglu", pIl);
      }
      break;
    case cLlmFfnGeglu:
      {
        pCur = ggml_geglu(vCtx0, pCur);
        mCb(pCur, "ffn_geglu", pIl);
      }
      break;
    case cLlmFfnGegluErf:
      {
        pCur = ggml_geglu_erf(vCtx0, pCur);
        mCb(pCur, "ffn_geglu_erf", pIl);
      }
      break;
    case cLlmFfnReglu:
      {
        pCur = ggml_reglu(vCtx0, pCur);
        mCb(pCur, "ffn_reglu", pIl);
      }
      break;
    case cLlmFfnSitu:
      GGML_ABORT("not yet supported");
    default:
      GGML_ABORT("fatal error");
  }

  if (pGate && pTypeGate == cLlmFfnPar) {
    pCur = ggml_mul(vCtx0, pCur, vTmp);
    mCb(pCur, "ffn_gate_par", pIl);
  }

  if (pDown) {
    pCur = mBuildLoraMm(pDown, pCur);
    if (cArch == cLlmArchGlm4 || cArch == cLlmArchGlm4Moe || cArch == cLlmArchJais2) {
      // GLM4, GLM4_MOE, and JAIS2 seem to have numerical issues with half-precision accumulators
      ggml_prec_set_acc(pCur, GGML_PREC_F32);
    }
  }

  if (pDownB) {
    mCb(pCur, "ffn_down", pIl);
  }

  if (pDownB) {
    pCur = ggml_add(vCtx0, pCur, pDownB);
  }

  if (pDownS) {
    pCur = ggml_mul(vCtx0, pCur, pDownS);
    mCb(pCur, "ffn_down_s", pIl);
  }

  return pCur;
}

// input embeddings with optional lora
ggml_tensor * llm_graph_context::mBuildInpEmbd(ggml_tensor * pTokEmbd, float pTokScale) const {
  const int64_t cNEmbdInp = vHparams.mNEmbdInp();
  const int64_t cNEmbd    = vHparams.n_embd;

  assert(n_embd_inp >= n_embd);

  auto vInp = std::make_unique<llm_graph_input_embd>(cNEmbdInp);

  // mixed path (ubatch.is_mixed()): set_rows the token rows into a copy of the embd rows, with its own inputs as select branches must not share tensors
  // TODO: use inp->tokens and inp->embd once ggml_build_forward_select allows it
  const bool cHasMixed = fLlmArchSupportsMixedBatch(cArch) && vCparams.vCtxType == cDgsContextTypeDefault;

  const int64_t cNTokRows = cHasMixed ? fLlmGraphNTokRows(vUbatch) : 0;

  // we have 3 standard paths to produce the input embeddings for the first layer:
  // - embd0: extract from the token embeddings weight (`tok_embd`) using the input token ids
  // - embd1: pass raw embeddings, skipping the `tok_embd`
  // - embd2: mixed path of both tokens ids + raw embeddings (if supported)
  ggml_tensor * vEmbd0 = nullptr;
  ggml_tensor * vEmbd1 = nullptr;
  ggml_tensor * vEmbd2 = nullptr;

  // construct the input tensors
  {
    vInp->vTokens = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, vUbatch.vNTokens);
    mCb(vInp->vTokens, "inp_tokens", -1);
    ggml_set_input(vInp->vTokens);
    vRes->vTInpTokens = vInp->vTokens;

    vInp->vEmbd = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, cNEmbdInp, vUbatch.vNTokens);
    mCb(vInp->vEmbd, "inp_embd", -1);
    ggml_set_input(vInp->vEmbd);

    if (cHasMixed) {
      vInp->vMixedTokens = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNTokRows);
      mCb(vInp->vMixedTokens, "inp_mixed_tokens", -1);
      ggml_set_input(vInp->vMixedTokens);
    }
  }

  // the embeddings placeholders for the 3 paths
  // we use ggml_build_forward_order to make the GET_ROWS ops stick at the beginning of the compute graph
  // this way the embeddings remain in the host buffer, and the GET_ROWS run before any other computations
  {
    vEmbd0 = ggml_get_rows(vCtx0, pTokEmbd, vInp->vTokens);
    ggml_build_forward_order(vGf, vEmbd0);

    vEmbd1 = vInp->vEmbd;

    if (cHasMixed) {
      vEmbd2 = ggml_get_rows(vCtx0, pTokEmbd, vInp->vMixedTokens);
      ggml_build_forward_order(vGf, vEmbd2);
    }
  }

  // helper for extracting token embeddings with lora and padding
  // TODO: when lora is active, this is likely going to cause issues similar to 
  //       need to add lora tests and refactor the logic to make the lora GET_ROWS go at the front of the graph
  auto vBuildTok = [&](ggml_tensor * pCur, ggml_tensor * pIds) {
    // a Hadamard-latent table stores rotated rows: restore the primal basis, h = s * (H z)
    if (vHdmd) {
      if (const auto cIt = vHdmd->dInv.find(pTokEmbd); cIt != vHdmd->dInv.end()) {
        pCur = fDgsMulMatHadamard(vCtx0, pCur, cIt->second.vRot);
        if (cIt->second.vSigns) {
          pCur = ggml_mul(vCtx0, pCur, cIt->second.vSigns);
        }
      }
    }

    // apply lora for embedding tokens if needed
    for (const auto & lora : *dLoras) {
      dgs_adapter_lora_weight * vLw = lora.first->mGetWeight(pTokEmbd);
      if (vLw == nullptr) {
        continue;
      }

      const float cAdapterScale = lora.second;
      const float cScale        = vLw->mGetScale(lora.first->vAlpha, cAdapterScale);

      ggml_tensor * vInpLDelta = ggml_scale(vCtx0,
                                            ggml_mul_mat(vCtx0, vLw->vB,  // non-transposed lora_b
                                                         ggml_get_rows(vCtx0, vLw->vA, pIds)),
                                            cScale);

      pCur = ggml_add(vCtx0, pCur, vInpLDelta);
    }

    if (cNEmbdInp != cNEmbd) {
      pCur = ggml_pad(vCtx0, pCur, vHparams.mNEmbdInp() - cNEmbd, 0, 0, 0);
    }

    return pCur;
  };

  // select one of the 3 inputs, based on the batch contents
  std::array<ggml_tensor *, 3> aInps = {};

  // token embeddings path (ubatch.token != nullptr)
  aInps[0] = vBuildTok(vEmbd0, vInp->vTokens);

  // vector embeddings path (ubatch.embd != nullptr)
  aInps[1] = vEmbd1;

  assert(ggml_are_same_shape(inps[0], inps[1]));
  assert(ggml_are_same_stride(inps[0], inps[1]));

  if (cHasMixed) {
    vInp->vMixedSlots = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I64, cNTokRows);
    mCb(vInp->vMixedSlots, "inp_mixed_slots", -1);
    ggml_set_input(vInp->vMixedSlots);

    vInp->vMixedEmbd = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, cNEmbdInp, vUbatch.vNTokens);
    mCb(vInp->vMixedEmbd, "inp_mixed_embd", -1);
    ggml_set_input(vInp->vMixedEmbd);

    // note: set_rows writes into its destination, so it gets a copy of the input
    ggml_tensor * vEmbdMixed = vBuildTok(vEmbd2, vInp->vMixedTokens);
    aInps[2]                 = ggml_set_rows(vCtx0, ggml_dup(vCtx0, vInp->vMixedEmbd), vEmbdMixed, vInp->vMixedSlots);

    assert(ggml_are_same_shape(inps[0], inps[2]));
    assert(ggml_are_same_stride(inps[0], inps[2]));
  }

  const int cIdx = vUbatch.mIsMixed() ? 2 : vUbatch.token ? 0 : 1;

  ggml_tensor * vCur = ggml_build_forward_select(vGf, aInps.data(), cHasMixed ? 3 : 2, cIdx);

  if (cNEmbdInp != cNEmbd) {
    vCur = ggml_view_2d(vCtx0, vCur, cNEmbd, cNTokens, vCur->nb[1], 0);
  }

  vRes->vTInpEmbd = vCur;

  // NOTE: For deepstack models, only apply scale to token inputs (ie text-only input).
  //  Raw embeddings are assumed to be multimodal inputs that should not be scaled.
  const bool cScaleTokOnly = vHparams.vFEmbeddingScale != 0.0f && vHparams.vNDeepstackLayers > 0;

  // For Granite architecture
  if (vHparams.vFEmbeddingScale != 0.0f && !cScaleTokOnly) {
    if (!ggml_is_contiguous(vCur)) {
      vCur = ggml_cont(vCtx0, vCur);
    }
    vCur = ggml_scale(vCtx0, vCur, vHparams.vFEmbeddingScale);
  }

  // scale the token rows only, applied after the select so that the graph is the same for any batch contents
  vInp->vScaleTok = pTokScale * (cScaleTokOnly ? vHparams.vFEmbeddingScale : 1.0f);
  if (vInp->vScaleTok != 1.0f) {
    vInp->vScaleRows = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, 1, vUbatch.vNTokens);
    mCb(vInp->vScaleRows, "inp_scale_rows", -1);
    ggml_set_input(vInp->vScaleRows);

    if (!ggml_is_contiguous(vCur)) {
      vCur = ggml_cont(vCtx0, vCur);
    }
    vCur = ggml_mul(vCtx0, vCur, vInp->vScaleRows);
  }

  mCb(vCur, "embd", -1);

  vRes->mAddInput(std::move(vInp));

  // make sure the produced embeddings are immediately materialized in the ggml graph
  ggml_build_forward_expand(vGf, vCur);

  return vCur;
}

ggml_tensor * llm_graph_context::mBuildInpPos() const {
  auto vInp = std::make_unique<llm_graph_input_pos>(vHparams.mNPosPerEmbd(), vHparams.vRopeSectionOrder);

  auto & vCur = vInp->vPos;

  vCur = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, (int64_t) cNTokens * vHparams.mNPosPerEmbd());
  ggml_set_input(vCur);
  mCb(vCur, "inp_pos", -1);

  vRes->mAddInput(std::move(vInp));

  return vCur;
}

ggml_tensor * llm_graph_context::mBuildInpAttnScale() const {
  auto vInp = std::make_unique<llm_graph_input_attn_temp>(vHparams.vNAttnTempFloorScale, vHparams.vFAttnTempScale,
                                                          vHparams.vFAttnTempOffset);

  auto & vCur = vInp->vAttnScale;

  // this need to be 1x1xN for broadcasting
  vCur = ggml_new_tensor_3d(vCtx0, GGML_TYPE_F32, 1, 1, cNTokens);
  ggml_set_input(vCur);
  mCb(vCur, "inp_attn_scale", -1);

  vRes->mAddInput(std::move(vInp));

  return vCur;
}

ggml_tensor * llm_graph_context::mBuildInpOutIds() const {
  // note: when all tokens are output, we could skip this optimization to spare the ggml_get_rows() calls,
  //       but this would make the graph topology depend on the number of output tokens, which can interfere with
  //       features that require constant topology such as pipeline parallelism
  //if (n_outputs < n_tokens) {
  //    return nullptr;
  //}

  auto vInp = std::make_unique<llm_graph_input_out_ids>(vHparams, vCparams, cNOutputs);

  auto & vCur = vInp->vOutIds;

  vCur = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNOutputs);
  ggml_set_input(vCur);
  ggml_set_name(vCur, "out_ids");

  vRes->mAddInput(std::move(vInp));

  return vCur;
}

ggml_tensor * llm_graph_context::mBuildInpMean() const {
  auto vInp = std::make_unique<llm_graph_input_mean>(vCparams);

  auto & vCur = vInp->vMean;

  vCur = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, cNTokens, vUbatch.vNSeqsUnq);
  ggml_set_input(vCur);
  ggml_set_name(vCur, "mean");

  vRes->mAddInput(std::move(vInp));

  return vCur;
}

ggml_tensor * llm_graph_context::mBuildInpCls() const {
  auto vInp = std::make_unique<llm_graph_input_cls>(vCparams, cArch);

  auto & vCur = vInp->vCls;

  vCur = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, vUbatch.vNSeqsUnq);
  ggml_set_input(vCur);
  ggml_set_name(vCur, "cls");

  vRes->mAddInput(std::move(vInp));

  return vCur;
}

ggml_tensor * llm_graph_context::mBuildInpCrossEmbd() const {
  auto vInp = std::make_unique<llm_graph_input_cross_embd>(vCross);

  auto & vCur = vInp->vCrossEmbd;

  // if we have the output embeddings from the encoder, use them directly
  // TODO: needs more work to be correct, for now just use the tensor shape
  //if (cross->t_embd) {
  //    cur = ggml_view_tensor(ctx0, cross->t_embd);

  //    return cur;
  //}

  const auto cNEmbd = !vCross->lVEmbd.empty() ? vCross->vNEmbd : vHparams.mNEmbdInp();
  const auto cNEnc  = !vCross->lVEmbd.empty() ? vCross->vNEnc : vHparams.n_ctx_train;

  vCur = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, cNEmbd, cNEnc);
  ggml_set_input(vCur);
  ggml_set_name(vCur, "cross_embd");

  vRes->mAddInput(std::move(vInp));

  return vCur;
}

ggml_tensor * llm_graph_context::mBuildInpPosBucketEnc() const {
  auto vInp = std::make_unique<llm_graph_input_pos_bucket>(vHparams);

  auto & vCur = vInp->vPosBucket;

  vCur = ggml_new_tensor_2d(vCtx0, GGML_TYPE_I32, cNTokens, cNTokens);
  ggml_set_input(vCur);
  ggml_set_name(vCur, "pos_bucket_enc");

  vRes->mAddInput(std::move(vInp));

  return vCur;
}

ggml_tensor * llm_graph_context::mBuildInpPosBucketDec() const {
  const auto * vMctxCur = static_cast<const dgs_kv_cache_context *>(vMctx);

  auto vInp = std::make_unique<llm_graph_input_pos_bucket_kv>(vHparams, vMctxCur);

  const auto cNKv = vMctxCur->mGetNKv();

  auto & vCur = vInp->vPosBucket;

  vCur = ggml_new_tensor_2d(vCtx0, GGML_TYPE_I32, cNKv, cNTokens);
  ggml_set_input(vCur);
  ggml_set_name(vCur, "pos_bucket_dec");

  vRes->mAddInput(std::move(vInp));

  return vCur;
}

ggml_tensor * llm_graph_context::mBuildPosBias(ggml_tensor * pPosBucket, ggml_tensor * pAttnRelB) const {
  ggml_tensor * vPosBucket1d = ggml_reshape_1d(vCtx0, pPosBucket, pPosBucket->ne[0] * pPosBucket->ne[1]);
  mCb(vPosBucket1d, "pos_bucket_1d", -1);

  ggml_tensor * vPosBias = ggml_get_rows(vCtx0, pAttnRelB, vPosBucket1d);

  vPosBias = ggml_reshape_3d(vCtx0, vPosBias, vPosBias->ne[0], pPosBucket->ne[0], pPosBucket->ne[1]);
  vPosBias = ggml_permute(vCtx0, vPosBias, 2, 0, 1, 3);
  vPosBias = ggml_cont(vCtx0, vPosBias);

  mCb(vPosBias, "pos_bias", -1);

  return vPosBias;
}

ggml_tensor * llm_graph_context::mBuildAttnMha(ggml_tensor * pQ,
                                               ggml_tensor * pK,
                                               ggml_tensor * pV,
                                               ggml_tensor * pKqB,
                                               ggml_tensor * pKqMask,
                                               ggml_tensor * pSinks,
                                               ggml_tensor * pVMla,
                                               int64_t       pNKvMax,
                                               float         pKqScale,
                                               int           pIl) const {
  const bool cVTrans = pV->nb[1] > pV->nb[2];

  // split the batch into streams if needed
  const auto cNStream = pK->ne[3];

  // the stream dim steps over one stream's worth of dim-2 (tokens): (ne[2]/n_stream) rows of stride nb[2].
  // q->nb[3]/n_stream only equals that for a contiguous q; nope-only MLA (glm5-next) passes a permuted
  // q_absorbed where nb[3] != ne[2]*nb[2], so using nb[3] read another head's queries for streams s >= 1.
  pQ = ggml_view_4d(vCtx0, pQ, pQ->ne[0], pQ->ne[1], pQ->ne[2] / cNStream, cNStream, pQ->nb[1], pQ->nb[2],
                    pQ->nb[2] * (pQ->ne[2] / cNStream), 0);

  pQ = ggml_permute(vCtx0, pQ, 0, 2, 1, 3);
  pK = ggml_permute(vCtx0, pK, 0, 2, 1, 3);
  pV = ggml_permute(vCtx0, pV, 0, 2, 1, 3);

  ggml_tensor * vCur;

  const bool cUseFlashAttn = vCparams.vFlashAttn && pKqB == nullptr;
  if (cUseFlashAttn) {
    GGML_ASSERT(pKqB == nullptr && "Flash attention does not support KQ bias yet");

    if (cVTrans) {
      pV = ggml_transpose(vCtx0, pV);
    }

    // this can happen when KV cache is not used (e.g. an embedding model with non-causal attn)
    if (pK->type == GGML_TYPE_F32) {
      pK = ggml_cast(vCtx0, pK, GGML_TYPE_F16);
    }

    if (pV->type == GGML_TYPE_F32) {
      pV = ggml_cast(vCtx0, pV, GGML_TYPE_F16);
    }

    vCur = ggml_flash_attn_ext(vCtx0, pQ, pK, pV, pKqMask, pKqScale, vHparams.vFMaxAlibiBias,
                               vHparams.vAttnSoftCap ? vHparams.vFAttnLogitSoftcapping : 0.0f);
    vRes->mAddFusedNode({ cLlmFusedOpFlashAttn, vCur, pIl });

    ggml_flash_attn_ext_add_sinks(vCur, pSinks);
    GGML_ASSERT(pNKvMax >= 0 && pNKvMax <= INT32_MAX);
    ggml_flash_attn_ext_set_n_kv_max(vCur, static_cast<int32_t>(pNKvMax));
    ggml_prec_set_acc(vCur, GGML_PREC_F32);

    if (pVMla) {
#if 0
            // v_mla can be applied as a matrix-vector multiplication with broadcasting across dimension 3 == n_tokens.
            // However, the code is optimized for dimensions 0 and 1 being large, so this is inefficient.
            cur = ggml_reshape_4d(ctx0, cur, v_mla->ne[0], 1, n_head, n_tokens);
            cur = ggml_mul_mat(ctx0, v_mla, cur);
#else
      // It's preferable to do the calculation as a matrix-matrix multiplication with n_tokens in dimension 1.
      // The permutations are noops and only change how the tensor data is interpreted.
      vCur = ggml_permute(vCtx0, vCur, 0, 2, 1, 3);
      vCur = ggml_mul_mat(vCtx0, pVMla, vCur);
      mCb(vCur, "fattn_mla", pIl);
      vCur = ggml_permute(vCtx0, vCur, 0, 2, 1, 3);
      vCur = ggml_cont(vCtx0, vCur);  // Needed because ggml_reshape_2d expects contiguous inputs.
#endif
    }

    vCur = ggml_reshape_2d(vCtx0, vCur, vCur->ne[0] * vCur->ne[1], vCur->ne[2] * vCur->ne[3]);
  } else {
    ggml_tensor * vKq = ggml_mul_mat(vCtx0, pK, pQ);
    mCb(vKq, "kq", pIl);

    // note: this op tends to require high floating point range
    //       while for some models F16 is enough, for others it is not, so we default to F32 here
    ggml_prec_set_acc(vKq, GGML_PREC_F32);

    if (cArch == cLlmArchGrok) {
      // need to do the following:
      // multiply by attn_output_multiplier
      // and then :
      // kq = 30 * tanh(kq / 30)
      // before the softmax below

      vKq = ggml_tanh(vCtx0, ggml_scale(vCtx0, vKq, vHparams.vFAttnOutScale / vHparams.vFAttnLogitSoftcapping));
      mCb(vKq, "kq_tanh", pIl);
      vKq = ggml_scale(vCtx0, vKq, vHparams.vFAttnLogitSoftcapping);
      mCb(vKq, "kq_scaled", pIl);
    }

    if (vHparams.vAttnSoftCap) {
      vKq = ggml_scale(vCtx0, vKq, 1.0f / vHparams.vFAttnLogitSoftcapping);
      mCb(vKq, "kq_scaled_1", pIl);
      vKq = ggml_tanh(vCtx0, vKq);
      mCb(vKq, "kq_tanh", pIl);
      vKq = ggml_scale(vCtx0, vKq, vHparams.vFAttnLogitSoftcapping);
      mCb(vKq, "kq_scaled_2", pIl);
    }

    if (pKqB) {
      vKq = ggml_add(vCtx0, vKq, pKqB);
      mCb(vKq, "kq_plus_kq_b", pIl);
    }

    vKq = ggml_soft_max_ext(vCtx0, vKq, pKqMask, pKqScale, vHparams.vFMaxAlibiBias);
    ggml_soft_max_add_sinks(vKq, pSinks);
    mCb(vKq, "kq_soft_max", pIl);

    if (!cVTrans) {
      // note: avoid this branch
      pV = ggml_cont(vCtx0, ggml_transpose(vCtx0, pV));
      mCb(pV, "v_cont", pIl);
    }

    ggml_tensor * vKqv = ggml_mul_mat(vCtx0, pV, vKq);
    mCb(vKqv, "kqv", pIl);

    // for MLA with the absorption optimization, we need to "decompress" from MQA back to MHA
    if (pVMla) {
      vKqv = ggml_mul_mat(vCtx0, pVMla, vKqv);
      mCb(vKqv, "kqv_mla", pIl);
    }

    vCur = ggml_permute(vCtx0, vKqv, 0, 2, 1, 3);

    // recombine streams
    vCur = ggml_cont_2d(vCtx0, vCur, vCur->ne[0] * vCur->ne[1], vCur->ne[2] * vCur->ne[3]);

    if (!vCparams.vOffloadKqv) {
      // all nodes between the KV store and the attention output are run on the CPU
      ggml_backend_sched_set_tensor_backend(vSched, vCur, vBackendCpu);
    }
  }

  ggml_build_forward_expand(vGf, vCur);

  return vCur;
}

llm_graph_input_attn_no_cache * llm_graph_context::mBuildAttnInpNoCache() const {
  auto vInp = std::make_unique<llm_graph_input_attn_no_cache>(vHparams, vCparams);

  // flash attention requires an f16 mask
  const auto cTypeMask = vCparams.vFlashAttn ? GGML_TYPE_F16 : GGML_TYPE_F32;

  // note: there is no KV cache, so the KV values are the tokens of the batch, padded like the KV cache (see dgs_kv_cache::get_n_kv)
  const int64_t cNKv = GGML_PAD(cNTokens, 256);

  vInp->vSelfKqMask = ggml_new_tensor_4d(vCtx0, cTypeMask, cNKv, cNTokens, 1, 1);
  ggml_set_input(vInp->vSelfKqMask);
  mCb(vInp->vSelfKqMask, "self_kq_mask", -1);

  vInp->vSelfKqMaskCnv = vInp->vSelfKqMask;

  if (vHparams.vSwaType != cDgsSwaTypeNone) {
    vInp->vSelfKqMaskSwa = ggml_new_tensor_4d(vCtx0, cTypeMask, cNKv, cNTokens, 1, 1);
    ggml_set_input(vInp->vSelfKqMaskSwa);

    vInp->vSelfKqMaskSwaCnv = vInp->vSelfKqMaskSwa;
  } else {
    vInp->vSelfKqMaskSwa    = nullptr;
    vInp->vSelfKqMaskSwaCnv = nullptr;
  }

  return (llm_graph_input_attn_no_cache *) vRes->mAddInput(std::move(vInp));
}

ggml_tensor * llm_graph_context::build_attn(llm_graph_input_attn_no_cache * pInp,
                                            ggml_tensor *                   pWo,
                                            ggml_tensor *                   pWoB,
                                            ggml_tensor *                   pWoS,
                                            ggml_tensor *                   pQCur,
                                            ggml_tensor *                   pKCur,
                                            ggml_tensor *                   pVCur,
                                            ggml_tensor *                   pKqB,
                                            ggml_tensor *                   pSinks,
                                            ggml_tensor *                   pVMla,
                                            float                           pKqScale,
                                            int                             pIl) const {
  GGML_UNUSED(cNTokens);

  // these nodes are added to the graph together so that they are not reordered
  // by doing so, the number of splits in the graph is reduced
  ggml_build_forward_expand(vGf, pQCur);
  ggml_build_forward_expand(vGf, pKCur);
  ggml_build_forward_expand(vGf, pVCur);

  const bool cIsSwa = vHparams.mIsSwa(pIl);

  const auto & vKqMask = cIsSwa ? pInp->mGetKqMaskSwa() : pInp->mGetKqMask();

  // [TAG_NO_CACHE_PAD]
  // TODO: if ubatch.equal_seqs() == true, we can split the three tensors below into ubatch.n_seqs_unq streams
  //       but it might not be worth it: 
  //assert(!ubatch.equal_seqs() || (k_cur->ne[3] == 1 && k_cur->ne[3] == ubatch.n_seqs_unq));

  ggml_tensor * vQ = pQCur;
  ggml_tensor * vK = pKCur;
  ggml_tensor * vV = pVCur;

  // the KV values are padded with zeros up to the KV dimension of the mask
  const int64_t cNKvPad = vKqMask->ne[0] - vK->ne[2];

  vK = ggml_pad(vCtx0, vK, 0, 0, cNKvPad, 0);
  vV = ggml_pad(vCtx0, vV, 0, 0, cNKvPad, 0);

  ggml_tensor * vCur = mBuildAttnMha(vQ, vK, vV, pKqB, vKqMask, pSinks, pVMla, 0, pKqScale, pIl);
  mCb(vCur, "kqv_out", pIl);

  if (pWo) {
    vCur = mBuildLoraMm(pWo, vCur, pWoS);
  }

  if (pWoB) {
    //cb(cur, "kqv_wo", il);
  }

  if (pWoB) {
    vCur = ggml_add(vCtx0, vCur, pWoB);
  }

  return vCur;
}

static std::unique_ptr<llm_graph_input_attn_kv> fBuildAttnInpKvImpl(ggml_context *               pCtx0,
                                                                    const dgs_ubatch &           pUbatch,
                                                                    const dgs_hparams &          pHparams,
                                                                    const dgs_cparams &          pCparams,
                                                                    const dgs_kv_cache_context * pMctxCur) {
  auto vInp = std::make_unique<llm_graph_input_attn_kv>(pHparams, pCparams, pMctxCur);

  {
    GGML_ASSERT(pHparams.vSwaType == cDgsSwaTypeNone && "Use dgs_kv_cache_iswa for SWA");

    vInp->vSelfKIdxs = pMctxCur->mBuildInputKIdxs(pCtx0, pUbatch);
    vInp->vSelfVIdxs = pMctxCur->mBuildInputVIdxs(pCtx0, pUbatch);

    vInp->vSelfKqMask    = fBuildAttnInpKqMask(pCtx0, pMctxCur, pUbatch, pCparams);
    vInp->vSelfKqMaskCnv = vInp->vSelfKqMask;
  }

  vInp->vSelfKRot = pMctxCur->mBuildInputKRot(pCtx0);
  vInp->vSelfVRot = pMctxCur->mBuildInputVRot(pCtx0);

  return vInp;
}

llm_graph_input_attn_kv * llm_graph_context::mBuildAttnInpKv() const {
  const auto * vMctxCur = static_cast<const dgs_kv_cache_context *>(vMctx);

  auto vInp = fBuildAttnInpKvImpl(vCtx0, vUbatch, vHparams, vCparams, vMctxCur);

  return (llm_graph_input_attn_kv *) vRes->mAddInput(std::move(vInp));
}

ggml_tensor * llm_graph_context::build_attn(llm_graph_input_attn_kv * pInp,
                                            ggml_tensor *             pWo,
                                            ggml_tensor *             pWoB,
                                            ggml_tensor *             pWoS,
                                            ggml_tensor *             pQCur,
                                            ggml_tensor *             pKCur,
                                            ggml_tensor *             pVCur,
                                            ggml_tensor *             pKqB,
                                            ggml_tensor *             pSinks,
                                            ggml_tensor *             pVMla,  // TODO: remove
                                            float                     pKqScale,
                                            int                       pIl) const {
  GGML_ASSERT(pVMla == nullptr);

  if (pInp->vSelfKRot) {
    pQCur = fDgsMulMatHadamard(vCtx0, pQCur, pInp->vSelfKRot);
    pKCur = fDgsMulMatHadamard(vCtx0, pKCur, pInp->vSelfKRot);
  }

  if (pInp->vSelfVRot) {
    pVCur = fDgsMulMatHadamard(vCtx0, pVCur, pInp->vSelfVRot);
  }

  // these nodes are added to the graph together so that they are not reordered
  // by doing so, the number of splits in the graph is reduced
  // expand k later to enable rope fusion which directly writes into k-v cache
  ggml_build_forward_expand(vGf, pQCur);
  ggml_build_forward_expand(vGf, pVCur);
  ggml_build_forward_expand(vGf, pKCur);

  const auto * vMctxCur = pInp->vMctx;

  ggml_tensor * vQ = pQCur;
  ggml_tensor * vK;
  ggml_tensor * vV;

  {
    {
      const auto & vKIdxs = pInp->mGetKIdxs();
      const auto & vVIdxs = pInp->mGetVIdxs();

      ggml_build_forward_expand(vGf, vMctxCur->mCpyK(vCtx0, pKCur, vKIdxs, pIl));
      ggml_build_forward_expand(vGf, vMctxCur->mCpyV(vCtx0, pVCur, vVIdxs, pIl));
    }

    vK = vMctxCur->mGetK(vCtx0, pIl);
    vV = vMctxCur->mGetV(vCtx0, pIl);
  }

  ggml_tensor * vKqMask = pInp->mGetKqMask();

  ggml_tensor * vCur = mBuildAttnMha(vQ, vK, vV, pKqB, vKqMask, pSinks, pVMla, 0, pKqScale, pIl);
  mCb(vCur, "kqv_out", pIl);

  if (pInp->vSelfVRot) {
    vCur = fDgsMulMatHadamard(vCtx0, vCur, pInp->vSelfVRot);
  }

  if (pWo) {
    if (cArch == cLlmArchGlm4 || cArch == cLlmArchGlm4Moe || cArch == cLlmArchJais2) {
      // GLM4, GLM4_MOE, and JAIS2 seem to have numerical issues with half-precision accumulators
      vCur = mBuildLoraMm(pWo, vCur);
      ggml_prec_set_acc(vCur, GGML_PREC_F32);
      if (pWoS) {
        vCur = ggml_mul(vCtx0, vCur, pWoS);
      }
    } else {
      vCur = mBuildLoraMm(pWo, vCur, pWoS);
    }
  }

  if (pWoB) {
    vCur = ggml_add(vCtx0, vCur, pWoB);
  }

  return vCur;
}

static std::unique_ptr<llm_graph_input_attn_k> fBuildAttnInpKImpl(ggml_context *               pCtx0,
                                                                  const dgs_ubatch &           pUbatch,
                                                                  const dgs_hparams &          pHparams,
                                                                  const dgs_cparams &          pCparams,
                                                                  const dgs_kv_cache_context * pMctxCur) {
  auto vInp = std::make_unique<llm_graph_input_attn_k>(pHparams, pCparams, pMctxCur);

  {
    GGML_ASSERT(pHparams.vSwaType == cDgsSwaTypeNone && "Use dgs_kv_cache_iswa for SWA");

    vInp->vSelfKIdxs = pMctxCur->mBuildInputKIdxs(pCtx0, pUbatch);

    vInp->vSelfKqMask    = fBuildAttnInpKqMask(pCtx0, pMctxCur, pUbatch, pCparams);
    vInp->vSelfKqMaskCnv = vInp->vSelfKqMask;
  }

  return vInp;
}

llm_graph_input_attn_k * llm_graph_context::mBuildAttnInpK() const {
  const auto * vMctxCur = static_cast<const dgs_kv_cache_context *>(vMctx);

  auto vInp = fBuildAttnInpKImpl(vCtx0, vUbatch, vHparams, vCparams, vMctxCur);

  return (llm_graph_input_attn_k *) vRes->mAddInput(std::move(vInp));
}

ggml_tensor * llm_graph_context::build_attn(llm_graph_input_attn_k * pInp,
                                            ggml_tensor *            pWo,
                                            ggml_tensor *            pWoB,
                                            ggml_tensor *            pWoS,
                                            ggml_tensor *            pQCur,
                                            ggml_tensor *            pKCur,
                                            ggml_tensor *            pVCur,
                                            ggml_tensor *            pKqB,
                                            ggml_tensor *            pSinks,
                                            ggml_tensor *            pVMla,
                                            float                    pKqScale,
                                            int                      pIl) const {
  // these nodes are added to the graph together so that they are not reordered
  // by doing so, the number of splits in the graph is reduced
  // expand k later to enable rope fusion which directly writes into k-v cache
  ggml_build_forward_expand(vGf, pQCur);
  ggml_build_forward_expand(vGf, pVCur);
  ggml_build_forward_expand(vGf, pKCur);

  const auto * vMctxCur = pInp->vMctx;

  // store to KV cache
  {
    const auto & vKIdxs = pInp->mGetKIdxs();

    ggml_build_forward_expand(vGf, vMctxCur->mCpyK(vCtx0, pKCur, vKIdxs, pIl));
  }

  const auto & vKqMask = pInp->mGetKqMask();

  ggml_tensor * vQ = pQCur;
  ggml_tensor * vK = vMctxCur->mGetK(vCtx0, pIl);
  ggml_tensor * vV =
    ggml_view_4d(vCtx0, vK, pVCur->ne[0], vK->ne[1], vK->ne[2], vK->ne[3], vK->nb[1], vK->nb[2], vK->nb[3], 0);

  ggml_tensor * vCur = mBuildAttnMha(vQ, vK, vV, pKqB, vKqMask, pSinks, pVMla, 0, pKqScale, pIl);
  mCb(vCur, "kqv_out", pIl);

  if (pWo) {
    if (cArch == cLlmArchGlm4 || cArch == cLlmArchGlm4Moe) {
      // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
      vCur = mBuildLoraMm(pWo, vCur);
      ggml_prec_set_acc(vCur, GGML_PREC_F32);
      if (pWoS) {
        vCur = ggml_mul(vCtx0, vCur, pWoS);
      }
    } else {
      vCur = mBuildLoraMm(pWo, vCur, pWoS);
    }
  }

  if (pWoB) {
    vCur = ggml_add(vCtx0, vCur, pWoB);
  }

  return vCur;
}

ggml_tensor * llm_graph_context::build_attn(llm_graph_input_attn_k_dsa * pInp,
                                            ggml_tensor *                pWo,
                                            ggml_tensor *                pWoB,
                                            ggml_tensor *                pWoS,
                                            ggml_tensor *                pQCur,
                                            ggml_tensor *                pKCur,
                                            ggml_tensor *                pVCur,
                                            ggml_tensor *                pKqB,
                                            ggml_tensor *                pSinks,
                                            ggml_tensor *                pVMla,
                                            ggml_tensor *                pTopK,
                                            float                        pKqScale,
                                            int                          pIl) const {
  // these nodes are added to the graph together so that they are not reordered
  // by doing so, the number of splits in the graph is reduced
  // expand k later to enable rope fusion which directly writes into k-v cache
  ggml_build_forward_expand(vGf, pQCur);
  ggml_build_forward_expand(vGf, pVCur);
  ggml_build_forward_expand(vGf, pKCur);

  const auto * vMctxCur = pInp->vMctx->mGetMla();

  // store to KV cache
  {
    const auto & vKIdxs = pInp->mGetKIdxsMla();

    ggml_build_forward_expand(vGf, vMctxCur->mCpyK(vCtx0, pKCur, vKIdxs, pIl));
  }

  const auto & vKqMask = pInp->mGetKqMaskMla();

  // prepare new kq mask - starts filled with -INFINITY
  ggml_tensor * vKqMaskAll = ggml_fill(vCtx0, vKqMask, -INFINITY);

  // reshape KQ mask into tensor with rows of size 1:
  // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
  vKqMaskAll = ggml_view_4d(vCtx0, vKqMaskAll, 1, vKqMaskAll->ne[0], vKqMaskAll->ne[1], vKqMaskAll->ne[3],
                            vKqMaskAll->nb[0], vKqMaskAll->nb[1], vKqMaskAll->nb[2], 0);

  // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
  ggml_tensor * vTopK3d = ggml_view_4d(vCtx0, pTopK, pTopK->ne[0], pTopK->ne[1], pTopK->ne[3], 1, pTopK->nb[1],
                                       pTopK->nb[2], pTopK->ne[3] * pTopK->nb[3], 0);

  // prepare zero-filled tensor with rows of size 1: [1, n_top_k, n_batch, n_stream]
  // this will be our source of zero values for unmasking top k mask elements
  ggml_tensor * vZeros = ggml_new_tensor_4d(vCtx0, GGML_TYPE_F32, 1, vTopK3d->ne[0], vTopK3d->ne[1], vTopK3d->ne[2]);
  vZeros               = ggml_fill(vCtx0, vZeros, 0.0f);

  // modify KQ mask by unmasking elements that are in top_k indices
  // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
  ggml_tensor * vKqMaskTopK = ggml_set_rows(vCtx0, vKqMaskAll, vZeros, vTopK3d);

  // reshape to restore the original shape of KQ mask:
  // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
  vKqMaskTopK = ggml_view_4d(vCtx0, vKqMaskTopK, vKqMaskTopK->ne[1], vKqMaskTopK->ne[2], 1, vKqMaskTopK->ne[3],
                             vKqMaskTopK->nb[2], vKqMaskTopK->nb[3], vKqMaskTopK->nb[3], 0);

  // combine with the original kq mask
  vKqMaskTopK = ggml_add(vCtx0, vKqMaskTopK, vKqMask);

  ggml_tensor * vQ = pQCur;
  ggml_tensor * vK = vMctxCur->mGetK(vCtx0, pIl);
  ggml_tensor * vV =
    ggml_view_4d(vCtx0, vK, pVCur->ne[0], vK->ne[1], vK->ne[2], vK->ne[3], vK->nb[1], vK->nb[2], vK->nb[3], 0);

  ggml_tensor * vCur = mBuildAttnMha(vQ, vK, vV, pKqB, vKqMaskTopK, pSinks, pVMla, pTopK->ne[0], pKqScale, pIl);
  mCb(vCur, "kqv_out", pIl);

  if (pWo) {
    vCur = mBuildLoraMm(pWo, vCur, pWoS);
  }

  if (pWoB) {
    vCur = ggml_add(vCtx0, vCur, pWoB);
  }

  return vCur;
}

ggml_tensor * llm_graph_context::build_attn(llm_graph_input_attn_kv_iswa * pInp,
                                            ggml_tensor *                  pWo,
                                            ggml_tensor *                  pWoB,
                                            ggml_tensor *                  pWoS,
                                            ggml_tensor *                  pQCur,
                                            ggml_tensor *                  pKCur,
                                            ggml_tensor *                  pVCur,
                                            ggml_tensor *                  pKqB,
                                            ggml_tensor *                  pSinks,
                                            ggml_tensor *                  pVMla,
                                            float                          pKqScale,
                                            int                            pIl) const {
  const bool cIsSwa = vHparams.mIsSwa(pIl);

  auto * vKRot = cIsSwa ? pInp->vSelfKRotSwa : pInp->vSelfKRot;
  auto * vVRot = cIsSwa ? pInp->vSelfVRotSwa : pInp->vSelfVRot;

  if (vKRot) {
    pQCur = fDgsMulMatHadamard(vCtx0, pQCur, vKRot);
    if (pKCur) {
      pKCur = fDgsMulMatHadamard(vCtx0, pKCur, vKRot);
    }
  }
  if (vVRot) {
    if (pVCur) {
      pVCur = fDgsMulMatHadamard(vCtx0, pVCur, vVRot);
    }
  }

  // these nodes are added to the graph together so that they are not reordered
  // by doing so, the number of splits in the graph is reduced
  ggml_build_forward_expand(vGf, pQCur);

  if (pKCur) {
    ggml_build_forward_expand(vGf, pKCur);
  }

  if (pVCur) {
    ggml_build_forward_expand(vGf, pVCur);
  }

  const auto * vMctxIswa = pInp->vMctx;

  const auto * vMctxCur = cIsSwa ? vMctxIswa->mGetSwa() : vMctxIswa->mGetBase();

  // whole seq fits into batch in training mode

  const bool cStoreKv = true;

  // optionally store to KV cache
  if (cStoreKv && pKCur) {
    const auto & vKIdxs = cIsSwa ? pInp->mGetKIdxsSwa() : pInp->mGetKIdxs();

    ggml_build_forward_expand(vGf, vMctxCur->mCpyK(vCtx0, pKCur, vKIdxs, pIl));
  }

  if (cStoreKv && pVCur) {
    const auto & vVIdxs = cIsSwa ? pInp->mGetVIdxsSwa() : pInp->mGetVIdxs();

    ggml_build_forward_expand(vGf, vMctxCur->mCpyV(vCtx0, pVCur, vVIdxs, pIl));
  }

  const auto & vKqMask = cIsSwa ? pInp->mGetKqMaskSwa() : pInp->mGetKqMask();

  ggml_tensor * vQ = pQCur;
  ggml_tensor * vK = vMctxCur->mGetK(vCtx0, pIl);
  ggml_tensor * vV = vMctxCur->mGetV(vCtx0, pIl);

  ggml_tensor * vCur = mBuildAttnMha(vQ, vK, vV, pKqB, vKqMask, pSinks, pVMla, 0, pKqScale, pIl);
  mCb(vCur, "kqv_out", pIl);

  if (vVRot) {
    vCur = fDgsMulMatHadamard(vCtx0, vCur, vVRot);
  }

  if (pWo) {
    vCur = mBuildLoraMm(pWo, vCur, pWoS);
  }

  if (pWoB) {
    //cb(cur, "kqv_wo", il);
  }

  if (pWoB) {
    vCur = ggml_add(vCtx0, vCur, pWoB);
  }

  return vCur;
}

ggml_tensor * llm_graph_context::build_attn(llm_graph_input_attn_k_iswa * pInp,
                                            ggml_tensor *                 pWo,
                                            ggml_tensor *                 pWoB,
                                            ggml_tensor *                 pWoS,
                                            ggml_tensor *                 pQCur,
                                            ggml_tensor *                 pKCur,
                                            ggml_tensor *                 pVCur,
                                            ggml_tensor *                 pKqB,
                                            ggml_tensor *                 pSinks,
                                            ggml_tensor *                 pVMla,
                                            float                         pKqScale,
                                            int                           pIl) const {
  const bool cIsSwa = vHparams.mIsSwa(pIl);

  auto * vKRot = cIsSwa ? pInp->vSelfKRotSwa : pInp->vSelfKRot;

  if (vKRot) {
    pQCur = fDgsMulMatHadamard(vCtx0, pQCur, vKRot);
    if (pKCur) {
      pKCur = fDgsMulMatHadamard(vCtx0, pKCur, vKRot);
    }
  }

  // these nodes are added to the graph together so that they are not reordered
  // by doing so, the number of splits in the graph is reduced
  ggml_build_forward_expand(vGf, pQCur);

  if (pKCur) {
    ggml_build_forward_expand(vGf, pKCur);
  }

  const auto * vMctxIswa = pInp->vMctx;
  const auto * vMctxCur  = cIsSwa ? vMctxIswa->mGetSwa() : vMctxIswa->mGetBase();

  // optionally store to KV cache
  if (pKCur) {
    const auto & vKIdxs = cIsSwa ? pInp->mGetKIdxsSwa() : pInp->mGetKIdxs();

    ggml_build_forward_expand(vGf, vMctxCur->mCpyK(vCtx0, pKCur, vKIdxs, pIl));
  }

  const auto & vKqMask = cIsSwa ? pInp->mGetKqMaskSwa() : pInp->mGetKqMask();

  // MLA-style attention: the cached K is used as V
  ggml_tensor * vQ = pQCur;
  ggml_tensor * vK = vMctxCur->mGetK(vCtx0, pIl);
  ggml_tensor * vV =
    ggml_view_4d(vCtx0, vK, pVCur->ne[0], vK->ne[1], vK->ne[2], vK->ne[3], vK->nb[1], vK->nb[2], vK->nb[3], 0);

  ggml_tensor * vCur = mBuildAttnMha(vQ, vK, vV, pKqB, vKqMask, pSinks, pVMla, 0, pKqScale, pIl);
  mCb(vCur, "kqv_out", pIl);

  if (vKRot) {
    vCur = fDgsMulMatHadamard(vCtx0, vCur, vKRot);
  }

  if (pWo) {
    vCur = mBuildLoraMm(pWo, vCur, pWoS);
  }

  if (pWoB) {
    vCur = ggml_add(vCtx0, vCur, pWoB);
  }

  return vCur;
}

llm_graph_input_attn_cross * llm_graph_context::mBuildAttnInpCross() const {
  auto vInp = std::make_unique<llm_graph_input_attn_cross>(vCross);

  const int32_t cNEnc = !vCross->lVEmbd.empty() ? vCross->vNEnc : vHparams.n_ctx_train;

  // flash attention requires an f16 mask
  const auto cTypeMask = vCparams.vFlashAttn ? GGML_TYPE_F16 : GGML_TYPE_F32;

  vInp->vCrossKqMask = ggml_new_tensor_4d(vCtx0, cTypeMask, cNEnc, cNTokens, 1, 1);
  ggml_set_input(vInp->vCrossKqMask);

  vInp->vCrossKqMaskCnv = vInp->vCrossKqMask;

  return (llm_graph_input_attn_cross *) vRes->mAddInput(std::move(vInp));
}

ggml_tensor * llm_graph_context::build_attn(llm_graph_input_attn_cross * pInp,
                                            ggml_tensor *                pWo,
                                            ggml_tensor *                pWoB,
                                            ggml_tensor *                pWoS,
                                            ggml_tensor *                pQCur,
                                            ggml_tensor *                pKCur,
                                            ggml_tensor *                pVCur,
                                            ggml_tensor *                pKqB,
                                            ggml_tensor *                pSinks,
                                            ggml_tensor *                pVMla,
                                            float                        pKqScale,
                                            int                          pIl) const {
  // these nodes are added to the graph together so that they are not reordered
  // by doing so, the number of splits in the graph is reduced
  ggml_build_forward_expand(vGf, pQCur);
  ggml_build_forward_expand(vGf, pKCur);
  ggml_build_forward_expand(vGf, pVCur);

  const auto & vKqMask = pInp->mGetKqMaskCross();

  ggml_tensor * vQ = pQCur;
  ggml_tensor * vK = pKCur;
  ggml_tensor * vV = pVCur;

  ggml_tensor * vCur = mBuildAttnMha(vQ, vK, vV, pKqB, vKqMask, pSinks, pVMla, 0, pKqScale, pIl);
  mCb(vCur, "kqv_out", pIl);

  if (pWo) {
    vCur = mBuildLoraMm(pWo, vCur, pWoS);
  }

  if (pWoB) {
    //cb(cur, "kqv_wo", il);
  }

  if (pWoB) {
    vCur = ggml_add(vCtx0, vCur, pWoB);
  }

  return vCur;
}

static std::unique_ptr<llm_graph_input_attn_k_dsa> fBuildAttnInpKDsaImpl(ggml_context *                   pCtx0,
                                                                         const dgs_ubatch &               pUbatch,
                                                                         const dgs_hparams &              pHparams,
                                                                         const dgs_cparams &              pCparams,
                                                                         const dgs_kv_cache_dsa_context * pMctxCur) {
  auto vInp = std::make_unique<llm_graph_input_attn_k_dsa>(pHparams, pCparams, pMctxCur);

  {
    vInp->vSelfKIdxsMla = pMctxCur->mGetMla()->mBuildInputKIdxs(pCtx0, pUbatch);

    vInp->vSelfKqMaskMla    = fBuildAttnInpKqMask(pCtx0, pMctxCur->mGetMla(), pUbatch, pCparams);
    vInp->vSelfKqMaskMlaCnv = vInp->vSelfKqMaskMla;
  }

  {
    vInp->vSelfKIdxsLid = pMctxCur->mGetLid()->mBuildInputKIdxs(pCtx0, pUbatch);

    // ensure that mask type matches fused lightning indexer use (requires f16 mask)
    auto vCparamsCopy       = pCparams;
    vCparamsCopy.vFlashAttn = pCparams.vFusedLid;

    vInp->vSelfKqMaskLid    = fBuildAttnInpKqMask(pCtx0, pMctxCur->mGetLid(), pUbatch, vCparamsCopy);
    vInp->vSelfKqMaskLidCnv = vInp->vSelfKqMaskLid;

    vInp->vSelfKRotLid = pMctxCur->mGetLid()->mBuildInputKRot(pCtx0);
  }

  return vInp;
}

llm_graph_input_attn_k_dsa * llm_graph_context::mBuildAttnInpKDsa() const {
  const auto * vMctxCur = static_cast<const dgs_kv_cache_dsa_context *>(vMctx);

  auto vInp = fBuildAttnInpKDsaImpl(vCtx0, vUbatch, vHparams, vCparams, vMctxCur);

  return (llm_graph_input_attn_k_dsa *) vRes->mAddInput(std::move(vInp));
}

llm_graph_input_attn_k_dsa_iswa * llm_graph_context::mBuildAttnInpKDsaIswa() const {
  const auto * vMctxCur = static_cast<const dgs_kv_cache_dsa_iswa_context *>(vMctx);

  auto vInpDsa = fBuildAttnInpKDsaImpl(vCtx0, vUbatch, vHparams, vCparams, vMctxCur->mGetDsa());

  // build_attn_inp_k_impl rejects SWA caches, so construct the input directly
  auto vInpSwa = std::make_unique<llm_graph_input_attn_k>(vHparams, vCparams, vMctxCur->mGetSwa());

  vInpSwa->vSelfKIdxs = vMctxCur->mGetSwa()->mBuildInputKIdxs(vCtx0, vUbatch);

  vInpSwa->vSelfKqMask    = fBuildAttnInpKqMask(vCtx0, vMctxCur->mGetSwa(), vUbatch, vCparams);
  vInpSwa->vSelfKqMaskCnv = vInpSwa->vSelfKqMask;

  auto vInp = std::make_unique<llm_graph_input_attn_k_dsa_iswa>(std::move(vInpDsa), std::move(vInpSwa), vMctxCur);

  return (llm_graph_input_attn_k_dsa_iswa *) vRes->mAddInput(std::move(vInp));
}

llm_graph_input_attn_kv_msa * llm_graph_context::mBuildAttnInpKvMsa(bool pMsaEnabled) const {
  const auto * vMctxCur = static_cast<const dgs_kv_cache_msa_context *>(vMctx);

  auto vInp = std::make_unique<llm_graph_input_attn_kv_msa>(vHparams, vCparams, vMctxCur);

  const auto * vMctxBase = vMctxCur->mGetBase();
  const auto * vMctxIdx  = vMctxCur->mGetIdx();

  {
    GGML_ASSERT(vHparams.vSwaType == cDgsSwaTypeNone && "Use dgs_kv_cache_iswa for SWA");

    vInp->vSelfKIdxs = vMctxBase->mBuildInputKIdxs(vCtx0, vUbatch);
    vInp->vSelfVIdxs = vMctxBase->mBuildInputVIdxs(vCtx0, vUbatch);

    vInp->vSelfKqMask    = fBuildAttnInpKqMask(vCtx0, vMctxBase, vUbatch, vCparams);
    vInp->vSelfKqMaskCnv = vInp->vSelfKqMask;
  }

  vInp->vSelfKRot = vMctxBase->mBuildInputKRot(vCtx0);
  vInp->vSelfVRot = vMctxBase->mBuildInputVRot(vCtx0);

  if (pMsaEnabled) {
    vInp->vSelfKIdxsIdx = vMctxIdx->mBuildInputKIdxs(vCtx0, vUbatch);
  }

  return (llm_graph_input_attn_kv_msa *) vRes->mAddInput(std::move(vInp));
}

// TODO: maybe separate the inner implementation into a separate function
//       like with the non-sliding window equivalent
//       once sliding-window hybrid caches are a thing.
llm_graph_input_attn_kv_iswa * llm_graph_context::mBuildAttnInpKvIswa() const {
  const auto * vMctxCur = static_cast<const dgs_kv_cache_iswa_context *>(vMctx);

  auto vInp = std::make_unique<llm_graph_input_attn_kv_iswa>(vHparams, vCparams, vMctxCur);

  {
    vInp->vSelfKIdxs = vMctxCur->mGetBase()->mBuildInputKIdxs(vCtx0, vUbatch);
    vInp->vSelfVIdxs = vMctxCur->mGetBase()->mBuildInputVIdxs(vCtx0, vUbatch);

    vInp->vSelfKqMask    = fBuildAttnInpKqMask(vCtx0, vMctxCur->mGetBase(), vUbatch, vCparams);
    vInp->vSelfKqMaskCnv = vInp->vSelfKqMask;
  }

  {
    GGML_ASSERT(vHparams.vSwaType != cDgsSwaTypeNone && "Use dgs_kv_cache for non-SWA");

    vInp->vSelfKIdxsSwa = vMctxCur->mGetSwa()->mBuildInputKIdxs(vCtx0, vUbatch);
    vInp->vSelfVIdxsSwa = vMctxCur->mGetSwa()->mBuildInputVIdxs(vCtx0, vUbatch);

    vInp->vSelfKqMaskSwa    = fBuildAttnInpKqMask(vCtx0, vMctxCur->mGetSwa(), vUbatch, vCparams);
    vInp->vSelfKqMaskSwaCnv = vInp->vSelfKqMaskSwa;
  }

  vInp->vSelfKRot = vMctxCur->mGetBase()->mBuildInputKRot(vCtx0);
  vInp->vSelfVRot = vMctxCur->mGetBase()->mBuildInputVRot(vCtx0);

  vInp->vSelfKRotSwa = vMctxCur->mGetSwa()->mBuildInputKRot(vCtx0);
  vInp->vSelfVRotSwa = vMctxCur->mGetSwa()->mBuildInputVRot(vCtx0);

  return (llm_graph_input_attn_kv_iswa *) vRes->mAddInput(std::move(vInp));
}

llm_graph_input_attn_k_iswa * llm_graph_context::mBuildAttnInpKIswa() const {
  const auto * vMctxCur = static_cast<const dgs_kv_cache_iswa_context *>(vMctx);

  auto vInp = std::make_unique<llm_graph_input_attn_k_iswa>(vHparams, vCparams, vMctxCur);

  {
    vInp->vSelfKIdxs = vMctxCur->mGetBase()->mBuildInputKIdxs(vCtx0, vUbatch);

    vInp->vSelfKqMask    = fBuildAttnInpKqMask(vCtx0, vMctxCur->mGetBase(), vUbatch, vCparams);
    vInp->vSelfKqMaskCnv = vInp->vSelfKqMask;
  }

  {
    GGML_ASSERT(vHparams.vSwaType != cDgsSwaTypeNone && "Use dgs_kv_cache for non-SWA");

    vInp->vSelfKIdxsSwa = vMctxCur->mGetSwa()->mBuildInputKIdxs(vCtx0, vUbatch);

    vInp->vSelfKqMaskSwa    = fBuildAttnInpKqMask(vCtx0, vMctxCur->mGetSwa(), vUbatch, vCparams);
    vInp->vSelfKqMaskSwaCnv = vInp->vSelfKqMaskSwa;
  }

  vInp->vSelfKRot = vMctxCur->mGetBase()->mBuildInputKRot(vCtx0);

  vInp->vSelfKRotSwa = vMctxCur->mGetSwa()->mBuildInputKRot(vCtx0);

  return (llm_graph_input_attn_k_iswa *) vRes->mAddInput(std::move(vInp));
}

llm_graph_input_dsv4 * llm_graph_context::mBuildInpDsv4() const {
  const auto * vMctxCur = static_cast<const dgs_kv_cache_dsv4_context *>(vMctx);
  const auto * vRawCtx  = vMctxCur->mGetRaw();

  auto vInpRaw = std::make_unique<llm_graph_input_dsv4_raw>(vCparams, vRawCtx);

  const int64_t cNStream = vMctxCur->mGetCsaPlan(vUbatch).vNStream;

  GGML_ASSERT(vHparams.vSwaType != cDgsSwaTypeNone && "DSV4 expects SWA raw cache");

  vInpRaw->vSelfKIdxs     = vRawCtx->mBuildInputKIdxs(vCtx0, vUbatch);
  vInpRaw->vSelfKqMask    = fDsv4BuildRawKqMask(vCtx0, vRawCtx, vUbatch, vCparams, cNStream);
  vInpRaw->vSelfKqMaskCnv = vInpRaw->vSelfKqMask;

  vInpRaw->vSelfKRot = vRawCtx->mBuildInputKRot(vCtx0);
  auto vInp          = std::make_unique<llm_graph_input_dsv4>(vCparams, std::move(vInpRaw), vMctxCur);

  fDsv4BuildCompInputs(vCtx0, vInp->vInpCsa, vMctxCur->mGetCsaPlan(vUbatch), "csa", vCparams, cNStream);
  fDsv4BuildCompInputs(vCtx0, vInp->vInpHca, vMctxCur->mGetHcaPlan(vUbatch), "hca", vCparams, cNStream);
  fDsv4BuildCompInputs(vCtx0, vInp->vInpLid, vMctxCur->mGetLidPlan(vUbatch), "lid", vCparams, cNStream);
  vInp->vInpCsa.vKRot = vMctxCur->mGetCsa()->mBuildInputKRot(vCtx0);
  vInp->vInpHca.vKRot = vMctxCur->mGetHca()->mBuildInputKRot(vCtx0);
  vInp->vInpLid.vKRot = vMctxCur->mGetLid()->mBuildInputKRot(vCtx0);

  return (llm_graph_input_dsv4 *) vRes->mAddInput(std::move(vInp));
}

ggml_tensor * llm_graph_context::build_rs(ggml_tensor *                 pS,
                                          ggml_tensor *                 pStateCopy,
                                          ggml_tensor *                 pStateCopyMain,
                                          int32_t                       pStateSize,
                                          int32_t                       pNSeqs,
                                          uint32_t                      pNRs,
                                          uint32_t                      pRsHead,
                                          uint32_t                      pRsSize,
                                          int32_t                       pRsZero,
                                          const llm_graph_get_rows_fn & pGetStateRows) const {
  GGML_UNUSED(pRsSize);
  ggml_tensor * vStates = ggml_reshape_2d(vCtx0, pS, pStateSize, pS->ne[1]);

  // Clear a single state which will then be copied to the other cleared states.
  // Note that this is a no-op when the view is zero-sized.
  ggml_tensor * vStateZero =
    ggml_view_1d(vCtx0, vStates, pStateSize * (pRsZero >= 0), pRsZero * vStates->nb[1] * (pRsZero >= 0));
  ggml_build_forward_expand(vGf, ggml_scale_inplace(vCtx0, vStateZero, 0));

  // copy states
  // NOTE: assuming the copy destinations are ALL contained between rs_head and rs_head + n_rs
  // one gather of the states i0..n_rs (ubatch states then extra states), sized by n_rs so the reserve covers every split
  // {state_size, rs_size} -> {state_size, n_rs - i0}
  const int64_t cI0 = pNRs - pStateCopy->ne[0];

  ggml_tensor * vStatesAll = ggml_get_rows(vCtx0, vStates, pStateCopy);

  ggml_tensor * vOutputStates = pGetStateRows ?
                                  pGetStateRows(vCtx0, vStates, pStateCopyMain) :
                                  ggml_view_2d(vCtx0, vStatesAll, pStateSize, pNSeqs, vStatesAll->nb[1], 0);
  ggml_build_forward_expand(vGf, vOutputStates);

  // copy extra states which won't be changed further (between n_seqs and n_rs)
  ggml_tensor * vStatesExtra =
    ggml_view_2d(vCtx0, vStatesAll, pStateSize, pNRs - pNSeqs, vStatesAll->nb[1], (pNSeqs - cI0) * vStatesAll->nb[1]);
  ggml_build_forward_expand(
    vGf, ggml_cpy(vCtx0, vStatesExtra,
                  ggml_view_2d(vCtx0, pS, pStateSize, (pNRs - pNSeqs), pS->nb[1], (pRsHead + pNSeqs) * pS->nb[1])));

  return vOutputStates;
}

static std::unique_ptr<llm_graph_input_rs> fBuildRsInpImpl(ggml_context *                       pCtx0,
                                                           const dgs_ubatch &                   pUbatch,
                                                           const dgs_memory_recurrent_context * pMctxCur) {
  auto vInp = std::make_unique<llm_graph_input_rs>(pMctxCur);

  const int64_t cNRs   = pMctxCur->mGetNRs();
  const int64_t cNSeqs = pUbatch.vNSeqs;

  vInp->vSCopy = ggml_new_tensor_1d(pCtx0, GGML_TYPE_I32, cNRs);
  ggml_set_input(vInp->vSCopy);
  ggml_set_name(vInp->vSCopy, "rs_s_copy");

  vInp->vSCopyMain = ggml_view_1d(pCtx0, vInp->vSCopy, cNSeqs, 0);
  vInp->vSCopyTail = ggml_view_1d(pCtx0, vInp->vSCopy, cNRs - 1, vInp->vSCopy->nb[0]);

  vInp->vHead = pMctxCur->mGetHead();
  vInp->vRsZ  = pMctxCur->mGetRsZ();

  return vInp;
}

llm_graph_input_rs * llm_graph_context::mBuildRsInp() const {
  const auto * vMctxCur = static_cast<const dgs_memory_recurrent_context *>(vMctx);

  auto vInp = fBuildRsInpImpl(vCtx0, vUbatch, vMctxCur);

  return (llm_graph_input_rs *) vRes->mAddInput(std::move(vInp));
}

ggml_tensor * llm_graph_context::build_rs(llm_graph_input_rs *          pInp,
                                          ggml_tensor *                 pS,
                                          int32_t                       pStateSize,
                                          int32_t                       pNSeqs,
                                          const llm_graph_get_rows_fn & pGetStateRows) const {
  const auto * vKvState = pInp->vMctx;

  // a custom getter reads the states of the ubatch straight from the cache, so the gather skips the first
  // state: it still holds the n_rs - n_seqs extra states and copies no state of a single sequence ubatch
  ggml_tensor * vStateCopy = pGetStateRows ? pInp->vSCopyTail : pInp->vSCopy;

  return build_rs(pS, vStateCopy, pInp->vSCopyMain, pStateSize, pNSeqs, vKvState->mGetNRs(), vKvState->mGetHead(),
                  vKvState->mGetSize(), vKvState->mGetRsZ(), pGetStateRows);
}

ggml_tensor * llm_graph_context::mBuildRwkvTokenShiftLoad(llm_graph_input_rs * pInp,
                                                          const dgs_ubatch &   pUbatch,
                                                          int                  pIl) const {
  const auto * vMctxCur = static_cast<const dgs_memory_recurrent_context *>(vMctx);

  const auto cTokenShiftCount = vHparams.vTokenShiftCount;

  const int64_t cNSeqs = pUbatch.vNSeqs;

  ggml_tensor * vTokenShiftAll = vMctxCur->mGetRL(pIl);

  ggml_tensor * vTokenShift = build_rs(pInp, vTokenShiftAll, vHparams.mNEmbdR(), cNSeqs);

  vTokenShift = ggml_reshape_3d(vCtx0, vTokenShift, vHparams.n_embd, cTokenShiftCount, cNSeqs);

  return vTokenShift;
}

ggml_tensor * llm_graph_context::mBuildRwkvTokenShiftStore(ggml_tensor *      pTokenShift,
                                                           const dgs_ubatch & pUbatch,
                                                           int                pIl) const {
  const auto * vMctxCur = static_cast<const dgs_memory_recurrent_context *>(vMctx);

  const auto cTokenShiftCount = vHparams.vTokenShiftCount;
  const auto cNEmbd           = vHparams.n_embd;

  const int64_t cNSeqs = pUbatch.vNSeqs;

  const auto cKvHead = vMctxCur->mGetHead();

  return ggml_cpy(vCtx0, ggml_view_1d(vCtx0, pTokenShift, cNEmbd * cNSeqs * cTokenShiftCount, 0),
                  ggml_view_1d(vCtx0, vMctxCur->mGetRL(pIl), vHparams.mNEmbdR() * cNSeqs,
                               vHparams.mNEmbdR() * cKvHead * ggml_element_size(vMctxCur->mGetRL(pIl))));
}

llm_graph_input_mem_hybrid * llm_graph_context::mBuildInpMemHybrid() const {
  const auto * vMctxCur = static_cast<const dgs_memory_hybrid_context *>(vMctx);

  auto vInpRs   = fBuildRsInpImpl(vCtx0, vUbatch, vMctxCur->mGetRecr());
  auto vInpAttn = fBuildAttnInpKvImpl(vCtx0, vUbatch, vHparams, vCparams, vMctxCur->mGetAttn());

  auto vInp = std::make_unique<llm_graph_input_mem_hybrid>(vCparams, std::move(vInpAttn), std::move(vInpRs), vMctxCur);

  return (llm_graph_input_mem_hybrid *) vRes->mAddInput(std::move(vInp));
}

llm_graph_input_mem_hybrid_k * llm_graph_context::mBuildInpMemHybridK() const {
  const auto * vMctxCur = static_cast<const dgs_memory_hybrid_context *>(vMctx);

  auto vInpRs   = fBuildRsInpImpl(vCtx0, vUbatch, vMctxCur->mGetRecr());
  auto vInpAttn = fBuildAttnInpKImpl(vCtx0, vUbatch, vHparams, vCparams, vMctxCur->mGetAttn());

  auto vInp =
    std::make_unique<llm_graph_input_mem_hybrid_k>(vCparams, std::move(vInpAttn), std::move(vInpRs), vMctxCur);

  return (llm_graph_input_mem_hybrid_k *) vRes->mAddInput(std::move(vInp));
}

llm_graph_input_mem_hybrid_iswa * llm_graph_context::mBuildInpMemHybridIswa() const {
  const auto * vMctxCur = static_cast<const dgs_memory_hybrid_iswa_context *>(vMctx);

  auto vInpRs = fBuildRsInpImpl(vCtx0, vUbatch, vMctxCur->mGetRecr());

  // build iswa attention input
  const auto * vAttnCtx = vMctxCur->mGetAttn();

  auto vInpAttn = std::make_unique<llm_graph_input_attn_kv_iswa>(vHparams, vCparams, vAttnCtx);

  {
    vInpAttn->vSelfKIdxs = vAttnCtx->mGetBase()->mBuildInputKIdxs(vCtx0, vUbatch);
    vInpAttn->vSelfVIdxs = vAttnCtx->mGetBase()->mBuildInputVIdxs(vCtx0, vUbatch);

    vInpAttn->vSelfKqMask    = fBuildAttnInpKqMask(vCtx0, vAttnCtx->mGetBase(), vUbatch, vCparams);
    vInpAttn->vSelfKqMaskCnv = vInpAttn->vSelfKqMask;
  }

  {
    vInpAttn->vSelfKIdxsSwa = vAttnCtx->mGetSwa()->mBuildInputKIdxs(vCtx0, vUbatch);
    vInpAttn->vSelfVIdxsSwa = vAttnCtx->mGetSwa()->mBuildInputVIdxs(vCtx0, vUbatch);

    vInpAttn->vSelfKqMaskSwa    = fBuildAttnInpKqMask(vCtx0, vAttnCtx->mGetSwa(), vUbatch, vCparams);
    vInpAttn->vSelfKqMaskSwaCnv = vInpAttn->vSelfKqMaskSwa;
  }

  auto vInp =
    std::make_unique<llm_graph_input_mem_hybrid_iswa>(vCparams, std::move(vInpAttn), std::move(vInpRs), vMctxCur);

  return (llm_graph_input_mem_hybrid_iswa *) vRes->mAddInput(std::move(vInp));
}

void llm_graph_context::mBuildDenseOut(ggml_tensor * pDense2, ggml_tensor * pDense2B, ggml_tensor * pDense3) const {
  if (!vCparams.vEmbeddings || !(pDense2 || pDense2B || pDense3)) {
    return;
  }
  ggml_tensor * vCur = vRes->vTEmbdPooled != nullptr ? vRes->vTEmbdPooled : vRes->vTEmbd;
  GGML_ASSERT(vCur != nullptr && "missing t_embd_pooled/t_embd");

  if (pDense2) {
    vCur = ggml_mul_mat(vCtx0, pDense2, vCur);
  }
  if (pDense2B) {
    vCur = ggml_add(vCtx0, vCur, pDense2B);
  }
  if (pDense3) {
    vCur = ggml_mul_mat(vCtx0, pDense3, vCur);
  }
  mCb(vCur, "result_embd_pooled", -1);
  vRes->vTEmbdPooled = vCur;
  ggml_build_forward_expand(vGf, vCur);
}

void llm_graph_context::mBuildPooling(ggml_tensor * pCls,
                                      ggml_tensor * pClsB,
                                      ggml_tensor * pClsOut,
                                      ggml_tensor * pClsOutB,
                                      ggml_tensor * pClsNorm) const {
  if (!vCparams.vEmbeddings) {
    return;
  }

  ggml_tensor * vInp = vRes->vTEmbd;

  //// find result_norm tensor for input
  //for (int i = ggml_graph_n_nodes(gf) - 1; i >= 0; --i) {
  //    inp = ggml_graph_node(gf, i);
  //    if (strcmp(inp->name, "result_norm") == 0 || strcmp(inp->name, "result_embd") == 0) {
  //        break;
  //    }

  //    inp = nullptr;
  //}

  GGML_ASSERT(vInp != nullptr && "missing result_norm/result_embd tensor");

  ggml_tensor * vCur;

  switch (cPoolingType) {
    case cDgsPoolingTypeNone:
      {
        vCur = vInp;
      }
      break;
    case cDgsPoolingTypeMean:
      {
        ggml_tensor * vInpMean = mBuildInpMean();
        vCur                   = ggml_mul_mat(vCtx0, ggml_cont(vCtx0, ggml_transpose(vCtx0, vInp)), vInpMean);
      }
      break;
    case cDgsPoolingTypeCls:
    case cDgsPoolingTypeLast:
      {
        ggml_tensor * vInpCls = mBuildInpCls();
        vCur                  = ggml_get_rows(vCtx0, vInp, vInpCls);
      }
      break;
    case cDgsPoolingTypeRank:
      {
        if (vHparams.vPoolingTypeCls == cDgsPoolingTypeMean) {
          // modern bert with classifier_pooling = "mean" builds mean first then applies prediction head and classifier
          // https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modular_modernbert.py#L1404-1411
          ggml_tensor * vInpMean = mBuildInpMean();
          vCur                   = ggml_mul_mat(vCtx0, ggml_cont(vCtx0, ggml_transpose(vCtx0, vInp)), vInpMean);
        } else {
          ggml_tensor * vInpCls = mBuildInpCls();
          vCur                  = ggml_get_rows(vCtx0, vInp, vInpCls);
        }

        // classification head
        // https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/roberta/modeling_roberta.py#L1566
        if (pCls) {
          vCur = ggml_mul_mat(vCtx0, pCls, vCur);
          if (pClsB) {
            vCur = ggml_add(vCtx0, vCur, pClsB);
          }
          vCur = ggml_unary(vCtx0, vCur, vHparams.vActCls);
          if (pClsNorm) {
            // head norm
            vCur = mBuildNorm(vCur, pClsNorm, NULL, cLlmNorm, -1);
          }
        }

        // some models don't have `cls_out`, for example: https://huggingface.co/jinaai/jina-reranker-v1-tiny-en
        // https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/blob/cb5347e43979c3084a890e3f99491952603ae1b7/modeling_bert.py#L884-L896
        // Single layer classification head (direct projection)
        // https://github.com/huggingface/transformers/blob/f4fc42216cd56ab6b68270bf80d811614d8d59e4/src/transformers/models/bert/modeling_bert.py#L1476
        if (pClsOut) {
          vCur = ggml_mul_mat(vCtx0, pClsOut, vCur);
          if (pClsOutB) {
            vCur = ggml_add(vCtx0, vCur, pClsOutB);
          }
        }

        // softmax for qwen3 reranker
        if (cArch == cLlmArchQwen3 || cArch == cLlmArchQwen3vl) {
          vCur = ggml_soft_max(vCtx0, vCur);
        }
      }
      break;
    default:
      {
        GGML_ABORT("unknown pooling type");
      }
  }

  mCb(vCur, "result_embd_pooled", -1);
  vRes->vTEmbdPooled = vCur;

  ggml_build_forward_expand(vGf, vCur);
}

void llm_graph_context::mBuildSampling() const {
  if (dSamplers.empty() || !vRes->vTLogits) {
    return;
  }

  std::array<ggml_tensor *, 2> aOuts;
  aOuts[0] = vRes->vTLogits;

  auto vInpSampling = std::make_unique<llm_graph_input_sampling>(dSamplers);
  vRes->mAddInput(std::move(vInpSampling));

  std::map<dgs_seq_id, std::vector<uint32_t>> dSamplingRows;
  uint32_t                                    vNRows = 0;
  for (uint32_t vI = 0; vI < vUbatch.vNTokens; ++vI) {
    if (vUbatch.vOutput[vI]) {
      dSamplingRows[vUbatch.vSeqId[vI][0]].push_back(vNRows++);
    }
  }

  vRes->lTSampled.resize(vNRows, nullptr);
  vRes->lTSampledProbs.resize(vNRows, nullptr);
  vRes->lTSampledLogits.resize(vNRows, nullptr);
  vRes->lTCandidates.resize(vNRows, nullptr);

  // res->t_logits will contain logits for all tokens that want the logits calculated (logits=1 or output=1)
  GGML_ASSERT(vRes->vTLogits != nullptr && "missing t_logits tensor");

  // the padding row gives the chains without a row of the ubatch a valid input, even with no output at all
  ggml_tensor * vLogitsT = ggml_pad(vCtx0, vRes->vTLogits, 0, 1, 0, 0);

  for (const auto & vEntry : dSamplers) {
    if (vEntry.second->vIface->vBackendReset) {
      vEntry.second->vIface->vBackendReset(vEntry.second);
    }
  }

  static const std::vector<uint32_t> lNoRows;

  // every sampler builds n_outputs_max_per_seq chains, like the reserve, so the graph keeps its topology
  // whatever rows the ubatch outputs: a chain without a row works on the first row and is not selected
  for (const auto & [seq_id, sampler] : dSamplers) {
    const auto   lIt   = dSamplingRows.find(seq_id);
    const auto & lRows = lIt != dSamplingRows.end() ? lIt->second : lNoRows;

    for (uint32_t vI = 0; vI < vCparams.vNOutputsMaxPerSeq; ++vI) {
      const bool     cActive = vI < lRows.size();
      const uint32_t cRow    = cActive ? lRows[vI] : 0;
      const int      cIOut   = cActive ? 1 : 0;

      ggml_tensor * vLogitsSeq = ggml_view_1d(vCtx0, vLogitsT, vLogitsT->ne[0], cRow * vLogitsT->nb[1]);
      ggml_format_name(vLogitsSeq, "logits_seq_%d_%u", seq_id, vI);

      struct dgs_sampler_data data = {
        /*.logits       =*/vLogitsSeq,
        /*.probs        =*/nullptr,
        /*.sampled      =*/nullptr,
        /*.candidates   =*/nullptr,
      };

      assert(sampler->iface->backend_apply);
      sampler->vIface->vBackendApply(sampler, vCtx0, vGf, &data);

      if (data.vSampled != nullptr) {
        if (cActive) {
          vRes->lTSampled[cRow] = data.vSampled;
        }
        aOuts[1] = data.vSampled;
        ggml_build_forward_select(vGf, aOuts.data(), aOuts.size(), cIOut);
      }

      if (data.vProbs != nullptr) {
        if (cActive) {
          vRes->lTSampledProbs[cRow] = data.vProbs;
        }
        aOuts[1] = data.vProbs;
        ggml_build_forward_select(vGf, aOuts.data(), aOuts.size(), cIOut);
      }

      if (data.vLogits != nullptr) {
        if (cActive) {
          vRes->lTSampledLogits[cRow] = data.vLogits;
        }
        aOuts[1] = data.vLogits;
        ggml_build_forward_select(vGf, aOuts.data(), aOuts.size(), cIOut);
      }

      if (data.vCandidates != nullptr) {
        if (cActive) {
          vRes->lTCandidates[cRow] = data.vCandidates;
        }
        aOuts[1] = data.vCandidates;
        ggml_build_forward_select(vGf, aOuts.data(), aOuts.size(), cIOut);
      }
    }
  }

  // TODO: Call backend_accept after all samplers have been applied.
  /*
    for (const auto & [seq_id, sampler] : samplers) {
        const auto it = sampling_rows.find(seq_id);
        if (it == sampling_rows.end()) {
            continue;
        }

        for (uint32_t row : it->second) {
            ggml_tensor * selected_token = res->t_sampled[row];
            if (selected_token != nullptr && sampler->iface->backend_accept) {
                sampler->iface->backend_accept(sampler, ctx0, gf, selected_token);
            }
        }
    }
    */
}

int32_t fDgsRelativePositionBucket(dgs_pos pX, dgs_pos pY, uint64_t pNBuckets, bool pBidirectional) {
  // TODO move to hparams if a T5 variant appears that uses a different value
  const int64_t cMaxDistance = 128;

  if (pBidirectional) {
    pNBuckets >>= 1;
  }

  const int64_t cMaxExact = pNBuckets >> 1;

  int32_t vRelativePosition = pX - pY;
  int32_t vRelativeBucket   = 0;

  if (pBidirectional) {
    vRelativeBucket += (vRelativePosition > 0) * pNBuckets;
    vRelativePosition = std::abs(vRelativePosition);
  } else {
    vRelativePosition = -std::min<int32_t>(vRelativePosition, 0);
  }

  int32_t vRelativePositionIfLarge =
    floorf(cMaxExact +
           logf(1.0 * vRelativePosition / cMaxExact) * (pNBuckets - cMaxExact) / log(1.0 * cMaxDistance / cMaxExact));
  vRelativePositionIfLarge = std::min<int32_t>(vRelativePositionIfLarge, pNBuckets - 1);
  vRelativeBucket += (vRelativePosition < cMaxExact ? vRelativePosition : vRelativePositionIfLarge);

  return vRelativeBucket;
}
