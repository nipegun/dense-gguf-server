#include "dgs-context.h"

#include "dgs-arch.h"
#include "dgs-batch.h"
#include "dgs-ext.h"
#include "dgs-graph.h"
#include "dgs-impl.h"
#include "dgs-io.h"
#include "dgs-memory.h"
#include "dgs-mmap.h"
#include "dgs-model.h"
#include "dgs-sampler.h"
#include "dgs.h"
#include "ggml.h"

#include <cinttypes>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

//
// dgs_context
//

// check that each folded weight in the graph gets its Hadamard transform, and each latent lookup gets the inverse
// without this check, an arch that skips the transform helpers loads and computes wrong results
static void fDgsVerifyHadamardGraph(ggml_cgraph *                  pGf,
                                    const dgs_hadamard_rotations & pRotations,
                                    const dgs_hadamard_rotations & pInverses) {
  const auto & forward = pRotations;

  auto vUnwrap = [](const ggml_tensor * pT) {
    while (pT && (pT->op == GGML_OP_RESHAPE || pT->op == GGML_OP_VIEW)) {
      pT = pT->src[0];
    }
    return pT;
  };

  std::map<const ggml_tensor *, bool> dLookups;  // get_rows results of latent tables

  for (int vI = 0; vI < ggml_graph_n_nodes(pGf); ++vI) {
    const ggml_tensor * node = ggml_graph_node(pGf, vI);

    if (node->op == GGML_OP_GET_ROWS && pInverses.count(node->src[0])) {
      dLookups.emplace(node, false);
      continue;
    }

    if (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_MUL_MAT_ID) {
      continue;
    }

    if (node->op == GGML_OP_MUL_MAT && ((const int32_t *) node->op_params)[1] == GGML_HINT_SRC0_IS_HADAMARD) {
      const auto cLk = dLookups.find(vUnwrap(node->src[1]));
      if (cLk != dLookups.end()) {
        cLk->second = true;
      }
      continue;
    }

    const auto cIt = forward.find(node->src[0]);
    if (cIt == forward.end()) {
      if (pInverses.count(node->src[0])) {
        throw std::runtime_error(
          format("Hadamard-latent table '%s' is used as a head without a forward transform", node->src[0]->name));
      }
      continue;
    }
    const ggml_tensor * vSrc         = vUnwrap(node->src[1]);
    const bool          cTransformed = vSrc && vSrc->op == GGML_OP_MUL_MAT &&
                                       ((const int32_t *) vSrc->op_params)[1] == GGML_HINT_SRC0_IS_HADAMARD &&
                                       vSrc->src[0] == cIt->second.vRot;
    if (!cTransformed) {
      throw std::runtime_error(
        format("Hadamard-folded weight '%s' is consumed without its activation transform; "
               "this graph's matmul path does not support prism.hadamard folding",
               node->src[0]->name));
    }
  }

  for (const auto & [node, ok] : dLookups) {
    if (!ok) {
      for (int vI = 0; vI < ggml_graph_n_nodes(pGf); ++vI) {
        const ggml_tensor * vN2 = ggml_graph_node(pGf, vI);
        for (int vS = 0; vS < GGML_MAX_SRC && vN2->src[vS]; ++vS) {
          if (vUnwrap(vN2->src[vS]) == node) {
            DGS_LOG_WARN("%s: latent lookup '%s' consumed by op=%s name='%s' src%d hint=%d\n", __func__, node->name,
                         ggml_op_name(vN2->op), vN2->name, vS, ((const int32_t *) vN2->op_params)[1]);
          }
        }
      }
      throw std::runtime_error(
        format("Hadamard-latent table '%s' is read without the inverse transform", node->src[0]->name));
    }
  }
}

static llm_graph_type fCtxTypeToGraphType(dgs_context_type pCtxType) {
  switch (pCtxType) {
    case cDgsContextTypeDefault:
      return cLlmGraphTypeDefault;
    case cDgsContextTypeMtp:
      return cLlmGraphTypeDecoderMtp;
  }
  throw std::runtime_error("Unsupported ctx type");
}

struct llm_fused_op_probe {
  llm_fused_op vOp;
  const char * name;
  uint32_t     vNTokensPerSeq;
};

static const llm_fused_op_probe cLlmFusedOpFlashAttnProbe = {
  /*.op               =*/cLlmFusedOpFlashAttn,
  /*.name             =*/"Flash Attention",
  /*.n_tokens_per_seq =*/1,
};

static const llm_fused_op_probe cLlmFusedOpGdnArProbe = {
  /*.op               =*/cLlmFusedOpGdnAr,
  /*.name             =*/"fused Gated Delta Net (autoregressive)",
  /*.n_tokens_per_seq =*/1,
};

static const llm_fused_op_probe cLlmFusedOpGdnChProbe = {
  /*.op               =*/cLlmFusedOpGdnCh,
  /*.name             =*/"fused Gated Delta Net (chunked)",
  /*.n_tokens_per_seq =*/16,
};

static const llm_fused_op_probe cLlmFusedOpLidProbe = {
  /*.op               =*/cLlmFusedOpLightningIndexer,
  /*.name             =*/"Lightning Indexer",
  /*.n_tokens_per_seq =*/1,
};

static const llm_fused_op_probe cLlmFusedOpDsv4HcPreProbe = {
  /*.op               =*/cLlmFusedOpDsv4HcPre,
  /*.name             =*/"fused DeepSeek V4 HC pre",
  /*.n_tokens_per_seq =*/1,
};

static const llm_fused_op_probe cLlmFusedOpDsv4HcCombProbe = {
  /*.op               =*/cLlmFusedOpDsv4HcComb,
  /*.name             =*/"fused DeepSeek V4 HC comb",
  /*.n_tokens_per_seq =*/1,
};

static const llm_fused_op_probe cLlmFusedOpDsv4HcPostProbe = {
  /*.op               =*/cLlmFusedOpDsv4HcPost,
  /*.name             =*/"fused DeepSeek V4 HC post",
  /*.n_tokens_per_seq =*/1,
};

dgs_context::dgs_context(const dgs_model & pModel, dgs_context_params pParams) :
  vModel(pModel),
  vCvec(std::make_unique<dgs_adapter_cvec>()),
  dLoras(std::make_unique<dgs_adapter_loras>()),
  // MTP uses the embd input for the hidden state
  vBalloc(std::make_unique<dgs_batch_allocr>(
    pModel.hparams.mNPosPerEmbd(),
    fLlmArchSupportsMixedBatch(pModel.vArch) && pParams.vCtxType == cDgsContextTypeDefault)) {
  // TODO warning when creating dgs_context with awkward ctx size that is not a power of 2,
  //     may need to be backend-dependent
  DGS_LOG_INFO("%s: constructing dgs_context\n", __func__);

  vTStartUs = pModel.vTStartUs;
  vTLoadUs  = pModel.vTLoadUs;

  const auto & vHparams = pModel.hparams;

  vCparams.vNSeqMax = std::max(1u, pParams.vNSeqMax);
  if (vCparams.vNSeqMax > DGS_MAX_SEQ) {
    throw std::runtime_error("n_seq_max must be <= " + std::to_string(DGS_MAX_SEQ));
  }

  vCparams.vNRsSeq = pParams.vNRsSeq;
  if (vCparams.vNRsSeq > 0 && !fLlmArchSupportsRsRollback(pModel.vArch)) {
    DGS_LOG_DEBUG("%s: n_rs_seq=%u requested but model does not support recurrent partial rollback; clamping to 0\n",
                  __func__, vCparams.vNRsSeq);
    vCparams.vNRsSeq = 0;
  }

  vCparams.vNThreads        = pParams.vNThreads;
  vCparams.vNThreadsBatch   = pParams.vNThreadsBatch;
  vCparams.vYarnExtFactor   = pParams.vYarnExtFactor >= 0.0f ? pParams.vYarnExtFactor : vHparams.vYarnExtFactor;
  vCparams.vYarnAttnFactor  = pParams.vYarnAttnFactor >= 0.0f ? pParams.vYarnAttnFactor : vHparams.vYarnAttnFactor;
  vCparams.vYarnBetaFast    = pParams.vYarnBetaFast >= 0.0f ? pParams.vYarnBetaFast : vHparams.vYarnBetaFast;
  vCparams.vYarnBetaSlow    = pParams.vYarnBetaSlow >= 0.0f ? pParams.vYarnBetaSlow : vHparams.vYarnBetaSlow;
  vCparams.vEmbeddings      = pParams.vEmbeddings;
  vCparams.vEmbeddingsNextn = false;
  vCparams.vEmbeddingsNextnMasked = false;
  vCparams.vOffloadKqv            = pParams.vOffloadKqv;
  vCparams.vNoPerf                = pParams.vNoPerf;
  vCparams.vWarmup                = false;

  // +1: id n_layer() taps the output of the last layer ("input" of the head)
  vCparams.lEmbeddingsLayerInp.resize(vHparams.n_layer() + 1, false);
  lEmbdLayerInp.resize(vHparams.n_layer() + 1);

  vCparams.vCtxType         = pParams.vCtxType;
  vCparams.vRopeScalingType = pParams.vRopeScalingType;
  vCparams.vPoolingType     = pParams.vPoolingType;

  vCparams.vNCtx          = pParams.vNCtx == 0 ? vHparams.n_ctx_train : pParams.vNCtx;
  vCparams.vRopeFreqBase  = pParams.vRopeFreqBase == 0.0f ? vHparams.vRopeFreqBaseTrain : pParams.vRopeFreqBase;
  vCparams.vRopeFreqScale = pParams.vRopeFreqScale == 0.0f ? vHparams.vRopeFreqScaleTrain : pParams.vRopeFreqScale;

  vCparams.vNCtxOrigYarn = pParams.vYarnOrigCtx != 0   ? pParams.vYarnOrigCtx :
                           vHparams.vNCtxOrigYarn != 0 ? vHparams.vNCtxOrigYarn :
                                                         vHparams.n_ctx_train;

  vCparams.vCbEval         = pParams.vCbEval;
  vCparams.vCbEvalUserData = pParams.vCbEvalUserData;

  vCparams.vCtxOther = nullptr;

  // TODO: more generic
  if (pModel.vArch == cLlmArchGemma4Assistant) {
    if (pParams.vCtxOther == nullptr) {
      // TODO: change from runtime_error to dgs_exception to avoid printing error message
      throw std::runtime_error(
        "Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)");
    }

    vCparams.vCtxOther = pParams.vCtxOther;
  }

  if (pModel.vArch == cLlmArchEagle3 || pModel.vArch == cLlmArchDflash) {
    if (pModel.vTokEmbd == nullptr || pModel.vOutput == nullptr) {
      if (pParams.vCtxOther == nullptr) {
        throw std::runtime_error(pModel.mArchName() +
                                 " requires ctx_other to be set (this warning is normal during memory fitting)");
      }
      vCparams.vCtxOther = pParams.vCtxOther;
    }
  }

  if (vCparams.vRopeScalingType == cDgsRopeScalingTypeUnspecified) {
    vCparams.vRopeScalingType = vHparams.vRopeScalingTypeTrain;
  }

  if (vCparams.vRopeScalingType == cDgsRopeScalingTypeNone) {
    vCparams.vRopeFreqScale = 1.0f;  // never scale if scaling type is none
  }

  if (vCparams.vYarnExtFactor < 0.0f) {  // negative indicates 'not set'
    vCparams.vYarnExtFactor = vCparams.vRopeScalingType == cDgsRopeScalingTypeYarn ? 1.0f : 0.0f;
  }

  if (vCparams.vYarnExtFactor != 0) {
    static auto vGetMscale = [](float pScale, float pMscale) {
      return pScale <= 1.0f ? 1.0f : (0.1f * pMscale * logf(pScale) + 1.0f);
    };

    const float cFactor = 1.0f / vCparams.vRopeFreqScale;

    // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
    if (vHparams.vRopeYarnLogMul != 0.0f) {
      // note: here we assume `mscale == 1.0f`
      // TODO: start reading the actual value of mscale and handle the case where it is not 1.0f
      float       vMscale        = 1.0f;
      const float cMscaleAllDims = vHparams.vRopeYarnLogMul;

      // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
      // special-case DEEPSEEK v2:
      // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L42-L43
      if (pModel.vArch == cLlmArchDeepseek2 && cMscaleAllDims != 1.0f) {
        vMscale = cMscaleAllDims;
      }

      vCparams.vYarnAttnFactor = vGetMscale(cFactor, vMscale) / vGetMscale(cFactor, cMscaleAllDims);

      DGS_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n", __func__,
                   vCparams.vYarnAttnFactor, vMscale, cMscaleAllDims);
    } else {
      vCparams.vYarnAttnFactor = vGetMscale(cFactor, 1.0f);
    }

    // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor:
    //
    vCparams.vYarnAttnFactor *= 1.0f / (1.0f + 0.1f * logf(cFactor));
  }

  vCparams.vYarnAttnFactor *= vHparams.vRopeAttnFactor;

  if (vCparams.vPoolingType == cDgsPoolingTypeUnspecified) {
    if (vHparams.vPoolingType == cDgsPoolingTypeUnspecified) {
      vCparams.vPoolingType = cDgsPoolingTypeNone;
    } else {
      vCparams.vPoolingType = vHparams.vPoolingType;
    }
  }

  if (pParams.vAttentionType == cDgsAttentionTypeUnspecified) {
    vCparams.vCausalAttn = vHparams.vCausalAttn;
  } else {
    vCparams.vCausalAttn = pParams.vAttentionType == cDgsAttentionTypeCausal;
  }

  vCparams.vFlashAttn = pParams.vFlashAttnType != cDgsFlashAttnTypeDisabled;
  vCparams.vAutoFa    = pParams.vFlashAttnType == cDgsFlashAttnTypeAuto;

  vCparams.vFusedGdnAr = true;
  vCparams.vFusedGdnCh = true;
  vCparams.vAutoFgdn   = false;

  vCparams.vFusedLid = true;
  vCparams.vAutoFlid = false;

  vCparams.vFusedDsv4HcPre  = true;
  vCparams.vFusedDsv4HcComb = true;
  vCparams.vFusedDsv4HcPost = true;
  vCparams.vAutoFhc         = true;

  // with causal attention, the batch size is limited by the context size
  vCparams.vNBatch = vCparams.vCausalAttn ? std::min(vCparams.vNCtx, pParams.vNBatch) : pParams.vNBatch;

  vCparams.vNUbatch = std::min(vCparams.vNBatch, pParams.vNUbatch == 0 ? pParams.vNBatch : pParams.vNUbatch);

  vCparams.vNOutputsMax =
    pParams.vNOutputsMax == 0 || fDgsModelHasEncoder(&pModel) ? vCparams.vNBatch : pParams.vNOutputsMax;
  vCparams.vNOutputsMaxPerSeq = pParams.vNOutputsMaxPerSeq == 0 ?
                                  vCparams.vNOutputsMax :
                                  std::min(pParams.vNOutputsMaxPerSeq, vCparams.vNOutputsMax);

  // Initialize backend samplers here so they are part of the sampling graph
  // before the reserve passes run later in this function. This avoids a later
  // re-reserve when graph nodes change.
  if (pParams.vSamplers != nullptr && pParams.vNSamplers > 0) {
    for (size_t vI = 0; vI < pParams.vNSamplers; ++vI) {
      const auto & vConfig = pParams.vSamplers[vI];

      if (fDgsSamplerChainGet(vConfig.vSampler, -1) == nullptr) {
        throw std::runtime_error("the backend samplers must be of type dgs_sampler_chain");
      }

      if (mSetSampler(vConfig.vSeqId, vConfig.vSampler)) {
        const int cNSamplers = fDgsSamplerChainN(vConfig.vSampler);

        DGS_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, vConfig.vSeqId, cNSamplers);
      }
    }
  }

  vCparams.vOpOffload = pParams.vOpOffload;
  vCparams.vKvUnified = pParams.vKvUnified;

  // initialized later
  vCparams.vPipelineParallel = false;

  {
    const char * vDgsGraphReuseDisable = getenv("DGS_GRAPH_REUSE_DISABLE");
    vGraphReuseDisable = vDgsGraphReuseDisable ? (atoi(vDgsGraphReuseDisable) != 0) : vGraphReuseDisable;

    if (vGraphReuseDisable) {
      DGS_LOG_WARN("%s: graph reuse disabled\n", __func__);
    }
  }

  vCparams.vNCtx = GGML_PAD(vCparams.vNCtx, 256);

  if (vCparams.vKvUnified) {
    vCparams.vNCtxSeq = vCparams.vNCtx;
  } else {
    vCparams.vNCtxSeq = vCparams.vNCtx / vCparams.vNSeqMax;
    vCparams.vNCtxSeq = GGML_PAD(vCparams.vNCtxSeq, 256);

    if (vCparams.vNCtxSeq == 0) {
      throw std::runtime_error("n_ctx_seq == 0");
    }

    if (vCparams.vNCtx != vCparams.vNCtxSeq * vCparams.vNSeqMax) {
      vCparams.vNCtx = vCparams.vNCtxSeq * vCparams.vNSeqMax;
      DGS_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, vCparams.vNCtx);
    }
  }

  DGS_LOG_INFO("%s: n_seq_max             = %u\n", __func__, vCparams.vNSeqMax);
  DGS_LOG_INFO("%s: n_ctx                 = %u\n", __func__, vCparams.vNCtx);
  DGS_LOG_INFO("%s: n_ctx_seq             = %u\n", __func__, vCparams.vNCtxSeq);
  DGS_LOG_INFO("%s: n_batch               = %u\n", __func__, vCparams.vNBatch);
  DGS_LOG_INFO("%s: n_ubatch              = %u\n", __func__, vCparams.vNUbatch);
  DGS_LOG_INFO("%s: causal_attn           = %d\n", __func__, vCparams.vCausalAttn);
  DGS_LOG_INFO("%s: flash_attn            = %s\n", __func__, fDgsFlashAttnTypeName(pParams.vFlashAttnType));
  DGS_LOG_INFO("%s: kv_unified            = %s\n", __func__, vCparams.vKvUnified ? "true" : "false");
  DGS_LOG_INFO("%s: freq_base             = %.1f\n", __func__, vCparams.vRopeFreqBase);
  DGS_LOG_INFO("%s: freq_scale            = %g\n", __func__, vCparams.vRopeFreqScale);
  DGS_LOG_INFO("%s: n_rs_seq              = %u\n", __func__, vCparams.vNRsSeq);
  DGS_LOG_INFO("%s: n_outputs_max         = %u\n", __func__, vCparams.vNOutputsMax);
  DGS_LOG_INFO("%s: n_outputs_max_per_seq = %u\n", __func__, vCparams.vNOutputsMaxPerSeq);

  if (vCparams.vNCtxSeq < vHparams.n_ctx_train) {
    DGS_LOG_INFO("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                 __func__, vCparams.vNCtxSeq, vHparams.n_ctx_train);
  }

  if (vCparams.vNCtxSeq > vHparams.n_ctx_train) {
    DGS_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n", __func__,
                 vCparams.vNCtxSeq, vHparams.n_ctx_train);
  }

  if (!vHparams.vVocabOnly) {
    // GPU backends
    for (const auto & vDev : pModel.lDevices) {
      ggml_backend_t vBackend = ggml_backend_dev_init(vDev.dev, nullptr);
      if (vBackend == nullptr) {
        throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(vDev.dev)));
      }
      lBackends.emplace_back(vBackend);
    }

    // add ACCEL backends (such as BLAS)
    for (size_t vI = 0; vI < ggml_backend_dev_count(); ++vI) {
      ggml_backend_dev_t vDev = ggml_backend_dev_get(vI);
      if (ggml_backend_dev_type(vDev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
        ggml_backend_t vBackend = ggml_backend_dev_init(vDev, nullptr);
        if (vBackend == nullptr) {
          throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(vDev)));
        }
        lBackends.emplace_back(vBackend);
      }
    }

    // add CPU backend
    vBackendCpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (vBackendCpu == nullptr) {
      throw std::runtime_error("failed to initialize CPU backend");
    }
    lBackends.emplace_back(vBackendCpu);

    // create a list of the set_n_threads functions in the backends
    for (auto & vBackend : lBackends) {
      ggml_backend_dev_t vDev = ggml_backend_get_device(vBackend.get());
      ggml_backend_reg_t vReg = vDev ? ggml_backend_dev_backend_reg(vDev) : nullptr;
      if (vReg) {
        auto vGgmlBackendSetNThreadsFn =
          (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(vReg, "ggml_backend_set_n_threads");
        if (vGgmlBackendSetNThreadsFn) {
          lSetNThreadsFns.emplace_back(vBackend.get(), vGgmlBackendSetNThreadsFn);
        }
      }
    }

    fDgsSetAbortCallback(this, pParams.vAbortCallback, pParams.vAbortCallbackData);

    // graph outputs buffer
    {
      if (mOutputReserve(pParams.vNSeqMax) < pParams.vNSeqMax) {
        throw std::runtime_error("failed to reserve initial output buffer");
      }

      DGS_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(vBufOutput.get()),
                   ggml_backend_buffer_get_size(vBufOutput.get()) / 1024.0 / 1024.0);
    }
  }

  // init the memory module
  if (!vHparams.vVocabOnly) {
    dgs_memory_params vParamsMem = {
      /*.type_k    =*/pParams.vTypeK,
      /*.type_v    =*/pParams.vTypeV,
      /*.swa_full  =*/pParams.vSwaFull,
      /*.ctx_type  =*/vCparams.vCtxType,
      /*.mem_other =*/fDgsGetMemory(vCparams.vCtxOther),
    };

    vMemory.reset(pModel.mCreateMemory(vParamsMem, vCparams));
  }

  // init backends
  if (!vHparams.vVocabOnly) {
    DGS_LOG_DEBUG("%s: enumerating backends\n", __func__);

    lBackendBuft.clear();
    lBackendPtrs.clear();
    lBackendBufExpSize.clear();

    for (auto & vBackend : lBackends) {
      auto * vBuft        = ggml_backend_get_default_buffer_type(vBackend.get());
      auto   vBackendType = ggml_backend_dev_type(ggml_backend_get_device(vBackend.get()));

      if (vBackendType == GGML_BACKEND_DEVICE_TYPE_CPU && !pModel.lDevices.empty()) {
        // use the host buffer of the first device CPU for faster transfer of the intermediate state
        const auto & vDev      = pModel.lDevices[0];
        auto *       vHostBuft = ggml_backend_dev_host_buffer_type(vDev.dev);
        if (vHostBuft) {
          vBuft = vHostBuft;
        }
      }

      lBackendBuft.push_back(vBuft);
      lBackendPtrs.push_back(vBackend.get());
      lBackendBufExpSize.push_back(0);
    }

    DGS_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, lBackendPtrs.size());

    // TODO: move these checks to ggml_backend_sched
    // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
    bool vPipelineParallel = pModel.mNDevices() > 1 && pModel.mNGpuLayers() > pModel.hparams.n_layer_all &&
                             pModel.mSplitMode() == cDgsSplitModeLayer && vCparams.vOffloadKqv &&
                             !pModel.mHasTensorOverrides();

    // pipeline parallelism requires support for async compute and events in all devices
    if (vPipelineParallel) {
      for (auto & vBackend : lBackends) {
        auto vDevType = ggml_backend_dev_type(ggml_backend_get_device(vBackend.get()));
        if (vDevType == GGML_BACKEND_DEVICE_TYPE_CPU) {
          // ignore CPU backend
          // TODO: should we ignore ACCEL types too?
          continue;
        }
        auto *                 vDev = ggml_backend_get_device(vBackend.get());
        ggml_backend_dev_props vProps;
        ggml_backend_dev_get_props(vDev, &vProps);
        if (!vProps.caps.async || !vProps.caps.events) {
          // device does not support async compute or events
          vPipelineParallel = false;
          break;
        }
      }
    }

    vCparams.vPipelineParallel = vPipelineParallel;

    if (vCparams.vPipelineParallel) {
      DGS_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);
    }

    mSchedReserve();

    if (!vCparams.vFlashAttn) {
      if (ggml_is_quantized(pParams.vTypeV)) {
        throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
      }
    }
  }

  // Initialize the full vocabulary token ids for backend samplers.
  {
    const int cNVocab = pModel.vocab.n_tokens();

    vSampling.lTokenIdsFullVocab.resize(cNVocab);
    for (int vI = 0; vI < cNVocab; ++vI) {
      vSampling.lTokenIdsFullVocab[vI] = vI;
    }
  }
}

dgs_context::~dgs_context() {
  // wait for any pending asynchronous copies into the output buffers before they are freed
  mSynchronize();

  if (!vModel.hparams.vNoAlloc) {
    for (size_t vI = 0; vI < lBackendPtrs.size(); ++vI) {
      ggml_backend_t             vBackend = lBackendPtrs[vI];
      ggml_backend_buffer_type_t vBuft    = lBackendBuft[vI];

      const size_t cSizeExp = lBackendBufExpSize[vI];
      const size_t cSizeAct = ggml_backend_sched_get_buffer_size(vSched.get(), vBackend);
      if (cSizeExp == cSizeAct) {
        DGS_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n", __func__,
                      ggml_backend_buft_name(vBuft), cSizeAct / (1024.0 * 1024.0), cSizeExp / (1024.0 * 1024.0));
      } else {
        DGS_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n", __func__,
                     ggml_backend_buft_name(vBuft), cSizeAct / (1024.0 * 1024.0), cSizeExp / (1024.0 * 1024.0));
      }
    }
  }
}

void dgs_context::mResolveFusedOps(const dgs_memory_context_i * pMctx, uint32_t pNSeqs) {
  const char * vFunc    = __func__;
  auto         vResolve = [&](const llm_fused_op_probe & pProbe, bool & pEnabled) {
    if (!pEnabled) {
      return;
    }

    const uint32_t cNTokensProbe = pProbe.vNTokensPerSeq * pNSeqs;

    auto * vGf = mGraphReserve(cNTokensProbe, pNSeqs, cNTokensProbe, pMctx, true);
    if (!vGf) {
      throw std::runtime_error(std::string("failed to reserve graph for ") + pProbe.name + " check");
    }

    bool vDeviceMismatch = false;
    for (const auto & node : mGetGfResReserve()->mGetFusedNodes()) {
      if (node.vOp != pProbe.vOp) {
        continue;
      }

      GGML_ASSERT(node.vIl >= 0);

      ggml_backend_t     vBackendFused = ggml_backend_sched_get_tensor_backend(vSched.get(), node.vTensor);
      ggml_backend_dev_t vDeviceFused  = vBackendFused ? ggml_backend_get_device(vBackendFused) : nullptr;

      // TODO: make this descriptor-specific; model.dev_layer() preserves the current behavior,
      // but is still wrong for cases like --no-kv-offload.
      ggml_backend_dev_t vDeviceLayer = vModel.mDevLayer(node.vIl);

      if (vDeviceFused != vDeviceLayer) {
        DGS_LOG_WARN(
          "%s: layer %d is assigned to device %s but %s "
          "is assigned to device %s (usually due to missing support)\n",
          vFunc, node.vIl, vDeviceLayer ? ggml_backend_dev_name(vDeviceLayer) : "none", pProbe.name,
          vDeviceFused ? ggml_backend_dev_name(vDeviceFused) : "none");
        vDeviceMismatch = true;
        break;
      }
    }

    if (vDeviceMismatch) {
      pEnabled = false;
      DGS_LOG_WARN("%s: %s not supported, set to disabled\n", vFunc, pProbe.name);
    } else {
      pEnabled = true;
      DGS_LOG_INFO("%s: %s enabled\n", vFunc, pProbe.name);
    }
  };

  if (vCparams.vAutoFa) {
    vResolve(cLlmFusedOpFlashAttnProbe, vCparams.vFlashAttn);
    vCparams.vAutoFa = false;
  }

  if (vCparams.vAutoFgdn) {
    DGS_LOG_INFO("%s: resolving fused Gated Delta Net support:\n", vFunc);
    vResolve(cLlmFusedOpGdnArProbe, vCparams.vFusedGdnAr);
    vResolve(cLlmFusedOpGdnChProbe, vCparams.vFusedGdnCh);
    vCparams.vAutoFgdn = false;
  }

  if (vCparams.vAutoFlid) {
    DGS_LOG_INFO("%s: resolving fused Lightning Indexer support:\n", vFunc);
    vResolve(cLlmFusedOpLidProbe, vCparams.vFusedLid);
    vCparams.vAutoFlid = false;
  }

  if (vCparams.vAutoFhc) {
    DGS_LOG_INFO("%s: resolving fused DeepSeek V4 HC support:\n", vFunc);
    vResolve(cLlmFusedOpDsv4HcPreProbe, vCparams.vFusedDsv4HcPre);
    vResolve(cLlmFusedOpDsv4HcCombProbe, vCparams.vFusedDsv4HcComb);
    vResolve(cLlmFusedOpDsv4HcPostProbe, vCparams.vFusedDsv4HcPost);
    vCparams.vAutoFhc = false;
  }
}

static int fDgsGraphNInputTensors(ggml_cgraph * pGf) {
  std::unordered_map<const ggml_tensor *, std::vector<ggml_tensor *>> dUsers;
  for (int vI = 0; vI < ggml_graph_n_nodes(pGf); ++vI) {
    ggml_tensor * node = ggml_graph_node(pGf, vI);
    if (node->flags & GGML_TENSOR_FLAG_INPUT) {
      dUsers[node].push_back(node);
    }
    for (int vJ = 0; vJ < GGML_MAX_SRC; ++vJ) {
      ggml_tensor * vSrc = node->src[vJ];
      if (!vSrc) {
        break;
      }
      if (vSrc->flags & GGML_TENSOR_FLAG_INPUT) {
        dUsers[vSrc].push_back(node);
      }
    }
  }

  for (const auto & [tensor, nodes] : dUsers) {
    GGML_ASSERT(tensor->op == GGML_OP_NONE);
    for (const ggml_tensor * node : nodes) {
      DGS_LOG_DEBUG("%s: input tensor '%32s' [%s, ne = { %5" PRId64 ", %5" PRId64 ", %5" PRId64 ", %5" PRId64
                    " }] is used by node '%s' (%s)\n",
                    __func__, tensor->name, ggml_type_name(tensor->type), tensor->ne[0], tensor->ne[1], tensor->ne[2],
                    tensor->ne[3], node->name, ggml_op_name(node->op));
    }
  }

  return (int) dUsers.size();
}

void dgs_context::mSchedNew(size_t pMaxNodes, bool pParallel) {
  vSched.reset(ggml_backend_sched_new(lBackendPtrs.data(), lBackendBuft.data(), lBackendPtrs.size(), pMaxNodes,
                                      pParallel, vCparams.vOpOffload));

  ggml_backend_sched_set_realloc_callback(vSched.get(), mSchedGraphRealloc, this);
}

void dgs_context::mSchedReserve() {
  if (!vSchedNeedReserve) {
    return;
  }

  vSchedNeedReserve = false;

  DGS_LOG_INFO("%s: reserving ...\n", __func__);

  mSynchronize();

  const int64_t cTStartUs = ggml_time_us();

  const uint32_t cNSeqs   = vCparams.vNSeqMax;
  const uint32_t cNTokens = std::min(vCparams.vNCtx, vCparams.vNUbatch);

  const size_t cMaxNodes = this->mGraphMaxNodes(cNTokens);

  DGS_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, cMaxNodes);

  for (auto & vRes : aGfResPrev) {
    vRes.reset();
  }
  vGfResReserve.reset(new llm_graph_result(cMaxNodes));
  vGfResPrevActive = nullptr;

  mSchedNew(cMaxNodes, vCparams.vPipelineParallel);

  dgs_memory_context_ptr vMctx;
  if (vMemory) {
    DGS_LOG_DEBUG("%s: reserving full memory module\n", __func__);
    vMctx = vMemory->mInitFull();
    if (!vMctx) {
      throw std::runtime_error("failed to initialize memory module");
    }
  }

  // avoid reserving graphs with zero outputs - assume one output per sequence
  const int cNOutputs = cNSeqs;

  DGS_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, cNTokens, cNSeqs, cNOutputs);

  mResolveFusedOps(vMctx.get(), cNSeqs);

  // reserve worst-case graph
  int vNSplitsPp       = -1;
  int vNNodesPp        = -1;
  int vNInputsPp       = -1;
  int vNInputTensorsPp = -1;

  int vNSplitsTg       = -1;
  int vNNodesTg        = -1;
  int vNInputsTg       = -1;
  int vNInputTensorsTg = -1;

  const uint32_t cNOutputsPp = std::min(cNTokens, vCparams.vNOutputsMax);

  // reserve pp (prompt processing) graph first so that buffers are only allocated once
  {
    auto * vGf = mGraphReserve(cNTokens, cNSeqs, cNOutputsPp, vMctx.get(), vModel.hparams.vNoAlloc,
                               vModel.hparams.vNoAlloc ? lBackendBufExpSize.data() : nullptr);
    if (!vGf) {
      if (vCparams.vPipelineParallel) {
        DGS_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", __func__);
        vCparams.vPipelineParallel = false;
        mSchedNew(cMaxNodes, false);
        vGf = mGraphReserve(cNTokens, cNSeqs, cNOutputsPp, vMctx.get());
      }
      if (!vGf) {
        throw std::runtime_error("failed to allocate compute pp buffers");
      }
    }

    vNSplitsPp       = ggml_backend_sched_get_n_splits(vSched.get());
    vNNodesPp        = ggml_graph_n_nodes(vGf);
    vNInputsPp       = mGetGfResReserve()->lInputs.size();
    vNInputTensorsPp = this->vNInputTensors;

    vGraphHasLogits = mGetGfResReserve()->mGetLogits() != nullptr;
  }

  // reserve with tg (token generation) graph to get the number of splits and nodes
  {
    auto * vGf = mGraphReserve(cNSeqs, cNSeqs, cNSeqs, vMctx.get(), vModel.hparams.vNoAlloc);
    if (!vGf) {
      throw std::runtime_error("failed to allocate compute tg buffers");
    }

    vNSplitsTg       = ggml_backend_sched_get_n_splits(vSched.get());
    vNNodesTg        = ggml_graph_n_nodes(vGf);
    vNInputsTg       = mGetGfResReserve()->lInputs.size();
    vNInputTensorsTg = this->vNInputTensors;
  }

  // reserve again with pp graph to avoid ggml-alloc reallocations during inference
  {
    // TODO: the worst case graph is not always reached for `n_seqs > 1`
    //       need to implement a more robust mechanism that tries a few different inputs and analyzes the results
    ggml_cgraph * vGf = nullptr;
    switch (vModel.vArch) {
      default:
        vGf = mGraphReserve(cNTokens, cNSeqs, cNOutputsPp, vMctx.get(), vModel.hparams.vNoAlloc);
    };

    if (!vGf) {
      throw std::runtime_error("failed to allocate compute pp buffers");
    }
  }

  for (size_t vI = 0; vI < lBackendPtrs.size(); ++vI) {
    ggml_backend_t             vBackend = lBackendPtrs[vI];
    ggml_backend_buffer_type_t vBuft    = lBackendBuft[vI];
    if (!vModel.hparams.vNoAlloc) {
      lBackendBufExpSize[vI] = ggml_backend_sched_get_buffer_size(vSched.get(), vBackend);
    }
    if (lBackendBufExpSize[vI] > 1) {
      DGS_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__, ggml_backend_buft_name(vBuft),
                   lBackendBufExpSize[vI] / 1024.0 / 1024.0);
    }
  }

  {
    const bool cDiff = vNNodesPp != vNNodesTg || vNSplitsPp != vNSplitsTg || vNInputsPp != vNInputsTg ||
                       vNInputTensorsPp != vNInputTensorsTg;

    const auto cVal = [cDiff](int pVPp, int pVTg) -> std::string {
      return cDiff ? format("%d / %d", pVPp, pVTg) : format("%d", pVPp);
    };

    DGS_LOG_INFO("%s: graph%s: nodes = %s, splits = %s, input objects = %s, input tensors = %s\n", __func__,
                 cDiff ? format(" (pp bs=%d, tg bs=%d)", cNTokens, cNSeqs).c_str() : "",
                 cVal(vNNodesPp, vNNodesTg).c_str(), cVal(vNSplitsPp, vNSplitsTg).c_str(),
                 cVal(vNInputsPp, vNInputsTg).c_str(), cVal(vNInputTensorsPp, vNInputTensorsTg).c_str());
  }

  const int64_t cTEndUs = ggml_time_us();

  DGS_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n", __func__, (cTEndUs - cTStartUs) / 1000.0,
               ggml_backend_sched_get_n_copies(vSched.get()));
}

void dgs_context::mSynchronize() {
  if (!vSched) {
    return;
  }

  ggml_backend_sched_synchronize(vSched.get());

  // FIXME: if multiple single tokens are evaluated without a synchronization,
  // the stats will be added to the prompt evaluation stats
  // this should only happen when using batch size 1 to evaluate a batch

  // add the evaluation to the stats
  if (vNQueuedTokens == 1) {
    if (!vCparams.vNoPerf) {
      vTEvalUs += ggml_time_us() - vTComputeStartUs;
    }
    vNEval++;
  } else if (vNQueuedTokens > 1) {
    if (!vCparams.vNoPerf) {
      vTPEvalUs += ggml_time_us() - vTComputeStartUs;
    }
    vNPEval += vNQueuedTokens;
  }

  // get a more accurate load time, upon first eval
  if (vNQueuedTokens > 0 && !vHasEvaluatedOnce) {
    vTLoadUs          = ggml_time_us() - vTStartUs;
    vHasEvaluatedOnce = true;
  }

  vNQueuedTokens   = 0;
  vTComputeStartUs = 0;
}

const dgs_model & dgs_context::mGetModel() const {
  return vModel;
}

const dgs_cparams & dgs_context::mGetCparams() const {
  return vCparams;
}

ggml_backend_sched_t dgs_context::mGetSched() const {
  return vSched.get();
}

uint32_t dgs_context::mNCtx() const {
  return vCparams.vNCtx;
}

uint32_t dgs_context::mNCtxSeq() const {
  return vCparams.vNCtxSeq;
}

uint32_t dgs_context::mNBatch() const {
  return vCparams.vNBatch;
}

uint32_t dgs_context::mNUbatch() const {
  return vCparams.vNUbatch;
}

uint32_t dgs_context::mNSeqMax() const {
  return vCparams.vNSeqMax;
}

uint32_t dgs_context::mNThreads() const {
  return vCparams.vNThreads;
}

uint32_t dgs_context::mNThreadsBatch() const {
  return vCparams.vNThreadsBatch;
}

dgs_memory_t dgs_context::mGetMemory() const {
  return vMemory.get();
}

bool dgs_context::mMemoryUpdate(bool pOptimize) {
  if (!vMemory) {
    return false;
  }

  {
    const auto cMctx = vMemory->mInitUpdate(this, pOptimize);
    switch (cMctx->mGetStatus()) {
      case cDgsMemoryStatusSuccess:
        {
          // noop
        }
        break;
      case cDgsMemoryStatusNoUpdate:
        {
          // no updates need to be performed
          return false;
        }
      case cDgsMemoryStatusFailedPrepare:
      case cDgsMemoryStatusFailedCompute:
        {
          DGS_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
          return false;
        }
    }

    // reset the previous graph results to make sure that they won't be reused
    // TODO: make mctx->apply() report if a graph reserve is needed, then reset graph results only if the memory module reset the scheduler
    for (auto & vRes : aGfResPrev) {
      if (vRes) {
        vRes->reset();
      }
    }
    vGfResPrevActive = nullptr;

    if (!cMctx->apply()) {
      DGS_LOG_ERROR("%s: failed to apply memory update\n", __func__);
    }
  }

  // if the memory module did any computation, we have to reserve a new worst-case graph
  {
    const auto cMctx = vMemory->mInitFull();
    if (!cMctx) {
      throw std::runtime_error("failed to initialize memory context");
    }

    const uint32_t cNSeqs   = vCparams.vNSeqMax;
    const uint32_t cNTokens = std::min(vCparams.vNCtx, vCparams.vNUbatch);

    const uint32_t cNOutputsMax = std::min(cNTokens, vCparams.vNOutputsMax);

    auto * vGf = mGraphReserve(cNTokens, cNSeqs, cNOutputsMax, cMctx.get());
    if (!vGf) {
      DGS_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
    }
  }

  return true;
}

enum fDgsPoolingType dgs_context::mPoolingType() const {
  return vCparams.vPoolingType;
}

float * dgs_context::mGetLogits() {
  mOutputReorder();

  return vLogits.data;
}

int64_t dgs_context::mOutputResolveRow(int32_t pI) const {
  int64_t vJ = -1;

  // support negative indices (last output row)
  if (pI < 0) {
    vJ = vNOutputs + pI;
    if (vJ < 0) {
      throw std::runtime_error(format("negative index out of range [0, %d)", vNOutputs));
    }
  } else if ((size_t) pI >= lOutputIds.size()) {
    throw std::runtime_error(format("out of range [0, %zu)", lOutputIds.size()));
  } else {
    // use output_ids to translate the batch token index into a row number
    // that holds this token's data.
    vJ = lOutputIds[pI];
  }

  if (vJ < 0) {
    // the batch token was not configured to output anything
    throw std::runtime_error(format("batch.logits[%d] != true", pI));
  }

  if (vJ >= vNOutputs) {
    throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", vJ, vNOutputs));
  }

  return vJ;
}

float * dgs_context::mGetLogitsIth(int32_t pI) {
  mOutputReorder();

  try {
    if (vLogits.data == nullptr) {
      throw std::runtime_error("no logits");
    }

    const int64_t cJ = mOutputResolveRow(pI);
    return vLogits.data + cJ * vModel.vocab.n_tokens();
  } catch (const std::exception & vErr) {
    DGS_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, pI, vErr.what());
#ifndef NDEBUG
    GGML_ABORT("fatal error");
#else
    return nullptr;
#endif
  }
}

float * dgs_context::mGetEmbeddings() {
  mOutputReorder();

  return vEmbd.data;
}

dgs_token * dgs_context::mGetSampledTokens() const {
  return vSampling.vSampled.data;
}

float * dgs_context::mGetEmbeddingsIth(int32_t pI) {
  mOutputReorder();

  try {
    if (vEmbd.data == nullptr) {
      throw std::runtime_error("no embeddings");
    }

    const int64_t  cJ        = mOutputResolveRow(pI);
    const uint32_t cNEmbdOut = vModel.hparams.mNEmbdOut();
    return vEmbd.data + cJ * cNEmbdOut;
  } catch (const std::exception & vErr) {
    DGS_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, pI, vErr.what());
#ifndef NDEBUG
    GGML_ABORT("fatal error");
#else
    return nullptr;
#endif
  }
}

float * dgs_context::mGetEmbeddingsSeq(dgs_seq_id pSeqId) {
  auto lIt = dEmbdSeq.find(pSeqId);
  if (lIt == dEmbdSeq.end()) {
    return nullptr;
  }

  return lIt->second.data();
}

float * dgs_context::mGetEmbeddingsNextn() {
  mOutputReorder();

  return vEmbdNextn.data;
}

float * dgs_context::mGetEmbeddingsNextnIth(int32_t pI) {
  mOutputReorder();

  try {
    if (vEmbdNextn.data == nullptr) {
      throw std::runtime_error("no nextn embeddings");
    }

    const uint32_t cNEmbd = vModel.hparams.mNEmbdOut();

    if (!vCparams.vEmbeddingsNextnMasked) {
      // unmasked: nextn rows are stored densely, indexed by raw token position.
      if (pI < 0 || (size_t) (pI + 1) * cNEmbd > vEmbdNextn.size) {
        throw std::runtime_error(format("out of range [0, %zu)", vEmbdNextn.size / cNEmbd));
      }
      return vEmbdNextn.data + (size_t) pI * cNEmbd;
    }

    const int64_t cJ = mOutputResolveRow(pI);
    return vEmbdNextn.data + cJ * cNEmbd;
  } catch (const std::exception & vErr) {
    DGS_LOG_ERROR("%s: invalid nextn embeddings id %d, reason: %s\n", __func__, pI, vErr.what());
#ifndef NDEBUG
    GGML_ABORT("fatal error");
#else
    return nullptr;
#endif
  }
}

float * dgs_context::mGetEmbeddingsLayerInp(uint32_t pLid) {
  mOutputReorder();

  GGML_ASSERT(pLid < lEmbdLayerInp.size() && lEmbdLayerInp[pLid].mHasData());

  return lEmbdLayerInp[pLid].data;
}

dgs_token dgs_context::mGetSampledTokenIth(int32_t pIdx) {
  mOutputReorder();

  if (!vSampling.vSampled.mHasData()) {
    return DGS_TOKEN_NULL;
  }

  try {
    const int64_t cRow = mOutputResolveRow(pIdx);
    GGML_ASSERT(cRow < (int64_t) vSampling.vSampled.size);
    return vSampling.vSampled.data[cRow];
  } catch (const std::exception & vErr) {
    DGS_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, pIdx, vErr.what());
    return DGS_TOKEN_NULL;
  }
}

float * dgs_context::mGetSampledProbsIth(int32_t pIdx) {
  mOutputReorder();

  if (!vSampling.vProbs.mHasData()) {
    return nullptr;
  }

  try {
    const int64_t cRow = mOutputResolveRow(pIdx);
    if ((size_t) cRow >= vSampling.lProbsCount.size() || vSampling.lProbsCount[cRow] == 0) {
      return nullptr;
    }
    return vSampling.vProbs.data + cRow * vModel.vocab.n_tokens();
  } catch (const std::exception & vErr) {
    DGS_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, pIdx, vErr.what());
    return nullptr;
  }
}

float * dgs_context::mGetSampledLogitsIth(int32_t pIdx) {
  mOutputReorder();

  if (!vSampling.vLogits.mHasData()) {
    return nullptr;
  }

  try {
    const int64_t cRow = mOutputResolveRow(pIdx);
    if ((size_t) cRow >= vSampling.lLogitsCount.size() || vSampling.lLogitsCount[cRow] == 0) {
      return nullptr;
    }
    return vSampling.vLogits.data + cRow * vModel.vocab.n_tokens();
  } catch (const std::exception & vErr) {
    DGS_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, pIdx, vErr.what());
    return nullptr;
  }
}

const dgs_token * dgs_context::mGetSampledCandidatesIth(int32_t pIdx) {
  mOutputReorder();

  try {
    const int64_t cRow = mOutputResolveRow(pIdx);
    if (vSampling.vCandidates.mHasData() && (size_t) cRow < vSampling.lCandidatesCount.size() &&
        vSampling.lCandidatesCount[cRow] > 0) {
      return vSampling.vCandidates.data + cRow * vModel.vocab.n_tokens();
    }
  } catch (const std::exception & vErr) {
    // fallback to full vocab list
    GGML_UNUSED(vErr);
  }

  return vSampling.lTokenIdsFullVocab.data();
}

size_t dgs_context::mGetSampledCandidatesCount(int32_t pIdx) {
  mOutputReorder();

  if (!vSampling.vCandidates.mHasData()) {
    return 0;
  }

  try {
    const int64_t cRow = mOutputResolveRow(pIdx);
    if ((size_t) cRow >= vSampling.lCandidatesCount.size()) {
      return 0;
    }
    return vSampling.lCandidatesCount[cRow];
  } catch (const std::exception & vErr) {
    DGS_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, pIdx, vErr.what());
    return 0;
  }
}

size_t dgs_context::mGetSampledLogitsCount(int32_t pIdx) {
  mOutputReorder();

  if (!vSampling.vLogits.mHasData()) {
    return vModel.vocab.n_tokens();
  }

  try {
    const int64_t cRow = mOutputResolveRow(pIdx);
    if ((size_t) cRow >= vSampling.lLogitsCount.size()) {
      return 0;
    }
    return vSampling.lLogitsCount[cRow];
  } catch (const std::exception & vErr) {
    DGS_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, pIdx, vErr.what());
    return 0;
  }
}

size_t dgs_context::mGetSampledProbsCount(int32_t pIdx) {
  mOutputReorder();

  if (!vSampling.vProbs.mHasData()) {
    return 0;
  }

  try {
    const int64_t cRow = mOutputResolveRow(pIdx);
    if ((size_t) cRow >= vSampling.lProbsCount.size()) {
      return 0;
    }
    return vSampling.lProbsCount[cRow];
  } catch (const std::exception & vErr) {
    DGS_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, pIdx, vErr.what());
    return 0;
  }
}

void dgs_context::mAttachThreadpool(ggml_threadpool_t pThreadpool, ggml_threadpool_t pThreadpoolBatch) {
  DGS_LOG_DEBUG("%s: call\n", __func__);

  this->vThreadpool      = pThreadpool;
  this->vThreadpoolBatch = pThreadpoolBatch ? pThreadpoolBatch : pThreadpool;
}

void dgs_context::mDetachThreadpool() {
  DGS_LOG_DEBUG("%s: call\n", __func__);

  this->vThreadpool      = nullptr;
  this->vThreadpoolBatch = nullptr;
}

void dgs_context::mSetNThreads(int32_t pNThreads, int32_t pNThreadsBatch) {
  DGS_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, pNThreads, pNThreadsBatch);

  vCparams.vNThreads      = pNThreads;
  vCparams.vNThreadsBatch = pNThreadsBatch;
}

void dgs_context::mSetAbortCallback(bool (*pAbortCallback)(void * data), void * pAbortCallbackData) {
  DGS_LOG_DEBUG("%s: call\n", __func__);

  this->vAbortCallback     = pAbortCallback;
  this->vAbortCallbackData = pAbortCallbackData;

  for (auto & vBackend : lBackends) {
    auto * vReg = ggml_backend_dev_backend_reg(ggml_backend_get_device(vBackend.get()));
    if (vReg) {
      auto * vSetAbortCallbackFn =
        (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(vReg, "ggml_backend_set_abort_callback");
      if (vSetAbortCallbackFn) {
        vSetAbortCallbackFn(vBackend.get(), this->vAbortCallback, this->vAbortCallbackData);
      }
    }
  }
}

void dgs_context::mSetEmbeddings(bool value) {
  DGS_LOG_DEBUG("%s: value = %d\n", __func__, value);

  vCparams.vEmbeddings = value;

  // TODO: not sure yet if we want to reserve here
  //sched_need_reserve = true;
}

void dgs_context::mSetEmbeddingsNextn(bool value, bool pMasked) {
  DGS_LOG_DEBUG("%s: value = %d, masked = %d\n", __func__, value, pMasked);

  if (vCparams.vEmbeddingsNextn != value || vCparams.vEmbeddingsNextnMasked != pMasked) {
    // these flags change the graph shape
    vSchedNeedReserve = true;
  }

  vCparams.vEmbeddingsNextn       = value;
  vCparams.vEmbeddingsNextnMasked = pMasked;
}

void dgs_context::mSetEmbeddingsLayerInp(uint32_t pLid, bool pEnable) {
  DGS_LOG_DEBUG("%s: lid = %d, enable = %d\n", __func__, pLid, pEnable);

  GGML_ASSERT(pLid <= vModel.hparams.n_layer());

  vCparams.lEmbeddingsLayerInp[pLid] = pEnable;

  // note: without this reserve, the draft acceptance drops to zero. not sure why - this is unexpected
  vSchedNeedReserve = true;
}

void dgs_context::mSetNextnLayerOffset(int32_t pOffset) {
  vCparams.vNextnLayerOffset = pOffset;
}

void dgs_context::mSetCausalAttn(bool value) {
  DGS_LOG_DEBUG("%s: value = %d\n", __func__, value);

  if (vCparams.vCausalAttn == value) {
    return;
  }

  vCparams.vCausalAttn = value;

  // no scheduler reserve needed because graph shapes must not depend on causal_attn, a flip only rebuilds the graph
  //sched_need_reserve = true;
}

bool dgs_context::mGetCausalAttn() const {
  return vCparams.vCausalAttn;
}

void dgs_context::mSetWarmup(bool value) {
  DGS_LOG_DEBUG("%s: value = %d\n", __func__, value);

  if (vCparams.vWarmup == value) {
    return;
  }

  vCparams.vWarmup = value;

  // warmups are usually with small batches, so no need to reserve
  //sched_need_reserve = true;
}

bool dgs_context::mSetSampler(dgs_seq_id pSeqId, dgs_sampler * pSampler) {
  if (!pSampler && vSampling.dSamplers.count(pSeqId) == 0) {
    return true;
  }

  DGS_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) pSeqId, (void *) pSampler);

  if (pSampler && vModel.mSplitMode() == cDgsSplitModeTensor) {
    static bool vWarned = false;
    if (!vWarned) {
      DGS_LOG_WARN("%s: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU\n", __func__);
      vWarned = true;
    }
    if (vSampling.dSamplers.count(pSeqId) > 0) {
      vSchedNeedReserve = true;
    }
    vSampling.dSamplers.erase(pSeqId);
    return false;
  }

  const bool cCanOffload =
    pSampler && pSampler->vIface->vBackendInit && pSampler->vIface->vBackendApply && fDgsSamplerChainN(pSampler) > 0;

  if (pSampler && cCanOffload) {
    auto * vBuft = ggml_backend_dev_buffer_type(vModel.mDevOutput());

    pSampler->vIface->vBackendInit(pSampler, vBuft, vCparams.vNOutputsMaxPerSeq);

    vSampling.dSamplers[pSeqId] = pSampler;

    vSchedNeedReserve = true;

    return true;
  }

  if (pSampler && !cCanOffload) {
    DGS_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__,
                 fDgsSamplerName(pSampler), pSeqId);

    if (vSampling.dSamplers.count(pSeqId) > 0) {
      vSchedNeedReserve = true;
    }

    vSampling.dSamplers.erase(pSeqId);

    return false;
  }

  vSampling.dSamplers.erase(pSeqId);

  vSchedNeedReserve = true;

  return true;
}

void dgs_context::mSetAdaptersLora(dgs_adapter_lora ** pAdapters, size_t pNAdapters, float * pScales) {
  DGS_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) pAdapters);

  if (mAdaptersLoraAreSame(pAdapters, pNAdapters, pScales)) {
    return;
  }

  dLoras.reset(new dgs_adapter_loras());

  for (size_t vI = 0; vI < pNAdapters; vI++) {
    if (pScales[vI] != 0.0f) {
      dLoras->insert({ pAdapters[vI], pScales[vI] });
    }
  }

  vSchedNeedReserve = true;
}

bool dgs_context::mAdaptersLoraAreSame(dgs_adapter_lora ** pAdapters, size_t pNAdapters, float * pScales) {
  DGS_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) pAdapters);

  // Adapters with a zero scale are never added to `loras`, so also ignore them for the comparison.
  size_t vNNonZero = 0;

  for (size_t vI = 0; vI < pNAdapters; vI++) {
    if (pScales[vI] == 0.0f) {
      continue;
    }
    vNNonZero++;

    auto vIt = dLoras->find(pAdapters[vI]);

    if (vIt == dLoras->end() || vIt->second != pScales[vI]) {
      return false;
    }
  }

  if (vNNonZero != dLoras->size()) {
    return false;
  }

  return true;
}

bool dgs_context::mSetAdapterCvec(const float * data, size_t pLen, int32_t pNEmbd, int32_t pIlStart, int32_t pIlEnd) {
  DGS_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, pIlStart, pIlEnd);

  bool vRes = vCvec->apply(vModel, data, pLen, pNEmbd, pIlStart, pIlEnd);

  vSchedNeedReserve = true;

  return vRes;
}

llm_graph_result * dgs_context::mProcessUbatch(const dgs_ubatch &     pUbatch,
                                               llm_graph_type         pGtype,
                                               dgs_memory_context_i * pMctx,
                                               ggml_status &          pRet) {
  if (pMctx && !pMctx->apply()) {
    DGS_LOG_ERROR("%s: failed to apply memory context\n", __func__);
    pRet = GGML_STATUS_FAILED;
    return nullptr;
  }

  auto * vRes = mGetGfResPrev();
  auto * vGf  = vRes->mGetGf();

  // the new graph parameters
  // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters
  const auto cGparams = mGraphParams(vRes, pUbatch, pMctx, pGtype);

  if (!vGraphReuseDisable && vGfResPrevActive == vRes && vRes->mCanReuse(cGparams)) {
    //DGS_LOG_DEBUG("%s: reusing previous graph\n", __func__);

    // with pipeline parallelism, the previous graph_compute_async may still be running
    // on the GPU. we must synchronize before set_inputs to avoid overwriting input tensors
    // that the previous compute is still reading.
    if (vCparams.vPipelineParallel) {
      ggml_backend_sched_synchronize(vSched.get());
    }

    vNReused++;
  } else {
    vGfResPrevActive = nullptr;
    vRes->reset();

    ggml_backend_sched_reset(vSched.get());
    ggml_backend_sched_set_eval_callback(vSched.get(), vCparams.vCbEval, vCparams.vCbEvalUserData);

    //const auto t_start_us = ggml_time_us();

    vGf = vModel.mBuildGraph(cGparams);

    //DGS_LOG_INFO("graph build time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);

    if (!vGf) {
      DGS_LOG_ERROR("%s: failed to initialize graph\n", __func__);
      pRet = GGML_STATUS_FAILED;
      return nullptr;
    }

    if (!ggml_backend_sched_alloc_graph(vSched.get(), vGf)) {
      DGS_LOG_ERROR("%s: failed to allocate graph\n", __func__);
      pRet = GGML_STATUS_ALLOC_FAILED;
      return nullptr;
    }

    vGfResPrevActive = vRes;
  }

  // set the input data for the input tensors
  {
    //const auto t_start_us = ggml_time_us();

    // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
    vRes->mSetInputs(&pUbatch);

    //DGS_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
  }

  const auto cStatus = mGraphCompute(vRes->mGetGf(), pUbatch.vNTokens > 1);
  if (cStatus != GGML_STATUS_SUCCESS) {
    DGS_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, cStatus);
    pRet = cStatus;
    return nullptr;
  }

  pRet = GGML_STATUS_SUCCESS;

  return vRes;
}

int dgs_context::mEncode(const dgs_batch_ext & pBatchInp) {
  if (pBatchInp.lTokens.empty()) {
    DGS_LOG_ERROR("%s: n_tokens == 0\n", __func__);
    return -1;
  }

  const auto & vHparams = vModel.hparams;

  if (pBatchInp.vNEmbd > 0 && pBatchInp.vNEmbd != vHparams.mNEmbdInpEnc()) {
    DGS_LOG_ERROR("%s: embd row width %zu does not match the encoder input %u\n", __func__, pBatchInp.vNEmbd,
                  vHparams.mNEmbdInpEnc());
    return -1;
  }

  // eagle3/DFlash: features as encoder input, and non-draft paths fall back to model's input dim
  const int64_t cNVocab = vModel.vocab.n_tokens();

  // note: during encode, we always output all tokens and skip position continuity checks (output_all=true)
  if (!vBalloc->init(pBatchInp, vModel.vocab, true)) {
    DGS_LOG_ERROR("%s: failed to initialize batch\n", __func__);
    return -1;
  }

  const uint32_t cNTokens = vBalloc->mGetNTokens();

  // [TAG_NO_CACHE_PAD]
  // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
  const dgs_ubatch cUbatch = vBalloc->mSplitSimple(cNTokens);

  // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
  GGML_ASSERT(vCparams.vNUbatch >= cNTokens && "encoder requires n_ubatch >= n_tokens");

  // TODO: this clear of the buffer can easily be forgotten - need something better
  // sync first so any in-flight async copies into embd_seq complete before it is freed
  if (!dEmbdSeq.empty()) {
    mSynchronize();
  }
  dEmbdSeq.clear();

  if (vTComputeStartUs == 0) {
    vTComputeStartUs = ggml_time_us();
  }

  mSchedReserve();

  vNQueuedTokens += cNTokens;

  // reserve output buffer
  if (mOutputReserve(cNTokens) < cNTokens) {
    DGS_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, cNTokens);
    return -2;
  };

  for (uint32_t vI = 0; vI < cNTokens; ++vI) {
    lOutputIds[vI] = vI;
  }

  vNOutputs = cNTokens;

  const auto cCausalAttnOrg = vCparams.vCausalAttn;

  // always use non-causal attention for encoder graphs
  // TODO: this is a tmp solution until we have a proper way to support enc-dec models
  vCparams.vCausalAttn = false;

  ggml_status  vStatus;
  const auto * vRes = mProcessUbatch(cUbatch, cLlmGraphTypeEncoder, nullptr, vStatus);

  vCparams.vCausalAttn = cCausalAttnOrg;

  if (!vRes) {
    switch (vStatus) {
      case GGML_STATUS_ABORTED:
        return 2;
      case GGML_STATUS_ALLOC_FAILED:
        return -2;
      case GGML_STATUS_FAILED:
        return -3;
      case GGML_STATUS_SUCCESS:
        GGML_ABORT("should not happen");
    }
  }

  auto * vTLogits = vRes->mGetLogits();
  auto * vTEmbd   = vRes->mGetEmbdPooled() ? vRes->mGetEmbdPooled() : vRes->mGetEmbd();
  auto * vTHNextn = vCparams.vEmbeddingsNextn ? vRes->mGetHNextn() : nullptr;

  // extract logits
  if (vLogits.data && vTLogits) {
    ggml_backend_t vBackendRes = ggml_backend_sched_get_tensor_backend(vSched.get(), vTLogits);
    GGML_ASSERT(vBackendRes != nullptr);
    GGML_ASSERT(vLogits.data != nullptr);

    ggml_backend_tensor_get_async(vBackendRes, vTLogits, vLogits.data, 0, cNTokens * cNVocab * sizeof(float));
  }

  // extract embeddings
  if (vEmbd.data && vTEmbd) {
    ggml_backend_t vBackendEmbd = ggml_backend_sched_get_tensor_backend(vSched.get(), vTEmbd);
    GGML_ASSERT(vBackendEmbd != nullptr);

    switch (vCparams.vPoolingType) {
      case cDgsPoolingTypeNone:
        {
          // extract token embeddings
          GGML_ASSERT(vEmbd.data != nullptr);
          const uint32_t cNEmbdOut = vHparams.mNEmbdOut();

          GGML_ASSERT(cNTokens * cNEmbdOut <= (int64_t) vEmbd.size);
          ggml_backend_tensor_get_async(vBackendEmbd, vTEmbd, vEmbd.data, 0, cNTokens * cNEmbdOut * sizeof(float));
        }
        break;
      case cDgsPoolingTypeMean:
      case cDgsPoolingTypeCls:
      case cDgsPoolingTypeLast:
        {
          // extract sequence embeddings
          auto & dEmbdSeqOut = dEmbdSeq;

          for (uint32_t vS = 0; vS < cUbatch.vNSeqsUnq; ++vS) {
            const dgs_seq_id cSeqId  = cUbatch.vSeqIdUnq[vS];
            const int32_t    cSeqIdx = cUbatch.vSeqIdx[cSeqId];

            // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
            // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
            const uint32_t cNEmbdOut = vHparams.mNEmbdOut();
            dEmbdSeqOut[cSeqId].resize(cNEmbdOut);
            ggml_backend_tensor_get_async(vBackendEmbd, vTEmbd, dEmbdSeqOut[cSeqId].data(),
                                          (cNEmbdOut * cSeqIdx) * sizeof(float), cNEmbdOut * sizeof(float));
          }
        }
        break;
      case cDgsPoolingTypeRank:
        {
          // extract the rerank score - n_cls_out floats per sequence
          auto & dEmbdSeqOut = dEmbdSeq;

          const uint32_t cNClsOut = vHparams.vNClsOut;

          for (uint32_t vS = 0; vS < cUbatch.vNSeqsUnq; ++vS) {
            const dgs_seq_id cSeqId  = cUbatch.vSeqIdUnq[vS];
            const int32_t    cSeqIdx = cUbatch.vSeqIdx[cSeqId];

            dEmbdSeqOut[cSeqId].resize(cNClsOut);
            ggml_backend_tensor_get_async(vBackendEmbd, vTEmbd, dEmbdSeqOut[cSeqId].data(),
                                          (cNClsOut * cSeqIdx) * sizeof(float), cNClsOut * sizeof(float));
          }
        }
        break;
      case cDgsPoolingTypeUnspecified:
        {
          GGML_ABORT("unknown pooling type");
        }
    }
  }

  // extract nextn embeddings (hidden state before the final output norm)
  if (vEmbdNextn.data && vTHNextn && vCparams.vPoolingType == cDgsPoolingTypeNone) {
    ggml_backend_t vBackendH = ggml_backend_sched_get_tensor_backend(vSched.get(), vTHNextn);
    GGML_ASSERT(vBackendH != nullptr);

    const uint32_t cNEmbd = vHparams.mNEmbdOut();
    GGML_ASSERT(cNTokens * cNEmbd <= (int64_t) vEmbdNextn.size);
    ggml_backend_tensor_get_async(vBackendH, vTHNextn, vEmbdNextn.data, 0, cNTokens * cNEmbd * sizeof(float));
  }

  // TODO: hacky solution
  if (vModel.vArch == cLlmArchT5 && vTEmbd) {
    //cross.t_embd = t_embd;

    mSynchronize();

    vCross.vNEmbd = vTEmbd->ne[0];
    vCross.vNEnc  = vTEmbd->ne[1];
    vCross.lVEmbd.resize(vCross.vNEmbd * vCross.vNEnc);
    memcpy(vCross.lVEmbd.data(), vEmbd.data, ggml_nbytes(vTEmbd));

    const auto & vBatch = vBalloc->mGetBatch();

    // remember the sequence ids used during the encoding - needed for cross attention later
    vCross.sSeqIdsEnc.resize(cNTokens);
    for (uint32_t vI = 0; vI < cNTokens; vI++) {
      vCross.sSeqIdsEnc[vI].clear();

      for (int vS = 0; vS < vBatch.vNSeqId[vI]; vS++) {
        const dgs_seq_id cSeqId = vBatch.vSeqId[vI][vS];

        vCross.sSeqIdsEnc[vI].insert(cSeqId);
      }
    }
  }

  return 0;
}

template <typename T>
static void fCopyTensorAsyncRows(const std::vector<ggml_tensor *> & pTensors,
                                 const buffer_view<T> &             pDst,
                                 size_t                             pStride,
                                 uint32_t                           pRowOffset,
                                 ggml_backend_sched_t               pSched,
                                 std::vector<uint32_t> *            pCounts = nullptr) {
  if (!pDst.mHasData()) {
    return;
  }

  for (size_t vI = 0; vI < pTensors.size(); ++vI) {
    auto * vTensor = pTensors[vI];
    if (vTensor == nullptr) {
      continue;
    }

    const uint32_t cRow       = pRowOffset + vI;
    const size_t   cNElements = ggml_nelements(vTensor);
    GGML_ASSERT(ggml_is_contiguous(vTensor) && "sampling tensor must be contiguous for async copy");
    GGML_ASSERT(cNElements <= pStride);
    GGML_ASSERT((size_t) cRow * pStride + cNElements <= pDst.size);

    ggml_backend_t vBackend = ggml_backend_sched_get_tensor_backend(pSched, vTensor);
    T *            vRowPtr  = pDst.data + (size_t) cRow * pStride;
    ggml_backend_tensor_get_async(vBackend, vTensor, vRowPtr, 0, ggml_nbytes(vTensor));

    if (pCounts) {
      GGML_ASSERT(cRow < pCounts->size());
      (*pCounts)[cRow] = cNElements;
    }
  }
}

static bool fNeedsRawLogits(const dgs_ubatch & pUbatch, const std::map<dgs_seq_id, dgs_sampler *> & pSamplers) {
  for (uint32_t vI = 0; vI < pUbatch.vNTokens; vI++) {
    if (!pUbatch.vOutput[vI]) {
      continue;
    }

    // Check if the output token has at least one sequence without a backend sampler.
    for (int32_t vJ = 0; vJ < pUbatch.vNSeqId[vI]; ++vJ) {
      dgs_seq_id vSeqId = pUbatch.vSeqId[vI][vJ];
      if (pSamplers.find(vSeqId) == pSamplers.end()) {
        return true;
      }
    }
  }
  return false;  // all sequences use backend sampling
}

int dgs_context::mDecode(const dgs_batch_ext & pBatchInp) {
  if (!vMemory) {
    DGS_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
    return mEncode(pBatchInp);
  }

  if (pBatchInp.lTokens.empty()) {
    DGS_LOG_ERROR("%s: n_tokens == 0\n", __func__);
    return -1;
  }

  if (pBatchInp.vNEmbd > 0 && pBatchInp.vNEmbd != pBatchInp.cNEmbdInp) {
    DGS_LOG_ERROR("%s: embd row width %zu does not match the decoder input %zu\n", __func__, pBatchInp.vNEmbd,
                  pBatchInp.cNEmbdInp);
    return -1;
  }

  const auto & vVocab   = vModel.vocab;
  const auto & vHparams = vModel.hparams;

  const int64_t cNVocab = vVocab.n_tokens();

  // when computing embeddings, all tokens are output
  const bool cOutputAll   = vCparams.vEmbeddings;
  // non-causal attention needs the whole sequence in one ubatch
  const bool cEmbdAll     = cOutputAll || !vCparams.vCausalAttn;
  const bool cHasSamplers = !vSampling.dSamplers.empty();

  const uint32_t cNSeqMax = vCparams.vKvUnified ? DGS_MAX_SEQ : vCparams.vNSeqMax;

  // TODO: avoid this workaround in the future
  // embedding contexts output every token even when no token is explicitly marked as output
  if (cHasSamplers) {
    std::vector<int32_t> lSeqOutputCount(cNSeqMax, 0);

    for (const auto & vTok : pBatchInp.lTokens) {
      if (!cOutputAll && !vTok.vOutput) {
        continue;
      }

      for (auto vSeqId : vTok.sSeqIds) {
        if (vSeqId < 0 || (uint32_t) vSeqId >= cNSeqMax) {
          continue;
        }

        lSeqOutputCount[vSeqId]++;
        auto vSampler = vSampling.dSamplers.find(vSeqId);
        if (vSampler != vSampling.dSamplers.end() && lSeqOutputCount[vSeqId] > (int32_t) vCparams.vNOutputsMaxPerSeq) {
          DGS_LOG_ERROR(
            "%s: backend sampling supports at most %u outputs per sequence "
            "(seq_id %d had %d)\n",
            __func__, vCparams.vNOutputsMaxPerSeq, vSeqId, lSeqOutputCount[vSeqId]);
          return -1;
        }
      }
    }
  }

  if (!vBalloc->init(pBatchInp, vVocab, cOutputAll)) {
    DGS_LOG_ERROR("%s: failed to initialize batch\n", __func__);
    return -1;
  }

  const uint32_t cNTokensAll  = vBalloc->mGetNTokens();
  const uint32_t cNOutputsAll = vBalloc->mGetNOutputs();

  if (cOutputAll) {
    // require that all tokens are output
    if (cNOutputsAll != cNTokensAll) {
      DGS_LOG_ERROR(
        "%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n", __func__,
        cNOutputsAll, cNTokensAll);
      return -1;
    }
  }

  GGML_ASSERT(cNTokensAll <= vCparams.vNBatch);

  GGML_ASSERT((vCparams.vCausalAttn || vCparams.vNUbatch >= cNTokensAll) &&
              "non-causal attention requires n_ubatch >= n_tokens");

  // TODO: this clear of the buffer can easily be forgotten - need something better
  // sync first so any in-flight async copies into embd_seq complete before it is freed
  if (!dEmbdSeq.empty()) {
    mSynchronize();
  }
  dEmbdSeq.clear();

  if (vTComputeStartUs == 0) {
    vTComputeStartUs = ggml_time_us();
  }
  vNQueuedTokens += cNTokensAll;

  lOutputSwaps.clear();
  lEmbdBatchIdxs.clear();

  mSchedReserve();

  bool vDidOptimize = false;

  // handle any pending shifts/copies
  mMemoryUpdate(false);

  dgs_memory_context_ptr vMctx;

  while (true) {
    vMctx = vMemory->mInitBatch(*vBalloc, vCparams.vNUbatch, cEmbdAll);
    if (!vMctx) {
      return -2;
    }

    switch (vMctx->mGetStatus()) {
      case cDgsMemoryStatusSuccess:
        {
        }
        break;
      case cDgsMemoryStatusNoUpdate:
        {
          DGS_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, vMctx->mGetStatus());

          return -2;
        }
      case cDgsMemoryStatusFailedPrepare:
        {
          if (!vDidOptimize) {
            vDidOptimize = true;

            if (mMemoryUpdate(true)) {
              DGS_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__, vBalloc->mGetNTokens());

              continue;
            }
          }

          DGS_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, vBalloc->mGetNTokens());

          return 1;
        }
      case cDgsMemoryStatusFailedCompute:
        {
          DGS_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__, vBalloc->mGetNTokens());

          return -2;
        }
    }

    break;
  }

  // reserve output buffer
  if (mOutputReserve(cNOutputsAll) < cNOutputsAll) {
    DGS_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, cNOutputsAll);
    return -2;
  };

  // start a new sampling transaction for this logical batch
  for (const auto & vEntry : vSampling.dSamplers) {
    fDgsSamplerBackendBegin(vEntry.second);
  }

  int64_t vNOutputsPrev = 0;
  int64_t vNTokensPrev  = 0;

  do {
    const auto & vUbatch = vMctx->mGetUbatch();

    // count the outputs in this ubatch
    {
      int32_t vNOutputsNew = 0;

      if (cNOutputsAll == cNTokensAll) {
        vNOutputsNew = vUbatch.vNTokens;
      } else {
        for (uint32_t vI = 0; vI < vUbatch.vNTokens; vI++) {
          vNOutputsNew += (int32_t) (vUbatch.vOutput[vI] != 0);
        }
      }

      // needs to happen before the graph is built
      vNOutputs = vNOutputsNew;
    }

    ggml_status vStatus;

    const auto * vRes = mProcessUbatch(vUbatch, fCtxTypeToGraphType(vCparams.vCtxType), vMctx.get(), vStatus);

    if (!vRes) {
      // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
      dgs_pos aPosMin[DGS_MAX_SEQ];
      for (int vS = 0; vS < DGS_MAX_SEQ; ++vS) {
        aPosMin[vS] = std::numeric_limits<dgs_pos>::max();
      }

      for (uint32_t vI = 0; vI < vUbatch.vNTokens; ++vI) {
        const auto & vSeqId = vUbatch.vSeqId[vI][0];

        aPosMin[vSeqId] = std::min(aPosMin[vSeqId], vUbatch.vPos[vI]);
      }

      for (int vS = 0; vS < DGS_MAX_SEQ; ++vS) {
        if (aPosMin[vS] == std::numeric_limits<dgs_pos>::max()) {
          continue;
        }

        DGS_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, vS,
                     aPosMin[vS]);

        vMemory->mSeqRm(vS, aPosMin[vS], -1);
      }

      switch (vStatus) {
        case GGML_STATUS_ABORTED:
          return 2;
        case GGML_STATUS_ALLOC_FAILED:
          return -2;
        case GGML_STATUS_FAILED:
          return -3;
        case GGML_STATUS_SUCCESS:
          GGML_ABORT("should not happen");
      }
    }

    // plot the computation graph in dot format (for debugging purposes)
    //if (n_past%100 == 0) {
    //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
    //}

    auto * vTLogits = vRes->mGetLogits();
    auto * vTEmbd   = vCparams.vEmbeddings ? vRes->mGetEmbd() : nullptr;
    auto * vTHNextn = vCparams.vEmbeddingsNextn ? vRes->mGetHNextn() : nullptr;

    if (vTEmbd && vRes->mGetEmbdPooled()) {
      vTEmbd = vRes->mGetEmbdPooled();
    }

    // extract logits
    if (vLogits.data && vTLogits && vNOutputs > 0 && fNeedsRawLogits(vUbatch, vSampling.dSamplers)) {
      ggml_backend_t vBackendRes = ggml_backend_sched_get_tensor_backend(vSched.get(), vTLogits);
      GGML_ASSERT(vBackendRes != nullptr);
      GGML_ASSERT(vLogits.data != nullptr);

      float * vLogitsOut = vLogits.data + vNOutputsPrev * cNVocab;

      if (vNOutputs) {
        GGML_ASSERT(vNOutputsPrev + vNOutputs <= cNOutputsAll);
        GGML_ASSERT((vNOutputsPrev + vNOutputs) * cNVocab <= (int64_t) vLogits.size);
        ggml_backend_tensor_get_async(vBackendRes, vTLogits, vLogitsOut, 0, vNOutputs * cNVocab * sizeof(float));
      }
    }

    // extract embeddings
    if (vEmbd.data && vTEmbd && vNOutputs > 0) {
      ggml_backend_t vBackendEmbd = ggml_backend_sched_get_tensor_backend(vSched.get(), vTEmbd);
      GGML_ASSERT(vBackendEmbd != nullptr);

      switch (vCparams.vPoolingType) {
        case cDgsPoolingTypeNone:
          {
            // extract token embeddings
            GGML_ASSERT(vEmbd.data != nullptr);
            const uint32_t cNEmbdOut = vHparams.mNEmbdOut();
            float *        vEmbdOut  = vEmbd.data + vNOutputsPrev * cNEmbdOut;

            if (vNOutputs) {
              GGML_ASSERT(vNOutputsPrev + vNOutputs <= cNOutputsAll);
              GGML_ASSERT((vNOutputsPrev + vNOutputs) * cNEmbdOut <= (int64_t) vEmbd.size);
              ggml_backend_tensor_get_async(vBackendEmbd, vTEmbd, vEmbdOut, 0, vNOutputs * cNEmbdOut * sizeof(float));
            }
          }
          break;
        case cDgsPoolingTypeMean:
        case cDgsPoolingTypeCls:
        case cDgsPoolingTypeLast:
          {
            // extract sequence embeddings (cleared before processing each batch)
            auto & dEmbdSeqOut = dEmbdSeq;

            // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
            // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
            const uint32_t cNEmbdOut = vHparams.mNEmbdOut();

            for (uint32_t vS = 0; vS < vUbatch.vNSeqsUnq; ++vS) {
              const dgs_seq_id cSeqId  = vUbatch.vSeqIdUnq[vS];
              const int32_t    cSeqIdx = vUbatch.vSeqIdx[cSeqId];

              dEmbdSeqOut[cSeqId].resize(cNEmbdOut);
              ggml_backend_tensor_get_async(vBackendEmbd, vTEmbd, dEmbdSeqOut[cSeqId].data(),
                                            (cNEmbdOut * cSeqIdx) * sizeof(float), cNEmbdOut * sizeof(float));
            }
          }
          break;
        case cDgsPoolingTypeRank:
          {
            // extract the rerank score - n_cls_out floats per sequence
            auto & dEmbdSeqOut = dEmbdSeq;

            const uint32_t cNClsOut = vHparams.vNClsOut;

            for (uint32_t vS = 0; vS < vUbatch.vNSeqsUnq; ++vS) {
              const dgs_seq_id cSeqId  = vUbatch.vSeqIdUnq[vS];
              const int32_t    cSeqIdx = vUbatch.vSeqIdx[cSeqId];

              dEmbdSeqOut[cSeqId].resize(cNClsOut);
              ggml_backend_tensor_get_async(vBackendEmbd, vTEmbd, dEmbdSeqOut[cSeqId].data(),
                                            (cNClsOut * cSeqIdx) * sizeof(float), cNClsOut * sizeof(float));
            }
          }
          break;
        case cDgsPoolingTypeUnspecified:
          {
            GGML_ABORT("unknown pooling type");
          }
      }
    }

    // [TAG_EXTRACT_TARGET_EMBEDDINGS]
    bool vExtractAllIdxs = mExtractLayerInputs(vRes, vNTokensPrev, vUbatch.vNTokens);

    // extract nextn embeddings before
    // only meaningful in DGS_POOLING_TYPE_NONE (per-token); other pooling modes are ignored.
    {
      const bool    cMasked = vCparams.vEmbeddingsNextnMasked;
      const int64_t cNRows  = cMasked ? vNOutputs : (int64_t) vUbatch.vNTokens;
      const int64_t cOffset = cMasked ? vNOutputsPrev : vNTokensPrev;

      if (vEmbdNextn.data && vTHNextn && cNRows > 0 && vCparams.vPoolingType == cDgsPoolingTypeNone) {
        ggml_backend_t vBackendH = ggml_backend_sched_get_tensor_backend(vSched.get(), vTHNextn);
        GGML_ASSERT(vBackendH != nullptr);

        const uint32_t cNEmbd        = vHparams.mNEmbdOut();
        float *        vEmbdNextnOut = vEmbdNextn.data + cOffset * cNEmbd;

        GGML_ASSERT((cOffset + cNRows) * cNEmbd <= (int64_t) vEmbdNextn.size);
        ggml_backend_tensor_get_async(vBackendH, vTHNextn, vEmbdNextnOut, 0, cNRows * cNEmbd * sizeof(float));
        vExtractAllIdxs = vExtractAllIdxs || !cMasked;
      }
    }

    if (vExtractAllIdxs) {
      GGML_ASSERT(vUbatch.data && vUbatch.data->lBatchIdxs.size() == vUbatch.vNTokens);
      GGML_ASSERT(lEmbdBatchIdxs.size() == (size_t) vNTokensPrev);
      const auto & lBatchIdxs = vUbatch.data->lBatchIdxs;
      lEmbdBatchIdxs.insert(lEmbdBatchIdxs.end(), lBatchIdxs.begin(), lBatchIdxs.end());
    }

    if (cHasSamplers) {
      const auto cStride = cNVocab;

      // async copy the sampling data from the backend to the host
      fCopyTensorAsyncRows(vRes->lTSampled, vSampling.vSampled, 1, vNOutputsPrev, vSched.get());
      fCopyTensorAsyncRows(vRes->lTSampledLogits, vSampling.vLogits, cStride, vNOutputsPrev, vSched.get(),
                           &vSampling.lLogitsCount);
      fCopyTensorAsyncRows(vRes->lTSampledProbs, vSampling.vProbs, cStride, vNOutputsPrev, vSched.get(),
                           &vSampling.lProbsCount);
      fCopyTensorAsyncRows(vRes->lTCandidates, vSampling.vCandidates, cStride, vNOutputsPrev, vSched.get(),
                           &vSampling.lCandidatesCount);
    }

    vNOutputsPrev += vNOutputs;
    vNTokensPrev += vUbatch.vNTokens;
  } while (vMctx->mNext());

  // set to total number of outputs in the batch, for use in dgs_get_logits_ith
  vNOutputs = cNOutputsAll;

  // set output mappings
  if (vNOutputs > 0) {
    bool vSortedOutput = true;

    auto & lOutIds = vBalloc->mGetOutIds();

    GGML_ASSERT(lOutIds.size() == (size_t) vNOutputs);

    for (int64_t vI = 0; vI < vNOutputs; ++vI) {
      int64_t vOutId     = lOutIds[vI];
      lOutputIds[vOutId] = vI;
      if (vOutId != vI) {
        vSortedOutput = false;
      }
    }

    // make the outputs have the same order they had in the user-provided batch
    // note: this is mostly relevant for recurrent models atm
    if (!vSortedOutput && vNOutputs > 1) {
      GGML_ASSERT((size_t) vNOutputs == lOutIds.size());

      // TODO: is there something more efficient which also minimizes swaps?
      // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
      for (uint32_t vI = 0; vI < vNOutputs - 1; ++vI) {
        uint32_t vJMin = vI;
        for (uint32_t vJ = vI + 1; vJ < vNOutputs; ++vJ) {
          if (lOutIds[vJ] < lOutIds[vJMin]) {
            vJMin = vJ;
          }
        }
        if (vJMin == vI) {
          continue;
        }
        std::swap(lOutIds[vI], lOutIds[vJMin]);

        // remember the swaps and apply them lazily upon logits/embeddings access
        lOutputSwaps.push_back({ vI, vJMin });
      }

      std::fill(lOutputIds.begin(), lOutputIds.end(), -1);

      for (uint32_t vI = 0; vI < vNOutputs; ++vI) {
        lOutputIds[lOutIds[vI]] = vI;
      }
    }
  }

  // wait for the computation to finish (automatically done when obtaining the model output)
  //synchronize();

  return 0;
}

//
// output
//

uint32_t dgs_context::mOutputReserve(int32_t pNOutputs) {
  const auto & vHparams = vModel.hparams;
  const auto & vVocab   = vModel.vocab;

  const int64_t cNOutputsMax = std::max<int64_t>(pNOutputs, mNSeqMax());

  const auto cNBatch   = vCparams.vNBatch;
  const auto cNVocab   = vVocab.n_tokens();
  const auto cNEmbd    = vHparams.n_embd;
  const auto cNEmbdOut = vHparams.mNEmbdOut();

  // LFM2 builds logits only when embeddings are off, and set_embeddings() does not re-reserve
  bool vHasLogits    = !vCparams.vEmbeddings || vGraphHasLogits;
  bool vHasEmbd      = vCparams.vEmbeddings;
  bool vHasEmbdNextn = vCparams.vEmbeddingsNextn;

  // TODO: hacky enc-dec support
  if (vModel.vArch == cLlmArchT5) {
    vHasLogits = true;
    vHasEmbd   = true;
  }

  size_t vBackendFloatCount      = 0;
  size_t vBackendTokenCount      = 0;
  size_t vEmbdLayerInpFloatCount = 0;

  vLogits.size    = vHasLogits ? cNVocab * cNOutputsMax : 0;
  vEmbd.size      = vHasEmbd ? cNEmbdOut * cNOutputsMax : 0;
  vEmbdNextn.size = vHasEmbdNextn ? cNEmbdOut * cNOutputsMax : 0;

  if (vHasEmbdNextn && !vCparams.vEmbeddingsNextnMasked) {
    // unmasked: nextn row exists for every token in the batch, not just
    // those flagged via batch.logits[i] -> size by token count instead.
    vEmbdNextn.size = (size_t) cNEmbdOut * cNBatch;
  }

  for (bool vEnabled : vCparams.lEmbeddingsLayerInp) {
    if (vEnabled) {
      vEmbdLayerInpFloatCount += (size_t) cNEmbd * cNBatch;
    }
  }

  // Allocate backend sampling output buffers if there are backend samplers configured.
  const bool cHasSampling = !vSampling.dSamplers.empty();
  if (cHasSampling) {
    vBackendFloatCount = 2 * cNVocab * cNOutputsMax;    // logits + probs
    vBackendTokenCount = (1 + cNVocab) * cNOutputsMax;  // sampled + candidates
  }

  if (lOutputIds.empty()) {
    // init, never resized afterwards
    lOutputIds.resize(cNBatch);
  }

  const size_t cPrevSize = vBufOutput ? ggml_backend_buffer_get_size(vBufOutput.get()) : 0;
  const size_t cNewSize =
    (vLogits.size + vEmbd.size + vEmbdNextn.size + vEmbdLayerInpFloatCount + vBackendFloatCount) * sizeof(float) +
    (vBackendTokenCount) * sizeof(dgs_token);

  // alloc only when more than the current capacity is required
  // TODO: also consider shrinking the buffer
  if (!vBufOutput || cPrevSize < cNewSize) {
    if (vBufOutput) {
#ifndef NDEBUG
      // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
      DGS_LOG_DEBUG("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__,
                    prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
      mSynchronize();

      // TODO: not needed?
      vBufOutput      = nullptr;
      vLogits.data    = nullptr;
      vEmbd.data      = nullptr;
      vEmbdNextn.data = nullptr;
      for (auto & vLayerInp : lEmbdLayerInp) {
        vLayerInp = { nullptr, 0 };
      }
    }

    auto * vBuft              = ggml_backend_cpu_buffer_type();
    // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
    auto * vOutputDev         = vModel.mDevOutput();
    auto * vOutputDevHostBuft = vOutputDev ? ggml_backend_dev_host_buffer_type(vOutputDev) : nullptr;
    if (vOutputDevHostBuft) {
      vBuft = vOutputDevHostBuft;
    }
    vBufOutput.reset(ggml_backend_buft_alloc_buffer(vBuft, cNewSize));
    if (vBufOutput == nullptr) {
      DGS_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, cNewSize / (1024.0 * 1024.0));
      return 0;
    }
    ggml_backend_buffer_clear(vBufOutput.get(), 0);
  }

  float * vOutputBase = (float *) ggml_backend_buffer_get_base(vBufOutput.get());

  size_t    vOffset = 0;
  uint8_t * vBase   = (uint8_t *) vOutputBase;

  vLogits = vHasLogits ? buffer_view<float>{ vOutputBase, vLogits.size } : buffer_view<float>{ nullptr, 0 };
  vOffset += vLogits.size * sizeof(float);

  vEmbd = vHasEmbd ? buffer_view<float>{ (float *) (vBase + vOffset), vEmbd.size } : buffer_view<float>{ nullptr, 0 };
  vOffset += vEmbd.size * sizeof(float);

  vEmbdNextn = vHasEmbdNextn ? buffer_view<float>{ (float *) (vBase + vOffset), vEmbdNextn.size } :
                               buffer_view<float>{ nullptr, 0 };
  vOffset += vEmbdNextn.size * sizeof(float);

  for (uint32_t vIl = 0; vIl < lEmbdLayerInp.size(); ++vIl) {
    if (vCparams.lEmbeddingsLayerInp[vIl]) {
      lEmbdLayerInp[vIl] = buffer_view<float>{ (float *) (vBase + vOffset), (size_t) cNEmbd * cNBatch };
      vOffset += lEmbdLayerInp[vIl].size * sizeof(float);
    } else {
      lEmbdLayerInp[vIl] = buffer_view<float>{ nullptr, 0 };
    }
  }

  if (cHasSampling) {
    vSampling.vLogits = { (float *) (vBase + vOffset), (size_t) (cNVocab * cNOutputsMax) };
    vOffset += vSampling.vLogits.size * sizeof(float);

    vSampling.vProbs = { (float *) (vBase + vOffset), (size_t) (cNVocab * cNOutputsMax) };
    vOffset += vSampling.vProbs.size * sizeof(float);

    vSampling.vSampled = { (dgs_token *) (vBase + vOffset), (size_t) cNOutputsMax };
    vOffset += vSampling.vSampled.size * sizeof(dgs_token);

    vSampling.vCandidates = { (dgs_token *) (vBase + vOffset), (size_t) (cNVocab * cNOutputsMax) };
    vOffset += vSampling.vCandidates.size * sizeof(dgs_token);

    // The count vectors keep track of the actual number of logits/probs/candidates
    // copied from the backend for each output row.

    vSampling.lLogitsCount.resize(cNOutputsMax);
    vSampling.lProbsCount.resize(cNOutputsMax);
    vSampling.lCandidatesCount.resize(cNOutputsMax);

    std::fill(vSampling.lLogitsCount.begin(), vSampling.lLogitsCount.end(), 0);
    std::fill(vSampling.lProbsCount.begin(), vSampling.lProbsCount.end(), 0);
    std::fill(vSampling.lCandidatesCount.begin(), vSampling.lCandidatesCount.end(), 0);

    std::fill_n(vSampling.vSampled.data, vSampling.vSampled.size, DGS_TOKEN_NULL);
  } else {
    vSampling.vLogits     = { nullptr, 0 };
    vSampling.vProbs      = { nullptr, 0 };
    vSampling.vSampled    = { nullptr, 0 };
    vSampling.vCandidates = { nullptr, 0 };

    vSampling.lLogitsCount.clear();
    vSampling.lProbsCount.clear();
    vSampling.lCandidatesCount.clear();
  }

  // set all ids as invalid (negative)
  std::fill(lOutputIds.begin(), lOutputIds.end(), -1);

  this->vNOutputs = 0;

  GGML_ASSERT(cNOutputsMax <= vCparams.vNOutputsMax);

  return cNOutputsMax;
}

bool dgs_context::mExtractLayerInputs(const llm_graph_result * pRes, size_t pTokenOffset, size_t pNTokens) {
  bool vExtracted = false;
  for (uint32_t vIl = 0; vIl < vCparams.lEmbeddingsLayerInp.size(); ++vIl) {
    if (!vCparams.lEmbeddingsLayerInp[vIl]) {
      continue;
    }
    if (!lEmbdLayerInp[vIl].mHasData()) {
      GGML_ABORT("output layer input buffer not allocated");
    }
    ggml_tensor * vT = pRes->mGetLayerInp((int) vIl);
    if (!vT) {
      GGML_ABORT("layer input tensor not found");
    }

    const size_t cNbytes  = ggml_nbytes(vT);
    const size_t cNfloats = cNbytes / sizeof(float);
    GGML_ASSERT(pNTokens > 0);
    GGML_ASSERT(cNfloats % pNTokens == 0);

    const size_t cRowFloats = cNfloats / pNTokens;
    GGML_ASSERT(cRowFloats == vModel.hparams.n_embd);
    const size_t cDstOffset = pTokenOffset * cRowFloats;
    GGML_ASSERT(cDstOffset + cNfloats <= lEmbdLayerInp[vIl].size);

    ggml_backend_t vBackend = ggml_backend_sched_get_tensor_backend(vSched.get(), vT);
    GGML_ASSERT(vBackend != nullptr);
    // Tensor-split backends require a zero source offset.
    ggml_backend_tensor_get_async(vBackend, vT, lEmbdLayerInp[vIl].data + cDstOffset, 0, cNbytes);
    vExtracted = true;
  }
  return vExtracted;
}

void dgs_context::mOutputReorder() {
  const uint64_t cNVocab   = vModel.vocab.n_tokens();
  const uint64_t cNEmbd    = vModel.hparams.n_embd;
  const uint64_t cNEmbdOut = vModel.hparams.mNEmbdOut();

  for (size_t vS = 0; vS < lOutputSwaps.size(); ++vS) {
    const uint64_t cI0 = lOutputSwaps[vS].vI0;
    const uint64_t cI1 = lOutputSwaps[vS].vI1;

    if (vLogits.size > 0) {
      for (uint64_t vK = 0; vK < cNVocab; vK++) {
        std::swap(vLogits.data[cI0 * cNVocab + vK], vLogits.data[cI1 * cNVocab + vK]);
      }
    }

    if (vEmbd.size > 0) {
      for (uint64_t vK = 0; vK < cNEmbdOut; vK++) {
        std::swap(vEmbd.data[cI0 * cNEmbdOut + vK], vEmbd.data[cI1 * cNEmbdOut + vK]);
      }
    }

    if (vEmbdNextn.size > 0 && vCparams.vEmbeddingsNextnMasked) {
      for (uint64_t vK = 0; vK < cNEmbdOut; vK++) {
        std::swap(vEmbdNextn.data[cI0 * cNEmbdOut + vK], vEmbdNextn.data[cI1 * cNEmbdOut + vK]);
      }
    }

    if (!vSampling.dSamplers.empty()) {
      assert(sampling.logits.size > 0);
      assert(sampling.probs.size > 0);
      assert(sampling.candidates.size > 0);
      assert(sampling.sampled.size > 0);
      assert(sampling.logits_count.size() > 0);
      assert(sampling.probs_count.size() > 0);
      assert(sampling.candidates_count.size() > 0);

      for (uint64_t vK = 0; vK < cNVocab; ++vK) {
        std::swap(vSampling.vLogits.data[cI0 * cNVocab + vK], vSampling.vLogits.data[cI1 * cNVocab + vK]);
      }

      for (uint64_t vK = 0; vK < cNVocab; ++vK) {
        std::swap(vSampling.vProbs.data[cI0 * cNVocab + vK], vSampling.vProbs.data[cI1 * cNVocab + vK]);
      }

      for (uint64_t vK = 0; vK < cNVocab; ++vK) {
        std::swap(vSampling.vCandidates.data[cI0 * cNVocab + vK], vSampling.vCandidates.data[cI1 * cNVocab + vK]);
      }

      std::swap(vSampling.vSampled.data[cI0], vSampling.vSampled.data[cI1]);
      std::swap(vSampling.lLogitsCount[cI0], vSampling.lLogitsCount[cI1]);
      std::swap(vSampling.lProbsCount[cI0], vSampling.lProbsCount[cI1]);
      std::swap(vSampling.lCandidatesCount[cI0], vSampling.lCandidatesCount[cI1]);
    }
  }

  lOutputSwaps.clear();

  // [TAG_EXTRACT_TARGET_EMBEDDINGS]
  // Layer inputs and unmasked NextN embeddings contain all token rows, independent of logits selection.
  for (size_t vI = 0; vI < lEmbdBatchIdxs.size(); ++vI) {
    while (lEmbdBatchIdxs[vI] != (int32_t) vI) {
      const int32_t cJ = lEmbdBatchIdxs[vI];
      GGML_ASSERT(cJ >= 0 && (size_t) cJ < lEmbdBatchIdxs.size());
      if (vEmbdNextn.mHasData() && !vCparams.vEmbeddingsNextnMasked) {
        for (size_t vK = 0; vK < cNEmbdOut; ++vK) {
          std::swap(vEmbdNextn.data[vI * cNEmbdOut + vK], vEmbdNextn.data[cJ * cNEmbdOut + vK]);
        }
      }
      for (auto & layer : lEmbdLayerInp) {
        if (layer.mHasData()) {
          for (size_t vK = 0; vK < cNEmbd; ++vK) {
            std::swap(layer.data[vI * cNEmbd + vK], layer.data[cJ * cNEmbd + vK]);
          }
        }
      }
      std::swap(lEmbdBatchIdxs[vI], lEmbdBatchIdxs[cJ]);
    }
  }
  lEmbdBatchIdxs.clear();
}

//
// graph
//

uint32_t dgs_context::mGraphMaxNodes(uint32_t pNTokens) const {
  uint32_t vRes;
  if (vModel.vArch == cLlmArchKimiK3 || vModel.vArch == cLlmArchGlm5Next) {
    // the n_tokens*40 budget below is exhausted at ubatch 3840
    vRes = std::max<uint32_t>(pNTokens * 160, 64u * vModel.mNTensors());
  } else if (vModel.vArch == cLlmArchHrmText) {
    // the 128-slot looped graph needs roughly one stack per token budget
    vRes = std::max<uint32_t>(pNTokens * 80, 64u * vModel.mNTensors());
  } else if (vModel.vArch == cLlmArchQwen3next || vModel.vArch == cLlmArchKimiLinear ||
             vModel.vArch == cLlmArchBailingmoe3 || vModel.vArch == cLlmArchQwen35 ||
             vModel.vArch == cLlmArchQwen35moe || vModel.vArch == cLlmArchClef || vModel.vArch == cLlmArchQwen4exp ||
             vModel.vArch == cLlmArchDeepseek4 || (vModel.vArch == cLlmArchDflash && vModel.hparams.vDsv4HcMult > 0) ||
             vModel.vArch == cLlmArchNanbeige || vModel.vArch == cLlmArchMinimax01 ||
             vModel.vArch == cLlmArchMinimaxM3 || vModel.vArch == cLlmArchHyV4) {
    vRes = std::max<uint32_t>(pNTokens * 40, 32u * vModel.mNTensors());
  } else if (vModel.vArch == cLlmArchDflash && vModel.hparams.vDflashSelectorRank > 0) {
    // DFlash2's convolutions and selector are shape work rather than matmuls,
    // so they cost ~8.6 nodes per tensor against ~5.9 for a plain DFlash draft
    vRes = std::max<uint32_t>(1024u, 12u * vModel.mNTensors());
  } else {
    vRes = std::max<uint32_t>(1024u, 8u * vModel.mNTensors());
    for (const auto & lora : vModel.sLoras) {
      vRes += lora->mGetNNodes();
    }
  }

  // every sampler builds n_outputs_max_per_seq chains, see llm_graph_context::build_sampling
  for (const auto & [seq_id, sampler] : vSampling.dSamplers) {
    vRes += fDgsSamplerBackendNNodes(sampler) * vCparams.vNOutputsMaxPerSeq;
  }

  return vRes;
}

llm_graph_result * dgs_context::mGetGfResReserve() const {
  return static_cast<llm_graph_result *>(vGfResReserve.get());
}

llm_graph_result * dgs_context::mGetGfResPrev() {
  auto & vRes = aGfResPrev[vNOutputs > 0];
  if (!vRes) {
    vRes.reset(new llm_graph_result(vGfResReserve->mGetMaxNodes()));
  }
  return vRes.get();
}

// pack sampler outputs into as few sequences as possible before using sequences without samplers
static void fUbatchPrepareReserve(dgs_ubatch &                                pUbatch,
                                  uint32_t                                    pNOutputs,
                                  const std::map<dgs_seq_id, dgs_sampler *> & pSamplers,
                                  uint32_t                                    pNOutputsMaxPerSeq) {
  const uint32_t cNSeqs      = pUbatch.vNSeqs;
  const uint32_t cNSeqTokens = pUbatch.vNSeqTokens;

  for (uint32_t vS = 0; vS < cNSeqs; ++vS) {
    for (uint32_t vT = 0; vT < cNSeqTokens; ++vT) {
      const uint32_t cI   = vS * cNSeqTokens + vT;
      pUbatch.vNSeqId[cI] = 1;
      pUbatch.vSeqId[cI]  = &pUbatch.vSeqIdUnq[vS];
    }
  }

  // sequences with a sampler that fit in this ubatch
  std::vector<uint32_t> lSamplerSeqs;
  std::vector<bool>     lHasSampler(cNSeqs, false);
  for (const auto & vEntry : pSamplers) {
    const dgs_seq_id cSeqId = vEntry.first;
    if (cSeqId < 0 || (uint32_t) cSeqId >= cNSeqs) {
      continue;
    }

    lSamplerSeqs.push_back(cSeqId);
    lHasSampler[cSeqId] = true;
  }

  uint32_t vNOutputsSet = 0;

  const uint32_t cNOutputsPerSeq = std::min(cNSeqTokens, pNOutputsMaxPerSeq);
  for (uint32_t vS : lSamplerSeqs) {
    if (vNOutputsSet >= pNOutputs) {
      break;
    }

    for (uint32_t vT = 0; vT < cNOutputsPerSeq && vNOutputsSet < pNOutputs; ++vT) {
      pUbatch.vOutput[vS * cNSeqTokens + vT] = true;
      ++vNOutputsSet;
    }
  }

  // use sequences without samplers for any remaining outputs
  for (uint32_t vT = 0; vT < cNSeqTokens && vNOutputsSet < pNOutputs; ++vT) {
    for (uint32_t vS = 0; vS < cNSeqs && vNOutputsSet < pNOutputs; ++vS) {
      if (lHasSampler[vS]) {
        continue;
      }

      pUbatch.vOutput[vS * cNSeqTokens + vT] = true;
      ++vNOutputsSet;
    }
  }
}

ggml_cgraph * dgs_context::mGraphReserve(uint32_t                     pNTokens,
                                         uint32_t                     pNSeqs,
                                         uint32_t                     pNOutputs,
                                         const dgs_memory_context_i * pMctx,
                                         bool                         pSplitOnly,
                                         size_t *                     pSizes) {
  DGS_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__,
                pNTokens, pNSeqs, pNOutputs);
  GGML_ASSERT(pNOutputs >= 1);

  if (pNTokens % pNSeqs != 0) {
    pNTokens = ((pNTokens + (pNSeqs - 1)) / pNSeqs) * pNSeqs;  // round to next multiple of n_seqs
    DGS_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", __func__,
                  pNTokens, pNSeqs, pNOutputs);
  }

  ggml_backend_sched_reset(vSched.get());

  // when the scheduler is reset, we cannot reuse old graphs, so we reset the previous graph results
  for (auto & vRes : aGfResPrev) {
    if (vRes) {
      vRes->reset();
    }
  }
  vGfResPrevActive = nullptr;

  // store the n_outputs as it is, and restore it afterwards
  // TODO: not sure if needed, might simplify in the future by removing this
  const auto cSaveNOutputs = this->vNOutputs;

  this->vNOutputs = pNOutputs;

  dgs_batch_allocr vBalloc(vModel.hparams.mNPosPerEmbd());
  dgs_ubatch       vUbatch = vBalloc.mUbatchReserve(pNTokens / pNSeqs, pNSeqs);

  fUbatchPrepareReserve(vUbatch, pNOutputs, vSampling.dSamplers, vCparams.vNOutputsMaxPerSeq);

  auto * vRes = vGfResReserve.get();

  const auto cGparams = mGraphParams(vRes, vUbatch, pMctx, fCtxTypeToGraphType(vCparams.vCtxType));

  vRes->reset();

  auto * vGf = vModel.mBuildGraph(cGparams);

  // check the graph before scheduling: cross-backend copies break the producer chain that the check follows
  if (!vHadamardVerified && vGf && (!vModel.vHdmd.dRot.empty() || !vModel.vHdmd.dInv.empty())) {
    fDgsVerifyHadamardGraph(vGf, vModel.vHdmd.dRot, vModel.vHdmd.dInv);
    vHadamardVerified = true;
  }

  this->vNInputTensors = fDgsGraphNInputTensors(vGf);
  this->vNOutputs      = cSaveNOutputs;

  // initialize scheduler with the specified graph
  if (pSplitOnly) {
    if (pSizes) {
      ggml_backend_sched_reserve_size(vSched.get(), vGf, pSizes);
    } else {
      ggml_backend_sched_split_graph(vSched.get(), vGf);
    }
  } else if (!ggml_backend_sched_reserve(vSched.get(), vGf)) {
    GGML_ASSERT(!pSizes);
    DGS_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
    return nullptr;
  }

  return vGf;
}

llm_graph_params dgs_context::mGraphParams(llm_graph_result *           pRes,
                                           const dgs_ubatch &           pUbatch,
                                           const dgs_memory_context_i * pMctx,
                                           llm_graph_type               pGtype) const {
  return {
    /*.arch        =*/vModel.vArch,
    /*.hparams     =*/vModel.hparams,
    /*.cparams     =*/vCparams,
    /*.ubatch      =*/pUbatch,
    /*.gtype       =*/pGtype,
    /*.sched       =*/vSched.get(),
    /*.backend_cpu =*/vBackendCpu,
    /*.cvec        =*/vCvec.get(),
    /*.loras       =*/dLoras.get(),
    /*.mctx        =*/pMctx,
    /*.cross       =*/&vCross,
    /*.prec_policy =*/&vModel.vPrecPolicy,
    /*.hdmd        =*/vModel.vHdmd.dRot.empty() ? nullptr : &vModel.vHdmd,
    /*.samplers    =*/vSampling.dSamplers,
    /*.n_outputs   =*/vNOutputs,
    /*.cb          =*/mGraphGetCb(),
    /*.res         =*/pRes,
  };
}

void dgs_context::mSchedGraphRealloc(ggml_backend_sched_t pSched,
                                     ggml_cgraph *        pGf,
                                     bool                 pBackendIdsChanged,
                                     bool                 pGraphSizeChanged,
                                     void *               pUserData) {
  // reallocating after the backend assignments changed is expected, trace only the unexpected ones
  if (pBackendIdsChanged) {
    return;
  }

  auto * vLctx = static_cast<dgs_context *>(pUserData);

  if (pGraphSizeChanged) {
    vLctx->vNReallocsDiff++;
  } else {
    vLctx->vNReallocsSame++;
  }

  DGS_LOG_INFO("%s: graph reallocation: nodes = %d (changed = %d), splits = %d, n_same = %d, n_diff = %d\n", __func__,
               ggml_graph_n_nodes(pGf), ggml_backend_sched_get_n_splits(pSched), pGraphSizeChanged,
               vLctx->vNReallocsSame, vLctx->vNReallocsDiff);
}

ggml_status dgs_context::mGraphCompute(ggml_cgraph * pGf, bool pBatched) {
  int               vNThreads = pBatched ? vCparams.vNThreadsBatch : vCparams.vNThreads;
  ggml_threadpool_t vTp       = pBatched ? vThreadpoolBatch : vThreadpool;

  if (vBackendCpu != nullptr) {
    auto * vReg             = ggml_backend_dev_backend_reg(ggml_backend_get_device(vBackendCpu));
    auto * vSetThreadpoolFn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(
      vReg, "ggml_backend_cpu_set_threadpool");
    if (vSetThreadpoolFn) {
      vSetThreadpoolFn(vBackendCpu, vTp);
    }
  }

  // set the number of threads for all the backends
  for (const auto & vSetNThreadsFn : lSetNThreadsFns) {
    vSetNThreadsFn.second(vSetNThreadsFn.first, vNThreads);
  }

  auto vStatus = ggml_backend_sched_graph_compute_async(vSched.get(), pGf);
  if (vStatus != GGML_STATUS_SUCCESS) {
    DGS_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, vStatus);
  }

  // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

  return vStatus;
}

llm_graph_cb dgs_context::mGraphGetCb() const {
  return [&](const dgs_ubatch & pUbatch, ggml_tensor * pCur, const char * name, int pIl) {
    if (pIl >= 0) {
      ggml_format_name(pCur, "%s-%d", name, pIl);
    } else {
      ggml_set_name(pCur, name);
    }

    // - norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
    // - force the last op of the layer on the specified backend to avoid running it on the backend of the next layer due to scheduling
    // FIXME: fix in ggml_backend_sched
    const bool cFullOffload = vModel.mNGpuLayers() > vModel.hparams.n_layer_all;
    if (pUbatch.vNTokens < 32 || cFullOffload) {
      if (pIl != -1 && (strcmp(name, "norm") == 0 || strcmp(name, "l_last") == 0)) {
        const auto & vDevLayer = vModel.mDevLayer(pIl);
        for (const auto & vBackend : lBackends) {
          if (ggml_backend_get_device(vBackend.get()) == vDevLayer) {
            if (ggml_backend_supports_op(vBackend.get(), pCur)) {
              ggml_backend_sched_set_tensor_backend(vSched.get(), pCur, vBackend.get());
            }
          }
        }
      }
    }
  };
}

//
// state save/load
//

class dgs_io_write_dummy : public dgs_io_write_i {
public:
  dgs_io_write_dummy(bool pSkipTensors) : cSkipTensors(pSkipTensors) {}

  void write(const void * /* src */, size_t size) override { vSizeWritten += size; }

  void mWriteTensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
    if (cSkipTensors) {
      return;
    }

    vSizeWritten += size;
  }

  size_t mNBytes() override { return vSizeWritten; }

private:
  const bool cSkipTensors;

  size_t vSizeWritten = 0;
};

class dgs_io_write_host : public dgs_io_write_i {
public:
  dgs_io_write_host(uint8_t * pP, size_t pLen) : vPtr(pP), vBufSize(pLen) {}

  ~dgs_io_write_host() {
    // TODO: add backend support to batch tensor_get? or some other way to speed this up
    for (const auto & vWinfo : lWinfos) {
      ggml_backend_tensor_get(vWinfo.vTensor, vWinfo.vPtr, vWinfo.vOffset, vWinfo.size);
    }
  }

  void write(const void * pSrc, size_t size) override {
    if (size > vBufSize) {
      throw std::runtime_error("unexpectedly reached end of buffer");
    }
    memcpy(vPtr, pSrc, size);
    vPtr += size;
    vSizeWritten += size;
    vBufSize -= size;
  }

  void mWriteTensor(ggml_tensor * pTensor, size_t pOffset, size_t size) override {
    if (size > vBufSize) {
      throw std::runtime_error("unexpectedly reached end of buffer");
    }

    // save the write for later during destruction
    lWinfos.push_back({ pTensor, vPtr, size, pOffset });

    vPtr += size;
    vSizeWritten += size;
    vBufSize -= size;
  }

  size_t mNBytes() override { return vSizeWritten; }

private:
  uint8_t * vPtr;
  size_t    vBufSize     = 0;
  size_t    vSizeWritten = 0;

  struct write_info {
    ggml_tensor * vTensor;
    uint8_t *     vPtr;
    size_t        size;
    size_t        vOffset;
  };

  std::vector<write_info> lWinfos;
};

class dgs_io_read_host : public dgs_io_read_i {
public:
  dgs_io_read_host(const uint8_t * pP, size_t pLen) : vPtr(pP), vBufSize(pLen) {}

  ~dgs_io_read_host() {
    // flush the reads
    for (const auto & vRinfo : lRinfos) {
      ggml_backend_tensor_set(vRinfo.vTensor, vRinfo.vPtr, vRinfo.vOffset, vRinfo.size);
    }
  }

  void read(void * pDst, size_t size) override {
    if (size > vBufSize) {
      throw std::runtime_error("unexpectedly reached end of buffer");
    }
    memcpy(pDst, vPtr, size);
    vPtr += size;
    vSizeRead += size;
    vBufSize -= size;
  }

  void mReadTensor(ggml_tensor * pTensor, size_t pOffset, size_t size) override {
    if (size > vBufSize) {
      throw std::runtime_error("unexpectedly reached end of buffer");
    }

    // save for later during destruction
    lRinfos.push_back({ pTensor, vPtr, size, pOffset });

    vPtr += size;
    vSizeRead += size;
    vBufSize -= size;
  }

  void mDiscard() override { lRinfos.clear(); }

  size_t mNBytes() override { return vSizeRead; }

private:
  const uint8_t * vPtr;
  size_t          vBufSize  = 0;
  size_t          vSizeRead = 0;

  struct read_info {
    ggml_tensor *   vTensor;
    const uint8_t * vPtr;
    size_t          size;
    size_t          vOffset;
  };

  std::vector<read_info> lRinfos;
};

class dgs_io_write_file : public dgs_io_write_i {
public:
  dgs_io_write_file(dgs_file * pF) : vFile(pF) {}

  void write(const void * pSrc, size_t size) override {
    vFile->mWriteRaw(pSrc, size);
    vSizeWritten += size;
  }

  void mWriteTensor(ggml_tensor * pTensor, size_t pOffset, size_t size) override {
    lTempBuffer.resize(size);
    ggml_backend_tensor_get(pTensor, lTempBuffer.data(), pOffset, size);
    write(lTempBuffer.data(), lTempBuffer.size());
  }

  size_t mNBytes() override { return vSizeWritten; }

private:
  dgs_file *           vFile;
  size_t               vSizeWritten = 0;
  std::vector<uint8_t> lTempBuffer;
};

class dgs_io_read_file : public dgs_io_read_i {
public:
  dgs_io_read_file(dgs_file * pF) : vFile(pF) {}

  void read(void * pDst, size_t size) override {
    vFile->mReadRaw(pDst, size);
    vSizeRead += size;
  }

  void mReadTensor(ggml_tensor * pTensor, size_t pOffset, size_t size) override {
    lTempBuffer.resize(size);
    read(lTempBuffer.data(), size);
    ggml_backend_tensor_set(pTensor, lTempBuffer.data(), pOffset, size);
  }

  size_t mNBytes() override { return vSizeRead; }

private:
  dgs_file *           vFile;
  size_t               vSizeRead = 0;
  std::vector<uint8_t> lTempBuffer;
};

class dgs_io_write_device : public dgs_io_write_i {
public:
  dgs_io_write_device(uint8_t * pP, size_t pLen, dgs_memory_buffers & pMbufs) :
    vPtr(pP),
    vBufSize(pLen),
    dMbufs(pMbufs) {}

  ~dgs_io_write_device() {
    dgs_memory_buffers dMbufsNew;

    for (const auto & vWinfo : lWinfos) {
      auto * vBuft = ggml_backend_buffer_get_type(vWinfo.vTensor->buffer);

      dMbufsNew[vBuft].vNTensors++;
      dMbufsNew[vBuft].vTotalSize += vWinfo.size;
    }

    for (auto & [buft, mbuf] : dMbufsNew) {
      ggml_init_params vParams = {
        /*.mem_size   =*/2 * mbuf.vNTensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
      };

      mbuf.vCtx.reset(ggml_init(vParams));

      mbuf.lOrg.reserve(mbuf.vNTensors);
      mbuf.lCpy.reserve(mbuf.vNTensors);
    }

    for (const auto & vWinfo : lWinfos) {
      auto * vBuft = ggml_backend_buffer_get_type(vWinfo.vTensor->buffer);

      const int64_t n = vWinfo.size / ggml_element_size(vWinfo.vTensor);

      auto & vMbuf = dMbufsNew[vBuft];

      vMbuf.lOrg.push_back(ggml_view_1d(vMbuf.vCtx.get(), vWinfo.vTensor, n, vWinfo.vOffset));
      vMbuf.lCpy.push_back(ggml_new_tensor_1d(vMbuf.vCtx.get(), vWinfo.vTensor->type, n));
    }

    for (auto & [buft, mbuf] : dMbufsNew) {
      auto & vMbufCur = dMbufs[buft];

      bool vNeedAlloc = false;

      vNeedAlloc = vNeedAlloc || (!vMbufCur.vBuf);
      vNeedAlloc = vNeedAlloc || (vMbufCur.lOrg.size() != mbuf.lOrg.size());
      vNeedAlloc = vNeedAlloc || (vMbufCur.vTotalSize != mbuf.vTotalSize);

      if (!vNeedAlloc) {
        for (size_t vI = 0; vI < vMbufCur.lOrg.size(); ++vI) {
          auto * vOrg0 = vMbufCur.lOrg[vI];
          auto * vOrg1 = mbuf.lOrg[vI];

          if (!ggml_are_same_shape(vOrg0, vOrg1)) {
            vNeedAlloc = true;
            break;
          }

          if (vOrg0->view_src != vOrg1->view_src || vOrg0->view_offs != vOrg1->view_offs) {
            vNeedAlloc = true;
            break;
          }
        }
      }

      if (vNeedAlloc) {
        if (!vMbufCur.vBuf || vMbufCur.vTotalSize != mbuf.vTotalSize) {
          vMbufCur = std::move(mbuf);

          vMbufCur.vBuf.reset(ggml_backend_alloc_ctx_tensors_from_buft(vMbufCur.vCtx.get(), buft));

          DGS_LOG_INFO("%s: allocated '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft),
                       mbuf.vTotalSize / 1024.0 / 1024.0);
        } else {
          //DGS_LOG_INFO("%s: reallocating tensors in '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);

          // save the old buffer and allocate the new tensors in it
          auto vBuf = std::move(vMbufCur.vBuf);

          vMbufCur = std::move(mbuf);

          ggml_tallocr vTalloc = ggml_tallocr_new(vBuf.get());

          for (size_t vI = 0; vI < vMbufCur.lOrg.size(); ++vI) {
            ggml_backend_view_init(vMbufCur.lOrg[vI]);
            ggml_tallocr_alloc(&vTalloc, vMbufCur.lCpy[vI]);
          }

          vMbufCur.vBuf = std::move(vBuf);
        }
      }

      for (size_t vI = 0; vI < vMbufCur.lOrg.size(); ++vI) {
        ggml_backend_tensor_copy(vMbufCur.lOrg[vI], vMbufCur.lCpy[vI]);
      }
    }
  }

  void write(const void * pSrc, size_t size) override {
    if (size > vBufSize) {
      throw std::runtime_error("unexpectedly reached end of buffer");
    }
    memcpy(vPtr, pSrc, size);
    vPtr += size;
    vSizeWritten += size;
    vBufSize -= size;
  }

  void mWriteTensor(ggml_tensor * pTensor, size_t pOffset, size_t size) override {
    // save the write for later during destruction
    lWinfos.push_back({ pTensor, vPtr, size, pOffset });
  }

  size_t mNBytes() override { return vSizeWritten; }

private:
  uint8_t * vPtr;
  size_t    vBufSize     = 0;
  size_t    vSizeWritten = 0;

  struct write_info {
    ggml_tensor * vTensor;
    uint8_t *     vPtr;
    size_t        size;
    size_t        vOffset;
  };

  std::vector<write_info> lWinfos;

  dgs_memory_buffers & dMbufs;
};

class dgs_io_read_device : public dgs_io_read_i {
public:
  dgs_io_read_device(const uint8_t * pP, size_t pLen, const dgs_memory_buffers & pMbufs) :
    vPtr(pP),
    vBufSize(pLen),
    dMbufs(pMbufs) {}

  ~dgs_io_read_device() {
    dgs_memory_buffers dMbufsNew;

    for (const auto & vRinfo : lRinfos) {
      auto * vBuft = ggml_backend_buffer_get_type(vRinfo.vTensor->buffer);

      dMbufsNew[vBuft].vNTensors++;
      dMbufsNew[vBuft].vTotalSize += vRinfo.size;
    }

    for (auto & [buft, mbuf] : dMbufsNew) {
      ggml_init_params vParams = {
        /*.mem_size   =*/mbuf.vNTensors * ggml_tensor_overhead(),
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
      };

      mbuf.vCtx.reset(ggml_init(vParams));

      mbuf.lOrg.reserve(mbuf.vNTensors);
    }

    for (const auto & vRinfo : lRinfos) {
      auto * vBuft = ggml_backend_buffer_get_type(vRinfo.vTensor->buffer);

      const int64_t n = vRinfo.size / ggml_element_size(vRinfo.vTensor);

      auto & vMbuf = dMbufsNew[vBuft];

      vMbuf.lOrg.push_back(ggml_view_1d(vMbuf.vCtx.get(), vRinfo.vTensor, n, vRinfo.vOffset));

      ggml_backend_view_init(vMbuf.lOrg.back());
    }

    for (auto & [buft, mbuf] : dMbufsNew) {
      const auto & vMbufCur = dMbufs.at(buft);

      if (!vMbufCur.vBuf || vMbufCur.vTotalSize != mbuf.vTotalSize) {
        GGML_ABORT("%s: memory buffer mismatch\n", __func__);
      }

      if (vMbufCur.vNTensors == mbuf.vNTensors) {
        // an equal tensor count does not imply the same chunking, e.g. save ranges [2,1] vs restore runs [1,2]
        bool vSameChunking = true;
        for (size_t vI = 0; vI < vMbufCur.lOrg.size(); ++vI) {
          if (ggml_nbytes(vMbufCur.lCpy[vI]) != ggml_nbytes(mbuf.lOrg[vI])) {
            vSameChunking = false;
            break;
          }
        }

        if (vSameChunking) {
          // same chunking: copy 1:1 by index
          for (size_t vI = 0; vI < vMbufCur.lOrg.size(); ++vI) {
            ggml_backend_tensor_copy(vMbufCur.lCpy[vI], mbuf.lOrg[vI]);
          }
          continue;
        }
      }

      // different chunking: copy the write-side data (mbuf_cur.cpy) into the read-side targets (mbuf.org)
      // with a byte cursor. Write and read enumerate the same logical data in the same order but may chunk
      // it differently (even with an equal number of tensors), so copy across tensor boundaries rather than
      // 1:1 by index.
      const size_t cTotal = vMbufCur.vTotalSize;

      ggml_init_params vParamsScratch = {
        /*.mem_size   =*/2 * (vMbufCur.lCpy.size() + mbuf.lOrg.size()) * ggml_tensor_overhead(),
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
      };
      ggml_context * vCtxScratch = ggml_init(vParamsScratch);

      size_t vSrcPos  = 0;
      size_t vDstPos  = 0;
      size_t vSrcJ    = 0;
      size_t vDstI    = 0;
      size_t vSrcBase = 0;
      size_t vDstBase = 0;

      while (vSrcPos < cTotal) {
        const auto & vSrcT = vMbufCur.lCpy[vSrcJ];
        const auto & vDstT = mbuf.lOrg[vDstI];

        const size_t cSrcSize = ggml_nbytes(vSrcT);
        const size_t cDstSize = ggml_nbytes(vDstT);

        const size_t cSrcOff = vSrcPos - vSrcBase;
        const size_t cDstOff = vDstPos - vDstBase;

        const size_t cNCopy = std::min(cSrcSize - cSrcOff, cDstSize - cDstOff);

        const size_t  cEl  = ggml_element_size(vSrcT);
        const int64_t cNEl = (int64_t) (cNCopy / cEl);

        auto * vSrcV = ggml_view_1d(vCtxScratch, vSrcT, cNEl, cSrcOff);
        ggml_backend_view_init(vSrcV);
        auto * vDstV = ggml_view_1d(vCtxScratch, vDstT, cNEl, cDstOff);
        ggml_backend_view_init(vDstV);

        ggml_backend_tensor_copy(vSrcV, vDstV);

        vSrcPos += cNCopy;
        vDstPos += cNCopy;

        if (vSrcPos - vSrcBase == cSrcSize) {
          vSrcBase = vSrcPos;
          ++vSrcJ;
        }
        if (vDstPos - vDstBase == cDstSize) {
          vDstBase = vDstPos;
          ++vDstI;
        }
      }

      GGML_ASSERT(vSrcPos == cTotal && vDstPos == cTotal);
      // any tensors left unvisited hold no data
      for (size_t vI = vSrcJ; vI < vMbufCur.lCpy.size(); ++vI) {
        GGML_ASSERT(ggml_nbytes(vMbufCur.lCpy[vI]) == 0);
      }
      for (size_t vI = vDstI; vI < mbuf.lOrg.size(); ++vI) {
        GGML_ASSERT(ggml_nbytes(mbuf.lOrg[vI]) == 0);
      }

      ggml_free(vCtxScratch);
    }

    GGML_ASSERT(vBufSize == 0);
  }

  void read(void * pDst, size_t size) override {
    if (size > vBufSize) {
      throw std::runtime_error("unexpectedly reached end of buffer");
    }
    memcpy(pDst, vPtr, size);
    vPtr += size;
    vSizeRead += size;
    vBufSize -= size;
  }

  void mReadTensor(ggml_tensor * pTensor, size_t pOffset, size_t size) override {
    // save for later during destruction
    lRinfos.push_back({ pTensor, vPtr, size, pOffset });
  }

  void mDiscard() override {
    lRinfos.clear();
    vBufSize = 0;
  }

  size_t mNBytes() override { return vSizeRead; }

private:
  const uint8_t * vPtr;
  size_t          vBufSize  = 0;
  size_t          vSizeRead = 0;

  struct read_info {
    ggml_tensor *   vTensor;
    const uint8_t * vPtr;
    size_t          size;
    size_t          vOffset;
  };

  std::vector<read_info> lRinfos;

  const dgs_memory_buffers & dMbufs;
};

size_t dgs_context::mStateGetSize() {
  dgs_io_write_dummy vIo(false);
  try {
    return mStateWriteData(vIo);
  } catch (const std::exception & vErr) {
    DGS_LOG_ERROR("%s: error getting state size: %s\n", __func__, vErr.what());
    return 0;
  }
}

size_t dgs_context::mStateGetData(uint8_t * pDst, size_t size) {
  dgs_io_write_host vIo(pDst, size);
  try {
    return mStateWriteData(vIo);
  } catch (const std::exception & vErr) {
    DGS_LOG_ERROR("%s: error saving state: %s\n", __func__, vErr.what());
    return 0;
  }
}

size_t dgs_context::mStateSetData(const uint8_t * pSrc, size_t size) {
  dgs_io_read_host vIo(pSrc, size);
  try {
    return mStateReadData(vIo);
  } catch (const std::exception & vErr) {
    DGS_LOG_ERROR("%s: error loading state: %s\n", __func__, vErr.what());
    vIo.mDiscard();
    return 0;
  }
}

static constexpr uint32_t cIoMagic = 0xaf143cd8;

size_t dgs_context::mStateSeqGetSize(dgs_seq_id pSeqId, dgs_state_seq_flags pFlags) {
  dgs_io_write_dummy vIo(pFlags & DGS_STATE_SEQ_FLAGS_ON_DEVICE);
  try {
    vIo.write(&cIoMagic, sizeof(cIoMagic));
    vIo.write(&pSeqId, sizeof(pSeqId));

    return mStateSeqWriteData(vIo, pSeqId, pFlags);
  } catch (const std::exception & vErr) {
    DGS_LOG_ERROR("%s: error getting state size: %s\n", __func__, vErr.what());
    return 0;
  }
}

size_t dgs_context::mStateSeqGetData(dgs_seq_id pSeqId, uint8_t * pDst, size_t size, dgs_state_seq_flags pFlags) {
  std::unique_ptr<dgs_io_write_i> vIo;
  if (pFlags & DGS_STATE_SEQ_FLAGS_ON_DEVICE) {
    vIo = std::make_unique<dgs_io_write_device>(pDst, size, dMemStorage[pSeqId]);
  } else {
    vIo = std::make_unique<dgs_io_write_host>(pDst, size);
  }

  try {
    vIo->write(&cIoMagic, sizeof(cIoMagic));
    vIo->write(&pSeqId, sizeof(pSeqId));

    return mStateSeqWriteData(*vIo, pSeqId, pFlags);
  } catch (const std::exception & vErr) {
    DGS_LOG_ERROR("%s: error saving state: %s\n", __func__, vErr.what());
    return 0;
  }
}

size_t dgs_context::mStateSeqSetData(dgs_seq_id pSeqId, const uint8_t * pSrc, size_t size, dgs_state_seq_flags pFlags) {
  std::unique_ptr<dgs_io_read_i> vIo;
  if (pFlags & DGS_STATE_SEQ_FLAGS_ON_DEVICE) {
    // create a temporary io to read the magic and the src seq_id
    vIo = std::make_unique<dgs_io_read_host>(pSrc, size);

    uint32_t vMagicRead;
    vIo->read(&vMagicRead, sizeof(vMagicRead));
    if (cIoMagic != vMagicRead) {
      throw std::runtime_error("wrong sequence state magic");
    }

    dgs_seq_id vSeqIdRead;
    vIo->read(&vSeqIdRead, sizeof(vSeqIdRead));

    GGML_ASSERT(dMemStorage.find(vSeqIdRead) != dMemStorage.end());

    vIo = std::make_unique<dgs_io_read_device>(pSrc, size, dMemStorage[vSeqIdRead]);
  } else {
    vIo = std::make_unique<dgs_io_read_host>(pSrc, size);
  }

  try {
    uint32_t vMagicRead;
    vIo->read(&vMagicRead, sizeof(vMagicRead));
    if (cIoMagic != vMagicRead) {
      throw std::runtime_error("wrong sequence state magic");
    }

    dgs_seq_id vSeqIdRead;
    vIo->read(&vSeqIdRead, sizeof(vSeqIdRead));

    return mStateSeqReadData(*vIo, pSeqId, pFlags);
  } catch (const std::exception & vErr) {
    DGS_LOG_ERROR("%s: error loading state: %s\n", __func__, vErr.what());
    vIo->mDiscard();
    return 0;
  }
}

bool dgs_context::mStateLoadFile(const char * pFilepath,
                                 dgs_token *  pTokensOut,
                                 size_t       pNTokenCapacity,
                                 size_t *     pNTokenCountOut) {
  dgs_file vFile(pFilepath, "rb");

  // sanity checks
  {
    const uint32_t cMagic   = vFile.mReadU32();
    const uint32_t cVersion = vFile.mReadU32();

    if (cMagic != DGS_SESSION_MAGIC || cVersion != DGS_SESSION_VERSION) {
      DGS_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, cMagic, cVersion);
      return false;
    }
  }

  // load the prompt
  {
    const uint32_t cNTokenCount = vFile.mReadU32();

    if (cNTokenCount > pNTokenCapacity) {
      DGS_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, cNTokenCount,
                    pNTokenCapacity);
      return false;
    }

    vFile.mReadRaw(pTokensOut, sizeof(dgs_token) * cNTokenCount);
    *pNTokenCountOut = cNTokenCount;
  }

  // restore the context state
  {
    const size_t cNStateSizeCur = vFile.size() - vFile.mTell();

    dgs_io_read_file vIo(&vFile);
    const size_t     cNRead = mStateReadData(vIo);

    if (cNRead != cNStateSizeCur) {
      DGS_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, cNStateSizeCur,
                    cNRead);
      return false;
    }
  }

  return true;
}

bool dgs_context::mStateSaveFile(const char * pFilepath, const dgs_token * pTokens, size_t pNTokenCount) {
  dgs_file vFile(pFilepath, "wb");

  vFile.mWriteU32(DGS_SESSION_MAGIC);
  vFile.mWriteU32(DGS_SESSION_VERSION);

  // save the prompt
  vFile.mWriteU32((uint32_t) pNTokenCount);
  vFile.mWriteRaw(pTokens, sizeof(dgs_token) * pNTokenCount);

  // save the context state using stream saving
  dgs_io_write_file vIo(&vFile);
  mStateWriteData(vIo);

  return true;
}

size_t dgs_context::mStateSeqLoadFile(dgs_seq_id   pSeqId,
                                      const char * pFilepath,
                                      dgs_token *  pTokensOut,
                                      size_t       pNTokenCapacity,
                                      size_t *     pNTokenCountOut) {
  dgs_file vFile(pFilepath, "rb");

  // version checks
  {
    const uint32_t cMagic   = vFile.mReadU32();
    const uint32_t cVersion = vFile.mReadU32();

    if (cMagic != DGS_STATE_SEQ_MAGIC || cVersion != DGS_STATE_SEQ_VERSION) {
      DGS_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, cMagic, cVersion);
      return 0;
    }
  }

  // load the prompt
  {
    const uint32_t cNTokenCount = vFile.mReadU32();

    if (pTokensOut == nullptr) {
      const size_t cNTokenMax = (vFile.size() - vFile.mTell()) / sizeof(dgs_token);
      if (cNTokenCount > cNTokenMax) {
        DGS_LOG_ERROR("%s: token count in sequence state file exceeds the file size! %u > %zu\n", __func__,
                      cNTokenCount, cNTokenMax);
        return 0;
      }

      *pNTokenCountOut = cNTokenCount;
      return vFile.mTell();
    }

    if (cNTokenCount > pNTokenCapacity) {
      DGS_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, cNTokenCount,
                    pNTokenCapacity);
      return 0;
    }

    vFile.mReadRaw(pTokensOut, sizeof(dgs_token) * cNTokenCount);
    *pNTokenCountOut = cNTokenCount;
  }

  // restore the context state
  {
    const size_t     cStateSize = vFile.size() - vFile.mTell();
    dgs_io_read_file vIo(&vFile);
    const size_t     cNread = mStateSeqReadData(vIo, pSeqId, 0);
    if (!cNread) {
      DGS_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
      return 0;
    }
    GGML_ASSERT(cNread <= cStateSize);
    GGML_ASSERT(cNread + sizeof(uint32_t) * 3 + sizeof(dgs_token) * *pNTokenCountOut == vFile.mTell());
  }

  return vFile.mTell();
}

size_t dgs_context::mStateSeqSaveFile(dgs_seq_id        pSeqId,
                                      const char *      pFilepath,
                                      const dgs_token * pTokens,
                                      size_t            pNTokenCount) {
  dgs_file vFile(pFilepath, "wb");

  vFile.mWriteU32(DGS_STATE_SEQ_MAGIC);
  vFile.mWriteU32(DGS_STATE_SEQ_VERSION);

  // save the prompt
  vFile.mWriteU32((uint32_t) pNTokenCount);
  vFile.mWriteRaw(pTokens, sizeof(dgs_token) * pNTokenCount);

  // save the context state using stream saving
  dgs_io_write_file vIo(&vFile);
  mStateSeqWriteData(vIo, pSeqId, 0);

  const size_t cRes = vFile.mTell();
  GGML_ASSERT(cRes == sizeof(uint32_t) * 3 + sizeof(dgs_token) * pNTokenCount + vIo.mNBytes());

  return cRes;
}

size_t dgs_context::mStateWriteData(dgs_io_write_i & pIo) {
  DGS_LOG_DEBUG("%s: writing state\n", __func__);

  // write model info
  {
    DGS_LOG_DEBUG("%s: - writing model info\n", __func__);

    const std::string cArchStr = fLlmArchName(vModel.vArch);
    pIo.mWriteString(cArchStr);
    // TODO: add more model-specific info which should prevent loading the session file if not identical
  }

  if (vMemory != nullptr) {
    DGS_LOG_DEBUG("%s: - writing memory module\n", __func__);
    vMemory->mStateWrite(pIo);
  }

  return pIo.mNBytes();
}

size_t dgs_context::mStateReadData(dgs_io_read_i & pIo) {
  DGS_LOG_DEBUG("%s: reading state\n", __func__);

  // read model info
  {
    DGS_LOG_DEBUG("%s: - reading model info\n", __func__);

    const std::string cCurArchStr = fLlmArchName(vModel.vArch);

    std::string vArchStr;
    pIo.mReadString(vArchStr);
    if (cCurArchStr != vArchStr) {
      throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", vArchStr.c_str(), cCurArchStr.c_str()));
    }
    // TODO: add more info which needs to be identical but which is not verified otherwise
  }

  if (vMemory) {
    DGS_LOG_DEBUG("%s: - reading memory module\n", __func__);

    vMemory->mStateRead(pIo);
  }

  return pIo.mNBytes();
}

size_t dgs_context::mStateSeqWriteData(dgs_io_write_i & pIo, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags) {
  if (vMemory) {
    vMemory->mStateWrite(pIo, pSeqId, pFlags);
  }

  return pIo.mNBytes();
}

size_t dgs_context::mStateSeqReadData(dgs_io_read_i & pIo, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags) {
  if (vMemory) {
    vMemory->mStateRead(pIo, pSeqId, pFlags);
  }

  return pIo.mNBytes();
}

//
// perf
//

dgs_perf_context_data dgs_context::mPerfGetData() const {
  dgs_perf_context_data data = {};

  data.vTStartMs = 1e-3 * vTStartUs;
  data.vTLoadMs  = 1e-3 * vTLoadUs;
  data.vTPEvalMs = 1e-3 * vTPEvalUs;
  data.vTEvalMs  = 1e-3 * vTEvalUs;
  data.vNPEval   = std::max(1, vNPEval);
  data.vNEval    = std::max(1, vNEval);
  data.vNReused  = std::max(0, vNReused);

  data.vNReallocsSame = vNReallocsSame;
  data.vNReallocsDiff = vNReallocsDiff;

  return data;
}

void dgs_context::mPerfReset() {
  vTStartUs = ggml_time_us();
  vTEvalUs = vNEval = 0;
  vTPEvalUs = vNPEval = 0;
  vNReused            = 0;

  // note: the reallocation counters are not reset - they track the lifetime totals for this context
}

dgs_memory_breakdown dgs_context::mMemoryBreakdown() const {
  std::map<ggml_backend_buffer_type_t, dgs_memory_breakdown_data> dRet;
  for (const auto & [buft, size] : vModel.mMemoryBreakdown()) {
    dRet[buft].vModel += size;
  }
  if (vMemory) {
    for (const auto & [buft, size] : vMemory->mMemoryBreakdown()) {
      dRet[buft].vContext += size;
    }
  }

  if (vModel.hparams.vNoAlloc) {
    for (size_t vI = 0; vI < lBackends.size(); ++vI) {
      ggml_backend_t             vBackend = lBackends[vI].get();
      ggml_backend_buffer_type_t vBuft    = ggml_backend_sched_get_buffer_type(vSched.get(), vBackend);
      dRet[vBuft].vCompute += lBackendBufExpSize[vI];
    }
  } else {
    for (const auto & vBackendPtr : lBackends) {
      ggml_backend_t             vBackend = vBackendPtr.get();
      ggml_backend_buffer_type_t vBuft    = ggml_backend_sched_get_buffer_type(vSched.get(), vBackend);
      dRet[vBuft].vCompute += ggml_backend_sched_get_buffer_size(vSched.get(), vBackend);
    }
  }
  return dRet;
}

//
// training
//

//
// interface implementation
//

dgs_context_params fDgsContextDefaultParams() {
  dgs_context_params vResult = {
    /*.n_ctx                       =*/512,
    /*.n_batch                     =*/2048,
    /*.n_ubatch                    =*/512,
    /*.n_seq_max                   =*/1,
    /*.n_rs_seq                    =*/0,
    /*.n_outputs_max               =*/0,
    /*.n_outputs_max_per_seq       =*/1,
    /*.n_threads                   =*/GGML_DEFAULT_N_THREADS,  // TODO: better default
    /*.n_threads_batch             =*/GGML_DEFAULT_N_THREADS,
    /*.ctx_type                    =*/cDgsContextTypeDefault,
    /*.rope_scaling_type           =*/cDgsRopeScalingTypeUnspecified,
    /*.pooling_type                =*/cDgsPoolingTypeUnspecified,
    /*.attention_type              =*/cDgsAttentionTypeUnspecified,
    /*.flash_attn_type             =*/cDgsFlashAttnTypeAuto,
    /*.rope_freq_base              =*/0.0f,
    /*.rope_freq_scale             =*/0.0f,
    /*.yarn_ext_factor             =*/-1.0f,
    /*.yarn_attn_factor            =*/-1.0f,
    /*.yarn_beta_fast              =*/-1.0f,
    /*.yarn_beta_slow              =*/-1.0f,
    /*.yarn_orig_ctx               =*/0,
    /*.cb_eval                     =*/nullptr,
    /*.cb_eval_user_data           =*/nullptr,
    /*.type_k                      =*/GGML_TYPE_F16,
    /*.type_v                      =*/GGML_TYPE_F16,
    /*.abort_callback              =*/nullptr,
    /*.abort_callback_data         =*/nullptr,
    /*.embeddings                  =*/false,
    /*.offload_kqv                 =*/true,
    /*.no_perf                     =*/true,
    /*.op_offload                  =*/true,
    /*.swa_full                    =*/true,
    /*.kv_unified                  =*/false,
    /*.sampler                     =*/nullptr,
    /*.n_sampler                   =*/0,
    /*.ctx_other                   =*/nullptr,
  };

  return vResult;
}

dgs_context * fDgsInitFromModel(dgs_model * pModel, dgs_context_params pParams) {
  if (!pModel) {
    DGS_LOG_ERROR("%s: model cannot be NULL\n", __func__);
    return nullptr;
  }

  if (pParams.vNBatch == 0 && pParams.vNUbatch == 0) {
    DGS_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
    return nullptr;
  }

  if (pParams.vNCtx == 0 && pModel->hparams.n_ctx_train == 0) {
    DGS_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
    return nullptr;
  }

  if (pParams.vFlashAttnType != cDgsFlashAttnTypeDisabled && pModel->vArch == cLlmArchGrok) {
    DGS_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
    pParams.vFlashAttnType = cDgsFlashAttnTypeDisabled;
  }

  if (pModel->mSplitMode() == cDgsSplitModeTensor) {
    if (pParams.vFlashAttnType == cDgsFlashAttnTypeAuto) {
      DGS_LOG_INFO("%s: enabling flash_attn since it is required for SPLIT_MODE_TENSOR\n", __func__);
      pParams.vFlashAttnType = cDgsFlashAttnTypeEnabled;
    }
    if (pParams.vFlashAttnType != cDgsFlashAttnTypeEnabled) {
      DGS_LOG_ERROR("%s: SPLIT_MODE_TENSOR requires flash_attn to be enabled\n", __func__);
      return nullptr;
    }
    if (pModel->vGetSplitStateUd.vNDevices == 1) {
      DGS_LOG_WARN("%s: SPLIT_MODE_TENSOR being used for a single device is not recommended\n", __func__);
    }
  }

  if ((pModel->hparams.mIsMla() || pModel->vArch == cLlmArchDeepseek4) && pParams.vTypeK != pParams.vTypeV) {
    DGS_LOG_ERROR("%s: model does not support different K (%s) and V (%s) cache types\n", __func__,
                  ggml_type_name(pParams.vTypeK), ggml_type_name(pParams.vTypeV));
    return nullptr;
  }

  if (ggml_is_quantized(pParams.vTypeV) && pParams.vFlashAttnType != cDgsFlashAttnTypeEnabled) {
    if (pParams.vFlashAttnType == cDgsFlashAttnTypeAuto) {
      DGS_LOG_INFO("%s: enabling flash_attn since it is required for quantized V cache\n", __func__);
      pParams.vFlashAttnType = cDgsFlashAttnTypeEnabled;
    }
    if (pParams.vFlashAttnType == cDgsFlashAttnTypeDisabled) {
      DGS_LOG_ERROR("%s: quantized V cache requires flash_attn to be enabled\n", __func__);
      return nullptr;
    }
  }

  if (pParams.vFlashAttnType != cDgsFlashAttnTypeDisabled && ggml_is_quantized(pParams.vTypeK)) {
    const uint32_t cBlckSize = ggml_blck_size(pParams.vTypeK);
    for (uint32_t vIl = 0; vIl < pModel->hparams.n_layer(); ++vIl) {
      if (pModel->hparams.n_embd_head_k(vIl) % cBlckSize != 0) {
        DGS_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n", __func__,
                      ggml_type_name(pParams.vTypeK), cBlckSize, pModel->hparams.n_embd_head_k(vIl));
        return nullptr;
      }
    }
  }

  if (pParams.vFlashAttnType != cDgsFlashAttnTypeDisabled && ggml_is_quantized(pParams.vTypeV)) {
    const uint32_t cBlckSize = ggml_blck_size(pParams.vTypeV);
    for (uint32_t vIl = 0; vIl < pModel->hparams.n_layer(); ++vIl) {
      if (pModel->hparams.n_embd_head_v(vIl) % cBlckSize != 0) {
        DGS_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_v=%u\n", __func__,
                      ggml_type_name(pParams.vTypeV), cBlckSize, pModel->hparams.n_embd_head_v(vIl));
        return nullptr;
      }
    }
  }

  if (pParams.vPoolingType != cDgsPoolingTypeUnspecified && pParams.vPoolingType != pModel->hparams.vPoolingType) {
    //user-specified pooling-type is different from the model default
    DGS_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                 pModel->hparams.vPoolingType, pParams.vPoolingType);
  }

  // router_layer >= 0 means n_layer_nextn is repurposed for a router layer, not real MTP
  if (pParams.vCtxType == cDgsContextTypeMtp &&
      (pModel->hparams.n_layer_nextn == 0 || pModel->hparams.vRouterLayer >= 0)) {
    DGS_LOG_WARN("%s: context type MTP requested but model doesn't contain MTP layers\n", __func__);
    return nullptr;
  }

  try {
    auto *       vCtx     = new dgs_context(*pModel, pParams);
    const auto & vCparams = vCtx->mGetCparams();

    if (vCparams.vRopeScalingType == cDgsRopeScalingTypeYarn &&
        vCparams.vRopeFreqScale != pModel->hparams.vRopeFreqScaleTrain) {
      DGS_LOG_INFO("%s: custom YaRN scaling detected, re-adjusting n_ctx_train(%u)...\n", __func__,
                   pModel->hparams.n_ctx_train);
      pModel->hparams.n_ctx_train = vCparams.vNCtxOrigYarn / vCparams.vRopeFreqScale;
      DGS_LOG_INFO("%s: n_ctx_train adjusted to %u\n", __func__, pModel->hparams.n_ctx_train);
    }

    return vCtx;
  } catch (const std::exception & vErr) {
    DGS_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, vErr.what());
  }

  return nullptr;
}

void fDgsFree(dgs_context * pCtx) {
  delete pCtx;
}

uint32_t fDgsNCtx(const dgs_context * pCtx) {
  return pCtx->mNCtx();
}

uint32_t fDgsNCtxSeq(const dgs_context * pCtx) {
  return pCtx->mNCtxSeq();
}

uint32_t fDgsNBatch(const dgs_context * pCtx) {
  return pCtx->mNBatch();
}

uint32_t fDgsNUbatch(const dgs_context * pCtx) {
  return pCtx->mNUbatch();
}

uint32_t fDgsNSeqMax(const dgs_context * pCtx) {
  return pCtx->mNSeqMax();
}

uint32_t fDgsNRsSeq(const dgs_context * pCtx) {
  return pCtx->mGetCparams().vNRsSeq;
}

const dgs_model * fDgsGetModel(const dgs_context * pCtx) {
  return &pCtx->mGetModel();
}

enum fDgsPoolingType fDgsPoolingType(const dgs_context * pCtx) {
  return pCtx->mPoolingType();
}

void fDgsAttachThreadpool(dgs_context * pCtx, ggml_threadpool_t pThreadpool, ggml_threadpool_t pThreadpoolBatch) {
  pCtx->mAttachThreadpool(pThreadpool, pThreadpoolBatch);
}

void fDgsDetachThreadpool(dgs_context * pCtx) {
  pCtx->mDetachThreadpool();
}

void fDgsSetNThreads(dgs_context * pCtx, int32_t pNThreads, int32_t pNThreadsBatch) {
  pCtx->mSetNThreads(pNThreads, pNThreadsBatch);
}

int32_t fDgsNThreads(dgs_context * pCtx) {
  return pCtx->mNThreads();
}

int32_t fDgsNThreadsBatch(dgs_context * pCtx) {
  return pCtx->mNThreadsBatch();
}

void fDgsSetAbortCallback(dgs_context * pCtx, bool (*pAbortCallback)(void * data), void * pAbortCallbackData) {
  pCtx->mSetAbortCallback(pAbortCallback, pAbortCallbackData);
}

void fDgsSetEmbeddings(dgs_context * pCtx, bool pEmbeddings) {
  pCtx->mSetEmbeddings(pEmbeddings);
}

void fDgsSetCausalAttn(dgs_context * pCtx, bool pCausalAttn) {
  pCtx->mSetCausalAttn(pCausalAttn);
}

bool fDgsGetCausalAttn(const dgs_context * pCtx) {
  return pCtx->mGetCausalAttn();
}

void fDgsSynchronize(dgs_context * pCtx) {
  pCtx->mSynchronize();
}

float * fDgsGetLogits(dgs_context * pCtx) {
  pCtx->mSynchronize();

  return pCtx->mGetLogits();
}

float * fDgsGetLogitsIth(dgs_context * pCtx, int32_t pI) {
  pCtx->mSynchronize();

  float * vRes = nullptr;

  vRes = pCtx->mGetSampledLogitsIth(pI);

  if (!vRes) {
    vRes = pCtx->mGetLogitsIth(pI);
  }

  return vRes;
}

float * fDgsGetEmbeddings(dgs_context * pCtx) {
  pCtx->mSynchronize();

  return pCtx->mGetEmbeddings();
}

float * fDgsGetEmbeddingsIth(dgs_context * pCtx, int32_t pI) {
  pCtx->mSynchronize();

  return pCtx->mGetEmbeddingsIth(pI);
}

float * fDgsGetEmbeddingsSeq(dgs_context * pCtx, dgs_seq_id pSeqId) {
  pCtx->mSynchronize();

  return pCtx->mGetEmbeddingsSeq(pSeqId);
}

void fDgsSetEmbeddingsNextn(dgs_context * pCtx, bool value, bool pMasked) {
  pCtx->mSetEmbeddingsNextn(value, pMasked);
}

void fDgsSetEmbeddingsLayerInp(dgs_context * pCtx, uint32_t pLid, bool value) {
  pCtx->mSetEmbeddingsLayerInp(pLid, value);
}

void fDgsSetNextnLayerOffset(dgs_context * pCtx, int32_t pOffset) {
  pCtx->mSetNextnLayerOffset(pOffset);
}

dgs_memory_t fDgsGetMemory(const struct dgs_context * pCtx) {
  if (!pCtx) {
    return nullptr;
  }

  return pCtx->mGetMemory();
}

float * fDgsGetEmbeddingsNextn(dgs_context * pCtx) {
  pCtx->mSynchronize();

  return pCtx->mGetEmbeddingsNextn();
}

float * fDgsGetEmbeddingsNextnIth(dgs_context * pCtx, int32_t pI) {
  pCtx->mSynchronize();

  return pCtx->mGetEmbeddingsNextnIth(pI);
}

float * fDgsGetEmbeddingsLayerInp(dgs_context * pCtx, uint32_t pLid) {
  pCtx->mSynchronize();

  return pCtx->mGetEmbeddingsLayerInp(pLid);
}

bool fDgsSetSampler(dgs_context * pCtx, dgs_seq_id pSeqId, dgs_sampler * pSmpl) {
  return pCtx->mSetSampler(pSeqId, pSmpl);
}

dgs_token fDgsGetSampledTokenIth(dgs_context * pCtx, int32_t pI) {
  pCtx->mSynchronize();

  return pCtx->mGetSampledTokenIth(pI);
}

float * fDgsGetSampledProbsIth(dgs_context * pCtx, int32_t pI) {
  pCtx->mSynchronize();

  return pCtx->mGetSampledProbsIth(pI);
}

float * fDgsGetSampledLogitsIth(dgs_context * pCtx, int32_t pI) {
  pCtx->mSynchronize();

  return pCtx->mGetSampledLogitsIth(pI);
}

dgs_token * fDgsGetSampledCandidatesIth(dgs_context * pCtx, int32_t pI) {
  pCtx->mSynchronize();

  return const_cast<dgs_token *>(pCtx->mGetSampledCandidatesIth(pI));
}

uint32_t fDgsGetSampledCandidatesCountIth(dgs_context * pCtx, int32_t pI) {
  pCtx->mSynchronize();

  return static_cast<uint32_t>(pCtx->mGetSampledCandidatesCount(pI));
}

uint32_t fDgsGetSampledLogitsCountIth(dgs_context * pCtx, int32_t pI) {
  pCtx->mSynchronize();

  return static_cast<uint32_t>(pCtx->mGetSampledLogitsCount(pI));
}

uint32_t fDgsGetSampledProbsCountIth(dgs_context * pCtx, int32_t pI) {
  pCtx->mSynchronize();

  return static_cast<uint32_t>(pCtx->mGetSampledProbsCount(pI));
}

struct ggml_cgraph * fDgsGraphReserve(struct dgs_context * pCtx,
                                      uint32_t             pNTokens,
                                      uint32_t             pNSeqs,
                                      uint32_t             pNOutputs) {
  auto                   vMemory = pCtx->mGetMemory();
  dgs_memory_context_ptr vMctx;
  if (vMemory) {
    vMctx = vMemory->mInitFull();
  }
  return pCtx->mGraphReserve(pNTokens, pNSeqs, pNOutputs, vMctx.get());
}

// dgs adapter API

int32_t fDgsSetAdaptersLora(dgs_context * pCtx, dgs_adapter_lora ** pAdapters, size_t pNAdapters, float * pScales) {
  if (pAdapters == nullptr || pScales == nullptr) {
    GGML_ASSERT(pNAdapters == 0 && "invalid dgs_set_adapters_lora call");
  }

  pCtx->mSetAdaptersLora(pAdapters, pNAdapters, pScales);

  return 0;
}

int32_t fDgsSetAdapterCvec(dgs_context * pCtx,
                           const float * data,
                           size_t        pLen,
                           int32_t       pNEmbd,
                           int32_t       pIlStart,
                           int32_t       pIlEnd) {
  bool vRes = pCtx->mSetAdapterCvec(data, pLen, pNEmbd, pIlStart, pIlEnd);

  return vRes ? 0 : -1;
}

//
// memory
//

void fDgsMemoryClear(dgs_memory_t pMem, bool data) {
  if (!pMem) {
    return;
  }

  pMem->clear(data);
}

bool fDgsMemorySeqRm(dgs_memory_t pMem, dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1) {
  if (!pMem) {
    return true;
  }

  return pMem->mSeqRm(pSeqId, pP0, pP1);
}

void fDgsMemorySeqCp(dgs_memory_t pMem, dgs_seq_id pSeqIdSrc, dgs_seq_id pSeqIdDst, dgs_pos pP0, dgs_pos pP1) {
  if (!pMem) {
    return;
  }

  pMem->mSeqCp(pSeqIdSrc, pSeqIdDst, pP0, pP1);
}

void fDgsMemorySeqKeep(dgs_memory_t pMem, dgs_seq_id pSeqId) {
  if (!pMem) {
    return;
  }

  pMem->mSeqKeep(pSeqId);
}

void fDgsMemorySeqAdd(dgs_memory_t pMem, dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1, dgs_pos pDelta) {
  if (!pMem) {
    return;
  }

  pMem->mSeqAdd(pSeqId, pP0, pP1, pDelta);
}

void fDgsMemorySeqDiv(dgs_memory_t pMem, dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1, int pD) {
  if (!pMem) {
    return;
  }

  pMem->mSeqDiv(pSeqId, pP0, pP1, pD);
}

dgs_pos fDgsMemorySeqPosMin(dgs_memory_t pMem, dgs_seq_id pSeqId) {
  if (!pMem) {
    return -1;
  }

  return pMem->mSeqPosMin(pSeqId);
}

dgs_pos fDgsMemorySeqPosMax(dgs_memory_t pMem, dgs_seq_id pSeqId) {
  if (!pMem) {
    return -1;
  }

  return pMem->mSeqPosMax(pSeqId);
}

bool fDgsMemoryCanShift(dgs_memory_t pMem) {
  if (!pMem) {
    return false;
  }

  return pMem->mGetCanShift();
}

// dgs state API

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t fDgsStateGetSize(dgs_context * pCtx) {
  return pCtx->mStateGetSize();
}

size_t fDgsStateGetData(dgs_context * pCtx, uint8_t * pDst, size_t size) {
  pCtx->mSynchronize();

  return pCtx->mStateGetData(pDst, size);
}

// Sets the state reading from the specified source address
size_t fDgsStateSetData(dgs_context * pCtx, const uint8_t * pSrc, size_t size) {
  pCtx->mSynchronize();

  return pCtx->mStateSetData(pSrc, size);
}

bool fDgsStateLoadFile(dgs_context * pCtx,
                       const char *  pPathSession,
                       dgs_token *   pTokensOut,
                       size_t        pNTokenCapacity,
                       size_t *      pNTokenCountOut) {
  pCtx->mSynchronize();

  try {
    return pCtx->mStateLoadFile(pPathSession, pTokensOut, pNTokenCapacity, pNTokenCountOut);
  } catch (const std::exception & vErr) {
    DGS_LOG_ERROR("%s: error loading session file: %s\n", __func__, vErr.what());
    return false;
  }
}

bool fDgsStateSaveFile(dgs_context * pCtx, const char * pPathSession, const dgs_token * pTokens, size_t pNTokenCount) {
  pCtx->mSynchronize();

  try {
    return pCtx->mStateSaveFile(pPathSession, pTokens, pNTokenCount);
  } catch (const std::exception & vErr) {
    DGS_LOG_ERROR("%s: error saving session file: %s\n", __func__, vErr.what());
    return false;
  }
}

size_t fDgsStateSeqGetSize(dgs_context * pCtx, dgs_seq_id pSeqId) {
  return fDgsStateSeqGetSizeExt(pCtx, pSeqId, 0);
}

size_t fDgsStateSeqGetData(dgs_context * pCtx, uint8_t * pDst, size_t size, dgs_seq_id pSeqId) {
  return fDgsStateSeqGetDataExt(pCtx, pDst, size, pSeqId, 0);
}

size_t fDgsStateSeqSetData(dgs_context * pCtx, const uint8_t * pSrc, size_t size, dgs_seq_id pSeqId) {
  return fDgsStateSeqSetDataExt(pCtx, pSrc, size, pSeqId, 0);
}

size_t fDgsStateSeqGetSizeExt(dgs_context * pCtx, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags) {
  return pCtx->mStateSeqGetSize(pSeqId, pFlags);
}

size_t fDgsStateSeqGetDataExt(dgs_context *       pCtx,
                              uint8_t *           pDst,
                              size_t              size,
                              dgs_seq_id          pSeqId,
                              dgs_state_seq_flags pFlags) {
  pCtx->mSynchronize();

  return pCtx->mStateSeqGetData(pSeqId, pDst, size, pFlags);
}

size_t fDgsStateSeqSetDataExt(dgs_context *       pCtx,
                              const uint8_t *     pSrc,
                              size_t              size,
                              dgs_seq_id          pSeqId,
                              dgs_state_seq_flags pFlags) {
  pCtx->mSynchronize();

  return pCtx->mStateSeqSetData(pSeqId, pSrc, size, pFlags);
}

size_t fDgsStateSeqSaveFile(dgs_context *     pCtx,
                            const char *      pFilepath,
                            dgs_seq_id        pSeqId,
                            const dgs_token * pTokens,
                            size_t            pNTokenCount) {
  pCtx->mSynchronize();

  try {
    return pCtx->mStateSeqSaveFile(pSeqId, pFilepath, pTokens, pNTokenCount);
  } catch (const std::exception & vErr) {
    DGS_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, vErr.what());
    return 0;
  }
}

size_t fDgsStateSeqLoadFile(dgs_context * pCtx,
                            const char *  pFilepath,
                            dgs_seq_id    pDestSeqId,
                            dgs_token *   pTokensOut,
                            size_t        pNTokenCapacity,
                            size_t *      pNTokenCountOut) {
  pCtx->mSynchronize();

  try {
    return pCtx->mStateSeqLoadFile(pDestSeqId, pFilepath, pTokensOut, pNTokenCapacity, pNTokenCountOut);
  } catch (const std::exception & vErr) {
    DGS_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, vErr.what());
    return 0;
  }
}

// compat: dgs_batch -> dgs_batch_ext -> encode/decode

int dgs_context::mEncode(const dgs_batch & pBatchInp) {
  dgs_batch_compat vCompat(this, pBatchInp, vModel.hparams.mNEmbdInpEnc());
  return mEncode(*vCompat.vBatchExt);
}

int dgs_context::mDecode(const dgs_batch & pBatchInp) {
  dgs_batch_compat vCompat(this, pBatchInp);
  return mDecode(*vCompat.vBatchExt);
}

///

int32_t fDgsEncode(dgs_context * pCtx, dgs_batch pBatch) {
  const int cRet = pCtx->mEncode(pBatch);
  if (cRet != 0) {
    DGS_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, cRet);
  }

  return cRet;
}

int32_t fDgsDecode(dgs_context * pCtx, dgs_batch pBatch) {
  const int cRet = pCtx->mDecode(pBatch);
  if (cRet != 0 && cRet != 1) {
    DGS_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, cRet);
  }

  return cRet;
}

//
// perf
//

dgs_perf_context_data fDgsPerfContext(const dgs_context * pCtx) {
  dgs_perf_context_data data = {};

  if (pCtx == nullptr) {
    return data;
  }

  data = pCtx->mPerfGetData();

  return data;
}

void fDgsPerfContextPrint(const dgs_context * pCtx) {
  const auto data = fDgsPerfContext(pCtx);

  const double cTEndMs = 1e-3 * ggml_time_us();

  DGS_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.vTLoadMs);
  DGS_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
               __func__, data.vTPEvalMs, data.vNPEval, data.vTPEvalMs / data.vNPEval,
               1e3 / data.vTPEvalMs * data.vNPEval);
  DGS_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
               __func__, data.vTEvalMs, data.vNEval, data.vTEvalMs / data.vNEval, 1e3 / data.vTEvalMs * data.vNEval);
  DGS_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (cTEndMs - data.vTStartMs),
               (data.vNPEval + data.vNEval));
  DGS_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.vNReused);
  DGS_LOG_INFO("%s:   graph reallocs = %10d    / %5d        (same size) / (changed size)\n", __func__,
               data.vNReallocsSame, data.vNReallocsDiff);
}

void fDgsPerfContextReset(dgs_context * pCtx) {
  pCtx->mPerfReset();
}

//
// training
//

int32_t fDgsProcess(dgs_context * pCtx, dgs_process_type type, dgs_batch_ext * pBatch) {
  switch (type) {
    case cDgsProcessTypeEncode:
      return pCtx->mEncode(*pBatch);
    case cDgsProcessTypeDecode:
      return pCtx->mDecode(*pBatch);
  }
  return -1;
}

//
// ext
//

dgs_memory_breakdown fDgsGetMemoryBreakdown(const struct dgs_context * pCtx) {
  return pCtx->mMemoryBreakdown();
}

dgs_context * fDgsGetCtxOther(struct dgs_context * pCtx) {
  return pCtx->mGetCparams().vCtxOther;
}
