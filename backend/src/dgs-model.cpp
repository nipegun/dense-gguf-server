#include "dgs-model.h"

#include "dgs-arch.h"
#include "dgs-cparams.h"
#include "dgs-ext.h"
#include "dgs-hparams.h"
#include "dgs-impl.h"
#include "dgs-kv-cache-dsa-iswa.h"
#include "dgs-kv-cache-dsa.h"
#include "dgs-kv-cache-dsv4.h"
#include "dgs-kv-cache-iswa.h"
#include "dgs-kv-cache-msa.h"
#include "dgs-kv-cache.h"
#include "dgs-memory-hybrid-idx.h"
#include "dgs-memory-hybrid-iswa.h"
#include "dgs-memory-hybrid.h"
#include "dgs-memory-recurrent.h"
#include "dgs-mmap.h"
#include "dgs-model-loader.h"
#include "dgs-model-policy.h"
#include "dgs.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "models/models.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static dgs_model * fDgsModelMapping(llm_arch pArch, const dgs_model_params & pParams) {
  switch (pArch) {
    case cLlmArchQwen35:
      return new dgs_model_qwen35(pParams);
    default:
      throw std::runtime_error(std::string("unsupported model architecture: '") + fLlmArchName(pArch) + "'");
  }
}

dgs_model * fDgsModelCreate(llm_arch pArch, const dgs_model_params & pParams) {
  dgs_model * vModel = fDgsModelMapping(pArch, pParams);

  if (vModel != nullptr) {
    vModel->vArch = pArch;
    if (pParams.vSplitMode == cDgsSplitModeTensor && !fLlmArchSupportsSmTensor(pArch)) {
      throw std::runtime_error(std::string("DGS_SPLIT_MODE_TENSOR not implemented for architecture '") +
                               fLlmArchName(pArch) + "'");
    }
  }

  return vModel;
}

static void fRequireDenseModel(const dgs_model_loader & pLoader) {
  const auto * cMetadata          = pLoader.vMetadata;
  const auto   vArchitecture      = pLoader.mGetArch();
  const bool   cSingleBranchDense = vArchitecture == cLlmArchHunyuanDense || vArchitecture == cLlmArchHunyuanVl;
  for (int64_t vIndex = 0; vIndex < gguf_get_n_kv(cMetadata); ++vIndex) {
    const std::string vKey = gguf_get_key(cMetadata, vIndex);
    if (vKey.find(".expert_count") == std::string::npos && vKey.find(".expert_used_count") == std::string::npos &&
        vKey.find(".expert_shared_count") == std::string::npos &&
        vKey.find(".expert_group_count") == std::string::npos) {
      continue;
    }
    const auto cType       = gguf_get_kv_type(cMetadata, vIndex);
    bool       vHasExperts = true;
    if (cType == GGUF_TYPE_UINT32) {
      const auto cCount = gguf_get_val_u32(cMetadata, vIndex);
      vHasExperts       = cCount > (cSingleBranchDense ? 1u : 0u);
    } else if (cType == GGUF_TYPE_ARRAY && gguf_get_arr_type(cMetadata, vIndex) == GGUF_TYPE_UINT32) {
      const auto * aCounts = static_cast<const uint32_t *>(gguf_get_arr_data(cMetadata, vIndex));
      vHasExperts          = false;
      for (size_t vLayer = 0; vLayer < gguf_get_arr_n(cMetadata, vIndex); ++vLayer) {
        vHasExperts |= aCounts[vLayer] > (cSingleBranchDense ? 1u : 0u);
      }
    }
    if (vHasExperts) {
      throw std::runtime_error("DGS supports dense models only: expert metadata '" + vKey + "' is not supported");
    }
  }
  for (const auto & vWeight : pLoader.dWeightsMap) {
    const auto & vName = vWeight.first;
    if (vName.find("_exps") != std::string::npos || vName.find("_shexp") != std::string::npos ||
        vName.find("ffn_gate_inp") != std::string::npos || vName.find("ffn_exp_probs") != std::string::npos) {
      throw std::runtime_error("DGS supports dense models only: expert tensor '" + vName + "' is not supported");
    }
  }
}

dgs_model * fDgsModelCreate(dgs_model_loader & pMl, const dgs_model_params & pParams) {
  fRequireDenseModel(pMl);
  fDgsRequireModelProfile(pMl);
  llm_arch vArch = pMl.mGetArch();
  if (vArch == cLlmArchUnknown) {
    throw std::runtime_error("unknown model architecture: '" + pMl.mGetArchName() + "'");
  }

  return fDgsModelCreate(vArch, pParams);
}

struct ggml_backend_meta_split_state fDgsMetaDeviceGetSplitState(const struct ggml_tensor * pTensor, void * pUserdata) {
  const dgs_meta_device_get_split_state_userdata * vUd = (const dgs_meta_device_get_split_state_userdata *) pUserdata;
  const dgs_hparams &                              vHparams    = vUd->vModel->hparams;
  const std::string                                cTensorName = pTensor->name;
  const bool                                       cIsDsv4 =
    vUd->vModel->vArch == cLlmArchDeepseek4 || (vUd->vModel->vArch == cLlmArchDflash && vHparams.vDsv4HcMult > 0);

  static const std::regex cPatternQWeight("blk\\.\\d*\\.attn_q.weight");
  static const std::regex cPatternKvWeight("blk\\.\\d*\\.attn_(k|v).weight");
  static const std::regex cPatternVExpsWeight("blk\\.\\d*\\.attn_v_exps.weight");  // K2 Horizon MoVA
  static const std::regex cPatternQkvWeight("blk\\.\\d*\\.attn_qkv.weight");
  static const std::regex cPatternQBias("blk\\.\\d*\\.attn_q\\.bias");
  static const std::regex cPatternKvBias("blk\\.\\d*\\.attn_(k|v)\\.bias");
  static const std::regex cPatternQkvBias("blk\\.\\d*\\.attn_qkv.bias");
  static const std::regex cPatternQkNorm("blk\\.\\d*\\.attn_(q|k)_norm\\.weight");
  static const std::regex cPatternKvCache("cache_(k|v)_l\\d*");
  static const std::regex cPatternIdxCache("cache_idx_(k|v)_l\\d*");
  static const std::regex cPatternDsv4State("dsv4_(csa|hca|lid)_state_(kv|score)_l\\d*");
  static const std::regex cPatternAttnSinks("blk\\.\\d*\\.attn_sinks.weight");
  static const std::regex cPatternAttnOutWeight("blk\\.\\d*\\.attn_output.weight");
  static const std::regex cPatternAttnOutBias("blk\\.\\d*\\.attn_output.bias");
  static const std::regex cPatternAttnOutAWeight("blk\\.\\d*\\.attn_output_a\\.weight");
  static const std::regex cPatternAttnOutBWeight("blk\\.\\d*\\.attn_output_b\\.weight");
  static const std::regex cPatternAttnQBWeight("blk\\.\\d*\\.attn_q_b\\.weight");
  static const std::regex cPatternAttnGateWeight("blk\\.\\d*\\.attn_gate.weight");

  static const std::regex cPatternSsmDt("blk\\.\\d*\\.ssm_dt.bias");
  static const std::regex cPatternSsmA("blk\\.\\d*\\.ssm_a");
  static const std::regex cPatternSsmAlpha("blk\\.\\d*\\.ssm_alpha.weight");
  static const std::regex cPatternSsmBeta("blk\\.\\d*\\.ssm_beta.weight");
  static const std::regex cPatternSsmBetaAlpha("blk\\.\\d*\\.ssm_ba.weight");
  static const std::regex cPatternRCache("cache_r_l\\d*");
  static const std::regex cPatternPleRCache("cache_ple_r_l\\d*");
  static const std::regex cPatternSCache("cache_s_l\\d*");
  static const std::regex cPatternSsmConv1d("blk\\.\\d*\\.ssm_conv1d.weight");
  static const std::regex cPatternSsmOutWeight("blk\\.\\d*\\.ssm_out.weight");

  static const std::regex cPatternFfnUpWeight("blk\\.\\d*\\.ffn_up(_exps)?.weight");
  static const std::regex cPatternFfnUpBias("blk\\.\\d*\\.ffn_up(_exps)?.bias");
  static const std::regex cPatternFfnGateWeight("blk\\.\\d*\\.ffn_gate(_exps)?.weight");
  static const std::regex cPatternFfnGateBias("blk\\.\\d*\\.ffn_gate(_exps)?.bias");
  static const std::regex cPatternFfnGateUpWeight("blk\\.\\d*\\.ffn_gate_up(_exps)?.weight");
  static const std::regex cPatternFfnDownWeight("blk\\.\\d*\\.ffn_down(_exps)?.weight");
  static const std::regex cPatternFfnDownBias("blk\\.\\d*\\.ffn_down.bias");
  static const std::regex cPatternFfnDownExpsBias("blk\\.\\d*\\.ffn_down_exps.bias");
  static const std::regex cPatternFfnUpShexpWeight("blk\\.\\d*\\.ffn_up_shexp.weight");
  static const std::regex cPatternFfnGateShexpWeight("blk\\.\\d*\\.ffn_gate_shexp.weight");
  static const std::regex cPatternFfnDownShexpWeight("blk\\.\\d*\\.ffn_down_shexp.weight");

  static const std::regex cPatternOutputWeight("output\\.weight");
  static const std::regex cPatternOutputBias("output\\.bias");

  struct tensor_config {
    ggml_backend_meta_split_axis vAxis;

    const ggml_tensor * vTensorAxis0;

    uint32_t vIl;
    size_t   vRotation;  // when assigning tensor slices, rotate how the rounding is done for more even allocation
  };

  auto vGetTensorConfigImpl = [&](const ggml_backend_meta_split_axis pAxis, const std::string & pSuffix = "",
                                  const std::string & pSuffixFallback = "") -> tensor_config {
    // the layers in a tensor can be inhomogeneous, if the pattern is cleanly divided by the number of GPUs there can be aliasing effects,
    //     count only the same type of previous layers to avoid this
    auto vGetIlEff = [&](const size_t pIl) {
      size_t     vRet      = 0;
      const bool cIlIsRecr = vHparams.mIsRecr(pIl);
      const bool cIlIsSwa  = vHparams.mIsSwa(pIl);
      for (size_t vIlPrev = 0; vIlPrev < pIl; vIlPrev++) {
        vRet += vHparams.mIsRecr(vIlPrev) == cIlIsRecr && vHparams.mIsSwa(vIlPrev) == cIlIsSwa;
      }
      return vRet;
    };

    uint32_t    vIl;
    std::string vPrefix;
    size_t      vRotation;
    if (cTensorName.substr(0, 4) == "blk.") {
      const size_t cLengthPrefix = cTensorName.find('.', 4);
      GGML_ASSERT(cLengthPrefix != std::string::npos);
      vPrefix   = cTensorName.substr(0, cLengthPrefix + 1);
      vIl       = std::stoull(cTensorName.substr(4, cLengthPrefix));
      vRotation = vGetIlEff(vIl) % vUd->vNDevices;
    } else if (cTensorName.substr(0, 6) == "cache_") {
      const size_t cLayerIndexStart = cTensorName.find("_l", 6);
      GGML_ASSERT(cLayerIndexStart != std::string::npos);
      vIl       = std::stoull(cTensorName.substr(cLayerIndexStart + 2));
      vPrefix   = "blk." + std::to_string(vIl) + ".";
      vRotation = vGetIlEff(vIl) % vUd->vNDevices;
    } else {
      vIl       = 0;
      vRotation = vHparams.n_layer() % vUd->vNDevices;
    }
    const ggml_tensor * vTensorAxis0 = pSuffix.empty() ? pTensor : vUd->vModel->mGetTensor((vPrefix + pSuffix).c_str());
    if (vTensorAxis0 == nullptr) {
      GGML_ASSERT(!pSuffixFallback.empty());
      vTensorAxis0 = vUd->vModel->mGetTensor((vPrefix + pSuffixFallback).c_str());
    }
    GGML_ASSERT(vTensorAxis0 != nullptr);
    return { pAxis, vTensorAxis0, vIl, vRotation };
  };

  auto vGetTensorConfig = [&]() -> tensor_config {
    if (vUd->vModel->vArch == cLlmArchHrmText) {
      // aliased cache slots cannot satisfy the meta-split invariants, so replicate all tensors
      return { GGML_BACKEND_SPLIT_AXIS_MIRRORED, pTensor, 0, 0 };
    }
    if (cIsDsv4) {
      if (std::regex_match(cTensorName, cPatternKvCache) || std::regex_match(cTensorName, cPatternDsv4State)) {
        return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
      }
      if (std::regex_match(cTensorName, cPatternAttnSinks)) {
        return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output_a.weight");
      }
      if (std::regex_match(cTensorName, cPatternAttnQBWeight)) {
        return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output_a.weight");
      }
      if (std::regex_match(cTensorName, cPatternAttnOutAWeight)) {
        return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_2);
      }
      if (std::regex_match(cTensorName, cPatternAttnOutBWeight)) {
        return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_0);
      }
      if (std::regex_match(cTensorName, cPatternFfnUpShexpWeight) ||
          std::regex_match(cTensorName, cPatternFfnGateShexpWeight)) {
        return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down_shexp.weight");
      }
      if (std::regex_match(cTensorName, cPatternFfnDownShexpWeight)) {
        return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down_shexp.weight");
      }
    }

    // the qsa indexer has one key head and its projections are mirrored, so its cache cannot be split
    if (std::regex_match(cTensorName, cPatternIdxCache)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    }

    // the PLE table is model-level and its conv is mirrored, so every device runs the whole conv and needs the whole history
    if (std::regex_match(cTensorName, cPatternPleRCache)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    }

    // standard attention
    if (std::regex_match(cTensorName, cPatternQWeight) || std::regex_match(cTensorName, cPatternKvWeight)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
    }
    // routed value experts {n_embd, n_embd_v_gqa, n_expert} produce V, so they split like attn_v
    if (std::regex_match(cTensorName, cPatternVExpsWeight)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight");
    }
    if (std::regex_match(cTensorName, cPatternQBias) || std::regex_match(cTensorName, cPatternKvBias)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight", "ssm_out.weight");
    }
    if (std::regex_match(cTensorName, cPatternQkvWeight)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
    }
    if (std::regex_match(cTensorName, cPatternQkvBias)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight", "ssm_out.weight");
    }
    if (std::regex_match(cTensorName, cPatternQkNorm)) {
      return vGetTensorConfigImpl(pTensor->ne[1] == 1 ? GGML_BACKEND_SPLIT_AXIS_MIRRORED : GGML_BACKEND_SPLIT_AXIS_1,
                                  "attn_output.weight");
    }
    if (std::regex_match(cTensorName, cPatternKvCache) || std::regex_match(cTensorName, cPatternAttnSinks)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_0, "attn_output.weight");
    }
    if (std::regex_match(cTensorName, cPatternAttnOutWeight)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_0);
    }
    if (std::regex_match(cTensorName, cPatternAttnOutBias)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    }

    if (std::regex_match(cTensorName, cPatternAttnGateWeight)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_1, "attn_output.weight", "ssm_out.weight");
    }
    if (std::regex_match(cTensorName, cPatternSsmDt) || std::regex_match(cTensorName, cPatternSsmA)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_0, "ssm_out.weight");
    }
    if (std::regex_match(cTensorName, cPatternSsmAlpha) || std::regex_match(cTensorName, cPatternSsmBeta) ||
        std::regex_match(cTensorName, cPatternSsmBetaAlpha)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_1, "ssm_out.weight");
    }
    if (std::regex_match(cTensorName, cPatternRCache) || std::regex_match(cTensorName, cPatternSCache)) {
      if (vUd->vModel->vArch == cLlmArchLfm2 || vUd->vModel->vArch == cLlmArchLfm2moe) {
        // the LFM2 shortconv block runs fully mirrored, so its conv state must be mirrored too
        return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_MIRRORED, "");
      }
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_0, "ssm_out.weight");
    }
    if (std::regex_match(cTensorName, cPatternSsmConv1d)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_1, "ssm_out.weight");
    }
    if (std::regex_match(cTensorName, cPatternSsmOutWeight)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_0);
    }

    // FFN
    if (std::regex_match(cTensorName, cPatternFfnUpWeight) || std::regex_match(cTensorName, cPatternFfnGateWeight)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down.weight", "ffn_down_exps.weight");
    }
    if (std::regex_match(cTensorName, cPatternFfnUpBias) || std::regex_match(cTensorName, cPatternFfnGateBias)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down.weight", "ffn_down_exps.weight");
    }
    if (std::regex_match(cTensorName, cPatternFfnGateUpWeight)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_1, "ffn_down.weight", "ffn_down_exps.weight");
    }
    if (std::regex_match(cTensorName, cPatternFfnDownWeight)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_0, "ffn_down.weight", "ffn_down_exps.weight");
    }
    if (std::regex_match(cTensorName, cPatternFfnDownBias)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
    }
    if (std::regex_match(cTensorName, cPatternFfnDownExpsBias)) {
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_PARTIAL, "ffn_down_exps.weight");
    }

    // output
    if (std::regex_match(cTensorName, cPatternOutputWeight)) {
      if (cIsDsv4) {
        return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
      }
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_1);
    }
    if (std::regex_match(cTensorName, cPatternOutputBias)) {
      const ggml_tensor * vOutputWeight = vUd->vModel->mGetTensor("output.weight");
      GGML_ASSERT(vOutputWeight != nullptr);
      return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_0);
    }

    // everything else
    return vGetTensorConfigImpl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);
  };

  // if a model has fused tensors they need to be separated into "segments", see the comments on ggml_backend_meta_split_state struct
  auto vGetSplitSegments = [&](int pAxis, uint32_t pIl) -> std::vector<std::pair<int64_t, uint32_t>> {
    // TODO: deduplicate condition [TAG_SPLIT_QGATE_QWEN]
    if (vUd->vModel->vArch == cLlmArchQwen3next || vUd->vModel->vArch == cLlmArchQwen35 ||
        vUd->vModel->vArch == cLlmArchQwen35moe || vUd->vModel->vArch == cLlmArchClef ||
        vUd->vModel->vArch == cLlmArchQwen4exp) {
      // fused full attention layers with Q gate tensors that need n_embd doubled:
      if (!vHparams.mIsRecr(pIl) &&
          (std::regex_match(cTensorName, cPatternQkvWeight) || std::regex_match(cTensorName, cPatternQkvBias))) {
        const int64_t cNEmbd    = vHparams.n_head(pIl) * vHparams.n_embd_head_k(pIl) * 2;
        const int64_t cNEmbdGqa = vHparams.n_embd_v_gqa(pIl);
        GGML_ASSERT(vHparams.n_embd_k_gqa(pIl) == cNEmbdGqa);
        GGML_ASSERT(pTensor->ne[pAxis] == cNEmbd + 2 * cNEmbdGqa);
        return {
          { cNEmbd,    1 },
          { cNEmbdGqa, 2 }
        };
      }

      const int64_t cHeadKDim = vHparams.vSsmDState;
      const int64_t cHeadVDim = vHparams.vSsmDState;
      const int64_t cNKHeads  = vHparams.vSsmNGroup;
      const int64_t cNVHeads  = vHparams.vSsmDtRank;
      const int64_t cKeyDim   = cHeadKDim * cNKHeads;
      const int64_t cValueDim = cHeadVDim * cNVHeads;

      // both Qwen 3 Next and Qwen 3.5 support n_v_heads > n_k_heads but the broadcasting pattern is different:
      //   - Qwen 3 Next: [k0_v0, k0_v1, k1_v2, k1_v3] (this is the default split pattern)
      //   - Qwen 3.5:    [k0_v0, k1_v1, k0_v2, k1_v3] (needs segmenting of V on the scale of K to get the correct pattern)
      if (vUd->vModel->vArch == cLlmArchQwen3next) {
        if (std::regex_match(cTensorName, cPatternQkvWeight) || std::regex_match(cTensorName, cPatternSsmConv1d)) {
          GGML_ASSERT(pTensor->ne[pAxis] == 2 * cKeyDim + cValueDim);
          return {
            { cKeyDim,   2 },
            { cValueDim, 1 }
          };
        }
        if (std::regex_match(cTensorName, cPatternRCache)) {
          return {
            { cKeyDim * (vHparams.vSsmDConv - 1),   2 },
            { cValueDim * (vHparams.vSsmDConv - 1), 1 }
          };
        }
      } else {
        const int64_t cHeadRatio = cNVHeads / cNKHeads;
        if (std::regex_match(cTensorName, cPatternQkvWeight) || std::regex_match(cTensorName, cPatternSsmConv1d)) {
          GGML_ASSERT(pTensor->ne[pAxis] == 2 * cKeyDim + cValueDim);
          return {
            { cKeyDim, 2 + cHeadRatio }
          };
        }
        if (std::regex_match(cTensorName, cPatternAttnGateWeight) ||
            std::regex_match(cTensorName, cPatternSsmOutWeight)) {
          return {
            { cKeyDim, cHeadRatio }
          };
        }
        if (std::regex_match(cTensorName, cPatternSsmDt) || std::regex_match(cTensorName, cPatternSsmA) ||
            std::regex_match(cTensorName, cPatternSsmAlpha) || std::regex_match(cTensorName, cPatternSsmBeta)) {
          return {
            { cNKHeads, cHeadRatio }
          };
        }
        if (std::regex_match(cTensorName, cPatternRCache)) {
          return {
            { cKeyDim * (vHparams.vSsmDConv - 1), 2 + cHeadRatio }
          };
        }
        if (std::regex_match(cTensorName, cPatternSCache)) {
          return {
            { cNKHeads * cHeadVDim * cHeadVDim, cHeadRatio }
          };
        }
      }

      // the FFN is the same for Qwen 3 Next and Qwen 3.5:
      if (std::regex_match(cTensorName, cPatternFfnGateUpWeight)) {
        const int64_t cNFfExp = vHparams.mNFfExp(pIl);
        GGML_ASSERT(pTensor->ne[pAxis] == 2 * cNFfExp);
        return {
          { cNFfExp, 2 }
        };
      }
      return {
        { pTensor->ne[pAxis], 1 }
      };
    }

    if (std::regex_match(cTensorName, cPatternQkvWeight) || std::regex_match(cTensorName, cPatternQkvBias)) {
      const int64_t cNEmbdQ = vHparams.n_head(pIl) * vHparams.n_embd_head_k(pIl);
      const int64_t cNEmbdK = vHparams.n_embd_k_gqa(pIl);
      const int64_t cNEmbdV = vHparams.n_embd_v_gqa(pIl);
      GGML_ASSERT(pTensor->ne[pAxis] == cNEmbdQ + cNEmbdK + cNEmbdV);
      if (cNEmbdK == cNEmbdV) {
        return {
          { cNEmbdQ, 1 },
          { cNEmbdK, 2 }
        };
      }
      // uneven K/V head sizes (e.g. MiMo d_k=192 d_v=128): split K and V as separate
      // segments so each device gets whole heads of both
      return {
        { cNEmbdQ, 1 },
        { cNEmbdK, 1 },
        { cNEmbdV, 1 }
      };
    }
    if (std::regex_match(cTensorName, cPatternFfnUpWeight) || std::regex_match(cTensorName, cPatternFfnUpBias)) {
      const int64_t cNFf = vHparams.n_ff(pIl);
      // some models such as Phi 3 have fused up + gate tensors named "up" tensors, which need to be segmented
      if (pTensor->ne[pAxis] == 2 * cNFf) {
        return {
          { cNFf, 2 }
        };
      }
      return {
        { pTensor->ne[pAxis], 1 }
      };
    }
    if (std::regex_match(cTensorName, cPatternFfnGateUpWeight)) {
      const int64_t cNFfExp = vHparams.mNFfExp(pIl);
      GGML_ASSERT(pTensor->ne[pAxis] == 2 * cNFfExp);
      return {
        { cNFfExp, 2 }
      };
    }
    return {
      { pTensor->ne[pAxis], 1 }
    };
  };

  auto vGetSplitGranularity = [&](int64_t pBlckSize, uint32_t pIl,
                                  const std::vector<std::pair<int64_t, uint32_t>> & pSegments) -> std::vector<int64_t> {
    // for better performance it may make sense to round up blck_size to a higher power of 2 so that more efficient kernels can be used
    if (vHparams.mIsRecr(pIl)) {
      // linear attention
      const int64_t cHeadDim        = vHparams.vSsmDState;
      const int64_t cBlckSizePerf   = std::lcm(pBlckSize, 128);
      const int64_t cGranularityQkv = std::lcm(cBlckSizePerf, cHeadDim);
      if (std::regex_match(cTensorName, cPatternQkvWeight) || std::regex_match(cTensorName, cPatternAttnGateWeight) ||
          std::regex_match(cTensorName, cPatternSsmConv1d) || std::regex_match(cTensorName, cPatternSsmOutWeight)) {
        return std::vector<int64_t>(pSegments.size(), cGranularityQkv);
      }
      if (std::regex_match(cTensorName, cPatternSsmDt) || std::regex_match(cTensorName, cPatternSsmA) ||
          std::regex_match(cTensorName, cPatternSsmAlpha) || std::regex_match(cTensorName, cPatternSsmBeta)) {
        return std::vector<int64_t>(pSegments.size(), cGranularityQkv / cHeadDim);
      }
      if (std::regex_match(cTensorName, cPatternSsmBetaAlpha)) {
        return std::vector<int64_t>(pSegments.size(), 2 * (cGranularityQkv / cHeadDim));
      }
      if (std::regex_match(cTensorName, cPatternRCache)) {
        return std::vector<int64_t>(pSegments.size(), cGranularityQkv * (vHparams.vSsmDConv - 1));
      }
      if (std::regex_match(cTensorName, cPatternSCache)) {
        return std::vector<int64_t>(pSegments.size(), cGranularityQkv * cHeadDim);
      }
    } else {
      // regular attention
      const uint32_t cNGqa   = vHparams.mNGqa(pIl);
      const uint32_t cNEmbdQ = cNGqa * vHparams.n_embd_head_k(pIl);

      // to handle head sizes like 80, only increase granularity while it doesn't cause underutilization
      int64_t vBlckSizePerf = pBlckSize;
      while (vBlckSizePerf < 128 && vBlckSizePerf * vUd->vNDevices < cNEmbdQ) {
        vBlckSizePerf *= 2;
      }

      const int64_t cGranularityQ = std::lcm(cNEmbdQ, vBlckSizePerf);
      const int64_t cGranularityHead =
        cGranularityQ / vHparams.n_embd_head_k(pIl);  // for tensors with one value per head
      if (std::regex_match(cTensorName, cPatternAttnSinks)) {
        GGML_ASSERT(pSegments.size() == 1);
        if (cIsDsv4) {
          return { vHparams.n_head(pIl) / vHparams.vDsv4OGroupCount };
        }
        return { cGranularityHead };
      }

      if (cIsDsv4) {
        if (std::regex_match(cTensorName, cPatternAttnQBWeight)) {
          GGML_ASSERT(pSegments.size() == 1);
          // the grouped output projection requires each device to hold whole groups of heads
          const int64_t cNHeadGroup = vHparams.n_head(pIl) / vHparams.vDsv4OGroupCount;
          return { cNHeadGroup * vHparams.n_embd_head_k(pIl) };
        }
        if (std::regex_match(cTensorName, cPatternAttnOutAWeight)) {
          GGML_ASSERT(pSegments.size() == 1);
          return { 1 };
        }
        if (std::regex_match(cTensorName, cPatternAttnOutBWeight)) {
          GGML_ASSERT(pSegments.size() == 1);
          // the boundaries must align with wo_a's per-group split, so quant blocks must not straddle groups
          GGML_ASSERT(vHparams.vDsv4OLoraRank % pBlckSize == 0);
          return { vHparams.vDsv4OLoraRank };
        }
      }
      if (std::regex_match(cTensorName, cPatternQWeight) || std::regex_match(cTensorName, cPatternQBias)) {
        GGML_ASSERT(pSegments.size() == 1);
        // some models have Q gate tensors, for those cases the granularity needs to be doubled:
        // TODO: deduplicate condition [TAG_SPLIT_QGATE_QWEN]
        if (vUd->vModel->vArch == cLlmArchQwen3next || vUd->vModel->vArch == cLlmArchQwen35 ||
            vUd->vModel->vArch == cLlmArchQwen35moe || vUd->vModel->vArch == cLlmArchClef ||
            vUd->vModel->vArch == cLlmArchQwen4exp) {
          return { std::lcm(2 * cNEmbdQ, vBlckSizePerf) };
        }
        return { cGranularityQ };
      }
      if (std::regex_match(cTensorName, cPatternAttnOutWeight)) {
        GGML_ASSERT(pSegments.size() == 1);
        return { cGranularityHead * vHparams.n_embd_head_v(pIl) };
      }
      if (std::regex_match(cTensorName, cPatternAttnGateWeight)) {
        GGML_ASSERT(pSegments.size() == 1);
        if (pTensor->ne[1] == vHparams.n_head(pIl)) {
          return { cGranularityHead };
        }
        return { cGranularityQ };
      }

      const int64_t cGranularityKv = cGranularityQ / cNGqa;
      // the V head size can differ from the K head size (e.g. MiMo d_k=192 d_v=128):
      // align V tensors to whole V heads at the same head-index scale as Q and K so all
      // three stay in lockstep per device
      const int64_t cGranularityV  = (cGranularityKv / vHparams.n_embd_head_k(pIl)) * vHparams.n_embd_head_v(pIl);
      if (std::regex_match(cTensorName, cPatternKvWeight) || std::regex_match(cTensorName, cPatternVExpsWeight) ||
          std::regex_match(cTensorName, cPatternKvBias) || std::regex_match(cTensorName, cPatternKvCache)) {
        GGML_ASSERT(pSegments.size() == 1);
        const bool cIsV =
          cTensorName.find("attn_v") != std::string::npos || cTensorName.find("cache_v") != std::string::npos;
        return { cIsV ? cGranularityV : cGranularityKv };
      }
      if (std::regex_match(cTensorName, cPatternQkvWeight) || std::regex_match(cTensorName, cPatternQkvBias)) {
        // fused full attention layers need Q gate tensors handled like above:
        // TODO: deduplicate condition [TAG_SPLIT_QGATE_QWEN]
        if (vUd->vModel->vArch == cLlmArchQwen3next || vUd->vModel->vArch == cLlmArchQwen35 ||
            vUd->vModel->vArch == cLlmArchQwen35moe || vUd->vModel->vArch == cLlmArchClef ||
            vUd->vModel->vArch == cLlmArchQwen4exp) {
          return { std::lcm(2 * cNEmbdQ, vBlckSizePerf), cGranularityKv };
        }
        if (pSegments.size() == 3) {
          // uneven K/V head sizes: per-segment granularity, V aligned to whole V heads
          return { cGranularityQ, cGranularityKv, cGranularityV };
        }
        GGML_ASSERT(pSegments.size() == 2);
        return { cGranularityQ, cGranularityKv };
      }
    }

    // FFN
    if (std::regex_match(cTensorName, cPatternFfnUpWeight) || std::regex_match(cTensorName, cPatternFfnUpBias) ||
        std::regex_match(cTensorName, cPatternFfnGateWeight) || std::regex_match(cTensorName, cPatternFfnGateBias) ||
        std::regex_match(cTensorName, cPatternFfnGateUpWeight) ||
        std::regex_match(cTensorName, cPatternFfnDownWeight) ||
        std::regex_match(cTensorName, cPatternFfnUpShexpWeight) ||
        std::regex_match(cTensorName, cPatternFfnGateShexpWeight) ||
        std::regex_match(cTensorName, cPatternFfnDownShexpWeight)) {
      const int64_t cBlckSizePerf = std::lcm(pBlckSize, 128);
      GGML_ASSERT(pSegments.size() == 1);
      return { cBlckSizePerf };
    }

    // everything else
    GGML_ASSERT(pSegments.size() == 1);
    return { 1 };
  };

  ggml_backend_meta_split_state vSplitState;
  memset(&vSplitState, 0, sizeof(vSplitState));
  tensor_config vTc = vGetTensorConfig();
  vSplitState.axis  = vTc.vAxis;
  if (vSplitState.axis >= 0 && vSplitState.axis < GGML_MAX_DIMS) {
    const int64_t      cBlckSize    = ggml_blck_size(vTc.vTensorAxis0->type);
    const float *      vTensorSplit = vUd->vModel->mTensorSplit();
    std::vector<float> lTensorSplitScan;
    lTensorSplitScan.reserve(vUd->vNDevices);
    for (size_t vJ = 0; vJ < vUd->vNDevices; vJ++) {
      lTensorSplitScan.push_back(vTensorSplit == nullptr ? 0.0f : vTensorSplit[(vJ + vTc.vRotation) % vUd->vNDevices]);
      if (vJ > 0) {
        lTensorSplitScan[vJ] += lTensorSplitScan[vJ - 1];
      }
    }
    const std::vector<std::pair<int64_t, uint32_t>> lSegments    = vGetSplitSegments(vSplitState.axis, vTc.vIl);
    const std::vector<int64_t>                      lGranularity = vGetSplitGranularity(cBlckSize, vTc.vIl, lSegments);
    for (size_t vIs = 0; vIs < lSegments.size(); vIs++) {
      const int64_t  cNeS = lSegments[vIs].first;
      const uint32_t cNrS = lSegments[vIs].second;
      const int64_t  cGS  = lGranularity[vIs];
      int64_t        vLow = 0;
      size_t         vJ   = 0;
      for (; vJ < vUd->vNDevices - 1; vJ++) {
        int64_t vHigh = lTensorSplitScan.back() == 0.0f ? cNeS * (vJ + 1) / vUd->vNDevices :
                                                          cNeS * lTensorSplitScan[vJ] / lTensorSplitScan.back();
        if (vHigh % cGS != 0) {
          vHigh -= vHigh % cGS;
        }
        vSplitState.ne[vIs * vUd->vNDevices + (vJ + vTc.vRotation) % vUd->vNDevices] = vHigh - vLow;
        vLow                                                                         = vHigh;
      }
      vSplitState.ne[vIs * vUd->vNDevices + (vJ + vTc.vRotation) % vUd->vNDevices] = cNeS - vLow;
      vSplitState.nr[vIs]                                                          = cNrS;
    }
    vSplitState.n_segments = lSegments.size();
  } else {
    memset(vSplitState.ne, 0, sizeof(vSplitState.ne));
    vSplitState.nr[0]      = 1;
    vSplitState.n_segments = 1;
    if (vSplitState.axis == GGML_BACKEND_SPLIT_AXIS_PARTIAL) {
      GGML_ASSERT(vTc.vTensorAxis0 != pTensor);
      const ggml_backend_meta_split_state cSourceSplitState = fDgsMetaDeviceGetSplitState(vTc.vTensorAxis0, pUserdata);
      GGML_ASSERT(cSourceSplitState.axis >= 0 && cSourceSplitState.axis < GGML_MAX_DIMS);
      for (size_t vJ = 0; vJ < vUd->vNDevices; vJ++) {
        for (size_t vIs = 0; vIs < cSourceSplitState.n_segments; vIs++) {
          vSplitState.ne[vJ] += cSourceSplitState.ne[vIs * vUd->vNDevices + vJ] * cSourceSplitState.nr[vIs];
        }
      }
    }
  }
  return vSplitState;
  GGML_UNUSED(pUserdata);
}

const char * fLlmTypeName(llm_type type) {
  switch (type) {
    case cLlmType14m:
      return "14M";
    case cLlmType17m:
      return "17M";
    case cLlmType22m:
      return "22M";
    case cLlmType33m:
      return "33M";
    case cLlmType47m:
      return "47M";
    case cLlmType60m:
      return "60M";
    case cLlmType70m:
      return "70M";
    case cLlmType80m:
      return "80M";
    case cLlmType109m:
      return "109M";
    case cLlmType137m:
      return "137M";
    case cLlmType140m:
      return "140M";
    case cLlmType149m:
      return "149M";
    case cLlmType160m:
      return "160M";
    case cLlmType190m:
      return "190M";
    case cLlmType220m:
      return "220M";
    case cLlmType230m:
      return "230M";
    case cLlmType250m:
      return "250M";
    case cLlmType256m:
      return "256M";
    case cLlmType270m:
      return "270M";
    case cLlmType335m:
      return "335M";
    case cLlmType350m:
      return "350M";
    case cLlmType360m:
      return "360M";
    case cLlmType395m:
      return "395M";
    case cLlmType410m:
      return "410M";
    case cLlmType450m:
      return "450M";
    case cLlmType475m:
      return "475M";
    case cLlmType558m:
      return "558M";
    case cLlmType700m:
      return "700M";
    case cLlmType770m:
      return "770M";
    case cLlmType780m:
      return "780M";
    case cLlmType950m:
      return "950M";
    case cLlmType0Point3B:
      return "0.3B";
    case cLlmType0Point5B:
      return "0.5B";
    case cLlmType0Point6B:
      return "0.6B";
    case cLlmType0Point8B:
      return "0.8B";
    case cLlmType1b:
      return "1B";
    case cLlmType1Point2B:
      return "1.2B";
    case cLlmType1Point3B:
      return "1.3B";
    case cLlmType1Point4B:
      return "1.4B";
    case cLlmType1Point5B:
      return "1.5B";
    case cLlmType1Point6B:
      return "1.6B";
    case cLlmType1Point7B:
      return "1.7B";
    case cLlmType1Point8B:
      return "1.8B";
    case cLlmType2b:
      return "2B";
    case cLlmType2Point6B:
      return "2.6B";
    case cLlmType2Point8B:
      return "2.8B";
    case cLlmType2Point9B:
      return "2.9B";
    case cLlmType3b:
      return "3B";
    case cLlmType4b:
      return "4B";
    case cLlmType6b:
      return "6B";
    case cLlmType6Point9B:
      return "6.9B";
    case cLlmType7b:
      return "7B";
    case cLlmType8b:
      return "8B";
    case cLlmType9b:
      return "9B";
    case cLlmType11b:
      return "11B";
    case cLlmType12b:
      return "12B";
    case cLlmType13b:
      return "13B";
    case cLlmType14b:
      return "14B";
    case cLlmType15b:
      return "15B";
    case cLlmType16b:
      return "16B";
    case cLlmType20b:
      return "20B";
    case cLlmType26b:
      return "26B";
    case cLlmType27b:
      return "27B";
    case cLlmType30b:
      return "30B";
    case cLlmType31b:
      return "31B";
    case cLlmType32b:
      return "32B";
    case cLlmType34b:
      return "34B";
    case cLlmType35b:
      return "35B";
    case cLlmType36b:
      return "36B";
    case cLlmType40b:
      return "40B";
    case cLlmType65b:
      return "65B";
    case cLlmType70b:
      return "70B";
    case cLlmType120b:
      return "120B";
    case cLlmType142b:
      return "142B";
    case cLlmType236b:
      return "236B";
    case cLlmType290b:
      return "290B";
    case cLlmType314b:
      return "314B";
    case cLlmType405b:
      return "405B";
    case cLlmType456b:
      return "456B";
    case cLlmType671b:
      return "671B";
    case cLlmTypeSmall:
      return "0.1B";
    case cLlmTypeMedium:
      return "0.4B";
    case cLlmTypeLarge:
      return "0.8B";
    case cLlmTypeXl:
      return "1.5B";
    case cLlmTypeA1Point7B:
      return "A1.7B";
    case cLlmTypeA2Point7B:
      return "A2.7B";
    case cLlmType8x7B:
      return "8x7B";
    case cLlmType8x22B:
      return "8x22B";
    case cLlmType16x12B:
      return "16x12B";
    case cLlmType16x3Point8B:
      return "16x3.8B";
    case cLlmType10b128x3Point66B:
      return "10B+128x3.66B";
    case cLlmType57bA14b:
      return "57B.A14B";
    case cLlmType17b16e:
      return "17Bx16E (Scout)";
    case cLlmType17b128e:
      return "17Bx128E (Maverick)";
    case cLlmTypeA13b:
      return "A13B";
    case cLlmType1bA400m:
      return "1B.A400M";
    case cLlmType3bA800m:
      return "3B.A800M";
    case cLlmType7bA1b:
      return "7B.A1B";
    case cLlmType8bA1b:
      return "8B.A1B";
    case cLlmType7Point9BA1Point3B:
      return "7.9B.A1.3B";
    case cLlmType12bA2Point5B:
      return "12B.A2.5B";
    case cLlmType16bA1b:
      return "16B.A1B";
    case cLlmType21bA3b:
      return "21B.A3B";
    case cLlmType24bA2b:
      return "24B.A2B";
    case cLlmType26bA4b:
      return "26B.A4B";
    case cLlmType30bA3b:
      return "30B.A3B";
    case cLlmType31bA3Point5B:
      return "31B.A3.5B";
    case cLlmType32bA9b:
      return "32B.A9B";
    case cLlmType35bA3b:
      return "35B.A3B";
    case cLlmType48bA3b:
      return "48B.A3B";
    case cLlmType75bA9b:
      return "75B.A9B";
    case cLlmType80bA3b:
      return "80B.A3B";
    case cLlmTypeA3b:
      return "A3B";
    case cLlmType100bA6b:
      return "100B.A6B";
    case cLlmType102bA12b:
      return "102B.A12B";
    case cLlmType106bA12b:
      return "106B.A12B";
    case cLlmType118bA8b:
      return "118B.A8B";
    case cLlmType120bA12b:
      return "120B.A12B";
    case cLlmType122bA10b:
      return "122B.A10B";
    case cLlmType124bA5Point1B:
      return "124B.A5.1B";
    case cLlmType196bA11b:
      return "196B.A11B";
    case cLlmType230bA10b:
      return "230B.A10B";
    case cLlmType428bA23b:
      return "428B.A23B";
    case cLlmType235bA22b:
      return "235B.A22B";
    case cLlmType288bA19b:
      return "288B.A19B";
    case cLlmType300bA47b:
      return "300B.A47B";
    case cLlmType310bA15b:
      return "310B.A15B";
    case cLlmType355bA32b:
      return "355B.A32B";
    case cLlmType397bA17b:
      return "397B.A17B";
    case cLlmType685bA37b:
      return "685B.A37B";
    case cLlmType744bA40b:
      return "744B.A40B";
    case cLlmType2Point8TA50b:
      return "2.8T.A50B";
    case cLlmType320bA18b:
      return "320B.A18B";
    case cLlmTypeE2b:
      return "E2B";
    case cLlmTypeE4b:
      return "E4B";
    default:
      return "?B";
  }
}

static const char * fDgsExpertGatingFuncName(dgs_expert_gating_func_type type) {
  switch (type) {
    case cDgsExpertGatingFuncTypeSoftmax:
      return "softmax";
    case cDgsExpertGatingFuncTypeSigmoid:
      return "sigmoid";
    case cDgsExpertGatingFuncTypeSqrtSoftplus:
      return "sqrtsoftplus";
    default:
      return "unknown";
  }
}

static const std::map<dgs_rope_scaling_type, const char *> dDgsRopeScalingTypes = {
  { cDgsRopeScalingTypeNone,     "none"     },
  { cDgsRopeScalingTypeLinear,   "linear"   },
  { cDgsRopeScalingTypeYarn,     "yarn"     },
  { cDgsRopeScalingTypeLongrope, "longrope" },
};

std::string fDgsRopeScalingTypeName(dgs_rope_scaling_type pRopeScalingType) {
  return dDgsRopeScalingTypes.at(pRopeScalingType);
}

static dgs_rope_scaling_type fDgsRopeScalingTypeFromString(const std::string & name) {
  for (const auto & vKv : dDgsRopeScalingTypes) {
    if (vKv.second == name) {
      return (dgs_rope_scaling_type) vKv.first;
    }
  }

  return cDgsRopeScalingTypeUnspecified;
}

static const std::map<dgs_rope_section_order, const char *> dDgsRopeSectionOrders = {
  { cDgsRopeSectionOrderTyxz, "tyxz" },
  { cDgsRopeSectionOrderZyxt, "zyxt" },
};

std::string fDgsRopeSectionOrderName(dgs_rope_section_order pRopeSectionOrder) {
  return dDgsRopeSectionOrders.at(pRopeSectionOrder);
}

static dgs_rope_section_order fDgsRopeSectionOrderFromString(const std::string & name) {
  for (const auto & vKv : dDgsRopeSectionOrders) {
    if (vKv.second == name) {
      return vKv.first;
    }
  }

  return cDgsRopeSectionOrderUnspecified;
}

// Maps GGUF activation names to the FFN op type used by the graph builders.
static const std::map<std::string, llm_ffn_op_type> dLlmFfnOpTypesFromString = {
  { "gelu",              cLlmFfnGegluErf },
  { "gelu_python",       cLlmFfnGegluErf },
  { "gelu_pytorch_tanh", cLlmFfnGeglu    },
  { "gelu_new",          cLlmFfnGeglu    },
  { "gelu_fast",         cLlmFfnGeglu    },
  { "gelu_accurate",     cLlmFfnGeglu    },
  { "gelu_python_tanh",  cLlmFfnGeglu    },
  { "geglu",             cLlmFfnGeglu    },
  { "silu",              cLlmFfnSwiglu   },
  { "swish",             cLlmFfnSwiglu   },
  { "swiglu",            cLlmFfnSwiglu   },
  { "relu",              cLlmFfnRelu     },
  { "reglu",             cLlmFfnReglu    },
};

// transformers names, "gelu" is the exact (erf) variant
static const std::map<std::string, ggml_unary_op> dLlmClsActTypesFromString = {
  { "gelu", GGML_UNARY_OP_GELU_ERF },
  { "silu", GGML_UNARY_OP_SILU     },
  { "tanh", GGML_UNARY_OP_TANH     },
};

llm_ffn_op_type fLlmFfnOpTypeFromString(const std::string & name, llm_ffn_op_type pFallback) {
  const auto cIt = dLlmFfnOpTypesFromString.find(name);
  if (cIt != dLlmFfnOpTypesFromString.end()) {
    return cIt->second;
  }
  return pFallback;
}

// CPU: ACCEL -> GPU host -> CPU extra -> CPU
static buft_list_t fMakeCpuBuftList(const std::vector<dgs_device> & pDevices, bool pUseExtraBufts, bool pNoHost) {
  buft_list_t lBuftList;

  // add ACCEL buffer types
  for (size_t vI = 0; vI < ggml_backend_dev_count(); ++vI) {
    ggml_backend_dev_t vDev = ggml_backend_dev_get(vI);
    if (ggml_backend_dev_type(vDev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
      auto * vBuft = ggml_backend_dev_buffer_type(vDev);
      // skip
      if (vBuft != ggml_backend_cpu_buffer_type()) {
        lBuftList.emplace_back(vDev, vBuft);
      }
    }
  }

  // add a host buffer type
  // storing the tensors in a host buffer is useful when the processing of large batches
  // is offloaded to a GPU device, since it reduces the time spent on data transfers
  // generally, this will be done using the first device in the list
  // a better approach would be to handle this on a weight-by-weight basis using the offload_op
  // function of the device to determine if it would benefit from being stored in a host buffer
  if (!pNoHost) {
    for (const auto & vDev : pDevices) {
      ggml_backend_buffer_type_t vBuft = ggml_backend_dev_host_buffer_type(vDev.dev);
      if (vBuft) {
        lBuftList.emplace_back(vDev.dev, vBuft);
        break;
      }
    }
  }

  // add extra buffer types
  if (pUseExtraBufts) {
    auto * vCpuDev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (vCpuDev == nullptr) {
      throw std::runtime_error(format("%s: no CPU backend found", __func__));
    }

    auto * vCpuReg                        = ggml_backend_dev_backend_reg(vCpuDev);
    auto   vGgmlBackendDevGetExtraBuftsFn = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(
      vCpuReg, "ggml_backend_dev_get_extra_bufts");
    if (vGgmlBackendDevGetExtraBuftsFn) {
      ggml_backend_buffer_type_t * vExtraBufts = vGgmlBackendDevGetExtraBuftsFn(vCpuDev);
      while (vExtraBufts && *vExtraBufts) {
        lBuftList.emplace_back(vCpuDev, *vExtraBufts);
        ++vExtraBufts;
      }
    }
  }

  // add the CPU buffer type
  for (size_t vI = 0; vI < ggml_backend_dev_count(); ++vI) {
    ggml_backend_dev_t vDev = ggml_backend_dev_get(vI);
    if (ggml_backend_dev_type(vDev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
      lBuftList.emplace_back(vDev, ggml_backend_dev_buffer_type(vDev));
    }
  }

  return lBuftList;
}

// GPU: split if DGS_SPLIT_MODE_ROW -> GPU
static buft_list_t fMakeGpuBuftList(ggml_backend_dev_t pDev, dgs_split_mode pSplitMode, const float * pTensorSplit) {
  buft_list_t lBuftList;

  // add the device split buffer type if requested and available
  if (pSplitMode == cDgsSplitModeRow) {
    ggml_backend_reg_t vReg = ggml_backend_dev_backend_reg(pDev);
    auto               vGgmlBackendSplitBufferTypeFn =
      (ggml_backend_split_buffer_type_t) ggml_backend_reg_get_proc_address(vReg, "ggml_backend_split_buffer_type");
    if (vGgmlBackendSplitBufferTypeFn) {
      size_t vDevIndex = [&]() {
        auto * vReg = ggml_backend_dev_backend_reg(pDev);
        for (size_t vI = 0; vI < ggml_backend_reg_dev_count(vReg); ++vI) {
          if (ggml_backend_reg_dev_get(vReg, vI) == pDev) {
            return vI;
          }
        }
        throw std::runtime_error(format("device %s not found in its backend reg", ggml_backend_dev_name(pDev)));
      }();
      auto * vBuft = vGgmlBackendSplitBufferTypeFn(vDevIndex, pTensorSplit);
      if (vBuft != nullptr) {
        lBuftList.emplace_back(pDev, vBuft);
      }
    } else {
      throw std::runtime_error(format("device %s does not support split buffers", ggml_backend_dev_name(pDev)));
    }
  }

  // add the device default buffer type
  lBuftList.emplace_back(pDev, ggml_backend_dev_buffer_type(pDev));

  // add the device extra buffer type (if any)
  ggml_backend_reg_t vReg = ggml_backend_dev_backend_reg(pDev);
  if (vReg) {
    auto vGgmlBackendDevGetExtraBuftsFn =
      (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(vReg, "ggml_backend_dev_get_extra_bufts");

    if (vGgmlBackendDevGetExtraBuftsFn) {
      ggml_backend_buffer_type_t * vExtraBufts = vGgmlBackendDevGetExtraBuftsFn(pDev);
      while (vExtraBufts && *vExtraBufts) {
        lBuftList.emplace_back(pDev, *vExtraBufts);
        ++vExtraBufts;
      }
    }
  }

  return lBuftList;
}

struct dgs_model::impl {
  impl()  = default;
  ~impl() = default;

  uint64_t vNElements = 0;

  size_t vNBytes = 0;

  std::string vDescStr;

  dgs_ftype vFtype = cDgsFtypeAllF32;

  // model memory mapped files
  dgs_mmaps lMappings;

  // objects representing data potentially being locked in memory
  dgs_mlocks lMlockBufs;
  dgs_mlocks lMlockMmaps;

  // contexts where the model tensors metadata is stored as well as the corresponding buffers:
  std::vector<std::pair<ggml_context_ptr, std::vector<ggml_backend_buffer_ptr>>> lCtxsBufs;

  buft_list_t                               lCpuBuftList;
  std::map<ggml_backend_dev_t, buft_list_t> dGpuBuftList;

  struct layer_dev {
    ggml_backend_dev_t vDev;
    buft_list_t *      lBuftList;
  };

  layer_dev              vDevInput  = {};
  layer_dev              vDevOutput = {};
  std::vector<layer_dev> lDevLayer;

  bool vHasTensorOverrides;

  std::vector<float> lTensorSplitOwned;
};

bool dgs_prec_policy::apply(ggml_tensor * pRes) const {
  if (!pRes || !pRes->src[0]) {
    return false;
  }

  const auto cIt = dPrecSrc1.find(pRes->src[0]);
  if (cIt == dPrecSrc1.end()) {
    return false;
  }

  return ggml_prec_set_src(pRes, cIt->second, 1);
}

void dgs_prec_policy::mLoad(dgs_model_loader & pMl, const dgs_model & pModel) {
  std::vector<std::string> lTensorNames;
  if (!pMl.get_arr(cLlmKvGeneralTensorExtraName, lTensorNames, false)) {
    return;
  }

  const gguf_context * vCtx = pMl.vMetadata;
  const std::string    cKey = pMl.vLlmKv(cLlmKvGeneralTensorExtraPrecA4);
  const int            cKid = gguf_find_key(vCtx, cKey.c_str());
  if (cKid < 0 || gguf_get_kv_type(vCtx, cKid) != GGUF_TYPE_ARRAY || gguf_get_arr_type(vCtx, cKid) != GGUF_TYPE_BOOL) {
    throw std::runtime_error(format("%s must be a bool array", cKey.c_str()));
  }

  const size_t cNValues = gguf_get_arr_n(vCtx, cKid);
  if (cNValues != lTensorNames.size()) {
    throw std::runtime_error(
      format("%s tensor/value length mismatch (%zu vs %zu)", cKey.c_str(), lTensorNames.size(), cNValues));
  }

  // tensors that can not use 4-bit activations, keep src1 at higher precision
  const int8_t *                  vValues = (const int8_t *) gguf_get_arr_data(vCtx, cKid);
  std::unordered_set<std::string> sWant;
  for (size_t vI = 0; vI < cNValues; ++vI) {
    if (vValues[vI] == 0) {
      sWant.insert(lTensorNames[vI]);
    }
  }

  // resolve names to tensor pointers
  for (const auto & [name, w] : pModel.lTensorsByName) {
    if (sWant.count(name)) {
      dPrecSrc1.emplace(w, GGML_PREC_Q8);
    }
  }
}

dgs_model::dgs_model(const dgs_model_params & pParams) : vParams(pParams), vPimpl(std::make_unique<impl>()) {
  if (pParams.vTensorSplit != nullptr) {
    // dgs_model_params stores tensor_split as a borrowed pointer, but the model
    // may need it later for tensor-parallel KV-cache split metadata.
    vPimpl->lTensorSplitOwned.assign(pParams.vTensorSplit, pParams.vTensorSplit + fDgsMaxDevices());
    this->vParams.vTensorSplit = vPimpl->lTensorSplitOwned.data();
  }
  vPimpl->vHasTensorOverrides = pParams.vTensorBuftOverrides && pParams.vTensorBuftOverrides[0].vPattern;
}

dgs_model::~dgs_model() {
  for (auto * lora : sLoras) {
    delete lora;
  }
}

void dgs_model_base::mLoadStats(dgs_model_loader & pMl) {
  vPimpl->vNElements = pMl.vNElements;
  vPimpl->vNBytes    = pMl.vNBytes;
}

void dgs_model_base::mLoadHparams(dgs_model_loader & pMl) {
  const gguf_context * vCtx = pMl.vMetadata;

  // get metadata as string
  for (int vI = 0; vI < gguf_get_n_kv(vCtx); vI++) {
    gguf_type type = gguf_get_kv_type(vCtx, vI);
    if (type == GGUF_TYPE_ARRAY) {
      continue;
    }
    const char *      name  = gguf_get_key(vCtx, vI);
    const std::string value = fGgufKvToStr(vCtx, vI);
    dGgufKv.emplace(name, value);
  }

  mLoadHparamsHadamard(pMl);

  // get general kv
  pMl.get_key(cLlmKvGeneralName, name, false);

  // everything past this point is not vocab-related
  // for CLIP models, we only need to load tensors, no hparams
  if (hparams.vVocabOnly || pMl.mGetArch() == cLlmArchClip) {
    return;
  }

  pMl.get_key(cLlmKvContextLength, hparams.n_ctx_train);
  pMl.get_key(cLlmKvEmbeddingLength, hparams.n_embd);
  pMl.get_key(cLlmKvEmbeddingLengthOut, hparams.vNEmbdOutImpl, false);
  pMl.get_key(cLlmKvAttentionCausal, hparams.vCausalAttn, false);
  pMl.get_key(cLlmKvPoolingType, hparams.vPoolingType, false);
  pMl.get_key(cLlmKvClassifierPoolingType, hparams.vPoolingTypeCls, false);
  std::string vActCls;
  if (pMl.get_key(cLlmKvClassifierActivation, vActCls, false)) {
    const auto cIt = dLlmClsActTypesFromString.find(vActCls);
    GGML_ASSERT(cIt != dLlmClsActTypesFromString.end() && "unsupported classifier activation");
    hparams.vActCls = cIt->second;
  }
  pMl.get_key(cLlmKvBlockCount, hparams.n_layer_all);
  GGML_ASSERT(hparams.n_layer_all > 0 && hparams.n_layer_all <= DGS_MAX_LAYERS);
  pMl.get_key(cLlmKvNextnPredictLayers, hparams.n_layer_nextn, false);
  GGML_ASSERT(hparams.n_layer_nextn <= hparams.n_layer_all);
  pMl.get_key(cLlmKvExpertCount, hparams.n_expert, false);
  std::fill(hparams.aNExpertUsedArr.begin(), hparams.aNExpertUsedArr.end(), 0);
  pMl.get_key_or_arr(cLlmKvExpertUsedCount, hparams.aNExpertUsedArr, hparams.n_layer_all, false);
  pMl.get_key(cLlmKvExpertGroupCount, hparams.vNExpertGroups, false);
  pMl.get_key(cLlmKvExpertGroupUsedCount, hparams.vNGroupUsed, false);

  if (vArch == cLlmArchHunyuanVl || vArch == cLlmArchHunyuanDense) {
    if (hparams.n_expert <= 1) {
      hparams.n_expert = 0;
      std::fill(hparams.aNExpertUsedArr.begin(), hparams.aNExpertUsedArr.end(), 0);
    }
  }

  if (hparams.n_expert != 0 || hparams.mNExpertUsedMax() != 0 || hparams.vNExpertGroups != 0) {
    throw std::runtime_error("DGS supports dense models only: expert metadata overrides are not supported");
  }

  if (vArch == cLlmArchWavtokenizerDec) {
    pMl.get_key(cLlmKvFeaturesLength, hparams.n_embd);
    pMl.get_key(cLlmKvEmbeddingLength, hparams.vNEmbdOutImpl);

    pMl.get_key(cLlmKvPosnetEmbeddingLength, hparams.vPosnet.vNEmbd);
    pMl.get_key(cLlmKvPosnetBlockCount, hparams.vPosnet.vNLayer);

    pMl.get_key(cLlmKvConvnextEmbeddingLength, hparams.vConvnext.vNEmbd);
    pMl.get_key(cLlmKvConvnextBlockCount, hparams.vConvnext.vNLayer);

    GGML_ASSERT(hparams.vPosnet.vNLayer <= hparams.n_layer_all);
    GGML_ASSERT(hparams.vConvnext.vNLayer <= hparams.n_layer_all);
  }

  // models may route a different number of experts per layer, so validate the maximum
  uint32_t vNExpertUsedMax = hparams.mNExpertUsedMax();

  GGML_ASSERT(hparams.n_expert <= DGS_MAX_EXPERTS);
  GGML_ASSERT(vNExpertUsedMax <= hparams.n_expert);
  if (hparams.n_expert > 0) {
    GGML_ASSERT(vNExpertUsedMax > 0);
    GGML_ASSERT(hparams.vNExpertGroups < hparams.n_expert);
    if (hparams.vNExpertGroups > 1) {
      GGML_ASSERT(hparams.n_expert % hparams.vNExpertGroups == 0);
      GGML_ASSERT(hparams.vNGroupUsed > 0);
      GGML_ASSERT(hparams.vNGroupUsed < hparams.vNExpertGroups);
    }
  } else {
    GGML_ASSERT(vNExpertUsedMax == 0);
    GGML_ASSERT(hparams.vNExpertGroups == 0);
  }

  std::fill(hparams.aNHeadArr.begin(), hparams.aNHeadArr.end(), 0);
  std::fill(hparams.aNHeadKvArr.begin(), hparams.aNHeadKvArr.end(), 0);
  std::fill(hparams.aNFfArr.begin(), hparams.aNFfArr.end(), 0);
  std::fill(hparams.aNFfExpArr.begin(), hparams.aNFfExpArr.end(), 0);

  std::fill(hparams.aRopeSections.begin(), hparams.aRopeSections.end(), 0);
  std::fill(hparams.aRopePattern.begin(), hparams.aRopePattern.end(), 1);
  std::fill(hparams.aIsSwaImpl.begin(), hparams.aIsSwaImpl.end(), 0);
  std::fill(hparams.aIsRecrImpl.begin(), hparams.aIsRecrImpl.end(), fLlmArchIsRecurrent(pMl.mGetArch()) ? 1 : 0);
  std::fill(hparams.aIsIndexerFullImpl.begin(), hparams.aIsIndexerFullImpl.end(), 0);

  std::fill(hparams.aXieluAlphaN.begin(), hparams.aXieluAlphaN.end(), 0.0f);
  std::fill(hparams.aXieluAlphaP.begin(), hparams.aXieluAlphaP.end(), 0.0f);
  std::fill(hparams.aXieluBeta.begin(), hparams.aXieluBeta.end(), 0.0f);
  std::fill(hparams.aXieluEps.begin(), hparams.aXieluEps.end(), 0.0f);

  std::fill(hparams.aSwigluClampExp.begin(), hparams.aSwigluClampExp.end(), 0.0f);
  std::fill(hparams.aSwigluClampShexp.begin(), hparams.aSwigluClampShexp.end(), 0.0f);

  pMl.get_key_or_arr(cLlmKvFeedForwardLength, hparams.aNFfArr, hparams.n_layer_all, false);
  pMl.get_key_or_arr(cLlmKvAttentionHeadCount, hparams.aNHeadArr, hparams.n_layer_all, false);

  // Populate deepstack_mapping_arr - initialized to -1 (no deepstack)
  std::fill(hparams.aDeepstackMappingArr.begin(), hparams.aDeepstackMappingArr.end(), -1);

  // n_head_kv is optional, default to n_head
  hparams.aNHeadKvArr = hparams.aNHeadArr;

  pMl.get_key_or_arr(cLlmKvAttentionHeadCountKv, hparams.aNHeadKvArr, hparams.n_layer_all, false);

  bool vRopeFinetuned = false;
  pMl.get_key(cLlmKvRopeScalingFinetuned, vRopeFinetuned, false);
  hparams.vRopeFinetuned = vRopeFinetuned;

  hparams.vNCtxOrigYarn = hparams.n_ctx_train;
  pMl.get_key(cLlmKvRopeScalingOrigCtxLen, hparams.vNCtxOrigYarn, false);

  // rope_freq_base (optional)
  hparams.vRopeFreqBaseTrain = 10000.0f;
  pMl.get_key(cLlmKvRopeFreqBase, hparams.vRopeFreqBaseTrain, false);

  std::string vRopeScaling("linear");
  pMl.get_key(cLlmKvRopeScalingType, vRopeScaling, false);
  hparams.vRopeScalingTypeTrain = fDgsRopeScalingTypeFromString(vRopeScaling);
  GGML_ASSERT(hparams.vRopeScalingTypeTrain != cDgsRopeScalingTypeUnspecified);

  std::string vRopeSectionOrder("tyxz");
  pMl.get_key(cLlmKvRopeSectionOrder, vRopeSectionOrder, false);
  hparams.vRopeSectionOrder = fDgsRopeSectionOrderFromString(vRopeSectionOrder);
  if (hparams.vRopeSectionOrder == cDgsRopeSectionOrderUnspecified) {
    throw std::runtime_error("unknown rope section order: " + vRopeSectionOrder);
  }

  // TODO: Handle SWA metadata similarly when models start implementing it
  // rope_freq_scale (inverse of the kv) is optional
  float vRopescale = 0.0f;
  pMl.get_key(cLlmKvRopeScalingFactor, vRopescale, false);
  hparams.vRopeFreqScaleTrain = vRopescale == 0.0f ? 1.0f : 1.0f / vRopescale;

  pMl.get_key(cLlmKvRopeScalingAttnFactor, hparams.vRopeAttnFactor, false);
  pMl.get_key(cLlmKvRopeScalingAlpha, hparams.vRopeScalingAlpha, false);

  // non-transformer models do not have attention heads
  if (hparams.n_head() > 0) {
    // gpt-neox n_rot = rotary_pct * (n_embd / n_head)
    // gpt-j n_rot = rotary_dim

    hparams.vNEmbdHeadKFull = hparams.n_embd / hparams.n_head();
    pMl.get_key(cLlmKvAttentionKeyLength, hparams.vNEmbdHeadKFull, false);

    hparams.vNEmbdHeadVFull = hparams.n_embd / hparams.n_head();
    pMl.get_key(cLlmKvAttentionValueLength, hparams.vNEmbdHeadVFull, false);

    // sanity check for n_rot (optional)
    hparams.vNRotFull = hparams.vNEmbdHeadKFull;

    pMl.get_key(cLlmKvRopeDimensionCount, hparams.vNRotFull, false);

    if (vArch == cLlmArchDgs || vArch == cLlmArchDeci || vArch == cLlmArchFalcon || vArch == cLlmArchDgsEmbed) {
      if (hparams.vNRotFull != hparams.vNEmbdHeadKFull) {
        throw std::runtime_error(format("invalid n_rot: %u, expected %u", hparams.vNRotFull, hparams.vNEmbdHeadKFull));
      }
    }
  } else {
    hparams.vNRotFull       = 0;
    hparams.vNEmbdHeadKFull = 0;
    hparams.vNEmbdHeadVFull = 0;
  }

  // head size and n_rot for SWA layers
  {
    hparams.vNEmbdHeadKSwa = hparams.vNEmbdHeadKFull;
    hparams.vNEmbdHeadVSwa = hparams.vNEmbdHeadVFull;
    pMl.get_key(cLlmKvAttentionKeyLengthSwa, hparams.vNEmbdHeadKSwa, false);
    pMl.get_key(cLlmKvAttentionValueLengthSwa, hparams.vNEmbdHeadVSwa, false);

    hparams.vNRotSwa = hparams.vNRotFull;
    pMl.get_key(cLlmKvRopeDimensionCountSwa, hparams.vNRotSwa, false);
  }

  // for classifier models
  pMl.get_arr(cLlmKvClassifierOutputLabels, lClassifierLabels, false);
  if (!lClassifierLabels.empty()) {
    hparams.vNClsOut = lClassifierLabels.size();
  }

  // per-arch hparams
  mLoadArchHparams(pMl);

  vPimpl->vNBytes = pMl.vNBytes;

  vPimpl->vDescStr = mArchName() + " " + mTypeName() + " " + pMl.mFtypeName();

  vPimpl->vFtype = pMl.vFtype;

  if (hparams.vFMaxAlibiBias > 0.0f) {
    hparams.vUseAlibi = true;
  }

  hparams.vRopeType = fDgsModelRopeType(this);

  if (hparams.vRopeSectionOrder != cDgsRopeSectionOrderTyxz && hparams.mNPosPerEmbd() != 4) {
    throw std::runtime_error("rope section order " + fDgsRopeSectionOrderName(hparams.vRopeSectionOrder) +
                             " requires M-RoPE");
  }
}

void dgs_model_base::mLoadVocab(dgs_model_loader & pMl) {
  const auto cKv = LLM_KV(vArch);

  vocab.mLoad(pMl, cKv);
}

bool dgs_model_base::mLoadTensors(dgs_model_loader & pMl) {
  const auto & vSplitMode   = vParams.vSplitMode;
  const bool   cUseMlock    = vParams.vLoadMode == cDgsLoadModeMlock || vParams.vLoadMode == cDgsLoadModeMmapMlock;
  const auto & vTensorSplit = vParams.vTensorSplit;

  const int cNLayerAll  = hparams.n_layer_all;
  const int cNGpuLayers = this->mNGpuLayers();

  const bool cUseMmapBuffer = true;

  this->vMl = &pMl;  // to be used by create_tensor() and load_arch_tensors()

  if (pMl.vUseMmap && vParams.vLoadMode == cDgsLoadModeAuto) {
    for (const auto & vDev : lDevices) {
      ggml_backend_dev_props vProps;
      ggml_backend_dev_get_props(vDev.dev, &vProps);
      if (!vProps.caps.mmap_support) {
        pMl.vUseMmap = false;
        break;
      }
    }
  }

  // resolve AUTO on systems without mmap support (e.g. iGPUs): fall back to OFF; see #28160
  if (pMl.vLazy.vMode == cDgsLazyModeAuto) {
    for (const auto & vDev : lDevices) {
      ggml_backend_dev_props vProps;
      ggml_backend_dev_get_props(vDev.dev, &vProps);
      if (!vProps.caps.mmap_support) {
        pMl.vLazy.vMode = cDgsLazyModeOff;
        break;
      }
    }
  }

  const char * vLoadModeName = vParams.vLoadMode == cDgsLoadModeAuto ?
                                 fDgsLoadModeName(pMl.vUseMmap ? cDgsLoadModeMmap : cDgsLoadModeNone) :
                                 fDgsLoadModeName(vParams.vLoadMode);

  DGS_LOG_INFO("%s: loading model tensors, this can take a while... (load_mode = %s)\n", __func__, vLoadModeName);

  // build a list of buffer types for the CPU and GPU devices
  vPimpl->lCpuBuftList = fMakeCpuBuftList(lDevices, vParams.vUseExtraBufts, vParams.vNoHost);
  for (const auto & vDev : lDevices) {
    buft_list_t lBuftList = fMakeGpuBuftList(vDev.dev, vSplitMode, vTensorSplit);
    // add CPU buffer types as a fallback
    lBuftList.insert(lBuftList.end(), vPimpl->lCpuBuftList.begin(), vPimpl->lCpuBuftList.end());
    vPimpl->dGpuBuftList.emplace(vDev.dev, std::move(lBuftList));
  }

  ggml_backend_dev_t vCpuDev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
  if (vCpuDev == nullptr) {
    throw std::runtime_error(format("%s: no CPU backend found", __func__));
  }

  // calculate the split points
  bool vAllZero = vTensorSplit == nullptr ||
                  std::all_of(vTensorSplit, vTensorSplit + mNDevices(), [](float pX) { return pX == 0.0f; });
  std::vector<float> lSplits(mNDevices());
  if (vAllZero) {
    // default split, by free memory
    for (size_t vI = 0; vI < mNDevices(); ++vI) {
      ggml_backend_dev_t vDev = lDevices[vI].dev;
      size_t             vTotal;
      size_t             free;
      ggml_backend_dev_memory(vDev, &free, &vTotal);

      // devices can return 0 bytes for free and total memory if they do not
      // have any to report. in this case, we will use the host memory as a fallback
      // fixes: 
      if (free == 0 && vTotal == 0) {
        ggml_backend_dev_memory(vCpuDev, &free, &vTotal);
      }
      lSplits[vI] = free;
    }
  } else {
    std::copy(vTensorSplit, vTensorSplit + mNDevices(), lSplits.begin());
  }

  // sum and normalize the splits to get the split points
  float vSplitSum = 0.0f;
  for (size_t vI = 0; vI < mNDevices(); ++vI) {
    vSplitSum += lSplits[vI];
    lSplits[vI] = vSplitSum;
  }
  for (size_t vI = 0; vI < mNDevices(); ++vI) {
    lSplits[vI] /= vSplitSum;
  }

  const int cIGpuStart        = std::max(cNLayerAll + 1 - cNGpuLayers, 0);
  const int cActGpuLayers     = lDevices.empty() ? 0 : std::min(cNGpuLayers, cNLayerAll + 1);
  auto      vGetLayerBuftList = [&](int pIl) -> dgs_model::impl::layer_dev {
    const bool cIsSwa = pIl < cNLayerAll && hparams.mIsSwa(pIl);
    if (pIl < cIGpuStart || (pIl - cIGpuStart) >= cActGpuLayers) {
      DGS_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", pIl, ggml_backend_dev_name(vCpuDev),
                         cIsSwa);
      return { vCpuDev, &vPimpl->lCpuBuftList };
    }
    const int cLayerGpu =
      std::upper_bound(lSplits.begin(), lSplits.begin() + mNDevices(), float(pIl - cIGpuStart) / cActGpuLayers) -
      lSplits.begin();
    auto * vDev = lDevices.at(cLayerGpu).dev;
    DGS_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", pIl, ggml_backend_dev_name(vDev),
                       cIsSwa);
    return { vDev, &vPimpl->dGpuBuftList.at(vDev) };
  };

  // assign the input layer
  // there is very little benefit to offloading the input layer, so always keep it on the CPU
  vPimpl->vDevInput = { vCpuDev, &vPimpl->lCpuBuftList };

  // assign the repeating layers to the devices according to the splits
  vPimpl->lDevLayer.resize(cNLayerAll);
  for (int vIl = 0; vIl < cNLayerAll; ++vIl) {
    vPimpl->lDevLayer[vIl] = vGetLayerBuftList(vIl);
  }

  // assign the output layer
  vPimpl->vDevOutput = vGetLayerBuftList(cNLayerAll);

  const auto cTensorNotRequired = dgs_model_loader::cTensorNotRequired;

  // create tensors for the weights
  {
    // TODO: move to a separate function
    const auto cTn = LLM_TN(vArch);

    const int64_t cNExpert = hparams.n_expert;

    if (cNExpert > 0 && hparams.mNExpertUsedMax() == 0) {
      throw std::runtime_error("model has expert layers but no expert layers are used");
    }

    lLayers.resize(cNLayerAll);

    // call the per-model loading function
    mLoadArchTensors(pMl);

    // generic pass: load optional per-tensor/per-expert ".scale" tensors (e.g. NVFP4 scale2)
    // this avoids having to add scale loading to every architecture
    for (int vI = 0; vI < cNLayerAll; ++vI) {
      auto & layer = lLayers[vI];

      // attention weight scales (per-tensor, shape {1})
      if (!layer.vWqS && layer.vWq) {
        layer.vWqS = mCreateTensor(cTn(cLlmTensorAttnQ, "scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vWkS && layer.vWk) {
        layer.vWkS = mCreateTensor(cTn(cLlmTensorAttnK, "scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vWvS && layer.vWv) {
        layer.vWvS = mCreateTensor(cTn(cLlmTensorAttnV, "scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vWoS && layer.vWo) {
        layer.vWoS = mCreateTensor(cTn(cLlmTensorAttnOut, "scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vWqkvS && layer.vWqkv) {
        layer.vWqkvS = mCreateTensor(cTn(cLlmTensorAttnQkv, "scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vWqkvGateS && layer.vWqkvGate) {
        layer.vWqkvGateS = mCreateTensor(cTn(cLlmTensorAttnGate, "scale", vI), { 1 }, cTensorNotRequired);
      }

      // dense FFN weight scales (per-tensor, shape {1})
      if (!layer.vFfnGateS && layer.vFfnGate) {
        layer.vFfnGateS = mCreateTensor(cTn(cLlmTensorFfnGate, "scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vFfnDownS && layer.vFfnDown) {
        layer.vFfnDownS = mCreateTensor(cTn(cLlmTensorFfnDown, "scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vFfnUpS && layer.vFfnUp) {
        layer.vFfnUpS = mCreateTensor(cTn(cLlmTensorFfnUp, "scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vFfnGateShexpS && layer.vFfnGateShexp) {
        layer.vFfnGateShexpS = mCreateTensor(cTn(cLlmTensorFfnGateShexp, "scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vFfnDownShexpS && layer.vFfnDownShexp) {
        layer.vFfnDownShexpS = mCreateTensor(cTn(cLlmTensorFfnDownShexp, "scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vFfnUpShexpS && layer.vFfnUpShexp) {
        layer.vFfnUpShexpS = mCreateTensor(cTn(cLlmTensorFfnUpShexp, "scale", vI), { 1 }, cTensorNotRequired);
      }

      // MoE expert weight scales (per-expert, shape {n_expert})
      if (!layer.vFfnGateExpsS && layer.vFfnGateExps) {
        layer.vFfnGateExpsS = mCreateTensor(cTn(cLlmTensorFfnGateExps, "scale", vI), { cNExpert }, cTensorNotRequired);
      }
      if (!layer.vFfnDownExpsS && layer.vFfnDownExps) {
        layer.vFfnDownExpsS = mCreateTensor(cTn(cLlmTensorFfnDownExps, "scale", vI), { cNExpert }, cTensorNotRequired);
      }
      if (!layer.vFfnUpExpsS && layer.vFfnUpExps) {
        layer.vFfnUpExpsS = mCreateTensor(cTn(cLlmTensorFfnUpExps, "scale", vI), { cNExpert }, cTensorNotRequired);
      }

      // recurrent / linear-attention weight scales (per-tensor, shape {1})
      if (!layer.vSsmInS && layer.vSsmIn) {
        layer.vSsmInS = mCreateTensor(cTn(cLlmTensorSsmIn, "scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vSsmOutS && layer.vSsmOut) {
        layer.vSsmOutS = mCreateTensor(cTn(cLlmTensorSsmOut, "scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vSsmAlphaS && layer.vSsmAlpha) {
        layer.vSsmAlphaS = mCreateTensor(cTn(cLlmTensorSsmAlpha, "scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vSsmBetaS && layer.vSsmBeta) {
        layer.vSsmBetaS = mCreateTensor(cTn(cLlmTensorSsmBeta, "scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vNextn.vEhProjS && layer.vNextn.vEhProj) {
        layer.vNextn.vEhProjS = mCreateTensor(cTn(cLlmTensorNextnEhProj, "scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vNextn.vSharedHeadHeadS && layer.vNextn.vSharedHeadHead) {
        layer.vNextn.vSharedHeadHeadS =
          mCreateTensor(cTn(cLlmTensorNextnSharedHeadHead, "scale", vI), { 1 }, cTensorNotRequired);
      }

      // input scales
      if (!layer.vWqInS && layer.vWq) {
        layer.vWqInS = mCreateTensor(cTn(cLlmTensorAttnQ, "input_scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vWkInS && layer.vWk) {
        layer.vWkInS = mCreateTensor(cTn(cLlmTensorAttnK, "input_scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vWvInS && layer.vWv) {
        layer.vWvInS = mCreateTensor(cTn(cLlmTensorAttnV, "input_scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vWoInS && layer.vWo) {
        layer.vWoInS = mCreateTensor(cTn(cLlmTensorAttnOut, "input_scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vWqkvInS && layer.vWqkv) {
        layer.vWqkvInS = mCreateTensor(cTn(cLlmTensorAttnQkv, "input_scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vWqkvGateInS && layer.vWqkvGate) {
        layer.vWqkvGateInS = mCreateTensor(cTn(cLlmTensorAttnGate, "input_scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vFfnGateInS && layer.vFfnGate) {
        layer.vFfnGateInS = mCreateTensor(cTn(cLlmTensorFfnGate, "input_scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vFfnDownInS && layer.vFfnDown) {
        layer.vFfnDownInS = mCreateTensor(cTn(cLlmTensorFfnDown, "input_scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vFfnUpInS && layer.vFfnUp) {
        layer.vFfnUpInS = mCreateTensor(cTn(cLlmTensorFfnUp, "input_scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vFfnGateExpsInS && layer.vFfnGateExps) {
        layer.vFfnGateExpsInS =
          mCreateTensor(cTn(cLlmTensorFfnGateExps, "input_scale", vI), { cNExpert }, cTensorNotRequired);
      }
      if (!layer.vFfnDownExpsInS && layer.vFfnDownExps) {
        layer.vFfnDownExpsInS =
          mCreateTensor(cTn(cLlmTensorFfnDownExps, "input_scale", vI), { cNExpert }, cTensorNotRequired);
      }
      if (!layer.vFfnUpExpsInS && layer.vFfnUpExps) {
        layer.vFfnUpExpsInS =
          mCreateTensor(cTn(cLlmTensorFfnUpExps, "input_scale", vI), { cNExpert }, cTensorNotRequired);
      }
      if (!layer.vFfnGateShexpInS && layer.vFfnGateShexp) {
        layer.vFfnGateShexpInS =
          mCreateTensor(cTn(cLlmTensorFfnGateShexp, "input_scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vFfnDownShexpInS && layer.vFfnDownShexp) {
        layer.vFfnDownShexpInS =
          mCreateTensor(cTn(cLlmTensorFfnDownShexp, "input_scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vFfnUpShexpInS && layer.vFfnUpShexp) {
        layer.vFfnUpShexpInS = mCreateTensor(cTn(cLlmTensorFfnUpShexp, "input_scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vSsmInInS && layer.vSsmIn) {
        layer.vSsmInInS = mCreateTensor(cTn(cLlmTensorSsmIn, "input_scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vSsmOutInS && layer.vSsmOut) {
        layer.vSsmOutInS = mCreateTensor(cTn(cLlmTensorSsmOut, "input_scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vSsmAlphaInS && layer.vSsmAlpha) {
        layer.vSsmAlphaInS = mCreateTensor(cTn(cLlmTensorSsmAlpha, "input_scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vSsmBetaInS && layer.vSsmBeta) {
        layer.vSsmBetaInS = mCreateTensor(cTn(cLlmTensorSsmBeta, "input_scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vNextn.vEhProjInS && layer.vNextn.vEhProj) {
        layer.vNextn.vEhProjInS =
          mCreateTensor(cTn(cLlmTensorNextnEhProj, "input_scale", vI), { 1 }, cTensorNotRequired);
      }
      if (!layer.vNextn.vSharedHeadHeadInS && layer.vNextn.vSharedHeadHead) {
        layer.vNextn.vSharedHeadHeadInS =
          mCreateTensor(cTn(cLlmTensorNextnSharedHeadHead, "input_scale", vI), { 1 }, cTensorNotRequired);
      }
    }
    // output scales
    if (vOutput && vOutput->type == GGML_TYPE_NVFP4) {
      // weight scale
      if (!vOutputS) {
        vOutputS = mCreateTensor(cTn(cLlmTensorOutput, "scale"), { 1 }, cTensorNotRequired);
      }
      // input scale
      if (!vOutputInS) {
        vOutputInS = mCreateTensor(cTn(cLlmTensorOutput, "input_scale"), { 1 }, cTensorNotRequired);
      }
    }
  }

  pMl.mDoneGettingTensors();

  if (vPerLayerTokEmbd && pMl.vLazy.has(vPerLayerTokEmbd)) {
    DGS_LOG_INFO("%s: enabling prefetch for '%s'\n", __func__, vPerLayerTokEmbd->name);

    sCanPrefetch.insert(vPerLayerTokEmbd);
  }

  // Tied NVFP4 output is valid when no separate LM-head scale tensors are present.
  // If sidecar scales exist, the output weight must be an actual output tensor.
  GGML_ASSERT(!(vOutput && vTokEmbd && strcmp(vOutput->name, vTokEmbd->name) == 0 && vOutput->type == GGML_TYPE_NVFP4 &&
                (vOutputS || vOutputInS)));
  // populate tensors_by_name
  for (auto & [_, ctx_ptr] : pMl.dCtxMap) {
    for (auto * vCur = ggml_get_first_tensor(ctx_ptr.get()); vCur != NULL;
         vCur        = ggml_get_next_tensor(ctx_ptr.get(), vCur)) {
      lTensorsByName.emplace_back(ggml_get_name(vCur), vCur);
    }
  }

  // per-tensor activation precision policy
  vPrecPolicy.mLoad(pMl, *this);

  pMl.mInitMappings(true, cUseMlock ? &vPimpl->lMlockMmaps : nullptr);
  vPimpl->lMappings.reserve(pMl.lMappings.size());

  // create the backend buffers
  std::vector<std::pair<ggml_context *, dgs_buf_map>> dCtxBufMaps;
  dCtxBufMaps.reserve(pMl.dCtxMap.size());

  // Ensure we have enough capacity for the maximum backend buffer we will potentially create
  const size_t cNMaxBackendBuffer = pMl.dCtxMap.size() * pMl.lFiles.size();
  vPimpl->lCtxsBufs.reserve(cNMaxBackendBuffer);

  for (auto & [ctx_key, ctx_ptr] : pMl.dCtxMap) {
    ggml_backend_buffer_type_t vBuft = ctx_key.vBuft;
    ggml_context *             vCtx  = ctx_ptr.get();

    // skip contexts without tensors
    if (ggml_get_first_tensor(vCtx) == nullptr) {
      continue;
    }

    dgs_buf_map dBufMap;
    dBufMap.reserve(cNMaxBackendBuffer);

    // check if it is possible to use buffer_from_host_ptr with this buffer type
    ggml_backend_dev_t vDev = ggml_backend_buft_get_device(vBuft);
    if (!vDev) {
      // FIXME: workaround for CPU backend buft having a NULL device
      vDev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
      if (!vDev) {
        throw std::runtime_error(format("%s: no CPU backend found", __func__));
      }
    }
    ggml_backend_dev_props vProps;
    ggml_backend_dev_get_props(vDev, &vProps);
    bool vBufferFromHostPtrSupported = vProps.caps.buffer_from_host_ptr;
    bool vIsDefaultBuft              = vBuft == ggml_backend_dev_buffer_type(vDev);

    std::vector<ggml_backend_buffer_ptr> lBufs;

    // a lazy context is mapped whatever the load mode, but the memory-fit pass maps nothing
    const bool cIsLazyMapped = ctx_key.vLazy && !pMl.vNoAlloc;

    if ((pMl.vUseMmap || cIsLazyMapped) && cUseMmapBuffer && vBufferFromHostPtrSupported && vIsDefaultBuft) {
      GGML_ASSERT(!pMl.vNoAlloc);
      for (uint32_t vIdx = 0; vIdx < pMl.lFiles.size(); vIdx++) {
        // only the mmap region containing the tensors in the model is mapped to the backend buffer
        // this is important for metal with apple silicon: if the entire model could be mapped to a metal buffer,
        //     then we could just use metal for all layers
        // this allows using partial offloading when the model size exceeds the metal buffer size, but not the RAM size
        void * vAddr = nullptr;
        size_t vFirst, vLast;  // NOLINT
        pMl.mGetMappingRange(&vFirst, &vLast, &vAddr, vIdx, vCtx);
        if (vFirst >= vLast) {
          continue;
        }
        const size_t          cMaxSize = ggml_get_max_tensor_size(vCtx);
        ggml_backend_buffer_t vBuf =
          ggml_backend_dev_buffer_from_host_ptr(vDev, (char *) vAddr + vFirst, vLast - vFirst, cMaxSize);
        if (vBuf == nullptr) {
          throw std::runtime_error(format("unable to allocate %s buffer", ggml_backend_buft_name(vBuft)));
        }
        lBufs.emplace_back(vBuf);
        dBufMap.emplace(vIdx, vBuf);
      }
    } else {
      ggml_backend_buffer_t vBuf;
      if (pMl.vNoAlloc) {
        vBuf = ggml_backend_buft_alloc_buffer(vBuft, /*size =*/0);  // dummy buffer
        for (ggml_tensor * vT = ggml_get_first_tensor(vCtx); vT != nullptr; vT = ggml_get_next_tensor(vCtx, vT)) {
          vT->buffer = vBuf;  // set dummy buffer for weights so that the backend scheduler won't try to allocate them
        }
      } else {
        vBuf = ggml_backend_alloc_ctx_tensors_from_buft(vCtx, vBuft);  // real buffer
      }
      if (vBuf == nullptr) {
        throw std::runtime_error(format("unable to allocate %s buffer", ggml_backend_buft_name(vBuft)));
      }
      if (cUseMlock && ggml_backend_buffer_is_host(vBuf)) {
        vPimpl->lMlockBufs.emplace_back(new dgs_mlock);
        auto & vMlockBuf = vPimpl->lMlockBufs.back();
        vMlockBuf->init(ggml_backend_buffer_get_base(vBuf));
        vMlockBuf->mGrowTo(ggml_backend_buffer_get_size(vBuf));
      }
      lBufs.emplace_back(vBuf);
      for (uint32_t vIdx = 0; vIdx < pMl.lFiles.size(); vIdx++) {
        dBufMap.emplace(vIdx, vBuf);
      }
    }

    for (auto & vBuf : lBufs) {
      // indicate that this buffer contains weights
      // this is used by ggml_backend_sched to improve op scheduling: ops that use a weight are preferably scheduled to the backend that contains the weight
      ggml_backend_buffer_set_usage(vBuf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    vPimpl->lCtxsBufs.emplace_back(std::move(ctx_ptr), std::move(lBufs));

    dCtxBufMaps.emplace_back(vCtx, dBufMap);
  }

  if (fDgsSupportsGpuOffload()) {
    const int cNGpu = std::min(cNGpuLayers, cNLayerAll);

    int vNRepeating = cNGpu;
    if (vNRepeating > 0) {
      DGS_LOG_INFO("%s: offloading output layer to GPU\n", __func__);
      vNRepeating--;
    }
    DGS_LOG_INFO("%s: offloading %d repeating layers to GPU\n", __func__, vNRepeating);

    const int cMaxBackendSupportedLayers = cNLayerAll + 1;
    const int cMaxOffloadableLayers      = cNLayerAll + 1;

    DGS_LOG_INFO("%s: offloaded %d/%d layers to GPU\n", __func__, std::min(cNGpuLayers, cMaxOffloadableLayers),
                 cMaxBackendSupportedLayers);
  }

  // print memory requirements per buffer type
  for (auto & [_, bufs] : vPimpl->lCtxsBufs) {
    for (auto & vBuf : bufs) {
      DGS_LOG_INFO("%s: %12s model buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(vBuf.get()),
                   ggml_backend_buffer_get_size(vBuf.get()) / 1024.0 / 1024.0);
    }
  }

  if (pMl.vNoAlloc) {
    return true;
  }

  // without mmap, load non-host buffers first: their tensors go through a staging buffer, which is cheapest while the fewest weights are resident
  if (!pMl.vUseMmap) {
    std::stable_partition(dCtxBufMaps.begin(), dCtxBufMaps.end(), [](const auto & pCtxBufMap) {
      const auto & vBufMap = pCtxBufMap.second;
      return !vBufMap.empty() && !ggml_backend_buffer_is_host(vBufMap.begin()->second);
    });
  }

  // load tensor data
  for (auto & [ctx, buf_map] : dCtxBufMaps) {
    if (!pMl.mLoadAllData(ctx, buf_map, cUseMlock ? &vPimpl->lMlockMmaps : NULL, vParams.vProgressCallback,
                          vParams.vProgressCallbackUserData)) {
      return false;
    }
  }

  if (cUseMmapBuffer) {
    for (auto & vMapping : pMl.lMappings) {
      vPimpl->lMappings.emplace_back(std::move(vMapping));
    }
  }

  mLoadTensorsHadamard();

  return true;
}

// read and check the prism.hadamard metadata, and record the folded weights (load_tensors_hadamard makes the tensors)
void dgs_model_base::mLoadHparamsHadamard(dgs_model_loader & pMl) {
  uint32_t vHadamardVersion = 0;
  pMl.get_key(cLlmKvPrismHadamardTiedOutput, vHdmd.vTiedOutput, false);
  if (pMl.get_key(cLlmKvPrismHadamardVersion, vHadamardVersion, false)) {
    if (vHadamardVersion != 1 && vHadamardVersion != 2) {
      throw std::runtime_error(format("unsupported prism.hadamard.version: %u", vHadamardVersion));
    }

    if ((vHadamardVersion == 2) != vHdmd.vTiedOutput) {
      throw std::runtime_error("prism.hadamard version 2 requires tied_output=true; version 1 forbids it");
    }
    if (vHdmd.vTiedOutput && pMl.mGetWeight("output.weight")) {
      throw std::runtime_error("prism.hadamard.tied_output requires output.weight to be absent");
    }

    uint32_t                 vBlockSize = 0;
    std::string              transform;
    std::string              vAxis;
    std::string              vSignMode;
    std::vector<std::string> lWeightNames;

    pMl.get_key(cLlmKvPrismHadamardBlockSize, vBlockSize);
    pMl.get_key(cLlmKvPrismHadamardTransform, transform);
    pMl.get_key(cLlmKvPrismHadamardAxis, vAxis);
    pMl.get_key(cLlmKvPrismHadamardSignMode, vSignMode);
    pMl.get_arr(cLlmKvPrismHadamardWeightNames, lWeightNames);

    if (vBlockSize == 0 || (vBlockSize & (vBlockSize - 1)) != 0) {
      throw std::runtime_error(format("invalid prism.hadamard.block_size: %u", vBlockSize));
    }
    if (transform != "normalized-sylvester-walsh-hadamard") {
      throw std::runtime_error(format("unsupported prism.hadamard.transform: %s", transform.c_str()));
    }
    if (vAxis != "input-last-dimension") {
      throw std::runtime_error(format("unsupported prism.hadamard.axis: %s", vAxis.c_str()));
    }
    if (vSignMode != "identity" && vSignMode != "explicit") {
      throw std::runtime_error(format("unsupported prism.hadamard.sign_mode: %s", vSignMode.c_str()));
    }
    if (lWeightNames.empty()) {
      throw std::runtime_error("prism.hadamard.weight_names is empty");
    }

    if (vSignMode == "explicit") {
      std::vector<int32_t> lSignWidths;
      std::vector<int32_t> lSignValues;
      pMl.get_arr(cLlmKvPrismHadamardSignWidths, lSignWidths);
      pMl.get_arr(cLlmKvPrismHadamardSignValues, lSignValues);
      // explicit mode with no widths gives an empty sign table, which acts as identity and changes the model
      if (lSignWidths.empty()) {
        throw std::runtime_error("prism.hadamard.sign_mode is explicit but sign_widths is empty");
      }
      size_t vOff = 0;
      for (const int32_t cWidth : lSignWidths) {
        if (cWidth <= 0 || (uint32_t) cWidth % vBlockSize != 0 || vOff + cWidth > lSignValues.size()) {
          throw std::runtime_error(format("invalid prism.hadamard sign width: %d", cWidth));
        }
        if (vHdmd.dSignData.count(cWidth)) {
          throw std::runtime_error(format("duplicate prism.hadamard sign width: %d", cWidth));
        }
        auto & lVec = vHdmd.dSignData[cWidth];
        lVec.assign(lSignValues.begin() + vOff, lSignValues.begin() + vOff + cWidth);
        for (const int32_t cV : lVec) {
          if (cV != 1 && cV != -1) {
            throw std::runtime_error("prism.hadamard sign values must be +/-1");
          }
        }
        vOff += cWidth;
      }
      if (vOff != lSignValues.size()) {
        throw std::runtime_error("prism.hadamard.sign_values length mismatch");
      }
    }

    pMl.get_key(cLlmKvPrismHadamardGdnVGrouped, vHdmd.vGdnVGrouped, false);

    // only build_lora_mm/build_lora_mm_id apply the transform: refuse archs and tensor kinds that can skip them
    switch (vArch) {
      case cLlmArchQwen35:

        break;
      default:
        throw std::runtime_error(
          format("prism.hadamard: arch '%s' is not verified to apply the activation transform to all folded weights",
                 fLlmArchName(vArch)));
    }

    // a folded weight W_f = W*D*H is the weight W with the +1/-1 signs D and the normalized block Hadamard H folded in
    // H*H = I and D*D = I, so W*x = W_f*(H*(D*x)): the graph applies D, then H, to the matmul input
    const auto cIsFoldableWeight = [](const std::string & name) {
      static const char * aKinds[] = {
        "attn_q",           "attn_k",         "attn_v",       "attn_qkv",       "attn_gate",   "attn_output",
        "ffn_gate",         "ffn_up",         "ffn_down",     "ffn_gate_exps",  "ffn_up_exps", "ffn_down_exps",
        "ffn_gate_up_exps", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp", "ssm_out",
      };
      if (name == "output.weight") {
        return true;  // the output head is built through build_lora_mm in every arch
      }
      if (name.compare(0, 4, "blk.") != 0) {
        return false;
      }
      size_t vPos = 4;
      while (vPos < name.size() && isdigit((unsigned char) name[vPos])) {
        vPos++;
      }
      if (vPos == 4 || vPos >= name.size() || name[vPos] != '.') {
        return false;
      }
      vPos++;
      for (const char * vKind : aKinds) {
        const std::string cSuffix = std::string(vKind) + ".weight";
        if (name.compare(vPos, std::string::npos, cSuffix) == 0) {
          return true;
        }
      }
      return false;
    };

    for (const auto & vWeightName : lWeightNames) {
      if (!cIsFoldableWeight(vWeightName)) {
        throw std::runtime_error(
          format("prism.hadamard: weight '%s' is not on a verified Hadamard-aware matmul path", vWeightName.c_str()));
      }
      if (!vHdmd.dWeightBlocks.emplace(vWeightName, vBlockSize).second) {
        throw std::runtime_error(format("duplicate prism.hadamard weight: %s", vWeightName.c_str()));
      }
    }

    // tables read by row lookup store latent rows: the inverse transform goes on the lookup result
    std::vector<std::string> lInverseNames;
    pMl.get_arr(cLlmKvPrismHadamardInverseWeightNames, lInverseNames, false);
    for (const auto & name : lInverseNames) {
      // the graph applies the inverse only after the token-embedding lookup, other latent tables stay rotated
      if (name != "token_embd.weight") {
        throw std::runtime_error(
          format("prism.hadamard: weight '%s' is not a verified inverse-after-lookup table", name.c_str()));
      }
      if (vHdmd.dWeightBlocks.count(name) || !vHdmd.dInverseBlocks.emplace(name, vBlockSize).second) {
        throw std::runtime_error(format("duplicate prism.hadamard inverse weight: %s", name.c_str()));
      }
    }
  }

  if (vHdmd.vTiedOutput) {
    if (vHadamardVersion != 2) {
      throw std::runtime_error("prism.hadamard.tied_output requires version 2");
    }
    const auto cIt = vHdmd.dInverseBlocks.find("token_embd.weight");
    if (cIt == vHdmd.dInverseBlocks.end()) {
      throw std::runtime_error("prism.hadamard.tied_output requires a latent token embedding");
    }
    vHdmd.dWeightBlocks.emplace("token_embd.weight", cIt->second);
  } else if (vHdmd.dInverseBlocks.count("token_embd.weight") && !pMl.mGetWeight("output.weight")) {
    throw std::runtime_error("a tied Hadamard output requires version 2 and tied_output=true");
  }
}

// make one Hadamard matrix per block size and one sign vector per width (the GGUF does not contain them)
// each tensor goes on the buffer type of its weights (never a CPU extra type), and each transform goes in hdmd.rot or hdmd.inv
void dgs_model_base::mLoadTensorsHadamard() {
  if (vHdmd.dWeightBlocks.empty() && vHdmd.dInverseBlocks.empty()) {
    return;
  }

  struct hadamard_rotation {
    uint32_t                   vBlockSize;
    ggml_backend_buffer_type_t vBuft;
    ggml_tensor *              vTensor;
  };

  std::vector<hadamard_rotation>                                           lRotations;
  std::map<std::pair<uint32_t, ggml_backend_buffer_type_t>, ggml_tensor *> dSignTensors;

  const std::pair<const std::unordered_map<std::string, uint32_t> *, dgs_hadamard_rotations *> dGroups[] = {
    { &vHdmd.dWeightBlocks,  &vHdmd.dRot },
    { &vHdmd.dInverseBlocks, &vHdmd.dInv },
  };
  // inverse transforms use the buffer type of the forward rotations, not the host type of a CPU-mapped table (no PCIe round trip per token)
  ggml_backend_buffer_type_t vPreferredBuft = nullptr;

  for (const auto & [blocks, target] : dGroups) {
    for (const auto & vEntry : *blocks) {
      const std::string & vWeightName = vEntry.first;
      const uint32_t      cBlockSize  = vEntry.second;
      const ggml_tensor * vWeight     = mGetTensor(vWeightName.c_str());
      if (vHdmd.vTiedOutput && vWeightName == "token_embd.weight") {
        vWeight = target == &vHdmd.dRot ? vOutput : vTokEmbd;
        if (!vWeight || strcmp(vWeight->name, "token_embd.weight") != 0) {
          throw std::runtime_error("prism.hadamard.tied_output is not bound to the token embedding");
        }
      }
      if (vWeight == nullptr) {
        throw std::runtime_error(format("prism.hadamard weight not found: %s", vWeightName.c_str()));
      }
      if (vWeight->ne[0] % cBlockSize != 0) {
        throw std::runtime_error(format("prism.hadamard block size %u does not divide input dimension %lld for %s",
                                        cBlockSize, (long long) vWeight->ne[0], vWeightName.c_str()));
      }
      if (vWeight->buffer == nullptr) {
        throw std::runtime_error(format("prism.hadamard weight has no buffer: %s", vWeightName.c_str()));
      }

      ggml_backend_buffer_type_t vBuft = ggml_backend_buffer_get_type(vWeight->buffer);
      // CPU extra buffer types (e.g. CPU_REPACK) only accept tensors they can repack
      if (ggml_backend_dev_t vDev = ggml_backend_buft_get_device(vBuft)) {
        if (ggml_backend_dev_type(vDev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
          vBuft = ggml_backend_dev_buffer_type(vDev);
        }
      }
      if (target == &vHdmd.dRot) {
        vPreferredBuft = vBuft;
      } else if (vPreferredBuft) {
        vBuft = vPreferredBuft;
      }
      auto lIt =
        std::find_if(lRotations.begin(), lRotations.end(), [cBlockSize, vBuft](const hadamard_rotation & pRotation) {
          return pRotation.vBlockSize == cBlockSize && pRotation.vBuft == vBuft;
        });

      if (lIt == lRotations.end()) {
        ggml_init_params vParams = {
          /*.mem_size   =*/ggml_tensor_overhead(),
          /*.mem_buffer =*/NULL,
          /*.no_alloc   =*/true,
        };
        ggml_context_ptr vCtx{ ggml_init(vParams) };
        if (!vCtx) {
          throw std::runtime_error("failed to create Hadamard rotation context");
        }

        ggml_tensor * vRotation = ggml_new_tensor_2d(vCtx.get(), GGML_TYPE_F32, cBlockSize, cBlockSize);
        char          aRotationName[GGML_MAX_NAME];
        snprintf(aRotationName, sizeof(aRotationName), "prism.hadamard.%u", cBlockSize);
        ggml_set_name(vRotation, aRotationName);

        ggml_backend_buffer_ptr vBuffer{ ggml_backend_alloc_ctx_tensors_from_buft(vCtx.get(), vBuft) };
        if (!vBuffer) {
          throw std::runtime_error(
            format("unable to allocate %s Hadamard rotation buffer", ggml_backend_buft_name(vBuft)));
        }
        ggml_backend_buffer_set_usage(vBuffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        std::vector<float> data((size_t) cBlockSize * cBlockSize);
        const float        cScale = 1.0f / sqrtf((float) cBlockSize);
        for (uint32_t vRow = 0; vRow < cBlockSize; ++vRow) {
          for (uint32_t vCol = 0; vCol < cBlockSize; ++vCol) {
            uint32_t vParity = vRow & vCol;
            vParity ^= vParity >> 16;
            vParity ^= vParity >> 8;
            vParity ^= vParity >> 4;
            vParity ^= vParity >> 2;
            vParity ^= vParity >> 1;
            data[(size_t) vRow * cBlockSize + vCol] = (vParity & 1) ? -cScale : cScale;
          }
        }
        ggml_backend_tensor_set(vRotation, data.data(), 0, data.size() * sizeof(float));

        std::vector<ggml_backend_buffer_ptr> lBuffers;
        lBuffers.emplace_back(std::move(vBuffer));
        vPimpl->lCtxsBufs.emplace_back(std::move(vCtx), std::move(lBuffers));
        lRotations.push_back({ cBlockSize, vBuft, vRotation });
        lIt = std::prev(lRotations.end());
      }

      ggml_tensor * vSignTensor = nullptr;
      if (!vHdmd.dSignData.empty()) {
        const uint32_t cWidth = (uint32_t) vWeight->ne[0];
        const auto     lSd    = vHdmd.dSignData.find(cWidth);
        if (lSd == vHdmd.dSignData.end()) {
          throw std::runtime_error(
            format("prism.hadamard has no sign vector for width %u (%s)", cWidth, vWeightName.c_str()));
        }
        const auto cKey = std::make_pair(cWidth, vBuft);
        auto       vSt  = dSignTensors.find(cKey);
        if (vSt == dSignTensors.end()) {
          ggml_init_params vParams = {
            /*.mem_size   =*/ggml_tensor_overhead(),
            /*.mem_buffer =*/NULL,
            /*.no_alloc   =*/true,
          };
          ggml_context_ptr vCtx{ ggml_init(vParams) };
          if (!vCtx) {
            throw std::runtime_error("failed to create Hadamard sign context");
          }

          ggml_tensor * vSigns = ggml_new_tensor_1d(vCtx.get(), GGML_TYPE_F32, cWidth);
          char          aSignName[GGML_MAX_NAME];
          snprintf(aSignName, sizeof(aSignName), "prism.hadamard.signs.%u", cWidth);
          ggml_set_name(vSigns, aSignName);

          ggml_backend_buffer_ptr vBuffer{ ggml_backend_alloc_ctx_tensors_from_buft(vCtx.get(), vBuft) };
          if (!vBuffer) {
            throw std::runtime_error(
              format("unable to allocate %s Hadamard sign buffer", ggml_backend_buft_name(vBuft)));
          }
          ggml_backend_buffer_set_usage(vBuffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

          std::vector<float> data(cWidth);
          for (uint32_t vI = 0; vI < cWidth; ++vI) {
            data[vI] = (float) lSd->second[vI];
          }
          ggml_backend_tensor_set(vSigns, data.data(), 0, data.size() * sizeof(float));

          std::vector<ggml_backend_buffer_ptr> lBuffers;
          lBuffers.emplace_back(std::move(vBuffer));
          vPimpl->lCtxsBufs.emplace_back(std::move(vCtx), std::move(lBuffers));
          vSt = dSignTensors.emplace(cKey, vSigns).first;
        }
        vSignTensor = vSt->second;
      }

      dgs_hadamard_transform transform{ lIt->vTensor, vSignTensor };
      // the GDN output projection reads its value heads in tiled order, see set_gdn_v_perm
      if (vHdmd.vGdnVGrouped && vWeightName.find(".ssm_out.") != std::string::npos &&
          !transform.mSetGdnVPerm(vWeight->ne[0], hparams.vSsmDtRank, hparams.vSsmNGroup)) {
        throw std::runtime_error(format("prism.hadamard: bad GDN head geometry for %s", vWeightName.c_str()));
      }
      target->emplace(vWeight, transform);
    }
  }

  DGS_LOG_INFO(
    "%s: loaded %zu Hadamard-folded weight(s) (%zu inverse-lookup) using %zu rotation(s) and %zu sign vector(s)\n",
    __func__, vHdmd.dRot.size() + vHdmd.dInv.size(), vHdmd.dInv.size(), lRotations.size(), dSignTensors.size());
}

ggml_tensor * dgs_model_base::mCreateTensor(dgs_model_loader &                     pMl,
                                            const LLM_TN_IMPL &                    pTn,
                                            const std::initializer_list<int64_t> & pNe,
                                            int                                    pFlags) {
  const buft_list_t * lBuftListLayer = nullptr;
  if (pTn.cBid != -1) {
    // blocks that are not model layers (e.g. the blocks of a head) go with the output
    lBuftListLayer = (size_t) pTn.cBid < vPimpl->lDevLayer.size() ? vPimpl->lDevLayer.at(pTn.cBid).lBuftList :
                                                                    vPimpl->vDevOutput.lBuftList;
  }
  return pMl.mCreateTensor(hparams, &vPimpl->lCpuBuftList, vPimpl->vDevInput.lBuftList, vPimpl->vDevOutput.lBuftList,
                           lBuftListLayer, pTn, pNe, pFlags);
}

std::string dgs_model::mArchName() const {
  return fLlmArchName(vArch);
}

std::string dgs_model::mTypeName() const {
  return fLlmTypeName(type);
}

std::string dgs_model::mDesc() const {
  return vPimpl->vDescStr;
}

dgs_ftype dgs_model::mFtype() const {
  return vPimpl->vFtype;
}

size_t dgs_model::size() const {
  return vPimpl->vNBytes;
}

size_t dgs_model::mNTensors() const {
  return lTensorsByName.size();
}

size_t dgs_model::mNDevices() const {
  return lDevices.size();
}

const float * dgs_model::mTensorSplit() const {
  return vParams.vTensorSplit;
}

uint32_t dgs_model::mNGpuLayers() const {
  // note: plus 1 for the "output" layer
  return vParams.vNGpuLayers >= 0 ? vParams.vNGpuLayers : hparams.n_layer_all + 1;
}

dgs_split_mode dgs_model::mSplitMode() const {
  return vParams.vSplitMode;
}

std::map<ggml_backend_buffer_type_t, size_t> dgs_model::mMemoryBreakdown() const {
  std::map<ggml_backend_buffer_type_t, size_t> dRet;
  for (const auto & [ctx, bufs] : vPimpl->lCtxsBufs) {
    if (hparams.vNoAlloc) {
      GGML_ASSERT(bufs.size() == 1);
      ggml_backend_buffer_t vBuf = bufs[0].get();
      GGML_ASSERT(ggml_backend_buffer_get_base(vBuf) == nullptr);
      ggml_backend_buffer_type_t vBuft = ggml_backend_buffer_get_type(vBuf);
      dRet[vBuft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), vBuft);
    } else {
      for (const auto & vBuf : bufs) {
        // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
        dRet[ggml_backend_buffer_get_type(vBuf.get())] += ggml_backend_buffer_get_size(vBuf.get());
      }
    }
  }
  return dRet;
}

uint64_t dgs_model::mNElements() const {
  return vPimpl->vNElements;
}

void dgs_model::mPrintInfo() const {
  const std::string cRopeScalingType = fDgsRopeScalingTypeName(hparams.vRopeScalingTypeTrain);

  auto vPrintF = [](const std::function<int32_t(uint32_t)> & pF, uint32_t n) {
    bool vIsVar = false;

    std::vector<int32_t> lV;
    for (uint32_t vI = 0; vI < n; ++vI) {
      lV.push_back(pF(vI));
      if (lV[vI] != lV[0]) {
        vIsVar = true;
      }
    }

    std::stringstream vSs;

    if (vIsVar) {
      vSs << "[";
      for (uint32_t vI = 0; vI < n; ++vI) {
        vSs << lV[vI];
        if (vI < n - 1) {
          vSs << ", ";
        }
      }
      vSs << "]";
    } else {
      vSs << lV[0];
    }

    return vSs.str();
  };

  // hparams
  DGS_LOG_INFO("%s: arch                  = %s\n", __func__, mArchName().c_str());
  DGS_LOG_INFO("%s: vocab_only            = %d\n", __func__, hparams.vVocabOnly);
  DGS_LOG_INFO("%s: no_alloc              = %d\n", __func__, hparams.vNoAlloc);

  if (!hparams.vVocabOnly) {
    DGS_LOG_INFO("%s: n_ctx_train           = %u\n", __func__, hparams.n_ctx_train);
    DGS_LOG_INFO("%s: n_embd_inp            = %u\n", __func__, hparams.mNEmbdInp());
    DGS_LOG_INFO("%s: n_embd                = %u\n", __func__, hparams.n_embd);
    DGS_LOG_INFO("%s: n_embd_out            = %u\n", __func__, hparams.mNEmbdOut());
    DGS_LOG_INFO("%s: n_layer               = %u\n", __func__, hparams.n_layer());
    DGS_LOG_INFO("%s: n_layer_all           = %u\n", __func__, hparams.n_layer_all);
    DGS_LOG_INFO("%s: n_head                = %s\n", __func__,
                 vPrintF([&](uint32_t pIl) { return hparams.n_head(pIl); }, hparams.n_layer_all).c_str());
    DGS_LOG_INFO("%s: n_head_kv             = %s\n", __func__,
                 vPrintF([&](uint32_t pIl) { return hparams.n_head_kv(pIl); }, hparams.n_layer_all).c_str());
    DGS_LOG_INFO("%s: n_rot                 = %u\n", __func__, hparams.vNRotFull);
    DGS_LOG_INFO("%s: n_swa                 = %u\n", __func__, hparams.vNSwa);
    DGS_LOG_INFO("%s: is_swa_any            = %u\n", __func__, hparams.mIsSwaAny());
    DGS_LOG_INFO("%s: non_causal_type       = %d\n", __func__, hparams.vNonCausalType);
    DGS_LOG_INFO("%s: n_embd_head_k         = %u\n", __func__, hparams.vNEmbdHeadKFull);
    DGS_LOG_INFO("%s: n_embd_head_v         = %u\n", __func__, hparams.vNEmbdHeadVFull);
    DGS_LOG_INFO("%s: n_gqa                 = %s\n", __func__,
                 vPrintF([&](uint32_t pIl) { return hparams.mNGqa(pIl); }, hparams.n_layer_all).c_str());
    DGS_LOG_INFO("%s: n_embd_k_gqa          = %s\n", __func__,
                 vPrintF([&](uint32_t pIl) { return hparams.n_embd_k_gqa(pIl); }, hparams.n_layer_all).c_str());
    DGS_LOG_INFO("%s: n_embd_v_gqa          = %s\n", __func__,
                 vPrintF([&](uint32_t pIl) { return hparams.n_embd_v_gqa(pIl); }, hparams.n_layer_all).c_str());
    DGS_LOG_INFO("%s: f_norm_eps            = %.1e\n", __func__, hparams.vFNormEps);
    DGS_LOG_INFO("%s: f_norm_rms_eps        = %.1e\n", __func__, hparams.vFNormRmsEps);
    DGS_LOG_INFO("%s: f_clamp_kqv           = %.1e\n", __func__, hparams.vFClampKqv);
    DGS_LOG_INFO("%s: f_max_alibi_bias      = %.1e\n", __func__, hparams.vFMaxAlibiBias);
    DGS_LOG_INFO("%s: f_logit_scale         = %.1e\n", __func__, hparams.vFLogitScale);
    DGS_LOG_INFO("%s: f_attn_scale          = %.1e\n", __func__, hparams.vFAttentionScale);
    DGS_LOG_INFO("%s: f_attn_value_scale    = %.4f\n", __func__, hparams.vFAttnValueScale);
    DGS_LOG_INFO("%s: n_ff                  = %s\n", __func__,
                 vPrintF([&](uint32_t pIl) { return hparams.n_ff(pIl); }, hparams.n_layer_all).c_str());
    DGS_LOG_INFO("%s: n_expert              = %u\n", __func__, hparams.n_expert);
    DGS_LOG_INFO("%s: n_expert_used         = %u\n", __func__, hparams.n_expert_used());
    DGS_LOG_INFO("%s: n_expert_groups       = %d\n", __func__, hparams.vNExpertGroups);
    DGS_LOG_INFO("%s: n_group_used          = %d\n", __func__, hparams.vNGroupUsed);
    DGS_LOG_INFO("%s: causal attn           = %d\n", __func__, hparams.vCausalAttn);
    DGS_LOG_INFO("%s: pooling type          = %d\n", __func__, hparams.vPoolingType);
    DGS_LOG_INFO("%s: rope type             = %d\n", __func__, hparams.vRopeType);
    DGS_LOG_INFO("%s: rope scaling          = %s\n", __func__, cRopeScalingType.c_str());
    DGS_LOG_INFO("%s: freq_base_train       = %.1f\n", __func__, hparams.vRopeFreqBaseTrain);
    DGS_LOG_INFO("%s: freq_scale_train      = %g\n", __func__, hparams.vRopeFreqScaleTrain);
    if (hparams.vSwaType != cDgsSwaTypeNone) {
      DGS_LOG_INFO("%s: freq_base_swa         = %.1f\n", __func__, hparams.vRopeFreqBaseTrainSwa);
      DGS_LOG_INFO("%s: freq_scale_swa        = %g\n", __func__, hparams.vRopeFreqScaleTrainSwa);
      DGS_LOG_INFO("%s: n_embd_head_k_swa     = %u\n", __func__, hparams.vNEmbdHeadKSwa);
      DGS_LOG_INFO("%s: n_embd_head_v_swa     = %u\n", __func__, hparams.vNEmbdHeadVSwa);
      DGS_LOG_INFO("%s: n_rot_swa             = %u\n", __func__, hparams.vNRotSwa);
    }
    DGS_LOG_INFO("%s: n_ctx_orig_yarn       = %u\n", __func__, hparams.vNCtxOrigYarn);
    DGS_LOG_INFO("%s: rope_yarn_log_mul     = %.4f\n", __func__, hparams.vRopeYarnLogMul);
    DGS_LOG_INFO("%s: rope_finetuned        = %s\n", __func__, hparams.vRopeFinetuned ? "yes" : "unknown");
    if (vArch == cLlmArchGranite &&
        std::any_of(hparams.aDeepstackMappingArr.begin(), hparams.aDeepstackMappingArr.end(),
                    [](const auto & pEntry) { return pEntry >= 0; })) {
      DGS_LOG_INFO(
        "%s: deepstack_mapping_arr = %s\n", __func__,
        vPrintF([&](uint32_t pIl) { return hparams.aDeepstackMappingArr[pIl]; }, hparams.n_layer_all).c_str());
    }
    // MRoPE (Multi-axis Rotary Position Embedding) sections
    if (const auto & aS = hparams.aRopeSections; aS[0] || aS[1] || aS[2] || aS[3]) {
      DGS_LOG_INFO("%s: mrope sections        = [%d, %d, %d, %d]\n", __func__, aS[0], aS[1], aS[2], aS[3]);
    }
    if (hparams.vRopeSectionOrder != cDgsRopeSectionOrderTyxz) {
      DGS_LOG_INFO("%s: rope section order    = %s\n", __func__,
                   fDgsRopeSectionOrderName(hparams.vRopeSectionOrder).c_str());
    }
    if (!lClassifierLabels.empty()) {
      DGS_LOG_INFO("%s: n_cls_out             = %u\n", __func__, hparams.vNClsOut);

      size_t vI = 0;
      for (const auto & vLabel : lClassifierLabels) {
        DGS_LOG_INFO("%s: cls_label[%2zu]         = %s\n", __func__, vI++, vLabel.c_str());
      }
    }

    if (vArch == cLlmArchMamba || vArch == cLlmArchMamba2 || vArch == cLlmArchJamba || vArch == cLlmArchFalconH1 ||
        vArch == cLlmArchPlamo2 || vArch == cLlmArchGraniteHybrid || vArch == cLlmArchQwen3next ||
        vArch == cLlmArchQwen35 || vArch == cLlmArchQwen35moe || vArch == cLlmArchNemotronH ||
        vArch == cLlmArchNemotronHMoe) {
      DGS_LOG_INFO("%s: ssm_d_conv            = %u\n", __func__, hparams.vSsmDConv);
      DGS_LOG_INFO("%s: ssm_d_inner           = %u\n", __func__, hparams.vSsmDInner);
      DGS_LOG_INFO("%s: ssm_d_state           = %u\n", __func__, hparams.vSsmDState);
      DGS_LOG_INFO("%s: ssm_dt_rank           = %u\n", __func__, hparams.vSsmDtRank);
      DGS_LOG_INFO("%s: ssm_n_group           = %u\n", __func__, hparams.vSsmNGroup);
      DGS_LOG_INFO("%s: ssm_dt_b_c_rms        = %d\n", __func__, hparams.vSsmDtBCRms);
    }

    DGS_LOG_INFO("%s: model type            = %s\n", __func__, mTypeName().c_str());
    if (vPimpl->vNElements >= 1e12) {
      DGS_LOG_INFO("%s: model params          = %.2f T\n", __func__, vPimpl->vNElements * 1e-12);
    } else if (vPimpl->vNElements >= 1e9) {
      DGS_LOG_INFO("%s: model params          = %.2f B\n", __func__, vPimpl->vNElements * 1e-9);
    } else if (vPimpl->vNElements >= 1e6) {
      DGS_LOG_INFO("%s: model params          = %.2f M\n", __func__, vPimpl->vNElements * 1e-6);
    } else {
      DGS_LOG_INFO("%s: model params          = %.2f K\n", __func__, vPimpl->vNElements * 1e-3);
    }

    // general kv
    DGS_LOG_INFO("%s: general.name          = %s\n", __func__, name.c_str());

    if (vArch == cLlmArchDeepseek) {
      DGS_LOG_INFO("%s: n_layer_dense_lead    = %d\n", __func__, hparams.vNLayerDenseLead);
      DGS_LOG_INFO("%s: n_ff_exp              = %d\n", __func__, hparams.mNFfExp());
      DGS_LOG_INFO("%s: n_expert_shared       = %d\n", __func__, hparams.vNExpertShared);
      DGS_LOG_INFO("%s: expert_weights_scale  = %.1f\n", __func__, hparams.vExpertWeightsScale);
    }

    if (vArch == cLlmArchDeepseek2 || vArch == cLlmArchDeepseek2ocr || vArch == cLlmArchDeepseek32 ||
        vArch == cLlmArchGlmDsa || vArch == cLlmArchDots3note || vArch == cLlmArchMistral4 || vArch == cLlmArchHyV4) {
      DGS_LOG_INFO("%s: n_layer_dense_lead    = %d\n", __func__, hparams.vNLayerDenseLead);
      DGS_LOG_INFO("%s: n_lora_q              = %d\n", __func__, hparams.vNLoraQ);
      DGS_LOG_INFO("%s: n_lora_kv             = %d\n", __func__, hparams.vNLoraKv);
      DGS_LOG_INFO("%s: n_embd_head_k_mla     = %d\n", __func__, hparams.mNEmbdHeadKMla());
      DGS_LOG_INFO("%s: n_embd_head_v_mla     = %d\n", __func__, hparams.mNEmbdHeadVMla());
      DGS_LOG_INFO("%s: n_ff_exp              = %d\n", __func__, hparams.mNFfExp());
      DGS_LOG_INFO("%s: n_expert_shared       = %d\n", __func__, hparams.vNExpertShared);
      DGS_LOG_INFO("%s: expert_weights_scale  = %.1f\n", __func__, hparams.vExpertWeightsScale);
      DGS_LOG_INFO("%s: expert_weights_norm   = %d\n", __func__, hparams.vExpertWeightsNorm);
      DGS_LOG_INFO("%s: expert_gating_func    = %s\n", __func__,
                   fDgsExpertGatingFuncName((dgs_expert_gating_func_type) hparams.vExpertGatingFunc));
    }

    if (vArch == cLlmArchQwen2moe) {
      DGS_LOG_INFO("%s: n_ff_exp              = %d\n", __func__, hparams.mNFfExp());
      DGS_LOG_INFO("%s: n_ff_shexp            = %d\n", __func__, hparams.vNFfShexp);
    }

    if (vArch == cLlmArchMellum || vArch == cLlmArchCohere2moe || vArch == cLlmArchQwen3moe ||
        vArch == cLlmArchOpenaiMoe || vArch == cLlmArchQwen3vlmoe || vArch == cLlmArchRnd1) {
      DGS_LOG_INFO("%s: n_ff_exp              = %d\n", __func__, hparams.mNFfExp());
    }

    if (vArch == cLlmArchMinicpm || vArch == cLlmArchGranite || vArch == cLlmArchGraniteMoe ||
        vArch == cLlmArchGraniteHybrid || vArch == cLlmArchGraniteSwitch || vArch == cLlmArchNemotronHMoe) {
      DGS_LOG_INFO("%s: f_embedding_scale     = %f\n", __func__, hparams.vFEmbeddingScale);
      DGS_LOG_INFO("%s: f_residual_scale      = %f\n", __func__, hparams.vFResidualScale);
      DGS_LOG_INFO("%s: f_attention_scale     = %f\n", __func__, hparams.vFAttentionScale);
      DGS_LOG_INFO("%s: n_ff_shexp            = %d\n", __func__, hparams.vNFfShexp);
    }

    if (vArch == cLlmArchBailingmoe) {
      DGS_LOG_INFO("%s: n_layer_dense_lead    = %d\n", __func__, hparams.vNLayerDenseLead);
      DGS_LOG_INFO("%s: n_ff_exp              = %d\n", __func__, hparams.mNFfExp());
      DGS_LOG_INFO("%s: n_expert_shared       = %d\n", __func__, hparams.vNExpertShared);
      DGS_LOG_INFO("%s: expert_weights_scale  = %.1f\n", __func__, hparams.vExpertWeightsScale);
      DGS_LOG_INFO("%s: expert_weights_norm   = %d\n", __func__, hparams.vExpertWeightsNorm);
    }

    if (vArch == cLlmArchBailingmoe2 || vArch == cLlmArchBailingmoe3) {
      DGS_LOG_INFO("%s: n_layer_dense_lead    = %d\n", __func__, hparams.vNLayerDenseLead);
      DGS_LOG_INFO("%s: n_ff_exp              = %d\n", __func__, hparams.mNFfExp());
      DGS_LOG_INFO("%s: n_ff_shexp            = %d\n", __func__, hparams.vNFfShexp);
      DGS_LOG_INFO("%s: n_expert_shared       = %d\n", __func__, hparams.vNExpertShared);
      DGS_LOG_INFO("%s: expert_weights_scale  = %.1f\n", __func__, hparams.vExpertWeightsScale);
      DGS_LOG_INFO("%s: expert_weights_norm   = %d\n", __func__, hparams.vExpertWeightsNorm);
      DGS_LOG_INFO("%s: expert_gating_func    = %s\n", __func__,
                   fDgsExpertGatingFuncName((dgs_expert_gating_func_type) hparams.vExpertGatingFunc));
      DGS_LOG_INFO("%s: n_layer_nextn         = %d\n", __func__, hparams.n_layer_nextn);
    }

    if (vArch == cLlmArchSmallthinker || vArch == cLlmArchLfm2moe) {
      DGS_LOG_INFO("%s: n_ff_exp              = %d\n", __func__, hparams.mNFfExp());
      DGS_LOG_INFO("%s: expert_gating_func    = %s\n", __func__,
                   fDgsExpertGatingFuncName((dgs_expert_gating_func_type) hparams.vExpertGatingFunc));
    }

    if (vArch == cLlmArchGrovemoe) {
      DGS_LOG_INFO("%s: n_ff_exp              = %d\n", __func__, hparams.mNFfExp());
      DGS_LOG_INFO("%s: n_ff_chexp            = %d\n", __func__, hparams.vNFfChexp);
      DGS_LOG_INFO("%s: n_group_experts       = %d\n", __func__, hparams.vNGroupExperts);
      DGS_LOG_INFO("%s: expert_group_scale    = %.2f\n", __func__, hparams.vExpertGroupScale);
    }
  }

  vocab.mPrintInfo();
}

ggml_backend_dev_t dgs_model::mDevLayer(int pIl) const {
  return vPimpl->lDevLayer.at(pIl).vDev;
}

ggml_backend_dev_t dgs_model::mDevOutput() const {
  return vPimpl->vDevOutput.vDev;
}

template <typename F> static bool buft_supported(ggml_backend_buffer_type_t pBuft, ggml_backend_dev_t pDev, F & pFn) {
  ggml_init_params vParams = {
    /*.mem_size   =*/ggml_tensor_overhead() * 8,
    /*.mem_buffer =*/NULL,
    /*.no_alloc   =*/true,
  };

  ggml_context_ptr vCtx{ ggml_init(vParams) };
  if (!vCtx) {
    throw std::runtime_error(format("failed to create ggml context"));
  }

  ggml_backend_buffer_ptr vBuf{ ggml_backend_buft_alloc_buffer(pBuft, 0) };
  ggml_tensor *           vOpTensor = pFn(vCtx.get());
  for (int vI = 0; vI < GGML_MAX_SRC; vI++) {
    if (vOpTensor->src[vI] != nullptr) {
      assert(op_tensor->src[i]->buffer == nullptr);
      vOpTensor->src[vI]->buffer = vBuf.get();
    }
  }

  bool vOpSupported = ggml_backend_dev_supports_op(pDev, vOpTensor);

  return vOpSupported;
}

template <typename F> static ggml_backend_buffer_type_t fSelectBuft(const buft_list_t & pBuftList, const F & pFn) {
  for (const auto & vCur : pBuftList) {
    ggml_backend_dev_t         vCurDev  = vCur.first;
    ggml_backend_buffer_type_t vCurBuft = vCur.second;
    if (buft_supported(vCurBuft, vCurDev, pFn)) {
      return vCurBuft;
    }
  }

  throw std::runtime_error(format("no suitable buffer type found"));
}

ggml_backend_buffer_type_t dgs_model::mSelectBuft(int pIl) const {
  return ::fSelectBuft(*vPimpl->lDevLayer.at(pIl).lBuftList, [&](ggml_context * pCtx) {
    ggml_tensor * vCur      = ggml_new_tensor_1d(pCtx, GGML_TYPE_F32, hparams.n_embd);
    ggml_tensor * vLayerDir = ggml_new_tensor_1d(pCtx, GGML_TYPE_F32, hparams.n_embd);
    return ggml_add(pCtx, vCur, vLayerDir);
  });
}

bool dgs_model::mHasTensorOverrides() const {
  return vPimpl->vHasTensorOverrides;
}

const ggml_tensor * dgs_model::mGetTensor(const char * name) const {
  auto lIt = std::find_if(lTensorsByName.begin(), lTensorsByName.end(),
                          [name](const std::pair<std::string, ggml_tensor *> & pIt) { return pIt.first == name; });
  if (lIt == lTensorsByName.end()) {
    return nullptr;
  }

  return lIt->second;
}

float dgs_model::mGetRopeFreqBase(const dgs_cparams & pCparams, int pIl) const {
  return hparams.mIsSwa(pIl) ? hparams.vRopeFreqBaseTrainSwa : pCparams.vRopeFreqBase;
}

float dgs_model::mGetRopeFreqScale(const dgs_cparams & pCparams, int pIl) const {
  return hparams.mIsSwa(pIl) ? hparams.vRopeFreqScaleTrainSwa : pCparams.vRopeFreqScale;
}

ggml_tensor * dgs_model::mGetRopeFactors(const dgs_cparams & pCparams, int pIl) const {
  const uint32_t cNCtxSeq = pCparams.vNCtxSeq;

  // choose long/short freq factors based on the context size
  if (lLayers[pIl].vRopeFreqs != nullptr) {
    return lLayers[pIl].vRopeFreqs;
  }

  if (cNCtxSeq > hparams.vNCtxOrigYarn) {
    return lLayers[pIl].vRopeLong;
  }

  return lLayers[pIl].vRopeShort;
}

dgs_memory_i * dgs_model::mCreateMemory(const dgs_memory_params & pParams, const dgs_cparams & pCparams) const {
  dgs_memory_i * vRes;

  // the non-causal LFM2 decision graph reads the whole prompt in one batch, nothing is kept
  if (vArch == cLlmArchLfm2 && !hparams.vCausalAttn && hparams.vNLayerDecision > 0) {
    return nullptr;
  }

  switch (vArch) {
    default:
      {
        // Dense MTP heads use a plain attention KV cache instead of the hybrid wrapper.
        const bool cMtpOnHybridQwen =
          pParams.vCtxType == cDgsContextTypeMtp && (vArch == cLlmArchQwen3next || vArch == cLlmArchQwen35 ||
                                                     vArch == cLlmArchQwen35moe || vArch == cLlmArchBailingmoe3);

        const bool cMtpOnHybridNemotron = pParams.vCtxType == cDgsContextTypeMtp && vArch == cLlmArchNemotronHMoe;

        if (fLlmArchIsRecurrent(vArch)) {
          vRes = new dgs_memory_recurrent(*this, GGML_TYPE_F32, GGML_TYPE_F32, pCparams.vOffloadKqv,
                                          std::max((uint32_t) 1, pCparams.vNSeqMax), pCparams.vNSeqMax,
                                          pCparams.vNRsSeq, nullptr);
        } else if (fLlmArchIsHybrid(vArch) && !cMtpOnHybridQwen && !cMtpOnHybridNemotron) {
          // The main difference between hybrid architectures is the
          // layer filters, so pick the right one here
          dgs_memory_hybrid::layer_filter_cb vFilterAttn  = nullptr;
          dgs_memory_hybrid::layer_filter_cb vFilterRecr  = nullptr;
          // only the sparse-attention architectures use dgs_memory_hybrid_idx
          // a null filter_idx means the GGUF has no indexer tensors
          dgs_memory_hybrid::layer_filter_cb vFilterIdx   = nullptr;
          const bool                         cNeedsMemIdx = (vArch == cLlmArchQwen4exp);
          if (vArch == cLlmArchFalconH1) {
            vFilterAttn = [&](uint32_t) {
              return true;
            };
            vFilterRecr = [&](uint32_t) {
              return true;
            };
          } else if (vArch == cLlmArchNemotronH || vArch == cLlmArchNemotronHMoe) {
            vFilterAttn = [&](uint32_t pIl) {
              return !hparams.mIsRecr(pIl) && hparams.n_ff(pIl) == 0;
            };
            vFilterRecr = [&](uint32_t pIl) {
              return hparams.mIsRecr(pIl) && hparams.n_ff(pIl) == 0;
            };
          } else if (vArch == cLlmArchQwen3next || vArch == cLlmArchQwen35 || vArch == cLlmArchQwen35moe ||
                     vArch == cLlmArchQwen4exp || vArch == cLlmArchMinimax01) {
            vFilterAttn = [&](uint32_t pIl) {
              return pIl < hparams.n_layer() && !hparams.mIsRecr(pIl);
            };
            vFilterRecr = [&](uint32_t pIl) {
              return pIl < hparams.n_layer() && hparams.mIsRecr(pIl);
            };

            if (vArch == cLlmArchQwen4exp && hparams.vIndexerHeadSize > 0) {
              // QSA runs on the dense-attention layers only
              vFilterIdx = [&](uint32_t pIl) {
                return pIl < hparams.n_layer() && !hparams.mIsRecr(pIl);
              };
            }

            // the MTP draft context holds the MTP block alone: its attention and indexer, no recurrent layer
            if (vArch == cLlmArchQwen4exp && pParams.vCtxType == cDgsContextTypeMtp) {
              vFilterAttn = [&](uint32_t pIl) {
                return pIl >= hparams.n_layer();
              };
              vFilterRecr = [&](uint32_t) {
                return false;
              };
              if (vFilterIdx) {
                vFilterIdx = [&](uint32_t pIl) {
                  return pIl >= hparams.n_layer();
                };
              }
            }
          }

          if (hparams.vSwaType != cDgsSwaTypeNone) {
            // Use hybrid-iswa for hybrid models with SWA
            vRes = new dgs_memory_hybrid_iswa(
              /* model             */ *this,
              /* attn_type_k       */ pParams.vTypeK,
              /* attn_type_v       */ pParams.vTypeV,
              /* attn_v_trans      */ !pCparams.vFlashAttn,
              /* attn_swa_full     */ pParams.vSwaFull,
              /* attn_kv_size      */ pCparams.vNCtxSeq,
              /* attn_n_ubatch     */ pCparams.vNUbatch,
              /* attn_n_pad        */ 1,
              /* recurrent_type_r  */ GGML_TYPE_F32,
              /* recurrent_type_s  */ GGML_TYPE_F32,
              /* recurrent_rs_size */ std::max((uint32_t) 1, pCparams.vNSeqMax),
              /* n_seq_max         */ pCparams.vNSeqMax,
              /* n_rs_seq          */ pCparams.vNRsSeq,
              /* offload           */ pCparams.vOffloadKqv,
              /* unified           */ pCparams.vKvUnified,
              /* filter_attn       */ std::move(vFilterAttn),
              /* filter_recr       */ std::move(vFilterRecr));
          } else if (cNeedsMemIdx) {
            // sparse attention over a per-token indexer cache, in its own memory type
            vRes = new dgs_memory_hybrid_idx(
              /* model             */ *this,
              /* attn_type_k       */ pParams.vTypeK,
              /* attn_type_v       */ pParams.vTypeV,
              /* attn_v_trans      */ !pCparams.vFlashAttn,
              /* attn_kv_size      */ pCparams.vNCtxSeq,
              /* attn_n_pad        */ 1,
              /* attn_n_swa        */ hparams.vNSwa,
              /* attn_swa_type     */ hparams.vSwaType,
              /* recurrent_type_k  */ GGML_TYPE_F32,
              /* recurrent_type_v  */ GGML_TYPE_F32,
              /* recurrent_kv_size */ std::max((uint32_t) 1, pCparams.vNSeqMax),
              /* n_seq_max         */ pCparams.vNSeqMax,
              /* n_rs_seq          */ pCparams.vNRsSeq,
              /* offload           */ pCparams.vOffloadKqv,
              /* unified           */ pCparams.vKvUnified,
              /* filter_attn       */ std::move(vFilterAttn),
              /* filter_recr       */ std::move(vFilterRecr),
              /* filter_idx        */ std::move(vFilterIdx));
          } else {
            vRes = new dgs_memory_hybrid(
              /* model             */ *this,
              /* attn_type_k       */ pParams.vTypeK,
              /* attn_type_v       */ pParams.vTypeV,
              /* attn_v_trans      */ !pCparams.vFlashAttn,
              /* attn_kv_size      */ pCparams.vNCtxSeq,
              /* attn_n_pad        */ 1,
              /* attn_n_swa        */ hparams.vNSwa,
              /* attn_swa_type     */ hparams.vSwaType,
              /* recurrent_type_k  */ GGML_TYPE_F32,
              /* recurrent_type_v  */ GGML_TYPE_F32,
              /* recurrent_kv_size */ std::max((uint32_t) 1, pCparams.vNSeqMax),
              /* n_seq_max         */ pCparams.vNSeqMax,
              /* n_rs_seq          */ pCparams.vNRsSeq,
              /* offload           */ pCparams.vOffloadKqv,
              /* unified           */ pCparams.vKvUnified,
              /* filter_attn       */ std::move(vFilterAttn),
              /* filter_recr       */ std::move(vFilterRecr));
          }
        } else {
          dgs_kv_cache::layer_filter_cb vFilter = nullptr;
          dgs_memory_i::layer_reuse_cb  vReuse  = nullptr;
          dgs_kv_cache::layer_share_cb  vShare  = nullptr;

          if (vArch == cLlmArchGemma3n || vArch == cLlmArchGemma4) {
            vReuse = [&](uint32_t pIl) {
              GGML_ASSERT(hparams.vNLayerKvFromStart >= 2);

              if (pIl >= (uint32_t) hparams.vNLayerKvFromStart) {
                return hparams.vNLayerKvFromStart - (hparams.mIsSwa(pIl) ? 2 : 1);
              }

              return -1;
            };
          }

          if (cMtpOnHybridQwen || cMtpOnHybridNemotron) {
            vFilter = [&](uint32_t pIl) {
              return pIl >= hparams.n_layer();
            };
          }

          // don't filter when n_layer_nextn is repurposed for a router layer the trunk attends
          // or when a model is entirely n_layer_nextn layers and has no trunk
          if (hparams.n_layer_nextn > 0 && hparams.n_layer() > 0 && hparams.vRouterLayer < 0) {
            if (pParams.vCtxType == cDgsContextTypeMtp) {
              vFilter = [&](uint32_t pIl) {
                return pIl >= hparams.n_layer();
              };
            } else {
              vFilter = [&](uint32_t pIl) {
                return pIl < hparams.n_layer();
              };
            }
          }

          if (hparams.vSwaType != cDgsSwaTypeNone) {
            GGML_ASSERT(hparams.mIsSwaAny());

            if (vArch == cLlmArchGemma4Assistant) {
              dgs_memory_t vMemOther = fDgsGetMemory(pCparams.vCtxOther);

              vShare = [&](int32_t pIl) {
                const dgs_model * vModelOther = fDgsGetModel(pCparams.vCtxOther);

                if (hparams.mIsSwa(pIl)) {
                  return fDgsModelNLayer(vModelOther) - 2;
                }

                return fDgsModelNLayer(vModelOther) - 1;
              };

              vRes =
                new dgs_kv_cache_iswa(*this, pParams.vTypeK, pParams.vTypeV, !pCparams.vFlashAttn, pCparams.vOffloadKqv,
                                      pParams.vSwaFull, pCparams.vKvUnified, pCparams.vNCtxSeq, pCparams.vNSeqMax,
                                      pCparams.vNUbatch, 1, vMemOther, vFilter, vReuse, vShare);
            } else {
              vRes =
                new dgs_kv_cache_iswa(*this, pParams.vTypeK, pParams.vTypeV, !pCparams.vFlashAttn, pCparams.vOffloadKqv,
                                      pParams.vSwaFull, pCparams.vKvUnified, pCparams.vNCtxSeq, pCparams.vNSeqMax,
                                      pCparams.vNUbatch, 1, nullptr, vFilter, vReuse, vShare);
            }
          } else {
            GGML_ASSERT(!hparams.mIsSwaAny());

            vRes = new dgs_kv_cache(*this, hparams, pParams.vTypeK, pParams.vTypeV, !pCparams.vFlashAttn,
                                    pCparams.vOffloadKqv, pCparams.vKvUnified, pCparams.vNCtxSeq, pCparams.vNSeqMax, 1,
                                    hparams.vNSwa, hparams.vSwaType, nullptr, vFilter, nullptr, nullptr);
          }
        }
      }
  }

  return vRes;
}

ggml_cgraph * dgs_model::mBuildGraph(const llm_graph_params & pParams) const {
  std::unique_ptr<llm_graph_context> vLlm = mBuildArchGraph(pParams);

  // add on pooling layer
  vLlm->mBuildPooling(vCls, vClsB, vClsOut, vClsOutB, vClsNorm);

  // add backend sampling layers (if any)
  vLlm->mBuildSampling();

  // if the gguf model was converted with --sentence-transformers-dense-modules
  // there will be two additional dense projection layers
  // dense linear projections are applied after pooling
  // TODO: move reranking logic here and generalize
  vLlm->mBuildDenseOut(vDense2OutLayers, vDense2OutLayersB, vDense3OutLayers);

  vLlm->vRes->mSetOutputs(pParams);

  return vLlm->vRes->mGetGf();
}

//
// interface implementation
//

dgs_model_params fDgsModelDefaultParams() {
  dgs_model_params vResult = {
    /*.devices                     =*/nullptr,
    /*.tensor_buft_overrides       =*/nullptr,
    /*.n_gpu_layers                =*/-1,
    /*.split_mode                  =*/cDgsSplitModeLayer,
    /*.load_mode                   =*/cDgsLoadModeAuto,
    /*.lazy_mode                   =*/cDgsLazyModeAuto,
    /*.main_gpu                    =*/0,
    /*.tensor_split                =*/nullptr,
    /*.progress_callback           =*/nullptr,
    /*.progress_callback_user_data =*/nullptr,
    /*.kv_overrides                =*/nullptr,
    /*.vocab_only                  =*/false,
    /*.check_tensors               =*/false,
    /*.use_extra_bufts             =*/true,
    /*.no_host                     =*/false,
    /*.no_alloc                    =*/false,
    /*.load_mtp                    =*/false,
  };

  return vResult;
}

const dgs_vocab * fDgsModelGetVocab(const dgs_model * pModel) {
  return &pModel->vocab;
}

void fDgsModelFree(dgs_model * pModel) {
  delete pModel;
}

int32_t fDgsModelNCtxTrain(const dgs_model * pModel) {
  return pModel->hparams.n_ctx_train;
}

int32_t fDgsModelNEmbd(const dgs_model * pModel) {
  return pModel->hparams.n_embd;
}

int32_t fDgsModelNEmbdInp(const dgs_model * pModel) {
  return pModel->hparams.mNEmbdInp();
}

int32_t fDgsModelNEmbdOut(const dgs_model * pModel) {
  return pModel->hparams.mNEmbdOut();
}

int32_t fDgsModelNLayer(const dgs_model * pModel) {
  return pModel->hparams.n_layer();
}

int32_t fDgsModelNLayerNextn(const dgs_model * pModel) {
  return pModel->hparams.n_layer_nextn;
}

int32_t fDgsModelDflashSelectorTopK(const dgs_model * pModel) {
  return pModel->hparams.vDflashSelectorTopK;
}

int32_t fDgsModelNHead(const dgs_model * pModel) {
  return pModel->hparams.n_head();
}

int32_t fDgsModelNHeadKv(const dgs_model * pModel) {
  return pModel->hparams.n_head_kv();
}

int32_t fDgsModelNSwa(const dgs_model * pModel) {
  // dsv4 kv-cache has SWA but it cannot be used as a rollback because of
  // other compression ratios, so we return 0 here
  if (pModel->vArch == cLlmArchDeepseek4) {
    return 0;
  }
  return pModel->hparams.vNSwa;
}

uint32_t fDgsModelNClsOut(const struct dgs_model * pModel) {
  return pModel->hparams.vNClsOut;
}

const char * fDgsModelClsLabel(const struct dgs_model * pModel, uint32_t pI) {
  if (pI < pModel->lClassifierLabels.size()) {
    return pModel->lClassifierLabels[pI].c_str();
  }

  return nullptr;
}

dgs_rope_type fDgsModelRopeType(const dgs_model * pModel) {
  switch (pModel->vArch) {
    case cLlmArchQwen35:

      return cDgsRopeTypeImrope;
  }

  return cDgsRopeTypeNone;
}

float fDgsModelRopeFreqScaleTrain(const dgs_model * pModel) {
  return pModel->hparams.vRopeFreqScaleTrain;
}

int32_t fDgsModelMetaValStr(const dgs_model * pModel, const char * pKey, char * pBuf, size_t pBufSize) {
  const auto & vIt = pModel->dGgufKv.find(pKey);
  if (vIt == pModel->dGgufKv.end()) {
    if (pBufSize > 0) {
      pBuf[0] = '\0';
    }
    return -1;
  }
  return snprintf(pBuf, pBufSize, "%s", vIt->second.c_str());
}

int32_t fDgsModelMetaCount(const dgs_model * pModel) {
  return (int) pModel->dGgufKv.size();
}

const char * fDgsModelMetaKeyStr(dgs_model_meta_key pKey) {
  switch (pKey) {
    case cDgsModelMetaKeySamplingSequence:
      return "general.sampling.sequence";
    case cDgsModelMetaKeySamplingTopK:
      return "general.sampling.top_k";
    case cDgsModelMetaKeySamplingTopP:
      return "general.sampling.top_p";
    case cDgsModelMetaKeySamplingMinP:
      return "general.sampling.min_p";
    case cDgsModelMetaKeySamplingXtcProbability:
      return "general.sampling.xtc_probability";
    case cDgsModelMetaKeySamplingXtcThreshold:
      return "general.sampling.xtc_threshold";
    case cDgsModelMetaKeySamplingTemp:
      return "general.sampling.temp";
    case cDgsModelMetaKeySamplingPenaltyLastN:
      return "general.sampling.penalty_last_n";
    case cDgsModelMetaKeySamplingPenaltyRepeat:
      return "general.sampling.penalty_repeat";
    case cDgsModelMetaKeySamplingMirostat:
      return "general.sampling.mirostat";
    case cDgsModelMetaKeySamplingMirostatTau:
      return "general.sampling.mirostat_tau";
    case cDgsModelMetaKeySamplingMirostatEta:
      return "general.sampling.mirostat_eta";
    default:
      return nullptr;
  }
}

int32_t fDgsModelMetaKeyByIndex(const dgs_model * pModel, int pI, char * pBuf, size_t pBufSize) {
  if (pI < 0 || pI >= (int) pModel->dGgufKv.size()) {
    if (pBufSize > 0) {
      pBuf[0] = '\0';
    }
    return -1;
  }
  auto vIt = pModel->dGgufKv.begin();
  std::advance(vIt, pI);
  return snprintf(pBuf, pBufSize, "%s", vIt->first.c_str());
}

int32_t fDgsModelMetaValStrByIndex(const dgs_model * pModel, int32_t pI, char * pBuf, size_t pBufSize) {
  if (pI < 0 || pI >= (int) pModel->dGgufKv.size()) {
    if (pBufSize > 0) {
      pBuf[0] = '\0';
    }
    return -1;
  }
  auto vIt = pModel->dGgufKv.begin();
  std::advance(vIt, pI);
  return snprintf(pBuf, pBufSize, "%s", vIt->second.c_str());
}

int32_t fDgsModelDesc(const dgs_model * pModel, char * pBuf, size_t pBufSize) {
  return snprintf(pBuf, pBufSize, "%s", pModel->mDesc().c_str());
}

dgs_ftype fDgsModelFtype(const dgs_model * pModel) {
  return pModel->mFtype();
}

uint64_t fDgsModelSize(const dgs_model * pModel) {
  return pModel->size();
}

const char * fDgsModelChatTemplate(const dgs_model * pModel, const char * name) {
  const auto   cKey = name ? LLM_KV(pModel->vArch, name)(cLlmKvTokenizerChatTemplate) :
                             LLM_KV(pModel->vArch)(cLlmKvTokenizerChatTemplate);
  const auto & vIt  = pModel->dGgufKv.find(cKey);
  if (vIt == pModel->dGgufKv.end()) {
    // one-off fix for very popular models (so we are not flooded with issues)
    // do not extend this list unless absolutely necessary
    // Mistral-Small-2503 does not have built-in chat template
    dgs_vocab_pre_type vPreType = pModel->vocab.mGetPreType();
    if (!name && vPreType == cDgsVocabPreTypeTekken && pModel->lLayers.size() == 40) {
      return "mistral-v7-tekken";
    }

    return nullptr;
  }

  return vIt->second.c_str();
}

uint64_t fDgsModelNParams(const dgs_model * pModel) {
  return pModel->mNElements();
}

bool fDgsModelHasEncoder(const dgs_model * pModel) {
  switch (pModel->vArch) {
    default:
      return false;
  }
}

bool fDgsModelHasDecoder(const dgs_model * pModel) {
  switch (pModel->vArch) {
    default:
      return true;
  }
}

dgs_token fDgsModelDecoderStartToken(const dgs_model * pModel) {
  return pModel->hparams.vDecStartTokenId;
}

bool fDgsModelIsRecurrent(const dgs_model * pModel) {
  return fLlmArchIsRecurrent(pModel->vArch);
}

bool fDgsModelIsHybrid(const dgs_model * pModel) {
  return fLlmArchIsHybrid(pModel->vArch);
}

bool fDgsModelIsDiffusion(const dgs_model * pModel) {
  return fLlmArchIsDiffusion(pModel->vArch);
}

const std::vector<std::pair<std::string, ggml_tensor *>> & fDgsInternalGetTensorMap(const dgs_model * pModel) {
  return pModel->lTensorsByName;
}

int32_t fDgsModelNExpert(const struct dgs_model * pModel) {
  return pModel->hparams.n_expert;
}

int32_t fDgsModelNDevices(const struct dgs_model * pModel) {
  return (int32_t) pModel->lDevices.size();
}

ggml_backend_dev_t fDgsModelGetDevice(const struct dgs_model * pModel, int pI) {
  if (pI < 0 || pI >= (int) pModel->lDevices.size()) {
    return nullptr;
  }
  return pModel->lDevices[pI].dev;
}

//
// dgs_model_base
//

dgs_model_base::dgs_model_base(const struct dgs_model_params & pParams) :
  dgs_model(pParams),
  vModel(this),
  cTn(vModel->vArch),
  cTensorDuplicated(dgs_model_loader::cTensorDuplicated),
  cTensorNotRequired(dgs_model_loader::cTensorNotRequired),
  cTensorSkip(dgs_model_loader::cTensorSkip),
  cTensorSkipIfVirtual(dgs_model_loader::cTensorSkipIfVirtual),
  cTensorAllowReshape(dgs_model_loader::cTensorAllowReshape),
  cTensorReadLazy(dgs_model_loader::cTensorReadLazy) {}

ggml_tensor * dgs_model_base::mCreateTensor(const LLM_TN_IMPL &                    pTn,
                                            const std::initializer_list<int64_t> & pNe,
                                            int                                    pFlags) {
  GGML_ASSERT(vMl != nullptr);
  return mCreateTensor(*vMl, pTn, pNe, pFlags);
}

void dgs_model_base::mCreateTensorGateUpExps(dgs_layer & layer,
                                             int         pBid,
                                             int64_t     pNEmbd,
                                             int64_t     pNFf,
                                             int64_t     pNExpert,
                                             int         pFlags) {
  if (pFlags & cTensorSkip) {
    const int cSkip = cTensorNotRequired | cTensorSkip;

    mCreateTensor(cTn(cLlmTensorFfnGateUpExps, "weight", pBid), { pNEmbd, pNFf * 2, pNExpert },
                  cSkip | cTensorSkipIfVirtual);
    mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", pBid), { pNEmbd, pNFf, pNExpert }, cSkip);
    mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", pBid), { pNEmbd, pNFf, pNExpert }, cSkip);
    return;
  }

  layer.vFfnGateUpExps =
    mCreateTensor(cTn(cLlmTensorFfnGateUpExps, "weight", pBid), { pNEmbd, pNFf * 2, pNExpert }, cTensorNotRequired);
  if (layer.vFfnGateUpExps == nullptr) {
    layer.vFfnGateExps = mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", pBid), { pNEmbd, pNFf, pNExpert }, pFlags);
    layer.vFfnUpExps   = mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", pBid), { pNEmbd, pNFf, pNExpert }, pFlags);
  }
}

void dgs_model_base::mCreateTensorQkv(dgs_layer & layer,
                                      int         pBid,
                                      int64_t     pNEmbd,
                                      int64_t     pNEmbdQ,
                                      int64_t     pNEmbdK,
                                      int64_t     pNEmbdV,
                                      int         pFlags) {
  const int64_t cNEmbdQkv = pNEmbdQ + pNEmbdK + pNEmbdV;

  if (pFlags & cTensorSkip) {
    const int cSkip = cTensorNotRequired | cTensorSkip;

    mCreateTensor(cTn(cLlmTensorAttnQkv, "weight", pBid), { pNEmbd, cNEmbdQkv }, cSkip | cTensorSkipIfVirtual);
    mCreateTensor(cTn(cLlmTensorAttnQkv, "bias", pBid), { cNEmbdQkv }, cSkip | cTensorSkipIfVirtual);
    mCreateTensor(cTn(cLlmTensorAttnQ, "weight", pBid), { pNEmbd, pNEmbdQ }, cSkip);
    mCreateTensor(cTn(cLlmTensorAttnK, "weight", pBid), { pNEmbd, pNEmbdK }, cSkip);
    mCreateTensor(cTn(cLlmTensorAttnV, "weight", pBid), { pNEmbd, pNEmbdV }, cSkip);
    mCreateTensor(cTn(cLlmTensorAttnQ, "bias", pBid), { pNEmbdQ }, cSkip);
    mCreateTensor(cTn(cLlmTensorAttnK, "bias", pBid), { pNEmbdK }, cSkip);
    mCreateTensor(cTn(cLlmTensorAttnV, "bias", pBid), { pNEmbdV }, cSkip);
    return;
  }

  layer.vWqkv = mCreateTensor(cTn(cLlmTensorAttnQkv, "weight", pBid), { pNEmbd, cNEmbdQkv },
                              cTensorNotRequired | cTensorSkipIfVirtual);
  if (layer.vWqkv) {
    layer.vWqkvB =
      mCreateTensor(cTn(cLlmTensorAttnQkv, "bias", pBid), { cNEmbdQkv }, cTensorNotRequired | cTensorSkipIfVirtual);
    // Current fused GGUF projections may use separate Q/K/V biases.
    if (!layer.vWqkvB) {
      layer.vWqB = mCreateTensor(cTn(cLlmTensorAttnQ, "bias", pBid), { pNEmbdQ }, cTensorNotRequired);
      layer.vWkB = mCreateTensor(cTn(cLlmTensorAttnK, "bias", pBid), { pNEmbdK }, cTensorNotRequired);
      layer.vWvB = mCreateTensor(cTn(cLlmTensorAttnV, "bias", pBid), { pNEmbdV }, cTensorNotRequired);
    }
  } else {
    layer.vWq  = mCreateTensor(cTn(cLlmTensorAttnQ, "weight", pBid), { pNEmbd, pNEmbdQ }, pFlags);
    layer.vWk  = mCreateTensor(cTn(cLlmTensorAttnK, "weight", pBid), { pNEmbd, pNEmbdK }, pFlags);
    layer.vWv  = mCreateTensor(cTn(cLlmTensorAttnV, "weight", pBid), { pNEmbd, pNEmbdV }, pFlags);
    layer.vWqB = mCreateTensor(cTn(cLlmTensorAttnQ, "bias", pBid), { pNEmbdQ }, cTensorNotRequired);
    layer.vWkB = mCreateTensor(cTn(cLlmTensorAttnK, "bias", pBid), { pNEmbdK }, cTensorNotRequired);
    layer.vWvB = mCreateTensor(cTn(cLlmTensorAttnV, "bias", pBid), { pNEmbdV }, cTensorNotRequired);
  }
}

dgs_model_base::nextn_flags_t dgs_model_base::mNextnFlags(dgs_model_loader & pMl, llm_tensor pTrunkProbe) const {
  int vTrunk = 0;
  int vMtp   = 0;

  // a file without the first trunk layer is MTP-only, a file without the first NextN layer is trunk-only
  if (hparams.n_layer_nextn > 0) {
    if (pMl.mGetWeight(cTn(pTrunkProbe, "weight", 0).str().c_str()) == nullptr) {
      vTrunk = cTensorNotRequired;
    }
    if (pMl.mGetWeight(cTn(cLlmTensorNextnEhProj, "weight", hparams.n_layer()).str().c_str()) == nullptr) {
      vMtp = cTensorNotRequired;
    }
  }

  if (!pMl.vLoadMtp) {
    vMtp |= cTensorSkip;
  }

  return { vTrunk, vMtp };
}

void dgs_model_base::mLoadSwaPattern(dgs_model_loader & pMl, uint32_t pNPattern, bool pDenseFirst) {
  if (pMl.get_arr(cLlmKvAttentionSlidingWindowPattern, hparams.aIsSwaImpl, false)) {
    return;
  }

  pMl.get_key(cLlmKvAttentionSlidingWindowPattern, pNPattern, false);
  hparams.mSetSwaPattern(pNPattern, pDenseFirst);
}

const int32_t * fDgsModelTargetLayerIds(const struct dgs_model * pModel) {
  const auto & lV = pModel->lTargetLayerIds;
  return lV.empty() ? nullptr : lV.data();
}

uint32_t fDgsModelTargetLayerIdsN(const struct dgs_model * pModel) {
  return (uint32_t) pModel->lTargetLayerIds.size();
}

uint32_t fDgsModelGetTokEmbd(const struct dgs_model * pModel, float * pOut) {
  if (pModel->vocab.n_tokens() == 0 || pModel->vTokEmbd == nullptr) {
    return 0;
  }

  const ggml_tensor * vTensor    = pModel->vTokEmbd;
  const size_t        cNelements = ggml_nelements(vTensor);
  GGML_ASSERT(cNelements <= UINT32_MAX);  // for the return type

  if (pOut == nullptr) {
    return (uint32_t) cNelements;
  }

  if (vTensor->type == GGML_TYPE_F32) {
    ggml_backend_tensor_get(vTensor, pOut, 0, cNelements * sizeof(float));
    return (uint32_t) cNelements;
  }

  std::vector<uint8_t> lBuf(ggml_nbytes(vTensor));
  ggml_backend_tensor_get(vTensor, lBuf.data(), 0, lBuf.size());

  const ggml_type_traits * vTraits = ggml_get_type_traits(vTensor->type);
  if (vTensor->type == GGML_TYPE_F16) {
    ggml_fp16_to_fp32_row((const ggml_fp16_t *) lBuf.data(), pOut, cNelements);
  } else if (vTensor->type == GGML_TYPE_BF16) {
    ggml_bf16_to_fp32_row((const ggml_bf16_t *) lBuf.data(), pOut, cNelements);
  } else if (ggml_is_quantized(vTensor->type) && vTraits->to_float != nullptr) {
    vTraits->to_float(lBuf.data(), pOut, cNelements);
  } else {
    GGML_ABORT("unsupported tensor type for dequantization: %s", ggml_type_name(vTensor->type));
  }

  return (uint32_t) cNelements;
}

dgs_model_mtp_breakdown fDgsModelGetMtpBreakdown(const struct dgs_model * pModel) {
  dgs_model_mtp_breakdown dBreakdown;
  const int               cFirstLayer = static_cast<int>(pModel->hparams.n_layer());
  for (const auto & [vName, vTensor] : pModel->lTensorsByName) {
    int vLayer = -1;
    if (vTensor && vTensor->buffer && sscanf(vName.c_str(), "blk.%d.", &vLayer) == 1 && vLayer >= cFirstLayer) {
      dBreakdown[ggml_backend_buffer_get_type(vTensor->buffer)] +=
        ggml_backend_buffer_get_alloc_size(vTensor->buffer, vTensor);
    }
  }
  return dBreakdown;
}
