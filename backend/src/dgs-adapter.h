#pragma once

#include "dgs.h"
#include "ggml-cpp.h"

#include <string>
#include <unordered_map>
#include <vector>

// TODO: pimpl

//
// dgs_adapter_cvec
//

struct dgs_adapter_cvec {
  ggml_tensor * mTensorFor(int pIl) const;

  ggml_tensor * mApplyTo(ggml_context * pCtx, ggml_tensor * pCur, int pIl) const;

  bool apply(const dgs_model & pModel,
             const float *     data,
             size_t            pLen,
             int32_t           pNEmbd,
             int32_t           pIlStart,
             int32_t           pIlEnd);

private:
  bool init(const dgs_model & pModel);

  int32_t vLayerStart = -1;
  int32_t vLayerEnd   = -1;

  std::vector<ggml_context_ptr>        lCtxs;
  std::vector<ggml_backend_buffer_ptr> lBufs;

  std::vector<ggml_tensor *> lTensors;  // per layer
};

using dgs_adapter_cvec_ptr = std::shared_ptr<dgs_adapter_cvec>;

//
// dgs_adapter_lora
//

struct dgs_adapter_lora_weight {
  ggml_tensor * vA = nullptr;
  ggml_tensor * vB = nullptr;

  // get actual scale based on rank and alpha
  float mGetScale(float pAlpha, float pAdapterScale) const {
    const float cRank  = (float) vB->ne[0];
    const float cScale = pAlpha ? pAdapterScale * pAlpha / cRank : pAdapterScale;
    return cScale;
  }

  dgs_adapter_lora_weight() = default;

  dgs_adapter_lora_weight(ggml_tensor * pA, ggml_tensor * pB) : vA(pA), vB(pB) {}
};

struct dgs_adapter_lora {
  dgs_model * vModel = nullptr;

  // map tensor name to lora_a_b
  std::unordered_map<std::string, dgs_adapter_lora_weight> dAbMap;

  std::vector<ggml_context_ptr>        lCtxs;
  std::vector<ggml_backend_buffer_ptr> lBufs;

  float vAlpha;

  // gguf metadata
  std::unordered_map<std::string, std::string> dGgufKv;

  // activated lora (aLoRA)
  std::vector<dgs_token> lAloraInvocationTokens;

  explicit dgs_adapter_lora(dgs_model * pModel) : vModel(pModel) {}

  ~dgs_adapter_lora() = default;

  dgs_adapter_lora_weight * mGetWeight(ggml_tensor * pW);

  uint32_t mGetNNodes() const {
    return dAbMap.size() * 6u;  // a, b, scale, add, 2 x mul_mat
  }
};

using dgs_adapter_loras     = std::unordered_map<dgs_adapter_lora *, float>;
using dgs_adapter_loras_ptr = std::unique_ptr<dgs_adapter_loras>;
