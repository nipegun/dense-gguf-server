#include "debug.h"

#include "common.h"
#include "log.h"

#include <cmath>
#include <regex>
#include <string>
#include <vector>

struct common_debug_cb_user_data::impl {
  std::vector<uint8_t>    data;
  std::vector<std::regex> lTensorFilters;
  bool                    vAbortOnNan{ false };
};

common_debug_cb_user_data::common_debug_cb_user_data() : vPimpl(std::make_unique<impl>()) {}

common_debug_cb_user_data::~common_debug_cb_user_data() = default;

common_debug_cb_user_data::common_debug_cb_user_data(common_params &                  pParams,
                                                     const std::vector<std::string> & pFilterPatterns,
                                                     bool                             pAbortOnNan) :
  vPimpl(std::make_unique<impl>()) {
  for (const auto & vPattern : pFilterPatterns) {
    try {
      std::string vAnchoredPattern = "^" + vPattern;
      vPimpl->lTensorFilters.emplace_back(vAnchoredPattern, std::regex::optimize);
    } catch (const std::regex_error & vE) {
      throw std::runtime_error("Invalid regex pattern '" + vPattern + "': " + vE.what());
    }
  }
  vPimpl->vAbortOnNan = pAbortOnNan;

  pParams.vCbEval         = fCommonDebugCbEval;
  pParams.vCbEvalUserData = this;
}

static std::string fCommonGgmlNeString(const ggml_tensor * pT) {
  std::string str;
  for (int vI = 0; vI < GGML_MAX_DIMS; ++vI) {
    str += std::to_string(pT->ne[vI]);
    if (vI + 1 < GGML_MAX_DIMS) {
      str += ", ";
    }
  }
  return str;
}

static float fCommonGgmlGetFloatValue(const uint8_t * data,
                                      ggml_type       type,
                                      const size_t *  pNb,
                                      size_t          pI0,
                                      size_t          pI1,
                                      size_t          pI2,
                                      size_t          pI3) {
  size_t vI = pI3 * pNb[3] + pI2 * pNb[2] + pI1 * pNb[1] + pI0 * pNb[0];
  float  vV;
  if (type == GGML_TYPE_F16) {
    vV = ggml_fp16_to_fp32(*(const ggml_fp16_t *) &data[vI]);
  } else if (type == GGML_TYPE_F32) {
    vV = *(const float *) &data[vI];
  } else if (type == GGML_TYPE_I64) {
    vV = (float) *(const int64_t *) &data[vI];
  } else if (type == GGML_TYPE_I32) {
    vV = (float) *(const int32_t *) &data[vI];
  } else if (type == GGML_TYPE_I16) {
    vV = (float) *(const int16_t *) &data[vI];
  } else if (type == GGML_TYPE_I8) {
    vV = (float) *(const int8_t *) &data[vI];
  } else if (type == GGML_TYPE_BF16) {
    vV = ggml_bf16_to_fp32(*(const ggml_bf16_t *) &data[vI]);
  } else {
    GGML_ABORT("fatal error");
  }
  return vV;
}

#define INDENT "    "

static void fCommonDebugPrintTensor(uint8_t *       data,
                                    ggml_type       type,
                                    const int64_t * pNe,
                                    const size_t *  pNb,
                                    int64_t         n,
                                    bool            pAbortOnNan) {
  GGML_ASSERT(n > 0);
  float vSum = 0;
  for (int64_t vI3 = 0; vI3 < pNe[3]; vI3++) {
    for (int64_t vI2 = 0; vI2 < pNe[2]; vI2++) {
      for (int64_t vI1 = 0; vI1 < pNe[1]; vI1++) {
        for (int64_t vI0 = 0; vI0 < pNe[0]; vI0++) {
          const float cV = fCommonGgmlGetFloatValue(data, type, pNb, vI0, vI1, vI2, vI3);
          vSum += cV;
        }
      }
    }
  }
  for (int64_t vI3 = 0; vI3 < pNe[3]; vI3++) {
    LOG(INDENT "[\n");
    for (int64_t vI2 = 0; vI2 < pNe[2]; vI2++) {
      if (vI2 == n && pNe[2] > 2 * n) {
        LOG(INDENT INDENT "..., \n");
        vI2 = pNe[2] - n;
      }
      LOG(INDENT INDENT "[\n");
      for (int64_t vI1 = 0; vI1 < pNe[1]; vI1++) {
        if (vI1 == n && pNe[1] > 2 * n) {
          LOG(INDENT INDENT INDENT "..., \n");
          vI1 = pNe[1] - n;
        }
        LOG(INDENT INDENT INDENT "[");
        for (int64_t vI0 = 0; vI0 < pNe[0]; vI0++) {
          if (vI0 == n && pNe[0] > 2 * n) {
            LOG("   ..., ");
            vI0 = pNe[0] - n;
          }
          const float cV = fCommonGgmlGetFloatValue(data, type, pNb, vI0, vI1, vI2, vI3);
          LOG("%12.4f", cV);
          if (vI0 < pNe[0] - 1) {
            LOG(", ");
          }
        }
        LOG("  ],\n");
      }
      LOG(INDENT INDENT "],\n");
    }
    LOG(INDENT "]\n");
    LOG(INDENT "sum = %f\n", vSum);
  }

  if (pAbortOnNan) {
    if (std::isnan(vSum)) {
      LOG("encountered NaN - aborting\n");
      exit(0);
    }
  }
}

/**
 * GGML operations callback during the graph execution.
 *
 * @param t current tensor
 * @param ask when ask is true, the scheduler wants to know if we are interested in data from this tensor
 *            if we return true, a follow-up call will be made with ask=false in which we can do the actual collection.
 *            see ggml_backend_sched_eval_callback
 * @param user_data user data to pass at each call back
 * @return true to receive data or continue the graph, false otherwise
 */
bool fCommonDebugCbEval(struct ggml_tensor * pT, bool pAsk, void * pUserData) {
  auto * vCbData = (common_debug_cb_user_data *) pUserData;
  auto * vPimpl  = vCbData->vPimpl.get();

  const struct ggml_tensor * vSrc0 = pT->src[0];
  const struct ggml_tensor * vSrc1 = pT->src[1];

  if (pAsk) {
    return true;  // Always retrieve data
  }

  bool vMatchesFilter = vPimpl->lTensorFilters.empty();

  if (!vMatchesFilter) {
    for (const auto & vFilter : vPimpl->lTensorFilters) {
      if (std::regex_search(pT->name, vFilter)) {
        vMatchesFilter = true;
        break;
      }
    }
  }

  char aSrc1Str[128] = { 0 };
  if (vSrc1) {
    snprintf(aSrc1Str, sizeof(aSrc1Str), "%s{%s}", vSrc1->name, fCommonGgmlNeString(vSrc1).c_str());
  }

  if (vMatchesFilter) {
    LOG("%s: %24s = (%s) %10s(%s{%s}, %s}) = {%s}\n", __func__, pT->name, ggml_type_name(pT->type), ggml_op_desc(pT),
        vSrc0->name, fCommonGgmlNeString(vSrc0).c_str(), vSrc1 ? aSrc1Str : "", fCommonGgmlNeString(pT).c_str());
  }

  const bool cIsHost = ggml_backend_buffer_is_host(pT->buffer);

  if (!cIsHost) {
    auto vNBytes = ggml_nbytes(pT);
    vPimpl->data.resize(vNBytes);
    ggml_backend_tensor_get(pT, vPimpl->data.data(), 0, vNBytes);
  }

  if (!ggml_is_quantized(pT->type) && vMatchesFilter) {
    uint8_t * data = cIsHost ? (uint8_t *) pT->data : vPimpl->data.data();
    fCommonDebugPrintTensor(data, pT->type, pT->ne, pT->nb, 3, vPimpl->vAbortOnNan);
  }

  return true;
}
