#pragma once

#include "ggml.h"  // for ggml_log_level

#include <string>
#include <type_traits>
#include <vector>

#ifdef __GNUC__
#  if defined(__MINGW32__) && !defined(__clang__)
#    define DGS_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#  else
#    define DGS_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#  endif
#else
#  define DGS_ATTRIBUTE_FORMAT(...)
#endif

//
// logging
//

DGS_ATTRIBUTE_FORMAT(2, 3)
void fDgsLogInternal(ggml_log_level pLevel, const char * format, ...);
void fDgsLogCallbackDefault(ggml_log_level pLevel, const char * pText, void * pUserData);

#define DGS_LOG(...)       fDgsLogInternal(GGML_LOG_LEVEL_NONE, __VA_ARGS__)
#define DGS_LOG_INFO(...)  fDgsLogInternal(GGML_LOG_LEVEL_INFO, __VA_ARGS__)
#define DGS_LOG_WARN(...)  fDgsLogInternal(GGML_LOG_LEVEL_WARN, __VA_ARGS__)
#define DGS_LOG_ERROR(...) fDgsLogInternal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define DGS_LOG_DEBUG(...) fDgsLogInternal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define DGS_LOG_CONT(...)  fDgsLogInternal(GGML_LOG_LEVEL_CONT, __VA_ARGS__)

//
// helpers
//

template <typename T> struct no_init {
  T value;
  no_init() = default;
};

template <typename dst_t, typename src_t> static inline dst_t fDgsCast(src_t pV) {
  if constexpr (std::is_same_v<src_t, dst_t>) {
    return pV;
  } else if constexpr (std::is_same_v<src_t, ggml_fp16_t> && std::is_same_v<dst_t, float>) {
    return ggml_fp16_to_fp32(pV);
  } else if constexpr (std::is_same_v<src_t, float> && std::is_same_v<dst_t, ggml_fp16_t>) {
    return ggml_fp32_to_fp16(pV);
  } else {
    static_assert(std::is_same_v<dst_t, void>, "unsupported type combination");
  }
}

static inline ggml_tensor * fDgsMulMatHadamard(ggml_context * pCtx, ggml_tensor * pCur, ggml_tensor * pRot) {
  const auto n = pRot->ne[0];

  ggml_tensor * vRes;

  if (!ggml_is_contiguous(pCur)) {
    vRes = ggml_cont_2d(pCtx, pCur, n, ggml_nelements(pCur) / n);
  } else {
    vRes = ggml_reshape_2d(pCtx, pCur, n, ggml_nelements(pCur) / n);
  }
  vRes = ggml_mul_mat(pCtx, pRot, vRes);
  ggml_mul_mat_set_hint(vRes, GGML_HINT_SRC0_IS_HADAMARD);
  vRes = ggml_reshape_4d(pCtx, vRes, pCur->ne[0], pCur->ne[1], pCur->ne[2], pCur->ne[3]);

  return vRes;
}

// Prefetch the host pages needed to gather these rows.
void fDgsPrefetchRows(const ggml_tensor * pTensor, const int32_t * pRows, size_t pNRows);

struct time_meas {
  time_meas(int64_t & pTAcc, bool pDisable = false);
  ~time_meas();

  const int64_t cTStartUs;

  int64_t & vTAcc;
};

template <typename T> struct buffer_view {
  T *    data;
  size_t size = 0;

  bool mHasData() const { return data && size > 0; }
};

void fDgsClearTensorData(ggml_tensor * pT, size_t pOffset, size_t size);

void fReplaceAll(std::string & pS, const std::string & pSearch, const std::string & pReplace);

// TODO: rename to dgs_format ?
DGS_ATTRIBUTE_FORMAT(1, 2)
std::string format(const char * pFmt, ...);

std::string fDgsFormatTensorShape(const std::vector<int64_t> & pNe);
std::string fDgsFormatTensorShape(const struct ggml_tensor * pT);

std::string fGgufKvToStr(const struct gguf_context * pCtxGguf, int pI);
