#include "dgs-impl.h"

#include "dgs-mmap.h"
#include "dgs.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cinttypes>
#include <climits>
#include <cstdarg>
#include <cstring>
#include <sstream>
#include <vector>

struct dgs_logger_state {
  ggml_log_callback vLogCallback         = fDgsLogCallbackDefault;
  void *            vLogCallbackUserData = nullptr;
};

static dgs_logger_state vGLoggerState;

void fDgsPrefetchRows(const ggml_tensor * pTensor, const int32_t * pRows, size_t pNRows) {
  if (!pTensor || !pTensor->data || !pTensor->buffer || !ggml_backend_buffer_is_host(pTensor->buffer) || pNRows == 0) {
    return;
  }

  GGML_ASSERT(ggml_is_matrix(pTensor));

  const size_t cRowBytes = ggml_row_size(pTensor->type, pTensor->ne[0]);
  const auto * vBase     = (const char *) pTensor->data;

  std::vector<dgs_memory_range> lMr;
  lMr.reserve(pNRows);
  for (size_t vI = 0; vI < pNRows; ++vI) {
    GGML_ASSERT(pRows[vI] >= 0 && pRows[vI] < pTensor->ne[1]);
    lMr.push_back({ vBase + (size_t) pRows[vI] * pTensor->nb[1], cRowBytes });
  }

  fDgsPrefetch(std::move(lMr));
}

time_meas::time_meas(int64_t & pTAcc, bool pDisable) : cTStartUs(pDisable ? -1 : ggml_time_us()), vTAcc(pTAcc) {}

time_meas::~time_meas() {
  if (cTStartUs >= 0) {
    vTAcc += ggml_time_us() - cTStartUs;
  }
}

void fDgsLogGet(ggml_log_callback * pLogCallback, void ** pUserData) {
  ggml_log_get(pLogCallback, pUserData);
}

void fDgsLogSet(ggml_log_callback pLogCallback, void * pUserData) {
  ggml_log_set(pLogCallback, pUserData);
  vGLoggerState.vLogCallback         = pLogCallback ? pLogCallback : fDgsLogCallbackDefault;
  vGLoggerState.vLogCallbackUserData = pUserData;
}

static void fDgsLogInternalV(ggml_log_level pLevel, const char * format, va_list pArgs) {
  va_list vArgsCopy;
  va_copy(vArgsCopy, pArgs);
  char aBuffer[128];
  int  vLen = vsnprintf(aBuffer, 128, format, pArgs);
  if (vLen < 128) {
    vGLoggerState.vLogCallback(pLevel, aBuffer, vGLoggerState.vLogCallbackUserData);
  } else {
    char * vBuffer2 = new char[vLen + 1];
    vsnprintf(vBuffer2, vLen + 1, format, vArgsCopy);
    vBuffer2[vLen] = 0;
    vGLoggerState.vLogCallback(pLevel, vBuffer2, vGLoggerState.vLogCallbackUserData);
    delete[] vBuffer2;
  }
  va_end(vArgsCopy);
}

void fDgsLogInternal(ggml_log_level pLevel, const char * format, ...) {
  va_list vArgs;
  va_start(vArgs, format);
  fDgsLogInternalV(pLevel, format, vArgs);
  va_end(vArgs);
}

void fDgsLogCallbackDefault(ggml_log_level pLevel, const char * pText, void * pUserData) {
  (void) pLevel;
  (void) pUserData;
  fputs(pText, stderr);
  fflush(stderr);
}

void fDgsClearTensorData(ggml_tensor * pT, size_t pOffset, size_t size) {
  static const std::vector<uint8_t> lZeros(1024 * 1024, 0);

  // not all backend buffers implement ggml_backend_tensor_memset(), so write zeros instead
  // TODO: make this a generic fallback in `ggml_backend_tensor_memset` when `set_tensor` is available
  for (size_t vOfs = 0; vOfs < size; vOfs += lZeros.size()) {
    ggml_backend_tensor_set(pT, lZeros.data(), pOffset + vOfs, std::min(size - vOfs, lZeros.size()));
  }
}

void fReplaceAll(std::string & pS, const std::string & pSearch, const std::string & pReplace) {
  if (pSearch.empty()) {
    return;
  }
  std::string vBuilder;
  vBuilder.reserve(pS.length());
  size_t vPos     = 0;
  size_t vLastPos = 0;
  while ((vPos = pS.find(pSearch, vLastPos)) != std::string::npos) {
    vBuilder.append(pS, vLastPos, vPos - vLastPos);
    vBuilder.append(pReplace);
    vLastPos = vPos + pSearch.length();
  }
  vBuilder.append(pS, vLastPos, std::string::npos);
  pS = std::move(vBuilder);
}

std::string format(const char * pFmt, ...) {
  va_list vAp;
  va_list vAp2;
  va_start(vAp, pFmt);
  va_copy(vAp2, vAp);
  int size = vsnprintf(NULL, 0, pFmt, vAp);
  GGML_ASSERT(size >= 0 && size < INT_MAX);  // NOLINT
  std::vector<char> lBuf(size + 1);
  int               vSize2 = vsnprintf(lBuf.data(), size + 1, pFmt, vAp2);
  GGML_ASSERT(vSize2 == size);
  va_end(vAp2);
  va_end(vAp);
  return std::string(lBuf.data(), size);
}

std::string fDgsFormatTensorShape(const std::vector<int64_t> & pNe) {
  char aBuf[256];
  snprintf(aBuf, sizeof(aBuf), "%6" PRId64, pNe.at(0));
  for (size_t vI = 1; vI < pNe.size(); vI++) {
    snprintf(aBuf + strlen(aBuf), sizeof(aBuf) - strlen(aBuf), ", %6" PRId64, pNe.at(vI));
  }
  return aBuf;
}

std::string fDgsFormatTensorShape(const struct ggml_tensor * pT) {
  char aBuf[256];
  snprintf(aBuf, sizeof(aBuf), "%6" PRId64, pT->ne[0]);
  for (int vI = 1; vI < GGML_MAX_DIMS; vI++) {
    snprintf(aBuf + strlen(aBuf), sizeof(aBuf) - strlen(aBuf), ", %6" PRId64, pT->ne[vI]);
  }
  return aBuf;
}

static std::string fGgufDataToStr(enum gguf_type type, const void * data, int pI) {
  switch (type) {
    case GGUF_TYPE_UINT8:
      return std::to_string(((const uint8_t *) data)[pI]);
    case GGUF_TYPE_INT8:
      return std::to_string(((const int8_t *) data)[pI]);
    case GGUF_TYPE_UINT16:
      return std::to_string(((const uint16_t *) data)[pI]);
    case GGUF_TYPE_INT16:
      return std::to_string(((const int16_t *) data)[pI]);
    case GGUF_TYPE_UINT32:
      return std::to_string(((const uint32_t *) data)[pI]);
    case GGUF_TYPE_INT32:
      return std::to_string(((const int32_t *) data)[pI]);
    case GGUF_TYPE_UINT64:
      return std::to_string(((const uint64_t *) data)[pI]);
    case GGUF_TYPE_INT64:
      return std::to_string(((const int64_t *) data)[pI]);
    case GGUF_TYPE_FLOAT32:
      return std::to_string(((const float *) data)[pI]);
    case GGUF_TYPE_FLOAT64:
      return std::to_string(((const double *) data)[pI]);
    case GGUF_TYPE_BOOL:
      return ((const int8_t *) data)[pI] != 0 ? "true" : "false";
    default:
      return format("unknown type %d", type);
  }
}

std::string fGgufKvToStr(const struct gguf_context * pCtxGguf, int pI) {
  const enum gguf_type type = gguf_get_kv_type(pCtxGguf, pI);

  switch (type) {
    case GGUF_TYPE_STRING:
      return gguf_get_val_str(pCtxGguf, pI);
    case GGUF_TYPE_ARRAY:
      {
        const enum gguf_type cArrType = gguf_get_arr_type(pCtxGguf, pI);
        int                  vArrN    = gguf_get_arr_n(pCtxGguf, pI);
        const void *         data     = cArrType == GGUF_TYPE_STRING ? nullptr : gguf_get_arr_data(pCtxGguf, pI);
        std::stringstream    vSs;
        vSs << "[";
        for (int vJ = 0; vJ < vArrN; vJ++) {
          if (cArrType == GGUF_TYPE_STRING) {
            std::string vVal = gguf_get_arr_str(pCtxGguf, pI, vJ);
            // escape quotes
            fReplaceAll(vVal, "\\", "\\\\");
            fReplaceAll(vVal, "\"", "\\\"");
            vSs << '"' << vVal << '"';
          } else if (cArrType == GGUF_TYPE_ARRAY) {
            vSs << "???";
          } else {
            vSs << fGgufDataToStr(cArrType, data, vJ);
          }
          if (vJ < vArrN - 1) {
            vSs << ", ";
          }
        }
        vSs << "]";
        return vSs.str();
      }
    default:
      return fGgufDataToStr(type, gguf_get_val_data(pCtxGguf, pI), 0);
  }
}
