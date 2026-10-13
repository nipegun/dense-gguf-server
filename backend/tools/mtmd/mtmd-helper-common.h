#pragma once

// shared internal utilities for the mtmd-helper-*.cpp translation units
// (mtmd-helper.cpp, mtmd-helper-gen.cpp)
// NOT part of the public mtmd-helper.h API

#include "dgs-cpp.h"
#include "dgs.h"
#include "ggml.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <vector>

//
// logging
//

struct mtmd_helper_logger {
  ggml_log_callback vDefaultCallback = [](ggml_log_level pLevel, const char * pText, void * pUserData) {
    (void) pLevel;
    (void) pUserData;
    fputs(pText, stderr);
    fflush(stderr);
  };

  ggml_log_callback vLogCallback = vDefaultCallback;
  void *            vLogCallbackUserData;

  void mLogV(enum ggml_log_level pLevel, const char * format, va_list pArgs) {
    if (format == NULL) {
      return;
    }
    va_list vArgsCopy;
    va_copy(vArgsCopy, pArgs);
    char aBuffer[128];
    int  vLen = vsnprintf(aBuffer, 128, format, pArgs);
    if (vLen < 128) {
      vLogCallback(pLevel, aBuffer, vLogCallbackUserData);
    } else {
      char * vBuffer2 = (char *) calloc(vLen + 1, sizeof(char));
      vsnprintf(vBuffer2, vLen + 1, format, vArgsCopy);
      vBuffer2[vLen] = 0;
      vLogCallback(pLevel, vBuffer2, vLogCallbackUserData);
      free(vBuffer2);
    }
    va_end(vArgsCopy);
  }

  void log(enum ggml_log_level pLevel, const char * format, ...) {
    va_list vArgs;
    va_start(vArgs, format);
    mLogV(pLevel, format, vArgs);
    va_end(vArgs);
  }
};

// inline, so all TUs including this header share one instance
inline mtmd_helper_logger g_logger;

#define LOG_DBG(...) g_logger.log(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LOG_INF(...) g_logger.log(GGML_LOG_LEVEL_INFO, __VA_ARGS__)
#define LOG_WRN(...) g_logger.log(GGML_LOG_LEVEL_WARN, __VA_ARGS__)
#define LOG_ERR(...) g_logger.log(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)

//
// embd batch
//

// helper struct to make working with embd batch easier
// note: this will be removed after dgs_batch_ext refactoring
struct decode_embd_batch {
  int                  vNPosPerEmbd;
  int                  vNMmprojEmbd;
  int32_t              vNTokens;
  const float *        vEmbd;     // [n_tokens, n_mmproj_embd], not owned
  std::vector<dgs_pos> lPos;      // [n_pos_per_embd, n_tokens], section-major
  std::vector<dgs_pos> lPosView;  // sliced positions of the last get_view()
  std::vector<int8_t>  lLogits;
  dgs_seq_id           vSeqId = 0;

  dgs_batch_ext_ptr vBatch;  // rendered sub-batch, see render()

  decode_embd_batch(const float * pEmbd, int32_t pNTokens, int pNPosPerEmbd, int pNMmprojEmbd) :
    vNPosPerEmbd(pNPosPerEmbd),
    vNMmprojEmbd(pNMmprojEmbd),
    vNTokens(pNTokens),
    vEmbd(pEmbd) {
    GGML_ASSERT(pNTokens > 0 && pNPosPerEmbd > 0 && pNMmprojEmbd > 0);
    lPos.resize((size_t) pNTokens * (size_t) pNPosPerEmbd);
    lLogits.resize(pNTokens);
  }

  void mSetPositionNormal(dgs_pos pPos0, dgs_seq_id pSeqId) {
    this->vSeqId = pSeqId;
    for (int vI = 0; vI < vNTokens; vI++) {
      lPos[vI] = pPos0 + vI;
    }
  }

  // M-RoPE for image
  void mSetPositionMrope2d(const std::vector<mtmd_decoder_pos> & pRelPos, dgs_seq_id pSeqId) {
    GGML_ASSERT(vNPosPerEmbd == 4);
    GGML_ASSERT(!pRelPos.empty() && (int32_t) pRelPos.size() == vNTokens);
    this->vSeqId = pSeqId;
    for (int32_t vI = 0; vI < vNTokens; vI++) {
      const size_t cIdx  = (size_t) vI;
      const size_t n     = (size_t) vNTokens;
      lPos[cIdx]         = pRelPos[vI].vT;
      lPos[cIdx + n]     = pRelPos[vI].vY;
      lPos[cIdx + n * 2] = pRelPos[vI].vX;
      lPos[cIdx + n * 3] = pRelPos[vI].vZ;
    }
  }

  // M-RoPE for audio
  void mSetPositionMrope1d(dgs_pos pPos0, dgs_seq_id pSeqId) {
    GGML_ASSERT(vNPosPerEmbd == 4);
    this->vSeqId = pSeqId;
    for (int vI = 0; vI < vNTokens; vI++) {
      const size_t cIdx  = (size_t) vI;
      const size_t n     = (size_t) vNTokens;
      lPos[cIdx]         = pPos0 + vI;
      lPos[cIdx + n]     = pPos0 + vI;
      lPos[cIdx + n * 2] = pPos0 + vI;
      lPos[cIdx + n * 3] = pPos0 + vI;
    }
  }

  // describe the entries [offset, offset + n) with section-major positions
  mtmd_helper_embd_batch mGetView(int pOffset, int n) {
    GGML_ASSERT(pOffset >= 0 && n > 0 && pOffset + n <= vNTokens);
    lPosView.clear();
    lPosView.reserve((size_t) n * (size_t) vNPosPerEmbd);
    for (int vJ = 0; vJ < vNPosPerEmbd; vJ++) {
      const size_t cSrc = (size_t) vJ * (size_t) vNTokens + (size_t) pOffset;
      lPosView.insert(lPosView.end(), lPos.data() + cSrc, lPos.data() + cSrc + n);
    }
    return {
      /*n_tokens =*/n,
      /*embd     =*/vEmbd + (size_t) pOffset * vNMmprojEmbd,
      /*n_embd   =*/vNMmprojEmbd,
      /*pos      =*/lPosView.data(),
      /*n_pos    =*/vNPosPerEmbd,
      /*seq_id   =*/vSeqId,
    };
  }

  // render the entries [offset, offset + n) into a batch owned by this object, ready for dgs_process()
  dgs_batch_ext * mRender(dgs_context * pLctx, int pOffset, int n) {
    GGML_ASSERT(pOffset >= 0 && n > 0 && pOffset + n <= vNTokens);
    if (!vBatch) {
      vBatch.reset(fDgsBatchExtInit(pLctx));
    }
    fDgsBatchExtClear(vBatch.get());
    for (int vI = pOffset; vI < pOffset + n; vI++) {
      const dgs_embd cE   = { vEmbd + (size_t) vI * vNMmprojEmbd, 1, (size_t) vNMmprojEmbd };
      const int32_t  cIdx = fDgsBatchExtAddEmbd(vBatch.get(), vSeqId, cE);
      GGML_ASSERT(cIdx >= 0);

      dgs_pos aP[GGML_MROPE_SECTIONS] = { 0, 0, 0, 0 };
      for (int vJ = 0; vJ < vNPosPerEmbd; vJ++) {
        aP[vJ] = lPos[(size_t) vJ * (size_t) vNTokens + (size_t) vI];
      }
      fDgsBatchExtSetPos(vBatch.get(), cIdx, aP);

      if (lLogits[vI]) {
        fDgsBatchExtSetOutputLogits(vBatch.get(), cIdx, true);
      }
    }
    return vBatch.get();
  }
};
