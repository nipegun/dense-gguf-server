#pragma once

#include "ggml.h"
#include "mtmd.h"

#include <stddef.h>
#include <stdint.h>

#include <map>

// !!! Internal header, to be used by mtmd only !!!

#define MTMD_INTERNAL_HEADER

struct clip_ctx;

struct clip_image_size {
  int vWidth;
  int vHeight;

  bool operator==(const clip_image_size & pOther) const { return vWidth == pOther.vWidth && vHeight == pOther.vHeight; }

  bool operator!=(const clip_image_size & pOther) const { return !(*this == pOther); }

  int mArea() const {
    // avoid overflow when computing area
    GGML_ASSERT(vWidth >= 0 && vWidth <= 46000);
    GGML_ASSERT(vHeight >= 0 && vHeight <= 46000);
    return vWidth * vHeight;
  }
};

struct clip_image_f32;
struct clip_image_f32_batch;

enum clip_modality {
  cClipModalityVision,
  cClipModalityAudio,
  cClipModalityGenAudio,
};

enum clip_flash_attn_type {
  cClipFlashAttnTypeAuto     = -1,
  cClipFlashAttnTypeDisabled = 0,
  cClipFlashAttnTypeEnabled  = 1,
};

struct clip_context_params {
  bool                             vUseGpu;
  ggml_backend_dev_t               vDevice;
  enum clip_flash_attn_type        vFlashAttnType;
  int                              vImageMinTokens;
  int                              vImageMaxTokens;
  bool                             vWarmup;
  ggml_backend_sched_eval_callback vCbEval;
  void *                           vCbEvalUserData;
  bool                             vNoAlloc;
  mtmd_progress_callback           vProgressCallback;
  void *                           vProgressCallbackUserData;
};

struct clip_init_result {
  struct clip_ctx * vCtxV;     // vision context
  struct clip_ctx * vCtxA;     // audio context
  struct clip_ctx * vCtxGenA;  // audio generation context
};

struct clip_init_result fClipInit(const char * pFname, struct clip_context_params pCtxParams);

// max number of output tokens per image, -1 if not dynamic size
int fClipGetImageMaxTokens(const struct clip_ctx * pCtx);

void fClipFree(struct clip_ctx * pCtx);

// TODO: should be enum, not string
const char * fClipPatchMergeType(const struct clip_ctx * pCtx);

int fClipNOutputTokens(const clip_ctx * pCtx, const clip_image_f32 * pImg);

// for M-RoPE, this will be the number of token positions in X and Y directions
// for other models, X will be the total number of tokens and Y will be 1
int fClipNOutputTokensX(const clip_ctx * pCtx, const clip_image_f32 * pImg);
int fClipNOutputTokensY(const clip_ctx * pCtx, const clip_image_f32 * pImg);

// this should be equal to the embedding dimension of the text model
int fClipNMmprojEmbd(const struct clip_ctx * pCtx);

// TODO: remove clip_image_encode() and always use batched version
bool fClipImageEncode(struct clip_ctx * pCtx, int pNThreads, const clip_image_f32 * pImg, std::vector<float> & pOutVec);
bool fClipImageBatchEncode(struct clip_ctx *                   pCtx,
                           int                                 pNThreads,
                           const struct clip_image_f32_batch * pImgs,
                           std::vector<float> &                pOutBatchEmbd);

enum clip_gen_process_type {
  cClipGenProcessGenUnknown,
  cClipGenProcessGenCode,  // h_state to codes
  cClipGenProcessGenWav,   // codes to raw PCM audio
};

struct clip_encode_params {
  int                          vNThreads = 1;
  const clip_image_f32_batch * vImgs     = nullptr;
  std::vector<float> *         lOutEmbd  = nullptr;

  // for audio gen, imgs has exactly one entry: hidden state from backbone (GEN_CODE) or unused (GEN_WAV)
  clip_gen_process_type vGenProcess = cClipGenProcessGenUnknown;

  // GEN_CODE: out_embd receives the embd to feed back to the backbone
  int32_t                vCode0    = 0;  // semantic code sampled by the backbone
  int32_t                vTopK     = 50;
  float                  vTopP     = 1.0f;
  std::vector<int32_t> * lOutCodes = nullptr;     // this frame's 16 sampled codes
  std::vector<float> *   lOutFeats = nullptr;     // continuous counterpart of out_codes
  uint32_t               vSeed     = UINT32_MAX;  // UINT32_MAX for random
  float                  vTemp     = 0.0f;        // sampling temperature, noise scale for flow-matching decoders
  bool *                 vOutIsEos = nullptr;

  // GEN_WAV
  const std::vector<int32_t> * lCodes    = nullptr;  // this frame's 16 RVQ codes
  const std::vector<float> *   lFeats    = nullptr;  // continuous counterpart of codes
  std::vector<float> *         lOutAudio = nullptr;  // decoded PCM samples, F32
  const std::vector<uint8_t> * lStateIn  = nullptr;  // state from previous call, null or wrong size means cold start
  std::vector<uint8_t> *       lStateOut = nullptr;  // state for the next call
};

bool fClipEncode(struct clip_ctx * pCtx, struct clip_encode_params * pParams);

bool fClipIsLlava(const struct clip_ctx * pCtx);
// note for contributor: this clip_is_(model) pattern is deprecated
//                       do NOT add new functions like this

bool fClipHasVisionEncoder(const struct clip_ctx * pCtx);
bool fClipHasAudioEncoder(const struct clip_ctx * pCtx);

bool fClipSupportBatch(const struct clip_ctx * pCtx);

int fClipModelNTemporalMerge(const struct clip_ctx * pCtx);  // TODO @ngxson : remove, refactor this

std::map<ggml_backend_dev_t, size_t> fClipGetMemUsage(const struct clip_ctx * pCtx);

struct clip_cap {
  bool vHasVision;
  bool vHasAudio;
};
struct clip_cap fClipGetCap(const char * pFname);
