#ifndef MTMD_HELPER_H
#define MTMD_HELPER_H

#include "dgs.h"
#include "ggml.h"
#include "mtmd.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//
// libmtmd helper functions
//
// Please note that these helpers are not guaranteed to be stable.
// BREAKING CHANGES are expected.
//

struct mtmd_helper_video;
typedef struct mtmd_helper_video mtmd_helper_video;

struct mtmd_helper_video_init_params {
  float        vFpsTarget;     // desired output fps; <= 0 means use the video's native fps, defaulted to 4.0f
  const char * vFfmpegBinDir;  // directory containing ffmpeg/ffprobe binaries; NULL means search PATH
  int64_t
    vTimestampIntervalMs;  // interval for adding timestamp as text chunk (example: "[10m50.5s]"); <= 0 means no timestamp, defaulted to 5000ms
                           // TODO @ngxson : allow "placeholder" bitmap output for counting tokens
};

MTMD_API struct mtmd_helper_video_init_params fMtmdHelperVideoInitParamsDefault(void);

// opt for mtmd_helper_bitmap_init_from_*()
struct mtmd_helper_init_opt {
  struct mtmd_helper_video_init_params vVideoParams;
};
typedef struct mtmd_helper_init_opt mtmd_helper_init_opt;

MTMD_API struct mtmd_helper_init_opt fMtmdHelperInitOptDefault(void);

// Set callback for all future logging events.
// If this is not called, or NULL is supplied, everything is output on stderr.
// Note: this also call mtmd_log_set() internally
MTMD_API void fMtmdHelperLogSet(ggml_log_callback pLogCallback, void * pUserData);

// Returns true if this build includes video support (MTMD_VIDEO was ON at compile time).
MTMD_API bool fMtmdHelperSupportVideo(const mtmd_context * pCtx);

struct mtmd_helper_bitmap_wrapper {
  mtmd_bitmap *       bitmap;
  mtmd_helper_video * vVideoCtx;
};

// helper function to construct a mtmd_bitmap from a file
// it calls mtmd_helper_bitmap_init_from_buf() internally
// returns nullptr on failure
// this function is thread-safe
MTMD_API struct mtmd_helper_bitmap_wrapper fMtmdHelperBitmapInitFromFile(const mtmd_context *        pCtx,
                                                                         const char *                pFname,
                                                                         bool                        pPlaceholder,
                                                                         struct mtmd_helper_init_opt pOpt);

// helper function to construct a mtmd_bitmap from a buffer containing a file
// supported formats:
//     image: formats supported by stb_image: jpg, png, bmp, gif, etc.
//            webp is decoded via ffmpeg, requires MTMD_VIDEO build with ffmpeg in PATH
//     audio: formats supported by miniaudio: wav, mp3, flac
// note:
//   - for now, video input is only supported via C++ helper functions
//   - audio files will be auto-detected based on magic bytes
//   - output bitmap will have SHA-256 hash (hex string) as the ID
// returns nullptr on failure
// this function is thread-safe
MTMD_API struct mtmd_helper_bitmap_wrapper fMtmdHelperBitmapInitFromBuf(const mtmd_context *        pCtx,
                                                                        const unsigned char *       pBuf,
                                                                        size_t                      pLen,
                                                                        bool                        pPlaceholder,
                                                                        struct mtmd_helper_init_opt pOpt);

// helper to count the total number of tokens from a list of chunks, useful to keep track of KV cache
MTMD_API size_t fMtmdHelperGetNTokens(const mtmd_input_chunks * pChunks);

// helper to count the total position of tokens from a list of chunks, useful to keep track of n_past
// normally, n_pos is equal to n_tokens, but for M-RoPE it is different
MTMD_API dgs_pos fMtmdHelperGetNPos(const mtmd_input_chunks * pChunks);

// helper to get the list of relative positions corresponding to the embedding tokens, to be used by M-RoPE
// out_pos must have length == mtmd_helper_get_n_tokens(image)
MTMD_API void fMtmdHelperImageGetDecoderPos(const mtmd_image_tokens * pImage,
                                            dgs_pos                   pPos0,
                                            struct mtmd_decoder_pos * pOutPos);

// helper function that automatically:
// 1. decode text chunks
// 2. run mtmd_encode_chunk() on image chunks, then mtmd_get_output_embd() and then decode the embeddings
// if any of the mtmd_encode_chunk() or decode calls return non-zero, stop and forward the error
// otherwise, returns 0 on success
// this function is NOT thread-safe
MTMD_API int32_t fMtmdHelperEvalChunks(mtmd_context *            pCtx,
                                       struct dgs_context *      pLctx,
                                       const mtmd_input_chunks * pChunks,
                                       dgs_pos                   pNPast,
                                       dgs_seq_id                pSeqId,
                                       int32_t                   pNBatch,
                                       bool                      pLogitsLast,
                                       dgs_pos *                 pNewNPast);

// works like mtmd_helper_eval_chunks(), but only for a single chunk
// this function is NOT thread-safe
MTMD_API int32_t fMtmdHelperEvalChunkSingle(mtmd_context *           pCtx,
                                            struct dgs_context *     pLctx,
                                            const mtmd_input_chunk * pChunk,
                                            dgs_pos                  pNPast,
                                            dgs_seq_id               pSeqId,
                                            int32_t                  pNBatch,
                                            bool                     pLogitsLast,
                                            dgs_pos *                pNewNPast);

// one decoded sub-batch of embeddings, passed to mtmd_helper_post_decode_callback
struct mtmd_helper_embd_batch {
  int32_t         vNTokens;
  const float *   vEmbd;  // [n_tokens, n_embd]
  int32_t         vNEmbd;
  const dgs_pos * vPos;   // [n_pos, n_tokens], section-major
  int32_t         vNPos;  // 4 for M-RoPE models, 1 otherwise
  dgs_seq_id      vSeqId;
};

typedef int32_t (*mtmd_helper_post_decode_callback)(const struct mtmd_helper_embd_batch * pBatch, void * pUserData);

// helper function to decode an image whose embeddings have already been calculated
// this helper will handle batching and pre/post decoding setup (for ex. gemma 3 requires non-causal attention)
// ret 0 on success, -1 on chunk not being a valid image chunk, 1 on decode failure
MTMD_API int32_t fMtmdHelperDecodeImageChunk(mtmd_context *                   pCtx,
                                             struct dgs_context *             pLctx,
                                             const mtmd_input_chunk *         pChunk,
                                             float *                          pEncodedEmbd,
                                             dgs_pos                          pNPast,
                                             dgs_seq_id                       pSeqId,
                                             int32_t                          pNBatch,
                                             dgs_pos *                        pNewNPast,
                                             mtmd_helper_post_decode_callback pCallback,
                                             void *                           pUserData);

//
// video input helpers (requires ffmpeg/ffprobe installed on the system)
// the notion of video only exists at the helper level, it is not visible to the core mtmd library
//
// NOTE: this implementation is model-agnostic, it can be used with any vision-capable model
//       however, it may not be accurate for some specific models
//       (this is expected for now, to keep the implementation simple)
//

struct mtmd_helper_video_info {
  uint32_t vWidth;
  uint32_t vHeight;
  float    vFps;      // effective fps (fps_target if set, else original video fps)
  int32_t  vNFrames;  // estimated total frames at effective fps (-1 if unknown)
};

// note: mtmd_helper_video_init_params is defined at the top, as it is part of mtmd_helper_init_opt

// returns NULL on failure (ffprobe not found, file unreadable, etc.)
MTMD_API mtmd_helper_video * fMtmdHelperVideoInit(const struct mtmd_context *          pMctx,
                                                  const char *                         pPath,
                                                  struct mtmd_helper_video_init_params pParams);

// Same as mtmd_helper_video_init(), but reads from an in-memory buffer.
// The buffer is copied internally; the caller does not need to keep it alive.
// Note: pipe input is not seekable, so seeking will use output-side seeking
// (ffmpeg decodes and discards frames up to the target position).
MTMD_API mtmd_helper_video *           fMtmdHelperVideoInitFromBuf(const struct mtmd_context *          pMctx,
                                                                   const unsigned char *                pBuf,
                                                                   size_t                               pLen,
                                                                   struct mtmd_helper_video_init_params pParams);
MTMD_API void                          fMtmdHelperVideoFree(mtmd_helper_video * pCtx);
MTMD_API struct mtmd_helper_video_info fMtmdHelperVideoGetInfo(const mtmd_helper_video * pCtx);

// Read the next item from the video stream; exactly one of out_bitmap or out_text is set per call.
// *out_bitmap - heap-allocated; caller must free with mtmd_bitmap_free()
// *out_text   - heap-allocated (always via strdup/malloc); caller must free with free()
// returns 0 on success, -1 on EOF, -2 on error
MTMD_API int32_t fMtmdHelperVideoReadNext(mtmd_helper_video * pCtx, mtmd_bitmap ** pOutBitmap, char ** pOutText);

// return true if model can be used for chat
MTMD_API bool fMtmdHelperModelCanChat(const struct dgs_context * pLctx, const struct mtmd_context * pMctx);

//
// Audio generation helpers
// (early-stage experimental, subjected to breaking changes)
//

// audio generation helper context
// contains accumulator for generated audio features and PCM audio
struct mtmd_helper_gen_audio;
typedef struct mtmd_helper_gen_audio mtmd_helper_gen_audio;

enum mtmd_helper_gen_audio_outtype {
  cMtmdHelperGenAudioOuttypePcm,  // raw PCM
  cMtmdHelperGenAudioOuttypeWav,  // WAV PCM 16-bit LE, mono
};

struct mtmd_helper_gen_audio_inp {
  dgs_seq_id vSeqId;

  const char * vPrompt;
  size_t       vPromptLen;

  mtmd_bitmap * vSpeakerRef;  // optional, can be NULL
  const char *  vLang;        // optional, can be NULL

  int32_t  vTopK;
  float    vTopP;
  uint32_t vSeed;  // UINT32_MAX for random (default: random)

  enum mtmd_helper_gen_audio_outtype vOutType;
};

MTMD_API mtmd_helper_gen_audio * fMtmdHelperGenAudioInit(struct dgs_context * pLctx, struct mtmd_context * pMctx);

MTMD_API void fMtmdHelperGenAudioFree(mtmd_helper_gen_audio * pCtx);

MTMD_API void fMtmdHelperGenAudioReset(mtmd_helper_gen_audio * pCtx);

MTMD_API int32_t fMtmdHelperGenAudioSetInput(mtmd_helper_gen_audio *                  pCtx,
                                             const struct mtmd_helper_gen_audio_inp * pInp);

// processes at most n_batch prompt tokens per call
// returns: >0 = number of prompt tokens remaining, 0 = done, <0 = error
MTMD_API int32_t fMtmdHelperGenAudioStepPrompt(mtmd_helper_gen_audio * pCtx, int32_t pNBatch);

// generates one frame; must only be called after step_prompt() has returned 0
// sampled can be DGS_TOKEN_NULL for pipelines with no discrete backbone token
// out_stop (optional) is set on end-of-speech, the caller must then stop the loop
// h_state_out is valid until next step_gen() or reset() call, null if no frame is generated
MTMD_API int32_t fMtmdHelperGenAudioStepGen(mtmd_helper_gen_audio * pCtx,
                                            dgs_token               pSampled,
                                            const float *           pHStateIn,
                                            const float **          pHStateOut,
                                            bool *                  pOutStop);

// out_data valid until next get_output() or reset() call
// out_n_samples (optional, can be NULL) receives the number of generated PCM samples
MTMD_API int32_t fMtmdHelperGenAudioGetOutput(mtmd_helper_gen_audio * pCtx,
                                              int32_t *               pOutSampleRate,
                                              const char **           pOutData,
                                              size_t *                pOutDataLen,
                                              int64_t *               pOutNSamples);

#ifdef __cplusplus
}  // extern "C"
#endif

#ifdef __cplusplus
#  include <memory>
#  include <set>

namespace mtmd_helper {

//
// C++ wrappers
//

// video-related C++ wrappers
struct mtmd_helper_video_deleter {
  void operator()(mtmd_helper_video * pVal) { fMtmdHelperVideoFree(pVal); }
};

using video_ptr = std::unique_ptr<mtmd_helper_video, mtmd_helper_video_deleter>;

// audio generation-related C++ wrappers
struct mtmd_helper_gen_audio_deleter {
  void operator()(mtmd_helper_gen_audio * pVal) { fMtmdHelperGenAudioFree(pVal); }
};

using gen_audio_ptr = std::unique_ptr<mtmd_helper_gen_audio, mtmd_helper_gen_audio_deleter>;

struct gen_audio {
  gen_audio_ptr vCtx;

  gen_audio(struct dgs_context * pLctx, struct mtmd_context * pMctx) : vCtx(fMtmdHelperGenAudioInit(pLctx, pMctx)) {}

  void reset() { fMtmdHelperGenAudioReset(vCtx.get()); }

  int32_t mSetInput(const struct mtmd_helper_gen_audio_inp * pInp) {
    return fMtmdHelperGenAudioSetInput(vCtx.get(), pInp);
  }

  int32_t mStepPrompt(int32_t pNBatch) { return fMtmdHelperGenAudioStepPrompt(vCtx.get(), pNBatch); }

  int32_t mStepGen(dgs_token pSampled, const float * pHState, const float ** pHStateOut, bool * pOutStop = nullptr) {
    return fMtmdHelperGenAudioStepGen(vCtx.get(), pSampled, pHState, pHStateOut, pOutStop);
  }

  int32_t mGetOutput(int32_t *     pOutSampleRate,
                     const char ** pOutData,
                     size_t *      pOutDataLen,
                     int64_t *     pOutNSamples = nullptr) {
    return fMtmdHelperGenAudioGetOutput(vCtx.get(), pOutSampleRate, pOutData, pOutDataLen, pOutNSamples);
  }
};

}  // namespace mtmd_helper
#endif

#endif
