#ifndef MTMD_H
#define MTMD_H

#include "dgs.h"
#include "ggml.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#  include <cinttypes>
#  include <map>
#  include <memory>
#  include <string>
#  include <vector>
#endif

/**
 * libmtmd: A library for multimodal support in dgs.cpp.
 *
 * WARNING: This API is experimental and subject to many BREAKING CHANGES.
 *          Issues related to API usage may receive lower priority support.
 *
 * For the usage, see an example in mtmd-cli.cpp
 *
 * For contributors:
 * - Make sure the C API is aligned with the libdgs C API (as in dgs.h)
 * - Do not include model name (e.g., qwen, gemma) in the API, use generic terms instead
 * - Keep the API minimal, do not expose internal details unless necessary
 *
 * IMPORTANT: The mtmd module does NOT accept pull requests that are fully or predominantly AI-generated.
 * We encourage human contributors to ensure the quality and reliability of the codebase.
 */

#ifdef DGS_SHARED
#  if defined(_WIN32) && !defined(__MINGW32__)
#    ifdef DGS_BUILD
#      define MTMD_API __declspec(dllexport)
#    else
#      define MTMD_API __declspec(dllimport)
#    endif
#  else
#    define MTMD_API __attribute__((visibility("default")))
#  endif
#else
#  define MTMD_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum mtmd_input_chunk_type {
  cMtmdInputChunkTypeText,
  cMtmdInputChunkTypeImage,
  cMtmdInputChunkTypeAudio,
  cMtmdInputChunkTypeCount,  // for validation
};

// opaque types
struct mtmd_context;
struct mtmd_bitmap;
struct mtmd_image_tokens;
struct mtmd_input_chunk;
struct mtmd_input_chunks;
struct mtmd_batch;

struct mtmd_input_text {
  const char * vText;
  size_t       vTextLen;
  bool         vAddSpecial;
  bool         vParseSpecial;
};

struct mtmd_input_part {
  // only text or bitmap can be set, not both
  const struct mtmd_input_text * vText;
  const struct mtmd_bitmap *     bitmap;
};

//
// C API
//

typedef struct mtmd_context      mtmd_context;
typedef struct mtmd_bitmap       mtmd_bitmap;
typedef struct mtmd_image_tokens mtmd_image_tokens;
typedef struct mtmd_input_chunk  mtmd_input_chunk;
typedef struct mtmd_input_chunks mtmd_input_chunks;
typedef struct mtmd_input_text   mtmd_input_text;
typedef struct mtmd_input_part   mtmd_input_part;
typedef struct mtmd_batch        mtmd_batch;

typedef bool (*mtmd_progress_callback)(float pProgress, void * pUserData);

struct mtmd_context_params {
  bool                     vUseGpu;
  ggml_backend_dev_t       vDevice;
  bool                     vPrintTimings;
  int                      vNThreads;
  const char *             vMediaMarker;
  enum dgs_flash_attn_type vFlashAttnType;
  bool                     vWarmup;  // whether to run a warmup encode pass after initialization

  // limit number of image tokens, only for vision models with dynamic resolution
  int vImageMinTokens;  // minimum number of tokens for image input (default: read from metadata)
  int vImageMaxTokens;  // maximum number of tokens for image input (default: read from metadata)

  // callback function passed over to mtmd proper
  ggml_backend_sched_eval_callback vCbEval;
  void *                           vCbEvalUserData;

  // batching params
  int32_t vBatchMaxTokens;  // maximum number of output tokens in a batch
  // (note: this is not a hard-limit, the first image will always be added even if it exceeds this limit)
  // (default: 1024)

  // Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
  // If the provided progress_callback returns true, model loading continues.
  // If it returns false, model loading is immediately aborted.
  mtmd_progress_callback vProgressCallback;
  void *                 vProgressCallbackUserData;
};

MTMD_API const char * fMtmdDefaultMarker(void);

MTMD_API struct mtmd_context_params fMtmdContextParamsDefault(void);

// initialize the mtmd context
// return nullptr on failure
MTMD_API mtmd_context * fMtmdInitFromFile(const char *                     pMmprojFname,
                                          const struct dgs_model *         pTextModel,
                                          const struct mtmd_context_params pCtxParams);

MTMD_API void fMtmdFree(mtmd_context * pCtx);

// whether we need to set non-causal mask before dgs_decode
// if chunk is nullptr, we assume the default case where chunk is an image chunk
MTMD_API bool fMtmdDecodeUseNonCausal(const mtmd_context * pCtx, const mtmd_input_chunk * pChunk);

// whether the current model use M-RoPE for dgs_decode
MTMD_API bool fMtmdDecodeUseMrope(const mtmd_context * pCtx);

// whether the current model supports vision input
MTMD_API bool fMtmdSupportVision(const mtmd_context * pCtx);

// whether the current model supports audio input
MTMD_API bool fMtmdSupportAudio(const mtmd_context * pCtx);

// get audio sample rate in Hz, for example 16000 for Whisper
// return -1 if audio is not supported
MTMD_API int fMtmdGetAudioSampleRate(const mtmd_context * pCtx);

// get the current marker string
MTMD_API const char * fMtmdGetMarker(const mtmd_context * pCtx);

// mtmd_bitmap
//
// if bitmap is image:
//     length of data must be nx * ny * 3
//     the data is in RGBRGBRGB... format
//     note: some video-capable models (i.e. qwen-vl) can merge consecutive bitmaps
//           into one chunk; mtmd_tokenize() handles this, but remember to set
//           mtmd_bitmap_set_mergeable(true) for every frame
// if bitmap is audio:
//     length of data must be n_samples * sizeof(float)
//     the data is in float format (PCM F32)
//
// if data == nullptr:
//     the bitmap is considered "empty", and will be treated as a placeholder for counting tokens
//     you can pass the bitmap via mtmd_tokenize(), then call mtmd_*_get_n_tokens() to count the tokens
//     note: passing a placeholder bitmap to mtmd_encode() will return an error
MTMD_API mtmd_bitmap *         fMtmdBitmapInit(uint32_t pNx, uint32_t pNy, const unsigned char * data);
MTMD_API mtmd_bitmap *         fMtmdBitmapInitFromAudio(size_t pNSamples, const float * data);
MTMD_API uint32_t              fMtmdBitmapGetNx(const mtmd_bitmap * bitmap);
MTMD_API uint32_t              fMtmdBitmapGetNy(const mtmd_bitmap * bitmap);
MTMD_API const unsigned char * fMtmdBitmapGetData(const mtmd_bitmap * bitmap);
MTMD_API size_t                fMtmdBitmapGetNBytes(const mtmd_bitmap * bitmap);
MTMD_API bool                  fMtmdBitmapIsAudio(const mtmd_bitmap * bitmap);
MTMD_API void                  fMtmdBitmapFree(mtmd_bitmap * bitmap);
// bitmap ID is optional, but useful for KV cache tracking
// these getters/setters are dedicated functions, so you can for example calculate the hash of the image based on mtmd_bitmap_get_data()
MTMD_API const char *          fMtmdBitmapGetId(const mtmd_bitmap * bitmap);
MTMD_API void                  fMtmdBitmapSetId(mtmd_bitmap * bitmap, const char * pId);
// if true, this bitmap can be merged (temporal merge) with an adjacent mergeable bitmap by certain video input models
MTMD_API void                  fMtmdBitmapSetMergeable(mtmd_bitmap * bitmap, bool pMergeable);

// mtmd_bitmap lazy
//
// this is a special bitmap that:
// - does not hold the actual data
// - can be expanded into one or more chunks (either media to text chunks)
// user must provide a callback to fill in the data when mtmd_tokenize() is called
// this is useful for large video inputs:
// - allow reading video frame by frame, without loading the entire video into memory
// - allow tracking the whole video with a single ID (for example, the file hash)

// set (*out_bitmap) to non-nullptr to emit a bitmap chunk; it will be freed automatically
// set (*out_text) to non-nullptr to emit a text chunk; it must be heap-allocated, null-terminated and will be freed automatically
// either out_bitmap or out_text can be set, but not both
// out_bitmap cannot be another lazy bitmap (no nested lazy allowed)
// return value:
//    0 on success
//   -1 on EOF (signal to mtmd_tokenize to move on)
//   -2 on error (signal to mtmd_tokenize to abort)
typedef int (*mtmd_bitmap_lazy_callback)(size_t         pChunkIdx,
                                         void *         pUserData,
                                         mtmd_bitmap ** pOutBitmap,
                                         char **        pOutText);

MTMD_API mtmd_bitmap * fMtmdBitmapInitLazy(const mtmd_context *      pCtx,
                                           const char *              pId,  // usually set to file hash
                                           void *                    pUserData,
                                           mtmd_bitmap_lazy_callback pCallback);

// mtmd_input_chunks
//
// this is simply a list of mtmd_input_chunk
// the elements can only be populated via mtmd_tokenize()
MTMD_API mtmd_input_chunks *      fMtmdInputChunksInit(void);
MTMD_API size_t                   fMtmdInputChunksSize(const mtmd_input_chunks * pChunks);
MTMD_API const mtmd_input_chunk * fMtmdInputChunksGet(const mtmd_input_chunks * pChunks, size_t pIdx);
MTMD_API void                     fMtmdInputChunksFree(mtmd_input_chunks * pChunks);

// mtmd_input_chunk
//
// the instance will be constructed via mtmd_tokenize()
// it will be freed along with mtmd_input_chunks
MTMD_API enum mtmd_input_chunk_type fMtmdInputChunkGetType(const mtmd_input_chunk * pChunk);
MTMD_API const dgs_token * fMtmdInputChunkGetTokensText(const mtmd_input_chunk * pChunk, size_t * pNTokensOutput);
MTMD_API const mtmd_image_tokens * fMtmdInputChunkGetTokensImage(const mtmd_input_chunk * pChunk);
MTMD_API size_t                    fMtmdInputChunkGetNTokens(const mtmd_input_chunk * pChunk);
// returns nullptr for ID on text chunk
MTMD_API const char *              fMtmdInputChunkGetId(const mtmd_input_chunk * pChunk);
// number of temporal positions (equals to max(t,h,w) for M-RoPE; equals to n_tokens otherwise)
MTMD_API dgs_pos                   fMtmdInputChunkGetNPos(const mtmd_input_chunk * pChunk);

// in case you want to use custom logic to handle the chunk (i.e. KV cache management)
// you can move the chunk ownership to your own code by copying it
// remember to free the chunk when you are done with it
MTMD_API mtmd_input_chunk * fMtmdInputChunkCopy(const mtmd_input_chunk * pChunk);
MTMD_API void               fMtmdInputChunkFree(mtmd_input_chunk * pChunk);

// similar to mtmd_input_chunk_copy, but returns a placeholder chunk
MTMD_API mtmd_input_chunk * fMtmdInputChunkGetPlaceholder(const mtmd_input_chunk * pChunk);

// save/load an input chunk to/from a buffer (useful for KV save/load)
// important: only chunk's metadata will be saved, the actual image/audio data will not be saved
// the loaded chunk will always be a placeholder, cannot be used for mtmd_encode() or mtmd_batch_encode()
// out_buf can be nullptr (to query expected_out_len)
// returns 0 on success, non-zero on failure
MTMD_API int32_t            fMtmdInputChunkSave(const mtmd_input_chunk * pChunk,
                                                char *                   pOutBuf,
                                                size_t                   pOutLen,
                                                size_t *                 pExpectedOutLen);
// returns nullptr on failure
MTMD_API mtmd_input_chunk * fMtmdInputChunkLoad(const char * pBuf, size_t pLen);

// mtmd_image_tokens
//
// the instance will be constructed via mtmd_tokenize()
// it will be freed along with mtmd_input_chunk
MTMD_API size_t       fMtmdImageTokensGetNTokens(const mtmd_image_tokens * pImageTokens);  // TODO: deprecate
MTMD_API const char * fMtmdImageTokensGetId(const mtmd_image_tokens * pImageTokens);       // TODO: deprecate
// number of temporal positions (equals to max(t,h,w) for M-RoPE; equals to n_tokens otherwise)
MTMD_API dgs_pos      fMtmdImageTokensGetNPos(const mtmd_image_tokens * pImageTokens);  // TODO: deprecate

struct mtmd_decoder_pos {
  uint32_t vT;
  uint32_t vX;
  uint32_t vY;
  uint32_t vZ;  // unused for now, reserved for future use
};
// get position for decoder attention, to be used by M-RoPE models
// i is the index of the embedding token, ranging from 0 to mtmd_image_tokens_get_n_tokens() - 1
// pos_0 is the absolute position of the first token
// return relative position (for example, embedding 0 will have position (0, 0, 0); remember to adjust it to the current absolute position)
MTMD_API struct mtmd_decoder_pos fMtmdImageTokensGetDecoderPos(const mtmd_image_tokens * pImageTokens,
                                                               dgs_pos                   pPos0,
                                                               size_t                    pI);

// tokenize an input text prompt and a list of bitmaps (image/audio)
// the prompt must have the input media marker (default: "<__media__>") in it
// the default marker is defined by mtmd_default_marker()
// the marker will be replaced with the media chunk
// for example:
//   "here is an image: <__media__>\ndescribe it in detail."
//   this will gives 3 chunks:
//   1. "here is an image: <start_of_image>"
//   2. (image/audio tokens)
//   3. "<end_of_image>\ndescribe it in detail."
// number of bitmaps must be equal to the number of markers in the prompt
// this function is thread-safe (shared ctx)
// return values:
//   0 on success
//   1 on number of bitmaps not matching the number of markers
//   2 on media preprocessing error
MTMD_API int32_t fMtmdTokenize(const mtmd_context *        pCtx,
                               mtmd_input_chunks *         pOutput,
                               const mtmd_input_text *     pText,
                               const mtmd_bitmap * const * bitmaps,
                               size_t                      pNBitmaps);

// same as mtmd_tokenize(), but takes an array of mtmd_input_part
// use cases:
// - when you don't want to use media markers (they will be tokenized as normal text)
// - when you want to control parse_special for each text part
// note: per-part add_special will be ignored
// return 1 if a part has both text and bitmap set (or neither)
MTMD_API int32_t fMtmdTokenizeFromParts(const mtmd_context *            pCtx,
                                        mtmd_input_chunks *             pOutput,
                                        const mtmd_input_part * const * pParts,
                                        size_t                          pNParts,
                                        bool                            pAddSpecial);

// text chunk will be ignored silently, only media chunk will be encoded
// returns 0 on success
// returns 1 on generic error
MTMD_API int32_t fMtmdEncodeChunk(mtmd_context * pCtx, const mtmd_input_chunk * pChunk);

// get output embeddings from the last encode pass
// the reading size (in bytes) is equal to:
// dgs_model_n_embd_inp(model) * mtmd_input_chunk_get_n_tokens(chunk) * sizeof(float)
MTMD_API float * fMtmdGetOutputEmbd(mtmd_context * pCtx);

// batch encoding API
// chunks are not owned by the batch, they will not be freed by mtmd_batch_free()
// batch is valid for a given context, cannot be shared across contexts
MTMD_API mtmd_batch * fMtmdBatchInit(mtmd_context * pCtx);
MTMD_API void         fMtmdBatchFree(mtmd_batch * pBatch);

// only media chunks are allowed, text chunks will be rejected
// returns 0 on success
// returns 1 on generic error
// returns 2 if the batch is too large (chunk won't be added)
// returns 3 if it cannot be batched with the existing chunks in the batch
MTMD_API int32_t fMtmdBatchAddChunk(mtmd_batch * pBatch, const mtmd_input_chunk * pChunk);

// returns 0 on success
// returns 1 on generic error
MTMD_API int32_t fMtmdBatchEncode(mtmd_batch * pBatch);
MTMD_API float * fMtmdBatchGetOutputEmbd(mtmd_batch * pBatch, const mtmd_input_chunk * pChunk);

// Set callback for all future logging events.
// If this is not called, or NULL is supplied, everything is output on stderr.
MTMD_API void fMtmdLogSet(ggml_log_callback pLogCallback, void * pUserData);

// EXPERIMENTAL API to get mmproj's capabilities without initializing the full context
// This is only intended to be used by dgs-server, breaking changes is expected
struct mtmd_caps {
  bool vInpVision;
  bool vInpAudio;
};
MTMD_API struct mtmd_caps fMtmdGetCapFromFile(const char * pMmprojFname);

/////////////////////////////////////////
// EXPERIMENTAL API for audio generation, subjected to breaking changes

// represent the pipeline type
enum mtmd_gen_audio_type {
  cMtmdGenAudioTypeNone,  // not supported
  cMtmdGenAudioTypeQwen3tts,
  cMtmdGenAudioTypePockettts,
};

struct mtmd_gen_audio_info {
  enum mtmd_gen_audio_type type;
  int32_t                  vSampleRate;    // in Hz, for example 24000 for qwen3tts
  const char *             vModelVariant;  // name of the weight variant, can be nullptr if not applicable
};

MTMD_API struct mtmd_gen_audio_info fMtmdGenAudioGetInfo(const mtmd_context * pCtx);

enum mtmd_gen_process_type {
  cMtmdGenProcessTypeGenCode,  // h_state to semantic (codes, mel-spectrogram, etc.)
  cMtmdGenProcessTypeGenWav,   // convert semantic to PCM audio
                               // for qwen3tts, this is code2wav
                               // for pocket-tts, this is mimi decoder
};

struct mtmd_gen_inp {
  enum mtmd_gen_process_type type;

  // for MTMD_GEN_PROCESS_TYPE_GEN_CODE
  int32_t  vCode0;  // the sampled codebook 0 entry from backbone
  float *  vEmbd;   // the hidden state from backbone, must have n_text_embd elements
  int32_t  vTopK;
  float    vTopP;
  uint32_t vSeed;  // UINT32_MAX for random
  float    vTemp;  // sampling temperature, or noise scale for flow-matching decoders

  // for MTMD_GEN_PROCESS_TYPE_GEN_WAV
  // pass either codes (discrete) or feats (continuous), depending on the pipeline
  int32_t *     vCodes;
  size_t        vNCodes;
  const float * vFeats;
  size_t        vNFeats;
  const char *  vStateData;
  size_t        vStateSize;
};

struct mtmd_gen_out {
  // note: output memory is allocated by the context, valid until next process() call

  // for MTMD_GEN_PROCESS_TYPE_GEN_CODE
  const int32_t * vCodes;
  size_t          vNCodes;
  const float *   vFeats;  // continuous counterpart of codes
  size_t          vNFeats;
  const float *   vEmbd;   // the generated hidden state, to be fed back to backbone
                           // it must have n_text_embd elements
  bool            vIsEos;  // only set by pipelines having the EOS head inside mmproj

  // for MTMD_GEN_PROCESS_TYPE_GEN_WAV
  const float * vAudio;
  size_t        vNSamples;
  const char *  vStateData;
  size_t        vStateSize;
};

// defaults tuned for the loaded pipeline, callers override only what they care about
MTMD_API struct mtmd_gen_inp fMtmdGenInpDefault(const mtmd_context * pCtx);

// note: this API is stateless, caller must handle state management and audio frame accumulation
MTMD_API int32_t fMtmdGenAudioProcess(mtmd_context *              pCtx,
                                      const struct mtmd_gen_inp * pInp,
                                      struct mtmd_gen_out *       pOut);

/////////////////////////////////////////

// test function, to be used in test-mtmd-c-api.c
MTMD_API mtmd_input_chunks * fMtmdTestCreateInputChunks(void);

#ifdef __cplusplus
}  // extern "C"
#endif

// Get memory usage of the current model in bytes, per backend device
// Note: this is an unstable API, used internally by fit_params; it WILL be removed or changed without deprecation
#ifdef __cplusplus
struct mtmd_memory_usage {
  std::map<ggml_backend_dev_t, size_t> dBackendMemUsage;
  // for models that use non-causal attention, max_tokens must not exceed n_ubatch of dgs_context
  int                                  vImageMaxTokens;
  bool                                 vUseNonCausal;
};
MTMD_API struct mtmd_memory_usage fMtmdGetMemoryUsage(const char * pMmprojFname, struct mtmd_context_params pCtxParams);
#endif

//
// C++ wrappers
//

#ifdef __cplusplus

namespace mtmd {

struct mtmd_context_deleter {
  void operator()(mtmd_context * pVal) { fMtmdFree(pVal); }
};

using context_ptr = std::unique_ptr<mtmd_context, mtmd_context_deleter>;

struct mtmd_bitmap_deleter {
  void operator()(mtmd_bitmap * pVal) { fMtmdBitmapFree(pVal); }
};

using bitmap_ptr = std::unique_ptr<mtmd_bitmap, mtmd_bitmap_deleter>;

struct mtmd_input_chunks_deleter {
  void operator()(mtmd_input_chunks * pVal) { fMtmdInputChunksFree(pVal); }
};

using input_chunks_ptr = std::unique_ptr<mtmd_input_chunks, mtmd_input_chunks_deleter>;

struct mtmd_input_chunk_deleter {
  void operator()(mtmd_input_chunk * pVal) { fMtmdInputChunkFree(pVal); }
};

using input_chunk_ptr = std::unique_ptr<mtmd_input_chunk, mtmd_input_chunk_deleter>;

struct mtmd_batch_deleter {
  void operator()(mtmd_batch * pVal) { fMtmdBatchFree(pVal); }
};

using batch_ptr = std::unique_ptr<mtmd_batch, mtmd_batch_deleter>;

struct bitmap {
  bitmap_ptr vPtr;

  bitmap() : vPtr(nullptr) {}

  bitmap(mtmd_bitmap * bitmap) : vPtr(bitmap) {}

  bitmap(bitmap && pOther) noexcept : vPtr(std::move(pOther.vPtr)) {}

  bitmap(uint32_t pNx, uint32_t pNy, const unsigned char * data) { vPtr.reset(fMtmdBitmapInit(pNx, pNy, data)); }

  ~bitmap() = default;

  uint32_t mNx() const { return fMtmdBitmapGetNx(vPtr.get()); }

  uint32_t mNy() const { return fMtmdBitmapGetNy(vPtr.get()); }

  const unsigned char * data() const { return fMtmdBitmapGetData(vPtr.get()); }

  size_t mNBytes() const { return fMtmdBitmapGetNBytes(vPtr.get()); }

  std::string mId() const { return fMtmdBitmapGetId(vPtr.get()); }

  void mSetId(const char * pId) const { fMtmdBitmapSetId(vPtr.get(), pId); }
};

struct bitmaps {
  std::vector<bitmap> lEntries;
  ~bitmaps() = default;

  // return list of pointers to mtmd_bitmap
  // example:
  //   auto bitmaps_c_ptr = bitmaps.c_ptr();
  //   int32_t res = mtmd_tokenize(... bitmaps_c_ptr.data(), bitmaps_c_ptr.size());
  std::vector<const mtmd_bitmap *> mCPtr() {
    std::vector<const mtmd_bitmap *> lRes(lEntries.size());
    for (size_t vI = 0; vI < lEntries.size(); vI++) {
      lRes[vI] = lEntries[vI].vPtr.get();
    }
    return lRes;
  }
};

struct input_chunks {
  input_chunks_ptr vPtr;
  input_chunks() = default;

  input_chunks(mtmd_input_chunks * pChunks) : vPtr(pChunks) {}

  ~input_chunks() = default;

  size_t size() const { return fMtmdInputChunksSize(vPtr.get()); }

  const mtmd_input_chunk * operator[](size_t pIdx) const { return fMtmdInputChunksGet(vPtr.get(), pIdx); }
};

}  // namespace mtmd

#endif

#endif
