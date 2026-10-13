// fix problem with std::min and std::max
#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#endif

#include "dgs.h"
#include "hash/hash.h"
#include "mtmd-helper-common.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <algorithm>
#include <cinttypes>
#include <vector>

//#define MTMD_AUDIO_DEBUG

#define MINIAUDIO_IMPLEMENTATION
#ifndef MTMD_AUDIO_DEBUG
#  define MA_NO_ENCODING
#endif
#define MA_NO_DEVICE_IO
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MA_API static
#include "miniaudio/miniaudio.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#ifdef MTMD_INTERNAL_HEADER
#  error "mtmd-helper is a public library outside of mtmd. it must not include internal headers"
#endif

#ifdef MTMD_VIDEO
#  include "sheredom/subprocess.h"

#  include <thread>
#  ifndef _WIN32
#    include <fcntl.h>
#    include <pthread.h>

#    include <csignal>
#  endif
#endif

//
// internal logging functions
//

void fMtmdHelperLogSet(ggml_log_callback pLogCallback, void * pUserData) {
  if (pLogCallback == nullptr) {
    pLogCallback = g_logger.vDefaultCallback;
  }
  g_logger.vLogCallback         = pLogCallback;
  g_logger.vLogCallbackUserData = pUserData;
  fMtmdLogSet(pLogCallback, pUserData);
}

//
// helper functions
//

size_t fMtmdHelperGetNTokens(const mtmd_input_chunks * pChunks) {
  size_t vNTokens = 0;
  for (size_t vI = 0; vI < fMtmdInputChunksSize(pChunks); vI++) {
    auto vChunk = fMtmdInputChunksGet(pChunks, vI);
    vNTokens += fMtmdInputChunkGetNTokens(vChunk);
  }
  return vNTokens;
}

dgs_pos fMtmdHelperGetNPos(const mtmd_input_chunks * pChunks) {
  dgs_pos vNPos = 0;
  for (size_t vI = 0; vI < fMtmdInputChunksSize(pChunks); vI++) {
    auto vChunk = fMtmdInputChunksGet(pChunks, vI);
    vNPos += fMtmdInputChunkGetNPos(vChunk);
  }
  return vNPos;
}

void fMtmdHelperImageGetDecoderPos(const mtmd_image_tokens * pChunks, dgs_pos pPos0, mtmd_decoder_pos * pOutPos) {
  size_t vNTokens = fMtmdImageTokensGetNTokens(pChunks);
  for (size_t vI = 0; vI < vNTokens; vI++) {
    pOutPos[vI] = fMtmdImageTokensGetDecoderPos(pChunks, pPos0, vI);
  }
}

// Helper class to set non-causal attention via RAII
class scope_non_causal {
public:
  scope_non_causal(dgs_context * pContext, bool pEnabled) : vContext(pContext), vEnabled(pEnabled) {
    if (vEnabled) {
      // TODO @ngxson : need to make sure only one image is processed at a time, and n_ubatch must be enough to hold the image
      fDgsSetCausalAttn(vContext, false);
    }
  }

  ~scope_non_causal() {
    if (vEnabled) {
      fDgsSetCausalAttn(vContext, true);
    }
  }

  scope_non_causal(const scope_non_causal &)             = delete;
  scope_non_causal & operator=(const scope_non_causal &) = delete;

private:
  dgs_context * vContext;
  bool          vEnabled;
};

// Helper function for decoding an image whose embeddings have already been calculated
int32_t fMtmdHelperDecodeImageChunk(mtmd_context *                   pCtx,
                                    struct dgs_context *             pLctx,
                                    const mtmd_input_chunk *         pChunk,
                                    float *                          pEncodedEmbd,
                                    dgs_pos                          pNPast,
                                    dgs_seq_id                       pSeqId,
                                    int32_t                          pNBatch,
                                    dgs_pos *                        pNewNPast,
                                    mtmd_helper_post_decode_callback pCallback,
                                    void *                           pUserData) {
  GGML_ASSERT(pNBatch > 0);
  auto         vChunkType = fMtmdInputChunkGetType(pChunk);
  const char * name       = vChunkType == cMtmdInputChunkTypeImage ? "image" : "audio";
  if (vChunkType == cMtmdInputChunkTypeText) {
    LOG_ERR("failed to decode chunk: input chunk not of image/audio type\n");
    return -1;
  }

  const dgs_model * vModel       = fDgsGetModel(pLctx);
  int               vNMmprojEmbd = fDgsModelNEmbdInp(vModel);
  int               vNPosPerEmbd = fMtmdDecodeUseMrope(pCtx) ? 4 : 1;

  int32_t           vNTokens     = fMtmdInputChunkGetNTokens(pChunk);
  int32_t           vIBatch      = 0;
  int32_t           vNImgBatches = (vNTokens + pNBatch - 1) / pNBatch;
  decode_embd_batch vBatchEmbd(pEncodedEmbd, vNTokens, vNPosPerEmbd, vNMmprojEmbd);

  if (fMtmdDecodeUseMrope(pCtx)) {
    if (vChunkType == cMtmdInputChunkTypeImage) {
      const auto cImageTokens = fMtmdInputChunkGetTokensImage(pChunk);
      if (!cImageTokens) {
        LOG_ERR("failed to decode chunk: image tokens are null\n");
        return -1;
      }
      const auto                    cNTokens = fMtmdImageTokensGetNTokens(cImageTokens);
      std::vector<mtmd_decoder_pos> lRelPos(cNTokens);
      fMtmdHelperImageGetDecoderPos(cImageTokens, pNPast, lRelPos.data());
      vBatchEmbd.mSetPositionMrope2d(lRelPos, pSeqId);
    } else if (vChunkType == cMtmdInputChunkTypeAudio) {
      vBatchEmbd.mSetPositionMrope1d(pNPast, pSeqId);
    } else {
      GGML_ABORT("invalid chunk type for M-RoPE");
    }
  } else {
    vBatchEmbd.mSetPositionNormal(pNPast, pSeqId);
  }

  const bool             cUseNonCausal = fMtmdDecodeUseNonCausal(pCtx, pChunk);
  const scope_non_causal cNonCausal(pLctx, cUseNonCausal);

  while (vIBatch < vNImgBatches) {  // split into batches
    int vPosOffset    = vIBatch * pNBatch;
    int vNTokensBatch = std::min(pNBatch, vNTokens - vPosOffset);

    LOG_INF("decoding %s batch %d/%d, n_tokens_batch = %d\n", name, vIBatch + 1, vNImgBatches, vNTokensBatch);

    int64_t vT1  = ggml_time_ms();
    int32_t vRet = fDgsProcess(pLctx, cDgsProcessTypeDecode, vBatchEmbd.mRender(pLctx, vPosOffset, vNTokensBatch));
    if (vRet != 0) {
      LOG_ERR("failed to decode %s\n", name);
      return vRet;
    }

    if (pCallback != nullptr) {
      const mtmd_helper_embd_batch cView = vBatchEmbd.mGetView(vPosOffset, vNTokensBatch);
      vRet                               = pCallback(&cView, pUserData);
      if (vRet != 0) {
        LOG_ERR("post-decode callback failed\n");
        return vRet;
      }
    }

    LOG_INF("%s decoded (batch %d/%d) in %" PRId64 " ms\n", name, vIBatch + 1, vNImgBatches, ggml_time_ms() - vT1);

    vIBatch++;
  }

  pNPast += fMtmdInputChunkGetNPos(pChunk);
  *pNewNPast = pNPast;

  return 0;
}

int32_t fMtmdHelperEvalChunkSingle(mtmd_context *           pCtx,
                                   struct dgs_context *     pLctx,
                                   const mtmd_input_chunk * pChunk,
                                   dgs_pos                  pNPast,
                                   dgs_seq_id               pSeqId,
                                   int32_t                  pNBatch,
                                   bool                     pLogitsLast,
                                   dgs_pos *                pNewNPast) {
  GGML_ASSERT(pNBatch > 0);
  int32_t vRet;
  auto    vChunkType = fMtmdInputChunkGetType(pChunk);

  if (vChunkType == cMtmdInputChunkTypeText) {
    size_t            vNTokens;
    const auto        cTokens = fMtmdInputChunkGetTokensText(pChunk, &vNTokens);
    // LOG_INF("decoding text chunk, n_tokens = %zu\n", n_tokens);
    dgs_batch_ext_ptr vTextBatch(fDgsBatchExtInit(pLctx));
    size_t            vI = 0;
    while (vI < vNTokens) {  // split into batches
      fDgsBatchExtClear(vTextBatch.get());
      int32_t vNAdded = 0;
      int32_t vIdx    = -1;
      for (; vI < vNTokens && vNAdded < pNBatch; vI++) {
        vIdx = fDgsBatchExtAddToken(vTextBatch.get(), pSeqId, cTokens[vI]);
        GGML_ASSERT(vIdx >= 0);
        dgs_pos vPos = pNPast++;
        fDgsBatchExtSetPos(vTextBatch.get(), vIdx, &vPos);
        vNAdded++;
      }
      bool vIsLastToken = (vI == vNTokens);
      if (pLogitsLast && vIsLastToken) {
        fDgsBatchExtSetOutputLogits(vTextBatch.get(), vIdx, true);
      }
      vRet = fDgsProcess(pLctx, cDgsProcessTypeDecode, vTextBatch.get());
      if (vRet != 0) {
        LOG_ERR("failed to decode text\n");
        return vRet;
      }
      *pNewNPast += vNAdded;
    }

  } else if (vChunkType == cMtmdInputChunkTypeImage || vChunkType == cMtmdInputChunkTypeAudio) {
    const char * name = vChunkType == cMtmdInputChunkTypeImage ? "image" : "audio";
    int64_t      vT0  = ggml_time_ms();

    LOG_INF("encoding %s slice...\n", name);

    vRet = fMtmdEncodeChunk(pCtx, pChunk);
    if (vRet != 0) {
      LOG_ERR("failed to encode %s slice\n", name);
      return vRet;
    }

    LOG_INF("%s slice encoded in %" PRId64 " ms\n", name, ggml_time_ms() - vT0);

    float * vEmbd = fMtmdGetOutputEmbd(pCtx);
    vRet =
      fMtmdHelperDecodeImageChunk(pCtx, pLctx, pChunk, vEmbd, pNPast, pSeqId, pNBatch, pNewNPast, nullptr, nullptr);
    if (vRet != 0) {
      LOG_ERR("failed to decode %s\n", name);
      return vRet;
    }
  } else {
    GGML_ABORT("chunk type not supported");
  }

  return 0;
}

int32_t fMtmdHelperEvalChunks(mtmd_context *            pCtx,
                              struct dgs_context *      pLctx,
                              const mtmd_input_chunks * pChunks,
                              dgs_pos                   pNPast,
                              dgs_seq_id                pSeqId,
                              int32_t                   pNBatch,
                              bool                      pLogitsLast,
                              dgs_pos *                 pNewNPast) {
  size_t vNChunks = fMtmdInputChunksSize(pChunks);
  if (vNChunks == 0) {
    LOG_WRN("no chunks to eval\n");
    return 0;
  }

  for (size_t vI = 0; vI < vNChunks; vI++) {
    bool vChunkLogitsLast = (vI == vNChunks - 1) && pLogitsLast;
    auto vChunk           = fMtmdInputChunksGet(pChunks, vI);

    int32_t vRes = fMtmdHelperEvalChunkSingle(pCtx, pLctx, vChunk, pNPast, pSeqId, pNBatch, vChunkLogitsLast, &pNPast);
    if (vRes != 0) {
      LOG_ERR("failed to eval chunk %zu\n", vI);
      return vRes;
    }
    *pNewNPast = pNPast;
  }

  return 0;
}

namespace audio_helpers {

static bool fIsAudioFile(const char * pBuf, size_t pLen) {
  if (pLen < 12) {
    return false;
  }

  // RIFF ref: https://en.wikipedia.org/wiki/Resource_Interchange_File_Format
  // WAV ref: https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
  bool vIsWav  = memcmp(pBuf, "RIFF", 4) == 0 && memcmp(pBuf + 8, "WAVE", 4) == 0;
  bool vIsMp3  = pLen >= 3 && (memcmp(pBuf, "ID3", 3) == 0 ||
                               // Check for MPEG sync word (simplified check)
                               ((unsigned char) pBuf[0] == 0xFF && ((unsigned char) pBuf[1] & 0xE0) == 0xE0));
  bool vIsFlac = memcmp(pBuf, "fLaC", 4) == 0;

  return vIsWav || vIsMp3 || vIsFlac;
}

// returns true if the buffer is a valid audio file
static bool fDecodeAudioFromBuf(const unsigned char * pBufIn,
                                size_t                pLen,
                                int                   pTargetSamplerRate,
                                std::vector<float> &  pPcmf32Mono) {
  ma_result         vResult;
  const int         cChannels      = 1;
  ma_decoder_config vDecoderConfig = ma_decoder_config_init(ma_format_f32, cChannels, pTargetSamplerRate);
  ma_decoder        vDecoder;

  vResult = ma_decoder_init_memory(pBufIn, pLen, &vDecoderConfig, &vDecoder);
  if (vResult != MA_SUCCESS) {
    return false;
  }

  ma_uint64 vFrameCount;
  ma_uint64 vFramesRead;
  vResult = ma_decoder_get_length_in_pcm_frames(&vDecoder, &vFrameCount);
  if (vResult != MA_SUCCESS) {
    ma_decoder_uninit(&vDecoder);
    return false;
  }

  pPcmf32Mono.resize(vFrameCount);
  vResult = ma_decoder_read_pcm_frames(&vDecoder, pPcmf32Mono.data(), vFrameCount, &vFramesRead);
  if (vResult != MA_SUCCESS) {
    ma_decoder_uninit(&vDecoder);
    return false;
  }

#ifdef MTMD_AUDIO_DEBUG
  // save audio to wav file
  ma_encoder_config config = ma_encoder_config_init(ma_encoding_format_wav, ma_format_f32, 1, target_sampler_rate);
  ma_encoder        encoder;
  ma_encoder_init_file("output.wav", &config, &encoder);
  ma_encoder_write_pcm_frames(&encoder, pcmf32_mono.data(), pcmf32_mono.size(), &frames_read);
  ma_encoder_uninit(&encoder);
#endif

  ma_decoder_uninit(&vDecoder);
  return true;
}

}  // namespace audio_helpers

static bool fIsWebpFile(const unsigned char * pBuf, size_t pLen) {
  // WEBP ref: https://developers.google.com/speed/webp/docs/riff_container
  return pLen >= 12 && memcmp(pBuf, "RIFF", 4) == 0 && memcmp(pBuf + 8, "WEBP", 4) == 0;
}

#ifdef MTMD_VIDEO
static mtmd_bitmap * fDecodeWebpWithFfmpeg(const mtmd_context *                  pMctx,
                                           const unsigned char *                 pBuf,
                                           size_t                                pLen,
                                           bool                                  pPlaceholder,
                                           const mtmd_helper_video_init_params & pParams);
static void          fMtmdHelperVideoSetId(mtmd_helper_video * pVctx, const std::string & pId);
#endif

mtmd_helper_bitmap_wrapper fMtmdHelperBitmapInitFromBuf(const mtmd_context *  pCtx,
                                                        const unsigned char * pBuf,
                                                        size_t                pLen,
                                                        bool                  pPlaceholder,
                                                        mtmd_helper_init_opt  pOpt) {
  // calculate the hash if needed
  std::string   vId;
  mtmd_bitmap * vResult = nullptr;

  GGML_UNUSED(pOpt);  // only used by video code paths

  if (!pPlaceholder) {
    // use sha256 to prevent cache poisoning
    vId = hash_sha256_hex(pBuf, pLen);
  }

  if (audio_helpers::fIsAudioFile((const char *) pBuf, pLen)) {
    std::vector<float> lPcmf32;
    const int          cSampleRate = fMtmdGetAudioSampleRate(pCtx);
    if (cSampleRate < 0) {
      LOG_ERR("This model does not support audio input\n");
      return { nullptr, nullptr };
    }
    if (!audio_helpers::fDecodeAudioFromBuf(pBuf, pLen, cSampleRate, lPcmf32)) {
      LOG_ERR("Unable to read WAV audio file from buffer\n");
      return { nullptr, nullptr };
    }
    vResult = fMtmdBitmapInitFromAudio(lPcmf32.size(), pPlaceholder ? nullptr : lPcmf32.data());
    fMtmdBitmapSetId(vResult, vId.empty() ? nullptr : vId.c_str());
    return { vResult, nullptr };
  }

  // otherwise, we assume it's an image
  if (!vResult) {
    int    vNx, vNy, vNc;
    auto * data = stbi_load_from_memory(pBuf, pLen, &vNx, &vNy, &vNc, 3);
    if (data) {
      vResult = fMtmdBitmapInit(vNx, vNy, pPlaceholder ? nullptr : data);
      fMtmdBitmapSetId(vResult, vId.empty() ? nullptr : vId.c_str());
      stbi_image_free(data);
      return { vResult, nullptr };
    }
    // otherwise, fallthrough to video decoding (if supported)
  }

#ifdef MTMD_VIDEO
  // stb_image does not support webp; decode it with ffmpeg as a single frame
  if (!vResult && fIsWebpFile(pBuf, pLen)) {
    vResult = fDecodeWebpWithFfmpeg(pCtx, pBuf, pLen, pPlaceholder, pOpt.vVideoParams);
    if (!vResult) {
      LOG_ERR("%s: failed to decode webp buffer\n", __func__);
      return { nullptr, nullptr };
    }
    fMtmdBitmapSetId(vResult, vId.empty() ? nullptr : vId.c_str());
    return { vResult, nullptr };
  }
#endif

  // last try: load as video
#ifdef MTMD_VIDEO
  if (!vResult) {
    auto vVideoCtx = fMtmdHelperVideoInitFromBuf(pCtx, pBuf, pLen, pOpt.vVideoParams);
    if (!vVideoCtx) {
      LOG_ERR("%s: failed to decode buffer as either image/audio/video\n", __func__);
      return { nullptr, nullptr };
    }
    fMtmdHelperVideoSetId(vVideoCtx, vId);  // propagate the hash to the frames
    vResult = fMtmdBitmapInitLazy(pCtx, vId.empty() ? nullptr : vId.c_str(), vVideoCtx,
                                  [](size_t, void * pUserData, mtmd_bitmap ** pOutBitmap, char ** pOutText) -> int {
                                    auto * vVctx = static_cast<mtmd_helper_video *>(pUserData);
                                    char * vText = nullptr;
                                    int    vRet  = fMtmdHelperVideoReadNext(vVctx, pOutBitmap, &vText);
                                    *pOutText    = vText;  // heap-allocated by read_next; freed automatically by mtmd
                                    return vRet;
                                  });
    return { vResult, vVideoCtx };
  }
#else
  if (!result) {
    LOG_ERR("%s: failed to decode buffer as either image or audio (video support not compiled in)\n", __func__);
    return { nullptr, nullptr };
  }
#endif

  // should not reach here
  return { nullptr, nullptr };
}

mtmd_helper_bitmap_wrapper fMtmdHelperBitmapInitFromFile(const mtmd_context * pCtx,
                                                         const char *         pFname,
                                                         bool                 pPlaceholder,
                                                         mtmd_helper_init_opt pOpt) {
#ifdef _WIN32
  int wlen = MultiByteToWideChar(CP_UTF8, 0, fname, -1, NULL, 0);
  if (!wlen) {
    LOG_ERR("Unable to convert filename to UTF-16: %s\n", fname);
    return { nullptr, nullptr };
  }
  std::vector<wchar_t> wfname(wlen);
  wlen = MultiByteToWideChar(CP_UTF8, 0, fname, -1, wfname.data(), wlen);
  if (!wlen) {
    LOG_ERR("Unable to convert filename to UTF-16: %s\n", fname);
    return { nullptr, nullptr };
  }
  FILE * f = _wfopen(wfname.data(), L"rb");
#else
  FILE * vF = fopen(pFname, "rb");
#endif
  if (!vF) {
    LOG_ERR("Unable to open file %s: %s\n", pFname, strerror(errno));
    return { nullptr, nullptr };
  }

  std::vector<unsigned char> lBuf;

  fseek(vF, 0, SEEK_END);
  long vFileSize = ftell(vF);
  fseek(vF, 0, SEEK_SET);
  if (vFileSize < 0) {
    LOG_ERR("Failed to get file size of %s\n", pFname);
    fclose(vF);
    return { nullptr, nullptr };
  }
  lBuf.resize(vFileSize);

  size_t vNRead = fread(lBuf.data(), 1, vFileSize, vF);
  fclose(vF);
  if (vNRead != (size_t) vFileSize) {
    LOG_ERR("Failed to read entire file %s", pFname);
    return { nullptr, nullptr };
  }

  return fMtmdHelperBitmapInitFromBuf(pCtx, lBuf.data(), lBuf.size(), pPlaceholder, pOpt);
}

bool fMtmdHelperSupportVideo(const mtmd_context * pCtx) {
#ifdef MTMD_VIDEO
  return fMtmdSupportVision(pCtx);
#else
  GGML_UNUSED(ctx);
  return false;
#endif
}

//
// Video input helpers
//

#ifdef MTMD_VIDEO

// MSVC deprecates strdup(), use a std::string version directly
static char * fMtmdStrdup(const std::string & pS) {
  size_t n  = pS.size() + 1;
  char * vP = (char *) malloc(n);
  if (vP) {
    memcpy(vP, pS.c_str(), n);
  }
  return vP;
}

struct mtmd_helper_video {
  const mtmd_context *   vMctx;
  std::string            vPath;
  std::vector<uint8_t>   lInputBuf;  // non-empty when initialized from buffer
  std::string            vFfmpegBin;
  std::string            vFfprobeBin;
  float                  vFpsTarget = 0.0f;
  mtmd_helper_video_info info       = {};
  std::string            vId;  // hash of the input video

  // RAII wrapper for managing subprocess
  struct subprocess_handle {
    struct subprocess_s vProc    = {};
    bool                vCreated = false;  // process exists and must be cleaned up
    bool                vAlive   = false;  // process can still give us data
    std::thread         vFeeder;

    subprocess_handle()                                      = default;
    subprocess_handle(const subprocess_handle &)             = delete;
    subprocess_handle & operator=(const subprocess_handle &) = delete;

    ~subprocess_handle() { mStop(); }

    void mStop() {
      // note: alive becomes false on stdout EOF, but the process still needs cleanup
      if (!vCreated) {
        return;
      }
      subprocess_terminate(&vProc);
#  ifdef _WIN32
      // no SIGPIPE on windows: a blocked feeder only gets a broken pipe once we close our read end of the child stdin
      if (proc.hStdInput) {
        CloseHandle(proc.hStdInput);
        proc.hStdInput = nullptr;
      }
#  endif
      // join before destroy: feeder holds a FILE* from subprocess_stdin;
      // subprocess_destroy closes it, so the thread must finish first
      if (vFeeder.joinable()) {
        vFeeder.join();
      }
      subprocess_join(&vProc, nullptr);  // reap the child, or else it stays a zombie
      subprocess_destroy(&vProc);
      vCreated = false;
      vAlive   = false;
    }

    FILE * mStdoutPipe() { return subprocess_stdout(&vProc); }

    // buf is tied to lifetime of mtmd_helper_video, so it's guaranteed to outlive the feeder thread
    void mStartFeeder(const std::vector<uint8_t> & pBuf) {
      vFeeder = std::thread([this, &pBuf]() {
#  ifndef _WIN32
        // ffmpeg can exit before it reads all the input, for example when ffprobe already got the metadata.
        // the write below must then fail with EPIPE, instead of killing the process with SIGPIPE
        sigset_t vSigpipeSet;
        sigemptyset(&vSigpipeSet);
        sigaddset(&vSigpipeSet, SIGPIPE);
        pthread_sigmask(SIG_BLOCK, &vSigpipeSet, nullptr);  // linux sends the signal to the writing thread
#  endif
        FILE * vF = subprocess_stdin(&vProc);
        if (!vF) {
          return;
        }
#  ifdef F_SETNOSIGPIPE
        fcntl(fileno(f), F_SETNOSIGPIPE, 1);  // macos/bsd send it to the process, so turn it off per fd
#  endif
        fwrite(pBuf.data(), 1, pBuf.size(), vF);
        fclose(vF);
        vProc.stdin_file = nullptr;  // prevent double-close in subprocess_destroy
      });
    }
  };

  subprocess_handle vSp;
  int32_t           vCurrentFrame = 0;

  std::string vPromptStart         = "Video:";
  int32_t     vTimestampIntervalMs = 5000;  // emit a timestamp text every N ms (0 = disabled)
  float       vNextTimestampMs     = 0.0f;  // next elapsed-ms threshold at which to emit

  std::vector<uint8_t> lFrameBuf;
  std::string          vPendingText;  // text queued to be returned before the next frame
  bool                 vStartEmitted = false;

  bool mIsBufInput() const { return !lInputBuf.empty(); }

  bool mProbe(float pFpsTargetArg) {
    const char * vInputArg = mIsBufInput() ? "pipe:0" : vPath.c_str();
    const char * aCmd[]    = {
      vFfprobeBin.c_str(),
      "-v",
      "quiet",
      "-show_entries",
      "stream=width,height,r_frame_rate,nb_frames,duration",
      "-select_streams",
      "v:0",
      "-of",
      "default=noprint_wrappers=1",
      vInputArg,
      nullptr,
    };

    LOG_DBG("%s: launching:", __func__);
    for (size_t vI = 0; aCmd[vI]; vI++) {
      LOG_DBG(" %s", aCmd[vI]);
    }
    LOG_DBG("\n");

    subprocess_handle vProbeSp;
    if (subprocess_create(aCmd, subprocess_option_search_user_path | subprocess_option_inherit_environment,
                          &vProbeSp.vProc) != 0) {
      LOG_ERR("%s: failed to launch ffprobe\n", __func__);
      return false;
    }
    vProbeSp.vCreated = true;
    vProbeSp.vAlive   = true;

    if (mIsBufInput()) {
      vProbeSp.mStartFeeder(lInputBuf);
    }

    uint32_t vWidth       = 0;
    uint32_t vHeight      = 0;
    float    vOrigFps     = 0.0f;
    float    vDuration    = -1.0f;
    int32_t  vNFramesOrig = -1;
    char     aLine[256];
    FILE *   vFp = vProbeSp.mStdoutPipe();

    while (fgets(aLine, sizeof(aLine), vFp)) {
      char * vEq = strchr(aLine, '=');
      if (!vEq) {
        continue;
      }
      *vEq              = '\0';
      const char * vKey = aLine;
      const char * vVal = vEq + 1;
      char *       vNl  = (char *) strchr(vVal, '\n');
      if (vNl) {
        *vNl = '\0';
      }

      if (strcmp(vKey, "width") == 0) {
        vWidth = (uint32_t) atoi(vVal);
      } else if (strcmp(vKey, "height") == 0) {
        vHeight = (uint32_t) atoi(vVal);
      } else if (strcmp(vKey, "r_frame_rate") == 0) {
        vOrigFps = mParseRational(vVal);
      } else if (strcmp(vKey, "nb_frames") == 0 && strcmp(vVal, "N/A") != 0) {
        vNFramesOrig = atoi(vVal);
      } else if (strcmp(vKey, "duration") == 0 && strcmp(vVal, "N/A") != 0) {
        vDuration = (float) atof(vVal);
      }
    }

    vProbeSp.mStop();

    if (vWidth == 0 || vHeight == 0 || vOrigFps <= 0.0f) {
      return false;
    }

    if (vDuration < 0.0f && vNFramesOrig > 0) {
      vDuration = (float) vNFramesOrig / vOrigFps;
    }

    vFpsTarget   = pFpsTargetArg > 0.0f ? pFpsTargetArg : vOrigFps;
    info.vWidth  = vWidth;
    info.vHeight = vHeight;
    info.vFps    = vFpsTarget;
    LOG_DBG("%s: %ux%u fps=%.2f duration=%.2fs n_frames=%d\n", __func__, vWidth, vHeight, vFpsTarget, vDuration,
            info.vNFrames);
    info.vNFrames = vDuration > 0.0f ? (int32_t) (vDuration * vFpsTarget + 0.5f) : -1;
    lFrameBuf.resize((size_t) vWidth * vHeight * 3);
    return true;
  }

  bool mStartFfmpeg(float pSeekSeconds) {
    char aSeekBuf[64];
    char aFpsBuf[64];

    std::vector<const char *> lCmd;
    lCmd.push_back(vFfmpegBin.c_str());

    if (!mIsBufInput() && pSeekSeconds > 0.0f) {
      // input-side seek: fast, keyframe-accurate; only valid for seekable file inputs
      snprintf(aSeekBuf, sizeof(aSeekBuf), "%.6f", pSeekSeconds);
      lCmd.push_back("-ss");
      lCmd.push_back(aSeekBuf);
    }

    lCmd.push_back("-nostdin");
    if (mIsBufInput()) {
      // remove the 64KB read-ahead limit of cache:, or else ffmpeg cannot reach a moov atom at end of file
      lCmd.push_back("-read_ahead_limit");
      lCmd.push_back("-1");
    }
    lCmd.push_back("-i");
    // cache:pipe:0 wraps stdin with a seekable in-memory cache, letting ffmpeg seek
    // backwards for container headers (e.g. MP4 moov atom at end of file)
    lCmd.push_back(mIsBufInput() ? "cache:pipe:0" : vPath.c_str());

    if (pSeekSeconds > 0.0f && mIsBufInput()) {
      // output-side seek: frame-accurate but decodes and discards frames up to seek point
      snprintf(aSeekBuf, sizeof(aSeekBuf), "%.6f", pSeekSeconds);
      lCmd.push_back("-ss");
      lCmd.push_back(aSeekBuf);
    }

    if (vFpsTarget > 0.0f) {
      snprintf(aFpsBuf, sizeof(aFpsBuf), "fps=%.6f", vFpsTarget);
      lCmd.push_back("-vf");
      lCmd.push_back(aFpsBuf);
    }

    lCmd.push_back("-f");
    lCmd.push_back("rawvideo");
    lCmd.push_back("-pix_fmt");
    lCmd.push_back("rgb24");
    lCmd.push_back("pipe:1");
    lCmd.push_back("-loglevel");
    lCmd.push_back("error");
    lCmd.push_back(nullptr);

    LOG_DBG("%s: launching:", __func__);
    for (size_t vI = 0; lCmd[vI]; vI++) {
      LOG_DBG(" %s", lCmd[vI]);
    }
    LOG_DBG("\n");

    int vRet = subprocess_create(
      lCmd.data(), subprocess_option_search_user_path | subprocess_option_inherit_environment, &vSp.vProc);

    vSp.vCreated = (vRet == 0);
    vSp.vAlive   = (vRet == 0);
    LOG_DBG("%s: subprocess_create ret=%d proc_alive=%d\n", __func__, vRet, (int) vSp.vAlive);

    if (vSp.vAlive && mIsBufInput()) {
      LOG_DBG("%s: starting feeder thread for %zu-byte buffer\n", __func__, lInputBuf.size());
      vSp.mStartFeeder(lInputBuf);
    }

    return vSp.vAlive;
  }

  void mStopFfmpeg() { vSp.mStop(); }

  mtmd_bitmap * mReadNextFrame() {
    if (!vSp.vAlive) {
      return nullptr;
    }

    FILE *       vFp        = vSp.mStdoutPipe();
    const size_t cFrameSize = (size_t) info.vWidth * info.vHeight * 3;
    LOG_DBG("%s: reading frame %d, expecting %zu bytes (%ux%u)\n", __func__, vCurrentFrame, cFrameSize, info.vWidth,
            info.vHeight);

    size_t vTotalRead = 0;
    while (vTotalRead < cFrameSize) {
      size_t n = fread(lFrameBuf.data() + vTotalRead, 1, cFrameSize - vTotalRead, vFp);
      if (n == 0) {
        // clean EOF only if no bytes read yet; partial frame is an error
        LOG_DBG("%s: fread returned 0 after %zu/%zu bytes (ferror=%d)\n", __func__, vTotalRead, cFrameSize,
                ferror(vFp));
        vSp.vAlive = false;
        return nullptr;
      }
      vTotalRead += n;
    }

    LOG_DBG("%s: frame %d read OK\n", __func__, vCurrentFrame);
    mtmd_bitmap * vFrame = fMtmdBitmapInit(info.vWidth, info.vHeight, lFrameBuf.data());
    fMtmdBitmapSetMergeable(vFrame, true);
    if (!vId.empty()) {
      // each frame gets a unique id in the form of {hash}+{frame}, so that it can be identified in cache
      std::string vFrameId = vId + "+" + std::to_string(vCurrentFrame);
      fMtmdBitmapSetId(vFrame, vFrameId.c_str());
    }
    vCurrentFrame++;
    return vFrame;
  }

  int32_t mReadNext(mtmd_bitmap ** pOutBitmap, char ** pOutText) {
    *pOutBitmap = nullptr;
    *pOutText   = nullptr;

    if (!vPendingText.empty()) {
      *pOutText = fMtmdStrdup(vPendingText);
      vPendingText.clear();
      return *pOutText ? 0 : -2;
    }

    LOG_DBG("%s: proc_alive=%d start_emitted=%d current_frame=%d\n", __func__, (int) vSp.vAlive, (int) vStartEmitted,
            vCurrentFrame);

    if (!vSp.vAlive) {
      return (vCurrentFrame == 0) ? -2 : -1;
    }

    if (!vStartEmitted) {
      vStartEmitted = true;
      if (!vPromptStart.empty()) {
        *pOutText = fMtmdStrdup(vPromptStart);
        return *pOutText ? 0 : -2;
      }
    }

    mtmd_bitmap * vFrame = mReadNextFrame();
    if (!vFrame) {
      return -1;
    }
    *pOutBitmap = vFrame;

    if (vTimestampIntervalMs > 0) {
      // current_frame was already incremented by read_next_frame(); undo for elapsed calc
      float vElapsedMs = (float) (vCurrentFrame - 1) / info.vFps * 1000.0f;
      if (vElapsedMs >= vNextTimestampMs) {
        char  aTsBuf[32];
        float vElapsedS = vElapsedMs / 1000.0f;
        int   vMinutes  = (int) (vElapsedS / 60);
        float vSeconds  = vElapsedS - vMinutes * 60.0f;
        snprintf(aTsBuf, sizeof(aTsBuf), "[%dm%.2fs]", vMinutes, vSeconds);
        vPendingText = aTsBuf;
        vNextTimestampMs += (float) vTimestampIntervalMs;
      }
    }

    return 0;
  }

  static float mParseRational(const char * pS) {
    int vNum = 0, vDen = 1;
    if (sscanf(pS, "%d/%d", &vNum, &vDen) == 2 && vDen > 0) {
      return (float) vNum / (float) vDen;
    }
    float vVal;
    if (sscanf(pS, "%f", &vVal) == 1) {
      return vVal;
    }
    return 0.0f;
  }
};
#endif

mtmd_helper_video_init_params fMtmdHelperVideoInitParamsDefault() {
  return {
    /* fps_target             */ 4.0f,
    /* ffmpeg_bin_dir         */ nullptr,
    /* timestamp_interval_ms  */ 5000,
  };
}

mtmd_helper_init_opt fMtmdHelperInitOptDefault() {
  return {
    /* video_params */ fMtmdHelperVideoInitParamsDefault(),
  };
}

static std::string fVideoResolveBin(const char * pBinDir, const char * name) {
  if (!pBinDir || pBinDir[0] == '\0') {
    return name;  // rely on PATH
  }
  std::string vResult = pBinDir;
  char        vLast   = vResult.back();
  if (vLast != '/' && vLast != '\\') {
#ifdef _WIN32
    result += '\\';
#else
    vResult += '/';
#endif
  }
  vResult += name;
#ifdef _WIN32
  result += ".exe";
#endif
  return vResult;
}

#ifdef MTMD_VIDEO
static void fMtmdHelperVideoSetId(mtmd_helper_video * pVctx, const std::string & pId) {
  pVctx->vId = pId;
}

static mtmd_bitmap * fDecodeWebpWithFfmpeg(const mtmd_context *                  pMctx,
                                           const unsigned char *                 pBuf,
                                           size_t                                pLen,
                                           bool                                  pPlaceholder,
                                           const mtmd_helper_video_init_params & pParams) {
  mtmd_helper_video vVctx;
  vVctx.vMctx = pMctx;
  vVctx.lInputBuf.assign(pBuf, pBuf + pLen);
  vVctx.vFfmpegBin  = fVideoResolveBin(pParams.vFfmpegBinDir, "ffmpeg");
  vVctx.vFfprobeBin = fVideoResolveBin(pParams.vFfmpegBinDir, "ffprobe");
  if (!vVctx.mProbe(0.0f)) {
    return nullptr;
  }
  if (pPlaceholder) {
    return fMtmdBitmapInit(vVctx.info.vWidth, vVctx.info.vHeight, nullptr);
  }
  // still image: the fps filter would output no frame, so disable it
  vVctx.vFpsTarget = 0.0f;
  if (!vVctx.mStartFfmpeg(0.0f)) {
    return nullptr;
  }
  mtmd_bitmap * vFrame = vVctx.mReadNextFrame();
  if (vFrame) {
    fMtmdBitmapSetMergeable(vFrame, false);
  }
  return vFrame;
}
#endif

mtmd_helper_video * fMtmdHelperVideoInit(const mtmd_context *          pMctx,
                                         const char *                  pPath,
                                         mtmd_helper_video_init_params pParams) {
#ifdef MTMD_VIDEO
  auto * vCtx = new mtmd_helper_video();

  vCtx->vMctx                = pMctx;
  vCtx->vPath                = pPath;
  vCtx->vFfmpegBin           = fVideoResolveBin(pParams.vFfmpegBinDir, "ffmpeg");
  vCtx->vFfprobeBin          = fVideoResolveBin(pParams.vFfmpegBinDir, "ffprobe");
  vCtx->vTimestampIntervalMs = pParams.vTimestampIntervalMs;

  if (!vCtx->mProbe(pParams.vFpsTarget)) {
    LOG_ERR("%s: ffprobe failed for '%s' (is ffprobe in PATH?)\n", __func__, pPath);
    delete vCtx;
    return nullptr;
  }

  if (!vCtx->mStartFfmpeg(0.0f)) {
    LOG_ERR("%s: failed to start ffmpeg for '%s' (is ffmpeg in PATH?)\n", __func__, pPath);
    delete vCtx;
    return nullptr;
  }

  return vCtx;
#else
  GGML_UNUSED(mctx);
  GGML_UNUSED(path);
  GGML_UNUSED(params);
  LOG_ERR("%s: video is not supported in this build (MTMD_VIDEO is set to OFF)\n", __func__);
  return nullptr;
#endif
}

mtmd_helper_video * fMtmdHelperVideoInitFromBuf(const mtmd_context *          pMctx,
                                                const unsigned char *         pBuf,
                                                size_t                        pLen,
                                                mtmd_helper_video_init_params pParams) {
#ifdef MTMD_VIDEO
  auto * vCtx = new mtmd_helper_video();

  vCtx->vMctx = pMctx;
  vCtx->lInputBuf.assign(pBuf, pBuf + pLen);
  vCtx->vFfmpegBin           = fVideoResolveBin(pParams.vFfmpegBinDir, "ffmpeg");
  vCtx->vFfprobeBin          = fVideoResolveBin(pParams.vFfmpegBinDir, "ffprobe");
  vCtx->vTimestampIntervalMs = pParams.vTimestampIntervalMs;

  if (!vCtx->mProbe(pParams.vFpsTarget)) {
    LOG_ERR("%s: ffprobe failed on buffer (is ffprobe in PATH?)\n", __func__);
    delete vCtx;
    return nullptr;
  }

  if (!vCtx->mStartFfmpeg(0.0f)) {
    LOG_ERR("%s: failed to start ffmpeg on buffer (is ffmpeg in PATH?)\n", __func__);
    delete vCtx;
    return nullptr;
  }

  return vCtx;
#else
  GGML_UNUSED(mctx);
  GGML_UNUSED(buf);
  GGML_UNUSED(len);
  GGML_UNUSED(params);
  LOG_ERR("%s: video is not supported in this build (MTMD_VIDEO is set to OFF)\n", __func__);
  return nullptr;
#endif
}

void fMtmdHelperVideoFree(mtmd_helper_video * pCtx) {
#ifdef MTMD_VIDEO
  if (!pCtx) {
    return;
  }
  pCtx->mStopFfmpeg();
  delete pCtx;
#else
  GGML_UNUSED(ctx);
  LOG_ERR("%s: video is not supported in this build (MTMD_VIDEO is set to OFF)\n", __func__);
#endif
}

mtmd_helper_video_info fMtmdHelperVideoGetInfo(const mtmd_helper_video * pCtx) {
#ifdef MTMD_VIDEO
  return pCtx->info;
#else
  GGML_UNUSED(ctx);
  GGML_ASSERT(false && "video is not supported in this build (MTMD_VIDEO is set to OFF)");
#endif
}

int32_t fMtmdHelperVideoReadNext(mtmd_helper_video * pCtx, mtmd_bitmap ** pOutBitmap, char ** pOutText) {
#ifdef MTMD_VIDEO
  if (!pCtx) {
    return -2;
  }
  return pCtx->mReadNext(pOutBitmap, pOutText);
#else
  GGML_UNUSED(ctx);
  GGML_UNUSED(out_bitmap);
  GGML_UNUSED(out_text);
  GGML_ASSERT(false && "video is not supported in this build (MTMD_VIDEO is set to OFF)");
#endif
}

bool fMtmdHelperModelCanChat(const dgs_context * pLctx, const mtmd_context * pMctx) {
  if (!pMctx) {
    return true;
  }

  auto * vModel = fDgsGetModel(pLctx);
  auto * vTmpl  = fDgsModelChatTemplate(vModel, nullptr);
  auto   info   = fMtmdGenAudioGetInfo(pMctx);

  // tts-only model cannot be used for chat (no chat template)
  bool vIsTtsOnly = info.type != cMtmdGenAudioTypeNone && vTmpl == nullptr;

  return !vIsTtsOnly;
}
