#pragma once

#include "clip-model.h"
#include "ggml.h"

#include <cstdint>
#include <string>
#include <vector>

#define MTMD_INTERNAL_HEADER

struct mtmd_audio_mel {
  int64_t vNLen;
  int64_t vNLenOrg;
  int64_t vNMel;

  std::vector<float> data;
};

struct mtmd_audio_mel_filters {
  int64_t vNMel;
  int64_t vNFft;

  std::vector<float> data;
};

// cache for audio processing, each processor instance owns its own cache
struct mtmd_audio_cache {
  std::vector<float> lSinVals;
  std::vector<float> lCosVals;

  std::vector<float> lHannWindow;

  mtmd_audio_mel_filters vFilters;

  void mFillSinCosTable(uint32_t n);

  void mFillHannWindow(uint32_t length, bool pPeriodic);

  // Build mel filterbank matrix [n_mel × n_fft_bins] at runtime.
  // n_fft_bins must be (N_fft / 2 + 1). Example: if N_fft=512 -> n_fft_bins=257.
  void mFillMelFilterbankMatrix(int64_t pNMel,
                                int64_t pNFft,
                                int     pSampleRate,              // e.g. 16000
                                float   pFmin           = 0.0f,   // e.g. 0.0
                                float   pFmax           = -1.0f,  // e.g. sr/2; pass -1 for auto
                                bool    pSlaneyAreaNorm = true,
                                float   pScale          = 1.0f,
                                bool    pUseHtk         = false);
};

struct mtmd_audio_preprocessor {
  const clip_hparams & vHparams;

  mtmd_audio_preprocessor(const clip_ctx * pCtx) : vHparams(*fClipGetHparams(pCtx)) {}

  mtmd_audio_preprocessor(const clip_hparams & pHparams) : vHparams(pHparams) {}

  virtual ~mtmd_audio_preprocessor() = default;
  virtual void mInitialize()         = 0;  // NOT thread-safe
  virtual bool mPreprocess(const float * pSamples, size_t pNSamples, std::vector<mtmd_audio_mel> & pOutput) const = 0;
};

struct mtmd_audio_preprocessor_whisper : mtmd_audio_preprocessor {
  mtmd_audio_preprocessor_whisper(const clip_ctx * pCtx) : mtmd_audio_preprocessor(pCtx) {}

  void mInitialize() override;
  bool mPreprocess(const float * pSamples, size_t pNSamples, std::vector<mtmd_audio_mel> & pOutput) const override;

private:
  mtmd_audio_cache vCache;
};

struct mtmd_audio_preprocessor_conformer : mtmd_audio_preprocessor {
  mtmd_audio_preprocessor_conformer(const clip_ctx * pCtx) : mtmd_audio_preprocessor(pCtx) {}

  mtmd_audio_preprocessor_conformer(const clip_hparams & pHparams) : mtmd_audio_preprocessor(pHparams) {}

  void mInitialize() override;
  bool mPreprocess(const float * pSamples, size_t pNSamples, std::vector<mtmd_audio_mel> & pOutput) const override;

private:
  mtmd_audio_cache vCache;
};

// same as conformer, the audio is cut to 30 s and padded to 0.5 s (d1-omni audio.py)
struct mtmd_audio_preprocessor_d1omni : mtmd_audio_preprocessor_conformer {
  using mtmd_audio_preprocessor_conformer::mtmd_audio_preprocessor_conformer;
  bool mPreprocess(const float * pSamples, size_t pNSamples, std::vector<mtmd_audio_mel> & pOutput) const override;
};

struct mtmd_audio_preprocessor_granite_speech : mtmd_audio_preprocessor {
  mtmd_audio_preprocessor_granite_speech(const clip_ctx * pCtx) : mtmd_audio_preprocessor(pCtx) {}

  void mInitialize() override;
  bool mPreprocess(const float * pSamples, size_t pNSamples, std::vector<mtmd_audio_mel> & pOutput) const override;

private:
  mtmd_audio_cache vCache;
};

struct mtmd_audio_preprocessor_gemma4a : mtmd_audio_preprocessor {
  mtmd_audio_preprocessor_gemma4a(const clip_ctx * pCtx) : mtmd_audio_preprocessor(pCtx) {}

  void mInitialize() override;
  bool mPreprocess(const float * pSamples, size_t pNSamples, std::vector<mtmd_audio_mel> & pOutput) const override;

private:
  mtmd_audio_cache vCache;
};

struct mtmd_audio_preprocessor_gemma4ua : mtmd_audio_preprocessor {
  mtmd_audio_preprocessor_gemma4ua(const clip_ctx * pCtx) : mtmd_audio_preprocessor(pCtx) {}

  void mInitialize() override;
  bool mPreprocess(const float * pSamples, size_t pNSamples, std::vector<mtmd_audio_mel> & pOutput) const override;
};

struct mtmd_audio_preprocessor_qwen3a : mtmd_audio_preprocessor {
  mtmd_audio_preprocessor_qwen3a(const clip_ctx * pCtx) : mtmd_audio_preprocessor(pCtx) {}

  void mInitialize() override;
  bool mPreprocess(const float * pSamples, size_t pNSamples, std::vector<mtmd_audio_mel> & pOutput) const override;

private:
  mtmd_audio_cache vCache;
};

struct mtmd_audio_preprocessor_dots3note : mtmd_audio_preprocessor {
  mtmd_audio_preprocessor_dots3note(const clip_ctx * pCtx) : mtmd_audio_preprocessor(pCtx) {}

  void mInitialize() override;
  bool mPreprocess(const float * pSamples, size_t pNSamples, std::vector<mtmd_audio_mel> & pOutput) const override;

private:
  mtmd_audio_cache vCache;
};

struct mtmd_audio_preprocessor_mimo_audio : mtmd_audio_preprocessor {
  mtmd_audio_preprocessor_mimo_audio(const clip_ctx * pCtx) : mtmd_audio_preprocessor(pCtx) {}

  void mInitialize() override;
  bool mPreprocess(const float * pSamples, size_t pNSamples, std::vector<mtmd_audio_mel> & pOutput) const override;

private:
  mtmd_audio_cache vCache;
};

struct mtmd_audio_preprocessor_qwen3tts_spk : mtmd_audio_preprocessor {
  mtmd_audio_preprocessor_qwen3tts_spk(const clip_ctx * pCtx) : mtmd_audio_preprocessor(pCtx) {}

  void mInitialize() override;
  bool mPreprocess(const float * pSamples, size_t pNSamples, std::vector<mtmd_audio_mel> & pOutput) const override;

private:
  mtmd_audio_cache vCache;
};

// mimi convolves the waveform directly, so this only pads it to a whole number of frames
struct mtmd_audio_preprocessor_pockettts : mtmd_audio_preprocessor {
  mtmd_audio_preprocessor_pockettts(const clip_ctx * pCtx) : mtmd_audio_preprocessor(pCtx) {}

  void mInitialize() override {}

  bool mPreprocess(const float * pSamples, size_t pNSamples, std::vector<mtmd_audio_mel> & pOutput) const override;
};

struct mtmd_audio_preprocessor_parakeet : mtmd_audio_preprocessor {
  mtmd_audio_preprocessor_parakeet(clip_ctx * pCtx) : mtmd_audio_preprocessor(pCtx) {}

  void mInitialize() override;
  bool mPreprocess(const float * pSamples, size_t pNSamples, std::vector<mtmd_audio_mel> & pOutput) const override;

private:
  mtmd_audio_cache vCache;

  static void mWorkerThread(int                        pIth,
                            const float *              pWindowFunc,
                            int                        pWindowSize,
                            const std::vector<float> & pSamples,
                            int                        pNSamples,
                            int                        pFrameSize,
                            int                        pFrameStep,
                            int                        pNThreads,
                            int                        pNFftBins,
                            const mtmd_audio_cache &   pCache,
                            mtmd_audio_mel &           pMel);
};

//
// streaming ISTFT - converts spectrogram frames back to audio one frame at a time
//
struct mtmd_audio_streaming_istft {
  mtmd_audio_streaming_istft(int pNFft, int pHopLength);

  // reset streaming state
  void reset();

  // process a single STFT frame (streaming)
  // frame_spectrum: [n_fft_bins x 2] interleaved real/imag
  // returns: up to hop_length samples
  std::vector<float> mProcessFrame(const float * pFrameSpectrum);

  // flush remaining samples at end of stream
  std::vector<float> mFlush();

private:
  int vNFft;
  int vHopLength;
  int vNFftBins;

  // Own cache for output processing
  mtmd_audio_cache vCache;

  // Streaming state
  std::vector<float> lOverlapBuffer;
  std::vector<float> lWindowSumBuffer;
  int                vPaddingToRemove;

  // Working buffers for IFFT
  std::vector<float> lIfftIn;
  std::vector<float> lIfftOut;
};
