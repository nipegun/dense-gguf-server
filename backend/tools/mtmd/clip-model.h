#pragma once

#include "clip-impl.h"
#include "clip.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <unordered_set>
#include <vector>

enum ffn_op_type {
  cFfnGelu,
  cFfnGeluErf,
  cFfnSilu,
  cFfnGeluQuick,
  cFfnReluSqr,
};

enum norm_type {
  cNormTypeNormal,
  cNormTypeRms,
};

enum patch_merge_type {
  cPatchMergeFlat,
  cPatchMergeSpatialUnpad,
};

// all algos are Pillow-compatible (matching PIL.Image.resize output)
enum resize_algo {
  cResizeAlgoBilinear,
  cResizeAlgoBicubic,
  cResizeAlgoLanczos,
};

// Padding style for img_tool::resize
//   PAD_NONE    - no padding; direct resize to target dimensions
//   PAD_CEIL    - aspect-preserving pad (default)
//   PAD_NEAREST - aspect-preserving pad with nearest-integer rounding (Pillow byte-parity)
enum pad_style {
  cPadNone,
  cPadCeil,
  cPadNearest,
};

struct clip_hparams {
  int32_t vImageSize     = 0;
  int32_t vPatchSize     = 0;
  int32_t vNEmbd         = 0;
  int32_t vNFf           = 0;
  int32_t vProjectionDim = 0;
  int32_t vNHead         = 0;
  int32_t vNHeadKv       = 0;
  // 0 = derive from n_embd; set when qkv width != n_embd
  int32_t vNEmbdHead     = 0;
  int32_t vNLayer        = 0;
  int32_t vNMerge        = 1;  // number of patch merges **per-side**

  // for preprocessor
  int32_t                vImageLongestEdge = 0;
  int32_t                vImageMinPixels   = -1;
  int32_t                vImageMaxPixels   = -1;
  resize_algo            vImageResizeAlgo  = cResizeAlgoBicubic;
  pad_style              vImageResizePad   = cPadCeil;  // padding style when resizing
  std::array<uint8_t, 3> aImagePadColor    = { 0, 0, 0 };

  // (preprocessor) for llava-uhd style models
  std::vector<clip_image_size> lImageResCandidates;
  int32_t                      vPreprocMinTiles   = 0;
  int32_t                      vPreprocMaxTiles   = 0;
  int32_t                      vMaxSliceNums      = 9;  // llava-uhd slice cap; per-model, carried in the GGUF
  int32_t                      vPreprocTileSize   = 0;  // local tile size (deepseek-ocr)
  resize_algo                  vImageResizeAlgoRf = cResizeAlgoBicubic;
  resize_algo                  vImageResizeAlgoOv = cResizeAlgoBicubic;
  pad_style                    vImagePadRf        = cPadCeil;  // padding style for the refined image (e.g. llava-1.6)
  pad_style                    vImagePadOv        = cPadNone;  // padding style for the overview image (e.g. llava-1.6)
  std::array<uint8_t, 3>       aImagePadColorRf   = { 0, 0, 0 };  // padding color for refined image
  std::array<uint8_t, 3>       aImagePadColorOv   = { 0, 0, 0 };  // padding color for overview image

  float aImageMean[3];
  float aImageStd[3];

  // for models using dynamic image size, we need to have a smaller image size to warmup
  // otherwise, user will get OOM every time they load the model
  int32_t vWarmupImageSize = 0;
  int32_t vWarmupAudioSize = 3000;

  ffn_op_type vFfnOp = cFfnGelu;

  patch_merge_type vMmPatchMergeType = cPatchMergeFlat;

  float vEps       = 1e-6;
  float vRopeTheta = 0.0;

  float                vSwigluClamp = 0.0f;
  int32_t              vNExpertUsed = 0;
  std::vector<int32_t> lFeatureLayers;
  int32_t              vAttnWindowSize = 0;
  int32_t              vNWaPattern     = 0;
  std::unordered_set<int32_t>
    sWaLayerIndexes;  // explicit layer indexes that use full attention (for irregular patterns like YoutuVL)
  std::vector<int32_t> lWaPatternMode;  // mimovl: per-layer window-attention mode

  // deepseek4v: resize solver caps the LLM token count of the aligner grid
  int32_t vDsv4MaxNToken  = 0;
  int32_t vDsv4MaxWhRatio = 0;

  // deepseek-ocr (sam)
  int32_t vSamNLayer = 0;
  int32_t vSamNHead  = 0;
  int32_t vSamNEmbd  = 0;

  // Granite4 Vision
  std::vector<int32_t> lProjSpatialOffsets;
  int32_t              vDownsampleQuerySide;
  int32_t              vDownsampleWindowSide;

  // Muse Glimmer vision (per-block sparse-window pattern, learned pos-emb, patch-temporal)
  // NOTE: these perhaps shouldn't have the architecture prefix
  int32_t vMuseGlimmerPatchTemporal = 0;
  int32_t vMuseGlimmerSparseFactor  = 0;

  // audio
  int32_t vNMelBins          = 0;  // whisper preprocessor
  int32_t vProjStackFactor   = 0;  // ultravox
  int32_t vSubsamplingFactor = 0;  // parakeet

  int32_t vAudioChunkSize          = 0;
  int32_t vAudioConvKernelSize     = 0;
  int32_t vAudioMaxPosEmb          = 0;
  int32_t vAudioProjWindowSize     = 0;
  int32_t vAudioProjDownsampleRate = 0;
  int32_t vAudioProjHeadCount      = 0;

  // audio-to-mel preprocessor params
  int32_t vAudioChunkLen   = -1;  // in seconds
  int32_t vAudioSampleRate = -1;
  int32_t vAudioNFft       = -1;
  int32_t vAudioWindowLen  = -1;
  int32_t vAudioHopLen     = -1;

  // parakeet
  std::vector<float> lMelFilters;
  std::vector<float> lWindow;

  // mimo-audio-tokenizer: residual vector quantizer
  int32_t              vRvqNumQuantizers = 0;
  std::vector<int32_t> lRvqCodebookSize;  // per-quantizer bin count (ragged, e.g. 1024/1024/256/128x17)

  // threshold for the "out_eos_score" graph output
  float vGenEosThreshold = 0.0f;

  // name of the weight variant, some pipelines tune themselves on it
  std::string vGenModelVariant;

  // pocket-tts
  static constexpr int32_t cPocketttsMaxSpkSeconds = 30;
  int32_t                  vSeanetNStage           = 0;
  std::vector<int32_t>     lSeanetRatios;        // encoder order (reversed compared to the config)
  int32_t                  vMimiDownsample = 0;  // encoder frame rate / model frame rate
  int32_t                  vMimiTfmContext = 0;  // attention window of the mimi transformers, in frames
  int32_t                  vFlowNStep      = 1;  // lsd_decode steps

  // qwen3tts code2wav
  int32_t vWavTfmNLayer      = 0;
  int32_t vWavTfmNEmbd       = 0;
  int32_t vWavTfmNFf         = 0;
  int32_t vWavTfmNHead       = 0;
  int32_t vWavTfmNHeadKv     = 0;
  float   vWavTfmEps         = 1e-5f;
  float   vWavTfmRopeTheta   = 10000.0f;
  int32_t vWavUpsampleNBlock = 0;
  int32_t vWavDacNBlock      = 0;
  int32_t vWavDacNRes        = 0;
  int32_t vWavTfmSwa         = 0;  // pre_transformer's KV cache size, in frames

  // mimo-v2.5: LLM-side connector (input_local_transformer)
  int32_t vAudioLocalNLayer    = 0;
  int32_t vAudioLocalGroupSize = 0;

  // legacy
  bool    vHasLlavaProjector = false;
  int     vMinicpmvVersion   = 0;
  int32_t vMinicpmvQueryNum  = 0;  // MiniCPM-V query number
  int32_t vInsertLayerId     = 0;  // MiniCPM-V 4.6 ViT merger insertion layer

  // custom value provided by user, can be undefined if not set
  int32_t vCustomImageMinTokens = -1;
  int32_t vCustomImageMaxTokens = -1;

  void mSetLimitImageTokens(int pNTokensMin, int pNTokensMax) {
    const int cPatchArea = vPatchSize * vPatchSize * vNMerge * vNMerge;
    vImageMinPixels      = (vCustomImageMinTokens > 0 ? vCustomImageMinTokens : pNTokensMin) * cPatchArea;
    vImageMaxPixels      = (vCustomImageMaxTokens > 0 ? vCustomImageMaxTokens : pNTokensMax) * cPatchArea;
    vWarmupImageSize     = static_cast<int>(std::sqrt(vImageMaxPixels));
  }

  // used by longest_edge preprocessor (no model-specific value for min/max tokens)
  void mSetLimitImageTokens() {
    const int cPatchArea = vPatchSize * vPatchSize * vNMerge * vNMerge;
    if (vCustomImageMinTokens > 0) {
      vImageMinPixels = vCustomImageMinTokens * cPatchArea;
    }
    if (vCustomImageMaxTokens > 0) {
      vImageMaxPixels = vCustomImageMaxTokens * cPatchArea;
    }
  }

  void mSetWarmupNTokens(int pNTokens) {
    int vNTokPerSide = static_cast<int>(std::sqrt(pNTokens));
    GGML_ASSERT(vNTokPerSide * vNTokPerSide == pNTokens && "n_tokens must be n*n");
    // do not warmup with more tokens than the max allowed
    if (vCustomImageMaxTokens > 0 && pNTokens > vCustomImageMaxTokens) {
      vNTokPerSide = std::max(1, static_cast<int>(std::sqrt(vCustomImageMaxTokens)));
    }
    vWarmupImageSize = vNTokPerSide * vPatchSize * vNMerge;
  }

  // sam vit deepseek-ocr
  std::vector<int32_t> mGlobalAttnIndices() const { return { 2, 5, 8, 11 }; }

  bool mIsGlobalAttn(int32_t layer) const {
    const auto lIndices = mGlobalAttnIndices();

    for (const auto & vIdx : lIndices) {
      if (layer == vIdx) {
        return true;
      }
    }

    return false;
  }

  bool mIsFeatureLayer(int32_t layer) const {
    return std::find(lFeatureLayers.begin(), lFeatureLayers.end(), layer) != lFeatureLayers.end();
  }
};

struct clip_layer {
  // layernorm 1 (or layer input norm, or pre-attention norm)
  ggml_tensor * vLn1W = nullptr;
  ggml_tensor * vLn1B = nullptr;

  // attention
  ggml_tensor * vKW   = nullptr;
  ggml_tensor * vKB   = nullptr;
  ggml_tensor * vQW   = nullptr;
  ggml_tensor * vQB   = nullptr;
  ggml_tensor * vVW   = nullptr;
  ggml_tensor * vVB   = nullptr;
  ggml_tensor * vQkvW = nullptr;
  ggml_tensor * vQkvB = nullptr;

  ggml_tensor * vOW = nullptr;
  ggml_tensor * vOB = nullptr;

  ggml_tensor * vAttnSinks = nullptr;

  ggml_tensor * vKNorm = nullptr;
  ggml_tensor * vQNorm = nullptr;

  ggml_tensor * vAttnPostNormW = nullptr;

  ggml_tensor * vFfUpW   = nullptr;
  ggml_tensor * vFfUpB   = nullptr;
  ggml_tensor * vFfGateW = nullptr;
  ggml_tensor * vFfGateB = nullptr;
  ggml_tensor * vFfDownW = nullptr;
  ggml_tensor * vFfDownB = nullptr;

  // MoE FFN (dots3note vision pyramid blocks)
  ggml_tensor * vFfGateInpW  = nullptr;
  ggml_tensor * vFfGateExpsW = nullptr;
  ggml_tensor * vFfUpExpsW   = nullptr;
  ggml_tensor * vFfDownExpsW = nullptr;
  ggml_tensor * vFfExpProbsB = nullptr;

  // layernorm 2 (or pre-FFN norm)
  ggml_tensor * vLn2W = nullptr;
  ggml_tensor * vLn2B = nullptr;

  ggml_tensor * vFfPostNormW = nullptr;

  // layer scale (no bias)
  ggml_tensor * vLs1W   = nullptr;
  ggml_tensor * vLs2W   = nullptr;
  ggml_tensor * vLsOutW = nullptr;  // gemma4

  // qwen3vl deepstack merger
  ggml_tensor * vDeepstackNormW = nullptr;
  ggml_tensor * vDeepstackNormB = nullptr;
  ggml_tensor * vDeepstackFc1W  = nullptr;
  ggml_tensor * vDeepstackFc1B  = nullptr;
  ggml_tensor * vDeepstackFc2W  = nullptr;
  ggml_tensor * vDeepstackFc2B  = nullptr;

  // sam rel_pos
  ggml_tensor * vRelPosW    = nullptr;
  ggml_tensor * vRelPosH    = nullptr;
  // lfm2
  ggml_tensor * vFfNormW    = nullptr;
  ggml_tensor * vFfNormB    = nullptr;
  ggml_tensor * vFfNorm1W   = nullptr;
  ggml_tensor * vFfNorm1B   = nullptr;
  ggml_tensor * vFfUp1W     = nullptr;
  ggml_tensor * vFfUp1B     = nullptr;
  ggml_tensor * vFfDown1W   = nullptr;
  ggml_tensor * vFfDown1B   = nullptr;
  ggml_tensor * vPosBiasU   = nullptr;
  ggml_tensor * vPosBiasV   = nullptr;
  ggml_tensor * vNormConvW  = nullptr;
  ggml_tensor * vNormConvB  = nullptr;
  ggml_tensor * vLinearPosW = nullptr;

  ggml_tensor * vConvNormW    = nullptr;
  ggml_tensor * vConvNormB    = nullptr;
  ggml_tensor * vConvNormMean = nullptr;  // parakeet
  ggml_tensor * vConvNormVar  = nullptr;  // parakeet
  ggml_tensor * vConvDwW      = nullptr;
  ggml_tensor * vConvDwB      = nullptr;
  ggml_tensor * vConvPw1W     = nullptr;
  ggml_tensor * vConvPw1B     = nullptr;
  ggml_tensor * vConvPw2W     = nullptr;
  ggml_tensor * vConvPw2B     = nullptr;

  // gemma4 audio conformer per-layer
  ggml_tensor * vAttnPreNormW  = nullptr;
  ggml_tensor * vAttnKRelW     = nullptr;
  ggml_tensor * vPerDimScaleW  = nullptr;
  ggml_tensor * vPerDimKScaleW = nullptr;
  ggml_tensor * vFfPostNorm1W  = nullptr;

  // granite_speech conformer per-layer
  ggml_tensor * vAttnRelPosEmb = nullptr;

  // granite_speech qformer cross-attention
  ggml_tensor * vCrossAttnQW    = nullptr;
  ggml_tensor * vCrossAttnQB    = nullptr;
  ggml_tensor * vCrossAttnKW    = nullptr;
  ggml_tensor * vCrossAttnKB    = nullptr;
  ggml_tensor * vCrossAttnVW    = nullptr;
  ggml_tensor * vCrossAttnVB    = nullptr;
  ggml_tensor * vCrossAttnOW    = nullptr;
  ggml_tensor * vCrossAttnOB    = nullptr;
  ggml_tensor * vCrossAttnNormW = nullptr;
  ggml_tensor * vCrossAttnNormB = nullptr;

  // qwen3tts speaker encoder: SE-Res2Net block, tdnn1/tdnn2 reuse conv_pw1_w/b and conv_pw2_w/b above
  ggml_tensor *              vSeConv1W = nullptr;
  ggml_tensor *              vSeConv1B = nullptr;
  ggml_tensor *              vSeConv2W = nullptr;
  ggml_tensor *              vSeConv2B = nullptr;
  std::vector<ggml_tensor *> lRes2ConvW;  // Res2Net hierarchical branches
  std::vector<ggml_tensor *> lRes2ConvB;

  bool mHasDeepstack() const { return vDeepstackFc1W != nullptr; }
};

// Expanded MobileNetV5 block structure for Gemma3n vision encoder
struct mobilenetv5_block {
  // Stage 0 (Edge Residual)
  ggml_tensor * vS0ConvExpW = nullptr;
  ggml_tensor * vS0Bn1W     = nullptr;
  ggml_tensor * vS0ConvPwlW = nullptr;
  ggml_tensor * vS0Bn2W     = nullptr;

  // Stage 1+ (Universal Inverted Residual)
  ggml_tensor * vDwStartW   = nullptr;
  ggml_tensor * vDwStartBnW = nullptr;

  ggml_tensor * vPwExpW   = nullptr;
  ggml_tensor * vPwExpBnW = nullptr;

  ggml_tensor * vDwMidW   = nullptr;
  ggml_tensor * vDwMidBnW = nullptr;

  ggml_tensor * vPwProjW   = nullptr;
  ggml_tensor * vPwProjBnW = nullptr;

  ggml_tensor * vLayerScaleW = nullptr;

  // Attention (MQA) components
  ggml_tensor * vAttnQW = nullptr;
  ggml_tensor * vAttnKW = nullptr;
  ggml_tensor * vAttnVW = nullptr;
  ggml_tensor * vAttnOW = nullptr;

  // Optional downsampling/norm in attention
  ggml_tensor * vAttnKDwW   = nullptr;
  ggml_tensor * vAttnKNormW = nullptr;
  ggml_tensor * vAttnVDwW   = nullptr;
  ggml_tensor * vAttnVNormW = nullptr;

  // Block norm (often present in attention blocks)
  ggml_tensor * vAttnNormW = nullptr;
};

struct yasa2_block {
  ggml_tensor * vDwW  = nullptr;
  ggml_tensor * vDwB  = nullptr;
  ggml_tensor * vLnW  = nullptr;
  ggml_tensor * vLnB  = nullptr;
  ggml_tensor * vPw1W = nullptr;
  ggml_tensor * vPw1B = nullptr;
  ggml_tensor * vGrnW = nullptr;
  ggml_tensor * vGrnB = nullptr;
  ggml_tensor * vPw2W = nullptr;
  ggml_tensor * vPw2B = nullptr;
};

struct yasa2_stage {
  ggml_tensor *            vDownLnW   = nullptr;
  ggml_tensor *            vDownLnB   = nullptr;
  ggml_tensor *            vDownConvW = nullptr;
  ggml_tensor *            vDownConvB = nullptr;
  std::vector<yasa2_block> lBlocks;
};

// QFormer projector block for models with 1 (or more) QFormer projectors
// Granite Speech, Granite4 Vision
struct qf_block {
  ggml_tensor *           vQfProjQuery     = nullptr;
  ggml_tensor *           vQfProjNormW     = nullptr;
  ggml_tensor *           vQfProjNormB     = nullptr;
  ggml_tensor *           vQfProjLinearW   = nullptr;
  ggml_tensor *           vQfProjLinearB   = nullptr;
  ggml_tensor *           vQfProjPostNormW = nullptr;
  ggml_tensor *           vQfProjPostNormB = nullptr;
  ggml_tensor *           vQfProjImgPos    = nullptr;  // Vision only
  std::vector<clip_layer> lQfProjLayers;
};

// pocket-tts SEANet stack, used in both directions:
// encoder = conv_in -> per stage (residual unit, strided conv) -> conv_out
// decoder = conv_in -> per stage (strided convtr, residual unit) -> conv_out
struct clip_seanet {
  // one residual unit: ELU -> dilated conv -> ELU -> pointwise conv, added to the input
  struct stage {
    ggml_tensor * vResConv1W  = nullptr;
    ggml_tensor * vResConv1B  = nullptr;
    ggml_tensor * vResConv2W  = nullptr;
    ggml_tensor * vResConv2B  = nullptr;
    ggml_tensor * vScaleConvW = nullptr;  // strided conv (encoder) or convtr (decoder)
    ggml_tensor * vScaleConvB = nullptr;
  };

  ggml_tensor *      vConvInW  = nullptr;
  ggml_tensor *      vConvInB  = nullptr;
  ggml_tensor *      vConvOutW = nullptr;
  ggml_tensor *      vConvOutB = nullptr;
  std::vector<stage> lStages;
};

// pocket-tts flow-matching decoder (SimpleMLPAdaLN)
struct clip_flow_net {
  // AdaLN res block: in_ln -> modulate -> Linear -> SiLU -> Linear, gated residual
  struct block {
    ggml_tensor * vNormW = nullptr;
    ggml_tensor * vNormB = nullptr;
    ggml_tensor * vUpW   = nullptr;
    ggml_tensor * vUpB   = nullptr;
    ggml_tensor * vDownW = nullptr;
    ggml_tensor * vDownB = nullptr;
    ggml_tensor * vAdaW  = nullptr;  // -> shift, scale, gate
    ggml_tensor * vAdaB  = nullptr;
  };

  // timestep embedder: cos/sin(t * freqs) -> Linear -> SiLU -> Linear -> RMSNorm
  struct time_embd {
    ggml_tensor * vFreqs = nullptr;
    ggml_tensor * vUpW   = nullptr;
    ggml_tensor * vUpB   = nullptr;
    ggml_tensor * vDownW = nullptr;
    ggml_tensor * vDownB = nullptr;
    ggml_tensor * vNorm  = nullptr;  // RMSNorm alpha
  };

  ggml_tensor *          vInputProjW = nullptr;
  ggml_tensor *          vInputProjB = nullptr;
  ggml_tensor *          vCondEmbdW  = nullptr;
  ggml_tensor *          vCondEmbdB  = nullptr;
  ggml_tensor *          vFinalAdaW  = nullptr;  // -> shift, scale
  ggml_tensor *          vFinalAdaB  = nullptr;
  ggml_tensor *          vFinalProjW = nullptr;
  ggml_tensor *          vFinalProjB = nullptr;
  std::vector<time_embd> lTime;
  std::vector<block>     lBlocks;
};

// qwen3tts code2wav: RVQ codes -> raw PCM
struct clip_code2wav {
  // "upsample" stage: one ConvNeXt block plus the causal ConvTranspose1d before it
  struct upsample_block {
    ggml_tensor * vConvW   = nullptr;  // causal ConvTranspose1d, 2x
    ggml_tensor * vConvB   = nullptr;
    ggml_tensor * vDwconvW = nullptr;  // depthwise causal conv, k=7
    ggml_tensor * vDwconvB = nullptr;
    ggml_tensor * vNormW   = nullptr;  // LayerNorm
    ggml_tensor * vNormB   = nullptr;
    ggml_tensor * vPw1W    = nullptr;  // pointwise expand
    ggml_tensor * vPw1B    = nullptr;
    ggml_tensor * vPw2W    = nullptr;  // pointwise project
    ggml_tensor * vPw2B    = nullptr;
    ggml_tensor * vGamma   = nullptr;  // layer scale
  };

  // one DAC residual unit: SnakeBeta -> dilated causal conv -> SnakeBeta -> pointwise causal conv
  struct dac_res {
    ggml_tensor * vAct1Alpha = nullptr;
    ggml_tensor * vAct1Beta  = nullptr;
    ggml_tensor * vConv1W    = nullptr;
    ggml_tensor * vConv1B    = nullptr;
    ggml_tensor * vAct2Alpha = nullptr;
    ggml_tensor * vAct2Beta  = nullptr;
    ggml_tensor * vConv2W    = nullptr;
    ggml_tensor * vConv2B    = nullptr;
  };

  // one DAC upsample block (SnakeBeta -> causal ConvTranspose1d -> 3 residual units)
  struct dac_block {
    ggml_tensor *        vSnakeAlpha = nullptr;
    ggml_tensor *        vSnakeBeta  = nullptr;
    ggml_tensor *        vConvW      = nullptr;  // causal ConvTranspose1d
    ggml_tensor *        vConvB      = nullptr;
    std::vector<dac_res> lRes;
  };

  // quantizer: RVQ codebook decode
  ggml_tensor * vQuantFirstInW  = nullptr;  // semantic RVQ, in_proj (1x1 conv, loaded as 2D)
  ggml_tensor * vQuantFirstOutW = nullptr;
  ggml_tensor * vQuantFirstCbW  = nullptr;  // codebook (1 layer)
  ggml_tensor * vQuantRestInW   = nullptr;  // acoustic RVQ
  ggml_tensor * vQuantRestOutW  = nullptr;
  ggml_tensor * vQuantRestCbW   = nullptr;  // codebooks, merged 3D [15, vocab, dim]

  ggml_tensor * vPreConvW = nullptr;
  ggml_tensor * vPreConvB = nullptr;

  ggml_tensor *           vTfmInProjW     = nullptr;
  ggml_tensor *           vTfmInProjB     = nullptr;
  ggml_tensor *           vTfmOutProjW    = nullptr;
  ggml_tensor *           vTfmOutProjB    = nullptr;
  ggml_tensor *           vTfmOutputNormW = nullptr;
  std::vector<clip_layer> lTfmLayers;  // reuses the generic block fields (ln_1/attn/ln_2/ffn/ls_1/ls_2)

  std::vector<upsample_block> lUpsample;

  ggml_tensor *          vDacEntryW = nullptr;
  ggml_tensor *          vDacEntryB = nullptr;
  std::vector<dac_block> lDac;
  ggml_tensor *          vDacPostSnakeAlpha = nullptr;
  ggml_tensor *          vDacPostSnakeBeta  = nullptr;
  ggml_tensor *          vDacPostConvW      = nullptr;
  ggml_tensor *          vDacPostConvB      = nullptr;
};

struct clip_model {
  clip_modality  vModality = cClipModalityVision;
  projector_type vProjType = cProjectorTypeMlp;
  clip_hparams   vHparams;

  // embeddings
  ggml_tensor * vClassEmbedding   = nullptr;
  ggml_tensor * vPatchEmbeddings0 = nullptr;
  ggml_tensor * vPatchEmbeddings1 =
    nullptr;  // second Conv2D kernel when we decouple Conv3D along temporal dimension (Qwen2VL)
  ggml_tensor * vPatchBias          = nullptr;
  ggml_tensor * vPositionEmbeddings = nullptr;
  ggml_tensor * vNormEmbdW          = nullptr;
  ggml_tensor * vNormEmbdB          = nullptr;

  // "indexed" patch embedding norms
  ggml_tensor * vPatchNorm1W = nullptr;
  ggml_tensor * vPatchNorm1B = nullptr;
  ggml_tensor * vPatchNorm2W = nullptr;
  ggml_tensor * vPatchNorm2B = nullptr;
  ggml_tensor * vPatchNorm3W = nullptr;
  ggml_tensor * vPatchNorm3B = nullptr;

  ggml_tensor * vPreLnW = nullptr;
  ggml_tensor * vPreLnB = nullptr;

  std::vector<clip_layer> lLayers;

  int32_t vNDeepstackLayers = 0;  // used by Qwen3-VL, calculated from clip_layer

  ggml_tensor * vPostLnW;
  ggml_tensor * vPostLnB;

  ggml_tensor * vMmFcW;
  ggml_tensor * vMmFcB;
  ggml_tensor * vMmFfnUpW    = nullptr;
  ggml_tensor * vMmFfnUpB    = nullptr;
  ggml_tensor * vMmFfnGateW  = nullptr;
  ggml_tensor * vMmFfnGateB  = nullptr;
  ggml_tensor * vMmFfnDownW  = nullptr;
  ggml_tensor * vMmFfnDownB  = nullptr;
  ggml_tensor * vMmPostNormW = nullptr;
  ggml_tensor * vMmPostNormB = nullptr;

  // LLaVA projection
  ggml_tensor * vMmInputNormW = nullptr;
  ggml_tensor * vMmInputNormB = nullptr;
  ggml_tensor * vMm0W         = nullptr;
  ggml_tensor * vMm0B         = nullptr;
  ggml_tensor * vMm2W         = nullptr;
  ggml_tensor * vMm2B         = nullptr;
  ggml_tensor * vMmMergerFc1W = nullptr;  // minimax-m3
  ggml_tensor * vMmMergerFc1B = nullptr;
  ggml_tensor * vMmMergerFc2W = nullptr;
  ggml_tensor * vMmMergerFc2B = nullptr;

  ggml_tensor * vImageNewline  = nullptr;
  ggml_tensor * vViewSeperator = nullptr;
  ggml_tensor * vTokEmbdSep    = nullptr;  // [n_embd_text, n_sep] rows of the text model tok_embd (MiniCPM-V 4.7)

  // Yi type models with mlp+normalization projection
  ggml_tensor * vMm1W = nullptr;  // Yi type models have 0, 1, 3, 4
  ggml_tensor * vMm1B = nullptr;
  ggml_tensor * vMm3W = nullptr;
  ggml_tensor * vMm3B = nullptr;
  ggml_tensor * vMm4W = nullptr;
  ggml_tensor * vMm4B = nullptr;
  ggml_tensor * vMm5W = nullptr;
  ggml_tensor * vMm5B = nullptr;
  ggml_tensor * vMm6W = nullptr;
  ggml_tensor * vMm6B = nullptr;

  // GLMV-Edge projection
  ggml_tensor * vMmModelAdapterConvW = nullptr;
  ggml_tensor * vMmModelAdapterConvB = nullptr;

  // MobileVLM projection
  ggml_tensor * vMmModelMlp1W               = nullptr;
  ggml_tensor * vMmModelMlp1B               = nullptr;
  ggml_tensor * vMmModelMlp3W               = nullptr;
  ggml_tensor * vMmModelMlp3B               = nullptr;
  ggml_tensor * vMmModelBlock1Block0Point0W = nullptr;
  ggml_tensor * vMmModelBlock1Block0Point1W = nullptr;
  ggml_tensor * vMmModelBlock1Block0Point1B = nullptr;
  ggml_tensor * vMmModelBlock1Block1Fc1W    = nullptr;
  ggml_tensor * vMmModelBlock1Block1Fc1B    = nullptr;
  ggml_tensor * vMmModelBlock1Block1Fc2W    = nullptr;
  ggml_tensor * vMmModelBlock1Block1Fc2B    = nullptr;
  ggml_tensor * vMmModelBlock1Block2Point0W = nullptr;
  ggml_tensor * vMmModelBlock1Block2Point1W = nullptr;
  ggml_tensor * vMmModelBlock1Block2Point1B = nullptr;
  ggml_tensor * vMmModelBlock2Block0Point0W = nullptr;
  ggml_tensor * vMmModelBlock2Block0Point1W = nullptr;
  ggml_tensor * vMmModelBlock2Block0Point1B = nullptr;
  ggml_tensor * vMmModelBlock2Block1Fc1W    = nullptr;
  ggml_tensor * vMmModelBlock2Block1Fc1B    = nullptr;
  ggml_tensor * vMmModelBlock2Block1Fc2W    = nullptr;
  ggml_tensor * vMmModelBlock2Block1Fc2B    = nullptr;
  ggml_tensor * vMmModelBlock2Block2Point0W = nullptr;
  ggml_tensor * vMmModelBlock2Block2Point1W = nullptr;
  ggml_tensor * vMmModelBlock2Block2Point1B = nullptr;

  // MobileVLM_V2 projection
  ggml_tensor * vMmModelMlp0W = nullptr;
  ggml_tensor * vMmModelMlp0B = nullptr;
  ggml_tensor * vMmModelMlp2W = nullptr;
  ggml_tensor * vMmModelMlp2B = nullptr;
  ggml_tensor * vMmModelPeg0W = nullptr;
  ggml_tensor * vMmModelPeg0B = nullptr;

  // MINICPMV projection
  ggml_tensor * vMmModelPosEmbedK = nullptr;
  ggml_tensor * vMmModelQuery     = nullptr;
  ggml_tensor * vMmModelProj      = nullptr;
  ggml_tensor * vMmModelProjB     = nullptr;
  ggml_tensor * vMmModelKvProj    = nullptr;
  ggml_tensor * vMmModelAttnQW    = nullptr;
  ggml_tensor * vMmModelAttnQB    = nullptr;
  ggml_tensor * vMmModelAttnKW    = nullptr;
  ggml_tensor * vMmModelAttnKB    = nullptr;
  ggml_tensor * vMmModelAttnVW    = nullptr;
  ggml_tensor * vMmModelAttnVB    = nullptr;
  ggml_tensor * vMmModelAttnOW    = nullptr;
  ggml_tensor * vMmModelAttnOB    = nullptr;
  ggml_tensor * vMmModelLnQW      = nullptr;
  ggml_tensor * vMmModelLnQB      = nullptr;
  ggml_tensor * vMmModelLnKvW     = nullptr;
  ggml_tensor * vMmModelLnKvB     = nullptr;
  ggml_tensor * vMmModelLnPostW   = nullptr;
  ggml_tensor * vMmModelLnPostB   = nullptr;

  // MiniCPM-V 4.6 ViT merger (window self-attention + ViT MLP downsample)
  ggml_tensor * vVitMergerLn1W    = nullptr;
  ggml_tensor * vVitMergerLn1B    = nullptr;
  ggml_tensor * vVitMergerAttnQW  = nullptr;
  ggml_tensor * vVitMergerAttnQB  = nullptr;
  ggml_tensor * vVitMergerAttnKW  = nullptr;
  ggml_tensor * vVitMergerAttnKB  = nullptr;
  ggml_tensor * vVitMergerAttnVW  = nullptr;
  ggml_tensor * vVitMergerAttnVB  = nullptr;
  ggml_tensor * vVitMergerAttnOW  = nullptr;
  ggml_tensor * vVitMergerAttnOB  = nullptr;
  ggml_tensor * vVitMergerDsLnW   = nullptr;
  ggml_tensor * vVitMergerDsLnB   = nullptr;
  ggml_tensor * vVitMergerDsUpW   = nullptr;
  ggml_tensor * vVitMergerDsUpB   = nullptr;
  ggml_tensor * vVitMergerDsDownW = nullptr;
  ggml_tensor * vVitMergerDsDownB = nullptr;

  // gemma3
  ggml_tensor * vMmInputProjW   = nullptr;
  ggml_tensor * vMmSoftEmbNormW = nullptr;

  // mobilenetv5 for gemma3n
  std::vector<mobilenetv5_block> lMobilenetBlocks;
  std::vector<int>               lMobilenetStageEnds;
  ggml_tensor *                  vMobilenetStemConvW = nullptr;
  ggml_tensor *                  vMobilenetStemConvB = nullptr;
  ggml_tensor *                  vMobilenetStemNormW = nullptr;
  ggml_tensor *                  vMmPostProjNormW    = nullptr;

  // Multi-Scale Fusion Adapter (MSFA) components
  ggml_tensor * vMsfaConcatConvW  = nullptr;
  ggml_tensor * vMsfaConcatNormW  = nullptr;
  ggml_tensor * vMsfaFfnExpandW   = nullptr;
  ggml_tensor * vMsfaFfnProjectW  = nullptr;
  ggml_tensor * vMsfaFfnExpandBn  = nullptr;
  ggml_tensor * vMsfaFfnProjectBn = nullptr;

  // yasa2
  ggml_tensor *            vYasaPatchW         = nullptr;
  ggml_tensor *            vYasaPatchB         = nullptr;
  ggml_tensor *            vYasaPatchLnW       = nullptr;
  ggml_tensor *            vYasaPatchLnB       = nullptr;
  ggml_tensor *            vYasaBackboneLnW    = nullptr;
  ggml_tensor *            vYasaBackboneLnB    = nullptr;
  ggml_tensor *            vYasaVisionPosEmbed = nullptr;
  std::vector<yasa2_stage> lYasaStages;

  // pixtral, glm4v
  ggml_tensor * vTokenEmbdImgBreak = nullptr;

  // deepseek4v sentinel embeddings (image_newline is reused for IMAGE_NEW_LINE)
  ggml_tensor * vTokenEmbdImgStart = nullptr;
  ggml_tensor * vTokenEmbdImgEnd   = nullptr;
  ggml_tensor * vTokenEmbdImgPad   = nullptr;
  ggml_tensor * vMmPatchMergerW    = nullptr;
  ggml_tensor * vMmPatchMergerB    = nullptr;

  // ultravox / whisper encoder
  ggml_tensor * vConv1d1W   = nullptr;
  ggml_tensor * vConv1d1B   = nullptr;
  ggml_tensor * vConv1d2W   = nullptr;
  ggml_tensor * vConv1d2B   = nullptr;
  ggml_tensor * vConvOutW   = nullptr;
  ggml_tensor * vConvOutB   = nullptr;
  ggml_tensor * vMmNormPreW = nullptr;
  ggml_tensor * vMmNormPreB = nullptr;
  ggml_tensor * vMmNormMidW = nullptr;

  // mimo-audio-tokenizer: post-transformer downsample + RVQ codebook
  ggml_tensor * vDownsampleConvW = nullptr;  // no bias
  ggml_tensor * vDownsampleNormW = nullptr;
  ggml_tensor * vDownsampleNormB = nullptr;
  ggml_tensor * vRvqCodebook     = nullptr;  // merged 3D [n_q, max_bins, dim]

  // mimo-v2.5: text-side RVQ code embedding ("text codebook")
  ggml_tensor * vMmACodeEmbd = nullptr;  // merged 3D [n_channels, vocab, dim]

  // mimo-v2.5: LLM-side connector (input_local_transformer, separate from the
  // audio_tokenizer's own encoder `layers`)
  std::vector<clip_layer> lMmALocalLayers;
  ggml_tensor *           vMmALocalNormW = nullptr;

  // qwen3a
  ggml_tensor * vConv2d1W = nullptr;
  ggml_tensor * vConv2d1B = nullptr;
  ggml_tensor * vConv2d2W = nullptr;
  ggml_tensor * vConv2d2B = nullptr;
  ggml_tensor * vConv2d3W = nullptr;
  ggml_tensor * vConv2d3B = nullptr;

  // qwen3tts speaker encoder (ECAPA-TDNN)
  // reused tensors: stem conv is conv1d_1_w/b, feature aggregation is conv_out_w/b, output proj is mm_fc_w/b
  ggml_tensor * vSpkAspAttnW = nullptr;
  ggml_tensor * vSpkAspAttnB = nullptr;
  ggml_tensor * vSpkAspTdnnW = nullptr;
  ggml_tensor * vSpkAspTdnnB = nullptr;

  // qwen3tts code_predictor
  ggml_tensor * vGenCodeProjInW  = nullptr;  // small_to_mtp_projection
  ggml_tensor * vGenCodeProjInB  = nullptr;
  ggml_tensor * vGenCodeEmbdW    = nullptr;  // per-codebook embedding, merged 3D
  ggml_tensor * vGenCodeHeadW    = nullptr;  // per-codebook output head, merged 3D
  ggml_tensor * vGenCodeOutEmbdW = nullptr;  // codebook-0 embedding, fed back into the talker
  ggml_tensor * vGenCodeNormW    = nullptr;  // final norm

  // qwen3tts code2wav: RVQ codes -> raw PCM
  clip_code2wav vC2w;

  // pocket-tts: SEANet stack, shared by the encoder (speaker path) and the decoder (gen path)
  clip_seanet vSeanet;

  // pocket-tts: voice latent -> backbone embd (speaker path)
  ggml_tensor * vSpkProjW    = nullptr;
  ggml_tensor * vDownsampleW = nullptr;

  // pocket-tts: flow-matching decoder, backbone hidden state -> next latent
  clip_flow_net           vFlow;
  ggml_tensor *           vGenOutEosW   = nullptr;
  ggml_tensor *           vGenOutEosB   = nullptr;
  ggml_tensor *           vGenInputLinW = nullptr;  // latent -> backbone embd
  ggml_tensor *           vGenEmbMean   = nullptr;
  ggml_tensor *           vGenEmbStd    = nullptr;
  ggml_tensor *           vGenQuantOutW = nullptr;  // latent -> decoder dim
  ggml_tensor *           vGenUpsampleW = nullptr;  // depthwise convtr, frame rate -> encoder frame rate
  std::vector<clip_layer> lGenTfmLayers;            // mimi decoder_transformer

  // cogvlm
  ggml_tensor * vMmPostFcNormW = nullptr;
  ggml_tensor * vMmPostFcNormB = nullptr;
  ggml_tensor * vMmHTo4hW      = nullptr;
  ggml_tensor * vMmGateW       = nullptr;
  ggml_tensor * vMm4hToHW      = nullptr;
  ggml_tensor * vMmBoi         = nullptr;
  ggml_tensor * vMmEoi         = nullptr;

  // hunyuanvl perceiver
  ggml_tensor * vMmPreNormW = nullptr;
  ggml_tensor * vMmImgBegin = nullptr;
  ggml_tensor * vMmImgEnd   = nullptr;

  // deepseek ocr sam
  ggml_tensor * vPatchEmbedProjW = nullptr;
  ggml_tensor * vPatchEmbedProjB = nullptr;
  ggml_tensor * vPosEmbed        = nullptr;

  ggml_tensor * vNeck0W;
  ggml_tensor * vNeck1W;
  ggml_tensor * vNeck1B;
  ggml_tensor * vNeck2W;
  ggml_tensor * vNeck3W;
  ggml_tensor * vNeck3B;
  ggml_tensor * vNet2;
  ggml_tensor * vNet3;

  int32_t vNSamLayers = 12;  // used by deepseek-ocr sam encoder

  std::vector<clip_layer> lSamLayers;

  // deepseek-ocr-2
  ggml_tensor * vResampleQuery768  = nullptr;
  ggml_tensor * vResampleQuery1024 = nullptr;

  // lfm2 audio
  std::array<ggml_tensor *, 7> aPreEncodeConvXW = { nullptr };
  std::array<ggml_tensor *, 7> aPreEncodeConvXB = { nullptr };
  ggml_tensor *                vPreEncodeOutW   = nullptr;
  ggml_tensor *                vPreEncodeOutB   = nullptr;

  // gemma4
  ggml_tensor * vStdBias  = nullptr;
  ggml_tensor * vStdScale = nullptr;

  // Gemma4ClippableLinear
  struct clamp_info {
    float vInpMax;
    float vInpMin;
    float vOutMax;
    float vOutMin;
  };

  std::map<std::string, clamp_info> dClampInfoMap;

  // gemma4 audio conformer
  std::array<ggml_tensor *, 2> aSscpConvW     = { nullptr };
  std::array<ggml_tensor *, 2> aSscpConvB     = { nullptr };
  std::array<ggml_tensor *, 2> aSscpNormW     = { nullptr };
  ggml_tensor *                vSscpInpProjW  = nullptr;
  ggml_tensor *                vSscpInpProjB  = nullptr;
  ggml_tensor *                vAudioOutProjW = nullptr;
  ggml_tensor *                vAudioOutProjB = nullptr;

  // granite_speech encoder
  ggml_tensor *         vInpProjW   = nullptr;
  ggml_tensor *         vInpProjB   = nullptr;
  ggml_tensor *         vCtcOutW    = nullptr;
  ggml_tensor *         vCtcOutB    = nullptr;
  ggml_tensor *         vCtcOutMidW = nullptr;
  ggml_tensor *         vCtcOutMidB = nullptr;
  // qformer projector(s)
  std::vector<qf_block> lQfProjBlocks;

  bool mAudioHasAvgpool() const {
    return vProjType == cProjectorTypeQwen2a || vProjType == cProjectorTypeVoxtral ||
           vProjType == cProjectorTypeMusicFlamingo;
  }

  bool mAudioHasStackFrames() const {
    return vProjType == cProjectorTypeUltravox || vProjType == cProjectorTypeVoxtral ||
           vProjType == cProjectorTypeMeralion;
  }
};

const clip_hparams * fClipGetHparams(const struct clip_ctx * pCtx);
