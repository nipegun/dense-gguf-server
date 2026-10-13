#pragma once

#include "dgs.h"

#include <array>
#include <bitset>
#include <cassert>
#include <cmath>

// bump if necessary
#define DGS_MAX_LAYERS    512
#define DGS_MAX_EXPERTS   1024  // Kimi K3
#define DGS_MAX_PLE_NGRAM 8     // qwen4exp
#define DGS_MAX_PLE_HEADS 64    // qwen4exp

enum dgs_expert_gating_func_type {
  cDgsExpertGatingFuncTypeNone          = 0,
  cDgsExpertGatingFuncTypeSoftmax       = 1,
  cDgsExpertGatingFuncTypeSigmoid       = 2,
  cDgsExpertGatingFuncTypeSoftmaxWeight = 3,  // applied to the router weights instead of the logits
  cDgsExpertGatingFuncTypeSqrtSoftplus  = 4,
};

enum dgs_swa_type {
  cDgsSwaTypeNone      = 0,
  cDgsSwaTypeStandard  = 1,
  cDgsSwaTypeChunked   = 2,
  cDgsSwaTypeSymmetric = 3,
};

// how the non-causal mask should be constructed with dgs_set_causal_attn(ctx, false)
// (e.g. mtmd decoding image tokens)
enum dgs_non_causal_type {
  cDgsNonCausalTypeAll     = 0,  // all layers non-causal, SWA still applied (gemma 3, qwen-vl, ...)
  cDgsNonCausalTypeSwaOnly = 1,  // SWA layers non-causal, dense layers stay causal (gemma 4)
  cDgsNonCausalTypeSwaFull =
    2,  // all layers non-causal, SWA not applied between tokens of the current ubatch (deepseek 4)
};

// M-RoPE: which input position slot feeds each RoPE section
enum dgs_rope_section_order {
  cDgsRopeSectionOrderUnspecified = -1,
  cDgsRopeSectionOrderTyxz        = 0,  // default, slot i feeds section i
  cDgsRopeSectionOrderZyxt        = 1,  // MiniCPM-V 4.7: time last
};

// forward declaration; full definition in dgs-graph.h
enum llm_ffn_op_type : int;

struct dgs_hparams_posnet {
  uint32_t vNEmbd;
  uint32_t vNLayer;
};

struct dgs_hparams_convnext {
  uint32_t vNEmbd;
  uint32_t vNLayer;
};

struct dgs_hparams {
  // note: use the `_impl` suffix to avoid name conflict between members and getters
  //       for example: n_embd_out() vs n_embd_out_impl

  bool vVocabOnly;
  bool vNoAlloc;
  bool vRopeFinetuned;
  bool vUseParRes;
  bool vSwinNorm;
  bool vNormBeforeResidual = false;
  bool vNormBeforeFc       = false;

  uint32_t n_ctx_train;  // context size the model was trained on
  uint32_t n_embd;
  uint32_t n_layer_all;
  uint32_t n_layer_nextn   = 0;
  uint32_t vNLayerDecision = 0;  // trailing blocks that form the decision head

  // granite-switch: index of the single-head "router" KV layer that encodes
  // per-token adapter selection. -1 when the model has no such layer.
  int32_t  vRouterLayer      = -1;
  uint32_t n_expert          = 0;
  uint32_t vNRelAttnBkts     = 0;
  uint32_t vNValueExpert     = 0;  // MoVA value experts (K2 Horizon)
  uint32_t vNValueExpertUsed = 0;

  // TODO: this needs to be reworked
  int32_t vNLayerKvFromStart = -1;  // if non-negative, the first n_layer_kv_from_start layers have KV cache

  // different head size for full_attention and SWA layers
  uint32_t
    vNEmbdHeadKFull;  // dimension of keys (d_k). d_q is assumed to be the same, but there are n_head q heads, and only n_head_kv k-v heads
  uint32_t vNEmbdHeadVFull;  // dimension of values (d_v) aka n_embd_head
  uint32_t vNEmbdHeadKSwa;
  uint32_t vNEmbdHeadVSwa;

  // different RoPE dimensions for full_attention and SWA layers
  uint32_t vNRotFull;
  uint32_t vNRotSwa;

  // note: deepseek2 using MLA converts into MQA with larger heads, then decompresses to MHA
  uint32_t vNEmbdHeadKMlaImpl = 0;
  uint32_t vNEmbdHeadVMlaImpl = 0;

  // for WavTokenizer
  struct dgs_hparams_posnet   vPosnet;
  struct dgs_hparams_convnext vConvnext;

  uint32_t vNShortconvLCache = 0;

  std::array<uint32_t, DGS_MAX_LAYERS> aNHeadArr;
  std::array<uint32_t, DGS_MAX_LAYERS> aNHeadKvArr;
  std::array<uint32_t, DGS_MAX_LAYERS> aNFfArr;

  // per-layer expert feed-forward size
  std::array<uint32_t, DGS_MAX_LAYERS> aNFfExpArr;
  // per-layer top-k expert routing count
  std::array<uint32_t, DGS_MAX_LAYERS> aNExpertUsedArr;

  uint32_t vNLayerDenseLead = 0;
  uint32_t vNLoraQ          = 0;
  uint32_t vNLoraKv         = 0;
  uint32_t vNFfShexp        = 0;
  uint32_t vNFfChexp        = 0;
  uint32_t vNExpertShared   = 0;
  uint32_t vNNormGroups     = 0;
  uint32_t vNExpertGroups   = 0;
  uint32_t vNGroupUsed      = 0;
  uint32_t vNGroupExperts   = 0;

  // MLA + SWA (i.e. dots3note)
  uint32_t vNLoraKvSwa       = 0;
  uint32_t vNEmbdHeadKMlaSwa = 0;
  uint32_t vNEmbdHeadVMlaSwa = 0;

  float    vExpertGroupScale   = 0.05f;
  float    vExpertWeightsScale = 0.0f;
  bool     vExpertWeightsNorm  = false;
  uint32_t vExpertGatingFunc   = cDgsExpertGatingFuncTypeNone;
  uint32_t vMoeEveryNLayers    = 0;
  uint32_t vMoeLatentSize      = 0;

  float vFNormEps;
  float vFNormRmsEps;
  float vFNormGroupEps;

  float vFAttnLogitSoftcapping   = 50.0f;
  float vFRouterLogitSoftcapping = 30.0f;
  float vFFinalLogitSoftcapping  = 30.0f;

  // for RWKV
  uint32_t vRescaleEveryNLayers = 0;
  uint32_t vTimeMixExtraDim     = 0;
  uint32_t vTimeDecayExtraDim   = 0;
  uint32_t vWkvHeadSize         = 0;
  uint32_t vTokenShiftCount     = 2;
  uint32_t vNLoraDecay          = 0;
  uint32_t vNLoraIclr           = 0;
  uint32_t vNLoraValueResMix    = 0;
  uint32_t vNLoraGate           = 0;

  float vRopeAttnFactor = 1.0f;
  float vRopeFreqBaseTrain;
  float vRopeFreqBaseTrainSwa = 10000.0f;
  float vRopeFreqScaleTrain;
  float vRopeFreqScaleTrainSwa = 1.0f;
  float vRopeScalingAlpha      = 0.0f;  // NTK-aware alpha for XDRoPE

  uint32_t vNCtxOrigYarn;
  float    vRopeYarnLogMul = 0.0f;

  float vYarnExtFactor  = -1.0f;
  float vYarnAttnFactor = 1.0f;
  float vYarnBetaFast   = 32.0f;
  float vYarnBetaSlow   = 1.0f;

  std::array<int, 4> aRopeSections;

  enum dgs_rope_section_order vRopeSectionOrder = cDgsRopeSectionOrderTyxz;

  // Per-layer RoPE enable flags (1 = use RoPE, 0 = NoPE)
  // by default, all layers use RoPE (controlled by rope_finetuned)
  std::array<uint32_t, DGS_MAX_LAYERS> aRopePattern;

  // Sliding Window Attention (SWA)
  dgs_swa_type vSwaType = cDgsSwaTypeNone;
  // the size of the sliding window (0 - no SWA)
  uint32_t     vNSwa    = 0;

  // see dgs_non_causal_type
  // note: for SWA_FULL, older tokens (outside the current ubatch) are still window-clipped
  dgs_non_causal_type vNonCausalType = cDgsNonCausalTypeAll;

  // if is_swa_impl[il] == 1, then layer il is SWA
  // if is_swa_impl[il] == 0, then layer il is dense (i.e. non-SWA)
  // by default, all layers are dense
  // note: using uint32_t type for compatibility reason
  std::array<uint32_t, DGS_MAX_LAYERS> aIsSwaImpl;

  // for hybrid state space models
  std::array<uint32_t, DGS_MAX_LAYERS> aIsRecrImpl;

  // for State Space Models
  uint32_t vSsmDConv  = 0;
  uint32_t vSsmDInner = 0;
  uint32_t vSsmDState = 0;
  uint32_t vSsmDtRank = 0;
  uint32_t vSsmNGroup = 0;

  // for MiniMax-Text-01 linear attention
  uint32_t vNEmbdHeadLa = 0;

  // for Kimi Linear KDA
  uint32_t vNEmbdHeadKda = 0;
  bool     vKdaSafeGate  = false;

  // kimi-k3
  uint32_t vNExpertLatent     = 0;  // routed_expert_hidden_size (0 = experts run at n_embd)
  uint32_t vAttnResBlockSize  = 0;  // 0 = no cross-layer attention residuals
  float    vKdaGateLowerBound = -INFINITY;
  float    vSituBeta          = 1.0f;
  float    vSituLinearBeta    = 0.0f;  // 0 = no linear-beta transform on the up branch

  // hrm-text (looped H/L stacks)
  uint32_t vNHrmLayersPerStack = 0;
  uint32_t vNHrmHCycles        = 0;
  uint32_t vNHrmLCycles        = 0;
  bool     vHrmPrefixLm        = false;

  bool vSsmDtBCRms = false;

  float vFClampKqv     = 0.0f;
  float vFMaxAlibiBias = 0.0f;
  float vFLogitScale   = 0.0f;

  // Additional scale factors (Granite/Granite MoE)
  float vFResidualScale  = 0.0f;
  float vFEmbeddingScale = 0.0f;
  float vFAttentionScale = 0.0f;

  // grok-2
  float    vFAttnOutScale  = 0.0f;
  uint32_t vAttnTempLength = 0;

  float vFAttnValueScale = 0.0f;

  bool vCausalAttn  = true;
  bool vUseAlibi    = false;
  bool vAttnSoftCap = false;
  bool vUseKqNorm   = false;

  // for Classifiers
  uint32_t vNClsOut = 1;

  // input embedding dimension (0 = use n_embd)
  uint32_t vNEmbdInpImpl = 0;

  // encoder input embedding dimension (0 = use n_embd_inp())
  // e.g. the eagle3 encoder fuses target_layers * target_hidden features
  uint32_t vNEmbdInpEncImpl = 0;

  // output embedding dimension (0 = use n_embd)
  uint32_t vNEmbdOutImpl = 0;

  uint32_t vDflashBlockSize      = 0;
  uint32_t vDflashConvKernelSize = 0;
  uint32_t vDflashConvGroupSize  = 0;
  uint32_t vDflashSelectorRank   = 0;
  uint32_t vDflashSelectorTopK   = 0;

  // dgs4 smallthinker
  uint32_t vNMoeLayerStep       = 0;
  uint32_t vNNoRopeLayerStep    = 4;
  uint32_t vNAttnTempFloorScale = 0;
  float    vFAttnTempScale      = 0.0f;
  float    vFAttnTempOffset     = 0.0f;  // offset position index

  // gemma3n altup
  uint32_t vNAltup     = 4;  // altup_num_inputs
  uint32_t vIAltupAct  = 0;  // altup_active_idx
  uint32_t vLaurelRank = 64;
  uint32_t vNEmbdAltup = 256;

  // needed for sentence-transformers dense layers
  uint32_t vDense2FeatIn  = 0;  // in_features of the 2_Dense
  uint32_t vDense2FeatOut = 0;  // out_features of the 2_Dense
  uint32_t vDense3FeatIn  = 0;  // in_features of the 3_Dense
  uint32_t vDense3FeatOut = 0;  // out_features of the 3_Dense

  // xIELU
  std::array<float, DGS_MAX_LAYERS> aXieluAlphaN;
  std::array<float, DGS_MAX_LAYERS> aXieluAlphaP;
  std::array<float, DGS_MAX_LAYERS> aXieluBeta;
  std::array<float, DGS_MAX_LAYERS> aXieluEps;

  // DSA (deepseek sparse attention)
  uint32_t vIndexerNHead           = 0;
  uint32_t vIndexerHeadSize        = 0;
  uint32_t vIndexerTopK            = 0;
  uint32_t vIndexerKpool           = 0;  // k-pool size
  bool     vIndexerKpoolSelectTail = true;
  // head-size slots per cached indexer row, the last one holds the pooled key
  uint32_t vIndexerKpoolRow        = 3;
  // pools are consecutive cells in sequence order, not runs of consecutive positions
  bool     vIndexerKpoolByOrder    = false;
  // MSA
  uint32_t vIndexerBlockSize       = 0;
  uint32_t vIndexerLocalBlocks     = 0;

  // Indexer is "full" (1) or "shared" (0)
  // Shared indexers reuse top-k from previous full layer
  std::array<uint32_t, DGS_MAX_LAYERS> aIsIndexerFullImpl;

  // DeepSeek-V4
  uint32_t                             vDsv4OGroupCount      = 0;
  uint32_t                             vDsv4OLoraRank        = 0;
  uint32_t                             vDsv4HcMult           = 0;
  uint32_t                             vDsv4HcSinkhornIters  = 0;
  uint32_t                             vDsv4HashLayerCount   = 0;
  float                                vDsv4CompressRopeBase = 0.0f;
  float                                vDsv4HcEps            = 0.0f;
  std::array<uint32_t, DGS_MAX_LAYERS> aDsv4CompressRatios;

  // 0 = full rank (DeepSeek-V4)
  uint32_t vHcLowRank = 0;

  // scale of the hyper-connection post gate (DeepSeek-V4 hardcodes 2.0)
  float vHcMagnitude = 0.0f;

  uint32_t                                vPleNgramSize     = 0;
  uint32_t                                vPleHeadsPerNgram = 0;
  uint32_t                                vPleConvKernel    = 0;
  uint32_t                                vPleNHeads        = 0;  // (ngram_size - 1) * heads_per_ngram
  uint32_t                                vPleHeadDim       = 0;
  uint32_t                                vPleEosTokenId    = 0;
  // the id the PLE hash stands in at image positions; 0 makes the loader fall back to EOS
  uint32_t                                vPleImageTokenId  = 0;
  // the file lists PLE layer indices, so this is never a per-layer gguf array and can hold one bit per layer
  std::bitset<DGS_MAX_LAYERS>             vIsPleImpl;
  // the hash multipliers reach ~2e13 and have to stay 64-bit
  std::array<uint64_t, DGS_MAX_PLE_NGRAM> aPleLayerMultipliers;
  // head offsets and vocab sizes are token-space indices; the gather truncates them to int32 anyway
  std::array<uint32_t, DGS_MAX_PLE_HEADS> aPleHeadOffsets;
  std::array<uint32_t, DGS_MAX_PLE_HEADS> aPleHeadVocabSizes;

  bool mIsPle(uint32_t pIl) const;

  // PLE conv history rows: (kernel - 1) * ngram_size; 0 without a PLE module
  uint32_t mPleConvState() const;

  // qwen3vl deepstack
  // When parsed from GGUF, this implies the first N layers consume the first
  // N deepstack embeddings. Use deepstack_mapping_arr if you need a more
  // complex mapping. If using deepstack_mapping_arr, also make sure to set
  // n_deepstack_layers to the number of unique deepstack layers so that
  // n_embd_imp is accurate (see granite.cpp).
  // TODO: can be expressed via the `new n_embd_inp_impl` and remove this param
  uint32_t vNDeepstackLayers = 0;

  // deepstack layer array (Granite4 Vision)
  // -1  => no deepstack
  // >=0 => input embedding index for deepstack injection
  std::array<int32_t, DGS_MAX_LAYERS> aDeepstackMappingArr;

  // gemma4 per-layer embedding
  uint32_t vNEmbdPerLayer = 0;

  // needed by encoder-decoder models (e.g. T5, FLAN-T5)
  dgs_token vDecStartTokenId = DGS_TOKEN_NULL;
  uint32_t  vDecNLayer       = 0;

  enum fDgsPoolingType       vPoolingType    = cDgsPoolingTypeNone;
  enum fDgsPoolingType       vPoolingTypeCls = cDgsPoolingTypeUnspecified;  // pooling before the classifier head (RANK)
  enum dgs_rope_type         vRopeType       = cDgsRopeTypeNone;
  enum dgs_rope_scaling_type vRopeScalingTypeTrain = cDgsRopeScalingTypeNone;

  // Resolved FFN gated activation flavor for archs that read
  // `<arch>.hidden_activation` from the GGUF (e.g. ModernBert derivatives).
  // Defaults to LLM_FFN_NONE (sentinel = 0); the mapping from the GGUF
  // string to a real op is done at hparam-load time via
  // llm_ffn_op_type_from_string() in dgs-model.cpp, mirroring how
  // rope_scaling_type_train is handled.
  enum llm_ffn_op_type vLlmFfnOp;
  enum ggml_unary_op   vActCls = GGML_UNARY_OP_TANH;  // activation of the classifier head (RANK)

  // Step35: optional per-layer clamps for (Swi)GLU
  std::array<float, DGS_MAX_LAYERS> aSwigluClampExp;    // clamping for expert FFN
  std::array<float, DGS_MAX_LAYERS> aSwigluClampShexp;  // shared expert

  // this value n_pattern means that every nth layer is dense (i.e. non-SWA)
  // dense_first means whether the pattern is start with a dense layer
  // note that if n_pattern == 0, all layers are SWA
  //           if n_pattern == 1, all layers are dense
  // example 1: n_pattern = 3, dense_first = false
  //   il == 0: swa
  //   il == 1: swa
  //   il == 2: dense
  //   il == 3: swa
  //   il == 4: swa
  //   il == 5: dense
  //   il == 6: swa
  //   etc ...
  // example 2: n_pattern = 2, dense_first = true
  //   il == 0: dense
  //   il == 1: swa
  //   il == 2: dense
  //   il == 3: swa
  //   etc ...
  void mSetSwaPattern(uint32_t pNPattern, bool pDenseFirst = false);

  // return true if one of the layers is SWA
  bool mIsSwaAny() const;

  bool mIsSwa(uint32_t pIl) const;

  bool mIsIndexerFull(uint32_t pIl) const;

  void mSetRecrPattern(uint32_t pNPattern, bool pDenseFirst = false);

  // whether or not the given layer is recurrent (for hybrid models)
  bool mIsRecr(uint32_t pIl) const;

  uint32_t n_head(uint32_t pIl = 0) const;

  uint32_t n_head_kv(uint32_t pIl = 0) const;

  uint32_t n_ff(uint32_t pIl = 0) const;

  uint32_t mNFfExp(uint32_t pIl = 0) const;

  uint32_t n_expert_used(uint32_t pIl = 0) const;

  // return the maximum n_expert_used across all layers
  uint32_t mNExpertUsedMax() const;

  uint32_t mNGqa(uint32_t pIl = 0) const;

  uint32_t n_rot(uint32_t pIl = 0) const;

  // dimension of main + auxiliary input embeddings
  uint32_t mNEmbdInp() const;

  // dimension of the encoder input embeddings
  uint32_t mNEmbdInpEnc() const;

  // dimension of output embeddings
  uint32_t mNEmbdOut() const;

  // dimension of key/value embeddings for each head (per layer)
  uint32_t n_embd_head_k(uint32_t pIl = 0) const;
  uint32_t n_embd_head_v(uint32_t pIl = 0) const;

  // dimension of key embeddings across all k-v heads
  uint32_t n_embd_k_gqa(uint32_t pIl = 0) const;

  // dimension of value embeddings across all k-v heads
  uint32_t n_embd_v_gqa(uint32_t pIl = 0) const;

  // true if any layer has a different n_embd_k_gqa/n_embd_v_gqa
  bool mIsNEmbdKGqaVariable() const;
  bool mIsNEmbdVGqaVariable() const;

  // return the maximum n_embd_k_gqa/n_embd_v_gqa across all layers
  uint32_t mNEmbdKGqaMax() const;
  uint32_t mNEmbdVGqaMax() const;

  // dimension of the rolling state embeddings
  // corresponds to Mamba's conv_states size or RWKV's token_shift states size
  uint32_t mNEmbdR() const;

  // dimension of the recurrent state embeddings
  uint32_t mNEmbdS() const;

  uint32_t mNPosPerEmbd() const;

  // note: currently only support if either all or none of the layers are MLA
  bool mIsMla() const;

  uint32_t mNEmbdHeadKMla() const;
  uint32_t mNEmbdHeadVMla() const;

  bool mHasKv(uint32_t pIl) const;

  bool mHasRope(uint32_t pIl) const;

  // number of effective layers (excludes nextn layers)
  uint32_t n_layer() const;

  // note that this function uses different SWA parameters from those in the hparams
  // note: inlined on purpose for performance reasons
  // TODO: think of a better place for this function
  // TODO: pack the SWA params in a struct?
  static bool mIsMaskedSwa(uint32_t pNSwa, dgs_swa_type pSwaType, dgs_pos pP0, dgs_pos pP1) {
    assert(p0 >= 0 && p1 >= 0);

    switch (pSwaType) {
      case cDgsSwaTypeNone:
        {
        }
        break;
      case cDgsSwaTypeStandard:
        {
          if (pP1 - pP0 >= (int32_t) pNSwa) {
            return true;
          }
        }
        break;
      case cDgsSwaTypeChunked:
        {
          const dgs_pos cPosChunkStart = (pP1 / pNSwa) * pNSwa;

          if (pP0 < cPosChunkStart) {
            return true;
          }
        }
        break;
      case cDgsSwaTypeSymmetric:
        {
          const int32_t cHalfNSwa = (int32_t) pNSwa / 2;
          const int32_t cPosDiff  = pP1 - pP0;

          // Mask if outside the symmetric window
          if (cPosDiff < -cHalfNSwa || cPosDiff > cHalfNSwa) {
            return true;
          }
        }
        break;
    }

    return false;
  }

  bool mUseMrope() const;
};

static_assert(std::is_trivially_copyable<dgs_hparams>::value, "dgs_hparams must be trivially copyable");
