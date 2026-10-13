#pragma once

#include "dgs-arch.h"
#include "dgs-graph.h"
#include "dgs-hparams.h"
#include "dgs-memory.h"
#include "dgs-vocab.h"
#include "dgs.h"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct dgs_cparams;
struct dgs_ubatch;
struct dgs_model_loader;
struct dgs_model;

// available models
enum llm_type {
  cLlmTypeUnknown,
  cLlmType14m,
  cLlmType17m,
  cLlmType22m,
  cLlmType33m,
  cLlmType47m,
  cLlmType60m,
  cLlmType70m,
  cLlmType80m,
  cLlmType109m,
  cLlmType137m,
  cLlmType140m,
  cLlmType149m,
  cLlmType160m,
  cLlmType190m,
  cLlmType220m,
  cLlmType230m,
  cLlmType250m,
  cLlmType256m,
  cLlmType270m,
  cLlmType335m,
  cLlmType350m,
  cLlmType360m,
  cLlmType395m,
  cLlmType410m,
  cLlmType450m,
  cLlmType475m,
  cLlmType558m,
  cLlmType700m,
  cLlmType770m,
  cLlmType780m,
  cLlmType950m,
  cLlmType0Point3B,
  cLlmType0Point5B,
  cLlmType0Point6B,
  cLlmType0Point8B,
  cLlmType1b,
  cLlmType1Point2B,
  cLlmType1Point3B,
  cLlmType1Point4B,
  cLlmType1Point5B,
  cLlmType1Point6B,
  cLlmType1Point7B,
  cLlmType1Point8B,
  cLlmType2b,
  cLlmType2Point6B,
  cLlmType2Point8B,
  cLlmType2Point9B,
  cLlmType3b,
  cLlmType4b,
  cLlmType6b,
  cLlmType6Point9B,
  cLlmType7b,
  cLlmType8b,
  cLlmType9b,
  cLlmType11b,
  cLlmType12b,
  cLlmType13b,
  cLlmType14b,
  cLlmType15b,
  cLlmType16b,
  cLlmType20b,
  cLlmType26b,
  cLlmType27b,
  cLlmType30b,
  cLlmType31b,
  cLlmType32b,
  cLlmType34b,
  cLlmType35b,
  cLlmType36b,
  cLlmType40b,
  cLlmType65b,
  cLlmType70b,
  cLlmType120b,
  cLlmType142b,
  cLlmType236b,
  cLlmType290b,
  cLlmType314b,
  cLlmType405b,
  cLlmType456b,
  cLlmType671b,
  cLlmTypeSmall,
  cLlmTypeMedium,
  cLlmTypeLarge,
  cLlmTypeXl,
  cLlmTypeA1Point7B,
  cLlmTypeA2Point7B,
  cLlmType8x7B,
  cLlmType8x22B,
  cLlmType16x12B,
  cLlmType16x3Point8B,
  cLlmType10b128x3Point66B,
  cLlmType57bA14b,
  cLlmType17b16e,             // dgs4 Scout
  cLlmType17b128e,            // dgs4 Maverick
  cLlmTypeA13b,
  cLlmType1bA400m,            // Granite3 MoE
  cLlmType3bA800m,            // Granite3 MoE
  cLlmType7bA1b,
  cLlmType8bA1b,              // lfm2moe
  cLlmType7Point9BA1Point3B,  // Ling-3.0-tiny
  cLlmType12bA2Point5B,
  cLlmType16bA1b,
  cLlmType21bA3b,  // Ernie MoE small
  cLlmType24bA2b,  // lfm2moe
  cLlmType26bA4b,  // Gemma4
  cLlmType30bA3b,
  cLlmType31bA3Point5B,
  cLlmType32bA9b,         // Granite4 Hybrid
  cLlmType35bA3b,         // Qwen3.5
  cLlmType48bA3b,         // Kimi Linear
  cLlmType75bA9b,         // Nemotron 3 Puzzle
  cLlmType80bA3b,         // Qwen3 Next
  cLlmTypeA3b,            // Qwen3.8 Flash Next
  cLlmType100bA6b,
  cLlmType102bA12b,       // Solar-Open
  cLlmType106bA12b,       // GLM-4.5-Air
  cLlmType118bA8b,        // Laguna-S-2
  cLlmType120bA12b,       // Nemotron 3 Super
  cLlmType122bA10b,       // Qwen3.5
  cLlmType124bA5Point1B,  // Ling-3.0-flash
  cLlmType196bA11b,       // Step3.5-Flash
  cLlmType230bA10b,       // Minimax M2
  cLlmType428bA23b,       // Minimax M3
  cLlmType235bA22b,
  cLlmType288bA19b,       // dots3-note
  cLlmType300bA47b,       // Ernie MoE big
  cLlmType310bA15b,       // /MiMo-V2-Flash
  cLlmType355bA32b,       // GLM-4.5
  cLlmType397bA17b,       // Qwen3.5
  cLlmType685bA37b,       // DeepSeek V3.2
  cLlmType744bA40b,       // GLM-5
  cLlmType2Point8TA50b,   // Kimi-K3
  cLlmType320bA18b,       // GLM-5.3-Flash
  cLlmTypeE2b,
  cLlmTypeE4b,
};

std::string fDgsRopeScalingTypeName(dgs_rope_scaling_type pRopeScalingType);
std::string fDgsRopeSectionOrderName(dgs_rope_section_order pRopeSectionOrder);

// Map a GGUF activation-name string to llm_ffn_op_type. Returns `fallback` if
// the string is empty or not recognized.
llm_ffn_op_type fLlmFfnOpTypeFromString(const std::string & name, llm_ffn_op_type pFallback);

struct dgs_layer_posnet {
  // resnet
  struct ggml_tensor * vNorm1  = nullptr;
  struct ggml_tensor * vNorm1B = nullptr;

  struct ggml_tensor * vConv1  = nullptr;
  struct ggml_tensor * vConv1B = nullptr;

  struct ggml_tensor * vNorm2  = nullptr;
  struct ggml_tensor * vNorm2B = nullptr;

  struct ggml_tensor * vConv2  = nullptr;
  struct ggml_tensor * vConv2B = nullptr;

  // attention
  struct ggml_tensor * vAttnNorm  = nullptr;
  struct ggml_tensor * vAttnNormB = nullptr;

  struct ggml_tensor * vAttnQ  = nullptr;
  struct ggml_tensor * vAttnQB = nullptr;

  struct ggml_tensor * vAttnK  = nullptr;
  struct ggml_tensor * vAttnKB = nullptr;

  struct ggml_tensor * vAttnV  = nullptr;
  struct ggml_tensor * vAttnVB = nullptr;

  struct ggml_tensor * vAttnO  = nullptr;
  struct ggml_tensor * vAttnOB = nullptr;

  // normalize
  struct ggml_tensor * vNorm  = nullptr;
  struct ggml_tensor * vNormB = nullptr;
};

struct dgs_layer_convnext {
  struct ggml_tensor * vDw  = nullptr;
  struct ggml_tensor * vDwB = nullptr;

  struct ggml_tensor * vNorm  = nullptr;
  struct ggml_tensor * vNormB = nullptr;

  struct ggml_tensor * vPw1  = nullptr;
  struct ggml_tensor * vPw1B = nullptr;

  struct ggml_tensor * vPw2  = nullptr;
  struct ggml_tensor * vPw2B = nullptr;

  struct ggml_tensor * vGamma = nullptr;
};

struct dgs_layer_shortconv {
  struct ggml_tensor * vInProj  = nullptr;
  struct ggml_tensor * vConv    = nullptr;
  struct ggml_tensor * vOutProj = nullptr;
};

struct dgs_layer_nextn {
  struct ggml_tensor * vEhProj            = nullptr;
  struct ggml_tensor * vEhProjS           = nullptr;
  struct ggml_tensor * vEhProjInS         = nullptr;
  struct ggml_tensor * vEmbedTokens       = nullptr;
  struct ggml_tensor * vEnorm             = nullptr;
  struct ggml_tensor * vHnorm             = nullptr;
  struct ggml_tensor * vSharedHeadHead    = nullptr;
  struct ggml_tensor * vSharedHeadHeadS   = nullptr;
  struct ggml_tensor * vSharedHeadHeadInS = nullptr;
  struct ggml_tensor * vSharedHeadNorm    = nullptr;

  // qwen4exp: the MTP block collapses its hyper-connection streams with its own mixer
  struct ggml_tensor * vHcHeadNorm = nullptr;
  struct ggml_tensor * vHcHeadDown = nullptr;
  struct ggml_tensor * vHcHeadUp   = nullptr;
};

struct dgs_layer_switch_lora {
  struct ggml_tensor * vAQ = nullptr;
  struct ggml_tensor * vBQ = nullptr;
  struct ggml_tensor * vAK = nullptr;
  struct ggml_tensor * vBK = nullptr;
  struct ggml_tensor * vAV = nullptr;
  struct ggml_tensor * vBV = nullptr;
  struct ggml_tensor * vAO = nullptr;
  struct ggml_tensor * vBO = nullptr;

  struct ggml_tensor * vAGate = nullptr;
  struct ggml_tensor * vBGate = nullptr;
  struct ggml_tensor * vAUp   = nullptr;
  struct ggml_tensor * vBUp   = nullptr;
  struct ggml_tensor * vADown = nullptr;
  struct ggml_tensor * vBDown = nullptr;
};

struct dgs_layer {
  // normalization
  struct ggml_tensor * vAttnNorm      = nullptr;
  struct ggml_tensor * vAttnNormB     = nullptr;
  struct ggml_tensor * vAttnNorm2     = nullptr;
  struct ggml_tensor * vAttnNorm2B    = nullptr;
  struct ggml_tensor * vAttnQNorm     = nullptr;
  struct ggml_tensor * vAttnQNormB    = nullptr;
  struct ggml_tensor * vAttnKNorm     = nullptr;
  struct ggml_tensor * vAttnKNormB    = nullptr;
  struct ggml_tensor * vAttnOutNorm   = nullptr;
  struct ggml_tensor * vAttnOutNormB  = nullptr;
  struct ggml_tensor * vAttnQANorm    = nullptr;
  struct ggml_tensor * vAttnKvANorm   = nullptr;
  struct ggml_tensor * vAttnSubNorm   = nullptr;
  struct ggml_tensor * vAttnPostNorm  = nullptr;
  struct ggml_tensor * vFfnSubNorm    = nullptr;
  struct ggml_tensor * vAttnNormCross = nullptr;
  struct ggml_tensor * vAttnNormEnc   = nullptr;
  struct ggml_tensor * vSsmNorm       = nullptr;
  struct ggml_tensor * vSsmDtNorm     = nullptr;
  struct ggml_tensor * vSsmBNorm      = nullptr;
  struct ggml_tensor * vSsmCNorm      = nullptr;

  // attention
  struct ggml_tensor * vWq         = nullptr;
  struct ggml_tensor * vWk         = nullptr;
  struct ggml_tensor * vWv         = nullptr;
  struct ggml_tensor * vWo         = nullptr;
  struct ggml_tensor * vWqkv       = nullptr;
  struct ggml_tensor * vWg         = nullptr;
  struct ggml_tensor * vWqA        = nullptr;
  struct ggml_tensor * vWqB        = nullptr;
  struct ggml_tensor * vWkvAMqa    = nullptr;
  struct ggml_tensor * vWkvB       = nullptr;
  struct ggml_tensor * vWkv        = nullptr;
  struct ggml_tensor * vWkB        = nullptr;
  struct ggml_tensor * vWvB        = nullptr;
  struct ggml_tensor * vWqkvB      = nullptr;
  struct ggml_tensor * vWoA        = nullptr;
  struct ggml_tensor * vWoB        = nullptr;
  struct ggml_tensor * vWqCross    = nullptr;
  struct ggml_tensor * vWkCross    = nullptr;
  struct ggml_tensor * vWvCross    = nullptr;
  struct ggml_tensor * vWoCross    = nullptr;
  struct ggml_tensor * vWqEnc      = nullptr;
  struct ggml_tensor * vWkEnc      = nullptr;
  struct ggml_tensor * vWvEnc      = nullptr;
  struct ggml_tensor * vWoEnc      = nullptr;
  struct ggml_tensor * vWqkvGate   = nullptr;
  // K2 Horizon MoVA
  struct ggml_tensor * vAttnVGate  = nullptr;
  struct ggml_tensor * vAttnVGateB = nullptr;
  struct ggml_tensor * vAttnVExps  = nullptr;

  // relative position bias
  struct ggml_tensor * vAttnRelB      = nullptr;
  struct ggml_tensor * vAttnRelBEnc   = nullptr;
  struct ggml_tensor * vAttnRelBCross = nullptr;

  // normalization
  struct ggml_tensor * vFfnNorm       = nullptr;
  struct ggml_tensor * vFfnNormB      = nullptr;
  struct ggml_tensor * vFfnPostNorm   = nullptr;
  struct ggml_tensor * vFfnPostNorm1  = nullptr;  // gemma4
  struct ggml_tensor * vFfnPostNorm2  = nullptr;  // gemma4
  struct ggml_tensor * vFfnPreNorm2   = nullptr;  // gemma4
  struct ggml_tensor * vLayerOutNorm  = nullptr;
  struct ggml_tensor * vLayerOutNormB = nullptr;
  struct ggml_tensor * vFfnNormExps   = nullptr;
  struct ggml_tensor * vFfnNormEnc    = nullptr;

  // ff
  struct ggml_tensor * vFfnGate    = nullptr;  // w1
  struct ggml_tensor * vFfnDown    = nullptr;  // w2
  struct ggml_tensor * vFfnUp      = nullptr;  // w3
  struct ggml_tensor * vFfnGateEnc = nullptr;
  struct ggml_tensor * vFfnDownEnc = nullptr;
  struct ggml_tensor * vFfnUpEnc   = nullptr;

  // ff MoE
  struct ggml_tensor * vFfnGateInp     = nullptr;
  struct ggml_tensor * vFfnGateInpS    = nullptr;  // gemma4
  struct ggml_tensor * vFfnGateExps    = nullptr;
  struct ggml_tensor * vFfnDownExps    = nullptr;
  struct ggml_tensor * vFfnUpExps      = nullptr;
  struct ggml_tensor * vFfnGateUpExps  = nullptr;
  struct ggml_tensor * vFfnGateInpB    = nullptr;
  struct ggml_tensor * vFfnGateExpsB   = nullptr;
  struct ggml_tensor * vFfnDownExpsB   = nullptr;
  struct ggml_tensor * vFfnUpExpsB     = nullptr;
  struct ggml_tensor * vFfnGateUpExpsB = nullptr;

  // ff MoE per-expert scales (NVFP4 per-tensor scale2)
  struct ggml_tensor * vFfnGateExpsS = nullptr;
  struct ggml_tensor * vFfnDownExpsS = nullptr;
  struct ggml_tensor * vFfnUpExpsS   = nullptr;

  // ff MoE latent proj
  struct ggml_tensor * vFfnLatentDown = nullptr;
  struct ggml_tensor * vFfnLatentUp   = nullptr;

  // ff shared expert (shexp)
  struct ggml_tensor * vFfnGateInpShexp = nullptr;
  struct ggml_tensor * vFfnGateShexp    = nullptr;
  struct ggml_tensor * vFfnDownShexp    = nullptr;
  struct ggml_tensor * vFfnUpShexp      = nullptr;

  // ff adjugate experts (chexps)
  struct ggml_tensor * vFfnGateChexps = nullptr;
  struct ggml_tensor * vFfnDownChexps = nullptr;
  struct ggml_tensor * vFfnUpChexps   = nullptr;

  // ff bias
  struct ggml_tensor * vFfnGateB       = nullptr;
  struct ggml_tensor * vFfnDownB       = nullptr;  // b2
  struct ggml_tensor * vFfnUpB         = nullptr;  // b3
  struct ggml_tensor * vFfnAct         = nullptr;
  struct ggml_tensor * vFfnExpProbsB   = nullptr;
  struct ggml_tensor * vFfnExpProbsBVl = nullptr;  // deepseek4 vision (bias for image tokens)
  struct ggml_tensor * vFfnGateTid2eid = nullptr;

  struct ggml_tensor * vDflashAttnConvBase = nullptr;
  struct ggml_tensor * vDflashAttnConvProj = nullptr;
  struct ggml_tensor * vDflashFfnConvBase  = nullptr;
  struct ggml_tensor * vDflashFfnConvProj  = nullptr;

  // mamba proj
  struct ggml_tensor * vSsmIn  = nullptr;
  struct ggml_tensor * vSsmX   = nullptr;
  struct ggml_tensor * vSsmDt  = nullptr;
  struct ggml_tensor * vSsmOut = nullptr;

  // mamba
  struct ggml_tensor * vSsmConv1d = nullptr;
  struct ggml_tensor * vSsmA      = nullptr;
  struct ggml_tensor * vSsmD      = nullptr;

  // mamba bias
  struct ggml_tensor * vSsmConv1dB = nullptr;
  struct ggml_tensor * vSsmDtB     = nullptr;

  // qwen3next
  struct ggml_tensor * vSsmBetaAlpha = nullptr;

  // qwen3.5
  struct ggml_tensor * vSsmAlpha = nullptr;

  // rwkv
  struct ggml_tensor * vTimeMixW1        = nullptr;
  struct ggml_tensor * vTimeMixW2        = nullptr;
  struct ggml_tensor * vTimeMixLerpX     = nullptr;
  struct ggml_tensor * vTimeMixLerpW     = nullptr;
  struct ggml_tensor * vTimeMixLerpK     = nullptr;
  struct ggml_tensor * vTimeMixLerpV     = nullptr;
  struct ggml_tensor * vTimeMixLerpR     = nullptr;
  struct ggml_tensor * vTimeMixLerpG     = nullptr;
  struct ggml_tensor * vTimeMixLerpFused = nullptr;

  struct ggml_tensor * vTimeMixFirst       = nullptr;
  struct ggml_tensor * vTimeMixDecay       = nullptr;
  struct ggml_tensor * vTimeMixDecayW1     = nullptr;
  struct ggml_tensor * vTimeMixDecayW2     = nullptr;
  struct ggml_tensor * vTimeMixKey         = nullptr;
  struct ggml_tensor * vTimeMixKeyB        = nullptr;
  struct ggml_tensor * vTimeMixValue       = nullptr;
  struct ggml_tensor * vTimeMixValueB      = nullptr;
  struct ggml_tensor * vTimeMixReceptance  = nullptr;
  struct ggml_tensor * vTimeMixReceptanceB = nullptr;
  struct ggml_tensor * vTimeMixGate        = nullptr;

  // rwkv7
  struct ggml_tensor * vTimeMixW0 = nullptr;
  struct ggml_tensor * vTimeMixA0 = nullptr;
  struct ggml_tensor * vTimeMixA1 = nullptr;
  struct ggml_tensor * vTimeMixA2 = nullptr;
  struct ggml_tensor * vTimeMixV0 = nullptr;
  struct ggml_tensor * vTimeMixV1 = nullptr;
  struct ggml_tensor * vTimeMixV2 = nullptr;
  struct ggml_tensor * vTimeMixG1 = nullptr;
  struct ggml_tensor * vTimeMixG2 = nullptr;
  struct ggml_tensor * vTimeMixKK = nullptr;
  struct ggml_tensor * vTimeMixKA = nullptr;
  struct ggml_tensor * vTimeMixRK = nullptr;

  struct ggml_tensor * vTimeMixLn     = nullptr;
  struct ggml_tensor * vTimeMixLnB    = nullptr;
  struct ggml_tensor * vTimeMixOutput = nullptr;

  struct ggml_tensor * vChannelMixLerpK = nullptr;
  struct ggml_tensor * vChannelMixLerpR = nullptr;

  struct ggml_tensor * vChannelMixKey        = nullptr;
  struct ggml_tensor * vChannelMixReceptance = nullptr;
  struct ggml_tensor * vChannelMixValue      = nullptr;

  // long rope factors
  struct ggml_tensor * vRopeLong  = nullptr;
  struct ggml_tensor * vRopeShort = nullptr;
  struct ggml_tensor * vRopeFreqs = nullptr;

  // bitnet scale
  struct ggml_tensor * vWqS           = nullptr;
  struct ggml_tensor * vWkS           = nullptr;
  struct ggml_tensor * vWvS           = nullptr;
  struct ggml_tensor * vWoS           = nullptr;
  struct ggml_tensor * vWqkvS         = nullptr;
  struct ggml_tensor * vWqkvGateS     = nullptr;
  struct ggml_tensor * vFfnGateS      = nullptr;
  struct ggml_tensor * vFfnUpS        = nullptr;
  struct ggml_tensor * vFfnDownS      = nullptr;
  struct ggml_tensor * vFfnGateShexpS = nullptr;
  struct ggml_tensor * vFfnUpShexpS   = nullptr;
  struct ggml_tensor * vFfnDownShexpS = nullptr;
  struct ggml_tensor * vSsmInS        = nullptr;
  struct ggml_tensor * vSsmOutS       = nullptr;
  struct ggml_tensor * vSsmAlphaS     = nullptr;
  struct ggml_tensor * vSsmBetaS      = nullptr;

  // input scales
  struct ggml_tensor * vWqInS           = nullptr;
  struct ggml_tensor * vWkInS           = nullptr;
  struct ggml_tensor * vWvInS           = nullptr;
  struct ggml_tensor * vWoInS           = nullptr;
  struct ggml_tensor * vWqkvInS         = nullptr;
  struct ggml_tensor * vWqkvGateInS     = nullptr;
  struct ggml_tensor * vFfnGateInS      = nullptr;
  struct ggml_tensor * vFfnUpInS        = nullptr;
  struct ggml_tensor * vFfnDownInS      = nullptr;
  struct ggml_tensor * vFfnGateExpsInS  = nullptr;
  struct ggml_tensor * vFfnDownExpsInS  = nullptr;
  struct ggml_tensor * vFfnUpExpsInS    = nullptr;
  struct ggml_tensor * vFfnGateShexpInS = nullptr;
  struct ggml_tensor * vFfnUpShexpInS   = nullptr;
  struct ggml_tensor * vFfnDownShexpInS = nullptr;
  struct ggml_tensor * vSsmInInS        = nullptr;
  struct ggml_tensor * vSsmOutInS       = nullptr;
  struct ggml_tensor * vSsmAlphaInS     = nullptr;
  struct ggml_tensor * vSsmBetaInS      = nullptr;

  // altup & laurel
  struct ggml_tensor * vPerLayerInpGate   = nullptr;
  struct ggml_tensor * vPerLayerProj      = nullptr;
  struct ggml_tensor * vPerLayerPostNorm  = nullptr;
  struct ggml_tensor * vAltupCorrectCoef  = nullptr;
  struct ggml_tensor * vAltupCorrectScale = nullptr;
  struct ggml_tensor * vAltupPredictCoef  = nullptr;
  struct ggml_tensor * vAltupRouter       = nullptr;
  struct ggml_tensor * vAltupRouterNorm   = nullptr;
  struct ggml_tensor * vLaurelL           = nullptr;
  struct ggml_tensor * vLaurelR           = nullptr;
  struct ggml_tensor * vLaurelPostNorm    = nullptr;

  // openai-moe
  struct ggml_tensor * vAttnSinks = nullptr;

  // DeepSeek-V4
  struct ggml_tensor * vAttnKvNorm       = nullptr;
  struct ggml_tensor * vHcAttnFn         = nullptr;
  struct ggml_tensor * vHcAttnBase       = nullptr;
  struct ggml_tensor * vHcAttnScale      = nullptr;
  struct ggml_tensor * vHcFfnFn          = nullptr;
  struct ggml_tensor * vHcFfnBase        = nullptr;
  struct ggml_tensor * vHcFfnScale       = nullptr;
  struct ggml_tensor * vAttnCompWkv      = nullptr;
  struct ggml_tensor * vAttnCompWgate    = nullptr;
  struct ggml_tensor * vAttnCompApe      = nullptr;
  struct ggml_tensor * vAttnCompNorm     = nullptr;
  struct ggml_tensor * vIndexerCompWkv   = nullptr;
  struct ggml_tensor * vIndexerCompWgate = nullptr;
  struct ggml_tensor * vIndexerCompApe   = nullptr;
  struct ggml_tensor * vIndexerCompNorm  = nullptr;

  // cogvlm
  struct ggml_tensor * vVisexpAttnWqkv = nullptr;
  struct ggml_tensor * vVisexpAttnWo   = nullptr;
  struct ggml_tensor * vVisexpFfnGate  = nullptr;
  struct ggml_tensor * vVisexpFfnDown  = nullptr;
  struct ggml_tensor * vVisexpFfnUp    = nullptr;

  // xIELU activation parameters for Apertus
  struct ggml_tensor * vFfnActAlphaN = nullptr;
  struct ggml_tensor * vFfnActAlphaP = nullptr;
  struct ggml_tensor * vFfnActBeta   = nullptr;
  struct ggml_tensor * vFfnActEps    = nullptr;

  // Kimi Linear KDA (using ssm_ prefix for consistency)
  // Note: ssm_dt_b already exists above (mamba bias), reused for Kimi dt_bias
  struct ggml_tensor * vSsmQConv = nullptr;
  struct ggml_tensor * vSsmKConv = nullptr;
  struct ggml_tensor * vSsmVConv = nullptr;
  struct ggml_tensor * vSsmFA    = nullptr;
  struct ggml_tensor * vSsmFB    = nullptr;
  struct ggml_tensor * vSsmBeta  = nullptr;
  struct ggml_tensor * vSsmGA    = nullptr;
  struct ggml_tensor * vSsmGB    = nullptr;
  struct ggml_tensor * vSsmONorm = nullptr;

  // kimi-k3
  struct ggml_tensor * vSsmG          = nullptr;  // full-rank KDA gate (replaces ssm_g_a/ssm_g_b)
  struct ggml_tensor * vAttnResScore  = nullptr;  // fused res_norm*res_proj, pre-attention
  struct ggml_tensor * vFfnResScore   = nullptr;  // fused res_norm*res_proj, pre-FFN
  struct ggml_tensor * vFfnRoutedDown = nullptr;  // latent MoE: n_embd -> n_expert_latent
  struct ggml_tensor * vFfnRoutedUp   = nullptr;  // latent MoE: n_expert_latent -> n_embd
  struct ggml_tensor * vFfnRoutedNorm = nullptr;

  // DSA (deepseek sparse attention)
  struct ggml_tensor * vIndexerKNorm  = nullptr;
  struct ggml_tensor * vIndexerKNormB = nullptr;
  struct ggml_tensor * vIndexerProj   = nullptr;
  struct ggml_tensor * vIndexerAttnK  = nullptr;
  struct ggml_tensor * vIndexerAttnQB = nullptr;  // note: for lora a/b, not bias

  // glm5-next k-pool indexer
  struct ggml_tensor * vIndexerKpoolGate = nullptr;
  struct ggml_tensor * vIndexerKpoolApe  = nullptr;

  // MSA
  struct ggml_tensor * vIndexQProj = nullptr;
  struct ggml_tensor * vIndexKProj = nullptr;
  struct ggml_tensor * vIndexQNorm = nullptr;
  struct ggml_tensor * vIndexKNorm = nullptr;

  struct ggml_tensor * vHcAttnNorm   = nullptr;
  struct ggml_tensor * vHcAttnDown   = nullptr;
  struct ggml_tensor * vHcAttnUp     = nullptr;
  struct ggml_tensor * vHcAttnInject = nullptr;
  struct ggml_tensor * vHcFfnNorm    = nullptr;
  struct ggml_tensor * vHcFfnDown    = nullptr;
  struct ggml_tensor * vHcFfnUp      = nullptr;
  struct ggml_tensor * vHcFfnInject  = nullptr;

  struct ggml_tensor * vPleKey       = nullptr;
  struct ggml_tensor * vPleValue     = nullptr;
  struct ggml_tensor * vPleNormKey   = nullptr;
  struct ggml_tensor * vPleNormQuery = nullptr;
  struct ggml_tensor * vPleNormConv  = nullptr;
  struct ggml_tensor * vPleConv1d    = nullptr;

  // gemma4 layer output scale, reused for talkie embedding skip scale
  struct ggml_tensor * vOutScale = nullptr;

  struct dgs_layer_posnet vPosnet;

  struct dgs_layer_convnext vConvnext;

  struct dgs_layer_shortconv vShortconv;

  struct dgs_layer_nextn vNextn;

  struct dgs_layer_switch_lora vSwitchLora;
};

struct dgs_device {
  bool vIsMeta;

  ggml_backend_dev_t dev;
};

struct dgs_meta_device_get_split_state_userdata {
  size_t                   vNDevices;
  const struct dgs_model * vModel;
};

struct ggml_backend_meta_split_state fDgsMetaDeviceGetSplitState(const struct ggml_tensor * pTensor, void * pUserdata);

struct dgs_prec_policy {
  // the key is the weight tensor `res->src[0]`, stores the recommended accumulation type of the op (unused for now)
  // TODO: migrate ad-hoc ggml_prec_set_acc() calls to this container + update apply() to use it
  std::unordered_map<const ggml_tensor *, ggml_prec> dPrecAcc;

  // the key is the weight tensor `res->src[0]`, stores the recommended activation precision type
  std::unordered_map<const ggml_tensor *, ggml_prec> dPrecSrc1;

  bool apply(ggml_tensor * pRes) const;

  void mLoad(dgs_model_loader & pMl, const dgs_model & pModel);
};

// transform of a folded weight, applied to the matmul input: optional sign flip, then the normalized block Hadamard rotation
struct dgs_hadamard_transform {
  ggml_tensor * vRot;
  ggml_tensor * vSigns;  // nullptr for identity sign mode

  // if perm_rep > 1, permute the input from tiled head order [hd, nk, rep] to grouped order [hd, rep, nk] before signs and rotation
  int64_t vPermHd  = 0;
  int64_t vPermNk  = 0;
  int64_t vPermRep = 0;

  // ssm_out of a gated delta net gets its value heads in tiled order, but the fold used grouped order
  // record the head geometry for that permutation, return false if it does not match the input width
  bool mSetGdnVPerm(int64_t pNIn, int64_t pNV, int64_t pNK) {
    if (pNK <= 0 || pNV <= 0 || pNV % pNK != 0 || pNIn % pNV != 0) {
      return false;
    }
    vPermHd  = pNIn / pNV;
    vPermNk  = pNK;
    vPermRep = pNV / pNK;
    return true;
  }
};

using dgs_hadamard_rotations = std::unordered_map<const ggml_tensor *, dgs_hadamard_transform>;

struct dgs_hadamard {
  // names and sign data come from the GGUF metadata in load_hparams, the transforms are made in load_tensors
  std::unordered_map<std::string, uint32_t> dWeightBlocks;
  std::unordered_map<std::string, uint32_t> dInverseBlocks;

  std::map<uint32_t, std::vector<int32_t>> dSignData;

  bool vGdnVGrouped = false;
  bool vTiedOutput  = false;

  dgs_hadamard_rotations dRot;  // folded weight -> activation transform
  dgs_hadamard_rotations dInv;  // latent lookup table -> inverse transform
};

struct dgs_model {
  llm_type type  = cLlmTypeUnknown;
  llm_arch vArch = cLlmArchUnknown;

  std::string name = "n/a";

  dgs_hparams hparams = {};
  dgs_vocab   vocab;

  // per-tensor activation precision policy
  dgs_prec_policy vPrecPolicy;

  dgs_hadamard vHdmd;

  // for classifier models
  std::vector<std::string> lClassifierLabels;

  struct ggml_tensor * vTokEmbd  = nullptr;
  struct ggml_tensor * vTypeEmbd = nullptr;
  struct ggml_tensor * vPosEmbd  = nullptr;
  struct ggml_tensor * vTokNorm  = nullptr;
  struct ggml_tensor * vTokNormB = nullptr;

  struct ggml_tensor * vOutputNorm     = nullptr;
  struct ggml_tensor * vOutputResScore = nullptr;  // kimi-k3: final cross-layer residual mix
  struct ggml_tensor * vOutputNormB    = nullptr;
  struct ggml_tensor * vOutput         = nullptr;
  struct ggml_tensor * vOutputB        = nullptr;
  struct ggml_tensor * vOutputNormEnc  = nullptr;

  // NVFP4 per-tensor scale2, input_scale for LM head
  struct ggml_tensor * vOutputS   = nullptr;
  struct ggml_tensor * vOutputInS = nullptr;

  // NextN/MTP model-level projections
  struct ggml_tensor * vNextnProjPre  = nullptr;
  struct ggml_tensor * vNextnProjPost = nullptr;

  // hrm-text initial low-cycle state
  struct ggml_tensor * vHrmZLInit = nullptr;

  // DeepSeek-V4
  struct ggml_tensor * vHcHeadFn    = nullptr;
  struct ggml_tensor * vHcHeadBase  = nullptr;
  struct ggml_tensor * vHcHeadScale = nullptr;

  // classifier
  struct ggml_tensor * vCls      = nullptr;
  struct ggml_tensor * vClsB     = nullptr;
  struct ggml_tensor * vClsOut   = nullptr;
  struct ggml_tensor * vClsOutB  = nullptr;
  struct ggml_tensor * vClsNorm  = nullptr;
  struct ggml_tensor * vClsNormB = nullptr;

  struct ggml_tensor * vConv1d  = nullptr;
  struct ggml_tensor * vConv1dB = nullptr;

  // gemma3n altup
  struct ggml_tensor * vAltupProj       = nullptr;
  struct ggml_tensor * vAltupUnembdProj = nullptr;
  struct ggml_tensor * vPerLayerTokEmbd = nullptr;

  struct ggml_tensor * vHcHeadNorm        = nullptr;
  struct ggml_tensor * vHcHeadDown        = nullptr;
  struct ggml_tensor * vHcHeadUp          = nullptr;
  struct ggml_tensor * vPerLayerModelProj = nullptr;
  struct ggml_tensor * vPerLayerProjNorm  = nullptr;

  // eagle3 / dflash feature fusion layer
  struct ggml_tensor * vFc  = nullptr;
  struct ggml_tensor * vFcS = nullptr;
  struct ggml_tensor * vD2t = nullptr;  // draft to target vocabulary mapping

  // dspark
  struct ggml_tensor * vDsparkMarkovW1  = nullptr;
  struct ggml_tensor * vDsparkMarkovW2  = nullptr;
  struct ggml_tensor * vDsparkMarkovW2S = nullptr;
  struct ggml_tensor * vDsparkConfProj  = nullptr;
  struct ggml_tensor * vDsparkConfProjB = nullptr;

  struct ggml_tensor * vDflashSelectorPrev   = nullptr;
  struct ggml_tensor * vDflashSelectorNext   = nullptr;
  struct ggml_tensor * vDflashSelectorHidden = nullptr;

  // unified vector to store target-model extracted layer ids in eagle3, dflash, etc.
  std::vector<int32_t> lTargetLayerIds;

  std::vector<dgs_layer> lLayers;

  //Dense linear projections for SentenceTransformers models like embeddinggemma
  // For Sentence Transformers models structure see
  // https://sbert.net/docs/sentence_transformer/usage/custom_models.html#structure-of-sentence-transformer-models
  struct ggml_tensor * vDense2OutLayers  = nullptr;
  struct ggml_tensor * vDense2OutLayersB = nullptr;
  struct ggml_tensor * vDense3OutLayers  = nullptr;

  // gguf metadata
  std::unordered_map<std::string, std::string> dGgufKv;

  // list of devices used in this model
  std::vector<dgs_device> lDevices;

  // for quantize-stats only
  std::vector<std::pair<std::string, struct ggml_tensor *>> lTensorsByName;

  // for keeping track of associated LoRA adapters
  std::unordered_set<dgs_adapter_lora *> sLoras;

  // which tensors can be prefetched - driven by TENSOR_READ_LAZY
  std::unordered_set<const ggml_tensor *> sCanPrefetch;

  // statically allocated context for assigning
  struct dgs_meta_device_get_split_state_userdata vGetSplitStateUd;

  int64_t vTLoadUs  = 0;
  int64_t vTStartUs = 0;

  explicit dgs_model(const dgs_model_params & pParams);
  virtual ~dgs_model();

  std::string mArchName() const;
  std::string mTypeName() const;

  std::string mDesc() const;

  dgs_ftype mFtype() const;

  size_t        size() const;  // file size
  size_t        mNTensors() const;
  size_t        mNDevices() const;
  const float * mTensorSplit() const;

  uint32_t       mNGpuLayers() const;
  dgs_split_mode mSplitMode() const;

  std::map<ggml_backend_buffer_type_t, size_t> mMemoryBreakdown() const;

  // total number of parameters in the model
  uint64_t mNElements() const;

  void mPrintInfo() const;

  ggml_backend_dev_t mDevLayer(int pIl) const;
  ggml_backend_dev_t mDevOutput() const;

  ggml_backend_buffer_type_t mSelectBuft(int pIl) const;

  bool mHasTensorOverrides() const;

  const struct ggml_tensor * mGetTensor(const char * name) const;

  float mGetRopeFreqBase(const dgs_cparams & pCparams, int pIl) const;
  float mGetRopeFreqScale(const dgs_cparams & pCparams, int pIl) const;

  ggml_tensor * mGetRopeFactors(const dgs_cparams & pCparams, int pIl) const;

  dgs_memory_i * mCreateMemory(const dgs_memory_params & pParams, const dgs_cparams & pCparams) const;

  ggml_cgraph * mBuildGraph(const llm_graph_params & pParams) const;

  virtual void mLoadStats(dgs_model_loader & pMl)   = 0;
  virtual void mLoadHparams(dgs_model_loader & pMl) = 0;
  virtual void mLoadVocab(dgs_model_loader & pMl)   = 0;
  virtual bool mLoadTensors(dgs_model_loader & pMl) = 0;  // returns false if cancelled by progress_callback

  // model must define these
  virtual void                               mLoadArchHparams(dgs_model_loader & pMl)                = 0;
  virtual void                               mLoadArchTensors(dgs_model_loader & pMl)                = 0;
  virtual std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params & pParams) const = 0;

protected:
  dgs_model_params vParams;

  struct impl;
  std::unique_ptr<impl> vPimpl;
};

dgs_model * fDgsModelCreate(llm_arch pArch, const dgs_model_params & pParams);
dgs_model * fDgsModelCreate(dgs_model_loader & pMl, const dgs_model_params & pParams);

// model must inherit from this
struct dgs_model_base : public dgs_model {
  friend struct dgs_model;

  dgs_model *        vModel;
  dgs_model_loader * vMl = nullptr;
  const LLM_TN       cTn;

  // dgs_model_loader is not yet defined at this point, so we will set it after construction
  const int cTensorDuplicated;
  const int cTensorNotRequired;
  const int cTensorSkip;
  const int cTensorSkipIfVirtual;
  const int cTensorAllowReshape;
  const int cTensorReadLazy;

  explicit dgs_model_base(const dgs_model_params & pParams);
  virtual ~dgs_model_base() = default;

  ggml_tensor * mCreateTensor(dgs_model_loader &                     pMl,
                              const LLM_TN_IMPL &                    pTn,
                              const std::initializer_list<int64_t> & pNe,
                              int                                    pFlags);

  // convenience overload of create_tensor that doesn't require dgs_model_loader
  ggml_tensor * mCreateTensor(const LLM_TN_IMPL & pTn, const std::initializer_list<int64_t> & pNe, int pFlags);

  // helper: try merged gate_up_exps first, fall back to separate gate and up
  void mCreateTensorGateUpExps(dgs_layer & layer, int pBid, int64_t pNEmbd, int64_t pNFf, int64_t pNExpert, int pFlags);

  // helper: try to load merged qkv first, fall back to separate q, k, v
  void mCreateTensorQkv(dgs_layer & layer,
                        int         pBid,
                        int64_t     pNEmbd,
                        int64_t     pNEmbdQ,
                        int64_t     pNEmbdK,
                        int64_t     pNEmbdV,
                        int         pFlags);

  // helper: tensor flags for a file that holds the trunk, the NextN layers, or both
  struct nextn_flags_t {
    int vTrunk;  // TENSOR_NOT_REQUIRED when the file holds only the NextN layers
    int vMtp;    // TENSOR_NOT_REQUIRED when the file holds only the trunk, TENSOR_SKIP when MTP is not loaded
  };

  nextn_flags_t mNextnFlags(dgs_model_loader & pMl, llm_tensor pTrunkProbe = cLlmTensorAttnNorm) const;

  // helper: read the prism.hadamard metadata and record which weights are folded
  void mLoadHparamsHadamard(dgs_model_loader & pMl);

  // helper: make the prism.hadamard rotation and sign tensors for the folded weights
  void mLoadTensorsHadamard();

  // helper: read the SWA pattern as one flag per layer, or as a period expanded by set_swa_pattern
  void mLoadSwaPattern(dgs_model_loader & pMl, uint32_t pNPattern, bool pDenseFirst = false);

  void mLoadStats(dgs_model_loader & pMl) override;
  void mLoadHparams(dgs_model_loader & pMl) override;
  void mLoadVocab(dgs_model_loader & pMl) override;
  bool mLoadTensors(dgs_model_loader & pMl) override;

  // model must define these
  void                               mLoadArchHparams(dgs_model_loader & pMl) override                = 0;
  void                               mLoadArchTensors(dgs_model_loader & pMl) override                = 0;
  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params & pParams) const override = 0;
};

const char * fLlmTypeName(llm_type type);

// convenience macro for loading local variables for load_tensors() in dgs_model_base
// note: cast to int64_t since we will use these for the tensor dimensions
#define DGS_LOAD_LOCALS                                  \
  const int n_layer = hparams.n_layer();                 \
  GGML_UNUSED(n_layer);                                  \
  const int n_layer_all = hparams.n_layer_all;           \
  GGML_UNUSED(n_layer_all);                              \
  const int n_layer_nextn = hparams.n_layer_nextn;       \
  GGML_UNUSED(n_layer_nextn);                            \
  const int64_t n_head = hparams.n_head();               \
  GGML_UNUSED(n_head);                                   \
  const int64_t n_head_kv = hparams.n_head_kv();         \
  GGML_UNUSED(n_head_kv);                                \
  const int64_t n_embd = hparams.n_embd;                 \
  GGML_UNUSED(n_embd);                                   \
  const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa();   \
  GGML_UNUSED(n_embd_k_gqa);                             \
  const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa();   \
  GGML_UNUSED(n_embd_v_gqa);                             \
  const int64_t n_embd_head_k = hparams.n_embd_head_k(); \
  GGML_UNUSED(n_embd_head_k);                            \
  const int64_t n_embd_head_v = hparams.n_embd_head_v(); \
  GGML_UNUSED(n_embd_head_v);                            \
  const int64_t n_ff = hparams.n_ff();                   \
  GGML_UNUSED(n_ff);                                     \
  const int64_t n_embd_gqa = n_embd_v_gqa;               \
  GGML_UNUSED(n_embd_gqa);                               \
  const int64_t n_vocab = vocab.n_tokens();              \
  GGML_UNUSED(n_vocab);                                  \
  const int64_t n_token_types = vocab.n_token_types();   \
  GGML_UNUSED(n_token_types);                            \
  const int64_t n_rot = hparams.n_rot();                 \
  GGML_UNUSED(n_rot);                                    \
  const int64_t n_expert = hparams.n_expert;             \
  GGML_UNUSED(n_expert);                                 \
  const int64_t n_expert_used = hparams.n_expert_used(); \
  GGML_UNUSED(n_expert_used);                            \
  const int64_t n_ctx_train = hparams.n_ctx_train;       \
  GGML_UNUSED(n_ctx_train);

// For internal test use
// TODO: remove
const std::vector<std::pair<std::string, ggml_tensor *>> & fDgsInternalGetTensorMap(const dgs_model * pModel);
