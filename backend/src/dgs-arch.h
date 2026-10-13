#pragma once

#include "ggml.h"  // ggml_op

#include <set>
#include <string>
#include <vector>

//
// gguf constants (sync with gguf.py)
//

enum llm_arch {
  cLlmArchClip,
  cLlmArchDgs,
  cLlmArchDgs4,
  cLlmArchDeci,
  cLlmArchFalcon,
  cLlmArchBaichuan,
  cLlmArchGrok,
  cLlmArchGpt2,
  cLlmArchGptj,
  cLlmArchGptneox,
  cLlmArchMpt,
  cLlmArchStarcoder,
  cLlmArchRefact,
  cLlmArchBert,
  cLlmArchModernBert,
  cLlmArchNomicBert,
  cLlmArchNomicBertMoe,
  cLlmArchNeoBert,
  cLlmArchJinaBertV2,
  cLlmArchJinaBertV3,
  cLlmArchEurobert,
  cLlmArchBloom,
  cLlmArchStablelm,
  cLlmArchQwen,
  cLlmArchQwen2,
  cLlmArchQwen2moe,
  cLlmArchQwen2vl,
  cLlmArchQwen3,
  cLlmArchQwen3moe,
  cLlmArchQwen3next,
  cLlmArchQwen3vl,
  cLlmArchQwen3vlmoe,
  cLlmArchQwen35,
  cLlmArchQwen35moe,
  cLlmArchClef,
  cLlmArchVllmDecision2,
  cLlmArchQwen4exp,
  cLlmArchPhi2,
  cLlmArchPhi3,
  cLlmArchPhimoe,
  cLlmArchPlamo,
  cLlmArchPlamo2,
  cLlmArchPlamo3,
  cLlmArchCodeshell,
  cLlmArchOrion,
  cLlmArchInternlm2,
  cLlmArchMinicpm,
  cLlmArchMinicpm3,
  cLlmArchGemma,
  cLlmArchGemma2,
  cLlmArchGemma3,
  cLlmArchGemma3n,
  cLlmArchGemma4,
  cLlmArchGemma4Assistant,
  cLlmArchGemmaEmbedding,
  cLlmArchGemmaEmbedding2,
  cLlmArchStarcoder2,
  cLlmArchMamba,
  cLlmArchMamba2,
  cLlmArchMaple,
  cLlmArchJamba,
  cLlmArchFalconH1,
  cLlmArchXverse,
  cLlmArchCommandR,
  cLlmArchCohere2,
  cLlmArchCohere2moe,
  cLlmArchDbrx,
  cLlmArchOlmo,
  cLlmArchOlmo2,
  cLlmArchOlmoe,
  cLlmArchMuseGlimmer,
  cLlmArchOpenelm,
  cLlmArchArctic,
  cLlmArchDeepseek,
  cLlmArchDeepseek2,
  cLlmArchDeepseek2ocr,
  cLlmArchDeepseek32,
  cLlmArchDeepseek4,
  cLlmArchChatglm,
  cLlmArchGlm4,
  cLlmArchGlm4Moe,
  cLlmArchGlmDsa,
  cLlmArchBitnet,
  cLlmArchT5,
  cLlmArchT5encoder,
  cLlmArchJais,
  cLlmArchJais2,
  cLlmArchNemotron,
  cLlmArchNemotronH,
  cLlmArchNemotronHMoe,
  cLlmArchExaone,
  cLlmArchExaone4,
  cLlmArchExaoneMoe,
  cLlmArchRwkv6,
  cLlmArchRwkv6qwen2,
  cLlmArchRwkv7,
  cLlmArchArwkv7,
  cLlmArchGranite,
  cLlmArchGraniteMoe,
  cLlmArchGraniteHybrid,
  cLlmArchGraniteSwitch,
  cLlmArchGraniteSwa,
  cLlmArchChameleon,
  cLlmArchWavtokenizerDec,
  cLlmArchPlm,
  cLlmArchBailingmoe,
  cLlmArchBailingmoe2,
  cLlmArchBailingmoe3,
  cLlmArchDots1,
  cLlmArchDots3note,
  cLlmArchArcee,
  cLlmArchAfmoe,
  cLlmArchLaguna,
  cLlmArchERNIE4Point5,
  cLlmArchERNIE4Point5Moe,
  cLlmArchHunyuanMoe,
  cLlmArchHunyuanDense,
  cLlmArchHunyuanVl,
  cLlmArchHyV3,
  cLlmArchHyV4,
  cLlmArchSmollm3,
  cLlmArchOpenaiMoe,
  cLlmArchLfm2,
  cLlmArchLfm2moe,
  cLlmArchDream,
  cLlmArchSmallthinker,
  cLlmArchLlada,
  cLlmArchLladaMoe,
  cLlmArchSeedOss,
  cLlmArchGrovemoe,
  cLlmArchApertus,
  cLlmArchMinimaxM2,
  cLlmArchCogvlm,
  cLlmArchRnd1,
  cLlmArchPanguEmbed,
  cLlmArchMistral3,
  cLlmArchMistral4,
  cLlmArchPaddleocr,
  cLlmArchMimo2,
  cLlmArchStep35,
  cLlmArchSPARK2Point5,
  cLlmArchDgsEmbed,
  cLlmArchMaincoder,
  cLlmArchKimiLinear,
  cLlmArchKimiK3,
  cLlmArchGlm5Next,
  cLlmArchTalkie,
  cLlmArchMellum,
  cLlmArchEagle3,
  cLlmArchMinimaxM3,
  cLlmArchDflash,
  cLlmArchNanbeige,
  cLlmArchQwen3tts,
  cLlmArchPockettts,
  cLlmArchMinimax01,
  cLlmArchHrmText,
  cLlmArchK2Horizon,
  cLlmArchUnknown,
};

enum llm_kv {
  cLlmKvGeneralType,
  cLlmKvGeneralArchitecture,
  cLlmKvGeneralQuantizationVersion,
  cLlmKvGeneralAlignment,
  cLlmKvGeneralFileType,
  cLlmKvGeneralSamplingSequence,
  cLlmKvGeneralSamplingTopK,
  cLlmKvGeneralSamplingTopP,
  cLlmKvGeneralSamplingMinP,
  cLlmKvGeneralSamplingXtcProbability,
  cLlmKvGeneralSamplingXtcThreshold,
  cLlmKvGeneralSamplingTemp,
  cLlmKvGeneralSamplingPenaltyLastN,
  cLlmKvGeneralSamplingPenaltyRepeat,
  cLlmKvGeneralSamplingMirostat,
  cLlmKvGeneralSamplingMirostatTau,
  cLlmKvGeneralSamplingMirostatEta,
  cLlmKvGeneralName,
  cLlmKvGeneralAuthor,
  cLlmKvGeneralVersion,
  cLlmKvGeneralUrl,
  cLlmKvGeneralDescription,
  cLlmKvGeneralLicense,
  cLlmKvGeneralSourceUrl,
  cLlmKvGeneralSourceHfRepo,
  cLlmKvGeneralTensorExtraName,
  cLlmKvGeneralTensorExtraPrecA4,

  cLlmKvVocabSize,
  cLlmKvContextLength,
  cLlmKvEmbeddingLength,
  cLlmKvEmbeddingLengthOut,
  cLlmKvEmbeddingLengthPerLayer,
  cLlmKvFeaturesLength,
  cLlmKvBlockCount,
  cLlmKvLeadingDenseBlockCount,
  cLlmKvAttnResBlockSize,
  cLlmKvActivationSituBeta,
  cLlmKvActivationSituLinearBeta,
  cLlmKvFeedForwardLength,
  cLlmKvExpertFeedForwardLength,
  cLlmKvExpertSharedFeedForwardLength,
  cLlmKvExpertChunkFeedForwardLength,
  cLlmKvSwigluClampExp,
  cLlmKvSwigluClampShexp,
  cLlmKvUseParallelResidual,
  cLlmKvTensorDataLayout,
  cLlmKvExpertCount,
  cLlmKvExpertUsedCount,
  cLlmKvExpertSharedCount,
  cLlmKvExpertGroupCount,
  cLlmKvExpertGroupUsedCount,
  cLlmKvExpertWeightsScale,
  cLlmKvExpertWeightsNorm,
  cLlmKvExpertLatentLength,
  cLlmKvExpertGatingFunc,
  cLlmKvExpertGroupScale,
  cLlmKvExpertsPerGroup,
  cLlmKvMoeEveryNLayers,
  cLlmKvMoeLatentSize,
  cLlmKvNextnPredictLayers,
  cLlmKvNumDeepstackLayers,
  cLlmKvDeepstackMapping,
  cLlmKvHiddenAct,
  cLlmKvPoolingType,
  cLlmKvLogitScale,
  cLlmKvDecoderStartTokenId,
  cLlmKvDecoderBlockCount,
  cLlmKvAttnLogitSoftcapping,
  cLlmKvRouterLogitSoftcapping,
  cLlmKvFinalLogitSoftcapping,
  cLlmKvSwinNorm,
  cLlmKvRescaleEveryNLayers,
  cLlmKvTimeMixExtraDim,
  cLlmKvTimeDecayExtraDim,
  cLlmKvResidualScale,
  cLlmKvEmbeddingScale,
  cLlmKvAdapterCount,
  cLlmKvAdapterTokenIdsActivate,
  cLlmKvAdapterTokenIdsSubstitute,
  cLlmKvAdapterLoraRank,
  cLlmKvAdapterRouterGain,
  cLlmKvTokenShiftCount,
  cLlmKvInterleaveMoeLayerStep,
  cLlmKvFullAttentionInterval,
  cLlmKvNumLoops,
  cLlmKvSkipLoopFinalNorm,
  cLlmKvHrmLayersPerStack,
  cLlmKvHrmHCycles,
  cLlmKvHrmLCycles,
  cLlmKvHrmPrefixLm,

  cLlmKvAttentionHeadCount,
  cLlmKvAttentionHeadCountKv,
  cLlmKvAttentionMaxAlibiBias,
  cLlmKvAttentionClampKqv,
  cLlmKvAttentionKeyLength,
  cLlmKvAttentionValueLength,
  cLlmKvAttentionLayernormEps,
  cLlmKvAttentionLayernormRmsEps,
  cLlmKvAttentionGroupnormEps,
  cLlmKvAttentionGroupnormGroups,
  cLlmKvAttentionCausal,
  cLlmKvAttentionQLoraRank,
  cLlmKvAttentionKvLoraRank,
  cLlmKvAttentionDecayLoraRank,
  cLlmKvAttentionIclrLoraRank,
  cLlmKvAttentionValueResidualMixLoraRank,
  cLlmKvAttentionGateLoraRank,
  cLlmKvAttentionRelativeBucketsCount,
  cLlmKvAttentionSlidingWindow,
  cLlmKvAttentionSlidingWindowPattern,
  cLlmKvAttentionScale,
  cLlmKvAttentionRopePattern,
  cLlmKvAttentionValueExpertCount,
  cLlmKvAttentionValueExpertUsedCount,

  cLlmKvAttentionOutputScale,
  cLlmKvAttentionValueScale,
  cLlmKvAttentionTemperatureLength,
  cLlmKvAttentionTemperatureScale,
  cLlmKvAttentionKeyLengthMla,
  cLlmKvAttentionValueLengthMla,
  cLlmKvAttentionKeyLengthSwa,
  cLlmKvAttentionValueLengthSwa,
  cLlmKvAttentionKeyLengthMlaSwa,
  cLlmKvAttentionValueLengthMlaSwa,
  cLlmKvAttentionKvLoraRankSwa,
  cLlmKvAttentionIndexerHeadCount,
  cLlmKvAttentionIndexerKeyLength,
  cLlmKvAttentionIndexerTopK,
  cLlmKvAttentionIndexerBlockSize,
  cLlmKvAttentionIndexerLocalBlocks,
  cLlmKvAttentionIndexerTypes,
  cLlmKvAttentionIndexerKpool,
  cLlmKvAttentionIndexerKpoolSelectTail,
  cLlmKvAttentionOutputGroupCount,
  cLlmKvAttentionOutputLoraRank,
  cLlmKvAttentionCompressRopeFreqBase,
  cLlmKvAttentionCompressRatios,
  cLlmKvAttentionSharedKvLayers,
  cLlmKvAttentionRecurrentLayers,

  cLlmKvHyperConnectionCount,
  cLlmKvHyperConnectionSinkhornIterations,
  cLlmKvHyperConnectionEpsilon,
  cLlmKvHyperConnectionMagnitude,
  cLlmKvHyperConnectionLowRank,

  cLlmKvPleLayers,
  cLlmKvPleNgramSize,
  cLlmKvPleHeadsPerNgram,
  cLlmKvPleConvKernel,
  cLlmKvPleLayerMultipliers,
  cLlmKvPleHeadOffsets,
  cLlmKvPleHeadVocabSizes,
  cLlmKvPleEosTokenId,
  cLlmKvPleImageTokenId,

  cLlmKvHashLayerCount,

  cLlmKvRopeDimensionCount,
  cLlmKvRopeDimensionCountSwa,
  cLlmKvRopeDimensionSections,
  cLlmKvRopeSectionOrder,
  cLlmKvRopeFreqBase,
  cLlmKvRopeFreqBaseSwa,
  cLlmKvRopeScalingType,
  cLlmKvRopeScalingFactor,
  cLlmKvRopeScalingAlpha,
  cLlmKvRopeScalingAttnFactor,
  cLlmKvRopeScalingOrigCtxLen,
  cLlmKvRopeScalingFinetuned,
  cLlmKvRopeScalingYarnLogMul,
  cLlmKvRopeScalingYarnExtFactor,
  cLlmKvRopeScalingYarnAttnFactor,
  cLlmKvRopeScalingYarnBetaFast,
  cLlmKvRopeScalingYarnBetaSlow,

  cLlmKvSplitNo,
  cLlmKvSplitCount,
  cLlmKvSplitTensorsCount,

  cLlmKvSsmInnerSize,
  cLlmKvSsmConvKernel,
  cLlmKvSsmStateSize,
  cLlmKvSsmTimeStepRank,
  cLlmKvSsmGroupCount,
  cLlmKvSsmDtBCRms,

  cLlmKvKdaHeadDim,
  cLlmKvKdaSafeGate,
  cLlmKvKdaGateLowerBound,

  cLlmKvWkvHeadSize,

  cLlmKvTokenizerModel,
  cLlmKvTokenizerPre,
  cLlmKvTokenizerList,
  cLlmKvTokenizerTokenType,
  cLlmKvTokenizerTokenTypeCount,
  cLlmKvTokenizerScores,
  cLlmKvTokenizerMerges,
  cLlmKvTokenizerBosId,
  cLlmKvTokenizerEosId,
  cLlmKvTokenizerEotId,
  cLlmKvTokenizerEomId,
  cLlmKvTokenizerUnkId,
  cLlmKvTokenizerSepId,
  cLlmKvTokenizerPadId,
  cLlmKvTokenizerClsId,
  cLlmKvTokenizerMaskId,
  cLlmKvTokenizerAddBos,
  cLlmKvTokenizerAddEos,
  cLlmKvTokenizerAddSep,
  cLlmKvTokenizerAddPrefix,
  cLlmKvTokenizerRemoveExtraWs,
  cLlmKvTokenizerPrecompiledCharsmap,
  cLlmKvTokenizerHfJson,
  cLlmKvTokenizerRwkv,
  cLlmKvTokenizerChatTemplate,
  cLlmKvTokenizerNormalizerLowercase,
  cLlmKvTokenizerNormalizerStripAccents,
  cLlmKvTokenizerFimPreId,
  cLlmKvTokenizerFimSufId,
  cLlmKvTokenizerFimMidId,
  cLlmKvTokenizerFimPadId,
  cLlmKvTokenizerFimRepId,
  cLlmKvTokenizerFimSepId,
  cLlmKvTokenizerSuppressTokens,

  cLlmKvAdapterType,
  cLlmKvAdapterLoraAlpha,
  cLlmKvAdapterLoraTaskName,
  cLlmKvAdapterLoraPromptPrefix,
  cLlmKvAdapterAloraInvocationTokens,

  cLlmKvPosnetEmbeddingLength,
  cLlmKvPosnetBlockCount,

  cLlmKvConvnextEmbeddingLength,
  cLlmKvConvnextBlockCount,

  cLlmKvClassifierOutputLabels,
  cLlmKvClassifierPoolingType,
  cLlmKvClassifierActivation,

  cLlmKvDecisionBlockCount,
  cLlmKvDecisionRoutingBlockCount,
  cLlmKvDecisionHeadCount,

  cLlmKvTargetLayers,
  cLlmKvTargetHiddenSize,
  cLlmKvDflashBlockSize,
  cLlmKvDflashConvKernelSize,
  cLlmKvDflashConvGroupSize,
  cLlmKvDflashSelectorRank,
  cLlmKvDflashSelectorTopK,
  cLlmKvNormBeforeResidual,
  cLlmKvNormBeforeFc,

  cLlmKvShortconvLCache,

  cLlmKvXieluAlphaN,
  cLlmKvXieluAlphaP,
  cLlmKvXieluBeta,
  cLlmKvXieluEps,

  cLlmKvPrismHadamardVersion,
  cLlmKvPrismHadamardTiedOutput,
  cLlmKvPrismHadamardBlockSize,
  cLlmKvPrismHadamardTransform,
  cLlmKvPrismHadamardAxis,
  cLlmKvPrismHadamardSignMode,
  cLlmKvPrismHadamardSignWidths,
  cLlmKvPrismHadamardSignValues,
  cLlmKvPrismHadamardWeightNames,
  cLlmKvPrismHadamardInverseWeightNames,
  cLlmKvPrismHadamardGdnVGrouped,

  // sentence-transformers dense layers in and out features
  cLlmKvDense2FeatIn,
  cLlmKvDense2FeatOut,
  cLlmKvDense3FeatIn,
  cLlmKvDense3FeatOut,
};

enum llm_tensor {
  cLlmTensorTokenEmbd,
  cLlmTensorTokenEmbdNorm,
  cLlmTensorTokenTypes,
  cLlmTensorPosEmbd,
  cLlmTensorDense2Out,
  cLlmTensorDense3Out,
  cLlmTensorOutput,
  cLlmTensorOutputNorm,
  cLlmTensorOutputNormLfm2,  // fix for wrong tensor name
  cLlmTensorRopeFreqs,
  cLlmTensorRopeFactorsLong,
  cLlmTensorRopeFactorsShort,
  cLlmTensorAttnQ,
  cLlmTensorAttnK,
  cLlmTensorAttnV,
  cLlmTensorAttnQkv,
  cLlmTensorAttnOut,
  cLlmTensorAttnNorm,
  cLlmTensorAttnNorm2,
  cLlmTensorAttnOutNorm,
  cLlmTensorAttnPostNorm,
  cLlmTensorAttnRotEmbd,
  cLlmTensorAttnSinks,
  cLlmTensorAttnGate,
  cLlmTensorFfnGateInp,
  cLlmTensorFfnGateInpShexp,
  cLlmTensorFfnNorm,
  cLlmTensorFfnPostNorm,
  cLlmTensorFfnPostNorm1,
  cLlmTensorFfnPostNorm2,
  cLlmTensorFfnPreNorm2,
  cLlmTensorFfnGate,
  cLlmTensorFfnDown,
  cLlmTensorFfnUp,
  cLlmTensorFfnAct,
  cLlmTensorFfnGateExp,
  cLlmTensorFfnUpExp,
  cLlmTensorFfnNormExps,
  cLlmTensorFfnDownExps,  // merged experts
  cLlmTensorFfnGateExps,
  cLlmTensorFfnUpExps,
  cLlmTensorFfnGateUpExps,
  cLlmTensorFfnDownShexp,
  cLlmTensorFfnGateShexp,
  cLlmTensorFfnUpShexp,
  cLlmTensorFfnDownChexps,
  cLlmTensorFfnGateChexps,
  cLlmTensorFfnUpChexps,
  cLlmTensorFfnExpProbsB,
  cLlmTensorFfnExpProbsBVl,
  cLlmTensorFfnLatentDown,
  cLlmTensorFfnLatentUp,
  cLlmTensorAttnQNorm,
  cLlmTensorAttnKNorm,
  cLlmTensorLayerOutNorm,
  cLlmTensorLayerOutScale,
  cLlmTensorPostAttnNorm,
  cLlmTensorPostMlpNorm,
  cLlmTensorPerLayerTokenEmbd,  // gemma3n
  cLlmTensorPerLayerModelProj,  // gemma3n
  cLlmTensorPerLayerInpGate,    // gemma3n
  cLlmTensorPerLayerProj,       // gemma3n
  cLlmTensorPerLayerProjNorm,   // gemma3n
  cLlmTensorPerLayerPostNorm,   // gemma3n
  cLlmTensorAltupProj,          // gemma3n
  cLlmTensorAltupUnembdProj,    // gemma3n
  cLlmTensorAltupCorrectCoef,   // gemma3n
  cLlmTensorAltupCorrectScale,  // gemma3n
  cLlmTensorAltupPredictCoef,   // gemma3n
  cLlmTensorAltupRouter,        // gemma3n
  cLlmTensorAltupRouterNorm,    // gemma3n
  cLlmTensorLaurelL,            // gemma3n
  cLlmTensorLaurelR,            // gemma3n
  cLlmTensorLaurelPostNorm,     // gemma3n
  cLlmTensorSsmIn,
  cLlmTensorSsmConv1d,
  cLlmTensorSsmX,
  cLlmTensorSsmDt,
  cLlmTensorSsmDtNorm,
  cLlmTensorSsmA,
  cLlmTensorSsmANoscan,  // qwen3next special case with MUL instead of SSM_SCAN
  cLlmTensorSsmBNorm,
  cLlmTensorSsmCNorm,
  cLlmTensorSsmD,
  cLlmTensorSsmNorm,
  cLlmTensorSsmOut,
  cLlmTensorSsmBetaAlpha,  // qwen3next
  cLlmTensorSsmAlpha,      // qwen3.5
  // Kimi Linear KDA (using SSM_ prefix for consistency)
  cLlmTensorSsmConv1dQ,      // kimi: Q conv1d weight
  cLlmTensorSsmConv1dK,      // kimi: K conv1d weight
  cLlmTensorSsmConv1dV,      // kimi: V conv1d weight
  cLlmTensorSsmFA,           // kimi: forget gate projection A
  cLlmTensorSsmFB,           // kimi: forget gate projection B
  cLlmTensorSsmBeta,         // kimi: beta mixing coefficient and qwen3.5
  cLlmTensorSsmGA,           // kimi: output gate projection A
  cLlmTensorSsmGB,           // kimi: output gate projection B
  cLlmTensorSsmG,            // kimi-k3: full-rank KDA gate
  cLlmTensorAttnResScore,    // kimi-k3: fused res_norm*res_proj (pre-attn)
  cLlmTensorFfnResScore,     // kimi-k3: fused res_norm*res_proj (pre-ffn)
  cLlmTensorOutputResScore,  // kimi-k3: fused res_norm*res_proj (final)
  cLlmTensorFfnRoutedDown,   // kimi-k3: latent MoE down
  cLlmTensorFfnRoutedUp,     // kimi-k3: latent MoE up
  cLlmTensorFfnRoutedNorm,   // kimi-k3: latent MoE norm
  cLlmTensorTimeMixW0,
  cLlmTensorTimeMixW1,
  cLlmTensorTimeMixW2,
  cLlmTensorTimeMixA0,
  cLlmTensorTimeMixA1,
  cLlmTensorTimeMixA2,
  cLlmTensorTimeMixV0,
  cLlmTensorTimeMixV1,
  cLlmTensorTimeMixV2,
  cLlmTensorTimeMixG1,
  cLlmTensorTimeMixG2,
  cLlmTensorTimeMixKK,
  cLlmTensorTimeMixKA,
  cLlmTensorTimeMixRK,
  cLlmTensorTimeMixLerpX,
  cLlmTensorTimeMixLerpW,
  cLlmTensorTimeMixLerpK,
  cLlmTensorTimeMixLerpV,
  cLlmTensorTimeMixLerpR,
  cLlmTensorTimeMixLerpG,
  cLlmTensorTimeMixLerpFused,
  cLlmTensorTimeMixFirst,
  cLlmTensorTimeMixDecay,
  cLlmTensorTimeMixDecayW1,
  cLlmTensorTimeMixDecayW2,
  cLlmTensorTimeMixKey,
  cLlmTensorTimeMixValue,
  cLlmTensorTimeMixReceptance,
  cLlmTensorTimeMixGate,
  cLlmTensorTimeMixLn,
  cLlmTensorTimeMixOutput,
  cLlmTensorChannelMixLerpK,
  cLlmTensorChannelMixLerpR,
  cLlmTensorChannelMixKey,
  cLlmTensorChannelMixReceptance,
  cLlmTensorChannelMixValue,
  cLlmTensorAttnQA,
  cLlmTensorAttnQB,
  cLlmTensorAttnKvAMqa,
  cLlmTensorAttnKvB,
  cLlmTensorAttnKv,
  cLlmTensorAttnKvNorm,
  cLlmTensorAttnOutA,
  cLlmTensorAttnOutB,
  cLlmTensorAttnKB,
  cLlmTensorAttnVB,
  cLlmTensorAttnQANorm,
  cLlmTensorAttnKvANorm,
  cLlmTensorHcHeadFn,
  cLlmTensorHcHeadBase,
  cLlmTensorHcHeadScale,
  cLlmTensorHcHeadNorm,  // qwen4exp
  cLlmTensorHcHeadDown,  // qwen4exp
  cLlmTensorHcHeadUp,    // qwen4exp
  cLlmTensorHcAttnFn,
  cLlmTensorHcAttnBase,
  cLlmTensorHcAttnScale,
  cLlmTensorHcFfnFn,
  cLlmTensorHcFfnBase,
  cLlmTensorHcFfnScale,
  cLlmTensorHcAttnNorm,    // qwen4exp
  cLlmTensorHcAttnDown,    // qwen4exp
  cLlmTensorHcAttnUp,      // qwen4exp
  cLlmTensorHcAttnInject,  // qwen4exp
  cLlmTensorHcFfnNorm,     // qwen4exp
  cLlmTensorHcFfnDown,     // qwen4exp
  cLlmTensorHcFfnUp,       // qwen4exp
  cLlmTensorHcFfnInject,   // qwen4exp
  cLlmTensorPleKey,        // qwen4exp
  cLlmTensorPleValue,      // qwen4exp
  cLlmTensorPleNormKey,    // qwen4exp
  cLlmTensorPleNormQuery,  // qwen4exp
  cLlmTensorPleNormConv,   // qwen4exp
  cLlmTensorPleConv1d,     // qwen4exp
  cLlmTensorAttnCompressorWkv,
  cLlmTensorAttnCompressorWgate,
  cLlmTensorAttnCompressorApe,
  cLlmTensorAttnCompressorNorm,
  cLlmTensorAttnSubNorm,
  cLlmTensorFfnSubNorm,
  cLlmTensorDecAttnNorm,
  cLlmTensorDecAttnQ,
  cLlmTensorDecAttnK,
  cLlmTensorDecAttnV,
  cLlmTensorDecAttnOut,
  cLlmTensorDecAttnRelB,
  cLlmTensorDecCrossAttnNorm,
  cLlmTensorDecCrossAttnNormKv,
  cLlmTensorDecCrossAttnQ,
  cLlmTensorDecCrossAttnK,
  cLlmTensorDecCrossAttnV,
  cLlmTensorDecCrossAttnOut,
  cLlmTensorDecCrossAttnRelB,
  cLlmTensorDecFfnNorm,
  cLlmTensorDecFfnGate,
  cLlmTensorDecFfnDown,
  cLlmTensorDecFfnUp,
  cLlmTensorDecOutputNorm,
  cLlmTensorEncAttnNorm,
  cLlmTensorEncAttnQ,
  cLlmTensorEncAttnK,
  cLlmTensorEncAttnV,
  cLlmTensorEncAttnOut,
  cLlmTensorEncAttnRelB,
  cLlmTensorEncFfnNorm,
  cLlmTensorEncFfnGate,
  cLlmTensorEncFfnDown,
  cLlmTensorEncFfnUp,
  cLlmTensorEncOutputNorm,
  cLlmTensorCls,
  cLlmTensorClsOut,
  cLlmTensorClsNorm,
  cLlmTensorDecisionHiddenNorm,
  cLlmTensorDecisionProjMemory,
  cLlmTensorDecisionProjQuestion,
  cLlmTensorDecisionProjOptionQuestion,
  cLlmTensorDecisionProjGlobal,
  cLlmTensorDecisionProjOptionContext,
  cLlmTensorDecisionProjOptionLexical,
  cLlmTensorDecisionOptionSummaryNorm,
  cLlmTensorDecisionFieldNorm,
  cLlmTensorDecisionOptionNorm,
  cLlmTensorDecisionScales,
  cLlmTensorDecisionScorer,
  cLlmTensorDecisionScorerOut,
  cLlmTensorDecisionQueryNorm,
  cLlmTensorDecisionKey,
  cLlmTensorDecisionQuery,
  cLlmTensorDecisionOptionMlp,
  cLlmTensorDecisionQueryMlp,
  cLlmTensorConv1d,
  cLlmTensorConvnextDw,
  cLlmTensorConvnextNorm,
  cLlmTensorConvnextPw1,
  cLlmTensorConvnextPw2,
  cLlmTensorConvnextGamma,
  cLlmTensorPosNetConv1,
  cLlmTensorPosNetConv2,
  cLlmTensorPosNetNorm,
  cLlmTensorPosNetNorm1,
  cLlmTensorPosNetNorm2,
  cLlmTensorPosNetAttnNorm,
  cLlmTensorPosNetAttnQ,
  cLlmTensorPosNetAttnK,
  cLlmTensorPosNetAttnV,
  cLlmTensorPosNetAttnOut,
  cLlmTensorShortconvConv,
  cLlmTensorShortconvInproj,
  cLlmTensorShortconvOutproj,
  cLlmTensorVisexpAttnQkv,
  cLlmTensorVisexpAttnOut,
  cLlmTensorVisexpFfnGate,
  cLlmTensorVisexpFfnDown,
  cLlmTensorVisexpFfnUp,
  cLlmTensorIndexerKNorm,
  cLlmTensorIndexerProj,
  cLlmTensorIndexerAttnK,
  cLlmTensorIndexerAttnQB,
  cLlmTensorIndexerQProj,
  cLlmTensorIndexerKProj,
  cLlmTensorIndexerQNorm,
  cLlmTensorIndexerCompressorWkv,
  cLlmTensorIndexerCompressorWgate,
  cLlmTensorIndexerCompressorApe,
  cLlmTensorIndexerCompressorNorm,
  cLlmTensorIndexerKpoolGate,  // glm5-next: k-pool gate scores
  cLlmTensorIndexerKpoolApe,   // glm5-next: k-pool position bias
  cLlmTensorFfnGateTid2eid,
  cLlmTensorNextnProjPre,
  cLlmTensorNextnProjPost,
  cLlmTensorNextnEhProj,
  cLlmTensorNextnEmbedTokens,
  cLlmTensorNextnEnorm,
  cLlmTensorNextnHnorm,
  cLlmTensorNextnSharedHeadHead,
  cLlmTensorNextnSharedHeadNorm,
  cLlmTensorNextnHcHeadNorm,  // qwen4exp: the MTP block's own hyper-connection mixer
  cLlmTensorNextnHcHeadDown,
  cLlmTensorNextnHcHeadUp,
  cLlmTensorMaskedEmbdCentroids,
  cLlmTensorMaskedEmbdOrdering,
  cLlmTensorHrmZLInit,
  cLlmTensorFc,
  cLlmTensorD2t,
  cLlmTensorDsparkMarkovW1,
  cLlmTensorDsparkMarkovW2,
  cLlmTensorDsparkConfProj,
  cLlmTensorDflashAttnConvBase,
  cLlmTensorDflashAttnConvProj,
  cLlmTensorDflashFfnConvBase,
  cLlmTensorDflashFfnConvProj,
  cLlmTensorDflashSelectorPrev,
  cLlmTensorDflashSelectorNext,
  cLlmTensorDflashSelectorHidden,
  cLlmTensorAttnVGate,
  cLlmTensorAttnVExps,
};

enum llm_tensor_layer {
  cLlmTensorLayerInput,
  cLlmTensorLayerRepeating,
  cLlmTensorLayerOutput,
};

struct LLM_KV {
  LLM_KV(llm_arch pArch, const char * pSuffix = nullptr);

  llm_arch     vArch;
  const char * vSuffix;

  std::string operator()(llm_kv pKv) const;
};

// helper to handle gguf constants
// usage:
//
//   const auto tn = LLM_TN(LLM_ARCH_DGS);
//
//   std::string name = tn(LLM_TENSOR_OUTPUT);                     -> "output"
//   std::string name = tn(LLM_TENSOR_TOKEN_EMBD, "bias");         -> "token_embd.bias"
//   std::string name = tn(LLM_TENSOR_ATTN_NORM, "weight", 3);     -> "blk.3.attn_norm.weight"
//
struct LLM_TN_IMPL {
  const llm_arch     cArch;
  const llm_tensor   cTensor;
  const char * const cSuffix;
  const int          cBid;
  const int          cXid;

  LLM_TN_IMPL(llm_arch pArch, llm_tensor pTensor, const char * pSuffix, int pBid, int pXid);

  std::string str() const;

  operator std::string() const { return str(); }

  friend bool operator==(const std::string & str, const LLM_TN_IMPL & pTn) { return str == pTn.str(); }

  friend bool operator!=(const std::string & str, const LLM_TN_IMPL & pTn) { return str != pTn.str(); }
};

struct LLM_TN {
  LLM_TN(llm_arch pArch) : vArch(pArch) {}

  llm_arch vArch;

  LLM_TN_IMPL operator()(llm_tensor pTensor, const char * pSuffix, int pBid = -1, int pXid = -1) const {
    return LLM_TN_IMPL(vArch, pTensor, pSuffix, pBid, pXid);
  }

  LLM_TN_IMPL operator()(llm_tensor pTensor, int pBid = -1, int pXid = -1) const {
    return LLM_TN_IMPL(vArch, pTensor, nullptr, pBid, pXid);
  }
};

struct llm_tensor_info {
  llm_tensor_layer layer;
  ggml_op          vOp;
};

std::vector<llm_arch> fLlmArchAll();

const char * fLlmArchName(llm_arch pArch);

llm_arch fLlmArchFromString(const std::string & name);

const llm_tensor_info & fLlmTensorInfoFor(llm_tensor pTensor);

bool fLlmArchIsRecurrent(const llm_arch & pArch);
bool fLlmArchIsHybrid(const llm_arch & pArch);
bool fLlmArchIsDiffusion(const llm_arch & pArch);
bool fLlmArchSupportsSmTensor(const llm_arch & pArch);
bool fLlmArchSupportsRsRollback(const llm_arch & pArch);
bool fLlmArchSupportsMixedBatch(const llm_arch & pArch);
