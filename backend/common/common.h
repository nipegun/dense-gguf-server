// Various helper functions and utilities

#pragma once

#include "dgs-cpp.h"
#include "dgs.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <list>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32) && !defined(_WIN32_WINNT)
#  define _WIN32_WINNT 0x0A00
#endif

#ifdef _WIN32
#  define DIRECTORY_SEPARATOR '\\'
#else
#  define DIRECTORY_SEPARATOR '/'
#endif  // _WIN32

#define COM_DBG(fmt, ...) LOG_DBG("cmn  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define COM_TRC(fmt, ...) LOG_TRC("cmn  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define COM_INF(fmt, ...) LOG_INF("cmn  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define COM_WRN(fmt, ...) LOG_WRN("cmn  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define COM_ERR(fmt, ...) LOG_ERR("cmn  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define COM_CNT(fmt, ...) LOG_CNT("" fmt, __VA_ARGS__)

#define die(msg)                       \
  do {                                 \
    fputs("error: " msg "\n", stderr); \
    exit(1);                           \
  } while (0)
#define die_fmt(fmt, ...)                             \
  do {                                                \
    fprintf(stderr, "error: " fmt "\n", __VA_ARGS__); \
    exit(1);                                          \
  } while (0)

struct common_time_meas {
  common_time_meas(int64_t & pTAcc, bool pDisable = false);
  ~common_time_meas();

  const int64_t cTStartUs;

  int64_t & vTAcc;
};

struct common_adapter_lora_info {
  std::string vPath;
  float       vScale;

  std::string vTaskName;
  std::string vPromptPrefix;

  struct dgs_adapter_lora * vPtr;
};

using dgs_tokens = std::vector<dgs_token>;

struct common_control_vector_load_info;

//
// CPU utils
//

struct common_cpu_params {
  int                      vNThreads                    = -1;
  bool                     aCpumask[GGML_MAX_N_THREADS] = { false };  // CPU affinity mask.
  bool                     vMaskValid                   = false;      // Default: any CPU
  enum ggml_sched_priority vPriority =
    GGML_SCHED_PRIO_NORMAL;     // Scheduling prio : (0 - normal, 1 - medium, 2 - high, 3 - realtime)
  bool     vStrictCpu = false;  // Use strict CPU placement
  uint32_t vPoll      = 50;     // Polling (busywait) level (0 - no polling, 100 - mostly polling)
};

int32_t fCommonCpuGetNumPhysicalCores();
int32_t fCommonCpuGetNumMath();

//
// Common params
//

enum dgs_example {
  cDgsExampleBatched,
  cDgsExampleDebug,
  cDgsExampleCommon,
  cDgsExampleSpeculative,
  cDgsExampleCompletion,
  cDgsExampleCli,
  cDgsExampleEmbedding,
  cDgsExamplePerplexity,
  cDgsExampleRetrieval,
  cDgsExamplePasskey,
  cDgsExampleImatrix,
  cDgsExampleBench,
  cDgsExampleServer,
  cDgsExampleCvectorGenerator,
  cDgsExampleExportLora,
  cDgsExampleMtmd,
  cDgsExampleLookup,
  cDgsExampleParallel,
  cDgsExampleTts,
  cDgsExampleDiffusion,
  cDgsExampleFinetune,
  cDgsExampleFitParams,
  cDgsExampleResults,
  cDgsExampleExportGraphOps,
  cDgsExampleDownload,
  cDgsExampleTokenize,

  cDgsExampleCount,
};

enum common_sampler_type {
  cCommonSamplerTypeNone        = 0,
  cCommonSamplerTypeDry         = 1,
  cCommonSamplerTypeTopK        = 2,
  cCommonSamplerTypeTopP        = 3,
  cCommonSamplerTypeMinP        = 4,
  //COMMON_SAMPLER_TYPE_TFS_Z       = 5,
  cCommonSamplerTypeTypicalP    = 6,
  cCommonSamplerTypeTemperature = 7,
  cCommonSamplerTypeXtc         = 8,
  cCommonSamplerTypeInfill      = 9,
  cCommonSamplerTypePenalties   = 10,
  cCommonSamplerTypeTopNSigma   = 11,
  cCommonSamplerTypeAdaptiveP   = 12,
};

// dimensionality reduction methods, used by cvector-generator
enum dimre_method {
  cDimreMethodPca,
  cDimreMethodMean,
};

enum common_conversation_mode {
  cCommonConversationModeDisabled = 0,
  cCommonConversationModeEnabled  = 1,
  cCommonConversationModeAuto     = 2,
};

enum common_grammar_trigger_type {
  cCommonGrammarTriggerTypeToken,
  cCommonGrammarTriggerTypeWord,
  cCommonGrammarTriggerTypePattern,
  cCommonGrammarTriggerTypePatternFull,
};

struct common_grammar_trigger {
  common_grammar_trigger_type type;
  std::string                 value;
  dgs_token                   token = DGS_TOKEN_NULL;
};

enum common_params_sampling_config : uint64_t {
  cCommonParamsSamplingConfigSamplers       = 1 << 0,
  cCommonParamsSamplingConfigTopK           = 1 << 1,
  cCommonParamsSamplingConfigTopP           = 1 << 2,
  cCommonParamsSamplingConfigMinP           = 1 << 3,
  cCommonParamsSamplingConfigXtcProbability = 1 << 4,
  cCommonParamsSamplingConfigXtcThreshold   = 1 << 5,
  cCommonParamsSamplingConfigTemp           = 1 << 6,
  cCommonParamsSamplingConfigPenaltyLastN   = 1 << 7,
  cCommonParamsSamplingConfigPenaltyRepeat  = 1 << 8,
  cCommonParamsSamplingConfigMirostat       = 1 << 9,
  cCommonParamsSamplingConfigMirostatTau    = 1 << 10,
  cCommonParamsSamplingConfigMirostatEta    = 1 << 11,
};

enum common_speculative_type {
  cCommonSpeculativeTypeNone,         // no speculative decoding
  cCommonSpeculativeTypeDraftSimple,  // standalone draft model speculative decoding
  cCommonSpeculativeTypeDraftEagle3,  // Eagle3 speculative decoding
  cCommonSpeculativeTypeDraftMtp,     // Multi-token prediction
  cCommonSpeculativeTypeDraftDflash,  // DFlash speculative decoding
  cCommonSpeculativeTypeDraftDspark,  // DSpark speculative decoding (DFlash + Markov head)
  cCommonSpeculativeTypeNgramSimple,  // simple self-speculative decoding based on n-grams
  cCommonSpeculativeTypeNgramMapK,    // self-speculative decoding with n-gram keys only
  cCommonSpeculativeTypeNgramMapK4v,  // self-speculative decoding with n-gram keys and 4 m-gram values
  cCommonSpeculativeTypeNgramMod,
  cCommonSpeculativeTypeNgramCache,   // self-speculative decoding with 3-level n-gram cache
  cCommonSpeculativeTypeCount         // number of types, unknown type
};

// Grammar type enumeration
enum common_grammar_type {
  cCommonGrammarTypeNone,          // no grammar set
  cCommonGrammarTypeUser,          // user-provided GBNF (--grammar / "grammar" API field)
  cCommonGrammarTypeOutputFormat,  // auto-generated from JSON schema (--json-schema / "json_schema" API field)
  cCommonGrammarTypeToolCalls,     // auto-generated by chat template parser for function calling
};

// Grammar variant struct with type and grammar string
struct common_grammar {
  common_grammar_type type = cCommonGrammarTypeNone;
  std::string         vGrammar;

  // Default constructor - no grammar
  common_grammar() = default;

  // Constructor with type and grammar string
  common_grammar(common_grammar_type pT, std::string pG) : type(pT), vGrammar(std::move(pG)) {
    GGML_ASSERT(type != cCommonGrammarTypeNone || !vGrammar.empty());
  }

  // Check if a grammar is set
  bool empty() const { return type == cCommonGrammarTypeNone || vGrammar.empty(); }
};

// Returns the raw grammar string, or empty string if no grammar is set.
inline const std::string & fCommonGrammarValue(const common_grammar & pG) {
  return pG.vGrammar;
}

// Returns true when the generation_prompt should be prefilled into the grammar sampler.
// Only output-format and tool-call grammars need prefill; user-supplied grammars must not be prefilled.
inline bool fCommonGrammarNeedsPrefill(const common_grammar & pG) {
  return pG.type == cCommonGrammarTypeOutputFormat || pG.type == cCommonGrammarTypeToolCalls;
}

// sampling parameters
struct common_params_sampling {
  uint32_t vSeed = DGS_DEFAULT_SEED;  // the seed used to initialize dgs_sampler

  int32_t vNPrev            = 64;     // number of previous tokens to remember
  int32_t vNProbs           = 0;      // if greater than 0, output the probabilities of top n_probs tokens.
  int32_t vMinKeep          = 0;      // 0 = disabled, otherwise samplers should return at least min_keep tokens
  int32_t vTopK             = 40;     // <= 0 to use vocab size
  float   vTopP             = 0.95f;  // 1.0 = disabled
  float   vMinP             = 0.05f;  // 0.0 = disabled
  float   vXtcProbability   = 0.00f;  // 0.0 = disabled
  float   vXtcThreshold     = 0.10f;  // > 0.5 disables XTC
  float   vTypP             = 1.00f;  // typical_p, 1.0 = disabled
  float   vTemp             = 0.80f;  // <= 0.0 to sample greedily, 0.0 to not output probabilities
  float   vDynatempRange    = 0.00f;  // 0.0 = disabled
  float   vDynatempExponent = 1.00f;  // controls how entropy maps to temperature in dynamic temperature sampler
  int32_t vPenaltyLastN     = 64;     // last n tokens to penalize (0 = disable penalty)
  float   vPenaltyRepeat    = 1.00f;  // 1.0 = disabled
  float   vPenaltyFreq      = 0.00f;  // 0.0 = disabled
  float   vPenaltyPresent   = 0.00f;  // 0.0 = disabled
  float   vDryMultiplier    = 0.0f;   // 0.0 = disabled;      DRY repetition penalty for tokens extending repetition:
  float   vDryBase =
    1.75f;  // 0.0 = disabled;      multiplier * base ^ (length of sequence before token - allowed length)
  int32_t vDryAllowedLength = 2;    // tokens extending repetitions beyond this receive penalty
  int32_t vDryPenaltyLastN  = 64;   // how many tokens to scan for repetitions (0 = disable penalty)
  float   vAdaptiveTarget = -1.0f;  // select tokens near this probability (valid range 0.0 to 1.0; negative = disabled)
  float   vAdaptiveDecay  = 0.90f;  // EMA decay for adaptation; history ≈ 1/(1-decay) tokens (0.0 - 0.99)
  int32_t vMirostat       = 0;      // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
  float   vTopNSigma      = -1.00f;  // -1.0 = disabled
  float   vMirostatTau    = 5.00f;   // target entropy
  float   vMirostatEta    = 0.10f;   // learning rate
  bool    vIgnoreEos      = false;
  bool    vNoPerf         = false;   // disable performance metrics
  bool    vTimingPerToken = false;

  uint64_t vUserSamplingConfig = 0;  // bitfield to track user-specified samplers

  std::vector<std::string> lDrySequenceBreakers = { "\n", ":", "\"", "*" };  // default sequence breakers for DRY

  std::vector<enum common_sampler_type> lSamplers = {
    cCommonSamplerTypePenalties, cCommonSamplerTypeDry,      cCommonSamplerTypeTopNSigma,
    cCommonSamplerTypeTopK,      cCommonSamplerTypeTypicalP, cCommonSamplerTypeTopP,
    cCommonSamplerTypeMinP,      cCommonSamplerTypeXtc,      cCommonSamplerTypeTemperature,
  };

  common_grammar                      vGrammar;  // optional grammar constraint (user / output-format / tool-calls)
  bool                                vGrammarLazy = false;
  std::vector<common_grammar_trigger> lGrammarTriggers;  // optional triggers (for lazy grammars)
  std::set<dgs_token>                 sPreservedTokens;

  std::vector<dgs_logit_bias> lLogitBias;     // logit biases to apply
  std::vector<dgs_logit_bias> lLogitBiasEog;  // pre-calculated logit biases for EOG tokens

  // The assistant generation prompt already prefilled into the prompt.
  // Fed to the grammar sampler (to advance past pre-existing tokens) and used
  // to determine the reasoning budget sampler's initial state.
  // Only applied when the grammar is of output-format or tool-calls type.
  std::string vGenerationPrompt;

  // reasoning budget sampler parameters
  // these are populated by the server/CLI based on chat template params
  int32_t                vReasoningBudgetTokens = -1;  // -1 = disabled, >= 0 = token budget
  std::vector<dgs_token> lReasoningBudgetStart;        // start tag token sequence
  std::vector<dgs_tokens>
                         lReasoningBudgetEnd;  // end tag token sequences; the first tag is used as the forcing sequence
  std::vector<dgs_token> lReasoningBudgetForced;   // forced sequence (message + first end tag)
  std::string            vReasoningBudgetMessage;  // message injected before end tag when budget exhausted
  bool vReasoningControl = false;  // create the budget sampler on demand so reasoning can be ended at runtime

  bool vBackendSampling = false;

  // print the parameters into a string
  std::string mPrint() const;
};

struct common_params_model {
  std::string vPath       = "";  // model local path
  std::string vUrl        = "";  // model url to download
  std::string vHfRepo     = "";  // HF repo
  std::string hf_file     = "";  // HF file
  std::string vDockerRepo = "";  // Docker repo

  std::string mGetName() const {
    if (!vHfRepo.empty()) {
      return vHfRepo;
    }
    if (!vDockerRepo.empty()) {
      return vDockerRepo;
    }
    return vPath;
  }

  bool empty() const { return mGetName().empty(); }
};

// draft-model-based speculative decoding parameters
struct common_params_speculative_draft {
  int32_t vNMax = 3;             // maximum number of tokens to draft during speculative decoding
  int32_t vNMin = 0;             // minimum number of draft tokens to use for speculative decoding

  float vPSplit = 0.1f;          // speculative decoding split probability
  float vPMin   = 0.0f;          // minimum speculative decoding probability (greedy)

  bool vBackendSampling = true;  // offload draft sampling to the backend (default: on)

  bool vProbabilistic = false;   // sample the draft and verify by rejection, instead of argmax and match

  common_params_model vMparams;

  dgs_context * vCtxTgt = nullptr;
  dgs_context * vCtxDft = nullptr;

  int32_t vNGpuLayers = -1;               // number of layers to store in VRAM for the draft model (-1 - use default)

  ggml_type vCacheTypeK = GGML_TYPE_F16;  // KV cache data type for the K
  ggml_type vCacheTypeV = GGML_TYPE_F16;  // KV cache data type for the V

  common_cpu_params vCpuparams;
  common_cpu_params vCpuparamsBatch;

  std::vector<ggml_backend_dev_t> lDevices;  // devices to use for offloading

  std::vector<dgs_model_tensor_buft_override> lTensorBuftOverrides;
};

struct common_params_speculative_ngram_mod {
  int32_t vNMatch = 24;

  int32_t vNMax = 64;
  int32_t vNMin = 48;
};

struct common_params_speculative_ngram_map {
  uint16_t vSizeN   = 12;  // ngram size for lookup
  uint16_t vSizeM   = 48;  // mgram size for speculative tokens
  uint16_t vMinHits = 1;   // minimum hits at ngram/mgram lookup for mgram to be proposed
};

struct common_params_speculative_ngram_cache {
  std::string vLookupCacheStatic;   // path of static ngram cache file for lookup decoding
  std::string vLookupCacheDynamic;  // path of dynamic ngram cache file for lookup decoding
};

struct common_params_speculative {
  std::vector<enum common_speculative_type> lTypes = { cCommonSpeculativeTypeNone };

  double              vSynthLen = -1.0;
  std::vector<double> lSynthRates;

  // used by Simple, MTP, Eagle3, etc. - all methods that require some kind of draft model
  common_params_speculative_draft vDraft;

  common_params_speculative_ngram_mod vNgramMod;
  common_params_speculative_ngram_map vNgramSimple;
  common_params_speculative_ngram_map vNgramMapK;
  common_params_speculative_ngram_map vNgramMapK4v;

  common_params_speculative_ngram_cache vNgramCache;

  bool mHasDft() const { return !vDraft.vMparams.empty(); }

  bool mHasSynth() const { return vSynthLen != -1.0 || !lSynthRates.empty(); }

  uint32_t mNeedNRsSeq() const {
    bool vNeedsRsSeq = std::any_of(lTypes.begin(), lTypes.end(), [&](auto pT) {
      return pT == cCommonSpeculativeTypeDraftMtp || pT == cCommonSpeculativeTypeDraftEagle3 ||
             pT == cCommonSpeculativeTypeDraftDflash || pT == cCommonSpeculativeTypeDraftDspark;
    });

    return vNeedsRsSeq ? vDraft.vNMax : 0u;
  }
};

struct common_params_diffusion {
  int32_t vSteps      = 128;
  bool    vVisualMode = false;

  float   vEps         = 0;       // epsilon for timesteps
  int32_t vBlockLength = 0;       // block length for generation

  int32_t vAlgorithm = 4;         // default algorithm: low-confidence
  float   vAlgTemp   = 0.0f;      // algorithm temperature

  float vCfgScale       = 0;      // classifier-free guidance scale
  bool  vAddGumbelNoise = false;  // add gumbel noise to the logits if temp > 0.0
};

// reasoning API response format (not to be confused as chat template's reasoning format)
// only used by server
enum common_reasoning_format {
  cCommonReasoningFormatNone,
  cCommonReasoningFormatAuto,  // Same as deepseek, using `message.reasoning_content`
  cCommonReasoningFormatDeepseek,  // Extract thinking tag contents and return as `message.reasoning_content`, including in streaming deltas.
                                   // do not extend this enum unless you absolutely have to
                                   // in most cases, use COMMON_REASONING_FORMAT_AUTO
};

struct common_params {
  int32_t vNPredict          = -1;     // max. number of new tokens to predict, -1 == no limit
  int32_t vNCtx              = 0;      // context size, 0 == context the model was trained with
  int32_t vNBatch            = 2048;   // logical batch size for prompt processing (must be >=32 to use BLAS)
  int32_t vNUbatch           = 512;    // physical batch size for prompt processing (must be >=32 to use BLAS)
  int32_t vNKeep             = 0;      // number of tokens to keep from initial prompt
  int32_t vNChunks           = -1;     // max number of chunks to process (-1 = unlimited)
  int32_t vNParallel         = 1;      // number of parallel sequences to decode
  int32_t vNSequences        = 1;      // number of sequences to decode
  int32_t vNOutputsMax       = 0;      // max outputs in a batch (0 = n_batch)
  int32_t vNOutputsMaxPerSeq = 1;      // max outputs per sequence
  int32_t vGrpAttnN          = 1;      // group-attention factor
  int32_t vGrpAttnW          = 512;    // group-attention width
  int32_t vNPrint            = -1;     // print token count every n tokens (-1 = disabled)
  float   vRopeFreqBase      = 0.0f;   // RoPE base frequency
  float   vRopeFreqScale     = 0.0f;   // RoPE frequency scaling factor
  float   vYarnExtFactor     = -1.0f;  // YaRN extrapolation mix factor
  float   vYarnAttnFactor    = -1.0f;  // YaRN magnitude scaling factor
  float   vYarnBetaFast      = -1.0f;  // YaRN low correction dim
  float   vYarnBetaSlow      = -1.0f;  // YaRN high correction dim
  int32_t vYarnOrigCtx       = 0;      // YaRN original context length

  // offload params
  std::vector<ggml_backend_dev_t> lDevices;  // devices to use for offloading

  int32_t vNGpuLayers       = -1;            // number of layers to store in VRAM, -1 is auto, <= -2 is all
  int32_t vMainGpu          = 0;             // the GPU that is used for scratch and small tensors
  float   aTensorSplit[128] = { 0 };         // how split tensors should be distributed across GPUs
  bool    vFitParams        = true;          // whether to fit unset model/context parameters to free device memory
  bool    vFitParamsPrint   = false;         // print the estimated required memory to run the model
  int32_t vFitParamsMinCtx  = 4096;          // minimum context size to set when trying to reduce memory use

  // margin per device in bytes for fitting parameters to free memory:
  std::vector<size_t> lFitParamsTarget = std::vector<size_t>(fDgsMaxDevices(), 1024 * 1024 * 1024);

  enum dgs_split_mode vSplitMode = cDgsSplitModeLayer;  // how to split the model across GPUs
  enum dgs_load_mode  vLoadMode  = cDgsLoadModeAuto;    // how to load the model

  enum dgs_lazy_mode vLazyMode = cDgsLazyModeAuto;      // on-demand reading of tensors marked by the arch

  common_cpu_params vCpuparams;
  common_cpu_params vCpuparamsBatch;

  ggml_backend_sched_eval_callback vCbEval         = nullptr;
  void *                           vCbEvalUserData = nullptr;

  ggml_numa_strategy vNuma = GGML_NUMA_STRATEGY_DISABLED;

  enum dgs_rope_scaling_type vRopeScalingType = cDgsRopeScalingTypeUnspecified;
  enum fDgsPoolingType       vPoolingType     = cDgsPoolingTypeUnspecified;    // pooling type for embeddings
  enum dgs_attention_type    vAttentionType   = cDgsAttentionTypeUnspecified;  // attention type for embeddings
  enum dgs_flash_attn_type   vFlashAttnType   = cDgsFlashAttnTypeAuto;         // whether to use Flash Attention

  struct common_params_sampling    vSampling;
  struct common_params_speculative vSpeculative;
  struct common_params_diffusion   vDiffusion;

  struct common_params_model vModel;

  std::set<std::string> sModelAlias;         // model aliases                                                 // NOLINT
  std::set<std::string> sModelTags;          // model tags (informational, not used for routing)              // NOLINT
  std::string           vHfToken      = "";  // HF token (aka bearer token)                                   // NOLINT
  std::string           vPrompt       = "";  // NOLINT
  std::string           vSystemPrompt = "";  // NOLINT
  std::string           vPromptFile   = "";  // store the external prompt file name                           // NOLINT
  std::string vPathPromptCache        = "";  // path to file for saving/loading prompt eval state             // NOLINT
  std::string vInputPrefix            = "";  // string to prefix user inputs with                             // NOLINT
  std::string vInputSuffix            = "";  // string to suffix user inputs with                             // NOLINT
  std::string vLogitsFile             = "";  // file for saving *all* logits                                  // NOLINT
  std::string vPathPromptsLogDir      = "";  // directory with logged prompts                                 // NOLINT

  // dgs-debug specific options
  std::string vLogitsOutputDir = "data";   // directory for saving logits output files                     // NOLINT
  bool        vSaveLogits      = false;    // whether to save logits to files                              // NOLINT
  std::vector<std::string> lTensorFilter;  // filter tensor names for debug output (regex)                 // NOLINT

  std::vector<std::string> lInFiles;       // all input files
  std::vector<std::string> lAntiprompt;    // strings upon which more user input is prompted (a.k.a. reverse prompts)
  std::vector<dgs_model_kv_override>          lKvOverrides;
  std::vector<dgs_model_tensor_buft_override> lTensorBuftOverrides;

  bool vLoraInitWithoutApply =
    false;  // only load lora to memory, but do not apply it to ctx (user can manually apply lora later using dgs_adapter_lora_apply)
  std::vector<common_adapter_lora_info> lLoraAdapters;           // lora adapter path with user defined scale

  std::vector<common_control_vector_load_info> lControlVectors;  // control vector with user defined scale

  int32_t vVerbosity               = 3;                          // LOG_LEVEL_INFO
  int32_t vControlVectorLayerStart = -1;                         // layer range for control vector
  int32_t vControlVectorLayerEnd   = -1;                         // layer range for control vector
  bool    vOffline                 = false;

  int32_t vPplStride = 0;  // stride for perplexity calculations. If left at 0, the pre-existing approach will be used.
  int32_t vPplOutputType  = 0;  // = 0 -> ppl output is as usual, = 1 -> ppl output is num_tokens, ppl, one per line
                                //                                       (which is more convenient to use for plotting)
                                //
  bool    vHellaswag      = false;  // compute HellaSwag score over random tasks from datafile supplied in prompt
  size_t  vHellaswagTasks = 400;    // number of tasks to use when computing the HellaSwag score

  bool   vWinogrande = false;       // compute Winogrande score over random tasks from datafile supplied in prompt
  size_t vWinograndeTasks =
    0;  // number of tasks to use when computing the Winogrande score. If 0, all tasks will be computed

  bool   vMultipleChoice = false;  // compute TruthfulQA score over random tasks from datafile supplied in prompt
  size_t vMultipleChoiceTasks =
    0;  // number of tasks to use when computing the TruthfulQA score. If 0, all tasks will be computed

  bool vKlDivergence = false;      // compute KL divergence

  bool vCheck = false;             // check rather than generate results for dgs-results

  bool vUsage            = false;  // print usage
  bool vCompletion       = false;  // print source-able completion script
  bool vUseColor         = false;  // use color to distinguish generations and inputs
  bool vSpecial          = false;  // enable special token output
  bool vInteractive      = false;  // interactive mode
  bool vInteractiveFirst = false;  // wait for user input immediately
  bool vPromptCacheAll   = false;  // save user input and generations to prompt cache
  bool vPromptCacheRo    = false;  // open the prompt cache read-only and do not update it

  bool vEscape         = true;     // escape "\n", "\r", "\t", "\'", "\"", and "\\"
  bool vMultilineInput = false;    // reverse the usage of `\`
  bool vSimpleIo       = false;    // improves compatibility with subprocesses and limited consoles
  bool vContBatching   = true;     // insert new sequences for decoding on-the-fly
  bool vNoPerf         = false;    // disable performance metrics
  bool vShowTimings    = true;     // show timing information on CLI
  bool vCtxShift       = false;    // context shift on infinite text generation
  bool vSwaFull =
    false;  // use full-size SWA cache (doc/MANUAL.md)
  bool vKvUnified = false;                // enable unified KV cache

  bool vInputPrefixBos = false;           // prefix BOS to user inputs, preceding input_prefix
  bool vVerbosePrompt  = false;           // print prompt tokens before generation
  bool vDisplayPrompt  = true;            // print prompt before generation
  bool vNoKvOffload    = false;           // disable KV offloading
  bool vWarmup         = true;            // warmup run
  bool vCheckTensors   = false;           // validate tensor data
  bool vNoOpOffload    = false;           // globally disable offload host tensor operations to device
  bool vNoExtraBufts   = false;           // disable extra buffer types (used for weight repacking)
  bool vNoHost         = false;           // bypass host buffer allowing extra buffers to be used
  bool vLoadMtp        = false;           // load MTP/NextN layers

  bool vSingleTurn = false;               // single turn chat conversation

  ggml_type vCacheTypeK = GGML_TYPE_F16;  // KV cache data type for the K
  ggml_type vCacheTypeV = GGML_TYPE_F16;  // KV cache data type for the V

  common_conversation_mode vConversationMode = cCommonConversationModeAuto;

  // multimodal models (see backend/mtmd)
  struct common_params_model vMmproj;
  bool                       vMmprojUseGpu = true;     // use GPU for multimodal model
  ggml_backend_dev_t         vMmprojDevice = nullptr;  // GPU device to use for multimodal model
  bool                       vNoMmproj     = false;    // explicitly disable multimodal model
  std::vector<std::string>   lImage;                   // path to image file(s) ; TODO: change the name to "media"
  int                        vImageMinTokens     = -1;
  int                        vImageMaxTokens     = -1;
  int                        vMtmdBatchMaxTokens = 1024;

  // for video input
  float       vVideoFps                 = 4.0f;
  int64_t     vVideoTimestampIntervalMs = 5000;
  std::string vVideoFfmpegBinDir        = "";

  // embedding
  bool    vEmbedding = false;  // get only sentence embedding
  int32_t vEmbdNormalize =
    2;  // normalisation for embeddings (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)
  std::string vEmbdOut =
    "";  // empty = default, "array" = [[],[]...], "json" = openai style, "json+" = same "json" + cosine similarity matrix
  std::string vEmbdSep = "\n";  // separator of embeddings
  std::string vClsSep  = "\t";  // separator of classification sequences

  // server params
  int32_t vPort              = 9931;            // server listens on this network port
  bool    vReusePort         = false;           // allow multiple sockets to bind to the same port
  int32_t vTimeoutRead       = 3600;            // http read timeout in seconds
  int32_t vTimeoutWrite      = vTimeoutRead;    // http write timeout in seconds
  int32_t vSsePingInterval   = 30;              // SSE ping interval in seconds
  int32_t vNThreadsHttp      = -1;              // number of threads to process HTTP requests (TODO: support threadpool)
  int32_t vNCacheReuse       = 0;               // min chunk size to reuse from the cache via KV shifting
  bool    vCachePrompt       = true;            // whether to enable prompt caching
  bool    vCacheIdleSlots    = true;            // save and clear idle slots upon starting a new task
  int32_t vNCtxCheckpoints   = 32;              // max number of context checkpoints per slot
  int32_t vKvUnifiedPerSlot  = 0;               // max context per parallel slot; 0 = unset
  int32_t vCheckpointMinStep = 8192;            // minimum spacing between context checkpoints
  int32_t vCacheRamMib       = 8192;            // -1 = no limit, 0 - disable, 1 = 1 MiB, etc.

  std::string              vPublicPath   = "";  // NOLINT
  std::string              vApiPrefix    = "/api";  // NOLINT
  std::string              vChatTemplate = "";      // NOLINT
  std::vector<std::string> lHostnames    = { "127.0.0.1" };
  bool                     vUseJinja     = true;    // NOLINT

  // server CORS params
  std::string vCorsOrigins         = "*";
  std::string vCorsMethods         = "GET, POST, DELETE, OPTIONS";
  std::string vCorsHeaders         = "*";
  bool        vCorsCredentials     = true;
  bool        vCorsOriginsExplicit = false;  // for --agent option

  bool                    vEnableChatTemplate     = true;
  bool                    vForcePureContentParser = false;
  common_reasoning_format vReasoningFormat        = cCommonReasoningFormatDeepseek;
  int                     vEnableReasoning        = -1;  // -1 = auto, 0 = disable, 1 = enable
  bool vPrefillAssistant = true;  // if true, any trailing assistant message will be prefilled into the response
  int  vSleepIdleSeconds = -1;    // if >0, server will sleep after this many seconds of idle time

  std::vector<std::string> lApiKeys;

  std::string vSslFileKey  = "";  // NOLINT
  std::string vSslFileCert = "";  // NOLINT

  std::map<std::string, std::string> dDefaultTemplateKwargs;
  bool                               vPreserveReasoningSpecified = false;

  // CLI params
  std::string vServerBase;  // if set, connect to this server instead of starting a new one

  // UI configs
  bool        vUi         = true;
  bool        vUiMcpProxy = false;
  std::string vUiConfigJson;

  // "advanced" endpoints are disabled by default for better security
  bool vEndpointSlots   = true;
  bool vEndpointProps   = false;  // only control POST requests, not GET
  bool vEndpointMetrics = false;

  // enable built-in tools
  std::vector<std::string> lServerTools;
  std::string              vServerToolsRuntime;

  // MCP server configs (Cursor-compatible JSON)
  bool        vShellMcpServer = false;
  std::string vMcpServersConfig;  // path to JSON file with MCP server definitions
  std::string vMcpServersJson;    // inline JSON with MCP server definitions

  // router server configs
  std::string vModelsDir      = "";    // directory containing models for the router server
  std::string vModelsPreset   = "";    // directory containing model presets for the router server
  int         vModelsMax      = 4;     // maximum number of models to load simultaneously
  bool        vModelsAutoload = true;  // automatically load models when requested via the router server
  std::string vModelsPresetHf = "";    // show a warning about remote presets on router loaded (if not empty)

  bool vLogJson = false;

  std::string vSlotSavePath;
  std::string vMediaPath;  // path to directory for loading media files

  float vSlotPromptSimilarity = 0.1f;

  // batched-bench params
  bool vIsPpShared   = false;
  bool vIsTgSeparate = false;

  std::vector<int32_t> lNPp;
  std::vector<int32_t> lNTg;
  std::vector<int32_t> lNPl;

  // retrieval params
  std::vector<std::string> lContextFiles;  // context files to embed

  int32_t vChunkSize = 64;                 // chunk size for context embedding

  std::string vChunkSeparator = "\n";      // chunk separator for context embedding

  // passkey params
  int32_t vNJunk = 250;  // number of times to repeat the junk text
  int32_t vIPos  = -1;   // position of the passkey in the junk text

  // imatrix params
  int32_t vNOutFreq  = 10;             // output the imatrix every n_out_freq iterations
  int32_t vNSaveFreq = 0;              // save the imatrix every n_save_freq iterations
  int32_t vIChunk    = 0;              // start processing from this chunk

  bool vProcessOutput        = false;  // collect data for the output tensor
  bool vComputePpl           = true;   // whether to compute perplexity
  bool vShowStatistics       = false;  // show imatrix statistics per tensor
  bool vActivationStatistics = false;  // generate data to calculate activation based statistics
  bool vParseSpecial         = false;  // whether to parse special tokens during imatrix tokenization

  // cvector-generator params
  int          vNPcaBatch           = 100;
  int          vNPcaIterations      = 1000;
  dimre_method vCvectorDimreMethod  = cDimreMethodPca;
  std::string  vCvectorPositiveFile = "tools/cvector-generator/positive.txt";
  std::string  vCvectorNegativeFile = "tools/cvector-generator/negative.txt";

  bool vSpmInfill = false;  // suffix/prefix/middle pattern for infill

  // batched-bench params
  bool vBatchedBenchOutputJsonl = false;

  // tokenize params
  bool vTokenizeIds       = false;  // if true, only print the token IDs
  bool vTokenizeStdin     = false;  // if true, read the prompt from stdin
  bool vTokenizeNoBos     = false;  // if true, do not add the BOS token
  bool vTokenizeShowCount = false;  // if true, print the total token count

  // common params
  std::string           vOutFile;  // output filename for all example programs
  // optional callback for model loading progress and cancellation:
  // called with a progress value between 0.0 and 1.0.
  // return false from callback to abort model loading or true to continue
  dgs_progress_callback vLoadProgressCallback         = NULL;
  void *                vLoadProgressCallbackUserData = NULL;
  bool                  vNoAlloc                      = false;  // Don't allocate model buffers

  // TTS params
  std::string vTtsLang        = "";
  std::string vTtsSpeakerFile = "";

  bool vIsGenDocs = false;  // whether we are running inside dgs-gen-docs
};

// call once at the start of a program if it uses libcommon
// initializes the logging system and prints info about the build
void fCommonInit();

void        fCommonParamsPrintInfo(const common_params & pParams, bool pPrintDevices = true);
std::string fCommonParamsGetSystemInfo(const common_params & pParams);

bool fParseCpuRange(const std::string & pRange, bool (&pBoolmask)[GGML_MAX_N_THREADS]);
bool fParseCpuMask(const std::string & pMask, bool (&pBoolmask)[GGML_MAX_N_THREADS]);
void fPostprocessCpuParams(common_cpu_params & pCpuparams, const common_cpu_params * pRoleModel = nullptr);
bool fSetProcessPriority(enum ggml_sched_priority pPrio);

//
// String utils
//

#ifdef __GNUC__
#  if defined(__MINGW32__) && !defined(__clang__)
#    define DGS_COMMON_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#  else
#    define DGS_COMMON_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#  endif
#else
#  define DGS_COMMON_ATTRIBUTE_FORMAT(...)
#endif

DGS_COMMON_ATTRIBUTE_FORMAT(1, 2)
std::string fStringFormat(const char * pFmt, ...);

std::string fStringStrip(const std::string & str);
std::string fStringGetSortableTimestamp();
std::string fStringLcs(std::string_view pA, std::string_view pB);

std::string              string_join(const std::vector<std::string> & pValues, const std::string & pSeparator);
std::vector<std::string> fStringSplit(const std::string & str, const std::string & pDelimiter);
std::string              fStringRepeat(const std::string & str, size_t n);

void fStringReplaceAll(std::string & pS, const std::string & pSearch, const std::string & pReplace);

std::string fRegexEscape(const std::string & pS);

template <class T> static std::vector<T> fStringSplit(const std::string & str, char pDelim) {
  static_assert(!std::is_same<T, std::string>::value, "Please use the specialized version for std::string");
  std::vector<T>     lValues;
  std::istringstream vStrStream(str);
  std::string        token;
  while (std::getline(vStrStream, token, pDelim)) {
    T                  value;
    std::istringstream vTokenStream(token);
    if (!(vTokenStream >> value)) {
      throw std::invalid_argument("invalid value: \"" + token + "\"");
    }
    lValues.push_back(value);
  }
  return lValues;
}

template <> inline std::vector<std::string> fStringSplit<std::string>(const std::string & str, char pDelim) {
  std::vector<std::string> lParts;
  size_t                   vBeginPos = 0;
  size_t                   vDelimPos = str.find(pDelim);
  while (vDelimPos != std::string::npos) {
    std::string vPart = str.substr(vBeginPos, vDelimPos - vBeginPos);
    lParts.emplace_back(vPart);
    vBeginPos = vDelimPos + 1;
    vDelimPos = str.find(pDelim, vBeginPos);
  }
  lParts.emplace_back(str.substr(vBeginPos));
  return lParts;
}

// remove when moving to c++20
inline bool fStringStartsWith(std::string_view str, std::string_view pPrefix) {
  return str.size() >= pPrefix.size() && str.compare(0, pPrefix.size(), pPrefix) == 0;
}

// remove when moving to c++20
inline bool fStringStartsWith(std::string_view str, char pPrefix) {
  return !str.empty() && str.front() == pPrefix;
}

// remove when moving to c++20
inline bool fStringEndsWith(std::string_view str, std::string_view pSuffix) {
  return str.size() >= pSuffix.size() && str.compare(str.size() - pSuffix.size(), pSuffix.size(), pSuffix) == 0;
}

inline bool fStringRemoveSuffix(std::string & str, std::string_view pSuffix) {
  if (fStringEndsWith(str, pSuffix)) {
    str.resize(str.size() - pSuffix.size());
    return true;
  }
  return false;
}

inline size_t fStringFindPartialStop(std::string_view str, std::string_view pStop) {
  if (!str.empty() && !pStop.empty()) {
    const size_t cMaxLen   = std::min(str.size(), pStop.size());
    const char   cLastChar = str.back();
    for (size_t vLen = cMaxLen; vLen > 0; --vLen) {
      if (pStop[vLen - 1] == cLastChar) {
        if (fStringEndsWith(str, pStop.substr(0, vLen))) {
          return str.size() - vLen;
        }
      }
    }
  }
  return std::string::npos;
}

bool fStringParseKvOverride(const char * data, std::vector<dgs_model_kv_override> & pOverrides);
void fStringProcessEscapes(std::string & pInput);

std::string fStringFrom(bool value);
std::string fStringFrom(const std::vector<int> & pValues);
std::string fStringFrom(const struct dgs_context * pCtx, const std::vector<dgs_token> & pTokens);

bool fGlobMatch(const std::string & pPattern, const std::string & str);

//
// Unicode utils
//

#ifdef _WIN32
std::wstring utf8_to_wstring(const std::string & str);
std::string  wstring_to_utf8(const std::wstring & str);
#endif

// returns the path as a UTF-8 string, preserving its separators
std::string fFsPathToUtf8(const std::filesystem::path & pPath);

//
// Environment utils
//

// portable environment access, an unset variable reads as an empty string
// and setting an empty value unsets the variable
std::string fCommonGetEnv(const std::string & name);
void        fCommonSetEnv(const std::string & name, const std::string & value);

// reads a path from the environment, an unset variable gives an empty path
std::filesystem::path fCommonGetPathFromEnv(const std::string & name);

//
// Filesystem utils
//

bool fFsValidateFilename(const std::string & pFilename, bool pAllowSubdirs = false);
bool fFsIsDirectory(const std::string & pPath);

// some old libstdc++ versions don't follow symlinks here, so adding a trailing "/" fixes it: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=101510
inline bool fCommonCreateDirectories(const std::filesystem::path & pPath, std::error_code & pEc) {
#if defined(__linux__)
  return std::filesystem::create_directories(pPath / "", pEc);
#else
  return std::filesystem::create_directories(path, ec);
#endif
}

std::filesystem::path fFsGetCacheDirectory();
std::filesystem::path fFsGetCacheFile(const std::string & pFilename);
std::filesystem::path fFsGetConfigDirectory();

void fFsWriteAtomic(const std::filesystem::path & pPath, const std::string & data);

//
// TTY utils
//

// Auto-detect if colors can be enabled based on terminal and environment
bool fTtyCanUseColors();
bool fTtyEnableAnsi();  // false when stdout or stderr is a console that cannot render ANSI sequences

// Check if the given file is attached to a terminal
bool fCommonIsTty(FILE * pFile);

//
// Model utils
//

struct common_sampler;

// typed decision models, see "<arch>.decision.type" in the model metadata
enum common_decision_type {
  cCommonDecisionTypeNone,     // not a decision model
  cCommonDecisionTypeOpenjev,  // logits of one label token per option, read at the last prompt token
  cCommonDecisionTypeLev,      // same as openjev, noul is read from a rating scale
  cCommonDecisionTypeKev,      // dot product of the hidden states of the last token and of one end token per option
  cCommonDecisionTypeNimble,   // same as openjev, the prompt lists all the questions of the request
  cCommonDecisionTypeLaya,     // score of one marker token per option, read from the embeddings output
  cCommonDecisionTypeClef,  // all questions in one prompt, score of option i read from the embeddings output at row i
  cCommonDecisionTypePplxDecider,  // same as openjev, label codes of 1 or 2 letters
  cCommonDecisionTypeVllmDecision2,  // score of each option read from the embeddings output, the options of a question are marked in the batch
  cCommonDecisionTypeVllmD3,      // same as pplx-decider, non-causal attention
  cCommonDecisionTypeLfm2D1,      // same as openjev, the labels depend on the question type
  cCommonDecisionTypeLfm2D1Omni,  // same as laya, other prompt layout
  cCommonDecisionTypeUnknown,     // a decision model of a type that is not supported
};

common_decision_type fCommonGetDecisionType(const struct dgs_model * pModel);

// metadata of a GGUF file, read without loading the model
struct common_gguf_info {
  common_decision_type vDecisionType =
    cCommonDecisionTypeUnknown;  // UNKNOWN if the file is missing, unreadable, or invalid
  uint32_t vNCtxTrain = 0;       // 0 if unknown
};

common_gguf_info fCommonGetGgufInfo(const std::string & pFname);

// note: defines the model, context, samplers, ets. lifetimes
struct common_init_result {
  common_init_result(common_params & pParams, bool pModelOnly = false);
  ~common_init_result();

  dgs_model *   mModel();
  dgs_context * mContext();

  common_sampler * mSampler(dgs_seq_id pSeqId);
  void             mResetSamplers();

  std::vector<dgs_adapter_lora_ptr> & lora();

private:
  struct impl;
  std::unique_ptr<impl> vPimpl;
};

using common_init_result_ptr = std::unique_ptr<common_init_result>;

common_init_result_ptr fCommonInitFromParams(common_params & pParams, bool pModelOnly = false);

struct dgs_model_params   fCommonModelParamsToDgs(common_params & pParams);
struct dgs_context_params fCommonContextParamsToDgs(const common_params & pParams);

// clear LoRA adapters from context, then apply new list of adapters
void fCommonSetAdapterLora(struct dgs_context * pCtx, std::vector<common_adapter_lora_info> & lora);

// model endpoint from env
std::string fCommonGetModelEndpoint();

// for testing purposes
char * fCommonGetModelOrExit(int, char *[]);

//
// Threadpool utils
//

struct ggml_threadpool_params fGgmlThreadpoolParamsFromCpuParams(const common_cpu_params & pParams);

struct common_threadpools {
  common_threadpools() = default;
  ~common_threadpools();

  common_threadpools(const common_threadpools &)             = delete;
  common_threadpools & operator=(const common_threadpools &) = delete;

  void init(dgs_context * pCtx, const common_params & pParams);

private:
  ggml_threadpool * vThreadpool      = nullptr;
  ggml_threadpool * vThreadpoolBatch = nullptr;

  decltype(ggml_threadpool_free) * vFreeFn = nullptr;
};

//
// Context utils
//

enum common_context_seq_rm_type {
  cCommonContextSeqRmTypeNo   = 0,  // seq_rm not supported (e.g. no memory module)
  cCommonContextSeqRmTypePart = 1,  // can seq_rm partial sequences
  cCommonContextSeqRmTypeFull = 2,  // can seq_rm full sequences only
  cCommonContextSeqRmTypeRs   = 3,  // can seq_rm partial sequences, bounded by n_rs_seq
};

// check if the dgs_context can remove sequences
// note: clears the memory of the context
common_context_seq_rm_type fCommonContextCanSeqRm(dgs_context * pCtx);

struct common_memory {
  dgs_context * vCtxTgt = nullptr;
  dgs_context * vCtxDft = nullptr;

  void init(dgs_context * pCtxTgt, dgs_context * pCtxDft = nullptr);

  // aborts execution on failure
  void mSeqRm(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1) const;
  void mSeqAdd(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1, dgs_pos pDelta) const;
  void mSeqCp(dgs_seq_id pSeqIdSrc, dgs_seq_id pSeqIdDst, dgs_pos pP0, dgs_pos pP1) const;
};

//
// Batch utils
//

// wrapper around dgs_batch_ext that provide getter functions for downstream code
// entries can exceed n_batch, use get_sub_batch() to decode them in chunks
struct common_batch {
  struct token {
    dgs_token                                vId;
    std::array<dgs_pos, GGML_MROPE_SECTIONS> aPos;    // only pos[0] is used for text tokens
    dgs_seq_id                               vSeqId;  // the first sequence id, see add_seq()
    bool                                     vOutput;
    dgs_embd                vEmbd;  // non-owning view of the data passed to add_embd()/set_embd(), data == NULL if none
    dgs_embd                vState;  // non-owning view of the data passed to set_embd_state(), data == NULL if none
    std::vector<dgs_seq_id> lSeqIdsExtra;        // see add_seq()
    int32_t                 vDecisionOrder = 0;  // see dgs_batch_ext_set_decision_order()
  };

  std::vector<token> lTokens;  // mirror of the entries, tokens[i] describes batch index i
  dgs_batch_ext_ptr  vBatch;

  int32_t vNPos = 1;  // positions per embedding entry, GGML_MROPE_SECTIONS for MROPE/IMROPE

  common_batch() = default;
  common_batch(struct dgs_context * pCtx);

  dgs_batch_ext * get() { return mGetSubBatch(0, size()); }

  // render entries [off, off + n) into batch, the result is overwritten by the next call
  dgs_batch_ext * mGetSubBatch(int32_t pOff, int32_t n);

  // content type of the batch, all entries carry the same combination
  bool mHasToken() const { return !lTokens.empty() && lTokens[0].vId != DGS_TOKEN_NULL; }

  bool mHasEmbd() const { return !lTokens.empty() && lTokens[0].vEmbd.data != nullptr; }

  void clear();

  // returns the batch index
  int32_t mAdd(dgs_token pId, dgs_pos pPos, dgs_seq_id pSeqId, bool pOutput);

  // same, with the entry shared by all seq_ids (must not be empty)
  int32_t mAdd(dgs_token pId, dgs_pos pPos, const std::vector<dgs_seq_id> & pSeqIds, bool pOutput);

  // add the entry at idx to another sequence, tokens[idx].seq_id keeps the first one
  bool mAddSeq(int32_t pIdx, dgs_seq_id pSeqId);

  bool mSetOutput(int32_t pIdx, bool value);

  // attach a token embedding to the entry at idx, can only be set once per entry
  bool mSetEmbd(int32_t pIdx, dgs_embd pEmbd);

  // attach a state embedding (e.g. the target hidden state for MTP) to the entry at idx, can only be set once per entry
  bool mSetEmbdState(int32_t pIdx, dgs_embd pState);

  // add an embedding-only entry (no token id)
  // pos points to n_pos positions
  int32_t mAddEmbd(dgs_embd pEmbd, const dgs_pos * pPos, dgs_seq_id pSeqId, bool pOutput);

  int32_t size() const { return (int32_t) lTokens.size(); }
};

// create a single-sequence batch from a list of tokens
// positions continue from the memory, last token always have output_logits set to true
common_batch fCommonBatchGetOne(struct dgs_context * pCtx, const dgs_token * pTokens, int32_t pNTokens);
common_batch fCommonBatchGetOne(struct dgs_context * pCtx, const dgs_tokens & pTokens);

// decodes a single batch of tokens for a prompt and manages session tokens
//
// Note: We save state before the last token so that we can replay it to ensure
// compatibility with all memory types. Recurrent/hybrid models cannot remove
// tokens from memory, so this approach works across all model architectures.
bool fCommonPromptBatchDecode(struct dgs_context * pCtx,
                              const dgs_tokens &   pAllTokens,
                              int                  pNNew,
                              int &                pNPast,
                              int                  pNBatch,
                              std::string_view     pStatePath,
                              bool                 pSaveState);

// replays the last token after loading state to regenerate logits
// used after loading session state to ensure the sampling context has valid logits
bool fCommonReplayLastToken(struct dgs_context * pCtx, dgs_token pLastToken, int32_t pPos);

//
// Vocab utils
//

// tokenizes a string into a vector of tokens
// should work similar to Python's `tokenizer.encode`
std::vector<dgs_token> fCommonTokenize(const struct dgs_context * pCtx,
                                       const std::string &        pText,
                                       bool                       pAddSpecial,
                                       bool                       pParseSpecial = false);

std::vector<dgs_token> fCommonTokenize(const struct dgs_vocab * pVocab,
                                       const std::string &      pText,
                                       bool                     pAddSpecial,
                                       bool                     pParseSpecial = false);

// tokenizes a token into a piece, optionally renders special/control tokens
// should work similar to Python's `tokenizer.id_to_piece`
std::string fCommonTokenToPiece(const struct dgs_context * pCtx, dgs_token token, bool pSpecial = true);

std::string fCommonTokenToPiece(const struct dgs_vocab * pVocab, dgs_token token, bool pSpecial = true);

// detokenizes a vector of tokens into a string
// should work similar to Python's `tokenizer.decode`
// optionally renders special/control tokens
std::string fCommonDetokenize(const struct dgs_context *     pCtx,
                              const std::vector<dgs_token> & pTokens,
                              bool                           pSpecial = true);

std::string fCommonDetokenize(const struct dgs_vocab *       pVocab,
                              const std::vector<dgs_token> & pTokens,
                              bool                           pSpecial = true);

//
// Embedding utils
//

// TODO: replace embd_norm with an enum
void fCommonEmbdNormalize(const float * pInp, float * pOut, int n, int pEmbdNorm);

float fCommonEmbdSimilarityCos(const float * pEmbd1, const float * pEmbd2, int n);

//
// Control vector utils
//

struct common_control_vector_data {
  int vNEmbd;

  // stores data for layers [1, n_layer] where n_layer = data.size() / n_embd
  std::vector<float> data;
};

struct common_control_vector_load_info {
  float vStrength;

  std::string vFname;
};

// Load control vectors, scale each by strength, and add them together.
// On error, returns {-1, empty}
common_control_vector_data fCommonControlVectorLoad(const std::vector<common_control_vector_load_info> & pLoadInfos);

//
// Split utils
//

namespace {

const char * const cLlmKvSplitNo           = "split.no";
const char * const cLlmKvSplitCount        = "split.count";
const char * const cLlmKvSplitTensorsCount = "split.tensors.count";

}  // namespace

//
// FFN offload utils
//

const char * const cLlmFfnDenseRegex = "\\.ffn_(up|down|gate)\\.";

inline std::string fLlmFfnBlockRegex(int pIdx, const char * pFfnRegex) {
  return fStringFormat("blk\\.%d%s", pIdx, pFfnRegex);
}

inline void fLlmAddNCpuFfnOverrides(int                                           n,
                                    const char *                                  pFfnRegex,
                                    std::vector<dgs_model_tensor_buft_override> & pOverrides) {
  // keep strings alive and avoid leaking memory by storing them in a static list
  static std::list<std::string> lBuftOverrideStrings;
  for (int vI = 0; vI < n; ++vI) {
    lBuftOverrideStrings.push_back(fLlmFfnBlockRegex(vI, pFfnRegex));
    pOverrides.push_back({ lBuftOverrideStrings.back().c_str(), ggml_backend_cpu_buffer_type() });
  }
}

//

//

// "adamw" or "sgd" (case insensitive)

//
// prompt utils
//

struct common_prompt_checkpoint {
  int64_t vNTokens;

  // (optional) id of the task that created the checkpoint
  int vIdTask = -1;

  dgs_pos vPosMin;
  dgs_pos vPosMax;

  std::vector<uint8_t> lDataTgt;
  std::vector<uint8_t> lDataDft;

  // (optional) speculative-decoding implementation state stashed with the checkpoint
  // (e.g. eagle3's deferred-boundary g_embd row)
  std::vector<uint8_t> lDataSpec;

  size_t size() const;

  bool empty() const;
  void clear();

  void mUpdatePos(int64_t pNTokens, dgs_pos pPosMin, dgs_pos pPosMax);

  void mUpdateTgt(dgs_context * pCtx, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags);

  void mUpdateDft(dgs_context * pCtx, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags);

  // return false if the state could not be restored
  bool mLoadTgt(dgs_context * pCtx, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags) const;

  bool mLoadDft(dgs_context * pCtx, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags) const;

  void mClearTgt();
  void mClearDft();
};
