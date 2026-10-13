#ifndef DGS_H
#define DGS_H

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef DGS_SHARED
#  if defined(_WIN32) && !defined(__MINGW32__)
#    ifdef DGS_BUILD
#      define DGS_API __declspec(dllexport)
#    else
#      define DGS_API __declspec(dllimport)
#    endif
#  else
#    define DGS_API __attribute__((visibility("default")))
#  endif
#else
#  define DGS_API
#endif

#define DGS_DEFAULT_SEED 0xFFFFFFFF

#define DGS_TOKEN_NULL -1

#define DGS_FILE_MAGIC_GGLA 0x67676c61u  // 'ggla'
#define DGS_FILE_MAGIC_GGSN 0x6767736eu  // 'ggsn'
#define DGS_FILE_MAGIC_GGSQ 0x67677371u  // 'ggsq'

#define DGS_SESSION_MAGIC   DGS_FILE_MAGIC_GGSN
#define DGS_SESSION_VERSION 11

#define DGS_STATE_SEQ_MAGIC   DGS_FILE_MAGIC_GGSQ
#define DGS_STATE_SEQ_VERSION 4

#ifdef __cplusplus
extern "C" {
#endif

//
// C interface
//
// TODO: show sample usage
//

struct dgs_vocab;
struct dgs_model;
struct dgs_context;
struct dgs_sampler;

typedef struct dgs_memory_i * dgs_memory_t;

typedef int32_t dgs_pos;
typedef int32_t dgs_token;
typedef int32_t dgs_seq_id;

enum fDgsVocabType {
  cDgsVocabTypeNone   = 0,  // For models without vocab
  cDgsVocabTypeSpm    = 1,  // LLaMA tokenizer based on byte-level BPE with byte fallback
  cDgsVocabTypeBpe    = 2,  // GPT-2 tokenizer based on byte-level BPE
  cDgsVocabTypeWpm    = 3,  // BERT tokenizer based on WordPiece
  cDgsVocabTypeUgm    = 4,  // T5 tokenizer based on Unigram
  cDgsVocabTypeRwkv   = 5,  // RWKV tokenizer based on greedy tokenization
  cDgsVocabTypePlamo2 = 6,  // PLaMo-2 tokenizer based on Aho-Corasick with dynamic programming
  cDgsVocabTypeTest   = 7,  // Dummy tokenizer for testing: rolling hash of fixed-size chunks -> tokens, tokens -> hex
  cDgsVocabTypePlamo3 = 8,  // PLaMo-3 tokenizer with pre-segmentation and dynamic programming
};

enum dgs_rope_type {
  cDgsRopeTypeNone   = -1,
  cDgsRopeTypeNorm   = 0,
  cDgsRopeTypeNeox   = GGML_ROPE_TYPE_NEOX,
  cDgsRopeTypeMrope  = GGML_ROPE_TYPE_MROPE,
  cDgsRopeTypeImrope = GGML_ROPE_TYPE_IMROPE,
  cDgsRopeTypeVision = GGML_ROPE_TYPE_VISION,
};

enum dgs_token_type {  //TODO: remove, required until per token attributes are available from GGUF file
  cDgsTokenTypeUndefined   = 0,
  cDgsTokenTypeNormal      = 1,
  cDgsTokenTypeUnknown     = 2,
  cDgsTokenTypeControl     = 3,
  cDgsTokenTypeUserDefined = 4,
  cDgsTokenTypeUnused      = 5,
  cDgsTokenTypeByte        = 6,
};

enum dgs_token_attr {
  cDgsTokenAttrUndefined   = 0,
  cDgsTokenAttrUnknown     = 1 << 0,
  cDgsTokenAttrUnused      = 1 << 1,
  cDgsTokenAttrNormal      = 1 << 2,
  cDgsTokenAttrControl     = 1 << 3,  // SPECIAL?
  cDgsTokenAttrUserDefined = 1 << 4,
  cDgsTokenAttrByte        = 1 << 5,
  cDgsTokenAttrNormalized  = 1 << 6,
  cDgsTokenAttrLstrip      = 1 << 7,
  cDgsTokenAttrRstrip      = 1 << 8,
  cDgsTokenAttrSingleWord  = 1 << 9,
};

// model file types
enum dgs_ftype {
  cDgsFtypeAllF32          = 0,
  cDgsFtypeMostlyF16       = 1,  // except 1d tensors
  cDgsFtypeMostlyQ4Point0  = 2,  // except 1d tensors
  cDgsFtypeMostlyQ4Point1  = 3,  // except 1d tensors
  // DGS_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,  // tok_embeddings.weight and output.weight are F16
  // DGS_FTYPE_MOSTLY_Q4_2       = 5,  // support has been removed
  // DGS_FTYPE_MOSTLY_Q4_3       = 6,  // support has been removed
  cDgsFtypeMostlyQ8Point0  = 7,   // except 1d tensors
  cDgsFtypeMostlyQ5Point0  = 8,   // except 1d tensors
  cDgsFtypeMostlyQ5Point1  = 9,   // except 1d tensors
  cDgsFtypeMostlyQ2K       = 10,  // except 1d tensors
  cDgsFtypeMostlyQ3KS      = 11,  // except 1d tensors
  cDgsFtypeMostlyQ3KM      = 12,  // except 1d tensors
  cDgsFtypeMostlyQ3KL      = 13,  // except 1d tensors
  cDgsFtypeMostlyQ4KS      = 14,  // except 1d tensors
  cDgsFtypeMostlyQ4KM      = 15,  // except 1d tensors
  cDgsFtypeMostlyQ5KS      = 16,  // except 1d tensors
  cDgsFtypeMostlyQ5KM      = 17,  // except 1d tensors
  cDgsFtypeMostlyQ6K       = 18,  // except 1d tensors
  cDgsFtypeMostlyIq2Xxs    = 19,  // except 1d tensors
  cDgsFtypeMostlyIq2Xs     = 20,  // except 1d tensors
  cDgsFtypeMostlyQ2KS      = 21,  // except 1d tensors
  cDgsFtypeMostlyIq3Xs     = 22,  // except 1d tensors
  cDgsFtypeMostlyIq3Xxs    = 23,  // except 1d tensors
  cDgsFtypeMostlyIq1S      = 24,  // except 1d tensors
  cDgsFtypeMostlyIq4Nl     = 25,  // except 1d tensors
  cDgsFtypeMostlyIq3S      = 26,  // except 1d tensors
  cDgsFtypeMostlyIq3M      = 27,  // except 1d tensors
  cDgsFtypeMostlyIq2S      = 28,  // except 1d tensors
  cDgsFtypeMostlyIq2M      = 29,  // except 1d tensors
  cDgsFtypeMostlyIq4Xs     = 30,  // except 1d tensors
  cDgsFtypeMostlyIq1M      = 31,  // except 1d tensors
  cDgsFtypeMostlyBf16      = 32,  // except 1d tensors
  //DGS_FTYPE_MOSTLY_Q4_0_4_4      = 33, // removed from gguf files, use Q4_0 and runtime repack
  //DGS_FTYPE_MOSTLY_Q4_0_4_8      = 34, // removed from gguf files, use Q4_0 and runtime repack
  //DGS_FTYPE_MOSTLY_Q4_0_8_8      = 35, // removed from gguf files, use Q4_0 and runtime repack
  cDgsFtypeMostlyTQ1Point0 = 36,  // except 1d tensors
  cDgsFtypeMostlyTQ2Point0 = 37,  // except 1d tensors
  cDgsFtypeMostlyNvfp4     = 39,  // except 1d tensors
  cDgsFtypeMostlyQ1Point0  = 40,  // except 1d tensors
  cDgsFtypeMostlyQ2Point0  = 41,  // except 1d tensors

  cDgsFtypeGuessed = 1024,        // not specified in the model file
};

// Get the model file type (quantization) as a string, e.g. "Q8_0" or "Q4_K - Medium"
DGS_API const char * fDgsFtypeName(enum dgs_ftype pFtype);

enum dgs_rope_scaling_type {
  cDgsRopeScalingTypeUnspecified = -1,
  cDgsRopeScalingTypeNone        = 0,
  cDgsRopeScalingTypeLinear      = 1,
  cDgsRopeScalingTypeYarn        = 2,
  cDgsRopeScalingTypeLongrope    = 3,
  cDgsRopeScalingTypeMaxValue    = cDgsRopeScalingTypeLongrope,
};

enum fDgsPoolingType {
  cDgsPoolingTypeUnspecified = -1,
  cDgsPoolingTypeNone        = 0,
  cDgsPoolingTypeMean        = 1,
  cDgsPoolingTypeCls         = 2,
  cDgsPoolingTypeLast        = 3,
  cDgsPoolingTypeRank        = 4,  // used by reranking models to attach the classification head to the graph
};

enum dgs_attention_type {
  cDgsAttentionTypeUnspecified = -1,
  cDgsAttentionTypeCausal      = 0,
  cDgsAttentionTypeNonCausal   = 1,
};

enum dgs_flash_attn_type {
  cDgsFlashAttnTypeAuto     = -1,
  cDgsFlashAttnTypeDisabled = 0,
  cDgsFlashAttnTypeEnabled  = 1,
};

DGS_API const char * fDgsFlashAttnTypeName(enum dgs_flash_attn_type pFlashAttnType);

enum dgs_split_mode {
  cDgsSplitModeNone   = 0,  // single GPU
  cDgsSplitModeLayer  = 1,  // split layers and KV across GPUs
  cDgsSplitModeRow    = 2,  // split layers and KV across GPUs, use tensor parallelism if supported
  cDgsSplitModeTensor = 3,
};

enum dgs_load_mode {
  cDgsLoadModeAuto      = -1,  // auto-detect based on device capabilities
  cDgsLoadModeNone      = 0,   // no special loading mode
  cDgsLoadModeMmap      = 1,   // memory map the model
  cDgsLoadModeMlock     = 2,   // force system to keep model in RAM rather than swapping or compressing
  cDgsLoadModeMmapMlock = 3,   // mmap + force system to keep model in RAM rather than swapping or compressing
  cDgsLoadModeDirectIo  = 4,   // use direct I/O if available
};

DGS_API const char *       fDgsLoadModeName(enum dgs_load_mode pLoadMode);
DGS_API enum dgs_load_mode fDgsLoadModeFromStr(const char * str);

enum dgs_lazy_mode {
  cDgsLazyModeOff  = 0,  // always read the whole tensor up front
  cDgsLazyModeAuto = 1,  // lazy only for marked tensors larger than 4 GiB (requires mmap)
  cDgsLazyModeOn   = 2,  // read the rows of tensors marked by the arch on demand (requires mmap)
};

enum dgs_context_type {
  cDgsContextTypeDefault = 0,
  cDgsContextTypeMtp     = 1,
};

// TODO: simplify
typedef struct dgs_token_data {
  dgs_token vId;     // token id
  float     vLogit;  // log-odds of the token
  float     vP;      // probability of the token
} dgs_token_data;

typedef struct dgs_token_data_array {
  // TODO: consider SoA
  // NOTE: this pointer can be modified by the samplers
  dgs_token_data * data;
  size_t           size;
  int64_t          vSelected;  // this is the index in the data array (i.e. not the token id)
  bool             vSorted;    // note: do not assume the data is sorted - always check this flag
} dgs_token_data_array;

typedef bool (*dgs_progress_callback)(float pProgress, void * pUserData);

// Input data for dgs_encode/dgs_decode
// A dgs_batch object can contain input about one or many sequences
// The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
//
// - token  : the token ids of the input (used when embd is NULL)
// - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
// - pos    : the positions of the respective token in the sequence
//            (if set to NULL, the token position will be tracked automatically by dgs_encode/dgs_decode)
// - seq_id : the sequence to which the respective token belongs
//            (if set to NULL, the sequence ID will be assumed to be 0)
// - logits : if zero, the logits (and/or the embeddings) for the respective token will not be output
//            (if set to NULL:
//               - if embeddings: all tokens are output
//               - if not:        only the last token is output
//            )
//
typedef struct dgs_batch {
  int32_t vNTokens;

  dgs_token *   token;
  float *       vEmbd;
  dgs_pos *     vPos;
  int32_t *     vNSeqId;
  dgs_seq_id ** vSeqId;
  int8_t *      vLogits;  // TODO: rename this to "output"
} dgs_batch;

enum dgs_model_kv_override_type {
  cDgsKvOverrideTypeInt,
  cDgsKvOverrideTypeFloat,
  cDgsKvOverrideTypeBool,
  cDgsKvOverrideTypeStr,
};

enum dgs_model_meta_key {
  cDgsModelMetaKeySamplingSequence,
  cDgsModelMetaKeySamplingTopK,
  cDgsModelMetaKeySamplingTopP,
  cDgsModelMetaKeySamplingMinP,
  cDgsModelMetaKeySamplingXtcProbability,
  cDgsModelMetaKeySamplingXtcThreshold,
  cDgsModelMetaKeySamplingTemp,
  cDgsModelMetaKeySamplingPenaltyLastN,
  cDgsModelMetaKeySamplingPenaltyRepeat,
  cDgsModelMetaKeySamplingMirostat,
  cDgsModelMetaKeySamplingMirostatTau,
  cDgsModelMetaKeySamplingMirostatEta,
};

enum dgs_process_type {
  cDgsProcessTypeEncode,
  cDgsProcessTypeDecode,
};

struct dgs_model_kv_override {
  enum dgs_model_kv_override_type vTag;

  char aKey[128];

  union {
    int64_t vValI64;
    double  vValF64;
    bool    vValBool;
    char    aValStr[128];
  };
};

struct dgs_model_tensor_buft_override {
  const char *               vPattern;
  ggml_backend_buffer_type_t vBuft;
};

struct dgs_model_params {
  // NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
  ggml_backend_dev_t * vDevices;

  // NULL-terminated list of buffer types to use for tensors that match a pattern
  const struct dgs_model_tensor_buft_override * vTensorBuftOverrides;

  int32_t             vNGpuLayers;  // number of layers to store in VRAM, a negative value means all layers
  enum dgs_split_mode vSplitMode;   // how to split the model across multiple GPUs
  enum dgs_load_mode  vLoadMode;    // how to load the model

  enum dgs_lazy_mode vLazyMode;     // on-demand reading of tensors marked by the arch

  // the GPU that is used for the entire model when split_mode is DGS_SPLIT_MODE_NONE
  int32_t vMainGpu;

  // proportion of the model (layers or rows) to offload to each GPU, size: dgs_max_devices()
  const float * vTensorSplit;

  // Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
  // If the provided progress_callback returns true, model loading continues.
  // If it returns false, model loading is immediately aborted.
  dgs_progress_callback vProgressCallback;

  // context pointer passed to the progress callback
  void * vProgressCallbackUserData;

  // override key-value pairs of the model meta data
  const struct dgs_model_kv_override * vKvOverrides;

  // Keep the booleans together to avoid misalignment during copy-by-value.
  bool vVocabOnly;      // only load the vocabulary, no weights
  bool vCheckTensors;   // validate model tensor data
  bool vUseExtraBufts;  // use extra buffer types (used for weight repacking)
  bool vNoHost;         // bypass host buffer allowing extra buffers to be used
  bool vNoAlloc;        // only load metadata and simulate memory allocations
  bool vLoadMtp;        // whether to load MTP layers
};

struct dgs_sampler_seq_config {
  dgs_seq_id           vSeqId;
  struct dgs_sampler * vSampler;
};

// NOTE: changing the default values of parameters marked as [EXPERIMENTAL] may cause crashes or incorrect results in certain configurations
struct dgs_context_params {
  uint32_t vNCtx;         // text context, 0 = from model
  uint32_t vNBatch;       // logical maximum batch size that can be submitted to dgs_decode
  uint32_t vNUbatch;      // physical maximum batch size
  uint32_t vNSeqMax;      // max number of sequences (i.e. distinct states for recurrent models)
  uint32_t vNRsSeq;       // number of recurrent-state snapshots per seq for rollback (0 = no rollback) [EXPERIMENTAL]
  uint32_t vNOutputsMax;  // max outputs in a ubatch (0 = n_batch)
  uint32_t vNOutputsMaxPerSeq;                  // max outputs per sequence (0 = n_outputs_max)
  int32_t  vNThreads;                           // number of threads to use for generation
  int32_t  vNThreadsBatch;                      // number of threads to use for batch processing

  enum dgs_context_type      vCtxType;          // set the context type (e.g. MTP)
  enum dgs_rope_scaling_type vRopeScalingType;  // RoPE scaling type, from `enum dgs_rope_scaling_type`
  enum fDgsPoolingType       vPoolingType;      // whether to pool (sum) embedding results by sequence id
  enum dgs_attention_type    vAttentionType;    // attention type to use for embeddings
  enum dgs_flash_attn_type   vFlashAttnType;    // when to enable Flash Attention

  float    vRopeFreqBase;    // RoPE base frequency, 0 = from model
  float    vRopeFreqScale;   // RoPE frequency scaling factor, 0 = from model
  float    vYarnExtFactor;   // YaRN extrapolation mix factor, negative = from model
  float    vYarnAttnFactor;  // YaRN magnitude scaling factor
  float    vYarnBetaFast;    // YaRN low correction dim
  float    vYarnBetaSlow;    // YaRN high correction dim
  uint32_t vYarnOrigCtx;     // YaRN original context size

  ggml_backend_sched_eval_callback vCbEval;
  void *                           vCbEvalUserData;

  enum ggml_type vTypeK;  // data type for K cache [EXPERIMENTAL]
  enum ggml_type vTypeV;  // data type for V cache [EXPERIMENTAL]

  // Abort callback
  // if it returns true, execution of dgs_decode() will be aborted
  // currently works only with CPU execution
  ggml_abort_callback vAbortCallback;
  void *              vAbortCallbackData;

  // Keep the booleans together and at the end of the struct to avoid misalignment during copy-by-value.
  bool vEmbeddings;  // if true, extract embeddings (together with logits)
  bool vOffloadKqv;  // offload the KQV ops (including the KV cache) to GPU
  bool vNoPerf;      // measure performance timings
  bool vOpOffload;   // offload host tensor operations to device
  bool vSwaFull;  // use full-size SWA cache (doc/MANUAL.md)
                  // NOTE: setting to false when n_seq_max > 1 can cause bad performance in some cases
  bool vKvUnified;  // use a unified buffer across the input sequences when computing the attention
  // try to disable when n_seq_max > 1 for improved performance when the sequences do not share a large prefix

  // [EXPERIMENTAL]
  // backend sampler chain configuration (make sure the caller keeps the sampler chains alive)
  // note: the samplers must be sampler chains (i.e. use dgs_sampler_chain_init)
  struct dgs_sampler_seq_config * vSamplers;
  size_t                          vNSamplers;

  // a source/target/parent context
  // can be utilized in various ways, for example by sharing results or dgs_memory between 2 contexts
  struct dgs_context * vCtxOther;
};

struct dgs_model_tensor_override {
  const char *   vPattern;
  enum ggml_type type;
};

struct dgs_model_imatrix_data {
  const char *  name;
  const float * data;
  size_t        size;
};

// model quantization parameters

typedef struct dgs_logit_bias {
  dgs_token token;
  float     vBias;
} dgs_logit_bias;

typedef struct dgs_sampler_chain_params {
  bool vNoPerf;  // whether to measure performance timings
} dgs_sampler_chain_params;

// lora adapter
struct dgs_adapter_lora;

DGS_API const char * fDgsVersion(void);

// Helpers for getting default parameters
// TODO: update API to start accepting pointers to params structs
DGS_API struct dgs_model_params         fDgsModelDefaultParams(void);
DGS_API struct dgs_context_params       fDgsContextDefaultParams(void);
DGS_API struct dgs_sampler_chain_params fDgsSamplerChainDefaultParams(void);

// Initialize the dgs + ggml backend
// Call once at the start of the program
DGS_API void fDgsBackendInit(void);

// Call once at the end of the program - currently only used for MPI
DGS_API void fDgsBackendFree(void);

// Optional: enable numa optimizations
// TODO: deprecate and make part of dgs_backend_init()
DGS_API void fDgsNumaInit(enum ggml_numa_strategy pNuma);

// Optional: an auto threadpool gets created in ggml if not passed explicitly
DGS_API void fDgsAttachThreadpool(struct dgs_context * pCtx,
                                  ggml_threadpool_t    pThreadpool,
                                  ggml_threadpool_t    pThreadpoolBatch);

DGS_API void fDgsDetachThreadpool(struct dgs_context * pCtx);

typedef void (*dgs_model_set_tensor_data_t)(struct ggml_tensor * pTensor, void * pUserdata);

// Create a new model from GGUF metadata as well as a function to set the tensor data
//   - tensors are created as GGML_TYPE_F32 by default,
//     override by adding a tensor with the same name but a different name to the context
DGS_API struct dgs_model * fDgsModelInitFromUser(
  struct gguf_context *       pMetadata,
  dgs_model_set_tensor_data_t pSetTensorData,    // function to initialize tensor data with
  void *                      pSetTensorDataUd,  // userdata for function
  struct dgs_model_params     pParams);

// Load a model from a file
// If the file is split into multiple parts, the file name must follow this pattern: <name>-%05d-of-%05d.gguf
// If the split file name does not follow this pattern, use dgs_model_load_from_splits
DGS_API struct dgs_model * fDgsModelLoadFromFile(const char * pPathModel, struct dgs_model_params pParams);

// Load a model from an open FILE pointer
// The GGUF is read from the current position, so it can be embedded in a larger file
// mmap needs the GGUF data section at a file offset to be aligned to the CPU tensor alignment (32 bytes)
DGS_API struct dgs_model * fDgsModelLoadFromFilePtr(FILE * pFile, struct dgs_model_params pParams);

// Load a model from multiple splits (support custom naming scheme)
// The paths must be in the correct order
DGS_API struct dgs_model * fDgsModelLoadFromSplits(const char **           pPaths,
                                                   size_t                  pNPaths,
                                                   struct dgs_model_params pParams);

DGS_API void fDgsModelFree(struct dgs_model * pModel);

DGS_API struct dgs_context * fDgsInitFromModel(struct dgs_model * pModel, struct dgs_context_params pParams);

// Frees all allocated memory
DGS_API void fDgsFree(struct dgs_context * pCtx);

DGS_API int64_t fDgsTimeUs(void);

DGS_API size_t fDgsMaxDevices(void);
DGS_API size_t fDgsMaxParallelSequences(void);
DGS_API size_t fDgsMaxTensorBuftOverrides(void);

DGS_API bool fDgsSupportsMmap(void);
DGS_API bool fDgsSupportsMlock(void);
DGS_API bool fDgsSupportsGpuOffload(void);
DGS_API bool fDgsSupportsRpc(void);

// NOTE: After creating a dgs_context, it is recommended to query the actual values using these functions
//       In some cases the requested values via dgs_context_params may differ from the actual values used by the context
DGS_API uint32_t fDgsNCtx(const struct dgs_context * pCtx);
DGS_API uint32_t fDgsNCtxSeq(const struct dgs_context * pCtx);
DGS_API uint32_t fDgsNBatch(const struct dgs_context * pCtx);
DGS_API uint32_t fDgsNUbatch(const struct dgs_context * pCtx);
DGS_API uint32_t fDgsNSeqMax(const struct dgs_context * pCtx);
DGS_API uint32_t fDgsNRsSeq(const struct dgs_context * pCtx);

DGS_API const struct dgs_model * fDgsGetModel(const struct dgs_context * pCtx);
DGS_API dgs_memory_t             fDgsGetMemory(const struct dgs_context * pCtx);
DGS_API enum fDgsPoolingType fDgsPoolingType(const struct dgs_context * pCtx);  // TODO: rename to dgs_get_pooling_type

DGS_API const struct dgs_vocab * fDgsModelGetVocab(const struct dgs_model * pModel);
DGS_API enum dgs_rope_type       fDgsModelRopeType(const struct dgs_model * pModel);

DGS_API int32_t fDgsModelNCtxTrain(const struct dgs_model * pModel);
DGS_API int32_t fDgsModelNEmbd(const struct dgs_model * pModel);
DGS_API int32_t fDgsModelNEmbdInp(const struct dgs_model * pModel);
DGS_API int32_t fDgsModelNEmbdOut(const struct dgs_model * pModel);
DGS_API int32_t fDgsModelNLayer(const struct dgs_model * pModel);
DGS_API int32_t fDgsModelNLayerNextn(const struct dgs_model * pModel);
DGS_API int32_t fDgsModelNHead(const struct dgs_model * pModel);
DGS_API int32_t fDgsModelNHeadKv(const struct dgs_model * pModel);
DGS_API int32_t fDgsModelNSwa(const struct dgs_model * pModel);

// Get the model's RoPE frequency scaling factor
DGS_API float fDgsModelRopeFreqScaleTrain(const struct dgs_model * pModel);

// Returns the number of classifier outputs (only valid for classifier models)
// Undefined behavior for non-classifier models
DGS_API uint32_t fDgsModelNClsOut(const struct dgs_model * pModel);

// Returns label of classifier output by index (<n_cls_out). Returns nullptr if no label provided
DGS_API const char * fDgsModelClsLabel(const struct dgs_model * pModel, uint32_t pI);

DGS_API enum fDgsVocabType fDgsVocabType(const struct dgs_vocab * pVocab);

DGS_API int32_t fDgsVocabNTokens(const struct dgs_vocab * pVocab);

// Functions to access the model's GGUF metadata scalar values
// - The functions return the length of the string on success, or -1 on failure
// - The output string is always null-terminated and cleared on failure
// - When retrieving a string, an extra byte must be allocated to account for the null terminator
// - GGUF array values are not supported by these functions

// Get metadata value as a string by key name
DGS_API int32_t fDgsModelMetaValStr(const struct dgs_model * pModel, const char * pKey, char * pBuf, size_t pBufSize);

// Get the number of metadata key/value pairs
DGS_API int32_t fDgsModelMetaCount(const struct dgs_model * pModel);

// Get sampling metadata key name. Returns nullptr if the key is invalid
DGS_API const char * fDgsModelMetaKeyStr(enum dgs_model_meta_key pKey);

// Get metadata key name by index
DGS_API int32_t fDgsModelMetaKeyByIndex(const struct dgs_model * pModel, int32_t pI, char * pBuf, size_t pBufSize);

// Get metadata value as a string by index
DGS_API int32_t fDgsModelMetaValStrByIndex(const struct dgs_model * pModel, int32_t pI, char * pBuf, size_t pBufSize);

// Get a string describing the model type
DGS_API int32_t fDgsModelDesc(const struct dgs_model * pModel, char * pBuf, size_t pBufSize);

// Get the model file type (quantization), e.g. DGS_FTYPE_MOSTLY_Q8_0
DGS_API enum dgs_ftype fDgsModelFtype(const struct dgs_model * pModel);

// Returns the total size of all the tensors in the model in bytes
DGS_API uint64_t fDgsModelSize(const struct dgs_model * pModel);

// Get the default chat template. Returns nullptr if not available
// If name is NULL, returns the default chat template
DGS_API const char * fDgsModelChatTemplate(const struct dgs_model * pModel, const char * name);

// Returns the total number of parameters in the model
DGS_API uint64_t fDgsModelNParams(const struct dgs_model * pModel);

// Returns true if the model contains an encoder that requires dgs_encode() call
DGS_API bool fDgsModelHasEncoder(const struct dgs_model * pModel);

// Returns true if the model contains a decoder that requires dgs_decode() call
DGS_API bool fDgsModelHasDecoder(const struct dgs_model * pModel);

// For encoder-decoder models, this function returns id of the token that must be provided
// to the decoder to start generating output sequence. For other models, it returns -1.
DGS_API dgs_token fDgsModelDecoderStartToken(const struct dgs_model * pModel);

// Returns true if the model is recurrent (like Mamba, RWKV, etc.)
DGS_API bool fDgsModelIsRecurrent(const struct dgs_model * pModel);

// Returns true if the model is hybrid (like Jamba, Granite, etc.)
DGS_API bool fDgsModelIsHybrid(const struct dgs_model * pModel);

// Returns true if the model is diffusion-based (like LLaDA, Dream, etc.)
DGS_API bool fDgsModelIsDiffusion(const struct dgs_model * pModel);

// Returns 0 on success

//
// Adapters
//

// Load a LoRA adapter from file
// The adapter is valid as long as the associated model is not freed
DGS_API struct dgs_adapter_lora * fDgsAdapterLoraInit(struct dgs_model * pModel, const char * pPathLora);

// Load a LoRA adapter from an open FILE pointer, reading from its current position
DGS_API struct dgs_adapter_lora * fDgsAdapterLoraInitFromFilePtr(struct dgs_model * pModel, FILE * pFile);

// Functions to access the adapter's GGUF metadata scalar values
// - The functions return the length of the string on success, or -1 on failure
// - The output string is always null-terminated and cleared on failure
// - When retrieving a string, an extra byte must be allocated to account for the null terminator
// - GGUF array values are not supported by these functions

// Get metadata value as a string by key name
DGS_API int32_t fDgsAdapterMetaValStr(const struct dgs_adapter_lora * pAdapter,
                                      const char *                    pKey,
                                      char *                          pBuf,
                                      size_t                          pBufSize);

// Get the number of metadata key/value pairs
DGS_API int32_t fDgsAdapterMetaCount(const struct dgs_adapter_lora * pAdapter);

// Get metadata key name by index
DGS_API int32_t fDgsAdapterMetaKeyByIndex(const struct dgs_adapter_lora * pAdapter,
                                          int32_t                         pI,
                                          char *                          pBuf,
                                          size_t                          pBufSize);

// Get metadata value as a string by index
DGS_API int32_t fDgsAdapterMetaValStrByIndex(const struct dgs_adapter_lora * pAdapter,
                                             int32_t                         pI,
                                             char *                          pBuf,
                                             size_t                          pBufSize);

// Manually free a LoRA adapter
// NOTE: loaded adapters that are not manually freed will be freed when the associated model is deleted
DGS_API void fDgsAdapterLoraFree(struct dgs_adapter_lora * pAdapter);

// Get the invocation tokens if the current lora is an alora
DGS_API uint64_t          fDgsAdapterGetAloraNInvocationTokens(const struct dgs_adapter_lora * pAdapter);
DGS_API const dgs_token * fDgsAdapterGetAloraInvocationTokens(const struct dgs_adapter_lora * pAdapter);

// The following functions operate on a dgs_context, hence the naming: dgs_verb_...

// Set LoRa adapters on the context. Will only modify if the adapters currently in context are different.
DGS_API int32_t fDgsSetAdaptersLora(struct dgs_context *       pCtx,
                                    struct dgs_adapter_lora ** pAdapters,
                                    size_t                     pNAdapters,
                                    float *                    pScales);

// Apply a loaded control vector to a dgs_context, or if data is NULL, clear
// the currently loaded vector.
// n_embd should be the size of a single layer's control, and data should point
// to an n_embd x n_layers buffer starting from layer 1.
// il_start and il_end are the layer range the vector should apply to (both inclusive)
// See dgs_control_vector_load in common to load a control vector.
DGS_API int32_t fDgsSetAdapterCvec(struct dgs_context * pCtx,
                                   const float *        data,
                                   size_t               pLen,
                                   int32_t              pNEmbd,
                                   int32_t              pIlStart,
                                   int32_t              pIlEnd);

//
// Memory
//

// Clear the memory contents
// If data == true, the data buffers will also be cleared together with the metadata
DGS_API void fDgsMemoryClear(dgs_memory_t pMem, bool data);

// Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
// Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails
// seq_id < 0 : match any sequence [TAG_DGS_SEQ_ID_NEG]
// p0 < 0     : [0,  p1]
// p1 < 0     : [p0, inf)
DGS_API bool fDgsMemorySeqRm(dgs_memory_t pMem, dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1);

// Copy all tokens that belong to the specified sequence to another sequence
// p0 < 0 : [0,  p1]
// p1 < 0 : [p0, inf)
DGS_API void fDgsMemorySeqCp(dgs_memory_t pMem, dgs_seq_id pSeqIdSrc, dgs_seq_id pSeqIdDst, dgs_pos pP0, dgs_pos pP1);

// Removes all tokens that do not belong to the specified sequence
DGS_API void fDgsMemorySeqKeep(dgs_memory_t pMem, dgs_seq_id pSeqId);

// Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
// p0 < 0 : [0,  p1]
// p1 < 0 : [p0, inf)
DGS_API void fDgsMemorySeqAdd(dgs_memory_t pMem, dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1, dgs_pos pDelta);

// Integer division of the positions by factor of `d > 1`
// p0 < 0 : [0,  p1]
// p1 < 0 : [p0, inf)
DGS_API void fDgsMemorySeqDiv(dgs_memory_t pMem, dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1, int pD);

// Returns the smallest position present in the memory for the specified sequence
// This is typically non-zero only for SWA caches
// Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
// Return -1 if the sequence is empty
DGS_API dgs_pos fDgsMemorySeqPosMin(dgs_memory_t pMem, dgs_seq_id pSeqId);

// Returns the largest position present in the memory for the specified sequence
// Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
// Return -1 if the sequence is empty
DGS_API dgs_pos fDgsMemorySeqPosMax(dgs_memory_t pMem, dgs_seq_id pSeqId);

// Check if the memory supports shifting
DGS_API bool fDgsMemoryCanShift(dgs_memory_t pMem);

//
// State / sessions
//

// Returns the *actual* size in bytes of the state
// (logits, embedding and memory)
// Only use when saving the state, not when restoring it, otherwise the size may be too small.
DGS_API size_t fDgsStateGetSize(struct dgs_context * pCtx);

// Copies the state to the specified destination address.
// Destination needs to have allocated enough memory.
// Returns the number of bytes copied
DGS_API size_t fDgsStateGetData(struct dgs_context * pCtx, uint8_t * pDst, size_t size);

// Set the state reading from the specified address
// Returns the number of bytes read
DGS_API size_t fDgsStateSetData(struct dgs_context * pCtx, const uint8_t * pSrc, size_t size);

// Save/load session file
DGS_API bool fDgsStateLoadFile(struct dgs_context * pCtx,
                               const char *         pPathSession,
                               dgs_token *          pTokensOut,
                               size_t               pNTokenCapacity,
                               size_t *             pNTokenCountOut);

DGS_API bool fDgsStateSaveFile(struct dgs_context * pCtx,
                               const char *         pPathSession,
                               const dgs_token *    pTokens,
                               size_t               pNTokenCount);

// Get the exact size needed to copy the state of a single sequence
DGS_API size_t fDgsStateSeqGetSize(struct dgs_context * pCtx, dgs_seq_id pSeqId);

// Copy the state of a single sequence into the specified buffer
DGS_API size_t fDgsStateSeqGetData(struct dgs_context * pCtx, uint8_t * pDst, size_t size, dgs_seq_id pSeqId);

// Copy the sequence data (originally copied with `dgs_state_seq_get_data`) into the specified sequence
// Returns:
//  - Positive: Ok
//  - Zero: Failed to load
DGS_API size_t fDgsStateSeqSetData(struct dgs_context * pCtx, const uint8_t * pSrc, size_t size, dgs_seq_id pDestSeqId);

DGS_API size_t fDgsStateSeqSaveFile(struct dgs_context * pCtx,
                                    const char *         pFilepath,
                                    dgs_seq_id           pSeqId,
                                    const dgs_token *    pTokens,
                                    size_t               pNTokenCount);

// If tokens_out is NULL, only the token count is reported through n_token_count_out and no state is loaded
DGS_API size_t fDgsStateSeqLoadFile(struct dgs_context * pCtx,
                                    const char *         pFilepath,
                                    dgs_seq_id           pDestSeqId,
                                    dgs_token *          pTokensOut,
                                    size_t               pNTokenCapacity,
                                    size_t *             pNTokenCountOut);

#define DGS_STATE_SEQ_FLAGS_NONE 0

// work only with partial states, such as SWA KV cache or recurrent cache (e.g. Mamba)
#define DGS_STATE_SEQ_FLAGS_PARTIAL_ONLY 1

// Keeps the tensor data on device buffers (i.e. not accessible in host memory, but faster save/load).
// Getting the state for a seq_id with this flag invalidates all prior states gotten for that seq_id with this flag.
#define DGS_STATE_SEQ_FLAGS_ON_DEVICE 2

typedef uint32_t dgs_state_seq_flags;

DGS_API size_t fDgsStateSeqGetSizeExt(struct dgs_context * pCtx, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags);

DGS_API size_t fDgsStateSeqGetDataExt(struct dgs_context * pCtx,
                                      uint8_t *            pDst,
                                      size_t               size,
                                      dgs_seq_id           pSeqId,
                                      dgs_state_seq_flags  pFlags);

DGS_API size_t fDgsStateSeqSetDataExt(struct dgs_context * pCtx,
                                      const uint8_t *      pSrc,
                                      size_t               size,
                                      dgs_seq_id           pDestSeqId,
                                      dgs_state_seq_flags  pFlags);

//
// Decoding
//

// Return batch for single sequence of tokens
// The sequence ID will be fixed to 0
// The position of the tokens will be tracked automatically by dgs_decode
//
// NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it
//
DGS_API struct dgs_batch fDgsBatchGetOne(dgs_token * pTokens, int32_t pNTokens);

// Allocates a batch of tokens on the heap that can hold a maximum of n_tokens
// Each token can be assigned up to n_seq_max sequence ids
// The batch has to be freed with dgs_batch_free()
// If embd != 0, dgs_batch.embd will be allocated with size of n_tokens * embd * sizeof(float)
// Otherwise, dgs_batch.token will be allocated to store n_tokens dgs_token
// The rest of the dgs_batch members are allocated with size n_tokens
// All members are left uninitialized
DGS_API struct dgs_batch fDgsBatchInit(int32_t pNTokens, int32_t pEmbd, int32_t pNSeqMax);

// Frees a batch of tokens allocated with dgs_batch_init()
DGS_API void fDgsBatchFree(struct dgs_batch pBatch);

// Process a batch of tokens.
// In contrast to dgs_decode() - this call does not use KV cache.
// For encode-decoder contexts, processes the batch using the encoder.
// Can store the encoder output internally for later use by the decoder's cross-attention layers.
//   0 - success
// < 0 - error. the memory state is restored to the state before this call
DGS_API int32_t fDgsEncode(struct dgs_context * pCtx, struct dgs_batch pBatch);

// Process a batch of tokens.
// Requires the context to have a memory.
// For encode-decoder contexts, processes the batch using the decoder.
// Positive return values does not mean a fatal error, but rather a warning.
// Upon fatal-error or abort, the ubatches that managed to be been processed will remain in the memory state of the context
//   To handle this correctly, query the memory state using dgs_memory_seq_pos_min() and dgs_memory_seq_pos_max()
// Upon other return values, the memory state is restored to the state before this call
//    0 - success
//    1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
//    2 - aborted     (processed ubatches will remain in the context's memory)
//   -1 - invalid input batch
// < -1 - fatal error (processed ubatches will remain in the context's memory)
DGS_API int32_t fDgsDecode(struct dgs_context * pCtx, struct dgs_batch pBatch);

//
// Extended batch API
//

struct dgs_batch_ext;

struct dgs_embd {
  const float * data;
  size_t        vNRows;  // number of embedding rows in data
  size_t        vNEmbd;  // size of one row
};

DGS_API struct dgs_batch_ext * fDgsBatchExtInit(struct dgs_context * pCtx);
DGS_API void                   fDgsBatchExtFree(struct dgs_batch_ext * pBatch);
DGS_API void                   fDgsBatchExtClear(struct dgs_batch_ext * pBatch);

// Add an input token to the batch, with default values:
//     id = DGS_TOKEN_NULL
//     embd = nullptr
//     pos = not set, the caller must set it with dgs_batch_ext_set_pos()
// Returns the batch index (>= 0)
// On error:
//     -1: batch is full
//     -2: token is invalid (id == DGS_TOKEN_NULL or invalid embd)
//     -3: invalid sequence id
DGS_API int32_t fDgsBatchExtAdd(struct dgs_batch_ext * pBatch, dgs_seq_id pSeqId);

// Add an input token to the batch, with a specified token ID or token embedding
DGS_API int32_t fDgsBatchExtAddToken(struct dgs_batch_ext * pBatch, dgs_seq_id pSeqId, dgs_token pId);
DGS_API int32_t fDgsBatchExtAddEmbd(struct dgs_batch_ext * pBatch, dgs_seq_id pSeqId, struct dgs_embd pEmbd);

// Add the token at index idx in the batch to another sequence id. The position will stays the same.
// Note: this should be called before other _set() functions
DGS_API bool fDgsBatchExtAddSeq(struct dgs_batch_ext * pBatch, int32_t pIdx, dgs_seq_id pSeqId);

// Set the token embedding for the token at index idx in the batch
// use it after dgs_batch_ext_add_token() to have an entry with both a token id and an embedding
DGS_API bool fDgsBatchExtSetEmbdToken(struct dgs_batch_ext * pBatch, int32_t pIdx, struct dgs_embd pEmbd);

// Set the "state" embedding for the token at index idx in the batch
// "state" here means extra hidden state carried over from a previous stage, e.g.:
//   - MTP: state from N layers of the target model
//   - Qwen3 VL (deepstack): state from N layers of the vision encoder
// Returns false if the context does not take a state embedding (currently only MTP contexts do)
DGS_API bool fDgsBatchExtSetEmbdState(struct dgs_batch_ext * pBatch, int32_t pIdx, struct dgs_embd pEmbd);

// Set if output embeddings should be available for the token at index idx in the batch
// Note: for now, this is equivalent to setting the output logits
DGS_API bool fDgsBatchExtSetOutputEmbd(struct dgs_batch_ext * pBatch, int32_t pIdx, bool value);

// Set output logits for the token at index idx in the batch
// Note: for now, this is equivalent to setting the output embd
DGS_API bool fDgsBatchExtSetOutputLogits(struct dgs_batch_ext * pBatch, int32_t pIdx, bool value);

// Set custom position for the token at index idx in the batch
// For M-RoPE models:
//     - Embedding tokens must have n_pos_per_embd positions per token, in order [t, y, x, z]; t is also the KV cache position
//     - Text token only requires one single position per token
DGS_API bool fDgsBatchExtSetPos(struct dgs_batch_ext * pBatch, int32_t pIdx, const dgs_pos * pPos);

// TODO: implement get_embeddings() and get_logits() for dgs_batch_ext

// Return values are the same as dgs_decode()
DGS_API int32_t fDgsProcess(struct dgs_context * pCtx, enum dgs_process_type type, struct dgs_batch_ext * pBatch);

// Set the number of threads used for decoding
// n_threads is the number of threads used for generation (single token)
// n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)
DGS_API void fDgsSetNThreads(struct dgs_context * pCtx, int32_t pNThreads, int32_t pNThreadsBatch);

// Get the number of threads used for generation of a single token.
DGS_API int32_t fDgsNThreads(struct dgs_context * pCtx);

// Get the number of threads used for prompt and batch processing (multiple token).
DGS_API int32_t fDgsNThreadsBatch(struct dgs_context * pCtx);

// Set whether the context outputs embeddings or not
// TODO: rename to avoid confusion with dgs_get_embeddings()
DGS_API void fDgsSetEmbeddings(struct dgs_context * pCtx, bool pEmbeddings);

// Set whether to use causal attention or not
// If set to true, the model will only attend to the past tokens
DGS_API void fDgsSetCausalAttn(struct dgs_context * pCtx, bool pCausalAttn);

// Returns whether the context is currently using causal attention
DGS_API bool fDgsGetCausalAttn(const struct dgs_context * pCtx);

// Set whether the model is in warmup mode or not
// If true, all model tensors are activated during dgs_decode() to load and cache their weights.
//
// note: using this can cause extra graph reallocations because it changes the graph topology with MoE models,
//       so it is generally not recommended to use in practice. will be removed in the future

// Set abort callback
DGS_API void fDgsSetAbortCallback(struct dgs_context * pCtx,
                                  ggml_abort_callback  pAbortCallback,
                                  void *               pAbortCallbackData);

// Wait until all computations are finished
// This is automatically done when using one of the functions below to obtain the computation results
// and is not necessary to call it explicitly in most cases
DGS_API void fDgsSynchronize(struct dgs_context * pCtx);

// Token logits obtained from the last call to dgs_decode()
// The logits for which dgs_batch.logits[i] != 0 are stored contiguously
// in the order they have appeared in the batch.
// Rows: number of tokens for which dgs_batch.logits[i] != 0
// Cols: n_vocab
// TODO: deprecate in favor of dgs_get_logits_ith
DGS_API float * fDgsGetLogits(struct dgs_context * pCtx);

// Logits for the ith token. For positive indices, Equivalent to:
// dgs_get_logits(ctx) + ctx->output_ids[i]*n_vocab
// Negative indices can be used to access logits in reverse order, -1 is the last logit.
// returns NULL for invalid ids.
DGS_API float * fDgsGetLogitsIth(struct dgs_context * pCtx, int32_t pI);

// Get all output token embeddings.
// when pooling_type == DGS_POOLING_TYPE_NONE or when using a generative model,
// the embeddings for which dgs_batch.logits[i] != 0 are stored contiguously
// in the order they have appeared in the batch.
// shape: [n_outputs*n_embd]
// Otherwise, returns NULL.
// TODO: deprecate in favor of dgs_get_embeddings_ith
DGS_API float * fDgsGetEmbeddings(struct dgs_context * pCtx);

// Get the embeddings for the ith token. For positive indices, Equivalent to:
// dgs_get_embeddings(ctx) + ctx->output_ids[i]*n_embd
// Negative indices can be used to access embeddings in reverse order, -1 is the last embedding.
// shape: [n_embd] (1-dimensional)
// returns NULL for invalid ids.
DGS_API float * fDgsGetEmbeddingsIth(struct dgs_context * pCtx, int32_t pI);

// Get the embeddings for a sequence id
// Returns NULL if pooling_type is DGS_POOLING_TYPE_NONE
// when pooling_type == DGS_POOLING_TYPE_RANK, returns float[n_cls_out] with the rank(s) of the sequence
// otherwise: float[n_embd] (1-dimensional)
DGS_API float * fDgsGetEmbeddingsSeq(struct dgs_context * pCtx, dgs_seq_id pSeqId);

//
// backend sampling API [EXPERIMENTAL]
// note: use only if the dgs_context was created with at least one dgs_sampler_seq_config
//

// Get the backend sampled token for the ith token.
// With multiple outputs, sampler state advances when the token is accepted,
// not when it is read through this function.
// When accepting multiple outputs, accept a contiguous prefix in output order.
// Returns DGS_TOKEN_NULL if no token was sampled.
DGS_API dgs_token fDgsGetSampledTokenIth(struct dgs_context * pCtx, int32_t pI);

// Get the backend sampled probabilities for the ith token
// The index matches dgs_get_sampled_token_ith().
// Returns NULL if no probabilities were generated.
DGS_API float *  fDgsGetSampledProbsIth(struct dgs_context * pCtx, int32_t pI);
DGS_API uint32_t fDgsGetSampledProbsCountIth(struct dgs_context * pCtx, int32_t pI);

// Get the backend sampled logits for the ith token
// Returns NULL if no logits were sampled.
DGS_API float *  fDgsGetSampledLogitsIth(struct dgs_context * pCtx, int32_t pI);
DGS_API uint32_t fDgsGetSampledLogitsCountIth(struct dgs_context * pCtx, int32_t pI);

// Get the backend sampled candidates (token ids) for the ith token
// These are needed to map probability/logit indices to vocab token ids.
// Returns NULL if no candidates were sampled.
DGS_API dgs_token * fDgsGetSampledCandidatesIth(struct dgs_context * pCtx, int32_t pI);
DGS_API uint32_t    fDgsGetSampledCandidatesCountIth(struct dgs_context * pCtx, int32_t pI);

//
// Vocab
//

DGS_API const char * fDgsVocabGetText(const struct dgs_vocab * pVocab, dgs_token token);

DGS_API float fDgsVocabGetScore(const struct dgs_vocab * pVocab, dgs_token token);

DGS_API enum dgs_token_attr fDgsVocabGetAttr(const struct dgs_vocab * pVocab, dgs_token token);

// Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)
DGS_API bool fDgsVocabIsEog(const struct dgs_vocab * pVocab, dgs_token token);

// Identify if Token Id is a control token or a render-able token
DGS_API bool fDgsVocabIsControl(const struct dgs_vocab * pVocab, dgs_token token);

// Special tokens
DGS_API dgs_token fDgsVocabBos(const struct dgs_vocab * pVocab);   // beginning-of-sentence
DGS_API dgs_token fDgsVocabEos(const struct dgs_vocab * pVocab);   // end-of-sentence
DGS_API dgs_token fDgsVocabEot(const struct dgs_vocab * pVocab);   // end-of-turn
DGS_API dgs_token fDgsVocabSep(const struct dgs_vocab * pVocab);   // sentence separator
DGS_API dgs_token fDgsVocabNl(const struct dgs_vocab * pVocab);    // next-line
DGS_API dgs_token fDgsVocabPad(const struct dgs_vocab * pVocab);   // padding
DGS_API dgs_token fDgsVocabMask(const struct dgs_vocab * pVocab);  // mask

DGS_API bool fDgsVocabGetAddBos(const struct dgs_vocab * pVocab);
DGS_API bool fDgsVocabGetAddEos(const struct dgs_vocab * pVocab);
DGS_API bool fDgsVocabGetAddSep(const struct dgs_vocab * pVocab);

// model-specific suppress tokens (gguf key: tokenizer.ggml.suppress_tokens)
DGS_API const dgs_token * fDgsVocabGetSuppressTokens(const struct dgs_vocab * pVocab, int32_t * pNSuppressTokens);

DGS_API dgs_token fDgsVocabFimPre(const struct dgs_vocab * pVocab);
DGS_API dgs_token fDgsVocabFimSuf(const struct dgs_vocab * pVocab);
DGS_API dgs_token fDgsVocabFimMid(const struct dgs_vocab * pVocab);
DGS_API dgs_token fDgsVocabFimPad(const struct dgs_vocab * pVocab);
DGS_API dgs_token fDgsVocabFimRep(const struct dgs_vocab * pVocab);
DGS_API dgs_token fDgsVocabFimSep(const struct dgs_vocab * pVocab);

// CLS is equivalent to BOS

//
// Tokenization
//
// The API is thread-safe.
//

/// @details Convert the provided text into tokens.
/// @param tokens The tokens pointer must be large enough to hold the resulting tokens.
/// @return Returns the number of tokens on success, no more than n_tokens_max
/// @return Returns a negative number on failure - the number of tokens that would have been returned
/// @return Returns INT32_MIN on overflow (e.g., tokenization result size exceeds int32_t limit)
/// @param add_special Allow to add BOS and EOS tokens if model is configured to do so.
/// @param parse_special Allow tokenizing special and/or control tokens which otherwise are not exposed and treated
///                      as plaintext. Does not insert a leading space.
DGS_API int32_t fDgsTokenize(const struct dgs_vocab * pVocab,
                             const char *             pText,
                             int32_t                  pTextLen,
                             dgs_token *              pTokens,
                             int32_t                  pNTokensMax,
                             bool                     pAddSpecial,
                             bool                     pParseSpecial);

// Token Id -> Piece.
// Uses the vocabulary in the provided context.
// Does not write null terminator to the buffer.
// User can skip up to 'lstrip' leading spaces before copying (useful when encoding/decoding multiple tokens with 'add_space_prefix')
// @param special If true, special tokens are rendered in the output.
DGS_API int32_t fDgsTokenToPiece(const struct dgs_vocab * pVocab,
                                 dgs_token                token,
                                 char *                   pBuf,
                                 int32_t                  length,
                                 int32_t                  pLstrip,
                                 bool                     pSpecial);

/// @details Convert the provided tokens into text (inverse of dgs_tokenize()).
/// @param text The char pointer must be large enough to hold the resulting text.
/// @return Returns the number of chars/bytes on success, no more than text_len_max.
/// @return Returns a negative number on failure - the number of chars/bytes that would have been returned.
/// @param remove_special Allow to remove BOS and EOS tokens if model is configured to do so.
/// @param unparse_special If true, special tokens are rendered in the output.
DGS_API int32_t fDgsDetokenize(const struct dgs_vocab * pVocab,
                               const dgs_token *        pTokens,
                               int32_t                  pNTokens,
                               char *                   pText,
                               int32_t                  pTextLenMax,
                               bool                     pRemoveSpecial,
                               bool                     pUnparseSpecial);

//
// Chat templates
//

/// Apply chat template. Inspired by hf apply_chat_template() on python.
///
/// NOTE: This function does not use a jinja parser. It only support a pre-defined list of template. See more: https://github.com/ggml-org/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
/// @param tmpl A Jinja template to use for this chat.
/// @param add_ass Whether to end the prompt with the token(s) that indicate the start of an assistant message.
/// @param buf A buffer to hold the output formatted prompt. The recommended alloc size is 2 * (total number of characters of all messages)
/// @param length The size of the allocated buffer
/// @return The total number of bytes of the formatted prompt. If is it larger than the size of buffer, you may need to re-alloc it and then re-apply the template.

// Get list of built-in chat templates

//
// Sampling API
//
// Sample usage:
//
//    // prepare the sampling chain at the start
//    auto sparams = dgs_sampler_chain_default_params();
//
//    dgs_sampler * smpl = dgs_sampler_chain_init(sparams);
//
//    dgs_sampler_chain_add(smpl, dgs_sampler_init_top_k(50));
//    dgs_sampler_chain_add(smpl, dgs_sampler_init_top_p(0.9, 1));
//    dgs_sampler_chain_add(smpl, dgs_sampler_init_temp (0.8));
//
//    // typically, the chain should end with a sampler such as "greedy", "dist" or "mirostat"
//    // this sampler will be responsible to select the actual token
//    dgs_sampler_chain_add(smpl, dgs_sampler_init_dist(seed));
//
//    ...
//
//    // decoding loop:
//    while (...) {
//        ...
//
//        dgs_decode(ctx, batch);
//
//        // sample from the logits of the last token in the batch
//        const dgs_token id = dgs_sampler_sample(smpl, ctx, -1);
//
//        ...
//    }
//
//    dgs_sampler_free(smpl);
//

typedef void * dgs_sampler_context_t;

struct dgs_sampler_data {
  struct ggml_tensor * vLogits;
  struct ggml_tensor * vProbs;
  struct ggml_tensor * vSampled;
  struct ggml_tensor * vCandidates;
};

// user code can implement the interface below in order to create custom dgs_sampler
struct dgs_sampler_i {
  const char * (*name)(const struct dgs_sampler * pSmpl);                   // can be NULL
  void (*accept)(struct dgs_sampler * pSmpl, dgs_token token);              // can be NULL
  void (*apply)(struct dgs_sampler * pSmpl, dgs_token_data_array * pCurP);  // required
  void (*reset)(struct dgs_sampler * pSmpl);                                // can be NULL
  struct dgs_sampler * (*clone)(const struct dgs_sampler * pSmpl);          // can be NULL if ctx is NULL
  void (*free)(struct dgs_sampler * pSmpl);                                 // can be NULL if ctx is NULL

  // [EXPERIMENTAL]
  // backend sampling interface:

  // return true if the backend supports all ops needed by the sampler and can handle up to n_outputs_max_per_seq outputs per sequence
  // note: call once per sampler
  bool (*vBackendInit)(struct dgs_sampler * pSmpl, ggml_backend_buffer_type_t pBuft, uint32_t pNOutputsMaxPerSeq);

  // call after .backend_apply()
  void (*vBackendAccept)(struct dgs_sampler *  pSmpl,
                         struct ggml_context * pCtx,
                         struct ggml_cgraph *  pGf,
                         struct ggml_tensor *  pSelectedToken);

  // call after .backend_init()
  void (*vBackendApply)(struct dgs_sampler *      pSmpl,
                        struct ggml_context *     pCtx,
                        struct ggml_cgraph *      pGf,
                        struct dgs_sampler_data * data);

  // called before graph execution to set inputs for the current ubatch
  void (*vBackendSetInput)(struct dgs_sampler * pSmpl);

  // called before rebuilding a sampling graph to clear any internal sampler state
  void (*vBackendReset)(struct dgs_sampler * pSmpl);

  // copy mutable state from src into dst while keeping dst's references to the current sampling graph
  // src and dst must have the same type and configuration
  void (*vCopyState)(const struct dgs_sampler * pSrc, struct dgs_sampler * pDst);
};

struct dgs_sampler {
  struct dgs_sampler_i * vIface;

  dgs_sampler_context_t vCtx;
};

// [EXPERIMENTAL]
// attach a sampler to the context
// note: prefer initializing the context with dgs_context_params.samplers when possible
DGS_API bool fDgsSetSampler(struct dgs_context * pCtx, dgs_seq_id pSeqId, struct dgs_sampler * pSmpl);

// mirror of dgs_sampler_i:
DGS_API struct dgs_sampler * fDgsSamplerInit(struct dgs_sampler_i * pIface, dgs_sampler_context_t pCtx);
DGS_API const char *         fDgsSamplerName(const struct dgs_sampler * pSmpl);
DGS_API void                 fDgsSamplerAccept(struct dgs_sampler * pSmpl, dgs_token token);
DGS_API void                 fDgsSamplerApply(struct dgs_sampler * pSmpl, dgs_token_data_array * pCurP);
DGS_API void                 fDgsSamplerReset(struct dgs_sampler * pSmpl);
DGS_API struct dgs_sampler * fDgsSamplerClone(const struct dgs_sampler * pSmpl);
DGS_API void                 fDgsSamplerCopy(const struct dgs_sampler * pSrc, struct dgs_sampler * pDst);
// important: do not free if the sampler has been added to a dgs_sampler_chain (via dgs_sampler_chain_add)
DGS_API void                 fDgsSamplerFree(struct dgs_sampler * pSmpl);

// dgs_sampler_chain
// a type of dgs_sampler that can chain multiple samplers one after another

DGS_API struct dgs_sampler * fDgsSamplerChainInit(struct dgs_sampler_chain_params pParams);

// important: takes ownership of the sampler object and will free it when dgs_sampler_free is called
DGS_API void fDgsSamplerChainAdd(struct dgs_sampler * pChain, struct dgs_sampler * pSmpl);

// return NULL if:
//   - the sampler is NULL
//   - the sampler is not a dgs_sampler_chain
//   - the index is out of bounds, unless i == -1
//   - if i == -1, returns the chain itself (can be used to check if the sampler is a chain)
DGS_API struct dgs_sampler * fDgsSamplerChainGet(struct dgs_sampler * pChain, int32_t pI);

// the total number of samplers in the chain
DGS_API int32_t fDgsSamplerChainN(const struct dgs_sampler * pChain);

// after removing a sampler, the chain will no longer own it, and it will not be freed when the chain is freed
DGS_API struct dgs_sampler * fDgsSamplerChainRemove(struct dgs_sampler * pChain, int32_t pI);

// available samplers:

DGS_API struct dgs_sampler * fDgsSamplerInitGreedy(void);

/// seed == DGS_DEFAULT_SEED to use a random seed.
DGS_API struct dgs_sampler * fDgsSamplerInitDist(uint32_t pSeed);

/// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
/// Setting k <= 0 makes this a noop
DGS_API struct dgs_sampler * fDgsSamplerInitTopK(int32_t pK);

/// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
DGS_API struct dgs_sampler * fDgsSamplerInitTopP(float pP, size_t pMinKeep);

/// @details Minimum P sampling as described in 
DGS_API struct dgs_sampler * fDgsSamplerInitMinP(float pP, size_t pMinKeep);

/// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
DGS_API struct dgs_sampler * fDgsSamplerInitTypical(float pP, size_t pMinKeep);

/// #details Updates the logits l_i` = l_i/t. When t <= 0.0f, the maximum logit is kept at it's original value, the rest are set to -inf
DGS_API struct dgs_sampler * fDgsSamplerInitTemp(float pT);

/// @details Dynamic temperature implementation (a.k.a. entropy) described in the paper https://arxiv.org/abs/2309.02772.
DGS_API struct dgs_sampler * fDgsSamplerInitTempExt(float pT, float pDelta, float pExponent);

/// @details XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
DGS_API struct dgs_sampler * fDgsSamplerInitXtc(float pP, float pT, size_t pMinKeep, uint32_t pSeed);

/// @details Top n sigma sampling as described in academic paper "Top-nσ: Not All Logits Are You Need" https://arxiv.org/pdf/2411.07641
DGS_API struct dgs_sampler * fDgsSamplerInitTopNSigma(float n);

/// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
/// @param candidates A vector of `dgs_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
/// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
/// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
/// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
/// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
DGS_API struct dgs_sampler * fDgsSamplerInitMirostat(int32_t  pNVocab,
                                                     uint32_t pSeed,
                                                     float    pTau,
                                                     float    pEta,
                                                     int32_t  pM);

/// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
/// @param candidates A vector of `dgs_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
/// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
/// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
/// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
DGS_API struct dgs_sampler * fDgsSamplerInitMirostatV2(uint32_t pSeed, float pTau, float pEta);

/// @details Initializes a GBNF grammar, see grammars/README.md for details.
/// @param vocab The vocabulary that this grammar will be used with.
/// @param grammar_str The production rules for the grammar, encoded as a string. Returns an empty grammar if empty. Returns NULL if parsing of grammar_str fails.
/// @param grammar_root The name of the start symbol for the grammar.
DGS_API struct dgs_sampler * fDgsSamplerInitGrammar(const struct dgs_vocab * pVocab,
                                                    const char *             pGrammarStr,
                                                    const char *             pGrammarRoot);

/// @details Lazy grammar sampler, introduced in 
/// @param trigger_patterns A list of patterns that will trigger the grammar sampler. Pattern will be matched from the start of the generation output, and grammar sampler will be fed content starting from its first match group.
/// @param trigger_tokens A list of tokens that will trigger the grammar sampler. Grammar sampler will be fed content starting from the trigger token included.
DGS_API struct dgs_sampler * fDgsSamplerInitGrammarLazyPatterns(const struct dgs_vocab * pVocab,
                                                                const char *             pGrammarStr,
                                                                const char *             pGrammarRoot,
                                                                const char **            pTriggerPatterns,
                                                                size_t                   pNumTriggerPatterns,
                                                                const dgs_token *        pTriggerTokens,
                                                                size_t                   pNumTriggerTokens);

/// NOTE: Avoid using on the full vocabulary as searching for repeated tokens can become slow. For example, apply top-k or top-p sampling first.
DGS_API struct dgs_sampler * fDgsSamplerInitPenalties(
  int32_t pNVocab,
  int32_t pPenaltyLastN,     // last n tokens to penalize (0 = disable penalty)
  float   pPenaltyRepeat,    // must be > 0.0, 1.0 = disabled
  float   pPenaltyFreq,      // must be finite, 0.0 = disabled
  float   pPenaltyPresent);  // must be finite, 0.0 = disabled

///  @details DRY sampler, designed by p-e-w, as described in: https://github.com/oobabooga/text-generation-webui/pull/5677, porting Koboldcpp implementation authored by pi6am: https://github.com/LostRuins/koboldcpp/pull/982
DGS_API struct dgs_sampler * fDgsSamplerInitDry(
  const struct dgs_vocab * pVocab,
  float                    pDryMultiplier,
  float                    pDryBase,
  int32_t                  pDryAllowedLength,
  int32_t                  pDryPenaltyLastN,  // last n tokens to penalize (0 = disable penalty)
  const char **            pSeqBreakers,
  size_t                   pNumBreakers);

/// adaptive-p: select tokens near a configurable target probability over time.
///
/// the adaptive-p sampler transforms the token probability distribution to favor tokens
/// that fall near a user-configurable probability target.
///
/// internally, the sampler maintains an exponential moving average of the *ORIGINAL*
/// probabilities of selected tokens at each sampling step. it uses this EMA to compute an
/// adapted target probability at each sampling step, thus maintaining the desired target
/// probability over time.
///
/// adaptive-p selects a token ID rather than just mutating candidates, so it must be last
/// in the sampler chain (like mirostat, dist, greedy).
///
/// only mild truncation before this sampler is recommended. we suggest applying min-p
/// before adaptive-p as the only other active sampler in the chain.
///
/// @param target select tokens near this probability (valid range 0.0 to 1.0; negative = disabled)
/// @param decay  EMA decay for adaptation; history ≈ 1/(1-decay) tokens (valid range 0.0 - 0.99)
/// @param seed   RNG seed
///
/// ref: 
///
DGS_API struct dgs_sampler * fDgsSamplerInitAdaptiveP(float pTarget, float pDecay, uint32_t pSeed);

DGS_API struct dgs_sampler * fDgsSamplerInitLogitBias(int32_t                pNVocab,
                                                      int32_t                pNLogitBias,
                                                      const dgs_logit_bias * pLogitBias);

// this sampler is meant to be used for fill-in-the-middle infilling
// it's supposed to be used after top_k + top_p sampling
//
// 1. if the sum of the EOG probs times the number of candidates is higher than the sum of the other probs -> pick EOG
// 2. combine probs of tokens that have the same prefix
//
// example:
//
// - before:
//   "hel":   0.5
//   "hell":  0.2
//   "hello": 0.1
//   "dummy": 0.1
//
// - after:
//   "hel":   0.8
//   "dummy": 0.1
//
// 3. discard non-EOG tokens with low prob
// 4. if no tokens are left -> pick EOT
//
DGS_API struct dgs_sampler * fDgsSamplerInitInfill(const struct dgs_vocab * pVocab);

// Returns the seed used by the sampler if applicable, DGS_DEFAULT_SEED otherwise
DGS_API uint32_t fDgsSamplerGetSeed(const struct dgs_sampler * pSmpl);

/// @details Sample and accept a token from the idx-th output of the last evaluation
// For multiple outputs from one sampler, call this function in output order without gaps.
//
// Shorthand for:
//    const auto * logits = dgs_get_logits_ith(ctx, idx);
//    dgs_token_data_array cur_p = { ... init from logits ... };
//    dgs_sampler_apply(smpl, &cur_p);
//    auto token = cur_p.data[cur_p.selected].id;
//    dgs_sampler_accept(smpl, token);
//    return token;
// Returns the sampled token
DGS_API dgs_token fDgsSamplerSample(struct dgs_sampler * pSmpl, struct dgs_context * pCtx, int32_t pIdx);

// TODO: extend in the future
//DGS_API void dgs_decode_with_sampler(struct dgs_context * ctx, struct dgs_sampler * smpl, struct dgs_batch batch, ...);

//
// Model split
//

/// @details Build a split GGUF final path for this chunk.
///          dgs_split_path(split_path, sizeof(split_path), "/models/ggml-model-q4_0", 2, 4) => split_path = "/models/ggml-model-q4_0-00002-of-00004.gguf"
//  Returns the split_path length.
DGS_API int32_t
fDgsSplitPath(char * pSplitPath, size_t pMaxlen, const char * pPathPrefix, int32_t pSplitNo, int32_t pSplitCount);

/// @details Extract the path prefix from the split_path if and only if the split_no and split_count match.
///          dgs_split_prefix(split_prefix, 64, "/models/ggml-model-q4_0-00002-of-00004.gguf", 2, 4) => split_prefix = "/models/ggml-model-q4_0"
//  Returns the split_prefix length.
DGS_API int32_t
fDgsSplitPrefix(char * pSplitPrefix, size_t pMaxlen, const char * pSplitPath, int32_t pSplitNo, int32_t pSplitCount);

// Print system information
DGS_API const char * fDgsPrintSystemInfo(void);

// Set callback for all future logging events.
// If this is not called, or NULL is supplied, everything is output on stderr.
// The logger state is global so these functions are NOT thread safe.
DGS_API void fDgsLogGet(ggml_log_callback * pLogCallback, void ** pUserData);
DGS_API void fDgsLogSet(ggml_log_callback pLogCallback, void * pUserData);

//
// Performance utils
//
// NOTE: Used by dgs.cpp examples/tools, avoid using in third-party apps. Instead, do your own performance measurements.
//

struct dgs_perf_context_data {
  // ms == milliseconds
  double vTStartMs;        // absolute start time
  double vTLoadMs;         // time needed for loading the model
  double vTPEvalMs;        // time needed for processing the prompt
  double vTEvalMs;         // time needed for generating tokens

  int32_t vNPEval;         // number of prompt tokens
  int32_t vNEval;          // number of generated tokens
  int32_t vNReused;        // number of times a ggml compute graph had been reused
  int32_t vNReallocsSame;  // number of graph reallocations during inference with an unchanged graph size
  int32_t vNReallocsDiff;  // number of graph reallocations during inference with a changed graph size
};

struct dgs_perf_sampler_data {
  double vTSampleMs;  // time needed for sampling in ms

  int32_t vNSample;   // number of sampled tokens
};

DGS_API struct dgs_perf_context_data fDgsPerfContext(const struct dgs_context * pCtx);
DGS_API void                         fDgsPerfContextPrint(const struct dgs_context * pCtx);
DGS_API void                         fDgsPerfContextReset(struct dgs_context * pCtx);

// NOTE: the following work only with samplers constructed via dgs_sampler_chain_init
DGS_API struct dgs_perf_sampler_data fDgsPerfSampler(const struct dgs_sampler * pChain);
DGS_API void                         fDgsPerfSamplerPrint(const struct dgs_sampler * pChain);
DGS_API void                         fDgsPerfSamplerReset(struct dgs_sampler * pChain);

#ifdef __cplusplus
}
#endif

#endif  // DGS_H
