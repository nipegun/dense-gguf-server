#pragma once

#include "dgs-adapter.h"
#include "dgs-cparams.h"
#include "dgs-ext.h"
#include "dgs-graph.h"
#include "dgs-impl.h"
#include "dgs-memory.h"
#include "dgs.h"
#include "ggml-cpp.h"

#include <array>
#include <map>
#include <vector>

struct dgs_model;
class dgs_batch_allocr;

class dgs_io_read_i;
class dgs_io_write_i;

// "memory" as in abstract memory for the context
struct dgs_memory_i;
struct dgs_memory_context_i;

// stores copy of the memory in device buffer. used for fast state save/load
struct dgs_memory_buffer {
  int    vNTensors  = 0;
  size_t vTotalSize = 0;

  ggml_backend_buffer_ptr vBuf;

  ggml_context_ptr vCtx;

  std::vector<ggml_tensor *> lOrg;
  std::vector<ggml_tensor *> lCpy;
};

using dgs_memory_buffers = std::map<ggml_backend_buffer_type_t, dgs_memory_buffer>;

struct dgs_context {
  // init scheduler and compute buffers, reserve worst-case graphs
  dgs_context(const dgs_model & pModel, dgs_context_params pParams);

  ~dgs_context();

  // reserve a new backend scheduler (if needed)
  // for example, when:
  //   - changing loras
  //   - changing samplers
  //   - changing attention type
  //   - etc.
  void mSchedReserve();

  void mSynchronize();

  const dgs_model &   mGetModel() const;
  const dgs_cparams & mGetCparams() const;

  ggml_backend_sched_t mGetSched() const;

  uint32_t mNCtx() const;
  uint32_t mNCtxSeq() const;
  uint32_t mNBatch() const;
  uint32_t mNUbatch() const;
  uint32_t mNSeqMax() const;

  uint32_t mNThreads() const;
  uint32_t mNThreadsBatch() const;

  dgs_memory_t mGetMemory() const;

  // return true if the memory was updated
  bool mMemoryUpdate(bool pOptimize);

  enum fDgsPoolingType mPoolingType() const;

  float * mGetLogits();
  float * mGetLogitsIth(int32_t pI);

  float * mGetEmbeddings();
  float * mGetEmbeddingsIth(int32_t pI);
  float * mGetEmbeddingsSeq(dgs_seq_id pSeqId);

  float * mGetEmbeddingsNextn();
  float * mGetEmbeddingsNextnIth(int32_t pI);

  float * mGetEmbeddingsLayerInp(uint32_t pLid);

  dgs_token * mGetSampledTokens() const;
  dgs_token   mGetSampledTokenIth(int32_t pIdx);

  float * mGetSampledLogitsIth(int32_t pIdx);
  size_t  mGetSampledLogitsCount(int32_t pIdx);

  float * mGetSampledProbsIth(int32_t pIdx);
  size_t  mGetSampledProbsCount(int32_t pIdx);

  const dgs_token * mGetSampledCandidatesIth(int32_t pIdx);
  size_t            mGetSampledCandidatesCount(int32_t pIdx);

  bool mGetCausalAttn() const;

  void mAttachThreadpool(ggml_threadpool_t pThreadpool, ggml_threadpool_t pThreadpoolBatch);

  void mDetachThreadpool();

  void mSetNThreads(int32_t pNThreads, int32_t pNThreadsBatch);

  void mSetAbortCallback(bool (*pAbortCallback)(void * data), void * pAbortCallbackData);

  void mSetEmbeddings(bool value);
  void mSetEmbeddingsNextn(bool value, bool pMasked);
  void mSetEmbeddingsLayerInp(uint32_t pLid, bool pEnable);
  void mSetNextnLayerOffset(int32_t pOffset);
  void mSetCausalAttn(bool value);
  void mSetWarmup(bool value);

  void mSetAdaptersLora(dgs_adapter_lora ** pAdapters, size_t pNAdapters, float * pScales);

  bool mAdaptersLoraAreSame(dgs_adapter_lora ** pAdapters, size_t pNAdapters, float * pScales);

  bool mSetAdapterCvec(const float * data, size_t pLen, int32_t pNEmbd, int32_t pIlStart, int32_t pIlEnd);

  // process a single ubatch with a specific graph type
  // if memory_context is provided, it will be applied first to the context's memory
  // ret contains the status of the graph computation
  // returns nullptr only if ret != GGML_STATUS_SUCCESS
  llm_graph_result * mProcessUbatch(const dgs_ubatch &     pUbatch,
                                    llm_graph_type         pGtype,
                                    dgs_memory_context_i * pMctx,
                                    ggml_status &          pRet);

  int mEncode(const dgs_batch_ext & pBatchInp);
  int mDecode(const dgs_batch_ext & pBatchInp);

  // compat version
  int mEncode(const dgs_batch & pBatchInp);
  int mDecode(const dgs_batch & pBatchInp);

  //
  // state save/load
  //

  size_t mStateGetSize();
  size_t mStateGetData(uint8_t * pDst, size_t size);
  size_t mStateSetData(const uint8_t * pSrc, size_t size);

  size_t mStateSeqGetSize(dgs_seq_id pSeqId, dgs_state_seq_flags pFlags);

  size_t mStateSeqGetData(dgs_seq_id pSeqId, uint8_t * pDst, size_t size, dgs_state_seq_flags pFlags);
  size_t mStateSeqSetData(dgs_seq_id pSeqId, const uint8_t * pSrc, size_t size, dgs_state_seq_flags pFlags);

  bool mStateLoadFile(const char * pFilepath, dgs_token * pTokensOut, size_t pNTokenCapacity, size_t * pNTokenCountOut);

  bool mStateSaveFile(const char * pFilepath, const dgs_token * pTokens, size_t pNTokenCount);

  size_t mStateSeqLoadFile(dgs_seq_id   pSeqId,
                           const char * pFilepath,
                           dgs_token *  pTokensOut,
                           size_t       pNTokenCapacity,
                           size_t *     pNTokenCountOut);

  size_t mStateSeqSaveFile(dgs_seq_id pSeqId, const char * pFilepath, const dgs_token * pTokens, size_t pNTokenCount);

  //
  // perf
  //

  dgs_perf_context_data mPerfGetData() const;
  void                  mPerfReset();

  dgs_memory_breakdown mMemoryBreakdown() const;

  //

  //

  // TODO: more flexible combinations of logical/physical batch size and context size


private:
  //
  // output
  //

  // Make sure enough space is available for outputs.
  // Returns max number of outputs for which space was reserved.
  uint32_t mOutputReserve(int32_t pNOutputs);

  void mOutputReorder();

  // map the output row index `i` to batch index
  int64_t mOutputResolveRow(int32_t pI) const;

  // async-copy enabled layer-input tensors (per cparams.output_layer_inp)
  // from backend into host-side embd_layer_inp buffers
  bool mExtractLayerInputs(const llm_graph_result * pRes, size_t pTokenOffset, size_t pNTokens);

  //
  // graph
  //

public:
  uint32_t mGraphMaxNodes(uint32_t pNTokens) const;

  // can reuse the llm_graph_result instance of the context (for example to update a memory module)
  llm_graph_result * mGetGfResReserve() const;

  // returns the result of ggml_backend_sched_graph_compute_async execution
  ggml_status mGraphCompute(ggml_cgraph * pGf, bool pBatched);

  // reserve a graph with a dummy ubatch of the specified size
  ggml_cgraph * mGraphReserve(uint32_t                     pNTokens,
                              uint32_t                     pNSeqs,
                              uint32_t                     pNOutputs,
                              const dgs_memory_context_i * pMctx,
                              bool                         pSplitOnly = false,
                              size_t *                     pSizes     = nullptr);

  bool mSetSampler(dgs_seq_id pSeqId, dgs_sampler * pSampler);

private:
  llm_graph_result * mGetGfResPrev();

  llm_graph_params mGraphParams(llm_graph_result *           pRes,
                                const dgs_ubatch &           pUbatch,
                                const dgs_memory_context_i * pMctx,
                                llm_graph_type               pGtype) const;

  llm_graph_cb mGraphGetCb() const;

  // create a new backend scheduler for a graph of the specified size and register the callbacks below
  void mSchedNew(size_t pMaxNodes, bool pParallel);

  // ggml_backend_sched realloc callback, counts and traces the graph reallocations of this context
  static void mSchedGraphRealloc(ggml_backend_sched_t pSched,
                                 ggml_cgraph *        pGf,
                                 bool                 pBackendIdsChanged,
                                 bool                 pGraphSizeChanged,
                                 void *               pUserData);

  // disable auto fused ops (Flash Attention, Gated Delta Net) whose op lands on a device
  // that differs from the layer it belongs to (usually due to missing backend support)
  void mResolveFusedOps(const dgs_memory_context_i * pMctx, uint32_t pNSeqs);

  // TODO: read/write lora adapters and cvec
  size_t mStateWriteData(dgs_io_write_i & pIo);
  size_t mStateReadData(dgs_io_read_i & pIo);

  size_t mStateSeqWriteData(dgs_io_write_i & pIo, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags);
  size_t mStateSeqReadData(dgs_io_read_i & pIo, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags);

  //
  // members
  //

  const dgs_model & vModel;

  dgs_cparams vCparams;

  dgs_adapter_cvec_ptr  vCvec;
  dgs_adapter_loras_ptr dLoras;

  dgs_cross vCross;  // TODO: tmp for handling cross-attention - need something better probably

  dgs_memory_ptr vMemory;

  // decode output (2-dimensional array: [n_outputs][n_vocab])
  buffer_view<float> vLogits = { nullptr, 0 };

  // embeddings output (2-dimensional array: [n_outputs][n_embd])
  // populated only when pooling_type == DGS_POOLING_TYPE_NONE
  buffer_view<float> vEmbd = { nullptr, 0 };

  // hidden state required by the nextn layers (2-dimensional array: [n_outputs][n_embd])
  // populated only when cparams.embeddings_nextn is enabled and the model graph
  // sets llm_graph_result::t_h_nextn
  buffer_view<float> vEmbdNextn = { nullptr, 0 };

  // host buffers for output layer input embeddings, per layer
  // populated when cparams.output_layer_inp[il] is true
  std::vector<buffer_view<float>> lEmbdLayerInp;
  std::vector<int32_t>            lEmbdBatchIdxs;  // extracted index -> original batch index

  struct sampling_info {
    // !samplers.empty() to check if any samplers are active
    std::map<dgs_seq_id, dgs_sampler *> dSamplers;

    buffer_view<float>     vLogits     = { nullptr, 0 };
    buffer_view<dgs_token> vSampled    = { nullptr, 0 };
    buffer_view<float>     vProbs      = { nullptr, 0 };
    buffer_view<dgs_token> vCandidates = { nullptr, 0 };

    std::vector<uint32_t> lLogitsCount;
    std::vector<uint32_t> lProbsCount;
    std::vector<uint32_t> lCandidatesCount;

    // optimization
    std::vector<dgs_token> lTokenIdsFullVocab;
  };

  sampling_info vSampling;

  // sequence embeddings output (map of [n_embd] vectors)
  // populated only when pooling_type != DGS_POOLING_TYPE_NONE
  std::map<dgs_seq_id, std::vector<float>> dEmbdSeq;

  // reuse the batch_allocr to avoid unnecessary memory allocations
  std::unique_ptr<dgs_batch_allocr> vBalloc;

  uint32_t vNInputTensors = 0;      // number of tensors marked as input during the last graph reserve
  uint32_t vNOutputs      = 0;      // number of actually-used outputs in the current ubatch or last logical batch

  std::vector<int32_t> lOutputIds;  // map batch token positions to ids of the logits and embd buffers

  struct swap_info {
    uint32_t vI0;
    uint32_t vI1;
  };

  std::vector<swap_info> lOutputSwaps;

  ggml_backend_sched_ptr vSched;

  bool vSchedNeedReserve = true;

  // whether the reserve graph has a logits tensor, set in sched_reserve()
  // true until then, because the first output_reserve() runs before any graph is reserved
  bool vGraphHasLogits = true;

  // graph reallocations during inference where the backend assignments did not change, split by whether the
  // graph size is the same as the previously scheduled one (unexpected) or different
  int32_t vNReallocsSame = 0;
  int32_t vNReallocsDiff = 0;

  ggml_backend_t                vBackendCpu = nullptr;
  std::vector<ggml_backend_ptr> lBackends;

  ggml_threadpool_t vThreadpool      = nullptr;
  ggml_threadpool_t vThreadpoolBatch = nullptr;

  ggml_abort_callback vAbortCallback     = nullptr;
  void *              vAbortCallbackData = nullptr;

  std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> lSetNThreadsFns;

  // pointers and buffer types used for the compute buffer of each backend
  std::vector<ggml_backend_t>             lBackendPtrs;
  std::vector<ggml_backend_buffer_type_t> lBackendBuft;
  std::vector<size_t>                     lBackendBufExpSize;  // expected buffer sizes

  // Separate arenas give batches with and without outputs distinct CUDA graph cache keys.
  std::array<llm_graph_result_ptr, 2> aGfResPrev;
  llm_graph_result_ptr                vGfResReserve;

  llm_graph_result * vGfResPrevActive = nullptr;

  // host buffer for the model output (logits and embeddings)
  ggml_backend_buffer_ptr vBufOutput;

  // keep copies of the per-sequence memory on the device
  std::map<dgs_seq_id, dgs_memory_buffers> dMemStorage;

  bool vHasEvaluatedOnce = false;

  // env: DGS_GRAPH_REUSE_DISABLE
  bool vGraphReuseDisable = false;

  // true after the prism.hadamard coverage check passes on a reserved graph
  bool vHadamardVerified = false;

  // perf
  mutable int64_t vTStartUs = 0;
  mutable int64_t vTLoadUs  = 0;
  mutable int64_t vTPEvalUs = 0;
  mutable int64_t vTEvalUs  = 0;

  mutable int64_t vTComputeStartUs = 0;
  mutable int64_t vNQueuedTokens   = 0;

  mutable int32_t vNPEval = 0;   // number of tokens in eval calls for the prompt (with batch size > 1)
  mutable int32_t vNEval  = 0;   // number of eval calls

  mutable int32_t vNReused = 0;  // number of times the previous graph was reused
};
