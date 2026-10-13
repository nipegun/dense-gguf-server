#pragma once

#include "dgs-kv-cache-iswa.h"
#include "dgs-kv-cache.h"

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

class dgs_dsv4_comp_state {
public:
  using stream_copy_info = dgs_kv_cache::stream_copy_info;

  stream_copy_info vScInfo;

  dgs_dsv4_comp_state(const dgs_model &                     pModel,
                      bool                                  pOffload,
                      bool                                  pUnified,
                      uint32_t                              pNSeqMax,
                      uint32_t                              pRatio,
                      uint32_t                              pStateSize,
                      uint32_t                              pNEmbdState,
                      uint32_t                              pNRsSeq,
                      const char *                          name,
                      const dgs_memory_i::layer_filter_cb & pFilter);

  void clear(dgs_seq_id pSeqId, bool data);
  void mSeqCp(dgs_seq_id pSeqIdSrc, dgs_seq_id pSeqIdDst);
  void mApplyCopies(const stream_copy_info & pScInfo) const;

  uint32_t mGetRatio() const;
  uint32_t mGetStateSize() const;
  uint32_t mGetNStream() const;
  uint32_t mGetNRsSeq() const;
  uint32_t mGetNRows() const;

  std::map<ggml_backend_buffer_type_t, size_t> mMemoryBreakdown() const;

  void mStateWrite(dgs_io_write_i &              pIo,
                   dgs_seq_id                    pSeqId,
                   dgs_state_seq_flags           pFlags,
                   const std::vector<uint32_t> & pRsIdx) const;
  void mStateRead(dgs_io_read_i & pIo, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags);

  ggml_tensor * mGetKv(ggml_context * pCtx, int32_t pIl) const;
  ggml_tensor * mGetScore(ggml_context * pCtx, int32_t pIl) const;
  ggml_tensor * mGetKvAll(ggml_context * pCtx, int32_t pIl) const;
  ggml_tensor * mGetScoreAll(ggml_context * pCtx, int32_t pIl) const;

  ggml_tensor * mCpyKv(ggml_context * pCtx, ggml_tensor * pCur, ggml_tensor * pIdxs, int32_t pIl) const;
  ggml_tensor * mCpyScore(ggml_context * pCtx, ggml_tensor * pCur, ggml_tensor * pIdxs, int32_t pIl) const;

private:
  struct layer {
    uint32_t vIl;

    ggml_tensor * vKv;
    ggml_tensor * vScore;

    std::vector<ggml_tensor *> lKvStream;
    std::vector<ggml_tensor *> lScoreStream;
  };

  const uint32_t cRatio;
  const uint32_t cStateSize;
  const uint32_t cNEmbdState;
  const uint32_t cNStream;
  const uint32_t cNRsSeq;

  std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> lCtxsBufs;

  std::vector<layer> lLayers;

  std::unordered_map<int32_t, int32_t> dMapLayerIds;

  size_t mTotalSize() const;
};

//
// dgs_kv_cache_dsv4
//

// DSV4 uses a normal raw/SWA token cache plus compressed K-only block caches.
// The compressed caches are storage only; DSV4-specific visibility and block
// planning are handled by dgs_kv_cache_dsv4_context / llm_graph_input_dsv4.
// FIXME: currently the cache only supports non-unified mode even if unified flag is passed
// FIXME: we currently conflate token_pos and buffer contents. See 

class dgs_kv_cache_dsv4 : public dgs_memory_i {
public:
  dgs_kv_cache_dsv4(const dgs_model &       pModel,
                    ggml_type               pTypeK,
                    ggml_type               pTypeV,
                    bool                    pVTrans,
                    bool                    pOffload,
                    bool                    pSwaFull,
                    bool                    pUnified,
                    uint32_t                pKvSize,
                    uint32_t                pNSeqMax,
                    uint32_t                pNUbatch,
                    uint32_t                pNPad,
                    uint32_t                pNRsSeq,
                    const layer_filter_cb & pFilter,
                    const layer_reuse_cb &  pReuse);

  ~dgs_kv_cache_dsv4() = default;

  //
  // dgs_memory_i
  //

  dgs_memory_context_ptr mInitBatch(dgs_batch_allocr & pBalloc, uint32_t pNUbatch, bool pEmbdAll) override;

  dgs_memory_context_ptr mInitFull() override;

  dgs_memory_context_ptr mInitUpdate(dgs_context * pLctx, bool pOptimize) override;

  bool mGetCanShift() const override;

  void clear(bool data) override;

  bool mSeqRm(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1) override;
  void mSeqCp(dgs_seq_id pSeqIdSrc, dgs_seq_id pSeqIdDst, dgs_pos pP0, dgs_pos pP1) override;
  void mSeqKeep(dgs_seq_id pSeqId) override;
  void mSeqAdd(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1, dgs_pos pShift) override;
  void mSeqDiv(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1, int pD) override;

  dgs_pos mSeqPosMin(dgs_seq_id pSeqId) const override;
  dgs_pos mSeqPosMax(dgs_seq_id pSeqId) const override;

  std::map<ggml_backend_buffer_type_t, size_t> mMemoryBreakdown() const override;

  void mStateWrite(dgs_io_write_i & pIo, dgs_seq_id pSeqId = -1, dgs_state_seq_flags pFlags = 0) const override;
  void mStateRead(dgs_io_read_i & pIo, dgs_seq_id pSeqId = -1, dgs_state_seq_flags pFlags = 0) override;

  //
  // dgs_kv_cache_dsv4 specific API
  //

  dgs_kv_cache_iswa *   mGetRaw() const;
  dgs_kv_cache *        mGetCsa() const;
  dgs_kv_cache *        mGetHca() const;
  dgs_kv_cache *        mGetLid() const;
  dgs_dsv4_comp_state * mGetCsaState() const;
  dgs_dsv4_comp_state * mGetHcaState() const;
  dgs_dsv4_comp_state * mGetLidState() const;

  uint32_t                      mGetNRsSeq() const;
  const std::vector<uint32_t> & mGetRsIdx() const;
  void                          mResetRsIdxForUbatches(const std::vector<dgs_ubatch> & pUbatches);

private:
  dgs_hparams vHparamsRaw;
  dgs_hparams vHparamsCsa;
  dgs_hparams vHparamsHca;
  dgs_hparams vHparamsLid;

  const uint32_t cNSeqMax;
  const uint32_t cNRsSeq;

  std::vector<uint32_t> lRsIdx;

  std::unique_ptr<dgs_kv_cache_iswa>   vKvRaw;
  std::unique_ptr<dgs_kv_cache>        vKvCsa;
  std::unique_ptr<dgs_kv_cache>        vKvHca;
  std::unique_ptr<dgs_kv_cache>        vKvLid;
  std::unique_ptr<dgs_dsv4_comp_state> vCsaState;
  std::unique_ptr<dgs_dsv4_comp_state> vHcaState;
  std::unique_ptr<dgs_dsv4_comp_state> vLidState;

  void mClearCompressed(dgs_seq_id pSeqId, bool data);
};

// DSV4 raw attention only uses the SWA half of kv_raw. The base half is kept
// for generic ISWA bookkeeping, but it has no DSV4 layers to expose here.
class dgs_kv_cache_dsv4_raw_context : public dgs_memory_context_i {
public:
  using slot_info_vec_t = dgs_kv_cache::slot_info_vec_t;

  dgs_kv_cache_dsv4_raw_context(dgs_kv_cache_iswa * pKv);

  dgs_kv_cache_dsv4_raw_context(dgs_kv_cache_iswa * pKv, dgs_context * pLctx, bool pOptimize);

  dgs_kv_cache_dsv4_raw_context(dgs_kv_cache_iswa *     pKv,
                                slot_info_vec_t         pSinfosBaseWrite,
                                slot_info_vec_t         pSinfosSwaWrite,
                                slot_info_vec_t         pSinfosSwaRead,
                                std::vector<dgs_ubatch> pUbatches,
                                std::vector<dgs_ubatch> pUbatchesWrite);

  bool mNext() override;
  bool apply() override;

  dgs_memory_status  mGetStatus() const override;
  const dgs_ubatch & mGetUbatch() const override;

  uint32_t mGetNKv() const;
  uint32_t mGetNWrite() const;

  ggml_tensor * mGetK(ggml_context * pCtx, int32_t pIl) const;
  ggml_tensor * mCpyK(ggml_context * pCtx, ggml_tensor * pKCur, ggml_tensor * pKIdxs, int32_t pIl) const;

  ggml_tensor * mBuildInputKIdxs(ggml_context * pCtx, const dgs_ubatch & pUbatch) const;
  ggml_tensor * mBuildInputKRot(ggml_context * pCtx) const;

  void mSetInputKIdxs(ggml_tensor * pDst) const;
  void mSetInputKqMask(ggml_tensor * pDst, const dgs_ubatch * pUbatch, bool pCausalAttn) const;
  void mSetInputKRot(ggml_tensor * pDst) const;

private:
  size_t vINext = 0;

  dgs_kv_cache * vKvSwa = nullptr;

  slot_info_vec_t         lSinfosWrite;
  slot_info_vec_t         lSinfosRead;
  std::vector<dgs_ubatch> lUbatches;
  std::vector<dgs_ubatch> lUbatchesWrite;

  const dgs_memory_context_ptr cCtxBaseMem;
  const dgs_memory_context_ptr cCtxSwaMem;

  uint32_t vNKv = 0;

  const dgs_memory_status cStatus;
};

// DSV4 compressed KV rows are graph outputs, not normal token KV writes.
// Keep a small context that exposes K tensors without generic apply() semantics.
class dgs_kv_cache_dsv4_comp_context {
public:
  using slot_info_vec_t = dgs_kv_cache::slot_info_vec_t;

  dgs_kv_cache_dsv4_comp_context(dgs_kv_cache * pKv);

  dgs_kv_cache_dsv4_comp_context(dgs_kv_cache * pKv, slot_info_vec_t pSinfos, std::vector<dgs_ubatch> pUbatches);

  bool mNext();

  uint32_t mGetNKv() const;

  ggml_tensor * mGetK(ggml_context * pCtx, int32_t pIl) const;
  ggml_tensor * mCpyK(ggml_context * pCtx, ggml_tensor * pKCur, ggml_tensor * pKIdxs, int32_t pIl) const;

  ggml_tensor * mBuildInputKRot(ggml_context * pCtx) const;
  void          mSetInputKRot(ggml_tensor * pDst) const;

private:
  dgs_kv_cache * vKv;

  size_t                  vICur = 0;
  slot_info_vec_t         lSinfos;
  std::vector<dgs_ubatch> lUbatches;

  uint32_t vNKv;
};

class dgs_kv_cache_dsv4_context : public dgs_memory_context_i {
public:
  using slot_info_vec_t  = dgs_kv_cache::slot_info_vec_t;
  using stream_copy_info = dgs_kv_cache::stream_copy_info;

  struct comp_plan {
    // Per-ubatch recipe for updating compressor state, committing completed
    // compressed rows, and masking the compressed attention source.

    // APE row ids, i.e. pos % ratio, for the compressor-state updates.
    std::vector<int32_t> lStatePos;

    // Current-ubatch source row ids and unique persistent-state
    // destination row ids for deterministic ring-state updates.
    std::vector<int32_t> lStatePersistSrcIdxs;
    std::vector<int32_t> lStatePersistDstIdxs;

    // Device-side rollback restore copies snapshot planes back to the
    // current compressor-state plane before the graph reads it.
    std::vector<int32_t> lStateRestoreSrcIdxs;
    std::vector<int32_t> lStateRestoreDstIdxs;

    // Device-side rollback snapshots copy rows from the graph-local
    // [persistent_state | current_ubatch_scratch] tensor into rollback
    // planes after the graph has computed current-token compressor state.
    std::vector<int32_t> lStateSnapshotSrcIdxs;
    std::vector<int32_t> lStateSnapshotDstIdxs;

    // Flattened source row ids used for state-backed commits. Source rows
    // index the graph-local [persistent_state | current_ubatch_scratch]
    // tensor. For overlapped compression the first half is previous rows
    // and the second half is current rows; a final synthetic zero/-inf row
    // may be addressed for the first block's previous half.
    std::vector<int32_t> lStateReadIdxs;

    // Final compressed-cache row ids written by state-backed commits.
    // A non-boundary CSA/LID decode step can target a masked scratch row.
    std::vector<int64_t> lStateWriteIdxs;

    // RoPE positions for state-backed commits.
    std::vector<int32_t> lStateWritePos;

    // Number of completed compressed rows visible for each query token.
    std::vector<int32_t> lNVisible;

    // Number of streams used by the attention graph for this ubatch.
    int64_t vNStream = 1;

    // Graph-width for compressed rows. This can be larger than n_visible
    // so masked padding rows do not force a new graph at every CSA block.
    int64_t vNKv = 0;
  };

  dgs_kv_cache_dsv4_context(dgs_memory_status pStatus);

  dgs_kv_cache_dsv4_context(dgs_kv_cache_dsv4 * pKv);

  dgs_kv_cache_dsv4_context(dgs_kv_cache_dsv4 * pKv,
                            dgs_context *       pLctx,
                            bool                pOptimize,
                            stream_copy_info    pScInfoCsa,
                            stream_copy_info    pScInfoHca,
                            stream_copy_info    pScInfoLid);

  dgs_kv_cache_dsv4_context(dgs_kv_cache_dsv4 *     pKv,
                            slot_info_vec_t         pSinfosRawBaseWrite,
                            slot_info_vec_t         pSinfosRawSwaWrite,
                            slot_info_vec_t         pSinfosRawSwaRead,
                            std::vector<dgs_ubatch> pUbatches,
                            std::vector<dgs_ubatch> pUbatchesRaw);

  virtual ~dgs_kv_cache_dsv4_context();

  //
  // dgs_memory_context_i
  //

  bool mNext() override;
  bool apply() override;

  dgs_memory_status  mGetStatus() const override;
  const dgs_ubatch & mGetUbatch() const override;

  //
  // dgs_kv_cache_dsv4_context specific API
  //

  const dgs_kv_cache_dsv4_raw_context *  mGetRaw() const;
  const dgs_kv_cache_dsv4_comp_context * mGetCsa() const;
  const dgs_kv_cache_dsv4_comp_context * mGetHca() const;
  const dgs_kv_cache_dsv4_comp_context * mGetLid() const;
  const dgs_dsv4_comp_state *            mGetCsaState() const;
  const dgs_dsv4_comp_state *            mGetHcaState() const;
  const dgs_dsv4_comp_state *            mGetLidState() const;

  const comp_plan & mGetCsaPlan() const;
  const comp_plan & mGetHcaPlan() const;
  const comp_plan & mGetLidPlan() const;

  const comp_plan & mGetCsaPlan(const dgs_ubatch & pUbatch) const;
  const comp_plan & mGetHcaPlan(const dgs_ubatch & pUbatch) const;
  const comp_plan & mGetLidPlan(const dgs_ubatch & pUbatch) const;

private:
  size_t vINext = 0;

  std::vector<dgs_ubatch> lUbatches;

  std::vector<comp_plan> lPlansCsa;
  std::vector<comp_plan> lPlansHca;
  std::vector<comp_plan> lPlansLid;

  const std::unique_ptr<dgs_kv_cache_dsv4_raw_context> cCtxRaw;
  const dgs_memory_context_ptr                         cCtxCsaMem;
  const dgs_memory_context_ptr                         cCtxHcaMem;
  const dgs_memory_context_ptr                         cCtxLidMem;

  const std::unique_ptr<dgs_kv_cache_dsv4_comp_context> cCtxCsa;
  const std::unique_ptr<dgs_kv_cache_dsv4_comp_context> cCtxHca;
  const std::unique_ptr<dgs_kv_cache_dsv4_comp_context> cCtxLid;

  dgs_dsv4_comp_state * vCsaState = nullptr;
  dgs_dsv4_comp_state * vHcaState = nullptr;
  dgs_dsv4_comp_state * vLidState = nullptr;

  stream_copy_info vScInfoCsa;
  stream_copy_info vScInfoHca;
  stream_copy_info vScInfoLid;

  bool              vReservePlans = false;
  mutable comp_plan vReservePlanCsa;
  mutable comp_plan vReservePlanHca;
  mutable comp_plan vReservePlanLid;

  const dgs_memory_status cStatus;
};
