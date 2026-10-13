#pragma once

#include "dgs-memory-hybrid.h"

#include <array>
#include <limits>
#include <memory>
#include <vector>

//
// dgs_memory_hybrid_idx
//

// dgs_memory_hybrid plus a third cache with one indexer key per token, for block-sparse attention (qwen4exp QSA)
// the indexer is a side buffer over the attention cells: same size, padding, streams and slots, so cell j is one token in both

class dgs_memory_hybrid_idx : public dgs_memory_hybrid {
public:
  dgs_memory_hybrid_idx(const dgs_model &       pModel,
                        /* attn */
                        ggml_type               pTypeK,
                        ggml_type               pTypeV,
                        bool                    pVTrans,
                        uint32_t                pKvSize,
                        uint32_t                pNPad,
                        uint32_t                pNSwa,
                        dgs_swa_type            pSwaType,
                        /* recurrent */
                        ggml_type               pTypeR,
                        ggml_type               pTypeS,
                        uint32_t                pRsSize,
                        /* common */
                        uint32_t                pNSeqMax,
                        uint32_t                pNRsSeq,
                        bool                    pOffload,
                        bool                    pUnified,
                        /* layer filters */
                        const layer_filter_cb & pFilterAttn,
                        const layer_filter_cb & pFilterRecr,
                        /* the indexer cache exists only if this is given */
                        const layer_filter_cb & pFilterIdx);

  // Defined out of line because kpool_layout is incomplete here.
  ~dgs_memory_hybrid_idx();

  //
  // dgs_memory_i
  //

  dgs_memory_context_ptr mInitBatch(dgs_batch_allocr & pBalloc, uint32_t pNUbatch, bool pEmbdAll) override;

  dgs_memory_context_ptr mInitFull() override;

  dgs_memory_context_ptr mInitUpdate(dgs_context * pLctx, bool pOptimize) override;

  void clear(bool data) override;

  bool mSeqRm(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1) override;
  void mSeqCp(dgs_seq_id pSeqIdSrc, dgs_seq_id pSeqIdDst, dgs_pos pP0, dgs_pos pP1) override;
  void mSeqKeep(dgs_seq_id pSeqId) override;
  void mSeqAdd(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1, dgs_pos pShift) override;
  void mSeqDiv(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1, int pD) override;

  std::map<ggml_backend_buffer_type_t, size_t> mMemoryBreakdown() const override;

  // state write/load

  void mStateWrite(dgs_io_write_i & pIo, dgs_seq_id pSeqId = -1, dgs_state_seq_flags pFlags = 0) const override;
  void mStateRead(dgs_io_read_i & pIo, dgs_seq_id pSeqId = -1, dgs_state_seq_flags pFlags = 0) override;

  //
  // dgs_memory_hybrid_idx specific API
  //

  dgs_kv_cache * mGetMemIdx() const;  // nullptr when the model carries no indexer

  // The model's indexer pool size.
  uint32_t mGetKpool() const { return vHparamsIdx.vIndexerKpool; }

  // Whether pools are kpool consecutive cells in sequence order (qwen4exp) instead of kpool consecutive positions.
  bool mGetKpoolByOrder() const { return vHparamsIdx.vIndexerKpoolByOrder; }

  // Which cells of a sequence make up which pool of kpool consecutive positions (or cells, in order mode).
  // It is kept here because it outlives the batch: pools are fixed by the positions relative to the
  // sequence's first one, so a ubatch only ever appends to it. Sequence edits drop it, see mem_idx_stale.
  struct kpool_layout;

  const kpool_layout & mKpoolLayoutUpdate();
  const kpool_layout & mKpoolLayoutGet() const;

  // The pooled keys persist in the idx cache across batches. A sequence edit can regroup the pools
  // from some position on, which stales every pooled key at or after it. POS_CLEAN means none.
  using stale_pos_t = std::array<dgs_pos, DGS_MAX_SEQ>;

  static constexpr dgs_pos cPosClean = std::numeric_limits<dgs_pos>::max();

  static stale_pos_t mStalePosClean() {
    stale_pos_t aRes;
    aRes.fill(cPosClean);
    return aRes;
  }

  const stale_pos_t & mMemIdxStaleGet() const { return aMemIdxStale; }

  void mMemIdxStaleClear() { aMemIdxStale.fill(cPosClean); }

private:
  // forget seq_id (all of it if seq_id < 0) in every cache at once, so a failed restore cannot leave the caches out of step
  // seq_id < 0 drops the whole context, as the caches themselves do on a failed restore
  void mStateDrop(dgs_seq_id pSeqId);

  // the indexer cache holds one key head per layer, so it needs its own hparams:
  // dgs_kv_cache keeps a reference to what it is given
  dgs_hparams vHparamsIdx;

  const std::unique_ptr<dgs_kv_cache> cMemIdx;

  // unique_ptr because kpool_layout is incomplete here
  std::unique_ptr<kpool_layout> vKpoolLay;

  // seq_id < 0 stales every sequence, p0 < 0 stales the sequence from its first position
  void mMemIdxStaleSet(dgs_seq_id pSeqId, dgs_pos pP0);

  // the position an edit at p0 stales the sequence from
  dgs_pos mMemIdxStalePos(dgs_seq_id pSeqId, dgs_pos pP0) const;

  stale_pos_t aMemIdxStale = mStalePosClean();
};

class dgs_memory_hybrid_idx_context : public dgs_memory_hybrid_context {
public:
  class kpool_access {
  public:
    ggml_tensor * mGatherKeyGate(ggml_tensor * pIdxs) const;
    ggml_tensor * mScatterPooled(ggml_tensor * pValues, ggml_tensor * pIdxs) const;
    ggml_tensor * mGatherPooled(ggml_tensor * pIdxs) const;

  private:
    friend class dgs_memory_hybrid_idx_context;

    kpool_access(ggml_context * pCtx, ggml_tensor * pK, int64_t pNEmbd);

    ggml_context * vCtx;
    ggml_tensor *  vKeyGate;
    ggml_tensor *  vPooled;
  };

  using slot_info_vec_t = dgs_kv_cache::slot_info_vec_t;

  // used for errors
  explicit dgs_memory_hybrid_idx_context(dgs_memory_status pStatus);

  // used to create a full-cache context
  explicit dgs_memory_hybrid_idx_context(dgs_memory_hybrid_idx * pMem);

  // used to create an update context
  dgs_memory_hybrid_idx_context(dgs_memory_hybrid_idx * pMem, dgs_context * pLctx, bool pOptimize);

  // used to create a batch processing context from a batch
  dgs_memory_hybrid_idx_context(dgs_memory_hybrid_idx * pMem,
                                slot_info_vec_t         pSinfosAttn,
                                slot_info_vec_t         pSinfosIdx,
                                std::vector<dgs_ubatch> pUbatches);

  ~dgs_memory_hybrid_idx_context();  // Defined out of line because kpool_state is incomplete here.

  //
  // dgs_memory_context_i
  //

  bool mNext() override;
  bool apply() override;

  //
  // dgs_memory_hybrid_idx_context specific API
  //

  // nullptr with no indexer
  const dgs_kv_cache_context * mGetIdx() const;

  // streams in the current slot info, the `ns` of get_k/get_v; 1 if unified
  uint32_t mGetNStream() const;

  // glm5-next and qwen4exp, complete pools of kpool cells per sequence, scored as whole pools.
  uint32_t     mGetNKpool() const;     // Padded pool count, where the last pool is always unused.
  uint32_t     mGetNKpoolNew() const;  // Pools to re-pool this ubatch, padded to a stable bound, never below 1.
  kpool_access mGetKpoolAccess(ggml_context * pCtx, int32_t pIl, int64_t pNEmbd) const;
  // sel_mask (F32 [n_sel, 1, 1, n_tokens], can be null): 0 for the live selection slots, -inf for the dead ones
  // new_pool_pos (I32 [4*n_new]): M-RoPE position of each new pool's first member, for pooled keys rotated at pooling time
  void         mSetInputKpool(ggml_tensor *      pPoolCells,
                              ggml_tensor *      pPoolIdxs,
                              ggml_tensor *      pPoolMask,
                              ggml_tensor *      pTailIdxs,
                              ggml_tensor *      pSelMask,
                              ggml_tensor *      pNewPoolIdxs,
                              ggml_tensor *      pNewPoolRep,
                              const dgs_ubatch * pUbatch,
                              ggml_tensor *      pNewPoolPos = nullptr) const;

private:
  dgs_memory_hybrid_idx * vMem = nullptr;

  // streams per ubatch, read from the slot infos before ctx_idx takes them
  // declared first, so it is initialised while sinfos_idx is still intact
  const std::vector<uint32_t> lNsUbatch;

  // the indexer cells of each ubatch, kept for pools in cache order (qwen4exp): token s*n + i of ubatch u
  // sits in cell idxs[s][i] of stream strm[s] of sinfos_kpool[u], and several cells can share a position
  const slot_info_vec_t lSinfosKpool;

  // null unless the model has an indexer
  const dgs_memory_context_ptr cCtxIdx;

  // mirrors the base class's ubatch cursor, which is private there
  size_t vICur = 0;

  // Which pools of the layout this ubatch must re-pool. The layout itself belongs to the memory.
  struct kpool_state;
  kpool_state         mKpoolBuildSizes() const;
  void                mKpoolBuildState(const dgs_ubatch & pUbatch);
  const kpool_state & mKpoolCur() const;

  // unique_ptr because kpool_state is incomplete here.
  std::unique_ptr<kpool_state> vKpoolSt;

  // The ubatch kpool_st was built for, guards against reads before apply.
  size_t vIKpool = SIZE_MAX;

  // Whether this context tracks k-pool states.
  bool mKpoolTrack() const;

  // Positions each sequence must re-pool from, cleared only after the first ubatch succeeds
  dgs_memory_hybrid_idx::stale_pos_t aMemIdxStaleBatch = dgs_memory_hybrid_idx::mStalePosClean();
};
