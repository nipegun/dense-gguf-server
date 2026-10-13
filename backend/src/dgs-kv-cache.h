#pragma once

#include "dgs-batch.h"
#include "dgs-graph.h"
#include "dgs-kv-cells.h"
#include "dgs-memory.h"

#include <unordered_map>
#include <vector>

struct dgs_cparams;
struct dgs_hparams;
struct dgs_model;
struct dgs_context;

//
// dgs_kv_cache
//

class dgs_kv_cache : public dgs_memory_i {
public:
  struct stream_copy_info {
    bool empty() const {
      assert(ssrc.size() == sdst.size());
      return lSsrc.empty();
    }

    std::vector<uint32_t> lSsrc;
    std::vector<uint32_t> lSdst;
  };

  // for each ubatch, create a slot_info that contains information about where the ubatch should be inserted in the
  //   KV cells. for example, cell indices for each token, such that: token[i] -> goes to cells[idxs[i]]
  struct slot_info {
    // data for ggml_set_rows
    using idx_vec_t = std::vector<uint32_t>;

    // number of streams: ns = s1 - s0 + 1
    uint32_t vS0;
    uint32_t vS1;

    std::vector<dgs_seq_id> lStrm;  // [ns]
    std::vector<idx_vec_t>  lIdxs;  // [ns]

    uint32_t mHead() const {
      GGML_ASSERT(lIdxs.size() == 1);
      GGML_ASSERT(!lIdxs[0].empty());

      return lIdxs[0][0];
    }

    void resize(size_t n) {
      lStrm.resize(n);
      lIdxs.resize(n);
    }

    size_t size() const {
      GGML_ASSERT(lIdxs.size() == lStrm.size());
      GGML_ASSERT(!lIdxs.empty());

      return lIdxs[0].size();
    }

    size_t mNStream() const { return lStrm.size(); }

    bool empty() const { return lIdxs.empty(); }

    void clear() { lIdxs.clear(); }

    // check if indices are contiguous starting from head()
    bool mIsContiguous() const {
      if (lIdxs.empty() || lIdxs[0].empty()) {
        return true;
      }
      if (lIdxs.size() > 1) {
        return false;
      }
      const uint32_t cH = lIdxs[0][0];
      for (size_t vI = 0; vI < lIdxs[0].size(); ++vI) {
        if (lIdxs[0][vI] != cH + vI) {
          return false;
        }
      }
      return true;
    }
  };

  using slot_info_vec_t = std::vector<slot_info>;

  // TODO: refactor the memory instances to not depend on `dgs_model`
  //       instead pass all necessary info (e.g. hparams, dev layers, arch, etc.) directly
  //       likely through `struct dgs_memory_params`
  dgs_kv_cache(const dgs_model &       pModel,
               const dgs_hparams &     pHparams,
               ggml_type               pTypeK,
               ggml_type               pTypeV,
               bool                    pVTrans,
               bool                    pOffload,
               bool                    pUnified,
               uint32_t                pKvSize,
               uint32_t                pNSeqMax,
               uint32_t                pNPad,
               uint32_t                pNSwa,
               dgs_swa_type            pSwaType,
               dgs_memory_t            pMemOther,
               const layer_filter_cb & pFilter,
               const layer_reuse_cb &  pReuse,
               const layer_share_cb &  pShare,
               // a model can hold more than one cache, so the tensor names have to stay unique
               const char *            pNameTag = "");

  ~dgs_kv_cache() = default;

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

  // state write/load

  void mStateWrite(dgs_io_write_i & pIo, dgs_seq_id pSeqId = -1, dgs_state_seq_flags pFlags = 0) const override;
  void mStateRead(dgs_io_read_i & pIo, dgs_seq_id pSeqId = -1, dgs_state_seq_flags pFlags = 0) override;

  //
  // dgs_kv_cache specific API
  //

  uint32_t mGetSize() const;
  uint32_t mGetNSeqMax() const;
  uint32_t mGetNStream() const;

  bool mGetHasShift() const;

  ggml_type mTypeK() const;
  ggml_type mTypeV() const;

  std::vector<uint32_t> mGetLayerIds() const;
  ggml_tensor *         mGetKStorage(int32_t pIl) const;

  const dgs_kv_cells & mGetCells(dgs_seq_id pSeqId) const;

  // The stream holding seq_id's cells.
  uint32_t mGetStream(dgs_seq_id pSeqId) const;

  // state_read, plus the cells the restored tokens were placed in
  // a cache that mirrors another one (the qwen4exp indexer) must not search for its own cells: two searches agree only by luck
  //   sinfos_out: if set, filled with the layout used; a stream with no cells leaves an empty entry
  //   sinfos_in : if set, the layout to use instead of searching. one entry per stream, cell count must match the blob
  void mStateReadSinfo(dgs_io_read_i &         pIo,
                       dgs_seq_id              pSeqId,
                       dgs_state_seq_flags     pFlags,
                       slot_info_vec_t *       pSinfosOut,
                       const slot_info_vec_t * pSinfosIn);

  // undo a state_read() of seq_id (-1 for the whole cache) that another memory module failed to complete
  void mStateClear(dgs_seq_id pSeqId);

  //
  // graph_build API
  //

  uint32_t mGetNKv(const slot_info & pSinfo) const;

  // get views of the current state of the cache
  ggml_tensor * mGetK(ggml_context * pCtx, int32_t pIl, uint32_t pNKv, const slot_info & pSinfo) const;
  ggml_tensor * mGetV(ggml_context * pCtx, int32_t pIl, uint32_t pNKv, const slot_info & pSinfo) const;

  // store k_cur and v_cur in the cache based on the provided head location
  ggml_tensor * mCpyK(ggml_context *    pCtx,
                      ggml_tensor *     pKCur,
                      ggml_tensor *     pKIdxs,
                      int32_t           pIl,
                      const slot_info & pSinfo) const;
  ggml_tensor * mCpyV(ggml_context *    pCtx,
                      ggml_tensor *     pVCur,
                      ggml_tensor *     pVIdxs,
                      int32_t           pIl,
                      const slot_info & pSinfo) const;

  //
  // preparation API
  //

  // find places for the provided ubatches in the cache, returns the slot infos
  // return empty vector on failure
  slot_info_vec_t mPrepare(const std::vector<dgs_ubatch> & pUbatches);

  bool mUpdate(dgs_context * pLctx, bool pDoShift, const stream_copy_info & pScInfo);

  // find a slot of kv cells that can hold the ubatch
  // if cont == true, then the slot must be continuous
  // return empty slot_info on failure
  slot_info mFindSlot(const dgs_ubatch & pUbatch, bool pCont) const;

  // emplace the ubatch context into slot: [sinfo.idxs[0...ubatch.n_tokens - 1]]
  void mApplyUbatch(const slot_info & pSinfo, const dgs_ubatch & pUbatch);

  //
  // input API
  //

  ggml_tensor * mBuildInputKIdxs(ggml_context * pCtx, const dgs_ubatch & pUbatch) const;
  ggml_tensor * mBuildInputVIdxs(ggml_context * pCtx, const dgs_ubatch & pUbatch) const;

  ggml_tensor * mBuildInputKRot(ggml_context * pCtx) const;
  ggml_tensor * mBuildInputVRot(ggml_context * pCtx) const;

  void mSetInputKIdxs(ggml_tensor * pDst, const dgs_ubatch * pUbatch, const slot_info & pSinfo) const;
  void mSetInputVIdxs(ggml_tensor * pDst, const dgs_ubatch * pUbatch, const slot_info & pSinfo) const;

  void mSetInputKShift(ggml_tensor * pDst) const;

  void mSetInputKqMask(ggml_tensor * pDst, const dgs_ubatch * pUbatch, bool pCausalAttn) const;
  void mSetInputPosBucket(ggml_tensor * pDst, const dgs_ubatch * pUbatch) const;

  void mSetInputKRot(ggml_tensor * pDst) const;
  void mSetInputVRot(ggml_tensor * pDst) const;

  // true if dgs_kv_cell_ext holds information that has to survive a state save/restore
  bool mHasCellExt() const;

  // for every token of the ubatch, the ids of the n tokens that precede it in its sequence
  // example for M-RoPE image case: tokens A B X X X C, where X is a 3-token image at pos 2 spanning positions 2..4:
  //   tok: A B X X X C
  //   pos: 0 1 2 2 2 5
  //   prev, n=2: A -> [NULL, NULL], B -> [NULL, A], 3rd X -> [X, X], C -> [X, X]
  // note: used by n-gram input embeddings
  void mGetPrevTokens(const dgs_ubatch & pUbatch, uint32_t n, std::vector<dgs_token> & pRes) const;

private:
  const dgs_model &   vModel;
  const dgs_hparams & vHparams;

  struct kv_layer {
    // layer index in the model
    // note: can be different from the layer index in the KV cache
    uint32_t vIl;

    ggml_tensor * vK;
    ggml_tensor * vV;

    std::vector<ggml_tensor *> lKStream;
    std::vector<ggml_tensor *> lVStream;
  };

  bool vVTrans = true;  // the value tensor is transposed

  const uint32_t cNSeqMax = 1;
  const uint32_t cNStream = 1;

  // required padding
  const uint32_t cNPad = 1;

  // SWA
  const uint32_t cNSwa = 0;

  // env: DGS_ATTN_ROT_DISABLE
  uint32_t vNRotK = 0;
  uint32_t vNRotV = 0;

  // if all layers participating in the cache have constant head size, the value is stored here
  // otherwise the value is -1
  int32_t vNEmbdHeadKAll = 0;
  int32_t vNEmbdHeadVAll = 0;

  // pre-computed hadamard martrices
  std::unordered_map<int64_t, std::vector<float>> dAttnRotHadamard;

  // env: DGS_KV_CACHE_DEBUG
  int vDebug = 0;

  // this is the SWA type of the cache - not to be confused with the model SWA type
  const dgs_swa_type cSwaType = cDgsSwaTypeNone;

  // ggml contexts for the KV cache along with the allocated backend buffers:
  std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> lCtxsBufs;

  // the current index from where we start searching for a free slot in the ring buffer of KV cells (see find_slot())
  // note: this is not part of the KV state and it's only used to speed-up the find_slot() method
  std::vector<uint32_t> lVHeads;

  // TODO: temporary until we refactor to be able to share the same cells between 2 kv caches [TAG_KV_CACHE_SHARE_CELLS]
  dgs_kv_cache * vOther;

  std::shared_ptr<dgs_kv_cells_vec> lVCellsImpl;

  dgs_kv_cells_vec & lVCells;

  // maps from a sequence id to a stream id
  std::vector<uint32_t> lSeqToStream;

  // pending stream copies that will be applied during the next update
  stream_copy_info vScInfo;

  std::vector<kv_layer> lLayers;

  // model layer id -> KV cache layer id
  std::unordered_map<int32_t, int32_t> dMapLayerIds;

  size_t mTotalSize() const;

  size_t mSizeKBytes() const;
  size_t mSizeVBytes() const;

  ggml_tensor * mBuildRopeShift(const dgs_cparams & pCparams,
                                ggml_context *      pCtx,
                                ggml_tensor *       pCur,
                                ggml_tensor *       pShift,
                                ggml_tensor *       pRot,
                                ggml_tensor *       pFactors,
                                float               pFreqBase,
                                float               pFreqScale,
                                uint32_t            pIl) const;

  ggml_cgraph * mBuildGraphShift(llm_graph_result * pRes, dgs_context * pLctx) const;

  struct cell_ranges_t {
    uint32_t vStrm;

    std::vector<std::pair<uint32_t, uint32_t>> data;  // ranges, from inclusive, to exclusive
  };

  void mStateWriteMeta(dgs_io_write_i & pIo, const cell_ranges_t & pCr, dgs_seq_id pSeqId = -1) const;
  void mStateWriteData(dgs_io_write_i & pIo, const cell_ranges_t & pCr) const;

  // sinfo_in, when set, replaces the find_slot call: the cells are given by the caller
  bool mStateReadMeta(dgs_io_read_i &   pIo,
                      uint32_t          pStrm,
                      uint32_t          pCellCount,
                      slot_info &       pSinfo,
                      dgs_seq_id        pDestSeqId = -1,
                      const slot_info * pSinfoIn   = nullptr);
  bool mStateReadData(dgs_io_read_i & pIo, uint32_t pStrm, uint32_t pCellCount, const slot_info & pSinfo);

  void mStateClear(dgs_seq_id pSeqId, uint32_t pStrm, const slot_info & pSinfo);
};

class dgs_kv_cache_context : public dgs_memory_context_i {
public:
  // some shorthands
  using slot_info_vec_t  = dgs_kv_cache::slot_info_vec_t;
  using stream_copy_info = dgs_kv_cache::stream_copy_info;

  // used for errors
  dgs_kv_cache_context(dgs_memory_status pStatus);

  // used to create a full-cache context
  dgs_kv_cache_context(dgs_kv_cache * pKv);

  // used to create an update context
  dgs_kv_cache_context(dgs_kv_cache * pKv, dgs_context * pLctx, bool pDoShift, stream_copy_info pScInfo);

  // used to create a batch processing context from a batch
  dgs_kv_cache_context(dgs_kv_cache * pKv, slot_info_vec_t pSinfos, std::vector<dgs_ubatch> pUbatches);

  virtual ~dgs_kv_cache_context();

  //
  // dgs_memory_context_i
  //

  bool mNext() override;
  bool apply() override;

  dgs_memory_status  mGetStatus() const override;
  const dgs_ubatch & mGetUbatch() const override;

  //
  // dgs_kv_cache_context specific API
  //

  uint32_t mGetNKv() const;

  ggml_type mTypeK() const;
  ggml_type mTypeV() const;

  // get views of the current state of the cache
  ggml_tensor * mGetK(ggml_context * pCtx, int32_t pIl) const;
  ggml_tensor * mGetV(ggml_context * pCtx, int32_t pIl) const;

  // store k_cur and v_cur in the cache based on the provided head location
  // note: the heads in k_cur and v_cur should be laid out contiguously in memory
  //   - k_cur  [n_embd_head_k, n_head_k, n_tokens]
  //   - k_idxs [n_tokens]
  //   - v_cur  [n_embd_head_v, n_head_v, n_tokens]
  //   - v_idxs [n_tokens] or [n_tokens*n_embd_v_gqa] depending if V cache is transposed
  ggml_tensor * mCpyK(ggml_context * pCtx, ggml_tensor * pKCur, ggml_tensor * pKIdxs, int32_t pIl) const;
  ggml_tensor * mCpyV(ggml_context * pCtx, ggml_tensor * pVCur, ggml_tensor * pVIdxs, int32_t pIl) const;

  // create destination indices for each head of the current batch for where it would be written in the KV cache
  // the indices address the global KV cache (not per stream) - this is not relevant for the user of this API, but
  //   helps understand the implementation logic of cpy_k and cpy_v
  ggml_tensor * mBuildInputKIdxs(ggml_context * pCtx, const dgs_ubatch & pUbatch) const;
  ggml_tensor * mBuildInputVIdxs(ggml_context * pCtx, const dgs_ubatch & pUbatch) const;

  ggml_tensor * mBuildInputKRot(ggml_context * pCtx) const;
  ggml_tensor * mBuildInputVRot(ggml_context * pCtx) const;

  void mSetInputKIdxs(ggml_tensor * pDst, const dgs_ubatch * pUbatch) const;
  void mSetInputVIdxs(ggml_tensor * pDst, const dgs_ubatch * pUbatch) const;

  void mSetInputKShift(ggml_tensor * pDst) const;
  void mSetInputKqMask(ggml_tensor * pDst, const dgs_ubatch * pUbatch, bool pCausalAttn) const;
  void mSetInputPosBucket(ggml_tensor * pDst, const dgs_ubatch * pUbatch) const;

  void mSetInputKRot(ggml_tensor * pDst) const;
  void mSetInputVRot(ggml_tensor * pDst) const;

  // see dgs_kv_cache::get_prev_tokens()
  void mGetPrevTokens(const dgs_ubatch & pUbatch, uint32_t n, std::vector<dgs_token> & pRes) const;

private:
  dgs_memory_status vStatus;

  dgs_kv_cache * vKv;
  dgs_context *  vLctx;

  //
  // update context
  //

  bool vDoShift = false;

  stream_copy_info vScInfo;

  //
  // batch processing context
  //

  // the index of the cur ubatch to process
  size_t vICur = 0;

  slot_info_vec_t lSinfos;

  std::vector<dgs_ubatch> lUbatches;

  //
  // data needed for building the compute graph for the current ubatch:
  //

  // a heuristic, to avoid attending the full cache if it is not yet utilized
  // as the cache gets filled, the benefit from this heuristic disappears
  int32_t vNKv;
};
