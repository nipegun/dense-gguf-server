#pragma once

#include "dgs-kv-cache.h"

#include <vector>

// dgs_kv_cache_msa

// uses two instances of dgs_kv_cache, one for K/V tensors, and one for the MSA indexer tensors
// both receive identical sequence operations and identical ubatches, so their cell layouts stay in synced.
// the context also exposes per-ubatch pos - cell translation maps populated from dgs_kv_cells via
// dgs_kv_cache::get_cells(), which the model graph uses to run MSA block selection in position space

class dgs_kv_cache_msa : public dgs_memory_i {
public:
  dgs_kv_cache_msa(const dgs_model &       pModel,
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
                   const layer_filter_cb & pFilter,
                   const layer_filter_cb & pFilterIdx,
                   const layer_reuse_cb &  pReuse);

  ~dgs_kv_cache_msa() = default;

  // dgs_memory_i

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

  // dgs_kv_cache_msa specific API

  dgs_kv_cache * mGetBase() const;
  dgs_kv_cache * mGetIdx() const;

  uint32_t mGetNPad() const { return cNPad; }

  uint32_t mGetNSeqMax() const { return cNSeqMax; }

  uint32_t mGetNSwa() const { return cNSwa; }

  dgs_swa_type mGetSwaType() const { return cSwaType; }

private:
  // keep the indexer KV cache hparams instance here as dgs_kv_cache stores only a reference
  dgs_hparams vHparamsIdx;

  const uint32_t cNStream = 1;
  const uint32_t cNSeqMax = 1;
  const uint32_t cNPad    = 1;

  const uint32_t     cNSwa    = 0;
  const dgs_swa_type cSwaType = cDgsSwaTypeNone;

  std::unique_ptr<dgs_kv_cache> vKvBase;
  std::unique_ptr<dgs_kv_cache> vKvIdx;
};

class dgs_kv_cache_msa_context : public dgs_memory_context_i {
public:
  using slot_info_vec_t = dgs_kv_cache::slot_info_vec_t;

  // used for errors
  dgs_kv_cache_msa_context(dgs_memory_status pStatus);

  // used to create a full-cache context
  dgs_kv_cache_msa_context(dgs_kv_cache_msa * pKv);

  // used to create an update context
  dgs_kv_cache_msa_context(dgs_kv_cache_msa * pKv, dgs_context * pLctx, bool pOptimize);

  // used to create a batch processing context from a batch
  dgs_kv_cache_msa_context(dgs_kv_cache_msa *      pKv,
                           slot_info_vec_t         pSinfosBase,
                           slot_info_vec_t         pSinfosIdx,
                           std::vector<dgs_ubatch> pUbatches);

  virtual ~dgs_kv_cache_msa_context();

  // dgs_memory_context_i

  bool mNext() override;
  bool apply() override;

  dgs_memory_status  mGetStatus() const override;
  const dgs_ubatch & mGetUbatch() const override;

  // dgs_kv_cache_msa_context specific API

  const dgs_kv_cache_context * mGetBase() const;
  const dgs_kv_cache_context * mGetIdx() const;

  // max position currently present in the cache plus one, padded MSA blocks are defined over token positions
  // so the block-selection tensors are sized by this value rather than by the number of cells
  uint32_t mGetNPos() const;

  // position <-> cell translation maps, populated from the base cache cells
  // the model graph relates cache contents to token positions only through these per ubatch inputs
  // value for empty or other-sequence cells is 0 so consumers must mask them
  void mSetInputCellPos(ggml_tensor * pDst, const dgs_ubatch * pUbatch, int32_t pDiv) const;
  // positions without a cell map to cell 0, consumers must mask them assumes one sequence per stream
  void mSetInputPosSlot(ggml_tensor * pDst, const dgs_ubatch * pUbatch) const;
  void mSetInputPosMask(ggml_tensor * pDst, const dgs_ubatch * pUbatch) const;

private:
  dgs_kv_cache_msa * vKv;

  // the index of the next ubatch to process
  size_t vINext = 0;

  std::vector<dgs_ubatch> lUbatches;

  const dgs_memory_context_ptr cCtxBase;
  const dgs_memory_context_ptr cCtxIdx;

  const dgs_memory_status cStatus;
};
