#pragma once

#include "dgs-kv-cache.h"

#include <vector>

//
// dgs_kv_cache_dsa
//

// utilizes two instances of dgs_kv_cache:
// - the first instance is for caching key tensors of the model,
// - the second instance is for caching lightning indexer key tensors

class dgs_kv_cache_dsa : public dgs_memory_i {
public:
  dgs_kv_cache_dsa(const dgs_model &       pModel,
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
                   const layer_filter_cb & pFilterMla,
                   const layer_filter_cb & pFilterLid,
                   const layer_reuse_cb &  pReuse);

  ~dgs_kv_cache_dsa() = default;

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
  // dgs_kv_cache_dsa specific API
  //

  dgs_kv_cache * mGetMla() const;
  dgs_kv_cache * mGetLid() const;

private:
  // we keep indexer KV cache hparams instance here as dgs_kv_cache stores only reference to it
  dgs_hparams    vHparamsLid;
  const uint32_t cNStream = 1;

  std::unique_ptr<dgs_kv_cache> vKvMla;
  std::unique_ptr<dgs_kv_cache> vKvLid;
};

class dgs_kv_cache_dsa_context : public dgs_memory_context_i {
public:
  using slot_info_vec_t = dgs_kv_cache::slot_info_vec_t;

  // used for errors
  dgs_kv_cache_dsa_context(dgs_memory_status pStatus);

  // used to create a full-cache context
  dgs_kv_cache_dsa_context(dgs_kv_cache_dsa * pKv);

  // used to create an update context
  dgs_kv_cache_dsa_context(dgs_kv_cache_dsa * pKv, dgs_context * pLctx, bool pOptimize);

  // used to create a batch processing context from a batch
  dgs_kv_cache_dsa_context(dgs_kv_cache_dsa *      pKv,
                           slot_info_vec_t         pSinfosBase,
                           slot_info_vec_t         pSinfosIk,
                           std::vector<dgs_ubatch> pUbatches);

  virtual ~dgs_kv_cache_dsa_context();

  //
  // dgs_memory_context_i
  //

  bool mNext() override;
  bool apply() override;

  dgs_memory_status  mGetStatus() const override;
  const dgs_ubatch & mGetUbatch() const override;

  //
  // dgs_kv_cache_dsa_context specific API
  //

  const dgs_kv_cache_context * mGetMla() const;
  const dgs_kv_cache_context * mGetLid() const;

private:
  //dgs_kv_cache_dsa * kv;

  // the index of the next ubatch to process
  size_t vINext = 0;

  std::vector<dgs_ubatch> lUbatches;

  const dgs_memory_context_ptr cCtxMla;
  const dgs_memory_context_ptr cCtxLid;

  const dgs_memory_status cStatus;
};
