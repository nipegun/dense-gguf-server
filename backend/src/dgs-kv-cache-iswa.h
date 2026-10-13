#pragma once

#include "dgs-kv-cache.h"

#include <vector>

//
// dgs_kv_cache_iswa
//

// utilizes two instances of dgs_kv_cache
//   the first instance is for the non-SWA layers of the model and the second instance is for the SWA layers

class dgs_kv_cache_iswa : public dgs_memory_i {
public:
  dgs_kv_cache_iswa(const dgs_model &       pModel,
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
                    dgs_memory_t            pMemOther,
                    const layer_filter_cb & pFilter,
                    const layer_reuse_cb &  pReuse,
                    const layer_share_cb &  pShare);

  dgs_kv_cache_iswa(const dgs_model &       pModel,
                    const dgs_hparams &     pHparams,
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
                    dgs_memory_t            pMemOther,
                    const layer_filter_cb & pFilter,
                    const layer_reuse_cb &  pReuse,
                    const layer_share_cb &  pShare);

  ~dgs_kv_cache_iswa() = default;

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
  // dgs_kv_cache_iswa specific API
  //

  dgs_kv_cache * mGetBase() const;
  dgs_kv_cache * mGetSwa() const;

private:
  const bool cUnified;

  std::unique_ptr<dgs_kv_cache> vKvBase;
  std::unique_ptr<dgs_kv_cache> vKvSwa;
};

class dgs_kv_cache_iswa_context : public dgs_memory_context_i {
public:
  using slot_info_vec_t = dgs_kv_cache::slot_info_vec_t;

  // used for errors
  dgs_kv_cache_iswa_context(dgs_memory_status pStatus);

  // used to create a full-cache context
  dgs_kv_cache_iswa_context(dgs_kv_cache_iswa * pKv);

  // used to create an update context
  dgs_kv_cache_iswa_context(dgs_kv_cache_iswa * pKv, dgs_context * pLctx, bool pOptimize);

  // used to create a batch processing context from a batch
  dgs_kv_cache_iswa_context(dgs_kv_cache_iswa *     pKv,
                            slot_info_vec_t         pSinfosBase,
                            slot_info_vec_t         pSinfosSwa,
                            std::vector<dgs_ubatch> pUbatches);

  virtual ~dgs_kv_cache_iswa_context();

  //
  // dgs_memory_context_i
  //

  bool mNext() override;
  bool apply() override;

  dgs_memory_status  mGetStatus() const override;
  const dgs_ubatch & mGetUbatch() const override;

  //
  // dgs_kv_cache_iswa_context specific API
  //

  const dgs_kv_cache_context * mGetBase() const;
  const dgs_kv_cache_context * mGetSwa() const;

private:
  //dgs_kv_cache_iswa * kv;

  // the index of the next ubatch to process
  size_t vINext = 0;

  std::vector<dgs_ubatch> lUbatches;

  const dgs_memory_context_ptr cCtxBase;
  const dgs_memory_context_ptr cCtxSwa;

  const dgs_memory_status cStatus;
};
