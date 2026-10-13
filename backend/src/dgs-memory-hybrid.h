#pragma once

#include "dgs-batch.h"
#include "dgs-graph.h"
#include "dgs-kv-cache.h"
#include "dgs-memory-recurrent.h"
#include "dgs-memory.h"

#include <memory>
#include <vector>

//
// dgs_memory_hybrid
//

// utilizes instances of dgs_memory_recurrent and dgs_kv_cache to
//   support models where each layer may be either attention-based or recurrent

class dgs_memory_hybrid : public dgs_memory_i {
public:
  dgs_memory_hybrid(const dgs_model &       pModel,
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
                    const layer_filter_cb & pFilterAttn = nullptr,
                    const layer_filter_cb & pFilterRecr = nullptr);

  ~dgs_memory_hybrid() = default;

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
  // dgs_memory_hybrid specific API
  //

  dgs_kv_cache *         mGetMemAttn() const;
  dgs_memory_recurrent * mGetMemRecr() const;

private:
  const dgs_hparams & vHparams;

  const std::unique_ptr<dgs_kv_cache>         cMemAttn;
  const std::unique_ptr<dgs_memory_recurrent> cMemRecr;
};

class dgs_memory_hybrid_context : public dgs_memory_context_i {
public:
  using slot_info_vec_t = dgs_kv_cache::slot_info_vec_t;

  // init failure
  explicit dgs_memory_hybrid_context(dgs_memory_status pStatus);

  // init full
  explicit dgs_memory_hybrid_context(dgs_memory_hybrid * pMem);

  // init update
  explicit dgs_memory_hybrid_context(dgs_memory_hybrid * pMem, dgs_context * pLctx, bool pOptimize);

  // init success
  dgs_memory_hybrid_context(dgs_memory_hybrid * pMem, slot_info_vec_t pSinfosAttn, std::vector<dgs_ubatch> pUbatches);

  ~dgs_memory_hybrid_context() = default;

  bool mNext() override;
  bool apply() override;

  dgs_memory_status  mGetStatus() const override;
  const dgs_ubatch & mGetUbatch() const override;

  //
  // dgs_memory_hybrid_context
  //

  const dgs_kv_cache_context *         mGetAttn() const;
  const dgs_memory_recurrent_context * mGetRecr() const;

private:
  // the index of the next ubatch to process
  size_t vINext = 0;

  std::vector<dgs_ubatch> lUbatches;

  const dgs_memory_context_ptr cCtxAttn;
  const dgs_memory_context_ptr cCtxRecr;

  const dgs_memory_status cStatus;
};
