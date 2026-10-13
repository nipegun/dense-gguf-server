#pragma once

#include "dgs-batch.h"
#include "dgs-graph.h"
#include "dgs-memory.h"

#include <map>
#include <set>
#include <vector>

//
// dgs_memory_recurrent
//

// TODO: extract the cache state used for graph computation into dgs_memory_recurrent_context_i
//       see the implementation of dgs_kv_cache_context_i for an example how to do it
class dgs_memory_recurrent : public dgs_memory_i {
public:
  dgs_memory_recurrent(const dgs_model &       pModel,
                       ggml_type               pTypeR,
                       ggml_type               pTypeS,
                       bool                    pOffload,
                       uint32_t                pMemSize,
                       uint32_t                pNSeqMax,
                       uint32_t                pNRsSeq,
                       const layer_filter_cb & pFilter);

  ~dgs_memory_recurrent() = default;

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

  dgs_pos mSeqPosMin(dgs_seq_id pSeqId) const override;
  dgs_pos mSeqPosMax(dgs_seq_id pSeqId) const override;

  std::map<ggml_backend_buffer_type_t, size_t> mMemoryBreakdown() const override;

  bool mPrepare(const std::vector<dgs_ubatch> & pUbatches);

  // find a contiguous slot of memory cells and emplace the ubatch there
  bool mFindSlot(const dgs_ubatch & pUbatch);

  bool mGetCanShift() const override;

  // state write/load

  void mStateWrite(dgs_io_write_i & pIo, dgs_seq_id pSeqId = -1, dgs_state_seq_flags pFlags = 0) const override;
  void mStateRead(dgs_io_read_i & pIo, dgs_seq_id pSeqId = -1, dgs_state_seq_flags pFlags = 0) override;

  uint32_t vHead = 0;  // the location where the batch will be placed in the cache (see find_slot())
  uint32_t size  = 0;  // total number of cells, shared across all sequences
  uint32_t vUsed = 0;  // used cells (i.e. at least one seq_id)

  // number of recurrent-state snapshots per seq for rollback; tensors are widened to (1 + n_rs_seq) groups
  uint32_t vNRsSeq = 0;

  // per-seq rollback index
  std::vector<uint32_t> lRsIdx;

  void mSetRsIdx(dgs_seq_id pSeqId, uint32_t pIdx);

  // computed before each graph build
  uint32_t n = 0;

  // first zero-ed state
  int32_t vRsZ = -1;

  // TODO: optimize for recurrent state needs
  struct mem_cell {
    dgs_pos vPos  = -1;
    int32_t vSrc  = -1;  // used to know where states should be copied from
    int32_t vSrc0 = -1;  // like src, but only used when setting the inputs (allowing to copy once)
    int32_t vTail = -1;

    std::set<dgs_seq_id> sSeqId;

    bool mHasSeqId(const dgs_seq_id & pId) const { return sSeqId.find(pId) != sSeqId.end(); }

    bool mIsEmpty() const { return sSeqId.empty(); }

    bool mIsSameSeq(const mem_cell & pOther) const { return sSeqId == pOther.sSeqId; }
  };

  std::vector<mem_cell> lCells;

  // per layer
  std::vector<ggml_tensor *> lRL;
  std::vector<ggml_tensor *> lSL;
  // a second conv history that must stay replicated across devices, so it cannot share the r row
  std::vector<ggml_tensor *> lPL;

private:
  //const dgs_model & model;
  const dgs_hparams & vHparams;

  const uint32_t cNSeqMax = 1;

  // ggml contexts for the KV cache along with the allocated backend buffers:
  std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> lCtxsBufs;

  // true if no layers - can happen if the layer filter removes all layers
  bool mIsEmpty() const;

  size_t mTotalSize() const;

  size_t mSizeRBytes() const;
  size_t mSizeSBytes() const;
  size_t mSizePBytes() const;

  void mStateWriteMeta(dgs_io_write_i &                                   pIo,
                       const std::vector<std::pair<uint32_t, uint32_t>> & pCellRanges,
                       dgs_seq_id                                         pSeqId = -1) const;
  void mStateWriteData(dgs_io_write_i & pIo, const std::vector<std::pair<uint32_t, uint32_t>> & pCellRanges) const;

  bool mStateReadMeta(dgs_io_read_i & pIo, uint32_t pCellCount, dgs_seq_id pDestSeqId = -1);
  bool mStateReadData(dgs_io_read_i & pIo, uint32_t pCellCount);

  void mStateClear(dgs_seq_id pSeqId, uint32_t pCellHead, uint32_t pCellCount);
};

class dgs_memory_recurrent_context : public dgs_memory_context_i {
public:
  // used for errors
  dgs_memory_recurrent_context(dgs_memory_status pStatus);

  // used to create a full-cache or update context
  dgs_memory_recurrent_context(dgs_memory_recurrent * pMem);

  // used to create a batch processing context from a batch
  dgs_memory_recurrent_context(dgs_memory_recurrent * pMem, std::vector<dgs_ubatch> pUbatches);

  virtual ~dgs_memory_recurrent_context();

  //
  // dgs_memory_context_i
  //

  bool mNext() override;
  bool apply() override;

  dgs_memory_status  mGetStatus() const override;
  const dgs_ubatch & mGetUbatch() const override;

  //
  // dgs_memory_recurrent_context specific API
  //

  uint32_t mGetNRs() const;
  uint32_t mGetHead() const;
  int32_t  mGetRsZ() const;
  uint32_t mGetSize() const;

  ggml_tensor * mGetRL(int32_t pIl) const;
  ggml_tensor * mGetSL(int32_t pIl) const;
  ggml_tensor * mGetPL(int32_t pIl) const;

  int32_t mSCopy(int pI) const;

private:
  const dgs_memory_status cStatus;

  dgs_memory_recurrent * vMem;

  size_t vINext = 0;

  std::vector<dgs_ubatch> lUbatches;

  //
  // data needed for building the compute graph for the current ubatch:
  // TODO: extract all the state like `head` and `n` here
  //

  const bool cIsFull = false;
};
