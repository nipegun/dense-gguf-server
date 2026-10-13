#pragma once

#include "dgs-graph.h"
#include "dgs.h"

#include <functional>
#include <map>
#include <memory>

struct dgs_ubatch;

class dgs_batch_allocr;

class dgs_io_write_i;
class dgs_io_read_i;

struct dgs_memory_params {
  // kv cache
  ggml_type vTypeK;
  ggml_type vTypeV;

  // use full-size SWA cache
  bool vSwaFull;

  dgs_context_type vCtxType;

  dgs_memory_t vMemOther;
};

enum dgs_memory_status {
  cDgsMemoryStatusSuccess = 0,
  cDgsMemoryStatusNoUpdate,
  cDgsMemoryStatusFailedPrepare,
  cDgsMemoryStatusFailedCompute,
};

// helper function for combining the status of two memory contexts
// useful for implementing hybrid memory types (e.g. iSWA)
dgs_memory_status fDgsMemoryStatusCombine(dgs_memory_status pS0, dgs_memory_status pS1);

// helper function for checking if a memory status indicates a failure
bool fDgsMemoryStatusIsFail(dgs_memory_status pStatus);

// the interface for managing the memory context during batch processing
// this interface is implemented per memory type. see:
//   - dgs_kv_cache_context
//   - dgs_kv_cache_iswa_context
//   ...
//
// the only method that should mutate the memory and the memory context is dgs_memory_i::apply()
struct dgs_memory_context_i {
  virtual ~dgs_memory_context_i() = default;

  // consume the current ubatch from the context and proceed to the next one
  // return false if we are done
  virtual bool mNext() = 0;

  // apply the memory state for the current ubatch to the memory object
  // return false on failure
  virtual bool apply() = 0;

  // get the current ubatch
  virtual const dgs_ubatch & mGetUbatch() const = 0;

  // get the status of the memory context - used for error handling and checking if any updates would be applied
  virtual dgs_memory_status mGetStatus() const = 0;
};

using dgs_memory_context_ptr = std::unique_ptr<dgs_memory_context_i>;

// general concept of LLM memory
// the KV cache is a type of LLM memory, but there can be other types
struct dgs_memory_i {
  // this callback is used to filter out layers that should not be included in the cache
  using layer_filter_cb = std::function<bool(int32_t pIl)>;

  // this callback is used to specify which layers should reuse memory from other layers
  // return negative value to indicate that the layer il should not reuse memory
  using layer_reuse_cb = std::function<int32_t(int32_t pIl)>;

  using layer_share_cb = std::function<int32_t(int32_t pIl)>;

  virtual ~dgs_memory_i() = default;

  // split the input batch into a set of ubatches and verify that they can fit into the cache
  // return a context object containing the ubatches and memory state required to process them
  // check the dgs_memory_context_i::get_status() for the result
  // embd_all: keep all the tokens of a sequence in one ubatch (all tokens are output, or the attention is non-causal)
  virtual dgs_memory_context_ptr mInitBatch(dgs_batch_allocr & pBalloc, uint32_t pNUbatch, bool pEmbdAll) = 0;

  // simulate full cache, used for allocating worst-case compute buffers
  virtual dgs_memory_context_ptr mInitFull() = 0;

  // prepare for any pending memory updates, such as shifts, copies, etc.
  // status == DGS_MEMORY_STATUS_NO_UPDATE if there is nothing to update
  virtual dgs_memory_context_ptr mInitUpdate(dgs_context * pLctx, bool pOptimize) = 0;

  // getters
  virtual bool mGetCanShift() const = 0;

  //
  // ops
  //

  // if data == true, the data buffers will also be cleared together with the metadata
  virtual void clear(bool data) = 0;

  virtual bool mSeqRm(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1)                          = 0;
  virtual void mSeqCp(dgs_seq_id pSeqIdSrc, dgs_seq_id pSeqIdDst, dgs_pos pP0, dgs_pos pP1) = 0;
  virtual void mSeqKeep(dgs_seq_id pSeqId)                                                  = 0;
  virtual void mSeqAdd(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1, dgs_pos pShift)         = 0;
  virtual void mSeqDiv(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1, int pD)                 = 0;

  virtual dgs_pos mSeqPosMin(dgs_seq_id pSeqId) const = 0;
  virtual dgs_pos mSeqPosMax(dgs_seq_id pSeqId) const = 0;

  virtual std::map<ggml_backend_buffer_type_t, size_t> mMemoryBreakdown() const = 0;

  //
  // state write/read
  //

  virtual void mStateWrite(dgs_io_write_i & pIo, dgs_seq_id pSeqId = -1, dgs_state_seq_flags pFlags = 0) const = 0;
  virtual void mStateRead(dgs_io_read_i & pIo, dgs_seq_id pSeqId = -1, dgs_state_seq_flags pFlags = 0)         = 0;
};

using dgs_memory_ptr = std::unique_ptr<dgs_memory_i>;
