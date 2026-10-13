#pragma once

// this is a staging header for new dgs.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "dgs.h"

#include <cstdint>
#include <map>

// Reserve a new compute graph. It is valid until the next call to dgs_graph_reserve.
DGS_API struct ggml_cgraph * fDgsGraphReserve(struct dgs_context * pCtx,
                                              uint32_t             pNTokens,
                                              uint32_t             pNSeqs,
                                              uint32_t             pNOutputs);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct dgs_memory_breakdown_data {
  size_t vModel   = 0;  // memory allocated for the model
  size_t vContext = 0;  // memory allocated for the context
  size_t vCompute = 0;  // memory allocated for temporary compute buffers

  size_t mTotal() const { return vModel + vContext + vCompute; }
};

struct dgs_device_memory_data {
  int64_t                   vTotal;
  int64_t                   free;
  dgs_memory_breakdown_data vMb;
};

// TODO: convert to C-style data structure
using dgs_memory_breakdown = std::map<ggml_backend_buffer_type_t, dgs_memory_breakdown_data>;

DGS_API int32_t fDgsModelNExpert(const struct dgs_model * pModel);
DGS_API int32_t fDgsModelNDevices(const struct dgs_model * pModel);

DGS_API ggml_backend_dev_t fDgsModelGetDevice(const struct dgs_model * pModel, int pI);

DGS_API dgs_memory_breakdown fDgsGetMemoryBreakdown(const struct dgs_context * pCtx);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
DGS_API void fDgsSetEmbeddingsNextn(struct dgs_context * pCtx, bool value, bool pMasked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
DGS_API void fDgsSetNextnLayerOffset(struct dgs_context * pCtx, int32_t pOffset);

// Marks the entries that a decision head (clef, vllm-decision2) reads, the default is 0
// See  for details
// A run of entries with the same value is one span, spans must be separated by entries with value 0
// An option belongs to the last question before it
enum dgs_decision_order {
  cDgsDecisionOrderNone           = 0,  // not read by the head
  cDgsDecisionOrderQuestionNoul   = 1,  // text of a question
  cDgsDecisionOrderQuestionChoice = 2,
  cDgsDecisionOrderQuestionScore  = 3,
  cDgsDecisionOrderOption         = 4,  // text of an option
};

// The embeddings output has one value per entry: row i is the score of option i
DGS_API bool fDgsBatchExtSetDecisionOrder(struct dgs_batch_ext * pBatch, int32_t pIdx, enum dgs_decision_order pOrder);

// mirrors:
// DGS_API float * dgs_get_embeddings(struct dgs_context * ctx);
DGS_API float * fDgsGetEmbeddingsNextn(struct dgs_context * pCtx);

// DGS_API float * dgs_get_embeddings_ith(struct dgs_context * ctx, int32_t i);
DGS_API float * fDgsGetEmbeddingsNextnIth(struct dgs_context * pCtx, int32_t pI);

// Set whether the context outputs the input embeddings of a specific layer
DGS_API void fDgsSetEmbeddingsLayerInp(struct dgs_context * pCtx, uint32_t pLid, bool value);

// mirrors:
// DGS_API float * dgs_get_embeddings(struct dgs_context * ctx);
DGS_API float * fDgsGetEmbeddingsLayerInp(struct dgs_context * pCtx, uint32_t pLid);

DGS_API dgs_context * fDgsGetCtxOther(struct dgs_context * pCtx);

//
// model/context data extraction
//

DGS_API int32_t fDgsModelDflashSelectorTopK(const struct dgs_model * pModel);

// returns pointer to the target-model layer indices
DGS_API const int32_t * fDgsModelTargetLayerIds(const struct dgs_model * pModel);
// returns the number of extracted layers from target model
DGS_API uint32_t        fDgsModelTargetLayerIdsN(const struct dgs_model * pModel);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
DGS_API uint32_t fDgsModelGetTokEmbd(const struct dgs_model * pModel, float * pOut);

// Allocated bytes of attached MTP weights, grouped by buffer type.
using dgs_model_mtp_breakdown = std::map<ggml_backend_buffer_type_t, size_t>;
DGS_API dgs_model_mtp_breakdown fDgsModelGetMtpBreakdown(const struct dgs_model * pModel);
