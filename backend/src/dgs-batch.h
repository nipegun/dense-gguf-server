#pragma once

#include "dgs-arch.h"
#include "dgs-cparams.h"
#include "dgs.h"

#include <array>
#include <bitset>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// keep this struct lightweight
struct dgs_ubatch {
  bool mEqualSeqs() const { return vBEqualSeqs != 0; }

  // typical for M-RoPE cases:
  //   0 - sequential position of the tokens/embeddings in the sequence
  //   1 - y position in the image
  //   2 - x position in the image
  //   3 - other
  bool mIsPos2d() const {
    // TODO @ngxson : we may need to check for model arch when more models use >1 positions
    return vNPos >= 3;
  }

  // mixed: type picks token or embd per row, pos has n_pos sections for all rows
  bool mIsMixed() const { return type != nullptr; }

  uint32_t vBEqualSeqs;  // note: this is a boolean, but we use an int32_t for alignment
                         //       otherwise address sanitizer complains
  // TODO: whole_seqs for embeddings?

  uint32_t vNTokens;     // total tokens (n_seq_tokens * n_seqs)
  uint32_t vNSeqTokens;  // tokens per sequence set
  uint32_t vNSeqs;       // sequence sets in the ubatch
  uint32_t vNSeqsUnq;    // unique sequence ids in the ubatch
  uint32_t vNPos;        // number of position inputs for each token/embedding

  // seq_id_unq: unique sequence ids in the ubatch
  // seq_idx:    indices of the unique sequence ids in the ubatch in [0, n_seqs_unq)
  //             used for extracting sequence pooled embeddings

  //                          // size                     | idx | val
  dgs_token * token;     // [n_tokens]               | i   | id, token
  float *     vEmbd;     // [n_embd, n_tokens]       | i   | embd
  float *   vEmbdState;  // [n_embd_state, n_tokens] | i   | hidden state carried over from a previous stage (e.g. MTP)
  dgs_pos * vPos;        // [n_tokens*n_pos]         | i   | pos
  int32_t * vNSeqId;     // [n_tokens]         | i   | -
  dgs_seq_id ** vSeqId;  // [n_tokens]         | s   | s0, s1, seq_id
  dgs_seq_id *  vSeqIdUnq;       // [n_seqs_unq]       | s   | seq_id
  int32_t *     vSeqIdx;         // [DGS_MAX_SEQ]    | -   | seq_idx
  int8_t *      vOutput;         // [n_tokens]         | i   | -
  int8_t *      type;            // [n_tokens]         | i   | -     (mixed ubatch only, 0 - token, 1 - embd)
  int32_t *     vDecisionOrder;  // [n_tokens], NULL if no entry has one, see dgs_batch_ext_set_decision_order()

  struct data_t {
    std::vector<dgs_token>    token;
    std::vector<float>        lEmbd;
    std::vector<float>        lEmbdState;
    std::vector<dgs_pos>      lPos;
    std::vector<int32_t>      lNSeqId;
    std::vector<dgs_seq_id *> lSeqId;  // these point into the seq_id_data below
    std::vector<dgs_seq_id>   lSeqIdUnq;
    std::vector<int32_t>      lSeqIdx;
    std::vector<int8_t>       lOutput;
    std::vector<int8_t>       type;
    std::vector<int32_t>      lBatchIdxs;  // original batch index for each token
    std::vector<int32_t>      lDecisionOrder;

    std::vector<dgs_seq_id> lSeqIdData;
  };

  // the dgs_ubatch pointers above point to this data if set. otherwise - point to external non-owning data
  std::shared_ptr<data_t> data;
};

// crash if a mixed ubatch reaches code that expects only tokens or only embd
#define ASSERT_EMBD_OR_TOKEN(ubatch) \
  GGML_ASSERT(!(ubatch).mIsMixed() && "mixed token/embd ubatch is not supported here")

struct dgs_hparams;

// DFlash batches carry the fused target features at the encoder input width (n_embd_inp_enc size).
// Other batches carry token embeddings (n_embd_inp size).
size_t fDgsBatchExtSelectNEmbdInp(dgs_context_type pCtxType, llm_arch pArch, const dgs_hparams & pHparams);

// MTP contexts also take the target model's hidden state (n_embd_out size), 0 = no state input
size_t fDgsBatchExtSelectNEmbdState(dgs_context_type pCtxType, const dgs_hparams & pHparams);

struct dgs_batch_ext {
  const size_t     cNTokensMax;   // max number of tokens that can be stored in the batch
  const size_t     cNEmbdInp;     // decoder embd row width
  const size_t     cNEmbdInpEnc;  // encoder embd row width (e.g. eagle3/dflash extracted features)
  const size_t     cNEmbdState;   // state embd row width, 0 if the context takes no state
  const dgs_seq_id cNSeqMax;      // max number of sequences
  dgs_memory_i *   vMem;          // memory for position inference
  const dgs_token  cNVocab;       // max token ID that we accept
  const size_t     cNPosPerEmbd;

  // actual embd row width of this batch, set by the first set_token_embd()
  // must be either n_embd_inp or n_embd_inp_enc; encode/decode verify it against the graph input
  size_t vNEmbd = 0;

  struct token {
    dgs_token                                vId            = DGS_TOKEN_NULL;
    bool                                     vHasEmbd       = false;  // whether embd_off is set
    size_t                                   vEmbdOff       = 0;      // index offset in the embd array
    bool                                     vHasState      = false;  // whether state_off is set
    size_t                                   vStateOff      = 0;      // index offset in the state array
    bool                                     vOutput        = false;  // TODO: have dedicated output flags
    int32_t                                  vDecisionOrder = 0;      // see dgs_batch_ext_set_decision_order()
    std::unordered_set<dgs_seq_id>           sSeqIds;
    std::array<dgs_pos, GGML_MROPE_SECTIONS> aPos = { 0, 0, 0, 0 };
  };

  std::vector<token> lTokens;
  std::vector<float> lEmbd;
  std::vector<float> lState;

  dgs_batch_ext(dgs_context * pCtx);

  // build without a dgs_context, used by tests
  dgs_batch_ext(size_t         pNTokensMax,
                size_t         pNEmbdInp,
                size_t         pNEmbdInpEnc,
                dgs_seq_id     pNSeqMax,
                dgs_memory_i * pMem,
                dgs_token      pNVocab,
                size_t         pNPosPerEmbd);

  void clear();

  // add an entry with an undefined position
  // the caller must set it explicitly via set_token_pos()
  int32_t mAddToken(dgs_seq_id pSeqId);

  bool mAddSeq(int32_t pIdx, dgs_seq_id pSeqId);
  bool mSetTokenId(int32_t pIdx, dgs_token pId);
  bool mSetTokenEmbd(int32_t pIdx, dgs_embd pEmbdIn);
  bool mSetTokenState(int32_t pIdx, dgs_embd pStateIn);
  bool mSetTokenPos(int32_t pIdx, const dgs_pos * pPosIn);
  bool mSetOutput(int32_t pIdx, bool pOutputLast);
  bool mSetDecisionOrder(int32_t pIdx, int32_t pOrder);
};

// a helper for sanitizing, fulfilling and splitting a batch
class dgs_batch_allocr {
public:
  dgs_batch_allocr(uint32_t pNPosPerEmbd, bool pAllowMixed = false);

  // convert a dgs_batch_ext to internal dgs_batch and sanitize it
  bool init(const dgs_batch_ext & pBatchInp, const dgs_vocab & pVocab, bool pOutputAll);

  const dgs_batch & mGetBatch() const;

  uint32_t mGetNTokens() const;
  uint32_t mGetNOutputs() const;
  uint32_t mGetNUsed() const;

  // the array of output indices in the order they were encountered during the ubatch splitting
  std::vector<int32_t> & mGetOutIds();

  // min/max positions of each sequence in the current ubatch
  dgs_pos mSeqPosMin(dgs_seq_id pSeqId) const;
  dgs_pos mSeqPosMax(dgs_seq_id pSeqId) const;

  // call once before splitting the batch to reset the internal state
  void mSplitReset();

  // simple split, unknown number of sequence sets of unequal lengths
  dgs_ubatch mSplitSimple(uint32_t pNUbatch);

  // make ubatches of equal-length sequences sets
  // if sequential == true, the tokens in the ubatch will have increasing sequential sequence ids
  // n_keep_tail = minimum trailing tokens of a seq that must land in the same ubatch
  dgs_ubatch mSplitEqual(uint32_t pNUbatch, bool pSequential, uint32_t pNKeepTail);

  // sequence-set-wise split - each ubatch contains a single sequence-set
  // TODO: extend to support multiple _full_ sequences in the ubatch
  dgs_ubatch mSplitSeq(uint32_t pNUbatch);

  // a helper method for creating a well-defined ubatch of tokens
  // TODO: support embeddings if needed in the future
  dgs_ubatch mUbatchReserve(uint32_t pNSeqTokens, uint32_t pNSeqs);

private:
  void clear();

  // create the next ubatch based on the provided batch indices (idxs) and the number of sequence sets (n_seqs)
  // return dgs_ubatch.n_tokens == 0 if the entire batch was consumed
  dgs_ubatch mUbatchAdd(const std::vector<int32_t> & pIdxs, uint32_t pNSeqs, bool pEqualSeqs);

  // for debugging, start with DGS_BATCH_DEBUG=2
  void mUbatchPrint(const dgs_ubatch & pUbatch, int pDebug);

  dgs_batch vBatch;

  // only for debugging purposes
  const dgs_vocab * vVocab;

  // TODO: this is more of a temporary solution until we have a better way to handle multiple positions per token/embd
  const uint32_t cNPosPerEmbd;

  const bool cAllowMixed;

  uint32_t vNEmbd;
  uint32_t vNEmbdState;
  uint32_t vNSeqMax;
  uint32_t vNOutputs;

  std::vector<dgs_token>  lTokenVec;   // owned token IDs built from dgs_batch_ext
  std::vector<float>      lEmbdVec;    // owned embeddings built from dgs_batch_ext
  std::vector<int8_t>     lIsEmbdVec;  // mixed batch only (= 1 if embd, 0 if text token)
  std::vector<float>      lStateVec;  // owned state embeddings built from dgs_batch_ext, dgs_batch has no slot for them
  std::vector<dgs_seq_id> lSeqIdData;  // flat storage for seq_id pointers below

  std::vector<dgs_pos>      lPos;
  std::vector<int32_t>      lNSeqId;
  std::vector<dgs_seq_id *> lSeqId;
  std::vector<dgs_seq_id>   lSeqIdUnq;
  std::vector<int32_t>      lSeqIdx;
  std::vector<int8_t>       lOutput;
  std::vector<int32_t>      lDecisionOrder;  // empty if no entry has one

  using pos_set_t = std::set<dgs_pos>;
  using seq_cpl_t = std::vector<bool>;

  // helper flag to quickly determine if there are any coupled sequences in the batch
  bool vHasCpl = false;

  std::vector<pos_set_t> sSeqPos;  // seq_pos[s]: the set of positions in sequence s
  std::vector<seq_cpl_t> lSeqCpl;  // seq_cpl[s0][s1]: if sequence s0 is coupled to sequence s1

  using idx_vec_t = std::vector<int32_t>;
  using seq_set_t = std::bitset<DGS_MAX_SEQ>;

  std::vector<seq_set_t> lSeqSet;                       // seq_set[i]: the sequence set of token i

  std::unordered_map<seq_set_t, idx_vec_t> dSeqSetMap;  // the indices at which the sequence set appears

  // batch indices of the output
  std::vector<int32_t> lOutIds;

  uint32_t vNUsed;

  // used[i] indicates if token i has already been used in a previous ubatch
  std::vector<bool> lUsed;

  int vDebug;
};

// RAII translation layer: converts a dgs_batch (old API) into a dgs_batch_ext
struct dgs_batch_compat {
  dgs_batch_ext * vBatchExt;

  // n_embd_row is the embd row width of batch_inp, 0 = use the decoder width
  dgs_batch_compat(dgs_context * pCtx, const dgs_batch & pBatchInp, size_t pNEmbdRow = 0);
  ~dgs_batch_compat();

  // fill an existing dgs_batch_ext from a dgs_batch (old API)
  // note: this is called directly by the tests, skipping dgs_context creation
  static void init(dgs_batch_ext & pBatchExt, const dgs_batch & pBatchInp, size_t pNEmbdRow = 0);
};
