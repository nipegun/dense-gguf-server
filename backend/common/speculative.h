#pragma once

#include "common.h"
#include "dgs.h"

struct common_speculative;

// comma separated list the provided types
std::string fCommonSpeculativeTypeNameStr(const std::vector<enum common_speculative_type> & pTypes);

// comma separated list of all types
const char * fCommonSpeculativeAllTypesStr();

// parse user provided types
std::vector<enum common_speculative_type> fCommonSpeculativeTypesFromNames(const std::vector<std::string> & pNames);

// infer the spec types from the GGUF metadata of a draft model; empty if unknown
std::vector<enum common_speculative_type> fCommonSpeculativeTypesFromGguf(const std::string & pPath);

// convert string to type
enum common_speculative_type fCommonSpeculativeTypeFromName(const std::string & name);

// convert type to string
std::string fCommonSpeculativeTypeToStr(enum common_speculative_type type);

// return the max number of draft tokens based on the speculative parameters
int32_t fCommonSpeculativeNMax(const common_params_speculative * pSpec);

// return the max number of draft tokens from the initialized implementations
int32_t fCommonSpeculativeNMax(const common_speculative * pSpec);

// validate and resolve the unconditional synthetic acceptance rates
std::vector<double> fCommonSpeculativeSynthRatesResolve(const common_params_speculative * pSpec, int32_t pNMax);

// return the conditional synthetic acceptance probabilities
const std::vector<double> & fCommonSpeculativeGetSynthProbs(const common_speculative * pSpec);

common_params fCommonBaseParamsToSpeculative(const common_params & pParams);

struct common_speculative_output_limits {
  int32_t vTotal;
  int32_t vPerSeq;
};

// return the output limits needed for speculative decoding
common_speculative_output_limits fCommonSpeculativeGetOutputLimits(int32_t pNBatch,
                                                                   int32_t pNParallel,
                                                                   int32_t pNDraft);

// return true if the target and draft models have compatible vocabs
bool fCommonSpeculativeAreCompatible(const dgs_model * pModelTgt, const dgs_model * pModelDft);

common_speculative * fCommonSpeculativeInit(common_params_speculative & pParams, uint32_t pNSeq);

void fCommonSpeculativeFree(common_speculative * pSpec);

struct common_speculative_draft_params {
  // this flag is used to chain the drafts through all the available implementations
  // after the first successful draft from an implementation, we set it
  //   to false to prevent further drafts for that sequence
  // at the end of the draft() call, all drafting flags will be reset to false
  bool vDrafting = false;

  // overrides individual configurations (-1 disabled)
  // can be used to constraint the max draft based on the remaining context size
  int32_t vNMax = -1;

  dgs_pos   vPos0;
  dgs_token vIdLast;

  // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
  const dgs_tokens * lPrompt;

  // the generated draft from the last _draft() call
  dgs_tokens * lResult;

  // candidate distribution per drafted token; set it to make draft-simple and draft-mtp sample
  std::vector<std::vector<dgs_token_data>> * lResultQ = nullptr;

  // the target's temp and seed, read only when the drafter samples probabilistically
  float    vTemp = 1.0f;
  uint32_t vSeed = DGS_DEFAULT_SEED;
};

common_speculative_draft_params & fCommonSpeculativeGetDraftParams(common_speculative * pSpec, dgs_seq_id pSeqId);

// optionally call once at the beginning of a new generation
void fCommonSpeculativeBegin(common_speculative * pSpec, dgs_seq_id pSeqId, const dgs_tokens & pPrompt);

// process the batch and update the internal state of the speculative context
bool fCommonSpeculativeProcess(common_speculative * pSpec, const common_batch & pBatch);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void fCommonSpeculativeDraft(common_speculative * pSpec);

// informs the speculative context that n_accepted tokens were accepted by the target model
void fCommonSpeculativeAccept(common_speculative * pSpec, dgs_seq_id, uint16_t pNAccepted);

// (optional) get/set internal state
bool fCommonSpeculativeGetState(common_speculative * pSpec, dgs_seq_id pSeqId, std::vector<uint8_t> & data);
void fCommonSpeculativeSetState(common_speculative * pSpec, dgs_seq_id pSeqId, const std::vector<uint8_t> & data);

// print statistics about the speculative decoding
void fCommonSpeculativePrintStats(const common_speculative * pSpec);

struct common_speculative_deleter {
  void operator()(common_speculative * pS) { fCommonSpeculativeFree(pS); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

struct common_speculative_init_result {
  common_speculative_init_result(common_params & pParams, dgs_model * pModelTgt, dgs_context * pCtxTgt);
  ~common_speculative_init_result();

  dgs_model *   mModel();
  dgs_context * mContext();

private:
  struct impl;
  std::unique_ptr<impl> vPimpl;
};

using common_speculative_init_result_ptr = std::unique_ptr<common_speculative_init_result>;

common_speculative_init_result_ptr fCommonSpeculativeInitFromParams(common_params & pParams,
                                                                    dgs_model *     pModelTgt,
                                                                    dgs_context *   pCtxTgt);
