#pragma once

#include "dgs.h"

#include <vector>

struct dgs_vocab;
struct dgs_grammar;

// sampler chain

struct dgs_sampler_chain {
  dgs_sampler_chain_params vParams;

  // has .backend_init() been called?
  bool vIsInit = false;

  uint32_t vNNodes = 0;

  struct info {
    bool vIsBackend;

    dgs_sampler * vPtr;
  };

  std::vector<info> lSamplers;

  // pre-allocated buffer for dgs_sampler_sample to avoid repeated allocations
  std::vector<dgs_token_data> lCur;

  // timing

  mutable int64_t vTSampleUs;

  mutable int32_t vNSample;
};

uint32_t fDgsSamplerBackendNNodes(const dgs_sampler * pSampler);
void     fDgsSamplerBackendBegin(dgs_sampler * pSampler);

struct dgs_sampler * fDgsSamplerInitDryTesting(float                                       pDryMultiplier,
                                               float                                       pDryBase,
                                               int32_t                                     pDryAllowedLength,
                                               int32_t                                     pDryPenaltyLastN,
                                               const std::vector<std::vector<dgs_token>> & pSeqBreakers);
