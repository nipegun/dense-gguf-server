#pragma once

#ifndef __cplusplus
#  error "This header is for C++ only"
#endif

#include "dgs.h"

#include <memory>

struct dgs_model_deleter {
  void operator()(dgs_model * pModel) { fDgsModelFree(pModel); }
};

struct dgs_context_deleter {
  void operator()(dgs_context * pContext) { fDgsFree(pContext); }
};

struct dgs_sampler_deleter {
  void operator()(dgs_sampler * pSampler) { fDgsSamplerFree(pSampler); }
};

struct dgs_adapter_lora_deleter {
  void operator()(dgs_adapter_lora * pAdapter) { fDgsAdapterLoraFree(pAdapter); }
};

struct dgs_batch_ext_deleter {
  void operator()(dgs_batch_ext * pBatch) { fDgsBatchExtFree(pBatch); }
};

typedef std::unique_ptr<dgs_model, dgs_model_deleter>               dgs_model_ptr;
typedef std::unique_ptr<dgs_context, dgs_context_deleter>           dgs_context_ptr;
typedef std::unique_ptr<dgs_sampler, dgs_sampler_deleter>           dgs_sampler_ptr;
typedef std::unique_ptr<dgs_adapter_lora, dgs_adapter_lora_deleter> dgs_adapter_lora_ptr;
typedef std::unique_ptr<dgs_batch_ext, dgs_batch_ext_deleter>       dgs_batch_ext_ptr;
