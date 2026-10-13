#pragma once

#include "dgs.h"
#include "ggml.h"

#include <vector>

enum common_params_fit_status {
  cCommonParamsFitStatusSuccess = 0,  // found allocations that are projected to fit
  cCommonParamsFitStatusFailure = 1,  // could not find allocations that are projected to fit
  cCommonParamsFitStatusError = 2,  // a hard error occurred, e.g. because no model could be found at the specified path
};

// a second model that shares the devices of the main model, e.g. a draft model
//   - its context follows the context of the main model, so its memory is measured again whenever that context changes
//   - shares_model tells the fit that the weights are already counted in the main model, as for an MTP context
struct common_fit_extra_model {
  const char *         vPathModel;
  dgs_model_params *   vMparams;
  dgs_context_params * vCparams;
  bool                 vSharesModel;
};

// fits mparams and cparams to free device memory (assumes system memory is unlimited)
//   - returns true if the parameters could be successfully modified to fit device memory
//   - this function is NOT thread safe because it modifies the global dgs logger state
//   - only parameters that have the same value as in dgs_default_model_params are modified
//     with the exception of the context size which is modified if and only if equal to 0
common_params_fit_status fCommonFitParams(
  const char *         pPathModel,
  dgs_model_params *   pMparams,
  dgs_context_params * pCparams,
  float *              pTensorSplit,  // writable buffer for tensor split, needs at least dgs_max_devices elements
  dgs_model_tensor_buft_override *
    pTensorBuftOverrides,  // writable buffer for overrides, needs at least dgs_max_tensor_buft_overrides elements
  size_t *                       pMargins,    // margins of memory to leave per device in bytes
  uint32_t                       pNCtxMin,    // minimum context size to set when trying to reduce memory use
  const common_fit_extra_model * pExtra,      // model to fit alongside the main one, nullptr if there is none
  ggml_log_level                 pLogLevel);  // minimum log level to print during fitting, lower levels go to debug log

// print estimated memory to stdout
void fCommonFitPrint(const char * pPathModel, dgs_model_params * pMparams, dgs_context_params * pCparams);

void fCommonMemoryBreakdownPrint(const dgs_context * pCtx);

struct common_device_memory_data {
  int64_t vTotal;
  int64_t free;
  size_t  vModel;
  size_t  vContext;
  size_t  vCompute;
};

using common_device_memory_data_vec = std::vector<common_device_memory_data>;

// Load a model + context with no_alloc and return the per-device memory breakdown.
common_device_memory_data_vec fCommonGetDeviceMemoryData(const char *                      pPathModel,
                                                         const dgs_model_params *          pMparams,
                                                         const dgs_context_params *        pCparams,
                                                         std::vector<ggml_backend_dev_t> & pDevs,
                                                         uint32_t &                        pHpNgl,
                                                         uint32_t &                        pHpNCtxTrain,
                                                         uint32_t &                        pHpNExpert,
                                                         ggml_log_level                    pLogLevel);
