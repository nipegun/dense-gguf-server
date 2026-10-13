#include "fit.h"

#include "../src/dgs-ext.h"
#include "json.h"
#include "log.h"

#include <array>
#include <cassert>
#include <cinttypes>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// this enum is only used in dgs_params_fit_impl but needs to be defined outside of it to fix a Windows compilation issue
// enum to identify part of a layer for distributing its tensors:
enum common_layer_fraction_t {
  cLayerFractionNone = 0,  // nothing
  cLayerFractionAttn = 1,  // attention
  cLayerFractionUp   = 2,  // attention + up
  cLayerFractionGate = 3,  // attention + up + gate
};

class common_params_fit_exception : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

static std::vector<dgs_device_memory_data> fCommonGetDeviceMemoryDataImpl(const char *                      pPathModel,
                                                                          const dgs_model_params *          pMparams,
                                                                          const dgs_context_params *        pCparams,
                                                                          std::vector<ggml_backend_dev_t> & pDevs,
                                                                          uint32_t &                        pHpNgl,
                                                                          uint32_t &     pHpNCtxTrain,
                                                                          uint32_t &     pHpNExpert,
                                                                          ggml_log_level pLogLevel) {
  struct user_data_t {
    struct {
      ggml_log_callback vCallback;
      void *            vUserData;
    } vOriginalLogger;

    ggml_log_level vMinLevel;  // prints below this log level go to debug log
  };

  user_data_t vUd;
  fDgsLogGet(&vUd.vOriginalLogger.vCallback, &vUd.vOriginalLogger.vUserData);
  vUd.vMinLevel = pLogLevel;

  fDgsLogSet(
    [](ggml_log_level pLevel, const char * pText, void * pUserData) {
      const user_data_t *  vUd       = (const user_data_t *) pUserData;
      const ggml_log_level cLevelEff = pLevel >= vUd->vMinLevel ? pLevel : GGML_LOG_LEVEL_DEBUG;
      vUd->vOriginalLogger.vCallback(cLevelEff, pText, vUd->vOriginalLogger.vUserData);
    },
    &vUd);

  dgs_model_params vMparamsCopy = *pMparams;
  vMparamsCopy.vNoAlloc         = true;
  vMparamsCopy.vLoadMode        = cDgsLoadModeNone;

  dgs_model * vModel = fDgsModelLoadFromFile(pPathModel, vMparamsCopy);
  if (vModel == nullptr) {
    fDgsLogSet(vUd.vOriginalLogger.vCallback, vUd.vOriginalLogger.vUserData);
    throw std::runtime_error("failed to load model");
  }

  dgs_context * vCtx = fDgsInitFromModel(vModel, *pCparams);
  if (vCtx == nullptr) {
    fDgsModelFree(vModel);
    fDgsLogSet(vUd.vOriginalLogger.vCallback, vUd.vOriginalLogger.vUserData);
    throw std::runtime_error("failed to create dgs_context from model");
  }

  const size_t                        cNd = fDgsModelNDevices(vModel);
  std::vector<dgs_device_memory_data> lRet(cNd + 1);

  dgs_memory_breakdown dMemoryBreakdown = fDgsGetMemoryBreakdown(vCtx);

  for (const auto & [buft, mb] : dMemoryBreakdown) {
    if (ggml_backend_buft_is_host(buft)) {
      lRet.back().vMb.vModel += mb.vModel;
      lRet.back().vMb.vContext += mb.vContext;
      lRet.back().vMb.vCompute += mb.vCompute;
      continue;
    }

    ggml_backend_dev_t vDev = ggml_backend_buft_get_device(buft);
    if (!vDev) {
      continue;
    }
    for (size_t vI = 0; vI < cNd; vI++) {
      if (vDev == fDgsModelGetDevice(vModel, vI)) {
        lRet[vI].vMb.vModel += mb.vModel;
        lRet[vI].vMb.vContext += mb.vContext;
        lRet[vI].vMb.vCompute += mb.vCompute;
        break;
      }
    }
  }

  {
    ggml_backend_dev_t vCpuDev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (vCpuDev == nullptr) {
      throw std::runtime_error("no CPU backend found");
    }
    size_t free;
    size_t vTotal;
    ggml_backend_dev_memory(vCpuDev, &free, &vTotal);
    lRet.back().free   = free;
    lRet.back().vTotal = vTotal;
  }
  for (size_t vI = 0; vI < cNd; vI++) {
    ggml_backend_dev_t vDev = fDgsModelGetDevice(vModel, vI);

    size_t free;
    size_t vTotal;
    ggml_backend_dev_memory(vDev, &free, &vTotal);

    // Some non-GPU accelerator backends, such as BLAS, report 0/0 and rely on
    // the host-memory fallback. For GPU-like backends, keep 0/0 so --fit does
    // not assign anything to a device with an unknown memory budget.
    if (free == 0 && vTotal == 0) {
      const enum ggml_backend_dev_type type = ggml_backend_dev_type(vDev);
      if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
        LOG_WRN("%s: device %s did not report memory; --fit will not use it\n", __func__, ggml_backend_dev_name(vDev));
      } else {
        free   = lRet.back().free;
        vTotal = lRet.back().vTotal;
      }
    }
    lRet[vI].free   = free;
    lRet[vI].vTotal = vTotal;
  }

  pDevs.clear();
  for (int vI = 0; vI < fDgsModelNDevices(vModel); vI++) {
    pDevs.push_back(fDgsModelGetDevice(vModel, vI));
  }

  pHpNgl = fDgsModelNLayer(vModel);
  if (pMparams->vLoadMtp) {
    pHpNgl += fDgsModelNLayerNextn(vModel);
  }
  pHpNCtxTrain = fDgsModelNCtxTrain(vModel);
  pHpNExpert   = fDgsModelNExpert(vModel);

  fCommonMemoryBreakdownPrint(vCtx);

  fDgsFree(vCtx);
  fDgsModelFree(vModel);
  fDgsLogSet(vUd.vOriginalLogger.vCallback, vUd.vOriginalLogger.vUserData);

  return lRet;
}

common_device_memory_data_vec fCommonGetDeviceMemoryData(const char *                      pPathModel,
                                                         const dgs_model_params *          pMparams,
                                                         const dgs_context_params *        pCparams,
                                                         std::vector<ggml_backend_dev_t> & pDevs,
                                                         uint32_t &                        pHpNgl,
                                                         uint32_t &                        pHpNCtxTrain,
                                                         uint32_t &                        pHpNExpert,
                                                         ggml_log_level                    pLogLevel) {
  std::vector<dgs_device_memory_data> impl =
    fCommonGetDeviceMemoryDataImpl(pPathModel, pMparams, pCparams, pDevs, pHpNgl, pHpNCtxTrain, pHpNExpert, pLogLevel);

  common_device_memory_data_vec lRet(impl.size());
  for (size_t vI = 0; vI < impl.size(); vI++) {
    lRet[vI].vTotal   = impl[vI].vTotal;
    lRet[vI].free     = impl[vI].free;
    lRet[vI].vModel   = impl[vI].vMb.vModel;
    lRet[vI].vContext = impl[vI].vMb.vContext;
    lRet[vI].vCompute = impl[vI].vMb.vCompute;
  }
  return lRet;
}

static void fCommonParamsFitImpl(const char *                            pPathModel,
                                 struct dgs_model_params *               pMparams,
                                 struct dgs_context_params *             pCparams,
                                 float *                                 pTensorSplit,
                                 struct dgs_model_tensor_buft_override * pTensorBuftOverrides,
                                 size_t *                                pMarginsS,
                                 uint32_t                                pNCtxMin,
                                 const common_fit_extra_model *          pExtra,
                                 enum ggml_log_level                     pLogLevel) {
  if (pMparams->vSplitMode == cDgsSplitModeTensor) {
    throw common_params_fit_exception("dgs_params_fit is not implemented for SPLIT_MODE_TENSOR, abort");
  }
  constexpr int64_t                           cMiB = 1024 * 1024;
  typedef std::vector<dgs_device_memory_data> dmds_t;
  const dgs_model_params                      cDefaultMparams = fDgsModelDefaultParams();

  std::vector<ggml_backend_dev_t> lDevs;
  uint32_t                        vHpNgl = 0;  // hparams.n_gpu_layers
  uint32_t                        vHpNct = 0;  // hparams.n_ctx_train
  uint32_t                        vHpNex = 0;  // hparams.n_expert

  // with non-unified kv, we need to take into account n_streams
  // for example, if memory can hold more than model's trained context size, we must extend the n_ctx to hold enough n_streams
  const uint32_t cNStreams = pCparams->vKvUnified ? 1 : std::max<uint32_t>(1, pCparams->vNSeqMax);
  const bool     cNCtxAuto = pCparams->vNCtx == 0;

  dmds_t   lDmdsExtra;      // memory of the extra model, laid out on the devices of the main model
  uint32_t vNCtxExtra = 0;  // context that memory was measured at

  // the extra model competes for the same memory as the main model, add it to every measurement
  // its memory is measured again whenever the context it follows changes
  auto vAddExtraMemory = [&](dmds_t & pDmds) {
    if (pExtra == nullptr) {
      return;
    }

    if (lDmdsExtra.empty() || vNCtxExtra != pCparams->vNCtx) {
      std::vector<ggml_backend_dev_t> lDevsExtra;
      uint32_t                        vNglExtra = 0;
      uint32_t                        vNctExtra = 0;
      uint32_t                        vNexExtra = 0;

      pExtra->vCparams->vNCtx = pCparams->vNCtx;

      LOG_TRC("%s: getting device memory data for the extra model at a context size of %" PRIu32 ":\n", __func__,
              pCparams->vNCtx);

      dmds_t lMeasured;
      try {
        lMeasured = fCommonGetDeviceMemoryDataImpl(pExtra->vPathModel, pExtra->vMparams, pExtra->vCparams, lDevsExtra,
                                                   vNglExtra, vNctExtra, vNexExtra, pLogLevel);
      } catch (const std::runtime_error & vE) {
        // the extra model is optional, fit the main model alone rather than giving up
        LOG_WRN("%s: failed to measure the memory of the extra model, fitting without it: %s\n", __func__, vE.what());
        lDmdsExtra = dmds_t(lDevs.size() + 1);
        vNCtxExtra = pCparams->vNCtx;
        return;
      }

      lDmdsExtra            = dmds_t(lDevs.size() + 1);
      lDmdsExtra.back().vMb = lMeasured.back().vMb;
      for (size_t vJe = 0; vJe < lDevsExtra.size(); vJe++) {
        for (size_t vId = 0; vId < lDevs.size(); vId++) {
          if (lDevsExtra[vJe] == lDevs[vId]) {
            lDmdsExtra[vId].vMb.vModel += lMeasured[vJe].vMb.vModel;
            lDmdsExtra[vId].vMb.vContext += lMeasured[vJe].vMb.vContext;
            lDmdsExtra[vId].vMb.vCompute += lMeasured[vJe].vMb.vCompute;
            break;
          }
        }
      }
      if (pExtra->vSharesModel) {
        for (dgs_device_memory_data & vDmd : lDmdsExtra) {
          vDmd.vMb.vModel = 0;
        }
      }

      vNCtxExtra = pCparams->vNCtx;
    }

    for (size_t vId = 0; vId < pDmds.size(); vId++) {
      pDmds[vId].vMb.vModel += lDmdsExtra[vId].vMb.vModel;
      pDmds[vId].vMb.vContext += lDmdsExtra[vId].vMb.vContext;
      pDmds[vId].vMb.vCompute += lDmdsExtra[vId].vMb.vCompute;
    }
  };

  // step 1: get data for default parameters and check whether any changes are necessary in the first place

  LOG_TRC("%s: getting device memory data for initial parameters:\n", __func__);
  dmds_t lDmdsFull =
    fCommonGetDeviceMemoryDataImpl(pPathModel, pMparams, pCparams, lDevs, vHpNgl, vHpNct, vHpNex, pLogLevel);

  // saturate instead of overflowing, this also preserves the UINT32_MAX sentinel of n_ctx_min:
  const uint32_t cNCtxMax      = (uint32_t) std::min<uint64_t>(uint64_t(vHpNct) * cNStreams, UINT32_MAX);
  const uint32_t cNCtxMinTotal = (uint32_t) std::min<uint64_t>(uint64_t(pNCtxMin) * cNStreams, UINT32_MAX);

  // dgs_context would use only hp_nct in total for n_ctx == 0, resolve the context before measuring anything else:
  if (cNCtxAuto) {
    pCparams->vNCtx = cNCtxMax;
    if (cNStreams > 1) {
      LOG_TRC("%s: context size unset and KV cache not unified -> using %" PRIu32 " for %" PRIu32 " sequences:\n",
              __func__, cNCtxMax, cNStreams);
      lDmdsFull =
        fCommonGetDeviceMemoryDataImpl(pPathModel, pMparams, pCparams, lDevs, vHpNgl, vHpNct, vHpNex, pLogLevel);
    }
  }
  vAddExtraMemory(lDmdsFull);

  const size_t cNd = lDevs.size();  // number of devices

  std::vector<int64_t>
    lMargins;  // this function uses int64_t rather than size_t for memory sizes to more conveniently handle deficits
  lMargins.reserve(cNd);
  if (cNd == 0) {
    lMargins.push_back(pMarginsS[0]);
  } else {
    for (size_t vId = 0; vId < cNd; vId++) {
      lMargins.push_back(pMarginsS[vId]);
    }
  }

  std::vector<std::string> lDevNames;
  {
    lDevNames.reserve(cNd);
    size_t vMaxLength = 0;
    for (const auto & vDev : lDevs) {
      std::string name = ggml_backend_dev_name(vDev);
      name += " (";
      name += ggml_backend_dev_description(vDev);
      name += ")";
      lDevNames.push_back(name);
      vMaxLength = std::max(vMaxLength, name.length());
    }
    for (std::string & vDn : lDevNames) {
      vDn.insert(vDn.end(), vMaxLength - vDn.length(), ' ');
    }
  }

  int64_t              vSumFree           = 0;
  int64_t              vSumProjectedFree  = 0;
  int64_t              vSumProjectedUsed  = 0;
  int64_t              vSumProjectedModel = 0;
  std::vector<int64_t> lProjectedFreePerDevice;
  lProjectedFreePerDevice.reserve(cNd);

  if (cNd == 0) {
    vSumProjectedUsed = lDmdsFull.back().vMb.mTotal();
    vSumFree          = lDmdsFull.back().vTotal;
    vSumProjectedFree = vSumFree - vSumProjectedUsed;
    LOG_TRC("%s: projected to use %" PRId64 " MiB of host memory vs. %" PRId64 " MiB of total host memory\n", __func__,
            vSumProjectedUsed / cMiB, vSumFree / cMiB);
    if (vSumProjectedFree >= lMargins[0]) {
      LOG_TRC("%s: will leave %" PRId64 " >= %" PRId64 " MiB of system memory, no changes needed\n", __func__,
              vSumProjectedFree / cMiB, lMargins[0] / cMiB);
      return;
    }
  } else {
    if (cNd > 1) {
      LOG_TRC("%s: projected memory use with initial parameters [MiB]:\n", __func__);
    }
    for (size_t vId = 0; vId < cNd; vId++) {
      const dgs_device_memory_data & vDmd = lDmdsFull[vId];

      const int64_t cProjectedUsed = vDmd.vMb.mTotal();
      const int64_t cProjectedFree = vDmd.free - cProjectedUsed;
      lProjectedFreePerDevice.push_back(cProjectedFree);

      vSumFree += vDmd.free;
      vSumProjectedUsed += cProjectedUsed;
      vSumProjectedFree += cProjectedFree;
      vSumProjectedModel += vDmd.vMb.vModel;

      if (cNd > 1) {
        LOG_TRC("%s:   - %s: %6" PRId64 " total, %6" PRId64 " used, %6" PRId64 " free vs. target of %6" PRId64 "\n",
                __func__, lDevNames[vId].c_str(), vDmd.vTotal / cMiB, cProjectedUsed / cMiB, cProjectedFree / cMiB,
                lMargins[vId] / cMiB);
      }
    }
    assert(sum_free >= 0 && sum_projected_used >= 0);
    LOG_TRC("%s: projected to use %" PRId64 " MiB of device memory vs. %" PRId64 " MiB of free device memory\n",
            __func__, vSumProjectedUsed / cMiB, vSumFree / cMiB);
    if (cNd == 1) {
      if (lProjectedFreePerDevice[0] >= lMargins[0]) {
        LOG_TRC("%s: will leave %" PRId64 " >= %" PRId64 " MiB of free device memory, no changes needed\n", __func__,
                lProjectedFreePerDevice[0] / cMiB, lMargins[0] / cMiB);
        return;
      }
    } else {
      bool vChangesNeeded = false;
      for (size_t vId = 0; vId < cNd; vId++) {
        if (lProjectedFreePerDevice[vId] < lMargins[vId]) {
          vChangesNeeded = true;
          break;
        }
      }
      if (!vChangesNeeded) {
        LOG_TRC("%s: targets for free memory can be met on all devices, no changes needed\n", __func__);
        return;
      }
    }
  }

  // step 2: try reducing memory use by reducing the context size

  {
    int64_t vGlobalSurplus = vSumProjectedFree;
    if (cNd == 0) {
      vGlobalSurplus -= lMargins[0];
    } else {
      for (size_t vId = 0; vId < cNd; vId++) {
        vGlobalSurplus -= lMargins[vId];
      }
    }
    if (vGlobalSurplus < 0) {
      if (cNd <= 1) {
        LOG_TRC("%s: cannot meet free memory target of %" PRId64 " MiB, need to reduce device memory by %" PRId64
                " MiB\n",
                __func__, lMargins[0] / cMiB, -vGlobalSurplus / cMiB);
      } else {
        LOG_TRC("%s: cannot meet free memory targets on all devices, need to use %" PRId64 " MiB less in total\n",
                __func__, -vGlobalSurplus / cMiB);
      }
      if (cNCtxAuto) {
        if (cNCtxMax > cNCtxMinTotal) {
          int64_t vSumUsedTarget = vSumFree;
          if (cNd == 0) {
            vSumUsedTarget -= lMargins[0];
          } else {
            for (size_t vId = 0; vId < cNd; vId++) {
              vSumUsedTarget -= lMargins[vId];
            }
          }
          if (cNd > 1) {
            // for multiple devices we need to be more conservative in terms of how much context we think can fit:
            //   - for dense models only whole layers can be assigned to devices
            //   - on average we expect a waste of 0.5 layers/tensors per device
            //   - use slightly more than the expected average for nd devices to be safe
            const int64_t cModelPerLayer = vSumProjectedModel / std::min(uint32_t(pMparams->vNGpuLayers), vHpNgl);
            vSumUsedTarget -= (cNd + 1) * cModelPerLayer / 2;
          }

          int64_t vSumProjectedUsedMinCtx = 0;
          pCparams->vNCtx                 = cNCtxMinTotal;
          dmds_t lDmdsMinCtx =
            fCommonGetDeviceMemoryDataImpl(pPathModel, pMparams, pCparams, lDevs, vHpNgl, vHpNct, vHpNex, pLogLevel);
          vAddExtraMemory(lDmdsMinCtx);
          if (cNd == 0) {
            vSumProjectedUsedMinCtx = lDmdsMinCtx.back().vMb.mTotal();
          } else {
            for (size_t vId = 0; vId < cNd; vId++) {
              vSumProjectedUsedMinCtx += lDmdsMinCtx[vId].vMb.mTotal();
            }
          }
          if (vSumUsedTarget > vSumProjectedUsedMinCtx) {
            // linear interpolation between minimum and maximum context size:
            pCparams->vNCtx += (cNCtxMax - cNCtxMinTotal) * (vSumUsedTarget - vSumProjectedUsedMinCtx) /
                               (vSumProjectedUsed - vSumProjectedUsedMinCtx);
            // round down context for CUDA backend, keep it divisible by the number of streams:
            const uint32_t cAlign = 256 * cNStreams;
            pCparams->vNCtx       = std::max(pCparams->vNCtx - pCparams->vNCtx % cAlign, cNCtxMinTotal);

            const int64_t cBytesPerCtx     = (vSumProjectedUsed - vSumProjectedUsedMinCtx) / (cNCtxMax - cNCtxMinTotal);
            const int64_t cMemoryReduction = (cNCtxMax - pCparams->vNCtx) * cBytesPerCtx;
            LOG_TRC("%s: context size reduced from %" PRIu32 " to %" PRIu32 " -> need %" PRId64
                    " MiB less memory in total\n",
                    __func__, cNCtxMax, pCparams->vNCtx, cMemoryReduction / cMiB);
            if (cNd <= 1) {
              LOG_TRC("%s: entire model can be fit by reducing context\n", __func__);
              return;
            }
            LOG_TRC("%s: entire model should be fit across devices by reducing context\n", __func__);
          } else {
            const int64_t cMemoryReduction = vSumProjectedUsed - vSumProjectedUsedMinCtx;
            LOG_TRC("%s: context size reduced from %" PRIu32 " to %" PRIu32 " -> need %" PRId64
                    " MiB less memory in total\n",
                    __func__, cNCtxMax, pCparams->vNCtx, cMemoryReduction / cMiB);
          }
        } else {
          if (pNCtxMin == UINT32_MAX) {
            LOG_TRC("%s: user has requested full context size of %" PRIu32 " -> no change\n", __func__, cNCtxMax);
          } else {
            LOG_TRC("%s: default model context size is %" PRIu32 " which is <= the min. context size of %" PRIu32
                    " -> no change\n",
                    __func__, cNCtxMax, cNCtxMinTotal);
          }
        }
      } else {
        LOG_TRC("%s: context size set by user to %" PRIu32 " -> no change\n", __func__, pCparams->vNCtx);
      }
    }
  }
  if (cNd == 0) {
    throw common_params_fit_exception("was unable to fit model into system memory by reducing context, abort");
  }

  if (pMparams->vNGpuLayers != cDefaultMparams.vNGpuLayers) {
    throw common_params_fit_exception("n_gpu_layers already set by user to " + std::to_string(pMparams->vNGpuLayers) +
                                      ", abort");
  }
  if (cNd > 1) {
    if (!pTensorSplit) {
      throw common_params_fit_exception("did not provide a buffer to write the tensor_split to, abort");
    }
    if (pMparams->vTensorSplit) {
      for (size_t vId = 0; vId < cNd; vId++) {
        if (pMparams->vTensorSplit[vId] != 0.0f) {
          throw common_params_fit_exception("model_params::tensor_split already set by user, abort");
        }
      }
    }
    if (pMparams->vSplitMode == cDgsSplitModeRow) {
      throw common_params_fit_exception("changing weight allocation for DGS_SPLIT_MODE_ROW not implemented, abort");
    }
  }
  if (!pTensorBuftOverrides) {
    throw common_params_fit_exception("did not provide buffer to set tensor_buft_overrides, abort");
  }
  if (pMparams->vTensorBuftOverrides &&
      (pMparams->vTensorBuftOverrides->vPattern || pMparams->vTensorBuftOverrides->vBuft)) {
    throw common_params_fit_exception("model_params::tensor_buft_overrides already set by user, abort");
  }

  // step 3: iteratively fill the back to front with "dense" layers
  //   - for a dense model simply fill full layers, giving each device a contiguous slice of the model

  // utility function that returns a static C string matching the tensors for a specific layer index and layer fraction:
  auto vGetOverflowPattern = [&](const size_t pIl, const common_layer_fraction_t pLf) -> const char * {
    constexpr size_t cNStrings = 1000;
    if (pIl >= cNStrings) {
      throw std::runtime_error("at most " + std::to_string(cNStrings) + " model layers are supported");
    }
    switch (pLf) {
      case cLayerFractionAttn:
        {
          static std::array<std::string, cNStrings> aPatterns;
          if (aPatterns[pIl].empty()) {
            aPatterns[pIl] = "blk\\." + std::to_string(pIl) + "\\.ffn_(gate|up|gate_up|down).*";
          }
          return aPatterns[pIl].c_str();
        }
      case cLayerFractionUp:
        {
          static std::array<std::string, cNStrings> aPatterns;
          if (aPatterns[pIl].empty()) {
            aPatterns[pIl] = "blk\\." + std::to_string(pIl) + "\\.ffn_(gate|gate_up|down).*";
          }
          return aPatterns[pIl].c_str();
        }
      case cLayerFractionGate:
        {
          static std::array<std::string, cNStrings> aPatterns;
          if (aPatterns[pIl].empty()) {
            aPatterns[pIl] = "blk\\." + std::to_string(pIl) + "\\.ffn_down.*";
          }
          return aPatterns[pIl].c_str();
        }

      default:
        GGML_ABORT("fatal error");
    }
  };

  struct ngl_t {
    uint32_t vNLayer = 0;  // number of total layers
    uint32_t vNPart  = 0;  // number of partial layers, <= n_layer

    common_layer_fraction_t vOverflowType = cLayerFractionAttn;

    uint32_t mNFull() const {
      assert(n_layer >= n_part);
      return vNLayer - vNPart;
    }
  };

  const size_t cNtbo = fDgsMaxTensorBuftOverrides();

  // utility function to set n_gpu_layers and tensor_split
  auto vSetNglTensorSplitTbo = [&](const std::vector<ngl_t> &                      pNglPerDevice,
                                   const std::vector<ggml_backend_buffer_type_t> & pOverflowBufts,
                                   dgs_model_params &                              pMparams) {
    pMparams.vNGpuLayers = 0;
    for (size_t vId = 0; vId < cNd; vId++) {
      pMparams.vNGpuLayers += pNglPerDevice[vId].vNLayer;
      if (cNd > 1) {
        pTensorSplit[vId] = pNglPerDevice[vId].vNLayer;
      }
    }
    assert(uint32_t(mparams.n_gpu_layers) <= hp_ngl + 1);
    uint32_t vIl0 = vHpNgl + 1 - pMparams.vNGpuLayers;  // start index for tensor buft overrides

    pMparams.vTensorSplit = pTensorSplit;

    size_t vItbo = 0;
    for (size_t vId = 0; vId < cNd; vId++) {
      vIl0 += pNglPerDevice[vId].mNFull();
      for (uint32_t vIl = vIl0; vIl < vIl0 + pNglPerDevice[vId].vNPart; vIl++) {
        if (vItbo + 1 >= cNtbo) {
          pTensorBuftOverrides[vItbo].vPattern = nullptr;
          pTensorBuftOverrides[vItbo].vBuft    = nullptr;
          vItbo++;
          pMparams.vTensorBuftOverrides = pTensorBuftOverrides;
          throw common_params_fit_exception("dgs_max_tensor_buft_overrides() == " + std::to_string(cNtbo) +
                                            " is insufficient for model");
        }
        pTensorBuftOverrides[vItbo].vPattern = vGetOverflowPattern(vIl, pNglPerDevice[vId].vOverflowType);
        pTensorBuftOverrides[vItbo].vBuft    = vIl == vIl0 ? pOverflowBufts[vId] : ggml_backend_cpu_buffer_type();
        vItbo++;
      }
      vIl0 += pNglPerDevice[vId].vNPart;
    }
    pTensorBuftOverrides[vItbo].vPattern = nullptr;
    pTensorBuftOverrides[vItbo].vBuft    = nullptr;
    vItbo++;
    pMparams.vTensorBuftOverrides = pTensorBuftOverrides;
  };

  // utility function that returns the memory use per device for given numbers of layers per device
  auto vGetMemoryForLayers = [&](
                               const char * pFuncName, const std::vector<ngl_t> & pNglPerDevice,
                               const std::vector<ggml_backend_buffer_type_t> & pOverflowBufts) -> std::vector<int64_t> {
    dgs_model_params vMparamsCopy = *pMparams;
    vSetNglTensorSplitTbo(pNglPerDevice, pOverflowBufts, vMparamsCopy);

    dmds_t lDmdNl =
      fCommonGetDeviceMemoryDataImpl(pPathModel, &vMparamsCopy, pCparams, lDevs, vHpNgl, vHpNct, vHpNex, pLogLevel);
    vAddExtraMemory(lDmdNl);

    LOG_TRC("%s: memory for test allocation by device:\n", pFuncName);
    for (size_t vId = 0; vId < cNd; vId++) {
      const ngl_t & n = pNglPerDevice[vId];
      LOG_TRC("%s: id=%zu, n_layer=%2" PRIu32 ", n_part=%2" PRIu32 ", overflow_type=%d, mem=%6" PRId64 " MiB\n",
              pFuncName, vId, n.vNLayer, n.vNPart, int(n.vOverflowType), lDmdNl[vId].vMb.mTotal() / cMiB);
    }

    std::vector<int64_t> lRet;
    lRet.reserve(cNd);
    for (size_t vId = 0; vId < cNd; vId++) {
      lRet.push_back(lDmdNl[vId].vMb.mTotal());
    }
    return lRet;
  };

  std::vector<int64_t> lTargets;  // maximum acceptable memory use per device
  lTargets.reserve(cNd);
  for (size_t vId = 0; vId < cNd; vId++) {
    lTargets.push_back(lDmdsFull[vId].free - lMargins[vId]);
    LOG_TRC("%s: id=%zu, target=%" PRId64 " MiB\n", __func__, vId, lTargets[vId] / cMiB);
  }

  std::vector<ggml_backend_buffer_type_t>
    lOverflowBufts;  // which bufts the first partial layer of a device overflows to:
  lOverflowBufts.reserve(cNd);
  for (size_t vId = 0; vId < cNd; vId++) {
    lOverflowBufts.push_back(ggml_backend_cpu_buffer_type());
  }

  std::vector<ngl_t>   lNglPerDevice(cNd);
  std::vector<int64_t> lMem = vGetMemoryForLayers(__func__, lNglPerDevice, lOverflowBufts);

  // optimize the number of layers per device using the method of false position:
  //   - ngl_per_device has 0 layers for each device, lower bound
  //   - try a "high" configuration where a device is given all unassigned layers
  //   - interpolate the memory use / layer between low and high linearly to get a guess where it meets our target
  //   - check memory use of our guess, replace either the low or high bound
  //   - once we only have a difference of a single layer, stop and return the lower bound that just barely still fits
  //   - the last device has the output layer, which cannot be a partial layer
  {
    LOG_TRC("%s: filling dense layers back-to-front:\n", __func__);
  }
  for (int vId = cNd - 1; vId >= 0; vId--) {
    uint32_t vNUnassigned = vHpNgl + 1;
    for (size_t vJd = vId + 1; vJd < cNd; ++vJd) {
      assert(n_unassigned >= ngl_per_device[jd].n_layer);
      vNUnassigned -= lNglPerDevice[vJd].vNLayer;
    }

    std::vector<ngl_t> lNglPerDeviceHigh = lNglPerDevice;
    lNglPerDeviceHigh[vId].vNLayer       = vNUnassigned;

    if (lNglPerDeviceHigh[vId].vNLayer > 0) {
      std::vector<int64_t> lMemHigh = vGetMemoryForLayers(__func__, lNglPerDeviceHigh, lOverflowBufts);
      if (lMemHigh[vId] > lTargets[vId]) {
        assert(ngl_per_device_high[id].n_layer > ngl_per_device[id].n_layer);
        uint32_t vDelta = lNglPerDeviceHigh[vId].vNLayer - lNglPerDevice[vId].vNLayer;
        LOG_TRC("%s: start filling device %" PRIu32 ", delta=%" PRIu32 "\n", __func__, vId, vDelta);
        while (vDelta > 1) {
          uint32_t vStepSize = int64_t(vDelta) * (lTargets[vId] - lMem[vId]) / (lMemHigh[vId] - lMem[vId]);
          vStepSize          = std::max(vStepSize, uint32_t(1));
          vStepSize          = std::min(vStepSize, vDelta - 1);

          std::vector<ngl_t> lNglPerDeviceTest = lNglPerDevice;
          lNglPerDeviceTest[vId].vNLayer += vStepSize;

          const std::vector<int64_t> lMemTest = vGetMemoryForLayers(__func__, lNglPerDeviceTest, lOverflowBufts);

          if (lMemTest[vId] <= lTargets[vId]) {
            lNglPerDevice = lNglPerDeviceTest;
            lMem          = lMemTest;
            LOG_TRC("%s: set ngl_per_device[%d].n_layer=%" PRIu32 "\n", __func__, vId, lNglPerDevice[vId].vNLayer);
          } else {
            lNglPerDeviceHigh = lNglPerDeviceTest;
            lMemHigh          = lMemTest;
            LOG_TRC("%s: set ngl_per_device_high[%d].n_layer=%" PRIu32 "\n", __func__, vId,
                    lNglPerDeviceHigh[vId].vNLayer);
          }
          vDelta = lNglPerDeviceHigh[vId].vNLayer - lNglPerDevice[vId].vNLayer;
        }
      } else {
        assert(ngl_per_device_high[id].n_layer == n_unassigned);
        lNglPerDevice = lNglPerDeviceHigh;
        lMem          = lMemHigh;
        LOG_TRC("%s: set ngl_per_device[%d].n_layer=%" PRIu32 "\n", __func__, vId, lNglPerDevice[vId].vNLayer);
      }
    }

    const int64_t cProjectedMargin = lDmdsFull[vId].free - lMem[vId];
    LOG_TRC("%s:   - %s: %2" PRIu32 " layers, %6" PRId64 " MiB used, %6" PRId64 " MiB free\n", __func__,
            lDevNames[vId].c_str(), lNglPerDevice[vId].vNLayer, lMem[vId] / cMiB, cProjectedMargin / cMiB);
  }
  vSetNglTensorSplitTbo(lNglPerDevice, lOverflowBufts, *pMparams);
}

enum common_params_fit_status fCommonFitParams(const char *                     pPathModel,
                                               dgs_model_params *               pMparams,
                                               dgs_context_params *             pCparams,
                                               float *                          pTensorSplit,
                                               dgs_model_tensor_buft_override * pTensorBuftOverrides,
                                               size_t *                         pMargins,
                                               uint32_t                         pNCtxMin,
                                               const common_fit_extra_model *   pExtra,
                                               ggml_log_level                   pLogLevel) {
  const int64_t            cT0Us   = fDgsTimeUs();
  common_params_fit_status vStatus = cCommonParamsFitStatusSuccess;
  try {
    fCommonParamsFitImpl(pPathModel, pMparams, pCparams, pTensorSplit, pTensorBuftOverrides, pMargins, pNCtxMin, pExtra,
                         pLogLevel);
    LOG_TRC("%s: successfully fit params to free device memory\n", __func__);
  } catch (const common_params_fit_exception & vE) {
    LOG_WRN("%s: failed to fit params to free device memory: %s\n", __func__, vE.what());
    vStatus = cCommonParamsFitStatusFailure;
  } catch (const std::runtime_error & vE) {
    LOG_ERR("%s: encountered an error while trying to fit params to free device memory: %s\n", __func__, vE.what());
    vStatus = cCommonParamsFitStatusError;
  }
  const int64_t cT1Us = fDgsTimeUs();
  LOG_TRC("%s: fitting params to free memory took %.2f seconds\n", __func__, (cT1Us - cT0Us) * 1e-6);
  return vStatus;
}

void fCommonMemoryBreakdownPrint(const struct dgs_context * pCtx) {
  //const auto & devices = ctx->get_model().devices;
  const auto * vModel = fDgsGetModel(pCtx);

  std::vector<ggml_backend_dev_t> lDevices;
  for (int vI = 0; vI < fDgsModelNDevices(vModel); vI++) {
    lDevices.push_back(fDgsModelGetDevice(vModel, vI));
  }

  dgs_memory_breakdown dMemoryBreakdown = fDgsGetMemoryBreakdown(pCtx);

  std::vector<std::array<std::string, 9>> lTableData;
  lTableData.reserve(lDevices.size());

  // same data as the table below, for --log-jsonl consumers
  common_json       vRows           = common_json::mArray();
  const std::string cTemplateHeader = "%s: | %s | %s   %s    %s   %s   %s   %s    %s |\n";
  const std::string cTemplateGpu    = "%s: | %s | %s = %s + (%s = %s + %s + %s) + %s |\n";
  const std::string cTemplateOther  = "%s: | %s | %s   %s    %s = %s + %s + %s    %s |\n";

  lTableData.push_back({ cTemplateHeader, "memory breakdown [MiB]", "total", "free", "self", "model", "context",
                         "compute", "unaccounted" });

  constexpr size_t               cMiB               = 1024 * 1024;
  const std::vector<std::string> lDescPrefixesStrip = {
    "NVIDIA ", "GeForce ", "Tesla ", "AMD ", "Radeon ", "Instinct "
  };

  // track seen buffer types to avoid double counting:
  std::set<ggml_backend_buffer_type_t> sSeenBufferTypes;

  // accumulative memory breakdown for each device and for host:
  std::vector<dgs_memory_breakdown_data> lMbDev(lDevices.size());
  dgs_memory_breakdown_data              vMbHost;

  for (const auto & vBuftMb : dMemoryBreakdown) {
    ggml_backend_buffer_type_t        vBuft = vBuftMb.first;
    const dgs_memory_breakdown_data & vMb   = vBuftMb.second;
    if (ggml_backend_buft_is_host(vBuft)) {
      vMbHost.vModel += vMb.vModel;
      vMbHost.vContext += vMb.vContext;
      vMbHost.vCompute += vMb.vCompute;
      sSeenBufferTypes.insert(vBuft);
      continue;
    }
    ggml_backend_dev_t vDev = ggml_backend_buft_get_device(vBuft);
    if (vDev) {
      int vIDev = -1;
      for (size_t vI = 0; vI < lDevices.size(); vI++) {
        if (lDevices[vI] == vDev) {
          vIDev = vI;
          break;
        }
      }
      if (vIDev != -1) {
        lMbDev[vIDev].vModel += vMb.vModel;
        lMbDev[vIDev].vContext += vMb.vContext;
        lMbDev[vIDev].vCompute += vMb.vCompute;
        sSeenBufferTypes.insert(vBuft);
        continue;
      }
    }
  }

  // print memory breakdown for each device:
  for (size_t vI = 0; vI < lDevices.size(); vI++) {
    ggml_backend_dev_t        vDev = lDevices[vI];
    dgs_memory_breakdown_data vMb  = lMbDev[vI];

    const std::string name  = ggml_backend_dev_name(vDev);
    std::string       vDesc = ggml_backend_dev_description(vDev);
    for (const std::string & vPrefix : lDescPrefixesStrip) {
      if (vDesc.length() >= vPrefix.length() && vDesc.substr(0, vPrefix.length()) == vPrefix) {
        vDesc = vDesc.substr(vPrefix.length());
      }
    }

    size_t free, vTotal;
    ggml_backend_dev_memory(vDev, &free, &vTotal);

    const size_t  cSelf = vMb.vModel + vMb.vContext + vMb.vCompute;
    const int64_t cUnaccounted =
      static_cast<int64_t>(vTotal) - static_cast<int64_t>(free) - static_cast<int64_t>(cSelf);

    lTableData.push_back({ cTemplateGpu, "  - " + name + " (" + vDesc + ")", std::to_string(vTotal / cMiB),
                           std::to_string(free / cMiB), std::to_string(cSelf / cMiB), std::to_string(vMb.vModel / cMiB),
                           std::to_string(vMb.vContext / cMiB), std::to_string(vMb.vCompute / cMiB),
                           std::to_string(cUnaccounted / static_cast<int64_t>(cMiB)) });

    vRows.push_back({
      { "kind",        "device"                                  },
      { "name",        name                                      },
      { "description", vDesc                                     },
      { "total",       vTotal / cMiB                             },
      { "free",        free / cMiB                               },
      { "self",        cSelf / cMiB                              },
      { "model",       vMb.vModel / cMiB                         },
      { "context",     vMb.vContext / cMiB                       },
      { "compute",     vMb.vCompute / cMiB                       },
      { "unaccounted", cUnaccounted / static_cast<int64_t>(cMiB) },
    });
  }

  // print memory breakdown for host:
  {
    const size_t cSelf = vMbHost.vModel + vMbHost.vContext + vMbHost.vCompute;
    lTableData.push_back({ cTemplateOther, "  - Host",
                           "",  // total
                           "",  // free
                           std::to_string(cSelf / cMiB), std::to_string(vMbHost.vModel / cMiB),
                           std::to_string(vMbHost.vContext / cMiB), std::to_string(vMbHost.vCompute / cMiB),
                           "" });  // unaccounted

    vRows.push_back({
      { "kind",    "host"                  },
      { "name",    "Host"                  },
      { "self",    cSelf / cMiB            },
      { "model",   vMbHost.vModel / cMiB   },
      { "context", vMbHost.vContext / cMiB },
      { "compute", vMbHost.vCompute / cMiB },
    });
  }

  // print memory breakdown for all remaining buffer types:
  for (const auto & vBuftMb : dMemoryBreakdown) {
    ggml_backend_buffer_type_t        vBuft = vBuftMb.first;
    const dgs_memory_breakdown_data & vMb   = vBuftMb.second;
    if (sSeenBufferTypes.count(vBuft) == 1) {
      continue;
    }
    const std::string name  = ggml_backend_buft_name(vBuft);
    const size_t      cSelf = vMb.vModel + vMb.vContext + vMb.vCompute;
    lTableData.push_back({ cTemplateOther, "  - " + name,
                           "",  // total
                           "",  // free
                           std::to_string(cSelf / cMiB), std::to_string(vMb.vModel / cMiB),
                           std::to_string(vMb.vContext / cMiB), std::to_string(vMb.vCompute / cMiB),
                           "" });  // unaccounted

    vRows.push_back({
      { "kind",    "buffer_type"       },
      { "name",    name                },
      { "self",    cSelf / cMiB        },
      { "model",   vMb.vModel / cMiB   },
      { "context", vMb.vContext / cMiB },
      { "compute", vMb.vCompute / cMiB },
    });

    sSeenBufferTypes.insert(vBuft);
  }

  for (size_t vJ = 1; vJ < lTableData[0].size(); vJ++) {
    size_t vMaxLen = 0;
    for (const auto & aTd : lTableData) {
      vMaxLen = std::max(vMaxLen, aTd[vJ].length());
    }
    for (auto & aTd : lTableData) {
      aTd[vJ].insert(vJ == 1 ? aTd[vJ].length() : 0, vMaxLen - aTd[vJ].length(), ' ');
    }
  }
  for (const auto & aTd : lTableData) {
    LOG_TRC(aTd[0].c_str(), __func__, aTd[1].c_str(), aTd[2].c_str(), aTd[3].c_str(), aTd[4].c_str(), aTd[5].c_str(),
            aTd[6].c_str(), aTd[7].c_str(), aTd[8].c_str());
  }

  LOG_JSON("fit_memory_breakdown", common_json({
                                     { "unit", "MiB" },
                                     { "rows", vRows },
  }));
}

void fCommonFitPrint(const char * pPathModel, dgs_model_params * pMparams, dgs_context_params * pCparams) {
  std::vector<ggml_backend_dev_t> lDevs;
  uint32_t                        vHpNgl = 0;  // hparams.n_gpu_layers
  uint32_t                        vHpNct = 0;  // hparams.n_ctx_train
  uint32_t                        vHpNex = 0;  // hparams.n_expert

  auto lDmd =
    fCommonGetDeviceMemoryDataImpl(pPathModel, pMparams, pCparams, lDevs, vHpNgl, vHpNct, vHpNex, GGML_LOG_LEVEL_ERROR);
  GGML_ASSERT(lDmd.size() == lDevs.size() + 1);

  for (size_t vId = 0; vId < lDevs.size(); vId++) {
    printf("%s ", ggml_backend_dev_name(lDevs[vId]));
    printf("%zu ", lDmd[vId].vMb.vModel / 1024 / 1024);
    printf("%zu ", lDmd[vId].vMb.vContext / 1024 / 1024);
    printf("%zu ", lDmd[vId].vMb.vCompute / 1024 / 1024);
    printf("\n");
  }

  printf("Host ");
  printf("%zu ", lDmd.back().vMb.vModel / 1024 / 1024);
  printf("%zu ", lDmd.back().vMb.vContext / 1024 / 1024);
  printf("%zu ", lDmd.back().vMb.vCompute / 1024 / 1024);
  printf("\n");
}
