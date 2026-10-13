#include "dgs.h"

#include "dgs-context.h"
#include "dgs-impl.h"
#include "dgs-mmap.h"
#include "dgs-model-loader.h"
#include "dgs-model.h"
#include "dgs-version.h"
#include "dgs-vocab.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <stdexcept>
#include <vector>

#if defined(_MSC_VER)
#  pragma warning(disable : 4244 4267)  // possible loss of data
#endif

//
// interface implementation
//

const char * fDgsFlashAttnTypeName(enum dgs_flash_attn_type pFlashAttnType) {
  switch (pFlashAttnType) {
    case cDgsFlashAttnTypeAuto:
      return "auto";
    case cDgsFlashAttnTypeDisabled:
      return "disabled";
    case cDgsFlashAttnTypeEnabled:
      return "enabled";
  }
  GGML_ABORT("fatal error");
}

const char * fDgsLoadModeName(enum dgs_load_mode pLoadMode) {
  switch (pLoadMode) {
    case cDgsLoadModeAuto:
      return "auto";
    case cDgsLoadModeNone:
      return "none";
    case cDgsLoadModeMmap:
      return "mmap";
    case cDgsLoadModeMlock:
      return "mlock";
    case cDgsLoadModeMmapMlock:
      return "mmap+mlock";
    case cDgsLoadModeDirectIo:
      return "dio";
  }
  GGML_ABORT("fatal error");
}

enum dgs_load_mode fDgsLoadModeFromStr(const char * str) {
  if (std::strcmp(str, "auto") == 0) {
    return cDgsLoadModeAuto;
  }
  if (std::strcmp(str, "none") == 0) {
    return cDgsLoadModeNone;
  }
  if (std::strcmp(str, "mmap") == 0) {
    return cDgsLoadModeMmap;
  }
  if (std::strcmp(str, "mlock") == 0) {
    return cDgsLoadModeMlock;
  }
  if (std::strcmp(str, "mmap+mlock") == 0) {
    return cDgsLoadModeMmapMlock;
  }
  if (std::strcmp(str, "dio") == 0) {
    return cDgsLoadModeDirectIo;
  }
  GGML_ABORT("unknown load mode: %s", str);
}

struct dgs_sampler_chain_params fDgsSamplerChainDefaultParams() {
  struct dgs_sampler_chain_params vResult = {
    /*.no_perf =*/true,
  };

  return vResult;
}

size_t fDgsMaxDevices(void) {
  return 16;
}

size_t fDgsMaxTensorBuftOverrides() {
  return 4096;
}

bool fDgsSupportsMmap(void) {
  return dgs_mmap::cSupported;
}

bool fDgsSupportsMlock(void) {
  return dgs_mlock::cSupported;
}

bool fDgsSupportsGpuOffload(void) {
  if (!ggml_backend_reg_count()) {
    ggml_backend_load_all();
  }
  return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) != nullptr ||
         ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU) != nullptr || fDgsSupportsRpc();
}

bool fDgsSupportsRpc(void) {
  if (!ggml_backend_reg_count()) {
    ggml_backend_load_all();
  }
  return ggml_backend_reg_by_name("RPC") != nullptr;
}

const char * fDgsVersion(void) {
  return DGS_VERSION;
}

void fDgsBackendInit(void) {
  ggml_time_init();

  // needed to initialize f16 tables
  {
    struct ggml_init_params vParams = { 0, NULL, false };
    struct ggml_context *   vCtx    = ggml_init(vParams);
    ggml_free(vCtx);
  }

  if (!ggml_backend_reg_count()) {
    ggml_backend_load_all();
  }
}

void fDgsNumaInit(enum ggml_numa_strategy pNuma) {
  if (pNuma != GGML_NUMA_STRATEGY_DISABLED) {
    auto * vDev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(vDev && "CPU backend is not loaded");
    auto * vReg = ggml_backend_dev_backend_reg(vDev);
    auto * vNumaInitFn =
      (decltype(ggml_numa_init) *) ggml_backend_reg_get_proc_address(vReg, "ggml_backend_cpu_numa_init");
    if (vNumaInitFn) {
      vNumaInitFn(pNuma);
    }
  }
}

void fDgsBackendFree(void) {
  ggml_quantize_free();
}

int64_t fDgsTimeUs(void) {
  return ggml_time_us();
}

// returns true on success
static bool fDgsPrepareModelDevices(const dgs_model_params & pParams, dgs_model * pModel) {
  // create list of devices to use with this model
  if (pParams.vDevices) {
    if (pParams.vSplitMode == cDgsSplitModeTensor) {
      size_t vNDevs = 0;
      while (pParams.vDevices[vNDevs]) {
        vNDevs++;
      }
      if (vNDevs == 0) {
        DGS_LOG_ERROR("%s: DGS_SPLIT_MODE_TENSOR needs >= 1 devices\n", __func__);
        return false;
      }
      DGS_LOG_INFO("%s: creating a Meta device with %zu devices\n", __func__, vNDevs);
      for (size_t vI = 0; vI < vNDevs; ++vI) {
        DGS_LOG_INFO("%s: - device %zu: %s\n", __func__, vI, ggml_backend_dev_name(pParams.vDevices[vI]));
      }
      pModel->vGetSplitStateUd.vNDevices = vNDevs;
      pModel->vGetSplitStateUd.vModel    = pModel;
      pModel->lDevices.push_back({ true, ggml_backend_meta_device(pParams.vDevices, vNDevs, fDgsMetaDeviceGetSplitState,
                                                                  &pModel->vGetSplitStateUd) });
    } else {
      for (ggml_backend_dev_t * vDev = pParams.vDevices; *vDev; ++vDev) {
        pModel->lDevices.push_back({ false, *vDev });
      }
    }
  } else {
    // default device selection

    // build list of available devices
    std::vector<dgs_device> lGpus;
    std::vector<dgs_device> lIgpus;
    std::vector<dgs_device> lRpcServers;

    if (pParams.vSplitMode == cDgsSplitModeTensor) {
      std::vector<ggml_backend_dev_t> lDevs;
      lDevs.reserve(ggml_backend_dev_count());
      for (size_t vI = 0; vI < ggml_backend_dev_count(); ++vI) {
        auto * vDev = ggml_backend_dev_get(vI);
        if (ggml_backend_dev_buffer_type(vDev) == ggml_backend_cpu_buffer_type()) {
          DGS_LOG_INFO("%s: skipping %s (%s) for tensor parallelism\n", __func__, ggml_backend_dev_name(vDev),
                       ggml_backend_dev_description(vDev));
          continue;
        }
        lDevs.push_back(vDev);
      }
      if (lDevs.empty()) {
        DGS_LOG_ERROR("%s: DGS_SPLIT_MODE_TENSOR needs >= 1 devices\n", __func__);
        return false;
      }

      DGS_LOG_INFO("%s: creating a Meta device for tensor parallelism from %zu devices:\n", __func__, lDevs.size());
      for (size_t vI = 0; vI < lDevs.size(); ++vI) {
        DGS_LOG_INFO("%s: - device %zu: %s (%s)\n", __func__, vI, ggml_backend_dev_name(lDevs[vI]),
                     ggml_backend_dev_description(lDevs[vI]));
      }

      GGML_ASSERT(!lDevs.empty());
      pModel->vGetSplitStateUd.vNDevices = lDevs.size();
      pModel->vGetSplitStateUd.vModel    = pModel;
      lGpus.push_back({ true, ggml_backend_meta_device(lDevs.data(), lDevs.size(), fDgsMetaDeviceGetSplitState,
                                                       &pModel->vGetSplitStateUd) });
    } else {
      for (size_t vI = 0; vI < ggml_backend_dev_count(); ++vI) {
        ggml_backend_dev_t vDev = ggml_backend_dev_get(vI);
        switch (ggml_backend_dev_type(vDev)) {
          case GGML_BACKEND_DEVICE_TYPE_CPU:
          case GGML_BACKEND_DEVICE_TYPE_ACCEL:
            // skip CPU backends since they are handled separately
            break;

          case GGML_BACKEND_DEVICE_TYPE_GPU:
            {
              ggml_backend_reg_t vReg = ggml_backend_dev_backend_reg(vDev);
              if (ggml_backend_reg_name(vReg) == std::string("RPC")) {
                lRpcServers.push_back({ false, vDev });
              } else {
                // check if there is already a GPU with the same device id
                ggml_backend_dev_props vProps;
                ggml_backend_dev_get_props(vDev, &vProps);
                auto lIt = std::find_if(lGpus.begin(), lGpus.end(), [&vProps](const dgs_device & pD) {
                  ggml_backend_dev_props vDProps;
                  ggml_backend_dev_get_props(pD.dev, &vDProps);
                  if (vProps.device_id && vDProps.device_id) {
                    return strcmp(vProps.device_id, vDProps.device_id) == 0;
                  }
                  return false;
                });

                if (lIt != lGpus.end()) {
                  DGS_LOG_INFO(
                    "%s: skipping device %s (%s) with id %s - already using device %s (%s) with the same id\n",
                    __func__, ggml_backend_dev_name(vDev), ggml_backend_dev_description(vDev),
                    vProps.device_id ? vProps.device_id : "unknown id", ggml_backend_dev_name(lIt->dev),
                    ggml_backend_dev_description(lIt->dev));
                } else {
                  lGpus.push_back({ false, vDev });
                }
              }
              break;
            }

          case GGML_BACKEND_DEVICE_TYPE_IGPU:
            // igpus.empty() - workaround for integrated devices seen by multiple backends
            // ggml_backend_dev_backend_reg - allow devices of the same backend regardless if integrated
            if (lIgpus.empty() ||
                ggml_backend_dev_backend_reg(vDev) == ggml_backend_dev_backend_reg(lIgpus.back().dev)) {
              lIgpus.push_back({ false, vDev });
            }
            break;
          case GGML_BACKEND_DEVICE_TYPE_META:
            GGML_ABORT("fatal error");
        }
      }
    }

    // add RPC servers at the front of the list to minimize network transfers
    pModel->lDevices.insert(pModel->lDevices.begin(), lRpcServers.begin(), lRpcServers.end());

    // add GPUs
    pModel->lDevices.insert(pModel->lDevices.end(), lGpus.begin(), lGpus.end());

    // add integrated GPUs only if no discrete GPUs were found
    // (RPC servers do not count, otherwise the local iGPU would be dropped on iGPU+RPC setups)
    if (lGpus.empty()) {
      pModel->lDevices.insert(pModel->lDevices.end(), lIgpus.begin(), lIgpus.end());
    }
  }

  // if using single GPU mode, remove all except the main GPU
  if (pParams.vSplitMode == cDgsSplitModeNone && !pModel->lDevices.empty()) {
    if (pParams.vMainGpu < 0) {
      pModel->lDevices.clear();
    } else {
      if (pParams.vMainGpu >= (int) pModel->lDevices.size()) {
        DGS_LOG_ERROR("%s: invalid value for main_gpu: %d (available devices: %zu)\n", __func__, pParams.vMainGpu,
                      pModel->lDevices.size());
        return false;
      }
      dgs_device vMainGpu = pModel->lDevices[pParams.vMainGpu];
      pModel->lDevices.clear();
      pModel->lDevices.push_back(vMainGpu);
    }
  }

  for (const auto & vDev : pModel->lDevices) {
    ggml_backend_dev_props vProps;
    ggml_backend_dev_get_props(vDev.dev, &vProps);
    DGS_LOG_INFO("%s: using device %s (%s) (%s) - %zu MiB free\n", __func__, ggml_backend_dev_name(vDev.dev),
                 ggml_backend_dev_description(vDev.dev), vProps.device_id ? vProps.device_id : "unknown id",
                 vProps.memory_free / 1024 / 1024);
  }

  return true;
}

// Returns 0 on success, -1 on error, and -2 on cancellation via dgs_progress_callback
static std::pair<int, dgs_model *> fDgsModelLoad(struct gguf_context *       pMetadata,
                                                 dgs_model_set_tensor_data_t pSetTensorData,
                                                 void *                      pSetTensorDataUd,
                                                 const std::string &         pFname,
                                                 std::vector<std::string> &  pSplits,
                                                 FILE *                      pFile,
                                                 dgs_model_params &          pParams) {
  try {
    dgs_model_loader vMl(pMetadata, pSetTensorData, pSetTensorDataUd, pFname, pSplits, pFile, pParams.vLoadMode,
                         pParams.vCheckTensors, pParams.vNoAlloc, pParams.vLoadMtp, pParams.vKvOverrides,
                         pParams.vTensorBuftOverrides);

    vMl.vLazy.vMode = pParams.vLazyMode;

    vMl.mPrintInfo();
    std::unique_ptr<dgs_model> vModelPtr(fDgsModelCreate(vMl, pParams));

    bool vOk = fDgsPrepareModelDevices(pParams, vModelPtr.get());
    if (!vOk) {
      return { -1, nullptr };
    }

    auto * vModel = dynamic_cast<dgs_model_base *>(vModelPtr.get());
    if (vModel == nullptr) {
      GGML_ABORT("fatal error: model does not implement dgs_model_base");
    }

    // loading time will be recalculated after the first eval, so
    // we take page faults deferred by mmap() into consideration
    vModel->vTLoadUs = 0;
    time_meas vTm(vModel->vTLoadUs);

    vModel->vTStartUs = vTm.cTStartUs;

    vModel->hparams.vVocabOnly = pParams.vVocabOnly;
    vModel->hparams.vNoAlloc   = pParams.vNoAlloc;

    try {
      vModel->mLoadHparams(vMl);
    } catch (const std::exception & vE) {
      throw std::runtime_error("error loading model hyperparameters: " + std::string(vE.what()));
    }
    if (vModel->vArch == cLlmArchClip) {
      throw std::runtime_error("CLIP cannot be used as main model, use it with --mmproj instead");
    }
    try {
      vModel->mLoadVocab(vMl);
    } catch (const std::exception & vE) {
      throw std::runtime_error("error loading model vocabulary: " + std::string(vE.what()));
    }

    vModel->mLoadStats(vMl);
    vModel->mPrintInfo();

    if (pParams.vVocabOnly) {
      DGS_LOG_INFO("%s: vocab only - skipping tensors\n", __func__);
      return { 0, vModelPtr.release() };
    }

    if (!vModel->mLoadTensors(vMl)) {
      return { -2, nullptr };
    }

    return { 0, vModelPtr.release() };
  } catch (const std::exception & vErr) {
    DGS_LOG_ERROR("%s: error loading model: %s\n", __func__, vErr.what());
    return { -1, nullptr };
  }
}

static struct dgs_model * fDgsModelLoadFromFileImpl(struct gguf_context *       pMetadata,
                                                    dgs_model_set_tensor_data_t pSetTensorData,
                                                    void *                      pSetTensorDataUd,
                                                    const std::string &         pPathModel,
                                                    std::vector<std::string> &  pSplits,
                                                    FILE *                      pFile,
                                                    struct dgs_model_params     pParams) {
  {
    int vNSourcesDefined = 0;
    if (pMetadata != nullptr) {
      vNSourcesDefined++;
    }
    if (!pPathModel.empty()) {
      vNSourcesDefined++;
    }
    if (pFile != nullptr) {
      vNSourcesDefined++;
    }
    if (vNSourcesDefined != 1) {
      DGS_LOG_ERROR("%s: exactly one out metadata, path_model, and file must be defined\n", __func__);
      return nullptr;
    }
  }
  // TODO: remove
  ggml_time_init();

  if (!pParams.vVocabOnly && ggml_backend_reg_count() == 0) {
    DGS_LOG_ERROR(
      "%s: no backends are loaded. hint: use ggml_backend_load() or ggml_backend_load_all() to load a backend before "
      "calling this function\n",
      __func__);
    return nullptr;
  }

  unsigned vCurPercentage = 0;
  if (pParams.vProgressCallback == NULL) {
    pParams.vProgressCallbackUserData = &vCurPercentage;
    pParams.vProgressCallback         = [](float pProgress, void * pCtx) {
      unsigned * vCurPercentageP = (unsigned *) pCtx;
      unsigned   vPercentage     = (unsigned) (100 * pProgress);
      while (vPercentage > *vCurPercentageP) {
        *vCurPercentageP = vPercentage;
        DGS_LOG_CONT(".");
        if (vPercentage >= 100) {
          DGS_LOG_CONT("\n");
        }
      }
      return true;
    };
  }

  const auto [status, model] =
    fDgsModelLoad(pMetadata, pSetTensorData, pSetTensorDataUd, pPathModel, pSplits, pFile, pParams);
  GGML_ASSERT(status <= 0);
  if (status < 0) {
    if (status == -1) {
      DGS_LOG_ERROR("%s: failed to load model\n", __func__);
    } else if (status == -2) {
      DGS_LOG_INFO("%s: cancelled model load\n", __func__);
    }

    if (model) {
      fDgsModelFree(model);
    }
    return nullptr;
  }

  return model;
}

struct dgs_model * fDgsModelInitFromUser(struct gguf_context *       pMetadata,
                                         dgs_model_set_tensor_data_t pSetTensorData,
                                         void *                      pSetTensorDataUd,
                                         struct dgs_model_params     pParams) {
  GGML_ASSERT(pMetadata != nullptr);
  std::string              vPathModel;
  std::vector<std::string> lSplits = {};
  pParams.vLoadMode                = cDgsLoadModeNone;
  pParams.vUseExtraBufts           = false;
  return fDgsModelLoadFromFileImpl(pMetadata, pSetTensorData, pSetTensorDataUd, vPathModel, lSplits, /*file*/ nullptr,
                                   pParams);
}

struct dgs_model * fDgsModelLoadFromFile(const char * pPathModel, struct dgs_model_params pParams) {
  std::vector<std::string> lSplits = {};
  return fDgsModelLoadFromFileImpl(nullptr, nullptr, nullptr, pPathModel, lSplits, /*file*/ nullptr, pParams);
}

struct dgs_model * fDgsModelLoadFromSplits(const char ** pPaths, size_t pNPaths, struct dgs_model_params pParams) {
  std::vector<std::string> lSplits;
  if (pNPaths == 0) {
    DGS_LOG_ERROR("%s: list of splits is empty\n", __func__);
    return nullptr;
  }
  lSplits.reserve(pNPaths);
  for (size_t vI = 0; vI < pNPaths; ++vI) {
    lSplits.push_back(pPaths[vI]);
  }
  return fDgsModelLoadFromFileImpl(nullptr, nullptr, nullptr, lSplits.front(), lSplits, /*file*/ nullptr, pParams);
}

struct dgs_model * fDgsModelLoadFromFilePtr(FILE * pFile, struct dgs_model_params pParams) {
  if (!pFile) {
    DGS_LOG_ERROR("%s: file is NULL\n", __func__);
    return nullptr;
  }
  std::string              vPathModel;
  std::vector<std::string> lSplits = {};
  return fDgsModelLoadFromFileImpl(nullptr, nullptr, nullptr, vPathModel, lSplits, pFile, pParams);
}

//
// chat templates
//

//
// model split
//

int32_t fDgsSplitPath(char *       pSplitPath,
                      size_t       pMaxlen,
                      const char * pPathPrefix,
                      int32_t      pSplitNo,
                      int32_t      pSplitCount) {
  static const char * const cSplitPathFormat = "%s-%05d-of-%05d.gguf";

  const int cWritten = snprintf(pSplitPath, pMaxlen, cSplitPathFormat, pPathPrefix, pSplitNo + 1, pSplitCount);

  if (cWritten < 0 || (size_t) cWritten >= pMaxlen) {
    return 0;
  }

  return (int32_t) cWritten;
}

int32_t fDgsSplitPrefix(char *       pSplitPrefix,
                        size_t       pMaxlen,
                        const char * pSplitPath,
                        int32_t      pSplitNo,
                        int32_t      pSplitCount) {
  const std::string cStrSplitPath(pSplitPath);

  char aPostfix[32];
  snprintf(aPostfix, sizeof(aPostfix), "-%05d-of-%05d.gguf", pSplitNo + 1, pSplitCount);

  const std::string cStrPostfix(aPostfix);
  if (cStrSplitPath.size() <= cStrPostfix.size()) {
    return 0;
  }

  const size_t cSizePrefix = cStrSplitPath.size() - cStrPostfix.size();

  if (cStrSplitPath.compare(cSizePrefix, std::string::npos, cStrPostfix) == 0) {
    const size_t cCopyLen = std::min(cSizePrefix + 1, pMaxlen);
    snprintf(pSplitPrefix, cCopyLen, "%s", pSplitPath);

    return (int32_t) cSizePrefix;
  }

  return 0;
}

const char * fDgsPrintSystemInfo(void) {
  static std::string vS;
  vS.clear();  // Clear the string, since it's static, otherwise it will accumulate data from previous calls.

  for (size_t vI = 0; vI < ggml_backend_reg_count(); vI++) {
    auto * vReg = ggml_backend_reg_get(vI);
    auto * vGetFeaturesFn =
      (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(vReg, "ggml_backend_get_features");
    if (vGetFeaturesFn) {
      ggml_backend_feature * vFeatures = vGetFeaturesFn(vReg);
      vS += ggml_backend_reg_name(vReg);
      vS += " : ";
      for (; vFeatures->name; vFeatures++) {
        vS += vFeatures->name;
        vS += " = ";
        vS += vFeatures->value;
        vS += " | ";
      }
    }
  }

  return vS.c_str();
}
