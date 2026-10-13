#pragma once

#include "dgs-arch.h"
#include "dgs-hparams.h"
#include "dgs-impl.h"
#include "dgs-mmap.h"
#include "dgs.h"
#include "ggml-cpp.h"

#include <cstddef>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>

using dgs_buf_map = std::unordered_map<uint32_t, ggml_backend_buffer_t>;

// lists of buffer types used for each layer
using buft_list_t = std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>>;

enum dgs_fver {
  cGgufFileVersionV1 = 1,
  cGgufFileVersionV2 = 2,
  cGgufFileVersionV3 = 3,
};

const char * fDgsFileVersionName(dgs_fver pVersion);

struct dgs_model_loader {
  // Holds information on a model weight
  struct dgs_tensor_weight {
    uint16_t vIdx;   // source file index
    size_t   vOffs;  // tensor data offset in the original file

    ggml_tensor * vTensor;

    dgs_tensor_weight(const dgs_file *            pFile,
                      uint16_t                    pIdx,
                      const struct gguf_context * pGgufCtx,
                      ggml_tensor *               pTensor) :
      vIdx(pIdx),
      vTensor(pTensor) {
      const int cTensorIdx = gguf_find_tensor(pGgufCtx, ggml_get_name(pTensor));
      if (cTensorIdx < 0) {
        throw std::runtime_error(format("tensor '%s' not found in the model", ggml_get_name(pTensor)));
      }

      vOffs = gguf_get_data_offset(pGgufCtx) + gguf_get_tensor_offset(pGgufCtx, cTensorIdx);
      if (vOffs + ggml_nbytes(pTensor) < vOffs || vOffs + ggml_nbytes(pTensor) > pFile->size()) {
        throw std::runtime_error(format(
          "tensor '%s' data is not within the file bounds, model is corrupted or incomplete", ggml_get_name(pTensor)));
      }
    }
  };

  // custom comparator to sort weights more nicely by layer
  struct weight_name_comparer {
    bool operator()(const std::string & pA, const std::string & pB) const {
      int vALayer = -1;
      int vBLayer = -1;
      sscanf(pA.c_str(), "blk.%d.", &vALayer);
      sscanf(pB.c_str(), "blk.%d.", &vBLayer);
      if (vALayer != vBLayer) {
        return vALayer < vBLayer;
      }
      return pA < pB;
    }
  };

  static const int cTensorNotRequired   = 1 << 0;
  static const int cTensorDuplicated    = 1 << 1;
  static const int cTensorSkip          = 1 << 2;
  static const int cTensorSkipIfVirtual = 1 << 3;
  static const int cTensorAllowReshape  = 1 << 4;
  static const int cTensorReadLazy =
    1 << 5;  // read rows on demand instead of loading whole tensor; requires mmap for now

  int vNKv      = 0;
  int vNTensors = 0;
  int vNCreated = 0;

  uint64_t vNElements = 0;
  size_t   vNBytes    = 0;

  bool vUseMmap     = false;
  bool vUseDirectIo = false;
  bool vCheckTensors;
  bool vNoAlloc;
  bool vLoadMtp;

  // handle TENSOR_READ_LAZY
  // use case: keep PLE / engrams embd tensors on disk, read them on demand
  struct lazy_read {
    // set by the caller before the create_tensor() calls
    enum dgs_lazy_mode vMode = cDgsLazyModeOff;

    // decide whether this tensor is read lazily
    // pass w to also record it, or nullptr to only ask
    bool mAdd(const std::string & name, const ggml_tensor * pT, const dgs_tensor_weight * pW);

    bool mAny() const { return !dRanges.empty(); }

    bool has(const ggml_tensor * pT) const { return sTensors.count(ggml_get_name(pT)) > 0; }

    const dgs_mmap::ranges & mForFile(uint32_t pIdx) const {
      static const dgs_mmap::ranges lNone;

      const auto lIt = dRanges.find(pIdx);
      return lIt == dRanges.end() ? lNone : lIt->second;
    }

    // lazy tensors are gathered on the host, so no offload setting applies to them
    static ggml_backend_buffer_type_t mBuft();

  private:
    std::map<uint32_t, dgs_mmap::ranges> dRanges;
    std::set<std::string>                sTensors;
  } vLazy;

  dgs_files lFiles;
  dgs_ftype vFtype;
  dgs_fver  vFver;

  dgs_mmaps lMappings;

  std::map<std::string, dgs_tensor_weight, weight_name_comparer> dWeightsMap;
  std::unordered_map<std::string, dgs_model_kv_override>         dKvOverrides;
  const dgs_model_tensor_buft_override *                         vTensorBuftOverrides;

  gguf_context_ptr              vMetadataPtr;
  struct gguf_context *         vMetadata;  // either metadata_ptr.get() or externally set
  dgs_model_set_tensor_data_t   vSetTensorData;
  void *                        vSetTensorDataUd;
  std::vector<ggml_context_ptr> lContexts;

  std::string vArchName;
  LLM_KV      vLlmKv = LLM_KV(cLlmArchUnknown);

  size_t                                 vSizeDone = 0;
  size_t                                 vSizeData = 0;
  std::vector<std::pair<size_t, size_t>> lMmapsUsed;

  // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
  struct ggml_backend_buft_comparator {
    bool operator()(const ggml_backend_buffer_type_t & pLhs, const ggml_backend_buffer_type_t & pRhs) const {
      return strcmp(ggml_backend_buft_name(pLhs), ggml_backend_buft_name(pRhs)) < 0;
    }
  };

  // lazy tensors need dedicated context
  struct ctx_key {
    ggml_backend_buffer_type_t vBuft;
    bool                       vLazy;
  };

  struct ctx_key_comparator {
    bool operator()(const ctx_key & pLhs, const ctx_key & pRhs) const {
      if (pLhs.vLazy != pRhs.vLazy) {
        return pLhs.vLazy < pRhs.vLazy;
      }
      return strcmp(ggml_backend_buft_name(pLhs.vBuft), ggml_backend_buft_name(pRhs.vBuft)) < 0;
    }
  };

  std::map<ctx_key, ggml_context_ptr, ctx_key_comparator> dCtxMap;

  // track tensors that had to be moved for debugging:
  size_t                     vNTensorsMoved = 0;
  std::string                vFirstTensorMovedName;
  std::string                vFirstTensorMovedTypeName;
  ggml_backend_buffer_type_t vFirstMovedFromBuft = nullptr;
  ggml_backend_buffer_type_t vFirstMovedToBuft   = nullptr;

  dgs_model_loader(
    struct gguf_context *                  pMetadata,
    dgs_model_set_tensor_data_t            pSetTensorData,
    void *                                 pSetTensorDataUd,
    const std::string &                    pFname,
    std::vector<std::string> &             pSplits,  // optional, only need if the split does not follow naming scheme
    FILE *                                 pFile,
    dgs_load_mode                          pLoadMode,
    bool                                   pCheckTensors,
    bool                                   pNoAlloc,
    bool                                   pLoadMtp,
    const dgs_model_kv_override *          pParamOverridesP,
    const dgs_model_tensor_buft_override * pParamTensorBuftOverridesP);

  template <typename T>
  typename std::enable_if<std::is_integral<T>::value, bool>::type get_arr_n(const std::string & pKey,
                                                                            T &                 pResult,
                                                                            bool                pRequired = true);

  template <typename T>
  typename std::enable_if<std::is_integral<T>::value, bool>::type get_arr_n(enum llm_kv pKid,
                                                                            T &         pResult,
                                                                            bool        pRequired = true);

  template <typename T> bool get_arr(const std::string & pKey, std::vector<T> & pResult, bool pRequired = true);

  template <typename T, size_t N_MAX>
  bool get_arr(const std::string & pKey, std::array<T, N_MAX> & pResult, bool pRequired = true);

  template <typename T> bool get_arr(enum llm_kv pKid, T & pResult, bool pRequired = true);

  template <typename T> bool get_key(const std::string & pKey, T & pResult, bool pRequired = true);

  template <typename T> bool get_key(enum llm_kv pKid, T & pResult, bool pRequired = true);

  template <typename T, size_t N_MAX>
  bool get_key_or_arr(const std::string & pKey, std::array<T, N_MAX> & pResult, uint32_t n, bool pRequired = true);

  template <typename T> bool get_key_or_arr(enum llm_kv pKid, T & pResult, uint32_t n, bool pRequired = true);

  bool get_key_or_arr(enum llm_kv pKid, uint32_t & pResult, bool pRequired = true);

  std::string mGetArchName() const;

  enum llm_arch mGetArch() const;

  const dgs_tensor_weight * mGetWeight(const char * name) const;

  const dgs_tensor_weight & mRequireWeight(const char * name) const;

  struct ggml_tensor * mGetTensorMeta(const char * name) const;

  struct ggml_tensor * mRequireTensorMeta(const std::string & name) const;

  const struct ggml_tensor * mCheckTensorDims(const std::string &          name,
                                              const std::vector<int64_t> & pNe,
                                              bool                         pRequired,
                                              bool                         pAllowReshape) const;

  struct ggml_tensor * mCreateTensor(const dgs_hparams &                    pHparams,
                                     const buft_list_t *                    pBuftListCpu,
                                     const buft_list_t *                    pBuftListInput,
                                     const buft_list_t *                    pBuftListOutput,
                                     const buft_list_t *                    pBuftListLayer,
                                     const LLM_TN_IMPL &                    pTn,
                                     const std::initializer_list<int64_t> & pNe,
                                     int                                    pFlags);

  void mDoneGettingTensors(bool pPartial = false) const;

  void mInitMappings(bool pPrefetch = true, dgs_mlocks * pMlockMmaps = nullptr);

  void mGetMappingRange(size_t * pFirst, size_t * pLast, void ** pAddr, int pIdx, ggml_context * pCtx) const;

  // release a weight's mmap pages
  void mUnmapWeight(const dgs_tensor_weight & pW) const;

  // read a byte range of a weight's data
  // with mmap, returns a pointer into the mapping, otherwise reads into buf and returns buf
  const void * mLoadDataRange(const dgs_tensor_weight & pW, size_t pOffs, size_t size, void * pBuf) const;

  // Returns false if cancelled by progress_callback
  bool mLoadAllData(struct ggml_context * pCtx,
                    dgs_buf_map &         pBufs,
                    dgs_mlocks *          pLmlocks,
                    dgs_progress_callback pProgressCallback,
                    void *                pProgressCallbackUserData);

  std::string mFtypeName() const;

  void mPrintInfo() const;
};
