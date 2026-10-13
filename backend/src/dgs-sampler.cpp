#include "dgs-sampler.h"

#include "dgs-grammar.h"
#include "dgs-impl.h"
#include "dgs-vocab.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>

// the ring buffer works similarly to std::deque, but with a fixed capacity
template <typename T> struct ring_buffer {
  ring_buffer(size_t pCap) : vCapacity(pCap), data(pCap) {}

  T & front() {
    if (vSz == 0) {
      throw std::runtime_error("ring buffer is empty");
    }
    return data[vFirst];
  }

  const T & front() const {
    if (vSz == 0) {
      throw std::runtime_error("ring buffer is empty");
    }
    return data[vFirst];
  }

  T & back() {
    if (vSz == 0) {
      throw std::runtime_error("ring buffer is empty");
    }
    return data[vPos];
  }

  const T & back() const {
    if (vSz == 0) {
      throw std::runtime_error("ring buffer is empty");
    }
    return data[vPos];
  }

  void push_back(const T & value) {
    if (vCapacity == 0) {
      throw std::runtime_error("ring buffer: capacity is zero");
    }

    if (vSz == vCapacity) {
      // advance the start when buffer is full
      vFirst = (vFirst + 1) % vCapacity;
    } else {
      vSz++;
    }
    data[vPos] = value;
    vPos       = (vPos + 1) % vCapacity;
  }

  T mPopFront() {
    if (vSz == 0) {
      throw std::runtime_error("ring buffer is empty");
    }
    T value = data[vFirst];
    vFirst  = (vFirst + 1) % vCapacity;
    vSz--;
    return value;
  }

  //T & operator[](size_t i) {
  //    if (i >= sz) {
  //        throw std::runtime_error("ring buffer: index out of bounds");
  //    }
  //    return data[(first + i) % capacity];
  //}

  //const T & at(size_t i) const {
  //    if (i >= sz) {
  //        throw std::runtime_error("ring buffer: index out of bounds");
  //    }
  //    return data[(first + i) % capacity];
  //}

  const T & mRat(size_t pI) const {
    if (pI >= vSz) {
      throw std::runtime_error("ring buffer: index out of bounds");
    }
    return data[(vFirst + vSz - pI - 1) % vCapacity];
  }

  std::vector<T> mToVector() const {
    std::vector<T> lResult;
    lResult.reserve(vSz);
    for (size_t vI = 0; vI < vSz; vI++) {
      lResult.push_back(data[(vFirst + vI) % vCapacity]);
    }
    return lResult;
  }

  void clear() {
    // here only reset the status of the buffer
    vSz    = 0;
    vFirst = 0;
    vPos   = 0;
  }

  bool empty() const { return vSz == 0; }

  size_t size() const { return vSz; }

  size_t vCapacity = 0;
  size_t vSz       = 0;
  size_t vFirst    = 0;
  size_t vPos      = 0;

  std::vector<T> data;
};

// writes result in res, does not mutate cur
static void fDgsTokenDataArrayPartialSort(const dgs_token_data_array &  pCur,
                                          int                           pNpartial,
                                          std::vector<dgs_token_data> & pRes) {
  static const auto cComp = [](const dgs_token_data & pA, const dgs_token_data & pB) {
    return pA.vLogit > pB.vLogit;
  };

  constexpr int   cNbuckets    = 128;
  constexpr float cBucketLow   = -10.0f;
  constexpr float cBucketHigh  = 10.0f;
  constexpr float cBucketScale = cNbuckets / (cBucketHigh - cBucketLow);
  constexpr float cBucketInter = -cBucketLow * cBucketScale;

  std::vector<int> lBucketIdx;
  std::vector<int> lHisto(cNbuckets, 0);

  std::vector<dgs_token_data *> lBucketPtrs;

  lBucketIdx.reserve(pCur.size);

  for (int vI = 0; vI < (int) pCur.size; ++vI) {
    const float cVal = pCur.data[vI].vLogit;
    int vIb = int(cBucketScale * cVal + cBucketInter);  //nbuckets * (val - bucket_low) / (bucket_high - bucket_low);
    vIb     = std::max(0, std::min(cNbuckets - 1, vIb));
    lBucketIdx.push_back(vIb);
    ++lHisto[vIb];
  }
  int vNhave = 0;
  int vIb    = cNbuckets - 1;
  for (; vIb >= 0; --vIb) {
    vNhave += lHisto[vIb];
    if (vNhave >= pNpartial) {
      break;
    }
  }
  pRes.resize(vNhave);
  auto * vPtr = pRes.data();
  lBucketPtrs.reserve(cNbuckets - vIb);
  for (int vJ = cNbuckets - 1; vJ >= vIb; --vJ) {
    lBucketPtrs.push_back(vPtr);
    vPtr += lHisto[vJ];
  }
  for (int vI = 0; vI < (int) pCur.size; ++vI) {
    int vJ = lBucketIdx[vI];
    if (vJ >= vIb) {
      *lBucketPtrs[cNbuckets - 1 - vJ]++ = pCur.data[vI];
    }
  }

  vPtr       = pRes.data();
  int vNdone = 0;
  for (int vJ = cNbuckets - 1; vJ > vIb; --vJ) {
    std::sort(vPtr, vPtr + lHisto[vJ], cComp);
    vPtr += lHisto[vJ];
    vNdone += lHisto[vJ];
  }
  std::partial_sort(vPtr, vPtr + pNpartial - vNdone, vPtr + lHisto[vIb], cComp);
}

// reduces the size of cur_p to npartial, keeping only the top npartial elements
static void fDgsTokenDataArrayPartialSortInplace(dgs_token_data_array * pCurP, int pNpartial) {
  static const auto cComp = [](const dgs_token_data & pA, const dgs_token_data & pB) {
    return pA.vLogit > pB.vLogit;
  };

  if (pNpartial <= 128) {
    std::partial_sort(pCurP->data, pCurP->data + pNpartial, pCurP->data + pCurP->size, cComp);

    pCurP->size    = pNpartial;
    pCurP->vSorted = true;

    return;
  }

  std::vector<dgs_token_data> lTmp;

  fDgsTokenDataArrayPartialSort(*pCurP, pNpartial, lTmp);

  std::copy(lTmp.data(), lTmp.data() + pNpartial, pCurP->data);

  pCurP->size    = pNpartial;
  pCurP->vSorted = true;
}

static int fDgsSampleDist(dgs_token_data_array * pCurP, std::mt19937 & pRng) {
  // iterator for the probabilities
#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif

  struct probs_iterator {
    typedef std::input_iterator_tag iterator_category;
    typedef float                   value_type;
    typedef float *                 pointer;
    typedef float &                 reference;
    typedef ptrdiff_t               difference_type;

    const dgs_token_data * data;

    bool operator==(const probs_iterator & pOther) const { return data == pOther.data; }

    bool operator!=(const probs_iterator & pOther) const { return data != pOther.data; }

    const float & operator*() const { return data->vP; }

    probs_iterator & operator++() {
      ++data;
      return *this;
    }

    probs_iterator operator++(int) {
      probs_iterator vTmp = *this;
      ++data;
      return vTmp;
    }
  };

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif

  std::discrete_distribution<int> vDist(probs_iterator{ pCurP->data }, probs_iterator{ pCurP->data + pCurP->size });

  return vDist(pRng);
}

/*
static void dgs_log_softmax(float * array, size_t size) {
    float max_l = *std::max_element(array, array + size);
    float sum = 0.f;
    for (size_t i = 0; i < size; ++i) {
        float p = expf(array[i] - max_l);
        sum += p;
        array[i] = p;
    }

    for (size_t i = 0; i < size; ++i) {
        array[i] = logf(array[i] / sum);
    }
}
*/

static void fDgsSamplerTempImpl(dgs_token_data_array * pCurP, float pTemp) {
  if (pCurP->size == 0) {
    return;
  }

  if (pTemp <= 0.0f) {
    // find the token with the highest logit and set the rest to -inf
    size_t vMaxI = 0;
    float  vMaxL = pCurP->data[0].vLogit;

    for (size_t vI = 1; vI < pCurP->size; ++vI) {
      if (pCurP->data[vI].vLogit > vMaxL) {
        pCurP->data[vMaxI].vLogit = -INFINITY;
        vMaxI                     = vI;
        vMaxL                     = pCurP->data[vI].vLogit;
      } else {
        pCurP->data[vI].vLogit = -INFINITY;
      }
    }

    return;
  }

  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    pCurP->data[vI].vLogit /= pTemp;
  }
}

static void fDgsSamplerSoftmaxImpl(dgs_token_data_array * pCurP, bool pDoSort) {
  GGML_ASSERT(pCurP->size > 0);

  // Sort the logits in descending order if requested
  if (pDoSort && !pCurP->vSorted) {
    fDgsTokenDataArrayPartialSortInplace(pCurP, pCurP->size);
  }

  float vMaxL = pCurP->data[0].vLogit;
  if (!pCurP->vSorted) {
    for (size_t vI = 1; vI < pCurP->size; ++vI) {
      vMaxL = std::max(vMaxL, pCurP->data[vI].vLogit);
    }
  }

  float vCumSum = 0.0f;

  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    float vP           = expf(pCurP->data[vI].vLogit - vMaxL);
    pCurP->data[vI].vP = vP;
    vCumSum += vP;
  }

  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    pCurP->data[vI].vP /= vCumSum;
  }
}

static void fDgsSamplerTopKImpl(dgs_token_data_array * pCurP, int32_t pK) {
  // if (k >= (int32_t)cur_p->size) {
  //     return;
  // }

  if (pK <= 0) {
    return;
  }

  pK = std::min(pK, (int) pCurP->size);

  // Sort scores in descending order
  if (!pCurP->vSorted) {
    fDgsTokenDataArrayPartialSortInplace(pCurP, pK);
  }

  pCurP->size = pK;
}

static uint32_t fGetRngSeed(uint32_t pSeed) {
  if (pSeed == DGS_DEFAULT_SEED) {
    // use system clock if std::random_device is not a true RNG
    static bool vIsRdPrng = std::random_device().entropy() == 0;
    if (vIsRdPrng) {
      return (uint32_t) std::chrono::system_clock::now().time_since_epoch().count();
    }
    std::random_device vRd;
    return vRd();
  }
  return pSeed;
}

// dgs_sampler API

struct dgs_sampler * fDgsSamplerInit(struct dgs_sampler_i * pIface, dgs_sampler_context_t pCtx) {
  return new dgs_sampler{
    /* .iface = */ pIface,
    /* .ctx   = */ pCtx,
  };
}

const char * fDgsSamplerName(const struct dgs_sampler * pSmpl) {
  if (!pSmpl->vIface) {
    return "(null)";
  }

  return pSmpl->vIface->name(pSmpl);
}

void fDgsSamplerAccept(struct dgs_sampler * pSmpl, dgs_token token) {
  if (!pSmpl) {
    return;
  }

  if (pSmpl->vIface->accept) {
    pSmpl->vIface->accept(pSmpl, token);
  }
}

void fDgsSamplerApply(struct dgs_sampler * pSmpl, struct dgs_token_data_array * pCurP) {
  if (!pSmpl) {
    return;
  }

  GGML_ASSERT(pSmpl->vIface->apply);
  pSmpl->vIface->apply(pSmpl, pCurP);
}

void fDgsSamplerReset(struct dgs_sampler * pSmpl) {
  if (!pSmpl) {
    return;
  }

  if (pSmpl->vIface->reset) {
    pSmpl->vIface->reset(pSmpl);
  }
}

struct dgs_sampler * fDgsSamplerClone(const struct dgs_sampler * pSmpl) {
  if (!pSmpl) {
    return nullptr;
  }

  if (pSmpl->vIface->clone) {
    return pSmpl->vIface->clone(pSmpl);
  }

  if (pSmpl->vCtx == nullptr) {
    return fDgsSamplerInit(
      /* .iface = */ pSmpl->vIface,
      /* .ctx   = */ nullptr);
  }

  GGML_ABORT("the sampler does not support cloning");
}

void fDgsSamplerFree(struct dgs_sampler * pSmpl) {
  if (pSmpl == nullptr) {
    return;
  }

  if (pSmpl->vIface->free) {
    pSmpl->vIface->free(pSmpl);
  }

  delete pSmpl;
}

// empty sampler

struct dgs_sampler_empty {
  const char * name;
};

static struct dgs_sampler * fDgsSamplerInitEmpty(const char * name);

static const char * fDgsSamplerEmptyName(const struct dgs_sampler * pSmpl) {
  auto * vCtx = (dgs_sampler_empty *) pSmpl->vCtx;
  return vCtx->name;
}

static void fDgsSamplerEmptyAccept(struct dgs_sampler * pSmpl, dgs_token token) {
  GGML_UNUSED(pSmpl);
  GGML_UNUSED(token);
}

static void fDgsSamplerEmptyApply(struct dgs_sampler * pSmpl, dgs_token_data_array * pCurP) {
  GGML_UNUSED(pSmpl);
  GGML_UNUSED(pCurP);
}

static void fDgsSamplerEmptyReset(struct dgs_sampler * pSmpl) {
  GGML_UNUSED(pSmpl);
}

static struct dgs_sampler * fDgsSamplerEmptyClone(const struct dgs_sampler * pSmpl) {
  auto * vCtx = (dgs_sampler_empty *) pSmpl->vCtx;
  return fDgsSamplerInitEmpty(vCtx->name);
}

static void fDgsSamplerEmptyFree(struct dgs_sampler * pSmpl) {
  delete (dgs_sampler_empty *) pSmpl->vCtx;
}

static bool fDgsSamplerEmptyBackendInit(struct dgs_sampler *       pSmpl,
                                        ggml_backend_buffer_type_t pBuft,
                                        uint32_t                   pNOutputsMaxPerSeq) {
  GGML_UNUSED(pSmpl);
  GGML_UNUSED(pBuft);
  GGML_UNUSED(pNOutputsMaxPerSeq);

  return true;
}

static void fDgsSamplerEmptyBackendAccept(struct dgs_sampler * pSmpl,
                                          ggml_context *       pCtx,
                                          ggml_cgraph *        pGf,
                                          struct ggml_tensor * pSelectedToken) {
  GGML_UNUSED(pSmpl);
  GGML_UNUSED(pCtx);
  GGML_UNUSED(pGf);
  GGML_UNUSED(pSelectedToken);
}

static void fDgsSamplerEmptyBackendApply(struct dgs_sampler *      pSmpl,
                                         struct ggml_context *     pCtx,
                                         struct ggml_cgraph *      pGf,
                                         struct dgs_sampler_data * data) {
  GGML_UNUSED(pSmpl);
  GGML_UNUSED(pCtx);
  GGML_UNUSED(pGf);
  GGML_UNUSED(data);
}

static void fDgsSamplerEmptyBackendSetInput(struct dgs_sampler * pSmpl) {
  GGML_UNUSED(pSmpl);
}

static struct dgs_sampler_i vDgsSamplerEmptyI = {
  /* .name              = */ fDgsSamplerEmptyName,
  /* .accept            = */ fDgsSamplerEmptyAccept,
  /* .apply             = */ fDgsSamplerEmptyApply,
  /* .reset             = */ fDgsSamplerEmptyReset,
  /* .clone             = */ fDgsSamplerEmptyClone,
  /* .free              = */ fDgsSamplerEmptyFree,
  /* .backend_init      = */ fDgsSamplerEmptyBackendInit,
  /* .backend_accept    = */ fDgsSamplerEmptyBackendAccept,
  /* .backend_apply     = */ fDgsSamplerEmptyBackendApply,
  /* .backend_set_input = */ fDgsSamplerEmptyBackendSetInput,
  /* .backend_reset     = */ nullptr,
  /* .copy_state        = */ nullptr,
};

struct dgs_sampler * fDgsSamplerInitEmpty(const char * name) {
  return fDgsSamplerInit(
    /* .iface = */ &vDgsSamplerEmptyI,
    /* .ctx   = */ new dgs_sampler_empty{
      /* .name = */ name,
    });
}

// common backend sampler functionality
//
// +name : means that the sampler is support and will run on the backend
// -name : means that a ggml operator is not supported by the backend
//
struct dgs_sampler_backend {
  dgs_sampler_backend(const char * name) : name(name), vNameExt(name), vIsInit(false), vSupport(false) {}

  const char * mGetName() {
    if (!vIsInit) {
      return name.c_str();
    }

    if (vSupport) {
      vNameExt = "+" + name;
    } else {
      vNameExt = "-" + name;
    }

    return vNameExt.c_str();
  }

  void init(bool pSupport) {
    GGML_ASSERT(this->vIsInit == false);

    this->vIsInit  = true;
    this->vSupport = pSupport;
  }

  // copy the state that is not tied to the current sampling graph
  // samplers that hold only immutable configuration can use this as is
  void mCopyState(const dgs_sampler_backend & pSrc) { GGML_UNUSED(pSrc); }

private:
  std::string name;
  std::string vNameExt;

  bool vIsInit;
  bool vSupport;
};

// .copy_state for samplers deriving from dgs_sampler_backend
template <typename T>
static void fDgsSamplerBackendCopyState(const struct dgs_sampler * pSrc, struct dgs_sampler * pDst) {
  ((T *) pDst->vCtx)->mCopyState(*(const T *) pSrc->vCtx);
}

struct dgs_sampler_backend_probe {
  ggml_context_ptr vCtx;
  ggml_cgraph *    vGf;
};

static dgs_sampler_backend_probe fDgsSamplerBackendProbeGraph(dgs_sampler * pSampler,
                                                              int64_t       pNCandidates,
                                                              uint32_t      pMaxNodes,
                                                              bool          pWithCandidates) {
  ggml_init_params vParams = {
    /*.mem_size   =*/pMaxNodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(pMaxNodes, false),
    /*.mem_buffer =*/nullptr,
    /*.no_alloc   =*/true,
  };

  ggml_context_ptr vCtxPtr{ ggml_init(vParams) };
  if (!vCtxPtr) {
    throw std::runtime_error(format("failed to create ggml context"));
  }

  auto * vCtx = vCtxPtr.get();
  auto * vGf  = ggml_new_graph_custom(vCtx, pMaxNodes, false);

  dgs_sampler_data data = {
    /*.logits       =*/ggml_new_tensor_1d(vCtx, GGML_TYPE_F32, pNCandidates),
    /*.probs        =*/nullptr,
    /*.sampled      =*/nullptr,
    /*.candidates   =*/pWithCandidates ? ggml_new_tensor_1d(vCtx, GGML_TYPE_I32, pNCandidates) : nullptr,
  };

  if (pSampler->vIface->vBackendReset) {
    pSampler->vIface->vBackendReset(pSampler);
  }
  pSampler->vIface->vBackendApply(pSampler, vCtx, vGf, &data);

  for (auto * vOutput : { data.vLogits, data.vProbs, data.vSampled, data.vCandidates }) {
    if (vOutput) {
      ggml_build_forward_expand(vGf, vOutput);
    }
  }

  if (pSampler->vIface->vBackendReset) {
    pSampler->vIface->vBackendReset(pSampler);
  }

  return { std::move(vCtxPtr), vGf };
}

static uint32_t fDgsSamplerBackendProbeNNodes(const dgs_sampler_backend_probe & pProbe) {
  uint32_t vNTensors = 0;
  for (auto * vTensor = ggml_get_first_tensor(pProbe.vCtx.get()); vTensor;
       vTensor        = ggml_get_next_tensor(pProbe.vCtx.get(), vTensor)) {
    ++vNTensors;
  }

  return std::max<uint32_t>(ggml_graph_n_nodes(pProbe.vGf), vNTensors);
}

// check if all ggml ops used by the sampler are supported by the backend
static bool fDgsSamplerBackendSupport(dgs_sampler * pSmpl, ggml_backend_buffer_type_t pBuft) {
  auto * vDevice = ggml_backend_buft_get_device(pBuft);
  if (!vDevice) {
    // CPU backend always supported
    return true;
  }

  auto vProbe = fDgsSamplerBackendProbeGraph(pSmpl, 128 * 1024, GGML_DEFAULT_GRAPH_SIZE, true);

  for (int vI = 0; vI < ggml_graph_n_nodes(vProbe.vGf); vI++) {
    struct ggml_tensor * vOp = ggml_graph_node(vProbe.vGf, vI);

    if (!ggml_backend_dev_supports_op(vDevice, vOp)) {
      DGS_LOG_WARN("%s: device '%s' does not have support for op %s needed for sampler '%s'\n", __func__,
                   ggml_backend_dev_name(vDevice), ggml_op_name(vOp->op), pSmpl->vIface->name(pSmpl));

      return false;
    }
  }

  return true;
}

// sampler chain

static const char * fDgsSamplerChainName(const struct dgs_sampler * /*smpl*/) {
  return "chain";
}

static void fDgsSamplerChainAccept(struct dgs_sampler * pSmpl, dgs_token token) {
  auto * vChain = (dgs_sampler_chain *) pSmpl->vCtx;

  time_meas vTm(vChain->vTSampleUs, vChain->vParams.vNoPerf);

  for (auto & vSmpl : vChain->lSamplers) {
    fDgsSamplerAccept(vSmpl.vPtr, token);
  }

  vChain->vNSample++;
}

static void fDgsSamplerChainApply(struct dgs_sampler * pSmpl, dgs_token_data_array * pCurP) {
  auto * vChain = (dgs_sampler_chain *) pSmpl->vCtx;

  time_meas vTm(vChain->vTSampleUs, vChain->vParams.vNoPerf);

  bool vIsBackend = vChain->vIsInit;

  for (auto & vSmpl : vChain->lSamplers) {
    if (vIsBackend && vSmpl.vIsBackend) {
      continue;
    }

    vIsBackend = false;

    if (vSmpl.vPtr->vIface->apply == nullptr) {
      continue;
    }

    fDgsSamplerApply(vSmpl.vPtr, pCurP);
  }
}

static void fDgsSamplerChainReset(struct dgs_sampler * pSmpl) {
  auto * vChain = (dgs_sampler_chain *) pSmpl->vCtx;

  for (auto & vSmpl : vChain->lSamplers) {
    fDgsSamplerReset(vSmpl.vPtr);
  }
}

static struct dgs_sampler * fDgsSamplerChainClone(const struct dgs_sampler * pSmpl) {
  const auto * vChainSrc = (const dgs_sampler_chain *) pSmpl->vCtx;

  auto * vResult = fDgsSamplerChainInit(vChainSrc->vParams);

  for (const auto & vSmpl : vChainSrc->lSamplers) {
    fDgsSamplerChainAdd(vResult, fDgsSamplerClone(vSmpl.vPtr));
  }

  return vResult;
}

static void fDgsSamplerChainFree(struct dgs_sampler * pSmpl) {
  auto * vChain = (dgs_sampler_chain *) pSmpl->vCtx;

  for (auto & vSmpl : vChain->lSamplers) {
    fDgsSamplerFree(vSmpl.vPtr);
  }

  delete vChain;
}

static bool fDgsSamplerChainBackendInit(struct dgs_sampler *       pSmpl,
                                        ggml_backend_buffer_type_t pBuft,
                                        uint32_t                   pNOutputsMaxPerSeq) {
  auto * vChain = (dgs_sampler_chain *) pSmpl->vCtx;

  GGML_ASSERT(vChain->vIsInit == false && "dgs_sampler_chain_backend_init() called twice");

  vChain->vIsInit = true;

  bool vRes           = true;
  bool vBackendPrefix = true;

  for (auto & vSmpl : vChain->lSamplers) {
    bool vCurPrefix = vBackendPrefix;

    // to be able to run a sampler on the backend, it has to:
    // - have the .backend_init() API implemented
    // - return true during .backend_init()
    // - support the requested per-sequence output limit
    if (vCurPrefix && vSmpl.vPtr->vIface->vBackendInit) {
      if (!vSmpl.vPtr->vIface->vBackendInit(vSmpl.vPtr, pBuft, pNOutputsMaxPerSeq)) {
        vCurPrefix = false;
      }
    } else {
      vCurPrefix = false;
    }

    vSmpl.vIsBackend = vCurPrefix;
    vBackendPrefix   = vCurPrefix;

    vRes = vRes && vCurPrefix;
  }

  auto vProbe     = fDgsSamplerBackendProbeGraph(pSmpl, 128 * 1024, GGML_DEFAULT_GRAPH_SIZE, false);
  vChain->vNNodes = fDgsSamplerBackendProbeNNodes(vProbe);

  return vRes;
}

static void fDgsSamplerChainBackendAccept(struct dgs_sampler * pSmpl,
                                          ggml_context *       pCtx,
                                          ggml_cgraph *        pGf,
                                          struct ggml_tensor * pSelectedToken) {
  auto * vChain = (dgs_sampler_chain *) pSmpl->vCtx;

  for (auto & vSmpl : vChain->lSamplers) {
    if (!vSmpl.vIsBackend) {
      break;
    }

    if (vSmpl.vPtr->vIface->vBackendAccept) {
      vSmpl.vPtr->vIface->vBackendAccept(vSmpl.vPtr, pCtx, pGf, pSelectedToken);
    }
  }
}

static void fDgsSamplerChainBackendApply(struct dgs_sampler *      pSmpl,
                                         struct ggml_context *     pCtx,
                                         struct ggml_cgraph *      pGf,
                                         struct dgs_sampler_data * data) {
  auto * vChain = (dgs_sampler_chain *) pSmpl->vCtx;

  GGML_ASSERT(vChain->vIsInit && "dgs_sampler_chain_backend_init() not called");

  for (auto & vSmpl : vChain->lSamplers) {
    if (!vSmpl.vIsBackend) {
      break;
    }

    if (vSmpl.vPtr->vIface->vBackendApply) {
      vSmpl.vPtr->vIface->vBackendApply(vSmpl.vPtr, pCtx, pGf, data);
    }
  }
}

static void fDgsSamplerChainBackendSetInput(struct dgs_sampler * pSmpl) {
  auto * vChain = (dgs_sampler_chain *) pSmpl->vCtx;

  for (auto & vSmpl : vChain->lSamplers) {
    if (!vSmpl.vIsBackend) {
      break;
    }

    if (vSmpl.vPtr->vIface->vBackendSetInput) {
      vSmpl.vPtr->vIface->vBackendSetInput(vSmpl.vPtr);
    }
  }
}

static void fDgsSamplerChainBackendReset(struct dgs_sampler * pSmpl) {
  auto * vChain = (dgs_sampler_chain *) pSmpl->vCtx;

  for (auto & vEntry : vChain->lSamplers) {
    if (!vEntry.vIsBackend) {
      break;
    }
    if (vEntry.vPtr->vIface->vBackendReset) {
      vEntry.vPtr->vIface->vBackendReset(vEntry.vPtr);
    }
  }
}

static void fDgsSamplerChainCopyState(const struct dgs_sampler * pSrc, struct dgs_sampler * pDst) {
  const auto * vSrcChain = (const dgs_sampler_chain *) pSrc->vCtx;
  auto *       vDstChain = (dgs_sampler_chain *) pDst->vCtx;

  GGML_ASSERT(vSrcChain->lSamplers.size() == vDstChain->lSamplers.size());

  for (size_t vI = 0; vI < vSrcChain->lSamplers.size(); ++vI) {
    fDgsSamplerCopy(vSrcChain->lSamplers[vI].vPtr, vDstChain->lSamplers[vI].vPtr);
  }

  // note: is_init, n_nodes and is_backend belong to the current sampling graph
  vDstChain->vParams    = vSrcChain->vParams;
  vDstChain->lCur       = vSrcChain->lCur;
  vDstChain->vTSampleUs = vSrcChain->vTSampleUs;
  vDstChain->vNSample   = vSrcChain->vNSample;
}

static struct dgs_sampler_i vDgsSamplerChainI = {
  /* .name              = */ fDgsSamplerChainName,
  /* .accept            = */ fDgsSamplerChainAccept,
  /* .apply             = */ fDgsSamplerChainApply,
  /* .reset             = */ fDgsSamplerChainReset,
  /* .clone             = */ fDgsSamplerChainClone,
  /* .free              = */ fDgsSamplerChainFree,
  /* .backend_init      = */ fDgsSamplerChainBackendInit,
  /* .backend_accept    = */ fDgsSamplerChainBackendAccept,
  /* .backend_apply     = */ fDgsSamplerChainBackendApply,
  /* .backend_set_input = */ fDgsSamplerChainBackendSetInput,
  /* .backend_reset     = */ fDgsSamplerChainBackendReset,
  /* .copy_state        = */ fDgsSamplerChainCopyState,
};

struct dgs_sampler * fDgsSamplerChainInit(struct dgs_sampler_chain_params pParams) {
  return fDgsSamplerInit(
    /* .iface = */ &vDgsSamplerChainI,
    /* .ctx   = */ new dgs_sampler_chain{
      /* .params               = */ pParams,
      /* .is_init              = */ false,
      /* .n_nodes              = */ 0,
      /* .samplers             = */ {},
      /* .cur                  = */ {},
      /* .t_sample_us          = */ 0,
      /* .n_sample             = */ 0,
    });
}

uint32_t fDgsSamplerBackendNNodes(const dgs_sampler * pSampler) {
  GGML_ASSERT(pSampler != nullptr);
  GGML_ASSERT(pSampler->vIface == &vDgsSamplerChainI);

  const auto * vChain = (const dgs_sampler_chain *) pSampler->vCtx;
  GGML_ASSERT(vChain->vIsInit);

  return vChain->vNNodes;
}

dgs_token fDgsSamplerSample(struct dgs_sampler * pSmpl, struct dgs_context * pCtx, int32_t pIdx) {
  const dgs_token   cSampledToken  = fDgsGetSampledTokenIth(pCtx, pIdx);
  const float *     vSampledProbs  = fDgsGetSampledProbsIth(pCtx, pIdx);
  const float *     vSampledLogits = fDgsGetSampledLogitsIth(pCtx, pIdx);
  const dgs_token * vSampledIds    = fDgsGetSampledCandidatesIth(pCtx, pIdx);

  // If a backend sampler has already sampled a token, return it.
  if (cSampledToken != DGS_TOKEN_NULL) {
    DGS_LOG_DEBUG("%s: Backend sampler selected token for idx %d. Skipping CPU samplers\n", __func__, pIdx);
    fDgsSamplerAccept(pSmpl, cSampledToken);
    return cSampledToken;
  }

  const dgs_model * vModel = fDgsGetModel(pCtx);
  const dgs_vocab * vVocab = fDgsModelGetVocab(vModel);

  const int cNVocab = fDgsVocabNTokens(vVocab);

  // use pre-allocated buffer from chain if available, otherwise allocate locally
  std::vector<dgs_token_data> * lCurPtr;
  std::vector<dgs_token_data>   lCurLocal;

  if (pSmpl->vIface == &vDgsSamplerChainI) {
    auto * vChain = (dgs_sampler_chain *) pSmpl->vCtx;
    lCurPtr       = &vChain->lCur;
  } else {
    lCurPtr = &lCurLocal;
  }

  auto & lCur = *lCurPtr;

  if (vSampledProbs) {
    const uint32_t cSampledProbsCount = fDgsGetSampledProbsCountIth(pCtx, pIdx);
    lCur.resize(cSampledProbsCount);
    for (uint32_t vI = 0; vI < cSampledProbsCount; ++vI) {
      lCur[vI] = dgs_token_data{ vSampledIds[vI], vSampledLogits[vI], vSampledProbs[vI] };
    }
  } else if (vSampledLogits) {
    const uint32_t cSampledLogitsCount = fDgsGetSampledLogitsCountIth(pCtx, pIdx);
    lCur.resize(cSampledLogitsCount);
    for (dgs_token vI = 0; vI < (int) cSampledLogitsCount; vI++) {
      lCur[vI] = dgs_token_data{ vSampledIds[vI], vSampledLogits[vI], 0.0f };
    }
  } else {
    const auto * vLogits = fDgsGetLogitsIth(pCtx, pIdx);
    GGML_ASSERT(vLogits != nullptr);
    lCur.resize(cNVocab);
    for (dgs_token vTokenId = 0; vTokenId < cNVocab; vTokenId++) {
      lCur[vTokenId] = dgs_token_data{ vTokenId, vLogits[vTokenId], 0.0f };
    }
  }

  dgs_token_data_array vCurP = {
    /* .data       = */ lCur.data(),
    /* .size       = */ lCur.size(),
    /* .selected   = */ -1,
    /* .sorted     = */ false,
  };

  fDgsSamplerApply(pSmpl, &vCurP);

  GGML_ASSERT(vCurP.vSelected >= 0 && vCurP.vSelected < (int32_t) vCurP.size);

  auto token = vCurP.data[vCurP.vSelected].vId;

  fDgsSamplerAccept(pSmpl, token);

  return token;
}

void fDgsSamplerChainAdd(struct dgs_sampler * pChain, struct dgs_sampler * pSmpl) {
  auto * vP = (dgs_sampler_chain *) pChain->vCtx;
  vP->lSamplers.push_back({
    /* .is_backend = */ false,
    /* .ptr        = */ pSmpl,
  });
}

struct dgs_sampler * fDgsSamplerChainGet(struct dgs_sampler * pChain, int32_t pI) {
  if (pChain == nullptr) {
    return nullptr;
  }

  if (pChain->vIface != &vDgsSamplerChainI) {
    return nullptr;
  }

  if (pI == -1) {
    return pChain;
  }

  const auto * vP = (const dgs_sampler_chain *) pChain->vCtx;

  if (pI < 0 || (size_t) pI >= vP->lSamplers.size()) {
    return nullptr;
  }

  return vP->lSamplers[pI].vPtr;
}

struct dgs_sampler * fDgsSamplerChainRemove(struct dgs_sampler * pChain, int32_t pI) {
  auto * vP = (dgs_sampler_chain *) pChain->vCtx;

  if (pI < 0 || (size_t) pI >= vP->lSamplers.size()) {
    return nullptr;
  }

  auto * vResult = vP->lSamplers[pI].vPtr;
  vP->lSamplers.erase(vP->lSamplers.begin() + pI);

  return vResult;
}

int32_t fDgsSamplerChainN(const struct dgs_sampler * pChain) {
  const auto * vP = (const dgs_sampler_chain *) pChain->vCtx;

  return vP->lSamplers.size();
}

//
// samplers
//

// greedy

struct dgs_sampler_greedy : public dgs_sampler_backend {};

static const char * fDgsSamplerGreedyName(const struct dgs_sampler * pSmpl) {
  auto * vSctx = (dgs_sampler_greedy *) pSmpl->vCtx;
  return vSctx->mGetName();
}

static void fDgsSamplerGreedyReset(struct dgs_sampler * pSmpl) {
  auto * vCtx = (dgs_sampler_greedy *) pSmpl->vCtx;
  GGML_UNUSED(vCtx);
}

static struct dgs_sampler * fDgsSamplerGreedyClone(const struct dgs_sampler * pSmpl) {
  const auto * vCtx    = (const dgs_sampler_greedy *) pSmpl->vCtx;
  auto *       vResult = fDgsSamplerInitGreedy();

  // copy the state
  {
    auto * vResultCtx = (dgs_sampler_greedy *) vResult->vCtx;

    GGML_UNUSED(vCtx);
    GGML_UNUSED(vResultCtx);
  }

  return vResult;
}

static void fDgsSamplerGreedyFree(struct dgs_sampler * pSmpl) {
  delete (dgs_sampler_greedy *) pSmpl->vCtx;
}

static void fDgsSamplerGreedyApply(struct dgs_sampler * /*smpl*/, dgs_token_data_array * pCurP) {
  pCurP->vSelected = 0;
  for (size_t vI = 1; vI < pCurP->size; ++vI) {
    if (pCurP->data[vI].vLogit > pCurP->data[pCurP->vSelected].vLogit) {
      pCurP->vSelected = vI;
    }
  }
}

static bool fDgsSamplerGreedyBackendInit(struct dgs_sampler *       pSmpl,
                                         ggml_backend_buffer_type_t pBuft,
                                         uint32_t                   pNOutputsMaxPerSeq) {
  auto * vSctx = (dgs_sampler_greedy *) pSmpl->vCtx;
  GGML_UNUSED(pNOutputsMaxPerSeq);

  const bool cRes = fDgsSamplerBackendSupport(pSmpl, pBuft);

  vSctx->init(cRes);

  return cRes;
}

static void fDgsSamplerGreedyBackendApply(struct dgs_sampler *      pSmpl,
                                          struct ggml_context *     pCtx,
                                          struct ggml_cgraph *      pGf,
                                          struct dgs_sampler_data * data) {
  GGML_UNUSED(pGf);
  GGML_UNUSED(pSmpl);

  struct ggml_tensor * vLogits = ggml_reshape_1d(pCtx, data->vLogits, ggml_nelements(data->vLogits));

  struct ggml_tensor * vCurl = ggml_argmax(pCtx, vLogits);
  ggml_set_name(vCurl, "greedy_argmax");

  if (data->vCandidates != nullptr) {
    struct ggml_tensor * vCandidates = ggml_reshape_2d(pCtx, data->vCandidates, 1, ggml_nelements(data->vCandidates));
    vCurl                            = ggml_get_rows(pCtx, vCandidates, vCurl);
    ggml_set_name(vCurl, "greedy_sampled_token");
  }

  data->vSampled = vCurl;
}

static struct dgs_sampler_i vDgsSamplerGreedyI = {
  /* .name              = */ fDgsSamplerGreedyName,
  /* .accept            = */ nullptr,
  /* .apply             = */ fDgsSamplerGreedyApply,
  /* .reset             = */ fDgsSamplerGreedyReset,
  /* .clone             = */ fDgsSamplerGreedyClone,
  /* .free              = */ fDgsSamplerGreedyFree,
  /* .backend_init      = */ fDgsSamplerGreedyBackendInit,
  /* .backend_accept    = */ nullptr,
  /* .backend_apply     = */ fDgsSamplerGreedyBackendApply,
  /* .backend_set_input = */ nullptr,
  /* .backend_reset     = */ nullptr,
  /* .copy_state        = */ fDgsSamplerBackendCopyState<dgs_sampler_greedy>,
};

struct dgs_sampler * fDgsSamplerInitGreedy() {
  return fDgsSamplerInit(
    /* .iface = */ &vDgsSamplerGreedyI,
    /* .ctx   = */ new dgs_sampler_greedy{
      ("greedy"),
    });
}

// dist

struct dgs_sampler_dist : public dgs_sampler_backend {
  const uint32_t cSeed;
  uint32_t       vSeedCur;

  std::mt19937 vRng;

  // TODO: refactor + fix naming
  // use a temporary RNG for multi-output sampling so rejected tokens do not advance rng
  bool         vBackendTransactional;
  std::mt19937 vRngBackend;
  size_t       vNBackendDrawsGenerated;
  size_t       vNBackendDrawsCommitted;

  // inputs for the current sampling graph
  std::vector<ggml_tensor *> lInpUniforms;

  void mCopyState(const dgs_sampler_dist & pSrc) {
    // note: inp_uniforms and backend_transactional belong to the current sampling graph
    vSeedCur                = pSrc.vSeedCur;
    vRng                    = pSrc.vRng;
    vRngBackend             = pSrc.vRngBackend;
    vNBackendDrawsGenerated = pSrc.vNBackendDrawsGenerated;
    vNBackendDrawsCommitted = pSrc.vNBackendDrawsCommitted;
  }
};

static const char * fDgsSamplerDistName(const struct dgs_sampler * pSmpl) {
  auto * vSctx = (dgs_sampler_dist *) pSmpl->vCtx;
  return vSctx->mGetName();
}

static void fDgsSamplerDistApply(struct dgs_sampler * pSmpl, dgs_token_data_array * pCurP) {
  auto * vCtx = (dgs_sampler_dist *) pSmpl->vCtx;

  // edge cases
  if (pCurP->size == 0) {
    pCurP->vSelected = -1;
    return;
  }

  pCurP->vSelected = 0;

  std::uniform_real_distribution<double> vDist(0.0f, 1.0f);

  if (pCurP->size == 1) {
    // keep the RNG state aligned with backend sampling, which draws once per output
    (void) vDist(vCtx->vRng);
    pCurP->data[0].vP = 1.0f;
    return;
  }

  // max logit for numerical stability
  float vMaxL = pCurP->data[0].vLogit;
  if (!pCurP->vSorted) {
    for (size_t vI = 1; vI < pCurP->size; ++vI) {
      vMaxL = std::max(vMaxL, pCurP->data[vI].vLogit);
    }
  }

  // apply softmax to obtain the probabilities
  double vSumCum = 0.0f;
  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    float vP           = expf(pCurP->data[vI].vLogit - vMaxL);
    pCurP->data[vI].vP = vP;
    vSumCum += vP;
  }

#if 1
  // sample from the obtained probabilities and normalize the probs in a single pass
  // this is ~3x faster on Mac with full gpt-oss vocab than the version below
  //
  const double cRnd = vDist(vCtx->vRng);

  double       vSumRun = 0.0f;
  const double cSumTgt = vSumCum * cRnd;

  bool vFound = false;
  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    if (!vFound) {
      // accumulate probs until we reach the target sum
      vSumRun += pCurP->data[vI].vP;
      if (vSumRun >= cSumTgt) {
        pCurP->vSelected = vI;
        vFound           = true;
      }
    }

    // normalize probs
    pCurP->data[vI].vP /= vSumCum;
  }

  // fallback to the last token (don't think this can happen)
  assert(found);
  if (!vFound) {
    pCurP->vSelected = pCurP->size - 1;
  }
#else
  // for clarity, this is the same as above but does one pass for normalization and one extra pass for sampling
  for (size_t i = 0; i < cur_p->size; ++i) {
    cur_p->data[i].p /= sum_cum;
  }

  cur_p->selected = fDgsSampleDist(cur_p, ctx->rng);
#endif
}

static void fDgsSamplerDistReset(struct dgs_sampler * pSmpl) {
  auto * vCtx    = (dgs_sampler_dist *) pSmpl->vCtx;
  vCtx->vSeedCur = fGetRngSeed(vCtx->cSeed);
  vCtx->vRng.seed(vCtx->vSeedCur);
  vCtx->vRngBackend             = vCtx->vRng;
  vCtx->vNBackendDrawsGenerated = 0;
  vCtx->vNBackendDrawsCommitted = 0;
}

static struct dgs_sampler * fDgsSamplerDistClone(const struct dgs_sampler * pSmpl) {
  const auto * vCtx    = (const dgs_sampler_dist *) pSmpl->vCtx;
  auto *       vResult = fDgsSamplerInitDist(vCtx->cSeed);

  // copy the state
  {
    auto * vResultCtx = (dgs_sampler_dist *) vResult->vCtx;

    vResultCtx->vSeedCur                = vCtx->vSeedCur;
    vResultCtx->vRng                    = vCtx->vRng;
    vResultCtx->vBackendTransactional   = vCtx->vBackendTransactional;
    vResultCtx->vRngBackend             = vCtx->vRngBackend;
    vResultCtx->vNBackendDrawsGenerated = vCtx->vNBackendDrawsGenerated;
    vResultCtx->vNBackendDrawsCommitted = vCtx->vNBackendDrawsCommitted;
  }

  return vResult;
}

static void fDgsSamplerDistFree(struct dgs_sampler * pSmpl) {
  delete (dgs_sampler_dist *) pSmpl->vCtx;
}

static bool fDgsSamplerDistBackendInit(struct dgs_sampler *       pSmpl,
                                       ggml_backend_buffer_type_t pBuft,
                                       uint32_t                   pNOutputsMaxPerSeq) {
  auto * vSctx = (dgs_sampler_dist *) pSmpl->vCtx;

  const bool cRes = fDgsSamplerBackendSupport(pSmpl, pBuft);

  vSctx->init(cRes);
  vSctx->vBackendTransactional   = pNOutputsMaxPerSeq > 1;
  vSctx->vRngBackend             = vSctx->vRng;
  vSctx->vNBackendDrawsGenerated = 0;
  vSctx->vNBackendDrawsCommitted = 0;

  return cRes;
}

static void fDgsSamplerDistBackendApply(struct dgs_sampler *      pSmpl,
                                        struct ggml_context *     pCtx,
                                        struct ggml_cgraph *      pGf,
                                        struct dgs_sampler_data * data) {
  GGML_UNUSED(pGf);

  auto * vSctx = (dgs_sampler_dist *) pSmpl->vCtx;

  ggml_tensor * vInpUniform = ggml_new_tensor_1d(pCtx, GGML_TYPE_F32, 1);
  ggml_format_name(vInpUniform, "uniform_%zu", vSctx->lInpUniforms.size());
  ggml_set_input(vInpUniform);
  vSctx->lInpUniforms.push_back(vInpUniform);

  // flatten
  struct ggml_tensor * vLogits = ggml_reshape_1d(pCtx, data->vLogits, ggml_nelements(data->vLogits));

  struct ggml_tensor * vProbs = ggml_soft_max(pCtx, vLogits);
  ggml_set_name(vProbs, "dist_probs");

  struct ggml_tensor * vCumsum = ggml_cumsum(pCtx, vProbs);
  ggml_set_name(vCumsum, "dist_cumsum");

  // The uniform tensor has a random value and we subtract this tensor with
  // the cumsum tensor (the uniform tensor will be broadcasted by ggml_sub).
  // Recall that each entry in cumsum is the cumulative probability up to that
  // index so values stay negative while the cumulative total is below the
  // random value, and become zero/positive once the threshold is crossed.
  struct ggml_tensor * vDiff = ggml_sub(pCtx, vCumsum, vInpUniform);
  ggml_set_name(vDiff, "dist_cumsum");

  // The ggml_step function produces a tensor where entries are 1 if the
  // corresponding entry in diff is > 0, and 0 otherwise. So all values up to
  // the index where the cumulative probability exceeds the random value are 0,
  // and all entries after that are 1.
  struct ggml_tensor * vMask = ggml_step(pCtx, vDiff);
  ggml_set_name(vMask, "dist_mask");

  // Taking the sum of the mask gives us the sum of elements after the threshold
  // we are interested in.
  struct ggml_tensor * vIdxf = ggml_sum(pCtx, vMask);
  ggml_set_name(vIdxf, "dist_index_f32");

  // Clamp to prevent out-of-bounds access when computing the index.
  vIdxf = ggml_clamp(pCtx, vIdxf, 1.0f, vMask->ne[0]);

  // Use ggml_scale_bias to scale the index value by -1 and then add the size
  // of the mask to that value so we get the correct index ((-1 * idxf) + n).
  struct ggml_tensor * vIdx = ggml_cast(pCtx, ggml_scale_bias(pCtx, vIdxf, -1.0f, vMask->ne[0]), GGML_TYPE_I32);
  ggml_set_name(vIdx, "dist_index_i32");

  // Map back to original vocab ids if a candidates tensor is available.
  struct ggml_tensor * vSampledToken = vIdx;
  if (data->vCandidates != nullptr) {
    struct ggml_tensor * vCandidates = ggml_reshape_2d(pCtx, data->vCandidates, 1, ggml_nelements(data->vCandidates));

    vSampledToken = ggml_get_rows(pCtx, vCandidates, vIdx);
    ggml_set_name(vSampledToken, "dist_sampled_token");
  }

  data->vSampled = vSampledToken;
  data->vProbs   = vProbs;
}

static void fDgsSamplerDistBackendSetInput(struct dgs_sampler * pSmpl) {
  auto * vSctx = (dgs_sampler_dist *) pSmpl->vCtx;

  GGML_ASSERT(!vSctx->lInpUniforms.empty());

  // We sample in double precision and cast to float to match rnd numbers of
  // dgs_sampler_dist which uses double precision (sampling from
  // std::uniform_real_distribution<double> and
  // std::uniform_real_distribution<float> with same rng will produce
  // different sequences).
  std::uniform_real_distribution<double> vDist(0.0f, 1.0f);

  auto & vRng = vSctx->vBackendTransactional ? vSctx->vRngBackend : vSctx->vRng;

  for (auto * vInpUniform : vSctx->lInpUniforms) {
    GGML_ASSERT(vInpUniform != nullptr);

    const float cRnd = vDist(vRng);
    ggml_backend_tensor_set(vInpUniform, &cRnd, 0, sizeof(float));

    if (vSctx->vBackendTransactional) {
      ++vSctx->vNBackendDrawsGenerated;
    }
  }
}

static void fDgsSamplerDistBackendReset(struct dgs_sampler * pSmpl) {
  auto * vSctx = (dgs_sampler_dist *) pSmpl->vCtx;
  vSctx->lInpUniforms.clear();
}

static void fDgsSamplerDistAccept(struct dgs_sampler * pSmpl, dgs_token token) {
  GGML_UNUSED(token);

  auto * vSctx = (dgs_sampler_dist *) pSmpl->vCtx;

  if (!vSctx->vBackendTransactional || vSctx->vNBackendDrawsCommitted >= vSctx->vNBackendDrawsGenerated) {
    return;
  }

  std::uniform_real_distribution<double> vDist(0.0f, 1.0f);
  (void) vDist(vSctx->vRng);
  ++vSctx->vNBackendDrawsCommitted;
}

static struct dgs_sampler_i vDgsSamplerDistI = {
  /* .name              = */ fDgsSamplerDistName,
  /* .accept            = */ fDgsSamplerDistAccept,
  /* .apply             = */ fDgsSamplerDistApply,
  /* .reset             = */ fDgsSamplerDistReset,
  /* .clone             = */ fDgsSamplerDistClone,
  /* .free              = */ fDgsSamplerDistFree,
  /* .backend_init      = */ fDgsSamplerDistBackendInit,
  /* .backend_accept    = */ nullptr,
  /* .backend_apply     = */ fDgsSamplerDistBackendApply,
  /* .backend_set_input = */ fDgsSamplerDistBackendSetInput,
  /* .backend_reset     = */ fDgsSamplerDistBackendReset,
  /* .copy_state        = */ fDgsSamplerBackendCopyState<dgs_sampler_dist>,
};

struct dgs_sampler * fDgsSamplerInitDist(uint32_t pSeed) {
  auto vSeedCur = fGetRngSeed(pSeed);
  return fDgsSamplerInit(
    /* .iface = */ &vDgsSamplerDistI,
    /* .ctx   = */ new dgs_sampler_dist{
      ("dist"),
      /* .seed                      = */ pSeed,
      /* .seed_cur                  = */ vSeedCur,
      /* .rng                       = */ std::mt19937(vSeedCur),
      /* .backend_transactional     = */ false,
      /* .rng_backend               = */ std::mt19937(vSeedCur),
      /* .n_backend_draws_generated = */ 0,
      /* .n_backend_draws_committed = */ 0,
      /* .inp_uniforms              = */ {},
    });
}

void fDgsSamplerBackendBegin(dgs_sampler * pSampler) {
  GGML_ASSERT(pSampler != nullptr);

  if (pSampler->vIface == &vDgsSamplerChainI) {
    auto * vChain = (dgs_sampler_chain *) pSampler->vCtx;
    for (auto & vEntry : vChain->lSamplers) {
      if (!vEntry.vIsBackend) {
        break;
      }
      fDgsSamplerBackendBegin(vEntry.vPtr);
    }
  } else if (pSampler->vIface == &vDgsSamplerDistI) {
    auto * vCtx = (dgs_sampler_dist *) pSampler->vCtx;
    if (vCtx->vBackendTransactional) {
      vCtx->vRngBackend             = vCtx->vRng;
      vCtx->vNBackendDrawsGenerated = 0;
      vCtx->vNBackendDrawsCommitted = 0;
    }
  }
}

// top-k

struct dgs_sampler_top_k : public dgs_sampler_backend {
  const int32_t cK;
};

static const char * fDgsSamplerTopKName(const struct dgs_sampler * pSmpl) {
  auto * vSctx = (dgs_sampler_top_k *) pSmpl->vCtx;
  return vSctx->mGetName();
}

static void fDgsSamplerTopKApply(struct dgs_sampler * pSmpl, dgs_token_data_array * pCurP) {
  auto * vCtx = (dgs_sampler_top_k *) pSmpl->vCtx;
  fDgsSamplerTopKImpl(pCurP, vCtx->cK);
}

static struct dgs_sampler * fDgsSamplerTopKClone(const struct dgs_sampler * pSmpl) {
  const auto * vCtx = (const dgs_sampler_top_k *) pSmpl->vCtx;
  return fDgsSamplerInitTopK(vCtx->cK);
}

static void fDgsSamplerTopKFree(struct dgs_sampler * pSmpl) {
  delete (dgs_sampler_top_k *) pSmpl->vCtx;
}

static bool fDgsSamplerTopKBackendInit(struct dgs_sampler *       pSmpl,
                                       ggml_backend_buffer_type_t pBuft,
                                       uint32_t                   pNOutputsMaxPerSeq) {
  auto * vSctx = (dgs_sampler_top_k *) pSmpl->vCtx;
  GGML_UNUSED(pNOutputsMaxPerSeq);

  const bool cRes = fDgsSamplerBackendSupport(pSmpl, pBuft);

  vSctx->init(cRes);

  return cRes;
}

static void fDgsSamplerTopKBackendApply(struct dgs_sampler *      pSmpl,
                                        struct ggml_context *     pCtx,
                                        struct ggml_cgraph *      pGf,
                                        struct dgs_sampler_data * data) {
  auto * vSctx = (dgs_sampler_top_k *) pSmpl->vCtx;

  struct ggml_tensor * vLogits = ggml_reshape_1d(pCtx, data->vLogits, ggml_nelements(data->vLogits));

  struct ggml_tensor * vTopK = ggml_top_k(pCtx, vLogits, vSctx->cK);
  ggml_set_name(vTopK, "top_k");

  if (data->vCandidates) {
    struct ggml_tensor * vCandidatesRows = ggml_reshape_2d(pCtx, data->vCandidates, 1, data->vCandidates->ne[0]);
    data->vCandidates                    = ggml_get_rows(pCtx, vCandidatesRows, vTopK);
    ggml_set_name(data->vCandidates, "top_k_candidates");
  } else {
    data->vCandidates = vTopK;
  }

  struct ggml_tensor * vLogitsRows = ggml_reshape_2d(pCtx, vLogits, 1, vLogits->ne[0]);
  data->vLogits                    = ggml_get_rows(pCtx, vLogitsRows, vTopK);
  ggml_set_name(data->vLogits, "top_k_rows");

  GGML_UNUSED(pGf);
}

static struct dgs_sampler_i vDgsSamplerTopKI = {
  /* .name              = */ fDgsSamplerTopKName,
  /* .accept            = */ nullptr,
  /* .apply             = */ fDgsSamplerTopKApply,
  /* .reset             = */ nullptr,
  /* .clone             = */ fDgsSamplerTopKClone,
  /* .free              = */ fDgsSamplerTopKFree,
  /* .backend_init      = */ fDgsSamplerTopKBackendInit,
  /* .backend_accept    = */ nullptr,
  /* .backend_apply     = */ fDgsSamplerTopKBackendApply,
  /* .backend_set_input = */ nullptr,
  /* .backend_reset     = */ nullptr,
  /* .copy_state        = */ fDgsSamplerBackendCopyState<dgs_sampler_top_k>,
};

struct dgs_sampler * fDgsSamplerInitTopK(int32_t pK) {
  const bool cIsEmpty = (pK <= 0);

  if (cIsEmpty) {
    return fDgsSamplerInitEmpty("?top-k");
  }

  return fDgsSamplerInit(
    /* .iface = */ &vDgsSamplerTopKI,
    /* .ctx   = */ new dgs_sampler_top_k{
      ("top-k"),
      /* .k = */ pK,
    });
}

// top-p

struct dgs_sampler_top_p : public dgs_sampler_backend {
  const float  cP;
  const size_t cMinKeep;

  std::vector<dgs_token_data> lBufSort;
};

static const char * fDgsSamplerTopPName(const struct dgs_sampler * pSmpl) {
  auto * vSctx = (dgs_sampler_top_p *) pSmpl->vCtx;
  return vSctx->mGetName();
}

static void fDgsSamplerTopPApply(struct dgs_sampler * pSmpl, dgs_token_data_array * pCurP) {
  auto * vCtx = (dgs_sampler_top_p *) pSmpl->vCtx;

  if (vCtx->cP >= 1.0f) {
    return;
  }

  fDgsSamplerSoftmaxImpl(pCurP, false);

  size_t vK     = pCurP->size;
  auto * vPdata = pCurP->data;

  auto & lBufSort = vCtx->lBufSort;

  // if not sorted, try adaptive top-k sorting
  if (!pCurP->vSorted && pCurP->size > 1024) {
    vK = std::min<size_t>(256, pCurP->size);
    fDgsTokenDataArrayPartialSort(*pCurP, vK, lBufSort);
    vPdata = lBufSort.data();
  } else if (!pCurP->vSorted) {
    // small candidates -> sort inplace
    fDgsTokenDataArrayPartialSortInplace(pCurP, vK);
  }

  // Compute the cumulative probabilities
  float  vCumSum  = 0.0f;
  size_t vLastIdx = pCurP->size;

  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    vCumSum += vPdata[vI].vP;

    // Check if the running sum is at least p or if we have kept at least min_keep tokens
    // we set the last index to i+1 to indicate that the current iterate should be included in the set
    if (vCumSum >= vCtx->cP && vI + 1 >= vCtx->cMinKeep) {
      vLastIdx = vI + 1;
      break;
    }

    // we exceeded the current top-k heuristic -> increase k and continue
    if (!pCurP->vSorted && vI == vK - 1) {
      vK = pCurP->size;
      fDgsTokenDataArrayPartialSort(*pCurP, vK, lBufSort);
      vPdata = lBufSort.data();
    }
  }

  // Resize the output vector to keep only the top-p tokens
  if (!pCurP->vSorted) {
    std::copy(lBufSort.data(), lBufSort.data() + vLastIdx, pCurP->data);
    pCurP->vSorted = true;
  }

  pCurP->size = vLastIdx;
}

static struct dgs_sampler * fDgsSamplerTopPClone(const struct dgs_sampler * pSmpl) {
  const auto * vCtx = (const dgs_sampler_top_p *) pSmpl->vCtx;
  return fDgsSamplerInitTopP(vCtx->cP, vCtx->cMinKeep);
}

static void fDgsSamplerTopPFree(struct dgs_sampler * pSmpl) {
  delete (dgs_sampler_top_p *) pSmpl->vCtx;
}

static bool fDgsSamplerTopPBackendInit(struct dgs_sampler *       pSmpl,
                                       ggml_backend_buffer_type_t pBuft,
                                       uint32_t                   pNOutputsMaxPerSeq) {
  auto * vSctx = (dgs_sampler_top_p *) pSmpl->vCtx;
  GGML_UNUSED(pNOutputsMaxPerSeq);

  const bool cRes = fDgsSamplerBackendSupport(pSmpl, pBuft);

  vSctx->init(cRes);

  return cRes;
}

static void fDgsSamplerTopPBackendApply(struct dgs_sampler *      pSmpl,
                                        struct ggml_context *     pCtx,
                                        struct ggml_cgraph *      pGf,
                                        struct dgs_sampler_data * data) {
  auto * vSctx = (dgs_sampler_top_p *) pSmpl->vCtx;

  // flatten
  struct ggml_tensor * vLogits = ggml_reshape_1d(pCtx, data->vLogits, ggml_nelements(data->vLogits));

  auto vGgmlSort = [pCtx](struct ggml_tensor * pA, struct ggml_tensor * pB) {
    GGML_ASSERT(ggml_nrows(pA) == 1);
    struct ggml_tensor * vAReshaped = ggml_reshape_2d(pCtx, pA, 1, pA->ne[0]);
    struct ggml_tensor * vASorted   = ggml_get_rows(pCtx, vAReshaped, pB);
    return vASorted;
  };

  // Get the sorted logits in descending order.
  struct ggml_tensor * vSortedIdx = ggml_argsort(pCtx, vLogits, GGML_SORT_ORDER_DESC);
  ggml_set_name(vSortedIdx, "top_p_sorted_idx");

  // Do the sorting via reshape + get_rows
  struct ggml_tensor * vSortedLogits = vGgmlSort(vLogits, vSortedIdx);
  ggml_set_name(vSortedLogits, "top_p_sorted_logits");

  vSortedLogits                 = ggml_reshape_1d(pCtx, vSortedLogits, ggml_nelements(vSortedLogits));
  struct ggml_tensor * vSoftmax = ggml_soft_max(pCtx, vSortedLogits);
  ggml_set_name(vSoftmax, "top_p_softmax");

  // If candidates are provided, sort them as well. Otherwise, set sorted indices as candidates.
  if (data->vCandidates) {
    data->vCandidates = vGgmlSort(data->vCandidates, vSortedIdx);
  } else {
    data->vCandidates = vSortedIdx;
  }
  ggml_set_name(data->vCandidates, "top_p_candidates");

  // Compute Cumulative Distribution Function (CDF) by means of GGML_OP_CUMSUM.
  struct ggml_tensor * vCdf = ggml_cumsum(pCtx, vSoftmax);
  ggml_set_name(vCdf, "top_p_cdf");

  // Invert CDF and add top-p value so that ggml_step yields 1 for values we want to keep
  struct ggml_tensor * vCdfScaled = ggml_scale_bias(pCtx, vCdf, -1.0f, vSctx->cP);
  ggml_set_name(vCdfScaled, "top_p_cdf_scaled");

  struct ggml_tensor * vMask = ggml_step(pCtx, vCdfScaled);
  ggml_set_name(vMask, "top_p_mask");

  // Taking the sum of the mask gives us the sum of elements after the threshold
  // we are interested in.
  struct ggml_tensor * vIdxf = ggml_sum(pCtx, vMask);
  ggml_set_name(vIdxf, "top_p_index_f32");

  // prevent out-of-bounds access
  vIdxf = ggml_clamp(pCtx, vIdxf, 0.0f, vMask->ne[0] - 1);

  // construct ones tensor to set the value in the mask
  struct ggml_tensor * vOnes = ggml_scale_bias(pCtx, vIdxf, 0.0f, 1.0f);
  ggml_set_name(vOnes, "top_p_ones");

  // Make top-p inclusive (i.e. return all values such that cum_sum/cdf >= p)
  struct ggml_tensor * vMaskReshaped = ggml_reshape_2d(pCtx, vMask, 1, vMask->ne[0]);

  vMaskReshaped = ggml_set_rows(pCtx, vMaskReshaped, vOnes, ggml_cast(pCtx, vIdxf, GGML_TYPE_I32));
  vMask         = ggml_reshape_1d(pCtx, vMaskReshaped, vMask->ne[0]);

  // Apply -INFINITY bias for masked-out tokens
  // log(1) = 0 (keep), log(0) = -INF (discard)
  struct ggml_tensor * vTopPBias = ggml_log(pCtx, vMask);
  ggml_set_name(vTopPBias, "top_p_bias");

  data->vLogits = ggml_add(pCtx, vSortedLogits, vTopPBias);
  ggml_set_name(data->vLogits, "top_p_logits");

  GGML_UNUSED(pGf);
}

static struct dgs_sampler_i vDgsSamplerTopPI = {
  /* .name              = */ fDgsSamplerTopPName,
  /* .accept            = */ nullptr,
  /* .apply             = */ fDgsSamplerTopPApply,
  /* .reset             = */ nullptr,
  /* .clone             = */ fDgsSamplerTopPClone,
  /* .free              = */ fDgsSamplerTopPFree,
  /* .backend_init      = */ fDgsSamplerTopPBackendInit,
  /* .backend_accept    = */ nullptr,
  /* .backend_apply     = */ fDgsSamplerTopPBackendApply,
  /* .backend_set_input = */ nullptr,
  /* .backend_reset     = */ nullptr,
  /* .copy_state        = */ fDgsSamplerBackendCopyState<dgs_sampler_top_p>,
};

struct dgs_sampler * fDgsSamplerInitTopP(float pP, size_t pMinKeep) {
  const bool cIsEmpty = pP >= 1.0f;

  if (cIsEmpty) {
    return fDgsSamplerInitEmpty("?top-p");
  }

  return fDgsSamplerInit(
    /* .iface = */ &vDgsSamplerTopPI,
    /* .ctx   = */ new dgs_sampler_top_p{
      ("top-p"),
      /* .p        = */ pP,
      /* .min_keep = */ pMinKeep,
      /* .buf_sort = */ {},
    });
}

// min-p

struct dgs_sampler_min_p : public dgs_sampler_backend {
  const float  cP;
  const size_t cMinKeep;
};

static const char * fDgsSamplerMinPName(const struct dgs_sampler * pSmpl) {
  auto * vSctx = (dgs_sampler_min_p *) pSmpl->vCtx;
  return vSctx->mGetName();
}

static void fDgsSamplerMinPApply(struct dgs_sampler * pSmpl, dgs_token_data_array * pCurP) {
  auto * vCtx = (dgs_sampler_min_p *) pSmpl->vCtx;

  if (vCtx->cP <= 0.0f || !pCurP->size) {
    return;
  }

  bool vMinPApplied = false;

  // if the cur_p aren't sorted, try the unsorted implementation first
  if (!pCurP->vSorted) {
    std::vector<dgs_token_data> lFilteredTokens;

    float vMaxLogit = -FLT_MAX;
    for (size_t vI = 0; vI < pCurP->size; ++vI) {
      vMaxLogit = std::max(vMaxLogit, pCurP->data[vI].vLogit);
    }
    const float cMinLogit = vMaxLogit + logf(vCtx->cP);  // min logit for p_i >= p * p_max

    for (size_t vI = 0; vI < pCurP->size; ++vI) {
      if (pCurP->data[vI].vLogit >= cMinLogit) {
        lFilteredTokens.push_back(pCurP->data[vI]);
      }
    }

    // if we have enough values the operation was a success
    if (!lFilteredTokens.empty() && lFilteredTokens.size() >= vCtx->cMinKeep) {
      std::copy(lFilteredTokens.begin(), lFilteredTokens.end(), pCurP->data);
      pCurP->size  = lFilteredTokens.size();
      vMinPApplied = true;
    }
  }

  // if the cur_p are sorted or the unsorted implementation failed, use this implementation
  if (!vMinPApplied) {
    // Sort the logits in descending order
    if (!pCurP->vSorted) {
      fDgsTokenDataArrayPartialSortInplace(pCurP, pCurP->size);
    }

    const float cMinLogit = pCurP->data[0].vLogit + logf(vCtx->cP);  // min logit for p_i >= p * p_max
    size_t      vI        = 1;                                       // first token always matches

    for (; vI < pCurP->size; ++vI) {
      if (pCurP->data[vI].vLogit < cMinLogit && vI >= vCtx->cMinKeep) {
        break;  // prob too small
      }
    }

    // Resize the output vector to keep only the matching tokens
    pCurP->size = vI;
  }
}

static struct dgs_sampler * fDgsSamplerMinPClone(const struct dgs_sampler * pSmpl) {
  const auto * vCtx = (const dgs_sampler_min_p *) pSmpl->vCtx;
  return fDgsSamplerInitMinP(vCtx->cP, vCtx->cMinKeep);
}

static void fDgsSamplerMinPFree(struct dgs_sampler * pSmpl) {
  delete (dgs_sampler_min_p *) pSmpl->vCtx;
}

static bool fDgsSamplerMinPBackendInit(struct dgs_sampler *       pSmpl,
                                       ggml_backend_buffer_type_t pBuft,
                                       uint32_t                   pNOutputsMaxPerSeq) {
  auto * vSctx = (dgs_sampler_min_p *) pSmpl->vCtx;
  GGML_UNUSED(pNOutputsMaxPerSeq);

  const bool cRes = fDgsSamplerBackendSupport(pSmpl, pBuft);

  vSctx->init(cRes);

  return cRes;
}

static void fDgsSamplerMinPBackendApply(struct dgs_sampler *      pSmpl,
                                        struct ggml_context *     pCtx,
                                        struct ggml_cgraph *      pGf,
                                        struct dgs_sampler_data * data) {
  auto * vSctx = (dgs_sampler_min_p *) pSmpl->vCtx;

  struct ggml_tensor * vLogits = ggml_reshape_1d(pCtx, data->vLogits, ggml_nelements(data->vLogits));

  struct ggml_tensor * vMaxIdx = ggml_argmax(pCtx, vLogits);
  ggml_set_name(vMaxIdx, "max_idx");

  struct ggml_tensor * vLogitsRows = ggml_reshape_2d(pCtx, vLogits, 1, vLogits->ne[0]);
  ggml_set_name(vLogitsRows, "logits_rows");

  struct ggml_tensor * vMaxLogit = ggml_get_rows(pCtx, vLogitsRows, vMaxIdx);
  ggml_set_name(vMaxLogit, "max_logit");

  // Calculate the threshold value.
  struct ggml_tensor * vThreshold = ggml_scale_bias(pCtx, vMaxLogit, 1.0f, logf(vSctx->cP));
  ggml_set_name(vThreshold, "min_p_threshold");

  // Subtract the threshold from logits.
  struct ggml_tensor * vSub = ggml_sub(pCtx, vLogits, vThreshold);

  // Create a mask where logits below the threshold are 0 (discard),
  // and others are 1 (keep).
  struct ggml_tensor * vMask = ggml_step(pCtx, vSub);
  ggml_set_name(vMask, "min_p_mask");

  // Apply -INFINITY bias for masked-out tokens
  // log(1) = 0 (keep), log(0) = -INF (discard)
  struct ggml_tensor * vMinPBias = ggml_log(pCtx, vMask);
  ggml_set_name(vMinPBias, "min_p_bias");

  data->vLogits = ggml_add(pCtx, vLogits, vMinPBias);
  ggml_set_name(data->vLogits, "min_p_logits");

  GGML_UNUSED(pGf);
}

static struct dgs_sampler_i vDgsSamplerMinPI = {
  /* .name              = */ fDgsSamplerMinPName,
  /* .accept            = */ nullptr,
  /* .apply             = */ fDgsSamplerMinPApply,
  /* .reset             = */ nullptr,
  /* .clone             = */ fDgsSamplerMinPClone,
  /* .free              = */ fDgsSamplerMinPFree,
  /* .backend_init      = */ fDgsSamplerMinPBackendInit,
  /* .backend_accept    = */ nullptr,
  /* .backend_apply     = */ fDgsSamplerMinPBackendApply,
  /* .backend_set_input = */ nullptr,
  /* .backend_reset     = */ nullptr,
  /* .copy_state        = */ fDgsSamplerBackendCopyState<dgs_sampler_min_p>,
};

struct dgs_sampler * fDgsSamplerInitMinP(float pP, size_t pMinKeep) {
  const bool cIsEmpty = (pP <= 0.0f);

  if (cIsEmpty) {
    return fDgsSamplerInitEmpty("?min-p");
  }

  return fDgsSamplerInit(
    /* .iface = */ &vDgsSamplerMinPI,
    /* .ctx   = */ new dgs_sampler_min_p{
      ("min-p"),
      /* .p        = */ pP,
      /* .min_keep = */ pMinKeep,
    });
}

// typical

struct dgs_sampler_typical {
  const float  cP;
  const size_t cMinKeep;
};

static const char * fDgsSamplerTypicalName(const struct dgs_sampler * /*smpl*/) {
  return "typical";
}

static void fDgsSamplerTypicalApply(struct dgs_sampler * pSmpl, dgs_token_data_array * pCurP) {
  auto * vCtx = (dgs_sampler_typical *) pSmpl->vCtx;

  // Reference implementation:
  // https://github.com/huggingface/transformers/compare/main...cimeister:typical-sampling:typical-pr
  if (vCtx->cP >= 1.0f) {
    return;
  }

  // Compute the softmax of logits and calculate entropy
  fDgsSamplerSoftmaxImpl(pCurP, true);

  float vEntropy = 0.0f;
  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    vEntropy += -pCurP->data[vI].vP * logf(pCurP->data[vI].vP);
  }

  // Compute the absolute difference between negative log probability and entropy for each candidate
  std::vector<float> lShiftedScores;
  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    float vShiftedScore = fabsf(-logf(pCurP->data[vI].vP) - vEntropy);
    lShiftedScores.push_back(vShiftedScore);
  }

  // Sort tokens based on the shifted_scores and their corresponding indices
  std::vector<size_t> lIndices(pCurP->size);
  std::iota(lIndices.begin(), lIndices.end(), 0);

  std::sort(lIndices.begin(), lIndices.end(),
            [&](size_t pA, size_t pB) { return lShiftedScores[pA] < lShiftedScores[pB]; });

  // Compute the cumulative probabilities
  float  vCumSum  = 0.0f;
  size_t vLastIdx = lIndices.size();

  for (size_t vI = 0; vI < lIndices.size(); ++vI) {
    size_t vIdx = lIndices[vI];
    vCumSum += pCurP->data[vIdx].vP;

    // Check if the running sum is greater than typical or if we have kept at least min_keep tokens
    if (vCumSum > vCtx->cP && (vCtx->cMinKeep == 0 || vI >= vCtx->cMinKeep - 1)) {
      vLastIdx = vI + 1;
      break;
    }
  }

  // Resize the output vector to keep only the locally typical tokens
  std::vector<dgs_token_data> lCurPNew;
  for (size_t vI = 0; vI < vLastIdx; ++vI) {
    size_t vIdx = lIndices[vI];
    lCurPNew.push_back(pCurP->data[vIdx]);
  }

  // Replace the data in cur_p with the cur_p_new data
  std::copy(lCurPNew.begin(), lCurPNew.end(), pCurP->data);
  pCurP->size    = lCurPNew.size();
  pCurP->vSorted = false;
}

static struct dgs_sampler * fDgsSamplerTypicalClone(const struct dgs_sampler * pSmpl) {
  const auto * vCtx = (const dgs_sampler_typical *) pSmpl->vCtx;
  return fDgsSamplerInitTypical(vCtx->cP, vCtx->cMinKeep);
}

static void fDgsSamplerTypicalFree(struct dgs_sampler * pSmpl) {
  delete (dgs_sampler_typical *) pSmpl->vCtx;
}

static struct dgs_sampler_i vDgsSamplerTypicalI = {
  /* .name              = */ fDgsSamplerTypicalName,
  /* .accept            = */ nullptr,
  /* .apply             = */ fDgsSamplerTypicalApply,
  /* .reset             = */ nullptr,
  /* .clone             = */ fDgsSamplerTypicalClone,
  /* .free              = */ fDgsSamplerTypicalFree,
  /* .backend_init      = */ nullptr,
  /* .backend_accept    = */ nullptr,
  /* .backend_apply     = */ nullptr,
  /* .backend_set_input = */ nullptr,
  /* .backend_reset     = */ nullptr,
  /* .copy_state        = */ nullptr,
};

struct dgs_sampler * fDgsSamplerInitTypical(float pP, size_t pMinKeep) {
  const bool cIsEmpty = (pP >= 1.0f);

  if (cIsEmpty) {
    return fDgsSamplerInitEmpty("?typical");
  }

  return fDgsSamplerInit(
    /* .iface = */ &vDgsSamplerTypicalI,
    /* .ctx   = */ new dgs_sampler_typical{
      /* .p        = */ pP,
      /* .min_keep = */ pMinKeep,
    });
}

// temp

struct dgs_sampler_temp : public dgs_sampler_backend {
  const float cTemp;
};

static const char * fDgsSamplerTempName(const struct dgs_sampler * pSmpl) {
  auto * vSctx = (dgs_sampler_temp *) pSmpl->vCtx;
  return vSctx->mGetName();
}

static void fDgsSamplerTempApply(struct dgs_sampler * pSmpl, dgs_token_data_array * pCurP) {
  const auto * vCtx = (dgs_sampler_temp *) pSmpl->vCtx;

  fDgsSamplerTempImpl(pCurP, vCtx->cTemp);
}

static struct dgs_sampler * fDgsSamplerTempClone(const struct dgs_sampler * pSmpl) {
  const auto * vCtx = (const dgs_sampler_temp *) pSmpl->vCtx;
  return fDgsSamplerInitTemp(vCtx->cTemp);
}

static void fDgsSamplerTempFree(struct dgs_sampler * pSmpl) {
  delete (dgs_sampler_temp *) pSmpl->vCtx;
}

static void fDgsSamplerBackendTempSampling(struct ggml_context *     pCtx,
                                           struct ggml_cgraph *      pGf,
                                           struct dgs_sampler_data * data,
                                           float                     pTemp) {
  if (pTemp <= 0.0f) {
    struct ggml_tensor * vLogits = ggml_reshape_1d(pCtx, data->vLogits, ggml_nelements(data->vLogits));

    // Find the most probable token index.
    struct ggml_tensor * vMaxIdx = ggml_argmax(pCtx, vLogits);
    ggml_set_name(vMaxIdx, "temp_max_idx");

    if (data->vCandidates) {
      struct ggml_tensor * vCandidatesRows =
        ggml_reshape_2d(pCtx, data->vCandidates, 1, ggml_nelements(data->vCandidates));
      data->vCandidates = ggml_get_rows(pCtx, vCandidatesRows, vMaxIdx);
    } else {
      data->vCandidates = vMaxIdx;
    }

    struct ggml_tensor * vLogitsRows = ggml_reshape_2d(pCtx, vLogits, 1, ggml_nelements(vLogits));
    data->vLogits                    = ggml_get_rows(pCtx, vLogitsRows, vMaxIdx);

    return;
  }

  data->vLogits = ggml_scale(pCtx, data->vLogits, 1.0f / pTemp);

  GGML_UNUSED(pGf);
}

static bool fDgsSamplerTempBackendInit(struct dgs_sampler *       pSmpl,
                                       ggml_backend_buffer_type_t pBuft,
                                       uint32_t                   pNOutputsMaxPerSeq) {
  auto * vSctx = (dgs_sampler_temp *) pSmpl->vCtx;
  GGML_UNUSED(pNOutputsMaxPerSeq);

  const bool cRes = fDgsSamplerBackendSupport(pSmpl, pBuft);

  vSctx->init(cRes);

  return cRes;
}

static void fDgsSamplerTempBackendApply(struct dgs_sampler *      pSmpl,
                                        struct ggml_context *     pCtx,
                                        struct ggml_cgraph *      pGf,
                                        struct dgs_sampler_data * data) {
  auto * vSctx = (dgs_sampler_temp *) pSmpl->vCtx;
  fDgsSamplerBackendTempSampling(pCtx, pGf, data, vSctx->cTemp);
}

static struct dgs_sampler_i vDgsSamplerTempI = {
  /* .name              = */ fDgsSamplerTempName,
  /* .accept            = */ nullptr,
  /* .apply             = */ fDgsSamplerTempApply,
  /* .reset             = */ nullptr,
  /* .clone             = */ fDgsSamplerTempClone,
  /* .free              = */ fDgsSamplerTempFree,
  /* .backend_init      = */ fDgsSamplerTempBackendInit,
  /* .backend_accept    = */ nullptr,
  /* .backend_apply     = */ fDgsSamplerTempBackendApply,
  /* .backend_set_input = */ nullptr,
  /* .backend_reset     = */ nullptr,
  /* .copy_state        = */ fDgsSamplerBackendCopyState<dgs_sampler_temp>,
};

struct dgs_sampler * fDgsSamplerInitTemp(float pTemp) {
  const bool cIsEmpty = pTemp == 1.0f;

  if (cIsEmpty) {
    return fDgsSamplerInitEmpty("?temp");
  }

  return fDgsSamplerInit(
    /* .iface = */ &vDgsSamplerTempI,
    /* .ctx   = */ new dgs_sampler_temp{
      ("temp"),
      /*.temp = */ pTemp,
    });
}

// temp-ext

struct dgs_sampler_temp_ext : public dgs_sampler_backend {
  const float cTemp;
  const float cDelta;
  const float cExponent;
};

static const char * fDgsSamplerTempExtName(const struct dgs_sampler * pSmpl) {
  auto * vSctx = (dgs_sampler_temp_ext *) pSmpl->vCtx;
  return vSctx->mGetName();
}

static void fDgsSamplerTempExtApply(struct dgs_sampler * pSmpl, dgs_token_data_array * pCurP) {
  auto * vCtx = (dgs_sampler_temp_ext *) pSmpl->vCtx;
  if (vCtx->cDelta > 0) {
    const float cMinTemp = std::max(0.0f, vCtx->cTemp - vCtx->cDelta);
    const float cMaxTemp = vCtx->cTemp + vCtx->cDelta;

    float vExponentVal = vCtx->cExponent;

    // no need to do anything if there is only one (or zero) candidates
    if (pCurP->size <= 1) {
      return;
    }

    // Calculate maximum possible entropy
    float vMaxEntropy = -logf(1.0f / pCurP->size);

    fDgsSamplerSoftmaxImpl(pCurP, true);

    // Calculate entropy of the softmax probabilities
    float vEntropy = 0.0f;
    for (size_t vI = 0; vI < pCurP->size; ++vI) {
      float vProb = pCurP->data[vI].vP;
      if (vProb > 0.0f) {  // Ensure no log(0)
        vEntropy -= vProb * logf(vProb);
      }
    }

    // Normalize the entropy (max_entropy cannot be 0 here because we checked cur_p->size != 1 above)
    float vNormalizedEntropy = vEntropy / vMaxEntropy;

    // Map the normalized entropy to the desired temperature range using the power function
    float vDynTemp = cMinTemp + (cMaxTemp - cMinTemp) * powf(vNormalizedEntropy, vExponentVal);

#ifdef DEBUG
    DGS_LOG_INFO("Your text maxtemp value is: %f\n", max_temp);
    DGS_LOG_INFO("Entropy: %f\n", entropy);
    DGS_LOG_INFO("Max Possible Entropy: %f\n", max_entropy);
    DGS_LOG_INFO("Normalized Entropy: %f\n", normalized_entropy);
    DGS_LOG_INFO("Exponent: %f\n", exponent_val);
    DGS_LOG_INFO("Dynamic Temperature (dyn_temp): %f\n", dyn_temp);
#endif

    // Apply the dynamically calculated temperature scaling
    fDgsSamplerTempImpl(pCurP, vDynTemp);

    // Re-compute softmax probabilities after scaling logits with dynamic temperature
    const double cMaxLDouble = pCurP->data[0].vLogit;

    double vCumSumDouble = 0.0;
    for (size_t vI = 0; vI < pCurP->size; ++vI) {
      double vP          = exp(pCurP->data[vI].vLogit - cMaxLDouble);
      pCurP->data[vI].vP = vP;  // Store the scaled probability
      vCumSumDouble += vP;
    }

    for (size_t vI = 0; vI < pCurP->size; ++vI) {
      pCurP->data[vI].vP /= vCumSumDouble;  // Re-normalize the probabilities
    }

#ifdef DEBUG
    // Print the updated top 25 probabilities after temperature scaling
    DGS_LOG_INFO("\nUpdated Top 25 Probabilities After Dynamic Temperature Scaling (in percentages):\n");
    for (size_t i = 0; i < 25 && i < cur_p->size; ++i) {
      DGS_LOG_INFO("Token %zu: %f%%\n", i + 1, cur_p->data[i].p * 100.0f);
    }
#endif
  } else {
    fDgsSamplerTempImpl(pCurP, vCtx->cTemp);
  }
}

static struct dgs_sampler * fDgsSamplerTempExtClone(const struct dgs_sampler * pSmpl) {
  const auto * vCtx = (const dgs_sampler_temp_ext *) pSmpl->vCtx;
  return fDgsSamplerInitTempExt(vCtx->cTemp, vCtx->cDelta, vCtx->cExponent);
}

static void fDgsSamplerTempExtFree(struct dgs_sampler * pSmpl) {
  delete (dgs_sampler_temp_ext *) pSmpl->vCtx;
}

static bool fDgsSamplerTempExtBackendInit(struct dgs_sampler *       pSmpl,
                                          ggml_backend_buffer_type_t pBuft,
                                          uint32_t                   pNOutputsMaxPerSeq) {
  auto * vSctx = (dgs_sampler_temp_ext *) pSmpl->vCtx;
  GGML_UNUSED(pNOutputsMaxPerSeq);

  const bool cRes = fDgsSamplerBackendSupport(pSmpl, pBuft);

  vSctx->init(cRes);

  return cRes;
}

static void fDgsSamplerTempExtBackendApply(struct dgs_sampler *      pSmpl,
                                           struct ggml_context *     pCtx,
                                           struct ggml_cgraph *      pGf,
                                           struct dgs_sampler_data * data) {
  auto * vSctx = (dgs_sampler_temp_ext *) pSmpl->vCtx;

  // Revert to standard temperature scaling if delta or temp are non-positive.
  if (vSctx->cDelta <= 0.0f || vSctx->cTemp <= 0.0f) {
    fDgsSamplerBackendTempSampling(pCtx, pGf, data, vSctx->cTemp);
    return;
  }

  struct ggml_tensor * vLogits = ggml_reshape_1d(pCtx, data->vLogits, ggml_nelements(data->vLogits));

  // Calculate min_temp, max_temp, and max_entropy.
  const float cMinTemp    = std::max(0.0f, vSctx->cTemp - vSctx->cDelta);
  const float cMaxTemp    = vSctx->cTemp + vSctx->cDelta;
  const float cMaxEntropy = logf(vLogits->ne[0]);

  // Calculate the probabilities.
  struct ggml_tensor * vProbs = ggml_soft_max(pCtx, vLogits);
  ggml_set_name(vProbs, "temp_ext_softmax_probs");

  // Clamp probabilities to avoid log(0) which would give -inf
  struct ggml_tensor * vProbsClamped = ggml_clamp(pCtx, vProbs, 1e-10f, 1.0f);
  ggml_set_name(vProbsClamped, "temp_ext_probs_clamped");

  // Calculate the entropy, entropy = -Σ(p * log(p)).
  struct ggml_tensor * vLogProbs = ggml_log(pCtx, vProbsClamped);
  struct ggml_tensor * vPLogP    = ggml_mul(pCtx, vProbsClamped, vLogProbs);
  struct ggml_tensor * vSumPLogP = ggml_sum(pCtx, vPLogP);
  struct ggml_tensor * vEntropy  = ggml_scale(pCtx, vSumPLogP, -1.0f);
  ggml_set_name(vLogProbs, "temp_ext_log_probs");
  ggml_set_name(vPLogP, "temp_ext_p_log_p");
  ggml_set_name(vSumPLogP, "temp_ext_sum_p_log_p");
  ggml_set_name(vEntropy, "temp_ext_entropy");

  // Normalize the entropy, norm_entropy = entropy / max_entropy
  struct ggml_tensor * vNormEntropy = ggml_scale(pCtx, vEntropy, 1.0f / cMaxEntropy);
  ggml_set_name(vNormEntropy, "temp_ext_norm_entropy");

  // Calculate the dynamic temperature:
  // dyn_temp = min_temp + (max_temp - min_temp) * powf(normalized_entropy, exponent);
  //
  // Calculate powf(normalized_entropy, exponent) as
  // norm_entropy^exponent = exp(exponent * log(norm_entropy))
  struct ggml_tensor * vLogNormEntropy = ggml_log(pCtx, vNormEntropy);
  struct ggml_tensor * vScaledLog      = ggml_scale(pCtx, vLogNormEntropy, vSctx->cExponent);
  struct ggml_tensor * vPowEntropy     = ggml_exp(pCtx, vScaledLog);
  // With pow_entropy computed we can now compute dyn_temp, scaling by
  // (max_temp - min_temp) and then adding min_temp.
  struct ggml_tensor * vDynTemp        = ggml_scale_bias(pCtx, vPowEntropy, cMaxTemp - cMinTemp, cMinTemp);
  ggml_set_name(vLogNormEntropy, "temp_ext_log_norm_entropy");
  ggml_set_name(vScaledLog, "temp_ext_scaled_log");
  ggml_set_name(vPowEntropy, "temp_ext_pow_entropy");
  ggml_set_name(vDynTemp, "temp_ext_dyn_temp");

  // Scale the logits by the dynamic temperature
  struct ggml_tensor * vScaledLogits = ggml_div(pCtx, vLogits, vDynTemp);
  ggml_set_name(vScaledLogits, "temp_ext_scaled_logits");

  data->vLogits = vScaledLogits;
}

static struct dgs_sampler_i vDgsSamplerTempExtI = {
  /* .name              = */ fDgsSamplerTempExtName,
  /* .accept            = */ nullptr,
  /* .apply             = */ fDgsSamplerTempExtApply,
  /* .reset             = */ nullptr,
  /* .clone             = */ fDgsSamplerTempExtClone,
  /* .free              = */ fDgsSamplerTempExtFree,
  /* .backend_init      = */ fDgsSamplerTempExtBackendInit,
  /* .backend_accept    = */ nullptr,
  /* .backend_apply     = */ fDgsSamplerTempExtBackendApply,
  /* .backend_set_input = */ nullptr,
  /* .backend_reset     = */ nullptr,
  /* .copy_state        = */ fDgsSamplerBackendCopyState<dgs_sampler_temp_ext>,
};

struct dgs_sampler * fDgsSamplerInitTempExt(float pTemp, float pDelta, float pExponent) {
  const bool cIsEmpty = pTemp == 1.0f && pDelta <= 0.0f;

  if (cIsEmpty) {
    return fDgsSamplerInitEmpty("?temp-ext");
  }

  auto * vRes = fDgsSamplerInit(
    /* .iface = */ &vDgsSamplerTempExtI,
    /* .ctx   = */ new dgs_sampler_temp_ext{
      ("temp-ext"),
      /* .temp     = */ pTemp,
      /* .delta    = */ pDelta,
      /* .exponent = */ pExponent,
    });

  return vRes;
}

// xtc

struct dgs_sampler_xtc {
  const float  cProbability;
  const float  cThreshold;
  const size_t cMinKeep;

  const uint32_t cSeed;
  uint32_t       vSeedCur;

  std::mt19937 vRng;
};

static const char * fDgsSamplerXtcName(const struct dgs_sampler * /*smpl*/) {
  return "xtc";
}

static void fDgsSampleXtcApply(struct dgs_sampler * pSmpl, dgs_token_data_array * pCurP) {
  auto * vCtx = (dgs_sampler_xtc *) pSmpl->vCtx;

  if (vCtx->cProbability <= 0.0f || vCtx->cThreshold > 0.5f || pCurP->size < 2) {
    return;
  }

  std::uniform_real_distribution<float> vDistribution(0.0f, 1.0f);
  float                                 vChance = vDistribution(vCtx->vRng);
  if (vChance > vCtx->cProbability) {
    return;
  }

  fDgsSamplerSoftmaxImpl(pCurP, true);

  int vPosLast = 0;

  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    if (pCurP->data[vI].vP >= vCtx->cThreshold) {
      vPosLast = vI;
    } else {
      break;
    }
  }

  if (pCurP->size - vPosLast >= vCtx->cMinKeep && vPosLast > 0) {
    pCurP->data += vPosLast;
    pCurP->size -= vPosLast;
  }
}

static struct dgs_sampler * fDgsSamplerXtcClone(const struct dgs_sampler * pSmpl) {
  const auto * vCtx    = (const dgs_sampler_xtc *) pSmpl->vCtx;
  auto *       vResult = fDgsSamplerInitXtc(vCtx->cProbability, vCtx->cThreshold, vCtx->cMinKeep, vCtx->cSeed);

  // copy the state
  {
    auto * vResultCtx = (dgs_sampler_xtc *) vResult->vCtx;

    vResultCtx->vRng = vCtx->vRng;
  }

  return vResult;
}

static void fDgsSamplerXtcFree(struct dgs_sampler * pSmpl) {
  delete (dgs_sampler_xtc *) pSmpl->vCtx;
}

static void fDgsSamplerXtcReset(struct dgs_sampler * pSmpl) {
  auto * vCtx    = (dgs_sampler_xtc *) pSmpl->vCtx;
  vCtx->vSeedCur = fGetRngSeed(vCtx->cSeed);
  vCtx->vRng.seed(vCtx->vSeedCur);
}

static struct dgs_sampler_i vDgsSamplerXtcI = {
  /* .name              = */ fDgsSamplerXtcName,
  /* .accept            = */ nullptr,
  /* .apply             = */ fDgsSampleXtcApply,
  /* .reset             = */ fDgsSamplerXtcReset,
  /* .clone             = */ fDgsSamplerXtcClone,
  /* .free              = */ fDgsSamplerXtcFree,
  /* .backend_init      = */ nullptr,
  /* .backend_accept    = */ nullptr,
  /* .backend_apply     = */ nullptr,
  /* .backend_set_input = */ nullptr,
  /* .backend_reset     = */ nullptr,
  /* .copy_state        = */ nullptr,
};

struct dgs_sampler * fDgsSamplerInitXtc(float pP, float pT, size_t pMinKeep, uint32_t pSeed) {
  const bool cIsEmpty = (pP <= 0.0f || pT > 0.5f);

  if (cIsEmpty) {
    return fDgsSamplerInitEmpty("?xtc");
  }

  const auto cSeedCur = fGetRngSeed(pSeed);

  return fDgsSamplerInit(
    /* .iface = */ &vDgsSamplerXtcI,
    /* .ctx   = */ new dgs_sampler_xtc{
      /* .probability   = */ pP,
      /* .threshold     = */ pT,
      /* .min_keep      = */ pMinKeep,
      /* .seed          = */ pSeed,
      /* .seed_cur      = */ cSeedCur,
      /* .rng           = */ std::mt19937(cSeedCur),
    });
}

// mirostat

struct dgs_sampler_mirostat {
  const int32_t cNVocab;

  const uint32_t cSeed;
  uint32_t       vSeedCur;

  const float cTau;
  const float cEta;

  const int32_t cM;

  float vMu;

  std::mt19937 vRng;
};

static const char * fDgsSamplerMirostatName(const struct dgs_sampler * /*smpl*/) {
  return "mirostat";
}

static void fDgsSamplerMirostatApply(struct dgs_sampler * pSmpl, dgs_token_data_array * pCurP) {
  auto * vCtx = (dgs_sampler_mirostat *) pSmpl->vCtx;

  fDgsSamplerSoftmaxImpl(pCurP, true);

  // Estimate s_hat using the most probable m tokens
  float vSHat    = 0.0;
  float vSumTiBi = 0.0;
  float vSumTiSq = 0.0;
  for (size_t vI = 0; vI < size_t(vCtx->cM - 1) && vI < pCurP->size - 1; ++vI) {
    float vTI = logf(float(vI + 2) / float(vI + 1));
    float vBI = logf(pCurP->data[vI].vP / pCurP->data[vI + 1].vP);
    vSumTiBi += vTI * vBI;
    vSumTiSq += vTI * vTI;
  }
  vSHat = vSumTiBi / vSumTiSq;

  // Compute k from the estimated s_hat and target surprise value
  float vEpsilonHat = vSHat - 1;
  float vK          = powf((vEpsilonHat * powf(2, vCtx->vMu)) / (1 - powf(vCtx->cNVocab, -vEpsilonHat)), 1 / vSHat);

  fDgsSamplerTopKImpl(pCurP, std::max(int(vK), 1));

  fDgsSamplerSoftmaxImpl(pCurP, true);

  const int cIdx = fDgsSampleDist(pCurP, vCtx->vRng);

  pCurP->vSelected = cIdx;

  float vObservedSurprise = -log2f(pCurP->data[cIdx].vP);
  float vE                = vObservedSurprise - vCtx->cTau;

  // Update mu using the learning rate and error
  vCtx->vMu = vCtx->vMu - vCtx->cEta * vE;
}

static struct dgs_sampler * fDgsSamplerMirostatClone(const struct dgs_sampler * pSmpl) {
  const auto * vCtx    = (const dgs_sampler_mirostat *) pSmpl->vCtx;
  auto *       vResult = fDgsSamplerInitMirostat(vCtx->cNVocab, vCtx->cSeed, vCtx->cTau, vCtx->cEta, vCtx->cM);

  // copy the state
  {
    auto * vResultCtx = (dgs_sampler_mirostat *) vResult->vCtx;

    vResultCtx->vMu  = vCtx->vMu;
    vResultCtx->vRng = vCtx->vRng;
  }

  return vResult;
}

static void fDgsSamplerMirostatReset(struct dgs_sampler * pSmpl) {
  auto * vCtx    = (dgs_sampler_mirostat *) pSmpl->vCtx;
  vCtx->vMu      = 2.0f * vCtx->cTau;
  vCtx->vSeedCur = fGetRngSeed(vCtx->cSeed);
  vCtx->vRng.seed(vCtx->vSeedCur);
}

static void fDgsSamplerMirostatFree(struct dgs_sampler * pSmpl) {
  delete (dgs_sampler_mirostat *) pSmpl->vCtx;
}

static struct dgs_sampler_i vDgsSamplerMirostatI = {
  /* .name              = */ fDgsSamplerMirostatName,
  /* .accept            = */ nullptr,
  /* .apply             = */ fDgsSamplerMirostatApply,
  /* .reset             = */ fDgsSamplerMirostatReset,
  /* .clone             = */ fDgsSamplerMirostatClone,
  /* .free              = */ fDgsSamplerMirostatFree,
  /* .backend_init      = */ nullptr,
  /* .backend_accept    = */ nullptr,
  /* .backend_apply     = */ nullptr,
  /* .backend_set_input = */ nullptr,
  /* .backend_reset     = */ nullptr,
  /* .copy_state        = */ nullptr,
};

struct dgs_sampler * fDgsSamplerInitMirostat(int32_t pNVocab, uint32_t pSeed, float pTau, float pEta, int32_t pM) {
  const auto cSeedCur = fGetRngSeed(pSeed);

  return fDgsSamplerInit(
    /* .iface = */ &vDgsSamplerMirostatI,
    /* .ctx   = */ new dgs_sampler_mirostat{
      /* .n_vocab  = */ pNVocab,
      /* .seed     = */ pSeed,
      /* .seed_cur = */ cSeedCur,
      /* .tau      = */ pTau,
      /* .eta      = */ pEta,
      /* .m        = */ pM,
      /* .mu       = */ 2.0f * pTau,
      /* .rng      = */ std::mt19937(cSeedCur),
    });
}

// mirostat v2

struct dgs_sampler_mirostat_v2 {
  const uint32_t cSeed;
  uint32_t       vSeedCur;

  const float cTau;
  const float cEta;

  float vMu;

  std::mt19937 vRng;
};

static const char * fDgsSamplerMirostatV2Name(const struct dgs_sampler * /*smpl*/) {
  return "mirostat-v2";
}

static void fDgsSamplerMirostatV2Apply(struct dgs_sampler * pSmpl, dgs_token_data_array * pCurP) {
  auto * vCtx = (dgs_sampler_mirostat_v2 *) pSmpl->vCtx;

  fDgsSamplerSoftmaxImpl(pCurP, true);

  // Truncate the words with surprise values greater than mu
  pCurP->size = std::distance(
    pCurP->data, std::find_if(pCurP->data, pCurP->data + pCurP->size,
                              [&](const dgs_token_data & pCandidate) { return -log2f(pCandidate.vP) > vCtx->vMu; }));

  if (pCurP->size == 0) {
    pCurP->size = 1;
  }

  // Normalize the probabilities of the remaining words
  fDgsSamplerSoftmaxImpl(pCurP, true);

  const int cIdx = fDgsSampleDist(pCurP, vCtx->vRng);

  pCurP->vSelected = cIdx;

  float vObservedSurprise = -log2f(pCurP->data[cIdx].vP);
  float vE                = vObservedSurprise - vCtx->cTau;

  // Update mu using the learning rate and error
  vCtx->vMu = vCtx->vMu - vCtx->cEta * vE;
}

static void fDgsSamplerMirostatV2Reset(struct dgs_sampler * pSmpl) {
  auto * vCtx    = (dgs_sampler_mirostat_v2 *) pSmpl->vCtx;
  vCtx->vMu      = 2.0f * vCtx->cTau;
  vCtx->vSeedCur = fGetRngSeed(vCtx->cSeed);
  vCtx->vRng.seed(vCtx->vSeedCur);
}

static struct dgs_sampler * fDgsSamplerMirostatV2Clone(const struct dgs_sampler * pSmpl) {
  const auto * vCtx = (const dgs_sampler_mirostat_v2 *) pSmpl->vCtx;

  auto * vResult = fDgsSamplerInitMirostatV2(vCtx->cSeed, vCtx->cTau, vCtx->cEta);

  // copy the state
  {
    auto * vResultCtx = (dgs_sampler_mirostat_v2 *) vResult->vCtx;

    vResultCtx->vMu  = vCtx->vMu;
    vResultCtx->vRng = vCtx->vRng;
  }

  return vResult;
}

static void fDgsSamplerMirostatV2Free(struct dgs_sampler * pSmpl) {
  delete (dgs_sampler_mirostat_v2 *) pSmpl->vCtx;
}

static struct dgs_sampler_i vDgsSamplerMirostatV2I = {
  /* .name              = */ fDgsSamplerMirostatV2Name,
  /* .accept            = */ nullptr,
  /* .apply             = */ fDgsSamplerMirostatV2Apply,
  /* .reset             = */ fDgsSamplerMirostatV2Reset,
  /* .clone             = */ fDgsSamplerMirostatV2Clone,
  /* .free              = */ fDgsSamplerMirostatV2Free,
  /* .backend_init      = */ nullptr,
  /* .backend_accept    = */ nullptr,
  /* .backend_apply     = */ nullptr,
  /* .backend_set_input = */ nullptr,
  /* .backend_reset     = */ nullptr,
  /* .copy_state        = */ nullptr,
};

struct dgs_sampler * fDgsSamplerInitMirostatV2(uint32_t pSeed, float pTau, float pEta) {
  auto vSeedCur = fGetRngSeed(pSeed);
  return fDgsSamplerInit(
    /* .iface = */ &vDgsSamplerMirostatV2I,
    /* .ctx   = */ new dgs_sampler_mirostat_v2{
      /* .seed     = */ pSeed,
      /* .seed_cur = */ vSeedCur,
      /* .tau      = */ pTau,
      /* .eta      = */ pEta,
      /* .mu       = */ 2.0f * pTau,
      /* .rng      = */ std::mt19937(vSeedCur),
    });
}

// grammar

struct dgs_sampler_grammar {
  const struct dgs_vocab * vVocab;

  std::string vGrammarStr;
  std::string vGrammarRoot;

  struct dgs_grammar * vGrammar;
};

static const char * fDgsSamplerGrammarName(const struct dgs_sampler * /*smpl*/) {
  return "grammar";
}

static void fDgsSamplerGrammarAcceptImpl(struct dgs_sampler * pSmpl, dgs_token token) {
  auto * vCtx = (dgs_sampler_grammar *) pSmpl->vCtx;
  if (vCtx->vGrammar) {
    fDgsGrammarAcceptImpl(*vCtx->vGrammar, token);
  }
}

static void fDgsSamplerGrammarApply(struct dgs_sampler * pSmpl, dgs_token_data_array * pCurP) {
  auto * vCtx = (dgs_sampler_grammar *) pSmpl->vCtx;
  if (vCtx->vGrammar) {
    fDgsGrammarApplyImpl(*vCtx->vGrammar, pCurP);
  }
}

// Fwd declare to break reset --> init_impl --> dgs_sampler_grammar_i --> reset cycle.
static struct dgs_sampler * fDgsSamplerInitGrammarImpl(const struct dgs_vocab * pVocab,
                                                       const char *             pGrammarStr,
                                                       const char *             pGrammarRoot,
                                                       bool                     pLazy,
                                                       const char **            pTriggerWords,
                                                       size_t                   pNumTriggerWords,
                                                       const dgs_token *        pTriggerTokens,
                                                       size_t                   pNumTriggerTokens,
                                                       const char **            pTriggerPatterns,
                                                       size_t                   pNumTriggerPatterns);

static void fDgsSamplerGrammarReset(struct dgs_sampler * pSmpl) {
  auto * vCtx = (dgs_sampler_grammar *) pSmpl->vCtx;
  if (!vCtx->vGrammar) {
    return;
  }

  std::vector<const char *> lTriggerPatternsC;
  lTriggerPatternsC.reserve(vCtx->vGrammar->lTriggerPatterns.size());
  for (auto & vTriggerPattern : vCtx->vGrammar->lTriggerPatterns) {
    lTriggerPatternsC.push_back(vTriggerPattern.vPattern.c_str());
  }

  auto * vGrammarNew =
    fDgsGrammarInitImpl(vCtx->vGrammar->vVocab, vCtx->vGrammarStr.c_str(), vCtx->vGrammarRoot.c_str(),
                        vCtx->vGrammar->vLazy, lTriggerPatternsC.data(), lTriggerPatternsC.size(),
                        vCtx->vGrammar->lTriggerTokens.data(), vCtx->vGrammar->lTriggerTokens.size());

  fDgsGrammarFreeImpl(vCtx->vGrammar);
  vCtx->vGrammar = vGrammarNew;
}

static struct dgs_sampler * fDgsSamplerGrammarClone(const struct dgs_sampler * pSmpl) {
  const auto * vCtx = (const dgs_sampler_grammar *) pSmpl->vCtx;

  auto * vResult =
    fDgsSamplerInitGrammarImpl(vCtx->vVocab, nullptr, nullptr, false, nullptr, 0, nullptr, 0, nullptr, 0);
  GGML_ASSERT(vResult);

  // copy the state
  {
    auto * vResultCtx = (dgs_sampler_grammar *) vResult->vCtx;

    if (vCtx->vGrammar) {
      vResultCtx->vGrammarStr  = vCtx->vGrammarStr;
      vResultCtx->vGrammarRoot = vCtx->vGrammarRoot;

      vResultCtx->vGrammar = fDgsGrammarCloneImpl(*vCtx->vGrammar);
    }
  }

  return vResult;
}

static void fDgsSamplerGrammarFree(struct dgs_sampler * pSmpl) {
  const auto * vCtx = (dgs_sampler_grammar *) pSmpl->vCtx;

  if (vCtx->vGrammar) {
    fDgsGrammarFreeImpl(vCtx->vGrammar);
  }

  delete vCtx;
}

static struct dgs_sampler_i vDgsSamplerGrammarI = {
  /* .name              = */ fDgsSamplerGrammarName,
  /* .accept            = */ fDgsSamplerGrammarAcceptImpl,
  /* .apply             = */ fDgsSamplerGrammarApply,
  /* .reset             = */ fDgsSamplerGrammarReset,
  /* .clone             = */ fDgsSamplerGrammarClone,
  /* .free              = */ fDgsSamplerGrammarFree,
  /* .backend_init      = */ nullptr,
  /* .backend_accept    = */ nullptr,
  /* .backend_apply     = */ nullptr,
  /* .backend_set_input = */ nullptr,
  /* .backend_reset     = */ nullptr,
  /* .copy_state        = */ nullptr,
};

static struct dgs_sampler * fDgsSamplerInitGrammarImpl(const struct dgs_vocab * pVocab,
                                                       const char *             pGrammarStr,
                                                       const char *             pGrammarRoot,
                                                       bool                     pLazy,
                                                       const char **            pTriggerWords,
                                                       size_t                   pNumTriggerWords,
                                                       const dgs_token *        pTriggerTokens,
                                                       size_t                   pNumTriggerTokens,
                                                       const char **            pTriggerPatterns,
                                                       size_t                   pNumTriggerPatterns) {
  auto * vCtx = new dgs_sampler_grammar;

  if (pGrammarStr != nullptr && pGrammarStr[0] != '\0') {
    std::string   vTriggerPattern;
    dgs_grammar * vGrammar = nullptr;
    // TODO: remove trigger_words support.
    if (pTriggerWords != nullptr && pNumTriggerWords > 0) {
      GGML_ASSERT(pTriggerPatterns == nullptr && pNumTriggerPatterns == 0);
      vTriggerPattern = "[\\s\\S]*?(";
      for (size_t vI = 0; vI < pNumTriggerWords; ++vI) {
        static const std::regex cSpecialChars("[.^$|()*+?\\[\\]{}\\\\]");
        if (vI > 0) {
          vTriggerPattern += "|";
        }
        vTriggerPattern += std::regex_replace(pTriggerWords[vI], cSpecialChars, "\\$0");
      }
      vTriggerPattern += ")[\\s\\S]*";

      std::array<const char *, 1> aTmpTriggerPatterns = { vTriggerPattern.c_str() };
      vGrammar = fDgsGrammarInitImpl(pVocab, pGrammarStr, pGrammarRoot, pLazy, aTmpTriggerPatterns.data(),
                                     aTmpTriggerPatterns.size(), pTriggerTokens, pNumTriggerTokens);
    } else {
      vGrammar = fDgsGrammarInitImpl(pVocab, pGrammarStr, pGrammarRoot, pLazy, pTriggerPatterns, pNumTriggerPatterns,
                                     pTriggerTokens, pNumTriggerTokens);
    }
    *vCtx = {
      /* .vocab        = */ pVocab,
      /* .grammar_str  = */ pGrammarStr,
      /* .grammar_root = */ pGrammarRoot,
      /* .grammar      = */ vGrammar,
    };
    if (!vCtx->vGrammar) {
      delete vCtx;
      return nullptr;
    }
  } else {
    *vCtx = {
      /* .vocab        = */ pVocab,
      /* .grammar_str  = */ {},
      /* .grammar_root = */ {},
      /* .grammar      = */ nullptr,
    };
  }

  return fDgsSamplerInit(
    /* .iface = */ &vDgsSamplerGrammarI,
    /* .ctx   = */ vCtx);
}

struct dgs_sampler * fDgsSamplerInitGrammar(const struct dgs_vocab * pVocab,
                                            const char *             pGrammarStr,
                                            const char *             pGrammarRoot) {
  return fDgsSamplerInitGrammarImpl(pVocab, pGrammarStr, pGrammarRoot, /* lazy= */ false, nullptr, 0, nullptr, 0,
                                    nullptr, 0);
}

struct dgs_sampler * fDgsSamplerInitGrammarLazyPatterns(const struct dgs_vocab * pVocab,
                                                        const char *             pGrammarStr,
                                                        const char *             pGrammarRoot,
                                                        const char **            pTriggerPatterns,
                                                        size_t                   pNumTriggerPatterns,
                                                        const dgs_token *        pTriggerTokens,
                                                        size_t                   pNumTriggerTokens) {
  return fDgsSamplerInitGrammarImpl(pVocab, pGrammarStr, pGrammarRoot, /* lazy= */ true, nullptr, 0, pTriggerTokens,
                                    pNumTriggerTokens, pTriggerPatterns, pNumTriggerPatterns);
}

// penalties

struct dgs_sampler_penalties : public dgs_sampler_backend {
  const int32_t cNVocab;
  const int32_t cPenaltyLastN;
  const float   cPenaltyRepeat;
  const float   cPenaltyFreq;
  const float   cPenaltyPresent;

  ring_buffer<dgs_token> vPrev;

  // a frequency map to count token occurrences
  std::unordered_map<dgs_token, int> dTokenCount;

  // backend graph inputs
  ggml_tensor * vInpTokenIds = nullptr;
  ggml_tensor * vInpCounts   = nullptr;

  // backend helpers
  int32_t vNMax          = 0;
  bool    vHasCandidates = false;

  std::vector<int32_t> lHostTokenIds;
  std::vector<int32_t> lHostCounts;

  void mCopyState(const dgs_sampler_penalties & pSrc) {
    // note: inp_token_ids/inp_counts belong to the current sampling graph
    vPrev       = pSrc.vPrev;
    dTokenCount = pSrc.dTokenCount;
  }

  static bool mIsDisabled(int32_t pPenaltyLastN, float pPenaltyRepeat, float pPenaltyFreq, float pPenaltyPresent) {
    return pPenaltyLastN == 0 || (pPenaltyRepeat == 1.0f && pPenaltyFreq == 0.0f && pPenaltyPresent == 0.0f);
  }

  bool mIsDisabled() const { return mIsDisabled(cPenaltyLastN, cPenaltyRepeat, cPenaltyFreq, cPenaltyPresent); }

  dgs_sampler_penalties(int32_t pNVocab,
                        int32_t pPenaltyLastN,
                        float   pPenaltyRepeat,
                        float   pPenaltyFreq,
                        float   pPenaltyPresent) :
    dgs_sampler_backend("penalties"),
    cNVocab(pNVocab),
    cPenaltyLastN(pPenaltyLastN),
    cPenaltyRepeat(pPenaltyRepeat),
    cPenaltyFreq(pPenaltyFreq),
    cPenaltyPresent(pPenaltyPresent),
    vPrev(pPenaltyLastN) {}
};

static const char * fDgsSamplerPenaltiesName(const struct dgs_sampler * pSmpl) {
  auto * vCtx = (dgs_sampler_penalties *) pSmpl->vCtx;
  return vCtx->mGetName();
}

static void fDgsSamplerPenaltiesAccept(struct dgs_sampler * pSmpl, dgs_token token) {
  auto * vCtx = (dgs_sampler_penalties *) pSmpl->vCtx;
  if (vCtx->cPenaltyLastN == 0) {
    return;
  }

  vCtx->dTokenCount[token]++;

  // if the ring buffer is full, remove the oldest token
  if (vCtx->vPrev.size() >= (size_t) vCtx->cPenaltyLastN) {
    const auto cOld = vCtx->vPrev.front();

    vCtx->dTokenCount[cOld]--;
    if (vCtx->dTokenCount[cOld] == 0) {
      vCtx->dTokenCount.erase(cOld);
    }
  }

  vCtx->vPrev.push_back(token);

#if 0
    // sanity check
    std::unordered_map<dgs_token, int> tmp;
    for (int i = 0; i < std::min<int>(ctx->penalty_last_n, ctx->prev.size()); ++i) {
        tmp[ctx->prev.rat(i)]++;
    }

    assert(ctx->token_count == tmp);
#endif
}

static void fDgsSamplerPenaltiesApply(struct dgs_sampler * pSmpl, dgs_token_data_array * pCurP) {
  auto * vCtx = (dgs_sampler_penalties *) pSmpl->vCtx;

  if (vCtx->mIsDisabled()) {
    return;
  }

  // Apply frequency and presence penalties to the cur_p
  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    const auto cTokenIter = vCtx->dTokenCount.find(pCurP->data[vI].vId);
    if (cTokenIter == vCtx->dTokenCount.end()) {
      continue;
    }

    const int count = cTokenIter->second;

    assert(count > 0 && count <= ctx->penalty_last_n);

    // The academic publication that described this technique actually just only divided, but that would cause tokens with negative logits to become more likely, which is obviously wrong.
    // This is common fix for this problem, which is to multiply by the penalty instead of dividing.
    if (pCurP->data[vI].vLogit <= 0) {
      pCurP->data[vI].vLogit *= vCtx->cPenaltyRepeat;
    } else {
      pCurP->data[vI].vLogit /= vCtx->cPenaltyRepeat;
    }

    pCurP->data[vI].vLogit -= float(count) * vCtx->cPenaltyFreq + float(count > 0) * vCtx->cPenaltyPresent;
  }

  pCurP->vSorted = false;
}

static void fDgsSamplerPenaltiesReset(struct dgs_sampler * pSmpl) {
  auto * vCtx = (dgs_sampler_penalties *) pSmpl->vCtx;
  vCtx->vPrev.clear();
  vCtx->dTokenCount.clear();
}

static struct dgs_sampler * fDgsSamplerPenaltiesClone(const struct dgs_sampler * pSmpl) {
  const auto * vCtx    = (const dgs_sampler_penalties *) pSmpl->vCtx;
  auto *       vResult = fDgsSamplerInitPenalties(vCtx->cNVocab, vCtx->cPenaltyLastN, vCtx->cPenaltyRepeat,
                                                  vCtx->cPenaltyFreq, vCtx->cPenaltyPresent);

  // copy the state
  {
    auto * vResultCtx = (dgs_sampler_penalties *) vResult->vCtx;

    vResultCtx->vPrev       = vCtx->vPrev;
    vResultCtx->dTokenCount = vCtx->dTokenCount;
  }

  return vResult;
}

static void fDgsSamplerPenaltiesFree(struct dgs_sampler * pSmpl) {
  delete (dgs_sampler_penalties *) pSmpl->vCtx;
}

static bool fDgsSamplerPenaltiesBackendInit(struct dgs_sampler *       pSmpl,
                                            ggml_backend_buffer_type_t pBuft,
                                            uint32_t                   pNOutputsMaxPerSeq) {
  auto * vSctx = (dgs_sampler_penalties *) pSmpl->vCtx;

  if (pNOutputsMaxPerSeq > 1) {
    vSctx->init(false);
    return false;
  }

  const bool cRes = fDgsSamplerBackendSupport(pSmpl, pBuft);

  vSctx->init(cRes);

  return cRes;
}

static void fDgsSamplerPenaltiesBackendApply(struct dgs_sampler *      pSmpl,
                                             struct ggml_context *     pCtx,
                                             struct ggml_cgraph *      pGf,
                                             struct dgs_sampler_data * data) {
  GGML_UNUSED(pGf);

  auto * vSctx = (dgs_sampler_penalties *) pSmpl->vCtx;

  if (vSctx->mIsDisabled()) {
    return;
  }

  GGML_ASSERT(vSctx->cNVocab > 0);

  vSctx->vHasCandidates = data->vCandidates != nullptr;
  vSctx->vNMax          = std::min(vSctx->cPenaltyLastN, vSctx->cNVocab);

  vSctx->vInpTokenIds = ggml_new_tensor_1d(pCtx, GGML_TYPE_I32, vSctx->vNMax);
  ggml_set_name(vSctx->vInpTokenIds, "penalties_token_ids");
  ggml_set_input(vSctx->vInpTokenIds);

  vSctx->vInpCounts = ggml_new_tensor_1d(pCtx, GGML_TYPE_I32, vSctx->vNMax);
  ggml_set_name(vSctx->vInpCounts, "penalties_counts");
  ggml_set_input(vSctx->vInpCounts);

  if ((int32_t) vSctx->lHostTokenIds.size() != vSctx->vNMax) {
    vSctx->lHostTokenIds.assign(vSctx->vNMax, 0);
    vSctx->lHostCounts.assign(vSctx->vNMax, 0);
  }

  // flatten
  ggml_tensor * vLogits    = ggml_reshape_1d(pCtx, data->vLogits, ggml_nelements(data->vLogits));
  ggml_tensor * vGathered  = vLogits;
  ggml_tensor * vCountsF32 = ggml_cast(pCtx, vSctx->vInpCounts, GGML_TYPE_F32);

  if (vSctx->vHasCandidates) {
    ggml_tensor * vCandidates  = ggml_reshape_1d(pCtx, data->vCandidates, ggml_nelements(data->vCandidates));
    const int64_t cNCandidates = vCandidates->ne[0];
    GGML_ASSERT(cNCandidates == ggml_nelements(vLogits));

    ggml_tensor * vCountsRows  = ggml_fill(pCtx, ggml_new_tensor_2d(pCtx, GGML_TYPE_F32, 1, vSctx->cNVocab), 0.0f);
    ggml_tensor * vScatterRows = ggml_reshape_2d(pCtx, vCountsF32, 1, vSctx->vNMax);
    vCountsRows                = ggml_set_rows(pCtx, vCountsRows, vScatterRows, vSctx->vInpTokenIds);
    vCountsF32                 = ggml_get_rows(pCtx, vCountsRows, vCandidates);
    vCountsF32                 = ggml_reshape_1d(pCtx, vCountsF32, cNCandidates);
  } else {
    ggml_tensor * vLogitsRows = ggml_reshape_2d(pCtx, vLogits, 1, ggml_nelements(vLogits));
    vGathered                 = ggml_get_rows(pCtx, vLogitsRows, vSctx->vInpTokenIds);
    vGathered                 = ggml_reshape_1d(pCtx, vGathered, vSctx->vNMax);
  }

  ggml_tensor * vActiveMask   = ggml_step(pCtx, vCountsF32);
  ggml_tensor * vInactiveMask = ggml_sub(pCtx, ggml_fill(pCtx, vActiveMask, 1.0f), vActiveMask);

  ggml_tensor * vPenalized = vGathered;

  if (vSctx->cPenaltyRepeat != 1.0f) {
    ggml_tensor * vPosMask = ggml_step(pCtx, vPenalized);
    ggml_tensor * vNegMask = ggml_sub(pCtx, ggml_fill(pCtx, vPosMask, 1.0f), vPosMask);

    ggml_tensor * vPosScale    = ggml_scale(pCtx, vPosMask, 1.0f / vSctx->cPenaltyRepeat);
    ggml_tensor * vNegScale    = ggml_scale(pCtx, vNegMask, vSctx->cPenaltyRepeat);
    ggml_tensor * vRepeatScale = ggml_add(pCtx, vPosScale, vNegScale);

    // scale inactive entries with 1 to avoid -INF * 0 = NaN for values masked by top-p
    vRepeatScale = ggml_mul(pCtx, vRepeatScale, vActiveMask);
    vRepeatScale = ggml_add(pCtx, vRepeatScale, vInactiveMask);
    vPenalized   = ggml_mul(pCtx, vGathered, vRepeatScale);
  }

  if (vSctx->cPenaltyFreq != 0.0f) {
    ggml_tensor * vPenaltyFreq = ggml_scale(pCtx, vCountsF32, vSctx->cPenaltyFreq);
    vPenalized                 = ggml_sub(pCtx, vPenalized, vPenaltyFreq);
  }

  if (vSctx->cPenaltyPresent != 0.0f) {
    ggml_tensor * vPenaltyPresent = ggml_scale(pCtx, vActiveMask, vSctx->cPenaltyPresent);
    vPenalized                    = ggml_sub(pCtx, vPenalized, vPenaltyPresent);
  }

  if (vSctx->vHasCandidates) {
    data->vLogits = vPenalized;
  } else {
    ggml_tensor * vLogitsRows  = ggml_reshape_2d(pCtx, vLogits, 1, ggml_nelements(vLogits));
    ggml_tensor * vScatterRows = ggml_reshape_2d(pCtx, vPenalized, 1, vSctx->vNMax);
    vLogitsRows                = ggml_set_rows(pCtx, vLogitsRows, vScatterRows, vSctx->vInpTokenIds);
    data->vLogits              = ggml_reshape_1d(pCtx, vLogitsRows, ggml_nelements(vLogits));
  }
}

static void fDgsSamplerPenaltiesBackendSetInput(struct dgs_sampler * pSmpl) {
  auto * vSctx = (dgs_sampler_penalties *) pSmpl->vCtx;

  if (!vSctx->vInpTokenIds || !vSctx->vInpCounts || vSctx->vNMax <= 0 || vSctx->cNVocab <= 0) {
    return;
  }

  if (vSctx->mIsDisabled()) {
    return;
  }

  // fill active entries from the map
  int32_t vNActive = 0;

  for (const auto & vIt : vSctx->dTokenCount) {
    GGML_ASSERT(vNActive < vSctx->vNMax);
    vSctx->lHostTokenIds[vNActive] = vIt.first;
    vSctx->lHostCounts[vNActive]   = vIt.second;
    ++vNActive;
  }

  // Sorting is required because backend_apply uses ggml_set_rows (a scatter-back operation)
  std::vector<std::pair<int32_t, int32_t>> lEntries;
  lEntries.reserve(vNActive);
  for (int32_t vI = 0; vI < vNActive; ++vI) {
    lEntries.emplace_back(vSctx->lHostTokenIds[vI], vSctx->lHostCounts[vI]);
  }
  std::sort(lEntries.begin(), lEntries.end(), [](const auto & pA, const auto & pB) { return pA.first < pB.first; });
  for (int32_t vI = 0; vI < vNActive; ++vI) {
    vSctx->lHostTokenIds[vI] = lEntries[vI].first;
    vSctx->lHostCounts[vI]   = lEntries[vI].second;
  }

  // Padding: Finds a filler token id that is not present in token_count.
  // Use it to do padding for the arrays, it avoids resizing every time.
  // The arrays must always have exactly n_max entries (the GPU tensor is a fixed size).
  int32_t vFiller = 0;
  if (vNActive < vSctx->vNMax) {
    while (vSctx->dTokenCount.find(vFiller) != vSctx->dTokenCount.end()) {
      ++vFiller;
    }
    GGML_ASSERT(vFiller < vSctx->cNVocab);
  }

  // Fill the rest of the arrays with the filler token id and count 0.
  // Inactive slots are padded with a unique dummy token ID (count = 0).
  // The uniqueness matters because ggml_set_rows with duplicate indices can produce non-deterministic or incorrect results.
  // Using a filler token with count 0 that isn't in the active set is safe, because the active_mask step in backend_apply filters them out via ggml_step(counts_f32)
  for (int32_t vI = vNActive; vI < vSctx->vNMax; ++vI) {
    vSctx->lHostTokenIds[vI] = vFiller;
    vSctx->lHostCounts[vI]   = 0;
  }

  ggml_backend_tensor_set(vSctx->vInpTokenIds, vSctx->lHostTokenIds.data(), 0, vSctx->vNMax * sizeof(int32_t));
  ggml_backend_tensor_set(vSctx->vInpCounts, vSctx->lHostCounts.data(), 0, vSctx->vNMax * sizeof(int32_t));
}

static void fDgsSamplerPenaltiesBackendReset(struct dgs_sampler * pSmpl) {
  auto * vSctx        = (dgs_sampler_penalties *) pSmpl->vCtx;
  vSctx->vInpTokenIds = nullptr;
  vSctx->vInpCounts   = nullptr;
}

static struct dgs_sampler_i vDgsSamplerPenaltiesI = {
  /* .name              = */ fDgsSamplerPenaltiesName,
  /* .accept            = */ fDgsSamplerPenaltiesAccept,
  /* .apply             = */ fDgsSamplerPenaltiesApply,
  /* .reset             = */ fDgsSamplerPenaltiesReset,
  /* .clone             = */ fDgsSamplerPenaltiesClone,
  /* .free              = */ fDgsSamplerPenaltiesFree,
  /* .backend_init      = */ fDgsSamplerPenaltiesBackendInit,
  /* .backend_accept    = */ nullptr,
  /* .backend_apply     = */ fDgsSamplerPenaltiesBackendApply,
  /* .backend_set_input = */ fDgsSamplerPenaltiesBackendSetInput,
  /* .backend_reset     = */ fDgsSamplerPenaltiesBackendReset,
  /* .copy_state        = */ fDgsSamplerBackendCopyState<dgs_sampler_penalties>,
};

struct dgs_sampler * fDgsSamplerInitPenalties(int32_t pNVocab,
                                              int32_t pPenaltyLastN,
                                              float   pPenaltyRepeat,
                                              float   pPenaltyFreq,
                                              float   pPenaltyPresent) {
  pPenaltyLastN = std::max(pPenaltyLastN, 0);

  if (dgs_sampler_penalties::mIsDisabled(pPenaltyLastN, pPenaltyRepeat, pPenaltyFreq, pPenaltyPresent)) {
    return fDgsSamplerInitEmpty("?penalties");
  }

  return fDgsSamplerInit(
    /* .iface = */ &vDgsSamplerPenaltiesI,
    /* .ctx   = */ new dgs_sampler_penalties(pNVocab, pPenaltyLastN, pPenaltyRepeat, pPenaltyFreq, pPenaltyPresent));
}

// top-n-sigma

struct dgs_sampler_top_n_sigma {
  const float n;
};

static const char * fDgsSamplerTopNSigmaName(const struct dgs_sampler * /*smpl*/) {
  return "top-n-sigma";
}

static void fDgsSamplerTopNSigmaApply(struct dgs_sampler * pSmpl, dgs_token_data_array * pCurP) {
  auto * vCtx = (dgs_sampler_top_n_sigma *) pSmpl->vCtx;

  if (vCtx->n <= 0.0f || pCurP->size <= 1) {
    return;
  }

  // find max logit and calculate mean
  float  max         = pCurP->data[0].vLogit;
  float  vLogitsSum  = 0;
  size_t vValidCount = 0;
  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    // Only count non-negative infinity values
    if (pCurP->data[vI].vLogit != -INFINITY) {
      max = std::max(max, pCurP->data[vI].vLogit);
      vLogitsSum += pCurP->data[vI].vLogit;
      vValidCount++;
    }
  }
  float vMean = vValidCount > 0 ? vLogitsSum / vValidCount : 0;

  // calculate standard deviation
  float vAcc = 0;
  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    // Skip -infinity in std calculation
    if (pCurP->data[vI].vLogit != -INFINITY) {
      vAcc += pow(pCurP->data[vI].vLogit - vMean, 2);
    }
  }
  float vStd = vValidCount > 0 ? sqrt(vAcc / vValidCount) : 0;

  // apply mask
  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    if (pCurP->data[vI].vLogit < max - (vCtx->n * vStd)) {
      pCurP->data[vI].vLogit = -INFINITY;
    }
  }
}

static struct dgs_sampler * fDgsSamplerTopNSigmaClone(const struct dgs_sampler * pSmpl) {
  const auto * vCtx = (const dgs_sampler_top_n_sigma *) pSmpl->vCtx;
  return fDgsSamplerInitTopNSigma(vCtx->n);
}

static void fDgsSamplerTopNSigmaFree(struct dgs_sampler * pSmpl) {
  delete (dgs_sampler_top_n_sigma *) pSmpl->vCtx;
}

static struct dgs_sampler_i vDgsSamplerTopNSigmaI = {
  /* .name              = */ fDgsSamplerTopNSigmaName,
  /* .accept            = */ nullptr,
  /* .apply             = */ fDgsSamplerTopNSigmaApply,
  /* .reset             = */ nullptr,
  /* .clone             = */ fDgsSamplerTopNSigmaClone,
  /* .free              = */ fDgsSamplerTopNSigmaFree,
  /* .backend_init      = */ nullptr,
  /* .backend_accept    = */ nullptr,
  /* .backend_apply     = */ nullptr,
  /* .backend_set_input = */ nullptr,
  /* .backend_reset     = */ nullptr,
  /* .copy_state        = */ nullptr,
};

struct dgs_sampler * fDgsSamplerInitTopNSigma(float n) {
  const bool cIsEmpty = (n <= 0.0f);

  if (cIsEmpty) {
    return fDgsSamplerInitEmpty("?top-n-sigma");
  }

  return fDgsSamplerInit(
    /* .iface = */ &vDgsSamplerTopNSigmaI,
    /* .ctx   = */ new dgs_sampler_top_n_sigma{
      /* .n = */ n,
    });
}

// DRY

struct dgs_sampler_dry {
  const float   cDryMultiplier;
  const float   cDryBase;
  const int32_t cDryAllowedLength;
  const int32_t cDryPenaltyLastN;

  std::unordered_multimap<dgs_token, std::vector<dgs_token>> lDryProcessedBreakers;
  std::vector<int>                                           lDryRepeatCount;
  std::unordered_map<dgs_token, int>                         dDryMaxTokenRepeat;
  ring_buffer<dgs_token>                                     vLastTokens;
};

// Ported from Koboldcpp, original PR: https://github.com/LostRuins/koboldcpp/pull/982 (Original author: pi6am)
static void fGetOverlappingTokenSequences(const dgs_vocab &                                            pVocab,
                                          const std::string &                                          str,
                                          std::unordered_multimap<dgs_token, std::vector<dgs_token>> & pTokenSequences,
                                          int pMaxTailLen = -1) {
  for (dgs_token vTokenId = 0; vTokenId < (dgs_token) pVocab.n_tokens(); vTokenId++) {
    std::string vWord = pVocab.mDetokenize({ vTokenId }, true);
    if (vWord.find(str) != std::string::npos) {
      pTokenSequences.emplace(vTokenId, std::vector<dgs_token>());
    } else {
      size_t vWordLen = vWord.size();
      size_t vStrLen  = str.size();
      size_t vPos     = -1;
      while ((vPos = vWord.find(str[0], vPos + 1)) != std::string::npos) {
        bool   vMatch = true;
        size_t vI;
        for (vI = 1; vI < vStrLen && vI + vPos < vWordLen; ++vI) {
          if (vWord[vPos + vI] != str[vI]) {
            vMatch = false;
            break;
          }
        }
        if (vMatch) {
          std::vector<dgs_token> lTokenization = pVocab.mTokenize(str.substr(vI), false, false);
          if (pMaxTailLen >= 0 && lTokenization.size() > (size_t) pMaxTailLen) {
            lTokenization.resize(pMaxTailLen);
          }

          // Ensure we don't already have a duplicate matching tokenization
          auto lIts   = pTokenSequences.equal_range(vTokenId);
          bool vFound = false;
          for (auto lIt = lIts.first; lIt != lIts.second; ++lIt) {
            if (lTokenization == lIt->second) {
              vFound = true;
              break;
            }
          }
          if (!vFound) {
            pTokenSequences.emplace(vTokenId, lTokenization);
          }
        }
      }
    }
  }
}

static const char * fDgsSamplerDryName(const struct dgs_sampler * /*smpl*/) {
  return "dry";
}

static void fDgsSamplerDryAccept(struct dgs_sampler * pSmpl, dgs_token token) {
  auto * vCtx = (dgs_sampler_dry *) pSmpl->vCtx;
  if (vCtx->cDryMultiplier == 0.0f || vCtx->cDryBase < 1.0f || vCtx->cDryPenaltyLastN == 0) {
    return;
  }

  vCtx->vLastTokens.push_back(token);
}

// Ported from Koboldcpp, original PR: https://github.com/LostRuins/koboldcpp/pull/982 (Original author: pi6am)
static void fDgsSamplerDryApply(struct dgs_sampler * pSmpl, dgs_token_data_array * pCurP) {
  auto * vCtx = (dgs_sampler_dry *) pSmpl->vCtx;

  if (vCtx->cDryMultiplier == 0.0f || vCtx->cDryBase < 1.0f || vCtx->cDryPenaltyLastN == 0) {
    return;
  }

  int vLastNRepeat = std::min((int) vCtx->vLastTokens.size(), vCtx->cDryPenaltyLastN);

  if (vLastNRepeat <= vCtx->cDryAllowedLength) {
    return;
  }

  vCtx->lDryRepeatCount.assign(vLastNRepeat, 0);
  vCtx->dDryMaxTokenRepeat.clear();

  // Step 1: Look for restart sequences to limit the maximum repetition length.
  // Work backwards through the context looking for any token that begins a restart sequence.
  //
  // The collection `restart_sequences` is a mapping from a "head" token to all "tail"
  // sequences that together comprise a restart sequence. This allows us to quickly check
  // whether each token is the head of a complete sequence. Most restart sequences are actually
  // a single token, and for these the "tail" is an empty vector.
  //
  // If the token is a "head", test all restart sequences that begin with this token
  // (there will often only be one sequence for each token, but if sequences like 'aaaq1' and
  // 'aaa1' are used as restart strings, both could start with 'aaa' when tokenized). The
  // longest matching sequence (if any) is used to limit the maximum repetition length.
  //
  // Note that in the case case of a short sequence contained in a longer one, this might fail to
  // find the smallest value for `rep_limit`. For example, if 'amniotic' and 'ni' are both used as
  // restart sequences, 'ni' will be found first, and since it's shorter it will fail to suppress
  // 'otic'. This is a minor issue since fully contained restart sequences are likely to be rare.
  //
  // This is theoretically worst-case O(N^2) for arbitrary restart sequences, which is why we
  // have already clamped the maximum tail sequence length when generating `restart_sequences`.
  // With clamping, this scan is O(N) in the context length.

  int vRepLimit = vLastNRepeat;
  for (int vI = 0; vI < vLastNRepeat; ++vI) {
    dgs_token token = vCtx->vLastTokens.mRat(vI);
    auto      lIts  = vCtx->lDryProcessedBreakers.equal_range(token);
    if (lIts.first == vCtx->lDryProcessedBreakers.end()) {
      continue;
    }
    int vLongestMatch = -1;
    for (auto lIt = lIts.first; lIt != lIts.second; ++lIt) {
      // Note that (*it) does not contain the head character, so seq_len will be
      // the restart sequence length minus 1.
      // In the common case of a single-token restart sequence, (*it) will be empty
      // and we will trivially match.
      int vSeqLen = (int) lIt->second.size();
      if (vSeqLen > vLongestMatch && vSeqLen <= (int) vI) {
        bool vMatch = true;
        for (int vOffset = 0; vOffset < vSeqLen; ++vOffset) {
          // The -1 when indexing `last_tokens` is because we already matched the head.
          if (lIt->second[vOffset] != vCtx->vLastTokens.mRat(vI - vOffset - 1)) {
            vMatch = false;
            break;
          }
        }
        if (vMatch) {
          vLongestMatch = vSeqLen;
        }
      }
    }
    if (vLongestMatch >= 0) {
      // We found a restart sequence starting `i` tokens from the end and continuing for
      // `longest_match` tokens.
      vRepLimit = vI - vLongestMatch;
      break;
    }
  }
  if (vRepLimit < vCtx->cDryAllowedLength) {
    return;
  }

  // Step 2: Iterate in reverse over the last N tokens of the context, using the "Z-algorithm" (in
  // the reverse direction) to efficiently compute the positions and lengths of suffixes appearing
  // elsewhere in the context. We limit the suffix length to `rep_limit` to respect restart sequences.
  //
  // This algorithm is not currently documented on Wikipedia, but there is a clear description here:
  // https://ivanyu.me/blog/2014/10/15/z-algorithm/
  //
  // The code below is adapted from the public domain implementation by the same author here:
  // https://github.com/ivanyu/string-algorithms/blob/master/z_algorithm.py
  //
  // Example:
  // Last N tokens: a b c c b c y a b c
  // Repeat counts: 0 0 3 1 0 2 0 0 0 0
  //                    ^
  //   This `3` means that the last three tokens of the context (a b c) also appear here.
  //
  // This step is worst case O(N) since the Z-algorithm is linear, despite the appearance of nested
  // for/while loops. This can be seen by observing that the `lt` and `rt` bounds are set after each
  // repeated suffix is detected (i.e. after each while loop when n > 0). These bound variables
  // ensure that the inner while loops only examine each token in the context once as the outer
  // for loop iterates over the context.

  {
    const int cLast = vLastNRepeat - 1;

    int vRt = 0;
    int vLt = 0;

    for (int vK = 1; vK < vLastNRepeat; ++vK) {
      if (vK > vRt) {
        // If k is outside the current Z-box, do naive computation.
        int n = 0;
        while (n + vK < vLastNRepeat && vCtx->vLastTokens.mRat(n) == vCtx->vLastTokens.mRat(n + vK)) {
          ++n;
        }
        vCtx->lDryRepeatCount[cLast - vK] = std::min(n, vRepLimit);
        if (n > 0) {
          vLt = vK;
          vRt = vK + n - 1;
        }
      } else {
        // If k is inside the current Z-box, consider two cases.

        int vP            = vK - vLt;  // Pair index.
        int vRightPartLen = vRt - vK + 1;

        if (vCtx->lDryRepeatCount[cLast - vP] < vRightPartLen) {
          int n                             = std::min(vCtx->lDryRepeatCount[cLast - vP], vRepLimit);
          vCtx->lDryRepeatCount[cLast - vK] = n;
        } else {
          int vI = vRt + 1;
          while (vI < vLastNRepeat && vCtx->vLastTokens.mRat(vI) == vCtx->vLastTokens.mRat(vI - vK)) {
            vI += 1;
          }

          int n                             = std::min(vI - vK, vRepLimit);
          vCtx->lDryRepeatCount[cLast - vK] = n;
          vLt                               = vK;
          vRt                               = vI - 1;
        }
      }
    }
  }

  // Step 3: Iterate over dry_repeat_count and last_tokens, examining the maximum repeat length
  // that would be generated by emitting each new token that would extend a sequence.
  //
  // Following the same example as above:
  // Last N tokens: a b c c b c y a b c
  // Repeat counts: 0 0 3 1 0 2 0 0 0 0
  //
  // For each non-zero, look ahead one token. This token, if emitted, would extend the repetition.
  // c: 3 -> 4 (from `a b c` to `a b c c`)
  // b: 1 -> 2 (from `c` to `c b`)
  // y: 2 -> 3 (from `b c` to `b c y`)

  for (int vI = 0; vI < vLastNRepeat - 1; ++vI) {
    int vRepeatLen = vCtx->lDryRepeatCount[vI];
    if (vRepeatLen >= vCtx->cDryAllowedLength) {
      // This token ends a repeat, so the next token would continue one.
      // By convention, the value of `repeat_len` only includes the tokens currently
      // in the context, not the new token that would be added.
      dgs_token    token = vCtx->vLastTokens.mRat(vLastNRepeat - 2 - vI);
      // Track the maximum sequence ending in this token.
      const auto & vIt   = vCtx->dDryMaxTokenRepeat.find(token);
      if (vIt == vCtx->dDryMaxTokenRepeat.end() || vIt->second < vRepeatLen) {
        vCtx->dDryMaxTokenRepeat[token] = vRepeatLen;
      }
    }
  }

  // Step 4: Apply logit penalties based on the maximum repeat length for relevant tokens.

  // Prevent floating point overflow in `pow(penalty_base, exponent)` by clamping to `max_exponent`.
  // Compute it from `penalty_base` and the approximate log of `std::numeric_limits<float>::max()`
  const float cFloatMaxLog = 88.7228391f;
  int         vMaxExponent = 0;
  if (vCtx->cDryBase > 1.000001f) {
    vMaxExponent = cFloatMaxLog / std::log(vCtx->cDryBase);
  }

  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    const auto & vAfKvp = vCtx->dDryMaxTokenRepeat.find(pCurP->data[vI].vId);
    if (vAfKvp != vCtx->dDryMaxTokenRepeat.end()) {
      // Check all sequence breakers starting with this token
      auto lRange                = vCtx->lDryProcessedBreakers.equal_range(pCurP->data[vI].vId);
      bool vIsSingleTokenBreaker = false;

      for (auto lIt = lRange.first; lIt != lRange.second; ++lIt) {
        if (lIt->second.empty()) {
          vIsSingleTokenBreaker = true;
          break;
        }
      }

      // Apply penalty only if it's not a single-token sequence breaker
      if (!vIsSingleTokenBreaker) {
        int vRepeatExp = vAfKvp->second - vCtx->cDryAllowedLength;
        if (vMaxExponent > 0 && vRepeatExp > vMaxExponent) {
          vRepeatExp = vMaxExponent;
        }
        float vPenalty = vCtx->cDryMultiplier * std::pow(vCtx->cDryBase, vRepeatExp);
        pCurP->data[vI].vLogit -= vPenalty;
      }
    }
  }

  pCurP->vSorted = false;
}

static void fDgsSamplerDryReset(struct dgs_sampler * pSmpl) {
  auto * vCtx = (dgs_sampler_dry *) pSmpl->vCtx;
  vCtx->vLastTokens.clear();
  vCtx->lDryRepeatCount.clear();
  vCtx->dDryMaxTokenRepeat.clear();
}

static struct dgs_sampler * fDgsSamplerDryClone(const struct dgs_sampler * pSmpl) {
  const auto * vCtx = (dgs_sampler_dry *) pSmpl->vCtx;

  dgs_vocab vDummyVocab;

  // dummy vocab is passed because it is only needed for raw sequence breaker processing, which we have already done and will simply be copying
  auto * vResult = fDgsSamplerInitDry(&vDummyVocab, vCtx->cDryMultiplier, vCtx->cDryBase, vCtx->cDryAllowedLength,
                                      vCtx->cDryPenaltyLastN, NULL, 0);

  // Copy the state, including the processed breakers
  {
    auto * vResultCtx                 = (dgs_sampler_dry *) vResult->vCtx;
    vResultCtx->lDryProcessedBreakers = vCtx->lDryProcessedBreakers;
    vResultCtx->lDryRepeatCount       = vCtx->lDryRepeatCount;
    vResultCtx->dDryMaxTokenRepeat    = vCtx->dDryMaxTokenRepeat;
    vResultCtx->vLastTokens           = vCtx->vLastTokens;
  }

  return vResult;
}

static void fDgsSamplerDryFree(struct dgs_sampler * pSmpl) {
  delete (dgs_sampler_dry *) pSmpl->vCtx;
}

static struct dgs_sampler_i vDgsSamplerDryI = {
  /* .name              = */ fDgsSamplerDryName,
  /* .accept            = */ fDgsSamplerDryAccept,
  /* .apply             = */ fDgsSamplerDryApply,
  /* .reset             = */ fDgsSamplerDryReset,
  /* .clone             = */ fDgsSamplerDryClone,
  /* .free              = */ fDgsSamplerDryFree,
  /* .backend_init      = */ nullptr,
  /* .backend_accept    = */ nullptr,
  /* .backend_apply     = */ nullptr,
  /* .backend_set_input = */ nullptr,
  /* .backend_reset     = */ nullptr,
  /* .copy_state        = */ nullptr,
};

struct dgs_sampler * fDgsSamplerInitDry(const struct dgs_vocab * pVocab,
                                        float                    pDryMultiplier,
                                        float                    pDryBase,
                                        int32_t                  pDryAllowedLength,
                                        int32_t                  pDryPenaltyLastN,
                                        const char **            pSeqBreakers,
                                        size_t                   pNumBreakers) {
  pDryPenaltyLastN = std::max(pDryPenaltyLastN, 0);
  std::unordered_multimap<dgs_token, std::vector<dgs_token>> lProcessedBreakers;
  const int                                                  cMaxCharLen = 40;
  const int                                                  cMaxSeqLen  = 20;

  const bool cDryEnabled = (pDryMultiplier != 0.0f && pDryBase >= 1.0f && pDryPenaltyLastN != 0);

  if (!cDryEnabled) {
    return fDgsSamplerInitEmpty("?dry");
  }

  if (cDryEnabled && pSeqBreakers != nullptr && pNumBreakers > 0) {
    // Process sequence breakers
    for (size_t vI = 0; vI < pNumBreakers; ++vI) {
      if (pSeqBreakers[vI] == nullptr || std::strlen(pSeqBreakers[vI]) == 0) {
        DGS_LOG_WARN("skipping null or empty DRY sequence breaker at index %zu\n", vI);
        continue;
      }

      std::string vSequenceBreak(pSeqBreakers[vI]);
      if (vSequenceBreak.empty()) {
        DGS_LOG_WARN("skipping empty DRY sequence breaker\n");
        continue;
      }

      if (vSequenceBreak.size() > cMaxCharLen) {
        DGS_LOG_WARN("truncating DRY sequence breaker to %d characters\n", cMaxCharLen);
        vSequenceBreak.resize(cMaxCharLen);
      }

      fGetOverlappingTokenSequences(*pVocab, vSequenceBreak, lProcessedBreakers, cMaxSeqLen);
    }
  }

  return fDgsSamplerInit(
    /* .iface = */ &vDgsSamplerDryI,
    /* .ctx   = */ new dgs_sampler_dry{
      /* .dry_multiplier         = */ pDryMultiplier,
      /* .dry_base               = */ pDryBase,
      /* .dry_allowed_length     = */ pDryAllowedLength,
      /* .dry_penalty_last_n     = */ pDryPenaltyLastN,
      /* .dry_processed_breakers = */ std::move(lProcessedBreakers),
      /* .dry_repeat_count       = */ cDryEnabled ? std::vector<int>(pDryPenaltyLastN, 0) : std::vector<int>{},
      /* .dry_max_token_repeat   = */ {},
      /* .last_tokens            = */
      cDryEnabled ? ring_buffer<dgs_token>(pDryPenaltyLastN) : ring_buffer<dgs_token>(0),
    });
}

// wrapper for test-sampling.cpp
struct dgs_sampler * fDgsSamplerInitDryTesting(float                                       pDryMultiplier,
                                               float                                       pDryBase,
                                               int32_t                                     pDryAllowedLength,
                                               int32_t                                     pDryPenaltyLastN,
                                               const std::vector<std::vector<dgs_token>> & pSeqBreakers) {
  dgs_vocab vDummyVocab;
  auto *    vResult =
    fDgsSamplerInitDry(&vDummyVocab, pDryMultiplier, pDryBase, pDryAllowedLength, pDryPenaltyLastN, NULL, 0);
  auto * vCtx = (dgs_sampler_dry *) vResult->vCtx;

  // Process the token-based sequence breakers
  vCtx->lDryProcessedBreakers.clear();
  if (pSeqBreakers.empty()) {
    DGS_LOG_WARN("empty DRY sequence breakers list in dgs_sampler_init_dry_testing\n");
  } else {
    for (const auto & lBreaker : pSeqBreakers) {
      if (lBreaker.empty()) {
        DGS_LOG_WARN("skipping DRY empty sequence breaker\n");
        continue;
      }
      dgs_token              vHeadToken = lBreaker[0];
      std::vector<dgs_token> lTailTokens(lBreaker.begin() + 1, lBreaker.end());
      vCtx->lDryProcessedBreakers.emplace(vHeadToken, std::move(lTailTokens));
    }

    if (vCtx->lDryProcessedBreakers.empty()) {
      DGS_LOG_WARN("no valid DRY sequence breakers processed in dgs_sampler_init_dry_testing\n");
    }
  }

  return vResult;
}

// adaptive-p sampler state
//
// maintains an exponential moving average of the *ORIGINAL* probabilities
// of selected tokens, used to compute an adapted target at each sampling step.
//
// see dgs.h for a full description of the sampler
//
//
struct dgs_sampler_adaptive_p {
  const float        cTarget;           // target probability (0.0 - 1.0; negative = disabled)
  const float        cDecay;            // EMA decay; history ~= 1/(1-decay) tokens (0.0 - 0.99)
  const uint32_t     cSeed;             // original RNG seed
  uint32_t           vSeedCur;          // actual RNG seed
  std::mt19937       vRng;              // RNG state
  float              vWeightedSum;      // sum(p_i * decay^i)
  float              vTotalWeight;      // sum(decay^i), converges to 1/(1-decay)
  std::vector<float> lOriginalProbs;    // pre-transform probs, cached for EMA update
  dgs_token          vPendingTokenId;   // token ID of selected token
  int32_t            vPendingTokenIdx;  // index of orig. prob. of selected token in original_probs
};

// adaptive probability transformation constants
static constexpr float cDistributionWidth = 0.3f;
static constexpr float cPeakLogitValue    = 5.0f;
static constexpr float cSharpness         = 10.0f;
static constexpr float cInvWidth          = 1.0f / cDistributionWidth;

static const char * fDgsSamplerAdaptivePName(const struct dgs_sampler * /*smpl*/) {
  return "adaptive-p";
}

static void fDgsSamplerAdaptivePApply(struct dgs_sampler * pSmpl, dgs_token_data_array * pCurP) {
  auto * vCtx = (dgs_sampler_adaptive_p *) pSmpl->vCtx;

  fDgsSamplerSoftmaxImpl(pCurP, false);

  if (vCtx->cTarget < 0.0f) {
    // at negative target values, adaptive-p is no-op
    // we simply sample from the existing distribution
    pCurP->vSelected = fDgsSampleDist(pCurP, vCtx->vRng);
    return;
  }

  // store the original probabilities
  vCtx->lOriginalProbs.resize(pCurP->size);
  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    vCtx->lOriginalProbs[vI] = pCurP->data[vI].vP;
  }

  // using the EMA, compute the adapted target probability for the current sampling step
  auto  vTarget        = std::clamp(vCtx->cTarget, 0.0f, 1.0f);
  float vAdaptedTarget = std::clamp(
    vCtx->vTotalWeight == 0.0f ? vTarget : 2.0f * vTarget - (vCtx->vWeightedSum / vCtx->vTotalWeight), 0.0f, 1.0f);

  // adaptive probability transform
  //
  // quadratic near target for fine differentiation, transitioning to linear decay in the
  // tails. unbounded negative logits ensure proper suppression of far-from-target tokens
  // after the softmax.
  //
  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    if (pCurP->data[vI].vLogit == -INFINITY) {
      // don't transform logits that are -INFINITY
      // (as masked out by e.g. min-p and top-p when using backend sampling)
      continue;
    }
    float vDist            = std::abs((pCurP->data[vI].vP - vAdaptedTarget) * cInvWidth);
    pCurP->data[vI].vLogit = cPeakLogitValue - cSharpness * vDist * vDist / (1.0f + vDist);
  }

  // softmax and sample from the transformed distribution
  fDgsSamplerSoftmaxImpl(pCurP, false);
  const int cIdx   = fDgsSampleDist(pCurP, vCtx->vRng);
  pCurP->vSelected = cIdx;

  // store the selected token ID for acceptance later
  vCtx->vPendingTokenId  = pCurP->data[cIdx].vId;
  vCtx->vPendingTokenIdx = cIdx;
}

static void fDgsSamplerAdaptivePAccept(struct dgs_sampler * pSmpl, dgs_token token) {
  auto * vCtx = (dgs_sampler_adaptive_p *) pSmpl->vCtx;
  if (vCtx->vPendingTokenId == token) {
    GGML_ASSERT(vCtx->vPendingTokenId != DGS_TOKEN_NULL);
    GGML_ASSERT(vCtx->vPendingTokenIdx != -1);
    // update EMA with the original probability of the selected token
    vCtx->vWeightedSum = vCtx->lOriginalProbs[vCtx->vPendingTokenIdx] + vCtx->cDecay * vCtx->vWeightedSum;
    vCtx->vTotalWeight = 1.0f + vCtx->cDecay * vCtx->vTotalWeight;
  }
  vCtx->vPendingTokenId  = DGS_TOKEN_NULL;
  vCtx->vPendingTokenIdx = -1;
}

static void fDgsSamplerAdaptivePReset(struct dgs_sampler * pSmpl) {
  auto * vCtx            = (dgs_sampler_adaptive_p *) pSmpl->vCtx;
  // ctx->target and ctx->decay never change after init, so it's safe to keep them as is.
  // original_probs is completely overwritten on every call to _apply.
  // so we only need to reset the EMA state and pending token.
  vCtx->vWeightedSum     = vCtx->cTarget / (1.0f - vCtx->cDecay);
  vCtx->vTotalWeight     = 1.0f / (1.0f - vCtx->cDecay);
  vCtx->vPendingTokenId  = DGS_TOKEN_NULL;
  vCtx->vPendingTokenIdx = -1;
  vCtx->vSeedCur         = fGetRngSeed(vCtx->cSeed);
  vCtx->vRng.seed(vCtx->vSeedCur);
}

static struct dgs_sampler * fDgsSamplerAdaptivePClone(const struct dgs_sampler * pSmpl) {
  const auto * vCtx       = (const dgs_sampler_adaptive_p *) pSmpl->vCtx;
  auto *       vResult    = fDgsSamplerInitAdaptiveP(vCtx->cTarget, vCtx->cDecay, vCtx->cSeed);
  auto *       vResultCtx = (dgs_sampler_adaptive_p *) vResult->vCtx;

  // copy everything (target, decay, seed, and RNG are already set)
  vResultCtx->vWeightedSum     = vCtx->vWeightedSum;
  vResultCtx->vTotalWeight     = vCtx->vTotalWeight;
  vResultCtx->vPendingTokenId  = vCtx->vPendingTokenId;
  vResultCtx->vPendingTokenIdx = vCtx->vPendingTokenIdx;

  return vResult;
}

static void fDgsSamplerAdaptivePFree(struct dgs_sampler * pSmpl) {
  delete (dgs_sampler_adaptive_p *) pSmpl->vCtx;
}

static struct dgs_sampler_i vDgsSamplerAdaptivePI = {
  /* .name              = */ fDgsSamplerAdaptivePName,
  /* .accept            = */ fDgsSamplerAdaptivePAccept,
  /* .apply             = */ fDgsSamplerAdaptivePApply,
  /* .reset             = */ fDgsSamplerAdaptivePReset,
  /* .clone             = */ fDgsSamplerAdaptivePClone,
  /* .free              = */ fDgsSamplerAdaptivePFree,
  /* .backend_init      = */ nullptr,
  /* .backend_accept    = */ nullptr,
  /* .backend_apply     = */ nullptr,
  /* .backend_set_input = */ nullptr,
  /* .backend_reset     = */ nullptr,
  /* .copy_state        = */ nullptr,
};

struct dgs_sampler * fDgsSamplerInitAdaptiveP(float pTarget, float pDecay, uint32_t pSeed) {
  auto  vSeedCur      = fGetRngSeed(pSeed);
  float vClampedDecay = std::clamp(pDecay, 0.0f, 0.99f);
  return fDgsSamplerInit(
    /* .iface = */ &vDgsSamplerAdaptivePI,
    /* .ctx   = */ new dgs_sampler_adaptive_p{ /* .target            = */ pTarget,
                                               /* .decay             = */ vClampedDecay,
                                               /* .seed              = */ pSeed,
                                               /* .seed_cur          = */ vSeedCur,
                                               /* .rng               = */ std::mt19937(vSeedCur),
                                               /* .weighted_sum      = */ pTarget / (1.0f - vClampedDecay),
                                               /* .total_weight      = */ 1.0f / (1.0f - vClampedDecay),
                                               /* .original_probs    = */ {},
                                               /* .pending_token_id  = */ DGS_TOKEN_NULL,
                                               /* .pending_token_idx = */ -1 });
}

// logit-bias

struct dgs_sampler_logit_bias : public dgs_sampler_backend {
  const int32_t cNVocab;

  const std::vector<dgs_logit_bias> lLogitBias;

  std::vector<dgs_logit_bias> lToSearch;

  struct ggml_tensor * vInpLogitBias;
  struct ggml_tensor * vInpLogitIdxs;
};

static const char * fDgsSamplerLogitBiasName(const struct dgs_sampler * pSmpl) {
  auto * vCtx = (dgs_sampler_logit_bias *) pSmpl->vCtx;
  return vCtx->mGetName();
}

static void fDgsSamplerLogitBiasApply(struct dgs_sampler * pSmpl, dgs_token_data_array * pCurP) {
  auto * vCtx = (dgs_sampler_logit_bias *) pSmpl->vCtx;

  if (vCtx->lLogitBias.empty()) {
    return;
  }

  vCtx->lToSearch.clear();

  // update the candidates that have not been shuffled in the vocabulary (i.e. idx == id)
  for (const auto & vLb : vCtx->lLogitBias) {
    if (vLb.token >= 0 && pCurP->size > (size_t) vLb.token && pCurP->data[vLb.token].vId == vLb.token) {
      pCurP->data[vLb.token].vLogit += vLb.vBias;
    } else {
      vCtx->lToSearch.push_back(vLb);
    }
  }

  if (vCtx->lToSearch.empty()) {
    return;
  }

  // search for the remaining candidates that were not found in the previous step
  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    for (const auto & vLb : vCtx->lToSearch) {
      if (pCurP->data[vI].vId == vLb.token) {
        pCurP->data[vI].vLogit += vLb.vBias;
        break;
      }
    }
  }
}

static struct dgs_sampler * fDgsSamplerLogitBiasClone(const struct dgs_sampler * pSmpl) {
  const auto * vCtx = (const dgs_sampler_logit_bias *) pSmpl->vCtx;
  return fDgsSamplerInitLogitBias(vCtx->cNVocab, vCtx->lLogitBias.size(), vCtx->lLogitBias.data());
}

static void fDgsSamplerLogitBiasFree(struct dgs_sampler * pSmpl) {
  delete (dgs_sampler_logit_bias *) pSmpl->vCtx;
}

static void fDgsSamplerLogitBiasBackendApply(struct dgs_sampler *      pSmpl,
                                             struct ggml_context *     pCtx,
                                             struct ggml_cgraph *      pGf,
                                             struct dgs_sampler_data * data) {
  GGML_UNUSED(pGf);
  GGML_UNUSED(pCtx);

  auto * vSctx = (dgs_sampler_logit_bias *) pSmpl->vCtx;
  if (vSctx->lLogitBias.empty()) {
    return;
  }

  const size_t n = vSctx->lLogitBias.size();

  if (vSctx->vInpLogitBias == nullptr) {
    GGML_ASSERT(vSctx->vInpLogitIdxs == nullptr);

    vSctx->vInpLogitBias = ggml_new_tensor_2d(pCtx, GGML_TYPE_F32, 1, n);
    ggml_set_name(vSctx->vInpLogitBias, "logit_bias");
    ggml_set_input(vSctx->vInpLogitBias);

    vSctx->vInpLogitIdxs = ggml_new_tensor_1d(pCtx, GGML_TYPE_I32, n);
    ggml_set_name(vSctx->vInpLogitIdxs, "logit_idxs");
    ggml_set_input(vSctx->vInpLogitIdxs);
  }

  ggml_tensor * vCur = ggml_fill(pCtx, data->vLogits, 0.0f);

  vCur = ggml_reshape_2d(pCtx, vCur, 1, ggml_nelements(vCur));
  vCur = ggml_set_rows(pCtx, vCur, vSctx->vInpLogitBias, vSctx->vInpLogitIdxs);
  vCur = ggml_reshape_1d(pCtx, vCur, ggml_nelements(vCur));

  data->vLogits = ggml_add(pCtx, data->vLogits, vCur);
}

static void fDgsSamplerLogitBiasBackendSetInput(struct dgs_sampler * pSmpl) {
  auto * vSctx = (dgs_sampler_logit_bias *) pSmpl->vCtx;
  if (vSctx->lLogitBias.empty()) {
    return;
  }

  GGML_ASSERT(vSctx->vInpLogitBias != nullptr);
  GGML_ASSERT(vSctx->vInpLogitIdxs != nullptr);

  const size_t n = vSctx->lLogitBias.size();

  std::vector<float>   lDataLogitBias(n, 0.0f);
  std::vector<int32_t> lDataLogitIdxs(n, 0);
  for (size_t vI = 0; vI < n; ++vI) {
    const auto & vLb = vSctx->lLogitBias[vI];
    GGML_ASSERT(vLb.token >= 0 && vLb.token < (int32_t) vSctx->cNVocab);
    lDataLogitBias[vI] = vLb.vBias;
    lDataLogitIdxs[vI] = vLb.token;
  }

  ggml_backend_tensor_set(vSctx->vInpLogitBias, lDataLogitBias.data(), 0, ggml_nbytes(vSctx->vInpLogitBias));
  ggml_backend_tensor_set(vSctx->vInpLogitIdxs, lDataLogitIdxs.data(), 0, ggml_nbytes(vSctx->vInpLogitIdxs));
}

static void fDgsSamplerLogitBiasBackendReset(struct dgs_sampler * pSmpl) {
  auto * vSctx         = (dgs_sampler_logit_bias *) pSmpl->vCtx;
  vSctx->vInpLogitBias = nullptr;
  vSctx->vInpLogitIdxs = nullptr;
}

static bool fDgsSamplerLogitBiasBackendInit(struct dgs_sampler *       pSmpl,
                                            ggml_backend_buffer_type_t pBuft,
                                            uint32_t                   pNOutputsMaxPerSeq) {
  GGML_UNUSED(pBuft);
  GGML_UNUSED(pNOutputsMaxPerSeq);

  auto * vSctx = (dgs_sampler_logit_bias *) pSmpl->vCtx;

  vSctx->init(true);

  if (vSctx->lLogitBias.empty()) {
    return true;
  }

  return true;
}

static struct dgs_sampler_i vDgsSamplerLogitBiasI = {
  /* .name              = */ fDgsSamplerLogitBiasName,
  /* .accept            = */ nullptr,
  /* .apply             = */ fDgsSamplerLogitBiasApply,
  /* .reset             = */ nullptr,
  /* .clone             = */ fDgsSamplerLogitBiasClone,
  /* .free              = */ fDgsSamplerLogitBiasFree,
  /* .backend_init      = */ fDgsSamplerLogitBiasBackendInit,
  /* .backend_accept    = */ nullptr,
  /* .backend_apply     = */ fDgsSamplerLogitBiasBackendApply,
  /* .backend_set_input = */ fDgsSamplerLogitBiasBackendSetInput,
  /* .backend_reset     = */ fDgsSamplerLogitBiasBackendReset,
  /* .copy_state        = */ fDgsSamplerBackendCopyState<dgs_sampler_logit_bias>,
};

struct dgs_sampler * fDgsSamplerInitLogitBias(int32_t pNVocab, int32_t pNLogitBias, const dgs_logit_bias * pLogitBias) {
  const bool cIsEmpty = pNLogitBias <= 0;

  if (cIsEmpty) {
    return fDgsSamplerInitEmpty("?logit-bias");
  }

  return fDgsSamplerInit(
    /* .iface = */ &vDgsSamplerLogitBiasI,
    /* .ctx   = */ new dgs_sampler_logit_bias{
      ("logit-bias"),
      /* .n_vocab        = */ pNVocab,
      /* .logit_bias     = */ std::vector<dgs_logit_bias>(pLogitBias, pLogitBias + pNLogitBias),
      /* .to_search      = */ {},
      /* .inp_logit_bias = */ nullptr,
      /* .inp_logit_idxs = */ nullptr,
    });
}

// infill

//#define GGML_DEBUG_SAMPLER_INFILL

struct dgs_sampler_infill {
  const struct dgs_vocab * vVocab;

  std::vector<char> lBuf0;
  std::vector<char> lBuf1;
};

static const char * fDgsSamplerInfillName(const struct dgs_sampler * /*smpl*/) {
  return "infill";
}

static void fDgsSamplerInfillApply(struct dgs_sampler * pSmpl, dgs_token_data_array * pCurP) {
  auto * vCtx = (dgs_sampler_infill *) pSmpl->vCtx;

  fDgsSamplerSoftmaxImpl(pCurP, true);

#if defined(GGML_DEBUG_SAMPLER_INFILL)
#  define LOG_DBG_CUR DGS_LOG_DEBUG
#else
#  define LOG_DBG_CUR(...)
#endif

  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    LOG_DBG_CUR("%s: cur_p[%3zu] = { id: %6d, p: %.6f, logit: %6.3f }\n", __func__, i, cur_p->data[i].id,
                cur_p->data[i].p, cur_p->data[i].logit);
  }

  float vPTxtSum = 0.0f;
  float vPEogSum = 0.0f;

  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    if (vCtx->vVocab->mIsEog(pCurP->data[vI].vId)) {
      vPEogSum += pCurP->data[vI].vP;
    } else {
      vPTxtSum += pCurP->data[vI].vP;
    }
  }

  const float cRat = vPEogSum == 0.0 ? INFINITY : vPTxtSum / vPEogSum;
  GGML_UNUSED(cRat);

  LOG_DBG_CUR("%s: p_txt_sum = %.2f, p_eog_sum = %.2f, rat = %.2f, n = %zu\n", __func__, p_txt_sum, p_eog_sum, rat,
              cur_p->size);

  if (3 * vPEogSum * pCurP->size > vPTxtSum) {
    LOG_DBG_CUR("%s: the ratio p_txt/p_eog = %.2f is too low -> sampling EOG\n", __func__, p_txt_sum / p_eog_sum);

    // keep just the EOG tokens
    const auto cSizeOrg = pCurP->size;

    pCurP->size = 0;

    float vPSum = 0.0f;

    for (size_t vI = 0; vI < cSizeOrg; ++vI) {
      if (vCtx->vVocab->mIsEog(pCurP->data[vI].vId)) {
        vPSum += pCurP->data[vI].vP;

        pCurP->data[pCurP->size++] = pCurP->data[vI];
      }
    }

    // normalize probs
    for (size_t vI = 0; vI < pCurP->size; ++vI) {
      pCurP->data[vI].vP /= vPSum;
    }

    return;
  }

  size_t vNCombined = 0;
  GGML_UNUSED(vNCombined);

  // combine tokens with common prefix
  for (size_t vI0 = 0; vI0 < pCurP->size; ++vI0) {
    for (size_t vI1 = 0; vI1 < pCurP->size; ++vI1) {
      if (pCurP->data[vI0].vLogit == -INFINITY) {
        break;
      }

      if (vI0 == vI1 || pCurP->data[vI1].vLogit == -INFINITY) {
        continue;
      }

      int vLen0 = vCtx->vVocab->mTokenToPiece(pCurP->data[vI0].vId, vCtx->lBuf0.data(), vCtx->lBuf0.size(), 0, false);
      if (vLen0 < 0) {
        vCtx->lBuf0.resize(vLen0);
        vLen0 = vCtx->vVocab->mTokenToPiece(pCurP->data[vI0].vId, vCtx->lBuf0.data(), vCtx->lBuf0.size(), 0, false);
        assert(len0 > 0);
      }

      int vLen1 = vCtx->vVocab->mTokenToPiece(pCurP->data[vI1].vId, vCtx->lBuf1.data(), vCtx->lBuf1.size(), 0, false);
      if (vLen1 < 0) {
        vCtx->lBuf1.resize(vLen1);
        vLen1 = vCtx->vVocab->mTokenToPiece(pCurP->data[vI1].vId, vCtx->lBuf1.data(), vCtx->lBuf1.size(), 0, false);
        assert(len1 > 0);
      }

      // token i0 is a prefix of token i1
      if (vLen0 > 0 && vLen0 <= vLen1 && memcmp(vCtx->lBuf0.data(), vCtx->lBuf1.data(), vLen0) == 0) {
        int vDst = vI0;
        int vSrc = vI1;

        // merge into the token with higher probability
        if (pCurP->data[vI1].vP > pCurP->data[vI0].vP) {
          std::swap(vDst, vSrc);
        }

        pCurP->data[vDst].vP += pCurP->data[vSrc].vP;
        pCurP->data[vSrc].vLogit = -INFINITY;
        pCurP->data[vSrc].vP     = 0.0f;

        vNCombined++;
      }
    }
  }

  size_t vNNonEog = 0;

  size_t vSizeOrg = pCurP->size;

  float vPSum  = 0.0f;
  float vThold = 0.2f;

  pCurP->size = 0;

  LOG_DBG_CUR("%s: n_combined = %zu, applying thold = %.3f\n", __func__, n_combined, thold);

  for (size_t vI = 0; vI < vSizeOrg; ++vI) {
    const bool cIsEog = vCtx->vVocab->mIsEog(pCurP->data[vI].vId);

    if (pCurP->data[vI].vP < vThold && !cIsEog) {
      continue;
    }

    if (!cIsEog) {
      ++vNNonEog;
    }

    vPSum += pCurP->data[vI].vP;

    // keep this token
    pCurP->data[pCurP->size++] = pCurP->data[vI];
  }

  LOG_DBG_CUR("%s: n_non_eog = %zu\n", __func__, n_non_eog);

  // if no non-EOG tokens are left -> reduce cur_p to single EOT token
  if (vNNonEog == 0) {
    pCurP->size        = 1;
    pCurP->data[0].vId = vCtx->vVocab->mTokenEot();
    if (pCurP->data[0].vId == DGS_TOKEN_NULL) {
      pCurP->data[0].vId = vCtx->vVocab->mTokenEos();
    }
    pCurP->data[0].vLogit = 1.0f;

    GGML_ASSERT(pCurP->data[0].vId != DGS_TOKEN_NULL);

    return;
  }

  // normalize probs
  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    pCurP->data[vI].vP /= vPSum;

    LOG_DBG_CUR("%s: cur_p[%3zu] = { id: %6d, p: %.6f, logit: %6.3f }\n", __func__, i, cur_p->data[i].id,
                cur_p->data[i].p, cur_p->data[i].logit);
  }

  vSizeOrg = pCurP->size;
  vPSum    = 0.0f;
  vThold   = 1.0 / (vNNonEog + 1);

  pCurP->size = 0;

  LOG_DBG_CUR("%s: applying thold = %.3f\n", __func__, thold);

  for (size_t vI = 0; vI < vSizeOrg; ++vI) {
    const bool cIsEog = vCtx->vVocab->mIsEog(pCurP->data[vI].vId);

    if (pCurP->data[vI].vP < vThold && !cIsEog) {
      continue;
    }

    vPSum += pCurP->data[vI].vP;

    pCurP->data[pCurP->size++] = pCurP->data[vI];
  }

  // normalize probs
  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    pCurP->data[vI].vP /= vPSum;

    LOG_DBG_CUR("%s: cur_p[%3zu] = { id: %6d, p: %.6f, logit: %6.3f }\n", __func__, i, cur_p->data[i].id,
                cur_p->data[i].p, cur_p->data[i].logit);
  }

#undef LOG_DBG_CUR
}

static struct dgs_sampler * fDgsSamplerInfillClone(const struct dgs_sampler * pSmpl) {
  const auto * vCtx = (const dgs_sampler_infill *) pSmpl->vCtx;
  return fDgsSamplerInitInfill(vCtx->vVocab);
}

static void fDgsSamplerInfillFree(struct dgs_sampler * pSmpl) {
  delete (dgs_sampler_infill *) pSmpl->vCtx;
}

static struct dgs_sampler_i vDgsSamplerInfillI = {
  /* .name              = */ fDgsSamplerInfillName,
  /* .accept            = */ nullptr,
  /* .apply             = */ fDgsSamplerInfillApply,
  /* .reset             = */ nullptr,
  /* .clone             = */ fDgsSamplerInfillClone,
  /* .free              = */ fDgsSamplerInfillFree,
  /* .backend_init      = */ nullptr,
  /* .backend_accept    = */ nullptr,
  /* .backend_apply     = */ nullptr,
  /* .backend_set_input = */ nullptr,
  /* .backend_reset     = */ nullptr,
  /* .copy_state        = */ nullptr,
};

struct dgs_sampler * fDgsSamplerInitInfill(const struct dgs_vocab * pVocab) {
  return fDgsSamplerInit(
    /* .iface = */ &vDgsSamplerInfillI,
    /* .ctx   = */ new dgs_sampler_infill{
      /* .vocab = */ pVocab,
      /* .buf0  = */ std::vector<char>(512),
      /* .buf1  = */ std::vector<char>(512),
    });
}

void fDgsSamplerCopy(const struct dgs_sampler * pSrc, struct dgs_sampler * pDst) {
  if (!pSrc || !pDst || pSrc == pDst) {
    return;
  }

  GGML_ASSERT(pSrc->vIface == pDst->vIface && "dgs_sampler_copy: cannot copy between different sampler types");

  if (pDst->vIface->vCopyState) {
    pDst->vIface->vCopyState(pSrc, pDst);
    return;
  }

  // build a temporary sampler carrying src's current state
  dgs_sampler * vTmp = fDgsSamplerClone(pSrc);

  // free dst's old state (frees dst->ctx, including children for a chain)
  if (pDst->vIface->free) {
    pDst->vIface->free(pDst);
  }

  // transplant tmp's state into dst, then destroy the (now empty) temp shell
  pDst->vCtx = vTmp->vCtx;
  vTmp->vCtx = nullptr;
  delete vTmp;
}

// utils

uint32_t fDgsSamplerGetSeed(const struct dgs_sampler * pSmpl) {
  if (pSmpl->vIface == &vDgsSamplerDistI) {
    return ((const dgs_sampler_dist *) pSmpl->vCtx)->vSeedCur;
  }

  if (pSmpl->vIface == &vDgsSamplerMirostatI) {
    return ((const dgs_sampler_mirostat *) pSmpl->vCtx)->vSeedCur;
  }

  if (pSmpl->vIface == &vDgsSamplerMirostatV2I) {
    return ((const dgs_sampler_mirostat_v2 *) pSmpl->vCtx)->vSeedCur;
  }

  if (pSmpl->vIface == &vDgsSamplerChainI) {
    const auto * vCtx = (const dgs_sampler_chain *) pSmpl->vCtx;
    for (auto lIt = vCtx->lSamplers.rbegin(); lIt != vCtx->lSamplers.rend(); ++lIt) {
      const uint32_t cSeed = fDgsSamplerGetSeed(lIt->vPtr);
      if (cSeed != DGS_DEFAULT_SEED) {
        return cSeed;
      }
    }
  }

  return DGS_DEFAULT_SEED;
}

// perf

struct dgs_perf_sampler_data fDgsPerfSampler(const struct dgs_sampler * pChain) {
  struct dgs_perf_sampler_data data = {};

  if (pChain == nullptr || pChain->vIface != &vDgsSamplerChainI) {
    GGML_ABORT("%s: invalid sampler passed - requires a sampler created with dgs_sampler_chain_init()\n", __func__);
  }

  const auto * vCtx = (const struct dgs_sampler_chain *) pChain->vCtx;

  data.vTSampleMs = 1e-3 * vCtx->vTSampleUs;
  data.vNSample   = std::max(0, vCtx->vNSample);

  return data;
}

void fDgsPerfSamplerPrint(const struct dgs_sampler * pChain) {
  const auto data = fDgsPerfSampler(pChain);

  DGS_LOG_INFO("%s:    samplers time = %10.2f ms / %5d runs\n", __func__, data.vTSampleMs, data.vNSample);
}

void fDgsPerfSamplerReset(struct dgs_sampler * pChain) {
  if (pChain == nullptr || pChain->vIface != &vDgsSamplerChainI) {
    GGML_ABORT("%s: invalid sampler passed - requires a sampler created with dgs_sampler_chain_init()\n", __func__);
  }

  auto * vCtx = (struct dgs_sampler_chain *) pChain->vCtx;

  vCtx->vTSampleUs = 0;
  vCtx->vNSample   = 0;
}
