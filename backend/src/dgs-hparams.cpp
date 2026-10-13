#include "dgs-hparams.h"

#include "ggml.h"

#include <algorithm>
#include <cassert>

void dgs_hparams::mSetSwaPattern(uint32_t pNPattern, bool pDenseFirst) {
  if (pDenseFirst) {
    for (uint32_t vIl = 0; vIl < n_layer(); ++vIl) {
      aIsSwaImpl[vIl] = pNPattern == 0 || (vIl % pNPattern != 0);
    }
  } else {
    for (uint32_t vIl = 0; vIl < n_layer(); ++vIl) {
      aIsSwaImpl[vIl] = pNPattern == 0 || (vIl % pNPattern < (pNPattern - 1));
    }
  }

  for (uint32_t vIl = n_layer(); vIl < n_layer_all; ++vIl) {
    aIsSwaImpl[vIl] = false;
  }
}

void dgs_hparams::mSetRecrPattern(uint32_t pNPattern, bool pDenseFirst) {
  if (pDenseFirst) {
    for (uint32_t vIl = 0; vIl < n_layer(); ++vIl) {
      aIsRecrImpl[vIl] = pNPattern == 0 || (vIl % pNPattern != 0);
    }
  } else {
    for (uint32_t vIl = 0; vIl < n_layer(); ++vIl) {
      aIsRecrImpl[vIl] = pNPattern == 0 || (vIl % pNPattern < (pNPattern - 1));
    }
  }

  for (uint32_t vIl = n_layer(); vIl < n_layer_all; ++vIl) {
    aIsRecrImpl[vIl] = false;
  }
}

bool dgs_hparams::mIsSwaAny() const {
  for (uint32_t vIl = 0; vIl < n_layer_all; ++vIl) {
    if (aIsSwaImpl[vIl]) {
      return true;
    }
  }

  return false;
}

uint32_t dgs_hparams::n_head(uint32_t pIl) const {
  if (pIl < n_layer_all) {
    return aNHeadArr[pIl];
  }

  GGML_ABORT("fatal error");
}

uint32_t dgs_hparams::n_head_kv(uint32_t pIl) const {
  if (pIl < n_layer_all) {
    return aNHeadKvArr[pIl];
  }

  GGML_ABORT("fatal error");
}

uint32_t dgs_hparams::n_ff(uint32_t pIl) const {
  if (pIl < n_layer_all) {
    return aNFfArr[pIl];
  }

  GGML_ABORT("fatal error");
}

uint32_t dgs_hparams::mNFfExp(uint32_t pIl) const {
  if (pIl < n_layer_all) {
    return aNFfExpArr[pIl];
  }

  GGML_ABORT("fatal error");
}

uint32_t dgs_hparams::n_expert_used(uint32_t pIl) const {
  if (pIl < n_layer_all) {
    return aNExpertUsedArr[pIl];
  }

  GGML_ABORT("fatal error");
}

uint32_t dgs_hparams::mNExpertUsedMax() const {
  uint32_t vVal = 0;
  for (uint32_t vIl = 0; vIl < n_layer_all; ++vIl) {
    vVal = std::max(vVal, n_expert_used(vIl));
  }

  return vVal;
}

uint32_t dgs_hparams::mNGqa(uint32_t pIl) const {
  const uint32_t cNHead   = this->n_head(pIl);
  const uint32_t cNHeadKv = this->n_head_kv(pIl);

  if (cNHeadKv == 0) {
    return 0;
  }

  return cNHead / cNHeadKv;
}

uint32_t dgs_hparams::n_rot(uint32_t pIl) const {
  if (pIl < n_layer_all) {
    return mIsSwa(pIl) ? vNRotSwa : vNRotFull;
  }

  GGML_ABORT("fatal error");
}

uint32_t dgs_hparams::mNEmbdInp() const {
  if (vNEmbdInpImpl > 0) {
    return vNEmbdInpImpl;
  }

  uint32_t vNEmbdInp = n_embd;

  if (vNDeepstackLayers > 0) {
    vNEmbdInp += n_embd * vNDeepstackLayers;
  }

  return vNEmbdInp;
}

uint32_t dgs_hparams::mNEmbdInpEnc() const {
  return vNEmbdInpEncImpl > 0 ? vNEmbdInpEncImpl : mNEmbdInp();
}

uint32_t dgs_hparams::mNEmbdOut() const {
  return vNEmbdOutImpl > 0 ? vNEmbdOutImpl : n_embd;
}

uint32_t dgs_hparams::n_embd_head_k(uint32_t pIl) const {
  if (pIl < n_layer_all) {
    return mIsSwa(pIl) ? vNEmbdHeadKSwa : vNEmbdHeadKFull;
  }

  GGML_ABORT("fatal error");
}

uint32_t dgs_hparams::n_embd_head_v(uint32_t pIl) const {
  if (pIl < n_layer_all) {
    return mIsSwa(pIl) ? vNEmbdHeadVSwa : vNEmbdHeadVFull;
  }

  GGML_ABORT("fatal error");
}

uint32_t dgs_hparams::n_embd_k_gqa(uint32_t pIl) const {
  const uint32_t cNHeadKv = this->n_head_kv(pIl);

  return n_embd_head_k(pIl) * cNHeadKv;
}

uint32_t dgs_hparams::n_embd_v_gqa(uint32_t pIl) const {
  const uint32_t cNHeadKv = this->n_head_kv(pIl);

  return n_embd_head_v(pIl) * cNHeadKv;
}

bool dgs_hparams::mIsNEmbdKGqaVariable() const {
  const uint32_t cVal = n_embd_k_gqa();
  for (uint32_t vIl = 0; vIl < n_layer_all; ++vIl) {
    if (cVal != n_embd_k_gqa(vIl)) {
      return true;
    }
  }

  return false;
}

bool dgs_hparams::mIsNEmbdVGqaVariable() const {
  const uint32_t cVal = n_embd_v_gqa();
  for (uint32_t vIl = 0; vIl < n_layer_all; ++vIl) {
    if (cVal != n_embd_v_gqa(vIl)) {
      return true;
    }
  }

  return false;
}

uint32_t dgs_hparams::mNEmbdKGqaMax() const {
  uint32_t vVal = n_embd_k_gqa();
  for (uint32_t vIl = 0; vIl < n_layer_all; ++vIl) {
    vVal = std::max(vVal, n_embd_k_gqa(vIl));
  }

  return vVal;
}

uint32_t dgs_hparams::mNEmbdVGqaMax() const {
  uint32_t vVal = n_embd_v_gqa();
  for (uint32_t vIl = 0; vIl < n_layer_all; ++vIl) {
    vVal = std::max(vVal, n_embd_v_gqa(vIl));
  }

  return vVal;
}

uint32_t dgs_hparams::mNEmbdR() const {
  if (vWkvHeadSize != 0) {
    // for RWKV models
    return vTokenShiftCount * n_embd;
  }

  if (vNShortconvLCache != 0) {
    // for LFM2 models
    return n_embd * (vNShortconvLCache - 1);
  }

  if (vNEmbdHeadKda != 0) {
    // for Kimi KDA layers
    // Conv state for Q, K, V: 3 * (d_conv - 1) * n_head * head_dim
    const uint32_t cDInner = n_head() * vNEmbdHeadKda;  // 32 * 128 = 4096
    return 3 * (vSsmDConv > 0 ? vSsmDConv - 1 : 3) * cDInner;
  }

  // TODO: maybe support other convolution strides than 1
  // NOTE: since the first column of the conv_state is shifted out each time, it's not actually needed
  // Corresponds to Mamba's conv_states size
  const uint32_t cNConv = (vSsmDConv > 0 ? vSsmDConv - 1 : 0) * (vSsmDInner + 2 * vSsmNGroup * vSsmDState);

  // PLE conv history needs its own row: Meta splits cache_r_l by head, so a history packed behind the first is unaddressable
  // it lives in cache_ple_r_l instead, mirrored like the rest of the PLE module
  return cNConv;
}

uint32_t dgs_hparams::mNEmbdS() const {
  if (vWkvHeadSize != 0) {
    // corresponds to RWKV's wkv_states size
    return n_embd * vWkvHeadSize;
  }

  if (vNEmbdHeadKda != 0) {
    // for Kimi KDA layers
    // Full recurrent state: head_dim * head_dim * n_head
    // h tensor shape for delta attention: [head_dim, head_dim, n_head]
    return vNEmbdHeadKda * vNEmbdHeadKda * n_head();  // 128 * 128 * 32 = 524288
  }

  if (vNEmbdHeadLa != 0) {
    // for MiniMax-Text-01 linear attention layers
    // Full recurrent state: head_dim * head_dim * n_head
    // tensor shape for linear attention: [head_dim, head_dim, n_head]
    return vNEmbdHeadLa * vNEmbdHeadLa * n_head();  // 128 * 128 * 64 = 1048576
  }

  // corresponds to Mamba's ssm_states size
  return vSsmDState * vSsmDInner;
}

bool dgs_hparams::mIsRecr(uint32_t pIl) const {
  if (pIl < n_layer_all) {
    return aIsRecrImpl[pIl];
  }

  GGML_ABORT("%s: il (%u) out of bounds (n_layer_all: %u)\n", __func__, pIl, n_layer_all);
}

uint32_t dgs_hparams::mPleConvState() const {
  if (vPleNHeads == 0 || vPleConvKernel == 0) {
    return 0;
  }

  // dilation equals the n-gram size, matching the reference module
  return (vPleConvKernel - 1) * vPleNgramSize * vDsv4HcMult * n_embd;
}

bool dgs_hparams::mIsPle(uint32_t pIl) const {
  if (pIl < n_layer_all) {
    return vIsPleImpl[pIl];
  }

  GGML_ABORT("%s: il (%u) out of bounds (n_layer_all: %u)\n", __func__, pIl, n_layer_all);
}

uint32_t dgs_hparams::mNPosPerEmbd() const {
  return (vRopeType == cDgsRopeTypeMrope || vRopeType == cDgsRopeTypeImrope) ? GGML_MROPE_SECTIONS : 1;
}

bool dgs_hparams::mIsSwa(uint32_t pIl) const {
  if (pIl < n_layer_all) {
    return aIsSwaImpl[pIl];
  }

  GGML_ABORT("%s: il (%u) out of bounds (n_layer_all: %u)\n", __func__, pIl, n_layer_all);
}

bool dgs_hparams::mIsMla() const {
  assert((n_embd_head_k_mla_impl == 0 && n_embd_head_v_mla_impl == 0) ||
         (n_embd_head_k_mla_impl != 0 && n_embd_head_v_mla_impl != 0));

  return vNEmbdHeadKMlaImpl != 0 && vNEmbdHeadVMlaImpl != 0;
}

bool dgs_hparams::mIsIndexerFull(uint32_t pIl) const {
  if (pIl < n_layer()) {
    return aIsIndexerFullImpl[pIl];
  }

  GGML_ABORT("%s: il (%u) out of bounds (n_layer: %u)\n", __func__, pIl, n_layer());
}

uint32_t dgs_hparams::mNEmbdHeadKMla() const {
  return mIsMla() ? vNEmbdHeadKMlaImpl : n_embd_head_k();
}

uint32_t dgs_hparams::mNEmbdHeadVMla() const {
  return mIsMla() ? vNEmbdHeadVMlaImpl : n_embd_head_v();
}

bool dgs_hparams::mHasKv(uint32_t pIl) const {
  if (vNLayerKvFromStart >= 0) {
    if (pIl < (uint32_t) vNLayerKvFromStart) {
      return true;
    }

    return false;
  }

  // by default, all layers have kv
  return true;
}

bool dgs_hparams::mHasRope(uint32_t pIl) const {
  // the router layer stores adapter routing signal, not positional info,
  // so it must not be RoPE-shifted
  if (vRouterLayer >= 0 && (int32_t) pIl == vRouterLayer) {
    return false;
  }

  if (pIl < n_layer_all) {
    return aRopePattern[pIl] != 0;
  }

  GGML_ABORT("%s: il (%u) out of bounds (n_layer_all: %u)\n", __func__, pIl, n_layer_all);
}

uint32_t dgs_hparams::n_layer() const {
  return n_layer_all - n_layer_nextn;
}

bool dgs_hparams::mUseMrope() const {
  return aRopeSections[0] > 0 && aRopeSections[1] > 0;
}
