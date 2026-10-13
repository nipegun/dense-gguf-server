#include "../dgs-model-policy.h"
#include "dgs-memory-recurrent.h"
#include "models.h"

void dgs_model_qwen35::mLoadArchHparams(dgs_model_loader & pMl) {
  pMl.get_key(cLlmKvAttentionLayernormRmsEps, hparams.vFNormRmsEps);
  pMl.get_key_or_arr(cLlmKvRopeDimensionSections, hparams.aRopeSections, 4, true);

  // Load linear attention (gated delta net) parameters
  pMl.get_key(cLlmKvSsmConvKernel, hparams.vSsmDConv);
  pMl.get_key(cLlmKvSsmInnerSize, hparams.vSsmDInner);
  pMl.get_key(cLlmKvSsmStateSize, hparams.vSsmDState);
  pMl.get_key(cLlmKvSsmTimeStepRank, hparams.vSsmDtRank);
  pMl.get_key(cLlmKvSsmGroupCount, hparams.vSsmNGroup);

  // Mark recurrent layers (linear attention layers). MTP layers are dense
  // attention-only and must be flagged non-recurrent.
  if (!pMl.get_key_or_arr(cLlmKvAttentionRecurrentLayers, hparams.aIsRecrImpl, hparams.n_layer_all, false)) {
    uint32_t vFullAttnInterval = 4;
    pMl.get_key(cLlmKvFullAttentionInterval, vFullAttnInterval, false);
    for (uint32_t vI = 0; vI < hparams.n_layer_all; ++vI) {
      hparams.aIsRecrImpl[vI] = (vI < hparams.n_layer()) && ((vI + 1) % vFullAttnInterval != 0);
    }
  }

  fDgsValidateModelProfile(pMl, hparams);

  switch (hparams.n_layer()) {
    case 24:
      type = cLlmType2b;
      break;
    case 32:
      type = hparams.n_embd == 2560 ? cLlmType4b : cLlmType9b;
      break;
    case 64:
      type = cLlmType27b;
      break;
    default:
      type = cLlmTypeUnknown;
  }
}

void dgs_model_qwen35::mLoadArchTensors(dgs_model_loader & pMl) {
  DGS_LOAD_LOCALS;

  const auto cNf         = mNextnFlags(pMl);
  const int  cTrunkFlags = cNf.vTrunk;
  const int  cMtpFlags   = cNf.vMtp;

  vTokEmbd = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), { n_embd, n_vocab }, 0);

  // output
  vOutputNorm = mCreateTensor(cTn(cLlmTensorOutputNorm, "weight"), { n_embd }, 0);
  vOutput     = mCreateTensor(cTn(cLlmTensorOutput, "weight"), { n_embd, n_vocab }, cTensorNotRequired);

  // optional projection of the embeddings output
  vClsOut  = mCreateTensor(cTn(cLlmTensorClsOut, "weight"), { n_embd, hparams.mNEmbdOut() }, cTensorNotRequired);
  vClsOutB = mCreateTensor(cTn(cLlmTensorClsOut, "bias"), { hparams.mNEmbdOut() }, cTensorNotRequired);

  // if output is NULL, init from the input tok embed
  if (vOutput == NULL) {
    vOutput = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), { n_embd, n_vocab }, cTensorDuplicated);
  }

  auto vLoadBlockTrunk = [&](int pIl, int pFlags) {
    auto & layer = lLayers[pIl];

    // Calculate dimensions from hyperparameters
    const int64_t cHeadKDim = hparams.vSsmDState;
    const int64_t cHeadVDim = hparams.vSsmDState;
    const int64_t cNKHeads  = hparams.vSsmNGroup;
    const int64_t cNVHeads  = hparams.vSsmDtRank;
    const int64_t cKeyDim   = cHeadKDim * cNKHeads;
    const int64_t cValueDim = cHeadVDim * cNVHeads;
    const int64_t cConvDim  = cKeyDim * 2 + cValueDim;

    layer.vAttnNorm     = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", pIl), { n_embd }, pFlags);
    layer.vAttnPostNorm = mCreateTensor(cTn(cLlmTensorAttnPostNorm, "weight", pIl), { n_embd }, pFlags);

    if (!hparams.mIsRecr(pIl)) {
      // Attention layers
      mCreateTensorQkv(layer, pIl, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, pFlags);
      layer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", pIl), { n_embd_head_k * n_head, n_embd }, pFlags);

      // Q/K normalization for attention layers
      layer.vAttnQNorm = mCreateTensor(cTn(cLlmTensorAttnQNorm, "weight", pIl), { n_embd_head_k }, pFlags);
      layer.vAttnKNorm = mCreateTensor(cTn(cLlmTensorAttnKNorm, "weight", pIl), { n_embd_head_k }, pFlags);
    } else {
      // Linear attention (gated delta net) specific tensors
      // Create tensors with calculated dimensions
      layer.vWqkv =
        mCreateTensor(cTn(cLlmTensorAttnQkv, "weight", pIl), { n_embd, cKeyDim * 2 + cValueDim }, cTensorNotRequired);
      layer.vWqkvGate =
        mCreateTensor(cTn(cLlmTensorAttnGate, "weight", pIl), { n_embd, cValueDim }, cTensorNotRequired);
      layer.vSsmConv1d =
        mCreateTensor(cTn(cLlmTensorSsmConv1d, "weight", pIl), { hparams.vSsmDConv, cConvDim }, pFlags);
      layer.vSsmDt    = mCreateTensor(cTn(cLlmTensorSsmDt, "bias", pIl), { hparams.vSsmDtRank }, pFlags);
      layer.vSsmA     = mCreateTensor(cTn(cLlmTensorSsmANoscan, pIl), { hparams.vSsmDtRank }, pFlags);
      layer.vSsmBeta  = mCreateTensor(cTn(cLlmTensorSsmBeta, "weight", pIl), { n_embd, cNVHeads }, pFlags);
      layer.vSsmAlpha = mCreateTensor(cTn(cLlmTensorSsmAlpha, "weight", pIl), { n_embd, cNVHeads }, pFlags);
      layer.vSsmNorm  = mCreateTensor(cTn(cLlmTensorSsmNorm, "weight", pIl), { cHeadVDim }, pFlags);
      layer.vSsmOut   = mCreateTensor(cTn(cLlmTensorSsmOut, "weight", pIl), { cValueDim, n_embd }, pFlags);
    }

    layer.vFfnGate = mCreateTensor(cTn(cLlmTensorFfnGate, "weight", pIl), { n_embd, n_ff }, pFlags);
    layer.vFfnDown = mCreateTensor(cTn(cLlmTensorFfnDown, "weight", pIl), { n_ff, n_embd }, pFlags);
    layer.vFfnUp   = mCreateTensor(cTn(cLlmTensorFfnUp, "weight", pIl), { n_embd, n_ff }, pFlags);
  };

  auto vLoadBlockMtp = [&](int pIl) {
    auto & layer = lLayers[pIl];

    // MTP block looks like a full-attention Qwen3.5 decoder block.
    layer.vAttnNorm     = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", pIl), { n_embd }, cMtpFlags);
    layer.vAttnPostNorm = mCreateTensor(cTn(cLlmTensorAttnPostNorm, "weight", pIl), { n_embd }, cMtpFlags);

    mCreateTensorQkv(layer, pIl, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, cMtpFlags);
    layer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", pIl), { n_embd_head_k * n_head, n_embd }, cMtpFlags);
    layer.vAttnQNorm = mCreateTensor(cTn(cLlmTensorAttnQNorm, "weight", pIl), { n_embd_head_k }, cMtpFlags);
    layer.vAttnKNorm = mCreateTensor(cTn(cLlmTensorAttnKNorm, "weight", pIl), { n_embd_head_k }, cMtpFlags);

    layer.vFfnGate = mCreateTensor(cTn(cLlmTensorFfnGate, "weight", pIl), { n_embd, n_ff }, cMtpFlags);
    layer.vFfnDown = mCreateTensor(cTn(cLlmTensorFfnDown, "weight", pIl), { n_ff, n_embd }, cMtpFlags);
    layer.vFfnUp   = mCreateTensor(cTn(cLlmTensorFfnUp, "weight", pIl), { n_embd, n_ff }, cMtpFlags);

    // NextN-specific tensors that define the MTP block.
    layer.vNextn.vEhProj = mCreateTensor(cTn(cLlmTensorNextnEhProj, "weight", pIl), { 2 * n_embd, n_embd }, cMtpFlags);
    layer.vNextn.vEnorm  = mCreateTensor(cTn(cLlmTensorNextnEnorm, "weight", pIl), { n_embd }, cMtpFlags);
    layer.vNextn.vHnorm  = mCreateTensor(cTn(cLlmTensorNextnHnorm, "weight", pIl), { n_embd }, cMtpFlags);
    layer.vNextn.vEmbedTokens    = mCreateTensor(cTn(cLlmTensorNextnEmbedTokens, "weight", pIl), { n_embd, n_vocab },
                                                 cMtpFlags | cTensorNotRequired);
    layer.vNextn.vSharedHeadHead = mCreateTensor(cTn(cLlmTensorNextnSharedHeadHead, "weight", pIl), { n_embd, n_vocab },
                                                 cMtpFlags | cTensorNotRequired);
    layer.vNextn.vSharedHeadNorm =
      mCreateTensor(cTn(cLlmTensorNextnSharedHeadNorm, "weight", pIl), { n_embd }, cMtpFlags | cTensorNotRequired);
  };

  for (int vI = 0; vI < n_layer; ++vI) {
    vLoadBlockTrunk(vI, cTrunkFlags);
  }
  for (int vI = n_layer; vI < n_layer_all; ++vI) {
    vLoadBlockMtp(vI);
  }
}

std::unique_ptr<llm_graph_context> dgs_model_qwen35::mBuildArchGraph(const llm_graph_params & pParams) const {
  if (pParams.vGtype == cLlmGraphTypeDecoderMtp) {
    return std::make_unique<graph_mtp>(*this, pParams);
  }
  return std::make_unique<graph>(*this, pParams);
}

dgs_model_qwen35::graph::graph(const dgs_model & pModel, const llm_graph_params & pParams) :
  llm_build_delta_net_base(pParams),
  vModel(pModel) {
  const int64_t cNEmbdHead = vHparams.n_embd_head_v();

  GGML_ASSERT(cNEmbdHead == vHparams.n_embd_head_k());

  int aSections[4];
  std::copy(std::begin(vHparams.aRopeSections), std::begin(vHparams.aRopeSections) + 4, aSections);

  ggml_tensor * vCur;
  ggml_tensor * vInpL;

  vInpL = mBuildInpEmbd(pModel.vTokEmbd);

  mCb(vInpL, "model.input_embed", -1);

  auto * vInp = mBuildInpMemHybrid();

  ggml_tensor * vInpPos    = mBuildInpPos();
  ggml_tensor * vInpOutIds = mBuildInpOutIds();

  // MTP/NextN layers are loaded as extra decoder blocks but not executed in the main pass.
  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    vRes->lTLayerInp[vIl] = vInpL;

    ggml_tensor * vInpSA = vInpL;

    vCur = mBuildNorm(vInpL, pModel.lLayers[vIl].vAttnNorm, nullptr, cLlmNormRms, vIl);
    mCb(vCur, "attn_norm", vIl);

    ggml_build_forward_expand(vGf, vCur);

    // Determine layer type and build appropriate attention mechanism
    if (vHparams.mIsRecr(vIl)) {
      // Linear attention layer (gated delta net)
      vCur = mBuildLayerAttnLinear(vInp->mGetRecr(), vCur, vIl);
    } else {
      // Full attention layer
      vCur = mBuildLayerAttn(vInp->mGetAttn(), vCur, vInpPos, aSections, vIl);
    }

    if (vIl == cNLayer - 1 && mCropBeforeNextn(vInpOutIds)) {
      vCur   = ggml_get_rows(vCtx0, vCur, vInpOutIds);
      vInpSA = ggml_get_rows(vCtx0, vInpSA, vInpOutIds);
    }

    // Residual connection
    vCur = ggml_add(vCtx0, vCur, vInpSA);
    mCb(vCur, "attn_residual", vIl);

    // Save the tensor before post-attention norm for residual connection
    ggml_tensor * vFfnResidual = vCur;

    // Post-attention norm
    ggml_tensor * vAttnPostNorm = mBuildNorm(vCur, pModel.lLayers[vIl].vAttnPostNorm, nullptr, cLlmNormRms, vIl);
    mCb(vAttnPostNorm, "attn_post_norm", vIl);

    // Dense FFN layer - without residual connection
    vCur = mBuildLayerFfn(vAttnPostNorm, vIl);
    mCb(vCur, "ffn_out", vIl);

    // Residual connection for FFN - add to the tensor from before post_attention_layernorm
    vCur = ggml_add(vCtx0, vCur, vFfnResidual);
    mCb(vCur, "post_ffn", vIl);

    vCur = mBuildCvec(vCur, vIl);
    mCb(vCur, "l_out", vIl);

    // Input for next layer
    vInpL = vCur;
  }
  vCur = vInpL;

  vCur = mBuildNorm(vCur, pModel.vOutputNorm, nullptr, cLlmNormRms, -1);

  mCb(vCur, "h_nextn", -1);
  vRes->vTHNextn = vCur;

  if (mCropAfterNextn(vInpOutIds)) {
    vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
  }

  mCb(vCur, "result_norm", -1);
  vRes->vTEmbd = vCur;

  if (pModel.vClsOut) {
    ggml_tensor * vEmbd = mBuildLoraMm(pModel.vClsOut, vCur);
    if (pModel.vClsOutB) {
      vEmbd = ggml_add(vCtx0, vEmbd, pModel.vClsOutB);
    }
    mCb(vEmbd, "result_embd_proj", -1);
    vRes->vTEmbd = vEmbd;
    ggml_build_forward_expand(vGf, vEmbd);
  }

  // LM head
  vCur = mBuildLoraMm(pModel.vOutput, vCur, pModel.vOutputS);

  mCb(vCur, "result_output", -1);
  vRes->vTLogits = vCur;

  ggml_build_forward_expand(vGf, vCur);
}

std::pair<ggml_tensor *, ggml_tensor *> dgs_model_qwen35::graph::mBuildQkvz(ggml_tensor * pInput, int pIl) {
  const int64_t cNSeqs      = vUbatch.vNSeqs;
  const int64_t cNSeqTokens = vUbatch.vNSeqTokens;

  ggml_tensor * vQkvMixed = mBuildLoraMm(vModel.lLayers[pIl].vWqkv, pInput, vModel.lLayers[pIl].vWqkvS);
  vQkvMixed               = ggml_reshape_3d(vCtx0, vQkvMixed, vQkvMixed->ne[0], cNSeqTokens, cNSeqs);
  mCb(vQkvMixed, "linear_attn_qkv_mixed", pIl);

  ggml_tensor * vZ = mBuildLoraMm(vModel.lLayers[pIl].vWqkvGate, pInput, vModel.lLayers[pIl].vWqkvGateS);
  mCb(vZ, "z", pIl);

  return { vQkvMixed, vZ };
}

ggml_tensor * dgs_model_qwen35::graph::mBuildNormGated(ggml_tensor * pInput,
                                                       ggml_tensor * pWeights,
                                                       ggml_tensor * pGate,
                                                       int           layer) {
  ggml_tensor * vNormalized = mBuildNorm(pInput, pWeights, nullptr, cLlmNormRms, layer);
  ggml_tensor * vGatedSilu  = ggml_silu(vCtx0, pGate);

  return ggml_mul(vCtx0, vNormalized, vGatedSilu);
}

ggml_tensor * dgs_model_qwen35::graph::mBuildLayerAttn(llm_graph_input_attn_kv * pInp,
                                                       ggml_tensor *             pCur,
                                                       ggml_tensor *             pInpPos,
                                                       int *                     pSections,
                                                       int                       pIl) {
  const int64_t cNEmbdHead = vHparams.n_embd_head_v();
  GGML_ASSERT(cNEmbdHead == vHparams.n_embd_head_k());

  // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

  // Qwen3Next uses a single Q projection that outputs query + gate
  auto [Qcur_full, Kcur, Vcur] = build_qkv(vModel.lLayers[pIl], pCur, cNEmbdHead * 2, cNHead, cNEmbdHead, cNHeadKv,
                                           cNEmbdHead, cNHeadKv, pIl, false);
  mCb(Qcur_full, "Qcur_full", pIl);
  mCb(Kcur, "Kcur", pIl);
  mCb(Vcur, "Vcur", pIl);

  ggml_tensor * vQcur =
    ggml_view_3d(vCtx0, Qcur_full, cNEmbdHead, cNHead, cNTokens, ggml_element_size(Qcur_full) * cNEmbdHead * 2,
                 ggml_element_size(Qcur_full) * cNEmbdHead * 2 * cNHead, 0);
  mCb(vQcur, "Qcur_reshaped", pIl);

  // Apply Q normalization
  vQcur = mBuildNorm(vQcur, vModel.lLayers[pIl].vAttnQNorm, nullptr, cLlmNormRms, pIl);
  mCb(vQcur, "Qcur_normed", pIl);

  // Apply K normalization
  Kcur = ggml_reshape_3d(vCtx0, Kcur, cNEmbdHead, cNHeadKv, cNTokens);
  Kcur = mBuildNorm(Kcur, vModel.lLayers[pIl].vAttnKNorm, nullptr, cLlmNormRms, pIl);
  mCb(Kcur, "Kcur_normed", pIl);

  ggml_tensor * vGate =
    ggml_view_3d(vCtx0, Qcur_full, cNEmbdHead, cNHead, cNTokens, ggml_element_size(Qcur_full) * cNEmbdHead * 2,
                 ggml_element_size(Qcur_full) * cNEmbdHead * 2 * cNHead, ggml_element_size(Qcur_full) * cNEmbdHead);
  vGate = ggml_cont_2d(vCtx0, vGate, cNEmbdHead * cNHead, cNTokens);
  mCb(vGate, "gate_reshaped", pIl);

  Vcur = ggml_reshape_3d(vCtx0, Vcur, cNEmbdHead, cNHeadKv, cNTokens);

  // Apply MRoPE
  vQcur = ggml_rope_multi(vCtx0, vQcur, pInpPos, nullptr, cNRot, pSections, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                          cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

  Kcur = ggml_rope_multi(vCtx0, Kcur, pInpPos, nullptr, cNRot, pSections, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                         cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

  mCb(vQcur, "Qcur", pIl);
  mCb(Kcur, "Kcur", pIl);
  mCb(Vcur, "Vcur", pIl);

  // Attention computation
  const float cKqScale =
    vHparams.vFAttentionScale == 0.0f ? 1.0f / sqrtf(float(cNEmbdHead)) : vHparams.vFAttentionScale;

  pCur = build_attn(pInp, nullptr, nullptr, nullptr, vQcur, Kcur, Vcur, nullptr, nullptr, nullptr, cKqScale, pIl);
  mCb(pCur, "attn_pregate", pIl);

  ggml_tensor * vGateSigmoid = ggml_sigmoid(vCtx0, vGate);
  mCb(vGateSigmoid, "gate_sigmoid", pIl);

  pCur = ggml_mul(vCtx0, pCur, vGateSigmoid);
  mCb(pCur, "attn_gated", pIl);

  pCur = mBuildLoraMm(vModel.lLayers[pIl].vWo, pCur, vModel.lLayers[pIl].vWoS);
  mCb(pCur, "attn_output", pIl);

  return pCur;
}

ggml_tensor * dgs_model_qwen35::graph::mBuildLayerAttnLinear(llm_graph_input_rs * pInp, ggml_tensor * pCur, int pIl) {
  const auto * vMctxCur = pInp->vMctx;

  const int64_t cDInner     = vHparams.vSsmDInner;
  const int64_t cNSeqs      = vUbatch.vNSeqs;
  const int64_t cHeadKDim   = vHparams.vSsmDState;
  const int64_t cNumKHeads  = vHparams.vSsmNGroup;
  const int64_t cNumVHeads  = vHparams.vSsmDtRank;
  const int64_t cHeadVDim   = cDInner / cNumVHeads;
  const int64_t cNSeqTokens = vUbatch.vNSeqTokens;

  GGML_ASSERT(cNSeqs != 0);
  GGML_ASSERT(vUbatch.mEqualSeqs());
  GGML_ASSERT(vUbatch.vNTokens == cNSeqTokens * cNSeqs);

  // Input projections
  auto          vQkvz     = mBuildQkvz(pCur, pIl);
  ggml_tensor * vQkvMixed = vQkvz.first;
  ggml_tensor * vZ        = vQkvz.second;

  ggml_tensor * vBeta = mBuildLoraMm(vModel.lLayers[pIl].vSsmBeta, pCur, vModel.lLayers[pIl].vSsmBetaS);
  vBeta               = ggml_reshape_4d(vCtx0, vBeta, 1, cNumVHeads, cNSeqTokens, cNSeqs);
  mCb(vBeta, "beta", pIl);

  vBeta = ggml_sigmoid(vCtx0, vBeta);
  mCb(vBeta, "beta_sigmoid", pIl);

  ggml_tensor * vAlpha = mBuildLoraMm(vModel.lLayers[pIl].vSsmAlpha, pCur, vModel.lLayers[pIl].vSsmAlphaS);
  vAlpha               = ggml_reshape_3d(vCtx0, vAlpha, cNumVHeads, cNSeqTokens, cNSeqs);
  mCb(vAlpha, "alpha", pIl);

  ggml_tensor * vAlphaBiased   = ggml_add(vCtx0, vAlpha, vModel.lLayers[pIl].vSsmDt);
  ggml_tensor * vAlphaSoftplus = ggml_softplus(vCtx0, vAlphaBiased);
  mCb(vAlphaSoftplus, "a_softplus", pIl);

  ggml_tensor * vGate = ggml_mul(vCtx0, vAlphaSoftplus, vModel.lLayers[pIl].vSsmA);  // -A_log.exp() * softplus
  mCb(vGate, "gate", pIl);

  vGate = ggml_reshape_4d(vCtx0, vGate, 1, cNumVHeads, cNSeqTokens, cNSeqs);

  ggml_tensor * vConvStatesAll = vMctxCur->mGetRL(pIl);
  ggml_tensor * vSsmStatesAll  = vMctxCur->mGetSL(pIl);

  ggml_tensor * vConvKernel     = vModel.lLayers[pIl].vSsmConv1d;
  const int64_t cConvKernelSize = vConvKernel->ne[0];
  const int64_t cConvChannels   = cDInner + 2 * vHparams.vSsmNGroup * vHparams.vSsmDState;

  ggml_tensor * vConvInput = mBuildConvState(pInp, vConvStatesAll, vQkvMixed, cConvKernelSize, cConvChannels, pIl);

  ggml_tensor * vState = build_rs(pInp, vSsmStatesAll, vHparams.mNEmbdS(), cNSeqs);
  vState               = ggml_reshape_4d(vCtx0, vState, cHeadVDim, cHeadVDim, cNumVHeads, cNSeqs);
  mCb(vState, "state_predelta", pIl);

  ggml_tensor * vConvOutputProper = ggml_ssm_conv(vCtx0, vConvInput, vConvKernel);
  mCb(vConvOutputProper, "conv_output_raw", pIl);

  ggml_tensor * vConvOutputSilu = ggml_silu(vCtx0, vConvOutputProper);
  mCb(vConvOutputSilu, "conv_output_silu", pIl);

  ggml_tensor * vConvQkvMix = vConvOutputSilu;

  // Calculate the total conv dimension
  int64_t vQkvDim = cHeadKDim * cNumKHeads * 2 + cHeadVDim * cNumVHeads;
  int64_t vNb1Qkv = ggml_row_size(vConvQkvMix->type, vQkvDim);

  // Extract the convolved Q, K, V from conv_output
  ggml_tensor * vQConv = ggml_view_4d(vCtx0, vConvQkvMix, cHeadKDim, cNumKHeads, cNSeqTokens, cNSeqs,
                                      ggml_row_size(vConvQkvMix->type, cHeadKDim), vNb1Qkv, vNb1Qkv * cNSeqTokens, 0);

  ggml_tensor * vKConv = ggml_view_4d(vCtx0, vConvQkvMix, cHeadKDim, cNumKHeads, cNSeqTokens, cNSeqs,
                                      ggml_row_size(vConvQkvMix->type, cHeadKDim), vNb1Qkv, vNb1Qkv * cNSeqTokens,
                                      cHeadKDim * cNumKHeads * ggml_element_size(vConvQkvMix));

  ggml_tensor * vVConv = ggml_view_4d(vCtx0, vConvQkvMix, cHeadVDim, cNumVHeads, cNSeqTokens, cNSeqs,
                                      ggml_row_size(vConvQkvMix->type, cHeadVDim), vNb1Qkv, vNb1Qkv * cNSeqTokens,
                                      ggml_row_size(vConvQkvMix->type, 2 * cHeadKDim * cNumKHeads));

  mCb(vQConv, "q_conv", pIl);
  mCb(vKConv, "k_conv", pIl);
  mCb(vVConv, "v_conv", pIl);

  const float cEpsNorm = vHparams.vFNormRmsEps;

  vQConv = fBuildGdnL2Norm(vCtx0, vQConv, cEpsNorm);
  vKConv = fBuildGdnL2Norm(vCtx0, vKConv, cEpsNorm);

  //q_conv = ggml_cont_4d(ctx0, q_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
  //k_conv = ggml_cont_4d(ctx0, k_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
  //v_conv = ggml_cont_4d(ctx0, v_conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

  // if head keys and value keys are different, repeat to force tensors into matching shapes
  // note: need explicit repeat only if we are not using the fused GDN.
  if (cNumKHeads != cNumVHeads && (!vCparams.vFusedGdnAr || !vCparams.vFusedGdnCh)) {
    GGML_ASSERT(cNumVHeads % cNumKHeads == 0);
    vQConv = ggml_repeat_4d(vCtx0, vQConv, cHeadKDim, cNumVHeads, cNSeqTokens, cNSeqs);
    vKConv = ggml_repeat_4d(vCtx0, vKConv, cHeadKDim, cNumVHeads, cNSeqTokens, cNSeqs);
  }

  mCb(vQConv, "q_conv_predelta", pIl);
  mCb(vKConv, "k_conv_predelta", pIl);
  mCb(vVConv, "v_conv_predelta", pIl);

  ggml_tensor * vOutput = mBuildRecurrentAttn(pInp, vSsmStatesAll, vQConv, vKConv, vVConv, vGate, vBeta, vState, pIl);

  // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
  ggml_tensor * vZ2d = ggml_reshape_4d(vCtx0, vZ, cHeadVDim, cNumVHeads, cNSeqTokens, cNSeqs);

  // Apply gated normalization: self.norm(core_attn_out, z)
  ggml_tensor * vAttnOutNorm = mBuildNormGated(vOutput, vModel.lLayers[pIl].vSsmNorm, vZ2d, pIl);

  // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_tokens, n_seqs, n_heads * head_dim]
  ggml_tensor * vFinalOutput = ggml_reshape_3d(vCtx0, vAttnOutNorm, cHeadVDim * cNumVHeads, cNSeqTokens, cNSeqs);
  mCb(vFinalOutput, "final_output", pIl);

  // Output projection
  pCur = mBuildLoraMm(vModel.lLayers[pIl].vSsmOut, vFinalOutput, vModel.lLayers[pIl].vSsmOutS);
  mCb(pCur, "linear_attn_out", pIl);

  // Reshape back to original dimensions
  pCur = ggml_reshape_2d(vCtx0, pCur, cNEmbd, cNSeqTokens * cNSeqs);

  return pCur;
}

ggml_tensor * dgs_model_qwen35::graph::mBuildLayerFfn(ggml_tensor * pCur, const int pIl) {
  GGML_ASSERT(vModel.lLayers[pIl].vFfnGateInp == nullptr);

  pCur = mBuildFfn(pCur, vModel.lLayers[pIl].vFfnUp, NULL, vModel.lLayers[pIl].vFfnUpS, vModel.lLayers[pIl].vFfnGate,
                   NULL, vModel.lLayers[pIl].vFfnGateS, vModel.lLayers[pIl].vFfnDown, NULL,
                   vModel.lLayers[pIl].vFfnDownS, NULL, cLlmFfnSilu, cLlmFfnPar, pIl);
  mCb(pCur, "ffn_out", pIl);

  return pCur;
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for Qwen3.5/3.6 dense series
dgs_model_qwen35::graph_mtp::graph_mtp(const dgs_model & pModel, const llm_graph_params & pParams) :
  llm_graph_context(pParams) {
  GGML_ASSERT(vHparams.n_layer_nextn > 0 && "QWEN35 MTP requires n_layer_nextn > 0");
  GGML_ASSERT(vHparams.n_layer_nextn == 1 && "QWEN35 MTP currently only supports a single MTP block");

  const int64_t cNEmbdHead = vHparams.n_embd_head_v();
  GGML_ASSERT(cNEmbdHead == vHparams.n_embd_head_k());

  // hparams.n_layer includes both main model layers and MTP layers. The MTP
  // layer is stored immediately after the main layers in model.layers[].
  const int    cIl   = vHparams.n_layer();
  const auto & layer = pModel.lLayers[cIl];

  GGML_ASSERT(layer.vNextn.vEhProj && "MTP block missing nextn.eh_proj");
  GGML_ASSERT(layer.vNextn.vEnorm && "MTP block missing nextn.enorm");
  GGML_ASSERT(layer.vNextn.vHnorm && "MTP block missing nextn.hnorm");

  int aSections[4];
  std::copy(std::begin(vHparams.aRopeSections), std::begin(vHparams.aRopeSections) + 4, aSections);

  // TODO: extract in a common llm_graph_context::build_inp_embd_h()
  auto vInp = std::make_unique<llm_graph_input_embd_h>(vHparams.mNEmbdInp(), vHparams.n_embd);

  vInp->vTokens = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNTokens);
  ggml_set_input(vInp->vTokens);

  vInp->vEmbd = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, vHparams.mNEmbdInp(), cNTokens);
  ggml_set_input(vInp->vEmbd);

  // TODO: make static using `ggml_build_forward_select()`
  //       see llm_graph_context::build_inp_embd() for reference
  ggml_tensor * vTokEmbd;
  ASSERT_EMBD_OR_TOKEN(vUbatch);
  if (vUbatch.token) {
    ggml_tensor * vTokEmbdW = layer.vNextn.vEmbedTokens ? layer.vNextn.vEmbedTokens : pModel.vTokEmbd;

    vTokEmbd = ggml_get_rows(vCtx0, vTokEmbdW, vInp->vTokens);

    // a Hadamard-latent table stores rotated rows; restore the primal basis
    if (vHdmd) {
      if (const auto cIt = vHdmd->dInv.find(vTokEmbdW); cIt != vHdmd->dInv.end()) {
        vTokEmbd = fDgsMulMatHadamard(vCtx0, vTokEmbd, cIt->second.vRot);
        if (cIt->second.vSigns) {
          vTokEmbd = ggml_mul(vCtx0, vTokEmbd, cIt->second.vSigns);
        }
      }
    }
  } else {
    vTokEmbd = vInp->vEmbd;
  }
  mCb(vTokEmbd, "mtp_tok_embd", cIl);

  vInp->vH = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, vHparams.n_embd, cNTokens);
  ggml_set_input(vInp->vH);
  ggml_set_name(vInp->vH, "mtp_h_input");

  ggml_tensor * vHEmbd = vInp->vH;

  vRes->mAddInput(std::move(vInp));

  ggml_tensor * vInpPos    = mBuildInpPos();
  ggml_tensor * vInpOutIds = mBuildInpOutIds();

  auto * vInpAttn = mBuildAttnInpKv();

  ggml_tensor * vHNorm = mBuildNorm(vHEmbd, layer.vNextn.vHnorm, nullptr, cLlmNormRms, cIl);
  mCb(vHNorm, "mtp_hnorm", cIl);

  ggml_tensor * vENorm = mBuildNorm(vTokEmbd, layer.vNextn.vEnorm, nullptr, cLlmNormRms, cIl);
  mCb(vENorm, "mtp_enorm", cIl);

  ggml_tensor * vConcat = ggml_concat(vCtx0, vENorm, vHNorm, /*dim=*/0);
  mCb(vConcat, "mtp_concat", cIl);

  ggml_tensor * vCur = mBuildLoraMm(layer.vNextn.vEhProj, vConcat, layer.vNextn.vEhProjS);
  mCb(vCur, "mtp_eh_proj", cIl);

  ggml_tensor * vInpSA = vCur;

  vCur = mBuildNorm(vCur, layer.vAttnNorm, nullptr, cLlmNormRms, cIl);
  mCb(vCur, "mtp_attn_norm", cIl);

  auto [Qcur_full, Kcur, Vcur] =
    build_qkv(layer, vCur, cNEmbdHead * 2, cNHead, cNEmbdHead, cNHeadKv, cNEmbdHead, cNHeadKv, cIl, false);
  mCb(Qcur_full, "mtp_Qcur_full", cIl);

  ggml_tensor * vQcur =
    ggml_view_3d(vCtx0, Qcur_full, cNEmbdHead, cNHead, cNTokens, ggml_element_size(Qcur_full) * cNEmbdHead * 2,
                 ggml_element_size(Qcur_full) * cNEmbdHead * 2 * cNHead, 0);
  vQcur = mBuildNorm(vQcur, layer.vAttnQNorm, nullptr, cLlmNormRms, cIl);
  mCb(vQcur, "mtp_Qcur_normed", cIl);

  ggml_tensor * vGate =
    ggml_view_3d(vCtx0, Qcur_full, cNEmbdHead, cNHead, cNTokens, ggml_element_size(Qcur_full) * cNEmbdHead * 2,
                 ggml_element_size(Qcur_full) * cNEmbdHead * 2 * cNHead, ggml_element_size(Qcur_full) * cNEmbdHead);
  vGate = ggml_cont_2d(vCtx0, vGate, cNEmbdHead * cNHead, cNTokens);
  mCb(vGate, "mtp_gate", cIl);

  Kcur = ggml_reshape_3d(vCtx0, Kcur, cNEmbdHead, cNHeadKv, cNTokens);
  Kcur = mBuildNorm(Kcur, layer.vAttnKNorm, nullptr, cLlmNormRms, cIl);
  mCb(Kcur, "mtp_Kcur_normed", cIl);

  Vcur = ggml_reshape_3d(vCtx0, Vcur, cNEmbdHead, cNHeadKv, cNTokens);
  mCb(Vcur, "mtp_Vcur", cIl);

  vQcur = ggml_rope_multi(vCtx0, vQcur, vInpPos, nullptr, cNRot, aSections, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                          cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
  Kcur  = ggml_rope_multi(vCtx0, Kcur, vInpPos, nullptr, cNRot, aSections, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                          cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);

  const float cKqScale =
    vHparams.vFAttentionScale == 0.0f ? 1.0f / sqrtf(float(cNEmbdHead)) : vHparams.vFAttentionScale;

  vCur = build_attn(vInpAttn, nullptr, nullptr, nullptr, vQcur, Kcur, Vcur, nullptr, nullptr, nullptr, cKqScale, cIl);
  mCb(vCur, "mtp_attn_pregate", cIl);

  vCur = ggml_mul(vCtx0, vCur, ggml_sigmoid(vCtx0, vGate));
  vCur = mBuildLoraMm(layer.vWo, vCur, layer.vWoS);
  mCb(vCur, "mtp_attn_out", cIl);

  vCur = ggml_add(vCtx0, vCur, vInpSA);
  mCb(vCur, "mtp_attn_residual", cIl);

  ggml_tensor * vFfnResidual = vCur;
  vCur                       = mBuildNorm(vCur, layer.vAttnPostNorm, nullptr, cLlmNormRms, cIl);
  mCb(vCur, "mtp_attn_post_norm", cIl);

  vCur = mBuildFfn(vCur, layer.vFfnUp, nullptr, layer.vFfnUpS, layer.vFfnGate, nullptr, layer.vFfnGateS, layer.vFfnDown,
                   nullptr, layer.vFfnDownS, nullptr, cLlmFfnSilu, cLlmFfnPar, cIl);
  mCb(vCur, "mtp_ffn_out", cIl);

  vCur = ggml_add(vCtx0, vCur, vFfnResidual);
  mCb(vCur, "mtp_post_ffn", cIl);

  ggml_tensor * vHeadNormW = layer.vNextn.vSharedHeadNorm ? layer.vNextn.vSharedHeadNorm : pModel.vOutputNorm;
  GGML_ASSERT(vHeadNormW && "QWEN35 MTP: missing both nextn.shared_head_norm and output_norm");
  vCur = mBuildNorm(vCur, vHeadNormW, nullptr, cLlmNormRms, -1);

  mCb(vCur, "h_nextn", -1);
  vRes->vTHNextn = vCur;

  vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
  mCb(vCur, "mtp_shared_head_norm", -1);

  ggml_tensor * vHeadW = layer.vNextn.vSharedHeadHead ? layer.vNextn.vSharedHeadHead : pModel.vOutput;
  ggml_tensor * vHeadS = layer.vNextn.vSharedHeadHead ? layer.vNextn.vSharedHeadHeadS : pModel.vOutputS;
  GGML_ASSERT(vHeadW && "QWEN35 MTP: missing LM head (nextn.shared_head_head or model.output)");
  vCur = mBuildLoraMm(vHeadW, vCur, vHeadS);
  mCb(vCur, "result_output", -1);

  vRes->vTLogits = vCur;
  ggml_build_forward_expand(vGf, vCur);
}
