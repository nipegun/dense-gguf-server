#pragma once

#include "dgs-graph.h"
#include "dgs-model-loader.h"
#include "dgs-model.h"

#include <algorithm>

// note: almost all graphs require at least sqrtf, so include cmath globally
#include <cmath>
#include <map>

class dgs_memory_hybrid_idx_context;

static inline ggml_tensor * fBuildGdnL2Norm(ggml_context * pCtx, ggml_tensor * pX, float pEps) {
  const float n = pX->ne[0];

  return ggml_scale(pCtx, ggml_rms_norm(pCtx, pX, pEps / n), 1.0f / sqrtf(n));
}

//
// base classes
//

struct llm_build_delta_net_base : public llm_graph_context {
  llm_build_delta_net_base(const llm_graph_params & pParams);

  virtual ~llm_build_delta_net_base() = default;

  // returns pair of output and new state
  std::pair<ggml_tensor *, ggml_tensor *> mBuildDeltaNetChunking(ggml_tensor * pQ,
                                                                 ggml_tensor * pK,
                                                                 ggml_tensor * pV,
                                                                 ggml_tensor * pG,
                                                                 ggml_tensor * pB,
                                                                 ggml_tensor * pS,
                                                                 int           pIl);

  // returns pair of output and new state
  std::pair<ggml_tensor *, ggml_tensor *> mBuildDeltaNetAutoregressive(ggml_tensor * pQ,
                                                                       ggml_tensor * pK,
                                                                       ggml_tensor * pV,
                                                                       ggml_tensor * pG,
                                                                       ggml_tensor * pB,
                                                                       ggml_tensor * pS,
                                                                       int           pIl);

  // use the ggml_gated_delta_net fused operator (K=1; state has shape [S_v, S_v, H_v, n_seqs])
  std::pair<ggml_tensor *, ggml_tensor *> mBuildDeltaNetFused(ggml_tensor * pQ,
                                                              ggml_tensor * pK,
                                                              ggml_tensor * pV,
                                                              ggml_tensor * pG,
                                                              ggml_tensor * pB,
                                                              ggml_tensor * pS,
                                                              int           pIl);

  // choose one of two implementations above based on the number of tokens
  std::pair<ggml_tensor *, ggml_tensor *> mBuildDeltaNet(ggml_tensor * pQ,
                                                         ggml_tensor * pK,
                                                         ggml_tensor * pV,
                                                         ggml_tensor * pG,
                                                         ggml_tensor * pB,
                                                         ggml_tensor * pS,
                                                         int           pIl);

  // read conv state from cache, concat with qkv_mixed, write back (single slot or per-token)
  // qkv_mixed: (qkv_dim, n_seq_tokens, n_seqs); returns conv_input: (kernel_size + n_seq_tokens - 1, channels, n_seqs)
  ggml_tensor * mBuildConvState(llm_graph_input_rs * pInp,
                                ggml_tensor *        pConvStatesAll,
                                ggml_tensor *        pQkvMixed,
                                int64_t              pConvKernelSize,
                                int64_t              pConvChannels,
                                int                  pIl);

  // run delta-net attention and write the new recurrent state(s) back to ssm_states_all
  // s: (head_v_dim, head_v_dim, num_v_heads, n_seqs); returns output: (head_v_dim, num_v_heads, n_seq_tokens, n_seqs)
  ggml_tensor * mBuildRecurrentAttn(llm_graph_input_rs * pInp,
                                    ggml_tensor *        pSsmStatesAll,
                                    ggml_tensor *        pQ,
                                    ggml_tensor *        pK,
                                    ggml_tensor *        pV,
                                    ggml_tensor *        pG,
                                    ggml_tensor *        pB,
                                    ggml_tensor *        pS,
                                    int                  pIl);
};

// Base class for RWKV7-related models

//
// models
//

// Quant-only stub for mmproj GGUFs
// none of these are ever called, they only exist to satisfy the dgs_model_base interface

// Qwen3 with the decision head of vllm-sr Decision 2.0: one score per option

struct dgs_model_qwen35 : public dgs_model_base {
  dgs_model_qwen35(const struct dgs_model_params & pParams) : dgs_model_base(pParams) {}

  void mLoadArchHparams(dgs_model_loader & pMl) override;
  void mLoadArchTensors(dgs_model_loader & pMl) override;

  struct graph : public llm_build_delta_net_base {
    graph(const dgs_model & pModel, const llm_graph_params & pParams);
  private:
    ggml_tensor * mBuildLayerAttn(llm_graph_input_attn_kv * pInpAttn,
                                  ggml_tensor *             pCur,
                                  ggml_tensor *             pInpPos,
                                  int *                     pSections,
                                  int                       pIl);

    ggml_tensor * mBuildLayerAttnLinear(llm_graph_input_rs * pInp, ggml_tensor * pCur, int pIl);

    ggml_tensor * mBuildLayerFfn(ggml_tensor * pCur, int pIl);

    ggml_tensor * mBuildNormGated(ggml_tensor * pInput, ggml_tensor * pWeights, ggml_tensor * pGate, int layer);

    // returns pair of qkv, z
    std::pair<ggml_tensor *, ggml_tensor *> mBuildQkvz(ggml_tensor * pInput, int pIl);

    const dgs_model & vModel;
  };

  struct graph_mtp : public llm_graph_context {
    graph_mtp(const dgs_model & pModel, const llm_graph_params & pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params & pParams) const override;
};

// Qwen3.5 backbone evaluated without memory, with a joint decision head on top of its hidden states
