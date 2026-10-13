#pragma once

#include <cstdint>

struct dgs_model_loader;
struct dgs_hparams;

struct dgs_model_profile {
  const char * vName;
  uint32_t     vLayers;
  uint32_t     vEmbedding;
  uint32_t     vFeedForward;
  uint32_t     vHeads;
  uint32_t     vKvHeads;
};

const dgs_model_profile & fDgsRequireModelProfile(const dgs_model_loader & pLoader);
void                      fDgsValidateModelProfile(const dgs_model_loader & pLoader, const dgs_hparams & pHparams);
