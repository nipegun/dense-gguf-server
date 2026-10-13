#pragma once

#include "../clip-graph.h"

struct clip_graph_qwen_vision_base : clip_graph {
  clip_graph_qwen_vision_base(clip_ctx * pCtx, const clip_image_f32 & pImg) : clip_graph(pCtx, pImg) {}

  ggml_tensor * mBuildInpWithTemporalMerge();
};

struct clip_graph_qwen3vl : clip_graph_qwen_vision_base {
  clip_graph_qwen3vl(clip_ctx * pCtx, const clip_image_f32 & pImg) : clip_graph_qwen_vision_base(pCtx, pImg) {}

  ggml_cgraph * mBuild() override;
};
