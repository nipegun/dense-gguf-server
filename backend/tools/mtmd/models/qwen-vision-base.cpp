#include "models.h"

ggml_tensor * clip_graph_qwen_vision_base::mBuildInpWithTemporalMerge() {
  ggml_tensor * vInpRaw = mBuildInpRaw();

  GGML_ASSERT(vImg.mNx() % (cPatchSize * 2) == 0);
  GGML_ASSERT(vImg.mNy() % (cPatchSize * 2) == 0);

  const size_t cNb1 = ggml_row_size(vInpRaw->type, vImg.mNx());
  const size_t cNb2 = ggml_row_size(vInpRaw->type, vImg.mNx() * vImg.mNy());

  if (vNBatch == 1) {
    // still image input
    return ggml_add(vCtx0, ggml_conv_2d(vCtx0, vModel.vPatchEmbeddings0, vInpRaw, cPatchSize, cPatchSize, 0, 0, 1, 1),
                    ggml_conv_2d(vCtx0, vModel.vPatchEmbeddings1, vInpRaw, cPatchSize, cPatchSize, 0, 0, 1, 1));
  } else if (vNBatch == 2) {
    // 2 frames input (video input)
    ggml_tensor * vInp0 = ggml_view_3d(vCtx0, vInpRaw, vImg.mNx(), vImg.mNy(), 3, cNb1, cNb2, 0);
    ggml_tensor * vInp1 = ggml_view_3d(vCtx0, vInpRaw, vImg.mNx(), vImg.mNy(), 3, cNb1, cNb2,
                                       cNb2 * 3);  // move to the second frame
    return ggml_add(vCtx0, ggml_conv_2d(vCtx0, vModel.vPatchEmbeddings0, vInp0, cPatchSize, cPatchSize, 0, 0, 1, 1),
                    ggml_conv_2d(vCtx0, vModel.vPatchEmbeddings1, vInp1, cPatchSize, cPatchSize, 0, 0, 1, 1));
  } else {
    GGML_ASSERT(false && "n_batch > 2 is not supported");
  }
}
