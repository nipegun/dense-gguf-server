#include "clip.h"

#include "clip-graph.h"
#include "clip-impl.h"
#include "clip-model.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "gguf.h"
#include "models/models.h"

#include <float.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

struct clip_logger_state vGLoggerState = { fClipLogCallbackDefault, NULL };

//#define CLIP_DEBUG_FUNCTIONS

#ifdef CLIP_DEBUG_FUNCTIONS
static void clip_image_write_image_to_ppm(const clip_image_u8 & img, const std::string & filename) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    LOG_ERR("Failed to open file for writing: %s\n", filename.c_str());
    return;
  }

  // PPM header: P6 format, width, height, and max color value
  const auto ppm_size = img.get_size();
  file << "P6\n" << ppm_size.width << " " << ppm_size.height << "\n255\n";

  // Write pixel data
  const auto & ppm_buf = img.get_ro_buf();
  for (size_t i = 0; i < ppm_buf.size(); i += 3) {
    // PPM expects binary data in RGB format, which matches our image buffer
    file.write(reinterpret_cast<const char *>(&ppm_buf[i]), 3);
  }

  file.close();
}

static void clip_image_save_to_bmp(const clip_image_u8 & img, const std::string & filename) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    LOG_ERR("Failed to open file for writing: %s\n", filename.c_str());
    return;
  }

  const auto bmp_size      = img.get_size();
  int        fileSize      = 54 + 3 * bmp_size.width * bmp_size.height;  // File header + info header + pixel data
  int        bytesPerPixel = 3;
  int        widthInBytes  = bmp_size.width * bytesPerPixel;
  int        paddingAmount = (4 - (widthInBytes % 4)) % 4;
  int        stride        = widthInBytes + paddingAmount;

  // Bitmap file header
  unsigned char fileHeader[14] = {
    'B', 'M',        // Signature
    0,   0,   0, 0,  // Image file size in bytes
    0,   0,   0, 0,  // Reserved
    54,  0,   0, 0   // Start of pixel array
  };

  // Total file size
  fileSize      = 54 + (stride * bmp_size.height);
  fileHeader[2] = (unsigned char) (fileSize);
  fileHeader[3] = (unsigned char) (fileSize >> 8);
  fileHeader[4] = (unsigned char) (fileSize >> 16);
  fileHeader[5] = (unsigned char) (fileSize >> 24);

  // Bitmap information header (BITMAPINFOHEADER)
  unsigned char infoHeader[40] = {
    40, 0, 0, 0,  // Size of this header (40 bytes)
    0,  0, 0, 0,  // Image width
    0,  0, 0, 0,  // Image height
    1,  0,        // Number of color planes
    24, 0,        // Bits per pixel
    0,  0, 0, 0,  // No compression
    0,  0, 0, 0,  // Image size (can be 0 for no compression)
    0,  0, 0, 0,  // X pixels per meter (not specified)
    0,  0, 0, 0,  // Y pixels per meter (not specified)
    0,  0, 0, 0,  // Total colors (color table not used)
    0,  0, 0, 0   // Important colors (all are important)
  };

  // Width and height in the information header
  infoHeader[4]  = (unsigned char) (bmp_size.width);
  infoHeader[5]  = (unsigned char) (bmp_size.width >> 8);
  infoHeader[6]  = (unsigned char) (bmp_size.width >> 16);
  infoHeader[7]  = (unsigned char) (bmp_size.width >> 24);
  infoHeader[8]  = (unsigned char) (bmp_size.height);
  infoHeader[9]  = (unsigned char) (bmp_size.height >> 8);
  infoHeader[10] = (unsigned char) (bmp_size.height >> 16);
  infoHeader[11] = (unsigned char) (bmp_size.height >> 24);

  // Write file headers
  file.write(reinterpret_cast<char *>(fileHeader), sizeof(fileHeader));
  file.write(reinterpret_cast<char *>(infoHeader), sizeof(infoHeader));

  // Pixel data
  std::vector<unsigned char> padding(3, 0);         // Max padding size to be added to each row
  for (int y = bmp_size.height - 1; y >= 0; --y) {  // BMP files are stored bottom-to-top
    for (int x = 0; x < bmp_size.width; ++x) {
      // Each pixel
      const auto    px       = img.get_pixel(x, y);
      unsigned char pixel[3] = { px[2],  // BMP stores pixels in BGR format
                                 px[1], px[0] };
      file.write(reinterpret_cast<char *>(pixel), 3);
    }
    // Write padding for the row
    file.write(reinterpret_cast<char *>(padding.data()), paddingAmount);
  }

  file.close();
}

// debug function to convert f32 to u8
static void clip_image_convert_f32_to_u8(const clip_image_f32 & src, clip_image_u8 & dst) {
  dst.set_size(src.get_size(), false);
  const auto &         src_buf = src.get_ro_buf();
  std::vector<uint8_t> dst_buf(src.n_elements());
  for (size_t i = 0; i < src.n_elements(); ++i) {
    dst_buf[i] = static_cast<uint8_t>(std::min(std::max(int(src_buf[i] * 255.0f), 0), 255));
  }
  dst.cpy_buf(dst_buf);
}
#endif

struct clip_ctx {
  clip_model vModel;

  gguf_context_ptr vCtxGguf;
  ggml_context_ptr vCtxData;

  std::vector<uint8_t> lBufComputeMeta;

  std::vector<ggml_backend_t>             lBackendPtrs;
  std::vector<ggml_backend_buffer_type_t> lBackendBuft;

  ggml_backend_t          vBackend    = nullptr;
  ggml_backend_t          vBackendCpu = nullptr;
  ggml_backend_buffer_ptr vBuf;

  int                    vMaxNodes = 8192;
  ggml_backend_sched_ptr vSched;
  clip_flash_attn_type   vFlashAttnType = cClipFlashAttnTypeAuto;
  bool                   vIsAllocated   = false;

  bool vDebugOutputEmbeddings = false;

  // for measuring memory usage
  bool                                 vNoAlloc = false;
  std::map<ggml_backend_dev_t, size_t> dMemUsage;
  std::map<ggml_backend_dev_t, size_t> dMemCompute;

  bool vSupportBatch = false;

  // for audio gen, reseeded only when the caller asks for another seed
  std::mt19937 vRng{ std::random_device{}() };
  uint32_t     vRngSeed = UINT32_MAX;

  clip_ctx(clip_context_params & pCtxParams) {
    vFlashAttnType = pCtxParams.vFlashAttnType;
    vNoAlloc       = pCtxParams.vNoAlloc;
    vBackendCpu    = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!vBackendCpu) {
      throw std::runtime_error("failed to initialize CPU backend");
    }
    if (pCtxParams.vUseGpu) {
      if (pCtxParams.vDevice != nullptr) {
        vBackend = ggml_backend_dev_init(pCtxParams.vDevice, nullptr);
        if (!vBackend) {
          throw std::runtime_error(fStringFormat("%s: failed to initialize \"%s\" backend\n", __func__,
                                                 ggml_backend_dev_name(pCtxParams.vDevice)));
        }
      } else {
        vBackend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        vBackend = vBackend ? vBackend : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
      }
    }

    if (vBackend) {
      LOG_INF("%s: CLIP using %s backend\n", __func__, ggml_backend_name(vBackend));
      lBackendPtrs.push_back(vBackend);
      lBackendBuft.push_back(ggml_backend_get_default_buffer_type(vBackend));
    } else {
      vBackend = vBackendCpu;
      LOG_INF("%s: CLIP using CPU backend\n", __func__);
    }

    if (pCtxParams.vImageMinTokens > 0) {
      vModel.vHparams.vCustomImageMinTokens = pCtxParams.vImageMinTokens;
    }
    if (pCtxParams.vImageMaxTokens > 0) {
      vModel.vHparams.vCustomImageMaxTokens = pCtxParams.vImageMaxTokens;
    }

    lBackendPtrs.push_back(vBackendCpu);
    lBackendBuft.push_back(ggml_backend_get_default_buffer_type(vBackendCpu));

    vSched.reset(
      ggml_backend_sched_new(lBackendPtrs.data(), lBackendBuft.data(), lBackendPtrs.size(), 8192, false, true));

    if (pCtxParams.vCbEval != nullptr) {
      ggml_backend_sched_set_eval_callback(vSched.get(), pCtxParams.vCbEval, pCtxParams.vCbEvalUserData);
    }

    vDebugOutputEmbeddings = std::getenv("MTMD_DEBUG_EMBEDDINGS") != nullptr;
  }

  ~clip_ctx() {
    ggml_backend_free(vBackend);
    if (vBackend != vBackendCpu) {
      ggml_backend_free(vBackendCpu);
    }
  }

  // this function is added so that we don't change too much of the existing code
  projector_type mProjType() const { return vModel.vProjType; }
};

//
// clip_graph
//

clip_graph::clip_graph(clip_ctx * pCtx, const clip_image_f32 & pImg) :
  vModel(pCtx->vModel),
  vHparams(vModel.vHparams),
  vProjType(pCtx->mProjType()),
  vImg(pImg),
  cPatchSize(vHparams.vPatchSize),
  cNPatchesX(pImg.mNx() / cPatchSize),
  cNPatchesY(pImg.mNy() / cPatchSize),
  cNPatches(cNPatchesX * cNPatchesY),
  cNEmbd(vHparams.vNEmbd),
  cNHead(vHparams.vNHead),
  cNHeadKv(vHparams.vNHeadKv),
  cDHead(vHparams.vNEmbdHead > 0 ? vHparams.vNEmbdHead : (cNHead > 0 ? cNEmbd / cNHead : 0)),
  cNLayer(vHparams.vNLayer),
  cNMmprojEmbd(fClipNMmprojEmbd(pCtx)),
  cEps(vHparams.vEps),
  vKqScale(cDHead > 0 ? 1.0f / sqrtf((float) cDHead) : 0.0f),
  cFlashAttnType(pCtx->vFlashAttnType) {
  struct ggml_init_params vParams = {
    /*.mem_size   =*/pCtx->lBufComputeMeta.size(),
    /*.mem_buffer =*/pCtx->lBufComputeMeta.data(),
    /*.no_alloc   =*/true,
  };
  vCtx0Ptr.reset(ggml_init(vParams));
  vCtx0 = vCtx0Ptr.get();
  vGf   = ggml_new_graph_custom(vCtx0, pCtx->vMaxNodes, false);
}

clip_graph::clip_graph(const clip_graph & pParent) :
  vModel(pParent.vModel),
  vHparams(pParent.vHparams),
  vProjType(pParent.vProjType),
  vImg(pParent.vImg),
  cPatchSize(pParent.cPatchSize),
  cNPatchesX(pParent.cNPatchesX),
  cNPatchesY(pParent.cNPatchesY),
  cNPatches(pParent.cNPatches),
  cNEmbd(pParent.cNEmbd),
  cNHead(pParent.cNHead),
  cNHeadKv(pParent.cNHeadKv),
  cDHead(pParent.cDHead),
  cNLayer(pParent.cNLayer),
  cNMmprojEmbd(pParent.cNMmprojEmbd),
  cEps(pParent.cEps),
  vKqScale(pParent.vKqScale),
  cFlashAttnType(pParent.cFlashAttnType) {
  // reuse from parent
  vCtx0 = pParent.vCtx0;
  vGf   = pParent.vGf;
}

ggml_tensor * clip_graph::mBuildMm(ggml_tensor * pW, ggml_tensor * pX) const {
  return ggml_mul_mat(vCtx0, pW, pX);
}

void clip_graph::mCb(ggml_tensor * pCur, const char * name, int pIl) const {
  if (pIl >= 0) {
    ggml_format_name(pCur, "%s-%d", name, pIl);
  } else {
    ggml_set_name(pCur, name);
  }
}

// siglip2 naflex
ggml_tensor * clip_graph::mResizePositionEmbeddings(uint32_t pInterpolationMode) {
  ggml_tensor *  vPosEmbd  = vModel.vPositionEmbeddings;
  const int      cHeight   = vImg.mNy() / cPatchSize;
  const int      cWidth    = vImg.mNx() / cPatchSize;
  const uint32_t cMode     = pInterpolationMode;
  const int      cNPerSide = (int) std::sqrt(vPosEmbd->ne[1]);

  GGML_ASSERT(vPosEmbd);

  if (cHeight == cNPerSide && cWidth == cNPerSide) {
    return vPosEmbd;
  }

  vPosEmbd = ggml_reshape_3d(vCtx0, vPosEmbd, cNEmbd, cNPerSide, cNPerSide);  // -> (n_embd, n_per_side, n_per_side)
  vPosEmbd = ggml_permute(vCtx0, vPosEmbd, 2, 0, 1, 3);                       // -> (n_per_side, n_per_side, n_embd)
  vPosEmbd = ggml_interpolate(vCtx0, vPosEmbd, cWidth, cHeight, cNEmbd, 1, cMode);  // -> (width, height, n_embd)
  vPosEmbd = ggml_permute(vCtx0, vPosEmbd, 1, 2, 0, 3);                             // -> (n_embd, width, height)
  vPosEmbd = ggml_cont_2d(vCtx0, vPosEmbd, cNEmbd, cWidth * cHeight);               // -> (n_embd, width * height)

  return vPosEmbd;
}

// build vision transformer (ViT) cgraph
// this function should cover most of the models
// if your model has specific features, you should probably duplicate this function
ggml_tensor * clip_graph::mBuildVit(ggml_tensor *                                                   pInp,
                                    int64_t                                                         pNPos,
                                    norm_type                                                       pNormT,
                                    ffn_op_type                                                     pFfnT,
                                    ggml_tensor *                                                   pLearnedPosEmbd,
                                    std::function<ggml_tensor *(ggml_tensor *, const clip_layer &)> pAddPos,
                                    const build_vit_opts &                                          pOpts) {
  // batch dim: inp is [n_embd, n_pos, B]
  const int64_t cB = pInp->ne[2];

  if (pLearnedPosEmbd) {
    pInp = ggml_add(vCtx0, pInp, pLearnedPosEmbd);
    mCb(pInp, "pos_embed", -1);
  }

  // flatten batch; unflatten again in attention
  pInp = ggml_reshape_2d(vCtx0, pInp, cNEmbd, pNPos * cB);

  ggml_tensor * vInpL = pInp;

  // pre-layernorm
  if (vModel.vPreLnW) {
    vInpL = mBuildNorm(vInpL, vModel.vPreLnW, vModel.vPreLnB, pNormT, cEps, -1);
    mCb(vInpL, "pre_ln", -1);
  }

  // loop over layers
  for (int vIl = 0; vIl < cNLayer; vIl++) {
    auto &        layer = vModel.lLayers[vIl];
    ggml_tensor * vCur  = vInpL;  // inpL = residual, cur = hidden_states

    ggml_tensor * vAttnMask = pOpts.vAttnMask;
    if (pOpts.lAttnMaskLayers.size() > (size_t) vIl) {
      vAttnMask = pOpts.lAttnMaskLayers[vIl];
    }

    // layernorm1
    vCur = mBuildNorm(vCur, layer.vLn1W, layer.vLn1B, pNormT, cEps, vIl);
    mCb(vCur, "layer_inp_normed", vIl);

    // self-attention
    {
      ggml_tensor * vQcur = nullptr;
      ggml_tensor * vKcur = nullptr;
      ggml_tensor * vVcur = nullptr;
      if (layer.vQkvW != nullptr) {
        // fused qkv
        vCur = mBuildMm(layer.vQkvW, vCur);
        if (layer.vQkvB != nullptr) {
          vCur = ggml_add(vCtx0, vCur, layer.vQkvB);
        }

        // Q/K/V as [d_head, n_head, n_pos, B], the batch stride is cur->nb[1]*n_pos.
        vQcur = ggml_view_4d(vCtx0, vCur, cDHead, cNHead, pNPos, cB,
                             /* nb1    */ ggml_row_size(vCur->type, cDHead),
                             /* nb2    */ vCur->nb[1],
                             /* nb3    */ vCur->nb[1] * pNPos,
                             /* offset */ 0);

        vKcur = ggml_view_4d(vCtx0, vCur, cDHead, cNHead, pNPos, cB,
                             /* nb1    */ ggml_row_size(vCur->type, cDHead),
                             /* nb2    */ vCur->nb[1],
                             /* nb3    */ vCur->nb[1] * pNPos,
                             /* offset */ ggml_row_size(vCur->type, cNHead * cDHead));

        vVcur = ggml_view_4d(vCtx0, vCur, cDHead, cNHead, pNPos, cB,
                             /* nb1    */ ggml_row_size(vCur->type, cDHead),
                             /* nb2    */ vCur->nb[1],
                             /* nb3    */ vCur->nb[1] * pNPos,
                             /* offset */ ggml_row_size(vCur->type, 2 * cNHead * cDHead));

        if (layer.vQNorm) {
          GGML_ASSERT(layer.vQNorm->ne[0] == vQcur->ne[0]);
          vQcur = mBuildNorm(vQcur, layer.vQNorm, NULL, pNormT, cEps, vIl);
          mCb(vQcur, "Qcur_norm", vIl);
        }

        if (layer.vKNorm) {
          GGML_ASSERT(layer.vKNorm->ne[0] == vKcur->ne[0]);
          vKcur = mBuildNorm(vKcur, layer.vKNorm, NULL, pNormT, cEps, vIl);
          mCb(vKcur, "Kcur_norm", vIl);
        }

      } else {
        // separate q, k, v
        vQcur = mBuildMm(layer.vQW, vCur);
        if (layer.vQB) {
          vQcur = ggml_add(vCtx0, vQcur, layer.vQB);
        }

        vKcur = mBuildMm(layer.vKW, vCur);
        if (layer.vKB) {
          vKcur = ggml_add(vCtx0, vKcur, layer.vKB);
        }

        vVcur = mBuildMm(layer.vVW, vCur);
        if (layer.vVB) {
          vVcur = ggml_add(vCtx0, vVcur, layer.vVB);
        }

        // if true, norm must be applied after reshaping to (d_head, n_head, n_pos)
        bool vNormPerHead = layer.vQNorm && layer.vQNorm->ne[0] == cDHead;

        if (!vNormPerHead) {
          if (layer.vQNorm) {
            vQcur = mBuildNorm(vQcur, layer.vQNorm, NULL, pNormT, cEps, vIl);
            mCb(vQcur, "Qcur_norm", vIl);
          }
          if (layer.vKNorm) {
            vKcur = mBuildNorm(vKcur, layer.vKNorm, NULL, pNormT, cEps, vIl);
            mCb(vKcur, "Kcur_norm", vIl);
          }
        }

        vQcur = ggml_reshape_4d(vCtx0, vQcur, cDHead, cNHead, pNPos, cB);
        vKcur = ggml_reshape_4d(vCtx0, vKcur, cDHead, cNHeadKv, pNPos, cB);
        vVcur = ggml_reshape_4d(vCtx0, vVcur, cDHead, cNHeadKv, pNPos, cB);

        if (vNormPerHead) {
          if (layer.vQNorm) {
            vQcur = mBuildNorm(vQcur, layer.vQNorm, NULL, pNormT, cEps, vIl);
            mCb(vQcur, "Qcur_norm_per_head", vIl);
          }
          if (layer.vKNorm) {
            vKcur = mBuildNorm(vKcur, layer.vKNorm, NULL, pNormT, cEps, vIl);
            mCb(vKcur, "Kcur_norm_per_head", vIl);
          }
        }
      }

      mCb(vQcur, "Qcur", vIl);
      mCb(vKcur, "Kcur", vIl);
      mCb(vVcur, "Vcur", vIl);

      if (pAddPos) {
        vQcur = pAddPos(vQcur, layer);
        vKcur = pAddPos(vKcur, layer);
        mCb(vQcur, "Qcur_pos", vIl);
        mCb(vKcur, "Kcur_pos", vIl);
      }

      if (vProjType == cProjectorTypeGemma4v) {
        vVcur = ggml_rms_norm(vCtx0, vVcur, cEps);
        mCb(vVcur, "Vcur_normed", vIl);
      }

      // build_attn returns a flat 2D [n_embd, n_pos*B]
      vCur = build_attn(layer.vOW, layer.vOB, vQcur, vKcur, vVcur, vAttnMask, vKqScale, vIl);
      mCb(vCur, "attn_out", vIl);
    }

    if (layer.vLs1W) {
      vCur = ggml_mul(vCtx0, vCur, layer.vLs1W);
      mCb(vCur, "attn_out_scaled", vIl);
    }

    if (layer.vAttnPostNormW) {
      vCur = mBuildNorm(vCur, layer.vAttnPostNormW, nullptr, pNormT, cEps, vIl);
      mCb(vCur, "attn_post_normed", vIl);
    }

    // re-add the layer input, e.g., residual
    vCur = ggml_add(vCtx0, vCur, vInpL);

    vInpL = vCur;  // inpL = residual, cur = hidden_states

    if (pOpts.vCallbackLayerOut) {
      pOpts.vCallbackLayerOut(vCur, vIl);
    }

    mCb(vCur, "ffn_inp", vIl);

    // layernorm2 (pre-ffn norm)
    vCur = mBuildNorm(vCur, layer.vLn2W, layer.vLn2B, pNormT, cEps, vIl);
    mCb(vCur, "ffn_inp_normed", vIl);

    // ffn
    vCur = layer.vFfGateExpsW ? mBuildMoeFfn(vCur, layer, pFfnT, vIl) :
                                mBuildFfn(vCur, layer.vFfUpW, layer.vFfUpB, layer.vFfGateW, layer.vFfGateB,
                                          layer.vFfDownW, layer.vFfDownB, pFfnT, vIl);

    mCb(vCur, "ffn_out", vIl);

    if (layer.vFfPostNormW) {
      vCur = mBuildNorm(vCur, layer.vFfPostNormW, nullptr, pNormT, cEps, vIl);
      mCb(vCur, "ffn_post_normed", vIl);
    }

    if (layer.vLs2W) {
      vCur = ggml_mul(vCtx0, vCur, layer.vLs2W);
      mCb(vCur, "ffn_out_scaled", vIl);
    }

    // residual 2
    vCur = ggml_add(vCtx0, vInpL, vCur);
    mCb(vCur, "layer_out", vIl);

    if (layer.vLsOutW) {
      vCur = ggml_mul(vCtx0, vCur, layer.vLsOutW);
      mCb(vCur, "layer_out_scaled", vIl);
    }

    vInpL = vCur;
  }

  if (vModel.mAudioHasAvgpool()) {
    ggml_tensor * vCur = vInpL;
    vCur               = ggml_transpose(vCtx0, vCur);
    vCur               = ggml_cont(vCtx0, vCur);
    vCur               = ggml_pool_1d(vCtx0, vCur, GGML_OP_POOL_AVG, 2, 2, 0);
    vCur               = ggml_transpose(vCtx0, vCur);
    vCur               = ggml_cont(vCtx0, vCur);
    vInpL              = vCur;
  }

  // post-layernorm
  if (vModel.vPostLnW && !pOpts.vSkipPostLn) {
    vInpL = mBuildNorm(vInpL, vModel.vPostLnW, vModel.vPostLnB, pNormT, cEps, -1);
  }

  // restore the batch dim
  GGML_ASSERT(vInpL->ne[1] % cB == 0);
  vInpL = ggml_reshape_3d(vCtx0, vInpL, cNEmbd, vInpL->ne[1] / cB, cB);
  return vInpL;
}

// build the input after conv2d (inp_raw --> patches)
// returns tensor with shape [n_embd, n_patches]
ggml_tensor * clip_graph::mBuildInp() {
  ggml_tensor * vInpRaw = mBuildInpRaw();
  ggml_tensor * vInp    = ggml_conv_2d(vCtx0, vModel.vPatchEmbeddings0, vInpRaw, cPatchSize, cPatchSize, 0, 0, 1, 1);
  vInp                  = ggml_reshape_3d(vCtx0, vInp, cNPatches, cNEmbd, vNBatch);
  vInp                  = ggml_cont(vCtx0, ggml_transpose(vCtx0, vInp));
  if (vModel.vPatchBias) {
    vInp = ggml_add(vCtx0, vInp, vModel.vPatchBias);
    mCb(vInp, "patch_bias", -1);
  }
  return vInp;
}

ggml_tensor * clip_graph::mBuildInpRaw(int pChannels) {
  ggml_tensor * vInpRaw = ggml_new_tensor_4d(vCtx0, GGML_TYPE_F32, vImg.mNx(), vImg.mNy(), pChannels, vNBatch);
  ggml_set_name(vInpRaw, "inp_raw");
  ggml_set_input(vInpRaw);
  return vInpRaw;
}

static std::string fGetAttnMaskName(int pIdx) {
  return pIdx == 0 ? "attn_mask" : "attn_mask_" + std::to_string(pIdx);
}

ggml_tensor * clip_graph::mBuildInpAttnMask(int64_t pNKv, int64_t pNQ, int pIdx) {
  const ggml_type type  = cFlashAttnType == cClipFlashAttnTypeEnabled ? GGML_TYPE_F16 : GGML_TYPE_F32;
  ggml_tensor *   vMask = ggml_new_tensor_2d(vCtx0, type, pNKv, pNQ);
  ggml_set_name(vMask, fGetAttnMaskName(pIdx).c_str());
  ggml_set_input(vMask);
  return vMask;
}

ggml_tensor * clip_graph::mBuildNorm(ggml_tensor * pCur,
                                     ggml_tensor * pMw,
                                     ggml_tensor * pMb,
                                     norm_type     type,
                                     float         pNormEps,
                                     int           pIl) const {
  pCur = type == cNormTypeRms ? ggml_rms_norm(vCtx0, pCur, pNormEps) : ggml_norm(vCtx0, pCur, pNormEps);

  if (pMw) {
    pCur = ggml_mul(vCtx0, pCur, pMw);
    mCb(pCur, "norm_w", pIl);
  }

  if (pMb) {
    pCur = ggml_add(vCtx0, pCur, pMb);
    mCb(pCur, "norm_b", pIl);
  }

  return pCur;
}

ggml_tensor * clip_graph::mBuildFfn(ggml_tensor * pCur,
                                    ggml_tensor * pUp,
                                    ggml_tensor * pUpB,
                                    ggml_tensor * pGate,
                                    ggml_tensor * pGateB,
                                    ggml_tensor * pDown,
                                    ggml_tensor * pDownB,
                                    ffn_op_type   pTypeOp,
                                    int           pIl) const {
  ggml_tensor * vTmp = pUp ? mBuildMm(pUp, pCur) : pCur;
  mCb(vTmp, "ffn_up", pIl);

  if (pUpB) {
    vTmp = ggml_add(vCtx0, vTmp, pUpB);
    mCb(vTmp, "ffn_up_b", pIl);
  }

  if (pGate) {
    pCur = mBuildMm(pGate, pCur);
    mCb(pCur, "ffn_gate", pIl);

    if (pGateB) {
      pCur = ggml_add(vCtx0, pCur, pGateB);
      mCb(pCur, "ffn_gate_b", pIl);
    }
  } else {
    pCur = vTmp;
  }

  // we only support parallel ffn for now
  switch (pTypeOp) {
    case cFfnSilu:
      if (pGate) {
        pCur = vHparams.vSwigluClamp > 0.0f ? ggml_swiglu_clamp(vCtx0, pCur, vTmp, vHparams.vSwigluClamp) :
                                              ggml_swiglu_split(vCtx0, pCur, vTmp);
        mCb(pCur, "ffn_swiglu", pIl);
      } else {
        pCur = ggml_silu(vCtx0, pCur);
        mCb(pCur, "ffn_silu", pIl);
      }
      break;
    case cFfnGelu:
      if (pGate) {
        pCur = ggml_geglu_split(vCtx0, pCur, vTmp);
        mCb(pCur, "ffn_geglu", pIl);
      } else {
        pCur = ggml_gelu(vCtx0, pCur);
        mCb(pCur, "ffn_gelu", pIl);
      }
      break;
    case cFfnGeluErf:
      if (pGate) {
        pCur = ggml_geglu_erf_split(vCtx0, pCur, vTmp);
        mCb(pCur, "ffn_geglu_erf", pIl);
      } else {
        pCur = ggml_gelu_erf(vCtx0, pCur);
        mCb(pCur, "ffn_gelu_erf", pIl);
      }
      break;
    case cFfnGeluQuick:
      if (pGate) {
        pCur = ggml_geglu_quick_split(vCtx0, pCur, vTmp);
        mCb(pCur, "ffn_geglu_quick", pIl);
      } else {
        pCur = ggml_gelu_quick(vCtx0, pCur);
        mCb(pCur, "ffn_gelu_quick", pIl);
      }
      break;
    case cFfnReluSqr:
      {
        pCur = ggml_relu(vCtx0, pCur);
        pCur = ggml_sqr(vCtx0, pCur);
        mCb(pCur, "ffn_relu_sqr", pIl);
      }
      break;
  }

  if (pDown) {
    pCur = mBuildMm(pDown, pCur);
  }

  if (pDownB) {
    mCb(pCur, "ffn_down", pIl);
  }

  if (pDownB) {
    pCur = ggml_add(vCtx0, pCur, pDownB);
  }

  return pCur;
}

// MoE FFN with sigmoid router and normalized top-k weights (dots3note vision)
// the router runs in fp32; exp_probs_b only affects expert selection, not the weights
ggml_tensor * clip_graph::mBuildMoeFfn(ggml_tensor *      pCur,
                                       const clip_layer & layer,
                                       ffn_op_type        pTypeOp,
                                       int                pIl) const {
  const int64_t cNTokens     = pCur->ne[1];
  const int64_t cNExpert     = layer.vFfGateExpsW->ne[2];
  const int64_t cNExpertUsed = std::min((int64_t) vHparams.vNExpertUsed, cNExpert);
  GGML_ASSERT(cNExpertUsed > 0);
  GGML_ASSERT(pTypeOp == cFfnSilu);

  ggml_tensor * vProbs = ggml_sigmoid(vCtx0, mBuildMm(layer.vFfGateInpW, pCur));  // [n_expert, n_tokens]
  mCb(vProbs, "ffn_moe_probs", pIl);

  ggml_tensor * vSel      = layer.vFfExpProbsB ? ggml_add(vCtx0, vProbs, layer.vFfExpProbsB) : vProbs;
  ggml_tensor * vSelected = ggml_top_k(vCtx0, vSel, cNExpertUsed);  // [n_expert_used, n_tokens]

  ggml_tensor * vWeights = ggml_get_rows(vCtx0, ggml_reshape_3d(vCtx0, vProbs, 1, cNExpert, cNTokens), vSelected);
  vWeights               = ggml_reshape_2d(vCtx0, vWeights, cNExpertUsed, cNTokens);
  vWeights               = ggml_div(vCtx0, vWeights, ggml_sum_rows(vCtx0, vWeights));
  vWeights               = ggml_reshape_3d(vCtx0, vWeights, 1, cNExpertUsed, cNTokens);
  mCb(vWeights, "ffn_moe_weights", pIl);

  pCur                = ggml_reshape_3d(vCtx0, pCur, pCur->ne[0], 1, cNTokens);
  ggml_tensor * vGate = ggml_mul_mat_id(vCtx0, layer.vFfGateExpsW, pCur, vSelected);  // [n_ff, n_expert_used, n_tokens]
  ggml_tensor * vUp   = ggml_mul_mat_id(vCtx0, layer.vFfUpExpsW, pCur, vSelected);
  pCur                = ggml_mul(vCtx0, ggml_silu(vCtx0, vGate), vUp);
  pCur = ggml_mul_mat_id(vCtx0, layer.vFfDownExpsW, pCur, vSelected);  // [n_embd, n_expert_used, n_tokens]
  pCur = ggml_mul(vCtx0, pCur, vWeights);

  // sum over the selected experts
  ggml_tensor * vOut = nullptr;
  for (int64_t vI = 0; vI < cNExpertUsed; vI++) {
    ggml_tensor * vV = ggml_view_2d(vCtx0, pCur, pCur->ne[0], cNTokens, pCur->nb[2], vI * pCur->nb[1]);
    vOut             = vOut ? ggml_add(vCtx0, vOut, vV) : vV;
  }
  if (cNExpertUsed == 1) {
    vOut = ggml_cont(vCtx0, vOut);
  }
  mCb(vOut, "ffn_moe_out", pIl);
  return vOut;
}

ggml_tensor * clip_graph::build_attn(ggml_tensor * pWo,
                                     ggml_tensor * pWoB,
                                     ggml_tensor * pQCur,
                                     ggml_tensor * pKCur,
                                     ggml_tensor * pVCur,
                                     ggml_tensor * pKqMask,
                                     float         pKqScale,
                                     int           pIl,
                                     ggml_tensor * pSinks) const {
  // these nodes are added to the graph together so that they are not reordered
  // by doing so, the number of splits in the graph is reduced
  // the order is fixed without the compute flag, so an unselected branch stays out of the compute set
  ggml_build_forward_order(vGf, pQCur);
  ggml_build_forward_order(vGf, pKCur);
  ggml_build_forward_order(vGf, pVCur);

  ggml_tensor * vQ = ggml_permute(vCtx0, pQCur, 0, 2, 1, 3);
  //cb(q, "q", il);

  ggml_tensor * vK = ggml_permute(vCtx0, pKCur, 0, 2, 1, 3);
  //cb(k, "k", il);

  ggml_tensor * vCur;

  if (cFlashAttnType == cClipFlashAttnTypeEnabled) {
    ggml_tensor * vV = ggml_permute(vCtx0, pVCur, 0, 2, 1, 3);

    vK = ggml_cast(vCtx0, vK, GGML_TYPE_F16);
    vV = ggml_cast(vCtx0, vV, GGML_TYPE_F16);
    // mask must be f16 here, use build_inp_attn_mask()
    GGML_ASSERT(!pKqMask || pKqMask->type == GGML_TYPE_F16);

    vCur = ggml_flash_attn_ext(vCtx0, vQ, vK, vV, pKqMask, pKqScale, 0.0f, 0.0f);
    ggml_prec_set_acc(vCur, GGML_PREC_F32);
    if (pSinks != nullptr) {
      ggml_flash_attn_ext_add_sinks(vCur, pSinks);
    }

    vCur = ggml_reshape_2d(vCtx0, vCur, vCur->ne[0] * vCur->ne[1], vCur->ne[2] * vCur->ne[3]);

  } else {
    ggml_tensor * vV = ggml_permute(vCtx0, pVCur, 1, 2, 0, 3);
    vV               = ggml_cont(vCtx0, vV);

    ggml_tensor * vKq = ggml_mul_mat(vCtx0, vK, vQ);
    // F32 may not needed for vision encoders?
    // ggml_prec_set_acc(kq, GGML_PREC_F32);

    vKq = ggml_soft_max_ext(vCtx0, vKq, pKqMask, pKqScale, 0.0f);
    if (pSinks != nullptr) {
      ggml_soft_max_add_sinks(vKq, pSinks);
    }

    ggml_tensor * vKqv = ggml_mul_mat(vCtx0, vV, vKq);
    vCur               = ggml_permute(vCtx0, vKqv, 0, 2, 1, 3);
    vCur               = ggml_cont_2d(vCtx0, vCur, vCur->ne[0] * vCur->ne[1], vCur->ne[2] * vCur->ne[3]);
  }

  mCb(vCur, "kqv_out", pIl);

  if (pWo) {
    vCur = mBuildMm(pWo, vCur);
  }

  if (pWoB) {
    vCur = ggml_add(vCtx0, vCur, pWoB);
  }

  return vCur;
}

// implementation of the 2D RoPE without adding a new op in ggml
ggml_tensor * clip_graph::mBuildRope2d(ggml_context * pCtx0,
                                       ggml_tensor *  pCur,
                                       ggml_tensor *  pPosA,  // first half
                                       ggml_tensor *  pPosB,  // second half
                                       const float    pFreqBase,
                                       const bool     pInterleaveFreq) {
  const int64_t cNDim = pCur->ne[0];

  // for example, if we have cur tensor of shape (n_dim=8, n_head, n_pos)
  // we will have a list of 4 inv_freq: 1e-0, 1e-1, 1e-2, 1e-3
  // first half of cur will use 1e-0, 1e-2 (even)
  // second half of cur will use 1e-1, 1e-3 (odd)
  // the trick here is to rotate just half of n_dim, so inv_freq will automatically be even
  //  ^ don't ask me why, it's math! -2(2i) / n_dim == -2i / (n_dim/2)
  // then for the second half, we use freq_scale to shift the inv_freq
  //  ^ why? replace (2i) with (2i+1) in the above equation
  const float cFreqScaleOdd = pInterleaveFreq ? std::pow(pFreqBase, (float) -2 / cNDim) : 1.0;

  // first half, dims [0, n_dim/2)
  pCur = ggml_rope_ext(pCtx0, pCur,
                       pPosA,      // positions
                       nullptr,    // freq factors
                       cNDim / 2,  // n_dims
                       0, 0, pFreqBase, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

  // second half, dims [n_dim/2, n_dim)
  pCur = ggml_rope_ext(pCtx0, pCur,
                       pPosB,      // positions
                       nullptr,    // freq factors
                       cNDim / 2,  // n_dims
                       0, 0, pFreqBase, cFreqScaleOdd, 0.0f, 1.0f, 0.0f, 0.0f);
  pCur = ggml_rope_set_offset(pCur, cNDim / 2);

  return pCur;
}

// Generic function to stack frames for audio processing
// Abstracts out the StackAudioFrames logic used by ultravox
ggml_tensor * clip_graph::mBuildStack(ggml_tensor * pCur, int32_t pStackFactor, int32_t pNEmbed) {
  if (pStackFactor <= 1) {
    return pCur;
  }

  int64_t vTotalElements = ggml_nelements(pCur);
  int64_t vStride        = pNEmbed * pStackFactor;

  // Calculate padded length
  int64_t vPaddedLen = GGML_PAD(vTotalElements, vStride);
  int64_t vPad       = vPaddedLen - vTotalElements;

  if (vPad > 0) {
    // Pad the tensor to make it divisible by stride
    pCur = ggml_view_1d(vCtx0, pCur, vTotalElements, 0);
    pCur = ggml_pad(vCtx0, pCur, vPad, 0, 0, 0);
  }

  // Reshape to [stride, padded_len / stride]
  pCur = ggml_view_2d(vCtx0, pCur, vStride, vPaddedLen / vStride, ggml_row_size(pCur->type, vStride), 0);
  return pCur;
}

// aka pixel_shuffle / pixel_unshuffle / patch_merger (Kimi-VL)
// support dynamic resolution
ggml_tensor * clip_graph::mBuildSuffix(ggml_tensor * pCur) {
  for (int vIdx : fClipSuffixRows(vImg.vSuffixType)) {
    GGML_ASSERT(vModel.vTokEmbdSep && vIdx < vModel.vTokEmbdSep->ne[1]);
    ggml_tensor * vRow = ggml_view_2d(vCtx0, vModel.vTokEmbdSep, vModel.vTokEmbdSep->ne[0], 1,
                                      vModel.vTokEmbdSep->nb[1], vIdx * vModel.vTokEmbdSep->nb[1]);
    pCur               = ggml_concat(vCtx0, pCur, ggml_cast(vCtx0, vRow, pCur->type), 1);
  }
  return pCur;
}

ggml_tensor * clip_graph::mBuildPatchMergePermute(ggml_tensor * pCur, int pScaleFactor) {
  GGML_ASSERT(pScaleFactor > 1);

  const int cNEmbd  = pCur->ne[0];
  int       vWidth  = vImg.mNx() / cPatchSize;
  int       vHeight = vImg.mNy() / cPatchSize;

  // pad width and height to factor
  const int64_t cPadWidth  = CLIP_ALIGN(vWidth, pScaleFactor) - vWidth;
  const int64_t cPadHeight = CLIP_ALIGN(vHeight, pScaleFactor) - vHeight;
  pCur                     = ggml_reshape_3d(vCtx0, pCur, cNEmbd, vWidth, vHeight);
  if (cPadWidth || cPadHeight) {
    pCur = ggml_pad(vCtx0, pCur, 0, cPadWidth, cPadHeight, 0);
    vWidth += cPadWidth;
    vHeight += cPadHeight;
  }

  // unshuffle h
  pCur = ggml_reshape_3d(vCtx0, pCur, cNEmbd * pScaleFactor, vWidth / pScaleFactor, vHeight);
  pCur = ggml_permute(vCtx0, pCur, 0, 2, 1, 3);

  // unshuffle w
  pCur = ggml_cont_3d(vCtx0, pCur, cNEmbd * pScaleFactor * pScaleFactor, vHeight / pScaleFactor, vWidth / pScaleFactor);
  pCur = ggml_permute(vCtx0, pCur, 0, 2, 1, 3);

  pCur = ggml_cont_2d(vCtx0, pCur, pCur->ne[0], pCur->ne[1] * pCur->ne[2]);
  mCb(pCur, "pixel_shuffle", -1);

  return pCur;
}

static std::unique_ptr<clip_graph> fClipGetGraphBuilder(clip_ctx *                   pCtx,
                                                        const clip_image_f32_batch & pImgs,
                                                        const clip_encode_params *   pParams = nullptr) {
  const clip_image_f32 &      vImg = pImgs.lEntries[0];
  std::unique_ptr<clip_graph> vBuilder;

  if (pCtx->mProjType() != cProjectorTypeQwen3vl) {
    throw std::runtime_error("Only the Qwen3.5/3.8 vision projector is supported");
  }
  vBuilder = std::make_unique<clip_graph_qwen3vl>(pCtx, vImg);

  vBuilder->vImgBatch = &pImgs;

  // TODO [QWEN_VIDEO]: improve this in the future
  vBuilder->vNBatch = pImgs.lEntries.size();

  return vBuilder;
}

//
// clip_model_loader
//

struct clip_model_loader {
  ggml_context_ptr vCtxMeta;
  gguf_context_ptr vCtxGguf;

  std::string vFname;

  size_t vModelSize = 0;  // in bytes

  bool vHasVision   = false;
  bool vHasAudio    = false;
  bool vHasGenAudio = false;

  mtmd_progress_callback vProgressCallback         = nullptr;
  void *                 vProgressCallbackUserData = nullptr;

  // TODO @ngxson : we should not pass clip_ctx here, it should be clip_model
  clip_model_loader(const char *           pFname,
                    bool                   pSkipTensors      = false,
                    mtmd_progress_callback pProgressCb       = nullptr,
                    void *                 pProgressUserData = nullptr) :
    vFname(pFname),
    vProgressCallback(pProgressCb),
    vProgressCallbackUserData(pProgressUserData) {
    struct ggml_context * vMeta = nullptr;

    struct gguf_init_params vParams = {
      /*.no_alloc = */ true,
      /*.ctx      = */ &vMeta,
    };

    vCtxGguf = gguf_context_ptr(gguf_init_from_file(pFname, vParams));
    if (!vCtxGguf.get()) {
      throw std::runtime_error(
        fStringFormat("%s: failed to load CLIP model from %s. Does this file exist?\n", __func__, pFname));
    }

    vCtxMeta.reset(vMeta);

    const int cNTensors = gguf_get_n_tensors(vCtxGguf.get());

    // print gguf info
    {
      std::string name;
      mGetString(KEY_NAME, name, false);
      std::string vDescription;
      mGetString(KEY_DESCRIPTION, vDescription, false);
      LOG_INF("%s: model name:   %s\n", __func__, name.c_str());
      LOG_INF("%s: description:  %s\n", __func__, vDescription.c_str());
      LOG_INF("%s: GGUF version: %d\n", __func__, gguf_get_version(vCtxGguf.get()));
      LOG_INF("%s: alignment:    %zu\n", __func__, gguf_get_alignment(vCtxGguf.get()));
      LOG_INF("%s: n_tensors:    %d\n", __func__, cNTensors);
      LOG_INF("%s: n_kv:         %d\n", __func__, (int) gguf_get_n_kv(vCtxGguf.get()));
      LOG_INF("\n");
    }

    // modalities
    {
      mGetBool(KEY_HAS_VISION_ENC, vHasVision, false);
      mGetBool(KEY_HAS_AUDIO_ENC, vHasAudio, false);
      mGetBool(KEY_HAS_GEN_AUDIO_ENC, vHasGenAudio, false);

      if (vHasVision) {
        LOG_INF("%s: has vision encoder\n", __func__);
      }
      if (vHasAudio) {
        LOG_INF("%s: has audio encoder\n", __func__);
      }
      if (vHasGenAudio) {
        LOG_INF("%s: has audio generation (gen) encoder\n", __func__);
      }
    }

    // tensors
    if (!pSkipTensors) {
      for (int vI = 0; vI < cNTensors; ++vI) {
        const char *   name        = gguf_get_tensor_name(vCtxGguf.get(), vI);
        const size_t   cOffset     = gguf_get_tensor_offset(vCtxGguf.get(), vI);
        enum ggml_type type        = gguf_get_tensor_type(vCtxGguf.get(), vI);
        ggml_tensor *  vCur        = ggml_get_tensor(vMeta, name);
        size_t         vTensorSize = ggml_nbytes(vCur);
        vModelSize += vTensorSize;
        LOG_DBG("%s: tensor[%d]: n_dims = %d, name = %s, tensor_size=%zu, offset=%zu, shape:[%" PRIu64 ", %" PRIu64
                ", %" PRIu64 ", %" PRIu64 "], type = %s\n",
                __func__, vI, ggml_n_dims(vCur), vCur->name, vTensorSize, cOffset, vCur->ne[0], vCur->ne[1],
                vCur->ne[2], vCur->ne[3], ggml_type_name(type));
      }
    }
  }

  void mLoadHparams(clip_model & pModel, clip_modality pModality) {
    auto &      vHparams = pModel.vHparams;
    std::string vLogFfnOp;  // for logging

    // sanity check
    if (pModality == cClipModalityVision) {
      GGML_ASSERT(vHasVision);
    } else if (pModality == cClipModalityAudio) {
      GGML_ASSERT(vHasAudio);
    } else if (pModality == cClipModalityGenAudio) {
      GGML_ASSERT(vHasGenAudio);
    }
    pModel.vModality = pModality;

    // projector type
    std::string vProjType;
    {
      // default key
      mGetString(KEY_PROJ_TYPE, vProjType, false);

      // for models with mixed modalities
      if (vProjType.empty()) {
        if (pModality == cClipModalityVision) {
          mGetString(KEY_VISION_PROJ_TYPE, vProjType, false);
        } else if (pModality == cClipModalityAudio) {
          mGetString(KEY_AUDIO_PROJ_TYPE, vProjType, false);
        } else if (pModality == cClipModalityGenAudio) {
          mGetString(KEY_GEN_AUDIO_PROJ_TYPE, vProjType, false);
        } else {
          GGML_ABORT("unknown modality");
        }
      }

      pModel.vProjType = fClipProjectorTypeFromString(vProjType);
      if (pModel.vProjType != cProjectorTypeQwen3vl || pModality != cClipModalityVision) {
        throw std::runtime_error("Only Qwen3.5/3.8 vision projectors are supported");
      }

      if (pModel.vProjType == cProjectorTypeUnknown) {
        throw std::runtime_error(fStringFormat("%s: unknown projector type: %s\n", __func__, vProjType.c_str()));
      }
    }

    const bool cIsVision   = pModel.vModality == cClipModalityVision;
    const bool cIsAudio    = pModel.vModality == cClipModalityAudio;
    const bool cIsGenAudio = pModel.vModality == cClipModalityGenAudio;

    // other hparams
    {
      const char * vPrefix = cIsVision ? "vision" : (cIsAudio ? "audio" : "gen.audio");
      mGetU32(fStringFormat(KEY_N_EMBD, vPrefix), vHparams.vNEmbd);
      mGetU32(fStringFormat(KEY_N_HEAD, vPrefix), vHparams.vNHead);
      mGetU32(fStringFormat(KEY_N_EMBD_HEAD, vPrefix), vHparams.vNEmbdHead, false);
      mGetU32(fStringFormat(KEY_N_FF, vPrefix), vHparams.vNFf);
      mGetU32(fStringFormat(KEY_N_BLOCK, vPrefix), vHparams.vNLayer);
      mGetU32(fStringFormat(KEY_PROJ_DIM, vPrefix), vHparams.vProjectionDim);
      mGetF32(fStringFormat(KEY_LAYER_NORM_EPS, vPrefix), vHparams.vEps);

      // n_head_kv is optional (for GQA), default to n_head
      vHparams.vNHeadKv = vHparams.vNHead;
      mGetU32(fStringFormat(KEY_N_HEAD_KV, vPrefix), vHparams.vNHeadKv, false);

      if (cIsVision) {
        mGetU32(KEY_IMAGE_SIZE, vHparams.vImageSize);
        mGetU32(KEY_PATCH_SIZE, vHparams.vPatchSize);
        mGetU32(KEY_MAX_SLICE_NUMS, vHparams.vMaxSliceNums, false);

      } else if (cIsAudio) {
        mGetU32(KEY_A_NUM_MEL_BINS, vHparams.vNMelBins);
        // some hparams are unused, but still need to set to avoid issues
        vHparams.vImageSize = 0;
        vHparams.vPatchSize = 1;

      } else if (cIsGenAudio) {
        // these are unused, but still need to be set to avoid issues
        vHparams.vImageSize = 0;
        vHparams.vPatchSize = 1;
        mGetString(KEY_GEN_AUDIO_VARIANT, vHparams.vGenModelVariant, false);

      } else {
        GGML_ASSERT(false && "unknown modality");
      }

      // for pinpoints, we need to convert it into a list of resolution candidates
      {
        std::vector<int> lPinpoints;
        mGetArrInt(KEY_IMAGE_GRID_PINPOINTS, lPinpoints, false);
        if (lPinpoints.size() % 2 != 0) {
          throw std::runtime_error(fStringFormat(
            "%s: image_grid_pinpoints must have an even number of elements, got %zu\n", __func__, lPinpoints.size()));
        }
        if (!lPinpoints.empty()) {
          for (size_t vI = 0; vI < lPinpoints.size(); vI += 2) {
            vHparams.lImageResCandidates.push_back({
              lPinpoints[vI],
              lPinpoints[vI + 1],
            });
          }
        }
      }

      // default warmup value
      vHparams.vWarmupImageSize = vHparams.vImageSize;

      {
        bool vUseGelu = false;
        bool vUseSilu = false;
        mGetBool(KEY_USE_GELU, vUseGelu, false);
        mGetBool(KEY_USE_SILU, vUseSilu, false);
        if (vUseGelu && vUseSilu) {
          throw std::runtime_error(fStringFormat("%s: both use_gelu and use_silu are set to true\n", __func__));
        }
        if (vUseGelu) {
          vHparams.vFfnOp = cFfnGelu;
          vLogFfnOp       = "gelu";
        } else if (vUseSilu) {
          vHparams.vFfnOp = cFfnSilu;
          vLogFfnOp       = "silu";
        } else {
          vHparams.vFfnOp = cFfnGeluQuick;
          vLogFfnOp       = "gelu_quick";
        }
      }

      {
        std::string vMmPatchMergeType;
        mGetString(KEY_MM_PATCH_MERGE_TYPE, vMmPatchMergeType, false);
        if (vMmPatchMergeType == "spatial_unpad") {
          vHparams.vMmPatchMergeType = cPatchMergeSpatialUnpad;
        }
      }

      if (cIsVision) {
        std::vector<float> lImageMean;
        std::vector<float> lImageStd;
        mGetArrF32(KEY_IMAGE_MEAN, lImageMean, false);
        mGetArrF32(KEY_IMAGE_STD, lImageStd, false);
        if (lImageMean.size() < 3 || lImageStd.size() < 3) {
          throw std::runtime_error(
            fStringFormat("%s: image_mean/image_std arrays must have at least 3 elements, got %zu and %zu\n", __func__,
                          lImageMean.size(), lImageStd.size()));
        }
        for (int vI = 0; vI < 3; ++vI) {
          vHparams.aImageMean[vI] = lImageMean[vI];
          vHparams.aImageStd[vI]  = lImageStd[vI];
        }
      }

      // Load the vision/audio feature layer indices if they are explicitly provided
      // NOTE: gguf conversions should standardize the values of the vision feature layer to be non-negative, since we use -1 to mark values as unset here.
      mGetArrInt(fStringFormat(KEY_FEATURE_LAYERS, vPrefix), vHparams.lFeatureLayers, false);
      for (const auto & vV : vHparams.lFeatureLayers) {
        if (vV > (int) vHparams.vNLayer) {
          throw std::runtime_error(
            fStringFormat("%s: feature layer index %d is out of range (n_layer: %d)", __func__, vV, vHparams.vNLayer));
        }
      }

      // model-specific params
      switch (pModel.vProjType) {
        case cProjectorTypeQwen3vl:

          {
            vHparams.vNMerge          = 2;  // default value for Qwen 2 and 2.5
            vHparams.vImageResizeAlgo = cResizeAlgoBicubic;
            mGetU32(KEY_SPATIAL_MERGE_SIZE, vHparams.vNMerge, false);
            mGetU32(KEY_WIN_ATTN_PATTERN, vHparams.vNWaPattern,
                    pModel.vProjType == cProjectorTypeQwen25vl);  // only 2.5 requires it
            // ref: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/preprocessor_config.json
            vHparams.mSetLimitImageTokens(8, 4096);
            // optional limits of the model, the custom values take precedence
            if (vHparams.vCustomImageMinTokens <= 0) {
              mGetU32(KEY_IMAGE_MIN_PIXELS, vHparams.vImageMinPixels, false);
            }
            if (vHparams.vCustomImageMaxTokens <= 0) {
              mGetU32(KEY_IMAGE_MAX_PIXELS, vHparams.vImageMaxPixels, false);
            }
            vHparams.mSetWarmupNTokens(46 * 46);  // avoid OOM on warmup
            const int cWarnMinPixels =
              1024 * vHparams.vNMerge * vHparams.vNMerge * vHparams.vPatchSize * vHparams.vPatchSize;
            if (vHparams.vImageMinPixels < cWarnMinPixels) {
              LOG_WRN(
                "%s: Qwen-VL models require at minimum 1024 image tokens to function correctly on grounding tasks\n",
                __func__);
              LOG_WRN("%s: if you encounter problems with accuracy, try adding --image-min-tokens 1024\n", __func__);
              LOG_WRN("%s: more info: doc/MANUAL.md\n\n", __func__);
            }
          }
          break;
        default:
          throw std::runtime_error(
            fStringFormat("%s: unknown vision projector type %s\n", __func__, vProjType.c_str()));
      }

      // sanity check
      {
        if (vHparams.vImageSize < 0) {
          // note: some models having hparams.image_size == 0, which means the image size is dynamic
          throw std::runtime_error(
            fStringFormat("%s: image_size (%d) cannot be negative\n", __func__, vHparams.vImageSize));
        }
        if (vHparams.vImageSize > 8192) {
          // cap prevents int32 overflow in n_patches = (image_size/patch_size)^2
          throw std::runtime_error(
            fStringFormat("%s: image_size (%d) is too large (max 8192)\n", __func__, vHparams.vImageSize));
        }
        if (vHparams.vPatchSize <= 0 || vHparams.vPatchSize >= 65536) {
          throw std::runtime_error(
            fStringFormat("%s: patch_size (%d) must be positive and less than 65536\n", __func__, vHparams.vPatchSize));
        }
        if (vHparams.vNEmbd <= 0) {
          throw std::runtime_error(
            fStringFormat("%s: n_embd (%d) must be greater than 0\n", __func__, vHparams.vNEmbd));
        }
        if (vHparams.vImageMaxPixels < vHparams.vImageMinPixels) {
          throw std::runtime_error(fStringFormat("%s: image_max_pixels (%d) is less than image_min_pixels (%d)\n",
                                                 __func__, vHparams.vImageMaxPixels, vHparams.vImageMinPixels));
        }
        if (vHparams.vNMerge <= 0 || vHparams.vNMerge >= 65536) {
          throw std::runtime_error(
            fStringFormat("%s: n_merge (%d) must be greater than 0 and less than 65536\n", __func__, vHparams.vNMerge));
        }
        if (vHparams.vAttnWindowSize > 4096) {
          throw std::runtime_error(
            fStringFormat("%s: attn_window_size (%d) is too large (max 4096)\n", __func__, vHparams.vAttnWindowSize));
        }
      }

      LOG_INF("%s: projector:          %s\n", __func__, vProjType.c_str());
      LOG_INF("%s: n_embd:             %d\n", __func__, vHparams.vNEmbd);
      LOG_INF("%s: n_head:             %d\n", __func__, vHparams.vNHead);
      LOG_INF("%s: n_ff:               %d\n", __func__, vHparams.vNFf);
      LOG_INF("%s: n_layer:            %d\n", __func__, vHparams.vNLayer);
      LOG_INF("%s: ffn_op:             %s\n", __func__, vLogFfnOp.c_str());
      LOG_INF("%s: projection_dim:     %d\n", __func__, vHparams.vProjectionDim);
      if (cIsVision) {
        LOG_INF("\n--- vision hparams ---\n");
        LOG_INF("%s: image_size:         %d\n", __func__, vHparams.vImageSize);
        LOG_INF("%s: patch_size:         %d\n", __func__, vHparams.vPatchSize);
        LOG_INF("%s: has_llava_proj:     %d\n", __func__, vHparams.vHasLlavaProjector);
        LOG_INF("%s: minicpmv_version:   %d\n", __func__, vHparams.vMinicpmvVersion);
        LOG_INF("%s: n_merge:            %d\n", __func__, vHparams.vNMerge);
        LOG_INF("%s: n_wa_pattern: %d\n", __func__, vHparams.vNWaPattern);
        if (!vHparams.sWaLayerIndexes.empty()) {
          LOG_INF("%s: wa_layer_indexes:  ", __func__);
          for (auto & layer : vHparams.sWaLayerIndexes) {
            LOG_INF("%d ", layer);
          }
          LOG_INF("\n");
        }
        if (vHparams.vImageMinPixels > 0) {
          LOG_INF("%s: image_min_pixels:   %d%s\n", __func__, vHparams.vImageMinPixels,
                  vHparams.vCustomImageMinTokens > 0 ? " (custom value)" : "");
        }
        if (vHparams.vImageMaxPixels > 0) {
          LOG_INF("%s: image_max_pixels:   %d%s\n", __func__, vHparams.vImageMaxPixels,
                  vHparams.vCustomImageMaxTokens > 0 ? " (custom value)" : "");
        }
        if (vHparams.vPreprocMaxTiles > 0) {
          LOG_INF("%s: preproc_tiles:      %d - %d\n", __func__, vHparams.vPreprocMinTiles, vHparams.vPreprocMaxTiles);
        }
      } else if (cIsAudio) {
        LOG_INF("\n--- audio hparams ---\n");
        LOG_INF("%s: n_mel_bins:         %d\n", __func__, vHparams.vNMelBins);
        LOG_INF("%s: proj_stack_factor:  %d\n", __func__, vHparams.vProjStackFactor);
        LOG_INF("%s: audio_chunk_len:    %d\n", __func__, vHparams.vAudioChunkLen);
        LOG_INF("%s: audio_sample_rate:  %d\n", __func__, vHparams.vAudioSampleRate);
        LOG_INF("%s: audio_n_fft:        %d\n", __func__, vHparams.vAudioNFft);
        LOG_INF("%s: audio_window_len:   %d\n", __func__, vHparams.vAudioWindowLen);
        LOG_INF("%s: audio_hop_len:      %d\n", __func__, vHparams.vAudioHopLen);

        // GEMMA4UA is encoder-free: it uses n_mel_bins as a raw-waveform frame size (640) and has no FFT/filterbank, so the mel-range and FFT
        // checks below do not apply to it.
        // pocket-tts is encoder-free in the same sense: mimi convolves the raw waveform
        const bool cFftBased =
          pModel.vProjType != cProjectorTypeGemma4ua && pModel.vProjType != cProjectorTypePocketttsSpkenc;

        // Validate audio hparams loaded from GGUF metadata
        if (vHparams.vNMelBins <= 0 || (cFftBased && vHparams.vNMelBins > 256)) {
          throw std::runtime_error(
            fStringFormat("%s: n_mel_bins (%d) must be in range [1, 256]\n", __func__, vHparams.vNMelBins));
        }
        if (cFftBased && (vHparams.vAudioSampleRate <= 0 || vHparams.vAudioNFft <= 0 || vHparams.vAudioHopLen <= 0 ||
                          vHparams.vAudioWindowLen <= 0)) {
          throw std::runtime_error(fStringFormat(
            "%s: audio hparams invalid: sample_rate=%d n_fft=%d window_len=%d hop_len=%d\n", __func__,
            vHparams.vAudioSampleRate, vHparams.vAudioNFft, vHparams.vAudioWindowLen, vHparams.vAudioHopLen));
        }
      }
      LOG_INF("\n");
      LOG_INF("%s: model size:         %.2f MiB\n", __func__, vModelSize / 1024.0 / 1024.0);
      LOG_INF("%s: metadata size:      %.2f MiB\n", __func__, ggml_get_mem_size(vCtxMeta.get()) / 1024.0 / 1024.0);
    }
  }

  void mLoadTensors(clip_ctx & pCtxClip) {
    auto &                        vModel   = pCtxClip.vModel;
    auto &                        vHparams = vModel.vHparams;
    std::map<std::string, size_t> dTensorOffset;
    std::vector<ggml_tensor *>    lTensorsToLoad;

    auto vFin = fOpenIfstreamBinary(vFname);
    if (!vFin) {
      throw std::runtime_error(fStringFormat("%s: failed to open %s\n", __func__, vFname.c_str()));
    }

    // TODO @ngxson : support both audio and video in the future
    const char * vPrefix = vModel.vModality == cClipModalityAudio    ? "a" :
                           vModel.vModality == cClipModalityGenAudio ? "a.gen.code" :
                                                                       "v";

    // get offsets
    for (int64_t vI = 0; vI < gguf_get_n_tensors(vCtxGguf.get()); ++vI) {
      const char * name   = gguf_get_tensor_name(vCtxGguf.get(), vI);
      dTensorOffset[name] = gguf_get_data_offset(vCtxGguf.get()) + gguf_get_tensor_offset(vCtxGguf.get(), vI);
    }

    // create data context
    struct ggml_init_params vParams = {
      /*.mem_size =*/static_cast<size_t>(gguf_get_n_tensors(vCtxGguf.get()) + 1) * ggml_tensor_overhead(),
      /*.mem_buffer =*/NULL,
      /*.no_alloc =*/true,
    };
    pCtxClip.vCtxData.reset(ggml_init(vParams));
    if (!pCtxClip.vCtxData) {
      throw std::runtime_error(fStringFormat("%s: failed to init ggml context\n", __func__));
    }

    // helper function
    std::unordered_set<std::string> sLoadedTensorNames;
    auto                            vGetTensor = [&](const std::string & name, bool pRequired = true) {
      // Each tensor should only be loaded once; duplicates indicate a bug
      if (sLoadedTensorNames.count(name)) {
        throw std::runtime_error(fStringFormat("%s: tensor already loaded: %s\n", __func__, name.c_str()));
      }
      ggml_tensor * vCur = ggml_get_tensor(vCtxMeta.get(), name.c_str());
      if (!vCur && pRequired) {
        throw std::runtime_error(fStringFormat("%s: unable to find tensor %s\n", __func__, name.c_str()));
      }
      if (vCur) {
        lTensorsToLoad.push_back(vCur);
        ggml_tensor * vDataTensor = ggml_dup_tensor(pCtxClip.vCtxData.get(), vCur);
        ggml_set_name(vDataTensor, vCur->name);
        sLoadedTensorNames.insert(name);
        vCur = vDataTensor;
        // add to weight memory counter
        pCtxClip.dMemUsage[ggml_backend_get_device(pCtxClip.vBackend)] += ggml_nbytes(vCur);
      }
      return vCur;
    };

    // pocket-tts: the encoder and the decoder share the same layout, only the prefix differs
    auto vLoadSeanet = [&](clip_seanet & pSeanet, bool pIsDecoder) {
      const char * vConvIn  = pIsDecoder ? TN_A_GEN_WAV_SEANET_CONV_IN : TN_A_SEANET_CONV_IN;
      const char * vConvOut = pIsDecoder ? TN_A_GEN_WAV_SEANET_CONV_OUT : TN_A_SEANET_CONV_OUT;
      const char * vRes1    = pIsDecoder ? TN_A_GEN_WAV_SEANET_RES_CONV1 : TN_A_SEANET_RES_CONV1;
      const char * vRes2    = pIsDecoder ? TN_A_GEN_WAV_SEANET_RES_CONV2 : TN_A_SEANET_RES_CONV2;
      const char * vScale   = pIsDecoder ? TN_A_GEN_WAV_SEANET_SCALE_CONV : TN_A_SEANET_SCALE_CONV;

      pSeanet.vConvInW  = vGetTensor(fStringFormat(vConvIn, "weight"));
      pSeanet.vConvInB  = vGetTensor(fStringFormat(vConvIn, "bias"));
      pSeanet.vConvOutW = vGetTensor(fStringFormat(vConvOut, "weight"));
      pSeanet.vConvOutB = vGetTensor(fStringFormat(vConvOut, "bias"));

      pSeanet.lStages.resize(vHparams.vSeanetNStage);
      for (int vI = 0; vI < vHparams.vSeanetNStage; vI++) {
        auto & stage      = pSeanet.lStages[vI];
        stage.vResConv1W  = vGetTensor(fStringFormat(vRes1, vI, "weight"));
        stage.vResConv1B  = vGetTensor(fStringFormat(vRes1, vI, "bias"));
        stage.vResConv2W  = vGetTensor(fStringFormat(vRes2, vI, "weight"));
        stage.vResConv2B  = vGetTensor(fStringFormat(vRes2, vI, "bias"));
        stage.vScaleConvW = vGetTensor(fStringFormat(vScale, vI, "weight"));
        stage.vScaleConvB = vGetTensor(fStringFormat(vScale, vI, "bias"));
      }
    };

    auto vGetVector = [&](const std::string & name) {
      std::vector<float> lResult;
      auto               vIt = dTensorOffset.find(name);
      if (vIt == dTensorOffset.end()) {
        return lResult;
      }

      const int64_t cIdx = gguf_find_tensor(vCtxGguf.get(), name.c_str());
      if (cIdx < 0) {
        throw std::runtime_error(fStringFormat("%s: failed to find tensor %s\n", __func__, name.c_str()));
      }

      if (const auto type = gguf_get_tensor_type(vCtxGguf.get(), cIdx); type != GGML_TYPE_F32) {
        throw std::runtime_error(fStringFormat("%s: %s must be %s, was %s\n", __func__, name.c_str(),
                                               ggml_type_name(GGML_TYPE_F32), ggml_type_name(type)));
      }

      const size_t cNBytes = gguf_get_tensor_size(vCtxGguf.get(), cIdx);
      if (cNBytes == 0) {
        throw std::runtime_error(fStringFormat("%s: tensor %s is empty\n", __func__, name.c_str()));
      }

      const size_t cNElems = cNBytes / sizeof(float);
      lResult.resize(cNElems);
      vFin.seekg(vIt->second, std::ios::beg);
      vFin.read(reinterpret_cast<char *>(lResult.data()), cNBytes);
      return lResult;
    };

    auto vGetScalar = [&](const std::string & name, float pDefaultVal) {
      auto lV = vGetVector(name);
      if (lV.empty()) {
        return pDefaultVal;
      }
      if (lV.size() != 1) {
        throw std::runtime_error(fStringFormat("%s: expected scalar tensor '%s' but got %d elements\n", __func__,
                                               name.c_str(), (int) lV.size()));
      }

      return lV[0];
    };

    vModel.vClassEmbedding = vGetTensor(TN_CLASS_EMBD, false);

    vModel.vPreLnW = vGetTensor(fStringFormat(TN_LN_PRE, vPrefix, "weight"), false);
    vModel.vPreLnB = vGetTensor(fStringFormat(TN_LN_PRE, vPrefix, "bias"), false);

    vModel.vPostLnW = vGetTensor(fStringFormat(TN_LN_POST, vPrefix, "weight"), false);
    vModel.vPostLnB = vGetTensor(fStringFormat(TN_LN_POST, vPrefix, "bias"), false);

    vModel.vPatchBias        = vGetTensor(TN_PATCH_BIAS, false);
    vModel.vPatchEmbeddings0 = vGetTensor(TN_PATCH_EMBD, false);
    vModel.vPatchEmbeddings1 = vGetTensor(TN_PATCH_EMBD_1, false);

    vModel.vNormEmbdW = vGetTensor(fStringFormat(TN_NORM_EMBD, "weight"), false);
    vModel.vNormEmbdB = vGetTensor(fStringFormat(TN_NORM_EMBD, "bias"), false);

    vModel.vPositionEmbeddings = vGetTensor(fStringFormat(TN_POS_EMBD, vPrefix), false);

    const bool cHasStandardLayers =
      (vModel.vProjType != cProjectorTypeGemma3nv && vModel.vProjType != cProjectorTypeQwen3ttsSpkenc &&
       vModel.vProjType != cProjectorTypePocketttsGen);

    // layers
    const int cNLayersToLoad = cHasStandardLayers ? vHparams.vNLayer : 0;
    vModel.lLayers.resize(cNLayersToLoad);
    for (int vIl = 0; vIl < cNLayersToLoad; ++vIl) {
      auto & layer         = vModel.lLayers[vIl];
      layer.vKW            = vGetTensor(fStringFormat(TN_ATTN_K, vPrefix, vIl, "weight"), false);
      layer.vQW            = vGetTensor(fStringFormat(TN_ATTN_Q, vPrefix, vIl, "weight"), false);
      layer.vVW            = vGetTensor(fStringFormat(TN_ATTN_V, vPrefix, vIl, "weight"), false);
      layer.vOW            = vGetTensor(fStringFormat(TN_ATTN_OUTPUT, vPrefix, vIl, "weight"));
      layer.vQkvW          = vGetTensor(fStringFormat(TN_ATTN_QKV, vPrefix, vIl, "weight"), false);
      layer.vKNorm         = vGetTensor(fStringFormat(TN_ATTN_K_NORM, vPrefix, vIl, "weight"), false);
      layer.vQNorm         = vGetTensor(fStringFormat(TN_ATTN_Q_NORM, vPrefix, vIl, "weight"), false);
      layer.vLn1W          = vGetTensor(fStringFormat(TN_LN_1, vPrefix, vIl, "weight"), false);
      layer.vLn2W          = vGetTensor(fStringFormat(TN_LN_2, vPrefix, vIl, "weight"), false);
      layer.vLs1W          = vGetTensor(fStringFormat(TN_LS_1, vPrefix, vIl, "weight"), false);            // no bias
      layer.vLs2W          = vGetTensor(fStringFormat(TN_LS_2, vPrefix, vIl, "weight"), false);            // no bias
      layer.vLsOutW        = vGetTensor(fStringFormat(TN_LS_OUT, vPrefix, vIl, "weight"), false);          // no bias
      layer.vAttnPostNormW = vGetTensor(fStringFormat(TN_ATTN_POST_NORM, vPrefix, vIl, "weight"), false);  // no bias
      layer.vFfPostNormW   = vGetTensor(fStringFormat(TN_FFN_POST_NORM, vPrefix, vIl, "weight"), false);   // no bias

      layer.vKB   = vGetTensor(fStringFormat(TN_ATTN_K, vPrefix, vIl, "bias"), false);
      layer.vQB   = vGetTensor(fStringFormat(TN_ATTN_Q, vPrefix, vIl, "bias"), false);
      layer.vVB   = vGetTensor(fStringFormat(TN_ATTN_V, vPrefix, vIl, "bias"), false);
      layer.vOB   = vGetTensor(fStringFormat(TN_ATTN_OUTPUT, vPrefix, vIl, "bias"), false);
      layer.vQkvB = vGetTensor(fStringFormat(TN_ATTN_QKV, vPrefix, vIl, "bias"), false);
      layer.vLn1B = vGetTensor(fStringFormat(TN_LN_1, vPrefix, vIl, "bias"), false);
      layer.vLn2B = vGetTensor(fStringFormat(TN_LN_2, vPrefix, vIl, "bias"), false);

      // MoE ffn (dots3note vision pyramid blocks); replaces the dense ffn when present
      layer.vFfGateInpW  = vGetTensor(fStringFormat(TN_FFN_GATE_INP, vPrefix, vIl, "weight"), false);
      layer.vFfGateExpsW = vGetTensor(fStringFormat(TN_FFN_GATE_EXPS, vPrefix, vIl, "weight"), false);
      layer.vFfUpExpsW   = vGetTensor(fStringFormat(TN_FFN_UP_EXPS, vPrefix, vIl, "weight"), false);
      layer.vFfDownExpsW = vGetTensor(fStringFormat(TN_FFN_DOWN_EXPS, vPrefix, vIl, "weight"), false);
      layer.vFfExpProbsB = vGetTensor(fStringFormat(TN_FFN_EXP_PROBS_B, vPrefix, vIl, "weight"), false);
      const bool cIsMoe  = layer.vFfGateExpsW != nullptr;

      // ffn
      layer.vFfUpW   = vGetTensor(fStringFormat(TN_FFN_UP, vPrefix, vIl, "weight"), !cIsMoe);
      layer.vFfUpB   = vGetTensor(fStringFormat(TN_FFN_UP, vPrefix, vIl, "bias"), false);
      layer.vFfGateW = vGetTensor(fStringFormat(TN_FFN_GATE, vPrefix, vIl, "weight"), false);
      layer.vFfGateB = vGetTensor(fStringFormat(TN_FFN_GATE, vPrefix, vIl, "bias"), false);
      layer.vFfDownW = vGetTensor(fStringFormat(TN_FFN_DOWN, vPrefix, vIl, "weight"), !cIsMoe);
      layer.vFfDownB = vGetTensor(fStringFormat(TN_FFN_DOWN, vPrefix, vIl, "bias"), false);

      // mimovl per-head attention sink bias
      layer.vAttnSinks = vGetTensor(fStringFormat(TN_ATTN_SINKS, vPrefix, vIl), false);

      // qwen3vl deepstack layer
      layer.vDeepstackNormW = vGetTensor(fStringFormat(TN_DEEPSTACK_NORM, vIl, "weight"), false);
      layer.vDeepstackNormB = vGetTensor(fStringFormat(TN_DEEPSTACK_NORM, vIl, "bias"), false);
      layer.vDeepstackFc1W  = vGetTensor(fStringFormat(TN_DEEPSTACK_FC1, vIl, "weight"), false);
      layer.vDeepstackFc1B  = vGetTensor(fStringFormat(TN_DEEPSTACK_FC1, vIl, "bias"), false);
      layer.vDeepstackFc2W  = vGetTensor(fStringFormat(TN_DEEPSTACK_FC2, vIl, "weight"), false);
      layer.vDeepstackFc2B  = vGetTensor(fStringFormat(TN_DEEPSTACK_FC2, vIl, "bias"), false);
      if (layer.mHasDeepstack()) {
        vModel.vNDeepstackLayers++;
      }
    }

    switch (vModel.vProjType) {
      case cProjectorTypeQwen3vl:
        {
          vModel.vMm0W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 0, "weight"));
          vModel.vMm0B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 0, "bias"));
          vModel.vMm1W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 2, "weight"));
          vModel.vMm1B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 2, "bias"));
        }
        break;
      default:
        GGML_ASSERT(false && "unknown projector type");
    }

    // load data
    {
      std::vector<uint8_t> lReadBuf;

      // start loading event
      if (vProgressCallback) {
        vProgressCallback(0.0, vProgressCallbackUserData);
      }

      // compute total tensor data size for progress reporting
      size_t vTotalDataSize = 0;
      for (auto & vT : lTensorsToLoad) {
        vTotalDataSize += ggml_nbytes(vT);
      }

      // alloc memory and offload data
      ggml_backend_buffer_type_t vBuft = ggml_backend_get_default_buffer_type(pCtxClip.vBackend);
      pCtxClip.vBuf.reset(ggml_backend_alloc_ctx_tensors_from_buft(pCtxClip.vCtxData.get(), vBuft));
      ggml_backend_buffer_set_usage(pCtxClip.vBuf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
      // read the weight from file
      if (!pCtxClip.vNoAlloc) {
        size_t vDataLoaded = 0;
        for (auto & vT : lTensorsToLoad) {
          ggml_tensor * vCur = ggml_get_tensor(pCtxClip.vCtxData.get(), vT->name);
          GGML_ASSERT(vCur && "tensor not found in ctx_data");
          auto vItOff = dTensorOffset.find(vT->name);
          GGML_ASSERT(vItOff != dTensorOffset.end() && "no offset for tensor");
          const size_t cOffset = vItOff->second;
          vFin.seekg(cOffset, std::ios::beg);
          if (!vFin) {
            throw std::runtime_error(fStringFormat("%s: failed to seek for tensor %s\n", __func__, vT->name));
          }
          size_t vNumBytes = ggml_nbytes(vCur);
          if (ggml_backend_buft_is_host(vBuft)) {
            // for the CPU and Metal backend, we can read directly into the tensor
            vFin.read(reinterpret_cast<char *>(vCur->data), vNumBytes);
          } else {
            // read into a temporary buffer first, then copy to device memory
            lReadBuf.resize(vNumBytes);
            vFin.read(reinterpret_cast<char *>(lReadBuf.data()), vNumBytes);
            ggml_backend_tensor_set(vCur, lReadBuf.data(), 0, vNumBytes);
          }
          vDataLoaded += vNumBytes;
          if (vProgressCallback && vTotalDataSize > 0) {
            const float cProgress = (float) vDataLoaded / (float) vTotalDataSize;
            if (!vProgressCallback(cProgress, vProgressCallbackUserData)) {
              throw std::runtime_error(fStringFormat("%s: model loading cancelled by progress_callback\n", __func__));
            }
          }
        }
        LOG_DBG("%s: loaded %zu tensors from %s\n", __func__, lTensorsToLoad.size(), vFname.c_str());
      } else {
        LOG_DBG("%s: no_alloc is set, skipping tensor data loading (%zu tensors)\n", __func__, lTensorsToLoad.size());
      }
      vFin.close();
    }
  }

  struct support_info_op {
    ggml_tensor * op;

    // true if the op runs on the accelerated ctx_clip.backend
    bool vIsAccel = true;
  };

  struct support_info_graph {
    // whether the clip_ctx.backend supports flash attention
    bool          vFattn   = true;
    ggml_tensor * vFattnOp = nullptr;  // for debugging

    std::vector<support_info_op> lOps;
  };

  static clip_image_f32_batch mGetDummyBatch(clip_ctx & pCtxClip) {
    // create a fake batch
    const auto &         vHparams = pCtxClip.vModel.vHparams;
    clip_image_f32_batch vBatch;
    clip_image_f32       vImg;
    if (pCtxClip.vModel.vModality == cClipModalityVision) {
      const int cSz = vHparams.vWarmupImageSize;
      vImg.mSetSize({ cSz, cSz }, false, false);
      LOG_INF("%s: warmup with image size = %d x %d\n", __func__, cSz, cSz);
    } else {
      // GEMMA4UA uses n_mel_bins as a raw-waveform frame size (640), not a mel-bin count,
      // so the [1, 256] bound only applies to FFT-based models.
      const bool cFftBased = pCtxClip.vModel.vProjType != cProjectorTypeGemma4ua;
      if (vHparams.vNMelBins <= 0 || (cFftBased && vHparams.vNMelBins > 256)) {
        throw std::runtime_error(
          fStringFormat("%s: invalid n_mel_bins (%d), must be in [1, 256]\n", __func__, vHparams.vNMelBins));
      }
      vImg.mSetSize({ vHparams.vWarmupAudioSize, vHparams.vNMelBins }, false, false);
      LOG_INF("%s: warmup with audio size = %d\n", __func__, vHparams.vWarmupAudioSize);
    }
    vBatch.lEntries.push_back(vImg);
    return vBatch;
  }

  static void mInitCtx(clip_ctx & pCtxClip) {
    pCtxClip.lBufComputeMeta.resize(pCtxClip.vMaxNodes * ggml_tensor_overhead() + ggml_graph_overhead());

    // check batching support
    auto vBatch            = mGetDummyBatch(pCtxClip);
    auto vBuilder          = fClipGetGraphBuilder(&pCtxClip, vBatch);
    pCtxClip.vSupportBatch = vBuilder->mSupportBatch();
  }

  static void mWarmup(clip_ctx & pCtxClip) {
    auto vBatch = mGetDummyBatch(pCtxClip);
    mWarmup(pCtxClip, vBatch);
  }

  static void mWarmup(clip_ctx & pCtxClip, const clip_image_f32_batch & pBatch) {
    support_info_graph info;

    if (pCtxClip.vFlashAttnType == cClipFlashAttnTypeAuto) {
      // try to enable flash attention to see if it's supported
      pCtxClip.vFlashAttnType = cClipFlashAttnTypeEnabled;
      info                    = mReserveComputeMeta(pCtxClip, pBatch);
      if (!info.vFattn && info.vFattnOp) {
        auto vOp = info.vFattnOp;
        LOG_WRN("%s: *****************************************************************\n", __func__);
        LOG_WRN("%s: WARNING: flash attention not supported by %s, memory usage will increase\n", __func__,
                ggml_backend_name(pCtxClip.vBackend));
        LOG_WRN("%s: op params: \n", __func__);
        static auto vPrintShape = [](const char * pFn, const char * name, ggml_tensor * pT) {
          LOG_WRN("%s:   %s: type = %s, ne = [%d %d %d %d], nb = [%d %d %d %d]\n", pFn, name, ggml_type_name(pT->type),
                  pT->ne[0], pT->ne[1], pT->ne[2], pT->ne[3], pT->nb[0], pT->nb[1], pT->nb[2], pT->nb[3]);
        };
        vPrintShape(__func__, " dst", vOp);
        vPrintShape(__func__, "src0", vOp->src[0]);
        vPrintShape(__func__, "src1", vOp->src[1]);
        vPrintShape(__func__, "src2", vOp->src[2]);
        LOG_WRN("%s: please report this on github as an issue\n", __func__);
        LOG_WRN("%s: *****************************************************************\n", __func__);
        pCtxClip.vFlashAttnType = cClipFlashAttnTypeDisabled;
        mReserveComputeMeta(pCtxClip, pBatch);
      }
    } else {
      info = mReserveComputeMeta(pCtxClip, pBatch);
      if (!info.vFattn && pCtxClip.vFlashAttnType == cClipFlashAttnTypeEnabled) {
        LOG_WRN(
          "%s: flash attention is not supported by the current backend; falling back to CPU (performance will be "
          "degraded)\n",
          __func__);
      }
    }

    pCtxClip.vIsAllocated = true;  // mark buffers as allocated

    LOG_INF("%s: flash attention is %s\n", __func__,
            (pCtxClip.vFlashAttnType == cClipFlashAttnTypeEnabled) ? "enabled" : "disabled");

    // print ops that are not supported by the GPU backend (if there is one)
    if (pCtxClip.vBackend && pCtxClip.vBackend != pCtxClip.vBackendCpu) {
      std::vector<support_info_op> lUnsupportedOps;
      for (const auto & vOp : info.lOps) {
        if (!vOp.vIsAccel) {
          lUnsupportedOps.push_back(vOp);
        }
      }
      if (!lUnsupportedOps.empty()) {
        LOG_WRN("%s: *****************************************************************\n", __func__);
        LOG_WRN("%s: WARNING: the CLIP graph uses unsupported operators by the backend\n", __func__);
        LOG_WRN("%s:          the performance will be suboptimal                      \n", __func__);
        LOG_WRN("%s:          list of unsupported ops (backend=%s):\n", __func__, ggml_backend_name(pCtxClip.vBackend));
        for (const auto & vOp : lUnsupportedOps) {
          LOG_WRN("%s: %16s: type = %s, ne = [%d %d %d %d]\n", __func__, ggml_op_name(vOp.op->op),
                  ggml_type_name(vOp.op->type), vOp.op->ne[0], vOp.op->ne[1], vOp.op->ne[2], vOp.op->ne[3]);
        }
        LOG_WRN("%s: flash attention is %s\n", __func__,
                (pCtxClip.vFlashAttnType == cClipFlashAttnTypeEnabled) ? "enabled" : "disabled");
        LOG_WRN("%s: please report this on github as an issue\n", __func__);
        LOG_WRN("%s: ref: doc/MANUAL.md\n", __func__);
        LOG_WRN("%s: *****************************************************************\n", __func__);
      }
    }
  }

  // only initialize backend buffers, but do not allocate them yet
  static support_info_graph mReserveComputeMeta(clip_ctx & pCtxClip, const clip_image_f32_batch & pBatch) {
    ggml_cgraph * vGf = fClipGetGraphBuilder(&pCtxClip, pBatch)->mBuild();
    ggml_backend_sched_reserve(pCtxClip.vSched.get(), vGf);

    pCtxClip.dMemCompute.clear();
    for (size_t vI = 0; vI < pCtxClip.lBackendPtrs.size(); ++vI) {
      ggml_backend_t             vBackend = pCtxClip.lBackendPtrs[vI];
      ggml_backend_buffer_type_t vBuft    = pCtxClip.lBackendBuft[vI];
      size_t                     size     = ggml_backend_sched_get_buffer_size(pCtxClip.vSched.get(), vBackend);
      if (size > 1) {
        LOG_INF("%s: %10s compute buffer size = %8.2f MiB\n", __func__, ggml_backend_buft_name(vBuft),
                size / 1024.0 / 1024.0);
      }
      pCtxClip.dMemCompute[ggml_backend_get_device(vBackend)] += size;
    }

    const int cNSplits = ggml_backend_sched_get_n_splits(pCtxClip.vSched.get());
    const int cNNodes  = ggml_graph_n_nodes(vGf);

    LOG_INF("%s: graph splits = %d, nodes = %d\n", __func__, cNSplits, cNNodes);

    support_info_graph vRes{
      /*.fattn    = */ true,
      /*.fattn_op = */ nullptr,
      /*.ops      = */ {},
    };

    // check op support
    for (int vI = 0; vI < ggml_graph_n_nodes(vGf); vI++) {
      ggml_tensor * node = ggml_graph_node(vGf, vI);
      vRes.lOps.push_back({ node, true });
      if (!ggml_backend_supports_op(pCtxClip.vBackend, node)) {
        vRes.lOps.back().vIsAccel = false;
        if (node->op == GGML_OP_FLASH_ATTN_EXT) {
          vRes.vFattn   = false;
          vRes.vFattnOp = node;
        }
      }
    }

    return vRes;
  }

  void mGetBool(const std::string & pKey, bool & pOutput, bool pRequired = true) const {
    const int cI = gguf_find_key(vCtxGguf.get(), pKey.c_str());
    if (cI < 0) {
      if (pRequired) {
        throw std::runtime_error("Key not found: " + pKey);
      }
      return;
    }
    pOutput = gguf_get_val_bool(vCtxGguf.get(), cI);
  }

  void mGetI32(const std::string & pKey, int & pOutput, bool pRequired = true) const {
    const int cI = gguf_find_key(vCtxGguf.get(), pKey.c_str());
    if (cI < 0) {
      if (pRequired) {
        throw std::runtime_error("Key not found: " + pKey);
      }
      return;
    }
    pOutput = gguf_get_val_i32(vCtxGguf.get(), cI);
  }

  void mGetU32(const std::string & pKey, int & pOutput, bool pRequired = true) const {
    const int cI = gguf_find_key(vCtxGguf.get(), pKey.c_str());
    if (cI < 0) {
      if (pRequired) {
        throw std::runtime_error("Key not found: " + pKey);
      }
      return;
    }
    const uint32_t cVal = gguf_get_val_u32(vCtxGguf.get(), cI);
    // sanity check
    if (cVal > (uint32_t) INT32_MAX) {
      throw std::runtime_error(
        fStringFormat("%s: value %u for key '%s' exceeds INT32_MAX\n", __func__, cVal, pKey.c_str()));
    }
    pOutput = (int) cVal;
  }

  void mGetF32(const std::string & pKey, float & pOutput, bool pRequired = true) const {
    const int cI = gguf_find_key(vCtxGguf.get(), pKey.c_str());
    if (cI < 0) {
      if (pRequired) {
        throw std::runtime_error("Key not found: " + pKey);
      }
      return;
    }
    pOutput = gguf_get_val_f32(vCtxGguf.get(), cI);
  }

  void mGetArrF32(const std::string & pKey, std::vector<float> & pOutput, bool pRequired = true) const {
    const int cI = gguf_find_key(vCtxGguf.get(), pKey.c_str());
    if (cI < 0) {
      if (pRequired) {
        throw std::runtime_error("Key not found: " + pKey);
      }
      return;
    }
    if (gguf_get_kv_type(vCtxGguf.get(), cI) != GGUF_TYPE_ARRAY) {
      throw std::runtime_error(fStringFormat("%s: key '%s' is not an array\n", __func__, pKey.c_str()));
    }
    const auto type = gguf_get_arr_type(vCtxGguf.get(), cI);
    if (type != GGUF_TYPE_FLOAT32) {
      throw std::runtime_error(fStringFormat("%s: array '%s' has type %d, expected %d (GGUF_TYPE_FLOAT32)\n", __func__,
                                             pKey.c_str(), type, GGUF_TYPE_FLOAT32));
    }
    const size_t n = gguf_get_arr_n(vCtxGguf.get(), cI);
    if (n > (size_t) std::numeric_limits<int>::max()) {
      throw std::runtime_error(
        fStringFormat("%s: array '%s' is too large (%zu elements)\n", __func__, pKey.c_str(), n));
    }
    pOutput.resize(n);
    const float * vValues = (const float *) gguf_get_arr_data(vCtxGguf.get(), cI);
    for (size_t vJ = 0; vJ < n; ++vJ) {
      pOutput[vJ] = vValues[vJ];
    }
  }

  void mGetString(const std::string & pKey, std::string & pOutput, bool pRequired = true) const {
    const int cI = gguf_find_key(vCtxGguf.get(), pKey.c_str());
    if (cI < 0) {
      if (pRequired) {
        throw std::runtime_error("Key not found: " + pKey);
      }
      return;
    }
    pOutput = std::string(gguf_get_val_str(vCtxGguf.get(), cI));
  }

  void mGetArrInt(const std::string & pKey, std::vector<int> & pOutput, bool pRequired = true) const {
    const int cI = gguf_find_key(vCtxGguf.get(), pKey.c_str());
    if (cI < 0) {
      if (pRequired) {
        throw std::runtime_error("Key not found: " + pKey);
      }
      return;
    }
    if (gguf_get_kv_type(vCtxGguf.get(), cI) != GGUF_TYPE_ARRAY) {
      throw std::runtime_error(fStringFormat("%s: key '%s' is not an array\n", __func__, pKey.c_str()));
    }
    const auto type = gguf_get_arr_type(vCtxGguf.get(), cI);
    if (type != GGUF_TYPE_INT32) {
      throw std::runtime_error(fStringFormat("%s: array '%s' has type %d, expected %d (GGUF_TYPE_INT32)\n", __func__,
                                             pKey.c_str(), type, GGUF_TYPE_INT32));
    }
    const size_t n = gguf_get_arr_n(vCtxGguf.get(), cI);
    if (n > (size_t) std::numeric_limits<int>::max()) {
      throw std::runtime_error(
        fStringFormat("%s: array '%s' is too large (%zu elements)\n", __func__, pKey.c_str(), n));
    }
    pOutput.resize(n);
    const int32_t * vValues = (const int32_t *) gguf_get_arr_data(vCtxGguf.get(), cI);
    for (size_t vJ = 0; vJ < n; ++vJ) {
      pOutput[vJ] = vValues[vJ];
    }
  }

  static void mSetLlavaUhdResCandidates(clip_model & pModel, const int pMaxPatchesPerSide) {
    auto & vHparams = pModel.vHparams;
    for (int vX = 1; vX <= pMaxPatchesPerSide; vX++) {
      for (int vY = 1; vY <= pMaxPatchesPerSide; vY++) {
        if (vX == 1 && vY == 1) {
          continue;  // skip the first point
        }
        vHparams.lImageResCandidates.push_back(clip_image_size{
          vX * vHparams.vImageSize,
          vY * vHparams.vImageSize,
        });
      }
    }
  }

  static void mSetInternvlDhrResCandidates(clip_model & pModel) {
    auto & vHparams = pModel.vHparams;
    int    vMinNum  = vHparams.vPreprocMinTiles;
    int    vMaxNum  = vHparams.vPreprocMaxTiles;
    if (vMinNum < 1) {
      return;  // avoid  divide by 0
    }
    for (int vA = vMinNum; vA <= vMaxNum; ++vA) {
      int vBLo = (vMinNum + vA - 1) / vA;
      int vBHi = vMaxNum / vA;
      vBLo     = std::max(vBLo, vMinNum);
      vBHi     = std::min(vBHi, vMaxNum);
      for (int vB = vBLo; vB <= vBHi; ++vB) {
        vHparams.lImageResCandidates.push_back(clip_image_size{
          vA * vHparams.vImageSize,
          vB * vHparams.vImageSize,
        });
      }
    }
  }
};

struct clip_init_result fClipInit(const char * pFname, struct clip_context_params pCtxParams) {
  clip_ctx * vCtxVision   = nullptr;
  clip_ctx * vCtxAudio    = nullptr;
  clip_ctx * vCtxGenAudio = nullptr;

  try {
    clip_model_loader vLoader(pFname,
                              /* skip_tensors */ false, pCtxParams.vProgressCallback,
                              pCtxParams.vProgressCallbackUserData);
    bool              vSkipAudio = false;

    if (vLoader.vHasVision) {
      vCtxVision = new clip_ctx(pCtxParams);
      vLoader.mLoadHparams(vCtxVision->vModel, cClipModalityVision);
      vLoader.mLoadTensors(*vCtxVision);
      vLoader.mInitCtx(*vCtxVision);
      if (pCtxParams.vWarmup) {
        vLoader.mWarmup(*vCtxVision);
      }

      // TODO: we don't support audio for Gemma 3N, but GGUF contains audio tensors
      // we can remove this check when we implement audio support for Gemma 3N
      vSkipAudio = vCtxVision->vModel.vProjType == cProjectorTypeGemma3nv;
    }

    if (vLoader.vHasAudio && !vSkipAudio) {
      vCtxAudio = new clip_ctx(pCtxParams);
      vLoader.mLoadHparams(vCtxAudio->vModel, cClipModalityAudio);
      vLoader.mLoadTensors(*vCtxAudio);
      vLoader.mInitCtx(*vCtxAudio);
      if (pCtxParams.vWarmup) {
        vLoader.mWarmup(*vCtxAudio);
      }
    }

    if (vLoader.vHasGenAudio) {
      vCtxGenAudio = new clip_ctx(pCtxParams);
      vLoader.mLoadHparams(vCtxGenAudio->vModel, cClipModalityGenAudio);
      vLoader.mLoadTensors(*vCtxGenAudio);
      // TODO: fix warmup
      vCtxGenAudio->lBufComputeMeta.resize(vCtxGenAudio->vMaxNodes * ggml_tensor_overhead() + ggml_graph_overhead());
    }

  } catch (const std::exception & vE) {
    LOG_ERR("%s: failed to load model '%s': %s\n", __func__, pFname, vE.what());

    delete vCtxVision;
    delete vCtxAudio;
    delete vCtxGenAudio;

    return { nullptr, nullptr, nullptr };
  }

  return { vCtxVision, vCtxAudio, vCtxGenAudio };
}

struct clip_cap fClipGetCap(const char * pFname) {
  clip_cap          vRes;
  clip_model_loader vLoader(pFname, /* skip_tensors= */ true);
  vRes.vHasVision = vLoader.vHasVision;
  vRes.vHasAudio  = vLoader.vHasAudio;
  return vRes;
}

int fClipGetImageMaxTokens(const clip_ctx * pCtx) {
  const auto & vHparams = pCtx->vModel.vHparams;
  if (pCtx->mProjType() == cProjectorTypeDeepseek4v) {
    return vHparams.vDsv4MaxNToken;
  }
  if (vHparams.vImageMaxPixels <= 0) {
    return -1;
  }
  const int cPatchArea = vHparams.vPatchSize * vHparams.vPatchSize * vHparams.vNMerge * vHparams.vNMerge;
  return vHparams.vImageMaxPixels / cPatchArea;
}

void fClipFree(clip_ctx * pCtx) {
  if (pCtx == nullptr) {
    return;
  }
  delete pCtx;
}

const char * fClipPatchMergeType(const struct clip_ctx * pCtx) {
  return pCtx->vModel.vHparams.vMmPatchMergeType == cPatchMergeSpatialUnpad ? "spatial_unpad" : "flat";
}

int fClipNOutputTokensX(const clip_ctx * pCtx, const clip_image_f32 * pImg) {
  const auto & vParams = pCtx->vModel.vHparams;
  const int    cNTotal = fClipNOutputTokens(pCtx, pImg);
  const auto & vProj   = pCtx->mProjType();
  switch (vProj) {
    case cProjectorTypeQwen3vl:

      return (pImg->mNx() / vParams.vPatchSize) / 2;
    default:
      break;
  }
  return cNTotal;
}

int fClipNOutputTokensY(const clip_ctx * pCtx, const clip_image_f32 * pImg) {
  const auto & vParams = pCtx->vModel.vHparams;
  const auto & vProj   = pCtx->mProjType();
  switch (vProj) {
    case cProjectorTypeQwen3vl:

      return (pImg->mNy() / vParams.vPatchSize) / 2;
    default:
      break;
  }
  return 1;
}

int fClipNOutputTokens(const clip_ctx * pCtx, const clip_image_f32 * pImg) {
  const auto & vParams = pCtx->vModel.vHparams;

  // for models with fixed size image, the input image is already pre-processed and resized to square
  int vPatchSize = vParams.vPatchSize;
  int vNPatches  = (pImg->mNx() / vPatchSize) * (pImg->mNy() / vPatchSize);

  projector_type vProj = pCtx->mProjType();

  switch (vProj) {
    case cProjectorTypeQwen3vl:

      {
        // dynamic size (2 conv, so double patch size)
        int vXPatch = pImg->mNx() / (vParams.vPatchSize * 2);
        int vYPatch = pImg->mNy() / (vParams.vPatchSize * 2);
        vNPatches   = vXPatch * vYPatch;
      }
      break;
    default:
      GGML_ABORT("unsupported projector type");
  }

  vNPatches += (int) fClipSuffixRows(pImg->vSuffixType).size();

  return vNPatches;
}

bool fClipImageEncode(struct clip_ctx *      pCtx,
                      int                    pNThreads,
                      const clip_image_f32 * pImg,
                      std::vector<float> &   pOutVec) {
  clip_image_f32_batch vImgs;
  clip_image_f32       vImgCopy = *pImg;
  vImgs.lEntries.push_back(std::move(vImgCopy));

  return fClipImageBatchEncode(pCtx, pNThreads, &vImgs, pOutVec);
}

bool fClipImageBatchEncode(clip_ctx *                   pCtx,
                           int                          pNThreads,
                           const clip_image_f32_batch * pImgsCPtr,
                           std::vector<float> &         pOutBatchEmbd) {
  clip_encode_params vParams;
  vParams.vImgs     = pImgsCPtr;
  vParams.vNThreads = pNThreads;
  vParams.lOutEmbd  = &pOutBatchEmbd;

  return fClipEncode(pCtx, &vParams);
}

// persisted state slots of the gen-audio decoder, per pipeline

bool fClipEncode(struct clip_ctx * pCtx, struct clip_encode_params * pParams) {
  const clip_image_f32_batch & vImgs      = *pParams->vImgs;
  int                          vNBatchCur = vImgs.lEntries.size();

  // [QWEN_VIDEO] for video models, the batch dimension is used as temporal dimension for merged frames
  if (!pCtx->vSupportBatch && vNBatchCur > fClipModelNTemporalMerge(pCtx)) {
    LOG_ERR("%s: batch size %d exceeds maximum supported batch/temporal-merge size %d\n", __func__, vNBatchCur,
            fClipModelNTemporalMerge(pCtx));
    return false;
  }

  // if buffers are not allocated, we need to do a warmup run to allocate them
  if (!pCtx->vIsAllocated) {
    clip_model_loader::mWarmup(*pCtx, *pParams->vImgs);
  }

  if (pParams->vSeed != pCtx->vRngSeed) {
    pCtx->vRngSeed = pParams->vSeed;
    pCtx->vRng.seed(pParams->vSeed == UINT32_MAX ? std::random_device{}() : pParams->vSeed);
  }

  // build the inference graph
  ggml_backend_sched_reset(pCtx->vSched.get());
  ggml_cgraph * vGf = fClipGetGraphBuilder(pCtx, vImgs, pParams)->mBuild();
  if (!ggml_backend_sched_alloc_graph(pCtx->vSched.get(), vGf)) {
    LOG_ERR("%s: failed to allocate compute graph\n", __func__);
    return false;
  }

  // set inputs
  const auto & vModel   = pCtx->vModel;
  const auto & vHparams = vModel.vHparams;

  const int cImageSizeWidth  = vImgs.lEntries[0].mNx();
  const int cImageSizeHeight = vImgs.lEntries[0].mNy();

  const int cPatchSize  = vHparams.vPatchSize;
  const int cNumPatches = ((cImageSizeWidth / cPatchSize) * (cImageSizeHeight / cPatchSize));
  const int cNPos       = cNumPatches + (vModel.vClassEmbedding ? 1 : 0);
  const int cPosW       = cImageSizeWidth / cPatchSize;
  const int cPosH       = cImageSizeHeight / cPatchSize;

  auto vGetInpTensor = [&vGf](const char * name) {
    ggml_tensor * vInp = ggml_graph_get_tensor(vGf, name);
    if (vInp == nullptr) {
      GGML_ABORT("Failed to get tensor %s", name);
    }
    if (!(vInp->flags & GGML_TENSOR_FLAG_INPUT)) {
      GGML_ABORT("Tensor %s is not an input tensor", name);
    }
    return vInp;
  };

  auto vSetInputF32 = [&vGetInpTensor](const char * name, const std::vector<float> & pValues) {
    ggml_tensor * vCur = vGetInpTensor(name);
    GGML_ASSERT(vCur->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(vCur) == (int64_t) pValues.size());
    ggml_backend_tensor_set(vCur, pValues.data(), 0, ggml_nbytes(vCur));
  };

  // mask from build_inp_attn_mask(), f16 if flash attn is enabled
  auto vSetInputAttnMask = [&vGetInpTensor](const std::vector<float> & pValues, int pIdx = 0) {
    ggml_tensor * vCur = vGetInpTensor(fGetAttnMaskName(pIdx).c_str());
    GGML_ASSERT(ggml_nelements(vCur) == (int64_t) pValues.size());
    if (vCur->type == GGML_TYPE_F16) {
      std::vector<ggml_fp16_t> lValuesF16(pValues.size());
      ggml_fp32_to_fp16_row(pValues.data(), lValuesF16.data(), pValues.size());
      ggml_backend_tensor_set(vCur, lValuesF16.data(), 0, ggml_nbytes(vCur));
    } else {
      GGML_ASSERT(vCur->type == GGML_TYPE_F32);
      ggml_backend_tensor_set(vCur, pValues.data(), 0, ggml_nbytes(vCur));
    }
  };

  auto vSetInputI32 = [&vGetInpTensor](const char * name, std::vector<int32_t> & pValues) {
    ggml_tensor * vCur = vGetInpTensor(name);
    GGML_ASSERT(vCur->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_nelements(vCur) == (int64_t) pValues.size());
    ggml_backend_tensor_set(vCur, pValues.data(), 0, ggml_nbytes(vCur));
  };

  // upload the decoder state from the previous call, or zero-fill on a cold start

  // rope positions and attention mask of the mimi transformers (pocket-tts).
  // the mask is causal with a sliding window, see _build_attention_mask() in the reference

  // set input pixel values
  if (!vImgs.vIsAudio) {
    size_t vNelem = 0;
    for (const auto & vImg : vImgs.lEntries) {
      vNelem += vImg.mNx() * vImg.mNy() * 3;
    }
    std::vector<float> lInpRaw(vNelem);

    // layout of data (note: the channel dim is unrolled to better visualize the layout):
    //
    // ┌──W──┐
    // │     H │  channel = R
    // ├─────┤ │
    // │     H │  channel = G
    // ├─────┤ │
    // │     H │  channel = B
    // └─────┘ │
    //   ──────┘ x B

    // IMPORTANT: [QWEN_VIDEO] the batch dim is currently used for temporal dim in Qwen-VL models
    // All entries must have the same spatial size (enforced by can_batch_with() during merging)
    {
      const int cNx = vImgs.lEntries[0].mNx();
      const int cNy = vImgs.lEntries[0].mNy();
      const int n   = cNx * cNy;

      for (int vB = 0; vB < vNBatchCur; vB++) {
        LOG_DBG("%s: copying image %d/%d to input buffer (nx=%d, ny=%d)\n", __func__, vB + 1, vNBatchCur, cNx, cNy);
        const auto & lBuf        = vImgs.lEntries[vB].mGetRoBuf();
        float *      vBatchEntry = lInpRaw.data() + vB * (3 * n);
        for (int vY = 0; vY < cNy; vY++) {
          for (int vX = 0; vX < cNx; vX++) {
            size_t vBaseSrc               = 3 * (vY * cNx + vX);
            size_t vBaseDst               = vY * cNx + vX;
            vBatchEntry[vBaseDst]         = lBuf[vBaseSrc];
            vBatchEntry[1 * n + vBaseDst] = lBuf[vBaseSrc + 1];
            vBatchEntry[2 * n + vBaseDst] = lBuf[vBaseSrc + 2];
          }
        }
      }
    }
    vSetInputF32("inp_raw", lInpRaw);

  } else if (pParams->vGenProcess != cClipGenProcessGenWav) {
    // audio input. GEN_WAV is not here: it takes codes or feats, set in the switch below
    GGML_ASSERT(vImgs.lEntries.size() == 1);

    const auto & vMelInp = vImgs.lEntries[0];
    const auto & lBuf    = vMelInp.mGetRoBuf();
    const int    cNStep  = vMelInp.mNx();
    const int    cNMel   = vMelInp.mNy();
    GGML_ASSERT((size_t) cNStep * cNMel == lBuf.size());

    vSetInputF32("inp_raw", lBuf);
  }

  // set input per projector
  switch (pCtx->vModel.vProjType) {
    case cProjectorTypeQwen3vl:

      {
        const int        cMergeRatio = vHparams.vNMerge;
        const int        cPw         = cImageSizeWidth / cPatchSize;
        const int        cPh         = cImageSizeHeight / cPatchSize;
        std::vector<int> lPositions(cNPos * 4);
        int              vPtr = 0;
        for (int vY = 0; vY < cPh; vY += cMergeRatio) {
          for (int vX = 0; vX < cPw; vX += cMergeRatio) {
            for (int vDy = 0; vDy < 2; vDy++) {
              for (int vDx = 0; vDx < 2; vDx++) {
                lPositions[vPtr]                   = vY + vDy;
                lPositions[cNumPatches + vPtr]     = vX + vDx;
                lPositions[2 * cNumPatches + vPtr] = vY + vDy;
                lPositions[3 * cNumPatches + vPtr] = vX + vDx;
                vPtr++;
              }
            }
          }
        }

        vSetInputI32("positions", lPositions);
      }
      break;
    default:
      GGML_ABORT("Unknown projector type");
  }

  // ggml_backend_cpu_set_n_threads(ctx->backend_cpu, n_threads);
  ggml_backend_dev_t vDev = ggml_backend_get_device(pCtx->vBackendCpu);
  ggml_backend_reg_t vReg = vDev ? ggml_backend_dev_backend_reg(vDev) : nullptr;
  if (vReg) {
    auto vGgmlBackendSetNThreadsFn =
      (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(vReg, "ggml_backend_set_n_threads");
    if (vGgmlBackendSetNThreadsFn) {
      vGgmlBackendSetNThreadsFn(pCtx->vBackendCpu, pParams->vNThreads);
    }
  }

  auto vStatus = ggml_backend_sched_graph_compute(pCtx->vSched.get(), vGf);
  if (vStatus != GGML_STATUS_SUCCESS) {
    LOG_ERR("%s: ggml_backend_sched_graph_compute failed with error %d\n", __func__, vStatus);
    return false;
  }

  // the last node is the embedding tensor, code2wav has no out_embd
  ggml_tensor * vEmbeddings = pParams->lOutEmbd ? ggml_graph_node(vGf, -1) : nullptr;

  if (vEmbeddings != nullptr) {
    // sanity check (assuming that all images in batch have the same number of tokens, so we only check the first one)
    const int cNTokensOut         = vEmbeddings->ne[1];
    const int cExpectedNTokensOut = fClipNOutputTokens(pCtx, &vImgs.lEntries[0]);
    if (cNTokensOut != cExpectedNTokensOut) {
      LOG_ERR("%s: expected output %d tokens, got %d\n", __func__, cExpectedNTokensOut, cNTokensOut);
      GGML_ABORT("Invalid number of output tokens");
    }

    LOG_DBG("%s: output embedding shape [%d, %d, %d]\n", __func__, (int) vEmbeddings->ne[0], (int) vEmbeddings->ne[1],
            (int) vEmbeddings->ne[2]);

    // copy output to user buffer if provided
    // if output is empty, skip the copy
    auto & lOutBatchEmbd = *pParams->lOutEmbd;
    if (!lOutBatchEmbd.empty()) {
      if (lOutBatchEmbd.size() != (size_t) ggml_nelements(vEmbeddings)) {
        LOG_ERR("%s: output buffer has %zu elements but expected %zu\n", __func__, lOutBatchEmbd.size(),
                (size_t) ggml_nelements(vEmbeddings));
        GGML_ABORT("Output buffer size mismatch");
      }
      ggml_backend_tensor_get(vEmbeddings, lOutBatchEmbd.data(), 0, ggml_nbytes(vEmbeddings));
    } else {
      LOG_WRN("%s: output buffer is empty, skipping copy\n", __func__);
    }
  }

  //
  // for audio gen models
  //

  // optional outputs: a pipeline yields codes or feats, and not all have an eos head
  if (pParams->lOutCodes != nullptr) {
    ggml_tensor * vCodes = ggml_graph_get_tensor(vGf, "out_codes");
    if (vCodes != nullptr) {
      auto & lOutCodes = *pParams->lOutCodes;
      lOutCodes.resize(ggml_nelements(vCodes));
      ggml_backend_tensor_get(vCodes, lOutCodes.data(), 0, ggml_nbytes(vCodes));
    }
  }
  if (pParams->lOutFeats != nullptr) {
    ggml_tensor * vFeats = ggml_graph_get_tensor(vGf, "out_feats");
    if (vFeats != nullptr) {
      auto & lOutFeats = *pParams->lOutFeats;
      lOutFeats.resize(ggml_nelements(vFeats));
      ggml_backend_tensor_get(vFeats, lOutFeats.data(), 0, ggml_nbytes(vFeats));
    }
  }
  if (pParams->vOutIsEos != nullptr) {
    ggml_tensor * vEos = ggml_graph_get_tensor(vGf, "out_eos_score");
    if (vEos != nullptr) {
      GGML_ASSERT(ggml_nelements(vEos) == 1);
      float vScore = 0.0f;
      ggml_backend_tensor_get(vEos, &vScore, 0, sizeof(float));
      *pParams->vOutIsEos = vScore > vHparams.vGenEosThreshold;
    }
  }
  if (pParams->lOutAudio != nullptr) {
    ggml_tensor * vAudio = ggml_graph_get_tensor(vGf, "out_audio");
    if (vAudio == nullptr) {
      GGML_ABORT("out_audio requested but graph has no \"out_audio\" tensor");
    }
    auto & lOutAudio = *pParams->lOutAudio;
    lOutAudio.resize(ggml_nelements(vAudio));
    ggml_backend_tensor_get(vAudio, lOutAudio.data(), 0, ggml_nbytes(vAudio));

    // drop the tail audio that comes from the code-0 rear padding
    const int64_t cNCodes   = pParams->lCodes ? vModel.vGenCodeHeadW->ne[2] + 1 : 0;
    const int64_t cNFramesW = vHparams.vWavTfmSwa;
    const int64_t cNFrames  = pParams->lCodes ? (int64_t) pParams->lCodes->size() / cNCodes : cNFramesW;
    if (cNFrames < cNFramesW) {
      const size_t cHop = lOutAudio.size() / cNFramesW;
      lOutAudio.resize((size_t) cNFrames * cHop);
    }
  }

  //
  // Debug: dump final embeddings if MTMD_DEBUG_EMBEDDINGS is set
  //

  if (pCtx->vDebugOutputEmbeddings && vEmbeddings != nullptr) {
    const int64_t      cNEmbd   = vEmbeddings->ne[0];
    const int64_t      cNTokens = vEmbeddings->ne[1];
    std::vector<float> lEmbData(ggml_nelements(vEmbeddings));
    ggml_backend_tensor_get(vEmbeddings, lEmbData.data(), 0, ggml_nbytes(vEmbeddings));

    LOG_INF("\n=== MTMD_DEBUG_EMBEDDINGS ===\n");
    LOG_INF("Shape: [%lld, %lld]\n", (long long) cNEmbd, (long long) cNTokens);

    // TEMP debugging (parity validation), will be removed before merge
    // when the env var holds a path, dump the raw data: [int32 n_tokens][int32 n_embd][f32 data]
    const char * vDumpPath = std::getenv("MTMD_DEBUG_EMBEDDINGS");
    if (vDumpPath && strcmp(vDumpPath, "1") != 0) {
      FILE * vF = fopen(vDumpPath, "wb");
      if (vF) {
        const int32_t aHdr[2] = { (int32_t) cNTokens, (int32_t) cNEmbd };
        fwrite(aHdr, sizeof(aHdr), 1, vF);
        fwrite(lEmbData.data(), sizeof(float), lEmbData.size(), vF);
        fclose(vF);
      }
    }

    // Print first few values of first token
    LOG_INF("Token 0 (first 16 values): ");
    for (int vI = 0; vI < std::min((int64_t) 16, cNEmbd); vI++) {
      LOG_INF("%.6f ", lEmbData[vI]);
    }
    LOG_INF("\n");

    // Print last few values of first token
    if (cNEmbd > 16) {
      LOG_INF("Token 0 (last 16 values):  ");
      for (int64_t vI = cNEmbd - 16; vI < cNEmbd; vI++) {
        LOG_INF("%.6f ", lEmbData[vI]);
      }
      LOG_INF("\n");
    }

    // Compute and print statistics
    float vSum = 0.0f, vSumSq = 0.0f, vMinVal = lEmbData[0], vMaxVal = lEmbData[0];
    for (size_t vI = 0; vI < lEmbData.size(); vI++) {
      vSum += lEmbData[vI];
      vSumSq += lEmbData[vI] * lEmbData[vI];
      vMinVal = std::min(vMinVal, lEmbData[vI]);
      vMaxVal = std::max(vMaxVal, lEmbData[vI]);
    }
    float vMean     = vSum / lEmbData.size();
    float vVariance = (vSumSq / lEmbData.size()) - (vMean * vMean);
    LOG_INF("Stats: mean=%.6f, std=%.6f, min=%.6f, max=%.6f, sum=%.6f\n", vMean, sqrtf(vVariance), vMinVal, vMaxVal,
            vSum);
    LOG_INF("=== END MTMD_DEBUG_EMBEDDINGS ===\n\n");
  }

  return true;
}

int fClipNMmprojEmbd(const struct clip_ctx * pCtx) {
  switch (pCtx->vModel.vProjType) {
    case cProjectorTypeQwen3vl:
      // main path + deepstack paths
      return pCtx->vModel.vMm1B->ne[0] * (1 + pCtx->vModel.vNDeepstackLayers);
    default:
      GGML_ABORT("Unknown projector type");
  }
}

bool fClipIsLlava(const struct clip_ctx * pCtx) {
  return pCtx->vModel.vHparams.vHasLlavaProjector;
}

bool fClipHasVisionEncoder(const struct clip_ctx * pCtx) {
  return pCtx->vModel.vModality == cClipModalityVision;
}

bool fClipHasAudioEncoder(const struct clip_ctx * pCtx) {
  return pCtx->vModel.vModality == cClipModalityAudio;
}

bool fClipSupportBatch(const struct clip_ctx * pCtx) {
  return pCtx->vSupportBatch;
}

// TODO @ngxson : this is no longer correct with mtmd_batch API
// this was only meant to be used by qwen-vl-based models, to fuse 2 input images into one (qwen-vl video support)
// this logic should be refactored in near future to distinctly handle "merge frames" and "batching"
int fClipModelNTemporalMerge(const struct clip_ctx * pCtx) {
  switch (pCtx->mProjType()) {
    case cProjectorTypeQwen3vl:

      return 2;
    default:
      return 1;
  }
}

//
// API used internally with mtmd
//

projector_type fClipGetProjectorType(const struct clip_ctx * pCtx) {
  return pCtx->mProjType();
}

const clip_hparams * fClipGetHparams(const struct clip_ctx * pCtx) {
  return &pCtx->vModel.vHparams;
}

std::map<ggml_backend_dev_t, size_t> fClipGetMemUsage(const struct clip_ctx * pCtx) {
  std::map<ggml_backend_dev_t, size_t> dResult = pCtx->dMemUsage;
  for (auto & [dev, size] : pCtx->dMemCompute) {
    dResult[dev] += size;
  }
  return dResult;
}

//
// API for debugging
//

void fClipSetDebugOutputEmbeddings(clip_ctx * pCtx, bool pEnable) {
  pCtx->vDebugOutputEmbeddings = pEnable;
}
