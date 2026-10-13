#pragma once

#include "clip-model.h"
#include "ggml.h"

#include <string>
#include <vector>

#define MTMD_INTERNAL_HEADER

struct mtmd_image_preproc_out {
  std::vector<clip_image_f32> lEntries;
  // grid size is required for llava-uhd style models

  clip_image_f32 vOverview;  // overview image (downscaled image)
  int            vGridX = 0;
  int            vGridY = 0;

  void mAppend(const clip_hparams & pHparams, const clip_image_u8 & pImg, bool pNormalized = true);
  void mAppend(const clip_hparams & pHparams, const std::vector<clip_image_u8> & pImgs, bool pNormalized = true);
  void mAppend(const clip_hparams & pHparams, clip_image_f32 & pImg, bool pNormalized = true);

  void mAppendOverview(const clip_hparams & pHparams, const clip_image_u8 & pImg, bool pNormalized = true);

  bool mHasOverview() const { return vOverview.mNx() > 0 || vOverview.mNy() > 0; }
};

// base class, models must inherit from this class
struct mtmd_image_preprocessor {
  const clip_hparams & vHparams;

  mtmd_image_preprocessor(const clip_ctx * pCtx) : vHparams(*fClipGetHparams(pCtx)) {}

  virtual ~mtmd_image_preprocessor()                                           = default;
  virtual mtmd_image_preproc_out mPreprocess(const clip_image_u8 & pImg) const = 0;
};

/**
 * implementation of LLaVA-UHD:
 *  - https://arxiv.org/pdf/2403.11703
 *  - https://github.com/thunlp/LLaVA-UHD
 *  - https://github.com/thunlp/LLaVA-UHD/blob/302301bc2175f7e717fb8548516188e89f649753/llava_uhd/train/llava-uhd/slice_logic.py#L118
 *
 * overview:
 *   - an image always have a single overview (downscaled image)
 *   - an image can have 0 or multiple slices, depending on the image size
 *   - each slice can then be considered as a separate image
 *
 * note: the term "slice" and "tile" are used interchangeably
 *
 * for example:
 *
 * [overview] --> [slice 1] --> [slice 2]
 *           |                |
 *           +--> [slice 3] --> [slice 4]
 *
 * NOTE: for the ordering of overview, set "ov_img_first" on the mtmd_context
 */
struct mtmd_image_preprocessor_llava_uhd : mtmd_image_preprocessor {
  mtmd_image_preprocessor_llava_uhd(const clip_ctx * pCtx) : mtmd_image_preprocessor(pCtx) {}

  mtmd_image_preproc_out mPreprocess(const clip_image_u8 & pImg) const override;

  struct slice_coordinates {
    int             vX;
    int             vY;
    clip_image_size size;
  };

  struct slice_instructions {
    clip_image_size                vOverviewSize;  // size of downscaled image
    clip_image_size                vRefinedSize;  // size of image right before slicing (must be multiple of slice size)
    clip_image_size                vGridSize;     // grid_size.width * grid_size.height = number of slices
    std::vector<slice_coordinates> lSlices;
  };

  virtual slice_instructions mGetSliceInstructions(const clip_image_size & pOriginalSize) const;

  struct slice_output {
    clip_image_u8              vOverview;
    std::vector<clip_image_u8> lSlices;
  };

  slice_output mSliceImage(const clip_image_u8 & pImg, const slice_instructions & pInst) const;

protected:
  // align slices to a multiple of the merger factor (integer merger tokens per slice)
  virtual int mGetSliceAlign() const {
    const int cMerge = vHparams.vNMerge > 0 ? vHparams.vNMerge : 1;
    return vHparams.vPatchSize * cMerge;
  }

  // rounding for snapping a length to a multiple of the align size
  virtual int mAlignRound(double pV) const { return static_cast<int>(std::round(pV)); }

  clip_image_size mGetBestResize(const clip_image_size & pOriginalSize,
                                 int                     pScaleResolution,
                                 int                     pPatchSize,
                                 bool                    pAllowUpscale = false) const;

  /**
     * Selects the best resolution from a list of possible resolutions based on the original size.
     *
     * For example, when given a list of resolutions:
     *  - 100x100
     *  - 200x100
     *  - 100x200
     *  - 200x200
     *
     * And an input image of size 111x200, then 100x200 is the best fit (least wasted resolution).
     *
     * @param original_size The original size of the image
     * @param possible_resolutions A list of possible resolutions
     * @return The best fit resolution
     */
  clip_image_size mSelectBestResolution(const clip_image_size &              pOriginalSize,
                                        const std::vector<clip_image_size> & pPossibleResolutions) const;

private:
  clip_image_size mResizeMaintainAspectRatio(const clip_image_size & pOrig, const clip_image_size & pTargetMax) const;
  int             mEnsureDivide(int length, int pPatchSize) const;
  clip_image_size mGetRefineSize(const clip_image_size & pOriginalSize,
                                 const clip_image_size & pGrid,
                                 int                     pScaleResolution,
                                 int                     pPatchSize,
                                 bool                    pAllowUpscale = false) const;
  clip_image_size mGetBestGrid(const int pMaxSliceNums, const int pMultiple, const float pLogRatio) const;
};

// downscale or upscale the input image to fixed size
struct mtmd_image_preprocessor_fixed_size : mtmd_image_preprocessor {
  mtmd_image_preprocessor_fixed_size(const clip_ctx * pCtx) : mtmd_image_preprocessor(pCtx) {}

  mtmd_image_preproc_out mPreprocess(const clip_image_u8 & pImg) const override;
};

// resize image to multiple of patch_size*n_merge, while preserving aspect ratio
// if image_resize_pad is true, the resized image will be padded, otherwise it will be either stretched or center-cropped depending on image_resize_pad
// this is used by models with native support for dynamic image size, for example: Qwen-VL, Pixtral, Kimi-VL, etc
struct mtmd_image_preprocessor_dyn_size : mtmd_image_preprocessor {
  mtmd_image_preprocessor_dyn_size(const clip_ctx * pCtx) : mtmd_image_preprocessor(pCtx) {}

  mtmd_image_preproc_out mPreprocess(const clip_image_u8 & pImg) const override;
};

// GLM 5.3 flash, similar to dyn_size, but each edge is aligned up to patch_size*n_merge, and max budget is met by a search over the height with the width scaled proportionally
struct mtmd_image_preprocessor_glm5v : mtmd_image_preprocessor {
  mtmd_image_preprocessor_glm5v(const clip_ctx * pCtx) : mtmd_image_preprocessor(pCtx) {}

  mtmd_image_preproc_out mPreprocess(const clip_image_u8 & pImg) const override;
};

// similar to mtmd_image_preprocessor_dyn_size, but resize the image to have longest edge equal to hparams.image_longest_edge, while preserving aspect ratio
struct mtmd_image_preprocessor_longest_edge : mtmd_image_preprocessor {
  mtmd_image_preprocessor_longest_edge(const clip_ctx * pCtx) : mtmd_image_preprocessor(pCtx) {}

  mtmd_image_preproc_out mPreprocess(const clip_image_u8 & pImg) const override;
};

// ref: inference/image_processor.py in the HF repo (DeepSeek-V4-Flash-Vision)
struct mtmd_image_preprocessor_deepseek4v : mtmd_image_preprocessor {
  mtmd_image_preprocessor_deepseek4v(const clip_ctx * pCtx) : mtmd_image_preprocessor(pCtx) {}

  mtmd_image_preproc_out mPreprocess(const clip_image_u8 & pImg) const override;

private:
  struct grid_info {
    int vNLlmH;
    int vNLlmW;
    int vNTokens;  // token count of the block (incl. newline/pad rows and start/end, excl. lead pads)
  };

  static grid_info mGridTokens(int pBestHeight, int pBestWidth, int pPatchSize, int pR);
  static void
  mSolveResizeRatio(int pHeight, int pWidth, int pP, int pR, int pMaxNToken, int & pBestHeight, int & pBestWidth);
  static void mSafeResize(int pHeight, int pWidth, int & pBestHeight, int & pBestWidth, int pP, int pR, int pMaxNToken);
};

// custom llava-uhd slicing logic for MiniCPM-V
struct mtmd_image_preprocessor_minicpmv : mtmd_image_preprocessor_llava_uhd {
  using mtmd_image_preprocessor_llava_uhd::mtmd_image_preprocessor_llava_uhd;
  slice_instructions mGetSliceInstructions(const clip_image_size & pOriginalSize) const override;

protected:
  // always patch_size * 4, even in 4x mode (the 2x2 vit_merger slot stays)
  int mGetSliceAlign() const override { return vHparams.vPatchSize * 4; }

  // Python's round() breaks ties to even, unlike std::round
  int mAlignRound(double pV) const override {
    const double cFl   = std::floor(pV);
    const double cDiff = pV - cFl;
    if (cDiff > 0.5) {
      return static_cast<int>(cFl) + 1;
    }
    if (cDiff < 0.5) {
      return static_cast<int>(cFl);
    }
    const int cLo = static_cast<int>(cFl);
    return (cLo % 2 == 0) ? cLo : cLo + 1;
  }
};

// custom llava-uhd slicing logic for LFM2
// ref: https://github.com/huggingface/transformers/blob/v5.1.0/src/transformers/models/lfm2_vl/image_processing_lfm2_vl_fast.py
struct mtmd_image_preprocessor_lfm2 : mtmd_image_preprocessor_llava_uhd {
  // ref: https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B/blob/main/processor_config.json
  static constexpr int   cMinTiles           = 2;
  static constexpr int   cMaxTiles           = 10;
  static constexpr float cMaxPixelsTolerance = 2.0f;
  static constexpr int   cTileSize           = 512;

  using mtmd_image_preprocessor_llava_uhd::mtmd_image_preprocessor_llava_uhd;
  mtmd_image_preproc_out mPreprocess(const clip_image_u8 & pImg) const override;
  slice_instructions     mGetSliceInstructions(const clip_image_size & pOriginalSize) const override;

  static bool mShouldTile(const clip_hparams & pHparams, const clip_image_size & pOriginalSize);

private:
  clip_image_size              mFindClosestAspectRatio(float                                pAspectRatio,
                                                       const std::vector<clip_image_size> & pTargetRatios,
                                                       int                                  pWidth,
                                                       int                                  pHeight) const;
  std::vector<clip_image_size> mGetTargetRatios() const;
  clip_image_size              mGetGridLayout(int pHeight, int pWidth) const;
};

struct mtmd_image_preprocessor_idefics3 : mtmd_image_preprocessor_llava_uhd {
  mtmd_image_preprocessor_idefics3(const clip_ctx * pCtx) : mtmd_image_preprocessor_llava_uhd(pCtx) {}

  mtmd_image_preproc_out mPreprocess(const clip_image_u8 & pImg) const override;
};

struct mtmd_image_preprocessor_internvl : mtmd_image_preprocessor_llava_uhd {
  mtmd_image_preprocessor_internvl(const clip_ctx * pCtx) : mtmd_image_preprocessor_llava_uhd(pCtx) {}

  mtmd_image_preproc_out mPreprocess(const clip_image_u8 & pImg) const override;
};

// stretch the image to a grid of tiles, add a thumbnail if there is more than 1 tile
// ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/cohere2_vision/image_processing_cohere2_vision.py
struct mtmd_image_preprocessor_cohere2v : mtmd_image_preprocessor_llava_uhd {
  mtmd_image_preprocessor_cohere2v(const clip_ctx * pCtx) : mtmd_image_preprocessor_llava_uhd(pCtx) {}

  mtmd_image_preproc_out mPreprocess(const clip_image_u8 & pImg) const override;
  slice_instructions     mGetSliceInstructions(const clip_image_size & pOriginalSize) const override;
};

// DeepSeek-OCR (v1/v2) global view + optional local tile grid
struct mtmd_image_preprocessor_deepseekocr : mtmd_image_preprocessor {
  mtmd_image_preprocessor_deepseekocr(const clip_ctx * pCtx) :
    mtmd_image_preprocessor(pCtx),
    vFuseRow(fClipGetProjectorType(pCtx) == cProjectorTypeDeepseekocr),
    vBaseSize(vHparams.vImageSize),
    vTileSize(vHparams.vPreprocTileSize),
    vMinTiles(vHparams.vPreprocMinTiles),
    vMaxTiles(vHparams.vPreprocMaxTiles) {}

  mtmd_image_preproc_out mPreprocess(const clip_image_u8 & pImg) const override;

private:
  bool vFuseRow;   // v1 fuses a tile-row into one image; v2 keeps tiles separate
  int  vBaseSize;  // global view
  int  vTileSize;  // each tile
  int  vMinTiles;
  int  vMaxTiles;

  std::vector<clip_image_size> mGetTargetRatios() const;
  clip_image_size              mFindClosestAspectRatio(float                                pAspectRatio,
                                                       const std::vector<clip_image_size> & pTargetRatios,
                                                       int                                  pWidth,
                                                       int                                  pHeight) const;
};

// custom image preprocessing for Step3VL
// ref: https://huggingface.co/stepfun-ai/Step3-VL-10B/blob/main/processing_step3.py
struct mtmd_image_preprocessor_step3vl : mtmd_image_preprocessor_llava_uhd {
  mtmd_image_preprocessor_step3vl(const clip_ctx * pCtx) : mtmd_image_preprocessor_llava_uhd(pCtx) {}

  mtmd_image_preproc_out    mPreprocess(const clip_image_u8 & pImg) const override;
  static slice_instructions mBuildSliceInstructions(const clip_hparams &    pParams,
                                                    const clip_image_size & pPreparedSize);

private:
  static constexpr int   cDefaultImageLongestEdge = 3024;
  static constexpr int   cDefaultImageCropSize    = 504;
  static constexpr float cSmallAspectRatioLimit   = 1.5f;
  static constexpr float cWideAspectRatioLimit    = 4.0f;
  static constexpr float cCropRoundingThreshold   = 0.2f;

  void                    mImgU8ResizeBilinearToF32(const clip_image_u8 & pSrc,
                                                    clip_image_f32 &      pDst,
                                                    int                   pTargetWidth,
                                                    int                   pTargetHeight,
                                                    const float           pMean[3],
                                                    const float           pStd[3]) const;
  static int              mGetImageLongestEdge(const clip_hparams & pParams);
  static int              mDetermineWindowSize(const clip_hparams & pParams, int pLonger, int pShorter);
  static int              mCalcCropExtent(int length, int pWindowSize);
  static std::vector<int> mCalcGrid(int length, int pWindowSize);
  static clip_image_u8    mPrepareImage(const clip_image_u8 & pImg, const clip_hparams & pParams);
  static clip_image_u8    mCropWithBlackPadding(const clip_image_u8 & pImage, int pX, int pY, int pW, int pH);
};

struct mtmd_image_preprocessor_youtuvl : mtmd_image_preprocessor {
  mtmd_image_preprocessor_youtuvl(const clip_ctx * pCtx) : mtmd_image_preprocessor(pCtx) {}

  mtmd_image_preproc_out mPreprocess(const clip_image_u8 & pImg) const override;
};

// llava-next "anyres": stacks the overview and all tiles into one image, assembled by clip in a single graph
struct mtmd_image_preprocessor_granite : mtmd_image_preprocessor_llava_uhd {
  mtmd_image_preprocessor_granite(const clip_ctx * pCtx) : mtmd_image_preprocessor_llava_uhd(pCtx) {}

  mtmd_image_preproc_out mPreprocess(const clip_image_u8 & pImg) const override;
};

// pick the patch grid closest to the input aspect ratio under the per-image token cap, stretch-resize.
struct mtmd_image_preprocessor_muse_glimmer : mtmd_image_preprocessor {
  mtmd_image_preprocessor_muse_glimmer(const clip_ctx * pCtx) : mtmd_image_preprocessor(pCtx) {}

  mtmd_image_preproc_out mPreprocess(const clip_image_u8 & pImg) const override;
};
