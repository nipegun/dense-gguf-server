#include "mtmd-image.h"

#include <algorithm>
#include <cmath>
#include <vector>

void mtmd_image_preproc_out::mAppend(const clip_hparams & pHparams, const clip_image_u8 & pImg, bool pNormalized) {
  clip_image_f32 vDst;
  vDst.mFromU8(pImg);
  if (pNormalized) {
    vDst.mNormalize(pHparams.aImageMean, pHparams.aImageStd);
  }
  lEntries.push_back(std::move(vDst));
}

void mtmd_image_preproc_out::mAppend(const clip_hparams &               pHparams,
                                     const std::vector<clip_image_u8> & pImgs,
                                     bool                               pNormalized) {
  for (const auto & vImg : pImgs) {
    mAppend(pHparams, vImg, pNormalized);
  }
}

void mtmd_image_preproc_out::mAppend(const clip_hparams & pHparams, clip_image_f32 & pImg, bool pNormalized) {
  if (pNormalized) {
    pImg.mNormalize(pHparams.aImageMean, pHparams.aImageStd);
  }
  lEntries.push_back(std::move(pImg));
}

void mtmd_image_preproc_out::mAppendOverview(const clip_hparams &  pHparams,
                                             const clip_image_u8 & pImg,
                                             bool                  pNormalized) {
  vOverview.mFromU8(pImg);
  if (pNormalized) {
    vOverview.mNormalize(pHparams.aImageMean, pHparams.aImageStd);
  }
}

// set of tools to manipulate images
// in the future, we can have HW acceleration by allowing this struct to access 3rd party lib like imagick or opencv
struct img_tool {
  static void resize(const clip_image_u8 &   pSrc,
                     clip_image_u8 &         pDst,
                     const clip_image_size & pTargetResolution,
                     resize_algo             pAlgo,
                     pad_style               pPadding  = cPadCeil,
                     std::array<uint8_t, 3>  pPadColor = { 0, 0, 0 }) {
    pDst.mSetSize(pTargetResolution, pSrc.mIsPlaceholder());

    if (pSrc.mIsPlaceholder()) {
      // no-op for placeholder image, just set the size and return
      return;
    }

    if (pDst.mGetSize() == pSrc.mGetSize()) {
      // no resize needed, simple copy
      pDst.mCpyBuf(pSrc.mGetRoBuf());
      return;
    }

    if (pPadding == cPadNone) {
      // direct resize
      mResizePillow(pSrc, pDst, pTargetResolution.vWidth, pTargetResolution.vHeight, pAlgo);
    } else {
      // resize with padding
      clip_image_u8 vResizedImage;
      float         vScaleW = static_cast<float>(pTargetResolution.vWidth) / pSrc.mGetSize().vWidth;
      float         vScaleH = static_cast<float>(pTargetResolution.vHeight) / pSrc.mGetSize().vHeight;
      float         vScale  = std::min(vScaleW, vScaleH);

      int vNewWidth, vNewHeight;
      if (pPadding == cPadNearest) {
        vNewWidth = std::min(static_cast<int>(std::round(pSrc.mGetSize().vWidth * vScale)), pTargetResolution.vWidth);
        vNewHeight =
          std::min(static_cast<int>(std::round(pSrc.mGetSize().vHeight * vScale)), pTargetResolution.vHeight);
      } else {
        vNewWidth  = std::min(static_cast<int>(std::ceil(pSrc.mGetSize().vWidth * vScale)), pTargetResolution.vWidth);
        vNewHeight = std::min(static_cast<int>(std::ceil(pSrc.mGetSize().vHeight * vScale)), pTargetResolution.vHeight);
      }

      mResizePillow(pSrc, vResizedImage, vNewWidth, vNewHeight, pAlgo);

      // fill dst with pad_color
      fill(pDst, pPadColor);

      int vOffsetX, vOffsetY;
      if (pPadding == cPadNearest) {
        vOffsetX = static_cast<int>(std::round((pTargetResolution.vWidth - vNewWidth) / 2.0f));
        vOffsetY = static_cast<int>(std::round((pTargetResolution.vHeight - vNewHeight) / 2.0f));
      } else {
        vOffsetX = (pTargetResolution.vWidth - vNewWidth) / 2;
        vOffsetY = (pTargetResolution.vHeight - vNewHeight) / 2;
      }
      mComposite(pDst, vResizedImage, vOffsetX, vOffsetY);
    }
  }

  static void mCrop(const clip_image_u8 & pImage, clip_image_u8 & pDst, int pX, int pY, int pW, int pH) {
    GGML_ASSERT(pX >= 0 && pY >= 0 && pW > 0 && pH > 0);
    GGML_ASSERT(pX + pW <= pImage.mGetSize().vWidth && pY + pH <= pImage.mGetSize().vHeight);
    pDst.mSetSize({ pW, pH }, pImage.mIsPlaceholder());

    if (pImage.mIsPlaceholder()) {
      // no-op for placeholder image, just set the size and return
      return;
    }

    for (int vI = 0; vI < pH; ++vI) {
      for (int vJ = 0; vJ < pW; ++vJ) {
        pDst.mSetPixel(vJ, vI, pImage.mGetPixel(pX + vJ, pY + vI));
      }
    }
  }

  struct calc_size_opt {
    int vAlignSize   = 1;
    int vMinPixels   = 0;  // 0 = disabled
    int vMaxPixels   = 0;  // 0 = disabled
    // applied before min/max_pixels, so min_pixels can push an edge back above longest_edge
    int vLongestEdge = 0;  // 0 = disabled
  };

  // calculate the size of the **resized** image, while preserving the aspect ratio and
  // aligning to the nearest multiple of align_size ("smart_resize" in transformers code)
  static clip_image_size mCalcSizePreservedRatio(const clip_image_size & pInpSize, const calc_size_opt & pOpts) {
    GGML_ASSERT(pOpts.vAlignSize > 0);
    const int cWidth  = pInpSize.vWidth;
    const int cHeight = pInpSize.vHeight;
    if (cWidth <= 0 || cHeight <= 0) {
      return { 0, 0 };
    }

    auto vRoundByFactor = [vF = pOpts.vAlignSize](float pX) {
      return static_cast<int>(std::round(pX / static_cast<float>(vF))) * vF;
    };
    auto vCeilByFactor = [vF = pOpts.vAlignSize](float pX) {
      return static_cast<int>(std::ceil(pX / static_cast<float>(vF))) * vF;
    };
    auto vFloorByFactor = [vF = pOpts.vAlignSize](float pX) {
      return static_cast<int>(std::floor(pX / static_cast<float>(vF))) * vF;
    };

    int vWBar, vHBar;
    if (pOpts.vLongestEdge > 0) {
      const float cScale =
        std::min(static_cast<float>(pOpts.vLongestEdge) / cWidth, static_cast<float>(pOpts.vLongestEdge) / cHeight);
      vWBar = vCeilByFactor(cWidth * cScale);
      vHBar = vCeilByFactor(cHeight * cScale);
    } else {
      // always align up first
      vWBar = std::max(pOpts.vAlignSize, vRoundByFactor(cWidth));
      vHBar = std::max(pOpts.vAlignSize, vRoundByFactor(cHeight));
    }

    if (pOpts.vMaxPixels > 0 && vHBar * vWBar > pOpts.vMaxPixels) {
      const auto cBeta = std::sqrt(static_cast<float>(cHeight) * cWidth / pOpts.vMaxPixels);
      vHBar            = std::max(pOpts.vAlignSize, vFloorByFactor(cHeight / cBeta));
      vWBar            = std::max(pOpts.vAlignSize, vFloorByFactor(cWidth / cBeta));
    } else if (pOpts.vMinPixels > 0 && vHBar * vWBar < pOpts.vMinPixels) {
      const auto cBeta = std::sqrt(static_cast<float>(pOpts.vMinPixels) / (static_cast<float>(cHeight) * cWidth));
      vHBar            = vCeilByFactor(cHeight * cBeta);
      vWBar            = vCeilByFactor(cWidth * cBeta);
    }

    return { vWBar, vHBar };
  }

  // draw src image into dst image at offset (offset_x, offset_y)
  static void mComposite(clip_image_u8 & pDst, const clip_image_u8 & pSrc, int pOffsetX, int pOffsetY) {
    if (pSrc.mIsPlaceholder()) {
      // no-op for placeholder image
      return;
    }

    const auto cSrcSize = pSrc.mGetSize();
    const auto cDstSize = pDst.mGetSize();
    for (int vY = 0; vY < cSrcSize.vHeight; ++vY) {
      for (int vX = 0; vX < cSrcSize.vWidth; ++vX) {
        int vDx = vX + pOffsetX;
        int vDy = vY + pOffsetY;
        // skip pixels that would be out of bounds in the destination
        if (vDx < 0 || vDy < 0 || vDx >= cDstSize.vWidth || vDy >= cDstSize.vHeight) {
          continue;
        }
        pDst.mSetPixel(vDx, vDy, pSrc.mGetPixel(vX, vY));
      }
    }
  }

  // fill the image with a solid color
  static void fill(clip_image_u8 & pImg, const std::array<uint8_t, 3> & pColor) {
    if (pImg.mIsPlaceholder()) {
      // no-op for placeholder image
      return;
    }

    const auto size = pImg.mGetSize();
    for (int vY = 0; vY < size.vHeight; ++vY) {
      for (int vX = 0; vX < size.vWidth; ++vX) {
        pImg.mSetPixel(vX, vY, pColor);
      }
    }
  }

private:
  // Pillow-compatible separable resampling (Bilinear, Bicubic and Lanczos)
  // Adapted from https://github.com/python-pillow/Pillow/blob/main/src/libImaging/Resample.c
  //
  // Key properties:
  // 1. Separable filtering: horizontal pass followed by vertical pass
  // 2. Pre-computes normalized filter coefficients for each output pixel
  // 3. Fixed-point integer arithmetic (22 fractional bits) for speed and determinism
  static bool mResizePillow(const clip_image_u8 & pImg,
                            clip_image_u8 &       pDst,
                            int                   pTargetWidth,
                            int                   pTargetHeight,
                            resize_algo           pAlgo) {
    // Fixed-point precision: 22 bits = 32 (int32_t) - 8 (uint8_t pixels) - 2 (headroom for accumulation)
    // This allows encoding fractional weights as integers: weight * 2^22
    const int cPrecisionBits = 32 - 8 - 2;

    // Filter support radius
    double vFilterSupport;
    switch (pAlgo) {
      case cResizeAlgoBilinear:
        vFilterSupport = 1.0;
        break;
      case cResizeAlgoBicubic:
        vFilterSupport = 2.0;
        break;
      case cResizeAlgoLanczos:
        vFilterSupport = 3.0;
        break;
      default:
        throw std::runtime_error("Unsupported resize algorithm");
    }

    // Returns filter weight for distance x from pixel center
    // Note: for bicubic, Pillow uses a = -0.5 while GGML/PyTorch use a = -0.75
    auto vResampleFilter = [pAlgo](double pX) -> double {
      if (pAlgo == cResizeAlgoLanczos) {
        if (-3.0 <= pX && pX < 3.0) {
          auto vSinc = [](double pV) {
            if (pV == 0.0) {
              return 1.0;
            }
            const double cPiV = pV * 3.141592653589793238462643383279502884;
            return std::sin(cPiV) / cPiV;
          };
          return vSinc(pX) * vSinc(pX / 3.0);
        }
        return 0.0;
      }

      if (pX < 0.0) {
        pX = -pX;
      }

      if (pAlgo == cResizeAlgoBilinear) {
        return pX < 1.0 ? 1.0 - pX : 0.0;
      }

      constexpr double cA = -0.5;
      if (pX < 1.0) {
        return ((cA + 2.0) * pX - (cA + 3.0)) * pX * pX + 1;
      }
      if (pX < 2.0) {
        return (((pX - 5) * pX + 8) * pX - 4) * cA;
      }
      return 0.0;  // Zero outside [-2, 2]
    };

    // Clipping function for 8-bit values
    auto vClip8 = [](int pVal) -> uint8_t {
      if (pVal < 0) {
        return 0;
      }
      if (pVal > 255) {
        return 255;
      }
      return static_cast<uint8_t>(pVal);
    };

    // Precompute filter coefficients for ONE dimension (horizontal or vertical)
    //
    // Parameters:
    //   inSize  - Number of pixels in input dimension (e.g., src_width or src_height)
    //   outSize - Number of pixels in output dimension (e.g., target_width or target_height)
    //   bounds  - [OUTPUT] Array of size outSize*2 storing input pixel ranges:
    //             bounds[xx*2+0] = first input pixel index for output pixel xx (xmin)
    //             bounds[xx*2+1] = number of input pixels for output pixel xx (xcnt)
    //   weights - [OUTPUT] Array of size outSize*ksize storing fixed-point filter weights:
    //             kk[xx*ksize + x] = weight for input pixel x contributing to output pixel xx
    //
    // Returns: kernel size (ksize) - number of input pixels that contribute to each output pixel
    auto vPrecomputeWeights = [&](int pInSize, int pOutSize, std::vector<int> & pBounds,
                                  std::vector<int32_t> & pWeights) -> int {
      GGML_ASSERT(pInSize > 0 && pOutSize > 0);
      double vSupport, vScale, vFilterscale;
      double vCenter, vWw, vSs;
      int    vXx, vX, vKsize, vXmin, vXmax;

      // Calculate scaling factor: ratio of input range to output size
      vFilterscale = vScale = static_cast<double>(pInSize) / pOutSize;
      // For upsampling (scale < 1), keep filterscale = 1 to maintain filter sharpness
      // For downsampling (scale > 1), widen filter to prevent aliasing
      if (vFilterscale < 1.0) {
        vFilterscale = 1.0;
      }

      // Determine filter support radius and kernel size
      vSupport = vFilterSupport * vFilterscale;                     // Widen filter when downsampling
      vKsize   = static_cast<int>(std::ceil(vSupport)) * 2 + 1;     // Total pixels in kernel

      std::vector<double> lPreWeights((size_t) pOutSize * vKsize);  // Temporary weights
      pBounds.resize((size_t) pOutSize * 2);

      // For each output pixel, compute its filter coefficients
      for (vXx = 0; vXx < pOutSize; vXx++) {
        // Calculate the center position in input space (pixel-center convention: +0.5)
        vCenter = (vXx + 0.5) * vScale;
        vWw     = 0.0;                 // Sum of weights for normalization
        vSs     = 1.0 / vFilterscale;  // Scale factor for filter function

        // Determine the range of input pixels that contribute to this output pixel
        vXmin = static_cast<int>(vCenter - vSupport + 0.5);
        if (vXmin < 0) {
          vXmin = 0;
        }

        vXmax = static_cast<int>(vCenter + vSupport + 0.5);
        if (vXmax > pInSize) {
          vXmax = pInSize;
        }

        vXmax -= vXmin;

        // Compute filter weights for each contributing input pixel
        for (vX = 0; vX < vXmax; vX++) {
          // Distance from input pixel center to output pixel center in input space
          double vW                               = vResampleFilter((vX + vXmin - vCenter + 0.5) * vSs);
          lPreWeights[(size_t) vXx * vKsize + vX] = vW;
          vWw += vW;  // Accumulate for normalization
        }

        // Normalize weights to sum to 1.0 (preserves brightness)
        for (vX = 0; vX < vXmax; vX++) {
          if (vWw != 0.0) {
            lPreWeights[(size_t) vXx * vKsize + vX] /= vWw;
          }
        }

        // Zero-pad remaining kernel positions
        for (; vX < vKsize; vX++) {
          lPreWeights[(size_t) vXx * vKsize + vX] = 0;
        }

        // Store input pixel range for this output pixel
        pBounds[vXx * 2 + 0] = vXmin;
        pBounds[vXx * 2 + 1] = vXmax;
      }

      // Convert floating-point coefficients to fixed-point integers
      // Formula: int32 = round(float * 2^PRECISION_BITS)
      pWeights.resize((size_t) pOutSize * vKsize);

      const double cFxpScale = std::ldexp(1.0, cPrecisionBits);  // 1.0 * 2^PRECISION_BITS

      for (size_t vI = 0; vI < (size_t) pOutSize * vKsize; vI++) {
        // Pillow adds +/- 0.5 then truncates toward zero; std::round would round twice
        const double cRounded = lPreWeights[vI] * cFxpScale + (lPreWeights[vI] < 0 ? -0.5 : 0.5);
        pWeights[vI]          = static_cast<int32_t>(cRounded);
      }

      return vKsize;
    };

    // Horizontal resampling pass
    // Resizes width from src to out_nx, preserving height
    auto vResampleHorizontal = [&](const uint8_t * pSrc, int pInNx, int pInNy, int pOutNx, int pKsize,
                                   const std::vector<int> & pBounds, const std::vector<int32_t> & pWeights) {
      std::vector<uint8_t> lOut((size_t) pOutNx * pInNy * 3);

      // Process each row independently
      for (int vYy = 0; vYy < pInNy; vYy++) {
        const uint8_t * vSrcRow = pSrc + (size_t) vYy * pInNx * 3;
        uint8_t *       vDstRow = lOut.data() + (size_t) vYy * pOutNx * 3;

        // For each output pixel in this row
        for (int vXx = 0; vXx < pOutNx; vXx++) {
          const int       cXmin = pBounds[vXx * 2 + 0];  // First input pixel index
          const int       cXcnt = pBounds[vXx * 2 + 1];  // Number of input pixels
          const int32_t * vK    = &pWeights[vXx * pKsize];
          const uint8_t * vP    = vSrcRow + (size_t) cXmin * 3;

          // Accumulators for RGB channels, with rounding bias (0.5 in fixed-point)
          int32_t vSs0 = 1 << (cPrecisionBits - 1);
          int32_t vSs1 = 1 << (cPrecisionBits - 1);
          int32_t vSs2 = 1 << (cPrecisionBits - 1);

          // Convolve: sum weighted input pixels
          for (int vX = 0; vX < cXcnt; vX++) {
            vSs0 += vP[0] * vK[vX];
            vSs1 += vP[1] * vK[vX];
            vSs2 += vP[2] * vK[vX];
            vP += 3;
          }

          // Convert back from fixed-point (divide by 2^PRECISION_BITS) and clamp to [0,255]
          vDstRow[vXx * 3 + 0] = vClip8(vSs0 >> cPrecisionBits);
          vDstRow[vXx * 3 + 1] = vClip8(vSs1 >> cPrecisionBits);
          vDstRow[vXx * 3 + 2] = vClip8(vSs2 >> cPrecisionBits);
        }
      }

      return lOut;
    };

    // Vertical resampling pass
    // Resizes height from src to out_ny, preserving width
    // Accumulates whole rows at once (contiguous access, auto-vectorizes well)
    auto vResampleVertical = [&](const uint8_t * pSrc, int pInNx, int pOutNy, int pKsize,
                                 const std::vector<int> & pBounds, const std::vector<int32_t> & pWeight) {
      const size_t         cRowElems = (size_t) pInNx * 3;
      std::vector<uint8_t> lOut(cRowElems * pOutNy);
      std::vector<int32_t> lAcc(cRowElems);

      // For each output row
      for (int vYy = 0; vYy < pOutNy; vYy++) {
        const int       cYmin = pBounds[vYy * 2 + 0];  // First input row index
        const int       cYcnt = pBounds[vYy * 2 + 1];  // Number of input rows
        const int32_t * vK    = &pWeight[vYy * pKsize];

        // Rounding bias (0.5 in fixed-point)
        std::fill(lAcc.begin(), lAcc.end(), 1 << (cPrecisionBits - 1));

        // Convolve: accumulate each weighted input row
        for (int vY = 0; vY < cYcnt; vY++) {
          const uint8_t * vSrcRow = pSrc + (size_t) (cYmin + vY) * cRowElems;
          const int32_t   cW      = vK[vY];
          for (size_t vI = 0; vI < cRowElems; vI++) {
            lAcc[vI] += vSrcRow[vI] * cW;
          }
        }

        // Convert back from fixed-point and clamp to [0,255]
        uint8_t * vDstRow = lOut.data() + (size_t) vYy * cRowElems;
        for (size_t vI = 0; vI < cRowElems; vI++) {
          vDstRow[vI] = vClip8(lAcc[vI] >> cPrecisionBits);
        }
      }

      return lOut;
    };

    // Main resampling logic using separable two-pass approach
    const int cSrcWidth  = pImg.mGetSize().vWidth;
    const int cSrcHeight = pImg.mGetSize().vHeight;

    // sanity check on the target size
    if (pTargetWidth <= 0 || pTargetWidth > 65536 || pTargetHeight <= 0 || pTargetHeight > 65536) {
      throw std::runtime_error("resize target " + std::to_string(pTargetWidth) + "x" + std::to_string(pTargetHeight) +
                               " is out of range (max 65536)");
    }

    bool vNeedHorizontal = (pTargetWidth != cSrcWidth);
    bool vNeedVertical   = (pTargetHeight != cSrcHeight);

    // Precompute filter coefficients for both dimensions
    std::vector<int>     lBoundsHoriz, lBoundsVert;
    std::vector<int32_t> lWeightsHoriz, lWeightsVert;
    int                  vKsizeHoriz = 0, vKsizeVert = 0;

    if (vNeedHorizontal) {
      vKsizeHoriz = vPrecomputeWeights(cSrcWidth, pTargetWidth, lBoundsHoriz, lWeightsHoriz);
    }

    if (vNeedVertical) {
      vKsizeVert = vPrecomputeWeights(cSrcHeight, pTargetHeight, lBoundsVert, lWeightsVert);
    }

    // Perform two-pass resampling
    const uint8_t * vSrc = pImg.mGetRoBuf().data();
    if (vNeedHorizontal && vNeedVertical) {
      auto lTemp =
        vResampleHorizontal(vSrc, cSrcWidth, cSrcHeight, pTargetWidth, vKsizeHoriz, lBoundsHoriz, lWeightsHoriz);
      pDst.mSetSize({ pTargetWidth, pTargetHeight }, false);
      pDst.mCpyBuf(vResampleVertical(lTemp.data(), pTargetWidth, pTargetHeight, vKsizeVert, lBoundsVert, lWeightsVert));
    } else if (vNeedHorizontal) {
      pDst.mSetSize({ pTargetWidth, cSrcHeight }, false);
      pDst.mCpyBuf(
        vResampleHorizontal(vSrc, cSrcWidth, cSrcHeight, pTargetWidth, vKsizeHoriz, lBoundsHoriz, lWeightsHoriz));
    } else if (vNeedVertical) {
      pDst.mSetSize({ cSrcWidth, pTargetHeight }, false);
      pDst.mCpyBuf(vResampleVertical(vSrc, cSrcWidth, pTargetHeight, vKsizeVert, lBoundsVert, lWeightsVert));
    } else {
      // No resizing needed - direct copy
      pDst.mSetSize(pImg.mGetSize(), false);
      pDst.mCpyBuf(pImg.mGetRoBuf());
    }

    return true;
  }
};

//
// mtmd_image_preprocessor_llava_uhd
//

mtmd_image_preproc_out mtmd_image_preprocessor_llava_uhd::mPreprocess(const clip_image_u8 & pImg) const {
  const clip_image_size cOriginalSize = pImg.mGetSize();
  const auto            cInst         = mGetSliceInstructions(cOriginalSize);
  auto                  vSliced       = mSliceImage(pImg, cInst);

  mtmd_image_preproc_out vOutput;
  vOutput.mAppendOverview(vHparams, vSliced.vOverview, true);
  vOutput.mAppend(vHparams, vSliced.lSlices, true);
  vOutput.vGridX = cInst.vGridSize.vWidth;
  vOutput.vGridY = cInst.vGridSize.vHeight;

  return vOutput;
}

mtmd_image_preprocessor_llava_uhd::slice_instructions mtmd_image_preprocessor_llava_uhd::mGetSliceInstructions(
  const clip_image_size & pOriginalSize) const {
  mtmd_image_preprocessor_llava_uhd::slice_instructions vRes;
  const int                                             cPatchSize      = mGetSliceAlign();
  const int                                             cSliceSize      = vHparams.vImageSize;
  const int                                             cOriginalWidth  = pOriginalSize.vWidth;
  const int                                             cOriginalHeight = pOriginalSize.vHeight;

  const bool cHasSlices    = pOriginalSize.vWidth > cSliceSize || pOriginalSize.vHeight > cSliceSize;
  const bool cHasPinpoints = !vHparams.lImageResCandidates.empty();

  if (!cHasSlices) {
    // skip slicing logic
    vRes.vOverviewSize = clip_image_size{ cSliceSize, cSliceSize };
    vRes.vRefinedSize  = clip_image_size{ 0, 0 };
    vRes.vGridSize     = clip_image_size{ 0, 0 };

    return vRes;
  }

  if (cHasPinpoints) {
    // has pinpoints, use them to calculate the grid size (e.g. llava-1.6)
    auto vRefineSize   = mSelectBestResolution(pOriginalSize, vHparams.lImageResCandidates);
    vRes.vOverviewSize = clip_image_size{ cSliceSize, cSliceSize };
    vRes.vRefinedSize  = vRefineSize;
    vRes.vGridSize     = clip_image_size{ 0, 0 };

    LOG_DBG("%s: using pinpoints for slicing\n", __func__);
    LOG_DBG("%s: original size: %d x %d, overview size: %d x %d, refined size: %d x %d\n", __func__, cOriginalWidth,
            cOriginalHeight, vRes.vOverviewSize.vWidth, vRes.vOverviewSize.vHeight, vRes.vRefinedSize.vWidth,
            vRes.vRefinedSize.vHeight);

    for (int vY = 0; vY < vRefineSize.vHeight; vY += cSliceSize) {
      for (int vX = 0; vX < vRefineSize.vWidth; vX += cSliceSize) {
        slice_coordinates vSlice;
        vSlice.vX           = vX;
        vSlice.vY           = vY;
        vSlice.size.vWidth  = std::min(cSliceSize, vRefineSize.vWidth - vX);
        vSlice.size.vHeight = std::min(cSliceSize, vRefineSize.vHeight - vY);
        vRes.lSlices.push_back(vSlice);
        LOG_DBG("%s: slice %d: x=%d, y=%d, size=%dx%d\n", __func__, (int) vRes.lSlices.size() - 1, vSlice.vX, vSlice.vY,
                vSlice.size.vWidth, vSlice.size.vHeight);
      }
    }

    vRes.vGridSize.vHeight = vRefineSize.vHeight / cSliceSize;
    vRes.vGridSize.vWidth  = vRefineSize.vWidth / cSliceSize;
    LOG_DBG("%s: grid size: %d x %d\n", __func__, vRes.vGridSize.vWidth, vRes.vGridSize.vHeight);

    return vRes;
  }

  // no pinpoints, dynamically calculate the grid size (e.g. minicpmv)

  auto vBestSize     = mGetBestResize(pOriginalSize, cSliceSize, cPatchSize, !cHasSlices);
  vRes.vOverviewSize = vBestSize;

  {
    const int   cMaxSliceNums = vHparams.vMaxSliceNums > 0 ? vHparams.vMaxSliceNums : 9;
    const float cLogRatio     = log((float) cOriginalWidth / cOriginalHeight);
    const float cRatio        = (float) cOriginalWidth * cOriginalHeight / (cSliceSize * cSliceSize);
    const int   cMultiple     = fmin(ceil(cRatio), cMaxSliceNums);

    auto vBestGrid    = mGetBestGrid(cMaxSliceNums, cMultiple, cLogRatio);
    auto vRefineSize  = mGetRefineSize(pOriginalSize, vBestGrid, cSliceSize, cPatchSize, true);
    vRes.vGridSize    = vBestGrid;
    vRes.vRefinedSize = vRefineSize;

    LOG_DBG("%s: original size: %d x %d, overview size: %d x %d, refined size: %d x %d, grid size: %d x %d\n", __func__,
            cOriginalWidth, cOriginalHeight, vRes.vOverviewSize.vWidth, vRes.vOverviewSize.vHeight,
            vRes.vRefinedSize.vWidth, vRes.vRefinedSize.vHeight, vRes.vGridSize.vWidth, vRes.vGridSize.vHeight);

    int vWidth  = vRefineSize.vWidth;
    int vHeight = vRefineSize.vHeight;
    int vGridX  = int(vWidth / vBestGrid.vWidth);
    int vGridY  = int(vHeight / vBestGrid.vHeight);
    for (int vPatchesY = 0, vIc = 0; vPatchesY < vRefineSize.vHeight && vIc < vBestGrid.vHeight;
         vPatchesY += vGridY, vIc += 1) {
      for (int vPatchesX = 0, vJc = 0; vPatchesX < vRefineSize.vWidth && vJc < vBestGrid.vWidth;
           vPatchesX += vGridX, vJc += 1) {
        slice_coordinates vSlice;
        vSlice.vX           = vPatchesX;
        vSlice.vY           = vPatchesY;
        vSlice.size.vWidth  = vGridX;
        vSlice.size.vHeight = vGridY;
        vRes.lSlices.push_back(vSlice);
        LOG_DBG("%s: slice %d: x=%d, y=%d, size=%dx%d\n", __func__, (int) vRes.lSlices.size() - 1, vSlice.vX, vSlice.vY,
                vSlice.size.vWidth, vSlice.size.vHeight);
      }
    }
  }

  return vRes;
}

mtmd_image_preprocessor_llava_uhd::slice_output mtmd_image_preprocessor_llava_uhd::mSliceImage(
  const clip_image_u8 &                                         pImg,
  const mtmd_image_preprocessor_llava_uhd::slice_instructions & pInst) const {
  slice_output vOutput;

  // resize to overview size
  img_tool::resize(pImg, vOutput.vOverview, pInst.vOverviewSize, vHparams.vImageResizeAlgoOv, vHparams.vImagePadOv,
                   vHparams.aImagePadColorOv);

  if (pInst.lSlices.empty()) {
    // no slices, just return the overview image
    return vOutput;
  }

  // resize to refined size
  clip_image_u8 vRefinedImg;
  img_tool::resize(pImg, vRefinedImg, pInst.vRefinedSize, vHparams.vImageResizeAlgoRf, vHparams.vImagePadRf,
                   vHparams.aImagePadColorRf);

  // create slices
  for (const auto & vSlice : pInst.lSlices) {
    int vX = vSlice.vX;
    int vY = vSlice.vY;
    int vW = vSlice.size.vWidth;
    int vH = vSlice.size.vHeight;

    clip_image_u8 vImgSlice;
    img_tool::mCrop(vRefinedImg, vImgSlice, vX, vY, vW, vH);
    vOutput.lSlices.push_back(std::move(vImgSlice));
  }

  return vOutput;
}

clip_image_size mtmd_image_preprocessor_llava_uhd::mGetBestResize(const clip_image_size & pOriginalSize,
                                                                  int                     pScaleResolution,
                                                                  int                     pPatchSize,
                                                                  bool                    pAllowUpscale) const {
  int vWidth  = pOriginalSize.vWidth;
  int vHeight = pOriginalSize.vHeight;
  if ((vWidth * vHeight > pScaleResolution * pScaleResolution) || pAllowUpscale) {
    float vR = static_cast<float>(vWidth) / vHeight;
    vHeight  = static_cast<int>(pScaleResolution / std::sqrt(vR));
    vWidth   = static_cast<int>(vHeight * vR);
  }
  clip_image_size vRes;
  vRes.vWidth  = mEnsureDivide(vWidth, pPatchSize);
  vRes.vHeight = mEnsureDivide(vHeight, pPatchSize);
  return vRes;
}

clip_image_size mtmd_image_preprocessor_llava_uhd::mResizeMaintainAspectRatio(
  const clip_image_size & pOrig,
  const clip_image_size & pTargetMax) const {
  float vScaleWidth  = static_cast<float>(pTargetMax.vWidth) / pOrig.vWidth;
  float vScaleHeight = static_cast<float>(pTargetMax.vHeight) / pOrig.vHeight;
  float vScale       = std::min(vScaleWidth, vScaleHeight);
  return clip_image_size{
    static_cast<int>(pOrig.vWidth * vScale),
    static_cast<int>(pOrig.vHeight * vScale),
  };
}

clip_image_size mtmd_image_preprocessor_llava_uhd::mSelectBestResolution(
  const clip_image_size &              pOriginalSize,
  const std::vector<clip_image_size> & pPossibleResolutions) const {
  clip_image_size vBestFit;
  int             vMinWastedArea          = std::numeric_limits<int>::max();
  int             vMaxEffectiveResolution = 0;

  for (const clip_image_size & vCandidate : pPossibleResolutions) {
    auto vTargetSize = mResizeMaintainAspectRatio(pOriginalSize, vCandidate);
    int  vEffectiveResolution =
      std::min(vTargetSize.vWidth * vTargetSize.vHeight, pOriginalSize.vWidth * pOriginalSize.vHeight);
    int vWastedArea = (vCandidate.vWidth * vCandidate.vHeight) - vEffectiveResolution;

    if (vEffectiveResolution > vMaxEffectiveResolution ||
        (vEffectiveResolution == vMaxEffectiveResolution && vWastedArea < vMinWastedArea)) {
      vMaxEffectiveResolution = vEffectiveResolution;
      vMinWastedArea          = vWastedArea;
      vBestFit                = vCandidate;
    }

    LOG_DBG("%s: candidate: %d x %d, target: %d x %d, wasted: %d, effective: %d\n", __func__, vCandidate.vWidth,
            vCandidate.vHeight, vTargetSize.vWidth, vTargetSize.vHeight, vWastedArea, vEffectiveResolution);
  }

  return vBestFit;
}

int mtmd_image_preprocessor_llava_uhd::mEnsureDivide(int length, int pPatchSize) const {
  return std::max(mAlignRound(static_cast<double>(length) / pPatchSize) * pPatchSize, pPatchSize);
}

clip_image_size mtmd_image_preprocessor_llava_uhd::mGetRefineSize(const clip_image_size & pOriginalSize,
                                                                  const clip_image_size & pGrid,
                                                                  int                     pScaleResolution,
                                                                  int                     pPatchSize,
                                                                  bool                    pAllowUpscale) const {
  int vWidth  = pOriginalSize.vWidth;
  int vHeight = pOriginalSize.vHeight;
  int vGridX  = pGrid.vWidth;
  int vGridY  = pGrid.vHeight;

  int vRefineWidth  = mEnsureDivide(vWidth, vGridX);
  int vRefineHeight = mEnsureDivide(vHeight, vGridY);

  clip_image_size vGridSize;
  vGridSize.vWidth  = vRefineWidth / vGridX;
  vGridSize.vHeight = vRefineHeight / vGridY;

  auto vBestGridSize   = mGetBestResize(vGridSize, pScaleResolution, pPatchSize, pAllowUpscale);
  int  vBestGridWidth  = vBestGridSize.vWidth;
  int  vBestGridHeight = vBestGridSize.vHeight;

  clip_image_size vRefineSize;
  vRefineSize.vWidth  = vBestGridWidth * vGridX;
  vRefineSize.vHeight = vBestGridHeight * vGridY;
  return vRefineSize;
}

clip_image_size mtmd_image_preprocessor_llava_uhd::mGetBestGrid(const int   pMaxSliceNums,
                                                                const int   pMultiple,
                                                                const float pLogRatio) const {
  std::vector<int> lCandidateSplitGridsNums;
  for (int vI : { pMultiple - 1, pMultiple, pMultiple + 1 }) {
    if (vI == 1 || vI > pMaxSliceNums) {
      continue;
    }
    lCandidateSplitGridsNums.push_back(vI);
  }

  std::vector<clip_image_size> lCandidateGrids;
  for (int vSplitGridsNums : lCandidateSplitGridsNums) {
    int vM = 1;
    while (vM <= vSplitGridsNums) {
      if (vSplitGridsNums % vM == 0) {
        lCandidateGrids.push_back(clip_image_size{ vM, vSplitGridsNums / vM });
      }
      ++vM;
    }
  }

  clip_image_size vBestGrid{ 1, 1 };
  float           vMinError = std::numeric_limits<float>::infinity();
  for (const auto & vGrid : lCandidateGrids) {
    float vError = std::abs(pLogRatio - std::log(1.0 * vGrid.vWidth / vGrid.vHeight));
    if (vError < vMinError) {
      vBestGrid = vGrid;
      vMinError = vError;
    }
  }
  return vBestGrid;
}

//
// mtmd_image_preprocessor_fixed_size
//

mtmd_image_preproc_out mtmd_image_preprocessor_fixed_size::mPreprocess(const clip_image_u8 & pImg) const {
  clip_image_u8 vResizedImage;
  int           vSz = vHparams.vImageSize;
  img_tool::resize(pImg, vResizedImage, { vSz, vSz }, vHparams.vImageResizeAlgo, vHparams.vImageResizePad,
                   vHparams.aImagePadColor);
  mtmd_image_preproc_out vOutput;
  vOutput.mAppend(vHparams, vResizedImage, true);
  return vOutput;
}

//
// mtmd_image_preprocessor_dyn_size
//

mtmd_image_preproc_out mtmd_image_preprocessor_dyn_size::mPreprocess(const clip_image_u8 & pImg) const {
  GGML_ASSERT(vHparams.vImageMinPixels > 0 && vHparams.vImageMaxPixels > 0);
  clip_image_u8         vResizedImage;
  const clip_image_size cOriginalSize = pImg.mGetSize();
  // the original pixtral model doesn't have n_merge
  const int             cCurMerge     = vHparams.vNMerge;
  const clip_image_size cTargetSize =
    img_tool::mCalcSizePreservedRatio(cOriginalSize, {
                                                       /* align_size   */ vHparams.vPatchSize * cCurMerge,
                                                       /* min_pixels   */ vHparams.vImageMinPixels,
                                                       /* max_pixels   */ vHparams.vImageMaxPixels,
                                                       /* longest_edge */ 0,
                                                     });
  img_tool::resize(pImg, vResizedImage, cTargetSize, vHparams.vImageResizeAlgo, vHparams.vImageResizePad,
                   vHparams.aImagePadColor);
  mtmd_image_preproc_out vOutput;
  vOutput.mAppend(vHparams, vResizedImage, true);
  return vOutput;
}

//
// mtmd_image_preprocessor_glm5v
//

// The canvas is ceil-aligned to patch_size*n_merge and fitted to the token budget.
// Only rescaled to meet the budget and sits top-left, with black padding on the right and bottom
mtmd_image_preproc_out mtmd_image_preprocessor_glm5v::mPreprocess(const clip_image_u8 & pImg) const {
  GGML_ASSERT(vHparams.vImageMinPixels > 0 && vHparams.vImageMaxPixels > 0);

  const int64_t cFactor = vHparams.vPatchSize * vHparams.vNMerge;
  const int64_t cMinPx  = vHparams.vImageMinPixels;  // single-frame pixel counts
  const int64_t cMaxPx  = vHparams.vImageMaxPixels;
  const int64_t cHeight = pImg.mGetSize().vHeight;
  const int64_t cWidth  = pImg.mGetSize().vWidth;

  auto vAlign = [cFactor](int64_t pV) {
    return (pV + cFactor - 1) / cFactor * cFactor;
  };

  // aligned canvas within the budget
  int64_t vCanvasH = vAlign(cHeight);
  int64_t vCanvasW = vAlign(cWidth);

  if (vCanvasH * vCanvasW < cMinPx) {
    const double cScale = std::sqrt((double) cMinPx / (double) (cHeight * cWidth));
    vCanvasH            = vAlign(std::max<int64_t>(1, (int64_t) std::ceil(cHeight * cScale)));
    vCanvasW            = vAlign(std::max<int64_t>(1, (int64_t) std::ceil(cWidth * cScale)));
  }

  if (vCanvasH * vCanvasW > cMaxPx) {
    // largest content height whose aligned canvas fits the budget
    int64_t vLo = 1, vHi = cHeight;
    int64_t vBestH = cFactor, vBestW = cFactor;
    while (vLo <= vHi) {
      const int64_t cCh = (vLo + vHi) / 2;
      const int64_t cCw = std::max<int64_t>(1, cWidth * cCh / cHeight);
      const int64_t cAh = vAlign(cCh);
      const int64_t cAw = vAlign(cCw);
      if (cAh * cAw <= cMaxPx) {
        vBestH = cAh;
        vBestW = cAw;
        vLo    = cCh + 1;
      } else {
        vHi = cCh - 1;
      }
    }
    vCanvasH = vBestH;
    vCanvasW = vBestW;
  }

  // Scaled to fit the canvas, and never upscaled, unless below the min budget
  double vScale = std::min((double) vCanvasH / cHeight, (double) vCanvasW / cWidth);
  if (cHeight * cWidth >= cMinPx) {
    vScale = std::min(1.0, vScale);
  }
  const int cContentH = (int) std::max<int64_t>(1, std::min<int64_t>(vCanvasH, (int64_t) std::floor(cHeight * vScale)));
  const int cContentW = (int) std::max<int64_t>(1, std::min<int64_t>(vCanvasW, (int64_t) std::floor(cWidth * vScale)));

  clip_image_u8 vContent;
  img_tool::resize(pImg, vContent, clip_image_size{ cContentW, cContentH }, vHparams.vImageResizeAlgo, cPadNone);

  clip_image_u8 vCanvas;
  vCanvas.mSetSize(clip_image_size{ (int) vCanvasW, (int) vCanvasH }, pImg.mIsPlaceholder());
  img_tool::fill(vCanvas, { 0, 0, 0 });
  img_tool::mComposite(vCanvas, vContent, 0, 0);

  mtmd_image_preproc_out vOutput;
  vOutput.mAppend(vHparams, vCanvas, true);
  return vOutput;
}

//
// mtmd_image_preprocessor_longest_edge
//

mtmd_image_preproc_out mtmd_image_preprocessor_longest_edge::mPreprocess(const clip_image_u8 & pImg) const {
  GGML_ASSERT(vHparams.vImageLongestEdge > 0);
  clip_image_u8         vResizedImage;
  const clip_image_size cOriginalSize = pImg.mGetSize();
  // the original pixtral model doesn't have n_merge
  const int             cCurMerge     = vHparams.vNMerge == 0 ? 1 : vHparams.vNMerge;
  const clip_image_size cTargetSize =
    img_tool::mCalcSizePreservedRatio(cOriginalSize, {
                                                       /* align_size   */ vHparams.vPatchSize * cCurMerge,
                                                       /* min_pixels   */ std::max(0, vHparams.vImageMinPixels),
                                                       /* max_pixels   */ std::max(0, vHparams.vImageMaxPixels),
                                                       /* longest_edge */ vHparams.vImageLongestEdge,
                                                     });
  img_tool::resize(pImg, vResizedImage, cTargetSize, vHparams.vImageResizeAlgo, vHparams.vImageResizePad,
                   vHparams.aImagePadColor);
  mtmd_image_preproc_out vOutput;
  vOutput.mAppend(vHparams, vResizedImage, true);
  return vOutput;
}

//
// mtmd_image_preprocessor_minicpmv
//

mtmd_image_preprocessor_llava_uhd::slice_instructions mtmd_image_preprocessor_minicpmv::mGetSliceInstructions(
  const clip_image_size & pOriginalSize) const {
  // overview only for small images, unlike generic llava-uhd which slices once one side exceeds scale resolution
  const int   cSliceSize = vHparams.vImageSize;
  const float cRatio     = (float) pOriginalSize.vWidth * pOriginalSize.vHeight / (cSliceSize * cSliceSize);
  if (cRatio <= 1.0f) {
    mtmd_image_preprocessor_llava_uhd::slice_instructions vInst;
    vInst.vOverviewSize = mGetBestResize(pOriginalSize, cSliceSize, mGetSliceAlign(), true);
    vInst.vRefinedSize  = clip_image_size{ 0, 0 };
    vInst.vGridSize     = clip_image_size{ 0, 0 };
    return vInst;
  }
  return mtmd_image_preprocessor_llava_uhd::mGetSliceInstructions(pOriginalSize);
}

//
// mtmd_image_preprocessor_lfm2
//

mtmd_image_preproc_out mtmd_image_preprocessor_lfm2::mPreprocess(const clip_image_u8 & pImg) const {
  const auto cInst = mGetSliceInstructions(pImg.mGetSize());
  if (!cInst.lSlices.empty()) {
    return mtmd_image_preprocessor_llava_uhd::mPreprocess(pImg);
  }

  // single tile: no thumbnail
  // note: not using output.overview here because it will emit <|img_thumbnail|> token, which we don't want in this case
  auto                   vSliced = mSliceImage(pImg, cInst);
  mtmd_image_preproc_out vOutput;
  vOutput.mAppend(vHparams, vSliced.vOverview, true);
  return vOutput;
}

bool mtmd_image_preprocessor_lfm2::mShouldTile(const clip_hparams & pHparams, const clip_image_size & pOriginalSize) {
  const int cAlignSize = pHparams.vPatchSize * pHparams.vNMerge;

  const auto cRoundByFactor = [cAlignSize](float pX) {
    return static_cast<int>(std::nearbyint(static_cast<double>(pX) / cAlignSize)) * cAlignSize;
  };

  const int cHBar = std::max(pHparams.vPatchSize, cRoundByFactor(pOriginalSize.vHeight));
  const int cWBar = std::max(pHparams.vPatchSize, cRoundByFactor(pOriginalSize.vWidth));

  return static_cast<double>(cHBar) * static_cast<double>(cWBar) >
         static_cast<double>(pHparams.vImageMaxPixels) * cMaxPixelsTolerance;
}

mtmd_image_preprocessor_llava_uhd::slice_instructions mtmd_image_preprocessor_lfm2::mGetSliceInstructions(
  const clip_image_size & pOriginalSize) const {
  mtmd_image_preprocessor_llava_uhd::slice_instructions vInst;
  const int                                             cAlignSize = vHparams.vPatchSize * vHparams.vNMerge;
  vInst.vOverviewSize                                              = img_tool::mCalcSizePreservedRatio(
    pOriginalSize, { cAlignSize, vHparams.vImageMinPixels, vHparams.vImageMaxPixels, 0 });

  const bool cNeedsTiling = mShouldTile(vHparams, pOriginalSize);

  if (!cNeedsTiling) {
    vInst.vRefinedSize = clip_image_size{ 0, 0 };
    vInst.vGridSize    = clip_image_size{ 0, 0 };
    return vInst;
  }

  const clip_image_size cGrid = mGetGridLayout(pOriginalSize.vHeight, pOriginalSize.vWidth);

  vInst.vGridSize    = cGrid;
  vInst.vRefinedSize = clip_image_size{ cTileSize * cGrid.vWidth, cTileSize * cGrid.vHeight };

  LOG_DBG("%s: original size: %d x %d, overview size: %d x %d, refined size: %d x %d, grid size: %d x %d\n", __func__,
          pOriginalSize.vWidth, pOriginalSize.vHeight, vInst.vOverviewSize.vWidth, vInst.vOverviewSize.vHeight,
          vInst.vRefinedSize.vWidth, vInst.vRefinedSize.vHeight, cGrid.vWidth, cGrid.vHeight);

  for (int vRow = 0; vRow < cGrid.vHeight; vRow++) {
    for (int vCol = 0; vCol < cGrid.vWidth; vCol++) {
      mtmd_image_preprocessor_llava_uhd::slice_coordinates vSlice;
      vSlice.vX   = vCol * cTileSize;
      vSlice.vY   = vRow * cTileSize;
      vSlice.size = clip_image_size{ cTileSize, cTileSize };
      vInst.lSlices.push_back(vSlice);
      LOG_DBG("%s: slice %d: x=%d, y=%d, size=%d x %d\n", __func__, (int) vInst.lSlices.size() - 1, vSlice.vX,
              vSlice.vY, vSlice.size.vWidth, vSlice.size.vHeight);
    }
  }

  return vInst;
}

clip_image_size mtmd_image_preprocessor_lfm2::mFindClosestAspectRatio(
  float                                pAspectRatio,
  const std::vector<clip_image_size> & pTargetRatios,
  int                                  pWidth,
  int                                  pHeight) const {
  float           vBestRatioDiff = std::numeric_limits<float>::max();
  clip_image_size vBestRatio     = { 1, 1 };
  const float     cArea          = static_cast<float>(pWidth * pHeight);

  for (const auto & vRatio : pTargetRatios) {
    const float cTargetAspectRatio = static_cast<float>(vRatio.vWidth) / vRatio.vHeight;
    const float cRatioDiff         = std::abs(pAspectRatio - cTargetAspectRatio);
    if (cRatioDiff < vBestRatioDiff) {
      vBestRatioDiff = cRatioDiff;
      vBestRatio     = vRatio;
    } else if (cRatioDiff == vBestRatioDiff) {
      const float cTargetArea = static_cast<float>(cTileSize * cTileSize * vRatio.vWidth * vRatio.vHeight);
      if (cArea > 0.5f * cTargetArea) {
        vBestRatio = vRatio;
      }
    }
  }
  return vBestRatio;
}

std::vector<clip_image_size> mtmd_image_preprocessor_lfm2::mGetTargetRatios() const {
  std::vector<clip_image_size> lRatios;
  for (int n = cMinTiles; n <= cMaxTiles; n++) {
    for (int vW = 1; vW <= n; vW++) {
      for (int vH = 1; vH <= n; vH++) {
        if (vW * vH >= cMinTiles && vW * vH <= cMaxTiles) {
          bool vFound = false;
          for (const auto & vR : lRatios) {
            if (vR.vWidth == vW && vR.vHeight == vH) {
              vFound = true;
              break;
            }
          }
          if (!vFound) {
            lRatios.push_back({ vW, vH });
          }
        }
      }
    }
  }
  std::sort(lRatios.begin(), lRatios.end(), [](const clip_image_size & pA, const clip_image_size & pB) {
    return pA.vWidth * pA.vHeight < pB.vWidth * pB.vHeight;
  });
  return lRatios;
}

clip_image_size mtmd_image_preprocessor_lfm2::mGetGridLayout(int pHeight, int pWidth) const {
  const float cAspectRatio = static_cast<float>(pWidth) / pHeight;
  const auto  lRatios      = mGetTargetRatios();
  return mFindClosestAspectRatio(cAspectRatio, lRatios, pWidth, pHeight);
}

//
// mtmd_image_preprocessor_idefics3
//

mtmd_image_preproc_out mtmd_image_preprocessor_idefics3::mPreprocess(const clip_image_u8 & pImg) const {
  // The refined size has two steps:
  // 1. Resize w/ aspect-ratio preserving such that the longer side is
  //      the preprocessor longest size
  // 2. Resize w/out preserving aspect ratio such that both sides are
  //      multiples of image_size (always rounding up)
  //
  // CITE: https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics3/image_processing_idefics3.py#L737
  const clip_image_size cOriginalSize = pImg.mGetSize();

  // old gguf files have no preprocessor longest size, custom token limits also need the generic size below
  if (vHparams.vImageLongestEdge > 0 && vHparams.vImageMinPixels <= 0 && vHparams.vImageMaxPixels <= 0) {
    const int    cTileSize    = vHparams.vImageSize;
    const int    cLongestEdge = vHparams.vImageLongestEdge;
    const double cAspectRatio = (double) cOriginalSize.vWidth / cOriginalSize.vHeight;

    clip_image_size vResizedSize;
    if (cOriginalSize.vWidth >= cOriginalSize.vHeight) {
      vResizedSize.vWidth  = cLongestEdge;
      vResizedSize.vHeight = (int) (cLongestEdge / cAspectRatio);
      vResizedSize.vHeight += vResizedSize.vHeight % 2;
    } else {
      vResizedSize.vHeight = cLongestEdge;
      vResizedSize.vWidth  = (int) (cLongestEdge * cAspectRatio);
      vResizedSize.vWidth += vResizedSize.vWidth % 2;
    }

    const int             cGridX       = (vResizedSize.vWidth + cTileSize - 1) / cTileSize;
    const int             cGridY       = (vResizedSize.vHeight + cTileSize - 1) / cTileSize;
    const clip_image_size cRefinedSize = clip_image_size{ cGridX * cTileSize, cGridY * cTileSize };

    clip_image_u8 vResizedImg;
    img_tool::resize(pImg, vResizedImg, vResizedSize, vHparams.vImageResizeAlgo, cPadNone);

    clip_image_u8 vRefinedImg;
    img_tool::resize(vResizedImg, vRefinedImg, cRefinedSize, vHparams.vImageResizeAlgo, cPadNone);

    clip_image_u8 vOverview;
    img_tool::resize(vRefinedImg, vOverview, { cTileSize, cTileSize }, vHparams.vImageResizeAlgo, cPadNone);

    std::vector<clip_image_u8> lSlices;
    for (int vY = 0; vY < cGridY; vY++) {
      for (int vX = 0; vX < cGridX; vX++) {
        clip_image_u8 vSlice;
        img_tool::mCrop(vRefinedImg, vSlice, vX * cTileSize, vY * cTileSize, cTileSize, cTileSize);
        lSlices.push_back(std::move(vSlice));
      }
    }

    LOG_DBG("%s: grid size: %d x %d (%d tiles) + overview\n", __func__, cGridX, cGridY, cGridX * cGridY);

    mtmd_image_preproc_out vOutput;
    vOutput.mAppendOverview(vHparams, vOverview, true);
    vOutput.mAppend(vHparams, lSlices, true);
    vOutput.vGridX = cGridX;
    vOutput.vGridY = cGridY;
    return vOutput;
  }

  const clip_image_size cRefinedSize = img_tool::mCalcSizePreservedRatio(
    cOriginalSize, { vHparams.vImageSize, std::max(0, vHparams.vImageMinPixels), std::max(0, vHparams.vImageMaxPixels),
                     vHparams.vImageLongestEdge });
  // LOG_INF("%s: original size: %d x %d, refined size: %d x %d\n",
  //         __func__, original_size.width, original_size.height,
  //         refined_size.width, refined_size.height);

  mtmd_image_preprocessor_llava_uhd::slice_instructions vInstructions;
  vInstructions.vOverviewSize = clip_image_size{ vHparams.vImageSize, vHparams.vImageSize };
  vInstructions.vRefinedSize  = cRefinedSize;
  vInstructions.vGridSize     = clip_image_size{
    static_cast<int>(std::ceil(static_cast<float>(cRefinedSize.vWidth) / vHparams.vImageSize)),
    static_cast<int>(std::ceil(static_cast<float>(cRefinedSize.vHeight) / vHparams.vImageSize)),
  };
  for (int vY = 0; vY < cRefinedSize.vHeight; vY += vHparams.vImageSize) {
    for (int vX = 0; vX < cRefinedSize.vWidth; vX += vHparams.vImageSize) {
      // LOG_INF("%s: adding slice at x=%d, y=%d\n", __func__, x, y);
      vInstructions.lSlices.push_back(mtmd_image_preprocessor_llava_uhd::slice_coordinates{
        /* x    */ vX,
        /* y    */ vY,
        /* size */
        clip_image_size{ std::min(vHparams.vImageSize, cRefinedSize.vWidth - vX),
                        std::min(vHparams.vImageSize, cRefinedSize.vHeight - vY) }
      });
    }
  }
  auto vSliced = mSliceImage(pImg, vInstructions);

  mtmd_image_preproc_out vOutput;
  vOutput.mAppendOverview(vHparams, vSliced.vOverview, true);
  vOutput.mAppend(vHparams, vSliced.lSlices, true);
  vOutput.vGridX = vInstructions.vGridSize.vWidth;
  vOutput.vGridY = vInstructions.vGridSize.vHeight;
  return vOutput;
}

//
// mtmd_image_preprocessor_cohere2v
//

mtmd_image_preproc_out mtmd_image_preprocessor_cohere2v::mPreprocess(const clip_image_u8 & pImg) const {
  const auto cInst   = mGetSliceInstructions(pImg.mGetSize());
  auto       vSliced = mSliceImage(pImg, cInst);

  mtmd_image_preproc_out vOutput;
  if (vSliced.lSlices.empty()) {
    vOutput.mAppendOverview(vHparams, vSliced.vOverview, true);
    return vOutput;
  }
  // slices first, then thumbnail
  vOutput.mAppend(vHparams, vSliced.lSlices, true);
  vOutput.mAppendOverview(vHparams, vSliced.vOverview, true);
  vOutput.vGridX = cInst.vGridSize.vWidth;
  vOutput.vGridY = cInst.vGridSize.vHeight;
  return vOutput;
}

mtmd_image_preprocessor_llava_uhd::slice_instructions mtmd_image_preprocessor_cohere2v::mGetSliceInstructions(
  const clip_image_size & pOriginalSize) const {
  const int cTile = vHparams.vImageSize;

  // pick the grid with the least upscale; if all grids need downscale, pick the one with the least downscale
  // grids are visited by tile count, then by width, same order as HF for ties
  double          vBestDown = -1.0;
  double          vBestUp   = std::numeric_limits<double>::max();
  clip_image_size vGridDown = { 1, 1 };
  clip_image_size vGridUp   = { 0, 0 };
  for (int n = 1; n <= vHparams.vPreprocMaxTiles; n++) {
    for (int vW = 1; vW <= n; vW++) {
      if (n % vW != 0) {
        continue;
      }
      const clip_image_size cG     = { vW, n / vW };
      const double          cScale = std::min((double) (cG.vWidth * cTile) / pOriginalSize.vWidth,
                                              (double) (cG.vHeight * cTile) / pOriginalSize.vHeight);
      if (cScale < 1.0) {
        if (cScale > vBestDown) {
          vBestDown = cScale;
          vGridDown = cG;
        }
      } else if (cScale < vBestUp) {
        vBestUp = cScale;
        vGridUp = cG;
      }
    }
  }
  const clip_image_size cGrid = vGridUp.vWidth > 0 ? vGridUp : vGridDown;

  slice_instructions vInst;
  vInst.vOverviewSize = { cTile, cTile };
  vInst.vRefinedSize  = { cTile * cGrid.vWidth, cTile * cGrid.vHeight };
  vInst.vGridSize     = cGrid;
  if (cGrid.vWidth * cGrid.vHeight > 1) {
    for (int vY = 0; vY < cGrid.vHeight; vY++) {
      for (int vX = 0; vX < cGrid.vWidth; vX++) {
        vInst.lSlices.push_back({
          vX * cTile, vY * cTile, { cTile, cTile }
        });
      }
    }
  }
  return vInst;
}

//
// mtmd_image_preprocessor_internvl
//

mtmd_image_preproc_out mtmd_image_preprocessor_internvl::mPreprocess(const clip_image_u8 & pImg) const {
  GGML_ASSERT(!vHparams.lImageResCandidates.empty());
  const clip_image_size cOriginalSize = pImg.mGetSize();
  const auto            cInst         = mGetSliceInstructions(cOriginalSize);
  auto                  vSliced       = mSliceImage(pImg, cInst);

  mtmd_image_preproc_out vOutput;
  // InternVL: slices first, then overview
  vOutput.mAppend(vHparams, vSliced.lSlices, true);
  vOutput.mAppendOverview(vHparams, vSliced.vOverview, true);
  vOutput.vGridX = cInst.vGridSize.vWidth;
  vOutput.vGridY = cInst.vGridSize.vHeight;
  return vOutput;
}

//
// mtmd_image_preprocessor_deepseekocr
//

std::vector<clip_image_size> mtmd_image_preprocessor_deepseekocr::mGetTargetRatios() const {
  std::vector<clip_image_size> lRatios;
  for (int n = vMinTiles; n <= vMaxTiles; n++) {
    for (int vW = 1; vW <= n; vW++) {
      for (int vH = 1; vH <= n; vH++) {
        if (vW * vH < vMinTiles || vW * vH > vMaxTiles) {
          continue;
        }
        bool vFound = false;
        for (const auto & vR : lRatios) {
          if (vR.vWidth == vW && vR.vHeight == vH) {
            vFound = true;
            break;
          }
        }
        if (!vFound) {
          lRatios.push_back({ vW, vH });
        }
      }
    }
  }
  std::sort(lRatios.begin(), lRatios.end(), [](const clip_image_size & pA, const clip_image_size & pB) {
    return pA.vWidth * pA.vHeight < pB.vWidth * pB.vHeight;
  });
  return lRatios;
}

clip_image_size mtmd_image_preprocessor_deepseekocr::mFindClosestAspectRatio(
  float                                pAspectRatio,
  const std::vector<clip_image_size> & pTargetRatios,
  int                                  pWidth,
  int                                  pHeight) const {
  float           vBestRatioDiff = std::numeric_limits<float>::max();
  clip_image_size vBestRatio     = { 1, 1 };
  const float     cArea          = static_cast<float>(pWidth * pHeight);

  for (const auto & vRatio : pTargetRatios) {
    const float cTargetAspectRatio = static_cast<float>(vRatio.vWidth) / vRatio.vHeight;
    const float cRatioDiff         = std::abs(pAspectRatio - cTargetAspectRatio);
    if (cRatioDiff < vBestRatioDiff) {
      vBestRatioDiff = cRatioDiff;
      vBestRatio     = vRatio;
    } else if (cRatioDiff == vBestRatioDiff) {
      const float cTargetArea = static_cast<float>(vTileSize * vTileSize * vRatio.vWidth * vRatio.vHeight);
      if (cArea > 0.5f * cTargetArea) {
        vBestRatio = vRatio;
      }
    }
  }
  return vBestRatio;
}

//
// DeepSeek-V4-Flash-Vision (deepseek4v)
//
// port of load_image / safe_resize / solve_resize_ratio / grid_tokens from inference/image_processor.py
// the resize solver picks the largest target size (multiple of patch_size) whose LLM token block fits max_n_token
//

// ref: grid_tokens()
mtmd_image_preprocessor_deepseek4v::grid_info mtmd_image_preprocessor_deepseek4v::mGridTokens(int pBestHeight,
                                                                                              int pBestWidth,
                                                                                              int pPatchSize,
                                                                                              int pR) {
  grid_info vG;
  vG.vNLlmH   = ((pBestHeight / pPatchSize) + pR - 1) / pR;
  vG.vNLlmW   = ((pBestWidth / pPatchSize) + pR - 1) / pR;
  vG.vNTokens = fDsv4GetBlockLayout(vG.vNLlmW, vG.vNLlmH, 0).vNOut;
  return vG;
}

// ref: solve_resize_ratio()
void mtmd_image_preprocessor_deepseek4v::mSolveResizeRatio(int   pHeight,
                                                           int   pWidth,
                                                           int   pP,
                                                           int   pR,
                                                           int   pMaxNToken,
                                                           int & pBestHeight,
                                                           int & pBestWidth) {
  const double cRatio = (double) pHeight / pWidth;
  const double cMaxWF = std::sqrt((pMaxNToken - 2) / cRatio + 0.25) - 0.5;
  const double cMaxHF = cMaxWF * cRatio;
  if (cMaxWF < 1.0) {
    const int cMaxW = 1;
    int       vMaxH = (pMaxNToken - 2) / (cMaxW + 1);
    if (vMaxH % 2 == 1) {
      vMaxH -= 1;
    }
    pBestWidth  = cMaxW * pP * pR;
    pBestHeight = vMaxH * pP * pR;
  } else if (cMaxHF < 2.0) {
    const int cMaxH = 2;
    // guard tiny budgets; cannot be hit with the current lower bound on max_n_token
    const int cMaxW = std::max(((pMaxNToken - 2) / cMaxH) - 1, 2);
    pBestWidth      = cMaxW * pP * pR;
    pBestHeight     = cMaxH * pP * pR;
  } else {
    const int cMaxWI = (int) std::floor(cMaxWF);
    int       vMaxHI = (int) std::floor(cMaxHF);
    if (vMaxHI % 2 == 1) {
      vMaxHI -= 1;
    }
    const double cBeta = std::min((double) cMaxWI * pP * pR / pWidth, (double) vMaxHI * pP * pR / pHeight);
    pBestWidth         = (int) std::floor(pWidth * cBeta / pP) * pP;
    pBestHeight        = (int) std::floor(pHeight * cBeta / pP) * pP;
  }
}

// ref: safe_resize()
void mtmd_image_preprocessor_deepseek4v::mSafeResize(int   pHeight,
                                                     int   pWidth,
                                                     int & pBestHeight,
                                                     int & pBestWidth,
                                                     int   pP,
                                                     int   pR,
                                                     int   pMaxNToken) {
  pMaxNToken -= 4 - 1;  // reserve room for the position-dependent lead pads (COMPRESS_PAD_TO - 1)
  grid_info vG      = mGridTokens(pBestHeight, pBestWidth, pP, pR);
  int       vBudget = pMaxNToken;
  while (vG.vNTokens > pMaxNToken) {
    mSolveResizeRatio(pHeight, pWidth, pP, pR, vBudget, pBestHeight, pBestWidth);
    vG = mGridTokens(pBestHeight, pBestWidth, pP, pR);
    vBudget -= 1;
  }
}

// ref: load_image()
mtmd_image_preproc_out mtmd_image_preprocessor_deepseek4v::mPreprocess(const clip_image_u8 & pImg) const {
  mtmd_image_preproc_out vOut;

  const int cP         = vHparams.vPatchSize;
  const int cR         = vHparams.vNMerge;
  const int cMaxNToken = vHparams.vDsv4MaxNToken;
  const int cMaxWh     = vHparams.vDsv4MaxWhRatio;

  const clip_image_size cOrig   = pImg.mGetSize();
  int                   vWidth  = cOrig.vWidth;
  int                   vHeight = cOrig.vHeight;
  if (cMaxWh > 0 && vWidth > vHeight * cMaxWh) {
    vWidth = vHeight * cMaxWh;
  }
  if (vHparams.vImageMinPixels > 0 && vWidth * vHeight > 0 && vWidth * vHeight < vHparams.vImageMinPixels) {
    const double cUp = std::sqrt((double) vHparams.vImageMinPixels / ((double) vWidth * vHeight));
    vWidth           = (int) (vWidth * cUp);
    vHeight          = (int) (vHeight * cUp);
  }
  int vBestWidth  = CLIP_ALIGN(vWidth, cP);
  int vBestHeight = CLIP_ALIGN(vHeight, cP);
  mSafeResize(vHeight, vWidth, vBestHeight, vBestWidth, cP, cR, cMaxNToken);

  clip_image_u8 vResized;
  if (cMaxWh > 0 && cOrig.vWidth >= cMaxWh * cOrig.vHeight) {
    // extreme aspect ratio: plain stretch resize, no padding
    img_tool::resize(pImg, vResized, { vBestWidth, vBestHeight }, vHparams.vImageResizeAlgo, cPadNone);
  } else {
    // aspect-preserving resize + centered padding (PIL ImageOps.pad)
    img_tool::resize(pImg, vResized, { vBestWidth, vBestHeight }, vHparams.vImageResizeAlgo, cPadNearest,
                     vHparams.aImagePadColor);
  }

  vOut.mAppend(vHparams, vResized);
  return vOut;
}

mtmd_image_preproc_out mtmd_image_preprocessor_deepseekocr::mPreprocess(const clip_image_u8 & pImg) const {
  mtmd_image_preproc_out vOutput;
  int                    vGridW   = 0;
  int                    vGridH   = 0;
  const auto             cImgSize = pImg.mGetSize();

  // global view: aspect-preserving fit-and-pad to base_size
  clip_image_u8 vPadded;
  img_tool::resize(pImg, vPadded, { vBaseSize, vBaseSize }, cResizeAlgoBicubic, cPadNearest, vHparams.aImagePadColor);
  vOutput.mAppendOverview(vHparams, vPadded, true);
  vOutput.vOverview.vAddViewsep = true;

  // if this condition doesn't hold, the output is overview only, no tiles
  if (cImgSize.vWidth > vTileSize || cImgSize.vHeight > vTileSize) {
    const float           cAspectRatio  = static_cast<float>(cImgSize.vWidth) / cImgSize.vHeight;
    const auto            lTargetRatios = mGetTargetRatios();
    const clip_image_size cGrid =
      mFindClosestAspectRatio(cAspectRatio, lTargetRatios, cImgSize.vWidth, cImgSize.vHeight);
    vGridW = cGrid.vWidth;
    vGridH = cGrid.vHeight;

    clip_image_u8 vRefined;
    img_tool::resize(pImg, vRefined, { vTileSize * vGridW, vTileSize * vGridH }, cResizeAlgoBicubic, cPadNone);

    for (int vRow = 0; vRow < vGridH; vRow++) {
      if (vFuseRow) {
        // concat all tiles in this row into a single image, along the H axis
        // output image size: w = tile_size, h = tile_size * grid_w
        // this is to ensure the whole row is always processed together
        clip_image_u8 vRowImg;
        vRowImg.mSetSize({ vTileSize, vTileSize * vGridW }, false);
        for (int vCol = 0; vCol < vGridW; vCol++) {
          for (int vPy = 0; vPy < vTileSize; vPy++) {
            for (int vPx = 0; vPx < vTileSize; vPx++) {
              vRowImg.mSetPixel(vPx, vCol * vTileSize + vPy,
                                vRefined.mGetPixel(vCol * vTileSize + vPx, vRow * vTileSize + vPy));
            }
          }
        }
        vOutput.mAppend(vHparams, vRowImg, true);
      } else {
        for (int vCol = 0; vCol < vGridW; vCol++) {
          clip_image_u8 vTile;
          img_tool::mCrop(vRefined, vTile, vCol * vTileSize, vRow * vTileSize, vTileSize, vTileSize);
          vOutput.mAppend(vHparams, vTile, true);
        }
      }
    }
    if (vFuseRow) {
      vGridW = 1;  // each fused row is one image; a single output column
    }
  }

  LOG_DBG("%s: grid size: %d x %d (%d tiles) + global view\n", __func__, vGridW, vGridH, vGridW * vGridH);
  LOG_DBG("%s: overview size: %d x %d\n", __func__, vPadded.mGetSize().vWidth, vPadded.mGetSize().vHeight);

  vOutput.vGridX = vGridW;
  vOutput.vGridY = vGridH;
  return vOutput;
}

//
// mtmd_image_preprocessor_step3vl
//

void mtmd_image_preprocessor_step3vl::mImgU8ResizeBilinearToF32(const clip_image_u8 & pSrc,
                                                                clip_image_f32 &      pDst,
                                                                int                   pTargetWidth,
                                                                int                   pTargetHeight,
                                                                const float           pMean[3],
                                                                const float           pStd[3]) const {
  const auto cSrcSize = pSrc.mGetSize();
  if (cSrcSize.vWidth == pTargetWidth && cSrcSize.vHeight == pTargetHeight) {
    pDst.mFromU8(pSrc);
    pDst.mNormalize(pMean, pStd);
    return;
  }

  pDst.mSetSize({ pTargetWidth, pTargetHeight }, false, false);

  if (pSrc.mIsPlaceholder()) {
    // no-op for placeholder image, just set the size and return
    return;
  }

  const float cScaleX = static_cast<float>(cSrcSize.vWidth) / pTargetWidth;
  const float cScaleY = static_cast<float>(cSrcSize.vHeight) / pTargetHeight;

  std::vector<float> lLocalBuf((size_t) 3 * (size_t) pTargetWidth * (size_t) pTargetHeight);

  for (int vY = 0; vY < pTargetHeight; ++vY) {
    const float cSrcY    = (static_cast<float>(vY) + 0.5f) * cScaleY - 0.5f;
    const int   cY0Floor = static_cast<int>(std::floor(cSrcY));
    const int   cY0      = std::max(0, std::min(cY0Floor, cSrcSize.vHeight - 1));
    const int   cY1      = std::max(0, std::min(cY0Floor + 1, cSrcSize.vHeight - 1));
    const float cLy      = cSrcY - cY0Floor;

    for (int vX = 0; vX < pTargetWidth; ++vX) {
      const float cSrcX    = (static_cast<float>(vX) + 0.5f) * cScaleX - 0.5f;
      const int   cX0Floor = static_cast<int>(std::floor(cSrcX));
      const int   cX0      = std::max(0, std::min(cX0Floor, cSrcSize.vWidth - 1));
      const int   cX1      = std::max(0, std::min(cX0Floor + 1, cSrcSize.vWidth - 1));
      const float cLx      = cSrcX - cX0Floor;

      const auto aP00 = pSrc.mGetPixel(cX0, cY0);
      const auto aP01 = pSrc.mGetPixel(cX1, cY0);
      const auto aP10 = pSrc.mGetPixel(cX0, cY1);
      const auto aP11 = pSrc.mGetPixel(cX1, cY1);

      const size_t cIdxDst = (size_t) 3 * ((size_t) vY * (size_t) pTargetWidth + (size_t) vX);
      for (int vC = 0; vC < 3; ++vC) {
        const float cV00 = (static_cast<float>(aP00[vC]) / 255.0f - pMean[vC]) / pStd[vC];
        const float cV01 = (static_cast<float>(aP01[vC]) / 255.0f - pMean[vC]) / pStd[vC];
        const float cV10 = (static_cast<float>(aP10[vC]) / 255.0f - pMean[vC]) / pStd[vC];
        const float cV11 = (static_cast<float>(aP11[vC]) / 255.0f - pMean[vC]) / pStd[vC];

        const float cTop        = cV00 + (cV01 - cV00) * cLx;
        const float cBot        = cV10 + (cV11 - cV10) * cLx;
        lLocalBuf[cIdxDst + vC] = cTop + (cBot - cTop) * cLy;
      }
    }
  }
  pDst.mCpyBuf(lLocalBuf);
}

int mtmd_image_preprocessor_step3vl::mGetImageLongestEdge(const clip_hparams & pParams) {
  return pParams.vImageLongestEdge > 0 ? pParams.vImageLongestEdge : cDefaultImageLongestEdge;
}

int mtmd_image_preprocessor_step3vl::mDetermineWindowSize(const clip_hparams & pParams, int pLonger, int pShorter) {
  const int   cImageSize   = pParams.vImageSize;
  const int   cCropSize    = cDefaultImageCropSize;
  const float cAspectRatio = static_cast<float>(pLonger) / pShorter;

  if (pLonger <= cImageSize) {
    return cAspectRatio > cSmallAspectRatioLimit ? pShorter : 0;
  }

  return cAspectRatio > cWideAspectRatioLimit ? std::min(pShorter, cCropSize) : cCropSize;
}

int mtmd_image_preprocessor_step3vl::mCalcCropExtent(int length, int pWindowSize) {
  const float cRatio = static_cast<float>(length) / pWindowSize;
  if (cRatio < 1.0f) {
    return length;
  }

  const float cDecimal = cRatio - std::floor(cRatio);
  const int   cRounded =
    cDecimal > cCropRoundingThreshold ? static_cast<int>(std::floor(cRatio)) + 1 : static_cast<int>(std::floor(cRatio));
  return pWindowSize * cRounded;
}

std::vector<int> mtmd_image_preprocessor_step3vl::mCalcGrid(int length, int pWindowSize) {
  const int        n = length <= pWindowSize ?
                         1 :
                         static_cast<int>(std::ceil(static_cast<float>(length - pWindowSize) / pWindowSize + 1.0f));
  std::vector<int> lStarts(n);

  for (int vI = 0; vI < n; ++vI) {
    lStarts[vI] = pWindowSize * vI;
  }

  if (n > 1 && lStarts.back() + pWindowSize > length) {
    lStarts.back() = length - pWindowSize;
  }

  return lStarts;
}

clip_image_u8 mtmd_image_preprocessor_step3vl::mPrepareImage(const clip_image_u8 & pImg, const clip_hparams & pParams) {
  clip_image_u8 vResized     = pImg;
  const auto    cImgSize     = pImg.mGetSize();
  const float   cAspectRatio = cImgSize.vHeight > 0 ? static_cast<float>(cImgSize.vWidth) / cImgSize.vHeight : 1.0f;
  if (std::min(cImgSize.vWidth, cImgSize.vHeight) < 32 &&
      (cAspectRatio > cWideAspectRatioLimit || cAspectRatio < 1.0f / cWideAspectRatioLimit)) {
    const int     cSquareSize = std::max(cImgSize.vWidth, cImgSize.vHeight);
    clip_image_u8 vPadded;
    vPadded.mSetSize({ cSquareSize, cSquareSize }, false);
    img_tool::fill(vPadded, { 0, 0, 0 });
    img_tool::mComposite(vPadded, pImg, 0, 0);
    vResized = std::move(vPadded);
  }

  const int  cMaxImageSize = mGetImageLongestEdge(pParams);
  const auto cResizedSize  = vResized.mGetSize();
  if (std::max(cResizedSize.vWidth, cResizedSize.vHeight) > cMaxImageSize) {
    const float cScale = static_cast<float>(cMaxImageSize) / std::max(cResizedSize.vWidth, cResizedSize.vHeight);
    const clip_image_size cNewSize = {
      std::max(1, static_cast<int>(std::floor(cResizedSize.vWidth * cScale))),
      std::max(1, static_cast<int>(std::floor(cResizedSize.vHeight * cScale))),
    };
    clip_image_u8 vScaled;
    img_tool::resize(vResized, vScaled, cNewSize, cResizeAlgoBilinear, cPadNone);
    vResized = std::move(vScaled);
  }

  return vResized;
}

clip_image_u8 mtmd_image_preprocessor_step3vl::mCropWithBlackPadding(const clip_image_u8 & pImage,
                                                                     int                   pX,
                                                                     int                   pY,
                                                                     int                   pW,
                                                                     int                   pH) {
  clip_image_u8 vDst;
  vDst.mSetSize({ pW, pH }, false);
  img_tool::fill(vDst, { 0, 0, 0 });

  const auto cImgSize = pImage.mGetSize();
  const int  cSrcX0   = std::max(0, pX);
  const int  cSrcY0   = std::max(0, pY);
  const int  cSrcX1   = std::min(cImgSize.vWidth, pX + pW);
  const int  cSrcY1   = std::min(cImgSize.vHeight, pY + pH);

  if (cSrcX0 >= cSrcX1 || cSrcY0 >= cSrcY1) {
    return vDst;
  }

  const int cDstX0 = cSrcX0 - pX;
  const int cDstY0 = cSrcY0 - pY;

  for (int vYy = 0; vYy < cSrcY1 - cSrcY0; ++vYy) {
    for (int vXx = 0; vXx < cSrcX1 - cSrcX0; ++vXx) {
      vDst.mSetPixel(cDstX0 + vXx, cDstY0 + vYy, pImage.mGetPixel(cSrcX0 + vXx, cSrcY0 + vYy));
    }
  }

  return vDst;
}

mtmd_image_preprocessor_step3vl::slice_instructions mtmd_image_preprocessor_step3vl::mBuildSliceInstructions(
  const clip_hparams &    pParams,
  const clip_image_size & pPreparedSize) {
  slice_instructions vInstructions;
  vInstructions.vOverviewSize = pPreparedSize;

  const int cWindowSize = mDetermineWindowSize(pParams, std::max(pPreparedSize.vWidth, pPreparedSize.vHeight),
                                               std::min(pPreparedSize.vWidth, pPreparedSize.vHeight));
  if (cWindowSize <= 0) {
    vInstructions.vRefinedSize = clip_image_size{ 0, 0 };
    vInstructions.vGridSize    = clip_image_size{ 0, 0 };
    return vInstructions;
  }

  const int cCropWidth       = mCalcCropExtent(pPreparedSize.vWidth, cWindowSize);
  const int cCropHeight      = mCalcCropExtent(pPreparedSize.vHeight, cWindowSize);
  vInstructions.vRefinedSize = clip_image_size{ cCropWidth, cCropHeight };

  const auto lXs          = mCalcGrid(cCropWidth, cWindowSize);
  const auto lYs          = mCalcGrid(cCropHeight, cWindowSize);
  vInstructions.vGridSize = clip_image_size{
    static_cast<int>(lXs.size()),
    static_cast<int>(lYs.size()),
  };

  for (int vY : lYs) {
    for (int vX : lXs) {
      vInstructions.lSlices.push_back(slice_coordinates{
        /* x    */ vX,
        /* y    */ vY,
        /* size */ clip_image_size{ cWindowSize, cWindowSize },
      });
    }
  }

  return vInstructions;
}

mtmd_image_preproc_out mtmd_image_preprocessor_step3vl::mPreprocess(const clip_image_u8 & pImg) const {
  clip_image_u8 vPrepared     = mPrepareImage(pImg, vHparams);
  const auto    cInstructions = mBuildSliceInstructions(vHparams, vPrepared.mGetSize());

  mtmd_image_preproc_out vOutput;
  // overview (normalized f32, already includes mean/std)
  mImgU8ResizeBilinearToF32(vPrepared, vOutput.vOverview, vHparams.vImageSize, vHparams.vImageSize, vHparams.aImageMean,
                            vHparams.aImageStd);

  if (cInstructions.lSlices.empty()) {
    vOutput.vGridX = 0;
    vOutput.vGridY = 0;
    return vOutput;
  }

  clip_image_u8 vImgForCrop   = vPrepared;
  const auto    cPreparedSize = vPrepared.mGetSize();
  if (cInstructions.vRefinedSize.vWidth != cPreparedSize.vWidth ||
      cInstructions.vRefinedSize.vHeight != cPreparedSize.vHeight) {
    clip_image_u8 vRefined;
    img_tool::resize(vPrepared, vRefined, cInstructions.vRefinedSize, cResizeAlgoBilinear, cPadNone);
    vImgForCrop = std::move(vRefined);
  }

  const int cCropSize = cDefaultImageCropSize;
  for (const auto & vSlice : cInstructions.lSlices) {
    // If the requested patch extends past the source image, pad the out-of-bounds area with black.
    clip_image_u8 vPatch =
      mCropWithBlackPadding(vImgForCrop, vSlice.vX, vSlice.vY, vSlice.size.vWidth, vSlice.size.vHeight);

    clip_image_f32 vPatchF32;
    mImgU8ResizeBilinearToF32(vPatch, vPatchF32, cCropSize, cCropSize, vHparams.aImageMean, vHparams.aImageStd);
    vOutput.mAppend(vHparams, vPatchF32, false);
  }

  vOutput.vGridX = cInstructions.vGridSize.vWidth;
  vOutput.vGridY = cInstructions.vGridSize.vHeight;

  return vOutput;
}

//
// mtmd_image_preprocessor_youtuvl
//

mtmd_image_preproc_out mtmd_image_preprocessor_youtuvl::mPreprocess(const clip_image_u8 & pImg) const {
  const int cPatchSize = vHparams.vPatchSize;      // typically 16
  const int cMergeSize = vHparams.vNMerge;         // typically 2
  const int cAlignSize = cPatchSize * cMergeSize;  // 32

  const int cMaxNumPatches = vHparams.vImageMaxPixels > 0 ? vHparams.vImageMaxPixels / (cPatchSize * cPatchSize) : 256;

  // Linear search for optimal scale to fit within max_num_patches
  const auto cImgSize      = pImg.mGetSize();
  float      vScale        = 1.0f;
  int        vTargetHeight = cImgSize.vHeight;
  int        vTargetWidth  = cImgSize.vWidth;

  auto vGetScaledImageSize = [cAlignSize](float pScale, int size) -> int {
    float vScaledSize = size * pScale;
    // Round up to nearest multiple of align_size
    int   vAligned    = static_cast<int>(std::ceil(vScaledSize / cAlignSize)) * cAlignSize;
    // Ensure at least one patch
    return std::max(cAlignSize, vAligned);
  };

  // Linear search with 0.02 step size
  while (vScale > 0.0f) {
    vTargetHeight = vGetScaledImageSize(vScale, cImgSize.vHeight);
    vTargetWidth  = vGetScaledImageSize(vScale, cImgSize.vWidth);

    int vNumPatchesH = vTargetHeight / cPatchSize;
    int vNumPatchesW = vTargetWidth / cPatchSize;
    int vNumPatches  = vNumPatchesH * vNumPatchesW;

    if (vNumPatches > cMaxNumPatches) {
      vScale -= 0.02f;
    } else {
      break;
    }
  }

  clip_image_size vNewSize = { vTargetWidth, vTargetHeight };

  // Resize the image
  clip_image_u8 vResized;
  img_tool::resize(pImg, vResized, vNewSize, vHparams.vImageResizeAlgo, vHparams.vImageResizePad);

  mtmd_image_preproc_out vOutput;
  vOutput.mAppend(vHparams, vResized, true);
  return vOutput;
}

mtmd_image_preproc_out mtmd_image_preprocessor_granite::mPreprocess(const clip_image_u8 & pImg) const {
  GGML_ASSERT(!vHparams.lImageResCandidates.empty());

  const clip_image_size cOrigSize = pImg.mGetSize();
  const int             cTileSize = vHparams.vImageSize;
  GGML_ASSERT(cTileSize > 0);

  // llava-next always encodes an overview plus a grid of tiles, even for small images
  const clip_image_size cRefinedSize = mSelectBestResolution(cOrigSize, vHparams.lImageResCandidates);
  const int             cGridX       = cRefinedSize.vWidth / cTileSize;
  const int             cGridY       = cRefinedSize.vHeight / cTileSize;

  // the tiles are stacked on the Y axis, a big grid overflows the stacked image height
  GGML_ASSERT(cGridX >= 0 && cGridX <= 1024 && cGridY >= 0 && cGridY <= 1024);

  clip_image_u8 vOverview;
  img_tool::resize(pImg, vOverview, { cTileSize, cTileSize }, vHparams.vImageResizeAlgoOv, vHparams.vImagePadOv,
                   vHparams.aImagePadColorOv);

  clip_image_u8 vRefined;
  img_tool::resize(pImg, vRefined, cRefinedSize, vHparams.vImageResizeAlgoRf, vHparams.vImagePadRf,
                   vHparams.aImagePadColorRf);

  // stack the overview and the tiles on the Y axis, so the whole grid goes through one graph
  clip_image_u8 vStacked;
  vStacked.mSetSize({ cTileSize, cTileSize * (1 + cGridX * cGridY) }, false);
  auto vCopyTile = [&](const clip_image_u8 & pSrc, int pSrcX, int pSrcY, int pDstIdx) {
    for (int vPy = 0; vPy < cTileSize; vPy++) {
      for (int vPx = 0; vPx < cTileSize; vPx++) {
        vStacked.mSetPixel(vPx, pDstIdx * cTileSize + vPy, pSrc.mGetPixel(pSrcX + vPx, pSrcY + vPy));
      }
    }
  };
  vCopyTile(vOverview, 0, 0, 0);
  for (int vTy = 0; vTy < cGridY; vTy++) {
    for (int vTx = 0; vTx < cGridX; vTx++) {
      vCopyTile(vRefined, vTx * cTileSize, vTy * cTileSize, 1 + vTy * cGridX + vTx);
    }
  }

  LOG_DBG("%s: grid size: %d x %d (%d tiles) + overview\n", __func__, cGridX, cGridY, cGridX * cGridY);

  mtmd_image_preproc_out vOutput;
  vOutput.mAppend(vHparams, vStacked, true);
  auto & vEntry          = vOutput.lEntries.back();
  vEntry.vAnyres.vGridX  = cGridX;
  vEntry.vAnyres.vGridY  = cGridY;
  vEntry.vAnyres.vOrigNx = cOrigSize.vWidth;
  vEntry.vAnyres.vOrigNy = cOrigSize.vHeight;
  return vOutput;
}

//
// mtmd_image_preprocessor_muse_glimmer
//

// Replicates transformers' get_aspect_ratio_preserving_size
static clip_image_size fMuseGlimmerGridSize(int pImgW, int pImgH, int pPatchHw, int pMaxTokens) {
  double       vINph  = (double) pImgH / pPatchHw;
  double       vINpw  = (double) pImgW / pPatchHw;
  const double cRatio = vINph > 0.0 ? vINpw / vINph : 1.0;
  if (vINph * vINpw > (double) pMaxTokens) {
    vINph = std::sqrt((double) pMaxTokens / cRatio);
    vINpw = vINph * cRatio;
  }
  const int    aHs[2]    = { (int) std::floor(vINph), (int) std::ceil(vINph) };
  const int    aWs[2]    = { (int) std::floor(vINpw), (int) std::ceil(vINpw) };
  const double cTargetAr = (double) pImgH / (double) pImgW;
  int          vBestNph  = -1;
  int          vBestNpw  = -1;
  double       vBestD    = 0.0;
  for (int vA = 0; vA < 2; ++vA) {
    for (int vB = 0; vB < 2; ++vB) {
      const int cNph = aHs[vA];
      const int cNpw = aWs[vB];
      if (cNph < 1 || cNpw < 1 || cNph * cNpw > pMaxTokens) {
        continue;
      }
      const double cD           = std::fabs((double) cNph / (double) cNpw - cTargetAr);
      const int    cNTokens     = cNph * cNpw;
      const int    cBestNTokens = vBestNph * vBestNpw;
      if (vBestNph < 0 || cD < vBestD || (cD == vBestD && cNTokens > cBestNTokens)) {
        vBestNph = cNph;
        vBestNpw = cNpw;
        vBestD   = cD;
      }
    }
  }
  if (vBestNph < 0) {  // no candidate fit under the cap: round and clamp
    vBestNph = std::max(1, (int) std::lround(vINph));
    vBestNpw = std::max(1, (int) std::lround(vINpw));
  }
  return clip_image_size{ vBestNpw * pPatchHw, vBestNph * pPatchHw };
}

mtmd_image_preproc_out mtmd_image_preprocessor_muse_glimmer::mPreprocess(const clip_image_u8 & pImg) const {
  const int cPatchHw   = vHparams.vPatchSize * vHparams.vNMerge;
  const int cPatchArea = vHparams.vPatchSize * vHparams.vPatchSize * vHparams.vNMerge * vHparams.vNMerge;
  GGML_ASSERT(cPatchArea > 0 && vHparams.vImageMaxPixels > 0);
  const int cMaxTokens = vHparams.vImageMaxPixels / cPatchArea;

  const clip_image_size cOriginalSize = pImg.mGetSize();
  const clip_image_size cTargetSize =
    fMuseGlimmerGridSize(cOriginalSize.vWidth, cOriginalSize.vHeight, cPatchHw, cMaxTokens);

  // PIL resizes directly to (target_w, target_h) -- a stretch, no padding.
  clip_image_u8 vResizedImage;
  img_tool::resize(pImg, vResizedImage, cTargetSize, vHparams.vImageResizeAlgo, cPadNone);

  mtmd_image_preproc_out vOutput;
  vOutput.mAppend(vHparams, vResizedImage, true);
  return vOutput;
}
