#include "mtmd.h"

#include "clip-impl.h"
#include "clip.h"
#include "dgs.h"
#include "mtmd-audio.h"
#include "mtmd-image.h"
#include "mtmd-internal.h"

// fix problem with std::min and std::max
#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#endif

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

// remember to bump this if the serialization format changes
#define MTMD_SERIALIZATION_VERSION 3

// oldest compat version that can be loaded
#define MTMD_SERIALIZATION_VERSION_MIN 2

struct mtmd_serialization {
  // note: using 64-bit here for future-proofing
  uint64_t          vVersion = MTMD_SERIALIZATION_VERSION;
  std::vector<char> data;
  size_t            vReadPos = 0;  // cursor used when reading

  // for writing
  mtmd_serialization(uint64_t pVersion) : vVersion(pVersion) { write(pVersion); }

  // for reading
  mtmd_serialization(uint64_t pVersion, const char * pBuf, size_t pLen) {
    // copy buf to data
    data.assign(pBuf, pBuf + pLen);
    uint64_t vVerIn = read<uint64_t>();
    if (vVerIn < MTMD_SERIALIZATION_VERSION_MIN || vVerIn > pVersion) {
      throw std::runtime_error("version mismatch");
    }
    this->vVersion = vVerIn;
  }

  template <typename T> void write(T value) {
    static_assert(std::is_trivially_copyable<T>::value && !std::is_same<T, bool>::value,
                  "T must be trivially copyable and not bool");
    const char * vP = reinterpret_cast<const char *>(&value);
    data.insert(data.end(), vP, vP + sizeof(T));
  }

  template <typename T> T read() {
    static_assert(std::is_trivially_copyable<T>::value && !std::is_same<T, bool>::value,
                  "T must be trivially copyable and not bool");
    if (vReadPos + sizeof(T) > data.size()) {
      throw std::runtime_error("read OOB");
    }
    T value;
    std::memcpy(&value, data.data() + vReadPos, sizeof(T));
    vReadPos += sizeof(T);
    return value;
  }
};

template <> void mtmd_serialization::write<bool>(bool value) {
  write<uint8_t>(value ? 1 : 0);
}

template <> bool mtmd_serialization::read<bool>() {
  return read<uint8_t>() != 0;
}

template <> void mtmd_serialization::write<std::string>(std::string value) {
  write<uint64_t>(value.size());
  data.insert(data.end(), value.begin(), value.end());
}

template <> std::string mtmd_serialization::read<std::string>() {
  uint64_t vLen = read<uint64_t>();
  if (vReadPos + vLen > data.size()) {
    throw std::runtime_error("read_string OOB");
  }
  std::string str(data.data() + vReadPos, vLen);
  vReadPos += vLen;
  return str;
}

// only mtmd.cpp needs these, so they're implemented here rather than in clip-impl.h
void clip_image_f32::serialize(mtmd_serialization & pSer) const {
  // remember to bump MTMD_SERIALIZATION_VERSION if this is changed
  // note: buf is intentionally NOT serialized; the loaded clip_image_f32 will always be a placeholder
  pSer.write(vAddViewsep);
  pSer.write(vAddNewline);
  pSer.write(vLeadPad);
  pSer.write((int32_t) vSuffixType);
  pSer.write((int32_t) vAnyres.vGridX);
  pSer.write((int32_t) vAnyres.vGridY);
  pSer.write((int32_t) vAnyres.vOrigNx);
  pSer.write((int32_t) vAnyres.vOrigNy);
  pSer.write((int32_t) vNx);
  pSer.write((int32_t) vNy);
}

void clip_image_f32::deserialize(mtmd_serialization & pSer) {
  vAddViewsep = pSer.read<bool>();
  vAddNewline = pSer.read<bool>();
  vLeadPad    = pSer.read<int32_t>();
  if (pSer.vVersion >= 3) {
    const int32_t cSuffixRaw = pSer.read<int32_t>();
    if (cSuffixRaw < 0 || cSuffixRaw >= cClipSuffixCount) {
      throw std::runtime_error("invalid suffix type");
    }
    vSuffixType     = (clip_suffix_type) cSuffixRaw;
    vAnyres.vGridX  = pSer.read<int32_t>();
    vAnyres.vGridY  = pSer.read<int32_t>();
    vAnyres.vOrigNx = pSer.read<int32_t>();
    vAnyres.vOrigNy = pSer.read<int32_t>();
  }
  vNx = pSer.read<int32_t>();
  vNy = pSer.read<int32_t>();
  lBuf.clear();  // always a placeholder after loading
}

void clip_image_f32_batch::serialize(mtmd_serialization & pSer) const {
  // remember to bump MTMD_SERIALIZATION_VERSION if this is changed
  pSer.write(vIsAudio);
  pSer.write<uint64_t>(lEntries.size());
  for (const auto & vEntry : lEntries) {
    vEntry.serialize(pSer);
  }
}

void clip_image_f32_batch::deserialize(mtmd_serialization & pSer) {
  vIsAudio                        = pSer.read<bool>();
  uint64_t         n              = pSer.read<uint64_t>();
  constexpr size_t cMinEntryBytes = sizeof(uint8_t) * 2 + sizeof(int32_t) * 2;
  if (n > (pSer.data.size() - pSer.vReadPos) / cMinEntryBytes) {
    throw std::runtime_error("entries count exceeds buffer size");
  }
  lEntries.clear();
  lEntries.reserve(n);
  for (uint64_t vI = 0; vI < n; vI++) {
    clip_image_f32 vEntry;
    vEntry.deserialize(pSer);
    lEntries.push_back(std::move(vEntry));
  }
}

// for still image data, layout is RGBRGBRGB...
// length of data must be nx * ny * 3 bytes
//
// for audio bitmap: nx = sample count, ny = 1, layout is F32 F32 F32 ...
// length of data must be nx * sizeof(float) bytes
struct mtmd_bitmap {
  uint32_t    vNx = 0;
  uint32_t    vNy = 0;
  std::string vId;  // optional user-defined id, for ex: can be set to image hash, useful for KV cache tracking
  bool        vIsAudio   = false;  // true if the bitmap is audio
  bool        vMergeable = false;  // [QWEN_VIDEO] set only on frames of the same video

  // lazy-loaded bitmap
  mtmd_bitmap_lazy_callback vLazyCallback = nullptr;
  void *                    vLazyUserData = nullptr;

  mtmd_bitmap(const unsigned char * data, uint32_t pNx, uint32_t pNy) : vNx(pNx), vNy(pNy), vIsAudio(false) {
    if (data) {
      size_t vDataSize = (size_t) pNx * pNy * 3;
      this->data.resize(vDataSize);
      std::memcpy(this->data.data(), data, vDataSize);
    }
  }

  mtmd_bitmap(const unsigned char * data, uint32_t pNSamples) : vNx(pNSamples), vNy(1), vIsAudio(true) {
    if (data) {
      size_t vDataSize = (size_t) vNx * sizeof(float);
      this->data.resize(vDataSize);
      std::memcpy(this->data.data(), data, vDataSize);
    }
  }

  const std::vector<unsigned char> & mGetRoBuf() const { return data; }

  bool mIsPlaceholder() const { return data.empty(); }

  size_t mNBytes() const { return data.size(); }

  bool mCanMergeWith(const mtmd_bitmap & pOther) const {
    // [QWEN_VIDEO] can (temporal) merge if both are images with same size
    return vMergeable && pOther.vMergeable && !vIsAudio && !pOther.vIsAudio && vNx == pOther.vNx && vNy == pOther.vNy;
  }

private:
  std::vector<unsigned char> data;
};

// position indexing for decoder model
enum mtmd_pos_type {
  cMtmdPosTypeNormal,     // number of positions equals to number of tokens
  cMtmdPosTypeMrope,      // qwen-vl mrope style, each image takes max(t,h,w) position indexes
  cMtmdPosTypeHunyuanvl,  // HunyuanVL mrope + BOI/EOI/newline layout with XD-RoPE dim-3
  cMtmdPosTypeCanvas,  // MiniCPM-V 4.7: overview + slices in one chunk, sharing one 2D canvas (see mtmd_image_tokens::canvas_tile_grid)
  cMtmdPosTypeCount,  // for validation
};

struct mtmd_image_tokens {
  uint32_t      vNx  = 0;  // number of tokens in x direction
  uint32_t      vNy  = 0;  // number of tokens in y direction
  mtmd_pos_type vPos = cMtmdPosTypeNormal;
  uint32_t      vImageIdx =
    0;  // 0-based position of this image among image chunks in the prompt(used by pos == MTMD_POS_TYPE_HUNYUANVL)
  uint32_t vNTemporalMerge = 1;  // for qwen-vl style temporal merge

  uint32_t mNTokens() const {
    if (vPos == cMtmdPosTypeHunyuanvl) {
      // [BOI] [row0 tokens + newline] ... [row(ny-1) tokens + newline] [EOI]
      return (vNx + 1) * vNy + 2;
    }
    if (vPos == cMtmdPosTypeCanvas) {
      uint32_t n = 0;
      for (size_t vK = 0; vK < vBatchF32.lEntries.size(); ++vK) {
        const auto [gw, gh] = mCanvasTileGrid(vK);
        n += gw * gh + (uint32_t) fClipSuffixRows(vBatchF32.lEntries[vK].vSuffixType).size();
      }
      return n;
    }
    uint32_t vNz = vBatchF32.lEntries.size();
    if (vNTemporalMerge > 1) {
      // [QWEN_VIDEO] this logic is quite ugly, it's mostly to make qwen-vl temporal merge work, can be improved in the future
      // TODO: simplify this by repeating the last frame until it fits the temporal merge
      if (vNz % vNTemporalMerge != 0) {
        vNz = vNz / vNTemporalMerge + 1;
      } else {
        vNz = vNz / vNTemporalMerge;
      }
    }
    return vNx * vNy * vNz;
  }

  clip_image_f32_batch vBatchF32;  // preprocessed image patches
  std::string          vId;        // optional user-defined ID, useful for KV cache tracking

  // true if one of entries in batch_f32 is a placeholder
  bool mIsPlaceholder() const {
    for (const auto & vEntry : vBatchF32.lEntries) {
      if (vEntry.mIsPlaceholder()) {
        return true;
      }
    }
    return false;
  }

  // MTMD_POS_TYPE_CANVAS: entries are [overview, slices row by row], nx/ny is the token grid of the last entry
  // returns the token grid (w, h) of entry k, scaled from its pixel size
  std::pair<uint32_t, uint32_t> mCanvasTileGrid(size_t pK) const {
    const auto & vRef = vBatchF32.lEntries.back();
    const auto & vE   = vBatchF32.lEntries[pK];
    return { (uint32_t) vE.mNx() * vNx / vRef.mNx(), (uint32_t) vE.mNy() * vNy / vRef.mNy() };
  }

  bool mCanBatchWith(const mtmd_image_tokens & pOther) {
    // a canvas chunk holds a whole image group, its layout is not given by nx/ny alone
    return vNx == pOther.vNx && vNy == pOther.vNy && vPos == pOther.vPos && vPos != cMtmdPosTypeCanvas;
  }

  mtmd_image_tokens clone() {
    return mtmd_image_tokens{ vNx, vNy, vPos, vImageIdx, vNTemporalMerge, vBatchF32.clone(), vId };
  }

  void serialize(mtmd_serialization & pSer) const {
    // remember to bump MTMD_SERIALIZATION_VERSION if this is changed
    pSer.write(vNx);
    pSer.write(vNy);
    pSer.write((uint32_t) vPos);
    pSer.write(vImageIdx);
    pSer.write(vNTemporalMerge);
    pSer.write(vId);
    vBatchF32.serialize(pSer);
  }

  void deserialize(mtmd_serialization & pSer) {
    vNx              = pSer.read<uint32_t>();
    vNy              = pSer.read<uint32_t>();
    uint32_t vPosRaw = pSer.read<uint32_t>();
    if (vPosRaw >= cMtmdPosTypeCount) {
      throw std::runtime_error("invalid pos type");
    }
    vPos            = (mtmd_pos_type) vPosRaw;
    vImageIdx       = pSer.read<uint32_t>();
    vNTemporalMerge = pSer.read<uint32_t>();
    vId             = pSer.read<std::string>();
    vBatchF32.deserialize(pSer);
  }
};

using mtmd_image_tokens_ptr = std::unique_ptr<mtmd_image_tokens>;

struct mtmd_audio_tokens {
  uint32_t             vNTokens = 0;  // number of tokens
  clip_image_f32_batch vBatchF32;     // preprocessed image patches
  std::string          vId;           // optional user-defined ID, useful for KV cache tracking

  // true if one of entries in batch_f32 is a placeholder
  bool mIsPlaceholder() const {
    for (const auto & vEntry : vBatchF32.lEntries) {
      if (vEntry.mIsPlaceholder()) {
        return true;
      }
    }
    return false;
  }

  mtmd_audio_tokens clone() { return mtmd_audio_tokens{ vNTokens, vBatchF32.clone(), vId }; }

  void serialize(mtmd_serialization & pSer) const {
    // remember to bump MTMD_SERIALIZATION_VERSION if this is changed
    pSer.write(vNTokens);
    pSer.write(vId);
    vBatchF32.serialize(pSer);
  }

  void deserialize(mtmd_serialization & pSer) {
    vNTokens = pSer.read<uint32_t>();
    vId      = pSer.read<std::string>();
    vBatchF32.deserialize(pSer);
  }
};

using mtmd_audio_tokens_ptr = std::unique_ptr<mtmd_audio_tokens>;

struct mtmd_input_chunk {
  mtmd_input_chunk_type  type;
  std::vector<dgs_token> lTokensText;
  mtmd_image_tokens_ptr  vTokensImage;
  mtmd_audio_tokens_ptr  vTokensAudio;

  bool mCanBatchWith(const mtmd_input_chunk & pOther) const {
    if (type != pOther.type) {
      return false;
    }

    if (vTokensImage && pOther.vTokensImage) {
      return vTokensImage->mCanBatchWith(*pOther.vTokensImage);
    }

    // TODO: allow batching audio chunks of the same size

    return false;
  }

  bool mIsPlaceholder() const {
    if (type == cMtmdInputChunkTypeImage) {
      return vTokensImage && vTokensImage->mIsPlaceholder();
    } else if (type == cMtmdInputChunkTypeAudio) {
      return vTokensAudio && vTokensAudio->mIsPlaceholder();
    }
    return false;
  }

  void serialize(mtmd_serialization & pSer) const {
    // remember to bump MTMD_SERIALIZATION_VERSION if this is changed
    pSer.write((uint32_t) type);

    pSer.write<uint64_t>(lTokensText.size());
    for (dgs_token vTok : lTokensText) {
      pSer.write((int32_t) vTok);
    }

    pSer.write(vTokensImage != nullptr);
    if (vTokensImage) {
      vTokensImage->serialize(pSer);
    }

    pSer.write(vTokensAudio != nullptr);
    if (vTokensAudio) {
      vTokensAudio->serialize(pSer);
    }
  }

  void deserialize(mtmd_serialization & pSer) {
    uint32_t vTypeRaw = pSer.read<uint32_t>();
    if (vTypeRaw >= cMtmdInputChunkTypeCount) {
      throw std::runtime_error("invalid chunk type");
    }
    type = (mtmd_input_chunk_type) vTypeRaw;

    uint64_t vNTokensText = pSer.read<uint64_t>();
    // reject before resize() so a tiny corrupted/malicious buffer can't force a huge allocation
    if (vNTokensText > (pSer.data.size() - pSer.vReadPos) / sizeof(int32_t)) {
      throw std::runtime_error("tokens_text length exceeds buffer size");
    }
    lTokensText.resize(vNTokensText);
    for (uint64_t vI = 0; vI < vNTokensText; vI++) {
      lTokensText[vI] = (dgs_token) pSer.read<int32_t>();
    }

    if (pSer.read<bool>()) {
      vTokensImage = std::make_unique<mtmd_image_tokens>();
      vTokensImage->deserialize(pSer);
    } else {
      vTokensImage.reset();
    }

    if (pSer.read<bool>()) {
      vTokensAudio = std::make_unique<mtmd_audio_tokens>();
      vTokensAudio->deserialize(pSer);
    } else {
      vTokensAudio.reset();
    }

    // catch buffers where the declared type doesn't match which payload is actually present,
    // so a mismatched chunk can't slip through and null-deref/abort later in an accessor
    if (type == cMtmdInputChunkTypeImage && !vTokensImage) {
      throw std::runtime_error("type is IMAGE but tokens_image is missing");
    }
    if (type == cMtmdInputChunkTypeAudio && !vTokensAudio) {
      throw std::runtime_error("type is AUDIO but tokens_audio is missing");
    }
  }
};

struct mtmd_input_chunks {
  std::vector<mtmd_input_chunk> lEntries;
};

struct mtmd_batch {
  mtmd_context *                        vCtx;
  std::vector<const mtmd_input_chunk *> lEntries;
  std::vector<float>                    lOutputEmbd;  // aggregated output embedding for the whole batch

  mtmd_batch(mtmd_context * pCtx) : vCtx(pCtx) {}

  int32_t mNTokens() const {
    int32_t n = 0;
    for (const auto * vChunk : lEntries) {
      n += fMtmdInputChunkGetNTokens(vChunk);
    }
    return n;
  }
};

// slice template, used by some llava-uhd models to correctly place the special tokens around image embeddings
// models not having it (llava-1.6) will process embeddings without any special tokens in-between
enum mtmd_slice_tmpl {
  cMtmdSliceTmplNone,
  cMtmdSliceTmplMinicpmv2Point5,
  cMtmdSliceTmplMinicpmv2Point6,
  cMtmdSliceTmplDgs4,
  cMtmdSliceTmplIdefics3,
  cMtmdSliceTmplLfm2,
  cMtmdSliceTmplStep3vl,
};

const char * fMtmdDefaultMarker() {
  return "<__media__>";
}

static clip_flash_attn_type fMtmdGetClipFlashAttnType(enum dgs_flash_attn_type pFlashAttnType) {
  switch (pFlashAttnType) {
    case cDgsFlashAttnTypeAuto:
      return cClipFlashAttnTypeAuto;
    case cDgsFlashAttnTypeDisabled:
      return cClipFlashAttnTypeDisabled;
    case cDgsFlashAttnTypeEnabled:
      return cClipFlashAttnTypeEnabled;
  }
  return cClipFlashAttnTypeAuto;
}

mtmd_context_params fMtmdContextParamsDefault() {
  mtmd_context_params vParams{
    /* use_gpu           */ true,
    /* device            */ nullptr,
    /* print_timings     */ true,
    /* n_threads         */ 4,
    /* media_marker      */ fMtmdDefaultMarker(),
    /* flash_attn_type   */ cDgsFlashAttnTypeAuto,
    /* warmup            */ true,
    /* image_min_tokens  */ -1,
    /* image_max_tokens  */ -1,
    /* cb_eval           */ nullptr,
    /* cb_eval_user_data */ nullptr,
    /* batch_max_tokens  */ 1024,
    /* progress_callback */ nullptr,
    /* progress_callback_user_data */ nullptr,
  };
  return vParams;
}

struct mtmd_context {
  struct clip_ctx *  vCtxV;     // vision
  struct clip_ctx *  vCtxA;     // audio
  std::vector<float> lOutEmbd;  // image embedding vector

  // generation context
  struct clip_ctx *    vCtxGenA;      // audio
  std::vector<int32_t> lGenOutCodes;  // this frame's 16 sampled codes (GEN_CODE)
  std::vector<float>   lGenOutFeats;  // this frame's continuous features, if any (GEN_CODE)
  std::vector<float>   lGenOutEmbd;   // next-step hidden state fed back to backbone (GEN_CODE)
  std::vector<float>   lGenOutAudio;  // decoded PCM samples for the current frame (GEN_WAV)
  std::vector<uint8_t> lGenOutState;  // state to feed into the next GEN_WAV call

  bool              vPrintTimings;
  int               vNThreads;
  std::string       vMediaMarker;
  const int         cNEmbdText = -1;       // -1 means llm context not provided, skip checking this
  const dgs_vocab * vVocab     = nullptr;  // can be nullptr if text_model is not provided
  mtmd_pos_type     vPosType;

  // these are not token, but strings used to mark the beginning and end of image/audio embeddings
  std::string vImgBeg;
  std::string vImgEnd;
  std::string vAudBeg;
  std::string vAudEnd;

  // for llava-uhd style models, we need special tokens in-between slices
  // minicpmv calls them "slices", dgs 4 calls them "tiles"
  mtmd_slice_tmpl        vSliceTmpl = cMtmdSliceTmplNone;
  std::vector<dgs_token> lTokOvImgStart;   // overview image
  std::vector<dgs_token> lTokOvImgEnd;     // overview image
  std::vector<dgs_token> lTokSlicesStart;  // start of all slices
  std::vector<dgs_token> lTokSlicesEnd;    // end of all slices
  std::vector<dgs_token> lTokSliImgStart;  // single slice start
  std::vector<dgs_token> lTokSliImgEnd;    // single slice end
  std::vector<dgs_token> lTokSliImgMid;    // between 2 slices
  std::vector<dgs_token> lTokRowEnd;       // end of row
  bool                   vTokRowEndTrail = false;
  bool                   vOvImgFirst     = false;

  // MiniCPM-V 4.6/4.7 prepends an <image_id>N</image_id> tag before <image>
  bool vUseImageId = false;

  // string template for slice image delimiters with row/col (idefics3)
  std::string vSliImgStartTmpl;

  std::unique_ptr<mtmd_audio_preprocessor> vAudioPreproc;
  std::unique_ptr<mtmd_image_preprocessor> vImagePreproc;

  // batching
  int32_t vBatchMaxTokens;

  // TODO @ngxson : add timings

  mtmd_context(const char *                pMmprojFname,
               const dgs_model *           pTextModel,
               const mtmd_context_params & pCtxParams,
               bool                        pNoAlloc = false) :
    vPrintTimings(pCtxParams.vPrintTimings),
    vNThreads(pCtxParams.vNThreads),
    vMediaMarker(pCtxParams.vMediaMarker),
    cNEmbdText(pTextModel ? fDgsModelNEmbdInp(pTextModel) : -1),
    vVocab(pTextModel ? fDgsModelGetVocab(pTextModel) : nullptr),
    vBatchMaxTokens(pCtxParams.vBatchMaxTokens) {
    if (vMediaMarker.empty()) {
      throw std::runtime_error("media_marker must not be empty");
    }

    if (pTextModel) {
      auto vDecoderRopeType = fDgsModelRopeType(pTextModel);
      switch (vDecoderRopeType) {
        case cDgsRopeTypeNone:
        case cDgsRopeTypeNorm:
        case cDgsRopeTypeNeox:
          {
            vPosType = cMtmdPosTypeNormal;
          }
          break;
        case cDgsRopeTypeMrope:
        case cDgsRopeTypeImrope:
          {
            vPosType = cMtmdPosTypeMrope;
          }
          break;
        default:
          throw std::runtime_error(fStringFormat("unsupported decoder rope type: %d\n", vDecoderRopeType));
      }
    }

    clip_context_params vCtxClipParams{
      /* use_gpu           */ pCtxParams.vUseGpu,
      /* device            */ pCtxParams.vDevice,
      /* flash_attn_type   */ fMtmdGetClipFlashAttnType(pCtxParams.vFlashAttnType),
      /* image_min_tokens  */ pCtxParams.vImageMinTokens,
      /* image_max_tokens  */ pCtxParams.vImageMaxTokens,
      /* warmup            */ pCtxParams.vWarmup,
      /* cb_eval           */ pCtxParams.vCbEval,
      /* cb_eval_user_data */ pCtxParams.vCbEvalUserData,
      /* no_alloc          */ pNoAlloc,
      /* progress_callback */ pCtxParams.vProgressCallback,
      /* progress_callback_user_data */ pCtxParams.vProgressCallbackUserData,
    };

    auto vRes = fClipInit(pMmprojFname, vCtxClipParams);
    vCtxV     = vRes.vCtxV;
    vCtxA     = vRes.vCtxA;
    vCtxGenA  = vRes.vCtxGenA;
    if (!vCtxV && !vCtxA) {
      throw std::runtime_error(fStringFormat("Failed to load CLIP model from %s\n", pMmprojFname));
    }

    // if both vision and audio mmproj are present, we need to validate their n_embd
    if (vCtxV && vCtxA) {
      int vNEmbdV = fClipNMmprojEmbd(vCtxV);
      int vNEmbdA = fClipNMmprojEmbd(vCtxA);
      if (vNEmbdV != vNEmbdA) {
        throw std::runtime_error(
          fStringFormat("mismatch between vision and audio mmproj (n_embd_v = %d, n_embd_a = %d)\n", vNEmbdV, vNEmbdA));
      }
    }

    // since we already validate n_embd of vision and audio mmproj,
    // we can safely assume that they are the same
    int vNEmbdClip = fClipNMmprojEmbd(vCtxV ? vCtxV : vCtxA);
    if (cNEmbdText > 0 && cNEmbdText != vNEmbdClip) {
      throw std::runtime_error(
        fStringFormat("mismatch between text model (n_embd = %d) and mmproj (n_embd = %d)\n"
                      "hint: you may be using wrong mmproj\n",
                      cNEmbdText, vNEmbdClip));
    }
    if (vCtxGenA) {
      int vNEmbdGen = fClipNMmprojEmbd(vCtxGenA);
      if (cNEmbdText > 0 && cNEmbdText != vNEmbdGen) {
        throw std::runtime_error(
          fStringFormat("mismatch between text model (n_embd = %d) and gen-audio mmproj (n_embd = %d)\n"
                        "hint: you may be using wrong mmproj\n",
                        cNEmbdText, vNEmbdGen));
      }
    }
    if (vCtxV) {
      mInitVision();
    }
    if (vCtxA) {
      mInitAudio();
    }
  }

  void mInitVision() {
    GGML_ASSERT(vCtxV != nullptr);
    vImagePreproc.reset();

    projector_type vProj = fClipGetProjectorType(vCtxV);

    switch (vProj) {
      case cProjectorTypeQwen3vl:

        {
          // <|vision_start|> ... (image embeddings) ... <|vision_end|>
          vImgBeg       = "<|vision_start|>";
          vImgEnd       = "<|vision_end|>";
          vImagePreproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(vCtxV);
        }
        break;
      default:
        throw std::runtime_error(fStringFormat("%s: unexpected vision projector type %d\n", __func__, vProj));
    }

    GGML_ASSERT(vImagePreproc != nullptr);
  }

  void mInitAudio() {
    GGML_ASSERT(vCtxA != nullptr);
    vAudioPreproc.reset();

    projector_type vProj = fClipGetProjectorType(vCtxA);

    LOG_WRN(
      "%s: audio input is in experimental stage and may have reduced quality:\n"
      "    doc/MANUAL.md\n",
      __func__);

    // set preprocessor
    switch (vProj) {
      default:
        throw std::runtime_error(fStringFormat("%s: unexpected audio projector type %d\n", __func__, vProj));
    }

    // initialize audio preprocessor
    GGML_ASSERT(vAudioPreproc != nullptr);
    vAudioPreproc->mInitialize();
  }

  // get clip ctx based on chunk type
  clip_ctx * mGetClipCtx(const mtmd_input_chunk * pChunk) const {
    if (pChunk->type == cMtmdInputChunkTypeImage) {
      return vCtxV;
    } else if (pChunk->type == cMtmdInputChunkTypeAudio) {
      return vCtxA;
    }
    GGML_ABORT("unknown chunk type");
  }

  projector_type mProjTypeV() const { return vCtxV ? fClipGetProjectorType(vCtxV) : cProjectorTypeUnknown; }

  projector_type mProjTypeA() const { return vCtxA ? fClipGetProjectorType(vCtxA) : cProjectorTypeUnknown; }

  int64_t mNEmbdOut() const {
    if (vCtxV) {
      return fClipNMmprojEmbd(vCtxV);
    } else if (vCtxA) {
      return fClipNMmprojEmbd(vCtxA);
    } else {
      throw std::runtime_error("no CLIP model loaded");
    }
  }

  ~mtmd_context() {
    fClipFree(vCtxA);
    fClipFree(vCtxV);
    fClipFree(vCtxGenA);
  }

private:
  dgs_token mLookupToken(const std::string & pTokenText) {
    if (vVocab == nullptr) {
      // TODO @ngxson : this case is currently hit by mtmd_get_memory_usage
      // but we should reconsider this if this case is needed in other places in the future
      return DGS_TOKEN_NULL;
    }
    const int cNVocab = fDgsVocabNTokens(vVocab);
    for (int vI = 0; vI < cNVocab; vI++) {
      if (mTokenToPiece(vVocab, vI, true) == pTokenText) {
        return vI;
      }
    }
    return DGS_TOKEN_NULL;
  }

  std::string mTokenToPiece(const dgs_vocab * pVocab, dgs_token token, bool pSpecial) {
    if (pVocab == nullptr) {
      throw std::runtime_error("dgs_vocab is not provided");
    }
    std::string vPiece;
    vPiece.resize(vPiece.capacity());  // using string internal cache, 15 bytes + '\n'
    const int cNChars = fDgsTokenToPiece(pVocab, token, &vPiece[0], vPiece.size(), 0, pSpecial);
    if (cNChars < 0) {
      vPiece.resize(-cNChars);
      int vCheck = fDgsTokenToPiece(pVocab, token, &vPiece[0], vPiece.size(), 0, pSpecial);
      GGML_ASSERT(vCheck == -cNChars);
    } else {
      vPiece.resize(cNChars);
    }
    return vPiece;
  }
};

mtmd_context * fMtmdInitFromFile(const char *                     pMmprojFname,
                                 const struct dgs_model *         pTextModel,
                                 const struct mtmd_context_params pCtxParams) {
  try {
    return new mtmd_context(pMmprojFname, pTextModel, pCtxParams);
  } catch (const std::exception & vE) {
    LOG_ERR("%s: error: %s\n", __func__, vE.what());
    return nullptr;
  }
}

void fMtmdFree(mtmd_context * pCtx) {
  delete pCtx;
}

std::vector<std::vector<const mtmd_bitmap *>> fMtmdGroupMergeableBitmaps(std::vector<mtmd_internal_part> & pParts,
                                                                         int                               pNMerge) {
  std::vector<std::vector<const mtmd_bitmap *>> lOutput;
  for (size_t vI = 0; vI < pParts.size(); vI++) {
    if (pParts[vI].bitmap == nullptr) {
      continue;  // text part
    }
    const bool cHasNext = pNMerge > 1 && vI + 1 < pParts.size() && pParts[vI + 1].bitmap != nullptr;
    if (cHasNext && pParts[vI].bitmap->mCanMergeWith(*pParts[vI + 1].bitmap)) {
      LOG_DBG("%s: merging 2 frames at part index %zu and %zu\n", __func__, vI, vI + 1);
      lOutput.push_back({ pParts[vI].bitmap, pParts[vI + 1].bitmap });
      pParts.erase(pParts.begin() + vI + 1);
      continue;
    }
    LOG_DBG("%s: no merging for part index %zu\n", __func__, vI);
    lOutput.push_back({ pParts[vI].bitmap });
  }
  return lOutput;
}

struct mtmd_tokenizer {
  const mtmd_context * vCtx;

  std::string       vInputText;  // note: can contain null bytes; do not use c_str()
  bool              vAddSpecial;
  bool              vParseSpecial;
  const dgs_vocab * vVocab;

  using part = mtmd_internal_part;
  std::vector<part>         lParts;
  // these will be freed when mtmd_tokenizer finishes
  std::vector<mtmd::bitmap> lBmFromLazy;  // TODO @ngxson : refactor, free bm_from_lazy progressively
  std::vector<const char *> lTextFromLazy;

  mtmd_input_chunks vCur;
  uint32_t          vNImagesAdded = 0;  // 0-based index assigned to the next image chunk

  ~mtmd_tokenizer() {
    // note: mtmd::bitmap is already RAII
    for (auto & str : lTextFromLazy) {
      free((void *) str);
    }
  }

  mtmd_tokenizer(const mtmd_context *        pCtx,
                 const mtmd_input_text *     pText,
                 const mtmd_bitmap * const * pBmps,
                 size_t                      pNBitmaps) :
    vCtx(pCtx) {
    vAddSpecial   = pText->vAddSpecial;
    vParseSpecial = pText->vParseSpecial;
    vVocab        = pCtx->vVocab;

    vInputText.assign(pText->vText, pText->vTextLen);

    std::vector<const mtmd_bitmap *> bitmaps(pBmps, pBmps + pNBitmaps);
    auto                             lPartsStr = mSplitText(vInputText, pCtx->vMediaMarker);
    size_t                           vIBm      = 0;
    for (const auto & vPart : lPartsStr) {
      if (vPart == pCtx->vMediaMarker) {
        if (vIBm >= bitmaps.size()) {
          throw std::runtime_error(fStringFormat(
            "number of media markers in text (%zu) exceeds number of bitmaps (%zu)", vIBm + 1, bitmaps.size()));
        }
        lParts.push_back({ "", bitmaps[vIBm++] });
      } else {
        lParts.push_back({ std::move(vPart), nullptr, vParseSpecial });
      }
    }

    size_t vNMarkers = 0;
    for (const auto & vPart : lParts) {
      if (vPart.bitmap != nullptr) {
        vNMarkers++;
      }
    }
    if (vNMarkers != bitmaps.size()) {
      throw std::runtime_error(fStringFormat(
        "number of media markers in text (%zu) does not match number of bitmaps (%zu)", vNMarkers, bitmaps.size()));
    }

    mExpandLazyBitmaps();
  }

  mtmd_tokenizer(const mtmd_context *            pCtx,
                 const mtmd_input_part * const * pInputParts,
                 size_t                          pNParts,
                 bool                            pAddSpecial) :
    vCtx(pCtx) {
    this->vAddSpecial = pAddSpecial;
    vParseSpecial     = true;  // only used for text returned by lazy bitmaps
    vVocab            = pCtx->vVocab;

    for (size_t vI = 0; vI < pNParts; vI++) {
      const mtmd_input_part * vP = pInputParts[vI];
      if (vP->vText != nullptr) {
        lParts.push_back({ std::string(vP->vText->vText, vP->vText->vTextLen), nullptr, vP->vText->vParseSpecial });
      } else {
        lParts.push_back({ "", vP->bitmap });
      }
    }

    mExpandLazyBitmaps();
  }

  void mExpandLazyBitmaps() {
    std::vector<part> lExpanded;
    lExpanded.reserve(lParts.size());
    for (auto & vP : lParts) {
      if (vP.bitmap != nullptr && vP.bitmap->vLazyCallback) {
        LOG_DBG("%s: expanding lazy bitmap\n", __func__);
        for (size_t vI = 0;; vI++) {
          char *        vOutStr = nullptr;
          mtmd_bitmap * vOutBm  = nullptr;
          int           vRes    = vP.bitmap->vLazyCallback(vI, vP.bitmap->vLazyUserData, &vOutBm, &vOutStr);
          if (vOutBm && vOutStr) {
            throw std::runtime_error(fStringFormat("lazy callback cannot return both bitmap and text"));
          }
          if (vRes == 0) {
            // OK, append the returned chunk; lazy part is not yet added
            if (vOutBm) {
              auto & vPtr = lBmFromLazy.emplace_back(vOutBm);  // remember to free it later
              lExpanded.push_back({ "", vPtr.vPtr.get() });
              LOG_DBG("%s: lazy callback returned bitmap with dimensions %d x %d\n", __func__, vOutBm->vNx,
                      vOutBm->vNy);
            } else if (vOutStr) {
              auto & vPtr = lTextFromLazy.emplace_back(vOutStr);  // remember to free it later
              lExpanded.push_back({ vPtr, nullptr, vParseSpecial });
              LOG_DBG("%s: lazy callback returned text: %s\n", __func__, vOutStr);
            }
          } else if (vRes == -1) {
            // EOF: lazy part removes itself (not added to expanded)
            break;
          } else if (vRes == -2) {
            // error
            throw std::runtime_error(fStringFormat("lazy callback returned error"));
          }
        }
      } else {
        lExpanded.push_back(std::move(vP));
      }
    }
    lParts = std::move(lExpanded);
  }

  int32_t mTokenize(mtmd_input_chunks * pOutput) {
    vCur.lEntries.clear();

    // [QWEN_VIDEO] handle frame merging for models that support it (i.e. qwen-vl)
    int vNMergeFrames = 1;
    if (vCtx->vCtxV) {
      vNMergeFrames = fClipModelNTemporalMerge(vCtx->vCtxV);
      GGML_ASSERT(
        vNMergeFrames <= 2 &&
        "we only support merging maximum 2 images for now; open an issue if this model supports merging more");
    }

    auto lMergedBitmaps = fMtmdGroupMergeableBitmaps(lParts, vNMergeFrames);

    size_t vIBm = 0;
    for (const auto & vP : lParts) {
      if (vP.bitmap != nullptr) {
        if (vIBm >= lMergedBitmaps.size()) {
          LOG_ERR("%s: error: number of bitmaps (%zu) does not match number of markers (%zu)\n", __func__,
                  lMergedBitmaps.size(), lParts.size() - 1);
          return 1;
        }
        auto    lBmps = lMergedBitmaps[vIBm++];
        int32_t vRes  = mAddMedia(lBmps);
        if (vRes != 0) {
          return vRes;
        }
      } else {
        mAddText(vP.vText, vP.vParseSpecial);
      }
    }

    if (vVocab != nullptr) {
      if (vAddSpecial && fDgsVocabGetAddBos(vVocab)) {
        // if first chunk is text, we add BOS token to first text chunk
        // otherwise, create a new text chunk with BOS token
        if (!vCur.lEntries.empty() && vCur.lEntries[0].type == cMtmdInputChunkTypeText) {
          // add BOS token to the beginning of first text chunk
          vCur.lEntries[0].lTokensText.insert(vCur.lEntries[0].lTokensText.begin(), fDgsVocabBos(vVocab));
        } else {
          // create a new text chunk with BOS token at the beginning
          mtmd_input_chunk vBosChunk{
            cMtmdInputChunkTypeText,
            { fDgsVocabBos(vVocab) },
            nullptr,  // image tokens
            nullptr,  // audio tokens
          };
          vCur.lEntries.insert(vCur.lEntries.begin(), std::move(vBosChunk));
        }
      }

      if (vAddSpecial && fDgsVocabGetAddEos(vVocab)) {
        // if last chunk is text, we add EOS token to it
        mAddText({ fDgsVocabEos(vVocab) });
      }
    }

    if (vIBm != lMergedBitmaps.size()) {
      LOG_ERR("%s: error: number of bitmaps (%zu) does not match number of markers (%zu)\n", __func__,
              lMergedBitmaps.size(), lParts.size() - 1);
      return 1;
    }

    *pOutput = std::move(vCur);

    return 0;
  }

  void mAddText(const std::string & pTxt, bool pParseSpecial) {
    if (vVocab == nullptr) {
      throw std::runtime_error("dgs_vocab is not provided");
    }
    LOG_DBG("%s: %s\n", __func__, pTxt.c_str());
    auto lTokens = mMtmdTokenizeTextInternal(vVocab, pTxt, /* add_special */ false, pParseSpecial);
    mAddText(lTokens);
  }

  void mAddText(const std::vector<dgs_token> & pTokens) {
    if (pTokens.empty()) {
      return;
    }
    // if last entry is also a text chunk, add tokens to it instead of creating new chunk
    if (!vCur.lEntries.empty() && vCur.lEntries.back().type == cMtmdInputChunkTypeText) {
      vCur.lEntries.back().lTokensText.insert(vCur.lEntries.back().lTokensText.end(), pTokens.begin(), pTokens.end());
    } else {
      mtmd_input_chunk vChunk{
        cMtmdInputChunkTypeText,
        pTokens,
        nullptr,  // image tokens
        nullptr,  // audio tokens
      };
      vCur.lEntries.emplace_back(std::move(vChunk));
    }
  }

  int32_t mAddMedia(std::vector<const mtmd_bitmap *> & bitmaps) {
    GGML_ASSERT(!bitmaps.empty());

    // note: only one type of media is supported per call, caller should enforce this
    const bool cIsVision = !bitmaps[0]->vIsAudio;

    if (cIsVision) {
      // handle image

      if (!vCtx->vCtxV) {
        LOG_ERR("%s: error: model does not support vision input\n", __func__);
        return 2;
      }

      if (!vCtx->vImgBeg.empty()) {
        mAddText(vCtx->vImgBeg, true);  // add image begin token
      }

      // TODO @ngxson : this is quite hacky because preprocessor only support batch with one single element, that need to be fixed in the future (e.g. by changing the preprocessor interface always take single input)

      mtmd_image_preproc_out vPreprocOut;

      for (const auto * vBmp : bitmaps) {
        // sanity check
        GGML_ASSERT(!vBmp->vIsAudio);
        GGML_ASSERT(vCtx->vImagePreproc != nullptr);
        if (vBmp->vNx <= 0 || vBmp->vNy <= 0) {
          LOG_ERR("%s: error: invalid bitmap dimensions: nx = %d, ny = %d\n", __func__, vBmp->vNx, vBmp->vNy);
          return 2;
        }

        // convert mtmd_bitmap to clip_image_u8
        clip_image_u8 vImgU8;
        vImgU8.mSetSize({ (int) vBmp->vNx, (int) vBmp->vNy }, vBmp->mIsPlaceholder());
        vImgU8.mCpyBuf(vBmp->mGetRoBuf());

        // preprocess image
        mtmd_image_preproc_out vTmpPreprocOut = vCtx->vImagePreproc->mPreprocess(vImgU8);

        // move entries and grid dimensions to the "global" preproc_out
        for (auto & vEntry : vTmpPreprocOut.lEntries) {
          vPreprocOut.lEntries.emplace_back(std::move(vEntry));
        }

        // for llava-uhd style, we need to handle grid too
        // we don't care about overwriting these values for now because the case where bitmaps.size() > 1 is only for frame merging (qwen-vl), not supported by llava-uhd
        if ((vTmpPreprocOut.vGridX > 0 && vTmpPreprocOut.vGridY > 0) || vTmpPreprocOut.mHasOverview()) {
          GGML_ASSERT(bitmaps.size() == 1);
          vPreprocOut.vGridX    = vTmpPreprocOut.vGridX;
          vPreprocOut.vGridY    = vTmpPreprocOut.vGridY;
          vPreprocOut.vOverview = std::move(vTmpPreprocOut.vOverview);
        }
      }

      LOG_DBG("%s: preproc_out has %zu entries, grid_x = %d, grid_y = %d, has_overview = %d\n", __func__,
              vPreprocOut.lEntries.size(), vPreprocOut.vGridX, vPreprocOut.vGridY, vPreprocOut.mHasOverview() ? 1 : 0);

      // handle llava-uhd style preprocessing
      // (output either a grid, or overview-only)
      const bool cHasTilingGrid = (vPreprocOut.vGridX > 0 && vPreprocOut.vGridY > 0) || vPreprocOut.mHasOverview();

      if (cHasTilingGrid && vCtx->mProjTypeV() == cProjectorTypeMINICPMV4Point7) {
        GGML_ASSERT(bitmaps.size() == 1);
        if (vCtx->vUseImageId) {
          mAddText("<image_id>" + std::to_string(vNImagesAdded) + "</image_id>", true);
        }
        mAddText(vCtx->lTokOvImgStart);
        // the separators after <image> are appended by clip, see add_canvas_chunk()
        mAddCanvasChunk(std::move(vPreprocOut), bitmaps[0]->vId);
      } else if (cHasTilingGrid) {
        // [QWEN_VIDEO] we do not support "frame merging" for dgs-uhd style, so no batching for now
        GGML_ASSERT(bitmaps.size() == 1);

        const int cNCol = vPreprocOut.vGridX;
        const int cNRow = vPreprocOut.vGridY;

        // split batch into chunks of single images
        auto lChunks = mSplitBatchToChunk(std::move(vPreprocOut), bitmaps[0]->vId);
        GGML_ASSERT(lChunks.size() > 0);

        // NOTE: preproc_out is invalidated after this point, do not use it anymore

        // split_batch_to_chunk must always put the overview image first
        auto vOvChunk = std::move(lChunks.front());
        lChunks.erase(lChunks.begin());

        // add overview image (first)
        if (vCtx->vOvImgFirst) {
          if (vCtx->vUseImageId) {
            mAddText("<image_id>" + std::to_string(vNImagesAdded) + "</image_id>", true);
          }
          mAddText(vCtx->lTokOvImgStart);
          vCur.lEntries.emplace_back(std::move(vOvChunk));
          mAddText(vCtx->lTokOvImgEnd);
        }

        // add slices (or tiles)
        if (!lChunks.empty()) {
          LOG_DBG("%s: adding %d slices (%d rows x %d cols)\n", __func__, (int) lChunks.size(), cNRow, cNCol);
          GGML_ASSERT((int) lChunks.size() == cNRow * cNCol);
          mAddText(vCtx->lTokSlicesStart);
          for (int vY = 0; vY < cNRow; vY++) {
            for (int vX = 0; vX < cNCol; vX++) {
              const bool cIsLastInRow = (vX == cNCol - 1);
              if (!vCtx->lTokSliImgStart.empty()) {
                mAddText(vCtx->lTokSliImgStart);
              } else if (!vCtx->vSliImgStartTmpl.empty()) {
                // If using a template to preceed a slice image
                const size_t cSz = std::snprintf(nullptr, 0, vCtx->vSliImgStartTmpl.c_str(), vY + 1, vX + 1) + 1;
                std::unique_ptr<char[]> vBuf(new char[cSz]);
                std::snprintf(vBuf.get(), cSz, vCtx->vSliImgStartTmpl.c_str(), vY + 1, vX + 1);
                mAddText(std::string(vBuf.get(), vBuf.get() + cSz - 1), true);
              }

              auto & vCurrChunk = lChunks[vY * cNCol + vX];
              auto & vCurrBatch = vCurrChunk.vTokensImage->vBatchF32;
              if (vCurrBatch.lEntries.size() != 1) {
                throw std::runtime_error(fStringFormat("%s: expect 1 image in batch_f32", __func__));
              }

              LOG_DBG("%s: adding slice image at row %d col %d\n", __func__, vY, vX);
              vCur.lEntries.emplace_back(std::move(vCurrChunk));

              mAddText(vCtx->lTokSliImgEnd);
              if (!cIsLastInRow) {
                mAddText(vCtx->lTokSliImgMid);
              }
            }
            if ((vY != cNRow - 1 || vCtx->vTokRowEndTrail)) {
              mAddText(vCtx->lTokRowEnd);
            }
          }
          mAddText(vCtx->lTokSlicesEnd);
        }

        // add overview image (last)
        if (!vCtx->vOvImgFirst) {
          mAddText(vCtx->lTokOvImgStart);
          vCur.lEntries.emplace_back(std::move(vOvChunk));
          mAddText(vCtx->lTokOvImgEnd);
        }
      } else {
        if (vPreprocOut.lEntries.size() == 0) {
          LOG_ERR(
            "%s: no image tokens produced by preprocessor (ref: doc/MANUAL.md)\n",
            __func__);
          return 2;
        }

        if (vCtx->mProjTypeV() == cProjectorTypeDeepseek4v) {
          // the text model perceives input in blocks of N tokens (N = COMPRESS_PAD_TO = 4, same as the CSA compress ratio)
          // image need to be aligned to block size, while adding IMAGE_PAD embeddings to the beginning
          // TODO @ngxson : maybe refactor this in the future
          constexpr int32_t cAlign = 4;
          size_t            vNPast = 0;
          for (const auto & vE : vCur.lEntries) {
            vNPast += fMtmdInputChunkGetNTokens(&vE);
          }
          vPreprocOut.lEntries[0].vLeadPad = cAlign - 1 - (int32_t) (vNPast % cAlign);
        }

        size_t vNTokens = 0;
        for (auto & vE : vPreprocOut.lEntries) {
          vNTokens += fClipNOutputTokens(vCtx->vCtxV, &vE);
          if (fClipModelNTemporalMerge(vCtx->vCtxV) == 2) {
            // [QWEN_VIDEO] pair input is merged to the same embd, so only count as one image
            break;
          }
        }

        mtmd_image_tokens_ptr vImageTokens(new mtmd_image_tokens);

        // [QWEN_VIDEO] improve this in the future
        vImageTokens->vNTemporalMerge = fClipModelNTemporalMerge(vCtx->vCtxV);

        if (fMtmdDecodeUseMrope(vCtx)) {
          // for Qwen2VL, we need this information for M-RoPE decoding positions
          vImageTokens->vNx = fClipNOutputTokensX(vCtx->vCtxV, &vPreprocOut.lEntries[0]);
          vImageTokens->vNy = fClipNOutputTokensY(vCtx->vCtxV, &vPreprocOut.lEntries[0]);
        } else {
          // other models, we only need the total number of tokens
          vImageTokens->vNx = vNTokens;
          vImageTokens->vNy = 1;
        }
        vImageTokens->vPos = vCtx->vPosType;
        // HunyuanVL wraps the image grid with BOI/EOI and adds one newline per row,
        // and uses XD-RoPE (dim-3 = image index). Override the position type so that
        // n_tokens() and mtmd_image_tokens_get_decoder_pos pick the HunyuanVL layout.
        if (vCtx->mProjTypeV() == cProjectorTypeHunyuanvl) {
          vImageTokens->vPos      = cMtmdPosTypeHunyuanvl;
          vImageTokens->vImageIdx = vNImagesAdded;
          GGML_ASSERT(vNTokens == (size_t) vImageTokens->mNTokens());
        }

        clip_image_f32_batch vBatchF32;
        vBatchF32.vIsAudio = false;
        vBatchF32.lEntries = std::move(vPreprocOut.lEntries);
        // do NOT use preproc_out from this point on, it's moved

        vImageTokens->vBatchF32 = std::move(vBatchF32);
        vImageTokens->vId       = bitmaps[0]->vId;  // optional

        LOG_DBG("image_tokens->nx = %d\n", vImageTokens->vNx);
        LOG_DBG("image_tokens->ny = %d\n", vImageTokens->vNy);
        LOG_DBG("batch_f32 size = %d\n", (int) vImageTokens->vBatchF32.lEntries.size());

        mtmd_input_chunk vChunk{
          cMtmdInputChunkTypeImage,
          {},       // text tokens
          std::move(vImageTokens),
          nullptr,  // audio tokens
        };
        vCur.lEntries.emplace_back(std::move(vChunk));
      }

      if (!vCtx->vImgEnd.empty()) {
        mAddText(vCtx->vImgEnd, true);  // add image end token
      }

      // advance image-chunk counter so the next image gets the next XD-RoPE dim-3 slot
      vNImagesAdded++;

    } else {
      // handle audio

      GGML_ASSERT(bitmaps.size() == 1);  // no batching support for now
      auto & bitmap = bitmaps[0];

      if (!vCtx->vCtxA) {
        LOG_ERR("%s: error: model does not support audio input\n", __func__);
        return 2;
      }

      if (bitmap->vNx == 0) {
        LOG_ERR("%s: error: empty audio data\n", __func__);
        return 2;
      }

      if (!vCtx->vAudBeg.empty()) {
        mAddText(vCtx->vAudBeg, true);  // add audio begin token
      }

      // sanity check
      GGML_ASSERT(vCtx->vAudioPreproc != nullptr);

      // preprocess audio
      std::vector<mtmd_audio_mel> lMelSpecChunks;
      {
        std::vector<float> lDummy;
        const float *      vSamples  = nullptr;
        size_t             vNSamples = 0;
        if (bitmap->mIsPlaceholder()) {
          // TODO @ngxson : skip underlay processing if bitmap is placeholder
          GGML_ASSERT(bitmap->vNy == 1);

          lDummy.resize(bitmap->vNx);
          vSamples  = lDummy.data();
          vNSamples = lDummy.size();
        } else {
          const auto & lBuf = bitmap->mGetRoBuf();
          GGML_ASSERT(lBuf.size() > sizeof(float));
          GGML_ASSERT(lBuf.size() % sizeof(float) == 0);

          vSamples  = (const float *) lBuf.data();
          vNSamples = lBuf.size() / sizeof(float);
        }
        bool vOk = vCtx->vAudioPreproc->mPreprocess(vSamples, vNSamples, lMelSpecChunks);
        if (!vOk) {
          LOG_ERR("Unable to preprocess audio\n");
          return 2;
        }
      }

      // consider each mel_spec as a separate audio chunk
      // TODO: maybe support batching, but this may come with memory cost
      for (auto & vMelSpec : lMelSpecChunks) {
        const bool cIsPlaceholder = vMelSpec.data.empty();

        // Validate dimensions fit in clip_image_size (int)
        GGML_ASSERT(vMelSpec.vNLen <= INT32_MAX && vMelSpec.vNLen >= 0);
        GGML_ASSERT(vMelSpec.vNMel <= INT32_MAX && vMelSpec.vNMel >= 0);
        clip_image_f32 vMelF32;
        vMelF32.mSetSize({ (int) vMelSpec.vNLen, (int) vMelSpec.vNMel }, cIsPlaceholder, /* is_audio */ true);
        vMelF32.mCpyBuf(vMelSpec.data);

        size_t vNTokens = fClipNOutputTokens(vCtx->vCtxA, &vMelF32);

        clip_image_f32_batch vBatchF32;
        vBatchF32.vIsAudio = true;
        vBatchF32.lEntries.push_back(std::move(vMelF32));

        mtmd_audio_tokens_ptr vAudioTokens(new mtmd_audio_tokens);
        vAudioTokens->vNTokens  = vNTokens;
        vAudioTokens->vBatchF32 = std::move(vBatchF32);
        vAudioTokens->vId       = bitmap->vId;  // optional

        LOG_DBG("audio_tokens->n_tokens = %d\n", vAudioTokens->vNTokens);

        mtmd_input_chunk vChunk{
          cMtmdInputChunkTypeAudio,
          {},       // text tokens
          nullptr,  // image tokens
          std::move(vAudioTokens),
        };
        vCur.lEntries.emplace_back(std::move(vChunk));
      }

      if (!vCtx->vAudEnd.empty()) {
        mAddText(vCtx->vAudEnd, true);  // add audio end token
      }
    }

    return 0;
  }

  // MiniCPM-V 4.7: the overview and all slices go in one chunk, clip appends the separators after each tile:
  //   [ov] </image><slice> [S00] </slice><slice> [S01] </slice>\n<slice> [S10] </slice><slice> [S11] </slice>
  void mAddCanvasChunk(mtmd_image_preproc_out && pPreprocOut, const std::string & pId) {
    const int cNCol   = pPreprocOut.vGridX;
    const int cNRow   = pPreprocOut.vGridY;
    auto &    lSlices = pPreprocOut.lEntries;
    GGML_ASSERT(pPreprocOut.mHasOverview());
    GGML_ASSERT((int) lSlices.size() == cNCol * cNRow);

    auto & vOv      = pPreprocOut.vOverview;
    vOv.vSuffixType = cClipSuffixMinicpmvOv;
    if (!lSlices.empty()) {
      vOv.vSuffixType    = cClipSuffixMinicpmvOvSlice;
      vOv.vAnyres.vGridX = cNCol;
      vOv.vAnyres.vGridY = cNRow;
    }
    for (int vY = 0; vY < cNRow; vY++) {
      for (int vX = 0; vX < cNCol; vX++) {
        auto & vSuffix = lSlices[vY * cNCol + vX].vSuffixType;
        if (vY == cNRow - 1 && vX == cNCol - 1) {
          vSuffix = cClipSuffixMinicpmvLast;
        } else if (vX == cNCol - 1) {
          vSuffix = cClipSuffixMinicpmvRowEnd;
        } else {
          vSuffix = cClipSuffixMinicpmvSlice;
        }
      }
    }

    mtmd_image_tokens_ptr vImageTokens(new mtmd_image_tokens);
    vImageTokens->vPos = cMtmdPosTypeCanvas;
    vImageTokens->vId  = pId;
    auto & lEntries    = vImageTokens->vBatchF32.lEntries;
    lEntries.push_back(std::move(vOv));
    for (auto & vSlice : lSlices) {
      lEntries.push_back(std::move(vSlice));
    }
    // token grid of the last entry, the grids of the other entries are scaled from it
    vImageTokens->vNx = fClipNOutputTokensX(vCtx->vCtxV, &lEntries.back());
    vImageTokens->vNy = fClipNOutputTokensY(vCtx->vCtxV, &lEntries.back());

    size_t vNTokens = 0;
    for (const auto & vEntry : lEntries) {
      vNTokens += fClipNOutputTokens(vCtx->vCtxV, &vEntry);
    }
    GGML_ASSERT(vNTokens == vImageTokens->mNTokens());

    mtmd_input_chunk vChunk{
      cMtmdInputChunkTypeImage,
      {},       // text tokens
      std::move(vImageTokens),
      nullptr,  // audio tokens
    };
    vCur.lEntries.emplace_back(std::move(vChunk));
  }

  std::vector<mtmd_input_chunk> mSplitBatchToChunk(mtmd_image_preproc_out && pPreprocOut, const std::string & pId) {
    std::vector<mtmd_input_chunk> lChunks;

    auto vProcessChunk = [&](clip_image_f32 && pImg) {
      mtmd_image_tokens_ptr vImageTokens(new mtmd_image_tokens);
      vImageTokens->vNx = fClipNOutputTokens(vCtx->vCtxV, &pImg);
      vImageTokens->vNy = 1;
      vImageTokens->vBatchF32.lEntries.push_back(std::move(pImg));
      vImageTokens->vId = pId;

      GGML_ASSERT(vImageTokens->vNx > 0);

      mtmd_input_chunk vChunk{
        cMtmdInputChunkTypeImage,
        {},       // text tokens
        std::move(vImageTokens),
        nullptr,  // audio tokens
      };
      lChunks.emplace_back(std::move(vChunk));
    };

    // overview image first
    auto & vOverview = pPreprocOut.vOverview;
    if (vOverview.mNx() == 0 || vOverview.mNy() == 0) {
      throw std::runtime_error(
        fStringFormat("%s: invalid overview image for llava-uhd style preprocessing\n", __func__));
    }
    vProcessChunk(std::move(pPreprocOut.vOverview));

    // then, process slices
    for (auto & vEntry : pPreprocOut.lEntries) {
      if (vEntry.mNx() == 0 || vEntry.mNy() == 0) {
        throw std::runtime_error(
          fStringFormat("%s: invalid image slice for llava-uhd style preprocessing\n", __func__));
      }
      vProcessChunk(std::move(vEntry));
    }

    return lChunks;
  }

  // for example: "a <__media__> b <__media__> c" --> "a", "<__media__>", "b", "<__media__>", "c"
  static std::vector<std::string> mSplitText(const std::string & pInput, const std::string & pDelimiter) {
    std::vector<std::string> lResult;
    if (pInput.empty()) {
      return lResult;
    }
    size_t vStart = 0;
    size_t vPos   = 0;
    while ((vPos = pInput.find(pDelimiter, vStart)) != std::string::npos) {
      if (vPos > vStart) {
        lResult.push_back(pInput.substr(vStart, vPos - vStart));
      }
      lResult.push_back(pDelimiter);
      vStart = vPos + pDelimiter.length();
    }
    if (vStart < pInput.length()) {
      lResult.push_back(pInput.substr(vStart));
    }
    return lResult;
  }

  // copied from common_tokenize
  static std::vector<dgs_token> mMtmdTokenizeTextInternal(const struct dgs_vocab * pVocab,
                                                          const std::string &      pText,
                                                          bool                     pAddSpecial,
                                                          bool                     pParseSpecial) {
    if (pVocab == nullptr) {
      throw std::runtime_error("dgs_vocab is not provided");
    }
    // upper limit for the number of tokens
    int                    vNTokens = pText.length() + 2 * pAddSpecial;
    std::vector<dgs_token> lResult(vNTokens);
    vNTokens =
      fDgsTokenize(pVocab, pText.data(), pText.length(), lResult.data(), lResult.size(), pAddSpecial, pParseSpecial);
    if (vNTokens == std::numeric_limits<int32_t>::min()) {
      throw std::runtime_error("Tokenization failed: input text too large, tokenization result exceeds int32_t limit");
    }
    if (vNTokens < 0) {
      lResult.resize(-vNTokens);
      int vCheck =
        fDgsTokenize(pVocab, pText.data(), pText.length(), lResult.data(), lResult.size(), pAddSpecial, pParseSpecial);
      GGML_ASSERT(vCheck == -vNTokens);
    } else {
      lResult.resize(vNTokens);
    }
    return lResult;
  }
};

int32_t fMtmdTokenize(const mtmd_context *        pCtx,
                      mtmd_input_chunks *         pOutput,
                      const mtmd_input_text *     pText,
                      const mtmd_bitmap * const * bitmaps,
                      size_t                      pNBitmaps) {
  try {
    mtmd_tokenizer vTokenizer(pCtx, pText, bitmaps, pNBitmaps);
    return vTokenizer.mTokenize(pOutput);
  } catch (const std::exception & vE) {
    LOG_ERR("%s: error: %s\n", __func__, vE.what());
    return 2;
  }
}

int32_t fMtmdTokenizeFromParts(const mtmd_context *            pCtx,
                               mtmd_input_chunks *             pOutput,
                               const mtmd_input_part * const * pParts,
                               size_t                          pNParts,
                               bool                            pAddSpecial) {
  for (size_t vI = 0; vI < pNParts; vI++) {
    if ((pParts[vI]->vText == nullptr) == (pParts[vI]->bitmap == nullptr)) {
      LOG_ERR("%s: part %zu must have either text or bitmap set, not both\n", __func__, vI);
      return 1;
    }
    if (pParts[vI]->vText != nullptr && pParts[vI]->vText->vText == nullptr) {
      LOG_ERR("%s: part %zu has null text pointer\n", __func__, vI);
      return 1;
    }
  }
  try {
    mtmd_tokenizer vTokenizer(pCtx, pParts, pNParts, pAddSpecial);
    return vTokenizer.mTokenize(pOutput);
  } catch (const std::exception & vE) {
    LOG_ERR("%s: error: %s\n", __func__, vE.what());
    return 2;
  }
}

static int32_t fMtmdEncodeImpl(mtmd_context *            pCtx,
                               const mtmd_image_tokens * pImageTokens,
                               std::vector<float> &      pOutEmbd) {
  clip_ctx * vCtxClip = pCtx->vCtxV;
  if (!vCtxClip) {
    LOG_ERR("%s: this API does not support non-vision input, please use mtmd_encode_chunk instead\n", __func__);
    return 1;
  }

  int  vNEmbdOut   = pCtx->mNEmbdOut();
  auto vNTokensOut = pImageTokens->mNTokens();
  pOutEmbd.resize((size_t) vNEmbdOut * vNTokensOut);

  if (pImageTokens->mIsPlaceholder()) {
    LOG_ERR("%s: image tokens batch is placeholder\n", __func__);
    return 1;
  }

  if (pImageTokens->vPos == cMtmdPosTypeCanvas) {
    // the tiles differ in size, encode them one by one
    size_t vOffset = 0;
    for (const auto & vEntry : pImageTokens->vBatchF32.lEntries) {
      clip_image_f32_batch vOne;
      vOne.lEntries.push_back(vEntry);
      std::vector<float> lEmbd((size_t) vNEmbdOut * fClipNOutputTokens(vCtxClip, &vEntry));
      if (!fClipImageBatchEncode(vCtxClip, pCtx->vNThreads, &vOne, lEmbd)) {
        return 1;
      }
      GGML_ASSERT(vOffset + lEmbd.size() <= pOutEmbd.size());
      std::copy(lEmbd.begin(), lEmbd.end(), pOutEmbd.begin() + vOffset);
      vOffset += lEmbd.size();
    }
    GGML_ASSERT(vOffset == pOutEmbd.size());
    return 0;
  }

  bool vOk = fClipImageBatchEncode(vCtxClip, pCtx->vNThreads, &pImageTokens->vBatchF32, pOutEmbd);

  return vOk ? 0 : 1;
}

static int32_t fMtmdEncodeChunkImpl(mtmd_context *           pCtx,
                                    const mtmd_input_chunk * pChunk,
                                    std::vector<float> &     pOutEmbd) {
  if (pChunk->type == cMtmdInputChunkTypeText) {
    LOG_WRN("mtmd_encode_chunk has no effect for text chunks\n");
    return 0;
  } else if (pChunk->type == cMtmdInputChunkTypeImage) {
    if (!pCtx->vCtxV) {
      LOG_ERR("%s: model does not support vision input\n", __func__);
      return 1;
    }
    if (pChunk->vTokensImage == nullptr) {
      LOG_ERR("%s: image tokens are null\n", __func__);
      return 1;
    }
    if (pChunk->vTokensImage->mIsPlaceholder()) {
      LOG_ERR("%s: image tokens batch is placeholder\n", __func__);
      return 1;
    }
    return fMtmdEncodeImpl(pCtx, pChunk->vTokensImage.get(), pOutEmbd);
  } else if (pChunk->type == cMtmdInputChunkTypeAudio) {
    if (!pCtx->vCtxA) {
      LOG_ERR("%s: model does not support audio input\n", __func__);
      return 1;
    }
    if (pChunk->vTokensAudio == nullptr) {
      LOG_ERR("%s: audio tokens are null\n", __func__);
      return 1;
    }
    if (pChunk->vTokensAudio->mIsPlaceholder()) {
      LOG_ERR("%s: audio tokens batch is placeholder\n", __func__);
      return 1;
    }
    int vNMmprojEmbd = pCtx->mNEmbdOut();
    pOutEmbd.resize((size_t) pChunk->vTokensAudio->vNTokens * vNMmprojEmbd);
    bool vOk = fClipImageBatchEncode(pCtx->vCtxA, pCtx->vNThreads, &pChunk->vTokensAudio->vBatchF32, pOutEmbd);
    return vOk ? 0 : 1;
  }

  LOG_ERR("%s: unknown chunk type %d\n", __func__, (int) pChunk->type);
  return 1;
}

int32_t fMtmdEncodeChunk(mtmd_context * pCtx, const mtmd_input_chunk * pChunk) {
  // this is the non-batching version
  try {
    return fMtmdEncodeChunkImpl(pCtx, pChunk, pCtx->lOutEmbd);
  } catch (const std::exception & vE) {
    LOG_ERR("%s: error: %s\n", __func__, vE.what());
    return 1;
  }
}

float * fMtmdGetOutputEmbd(mtmd_context * pCtx) {
  return pCtx->lOutEmbd.data();
}

//
// audio generation
//

mtmd_gen_audio_info fMtmdGenAudioGetInfo(const mtmd_context * pCtx) {
  mtmd_gen_audio_info info{};
  info.vModelVariant = "";
  if (!pCtx->vCtxGenA) {
    info.type = cMtmdGenAudioTypeNone;
    return info;
  }
  info.vModelVariant = fClipGetHparams(pCtx->vCtxGenA)->vGenModelVariant.c_str();
  switch (fClipGetProjectorType(pCtx->vCtxGenA)) {
    default:
      info.type = cMtmdGenAudioTypeNone;
      break;
  }
  return info;
}

mtmd_gen_inp fMtmdGenInpDefault(const mtmd_context * pCtx) {
  mtmd_gen_inp vInp{};
  vInp.type  = cMtmdGenProcessTypeGenCode;
  vInp.vSeed = UINT32_MAX;
  if (!pCtx->vCtxGenA) {
    return vInp;
  }

  switch (fClipGetProjectorType(pCtx->vCtxGenA)) {
    default:
      break;
  }
  return vInp;
}

static int32_t fMtmdGenAudioProcessImpl(mtmd_context * pCtx, const mtmd_gen_inp * pInp, mtmd_gen_out * pOut) {
  clip_ctx * vCtxClip = pCtx->vCtxGenA;
  if (!vCtxClip) {
    LOG_ERR("%s: model does not support audio generation\n", __func__);
    return 1;
  }

  *pOut = {};

  if (pInp->type == cMtmdGenProcessTypeGenCode) {
    const size_t cNEmbd = (size_t) fClipNMmprojEmbd(vCtxClip);

    clip_image_f32 vHiddenState;
    vHiddenState.mSetSize({ (int) cNEmbd, 1 }, false, true);
    vHiddenState.mCpyBuf(std::vector<float>(pInp->vEmbd, pInp->vEmbd + cNEmbd));

    clip_image_f32_batch vBatch;
    vBatch.vIsAudio = true;
    vBatch.lEntries.push_back(std::move(vHiddenState));

    std::vector<float>   lOutEmbd(cNEmbd);
    std::vector<int32_t> lOutCodes;
    std::vector<float>   lOutFeats;
    bool                 vIsEos = false;

    clip_encode_params vParams;
    vParams.vImgs       = &vBatch;
    vParams.vNThreads   = pCtx->vNThreads;
    vParams.vGenProcess = cClipGenProcessGenCode;
    vParams.lOutEmbd    = &lOutEmbd;
    vParams.lOutCodes   = &lOutCodes;
    vParams.lOutFeats   = &lOutFeats;
    vParams.vCode0      = pInp->vCode0;
    vParams.vTopK       = pInp->vTopK;
    vParams.vTopP       = pInp->vTopP;
    vParams.vSeed       = pInp->vSeed;
    vParams.vTemp       = pInp->vTemp;
    vParams.vOutIsEos   = &vIsEos;

    if (!fClipEncode(vCtxClip, &vParams)) {
      LOG_ERR("%s: clip_encode failed (gen_code)\n", __func__);
      return 1;
    }

    pCtx->lGenOutEmbd  = std::move(lOutEmbd);
    pCtx->lGenOutCodes = std::move(lOutCodes);
    pCtx->lGenOutFeats = std::move(lOutFeats);

    pOut->vEmbd   = pCtx->lGenOutEmbd.data();
    pOut->vCodes  = pCtx->lGenOutCodes.data();
    pOut->vNCodes = pCtx->lGenOutCodes.size();
    pOut->vFeats  = pCtx->lGenOutFeats.data();
    pOut->vNFeats = pCtx->lGenOutFeats.size();
    pOut->vIsEos  = vIsEos;
    return 0;
  }

  // MTMD_GEN_PROCESS_TYPE_GEN_WAV
  const bool cHasCodes = pInp->vCodes && pInp->vNCodes > 0;
  const bool cHasFeats = pInp->vFeats && pInp->vNFeats > 0;
  if (cHasCodes == cHasFeats) {
    LOG_ERR("%s: gen_wav requires exactly one of codes or feats\n", __func__);
    return 1;
  }
  std::vector<int32_t> lInCodes;
  std::vector<float>   lInFeats;
  if (cHasCodes) {
    lInCodes.assign(pInp->vCodes, pInp->vCodes + pInp->vNCodes);
  } else {
    lInFeats.assign(pInp->vFeats, pInp->vFeats + pInp->vNFeats);
  }
  std::vector<uint8_t> lInState;
  if (pInp->vStateData) {
    lInState.assign(pInp->vStateData, pInp->vStateData + pInp->vStateSize);
  }

  // gen_wav has no hidden-state input, the batch entry is an unused placeholder
  // TODO @ngxson : some models in the future may require hidden-state input, need to update this code later
  clip_image_f32 vDummy;
  vDummy.mSetSize({ 1, 1 }, false, true);
  vDummy.mCpyBuf(std::vector<float>(1, 0.0f));

  clip_image_f32_batch vBatch;
  vBatch.vIsAudio = true;
  vBatch.lEntries.push_back(std::move(vDummy));

  clip_encode_params vParams;
  vParams.vImgs       = &vBatch;
  vParams.vNThreads   = pCtx->vNThreads;
  vParams.vGenProcess = cClipGenProcessGenWav;
  // gen_wav draws no randomness, but keep the seed so it does not reseed mid-generation
  vParams.vSeed       = pInp->vSeed;
  vParams.lCodes      = cHasCodes ? &lInCodes : nullptr;
  vParams.lFeats      = cHasFeats ? &lInFeats : nullptr;
  vParams.lOutAudio   = &pCtx->lGenOutAudio;
  vParams.lStateIn    = pInp->vStateData ? &lInState : nullptr;
  vParams.lStateOut   = &pCtx->lGenOutState;

  if (!fClipEncode(vCtxClip, &vParams)) {
    LOG_ERR("%s: clip_encode failed (code2wav)\n", __func__);
    return 1;
  }

  pOut->vAudio     = pCtx->lGenOutAudio.data();
  pOut->vNSamples  = pCtx->lGenOutAudio.size();
  pOut->vStateData = (const char *) pCtx->lGenOutState.data();
  pOut->vStateSize = pCtx->lGenOutState.size();

  return 0;
}

int32_t fMtmdGenAudioProcess(mtmd_context * pCtx, const struct mtmd_gen_inp * pInp, struct mtmd_gen_out * pOut) {
  try {
    return fMtmdGenAudioProcessImpl(pCtx, pInp, pOut);
  } catch (const std::exception & vE) {
    LOG_ERR("%s: error: %s\n", __func__, vE.what());
    return 1;
  }
}

mtmd_batch * fMtmdBatchInit(mtmd_context * pCtx) {
  return new mtmd_batch(pCtx);
}

void fMtmdBatchFree(mtmd_batch * pBatch) {
  if (pBatch) {
    delete pBatch;
  }
}

int32_t fMtmdBatchAddChunk(mtmd_batch * pBatch, const mtmd_input_chunk * pChunk) {
  if (pChunk->type == cMtmdInputChunkTypeText) {
    LOG_ERR("%s: text chunk is not supported in batch\n", __func__);
    return 1;
  }

  auto * vCtx = pBatch->vCtx->mGetClipCtx(pChunk);
  if (!vCtx) {
    LOG_ERR("%s: model does not support input chunk type %d\n", __func__, (int) pChunk->type);
    return 1;
  }

  if (pBatch->lEntries.empty()) {
    // batch must have at least one chunk
    pBatch->lEntries.push_back(pChunk);
    return 0;
  }

  if (!fClipSupportBatch(vCtx)) {
    // if no batching support, batch can only have one single chunk
    return 2;  // "batch too large" error code
  }

  int32_t vNewNTokens = pBatch->mNTokens() + (int32_t) fMtmdInputChunkGetNTokens(pChunk);
  if (vNewNTokens > pBatch->vCtx->vBatchMaxTokens) {
    return 2;  // "batch too large" error code
  }

  auto & vFirstChunk = pBatch->lEntries[0];
  if (vFirstChunk->mCanBatchWith(*pChunk)) {
    pBatch->lEntries.push_back(pChunk);
    return 0;
  }

  return 3;  // "cannot batch" error code
}

static int32_t fMtmdBatchEncodeImpl(mtmd_batch * pBatch) {
  if (pBatch->lEntries.empty()) {
    LOG_ERR("%s: batch is empty\n", __func__);
    return 1;
  }
  for (const auto * vChunk : pBatch->lEntries) {
    if (vChunk->mIsPlaceholder()) {
      LOG_ERR("%s: chunk is placeholder\n", __func__);
      return 1;
    }
  }

  // represent the whole batch as one single chunk
  mtmd::input_chunk_ptr vBatchChunk(fMtmdInputChunkCopy(pBatch->lEntries[0]));
  if (vBatchChunk->vTokensImage) {
    auto & vB0F32 = vBatchChunk->vTokensImage->vBatchF32;
    // copy all entries from other chunks into the first chunk's batch_f32
    // note: skip first entry because it's already in batch_chunk
    for (size_t vIc = 1; vIc < pBatch->lEntries.size(); vIc++) {
      auto & vChunk = pBatch->lEntries[vIc];
      GGML_ASSERT(vChunk->vTokensImage);
      auto vB1F32 = vChunk->vTokensImage->vBatchF32.clone();
      for (size_t vI = 0; vI < vB1F32.lEntries.size(); vI++) {
        vB0F32.lEntries.push_back(std::move(vB1F32.lEntries[vI]));
      }
    }
  } else if (vBatchChunk->vTokensAudio) {
    auto & vB0F32 = vBatchChunk->vTokensAudio->vBatchF32;
    // copy all entries from other chunks into the first chunk's batch_f32
    // note: skip first entry because it's already in batch_chunk
    for (size_t vIc = 1; vIc < pBatch->lEntries.size(); vIc++) {
      auto & vChunk = pBatch->lEntries[vIc];
      GGML_ASSERT(vChunk->vTokensAudio);
      auto vB1F32 = vChunk->vTokensAudio->vBatchF32.clone();
      for (size_t vI = 0; vI < vB1F32.lEntries.size(); vI++) {
        vB0F32.lEntries.push_back(std::move(vB1F32.lEntries[vI]));
      }
    }
  } else {
    LOG_ERR("%s: unsupported chunk type\n", __func__);
    return 1;
  }

  LOG_DBG("%s: encoding batch with %zu entries and total %zu tokens\n", __func__, pBatch->lEntries.size(),
          fMtmdInputChunkGetNTokens(vBatchChunk.get()));
  int32_t vRes = fMtmdEncodeChunkImpl(pBatch->vCtx, vBatchChunk.get(), pBatch->lOutputEmbd);
  return vRes;
}

int32_t fMtmdBatchEncode(mtmd_batch * pBatch) {
  try {
    return fMtmdBatchEncodeImpl(pBatch);
  } catch (const std::exception & vE) {
    LOG_ERR("%s: error: %s\n", __func__, vE.what());
    return 1;
  }
}

float * fMtmdBatchGetOutputEmbd(mtmd_batch * pBatch, const mtmd_input_chunk * pChunk) {
  if (pBatch->lOutputEmbd.empty()) {
    LOG_ERR("%s: batch has not been encoded yet\n", __func__);
    return nullptr;
  }
  size_t       vOffset = 0;
  const size_t cNEmbd  = pBatch->vCtx->mNEmbdOut();
  for (const auto * vC : pBatch->lEntries) {
    size_t vOffsetPrev = vOffset;
    size_t vNTokens    = fMtmdInputChunkGetNTokens(vC);
    vOffset += vNTokens * cNEmbd;
    GGML_ASSERT(vOffsetPrev < pBatch->lOutputEmbd.size());
    GGML_ASSERT(vOffset <= pBatch->lOutputEmbd.size());
    if (vC == pChunk) {
      return &pBatch->lOutputEmbd.data()[vOffsetPrev];
    }
  }
  return nullptr;  // not found
}

bool fMtmdDecodeUseNonCausal(const mtmd_context * pCtx, const mtmd_input_chunk * pChunk) {
  auto vProjType = pCtx->mProjTypeV();
  if (pChunk && pChunk->type == cMtmdInputChunkTypeAudio) {
    vProjType = pCtx->mProjTypeA();
  }
  switch (vProjType) {
    default:
      return false;
  }
}

bool fMtmdDecodeUseMrope(const mtmd_context * pCtx) {
  return pCtx->vPosType == cMtmdPosTypeMrope;
}

bool fMtmdSupportVision(const mtmd_context * pCtx) {
  return pCtx->vCtxV != nullptr;
}

bool fMtmdSupportAudio(const mtmd_context * pCtx) {
  return pCtx->vCtxA != nullptr;
}

int fMtmdGetAudioSampleRate(const mtmd_context * pCtx) {
  if (!pCtx->vCtxA) {
    return -1;
  }
  return fClipGetHparams(pCtx->vCtxA)->vAudioSampleRate;
}

const char * fMtmdGetMarker(const mtmd_context * pCtx) {
  return pCtx->vMediaMarker.c_str();
}

//
// public API functions
//

// mtmd_bitmap

mtmd_bitmap * fMtmdBitmapInit(uint32_t pNx, uint32_t pNy, const unsigned char * data) {
  mtmd_bitmap * bitmap = new mtmd_bitmap(data, pNx, pNy);
  return bitmap;
}

mtmd_bitmap * fMtmdBitmapInitFromAudio(size_t pNSamples, const float * data) {
  mtmd_bitmap * bitmap = new mtmd_bitmap((const unsigned char *) data, pNSamples);
  GGML_ASSERT(bitmap->vIsAudio);
  if (!bitmap->mIsPlaceholder()) {
    GGML_ASSERT(bitmap->mGetRoBuf().size() == pNSamples * sizeof(float));
  }
  return bitmap;
}

uint32_t fMtmdBitmapGetNx(const mtmd_bitmap * bitmap) {
  return bitmap->vNx;
}

uint32_t fMtmdBitmapGetNy(const mtmd_bitmap * bitmap) {
  return bitmap->vNy;
}

const unsigned char * fMtmdBitmapGetData(const mtmd_bitmap * bitmap) {
  if (bitmap->mIsPlaceholder()) {
    return nullptr;
  }
  return bitmap->mGetRoBuf().data();
}

size_t fMtmdBitmapGetNBytes(const mtmd_bitmap * bitmap) {
  if (bitmap->mIsPlaceholder()) {
    return 0;
  }
  return bitmap->mGetRoBuf().size();
}

bool fMtmdBitmapIsAudio(const mtmd_bitmap * bitmap) {
  return bitmap->vIsAudio;
}

const char * fMtmdBitmapGetId(const mtmd_bitmap * bitmap) {
  return bitmap->vId.c_str();
}

void fMtmdBitmapSetId(mtmd_bitmap * bitmap, const char * pId) {
  if (pId) {
    bitmap->vId = std::string(pId);
  } else {
    bitmap->vId.clear();
  }
}

void fMtmdBitmapSetMergeable(mtmd_bitmap * bitmap, bool pMergeable) {
  bitmap->vMergeable = pMergeable;
}

mtmd_bitmap * fMtmdBitmapInitLazy(const mtmd_context *      pCtx,
                                  const char *              pId,
                                  void *                    pUserData,
                                  mtmd_bitmap_lazy_callback pCallback) {
  GGML_UNUSED(pCtx);  // reserved for future use
  mtmd_bitmap * bitmap  = new mtmd_bitmap(nullptr, 0, 0);
  bitmap->vLazyCallback = pCallback;
  bitmap->vLazyUserData = pUserData;
  fMtmdBitmapSetId(bitmap, pId);
  return bitmap;
}

void fMtmdBitmapFree(mtmd_bitmap * bitmap) {
  if (bitmap) {
    delete bitmap;
  }
}

// mtmd_input_chunks

mtmd_input_chunks * fMtmdInputChunksInit() {
  return new mtmd_input_chunks;
}

size_t fMtmdInputChunksSize(const mtmd_input_chunks * pChunks) {
  return pChunks->lEntries.size();
}

const mtmd_input_chunk * fMtmdInputChunksGet(const mtmd_input_chunks * pChunks, size_t pIdx) {
  if (pIdx >= pChunks->lEntries.size()) {
    return nullptr;
  }
  return &pChunks->lEntries[pIdx];
}

void fMtmdInputChunksFree(mtmd_input_chunks * pChunks) {
  if (pChunks) {
    delete pChunks;
  }
}

// mtmd_input_chunk

enum mtmd_input_chunk_type fMtmdInputChunkGetType(const mtmd_input_chunk * pChunk) {
  return pChunk->type;
}

const dgs_token * fMtmdInputChunkGetTokensText(const mtmd_input_chunk * pChunk, size_t * pNTokensOutput) {
  if (pChunk->type == cMtmdInputChunkTypeText) {
    *pNTokensOutput = pChunk->lTokensText.size();
    return pChunk->lTokensText.data();
  }
  *pNTokensOutput = 0;
  return nullptr;
}

const mtmd_image_tokens * fMtmdInputChunkGetTokensImage(const mtmd_input_chunk * pChunk) {
  if (pChunk->type == cMtmdInputChunkTypeImage) {
    return pChunk->vTokensImage.get();
  }
  return nullptr;
}

size_t fMtmdInputChunkGetNTokens(const mtmd_input_chunk * pChunk) {
  if (pChunk->type == cMtmdInputChunkTypeText) {
    return pChunk->lTokensText.size();
  } else if (pChunk->type == cMtmdInputChunkTypeImage) {
    return fMtmdImageTokensGetNTokens(pChunk->vTokensImage.get());
  } else if (pChunk->type == cMtmdInputChunkTypeAudio) {
    return pChunk->vTokensAudio->vNTokens;
  } else {
    GGML_ABORT("invalid chunk type");
  }
}

dgs_pos fMtmdInputChunkGetNPos(const mtmd_input_chunk * pChunk) {
  if (pChunk->type == cMtmdInputChunkTypeText) {
    return pChunk->lTokensText.size();
  } else if (pChunk->type == cMtmdInputChunkTypeImage) {
    return fMtmdImageTokensGetNPos(pChunk->vTokensImage.get());
  } else if (pChunk->type == cMtmdInputChunkTypeAudio) {
    return pChunk->vTokensAudio->vNTokens;
  } else {
    GGML_ABORT("invalid chunk type");
  }
}

const char * fMtmdInputChunkGetId(const mtmd_input_chunk * pChunk) {
  if (pChunk->type == cMtmdInputChunkTypeImage) {
    return pChunk->vTokensImage->vId.c_str();
  } else if (pChunk->type == cMtmdInputChunkTypeAudio) {
    return pChunk->vTokensAudio->vId.c_str();
  }
  return nullptr;
}

mtmd_input_chunk * fMtmdInputChunkCopy(const mtmd_input_chunk * pChunk) {
  mtmd_input_chunk * copy = new mtmd_input_chunk{
    pChunk->type,
    pChunk->lTokensText,
    nullptr,
    nullptr,
  };
  if (pChunk->vTokensImage) {
    // copy the image tokens
    copy->vTokensImage  = mtmd_image_tokens_ptr(new mtmd_image_tokens());
    *copy->vTokensImage = pChunk->vTokensImage->clone();
  }
  if (pChunk->vTokensAudio) {
    // copy the audio tokens
    copy->vTokensAudio  = mtmd_audio_tokens_ptr(new mtmd_audio_tokens());
    *copy->vTokensAudio = pChunk->vTokensAudio->clone();
  }
  return copy;
}

void fMtmdInputChunkFree(mtmd_input_chunk * pChunk) {
  if (pChunk) {
    delete pChunk;
  }
}

// returns 0 on success
static int32_t fMtmdInputChunkSaveImpl(const mtmd_input_chunk * pChunk, std::vector<char> & pOutBuf) {
  try {
    mtmd_serialization vSer(MTMD_SERIALIZATION_VERSION);
    pChunk->serialize(vSer);
    pOutBuf = std::move(vSer.data);
    return 0;
  } catch (const std::exception & vE) {
    LOG_ERR("%s: %s\n", __func__, vE.what());
    return -1;
  }
}

mtmd_input_chunk * fMtmdInputChunkGetPlaceholder(const mtmd_input_chunk * pChunk) {
  // this is hacky, but still faster than copy the whole batch data
  std::vector<char> lBuf;
  if (fMtmdInputChunkSaveImpl(pChunk, lBuf) != 0) {
    return nullptr;
  }
  return fMtmdInputChunkLoad(lBuf.data(), lBuf.size());
}

int32_t fMtmdInputChunkSave(const mtmd_input_chunk * pChunk, char * pOutBuf, size_t pOutLen, size_t * pExpectedOutLen) {
  std::vector<char> lBuf;
  if (fMtmdInputChunkSaveImpl(pChunk, lBuf) != 0) {
    return -1;
  }
  if (pExpectedOutLen) {
    *pExpectedOutLen = lBuf.size();
  }
  if (!pOutBuf) {
    // caller is only querying the required size
    return 0;
  }
  if (pOutLen < lBuf.size()) {
    LOG_ERR("%s: out_buf is too small, need %zu bytes, got %zu\n", __func__, lBuf.size(), pOutLen);
    return -1;
  }
  std::memcpy(pOutBuf, lBuf.data(), lBuf.size());
  return 0;
}

mtmd_input_chunk * fMtmdInputChunkLoad(const char * pBuf, size_t pLen) {
  try {
    mtmd_serialization    vSer(MTMD_SERIALIZATION_VERSION, pBuf, pLen);
    mtmd::input_chunk_ptr vChunk(new mtmd_input_chunk());
    vChunk->deserialize(vSer);
    return vChunk.release();
  } catch (const std::exception & vE) {
    LOG_ERR("%s: %s\n", __func__, vE.what());
    return nullptr;
  }
}

// mtmd_image_tokens

size_t fMtmdImageTokensGetNTokens(const mtmd_image_tokens * pImageTokens) {
  return pImageTokens->mNTokens();
}

// map a tile coordinate onto the canvas like the reference: round(linspace(0, canvas - 1, grid)), round() breaks ties to even
static uint32_t fMtmdCanvasScale(uint32_t pCoord, uint32_t pGrid, uint32_t pCanvas) {
  if (pGrid <= 1 || pCanvas <= 1) {
    return 0;
  }
  const double cV = (double) pCoord * (double) (pCanvas - 1) / (double) (pGrid - 1);
  return std::min((uint32_t) std::nearbyint(cV), pCanvas - 1);
}

// MTMD_POS_TYPE_CANVAS: every tile shares the <image> token before the chunk as origin
// the overview is stretched over the whole canvas, each slice fills its own cell; the time component is the origin, in slot z
// a tile takes one position in slot t (the KV cache position), the separators after it take one position each
static mtmd_decoder_pos fMtmdCanvasDecoderPos(const mtmd_image_tokens * pImageTokens, dgs_pos pPos0, size_t pI) {
  const auto &   lEntries = pImageTokens->vBatchF32.lEntries;
  const auto &   vGrid    = lEntries[0].vAnyres;
  const uint32_t cNx      = pImageTokens->vNx;
  const uint32_t cNy      = pImageTokens->vNy;
  const uint32_t cCanvasW = vGrid.mIsTiled() ? vGrid.vGridX * cNx : cNx;
  const uint32_t cCanvasH = vGrid.mIsTiled() ? vGrid.vGridY * cNy : cNy;
  const uint32_t cBase    = pPos0 - 1;

  mtmd_decoder_pos vPos;
  uint32_t         vT = pPos0;
  for (size_t vK = 0; vK < lEntries.size(); ++vK) {
    const auto [gw, gh] = pImageTokens->mCanvasTileGrid(vK);
    if (pI < gw * gh) {
      const uint32_t cRow = pI / gw;
      const uint32_t cCol = pI % gw;
      uint32_t       vH;
      uint32_t       vW;
      if (vK == 0) {
        vH = fMtmdCanvasScale(cRow, gh, cCanvasH);
        vW = fMtmdCanvasScale(cCol, gw, cCanvasW);
      } else {
        const uint32_t cS = vK - 1;
        vH                = (cS / vGrid.vGridX) * cNy + cRow;
        vW                = (cS % vGrid.vGridX) * cNx + cCol;
      }
      vPos.vT = vT;
      vPos.vX = cBase + vW;
      vPos.vY = cBase + vH;
      vPos.vZ = cBase;
      return vPos;
    }
    pI -= gw * gh;

    const size_t cNSep = fClipSuffixRows(lEntries[vK].vSuffixType).size();
    if (pI < cNSep) {
      const uint32_t cP = vT + 1 + pI;
      vPos.vT           = cP;
      vPos.vX           = cP;
      vPos.vY           = cP;
      vPos.vZ           = cP;
      return vPos;
    }
    pI -= cNSep;
    vT += 1 + cNSep;
  }
  GGML_ABORT("token index out of range");
}

mtmd_decoder_pos fMtmdImageTokensGetDecoderPos(const mtmd_image_tokens * pImageTokens, dgs_pos pPos0, size_t pI) {
  mtmd_decoder_pos vPos;
  switch (pImageTokens->vPos) {
    case cMtmdPosTypeMrope:
      {
        vPos.vT = pPos0;
        vPos.vX = pPos0 + (pI % pImageTokens->vNx);
        vPos.vY = pPos0 + (pI / pImageTokens->vNx);
        vPos.vZ = 0;  // unused for now
      }
      break;
    case cMtmdPosTypeNormal:
      {
        vPos.vT = pPos0 + pI;
        vPos.vX = pPos0 + pI;
        vPos.vY = pPos0 + pI;
        vPos.vZ = pPos0 + pI;
      }
      break;
    case cMtmdPosTypeHunyuanvl:
      {
        // HunyuanVL layout: [BOI] [row0 tokens + newline] ... [row(ny-1) tokens + newline] [EOI]
        // Total = 1 + ny*(nx+1) + 1. BOI and EOI use sequential positions in every dim;
        // content and row-newline tokens use (row, col) with XD-RoPE dim-3 = image_idx.
        const uint32_t cNx     = pImageTokens->vNx;
        const uint32_t cNTotal = pImageTokens->mNTokens();
        if (pI == 0) {
          // BOI
          vPos.vT = pPos0 + pI;
          vPos.vX = pPos0 + pI;
          vPos.vY = pPos0 + pI;
          vPos.vZ = pPos0 + pI;
        } else if (pI == cNTotal - 1) {
          // EOI
          vPos.vT = pPos0 + pI;
          vPos.vX = pPos0 + pI;
          vPos.vY = pPos0 + pI;
          vPos.vZ = pPos0 + pI;
        } else {
          // content token at (row, col), or the trailing newline of a row (col == nx)
          //   section 0 = sequential, section 1 = w(col), section 2 = h(row), section 3 = image_count.
          // set_position_mrope_2d writes .y -> section 1 and .x -> section 2
          const uint32_t cOffset = (uint32_t) pI - 1;
          const uint32_t cRow    = cOffset / (cNx + 1);
          const uint32_t cCol    = cOffset % (cNx + 1);
          vPos.vT                = pPos0 + pI;
          vPos.vX                = cRow;
          vPos.vY                = cCol;
          vPos.vZ                = pImageTokens->vImageIdx;
        }
      }
      break;
    case cMtmdPosTypeCanvas:
      {
        vPos = fMtmdCanvasDecoderPos(pImageTokens, pPos0, pI);
      }
      break;
    default:
      GGML_ABORT("invalid position type");
  }
  return vPos;
}

const char * fMtmdImageTokensGetId(const mtmd_image_tokens * pImageTokens) {
  return pImageTokens->vId.c_str();
}

dgs_pos fMtmdImageTokensGetNPos(const mtmd_image_tokens * pImageTokens) {
  switch (pImageTokens->vPos) {
    case cMtmdPosTypeMrope:
      return std::max(pImageTokens->vNx, pImageTokens->vNy);
    case cMtmdPosTypeNormal:
      return pImageTokens->mNTokens();
    case cMtmdPosTypeHunyuanvl:
      // HunyuanVL: the sequential (dim-0) position advances by the full token count
      // (includes BOI/EOI and row newline tokens), not by max(nx, ny)
      return pImageTokens->mNTokens();
    case cMtmdPosTypeCanvas:
      {
        // one position per tile, plus one per separator
        dgs_pos vNPos = 0;
        for (const auto & vEntry : pImageTokens->vBatchF32.lEntries) {
          vNPos += 1 + (dgs_pos) fClipSuffixRows(vEntry.vSuffixType).size();
        }
        return vNPos;
      }
    default:
      GGML_ABORT("invalid position type");
  }
}

// test function

mtmd_input_chunks * fMtmdTestCreateInputChunks() {
  mtmd_input_chunks * vChunks = fMtmdInputChunksInit();
  if (!vChunks) {
    return nullptr;
  }

  // create a text chunk
  std::vector<dgs_token> lTokensText = { 1, 2, 3, 4, 5 };
  mtmd_input_chunk       vChunkText{
    cMtmdInputChunkTypeText,
    std::move(lTokensText),
    nullptr,  // image tokens
    nullptr,  // audio tokens
  };
  vChunks->lEntries.emplace_back(std::move(vChunkText));

  // create an image chunk
  mtmd_image_tokens_ptr vImageTokens(new mtmd_image_tokens);
  vImageTokens->vNx = 4;
  vImageTokens->vNy = 4;
  vImageTokens->vBatchF32.lEntries.resize(16);
  vImageTokens->vId = "image_1";
  mtmd_input_chunk vChunkImage{
    cMtmdInputChunkTypeImage,
    {},       // text tokens
    std::move(vImageTokens),
    nullptr,  // audio tokens
  };
  vChunks->lEntries.emplace_back(std::move(vChunkImage));

  return vChunks;
}

void fMtmdLogSet(ggml_log_callback pLogCallback, void * pUserData) {
  vGLoggerState.vLogCallback         = pLogCallback ? pLogCallback : fClipLogCallbackDefault;
  vGLoggerState.vLogCallbackUserData = pUserData;
}

struct mtmd_caps fMtmdGetCapFromFile(const char * pFname) {
  try {
    auto      vTmp = fClipGetCap(pFname);
    mtmd_caps vCap;
    vCap.vInpAudio  = vTmp.vHasAudio;
    vCap.vInpVision = vTmp.vHasVision;
    return vCap;
  } catch (const std::exception & vE) {
    LOG_ERR("%s: failed to get capabilities from file '%s': %s\n", __func__, pFname, vE.what());
    return mtmd_caps{ false, false };
  }
}

//
// Debugging API (NOT intended for public use)
//

static void fStubLogCallback(enum ggml_log_level, const char *, void *) {
  // do nothing
}

mtmd_memory_usage fMtmdGetMemoryUsage(const char * pMmprojFname, struct mtmd_context_params pCtxParams) {
  mtmd::context_ptr vCtx;
  auto              vSavedLogCallback = vGLoggerState.vLogCallback;
  auto              vSavedLogUserData = vGLoggerState.vLogCallbackUserData;

  pCtxParams.vProgressCallback = nullptr;

  try {
    fMtmdLogSet(fStubLogCallback, nullptr);             // suppress logging
    vCtx.reset(new mtmd_context(pMmprojFname, nullptr, pCtxParams, true));
    fMtmdLogSet(vSavedLogCallback, vSavedLogUserData);  // restore log callback
    std::map<ggml_backend_dev_t, size_t> dTotalMem;
    auto                                 vMerge = [&](const struct clip_ctx * pC) {
      for (auto & [dev, size] : fClipGetMemUsage(pC)) {
        dTotalMem[dev] += size;
      }
    };
    if (vCtx->vCtxV) {
      vMerge(vCtx->vCtxV);
    }
    if (vCtx->vCtxA) {
      vMerge(vCtx->vCtxA);
    }
    mtmd_memory_usage vRes;
    vRes.dBackendMemUsage = std::move(dTotalMem);
    vRes.vImageMaxTokens  = vCtx->vCtxV ? fClipGetImageMaxTokens(vCtx->vCtxV) : -1;
    vRes.vUseNonCausal    = vCtx->vCtxV ? fMtmdDecodeUseNonCausal(vCtx.get(), nullptr) : false;
    return vRes;
  } catch (const std::exception & vE) {
    fMtmdLogSet(vSavedLogCallback, vSavedLogUserData);  // restore log callback
    LOG_ERR("%s: error: %s\n", __func__, vE.what());
    return { {}, -1, false };
  }
}
