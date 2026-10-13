#include "../../src/dgs-ext.h"
#include "dgs.h"
#include "mtmd-helper-common.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef MTMD_INTERNAL_HEADER
#  error "mtmd-helper is a public library outside of mtmd. it must not include internal headers"
#endif

//
// Audio generation helpers
//

// --tts-lang codes -> language names used by the codec_language special tokens
static const std::unordered_map<std::string, std::string> dTtsLangCodes = {
  { "zh", "chinese"    },
  { "en", "english"    },
  { "de", "german"     },
  { "it", "italian"    },
  { "pt", "portuguese" },
  { "es", "spanish"    },
  { "ja", "japanese"   },
  { "ko", "korean"     },
  { "fr", "french"     },
  { "ru", "russian"    },
};

static std::string fTtsResolveLang(const std::string & pLang) {
  auto vIt = dTtsLangCodes.find(pLang);
  return vIt != dTtsLangCodes.end() ? vIt->second : pLang;
}

static dgs_token fFindSpecialToken(const dgs_vocab * pVocab, const std::string & pPiece) {
  const int32_t n = fDgsVocabNTokens(pVocab);
  for (dgs_token vT = 0; vT < n; vT++) {
    if (pPiece == fDgsVocabGetText(pVocab, vT)) {
      return vT;
    }
  }
  return DGS_TOKEN_NULL;
}

static bool fWriteWav16(std::vector<char> & pBuf, const std::vector<float> & pPcm, int32_t pRate) {
  // RIFF chunk sizes are 32-bit; refuse to emit a file with a truncated header
  if (pPcm.size() > ((size_t) UINT32_MAX - 36) / 2) {
    return false;
  }
  const uint32_t cDataSz = (uint32_t) (pPcm.size() * 2);
  const uint32_t cRiffSz = 36 + cDataSz;
  const uint32_t cFmtSz = 16, cByteRate = (uint32_t) pRate * 2;
  const uint16_t cFmt = 1, cCh = 1, cAlign = 2, cBits = 16;
  const uint32_t cRate32 = (uint32_t) pRate;
  auto           vPut    = [&](const void * pP, size_t n) {
    const char * vC = (const char *) pP;
    pBuf.insert(pBuf.end(), vC, vC + n);
  };
  vPut("RIFF", 4);
  vPut(&cRiffSz, 4);
  vPut("WAVE", 4);
  vPut("fmt ", 4);
  vPut(&cFmtSz, 4);
  vPut(&cFmt, 2);
  vPut(&cCh, 2);
  vPut(&cRate32, 4);
  vPut(&cByteRate, 4);
  vPut(&cAlign, 2);
  vPut(&cBits, 2);
  vPut("data", 4);
  vPut(&cDataSz, 4);
  for (float vV : pPcm) {
    int16_t vS = (int16_t) (std::max(-1.0f, std::min(1.0f, vV)) * 32767.0f);
    vPut(&vS, 2);
  }
  return true;
}

class mtmd_gen_audio_pipeline {
public:
  mtmd_gen_audio_pipeline(dgs_context * pLctx, mtmd_context * pMctx) :
    vLctx(pLctx),
    vMctx(pMctx),
    vModel(fDgsGetModel(pLctx)),
    vVocab(fDgsModelGetVocab(vModel)),
    vNEmbd(fDgsModelNEmbd(vModel)),
    info(fMtmdGenAudioGetInfo(pMctx)) {}

  virtual ~mtmd_gen_audio_pipeline() = default;

  virtual void    reset()                                                                                           = 0;
  virtual int32_t mSetInput(const mtmd_helper_gen_audio_inp * pInp)                                                 = 0;
  // decodes at most n_batch prompt tokens; returns remaining count (0 = done), <0 on error
  virtual int32_t mStepPrompt(int32_t pNBatch)                                                                      = 0;
  // sampled can be DGS_TOKEN_NULL for pipelines with no discrete backbone token,
  // those read what they need from h_state_in instead
  // set out_stop on end-of-speech, h_state_out must be null if no frame is generated
  virtual int32_t mStepGen(dgs_token pSampled, const float * pHStateIn, const float ** pHStateOut, bool * pOutStop) = 0;
  virtual int32_t mGetOutput(int32_t *     pOutSampleRate,
                             const char ** pOutData,
                             size_t *      pOutDataLen,
                             int64_t *     pOutNSamples)                                                            = 0;

protected:
  dgs_context *       vLctx;
  mtmd_context *      vMctx;
  const dgs_model *   vModel;
  const dgs_vocab *   vVocab;
  int                 vNEmbd;
  mtmd_gen_audio_info info;
};

// Qwen3-TTS: backbone samples codec_0, code_predictor gives the other 15 codebooks,
// then code2wav decodes them to PCM
class qwen3tts_gen_audio_pipeline : public mtmd_gen_audio_pipeline {
public:
  using mtmd_gen_audio_pipeline::mtmd_gen_audio_pipeline;

  void reset() override {
    vSeqId = 0;
    vPos   = 0;
    lCodesBuf.clear();
    lC2wState.clear();
    lAudioPcm.clear();
    lOverlay.clear();
    lHStateBuf.clear();
    lOutBuf.clear();
    lPromptEmbdBuf.clear();
    vPromptBatch.reset();
    vNPrompt   = 0;
    vPromptPos = 0;
  }

  int32_t mSetInput(const mtmd_helper_gen_audio_inp * pInp) override {
    reset();
    vSeqId = pInp->vSeqId;

    if (!mEnsureCache()) {
      return 1;
    }

    const std::string cLang  = fTtsResolveLang((pInp->vLang && pInp->vLang[0]) ? pInp->vLang : "english");
    const dgs_token   cCLang = fFindSpecialToken(vVocab, ("<|codec_language_" + cLang + "|>").c_str());
    if (cCLang == DGS_TOKEN_NULL) {
      LOG_ERR("mtmd_helper_gen_audio: unknown language '%s'\n", cLang.c_str());
      return 1;
    }

    std::vector<float> lSpeakerEmbd;
    if (pInp->vSpeakerRef) {
      if (!mEncodeSpeaker(pInp->vSpeakerRef, lSpeakerEmbd)) {
        return 1;
      }
    }

    const int cNE  = vNEmbd;
    auto      vRow = [&](dgs_token pT) {
      return std::vector<float>(lTokEmbd.begin() + (size_t) pT * cNE, lTokEmbd.begin() + (size_t) (pT + 1) * cNE);
    };
    auto vSumRow = [&](dgs_token pA, dgs_token pB) {
      std::vector<float> lVa = vRow(pA), lVb = vRow(pB);
      for (int vI = 0; vI < cNE; vI++) {
        lVa[(size_t) vI] += lVb[(size_t) vI];
      }
      return lVa;
    };
    auto vSumVec = [&](dgs_token pA, const std::vector<float> & pVb) {
      std::vector<float> lVa = vRow(pA);
      for (int vI = 0; vI < cNE; vI++) {
        lVa[(size_t) vI] += pVb[(size_t) vI];
      }
      return lVa;
    };

    // upstream chat wrap, then slices: [0:3] role, [3:-5] utterance body
    const std::string cFull =
      "<|im_start|>assistant\n" + std::string(pInp->vPrompt, pInp->vPromptLen) + "<|im_end|>\n<|im_start|>assistant\n";
    std::vector<dgs_token> lIds(cFull.size() + 16);
    int                    vNIds =
      fDgsTokenize(vVocab, cFull.c_str(), (int32_t) cFull.size(), lIds.data(), (int32_t) lIds.size(), false, true);
    if (vNIds < 8) {
      LOG_ERR("mtmd_helper_gen_audio: tokenization failed\n");
      return 1;
    }
    lIds.resize((size_t) vNIds);

    std::vector<std::vector<float>> lPrompt;
    for (int vI = 0; vI < 3; vI++) {
      lPrompt.push_back(vRow(lIds[(size_t) vI]));
    }
    lPrompt.push_back(vSumRow(vTtsPad, vCThink));
    lPrompt.push_back(vSumRow(vTtsPad, vCThinkB));
    lPrompt.push_back(vSumRow(vTtsPad, cCLang));
    lPrompt.push_back(vSumRow(vTtsPad, vCThinkE));
    if (!lSpeakerEmbd.empty()) {
      lPrompt.push_back(vSumVec(vTtsPad, lSpeakerEmbd));
    }
    lPrompt.push_back(vSumRow(vTtsBos, vCodecPad));
    for (int vI = 3; vI < vNIds - 5; vI++) {
      lPrompt.push_back(vSumRow(lIds[(size_t) vI], vCodecPad));
    }
    lPrompt.push_back(vSumRow(vTtsEos, vCodecPad));
    lPrompt.push_back(vSumRow(vTtsPad, vCodecBos));

    vNPrompt = (int) lPrompt.size();

    // the talker uses the qwen3vl interleaved mrope, all sections are equal for a text/codec stream
    vMrope = fDgsModelRopeType(vModel) == cDgsRopeTypeMrope || fDgsModelRopeType(vModel) == cDgsRopeTypeImrope;
    const int cNPosPerEmbd = vMrope ? 4 : 1;

    lPromptEmbdBuf.resize((size_t) vNPrompt * (size_t) cNE);
    for (int vI = 0; vI < vNPrompt; vI++) {
      memcpy(lPromptEmbdBuf.data() + (size_t) vI * cNE, lPrompt[(size_t) vI].data(), (size_t) cNE * sizeof(float));
    }

    vPromptBatch.reset(new decode_embd_batch(lPromptEmbdBuf.data(), vNPrompt, cNPosPerEmbd, cNE));
    if (vMrope) {
      vPromptBatch->mSetPositionMrope1d(0, vSeqId);
    } else {
      vPromptBatch->mSetPositionNormal(0, vSeqId);
    }
    vPromptPos = 0;

    vPos                    = 0;
    const mtmd_gen_inp cDef = fMtmdGenInpDefault(vMctx);
    vTopK                   = pInp->vTopK > 0 ? pInp->vTopK : cDef.vTopK;
    vTopP                   = pInp->vTopP > 0 ? pInp->vTopP : cDef.vTopP;
    vSeed                   = pInp->vSeed;
    vOutType                = pInp->vOutType;

    // the prompt above holds the whole text stream up to tts_eos, so every generated
    // frame adds tts_pad on top of the codes embedding
    lOverlay = vRow(vTtsPad);

    return 0;
  }

  int32_t mStepPrompt(int32_t pNBatch) override {
    GGML_ASSERT(pNBatch > 0);
    if (vPromptPos >= vNPrompt) {
      return 0;
    }
    const int32_t cNTokensBatch = std::min(pNBatch, vNPrompt - vPromptPos);

    const bool cIsLastBatch = (vPromptPos + cNTokensBatch) == vNPrompt;
    if (cIsLastBatch) {
      vPromptBatch->lLogits[vPromptPos + cNTokensBatch - 1] = 1;
    }

    if (fDgsProcess(vLctx, cDgsProcessTypeDecode, vPromptBatch->mRender(vLctx, vPromptPos, cNTokensBatch)) != 0) {
      LOG_ERR("mtmd_helper_gen_audio: prompt decode failed\n");
      return -1;
    }

    vPos += cNTokensBatch;
    vPromptPos += cNTokensBatch;

    if (vPromptPos >= vNPrompt) {
      // prompt fully processed, its embedding buffer is no longer needed
      vPromptBatch.reset();
      lPromptEmbdBuf.clear();
      return 0;
    }
    return vNPrompt - vPromptPos;
  }

  int32_t mStepGen(dgs_token pSampled, const float * pHStateIn, const float ** pHStateOut, bool * pOutStop) override {
    if (pSampled == DGS_TOKEN_NULL) {
      LOG_ERR("mtmd_helper_gen_audio: qwen3tts requires a token sampled from the backbone\n");
      return 1;
    }

    // backbone signals end-of-speech with a token, no frame for this step
    if (pSampled == vCodecEos || fDgsVocabIsEog(vVocab, pSampled)) {
      *pOutStop   = true;
      *pHStateOut = nullptr;
      return 0;
    }

    mtmd_gen_inp vInp = fMtmdGenInpDefault(vMctx);
    vInp.type         = cMtmdGenProcessTypeGenCode;
    vInp.vCode0       = pSampled - vCodec0;
    vInp.vEmbd        = const_cast<float *>(pHStateIn);
    vInp.vTopK        = vTopK;
    vInp.vTopP        = vTopP;
    vInp.vSeed        = vSeed;
    mtmd_gen_out vOut{};
    if (fMtmdGenAudioProcess(vMctx, &vInp, &vOut) != 0) {
      LOG_ERR("mtmd_helper_gen_audio: gen_code process failed\n");
      return 1;
    }

    lCodesBuf.insert(lCodesBuf.end(), vOut.vCodes, vOut.vCodes + vOut.vNCodes);
    if (vOut.vNCodes > 0 && lCodesBuf.size() / vOut.vNCodes >= vWindowFrames) {
      if (!mFlushGenWav()) {
        return 1;
      }
    }

    std::vector<float> lFb(vOut.vEmbd, vOut.vEmbd + vNEmbd);
    for (int vI = 0; vI < vNEmbd; vI++) {
      lFb[(size_t) vI] += lOverlay[(size_t) vI];
    }

    const int         cNPosPerEmbd = vMrope ? 4 : 1;
    decode_embd_batch vBatchEmbd(lFb.data(), 1, cNPosPerEmbd, vNEmbd);
    if (vMrope) {
      vBatchEmbd.mSetPositionMrope1d(vPos, vSeqId);
    } else {
      vBatchEmbd.mSetPositionNormal(vPos, vSeqId);
    }
    vBatchEmbd.lLogits[0] = 1;
    vPos++;

    if (fDgsProcess(vLctx, cDgsProcessTypeDecode, vBatchEmbd.mRender(vLctx, 0, 1)) != 0) {
      LOG_ERR("mtmd_helper_gen_audio: decode failed\n");
      return 1;
    }

    const float * vHe = fDgsGetEmbeddingsIth(vLctx, -1);
    lHStateBuf.assign(vHe, vHe + vNEmbd);
    *pHStateOut = lHStateBuf.data();

    return 0;
  }

  int32_t mGetOutput(int32_t *     pOutSampleRate,
                     const char ** pOutData,
                     size_t *      pOutDataLen,
                     int64_t *     pOutNSamples) override {
    if (!mFlushGenWav()) {
      return 1;
    }

    *pOutSampleRate = info.vSampleRate;
    if (pOutNSamples) {
      *pOutNSamples = (int64_t) lAudioPcm.size();
    }

    if (vOutType == cMtmdHelperGenAudioOuttypePcm) {
      *pOutData    = (const char *) lAudioPcm.data();
      *pOutDataLen = lAudioPcm.size() * sizeof(float);
      return 0;
    }

    lOutBuf.clear();
    if (!fWriteWav16(lOutBuf, lAudioPcm, info.vSampleRate)) {
      LOG_ERR("mtmd_helper_gen_audio: output too large for WAV\n");
      return 1;
    }
    *pOutData    = lOutBuf.data();
    *pOutDataLen = lOutBuf.size();
    return 0;
  }

private:
  bool mEnsureCache() {
    if (vSpecialsOk) {
      return true;
    }
    vCodec0   = fFindSpecialToken(vVocab, "<|codec_0|>");
    vCodecBos = fFindSpecialToken(vVocab, "<|codec_bos|>");
    vCodecEos = fFindSpecialToken(vVocab, "<|codec_eos_token|>");
    vCodecPad = fFindSpecialToken(vVocab, "<|codec_pad|>");
    vCThink   = fFindSpecialToken(vVocab, "<|codec_think|>");
    vCThinkB  = fFindSpecialToken(vVocab, "<|codec_think_bos|>");
    vCThinkE  = fFindSpecialToken(vVocab, "<|codec_think_eos|>");
    vTtsPad   = fFindSpecialToken(vVocab, "<tts_pad>");
    vTtsBos   = fFindSpecialToken(vVocab, "<tts_text_bos>");
    vTtsEos   = fFindSpecialToken(vVocab, "<tts_text_eod>");
    for (dgs_token vT :
         { vCodec0, vCodecBos, vCodecEos, vCodecPad, vCThink, vCThinkB, vCThinkE, vTtsPad, vTtsBos, vTtsEos }) {
      if (vT == DGS_TOKEN_NULL) {
        LOG_ERR("mtmd_helper_gen_audio: missing a required special token in vocab\n");
        return false;
      }
    }
    const uint32_t cNTokEmbd = fDgsModelGetTokEmbd(vModel, nullptr);
    if (cNTokEmbd == 0) {
      LOG_ERR("mtmd_helper_gen_audio: model has no token embeddings\n");
      return false;
    }
    lTokEmbd.resize(cNTokEmbd);
    if (fDgsModelGetTokEmbd(vModel, lTokEmbd.data()) != cNTokEmbd) {
      LOG_ERR("mtmd_helper_gen_audio: token embedding copy failed\n");
      return false;
    }
    vSpecialsOk = true;
    return true;
  }

  // runs the reference wav through the speaker encoder, returns one x-vector embedding row
  bool mEncodeSpeaker(mtmd_bitmap * bitmap, std::vector<float> & pOut) {
    if (!fMtmdSupportAudio(vMctx)) {
      LOG_ERR("mtmd_helper_gen_audio: mmproj has no speaker/audio encoder\n");
      return false;
    }
    const std::string   cMarker = fMtmdDefaultMarker();
    mtmd_input_text     vText{ cMarker.c_str(), cMarker.size(), false, true };
    mtmd_input_chunks * vChunks = fMtmdInputChunksInit();
    const mtmd_bitmap * vBptr   = bitmap;
    bool                vOk     = fMtmdTokenize(vMctx, vChunks, &vText, &vBptr, 1) == 0;
    if (vOk) {
      vOk = false;
      for (size_t vI = 0; vI < fMtmdInputChunksSize(vChunks); vI++) {
        const mtmd_input_chunk * vChunk = fMtmdInputChunksGet(vChunks, vI);
        if (fMtmdInputChunkGetType(vChunk) != cMtmdInputChunkTypeAudio) {
          continue;
        }
        if (fMtmdEncodeChunk(vMctx, vChunk) != 0) {
          LOG_ERR("mtmd_helper_gen_audio: speaker encode failed\n");
          break;
        }
        const float * vEmbd = fMtmdGetOutputEmbd(vMctx);
        const size_t  n     = (size_t) fDgsModelNEmbdInp(vModel) * fMtmdInputChunkGetNTokens(vChunk);
        pOut.assign(vEmbd, vEmbd + n);
        vOk = true;
        break;
      }
    }
    fMtmdInputChunksFree(vChunks);
    return vOk;
  }

  // one GEN_WAV process() call over the buffered codes, state is carried across batches
  bool mFlushGenWav() {
    if (lCodesBuf.empty()) {
      return true;
    }
    mtmd_gen_inp vInp = fMtmdGenInpDefault(vMctx);
    vInp.type         = cMtmdGenProcessTypeGenWav;
    vInp.vCodes       = lCodesBuf.data();
    vInp.vNCodes      = lCodesBuf.size();
    vInp.vSeed        = vSeed;  // same seed as gen_code, else clip reseeds mid-generation
    vInp.vStateData   = lC2wState.empty() ? nullptr : (const char *) lC2wState.data();
    vInp.vStateSize   = lC2wState.size();
    mtmd_gen_out vOut{};
    if (fMtmdGenAudioProcess(vMctx, &vInp, &vOut) != 0) {
      LOG_ERR("mtmd_helper_gen_audio: gen_wav process failed\n");
      return false;
    }
    lAudioPcm.insert(lAudioPcm.end(), vOut.vAudio, vOut.vAudio + vOut.vNSamples);
    lC2wState.assign(vOut.vStateData, vOut.vStateData + vOut.vStateSize);
    lCodesBuf.clear();
    return true;
  }

  // vocab specials fixed across the whole session, looked up once
  bool               vSpecialsOk = false;
  dgs_token          vCodec0     = DGS_TOKEN_NULL;
  dgs_token          vCodecBos   = DGS_TOKEN_NULL;
  dgs_token          vCodecEos   = DGS_TOKEN_NULL;
  dgs_token          vCodecPad   = DGS_TOKEN_NULL;
  dgs_token          vCThink     = DGS_TOKEN_NULL;
  dgs_token          vCThinkB    = DGS_TOKEN_NULL;
  dgs_token          vCThinkE    = DGS_TOKEN_NULL;
  dgs_token          vTtsPad     = DGS_TOKEN_NULL;
  dgs_token          vTtsBos     = DGS_TOKEN_NULL;
  dgs_token          vTtsEos     = DGS_TOKEN_NULL;
  std::vector<float> lTokEmbd;  // whole token embedding matrix, n_vocab * n_embd

  // must match hparams.wav_tfm_swa hardcoded in clip.cpp
  size_t vWindowFrames = 72;

  // per-generation state, cleared by reset()
  dgs_seq_id                         vSeqId = 0;
  bool                               vMrope = false;
  int                                vPos   = 0;
  // prompt decode state, consumed batch-by-batch by step_prompt()
  std::vector<float>                 lPromptEmbdBuf;
  std::unique_ptr<decode_embd_batch> vPromptBatch;
  int                                vNPrompt   = 0;
  int                                vPromptPos = 0;
  int32_t                            vTopK      = 50;
  float                              vTopP      = 1.0f;
  uint32_t                           vSeed      = UINT32_MAX;
  std::vector<int32_t>               lCodesBuf;
  std::vector<uint8_t>               lC2wState;
  std::vector<float>                 lAudioPcm;
  std::vector<float>                 lOverlay;
  std::vector<float>                 lHStateBuf;
  mtmd_helper_gen_audio_outtype      vOutType = cMtmdHelperGenAudioOuttypeWav;
  std::vector<char>                  lOutBuf;
};

// settings that only live in the reference's per-pack yaml, not in the checkpoint
// the english packs share the same shapes and tokenizer, but disagree on these
// all three are 0 / false when the pack does not tune them, the model default is then used
struct pockettts_pack_settings {
  float vTemp           = 0.0f;
  int   vFramesAfterEos = 0;
  bool  vPadShortText   = false;
};

static pockettts_pack_settings fPocketttsPack(const char * pVariant) {
  static const std::unordered_map<std::string, pockettts_pack_settings> dPacks = {
    { "english",         { 0.3f, 0, false } },
    { "english_2026-01", { 0.7f, 0, true }  },
    { "english_2026-04", { 0.3f, 0, false } },
    { "french_24l",      { 0.7f, 8, false } },
  };
  auto vIt = dPacks.find(pVariant ? pVariant : "");
  if (vIt == dPacks.end()) {
    LOG_WRN("mtmd_helper_gen_audio: no tuned settings for pocket-tts variant \"%s\"\n", pVariant ? pVariant : "");
    return {};
  }
  return vIt->second;
}

// pocket-tts: the backbone emits no token, the flow net turns each hidden state into a latent
// the end-of-speech head also lives in the mmproj
class pockettts_gen_audio_pipeline : public mtmd_gen_audio_pipeline {
public:
  using mtmd_gen_audio_pipeline::mtmd_gen_audio_pipeline;

  void reset() override {
    vSeqId = 0;
    vPos   = 0;
    lFeatsBuf.clear();
    lDecState.clear();
    lAudioPcm.clear();
    lHStateBuf.clear();
    lOutBuf.clear();
    lPromptEmbdBuf.clear();
    vPromptBatch.reset();
    vNPrompt   = 0;
    vPromptPos = 0;
    vStepIdx   = 0;
    vEosStep   = -1;
    lChunks.clear();
    vChunkIdx    = 0;
    vNVoicePos   = 0;
    vChunkBudget = 0;
  }

  int32_t mSetInput(const mtmd_helper_gen_audio_inp * pInp) override {
    reset();
    vSeqId = pInp->vSeqId;

    if (!mEnsureCache()) {
      return 1;
    }

    std::vector<float> lVoice;
    if (pInp->vSpeakerRef) {
      if (!mEncodeSpeaker(pInp->vSpeakerRef, lVoice)) {
        return 1;
      }
    }

    vPack = fPocketttsPack(info.vModelVariant);

    const std::string cText = mPrepareText(std::string(pInp->vPrompt, pInp->vPromptLen), vPack.vPadShortText);
    if (cText.empty()) {
      LOG_ERR("mtmd_helper_gen_audio: empty prompt\n");
      return 1;
    }

    std::vector<dgs_token> lIds(cText.size() + 16);
    int                    vNIds =
      fDgsTokenize(vVocab, cText.c_str(), (int32_t) cText.size(), lIds.data(), (int32_t) lIds.size(), false, false);
    if (vNIds <= 0) {
      LOG_ERR("mtmd_helper_gen_audio: tokenization failed\n");
      return 1;
    }
    lIds.resize((size_t) vNIds);

    // long inputs degrade badly, so each chunk restarts from the voice conditioning
    // see split_into_best_sentences() in the reference
    lChunks   = mSplitChunks(lIds);
    vChunkIdx = 0;
    if (lChunks.size() > 1) {
      LOG_INF("mtmd_helper_gen_audio: %d tokens split into %zu chunks\n", vNIds, lChunks.size());
    }

    const int cNE = vNEmbd;

    // sequence order is voice, then text, then the audio BOS that starts generation
    if (!lVoice.empty()) {
      GGML_ASSERT(lVoice.size() % (size_t) cNE == 0);
      if (vBosBeforeVoice != DGS_TOKEN_NULL) {
        mPushEmbdRow(lPromptEmbdBuf, vBosBeforeVoice);
      }
      lPromptEmbdBuf.insert(lPromptEmbdBuf.end(), lVoice.begin(), lVoice.end());
    }
    // every later chunk rewinds to here and re-prompts, so the voice stays primed
    vNVoicePos = (int) (lPromptEmbdBuf.size() / (size_t) cNE);

    for (dgs_token vT : lChunks[0]) {
      mPushEmbdRow(lPromptEmbdBuf, vT);
    }
    mPushEmbdRow(lPromptEmbdBuf, vAudioBos);
    mArmChunkBudget(0);

    vNPrompt = (int) (lPromptEmbdBuf.size() / (size_t) cNE);
    vPromptBatch.reset(new decode_embd_batch(lPromptEmbdBuf.data(), vNPrompt, 1, cNE));
    vPromptBatch->mSetPositionNormal(0, vSeqId);
    vPromptPos = 0;

    vSeed    = pInp->vSeed;
    vOutType = pInp->vOutType;

    return 0;
  }

  int32_t mStepPrompt(int32_t pNBatch) override {
    GGML_ASSERT(pNBatch > 0);
    if (vPromptPos >= vNPrompt) {
      return 0;
    }
    const int32_t cNTokensBatch = std::min(pNBatch, vNPrompt - vPromptPos);

    if ((vPromptPos + cNTokensBatch) == vNPrompt) {
      vPromptBatch->lLogits[vPromptPos + cNTokensBatch - 1] = 1;
    }

    if (fDgsProcess(vLctx, cDgsProcessTypeDecode, vPromptBatch->mRender(vLctx, vPromptPos, cNTokensBatch)) != 0) {
      LOG_ERR("mtmd_helper_gen_audio: prompt decode failed\n");
      return -1;
    }

    vPos += cNTokensBatch;
    vPromptPos += cNTokensBatch;

    if (vPromptPos >= vNPrompt) {
      vPromptBatch.reset();
      lPromptEmbdBuf.clear();
      return 0;
    }
    return vNPrompt - vPromptPos;
  }

  int32_t mStepGen(dgs_token pSampled, const float * pHStateIn, const float ** pHStateOut, bool * pOutStop) override {
    (void) pSampled;  // the backbone output is continuous, there is no token to consume

    mtmd_gen_inp vInp = fMtmdGenInpDefault(vMctx);
    vInp.type         = cMtmdGenProcessTypeGenCode;
    vInp.vEmbd        = const_cast<float *>(pHStateIn);
    // clip only reseeds when the seed changes, so pass the same one on every step
    vInp.vSeed        = vSeed;
    if (vPack.vTemp > 0.0f) {
      vInp.vTemp = vPack.vTemp;
    }
    mtmd_gen_out vOut{};
    if (fMtmdGenAudioProcess(vMctx, &vInp, &vOut) != 0) {
      LOG_ERR("mtmd_helper_gen_audio: flow decode failed\n");
      return 1;
    }
    if (vOut.vIsEos && vEosStep < 0) {
      vEosStep = vStepIdx;
    }
    // the frame of the stopping step is discarded, matching _autoregressive_generation().
    // the budget is the reference's fallback for a chunk whose eos head never fires
    const bool cChunkDone = (vEosStep >= 0 && vStepIdx >= vEosStep + vFramesAfterEos) || vStepIdx >= vChunkBudget;
    if (cChunkDone) {
      if (vEosStep < 0) {
        LOG_WRN("mtmd_helper_gen_audio: chunk %zu hit its budget without end-of-speech\n", vChunkIdx);
      }
      return mFinishChunk(pHStateOut, pOutStop);
    }

    lFeatsBuf.insert(lFeatsBuf.end(), vOut.vFeats, vOut.vFeats + vOut.vNFeats);
    vStepIdx++;
    if (vOut.vNFeats > 0 && lFeatsBuf.size() / vOut.vNFeats >= vWindowFrames) {
      if (!mFlushGenWav()) {
        return 1;
      }
    }

    decode_embd_batch vBatchEmbd(vOut.vEmbd, 1, 1, vNEmbd);
    vBatchEmbd.mSetPositionNormal(vPos, vSeqId);
    vBatchEmbd.lLogits[0] = 1;
    vPos++;

    if (fDgsProcess(vLctx, cDgsProcessTypeDecode, vBatchEmbd.mRender(vLctx, 0, 1)) != 0) {
      LOG_ERR("mtmd_helper_gen_audio: decode failed\n");
      return 1;
    }

    const float * vHe = fDgsGetEmbeddingsIth(vLctx, -1);
    lHStateBuf.assign(vHe, vHe + vNEmbd);
    *pHStateOut = lHStateBuf.data();

    return 0;
  }

  int32_t mGetOutput(int32_t *     pOutSampleRate,
                     const char ** pOutData,
                     size_t *      pOutDataLen,
                     int64_t *     pOutNSamples) override {
    if (!mFlushGenWav()) {
      return 1;
    }

    *pOutSampleRate = info.vSampleRate;
    if (pOutNSamples) {
      *pOutNSamples = (int64_t) lAudioPcm.size();
    }

    if (vOutType == cMtmdHelperGenAudioOuttypePcm) {
      *pOutData    = (const char *) lAudioPcm.data();
      *pOutDataLen = lAudioPcm.size() * sizeof(float);
      return 0;
    }

    lOutBuf.clear();
    if (!fWriteWav16(lOutBuf, lAudioPcm, info.vSampleRate)) {
      LOG_ERR("mtmd_helper_gen_audio: output too large for WAV\n");
      return 1;
    }
    *pOutData    = lOutBuf.data();
    *pOutDataLen = lOutBuf.size();
    return 0;
  }

private:
  bool mEnsureCache() {
    if (vSpecialsOk) {
      return true;
    }
    // bos_before_voice is optional, some packs do not insert it
    vBosBeforeVoice = fFindSpecialToken(vVocab, "<|bos_before_voice|>");
    vAudioBos       = fFindSpecialToken(vVocab, "<|audio_bos|>");
    if (vAudioBos == DGS_TOKEN_NULL) {
      LOG_ERR("mtmd_helper_gen_audio: missing <|audio_bos|> in vocab\n");
      return false;
    }
    const uint32_t cNTokEmbd = fDgsModelGetTokEmbd(vModel, nullptr);
    if (cNTokEmbd == 0) {
      LOG_ERR("mtmd_helper_gen_audio: model has no token embeddings\n");
      return false;
    }
    lTokEmbd.resize(cNTokEmbd);
    if (fDgsModelGetTokEmbd(vModel, lTokEmbd.data()) != cNTokEmbd) {
      LOG_ERR("mtmd_helper_gen_audio: token embedding copy failed\n");
      return false;
    }
    GGML_ASSERT(vNEmbd > 0 && cNTokEmbd % (uint32_t) vNEmbd == 0);
    vSpecialsOk = true;
    return true;
  }

  // the table can be shorter than the vocab, so bound the row lookup
  void mPushEmbdRow(std::vector<float> & pDst, dgs_token pT) const {
    const size_t cNRows = lTokEmbd.size() / (size_t) vNEmbd;
    GGML_ASSERT(pT >= 0 && (size_t) pT < cNRows);
    pDst.insert(pDst.end(), lTokEmbd.begin() + (size_t) pT * vNEmbd, lTokEmbd.begin() + (size_t) (pT + 1) * vNEmbd);
  }

  // token ids of the pieces the reference splits on, see split_into_best_sentences().
  // the leading token is dropped, it is the tokenizer's dummy prefix
  std::vector<dgs_token> mPunctIds(const char * pS) const {
    std::vector<dgs_token> lIds(16);
    const int n = fDgsTokenize(vVocab, pS, (int32_t) strlen(pS), lIds.data(), (int32_t) lIds.size(), false, false);
    if (n <= 1) {
      return {};
    }
    return std::vector<dgs_token>(lIds.begin() + 1, lIds.begin() + n);
  }

  // cut after runs of boundary tokens, so punctuation stays with the sentence it ends
  static std::vector<std::vector<dgs_token>> mSplitOn(const std::vector<dgs_token> & pIds,
                                                      const std::vector<dgs_token> & pBoundary) {
    std::vector<std::vector<dgs_token>> lOut;
    size_t                              vStart           = 0;
    bool                                vPrevWasBoundary = false;
    for (size_t vI = 0; vI < pIds.size(); vI++) {
      const bool cIsBoundary = std::find(pBoundary.begin(), pBoundary.end(), pIds[vI]) != pBoundary.end();
      if (!cIsBoundary && vPrevWasBoundary) {
        lOut.emplace_back(pIds.begin() + vStart, pIds.begin() + vI);
        vStart = vI;
      }
      vPrevWasBoundary = cIsBoundary;
    }
    lOut.emplace_back(pIds.begin() + vStart, pIds.end());
    return lOut;
  }

  std::vector<std::vector<dgs_token>> mSplitChunks(const std::vector<dgs_token> & pIds) const {
    if ((int) pIds.size() <= cMaxChunkTokens) {
      return { pIds };
    }
    const std::vector<dgs_token> lEosPunct = mPunctIds(".!...?");
    const std::vector<dgs_token> lMidPunct = mPunctIds(",;:");

    // oversized sentences are split again on weaker punctuation, else words get skipped
    std::vector<std::vector<dgs_token>> lSegments;
    for (auto & lSeg : mSplitOn(pIds, lEosPunct)) {
      if ((int) lSeg.size() <= cMaxChunkTokens) {
        lSegments.push_back(std::move(lSeg));
        continue;
      }
      auto lSub = mSplitOn(lSeg, lMidPunct);
      if (lSub.size() > 1) {
        for (auto & lS : lSub) {
          lSegments.push_back(std::move(lS));
        }
      } else {
        lSegments.push_back(std::move(lSeg));
      }
    }

    std::vector<std::vector<dgs_token>> lOut;
    for (auto & lSeg : lSegments) {
      if (lSeg.empty()) {
        continue;
      }
      if (!lOut.empty() && (int) (lOut.back().size() + lSeg.size()) <= cMaxChunkTokens) {
        lOut.back().insert(lOut.back().end(), lSeg.begin(), lSeg.end());
      } else {
        lOut.push_back(std::move(lSeg));
      }
    }
    if (lOut.empty()) {
      lOut.push_back(pIds);
    }
    for (const auto & lC : lOut) {
      if ((int) lC.size() > cMaxChunkTokens) {
        LOG_WRN(
          "mtmd_helper_gen_audio: chunk of %zu tokens exceeds the %d token budget, "
          "generation may skip words\n",
          lC.size(), cMaxChunkTokens);
      }
    }
    return lOut;
  }

  // _estimate_max_gen_len() plus the per-chunk tail guess, both in frames
  void mArmChunkBudget(size_t pIdx) {
    const int cNTok = (int) lChunks[pIdx].size();
    vChunkBudget    = (int) std::ceil((cNTok / 3.0 + 2.0) * cFrameRate);
    // the pack may pin the tail, else the reference guesses it from the word count
    vFramesAfterEos = vPack.vFramesAfterEos > 0 ? vPack.vFramesAfterEos : (cNTok <= 6 ? 5 : 3);
    vStepIdx        = 0;
    vEosStep        = -1;
  }

  // ends the current chunk and, if there is another, re-prompts it on top of the voice
  int32_t mFinishChunk(const float ** pHStateOut, bool * pOutStop) {
    if (!mFlushGenWav()) {
      return 1;
    }
    // the decoder restarts too, the next chunk's audio is not continuous with this one
    lDecState.clear();

    if (vChunkIdx + 1 >= lChunks.size()) {
      *pOutStop   = true;
      *pHStateOut = nullptr;
      return 0;
    }
    vChunkIdx++;

    // drop this chunk's text and audio, keep the voice conditioning
    fDgsMemorySeqRm(fDgsGetMemory(vLctx), vSeqId, vNVoicePos, -1);
    vPos = vNVoicePos;

    const int cNE = vNEmbd;
    lPromptEmbdBuf.clear();
    for (dgs_token vT : lChunks[vChunkIdx]) {
      mPushEmbdRow(lPromptEmbdBuf, vT);
    }
    mPushEmbdRow(lPromptEmbdBuf, vAudioBos);
    mArmChunkBudget(vChunkIdx);

    const int cNRows = (int) (lPromptEmbdBuf.size() / (size_t) cNE);
    GGML_ASSERT(cNRows > 0);
    decode_embd_batch vBatch(lPromptEmbdBuf.data(), cNRows, 1, cNE);
    vBatch.mSetPositionNormal(vPos, vSeqId);
    vBatch.lLogits[cNRows - 1] = 1;
    if (fDgsProcess(vLctx, cDgsProcessTypeDecode, vBatch.mRender(vLctx, 0, cNRows)) != 0) {
      LOG_ERR("mtmd_helper_gen_audio: chunk prompt decode failed\n");
      return 1;
    }
    vPos += cNRows;
    lPromptEmbdBuf.clear();

    const float * vHe = fDgsGetEmbeddingsIth(vLctx, -1);
    lHStateBuf.assign(vHe, vHe + vNEmbd);
    *pHStateOut = lHStateBuf.data();
    *pOutStop   = false;
    return 0;
  }

  // same normalization as prepare_text_prompt() in the reference, it affects quality
  static std::string mPrepareText(const std::string & pIn, bool pPadShort) {
    std::string vS;
    vS.reserve(pIn.size() + 1);
    for (char vC : pIn) {
      if (vC == '\n' || vC == '\r') {
        vS += ' ';
      } else if (vC == ';') {
        vS += ',';
      } else {
        vS += vC;
      }
    }
    const size_t cB = vS.find_first_not_of(' ');
    const size_t cE = vS.find_last_not_of(' ');
    if (cB == std::string::npos) {
      return "";
    }
    vS = vS.substr(cB, cE - cB + 1);
    if (vS[0] >= 'a' && vS[0] <= 'z') {
      vS[0] = (char) (vS[0] - 'a' + 'A');
    }
    const unsigned char cLast = (unsigned char) vS.back();
    if (std::isalnum(cLast)) {
      vS += '.';
    }
    if (pPadShort && mCountWords(vS) < 5) {
      vS = std::string(8, ' ') + vS;
    }
    return vS;
  }

  static int mCountWords(const std::string & pS) {
    int  n       = 0;
    bool vInWord = false;
    for (char vC : pS) {
      if (vC == ' ') {
        vInWord = false;
      } else if (!vInWord) {
        vInWord = true;
        n++;
      }
    }
    return n;
  }

  // runs the reference wav through the mimi encoder, returns one row per 12.5Hz frame
  bool mEncodeSpeaker(mtmd_bitmap * bitmap, std::vector<float> & pOut) {
    if (!fMtmdSupportAudio(vMctx)) {
      LOG_ERR("mtmd_helper_gen_audio: mmproj has no voice encoder\n");
      return false;
    }
    const std::string   cMarker = fMtmdDefaultMarker();
    mtmd_input_text     vText{ cMarker.c_str(), cMarker.size(), false, true };
    mtmd_input_chunks * vChunks = fMtmdInputChunksInit();
    const mtmd_bitmap * vBptr   = bitmap;
    bool                vOk     = fMtmdTokenize(vMctx, vChunks, &vText, &vBptr, 1) == 0;
    if (vOk) {
      vOk = false;
      for (size_t vI = 0; vI < fMtmdInputChunksSize(vChunks); vI++) {
        const mtmd_input_chunk * vChunk = fMtmdInputChunksGet(vChunks, vI);
        if (fMtmdInputChunkGetType(vChunk) != cMtmdInputChunkTypeAudio) {
          continue;
        }
        if (fMtmdEncodeChunk(vMctx, vChunk) != 0) {
          LOG_ERR("mtmd_helper_gen_audio: voice encode failed\n");
          break;
        }
        const float * vEmbd = fMtmdGetOutputEmbd(vMctx);
        const size_t  n     = (size_t) fDgsModelNEmbdInp(vModel) * fMtmdInputChunkGetNTokens(vChunk);
        pOut.assign(vEmbd, vEmbd + n);
        vOk = true;
        break;
      }
    }
    fMtmdInputChunksFree(vChunks);
    return vOk;
  }

  // decodes the buffered latents, the mimi decoder state carries over between calls
  bool mFlushGenWav() {
    if (lFeatsBuf.empty()) {
      return true;
    }
    mtmd_gen_inp vInp = fMtmdGenInpDefault(vMctx);
    vInp.type         = cMtmdGenProcessTypeGenWav;
    vInp.vFeats       = lFeatsBuf.data();
    vInp.vNFeats      = lFeatsBuf.size();
    vInp.vSeed        = vSeed;
    vInp.vStateData   = lDecState.empty() ? nullptr : (const char *) lDecState.data();
    vInp.vStateSize   = lDecState.size();
    mtmd_gen_out vOut{};
    if (fMtmdGenAudioProcess(vMctx, &vInp, &vOut) != 0) {
      LOG_ERR("mtmd_helper_gen_audio: mimi decode failed\n");
      return false;
    }
    lAudioPcm.insert(lAudioPcm.end(), vOut.vAudio, vOut.vAudio + vOut.vNSamples);
    lDecState.assign(vOut.vStateData, vOut.vStateData + vOut.vStateSize);
    lFeatsBuf.clear();
    return true;
  }

  pockettts_pack_settings vPack;
  bool                    vSpecialsOk     = false;
  dgs_token               vBosBeforeVoice = DGS_TOKEN_NULL;
  dgs_token               vAudioBos       = DGS_TOKEN_NULL;
  std::vector<float>      lTokEmbd;

  dgs_seq_id                          vSeqId = 0;
  int                                 vPos   = 0;
  std::vector<float>                  lPromptEmbdBuf;
  std::unique_ptr<decode_embd_batch>  vPromptBatch;
  int                                 vNPrompt        = 0;
  int                                 vPromptPos      = 0;
  uint32_t                            vSeed           = UINT32_MAX;
  // end-of-speech is latched, then a few more frames are generated as tail padding
  int                                 vStepIdx        = 0;
  int                                 vEosStep        = -1;
  int                                 vFramesAfterEos = 3;
  static constexpr int                cMaxChunkTokens = 50;  // MAX_TOKEN_PER_CHUNK in the reference
  static constexpr double             cFrameRate      = 12.5;
  std::vector<std::vector<dgs_token>> lChunks;
  size_t                              vChunkIdx    = 0;
  int                                 vNVoicePos   = 0;  // KV positions held by the voice conditioning
  int                                 vChunkBudget = 0;

  // latents are decoded a window at a time, the decoder state bridges the windows
  size_t                        vWindowFrames = 8;
  std::vector<float>            lFeatsBuf;
  std::vector<uint8_t>          lDecState;
  std::vector<float>            lAudioPcm;
  std::vector<float>            lHStateBuf;
  mtmd_helper_gen_audio_outtype vOutType = cMtmdHelperGenAudioOuttypeWav;
  std::vector<char>             lOutBuf;
};

static std::unique_ptr<mtmd_gen_audio_pipeline> fMakePipeline(dgs_context * pLctx, mtmd_context * pMctx) {
  switch (fMtmdGenAudioGetInfo(pMctx).type) {
    case cMtmdGenAudioTypeQwen3tts:
      return std::unique_ptr<mtmd_gen_audio_pipeline>(new qwen3tts_gen_audio_pipeline(pLctx, pMctx));
    case cMtmdGenAudioTypePockettts:
      return std::unique_ptr<mtmd_gen_audio_pipeline>(new pockettts_gen_audio_pipeline(pLctx, pMctx));
    default:
      return nullptr;
  }
}

struct mtmd_helper_gen_audio {
  std::unique_ptr<mtmd_gen_audio_pipeline> vPipeline;
};

mtmd_helper_gen_audio * fMtmdHelperGenAudioInit(struct dgs_context * pLctx, struct mtmd_context * pMctx) {
  auto * vCtx     = new mtmd_helper_gen_audio();
  vCtx->vPipeline = fMakePipeline(pLctx, pMctx);
  return vCtx;
}

void fMtmdHelperGenAudioFree(mtmd_helper_gen_audio * pCtx) {
  delete pCtx;
}

void fMtmdHelperGenAudioReset(mtmd_helper_gen_audio * pCtx) {
  if (pCtx->vPipeline) {
    pCtx->vPipeline->reset();
  }
}

int32_t fMtmdHelperGenAudioSetInput(mtmd_helper_gen_audio * pCtx, const mtmd_helper_gen_audio_inp * pInp) {
  if (!pCtx->vPipeline) {
    LOG_ERR("mtmd_helper_gen_audio: unsupported or missing gen-audio pipeline\n");
    return 1;
  }
  return pCtx->vPipeline->mSetInput(pInp);
}

int32_t fMtmdHelperGenAudioStepPrompt(mtmd_helper_gen_audio * pCtx, int32_t pNBatch) {
  if (!pCtx->vPipeline) {
    return -1;
  }
  return pCtx->vPipeline->mStepPrompt(pNBatch);
}

int32_t fMtmdHelperGenAudioStepGen(mtmd_helper_gen_audio * pCtx,
                                   dgs_token               pSampled,
                                   const float *           pHStateIn,
                                   const float **          pHStateOut,
                                   bool *                  pOutStop) {
  if (!pCtx->vPipeline) {
    return 1;
  }
  bool          vStop = false;
  const int32_t cRet  = pCtx->vPipeline->mStepGen(pSampled, pHStateIn, pHStateOut, &vStop);
  if (pOutStop) {
    *pOutStop = vStop;
  }
  return cRet;
}

int32_t fMtmdHelperGenAudioGetOutput(mtmd_helper_gen_audio * pCtx,
                                     int32_t *               pOutSampleRate,
                                     const char **           pOutData,
                                     size_t *                pOutDataLen,
                                     int64_t *               pOutNSamples) {
  if (!pCtx->vPipeline) {
    return 1;
  }
  return pCtx->vPipeline->mGetOutput(pOutSampleRate, pOutData, pOutDataLen, pOutNSamples);
}
