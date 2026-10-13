#include "server-common.h"

#include "base64.hpp"
#include "chat.h"
#include "common.h"
#include "dgs.h"
#include "download.h"
#include "log.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <thread>
#include <type_traits>

#ifdef _WIN32
// windows.h defines min and max as macros, which breaks std::min and std::max
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <io.h>
#  include <windows.h>
#else
#  include <errno.h>
#  include <fcntl.h>
#  include <poll.h>
#  include <unistd.h>
#endif

json fFormatErrorResponse(const std::string & pMessage, const enum error_type type) {
  std::string vTypeStr;
  int         vCode = 500;
  switch (type) {
    case cErrorTypeInvalidRequest:
      vTypeStr = "invalid_request_error";
      vCode    = 400;
      break;
    case cErrorTypeAuthentication:
      vTypeStr = "authentication_error";
      vCode    = 401;
      break;
    case cErrorTypeNotFound:
      vTypeStr = "not_found_error";
      vCode    = 404;
      break;
    case cErrorTypeServer:
      vTypeStr = "server_error";
      vCode    = 500;
      break;
    case cErrorTypePermission:
      vTypeStr = "permission_error";
      vCode    = 403;
      break;
    case cErrorTypeNotSupported:
      vTypeStr = "not_supported_error";
      vCode    = 501;
      break;
    case cErrorTypeUnavailable:
      vTypeStr = "unavailable_error";
      vCode    = 503;
      break;
    case cErrorTypeExceedContextSize:
      vTypeStr = "exceed_context_size_error";
      vCode    = 400;
      break;
  }
  return json{
    { "code",    vCode    },
    { "message", pMessage },
    { "type",    vTypeStr },
  };
}

//
// server_slot_stats
//

json server_slot_stats::to_json() const {
  json vBase = {
    { "cache_n",                vNPromptCached       },

    { "prompt_n",               vNPromptProcessed    },
    { "prompt_ms",              mTPromptMs()         },
    { "prompt_per_token_ms",    mTPromptPerTokenMs() },
    { "prompt_per_second",      mNPromptTps()        },

    { "predicted_n",            vNGen                },
    { "predicted_ms",           mTGenMs()            },
    { "predicted_per_token_ms", mTGenPerTokenMs()    },
    { "predicted_per_second",   mNGenTps()           },
  };

  if (vNDraftTokens > 0) {
    vBase["draft_n"]          = vNDraftTokens;
    vBase["draft_n_accepted"] = vNDraftAccepted;
  }

  return vBase;
}

//
// random string / id
//

std::string fRandomString() {
  static const std::string str("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");

  std::random_device vRd;
  std::mt19937       vGenerator(vRd());

  std::string vResult(32, ' ');

  for (int vI = 0; vI < 32; ++vI) {
    vResult[vI] = str[vGenerator() % str.size()];
  }

  return vResult;
}

std::string fGenChatcmplid() {
  return "chatcmpl-" + fRandomString();
}

std::string fGenToolCallId() {
  return fRandomString();
}

const char * fGetMediaMarker() {
  static const std::string cMarker = []() {
    // allow user to pin a reproducible marker via env var
    const char * vEnv = getenv("DGS_MEDIA_MARKER");
    if (vEnv && vEnv[0] != '\0') {
      return std::string(vEnv);
    }
    return std::string("<__media_") + fRandomString() + "__>";
  }();
  return cMarker.c_str();
}

//
// model output modalities
//

std::vector<std::string> fServerModelOutputModalities(common_decision_type pDecisionType) {
  switch (pDecisionType) {
    case cCommonDecisionTypeOpenjev:
    case cCommonDecisionTypeLev:
    case cCommonDecisionTypeKev:
    case cCommonDecisionTypeNimble:
    case cCommonDecisionTypeLaya:
    case cCommonDecisionTypeClef:
    case cCommonDecisionTypePplxDecider:
    case cCommonDecisionTypeVllmDecision2:
    case cCommonDecisionTypeVllmD3:
    case cCommonDecisionTypeLfm2D1:
    case cCommonDecisionTypeLfm2D1Omni:
      return { "decisions" };
    default:
      // fallback when there is no decision type or the metadata is bad
      return { "text" };
  }
}

json fServerModelArchitectureJson(bool                             pInpImage,
                                  bool                             pInpAudio,
                                  bool                             pInpVideo,
                                  const std::vector<std::string> & pOutputModalities) {
  std::vector<std::string> lInputModalities = { "text" };
  if (pInpImage) {
    lInputModalities.push_back("image");
  }
  if (pInpAudio) {
    lInputModalities.push_back("audio");
  }
  if (pInpVideo) {
    lInputModalities.push_back("video");
  }

  return {
    { "input_modalities",  lInputModalities  },
    { "output_modalities", pOutputModalities },
  };
}

//
// lora utils
//

bool fLoraAllAlora(const std::vector<common_adapter_lora_info> & pLoras) {
  bool vFoundAlora = false;
  for (const auto & lora : pLoras) {
    if (lora.vScale != 0) {
      if (fDgsAdapterGetAloraNInvocationTokens(lora.vPtr) == 0) {
        return false;
      }
      vFoundAlora = true;
    }
  }
  return vFoundAlora;
}

bool fLoraShouldClearCache(const std::vector<common_adapter_lora_info> & pCurrent,
                           const std::vector<common_adapter_lora_info> & pNext) {
  // This should always be called after determining that the two sets are
  // _not_ equal. This assert is therefore some slightly wasted work and
  // should be safe to remove as long as this method is called correctly.
  GGML_ASSERT(!fAreLoraEqual(pCurrent, pNext));

  return (!(fLoraGetEnabledIds(pCurrent).empty() || fLoraAllAlora(pCurrent)) || !fLoraAllAlora(pNext));
}

std::map<int, float> fParseLoraRequest(const json & data) {
  std::map<int, float> lora;

  // set value
  for (const auto & vEntry : data) {
    int   vId    = fJsonValue(vEntry, "id", -1);
    float vScale = fJsonValue(vEntry, "scale", 0.0f);
    lora[vId]    = vScale;
  }

  return lora;
}

bool fAreLoraEqual(const std::vector<common_adapter_lora_info> & pL1,
                   const std::vector<common_adapter_lora_info> & pL2) {
  if (pL1.size() != pL2.size()) {
    return false;
  }
  for (size_t vI = 0; vI < pL1.size(); ++vI) {
    // we don't check lora.path to reduce the time complexity
    if (pL1[vI].vScale != pL2[vI].vScale || pL1[vI].vPtr != pL2[vI].vPtr) {
      return false;
    }
  }
  return true;
}

std::vector<size_t> fLoraGetEnabledIds(const std::vector<common_adapter_lora_info> & pLoras) {
  std::vector<size_t> lEnabledIds;
  for (size_t vI = 0; vI < pLoras.size(); ++vI) {
    if (pLoras[vI].vScale > 0) {
      lEnabledIds.push_back(vI);
    }
  }
  return lEnabledIds;
}

//
// base64 utils (TODO: use the base64::decode from base64.hpp)
//

static const std::string cBase64Chars =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  "abcdefghijklmnopqrstuvwxyz"
  "0123456789+/";

static inline bool fIsBase64(uint8_t pC) {
  return (isalnum(pC) || (pC == '+') || (pC == '/'));
}

static inline raw_buffer fBase64Decode(const std::string & pEncodedString) {
  int vI  = 0;
  int vJ  = 0;
  int vIn = 0;

  int vInLen = pEncodedString.size();

  uint8_t aCharArray4[4];
  uint8_t aCharArray3[3];

  raw_buffer lRet;

  while (vInLen-- && (pEncodedString[vIn] != '=') && fIsBase64(pEncodedString[vIn])) {
    aCharArray4[vI++] = pEncodedString[vIn];
    vIn++;
    if (vI == 4) {
      for (vI = 0; vI < 4; vI++) {
        aCharArray4[vI] = cBase64Chars.find(aCharArray4[vI]);
      }

      aCharArray3[0] = ((aCharArray4[0]) << 2) + ((aCharArray4[1] & 0x30) >> 4);
      aCharArray3[1] = ((aCharArray4[1] & 0xf) << 4) + ((aCharArray4[2] & 0x3c) >> 2);
      aCharArray3[2] = ((aCharArray4[2] & 0x3) << 6) + aCharArray4[3];

      for (vI = 0; (vI < 3); vI++) {
        lRet.push_back(aCharArray3[vI]);
      }

      vI = 0;
    }
  }

  if (vI) {
    for (vJ = vI; vJ < 4; vJ++) {
      aCharArray4[vJ] = 0;
    }

    for (vJ = 0; vJ < 4; vJ++) {
      aCharArray4[vJ] = cBase64Chars.find(aCharArray4[vJ]);
    }

    aCharArray3[0] = ((aCharArray4[0]) << 2) + ((aCharArray4[1] & 0x30) >> 4);
    aCharArray3[1] = ((aCharArray4[1] & 0xf) << 4) + ((aCharArray4[2] & 0x3c) >> 2);
    aCharArray3[2] = ((aCharArray4[2] & 0x3) << 6) + aCharArray4[3];

    for (vJ = 0; vJ < vI - 1; vJ++) {
      lRet.push_back(aCharArray3[vJ]);
    }
  }

  return lRet;
}

//
// server_tokens implementation
//

namespace {

constexpr uint32_t cServerTokensStateVersion = 1;

uint32_t server_tokens_state_u32(size_t value) {
  if (value > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("Server tokens state is too large");
  }
  return value;
}

class server_tokens_state_writer {
public:
  template <typename T> void write(T value) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
    const auto * vPtr = reinterpret_cast<const char *>(&value);
    data.insert(data.end(), vPtr, vPtr + sizeof(value));
  }

  template <typename T> void write(const std::vector<T> & pValues) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
    write(server_tokens_state_u32(pValues.size()));
    if (pValues.empty()) {
      return;
    }
    const auto * vPtr = reinterpret_cast<const char *>(pValues.data());
    data.insert(data.end(), vPtr, vPtr + pValues.size() * sizeof(T));
  }

  void mWriteMediaChunk(const mtmd_input_chunk * pChunk) {
    size_t vChunkSize = 0;
    if (fMtmdInputChunkSave(pChunk, nullptr, 0, &vChunkSize) != 0 || vChunkSize == 0) {
      throw std::runtime_error("Cannot serialize media chunk in server tokens");
    }
    std::vector<char> lChunkData(server_tokens_state_u32(vChunkSize));
    if (fMtmdInputChunkSave(pChunk, lChunkData.data(), lChunkData.size(), nullptr) != 0) {
      throw std::runtime_error("Cannot serialize media chunk in server tokens");
    }
    write(lChunkData);
  }

  std::vector<char> mTake() {
    data.resize((data.size() + sizeof(dgs_token) - 1) / sizeof(dgs_token) * sizeof(dgs_token), 0);
    return std::move(data);
  }

private:
  std::vector<char> data;
};

class server_tokens_state_reader {
public:
  server_tokens_state_reader(const char * data, size_t size) : data(data), size(size) {}

  template <typename T> T read() {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
    if (size - vPos < sizeof(T)) {
      throw std::runtime_error("Unexpected end of server tokens state");
    }
    T value;
    std::memcpy(&value, data + vPos, sizeof(value));
    vPos += sizeof(value);
    return value;
  }

  template <typename T> std::vector<T> mReadVector() {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
    const uint32_t cNValues = read<uint32_t>();
    // reject before resizing, so that a small corrupted payload cannot request a huge allocation
    if (cNValues > mRemaining() / sizeof(T)) {
      throw std::runtime_error("Unexpected end of server tokens state");
    }
    std::vector<T> lValues(cNValues);
    if (cNValues > 0) {
      std::memcpy(lValues.data(), data + vPos, lValues.size() * sizeof(T));
      vPos += lValues.size() * sizeof(T);
    }
    return lValues;
  }

  size_t mRemaining() const { return size - vPos; }

private:
  const char * data;
  size_t       size;
  size_t       vPos = 0;
};

}  // namespace

server_tokens::server_tokens(mtmd::input_chunks & pMtmdChunks, bool pHasMtmd) : has_mtmd(pHasMtmd) {
  for (size_t vI = 0; vI < pMtmdChunks.size(); ++vI) {
    push_back(pMtmdChunks[vI]);
  }
}

server_tokens::server_tokens(const dgs_tokens & pTokens, bool pHasMtmd) : has_mtmd(pHasMtmd), tokens(pTokens) {}

dgs_pos server_tokens::mPosNext(int64_t pNTokens) const {
  if (!has_mtmd) {
    if (pNTokens < 0) {
      return tokens.size();
    }

    return pNTokens;
  }

  if (pNTokens < 0) {
    dgs_pos vRes = tokens.size();

    for (auto vIt = map_idx_to_media.begin(); vIt != map_idx_to_media.end(); ++vIt) {
      const auto & vChunk = vIt->second;
      vRes += fMtmdInputChunkGetNPos(vChunk.get()) - fMtmdInputChunkGetNTokens(vChunk.get());
    }

    return vRes;
  }

  int64_t vIdx = 0;
  dgs_pos vPos = 0;

  GGML_ASSERT(pNTokens <= (int64_t) tokens.size());

  while (vIdx < pNTokens) {
    const auto cMediaIt = map_idx_to_media.find(vIdx);
    if (cMediaIt != map_idx_to_media.end()) {
      const auto &  vChunk = cMediaIt->second;
      const dgs_pos cNPos  = fMtmdInputChunkGetNPos(vChunk.get());
      const size_t  cNTok  = fMtmdInputChunkGetNTokens(vChunk.get());

      vPos += cNPos;
      vIdx += cNTok;
    } else {
      vPos++;
      vIdx++;
    }
  }

  return vPos;
}

size_t server_tokens::mSizeUpToPos(dgs_pos pMaxPos) const {
  if (!has_mtmd) {
    return std::min((size_t) pMaxPos, tokens.size());
  }

  size_t  vIdx = 0;
  dgs_pos vPos = 0;

  while (vIdx < tokens.size()) {
    const auto cMediaIt = map_idx_to_media.find(vIdx);
    if (cMediaIt != map_idx_to_media.end()) {
      const auto &  vChunk = cMediaIt->second;
      const dgs_pos cNPos  = fMtmdInputChunkGetNPos(vChunk.get());
      const size_t  cNTok  = fMtmdInputChunkGetNTokens(vChunk.get());

      vPos += cNPos;
      vIdx += cNTok;
    } else {
      vPos++;
      vIdx++;
    }

    if (vPos >= pMaxPos) {
      break;
    }
  }

  return vIdx;
}

std::string server_tokens::str() const {
  std::ostringstream vOss;
  vOss << "tokens: ";
  for (size_t vIdx = 0; vIdx < tokens.size(); ++vIdx) {
    dgs_token vT = tokens[vIdx];
    vOss << "idx:" << vIdx << " ";
    if (vT == DGS_TOKEN_NULL) {
      vOss << "<embd> ";
    } else {
      vOss << vT << " ";
    }
  }
  vOss << "\n";
  vOss << "image idx: ";
  for (const auto & vIt : map_idx_to_media) {
    vOss << vIt.first << ", ";
  }
  return vOss.str();
}

const mtmd::input_chunk_ptr & server_tokens::mFindChunk(size_t pIdx) const {
  auto vIt = map_idx_to_media.find(pIdx);
  if (vIt != map_idx_to_media.end()) {
    return vIt->second;
  }
  throw std::runtime_error("Chunk not found");
}

std::pair<const mtmd::input_chunk_ptr *, size_t> server_tokens::mFindNextMediaChunk(size_t pIdx) const {
  auto vIt = map_idx_to_media.upper_bound(pIdx);
  if (vIt != map_idx_to_media.end()) {
    return { &vIt->second, vIt->first };
  }
  return { nullptr, 0 };
}

void server_tokens::push_back(dgs_token pTok) {
  if (pTok == DGS_TOKEN_NULL) {
    throw std::runtime_error("Invalid token");
  }
  tokens.emplace_back(pTok);
}

void server_tokens::push_back(const mtmd_input_chunk * pChunk) {
  auto type = fMtmdInputChunkGetType(pChunk);
  if (type == cMtmdInputChunkTypeImage || type == cMtmdInputChunkTypeAudio) {
    GGML_ASSERT(has_mtmd);
    const size_t cNTokens  = fMtmdInputChunkGetNTokens(pChunk);
    size_t       vStartIdx = tokens.size();
    for (size_t vI = 0; vI < cNTokens; ++vI) {
      tokens.emplace_back(DGS_TOKEN_NULL);
    }
    mtmd::input_chunk_ptr vNewChunk(fMtmdInputChunkCopy(pChunk));
    map_idx_to_media[vStartIdx] = std::move(vNewChunk);
  } else if (type == cMtmdInputChunkTypeText) {
    size_t       vNTokens;
    const auto * vTextTokens = fMtmdInputChunkGetTokensText(pChunk, &vNTokens);
    for (size_t vI = 0; vI < vNTokens; ++vI) {
      push_back(vTextTokens[vI]);
    }
  } else {
    GGML_ABORT("Invalid chunk type");
  }
}

void server_tokens::mPushBackPlaceholder(const mtmd_input_chunk * pChunk) {
  auto type = fMtmdInputChunkGetType(pChunk);
  if (type == cMtmdInputChunkTypeImage || type == cMtmdInputChunkTypeAudio) {
    GGML_ASSERT(has_mtmd);
    mtmd::input_chunk_ptr vNewChunk(fMtmdInputChunkGetPlaceholder(pChunk));
    GGML_ASSERT(vNewChunk != nullptr && "failed to create placeholder chunk");
    const size_t cNTokens  = fMtmdInputChunkGetNTokens(pChunk);
    size_t       vStartIdx = tokens.size();
    for (size_t vI = 0; vI < cNTokens; ++vI) {
      tokens.emplace_back(DGS_TOKEN_NULL);
    }
    map_idx_to_media[vStartIdx] = std::move(vNewChunk);
  } else {
    push_back(pChunk);
  }
}

void server_tokens::push_back(server_tokens & pTokens) {
  size_t vStartIdx = size();
  for (size_t vI = 0; vI < pTokens.size(); vI++) {
    push_back(pTokens[vI]);
  }
  if (pTokens.has_mtmd) {
    // Assert if we are copying MTMD chunks to a server_tokens that does not have mtmd.
    // We could also just check, but this will prevent silently dropping MTMD data.
    GGML_ASSERT(has_mtmd);
    for (auto vIt = pTokens.map_idx_to_media.begin(); vIt != pTokens.map_idx_to_media.end();) {
      auto *                vChunk = pTokens.map_idx_to_media[vIt->first].get();
      mtmd::input_chunk_ptr vNewChunk(fMtmdInputChunkCopy(vChunk));
      map_idx_to_media[vStartIdx + vIt->first] = std::move(vNewChunk);
    }
  }
}

void server_tokens::insert(const dgs_tokens & pInpTokens) {
  tokens.insert(tokens.end(), pInpTokens.begin(), pInpTokens.end());
}

const dgs_tokens & server_tokens::mGetTokens() const {
  GGML_ASSERT(!has_mtmd);
  return tokens;
}

std::vector<char> server_tokens::serialize() const {
  static_assert(sizeof(dgs_token) == sizeof(uint32_t), "unexpected dgs_token size");

  server_tokens_state_writer vWriter;
  vWriter.write((dgs_token) DGS_TOKEN_NULL);
  vWriter.write(cServerTokensStateVersion);
  vWriter.write(tokens);

  std::vector<uint32_t> lMediaKeys;
  lMediaKeys.reserve(map_idx_to_media.size());
  for (const auto & vItem : map_idx_to_media) {
    lMediaKeys.push_back(server_tokens_state_u32(vItem.first));
  }
  vWriter.write(lMediaKeys);

  for (const auto & vItem : map_idx_to_media) {
    vWriter.mWriteMediaChunk(vItem.second.get());
  }

  return vWriter.mTake();
}

server_tokens server_tokens::deserialize(const dgs_tokens & pPacked, bool pHasMtmd) {
  static_assert(sizeof(dgs_token) == sizeof(uint32_t), "unexpected dgs_token size");

  if (pPacked.empty() || pPacked[0] != DGS_TOKEN_NULL) {
    // plain token list, as written by older versions
    return server_tokens(pPacked, pHasMtmd);
  }

  server_tokens_state_reader vReader(reinterpret_cast<const char *>(pPacked.data()),
                                     pPacked.size() * sizeof(dgs_token));
  vReader.read<dgs_token>();  // format marker
  if (vReader.read<uint32_t>() != cServerTokensStateVersion) {
    throw std::runtime_error("Unsupported server tokens state version");
  }

  const dgs_tokens lTokens = vReader.mReadVector<dgs_token>();

  // the media start indices, followed by the media chunks in the same order
  const std::vector<uint32_t> lMediaKeys = vReader.mReadVector<uint32_t>();
  if (!lMediaKeys.empty() && !pHasMtmd) {
    throw std::runtime_error("Cannot restore media tokens without an mmproj");
  }

  server_tokens vResult(lTokens, pHasMtmd);

  for (const uint32_t cKey : lMediaKeys) {
    const size_t            cStartIdx  = cKey;
    const std::vector<char> lChunkData = vReader.mReadVector<char>();
    if (lChunkData.empty()) {
      throw std::runtime_error("Cannot load media chunk from server tokens state");
    }

    mtmd::input_chunk_ptr vChunk(fMtmdInputChunkLoad(lChunkData.data(), lChunkData.size()));
    if (!vChunk) {
      throw std::runtime_error("Cannot load media chunk from server tokens state");
    }
    vResult.map_idx_to_media[cStartIdx] = std::move(vChunk);
  }

  if (vReader.mRemaining() >= sizeof(dgs_token)) {
    throw std::runtime_error("Trailing data in server tokens state");
  }

  return vResult;
}

dgs_tokens server_tokens::mGetTextTokens() const {
  dgs_tokens lRes;
  lRes.reserve(tokens.size());
  for (dgs_token vT : tokens) {
    if (vT != DGS_TOKEN_NULL) {
      lRes.push_back(vT);
    }
  }
  return lRes;
}

void server_tokens::mSetToken(dgs_pos pPos, dgs_token pId) {
  GGML_ASSERT(!has_mtmd);  // only allow this if mtmd is disabled
  tokens[pPos] = pId;
}

void server_tokens::mKeepFirst(size_t n) {
  GGML_ASSERT(n <= tokens.size());
  if (has_mtmd) {
    if (n == tokens.size()) {
      return;  // nothing to do
    }
    // we throw an error if we try to remove a token in the middle of an image
    // for ex. with input of 5 text tokens and 2 images:
    //    [0] [1] [2] [3] [4] [img0] [img0] [img0] [img1] [img1]
    // n  1   2   3   4   5   6      7      8      9      10
    // allowed to resize      ^                    ^
    // disallowed to resize          ^      ^             ^
    if (n > 0) {
      // make sure we never remove tokens in the middle of an image
      // note that the case where we keep a full image at the end is allowed:
      //   tokens[n - 1] == DGS_TOKEN_NULL && tokens[n] != DGS_TOKEN_NULL
      if (tokens[n - 1] == DGS_TOKEN_NULL && tokens[n] == DGS_TOKEN_NULL) {
        mFindChunk(n);  // will throw an error if the cut is not at a chunk boundary
      }
    }
    // remove all image chunks that are not used anymore
    for (auto vIt = map_idx_to_media.begin(); vIt != map_idx_to_media.end();) {
      size_t vIdx = vIt->first;
      if (vIdx >= n) {
        vIt = map_idx_to_media.erase(vIt);
      } else {
        ++vIt;
      }
    }
  }
  tokens.resize(n);
}

std::string server_tokens::mDetokenize(const dgs_context * pCtx, bool pSpecial) const {
  dgs_tokens lTextTokens;
  lTextTokens.reserve(tokens.size());
  for (const auto & vT : tokens) {
    if (vT != DGS_TOKEN_NULL) {
      lTextTokens.push_back(vT);
    }
  }
  return fCommonDetokenize(pCtx, lTextTokens, pSpecial);
}

size_t server_tokens::mGetCommonPrefix(const server_tokens & pB) const {
  const size_t cMaxIdx = std::min(tokens.size(), pB.tokens.size());

  if (!has_mtmd) {
    for (size_t vI = 0; vI < cMaxIdx; ++vI) {
      if (tokens[vI] == pB.tokens[vI]) {
        continue;
      }

      return vI;
    }

    return cMaxIdx;
  }

  for (size_t vI = 0; vI < cMaxIdx; ++vI) {
    const dgs_token cAi = tokens[vI];
    const dgs_token cBi = pB.tokens[vI];

    if (cAi == DGS_TOKEN_NULL && cBi == DGS_TOKEN_NULL) {
      const auto & vAChunk = mFindChunk(vI);
      const auto & vBChunk = pB.mFindChunk(vI);

      GGML_ASSERT(vAChunk && vBChunk);

      const std::string cIdAi = fMtmdInputChunkGetId(vAChunk.get());
      const std::string cIdBi = fMtmdInputChunkGetId(vBChunk.get());

      const size_t cNTokA = fMtmdInputChunkGetNTokens(vAChunk.get());
      const size_t cNTokB = fMtmdInputChunkGetNTokens(vBChunk.get());

      if (cIdAi == cIdBi && cNTokA == cNTokB) {
        GGML_ASSERT(cNTokA > 0 && "Invalid media chunk");  // should never happen
        vI += cNTokA - 1;                                  // will be +1 by the for loop
        continue;
      }

      return vI;
    }

    if (cAi == cBi) {
      continue;
    }

    return vI;
  }

  return cMaxIdx;  // all tokens are equal
}

common_chat_msg_spans server_tokens::mFindMessageSpans(const common_chat_msg_delimiters & pDelims) const {
  std::map<size_t, size_t> dSkips;
  for (const auto & vIt : map_idx_to_media) {
    dSkips[vIt.first] = fMtmdInputChunkGetNTokens(vIt.second.get());
  }
  return pDelims.mSplit(tokens, dSkips);
}

bool server_tokens::mValidate(const struct dgs_context * pCtx) const {
  const dgs_model * vModel  = fDgsGetModel(pCtx);
  const dgs_vocab * vVocab  = fDgsModelGetVocab(vModel);
  const int32_t     cNVocab = fDgsVocabNTokens(vVocab);
  size_t            vNMedia = 0;

  for (size_t vI = 0; vI < tokens.size(); ++vI) {
    const auto & vT = tokens[vI];
    if (vT == DGS_TOKEN_NULL) {
      try {
        const auto & vChunk = mFindChunk(vI);
        if (fMtmdInputChunkGetType(vChunk.get()) == cMtmdInputChunkTypeText) {
          return false;
        }
        const size_t  cNTokens = fMtmdInputChunkGetNTokens(vChunk.get());
        const dgs_pos cNPos    = fMtmdInputChunkGetNPos(vChunk.get());
        if (cNTokens == 0 || cNPos <= 0 || cNTokens > tokens.size() - vI) {
          return false;
        }
        for (size_t vJ = vI; vJ < vI + cNTokens; ++vJ) {
          if (tokens[vJ] != DGS_TOKEN_NULL) {
            return false;
          }
        }
        ++vNMedia;
        vI += cNTokens - 1;
      } catch (const std::exception & vE) {
        return false;
      }
    } else if (vT < 0 || vT >= cNVocab) {
      return false;
    }
  }
  return vNMedia == map_idx_to_media.size();
}

server_tokens server_tokens::clone() const {
  server_tokens vRes;
  vRes.has_mtmd = has_mtmd;
  vRes.tokens   = tokens;
  for (auto vIt = map_idx_to_media.begin(); vIt != map_idx_to_media.end(); ++vIt) {
    size_t                        vIdx   = vIt->first;
    const mtmd::input_chunk_ptr & vChunk = vIt->second;
    vRes.map_idx_to_media[vIdx]          = mtmd::input_chunk_ptr(fMtmdInputChunkCopy(vChunk.get()));
  }
  return vRes;
}

//
// tokenizer and input processing utils
//

bool fJsonIsArrayOfNumbers(const json & data) {
  if (data.is_array()) {
    for (const auto & vE : data) {
      if (!vE.mIsNumberInteger()) {
        return false;
      }
    }
    return true;
  }
  return false;
}

bool fJsonIsArrayOfMixedNumbersStrings(const json & data) {
  bool vSeenString = false;
  bool vSeenNumber = false;
  if (data.is_array()) {
    for (const auto & vE : data) {
      vSeenString |= vE.is_string();
      vSeenNumber |= vE.mIsNumberInteger();
      if (vSeenNumber && vSeenString) {
        return true;
      }
    }
  }
  return false;
}

bool fJsonIsArrayAndContainsNumbers(const json & data) {
  if (data.is_array()) {
    for (const auto & vE : data) {
      if (vE.mIsNumberInteger()) {
        return true;
      }
    }
    return false;
  }
  return false;
}

json fJsonGetNestedValues(const std::vector<std::string> & pPaths, const json & pJs) {
  json vResult = json::mObject();

  for (const std::string & vPath : pPaths) {
    json       vCurrent   = pJs;
    const auto lKeys      = fStringSplit<std::string>(vPath, /*separator*/ '/');
    bool       vValidPath = true;
    for (const std::string & vK : lKeys) {
      if (vValidPath && vCurrent.is_object() && vCurrent.contains(vK)) {
        vCurrent = vCurrent[vK];
      } else {
        vValidPath = false;
      }
    }
    if (vValidPath) {
      vResult[vPath] = vCurrent;
    }
  }
  return vResult;
}

dgs_tokens fTokenizeMixed(const dgs_vocab * pVocab, const json & pJsonPrompt, bool pAddSpecial, bool pParseSpecial) {
  // If `add_bos` is true, we only add BOS, when json_prompt is a string,
  // or the first element of the json_prompt array is a string.
  dgs_tokens lPromptTokens;

  if (pJsonPrompt.is_array()) {
    bool vFirst = true;
    for (const auto & vP : pJsonPrompt) {
      if (vP.is_string()) {
        auto vS = vP.template get<std::string>();

        dgs_tokens lP;
        if (vFirst) {
          lP     = fCommonTokenize(pVocab, vS, pAddSpecial, pParseSpecial);
          vFirst = false;
        } else {
          lP = fCommonTokenize(pVocab, vS, false, pParseSpecial);
        }

        lPromptTokens.insert(lPromptTokens.end(), lP.begin(), lP.end());
      } else {
        if (vFirst) {
          vFirst = false;
        }

        lPromptTokens.push_back(vP.template get<dgs_token>());
      }
    }
  } else {
    auto vS       = pJsonPrompt.template get<std::string>();
    lPromptTokens = fCommonTokenize(pVocab, vS, pAddSpecial, pParseSpecial);
  }

  return lPromptTokens;
}

size_t fValidateUtf8(const std::string & pText) {
  size_t vLen = pText.size();
  if (vLen == 0) {
    return 0;
  }

  // Check the last few bytes to see if a multi-byte character is cut off
  for (size_t vI = 1; vI <= 4 && vI <= vLen; ++vI) {
    unsigned char vC = pText[vLen - vI];
    // Check for start of a multi-byte sequence from the end
    if ((vC & 0xE0) == 0xC0) {
      // 2-byte character start: 110xxxxx
      // Needs at least 2 bytes
      if (vI < 2) {
        return vLen - vI;
      }
    } else if ((vC & 0xF0) == 0xE0) {
      // 3-byte character start: 1110xxxx
      // Needs at least 3 bytes
      if (vI < 3) {
        return vLen - vI;
      }
    } else if ((vC & 0xF8) == 0xF0) {
      // 4-byte character start: 11110xxx
      // Needs at least 4 bytes
      if (vI < 4) {
        return vLen - vI;
      }
    }
  }

  // If no cut-off multi-byte character is found, return full length
  return vLen;
}

server_tokens fProcessMtmdPrompt(mtmd_context *                  pMctx,
                                 const std::string &             pPrompt,
                                 const std::vector<raw_buffer> & pFiles,
                                 const mtmd_helper_init_opt &    pInitOpt,
                                 bool                            pIsPlaceholder) {
  // these will be freed upon going out of scope
  mtmd::bitmaps                       bitmaps;
  std::vector<mtmd_helper::video_ptr> lVideos;
  for (auto & lFile : pFiles) {
    auto vOut = fMtmdHelperBitmapInitFromBuf(pMctx, lFile.data(), lFile.size(), pIsPlaceholder, pInitOpt);
    if (!vOut.bitmap) {
      throw std::runtime_error("Failed to load image or audio file");
    }
    bitmaps.lEntries.emplace_back(vOut.bitmap);
    if (vOut.vVideoCtx) {
      lVideos.emplace_back(vOut.vVideoCtx);
    }
  }
  // process prompt
  std::vector<server_tokens> lInputs;
  // multimodal
  mtmd_input_text            vInpTxt = {
    pPrompt.data(),
    pPrompt.size(),
    /* add_special */ true,
    /* parse_special */ true,
  };
  mtmd::input_chunks vChunks(fMtmdInputChunksInit());
  auto               lBitmapsCPtr = bitmaps.mCPtr();
  int32_t vTokenized = fMtmdTokenize(pMctx, vChunks.vPtr.get(), &vInpTxt, lBitmapsCPtr.data(), lBitmapsCPtr.size());
  if (vTokenized != 0) {
    throw std::runtime_error("Failed to tokenize prompt");
  }
  auto vResult = server_tokens(vChunks, true);
  return vResult;
}

/**
 * tokenize a single input "prompt" object
 * use tokenize_input_prompts() if the input could be an array.
 * this supports these cases:
 * - "prompt": "string"
 * - "prompt": [12, 34, 56]
 * - "prompt": [12, 34, "string", 56, 78]
 * - "prompt": { "prompt_string": "string", "multimodal_data": [ "base64" ] }
 */
server_tokens fTokenizeInputSubprompt(const dgs_vocab *            pVocab,
                                      mtmd_context *               pMctx,
                                      const json &                 pJsonPrompt,
                                      bool                         pAddSpecial,
                                      bool                         pParseSpecial,
                                      const mtmd_helper_init_opt & pInitOpt) {
  constexpr char aJsonStringPromptKey[] = "prompt_string";
  constexpr char aJsonMtmdDataKey[]     = "multimodal_data";
  const bool     cHasMtmd               = pMctx != nullptr;
  if (pJsonPrompt.is_string() || fJsonIsArrayOfMixedNumbersStrings(pJsonPrompt)) {
    // string or mixed
    dgs_tokens lTmp = fTokenizeMixed(pVocab, pJsonPrompt, pAddSpecial, pParseSpecial);
    return server_tokens(lTmp, false);
  } else if (fJsonIsArrayOfNumbers(pJsonPrompt)) {
    // array of tokens
    dgs_tokens lTmp = pJsonPrompt.get<dgs_tokens>();
    return server_tokens(lTmp, false);
  } else if (pJsonPrompt.contains(aJsonStringPromptKey)) {
    // JSON object with prompt key.
    if (pJsonPrompt.contains(aJsonMtmdDataKey)) {
      if (!cHasMtmd) {
        throw std::runtime_error("Multimodal data provided, but model does not support multimodal requests.");
      }

      // JSON object with prompt and multimodal key.
      std::vector<raw_buffer> lFiles;
      for (const auto & vEntry : pJsonPrompt.at(aJsonMtmdDataKey)) {
        lFiles.push_back(fBase64Decode(vEntry));
      }
      return fProcessMtmdPrompt(pMctx, pJsonPrompt.at(aJsonStringPromptKey), lFiles, pInitOpt);
    } else {
      // Not multimodal, but contains a subobject.
      dgs_tokens lTmp = fTokenizeMixed(pVocab, pJsonPrompt.at(aJsonStringPromptKey), pAddSpecial, pParseSpecial);
      return server_tokens(lTmp, false);
    }
  } else {
    throw std::invalid_argument(
      "\"prompt\" elements must be a string, a list of tokens, a JSON object containing a prompt string, or a list of "
      "mixed strings & tokens.");
  }
}

std::vector<server_tokens> fTokenizeInputPrompts(const dgs_vocab *            pVocab,
                                                 mtmd_context *               pMctx,
                                                 const json &                 pJsonPrompt,
                                                 bool                         pAddSpecial,
                                                 bool                         pParseSpecial,
                                                 const mtmd_helper_init_opt & pInitOpt) {
  std::vector<server_tokens> lResult;
  if (pJsonPrompt.is_array() && !fJsonIsArrayAndContainsNumbers(pJsonPrompt)) {
    lResult.reserve(pJsonPrompt.size());
    for (const auto & vP : pJsonPrompt) {
      lResult.push_back(fTokenizeInputSubprompt(pVocab, pMctx, vP, pAddSpecial, pParseSpecial, pInitOpt));
    }
  } else {
    lResult.push_back(fTokenizeInputSubprompt(pVocab, pMctx, pJsonPrompt, pAddSpecial, pParseSpecial, pInitOpt));
  }
  if (lResult.empty()) {
    throw std::invalid_argument("\"prompt\" must not be empty");
  }
  return lResult;
}

//
// OAI utils
//

// used by /completions endpoint
json fOaicompatCompletionParamsParse(const json & pBody) {
  json vDgsParams;

  if (!pBody.contains("prompt")) {
    throw std::runtime_error("\"prompt\" is required");
  }

  // Handle "stop" field
  if (pBody.contains("stop") && pBody.at("stop").is_string()) {
    vDgsParams["stop"] = json::mArray({ pBody.at("stop").get<std::string>() });
  } else {
    vDgsParams["stop"] = fJsonValue(pBody, "stop", json::mArray());
  }

  // Handle "echo" field
  if (fJsonValue(pBody, "echo", false)) {
    throw std::runtime_error("Only no echo is supported");
  }

  // Params supported by OAI but unsupported by dgs.cpp
  static const std::vector<std::string> lUnsupportedParams{ "best_of", "suffix" };
  for (const auto & vParam : lUnsupportedParams) {
    if (pBody.contains(vParam)) {
      throw std::runtime_error("Unsupported param: " + vParam);
    }
  }

  // Copy remaining properties to dgs_params
  for (const auto & vItem : pBody.mItems()) {
    // Exception: if "n_predict" is present, we overwrite the value specified earlier by "max_tokens"
    if (!vDgsParams.contains(vItem.mKey()) || vItem.mKey() == "n_predict") {
      vDgsParams[vItem.mKey()] = vItem.value();
    }
  }

  return vDgsParams;
}

// url can be
// - http(s):// for remote files
// - file:// for local files (only allowed if media_path is set)
// - data: for base64 encoded data with uri scheme (e.g. data:image/png;base64,...)
// - raw base64 encoded data
void fHandleMedia(std::vector<raw_buffer> & pOutFiles, const std::string & pUrl, const std::string & pMediaPath) {
  if (!pMediaPath.empty()) {
    // should already be enforced by arg.cpp, but checking just in case
    GGML_ASSERT(pMediaPath.back() == DIRECTORY_SEPARATOR);
  }

  if (fStringStartsWith(pUrl, "http")) {
    // download remote image
    // TODO @ngxson : maybe make these params configurable
    common_remote_params vParams;
    vParams.vMaxSize = 1024 * 1024 * 10;  // 10MB
    vParams.vTimeout = 10;                // seconds
    SRV_INF("downloading image from '%s'\n", pUrl.c_str());
    auto lRes = fCommonRemoteGetContent(pUrl, vParams);
    if (200 <= lRes.first && lRes.first < 300) {
      SRV_INF("downloaded %zu bytes\n", lRes.second.size());
      raw_buffer data;
      data.insert(data.end(), lRes.second.begin(), lRes.second.end());
      pOutFiles.push_back(data);
    } else {
      throw std::runtime_error("Failed to download image");
    }

  } else if (fStringStartsWith(pUrl, "file://")) {
    if (pMediaPath.empty()) {
      throw std::invalid_argument("file:// URLs are not allowed unless --media-path is specified");
    }
    // load local image file
    std::string vFilePath = pUrl.substr(7);  // remove "file://"
    raw_buffer  data;
    if (!fFsValidateFilename(vFilePath, true)) {
      throw std::invalid_argument("file path is not allowed: " + vFilePath);
    }
    SRV_INF("loading image from local file '%s'\n", (pMediaPath + vFilePath).c_str());
    std::ifstream vFile(pMediaPath + vFilePath, std::ios::binary);
    if (!vFile) {
      throw std::invalid_argument("file does not exist or cannot be opened: " + vFilePath);
    }
    data.assign((std::istreambuf_iterator<char>(vFile)), std::istreambuf_iterator<char>());
    pOutFiles.push_back(data);

  } else if (fStringStartsWith(pUrl, "data:")) {
    // try to decode base64 image, video, or audio
    std::vector<std::string> lParts = fStringSplit<std::string>(pUrl, /*separator*/ ',');
    if (lParts.size() != 2) {
      throw std::invalid_argument("Invalid uri-encoded base64 value");
    } else if (!fStringStartsWith(lParts[0], "data:image/") && !fStringStartsWith(lParts[0], "data:video/") &&
               !fStringStartsWith(lParts[0], "data:audio/")) {
      throw std::invalid_argument("Invalid uri format: " + lParts[0]);
    } else if (!fStringEndsWith(lParts[0], "base64")) {
      throw std::invalid_argument("uri must be base64 encoded");
    } else {
      auto vBase64Data  = lParts[1];
      auto lDecodedData = fBase64Decode(vBase64Data);
      pOutFiles.push_back(lDecodedData);
    }

  } else {
    // try as raw base64 string
    auto lDecodedData = fBase64Decode(pUrl);
    if (lDecodedData.empty()) {
      throw std::runtime_error("Invalid base64 value");
    }
    pOutFiles.push_back(lDecodedData);
  }
}

// load media files from an OAI content array, then replace each media part with a media marker text part
static void fOaicompatContentLoadMedia(json &                     pContent,
                                       const server_chat_params & pOpt,
                                       std::vector<raw_buffer> &  pOutFiles) {
  for (auto & vP : pContent) {
    std::string type = fJsonValue(vP, "type", std::string());
    if (type == "image_url") {
      if (!pOpt.vAllowImage) {
        throw std::runtime_error(
          "image input is not supported - hint: if this is unexpected, you may need to provide the mmproj");
      }

      json        vImageUrl = fJsonValue(vP, "image_url", json::mObject());
      std::string vUrl      = fJsonValue(vImageUrl, "url", std::string());
      fHandleMedia(pOutFiles, vUrl, pOpt.vMediaPath);

      vP["type"] = "media_marker";
      vP["text"] = fGetMediaMarker();
      vP.erase("image_url");

    } else if (type == "input_audio") {
      if (!pOpt.vAllowAudio) {
        throw std::runtime_error(
          "audio input is not supported - hint: if this is unexpected, you may need to provide the mmproj");
      }

      // note: don't need to validate "format", it's redundant
      json        vInputAudio = fJsonValue(vP, "input_audio", json::mObject());
      std::string vUrl        = fJsonValue(vInputAudio, "data", fJsonValue(vInputAudio, "url", std::string()));
      fHandleMedia(pOutFiles, vUrl, pOpt.vMediaPath);

      vP["type"] = "media_marker";
      vP["text"] = fGetMediaMarker();
      vP.erase("input_audio");

    } else if (type == "input_video" || type == "video_url") {
      if (!pOpt.vAllowVideo) {
        throw std::runtime_error(
          "video input is not supported - hint: if this is unexpected, you may need to provide the mmproj");
      }

      // accept the OpenAI-style "video_url" key as an alias of "input_video"
      json        vInputVideo = fJsonValue(vP, type, json::mObject());
      std::string vUrl        = fJsonValue(vInputVideo, "data", fJsonValue(vInputVideo, "url", std::string()));
      fHandleMedia(pOutFiles, vUrl, pOpt.vMediaPath);

      vP["type"] = "media_marker";
      vP["text"] = fGetMediaMarker();
      vP.erase("input_video");
      vP.erase("video_url");

    } else if (type != "text") {
      throw std::invalid_argument("unsupported content[].type");
    }
  }
}

server_tokens fTokenizeOaiContentArray(const dgs_vocab *            pVocab,
                                       mtmd_context *               pMctx,
                                       const server_chat_params &   pOpt,
                                       json                         pContent,
                                       bool                         pAddSpecial,
                                       bool                         pParseSpecial,
                                       const mtmd_helper_init_opt & pInitOpt) {
  if (!pContent.is_array()) {
    throw std::invalid_argument("\"content\" must be an array");
  }

  std::vector<raw_buffer> lFiles;
  fOaicompatContentLoadMedia(pContent, pOpt, lFiles);

  std::string vPrompt;
  for (const auto & vP : pContent) {
    vPrompt += fJsonValue(vP, "text", std::string());
  }

  if (lFiles.empty()) {
    return server_tokens(fCommonTokenize(pVocab, vPrompt, pAddSpecial, pParseSpecial), false);
  }
  return fProcessMtmdPrompt(pMctx, vPrompt, lFiles, pInitOpt);
}

// used by /chat/completions endpoint
json fOaicompatChatParamsParse(const dgs_vocab *          pVocab,
                               json &                     pBody, /* openai api json semantics */
                               const server_chat_params & pOpt,
                               std::vector<raw_buffer> &  pOutFiles,
                               common_chat_session &      pOutSession) {
  json vDgsParams;

  auto vTools      = fJsonValue(pBody, "tools", json());
  auto vHasTools   = vTools.is_array() && !vTools.empty();
  auto stream      = fJsonValue(pBody, "stream", false);
  auto vToolChoice = fJsonValue(pBody, "tool_choice", std::string("auto"));

  if (!pOpt.vUseJinja) {
    if (vHasTools) {
      throw std::runtime_error("tools require the current Jinja template engine");
    }
    if (vToolChoice != "auto") {
      throw std::runtime_error("tool_choice requires the current Jinja template engine");
    }
  }

  // Handle "stop" field
  if (pBody.contains("stop") && pBody.at("stop").is_string()) {
    vDgsParams["stop"] = json::mArray({ pBody.at("stop").get<std::string>() });
  } else {
    vDgsParams["stop"] = fJsonValue(pBody, "stop", json::mArray());
  }

  auto vJsonSchema = fJsonValue(pBody, "json_schema", json());
  auto vGrammar    = fJsonValue(pBody, "grammar", std::string());
  if (!vJsonSchema.is_null() && !vGrammar.empty()) {
    throw std::runtime_error("Cannot use both json_schema and grammar");
  }

  // Handle "response_format" field
  if (pBody.contains("response_format")) {
    json        vResponseFormat = fJsonValue(pBody, "response_format", json::mObject());
    std::string vResponseType   = fJsonValue(vResponseFormat, "type", std::string());
    if (vResponseType == "json_object") {
      if (vResponseFormat.contains("schema") || vJsonSchema.empty()) {
        vJsonSchema = fJsonValue(vResponseFormat, "schema", json::mObject());
      }
    } else if (vResponseType == "json_schema") {
      auto vSchemaWrapper = fJsonValue(vResponseFormat, "json_schema", json::mObject());
      vJsonSchema         = fJsonValue(vSchemaWrapper, "schema", json::mObject());
    } else if (!vResponseType.empty() && vResponseType != "text") {
      throw std::invalid_argument("response_format type must be one of \"text\" or \"json_object\", but got: " +
                                  vResponseType);
    }
  }

  // an absent or empty schema means any object
  if (vJsonSchema.is_object() && vJsonSchema.empty()) {
    vJsonSchema["type"] = "object";
  }

  // get input files
  if (!pBody.contains("messages")) {
    throw std::invalid_argument("'messages' is required");
  }
  json & vMessages = pBody.at("messages");
  if (!vMessages.is_array()) {
    throw std::invalid_argument("Expected 'messages' to be an array");
  }
  for (auto & vMsg : vMessages) {
    std::string vRole = fJsonValue(vMsg, "role", std::string());
    if (vRole != "assistant" && !vMsg.contains("content")) {
      throw std::invalid_argument("All non-assistant messages must contain 'content'");
    }
    if (vRole == "assistant") {
      if (!vMsg.contains("content") && !vMsg.contains("tool_calls")) {
        throw std::invalid_argument("Assistant message must contain either 'content' or 'tool_calls'!");
      }
      if (!vMsg.contains("content")) {
        continue;  // avoid errors with no content
      }
    }
    json & vContent = vMsg.at("content");
    if (vContent.is_string() || vContent.is_null()) {
      continue;
    }

    if (!vContent.is_array()) {
      throw std::invalid_argument("Expected 'content' to be a string or an array");
    }

    fOaicompatContentLoadMedia(vContent, pOpt, pOutFiles);
  }

  auto caps = fCommonChatTemplatesGetCaps(pOpt.vTmpls.get());

  common_chat_templates_inputs vInputs;
  vInputs.lMessages             = fCommonChatMsgsParseOaicompat(vMessages);
  vInputs.lTools                = fCommonChatToolsParseOaicompat(vTools);
  vInputs.vToolChoice           = fCommonChatToolChoiceParseOaicompat(vToolChoice);
  vInputs.vJsonSchema           = vJsonSchema.is_null() ? "" : vJsonSchema.mDump();
  vInputs.vGrammar              = vGrammar;
  vInputs.vUseJinja             = pOpt.vUseJinja;
  vInputs.vParallelToolCalls    = fJsonValue(pBody, "parallel_tool_calls", caps["supports_parallel_tool_calls"]);
  vInputs.vAddGenerationPrompt  = fJsonValue(pBody, "add_generation_prompt", true);
  vInputs.vContinueFinalMessage = pBody.contains("continue_final_message") ?
                                    fCommonChatContinuationParse(pBody.at("continue_final_message")) :
                                    cCommonChatContinuationNone;
  if (vInputs.vContinueFinalMessage == cCommonChatContinuationNone && pOpt.vPrefillAssistant &&
      !vInputs.lMessages.empty() && vInputs.lMessages.back().vRole == "assistant") {
    if (vInputs.lMessages.size() >= 2 && vInputs.lMessages[vInputs.lMessages.size() - 2].vRole == "assistant") {
      throw std::invalid_argument("Cannot have 2 or more assistant messages at the end of the list.");
    }
    vInputs.vContinueFinalMessage = cCommonChatContinuationAuto;
    vInputs.vAddGenerationPrompt  = false;
  }
  if (vInputs.vContinueFinalMessage != cCommonChatContinuationNone && vInputs.vAddGenerationPrompt) {
    throw std::invalid_argument("Cannot set both add_generation_prompt and continue_final_message to true.");
  }
  if (vInputs.vContinueFinalMessage != cCommonChatContinuationNone && !vInputs.lMessages.empty() &&
      vInputs.lMessages.back().vRole == "assistant" && !vInputs.lMessages.back().lToolCalls.empty()) {
    throw std::invalid_argument("Cannot continue an assistant message that contains tool calls.");
  }
  vInputs.vReasoningFormat = pOpt.vReasoningFormat;
  if (pBody.contains("reasoning_format")) {
    vInputs.vReasoningFormat = fCommonReasoningFormatFromName(pBody.at("reasoning_format").get<std::string>());
  }
  vInputs.vEnableThinking = pOpt.vEnableThinking;
  if (!vInputs.lTools.empty() && vInputs.vToolChoice != cCommonChatToolChoiceNone) {
    if (pBody.contains("grammar")) {
      throw std::invalid_argument("Cannot use custom grammar constraints with tools.");
    }
  }

  // merge the template args provided from command line with the args provided in the user request
  auto vChatTemplateKwargsObject = fJsonValue(pBody, "chat_template_kwargs", json::mObject());
  vInputs.dChatTemplateKwargs    = pOpt.dChatTemplateKwargs;
  for (const auto & vItem : vChatTemplateKwargsObject.mItems()) {
    vInputs.dChatTemplateKwargs[vItem.mKey()] = vItem.value().mDump();
  }

  // parse the "enable_thinking" kwarg to override the default value
  auto vEnableThinkingKwarg = fJsonValue(vInputs.dChatTemplateKwargs, "enable_thinking", std::string(""));
  if (vEnableThinkingKwarg == "true") {
    vInputs.vEnableThinking = true;
  } else if (vEnableThinkingKwarg == "false") {
    vInputs.vEnableThinking = false;
  } else if (!vEnableThinkingKwarg.empty() && vEnableThinkingKwarg[0] == '"') {
    throw std::invalid_argument("invalid type for \"enable_thinking\" (expected boolean, got string)");
  }

  // Parse the OAI "reasoning_effort" field; "none" disables reasoning.
  if (pBody.contains("reasoning_effort")) {
    auto vReasoningEffort = fJsonValue(pBody, "reasoning_effort", std::string(""));
    if (vReasoningEffort == "none") {
      vInputs.vEnableThinking = false;
      vInputs.dChatTemplateKwargs.erase("reasoning_effort");
    } else if (!vReasoningEffort.empty()) {
      vInputs.dChatTemplateKwargs["reasoning_effort"] = json(vReasoningEffort).mDump();
    }
  }

  vInputs.vForcePureContent = pOpt.vForcePureContent;

  // Apply chat template to the list of messages
  common_chat_session_params vSessionParams;
  vSessionParams.vEcho = fJsonValue(pBody, "echo", false);
  pOutSession          = common_chat_session(pOpt.vTmpls.get(), pVocab, vInputs, vSessionParams);

  vDgsParams["prompt"] = pOutSession.mPrompt();

  // Reasoning budget: pass parameters through to sampling layer
  {
    int vReasoningBudget =
      fJsonValue(pBody, "reasoning_budget_tokens", fJsonValue(pBody, "thinking_budget_tokens", -1));
    if (vReasoningBudget == -1) {
      vReasoningBudget = pOpt.vReasoningBudget;
    }

    if (!pOutSession.mThinkingEndTags().empty()) {
      vDgsParams["reasoning_budget_tokens"]    = vReasoningBudget;
      vDgsParams["reasoning_budget_start_tag"] = pOutSession.mThinkingStartTag();
      vDgsParams["reasoning_budget_end_tags"]  = pOutSession.mThinkingEndTags();
      vDgsParams["reasoning_budget_message"] =
        fJsonValue(pBody, "reasoning_budget_message", pOpt.vReasoningBudgetMessage);
      vDgsParams["reasoning_control"] = fJsonValue(pBody, "reasoning_control", false);
    }
  }

  // Handle "logprobs" field
  // TODO: The response format of this option is not yet OAI-compatible, but seems like no one really using it; We may need to fix it in the future
  if (fJsonValue(pBody, "logprobs", false)) {
    if (vHasTools && stream) {
      throw std::invalid_argument("logprobs is not supported with tools + stream");
    }
    vDgsParams["n_probs"] = fJsonValue(pBody, "top_logprobs", 20);
  } else if (pBody.contains("top_logprobs") && !pBody.at("top_logprobs").is_null()) {
    throw std::invalid_argument("top_logprobs requires logprobs to be set to true");
  }

  // Copy remaining properties to dgs_params
  // This allows user to use dgs.cpp-specific params like "mirostat", ... via OAI endpoint.
  // See "launch_slot_with_task()" for a complete list of params supported by dgs.cpp
  for (const auto & vItem : pBody.mItems()) {
    // Exception: if "n_predict" is present, we overwrite the value specified earlier by "max_tokens"
    if (!vDgsParams.contains(vItem.mKey()) || vItem.mKey() == "n_predict") {
      vDgsParams[vItem.mKey()] = vItem.value();
    }
  }

  // the session owns these, the server applies them with server_task::apply_chat_session()
  for (const char * vKey : { "grammar_lazy", "grammar_triggers", "preserved_tokens" }) {
    vDgsParams.erase(vKey);
  }
  if (!pOutSession.mGrammar().empty()) {
    vDgsParams.erase("grammar");
    vDgsParams.erase("json_schema");
  }

  return vDgsParams;
}

json fFormatEmbeddingsResponseOaicompat(const json &        pRequest,
                                        const std::string & pModelName,
                                        const json &        pEmbeddings,
                                        bool                pUseBase64) {
  json    data     = json::mArray();
  int32_t vNTokens = 0;
  int     vI       = 0;
  for (const auto & vElem : pEmbeddings) {
    json vEmbeddingObj;

    if (pUseBase64) {
      const auto & lVec      = fJsonValue(vElem, "embedding", json::mArray()).get<std::vector<float>>();
      const char * vDataPtr  = reinterpret_cast<const char *>(lVec.data());
      size_t       vDataSize = lVec.size() * sizeof(float);
      vEmbeddingObj          = {
        { "embedding", base64::mEncode(vDataPtr, vDataSize) },
        { "index", vI++ },
        { "object", "embedding" },
        { "encoding_format", "base64" }
      };
    } else {
      vEmbeddingObj = {
        { "embedding", fJsonValue(vElem, "embedding", json::mArray()) },
        { "index", vI++ },
        { "object", "embedding" }
      };
    }
    data.push_back(vEmbeddingObj);

    vNTokens += fJsonValue(vElem, "tokens_evaluated", 0);
  }

  json vRes = json{
    { "model", fJsonValue(pRequest, "model", pModelName) },
    { "object", "list" },
    { "usage", json{ { "prompt_tokens", vNTokens }, { "total_tokens", vNTokens } } },
    { "data", data }
  };

  return vRes;
}

json fFormatResponseRerank(const json &               pRequest,
                           const std::string &        pModelName,
                           const json &               pRanks,
                           bool                       pIsTeiFormat,
                           std::vector<std::string> & pTexts,
                           int                        pTopN) {
  int32_t           vNTokens    = 0;
  bool              vReturnText = pIsTeiFormat && fJsonValue(pRequest, "return_text", false);
  std::vector<json> lElements;  // Temporary vector to hold unsorted elements
  std::string       vScoreLabel = pIsTeiFormat ? "score" : "relevance_score";
  for (const auto & vRank : pRanks) {
    int  vIndex = fJsonValue(vRank, "index", 0);
    json vElem  = json{
      { "index", vIndex },
      { vScoreLabel, fJsonValue(vRank, "score", 0.0) },
    };
    vNTokens += fJsonValue(vRank, "tokens_evaluated", 0);
    if (vReturnText) {
      vElem["text"] = std::move(pTexts[vIndex]);
    }
    lElements.push_back(vElem);
  }

  std::sort(lElements.begin(), lElements.end(), [vScoreLabel](const json & pA, const json & pB) {
    return fJsonValue(pA, vScoreLabel, 0.0) > fJsonValue(pB, vScoreLabel, 0.0);
  });

  lElements.resize(std::min(pTopN, (int) lElements.size()));
  json vResults = lElements;

  if (pIsTeiFormat) {
    return vResults;
  }

  json vRes = json{
    { "model", fJsonValue(pRequest, "model", pModelName) },
    { "object", "list" },
    { "usage", json{ { "prompt_tokens", vNTokens }, { "total_tokens", vNTokens } } },
    { "results", vResults }
  };

  return vRes;
}

//
// other utils
//

std::vector<dgs_token_data> fGetTokenProbabilities(dgs_context * pCtx, int pIdx, size_t pNTop) {
  std::vector<dgs_token_data> lCur;

  const auto *      vLogits     = fDgsGetLogitsIth(pCtx, pIdx);
  const dgs_token * vSampledIds = fDgsGetSampledCandidatesIth(pCtx, pIdx);

  const int cNLogits = fDgsGetSampledLogitsCountIth(pCtx, pIdx);

  lCur.resize(cNLogits);
  if (vSampledIds) {
    for (int vI = 0; vI < cNLogits; vI++) {
      lCur[vI] = dgs_token_data{ vSampledIds[vI], vLogits[vI], 0.0f };
    }
  } else {
    for (dgs_token vTokenId = 0; vTokenId < cNLogits; vTokenId++) {
      lCur[vTokenId] = dgs_token_data{ vTokenId, vLogits[vTokenId], 0.0f };
    }
  }

  // sort tokens by logits (partial: only the leading `n_top` need ordering)
  if (pNTop > lCur.size()) {
    pNTop = lCur.size();
  }
  if (pNTop > 0) {
    std::partial_sort(lCur.begin(), lCur.begin() + pNTop, lCur.end(),
                      [](const dgs_token_data & pA, const dgs_token_data & pB) { return pA.vLogit > pB.vLogit; });
  }

  // apply softmax
  float vMaxL = -std::numeric_limits<float>::infinity();
  if (pNTop > 0) {
    vMaxL = lCur[0].vLogit;  // partial_sort guarantees the absolute maximum is at index 0
  } else {
    for (const auto & vT : lCur) {
      vMaxL = std::max(vMaxL, vT.vLogit);
    }
  }
  float vCumSum = 0.0f;
  for (auto & vT : lCur) {
    float vP = expf(vT.vLogit - vMaxL);
    vT.vP    = vP;
    vCumSum += vP;
  }
  for (auto & vT : lCur) {
    vT.vP /= vCumSum;
  }

  return lCur;
}

std::string fSafeJsonToStr(const json & data) {
  return data.mDumpSafe();
}

// TODO: reuse dgs_detokenize
template <class Iter> static std::string fTokensToStr(const dgs_vocab * pCtx, Iter begin, Iter end) {
  std::string vRet;
  for (; begin != end; ++begin) {
    vRet += fCommonTokenToPiece(pCtx, *begin);
  }

  return vRet;
}

std::string fTokensToStr(dgs_context * pCtx, const dgs_tokens & pTokens) {
  auto vModel = fDgsGetModel(pCtx);
  return fTokensToStr(fDgsModelGetVocab(vModel), pTokens.begin(), pTokens.end());
}

std::string fTokensToStr(const dgs_vocab * pVocab, const dgs_tokens & pTokens) {
  return fTokensToStr(pVocab, pTokens.begin(), pTokens.end());
}

// format incomplete utf-8 multibyte character for output
std::string fTokensToOutputFormattedString(const dgs_context * pCtx, const dgs_token token) {
  std::string vOut = token == DGS_TOKEN_NULL ? "" : fCommonTokenToPiece(pCtx, token);

  // if the size is 1 and first bit is 1, meaning it's a partial character
  //   (size > 1 meaning it's already a known token)
  if (vOut.size() == 1 && (vOut[0] & 0x80) == 0x80) {
    std::stringstream vSs;
    vSs << std::hex << (vOut[0] & 0xff);
    std::string vRes(vSs.str());
    vOut = "byte: \\x" + vRes;
  }

  return vOut;
}

// format server-sent event (SSE), return the formatted string to send
// note: if data is a json array, it will be sent as multiple events, one per item
std::string fFormatOaiSse(const json & data) {
  std::ostringstream vSs;
  auto               vSendSingle = [&vSs](const json & data) {
    vSs << "data: " << fSafeJsonToStr(data)
        << "\n\n";  // required by RFC 8895 - A message is terminated by a blank line (two line terminators in a row).
  };

  if (data.is_array()) {
    for (const auto & vItem : data) {
      vSendSingle(vItem);
    }
  } else {
    vSendSingle(data);
  }

  return vSs.str();
}

std::string fFormatOaiRespSse(const json & data) {
  std::ostringstream vSs;
  auto               vSendSingle = [&vSs](const json & pEventObj) {
    vSs << "event: " << pEventObj.at("event").get<std::string>() << "\n";
    vSs << "data: " << fSafeJsonToStr(pEventObj.at("data")) << "\n\n";
  };

  if (data.is_array()) {
    for (const auto & vItem : data) {
      vSendSingle(vItem);
    }
  } else {
    vSendSingle(data);
  }

  return vSs.str();
}

std::string fFormatAnthropicSse(const json & data) {
  std::ostringstream vSs;

  auto vSendEvent = [&vSs](const json & pEventObj) {
    if (pEventObj.contains("event") && pEventObj.contains("data")) {
      vSs << "event: " << pEventObj.at("event").get<std::string>() << "\n";
      vSs << "data: " << fSafeJsonToStr(pEventObj.at("data")) << "\n\n";
    } else {
      vSs << "data: " << fSafeJsonToStr(pEventObj) << "\n\n";
    }
  };

  if (data.is_array()) {
    for (const auto & vEvent : data) {
      vSendEvent(vEvent);
    }
  } else {
    vSendEvent(data);
  }

  return vSs.str();
}

bool fIsValidUtf8(const std::string & str) {
  const unsigned char * vBytes = reinterpret_cast<const unsigned char *>(str.data());
  const unsigned char * end    = vBytes + str.length();

  while (vBytes < end) {
    if (*vBytes <= 0x7F) {
      // 1-byte sequence (0xxxxxxx)
      vBytes++;
    } else if ((*vBytes & 0xE0) == 0xC0) {
      // 2-byte sequence (110xxxxx 10xxxxxx)
      if (end - vBytes < 2 || (vBytes[1] & 0xC0) != 0x80) {
        return false;
      }
      vBytes += 2;
    } else if ((*vBytes & 0xF0) == 0xE0) {
      // 3-byte sequence (1110xxxx 10xxxxxx 10xxxxxx)
      if (end - vBytes < 3 || (vBytes[1] & 0xC0) != 0x80 || (vBytes[2] & 0xC0) != 0x80) {
        return false;
      }
      vBytes += 3;
    } else if ((*vBytes & 0xF8) == 0xF0) {
      // 4-byte sequence (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
      if (end - vBytes < 4 || (vBytes[1] & 0xC0) != 0x80 || (vBytes[2] & 0xC0) != 0x80 || (vBytes[3] & 0xC0) != 0x80) {
        return false;
      }
      vBytes += 4;
    } else {
      // Invalid UTF-8 lead byte
      return false;
    }
  }

  return true;
}

dgs_tokens fFormatPromptInfill(const dgs_vocab *  pVocab,
                               const json &       pInputPrefix,
                               const json &       pInputSuffix,
                               const json &       pInputExtra,
                               const int          pNBatch,
                               const int          pNPredict,
                               const int          pNCtx,
                               const bool         pSpmInfill,
                               const dgs_tokens & pTokensPrompt) {
  // TODO: optimize this block by reducing memory allocations and movement

  // use FIM repo-level pattern:
  // ref: https://arxiv.org/pdf/2409.12186
  //
  // [FIM_REP]myproject
  // [FIM_SEP]filename0
  // extra chunk 0
  // [FIM_SEP]filename1
  // extra chunk 1
  // ...
  // [FIM_SEP]filename
  // [FIM_PRE]prefix[FIM_SUF]suffix[FIM_MID]prompt
  //
  dgs_tokens lExtraTokens;
  lExtraTokens.reserve(pNCtx);

  auto lTokensPrefix = fTokenizeMixed(pVocab, pInputPrefix, false, false);
  auto lTokensSuffix = fTokenizeMixed(pVocab, pInputSuffix, false, false);

  if (fDgsVocabFimRep(pVocab) != DGS_TOKEN_NULL) {
    // TODO: make project name an input
    static const auto lKFimRepo = fCommonTokenize(pVocab, "myproject\n", false, false);

    lExtraTokens.push_back(fDgsVocabFimRep(pVocab));
    lExtraTokens.insert(lExtraTokens.end(), lKFimRepo.begin(), lKFimRepo.end());
  }
  for (const auto & vChunk : pInputExtra) {
    // { "text": string, "filename": string }
    const std::string cText     = fJsonValue(vChunk, "text", std::string());
    const std::string cFilename = fJsonValue(vChunk, "filename", std::string("tmp"));

    if (fDgsVocabFimSep(pVocab) != DGS_TOKEN_NULL) {
      const auto lKFimFile = fCommonTokenize(pVocab, cFilename + "\n", false, false);

      lExtraTokens.insert(lExtraTokens.end(), fDgsVocabFimSep(pVocab));
      lExtraTokens.insert(lExtraTokens.end(), lKFimFile.begin(), lKFimFile.end());
    } else {
      // chunk separator in binary form to avoid confusing the AI
      static const char aKChunkPrefixStr[]  = { 0x0a, 0x0a, 0x2d, 0x2d, 0x2d, 0x20, 0x73, 0x6e, 0x69, 0x70,
                                                0x70, 0x65, 0x74, 0x20, 0x2d, 0x2d, 0x2d, 0x0a, 0x0a, 0x00 };
      static const auto lKChunkPrefixTokens = fCommonTokenize(pVocab, aKChunkPrefixStr, false, false);

      lExtraTokens.insert(lExtraTokens.end(), lKChunkPrefixTokens.begin(), lKChunkPrefixTokens.end());
    }

    const auto lChunkTokens = fCommonTokenize(pVocab, cText, false, false);
    lExtraTokens.insert(lExtraTokens.end(), lChunkTokens.begin(), lChunkTokens.end());
  }

  if (fDgsVocabFimSep(pVocab) != DGS_TOKEN_NULL) {
    // TODO: current filename
    static const auto lKFimFile = fCommonTokenize(pVocab, "filename\n", false, false);

    lExtraTokens.insert(lExtraTokens.end(), fDgsVocabFimSep(pVocab));
    lExtraTokens.insert(lExtraTokens.end(), lKFimFile.begin(), lKFimFile.end());
  }

  // for now pick FIM context to fit in a batch (ratio prefix:suffix = 3:1, TODO: configurable?)
  const int cNPrefixTake = std::min<int>(lTokensPrefix.size(), 3 * (pNBatch / 4));
  const int cNSuffixTake =
    std::min<int>(lTokensSuffix.size(), std::max<int>(0, (pNBatch / 4) - (2 + pTokensPrompt.size())));

  SRV_DBG("n_prefix_take = %d, n_suffix_take = %d, total = %d\n", cNPrefixTake, cNSuffixTake,
          (cNPrefixTake + cNSuffixTake));

  // fill the rest of the context with extra chunks
  const int cNExtraTake = std::min<int>(std::max<int>(0, pNCtx - (pNBatch) -2 * pNPredict), lExtraTokens.size());

  lTokensPrefix.erase(lTokensPrefix.begin(), lTokensPrefix.begin() + lTokensPrefix.size() - cNPrefixTake);
  lTokensSuffix.resize(cNSuffixTake);

  lTokensPrefix.insert(lTokensPrefix.begin(), fDgsVocabFimPre(pVocab));
  lTokensPrefix.insert(lTokensPrefix.end(), pTokensPrompt.begin(), pTokensPrompt.end());
  lTokensSuffix.insert(lTokensSuffix.begin(), fDgsVocabFimSuf(pVocab));

  auto lEmbdInp = pSpmInfill ? lTokensSuffix : lTokensPrefix;
  auto lEmbdEnd = pSpmInfill ? lTokensPrefix : lTokensSuffix;

  if (fDgsVocabGetAddBos(pVocab)) {
    lEmbdInp.insert(lEmbdInp.begin(), fDgsVocabBos(pVocab));
  }

  SRV_DBG("extra: n_ctx = %d, n_extra_take = %d, n_extra = %d\n", pNCtx, cNExtraTake, (int) lExtraTokens.size());

  // put the extra context before the FIM prefix
  lEmbdInp.insert(lEmbdInp.begin(), lExtraTokens.end() - cNExtraTake, lExtraTokens.end());

  lEmbdInp.insert(lEmbdInp.end(), lEmbdEnd.begin(), lEmbdEnd.end());
  lEmbdInp.push_back(fDgsVocabFimMid(pVocab));

  return lEmbdInp;
}

server_tokens fFormatPromptRerank(const struct dgs_model *     pModel,
                                  const struct dgs_vocab *     pVocab,
                                  mtmd_context *               pMctx,
                                  const std::string &          pQuery,
                                  const std::string &          pDoc,
                                  const mtmd_helper_init_opt & pInitOpt) {
  server_tokens vResult = {};

  const char * vRerankPrompt = fDgsModelChatTemplate(pModel, "rerank");

  if (vRerankPrompt != nullptr) {
    std::string vPrompt = vRerankPrompt;
    fStringReplaceAll(vPrompt, "{query}", pQuery);
    fStringReplaceAll(vPrompt, "{document}", pDoc);
    server_tokens vTokens = fTokenizeInputSubprompt(pVocab, pMctx, vPrompt, false, true, pInitOpt);
    vResult.push_back(vTokens);
  } else {
    // Get EOS token - use SEP token as fallback if EOS is not available
    server_tokens vQueryTokens = fTokenizeInputSubprompt(pVocab, pMctx, pQuery, false, false, pInitOpt);
    server_tokens vDocTokens   = fTokenizeInputSubprompt(pVocab, pMctx, pDoc, false, false, pInitOpt);
    dgs_token     vEosToken    = fDgsVocabEos(pVocab);
    if (vEosToken == DGS_TOKEN_NULL) {
      vEosToken = fDgsVocabSep(pVocab);
    }

    if (fDgsVocabGetAddBos(pVocab)) {
      vResult.push_back(fDgsVocabBos(pVocab));
    }
    vResult.push_back(vQueryTokens);
    if (fDgsVocabGetAddEos(pVocab)) {
      vResult.push_back(vEosToken);
    }
    if (fDgsVocabGetAddSep(pVocab)) {
      vResult.push_back(fDgsVocabSep(pVocab));
    }
    vResult.push_back(vDocTokens);
    if (fDgsVocabGetAddEos(pVocab)) {
      vResult.push_back(vEosToken);
    }
  }

  return vResult;
}

//
// server_subproc
//

FILE * fServerReserveStdout() {
  fflush(stdout);
  // the reserved stream is not inherited, grandchildren get stdout and stderr of their own
#ifdef _WIN32
  int fd = _dup(_fileno(stdout));
  GGML_ASSERT(fd >= 0);
  SetHandleInformation((HANDLE) _get_osfhandle(fd), HANDLE_FLAG_INHERIT, 0);
  _dup2(_fileno(stderr), _fileno(stdout));
  SetStdHandle(STD_OUTPUT_HANDLE, GetStdHandle(STD_ERROR_HANDLE));
  FILE * f = _fdopen(fd, "w");
#else
  int vFd = fcntl(fileno(stdout), F_DUPFD_CLOEXEC, 0);
  GGML_ASSERT(vFd >= 0);
  dup2(fileno(stderr), fileno(stdout));
  FILE * vF = fdopen(vFd, "w");
#endif
  GGML_ASSERT(vF);
  return vF;
}

bool server_subproc::mHasOutput() {
  FILE * aFiles[cServerSubprocStreams] = { vSproc.mStdoutFile(), vSproc.mStderrFile() };
  for (int vI = 0; vI < cServerSubprocStreams; vI++) {
    if (aOutHandles[vI] >= 0) {
      continue;
    }
    if (!aFiles[vI]) {
      return false;
    }
#ifdef _WIN32
    HANDLE h = (HANDLE) _get_osfhandle(_fileno(files[i]));
    if (h == INVALID_HANDLE_VALUE) {
      return false;
    }
    out_handles[i] = (intptr_t) h;
#else
    int vFd = fileno(aFiles[vI]);
    if (vFd < 0) {
      return false;
    }
    fcntl(vFd, F_SETFL, fcntl(vFd, F_GETFL, 0) | O_NONBLOCK);
    aOutHandles[vI] = vFd;
#endif
  }
  return true;
}

bool server_subproc::mOutputClosed() const {
  return aOutClosed[cServerSubprocStdout] && aOutClosed[cServerSubprocStderr];
}

int server_subproc::mReadOutput(server_subproc_stream stream, char * pBuf, size_t pLen) {
  if (aOutClosed[stream]) {
    return -1;
  }
  if (!mHasOutput()) {
    aOutClosed[stream] = true;
    return -1;
  }
#ifdef _WIN32
  HANDLE h     = (HANDLE) out_handles[stream];
  DWORD  avail = 0;
  if (!PeekNamedPipe(h, NULL, 0, NULL, &avail, NULL)) {
    out_closed[stream] = true;  // pipe broken, child gone
    return -1;
  }
  if (avail == 0) {
    return 0;
  }
  DWORD to_read = avail < (DWORD) len ? avail : (DWORD) len;
  DWORD got     = 0;
  if (!ReadFile(h, buf, to_read, &got, NULL) || got == 0) {
    out_closed[stream] = true;
    return -1;
  }
  return (int) got;
#else
  while (true) {
    ssize_t vR = read((int) aOutHandles[stream], pBuf, pLen);
    if (vR > 0) {
      return (int) vR;
    }
    if (vR < 0 && errno == EINTR) {
      continue;
    }
    if (vR < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      return 0;
    }
    aOutClosed[stream] = true;  // EOF or error
    return -1;
  }
#endif
}

server_subproc::waiter::waiter() {
#ifndef _WIN32
  int aFds[2];
  GGML_ASSERT(pipe(aFds) == 0);
  for (int vFd : aFds) {
    fcntl(vFd, F_SETFL, fcntl(vFd, F_GETFL, 0) | O_NONBLOCK);
  }
  aWakeFd[0] = aFds[0];
  aWakeFd[1] = aFds[1];
#endif
}

server_subproc::waiter::~waiter() {
#ifndef _WIN32
  close((int) aWakeFd[0]);
  close((int) aWakeFd[1]);
#endif
}

void server_subproc::waiter::mWake() {
#ifndef _WIN32
  char vC = 1;
  (void) !write((int) aWakeFd[1], &vC, 1);
#endif
}

void server_subproc::waiter::mWait(const std::vector<server_subproc *> & pProcs,
                                   std::vector<bool> &                   pReady,
                                   int64_t                               pTimeoutMs) {
  pReady.assign(pProcs.size(), false);
#ifdef _WIN32
  // no waitable wait exists for anonymous pipes, so poll them in 50 ms steps
  bool any = false;
  for (size_t i = 0; i < procs.size(); i++) {
    server_subproc * p = procs[i];
    ready[i]           = !p->has_output();
    for (int s = 0; s < cServerSubprocStreams && !ready[i]; s++) {
      DWORD avail = 0;
      if (!p->out_closed[s] && (!PeekNamedPipe((HANDLE) p->out_handles[s], NULL, 0, NULL, &avail, NULL) || avail > 0)) {
        ready[i] = true;  // data or broken pipe, read_output() tells which
      }
    }
    any = any || ready[i];
  }
  if (!any) {
    int64_t step = timeout_ms < 0 ? 50 : std::min<int64_t>(timeout_ms, 50);
    std::this_thread::sleep_for(std::chrono::milliseconds(step));
  }
#else
  std::vector<pollfd> lPfds;
  lPfds.reserve(pProcs.size() * cServerSubprocStreams + 1);
  lPfds.push_back({ (int) aWakeFd[0], POLLIN, 0 });
  for (auto * vP : pProcs) {
    const bool cOpen = vP->mHasOutput();
    for (int vS = 0; vS < cServerSubprocStreams; vS++) {
      lPfds.push_back(
        { cOpen && !vP->aOutClosed[vS] ? (int) vP->aOutHandles[vS] : -1, POLLIN, 0 });  // poll() skips negative fds
    }
  }
  int vTimeout = pTimeoutMs < 0 ? -1 : (int) std::min<int64_t>(pTimeoutMs, std::numeric_limits<int>::max());
  int vR       = poll(lPfds.data(), lPfds.size(), vTimeout);
  if (vR < 0 && errno != EINTR) {
    LOG_ERR("%s: poll() failed: %s\n", __func__, strerror(errno));
  }
  if (lPfds[0].revents) {
    char aBuf[64];
    while (read((int) aWakeFd[0], aBuf, sizeof(aBuf)) > 0) {
    }
  }
  for (size_t vI = 0; vI < pProcs.size(); vI++) {
    pReady[vI] = !pProcs[vI]->mHasOutput();
    for (int vS = 0; vS < cServerSubprocStreams; vS++) {
      pReady[vI] = pReady[vI] || lPfds[1 + vI * cServerSubprocStreams + vS].revents != 0;
    }
  }
#endif
}
