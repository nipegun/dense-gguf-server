#include "dgs-model-policy.h"

#include "dgs-hparams.h"
#include "dgs-model-loader.h"

#include <array>
#include <cctype>
#include <stdexcept>
#include <string>

namespace {
constexpr std::array<dgs_model_profile, 4> aProfiles = {
  {
   { "qwen3.8-27b", 64, 5120, 17408, 24, 4 },
   { "qwen3.5-9b", 32, 4096, 12288, 16, 4 },
   { "qwen3.5-4b", 32, 2560, 9216, 16, 4 },
   { "qwen3.5-2b", 24, 2048, 6144, 8, 2 },
   }
};

std::string fMetadataString(const gguf_context * pMetadata, const char * pKey) {
  const auto cIndex = gguf_find_key(pMetadata, pKey);
  if (cIndex < 0 || gguf_get_kv_type(pMetadata, cIndex) != GGUF_TYPE_STRING) {
    return {};
  }
  return gguf_get_val_str(pMetadata, cIndex);
}

std::string fNormalizeName(const std::string & pName) {
  std::string vResult;
  for (const unsigned char vChar : pName) {
    if (std::isspace(vChar) || vChar == '_') {
      if (!vResult.empty() && vResult.back() != '-') {
        vResult += '-';
      }
    } else {
      vResult += static_cast<char>(std::tolower(vChar));
    }
  }
  return vResult;
}

bool fNamesProfile(const std::string & pName, const char * pProfile) {
  const auto cPosition = pName.find(pProfile);
  if (cPosition == std::string::npos) {
    return false;
  }
  const auto cEnd = cPosition + std::char_traits<char>::length(pProfile);
  return (cPosition == 0 || !std::isalnum(static_cast<unsigned char>(pName[cPosition - 1]))) &&
         (cEnd == pName.size() || !std::isalnum(static_cast<unsigned char>(pName[cEnd])));
}
}  // namespace

const dgs_model_profile & fDgsRequireModelProfile(const dgs_model_loader & pLoader) {
  if (pLoader.mGetArch() != cLlmArchQwen35 || gguf_get_version(pLoader.vMetadata) != GGUF_VERSION) {
    throw std::runtime_error("DGS requires current GGUF qwen35 models: Qwen3.8-27B or Qwen3.5-9B/4B/2B");
  }
  const auto cName         = fNormalizeName(fMetadataString(pLoader.vMetadata, "general.name"));
  const auto cBase         = fNormalizeName(fMetadataString(pLoader.vMetadata, "general.basename") + "-" +
                                            fMetadataString(pLoader.vMetadata, "general.size_label"));
  const auto cFamilyStart  = cName.find("qwen");
  size_t     vVersionStart = cFamilyStart == std::string::npos ? cName.size() : cFamilyStart + 4;
  while (vVersionStart < cName.size() && cName[vVersionStart] == '-') {
    ++vVersionStart;
  }
  const bool cExplicitVersion =
    vVersionStart < cName.size() && std::isdigit(static_cast<unsigned char>(cName[vVersionStart]));
  for (const auto & vProfile : aProfiles) {
    // An explicitly named version must not be reinterpreted using a different basename.
    if (fNamesProfile(cName, vProfile.vName) || (!cExplicitVersion && fNamesProfile(cBase, vProfile.vName))) {
      return vProfile;
    }
  }
  throw std::runtime_error(
    "Model is not allowed: GGUF general.name or general.basename/general.size_label must identify Qwen3.8-27B or "
    "Qwen3.5-9B/4B/2B");
}

void fDgsValidateModelProfile(const dgs_model_loader & pLoader, const dgs_hparams & pHparams) {
  const auto & cProfile = fDgsRequireModelProfile(pLoader);
  if (pHparams.n_layer() != cProfile.vLayers || pHparams.n_embd != cProfile.vEmbedding) {
    throw std::runtime_error("GGUF dimensions do not match the allowed Qwen model profile");
  }
  for (uint32_t vLayer = 0; vLayer < pHparams.n_layer(); ++vLayer) {
    if (pHparams.n_ff(vLayer) != cProfile.vFeedForward ||
        (!pHparams.mIsRecr(vLayer) &&
         (pHparams.n_head(vLayer) != cProfile.vHeads || pHparams.n_head_kv(vLayer) != cProfile.vKvHeads))) {
      throw std::runtime_error("GGUF layer dimensions do not match the allowed Qwen model profile");
    }
  }
}
