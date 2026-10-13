#include "server-decision.h"

#include "../../src/dgs-ext.h"  // staging API: dgs_decision_order

#include <algorithm>
#include <cctype>
#include <cmath>
#include <regex>
#include <stdexcept>

static const char * fDecisionQuestionTypeName(server_decision_question_type type) {
  switch (type) {
    case cServerDecisionQuestionChoice:
      return "choice";
    case cServerDecisionQuestionScore:
      return "score";
    case cServerDecisionQuestionNoul:
      return "noul";
  }
  return "";
}

// lev reads noul from a rating scale: 0 = certainly no, 8 = certainly yes
static const size_t cDecisionLevNRatings = 9;

static std::string fDecisionMetaStr(const dgs_model * pModel, const std::string & pKey) {
  char          aBuf[256];
  const int32_t n = fDgsModelMetaValStr(pModel, pKey.c_str(), aBuf, sizeof(aBuf));
  return n < 0 ? "" : std::string(aBuf);
}

//
// model-specific setup
//

void server_decision_context::init(const dgs_model * pModel) {
  *this = server_decision_context();  // the model can be reloaded

  const common_decision_type cModelType = fCommonGetDecisionType(pModel);
  if (cModelType == cCommonDecisionTypeNone) {
    return;
  }

  const std::string cPrefix   = fDecisionMetaStr(pModel, "general.architecture") + ".decision.";
  const std::string cTypeName = fDecisionMetaStr(pModel, cPrefix + "type");

  vVocab = fDgsModelGetVocab(pModel);

  const char * vTmplSrc = fDgsModelChatTemplate(pModel, "systemone");
  if (vTmplSrc == nullptr) {
    throw std::runtime_error("decision model has no \"systemone\" template");
  }
  vTmpl = std::make_shared<const common_chat_template>(vTmplSrc, "", "");

  const std::string cPrefixTemp = cPrefix + "temperature.";
  for (int32_t vI = 0; vI < fDgsModelMetaCount(pModel); vI++) {
    char aKey[256];
    char aVal[64];
    if (fDgsModelMetaKeyByIndex(pModel, vI, aKey, sizeof(aKey)) < 0 || !fStringStartsWith(aKey, cPrefixTemp)) {
      continue;
    }
    if (fDgsModelMetaValStrByIndex(pModel, vI, aVal, sizeof(aVal)) < 0) {
      continue;
    }
    const float cTemp = std::strtof(aVal, nullptr);
    if (cTemp <= 0.0f) {
      throw std::runtime_error(fStringFormat("invalid decision temperature: %s = %s", aKey, aVal));
    }
    dTemperatures[aKey + cPrefixTemp.size()] = cTemp;
  }

  // "<n_levels>.<level>" -> offset
  const std::string cPrefixBias = cPrefix + "score_bias.";
  for (int32_t vI = 0; vI < fDgsModelMetaCount(pModel); vI++) {
    char aKey[256];
    char aVal[64];
    if (fDgsModelMetaKeyByIndex(pModel, vI, aKey, sizeof(aKey)) < 0 || !fStringStartsWith(aKey, cPrefixBias)) {
      continue;
    }
    if (fDgsModelMetaValStrByIndex(pModel, vI, aVal, sizeof(aVal)) < 0) {
      continue;
    }
    const auto   lParts = fStringSplit<std::string>(aKey + cPrefixBias.size(), '.');
    const size_t n      = lParts.size() == 2 ? std::strtoul(lParts[0].c_str(), nullptr, 10) : 0;
    const size_t cLevel = lParts.size() == 2 ? std::strtoul(lParts[1].c_str(), nullptr, 10) : 0;
    if (n < 2 || n > 255 || cLevel >= n) {
      throw std::runtime_error(fStringFormat("invalid decision score bias: %s", aKey));
    }
    auto & lBias = dScoreBias[n];
    lBias.resize(n, 0.0f);
    lBias[cLevel] = std::strtof(aVal, nullptr);
  }

  if (cModelType == cCommonDecisionTypeOpenjev) {
    // one letter per option, each must be a single token
    const std::string cLetters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    for (const char cC : cLetters) {
      const auto lToks = fCommonTokenize(vVocab, std::string(1, cC), false, false);
      if (lToks.size() != 1) {
        throw std::runtime_error(fStringFormat("decision label '%c' is not a single token", cC));
      }
      lLabels.push_back(lToks[0]);
    }
    vNOptionsMax   = lLabels.size();
    vNoulTrueFirst = true;
  } else if (cModelType == cCommonDecisionTypeLev || cModelType == cCommonDecisionTypeNimble ||
             cModelType == cCommonDecisionTypePplxDecider || cModelType == cCommonDecisionTypeVllmD3) {
    // label codes are A..Z then AA..ZZ, only the ones that are a single token are used
    std::vector<std::string> lCodes;
    for (char vA = 'A'; vA <= 'Z'; vA++) {
      lCodes.push_back(std::string(1, vA));
    }
    for (char vA = 'A'; vA <= 'Z'; vA++) {
      for (char vB = 'A'; vB <= 'Z'; vB++) {
        lCodes.push_back(std::string{ vA, vB });
      }
    }
    for (const auto & vCode : lCodes) {
      const auto lToks = fCommonTokenize(vVocab, vCode, false, false);
      if (lToks.size() == 1 && lLabels.size() < 255) {
        lLabels.push_back(lToks[0]);
        lLabelTexts.push_back(vCode);
      }
    }
    vNOptionsMax = lLabels.size();
    if (cModelType == cCommonDecisionTypeVllmD3) {
      vCanonicalLegend = true;
    }
  } else if (cModelType == cCommonDecisionTypeKev) {
    // the hidden state of an option is read at the token that ends it
    const auto lToks = fCommonTokenize(vVocab, "<|box_end|>", false, true);
    if (lToks.size() != 1) {
      throw std::runtime_error("decision model has no <|box_end|> token");
    }
    vTokenMarker = lToks[0];
    vNOptionsMax = 255;
  } else if (cModelType == cCommonDecisionTypeLaya) {
    vTokenMarker = fDgsVocabMask(vVocab);
    vTokenSep    = fDgsVocabSep(vVocab);
    if (vTokenMarker == DGS_TOKEN_NULL || vTokenSep == DGS_TOKEN_NULL) {
      throw std::runtime_error("decision model has no mask or sep token");
    }
    vTextMarker = fCommonTokenToPiece(vVocab, vTokenMarker, true);

    const std::string cVal = fDecisionMetaStr(pModel, cPrefix + "max_head_tokens");
    vMaxHeadTokens         = std::strtoul(cVal.c_str(), nullptr, 10);
    if (vMaxHeadTokens == 0) {
      throw std::runtime_error("decision model has no valid max_head_tokens");
    }
    vNOptionsMax = 255;
  } else if (cModelType == cCommonDecisionTypeClef) {
    vNOptionsMax   = 255;
    vNoulTrueFirst = true;
    vChoiceSorted  = true;
  } else if (cModelType == cCommonDecisionTypeVllmDecision2) {
    vNOptionsMax       = 255;
    vNoulCriteriaOrder = true;
    vStrictPayload     = true;
    vCanonicalLegend   = true;
  } else if (cModelType == cCommonDecisionTypeLfm2D1) {
    vNOptionsMax   = 255;
    vNoulTrueFirst = true;
  } else if (cModelType == cCommonDecisionTypeLfm2D1Omni) {
    vTokenMarker = fDgsVocabMask(vVocab);
    if (vTokenMarker == DGS_TOKEN_NULL) {
      throw std::runtime_error("decision model has no mask token");
    }
    vNOptionsMax = 255;
  } else {
    throw std::runtime_error("unsupported decision model type: " + cTypeName);
  }
  type = cModelType;

  SRV_INF("decision model type: %s\n", cTypeName.c_str());
}

//
// request parsing
//

static bool fDecisionIsPayload(const json & pVal) {
  return pVal.is_string() || pVal.is_object() || pVal.is_array();
}

std::vector<server_decision_question> server_decision_context::mParseQuestions(const json & pBody) const {
  // lfm2-d1, lfm2-d1-omni and vllm-d3 accept a null state, for example to ask about images only
  if (!pBody.contains("state") || (pBody.at("state").is_null() && type != cCommonDecisionTypeLfm2D1 &&
                                   type != cCommonDecisionTypeLfm2D1Omni && type != cCommonDecisionTypeVllmD3)) {
    throw std::invalid_argument("\"state\" must be provided");
  }
  if (vStrictPayload && !fDecisionIsPayload(pBody.at("state"))) {
    throw std::invalid_argument("\"state\" must be text, an object or an array");
  }
  if (!pBody.contains("questions") || !pBody.at("questions").is_object() || pBody.at("questions").empty()) {
    throw std::invalid_argument("\"questions\" must be a non-empty object");
  }

  std::vector<server_decision_question> lQuestions;
  for (const auto & [id, q] : pBody.at("questions").mItems()) {
    auto vErr = [&vId = id](const std::string & pMsg) {
      return std::invalid_argument("questions." + vId + ": " + pMsg);
    };
    if (!q.is_object()) {
      throw vErr("must be an object");
    }
    // the vllm-d3 prompt has a default question
    const json cInstructions = q.contains("instructions") ? q.at("instructions") : json();
    if (cInstructions.is_null() && type != cCommonDecisionTypeVllmD3) {
      throw vErr("\"instructions\" must be provided");
    }
    if (vStrictPayload && (!fDecisionIsPayload(cInstructions) || cInstructions == "")) {
      throw vErr("\"instructions\" must be non-empty text, an object or an array");
    }

    server_decision_question vQuestion;
    vQuestion.vId           = id;
    vQuestion.vInstructions = cInstructions;

    const std::string cTypeName = fJsonValue(q, "type", std::string());
    const json        cCriteria = q.contains("criteria") ? q.at("criteria") : json();

    if (cTypeName == "choice") {
      vQuestion.type = cServerDecisionQuestionChoice;
      if (!cCriteria.is_object() || cCriteria.empty()) {
        throw vErr("\"criteria\" must be a non-empty object");
      }
      for (const auto & [key, description] : cCriteria.mItems()) {
        if (vStrictPayload && (key.empty() || !(description.is_null() || fDecisionIsPayload(description)))) {
          throw vErr("option keys must be non-empty, descriptions must be null, text, an object or an array");
        }
        vQuestion.lOptions.push_back({ key, description });
      }
      if (vStrictPayload && vQuestion.lOptions.size() < 2) {
        throw vErr("\"criteria\" must have at least 2 options");
      }
      if (vChoiceSorted) {
        std::sort(vQuestion.lOptions.begin(), vQuestion.lOptions.end(),
                  [](const auto & pA, const auto & pB) { return pA.vKey < pB.vKey; });
      }
    } else if (cTypeName == "score") {
      vQuestion.type = cServerDecisionQuestionScore;
      if (!cCriteria.is_array() || cCriteria.size() < 2 || cCriteria.size() > 10) {
        throw vErr("\"criteria\" must be an array of 2 to 10 levels");
      }
      for (size_t vI = 0; vI < cCriteria.size(); vI++) {
        if (vStrictPayload && !fDecisionIsPayload(cCriteria.at(vI))) {
          throw vErr("level descriptions must be text, an object or an array");
        }
        vQuestion.lOptions.push_back({ std::to_string(vI), cCriteria.at(vI) });
      }
    } else if (cTypeName == "noul") {
      vQuestion.type = cServerDecisionQuestionNoul;
      if (!cCriteria.is_null() && !cCriteria.is_object()) {
        throw vErr("\"criteria\" must be an object");
      }
      if (vStrictPayload && cCriteria.is_object()) {
        for (const auto & [key, description] : cCriteria.mItems()) {
          if ((key != "false" && key != "true") || !fDecisionIsPayload(description)) {
            throw vErr("\"criteria\" may only describe \"false\" and \"true\" with text, an object or an array");
          }
        }
      }
      for (const char * vKey : { "false", "true" }) {
        json vDescription;
        if (cCriteria.is_object() && cCriteria.contains(vKey)) {
          vDescription = cCriteria.at(vKey);
        } else if (cCriteria.is_object() && type == cCommonDecisionTypeLfm2D1Omni) {
          // lfm2-d1-omni also reads the descriptions under "no" and "yes"
          const char * vAlias = std::string(vKey) == "true" ? "yes" : "no";
          vDescription        = cCriteria.contains(vAlias) ? cCriteria.at(vAlias) : json();
        }
        vQuestion.lOptions.push_back({ vKey, vDescription });
      }
      if (vNoulTrueFirst || (vNoulCriteriaOrder && cCriteria.is_object() && cCriteria.size() == 2 &&
                             cCriteria.begin().mKey() == "true")) {
        std::swap(vQuestion.lOptions[0], vQuestion.lOptions[1]);
      }
    } else {
      throw vErr("\"type\" must be one of: choice, score, noul");
    }

    if (vQuestion.lOptions.size() > vNOptionsMax) {
      throw vErr(fStringFormat("too many options (%zu), this model supports at most %zu", vQuestion.lOptions.size(),
                               vNOptionsMax));
    }

    lQuestions.push_back(std::move(vQuestion));
  }
  return lQuestions;
}

//
// images
//

static const size_t cDecisionMaxImages = 8;

// any media, mtmd tells an audio clip from an image by its content
static void fDecisionLoadImage(const json & pUrl, std::vector<raw_buffer> & pFiles) {
  if (!pUrl.is_string() || !fStringStartsWith(pUrl.get<std::string>(), "data:")) {
    throw std::invalid_argument("images must be data URLs (data:image/...;base64,... or data:audio/...;base64,...)");
  }
  if (pFiles.size() >= cDecisionMaxImages) {
    throw std::invalid_argument(fStringFormat("too many images, the maximum is %zu", cDecisionMaxImages));
  }
  fHandleMedia(pFiles, pUrl.get<std::string>(), "");
}

// audio is base64 data, as a data URL or not (OpenAI input_audio)
static void fDecisionLoadAudio(const json & data, std::vector<raw_buffer> & pFiles) {
  if (!data.is_string() || fStringStartsWith(data.get<std::string>(), "http") ||
      fStringStartsWith(data.get<std::string>(), "file://")) {
    throw std::invalid_argument("audio must be base64 data");
  }
  if (pFiles.size() >= cDecisionMaxImages) {
    throw std::invalid_argument(fStringFormat("too many media files, the maximum is %zu", cDecisionMaxImages));
  }
  fHandleMedia(pFiles, data.get<std::string>(), "");
}

json server_decision_context::mParseState(const json & pBody, std::vector<raw_buffer> & pFiles) const {
  if (pBody.contains("videos") && !pBody.at("videos").is_null() && !pBody.at("videos").empty()) {
    throw std::invalid_argument("\"videos\" is not supported");
  }
  // "images" is an alias of "files"
  for (const char * vKey : { "files", "images" }) {
    if (!pBody.contains(vKey) || pBody.at(vKey).is_null()) {
      continue;
    }
    if (!pBody.at(vKey).is_array()) {
      throw std::invalid_argument(fStringFormat("\"%s\" must be an array", vKey));
    }
    for (const auto & vUrl : pBody.at(vKey)) {
      fDecisionLoadImage(vUrl, pFiles);
    }
  }

  const json & vState     = pBody.at("state");
  const bool   cIsWrapped = vState.is_object() && vState.contains("messages");
  const json & vMessages  = cIsWrapped ? vState.at("messages") : vState;
  if (!vMessages.is_array()) {
    return vState;
  }

  // chat messages: take the image and audio parts out of the content
  json vMessagesOut = json::mArray();
  for (const auto & vMsg : vMessages) {
    if (!vMsg.is_object() || !vMsg.contains("content") || !vMsg.at("content").is_array()) {
      vMessagesOut.push_back(vMsg);
      continue;
    }
    json vContent = json::mArray();
    for (const auto & vPart : vMsg.at("content")) {
      if (vPart.is_object() && fJsonValue(vPart, "type", std::string()) == "image_url" && vPart.contains("image_url")) {
        const json & vImageUrl = vPart.at("image_url");
        fDecisionLoadImage(vImageUrl.is_object() && vImageUrl.contains("url") ? vImageUrl.at("url") : vImageUrl,
                           pFiles);
      } else if (vPart.is_object() && fJsonValue(vPart, "type", std::string()) == "input_audio" &&
                 vPart.contains("input_audio")) {
        const json cInputAudio = fJsonValue(vPart, "input_audio", json::mObject());
        fDecisionLoadAudio(
          cInputAudio.contains("data") ? cInputAudio.at("data") : fJsonValue(cInputAudio, "url", json()), pFiles);
      } else {
        vContent.push_back(vPart);
      }
    }
    json vMsgOut       = vMsg;
    vMsgOut["content"] = vContent;
    vMessagesOut.push_back(vMsgOut);
  }

  if (!cIsWrapped) {
    return vMessagesOut;
  }
  json vStateOut        = vState;
  vStateOut["messages"] = vMessagesOut;
  return vStateOut;
}

//
// prompt
//

// replace text in all strings of a JSON value
static json fDecisionReplaceText(const json & pVal, const std::string & pSearch, const std::string & pReplace) {
  if (pVal.is_string()) {
    std::string str = pVal.get<std::string>();
    fStringReplaceAll(str, pSearch, pReplace);
    return str;
  }
  if (pVal.is_array()) {
    json vOut = json::mArray();
    for (const auto & vItem : pVal) {
      vOut.push_back(fDecisionReplaceText(vItem, pSearch, pReplace));
    }
    return vOut;
  }
  if (pVal.is_object()) {
    json vOut = json::mObject();
    for (const auto & [key, item] : pVal.mItems()) {
      vOut[key] = fDecisionReplaceText(item, pSearch, pReplace);
    }
    return vOut;
  }
  return pVal;
}

// sort the keys of all objects of a JSON value
static json fDecisionSortKeys(const json & pVal) {
  if (pVal.is_array()) {
    json vOut = json::mArray();
    for (const auto & vItem : pVal) {
      vOut.push_back(fDecisionSortKeys(vItem));
    }
    return vOut;
  }
  if (pVal.is_object()) {
    std::map<std::string, json> dSorted;
    for (const auto & [key, item] : pVal.mItems()) {
      dSorted[key] = fDecisionSortKeys(item);
    }
    json vOut = json::mObject();
    for (const auto & [key, item] : dSorted) {
      vOut[key] = item;
    }
    return vOut;
  }
  return pVal;
}

// kev flattens a JSON value into text, the keys of an object are kept as labels (kev/api.py: render)
static std::string fDecisionKevRender(const json & pVal, int pIndent = 0) {
  const std::string cPad(2 * pIndent, ' ');
  if (pVal.is_null()) {
    return "";
  }
  if (pVal.is_string()) {
    return pVal.get<std::string>();
  }
  if (pVal.is_boolean()) {
    return pVal.get<bool>() ? "True" : "False";
  }
  if (pVal.is_array()) {
    std::string vOut;
    for (const auto & vItem : pVal) {
      const std::string cText = fDecisionKevRender(vItem, pIndent + 1);
      vOut += (vOut.empty() ? "" : "\n") + cPad + "- " +
              cText.substr(std::min(cText.size(), cText.find_first_not_of(" \t\n\r")));
    }
    return vOut;
  }
  if (pVal.is_object()) {
    std::string vOut;
    for (const auto & [key, item] : pVal.mItems()) {
      const bool cIsNested = item.is_object() || item.is_array();
      vOut += (vOut.empty() ? "" : "\n") + cPad + key + (cIsNested ? ":\n" : ": ") +
              fDecisionKevRender(item, cIsNested ? pIndent + 1 : 0);
    }
    return vOut;
  }
  return pVal.mDump();
}

// kev text input: special tokens written in the text must not be parsed as such
static std::string fDecisionKevText(const json & pVal) {
  static const std::regex cReSpecial("<\\|([A-Za-z0-9_]+)\\|>");
  return std::regex_replace(fDecisionKevRender(pVal), cReSpecial, "<\xC2\xA6$1\xC2\xA6>");
}

// lfm2-d1-omni: special tokens written in the input must not be parsed as such, in keys too (d1-omni prompt.py: escape)
static json fDecisionD1omniEscape(const json & pVal) {
  static const std::regex cReSpecial("<\\|([A-Za-z0-9_]+)\\|>");
  if (pVal.is_string()) {
    return std::regex_replace(pVal.get<std::string>(), cReSpecial, "<\xC2\xA6$1\xC2\xA6>");
  }
  if (pVal.is_array()) {
    json vOut = json::mArray();
    for (const auto & vItem : pVal) {
      vOut.push_back(fDecisionD1omniEscape(vItem));
    }
    return vOut;
  }
  if (pVal.is_object()) {
    json vOut = json::mObject();
    for (const auto & [key, item] : pVal.mItems()) {
      vOut[std::regex_replace(key, cReSpecial, "<\xC2\xA6$1\xC2\xA6>")] = fDecisionD1omniEscape(item);
    }
    return vOut;
  }
  return pVal;
}

// given to the lfm2-d1-omni template: text between the pieces of the prompt, and at the start of the pieces that are cut to a token budget
static const std::string cD1omniMarker       = "<<d1omni:";
static const std::string cD1omniSep          = "<<d1omni:sep>>";
static const std::string cD1omniMarkState    = "<<d1omni:state>>";
static const std::string cD1omniMarkQuestion = "<<d1omni:question>>";
static const std::string cD1omniMarkOption   = "<<d1omni:option>>";

// max_length, image_text_length and audio_text_length of the model config, the converter checks them
static const size_t cD1omniMaxTokens      = 16384;
static const size_t cD1omniMaxTokensImage = 896;
static const size_t cD1omniMaxTokensAudio = 15360;

size_t server_decision_context::mNVariants(const server_decision_question & pQuestion) const {
  // lev shows the options of a choice in 2 orders, to cancel the preference for the first label
  if (type == cCommonDecisionTypeLev && pQuestion.type == cServerDecisionQuestionChoice &&
      pQuestion.lOptions.size() > 1) {
    return 2;
  }
  return 1;
}

size_t server_decision_context::mNOutputs(const server_decision_question & pQuestion) const {
  if (type == cCommonDecisionTypeLev && pQuestion.type == cServerDecisionQuestionNoul) {
    return cDecisionLevNRatings;
  }
  return pQuestion.lOptions.size();
}

// label codes follow prompt.py of the model repo
void server_decision_context::mD1Labels(const server_decision_question & pQuestion,
                                        std::vector<std::string> &       pTexts,
                                        std::vector<dgs_tokens> &        pGroups) const {
  const size_t cNOptions = pQuestion.lOptions.size();

  auto vGetSingleTokens = [&](const std::vector<std::string> & pForms) {
    dgs_tokens lOut;
    for (const auto & vForm : pForms) {
      const auto lToks = fCommonTokenize(vVocab, vForm, false, false);
      if (lToks.size() == 1 && std::find(lOut.begin(), lOut.end(), lToks[0]) == lOut.end()) {
        lOut.push_back(lToks[0]);
      }
    }
    return lOut;
  };

  if (pQuestion.type != cServerDecisionQuestionChoice) {
    for (const auto & vOpt : pQuestion.lOptions) {
      dgs_tokens lGroup;
      if (pQuestion.type == cServerDecisionQuestionScore) {
        lGroup = vGetSingleTokens({ vOpt.vKey });
      } else if (vOpt.vKey == "true") {
        lGroup = vGetSingleTokens({ "yes", "Yes", "YES" });
      } else {
        lGroup = vGetSingleTokens({ "no", "No", "NO" });
      }
      if (lGroup.empty()) {
        throw std::runtime_error("decision label is not a single token: " + vOpt.vKey);
      }
      pTexts.push_back(vOpt.vKey);
      pGroups.push_back(lGroup);
    }
    return;
  }

  bool vIsLetters = true;
  for (const auto & vOpt : pQuestion.lOptions) {
    vIsLetters = vIsLetters && vOpt.vKey.size() == 1 && std::isalpha((unsigned char) vOpt.vKey[0]);
  }

  std::vector<std::string> lCodes;
  for (size_t vI = 0; vI < cNOptions; vI++) {
    if (vIsLetters) {
      lCodes.push_back(pQuestion.lOptions[vI].vKey);
    } else if (cNOptions <= 26) {
      lCodes.push_back(std::string(1, 'A' + vI));
    } else {
      lCodes.push_back(fStringFormat("%02zu", vI));
    }
  }

  std::vector<std::string> lPool;
  for (char vC = 'A'; vC <= 'Z'; vC++) {
    lPool.push_back(std::string(1, vC));
  }
  for (int vI = 0; vI < 100; vI++) {
    lPool.push_back(fStringFormat("%02d", vI));
  }
  for (char vC = 'a'; vC <= 'z'; vC++) {
    lPool.push_back(std::string(1, vC));
  }
  for (int vI = 0; vI < 200; vI++) {
    lPool.push_back(fStringFormat("#%d", vI));
  }
  for (char vA = 'A'; vA <= 'Z'; vA++) {
    for (char vB = 'A'; vB <= 'Z'; vB++) {
      lPool.push_back(std::string{ vA, vB });
    }
  }

  dgs_tokens lUsed;
  auto       vTake = [&](const std::string & pCode) {
    const auto lToks = fCommonTokenize(vVocab, pCode, false, false);
    if (lToks.size() != 1 || std::find(lUsed.begin(), lUsed.end(), lToks[0]) != lUsed.end()) {
      return false;
    }
    lUsed.push_back(lToks[0]);
    dgs_tokens lGroup = { lToks[0] };
    for (const dgs_token cTok : vGetSingleTokens({ " " + pCode })) {
      if (cTok != lToks[0]) {
        lGroup.push_back(cTok);
      }
    }
    pTexts.push_back(pCode);
    pGroups.push_back(lGroup);
    return true;
  };
  for (const auto & vCode : lCodes) {
    bool vIsTaken = vTake(vCode);
    for (size_t vI = 0; !vIsTaken && vI < lPool.size(); vI++) {
      vIsTaken = vTake(lPool[vI]);
    }
    if (!vIsTaken) {
      throw std::invalid_argument(fStringFormat("no single-token label left for %zu options", cNOptions));
    }
  }
}

json server_decision_context::mRenderOptions(const server_decision_question & pQuestion, size_t pVariant) const {
  const size_t cNOptions = pQuestion.lOptions.size();

  std::vector<std::string> lD1Texts;
  std::vector<dgs_tokens>  lD1Groups;
  if (type == cCommonDecisionTypeLfm2D1) {
    mD1Labels(pQuestion, lD1Texts, lD1Groups);
  }

  // the second variant shows the options in the reverse order
  json vOptions = json::mArray();
  for (size_t vI = 0; vI < cNOptions; vI++) {
    const auto & vOpt    = pQuestion.lOptions[pVariant == 0 ? vI : cNOptions - 1 - vI];
    json         vOption = json{
      { "key",         vOpt.vKey         },
      { "description", vOpt.vDescription },
    };
    if (type == cCommonDecisionTypeKev) {
      vOption["key"] = fDecisionKevText(vOpt.vKey);
      if (!vOpt.vDescription.is_null()) {
        vOption["description"] = fDecisionKevText(vOpt.vDescription);
      }
    }
    if (!lLabelTexts.empty()) {
      vOption["label"] = lLabelTexts[vI];
    }
    if (!lD1Texts.empty()) {
      vOption["label"] = lD1Texts[vI];
    }
    vOptions.push_back(vOption);
  }
  return vOptions;
}

std::string server_decision_context::mRender(const json &                                  pState,
                                             const std::vector<server_decision_question> & pQuestions,
                                             const server_decision_question &              pQuestion,
                                             size_t                                        pVariant,
                                             size_t                                        pNImages,
                                             bool                                          pIsAudio) const {
  // the template is given raw JSON values, it serializes the ones that are not strings
  json vInp = json{
    { "id", pQuestion.vId },
    { "type", fDecisionQuestionTypeName(pQuestion.type) },
    { "instructions", pQuestion.vInstructions },
    { "state", pState },
    { "options", mRenderOptions(pQuestion, pVariant) },
  };

  // the nimble prompt lists all the questions of the request
  if (type == cCommonDecisionTypeNimble) {
    vInp["questions"] = json::mArray();
    for (const auto & vQ : pQuestions) {
      vInp["questions"].push_back(json{
        { "id", vQ.vId },
        { "type", fDecisionQuestionTypeName(vQ.type) },
        { "instructions", vQ.vInstructions },
        { "options", mRenderOptions(vQ, 0) },
      });
    }
  }

  // lev was trained with sorted keys
  if (type == cCommonDecisionTypeLev) {
    vInp = fDecisionSortKeys(vInp);
  }

  // the kev template only takes text
  if (type == cCommonDecisionTypeKev) {
    vInp["state"]        = fDecisionKevText(pState);
    vInp["instructions"] = fDecisionKevText(pQuestion.vInstructions);
  }

  // the input must not contain the marker of the options
  if (!vTextMarker.empty()) {
    vInp = fDecisionReplaceText(vInp, vTextMarker, " ");
  }

  if (type == cCommonDecisionTypeLfm2D1Omni) {
    vInp                  = fDecisionReplaceText(fDecisionD1omniEscape(vInp), cD1omniMarker, "<<d1omni ");
    vInp["audio"]         = pIsAudio;
    vInp["sep"]           = cD1omniSep;
    vInp["mark_state"]    = cD1omniMarkState;
    vInp["mark_question"] = cD1omniMarkQuestion;
    vInp["mark_option"]   = cD1omniMarkOption;
  }

  // the template puts one media marker per image
  json vImages = json::mArray();
  if (pNImages > 0) {
    vInp = fDecisionReplaceText(vInp, fGetMediaMarker(), " ");
    for (size_t vI = 0; vI < pNImages; vI++) {
      vImages.push_back(fGetMediaMarker());
    }
  }
  vInp["images"] = vImages;

  jinja::context vCtx(vTmpl->mSource());
  jinja::fGlobalFromJson(vCtx, vInp, false);
  jinja::runtime     vRuntime(vCtx);
  const jinja::value cResults = vRuntime.mExecute(vTmpl->prog);
  return jinja::runtime::mGatherStringParts(cResults)->mAsString().str();
}

void server_decision_context::mFillTask(const json &                                  pState,
                                        const std::vector<server_decision_question> & pQuestions,
                                        const server_decision_question &              pQuestion,
                                        size_t                                        pVariant,
                                        const std::vector<raw_buffer> &               pFiles,
                                        mtmd_context *                                pMctx,
                                        const mtmd_helper_init_opt &                  pInitOpt,
                                        server_task &                                 pTask) const {
  // each question has its own prompt, with the spans that the head reads
  if (type == cCommonDecisionTypeVllmDecision2) {
    mFillTaskJoint(pState, { pQuestion }, pFiles, pMctx, pInitOpt, pTask);
    return;
  }
  if (type == cCommonDecisionTypeLfm2D1Omni) {
    mFillTaskD1omni(pState, pQuestions, pQuestion, pVariant, pFiles, pMctx, pInitOpt, pTask);
    return;
  }

  const std::string cPrompt = mRender(pState, pQuestions, pQuestion, pVariant, pFiles.size());

  if (type == cCommonDecisionTypeOpenjev || type == cCommonDecisionTypeLev || type == cCommonDecisionTypeNimble ||
      type == cCommonDecisionTypePplxDecider || type == cCommonDecisionTypeVllmD3) {
    // lev reads the ratings of a noul question at its first labels, not at the digits
    pTask.decision.lLabels.assign(lLabels.begin(), lLabels.begin() + mNOutputs(pQuestion));
  }
  if (type == cCommonDecisionTypeLfm2D1) {
    std::vector<std::string> lTexts;
    std::vector<dgs_tokens>  lGroups;
    mD1Labels(pQuestion, lTexts, lGroups);
    for (const auto & lGroup : lGroups) {
      pTask.decision.lLabels.insert(pTask.decision.lLabels.end(), lGroup.begin(), lGroup.end());
      pTask.decision.lLabelGroups.push_back(lGroup.size());
    }
  }
  if (type == cCommonDecisionTypeOpenjev || type == cCommonDecisionTypeLev || type == cCommonDecisionTypeNimble ||
      type == cCommonDecisionTypePplxDecider || type == cCommonDecisionTypeVllmD3 ||
      type == cCommonDecisionTypeLfm2D1) {
    if (!pFiles.empty()) {
      pTask.vTokens = fProcessMtmdPrompt(pMctx, cPrompt, pFiles, pInitOpt);
      return;
    }
  }

  dgs_tokens lTokens = fCommonTokenize(vVocab, cPrompt, false, true);
  if (type == cCommonDecisionTypeLaya) {
    mFillTaskLaya(lTokens, pQuestion, pTask);
  }
  if (type == cCommonDecisionTypeKev) {
    // an option is read at its end token, the question at the last token
    for (size_t vI = 0; vI < lTokens.size(); vI++) {
      if (lTokens[vI] == vTokenMarker) {
        pTask.decision.lMarkers.push_back(vI);
      }
    }
    if (pTask.decision.lMarkers.size() != pQuestion.lOptions.size()) {
      throw std::runtime_error("unexpected layout of the decision prompt");
    }
    pTask.decision.pointer = lTokens.size() - 1;
  }
  pTask.vTokens = server_tokens(lTokens, false);
}

// the prompt is: [cls] question [sep] ([marker] option)* [sep] state [sep]
// options and question are cut to fit max_head_tokens, the same way the model was trained
void server_decision_context::mFillTaskLaya(dgs_tokens &                     pTokens,
                                            const server_decision_question & pQuestion,
                                            server_task &                    pTask) const {
  const size_t cNOptions = pQuestion.lOptions.size();

  std::vector<size_t> lMarkers;
  for (size_t vI = 0; vI < pTokens.size(); vI++) {
    if (pTokens[vI] == vTokenMarker) {
      lMarkers.push_back(vI);
    }
  }
  const auto cInvalid = std::runtime_error("unexpected layout of the decision prompt");
  if (lMarkers.size() != cNOptions || lMarkers[0] < 2 || pTokens[lMarkers[0] - 1] != vTokenSep ||
      pTokens.back() != vTokenSep) {
    throw cInvalid;
  }
  const size_t cHeadEnd = lMarkers[0] - 1;
  const size_t cOptsEnd = std::find(pTokens.begin() + lMarkers.back(), pTokens.end(), vTokenSep) - pTokens.begin();
  if (cOptsEnd + 1 >= pTokens.size()) {
    throw cInvalid;
  }

  // marker + text of each option
  std::vector<dgs_tokens> lOptions;
  size_t                  vNOptionsTokens = 0;
  auto                    vSetMax         = [&](size_t pNMax) {
    vNOptionsTokens = 0;
    for (auto & lOpt : lOptions) {
      lOpt.resize(std::min(lOpt.size(), pNMax));
      vNOptionsTokens += lOpt.size();
    }
  };
  for (size_t vI = 0; vI < cNOptions; vI++) {
    const size_t end = vI + 1 < cNOptions ? lMarkers[vI + 1] : cOptsEnd;
    lOptions.emplace_back(pTokens.begin() + lMarkers[vI], pTokens.begin() + end);
  }
  vSetMax(vMaxOptionTokens + 1);
  if (vNOptionsTokens + 16 > vMaxHeadTokens) {
    // too many or too long options, shrink them evenly
    vSetMax(std::max((size_t) 4, (vMaxHeadTokens - std::min(vMaxHeadTokens, (size_t) 16)) / cNOptions));
  }
  const size_t cNQuestionMax = std::max((size_t) 8, vMaxHeadTokens - std::min(vMaxHeadTokens, vNOptionsTokens));

  dgs_tokens lOut;
  lOut.push_back(pTokens[0]);
  lOut.insert(lOut.end(), pTokens.begin() + 1, pTokens.begin() + std::min(cHeadEnd, 1 + cNQuestionMax));
  lOut.push_back(vTokenSep);
  for (const auto & lOpt : lOptions) {
    pTask.decision.lMarkers.push_back(lOut.size());
    lOut.insert(lOut.end(), lOpt.begin(), lOpt.end());
  }
  lOut.insert(lOut.end(), pTokens.begin() + cOptsEnd, pTokens.end());
  pTokens = std::move(lOut);

  // the output has one score per question type
  pTask.decision.vColumn = pQuestion.type;
}

// each piece is cut to its budget as the model was trained (d1-omni prompt.py: encode), the state gets the room that is left
// the media come first, the text after them has a budget of its own
void server_decision_context::mFillTaskD1omni(const json &                                  pState,
                                              const std::vector<server_decision_question> & pQuestions,
                                              const server_decision_question &              pQuestion,
                                              size_t                                        pVariant,
                                              const std::vector<raw_buffer> &               pFiles,
                                              mtmd_context *                                pMctx,
                                              const mtmd_helper_init_opt &                  pInitOpt,
                                              server_task &                                 pTask) const {
  const auto cInvalid = std::runtime_error("unexpected layout of the decision prompt");

  // the prompt depends on the kind of media, mtmd tells an audio clip from an image by its content
  std::string   vMarkers;
  server_tokens vMedia(dgs_tokens(), false);
  bool          vIsAudio = false;
  if (!pFiles.empty()) {
    for (size_t vI = 0; vI < pFiles.size(); vI++) {
      vMarkers += fGetMediaMarker();
    }
    vMedia = fProcessMtmdPrompt(pMctx, vMarkers, pFiles, pInitOpt);
    for (size_t vI = 0; vI < vMedia.size(); vI++) {
      if (vMedia[vI] == DGS_TOKEN_NULL) {
        const auto & vChunk = vMedia.mFindChunk(vI);
        vIsAudio            = vIsAudio || fMtmdInputChunkGetType(vChunk.get()) == cMtmdInputChunkTypeAudio;
        vI += fMtmdInputChunkGetNTokens(vChunk.get()) - 1;
      }
    }
  }
  if (vIsAudio && pFiles.size() > 1) {
    throw std::invalid_argument("a request has images or one audio clip, not both");
  }
  const size_t cNMedia = vMedia.size();

  const std::string        cPrompt = mRender(pState, pQuestions, pQuestion, pVariant, pFiles.size(), vIsAudio);
  std::vector<std::string> lPieces = fStringSplit(cPrompt, cD1omniSep);
  if (lPieces.empty() || lPieces[0] != vMarkers) {
    throw cInvalid;
  }

  size_t vNMax = cD1omniMaxTokens;
  if (!pFiles.empty()) {
    vNMax = std::min(vIsAudio ? cD1omniMaxTokensAudio : cD1omniMaxTokensImage,
                     cD1omniMaxTokens - std::min(cD1omniMaxTokens, cNMedia));
    if (vNMax < 64) {
      throw std::invalid_argument(
        fStringFormat("the media take %zu of the %zu positions, send fewer images", cNMedia, cD1omniMaxTokens));
    }
  }

  // the options get max(96, min(24 n + 32, max / 2)) tokens, shared evenly
  const int64_t cNOptions     = pQuestion.lOptions.size();
  const int64_t cNBudget      = std::max<int64_t>(96, std::min<int64_t>(24 * cNOptions + 32, vNMax / 2));
  const int64_t cNOptionMax   = std::max<int64_t>(2, (cNBudget - 3 * cNOptions) / cNOptions);
  const int64_t cNQuestionMax = std::max<int64_t>(16, cNBudget);

  dgs_tokens lHead;  // before the state
  dgs_tokens lBody;  // the state
  dgs_tokens lTail;  // after the state
  bool       vHasState = false;
  for (size_t vIPiece = 1; vIPiece < lPieces.size(); vIPiece++) {
    std::string vPiece     = lPieces[vIPiece];
    int64_t     vNPieceMax = -1;
    bool        vIsState   = false;
    if (fStringStartsWith(vPiece, cD1omniMarkState)) {
      vPiece   = vPiece.substr(cD1omniMarkState.size());
      vIsState = true;
    } else if (fStringStartsWith(vPiece, cD1omniMarkQuestion)) {
      vPiece     = vPiece.substr(cD1omniMarkQuestion.size());
      vNPieceMax = cNQuestionMax;
    } else if (fStringStartsWith(vPiece, cD1omniMarkOption)) {
      vPiece     = vPiece.substr(cD1omniMarkOption.size());
      vNPieceMax = cNOptionMax;
    }

    dgs_tokens lTokens = fCommonTokenize(vVocab, vPiece, false, true);
    if (vNPieceMax >= 0 && (int64_t) lTokens.size() > vNPieceMax) {
      lTokens.resize(vNPieceMax);
    }

    if (vIsState) {
      if (vHasState) {
        throw cInvalid;
      }
      lBody     = std::move(lTokens);
      vHasState = true;
    } else {
      dgs_tokens & lDst = vHasState ? lTail : lHead;
      lDst.insert(lDst.end(), lTokens.begin(), lTokens.end());
    }
  }
  if (!vHasState) {
    throw cInvalid;
  }

  const size_t cNRoom = vNMax - std::min(vNMax, lHead.size() + lTail.size());
  lBody.resize(std::min(lBody.size(), cNRoom));

  dgs_tokens lTokens = std::move(lHead);
  lTokens.insert(lTokens.end(), lBody.begin(), lBody.end());
  lTokens.insert(lTokens.end(), lTail.begin(), lTail.end());
  lTokens.resize(std::min(lTokens.size(), vNMax));

  for (size_t vI = 0; vI < lTokens.size(); vI++) {
    if (lTokens[vI] == vTokenMarker) {
      pTask.decision.lMarkers.push_back(cNMedia + vI);
    }
  }
  if ((int64_t) pTask.decision.lMarkers.size() != cNOptions) {
    throw std::invalid_argument("the options do not fit in the context");
  }

  // the output has one score per question type
  pTask.decision.vColumn = pQuestion.type;
  if (pFiles.empty()) {
    pTask.vTokens = server_tokens(lTokens, false);
  } else {
    pTask.vTokens = std::move(vMedia);
    for (const dgs_token token : lTokens) {
      pTask.vTokens.push_back(token);
    }
  }
}

//
// joint prompt (clef)
//

// given to the template: text between the pieces of the prompt, and at the start of the span of a question or of an option
static const std::string cClefMarker       = "<<clef:";
static const std::string cClefSep          = "<<clef:sep>>";
static const std::string cClefMarkQuestion = "<<clef:question>>";
static const std::string cClefMarkOption   = "<<clef:option>>";

void server_decision_context::mFillTaskJoint(const json &                                  pState,
                                             const std::vector<server_decision_question> & pQuestions,
                                             const std::vector<raw_buffer> &               pFiles,
                                             mtmd_context *                                pMctx,
                                             const mtmd_helper_init_opt &                  pInitOpt,
                                             server_task &                                 pTask) const {
  json vInpQuestions = json::mArray();
  for (const auto & vQuestion : pQuestions) {
    json vOptions = json::mArray();
    for (const auto & vOpt : vQuestion.lOptions) {
      vOptions.push_back(json{
        { "key",         vOpt.vKey         },
        { "description", vOpt.vDescription },
      });
    }
    vInpQuestions.push_back(json{
      { "id",           vQuestion.vId                             },
      { "type",         fDecisionQuestionTypeName(vQuestion.type) },
      { "instructions", vQuestion.vInstructions                   },
      { "options",      vOptions                                  },
    });
  }

  // the template is given raw JSON values with sorted keys, and no marker in the input
  json vInp = json{
    { "state",     pState        },
    { "questions", vInpQuestions },
  };
  vInp = fDecisionReplaceText(fDecisionSortKeys(vInp), cClefMarker, "<<clef ");

  // the template puts one media marker per image
  json vImages = json::mArray();
  if (!pFiles.empty()) {
    vInp = fDecisionReplaceText(vInp, fGetMediaMarker(), " ");
    for (size_t vI = 0; vI < pFiles.size(); vI++) {
      vImages.push_back(fGetMediaMarker());
    }
  }
  vInp["images"]        = vImages;
  vInp["sep"]           = cClefSep;
  vInp["mark_question"] = cClefMarkQuestion;
  vInp["mark_option"]   = cClefMarkOption;

  jinja::context vCtx(vTmpl->mSource());
  jinja::fGlobalFromJson(vCtx, vInp, false);
  jinja::runtime     vRuntime(vCtx);
  const jinja::value cResults = vRuntime.mExecute(vTmpl->prog);
  const std::string  cPrompt  = jinja::runtime::mGatherStringParts(cResults)->mAsString().str();

  const auto cInvalid = std::runtime_error("unexpected layout of the decision prompt");

  // the model was trained with the pieces tokenized one by one
  const std::vector<std::string> lPieces = fStringSplit(cPrompt, cClefSep);
  size_t                         vIPiece = 0;

  pTask.vTokens = server_tokens(dgs_tokens(), false);
  if (!pFiles.empty()) {
    // the text before the images is tokenized with them, a vision token separates the pieces anyway
    std::string vHead;
    while (vIPiece < lPieces.size() && vHead.find(fGetMediaMarker()) == std::string::npos) {
      vHead += lPieces[vIPiece++];
    }
    if (vHead.find(fGetMediaMarker()) == std::string::npos || vHead.find(cClefMarker) != std::string::npos) {
      throw cInvalid;
    }
    pTask.vTokens = fProcessMtmdPrompt(pMctx, vHead, pFiles, pInitOpt);
    pTask.decision.lOrder.resize(pTask.vTokens.size(), cDgsDecisionOrderNone);
  }

  size_t vIQuestion = 0;
  for (; vIPiece < lPieces.size(); vIPiece++) {
    std::string vPiece = lPieces[vIPiece];
    int32_t     vOrder = cDgsDecisionOrderNone;
    if (fStringStartsWith(vPiece, cClefMarkQuestion)) {
      vPiece = vPiece.substr(cClefMarkQuestion.size());
      if (vIQuestion >= pQuestions.size()) {
        throw cInvalid;
      }
      switch (pQuestions[vIQuestion++].type) {
        case cServerDecisionQuestionNoul:
          vOrder = cDgsDecisionOrderQuestionNoul;
          break;
        case cServerDecisionQuestionChoice:
          vOrder = cDgsDecisionOrderQuestionChoice;
          break;
        case cServerDecisionQuestionScore:
          vOrder = cDgsDecisionOrderQuestionScore;
          break;
      }
    } else if (fStringStartsWith(vPiece, cClefMarkOption)) {
      vPiece = vPiece.substr(cClefMarkOption.size());
      vOrder = cDgsDecisionOrderOption;
      pTask.decision.vNScores++;
    }

    const dgs_tokens lPieceTokens = fCommonTokenize(vVocab, vPiece, false, true);
    if (vOrder != cDgsDecisionOrderNone && lPieceTokens.empty()) {
      throw std::invalid_argument("the instructions and the options of a question must not be empty");
    }
    for (const dgs_token token : lPieceTokens) {
      pTask.vTokens.push_back(token);
    }
    pTask.decision.lOrder.resize(pTask.vTokens.size(), vOrder);
  }

  size_t vNOptions = 0;
  for (const auto & vQuestion : pQuestions) {
    vNOptions += vQuestion.lOptions.size();
  }
  if (vIQuestion != pQuestions.size() || (size_t) pTask.decision.vNScores != vNOptions) {
    throw cInvalid;
  }
}

//
// answer
//

float server_decision_context::mGetTemperature(const server_decision_question & pQuestion) const {
  const size_t      n         = pQuestion.lOptions.size();
  const std::string cTypeName = fDecisionQuestionTypeName(pQuestion.type);

  // the temperature can depend on the number of options, the buckets are the ones used to fit it
  std::string vBucket;
  if (type == cCommonDecisionTypeLev) {
    vBucket = n <= 8 ? "small" : n <= 26 ? "mid" : "large";
  } else {
    vBucket = n <= 2 ? "2" : n <= 5 ? "3_5" : n <= 10 ? "6_10" : "11";
  }

  for (const auto & name : { cTypeName + "." + vBucket, cTypeName }) {
    const auto cIt = dTemperatures.find(name);
    if (cIt != dTemperatures.end()) {
      return cIt->second;
    }
  }
  return 1.0f;
}

// confidence formulas are the ones published by TypeSafe

static double fDecisionConfidenceChoice(const std::vector<double> & pProbs) {
  if (pProbs.size() < 2) {
    return 1.0;
  }
  const double cUniform = 1.0 / pProbs.size();
  const double cPMax    = *std::max_element(pProbs.begin(), pProbs.end());
  return std::max(0.0, (cPMax - cUniform) / (1.0 - cUniform));
}

static double fDecisionConfidenceScore(const std::vector<double> & pProbs) {
  if (pProbs.size() < 2) {
    return 1.0;
  }
  const size_t n     = pProbs.size();
  const size_t cMode = std::max_element(pProbs.begin(), pProbs.end()) - pProbs.begin();

  // mean distance to the mode, relative to the one of a uniform distribution around its center
  double vDist        = 0.0;
  double vDistUniform = 0.0;
  for (size_t vI = 0; vI < n; vI++) {
    vDist += pProbs[vI] * std::fabs((double) vI - (double) cMode);
    vDistUniform += std::fabs((double) vI - (n - 1) / 2.0) / n;
  }
  return std::max(0.0, 1.0 - vDist / vDistUniform);
}

// VLLM_DECISION2, VLLM_D3: normalized entropy, as in the reference implementation
static double fDecisionConfidenceEntropy(const std::vector<double> & pProbs) {
  if (pProbs.size() < 2) {
    return 1.0;
  }
  double vEntropy = 0.0;
  for (const double cP : pProbs) {
    if (cP > 0.0) {
      vEntropy -= cP * std::log(cP);
    }
  }
  return std::max(0.0, std::min(1.0, 1.0 - vEntropy / std::log((double) pProbs.size())));
}

double server_decision_context::mGetDecisionConfidence(server_decision_question_type pQuestionType,
                                                       const std::vector<double> &   pProbs) const {
  if (type == cCommonDecisionTypeVllmDecision2 || type == cCommonDecisionTypeVllmD3) {
    return fDecisionConfidenceEntropy(pProbs);
  }
  switch (pQuestionType) {
    case cServerDecisionQuestionChoice:
      return fDecisionConfidenceChoice(pProbs);
    case cServerDecisionQuestionScore:
      return fDecisionConfidenceScore(pProbs);
    case cServerDecisionQuestionNoul:
      break;
  }
  throw std::runtime_error("no confidence for this question type");
}

json server_decision_context::mFormatAnswer(const server_decision_question &        pQuestion,
                                            const std::vector<std::vector<float>> & pScores,
                                            bool                                    pHasMedia) const {
  const size_t n = mNOutputs(pQuestion);
  if (pScores.size() != mNVariants(pQuestion)) {
    throw std::runtime_error("decision result does not match the number of variants");
  }

  // softmax over the outputs of each variant, then the average of the variants
  // lfm2-d1-omni: image and audio answers are not calibrated
  const float cTemperature = pHasMedia && type == cCommonDecisionTypeLfm2D1Omni ? 1.0f : mGetTemperature(pQuestion);
  std::vector<double> lProbs(n, 0.0);
  for (size_t vV = 0; vV < pScores.size(); vV++) {
    const auto & lS = pScores[vV];
    if (lS.size() != n) {
      throw std::runtime_error("decision result does not match the number of options");
    }
    // a joint head returns NaN if it could not use the decision order
    if (std::any_of(lS.begin(), lS.end(), [](float pV) { return std::isnan(pV); })) {
      throw std::runtime_error("the model could not evaluate the decision");
    }
    std::vector<double> lLogits(lS.begin(), lS.end());
    if (pQuestion.type == cServerDecisionQuestionScore && dScoreBias.count(n) > 0) {
      for (size_t vI = 0; vI < n; vI++) {
        lLogits[vI] += dScoreBias.at(n)[vI];
      }
    }
    const double        cLogitMax = *std::max_element(lLogits.begin(), lLogits.end());
    std::vector<double> lP(n);
    double              vSum = 0.0;
    for (size_t vI = 0; vI < n; vI++) {
      lP[vI] = std::exp((lLogits[vI] - cLogitMax) / cTemperature);
      vSum += lP[vI];
    }
    for (size_t vI = 0; vI < n; vI++) {
      // the second variant is in the reverse order
      lProbs[vV == 0 ? vI : n - 1 - vI] += lP[vI] / vSum / pScores.size();
    }
  }

  json vAnswer = json{
    { "type", fDecisionQuestionTypeName(pQuestion.type) }
  };

  if (pQuestion.type == cServerDecisionQuestionNoul) {
    if (type == cCommonDecisionTypeLev) {
      double vExpected = 0.0;
      for (size_t vI = 0; vI < n; vI++) {
        vExpected += lProbs[vI] * vI / (n - 1);
      }
      vAnswer["noul"] = vExpected;
      return vAnswer;
    }
    for (size_t vI = 0; vI < n; vI++) {
      if (pQuestion.lOptions[vI].vKey == "true") {
        vAnswer["noul"] = lProbs[vI];
      }
    }
    return vAnswer;
  }

  json vProbabilities = json::mObject();
  for (size_t vI = 0; vI < n; vI++) {
    vProbabilities[pQuestion.lOptions[vI].vKey] = lProbs[vI];
  }

  if (pQuestion.type == cServerDecisionQuestionChoice) {
    const size_t cBest       = std::max_element(lProbs.begin(), lProbs.end()) - lProbs.begin();
    vAnswer["choice"]        = pQuestion.lOptions[cBest].vKey;
    vAnswer["probabilities"] = vProbabilities;
    vAnswer["confidence"]    = mGetDecisionConfidence(pQuestion.type, lProbs);
  } else {
    double vExpected = 0.0;
    json   vLegend   = json::mObject();
    for (size_t vI = 0; vI < n; vI++) {
      vExpected += vI * lProbs[vI];
      const json & vDescription = pQuestion.lOptions[vI].vDescription;
      vLegend[pQuestion.lOptions[vI].vKey] =
        vCanonicalLegend && !vDescription.is_string() ? json(fDecisionSortKeys(vDescription).mDump()) : vDescription;
    }
    vAnswer["score"]         = vExpected;
    vAnswer["legend"]        = vLegend;
    vAnswer["probabilities"] = vProbabilities;
    vAnswer["confidence"]    = mGetDecisionConfidence(pQuestion.type, lProbs);
  }
  return vAnswer;
}

//
// shared prompt prefix
//

std::vector<server_task> fServerDecisionGroupTasks(std::vector<server_task> && pTasks, size_t pNSlots) {
  pNSlots = std::max(pNSlots, (size_t) 1);

  std::vector<server_task> lGroups;
  for (size_t vI = 0; vI < pTasks.size(); vI += pNSlots) {
    const size_t  end     = std::min(pTasks.size(), vI + pNSlots);
    server_task & vParent = pTasks[vI];

    // every task must have at least one token of its own to evaluate
    size_t vNShared = vParent.vTokens.size() - 1;
    for (size_t vJ = vI + 1; vJ < end; vJ++) {
      vNShared = std::min(vNShared, vParent.vTokens.mGetCommonPrefix(pTasks[vJ].vTokens));
      vNShared = std::min(vNShared, pTasks[vJ].vTokens.size() - 1);
    }

    if (end - vI < 2 || vNShared == 0) {
      for (size_t vJ = vI; vJ < end; vJ++) {
        lGroups.push_back(std::move(pTasks[vJ]));
      }
      continue;
    }

    vParent.vNTokensShared = vNShared;
    for (size_t vJ = vI + 1; vJ < end; vJ++) {
      pTasks[vJ].vIdParent = vParent.vId;
      vParent.lChildTasks.push_back(std::move(pTasks[vJ]));
    }
    lGroups.push_back(std::move(vParent));
  }
  return lGroups;
}
