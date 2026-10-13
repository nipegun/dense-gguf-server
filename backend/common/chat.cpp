#include "chat.h"

#include "chat-auto-parser-helpers.h"
#include "chat-auto-parser.h"
#include "chat-peg-parser.h"
#include "common.h"
#include "ggml.h"
#include "jinja/caps.h"
#include "jinja/runtime.h"
#include "jinja/value.h"
#include "json-schema-to-grammar.h"
#include "json.h"
#include "log.h"
#include "parsers/parsers.h"
#include "peg-parser.h"
#include "sampling.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <functional>
#include <iomanip>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using json = common_json;

static std::string fFormatTime(const std::chrono::system_clock::time_point & pNow, const std::string & format) {
  auto               vTime      = std::chrono::system_clock::to_time_t(pNow);
  auto               vLocalTime = *std::localtime(&vTime);
  std::ostringstream vSs;
  vSs << std::put_time(&vLocalTime, format.c_str());
  auto vRes = vSs.str();
  return vRes;
}

static json fSafeArgsParse(const std::string & pToParse) {
  std::string vStripped = pToParse;
  if (pToParse.at(0) == '"' && pToParse.at(pToParse.length() - 1) == '"') {
    vStripped = pToParse.substr(1, pToParse.length() - 1);
  }
  try {
    return json::mParse(vStripped);
  } catch (const common_json_error & vE) {
    return vStripped;
  }
}

static std::string fStringDiff(const std::string & pLast, const std::string & pCurrent) {
  if (pLast.empty()) {
    return pCurrent;
  }
  if (!fStringStartsWith(pCurrent, pLast)) {
    if (fStringStartsWith(pLast, pCurrent)) {
      // This happens if the last generation ended on a partial stop word (not erased),
      // and the current ended on a stop word (erased).
      return "";
    }
    throw std::runtime_error("Invalid diff: '" + pLast + "' not found at start of '" + pCurrent + "'");
  }
  return pCurrent.substr(pLast.size());
}

static bool fHasContentOrToolCalls(const common_chat_msg & pMsg) {
  return !pMsg.vContent.empty() || !pMsg.lToolCalls.empty();
}

std::string common_chat_msg::mRenderContent(const std::string & pDelimiter) const {
  if (!vContent.empty() && !lContentParts.empty()) {
    throw std::runtime_error("Cannot specify both content and content_parts");
  }
  if (!vContent.empty()) {
    return vContent;
  }

  std::string vText;
  for (const auto & vPart : lContentParts) {
    if (vPart.type == "text") {
      if (!vText.empty()) {
        vText += pDelimiter;
      }
      vText += vPart.vText;
    }
  }
  return vText;
}

common_chat_role fCommonChatRoleFromString(const std::string & pRole) {
  if (pRole == "system") {
    return cCommonChatRoleSystem;
  }
  if (pRole == "assistant") {
    return cCommonChatRoleAssistant;
  }
  if (pRole == "user") {
    return cCommonChatRoleUser;
  }
  if (pRole == "tool") {
    return cCommonChatRoleTool;
  }
  return cCommonChatRoleUnknown;
}

const char * fCommonChatRoleToString(common_chat_role pRole) {
  switch (pRole) {
    case cCommonChatRoleSystem:
      return "system";
    case cCommonChatRoleAssistant:
      return "assistant";
    case cCommonChatRoleUser:
      return "user";
    case cCommonChatRoleTool:
      return "tool";
    case cCommonChatRoleUnknown:
      return "";
  }
  return "";
}

void common_chat_msg_delimiters::mTokenize(const dgs_vocab * pVocab) {
  for (auto & vD : lDelimiters) {
    vD.lTokens = fCommonTokenize(pVocab, vD.vDelimiter, false, true);
  }
}

common_chat_msg_spans common_chat_msg_delimiters::mSplit(const dgs_tokens &               pTokens,
                                                         const std::map<size_t, size_t> & pSkips) const {
  std::vector<std::pair<common_chat_role, size_t>> lMatches;

  auto vSkip = pSkips.begin();
  for (size_t vI = 0; vI < pTokens.size();) {
    if (vSkip != pSkips.end() && vI == vSkip->first) {
      vI += vSkip->second;
      ++vSkip;
      continue;
    }
    for (const auto & vD : lDelimiters) {
      if (vI + vD.lTokens.size() > pTokens.size()) {
        continue;
      }
      if (std::equal(vD.lTokens.begin(), vD.lTokens.end(), pTokens.begin() + vI)) {
        lMatches.emplace_back(vD.vRole, vI);
        break;
      }
    }
    vI++;
  }

  lMatches.emplace_back(cCommonChatRoleUnknown, pTokens.size());

  common_chat_msg_spans vSpans;
  for (size_t vI = 0; vI + 1 < lMatches.size(); vI++) {
    const auto & vCurr = lMatches[vI];
    const auto & vNext = lMatches[vI + 1];
    vSpans.mAdd(vCurr.first, vCurr.second, vNext.second - vCurr.second);
  }

  return vSpans;
}

json common_chat_msg::mToJsonOaicompat(bool pConcatTypedText) const {
  if (!vContent.empty() && !lContentParts.empty()) {
    throw std::runtime_error("Cannot specify both content and content_parts");
  }
  json vJmsg{
    { "role", vRole },
  };
  if (!vContent.empty()) {
    vJmsg["content"] = vContent;
  } else if (!lContentParts.empty()) {
    if (pConcatTypedText || mContainsMedia()) {
      std::string vText;
      bool        vLastWasMediaMarker = false;
      // join parts with newline, do not add newline before or after media markers
      for (const auto & vPart : lContentParts) {
        bool vAddNewLine = true;
        if (vPart.type == "text") {
          vAddNewLine         = !vLastWasMediaMarker && !vText.empty();
          vLastWasMediaMarker = false;
        } else if (vPart.type == "media_marker") {
          vAddNewLine         = false;
          vLastWasMediaMarker = true;
        } else {
          LOG_WRN("Ignoring content part type: %s\n", vPart.type.c_str());
          continue;
        }

        if (vAddNewLine) {
          vText += '\n';
        }

        vText += vPart.vText;
      }
      vJmsg["content"] = vText;
    } else {
      auto & vParts = vJmsg["content"] = json::mArray();
      for (const auto & vPart : lContentParts) {
        vParts.push_back({
          { "type", vPart.type  },
          { "text", vPart.vText },
        });
      }
    }
  } else {
    vJmsg["content"] = "";
  }
  if (!vReasoningContent.empty()) {
    vJmsg["reasoning_content"] = vReasoningContent;
  }
  if (!vToolName.empty()) {
    vJmsg["name"] = vToolName;
  }
  if (!vToolCallId.empty()) {
    vJmsg["tool_call_id"] = vToolCallId;
  }
  if (!lToolCalls.empty()) {
    vJmsg["tool_calls"] = json::mArray();
    auto & vJtoolCalls  = vJmsg["tool_calls"];
    for (const auto & vToolCall : lToolCalls) {
      json vTc{
        { "type",     "function" },
        { "function",
         {
            { "name", vToolCall.name },
            { "arguments", json(vToolCall.vArguments) },
          }                      },
      };
      if (!vToolCall.vId.empty()) {
        vTc["id"] = vToolCall.vId;
      }
      // Some templates generate and require an id (sometimes in a very specific format, e.g. Mistral Nemo).
      // We only generate a random id for the ones that don't generate one by themselves
      // (they also won't get to see it as their template likely doesn't use it, so it's all for the client)
      // {"id", tc.id.empty() ? gen_tool_call_id() : tc.id},
      vJtoolCalls.push_back(vTc);
    }
  }

  return vJmsg;
}

std::vector<common_chat_msg_diff> common_chat_msg_diff::mComputeDiffs(const common_chat_msg & pMsgPrv,
                                                                      const common_chat_msg & pMsgNew) {
  std::vector<common_chat_msg_diff> lDiffs;
  if (pMsgNew.lToolCalls.size() > pMsgPrv.lToolCalls.size()) {
    lDiffs.reserve(pMsgNew.lToolCalls.size() - pMsgPrv.lToolCalls.size() + 3);
  } else {
    lDiffs.reserve(3);
  }

  // TODO: these can become expensive for long messages - how to optimize?
  if (pMsgPrv.vReasoningContent != pMsgNew.vReasoningContent) {
    auto & vDiff                 = lDiffs.emplace_back();
    vDiff.vReasoningContentDelta = fStringDiff(pMsgPrv.vReasoningContent, pMsgNew.vReasoningContent);
  }
  if (pMsgPrv.vContent != pMsgNew.vContent) {
    auto & vDiff        = lDiffs.emplace_back();
    vDiff.vContentDelta = fStringDiff(pMsgPrv.vContent, pMsgNew.vContent);
  }

  if (pMsgNew.lToolCalls.size() < pMsgPrv.lToolCalls.size()) {
    std::string vErr = "Invalid diff: now finding less tool calls!\n";
    vErr += "  Previous (" + std::to_string(pMsgPrv.lToolCalls.size()) + "):\n";
    for (const auto & vTc : pMsgPrv.lToolCalls) {
      vErr += "    - name: '" + vTc.name + "', args: '" + vTc.vArguments + "'\n";
    }
    vErr += "  Current (" + std::to_string(pMsgNew.lToolCalls.size()) + "):\n";
    for (const auto & vTc : pMsgNew.lToolCalls) {
      vErr += "    - name: '" + vTc.name + "', args: '" + vTc.vArguments + "'\n";
    }
    vErr += "  Current msg text content:\n" + pMsgNew.vContent + "\n";
    throw std::runtime_error(vErr);
  }

  if (!pMsgPrv.lToolCalls.empty()) {
    const auto   cIdx  = pMsgPrv.lToolCalls.size() - 1;
    const auto & vPref = pMsgPrv.lToolCalls[cIdx];
    const auto & vNewf = pMsgNew.lToolCalls[cIdx];
    // Allow tool name to change during incremental parsing:
    // - empty -> non-empty (initial discovery)
    // - prefix -> longer string (name grows as more input is parsed)
    if (vPref.name != vNewf.name && !vPref.name.empty() && !vNewf.name.empty()) {
      // Check if one is a prefix of the other (for incremental parsing where names grow or shrink)
      bool vIsPrefix = (vNewf.name.rfind(vPref.name, 0) == 0);
      if (!vIsPrefix) {
        LOG_ERR("Tool call mismatch: prev='%s' new='%s'\n", vPref.name.c_str(), vNewf.name.c_str());
        throw std::runtime_error("Invalid diff: tool call mismatch!");
      }
    }
    const auto cArgsDiff = fStringDiff(vPref.vArguments, vNewf.vArguments);
    if (!cArgsDiff.empty() || vPref.vId != vNewf.vId || vPref.name != vNewf.name) {
      auto & vDiff         = lDiffs.emplace_back();
      vDiff.vToolCallIndex = cIdx;
      if (vPref.vId != vNewf.vId || vPref.name != vNewf.name) {
        vDiff.vToolCallDelta.vId  = vNewf.vId;
        vDiff.vToolCallDelta.name = vNewf.name;
      }
      vDiff.vToolCallDelta.vArguments = cArgsDiff;
    }
  }
  for (size_t vIdx = pMsgPrv.lToolCalls.size(); vIdx < pMsgNew.lToolCalls.size(); ++vIdx) {
    auto & vDiff         = lDiffs.emplace_back();
    vDiff.vToolCallIndex = vIdx;
    vDiff.vToolCallDelta = pMsgNew.lToolCalls[vIdx];
  }

  return lDiffs;
}

using chat_template_caps = jinja::caps;

struct common_chat_templates {
  bool vAddBos;
  bool vAddEos;
  bool vHasExplicitTemplate;  // Model had builtin template or template overridden was specified.
  std::unique_ptr<common_chat_template> vTemplateDefault;  // always set (defaults to chatml)
  std::unique_ptr<common_chat_template> vTemplateToolUse;
};

common_chat_tool_choice fCommonChatToolChoiceParseOaicompat(const std::string & pToolChoice) {
  if (pToolChoice == "auto") {
    return cCommonChatToolChoiceAuto;
  }
  if (pToolChoice == "none") {
    return cCommonChatToolChoiceNone;
  }
  if (pToolChoice == "required") {
    return cCommonChatToolChoiceRequired;
  }
  throw std::invalid_argument("Invalid tool_choice: " + pToolChoice);
}

bool fCommonChatTemplatesSupportEnableThinking(const common_chat_templates * pChatTemplates) {
  common_chat_templates_inputs vInputs;
  vInputs.vReasoningFormat = cCommonReasoningFormatDeepseek;
  common_chat_msg vMsg;
  vMsg.vRole                   = "user";
  vMsg.vContent                = "test";
  vInputs.lMessages            = { vMsg };
  vInputs.vEnableThinking      = true;
  vInputs.vAddGenerationPrompt = true;
  vInputs.vReasoningFormat     = cCommonReasoningFormatDeepseek;

  auto vParams = fCommonChatTemplatesApply(pChatTemplates, vInputs);
  return vParams.vSupportsThinking;
}

std::vector<common_chat_msg> fCommonChatMsgsParseOaicompat(const json & pMessages) {
  std::vector<common_chat_msg> lMsgs;

  try {
    if (!pMessages.is_array()) {
      throw std::invalid_argument("Expected 'messages' to be an array, got " + pMessages.mDump());
    }

    for (const auto & vMessage : pMessages) {
      if (!vMessage.is_object()) {
        throw std::invalid_argument("Expected 'message' to be an object, got " + vMessage.mDump());
      }

      common_chat_msg vMsg;
      if (!vMessage.contains("role")) {
        throw std::invalid_argument("Missing 'role' in message: " + vMessage.mDump());
      }
      vMsg.vRole = vMessage.at("role");

      auto vHasContent   = vMessage.contains("content");
      auto vHasToolCalls = vMessage.contains("tool_calls");
      if (vHasContent) {
        const auto & vContent = vMessage.at("content");
        if (vContent.is_string()) {
          vMsg.vContent = vContent;
        } else if (vContent.is_array()) {
          for (const auto & vPart : vContent) {
            if (!vPart.contains("type")) {
              throw std::invalid_argument("Missing content part type: " + vPart.mDump());
            }
            const auto & type = vPart.at("type");
            if (type != "text" && type != "media_marker") {
              throw std::invalid_argument("Unsupported content part type: " + type.mDump());
            }
            common_chat_msg_content_part vMsgPart;
            vMsgPart.type  = type;
            vMsgPart.vText = vPart.at("text");
            vMsg.lContentParts.push_back(vMsgPart);
          }
        } else if (!vContent.is_null()) {
          throw std::invalid_argument("Invalid 'content' type: expected string or array, got " + vContent.mDump() +
                                      " (ref: doc/MANUAL.md)");
        }
      }
      if (vHasToolCalls) {
        for (const auto & vToolCall : vMessage.at("tool_calls")) {
          common_chat_tool_call vTc;
          if (!vToolCall.contains("type")) {
            throw std::invalid_argument("Missing tool call type: " + vToolCall.mDump());
          }
          const auto & type = vToolCall.at("type");
          if (type != "function") {
            throw std::invalid_argument("Unsupported tool call type: " + vToolCall.mDump());
          }
          if (!vToolCall.contains("function")) {
            throw std::invalid_argument("Missing tool call function: " + vToolCall.mDump());
          }
          const auto & vFc = vToolCall.at("function");
          if (!vFc.contains("name")) {
            throw std::invalid_argument("Missing tool call name: " + vToolCall.mDump());
          }
          vTc.name           = vFc.at("name");
          const auto & vArgs = vFc.at("arguments");
          if (vArgs.is_string()) {
            vTc.vArguments = vArgs;
          } else {
            vTc.vArguments = vArgs.mDump();
          }
          if (vToolCall.contains("id")) {
            vTc.vId = vToolCall.at("id");
          }
          vMsg.lToolCalls.push_back(vTc);
        }
      }
      if (!vHasContent && !vHasToolCalls) {
        throw std::invalid_argument(
          "Expected 'content' or 'tool_calls' (ref: doc/MANUAL.md & "
          "doc/MANUAL.md)");
      }
      if (vMessage.contains("reasoning_content")) {
        vMsg.vReasoningContent = vMessage.at("reasoning_content");
      }
      if (vMessage.contains("name")) {
        vMsg.vToolName = vMessage.at("name");
      }
      if (vMessage.contains("tool_call_id")) {
        vMsg.vToolCallId = vMessage.at("tool_call_id");
      }

      lMsgs.push_back(vMsg);
    }
  } catch (const std::exception & vE) {
    // @ngxson : disable otherwise it's bloating the API response
    // printf("%s\n", std::string("; messages = ") + messages.dump(2));
    throw std::runtime_error("Failed to parse messages: " + std::string(vE.what()));
  }

  return lMsgs;
}

struct messages_inp_normalizer {
  const jinja::caps & caps;

  messages_inp_normalizer(const jinja::caps & pC) : caps(pC) {}

  // handle supports_string_content / supports_typed_content
  // if string=true and array=false, convert array to string
  // if string=false and array=true, convert string to array
  // if both are true, do nothing
  json mNormalize(const json & pMessages) {
    bool vOnlyString = caps.vSupportsStringContent && !caps.vSupportsTypedContent;
    bool vOnlyTyped  = !caps.vSupportsStringContent && caps.vSupportsTypedContent;
    if ((!vOnlyString && !vOnlyTyped) || !pMessages.is_array()) {
      return pMessages;
    }
    json vNormalized = json::mArray();
    for (const auto & vMsg : pMessages) {
      json copy = vMsg;
      if (copy.contains("content")) {
        json & vIt = copy.at("content");
        if (vOnlyTyped && vIt.is_string()) {
          vIt = json::mArray({
            json{
                 { "type", "text" },
                 { "text", vIt.get<std::string>() },
                 }
          });
        } else if (vOnlyString && vIt.is_array()) {
          vIt = mConcatContentParts(vIt);
        }
      }
      vNormalized.push_back(std::move(copy));
    }
    return vNormalized;
  }

  // join parts with newline, do not add newline before or after media markers
  static std::string mConcatContentParts(const json & pParts) {
    std::string vText;
    bool        vLastWasMediaMarker = false;
    for (const auto & vPart : pParts) {
      std::string type        = vPart.value("type", "");
      bool        vAddNewLine = true;
      if (type == "text") {
        vAddNewLine         = !vLastWasMediaMarker && !vText.empty();
        vLastWasMediaMarker = false;
      } else if (type == "media_marker") {
        vAddNewLine         = false;
        vLastWasMediaMarker = true;
      } else {
        LOG_WRN("Ignoring content part type: %s\n", type.c_str());
        continue;
      }

      if (vAddNewLine) {
        vText += '\n';
      }

      vText += vPart.value("text", "");
    }
    return vText;
  }
};

static json fRenderMessageToJson(const std::vector<common_chat_msg> & pMsgs, const jinja::caps & pC) {
  if (!pC.vSupportsStringContent && !pC.vSupportsTypedContent) {
    LOG_WRN(
      "%s: Neither string content nor typed content is supported by the template. This is unexpected and may lead to "
      "issues.\n",
      __func__);
  }

  json vMessages = json::mArray();
  for (const auto & vMsg : pMsgs) {
    vMessages.push_back(vMsg.mToJsonOaicompat(/* concat_typed_text= */ false));
  }
  return messages_inp_normalizer(pC).mNormalize(vMessages);
}

// DEPRECATED: only used in tests
json fCommonChatMsgsToJsonOaicompat(const std::vector<common_chat_msg> & pMsgs, bool pConcatTypedText) {
  jinja::caps vC;
  vC.vSupportsStringContent = true;
  vC.vSupportsTypedContent  = !pConcatTypedText;
  return fRenderMessageToJson(pMsgs, vC);
}

json fCommonChatToolsToJsonOaicompat(const std::vector<common_chat_tool> & pTools) {
  if (pTools.empty()) {
    return json();
  }

  auto vResult = json::mArray();
  for (const auto & vTool : pTools) {
    vResult.push_back({
      { "type",     "function" },
      { "function",
       {
          { "name", vTool.name },
          { "description", vTool.vDescription },
          { "parameters", json::mParse(vTool.vParameters) },
        }                      },
    });
  }
  return vResult;
}

json fCommonChatToolParameters(const json & pFunction) {
  if (pFunction.contains("parameters")) {
    const auto & vParams = pFunction.at("parameters");
    if (!vParams.is_null() && !(vParams.is_object() && vParams.empty())) {
      return vParams;
    }
  }
  return json{
    { "type",       "object"        },
    { "properties", json::mObject() }
  };
}

std::vector<common_chat_tool> fCommonChatToolsParseOaicompat(const json & pTools) {
  std::vector<common_chat_tool> lResult;

  try {
    if (!pTools.is_null()) {
      if (!pTools.is_array()) {
        throw std::invalid_argument("Expected 'tools' to be an array, got " + pTools.mDump());
      }
      for (const auto & vTool : pTools) {
        if (!vTool.contains("type")) {
          throw std::invalid_argument("Missing tool type: " + vTool.mDump());
        }
        const auto & type = vTool.at("type");
        if (!type.is_string() || type != "function") {
          throw std::invalid_argument("Unsupported tool type: " + vTool.mDump());
        }
        if (!vTool.contains("function")) {
          throw std::invalid_argument("Missing tool function: " + vTool.mDump());
        }

        const auto & vFunction = vTool.at("function");
        lResult.push_back({
          /* .name = */ vFunction.at("name"),
          /* .description = */ vFunction.value("description", ""),
          /* .parameters = */ vFunction.value("parameters", json::mObject()).mDump(),
        });
      }
    }
  } catch (const std::exception & vE) {
    throw std::runtime_error("Failed to parse tools: " + std::string(vE.what()) + "; tools = " + pTools.mDump(2));
  }

  return lResult;
}

common_chat_continuation fCommonChatContinuationParse(const common_json & value) {
  if (value.is_null()) {
    return cCommonChatContinuationNone;
  }
  if (value.is_boolean()) {
    return value.get<bool>() ? cCommonChatContinuationAuto : cCommonChatContinuationNone;
  }
  if (value.is_string()) {
    auto vValueStr = value.get<std::string>();
    if (vValueStr == "reasoning_content") {
      return cCommonChatContinuationReasoning;
    }
    if (vValueStr == "content") {
      return cCommonChatContinuationContent;
    }
  }
  throw std::invalid_argument(
    "Invalid continue_final_message: expected a boolean, \"content\" or \"reasoning_content\"");
}

bool fCommonChatVerifyTemplate(const std::string & pTmpl, bool pUseJinja) {
  GGML_UNUSED(pUseJinja);
  {
    try {
      common_chat_msg vMsg;
      vMsg.vRole    = "user";
      vMsg.vContent = "test";

      auto vTmpls = fCommonChatTemplatesInit(/* model= */ nullptr, pTmpl);

      common_chat_templates_inputs vInputs;
      vInputs.lMessages = { vMsg };

      fCommonChatTemplatesApply(vTmpls.get(), vInputs);
      return true;
    } catch (const std::exception & vE) {
      LOG_ERR("%s: failed to apply template: %s\n", __func__, vE.what());
      return false;
    }
  }
  return false;
}

std::string fCommonChatFormatSingle(const struct common_chat_templates * pTmpls,
                                    const std::vector<common_chat_msg> & pPastMsg,
                                    const common_chat_msg &              pNewMsg,
                                    bool                                 pAddAss,
                                    bool                                 pUseJinja) {
  common_chat_templates_inputs vInputs;
  vInputs.vUseJinja = pUseJinja;
  vInputs.vAddBos   = pTmpls->vAddBos;
  vInputs.vAddEos   = pTmpls->vAddEos;

  std::string vFmtPastMsg;
  if (!pPastMsg.empty()) {
    vInputs.lMessages            = pPastMsg;
    vInputs.vAddGenerationPrompt = false;
    vFmtPastMsg                  = fCommonChatTemplatesApply(pTmpls, vInputs).vPrompt;
  }
  std::ostringstream vSs;
  // if the past_msg ends with a newline, we must preserve it in the formatted version
  if (pAddAss && !vFmtPastMsg.empty() && vFmtPastMsg.back() == '\n') {
    vSs << "\n";
  };
  // format chat with new_msg
  vInputs.lMessages.push_back(pNewMsg);
  vInputs.vAddGenerationPrompt = pAddAss;
  auto vFmtNewMsg              = fCommonChatTemplatesApply(pTmpls, vInputs).vPrompt;
  // get the diff part
  vSs << vFmtNewMsg.substr(vFmtPastMsg.size(), vFmtNewMsg.size() - vFmtPastMsg.size());
  return vSs.str();
}

std::string fCommonChatFormatExample(const struct common_chat_templates *       pTmpls,
                                     bool                                       pUseJinja,
                                     const std::map<std::string, std::string> & pChatTemplateKwargs) {
  common_chat_templates_inputs vInputs;
  vInputs.vUseJinja           = pUseJinja;
  vInputs.vAddBos             = pTmpls->vAddBos;
  vInputs.vAddEos             = pTmpls->vAddEos;
  vInputs.dChatTemplateKwargs = pChatTemplateKwargs;
  auto vAddSimpleMsg          = [&](auto pRole, auto pContent) {
    common_chat_msg vMsg;
    vMsg.vRole    = pRole;
    vMsg.vContent = pContent;
    vInputs.lMessages.push_back(vMsg);
  };
  vAddSimpleMsg("system", "You are a helpful assistant");
  vAddSimpleMsg("user", "Hello");
  vAddSimpleMsg("assistant", "Hi there");
  vAddSimpleMsg("user", "How are you?");
  return fCommonChatTemplatesApply(pTmpls, vInputs).vPrompt;
}

void fCommonChatTemplatesFree(struct common_chat_templates * pTmpls) {
  delete pTmpls;
}

bool fCommonChatTemplatesWasExplicit(const struct common_chat_templates * pTmpls) {
  return pTmpls->vHasExplicitTemplate;
}

common_chat_prompt_preset fCommonChatGetAsrPrompt(const common_chat_templates * pChatTemplates) {
  GGML_UNUSED(pChatTemplates);
  common_chat_prompt_preset vAsrPreset;
  vAsrPreset.vSystem = "";
  vAsrPreset.vUser   = "Transcribe audio to text";

  return vAsrPreset;
}

std::string fCommonChatTemplatesSource(const struct common_chat_templates * pTmpls, const std::string & pVariant) {
  if (!pVariant.empty()) {
    if (pVariant == "tool_use") {
      if (pTmpls->vTemplateToolUse) {
        return pTmpls->vTemplateToolUse->mSource();
      }
      return "";
    }
    LOG_DBG("%s: unknown template variant: %s\n", __func__, pVariant.c_str());
  }
  return pTmpls->vTemplateDefault->mSource();
}

common_chat_templates_ptr fCommonChatTemplatesInit(const struct dgs_model * pModel,
                                                   const std::string &      pChatTemplateOverride,
                                                   const std::string &      pBosTokenOverride,
                                                   const std::string &      pEosTokenOverride) {
  std::string vDefaultTemplateSrc;
  std::string vTemplateToolUseSrc;

  bool vHasExplicitTemplate = !pChatTemplateOverride.empty();
  if (pChatTemplateOverride.empty()) {
    GGML_ASSERT(pModel != nullptr);
    const auto * str = fDgsModelChatTemplate(pModel, /* name */ nullptr);
    if (str) {
      vDefaultTemplateSrc  = str;
      vHasExplicitTemplate = true;
    }
    str = fDgsModelChatTemplate(pModel, /* name */ "tool_use");
    if (str) {
      vTemplateToolUseSrc  = str;
      vHasExplicitTemplate = true;
    }
  } else {
    vDefaultTemplateSrc = pChatTemplateOverride;
  }
  if (vDefaultTemplateSrc.empty()) {
    throw std::runtime_error(
      "The selected Qwen model requires a Jinja chat template in GGUF metadata or --chat-template-file");
  }

  std::string vTokenBos = pBosTokenOverride;
  std::string vTokenEos = pEosTokenOverride;
  bool        vAddBos   = false;
  bool        vAddEos   = false;
  if (pModel) {
    const auto * vVocab    = fDgsModelGetVocab(pModel);
    const auto   cGetToken = [&](dgs_token token, const char * name, const char * pJinjaVariableName) {
      if (token == DGS_TOKEN_NULL) {
        if (vDefaultTemplateSrc.find(pJinjaVariableName) != std::string::npos ||
            vTemplateToolUseSrc.find(pJinjaVariableName) != std::string::npos) {
          LOG_WRN(
            "common_chat_templates_init: warning: vocab does not have a %s token, jinja template won't "
            "work as intended.\n",
            name);
        }
        return std::string();
      }
      return fCommonTokenToPiece(vVocab, token, true);
    };
    vTokenBos = cGetToken(fDgsVocabBos(vVocab), "BOS", "bos_token");
    vTokenEos = cGetToken(fDgsVocabEos(vVocab), "EOS", "eos_token");
    vAddBos   = fDgsVocabGetAddBos(vVocab);
    vAddEos   = fDgsVocabGetAddEos(vVocab);
  }
  common_chat_templates_ptr vTmpls(new common_chat_templates());
  vTmpls->vHasExplicitTemplate = vHasExplicitTemplate;
  vTmpls->vAddBos              = vAddBos;
  vTmpls->vAddEos              = vAddEos;
  try {
    vTmpls->vTemplateDefault = std::make_unique<common_chat_template>(vDefaultTemplateSrc, vTokenBos, vTokenEos);
  } catch (const std::exception & vE) {
    LOG_ERR("%s: error: %s\n", __func__, vE.what());
    LOG_ERR("%s: failed to initialize chat template\n", __func__);
    LOG_ERR("%s: provide a valid Jinja template for the selected Qwen model\n", __func__);
    throw vE;
  }
  if (!vTemplateToolUseSrc.empty()) {
    try {
      vTmpls->vTemplateToolUse = std::make_unique<common_chat_template>(vTemplateToolUseSrc, vTokenBos, vTokenEos);
    } catch (const std::exception & vE) {
      LOG_ERR("%s: failed to parse tool use chat template (ignoring it): %s\n", __func__, vE.what());
    }
  }
  return vTmpls;
}

const char * fCommonChatFormatName(common_chat_format format) {
  switch (format) {
    case cCommonChatFormatContentOnly:
      return "Content-only";
    case cCommonChatFormatPegSimple:
      return "peg-simple";
    case cCommonChatFormatPegNative:
      return "peg-native";
    case cCommonChatFormatPegGemma4:
      return "peg-gemma4";
    case cCommonChatFormatPegMinimaxM3:
      return "peg-minimax-m3";
    default:
      throw std::runtime_error("Unknown chat format");
  }
}

const char * fCommonReasoningFormatName(common_reasoning_format format) {
  switch (format) {
    case cCommonReasoningFormatNone:
      return "none";
    case cCommonReasoningFormatAuto:
      return "auto";
    case cCommonReasoningFormatDeepseek:
      return "deepseek";
    default:
      throw std::runtime_error("Unknown reasoning format");
  }
}

common_reasoning_format fCommonReasoningFormatFromName(const std::string & format) {
  if (format == "none") {
    return cCommonReasoningFormatNone;
  }
  if (format == "auto") {
    return cCommonReasoningFormatAuto;
  }
  if (format == "deepseek") {
    return cCommonReasoningFormatDeepseek;
  }

  throw std::runtime_error("Unknown reasoning format: " + format);
}

std::string fCommonChatTemplateDirectApplyImpl(const common_chat_template &          pTmpl,
                                               const autoparser::generation_params & pInputs,
                                               const std::optional<json> &           pMessagesOverride,
                                               const std::optional<json> &           pToolsOverride,
                                               const std::optional<json> &           pAdditionalContext) {
  jinja::context vCtx(pTmpl.mSource());

  // messages_override is already built for this template, do not touch its content parts
  json vInp = json{
    { "messages",        pMessagesOverride.has_value() ?
                    *pMessagesOverride :
                    messages_inp_normalizer(pTmpl.mOriginalCaps()).mNormalize(pInputs.vMessages) },
    { "bos_token",       pTmpl.mBosToken()                                                                        },
    { "eos_token",       pTmpl.mEosToken()                                                                        },
    { "enable_thinking", pInputs.vEnableThinking                                                                  },
  };
  if (pToolsOverride.has_value() || !pInputs.vTools.empty()) {
    vInp["tools"] = pToolsOverride.has_value() ? *pToolsOverride : pInputs.vTools;
  }
  if (pInputs.vExtraContext.is_object()) {
    // TODO: do we need to merge, or replacing is fine?
    for (const auto & [k, v] : pInputs.vExtraContext.mItems()) {
      vInp[k] = v;
    }
  }
  if (pAdditionalContext.has_value()) {
    // TODO: merge properly instead of overwriting (matching old behavior)
    for (const auto & [k, v] : pAdditionalContext->mItems()) {
      vInp[k] = v;
    }
  }
  if (pInputs.vAddGenerationPrompt) {
    vInp["add_generation_prompt"] = true;
  }
  if (vInp.contains("preserve_reasoning") && vInp["preserve_reasoning"].is_boolean()) {
    bool vEnabled = vInp["preserve_reasoning"].get<bool>();
    jinja::fCapsApplyPreserveReasoning(vCtx, vEnabled);
  }
  if (vInp.contains("reasoning_effort") && vInp["reasoning_effort"].is_string() && !vInp["reasoning_effort"].empty()) {
    std::string vReasoningEffort = vInp["reasoning_effort"].get<std::string>();
    jinja::fCapsApplyReasoningEffort(vCtx, vReasoningEffort);
  }

  jinja::fGlobalFromJson(vCtx, vInp, pInputs.vMarkInput);

  // render
  jinja::runtime     vRuntime(vCtx);
  const jinja::value cResults = vRuntime.mExecute(pTmpl.prog);
  auto               vParts   = jinja::runtime::mGatherStringParts(cResults);

  std::string vResult = vParts->mAsString().str();

  // TODO: improve this later
  if (pInputs.vAddBos && fStringStartsWith(vResult, pTmpl.mBosToken())) {
    vResult = vResult.substr(pTmpl.mBosToken().size());
  }
  if (pInputs.vAddEos && fStringEndsWith(vResult, pTmpl.mEosToken())) {
    vResult = vResult.substr(0, vResult.size() - pTmpl.mEosToken().size());
  }
  return vResult;
}

std::string fCommonChatTemplateDirectApply(const common_chat_template &          pTmpl,
                                           const autoparser::generation_params & pInputs) {
  return fCommonChatTemplateDirectApplyImpl(pTmpl, pInputs, std::nullopt, std::nullopt, std::nullopt);
}

std::string fCommonChatTemplateGenerationPromptImpl(const common_chat_template &          pTmpl,
                                                    const autoparser::generation_params & pInputs,
                                                    const std::optional<json> &           pMessagesOverride,
                                                    const std::optional<json> &           pToolsOverride,
                                                    const std::optional<json> &           pAdditionalContext) {
  autoparser::generation_params vParams = pInputs;
  vParams.vAddGenerationPrompt          = false;
  vParams.vContinueFinalMessage         = cCommonChatContinuationNone;
  std::string vNoGenPrompt =
    fCommonChatTemplateDirectApplyImpl(pTmpl, vParams, pMessagesOverride, pToolsOverride, pAdditionalContext);
  vParams.vAddGenerationPrompt = true;
  std::string vGenPrompt =
    fCommonChatTemplateDirectApplyImpl(pTmpl, vParams, pMessagesOverride, pToolsOverride, pAdditionalContext);

  size_t vPrefixLen = 0;
  size_t vMinSize   = std::min(vNoGenPrompt.size(), vGenPrompt.size());
  while (vPrefixLen < vMinSize && vNoGenPrompt[vPrefixLen] == vGenPrompt[vPrefixLen]) {
    vPrefixLen++;
  }
  return vGenPrompt.substr(vPrefixLen);
}

std::string fCommonChatTemplateGenerationPrompt(const common_chat_template &          pTmpl,
                                                const autoparser::generation_params & pInputs) {
  return fCommonChatTemplateGenerationPromptImpl(pTmpl, pInputs, std::nullopt, std::nullopt, std::nullopt);
}

namespace workaround {

static void fMapDeveloperRoleToSystem(json & pMessages) {
  for (auto & vMessage : pMessages) {
    if (vMessage.contains("role")) {
      if (vMessage["role"] == "developer") {
        vMessage["role"] = "system";
      }
    }
  }
}

// if first message is system and template does not support it, merge it with next message
static void fSystemMessageNotSupported(json & pMessages) {
  if (!pMessages.empty() && pMessages.front().at("role") == "system") {
    if (pMessages.size() > 1) {
      LOG_DBG("Merging system prompt into next message\n");
      auto & vFirstMsg  = pMessages.front();
      auto & vSecondMsg = pMessages[1];
      vSecondMsg["content"] =
        vFirstMsg.at("content").get<std::string>() + "\n" + vSecondMsg.at("content").get<std::string>();
      pMessages.erase(0);
    } else {
      LOG_WRN("Removing system prompt due to template not supporting system role\n");
      pMessages.erase(0);
    }
  }
}

static void fRequiresNonNullContent(json & pMessages) {
  GGML_ASSERT(pMessages.is_array());
  for (auto & vMessage : pMessages) {
    if (vMessage.contains("tool_calls") && !vMessage.contains("content")) {
      vMessage["content"] = "";
    }
  }
}

static void fFuncArgsNotString(json & pMessages) {
  GGML_ASSERT(pMessages.is_array());
  for (auto & vMessage : pMessages) {
    if (vMessage.contains("tool_calls")) {
      for (auto & vToolCall : vMessage["tool_calls"]) {
        if (vToolCall.contains("function") && vToolCall["function"].contains("arguments")) {
          auto & vArgs = vToolCall["function"]["arguments"];
          if (vArgs.is_string()) {
            try {
              vArgs = json::mParse(vArgs.get<std::string>());
            } catch (const std::exception & vE) {
              throw std::runtime_error("Failed to parse tool call arguments as JSON: " + std::string(vE.what()));
            }
          }
        }
      }
    }
  }
}

// Trim leading/trailing whitespace from message contents before rendering. This
// has to run on the messages (not on the rendered JSON) because templates with
// string-only content caps concatenate typed content parts into a single string
// during rendering, after which the per-part whitespace can no longer be reached.
// Both the plain string content and the text of typed content parts are trimmed.
static void fTrimAllContent(std::vector<common_chat_msg> & pMessages) {
  for (auto & vMessage : pMessages) {
    vMessage.vContent          = fTrimWhitespace(vMessage.vContent);
    vMessage.vReasoningContent = fTrimWhitespace(vMessage.vReasoningContent);
    for (auto & vPart : vMessage.lContentParts) {
      if (vPart.type == "text") {
        vPart.vText = fTrimWhitespace(vPart.vText);
      }
    }
  }
}

}  // namespace workaround

static json fCommonChatExtraContext() {
  json                                  vCtx         = json::mObject();
  std::chrono::system_clock::time_point vNow         = std::chrono::system_clock::now();
  std::string                           vDatetimeStr = fFormatTime(vNow, "%b %d %Y");
  std::string                           vDateStr     = fFormatTime(vNow, "%d %b %Y");
  vCtx["datetime"]                                   = vDatetimeStr;
  vCtx["date_string"]                                = vDateStr;
  return vCtx;
}

// Older gemma4 templates need their tool responses rewritten before rendering

// Pick the dedicated handler for a template from its source, or null for the autoparser.
// Order matters: the first match wins, and later checks assume the earlier ones did not match.
static common_chat_params_init_fn fCommonChatTemplateDetectParamsInit(const std::string & pSrc) {
  if (pSrc.find("<tool_call>") != std::string::npos && pSrc.find("<function=") != std::string::npos &&
      pSrc.find("<parameter=") != std::string::npos) {
    return fCommonChatParamsInitQwen3Coder;
  }
  return nullptr;
}

common_chat_template::common_chat_template(const std::string & pSrc,
                                           const std::string & pBosToken,
                                           const std::string & pEosToken) {
  jinja::lexer vLexer;
  auto         vLexerRes = vLexer.mTokenize(pSrc);
  this->prog             = jinja::fParseFromTokens(vLexerRes);

  this->src     = vLexerRes.vSource;
  this->bos_tok = pBosToken;
  this->eos_tok = pEosToken;

  this->caps = jinja::fCapsGet(prog);
  // LOG_INF("%s: caps:\n%s\n", __func__, this->caps.to_string().c_str());

  this->params_init = fCommonChatTemplateDetectParamsInit(this->src);
  if (this->params_init) {
    return;
  }

  // The analysis depends only on the template, so run it once here instead of on every apply.
  // A failure is kept for apply to report, so a bad template still loads like it did before.
  try {
    analysis = std::make_unique<autoparser::autoparser>();
    analysis->mAnalyzeTemplate(*this);
  } catch (const std::exception & vE) {
    analysis.reset();
    analysis_error = vE.what();
  }
}

common_chat_template::~common_chat_template()                                   = default;
common_chat_template::common_chat_template(common_chat_template &&)             = default;
common_chat_template & common_chat_template::operator=(common_chat_template &&) = default;

std::optional<common_chat_params> fCommonChatTrySpecializedTemplate(const common_chat_template &          pTmpl,
                                                                    const autoparser::generation_params & pParams) {
  if (!pTmpl.params_init) {
    return std::nullopt;
  }
  return pTmpl.params_init(pTmpl, pParams);
}

static common_chat_params fCommonChatTemplatesApplyJinja(const struct common_chat_templates *        pTmpls,
                                                         const struct common_chat_templates_inputs & pInputs) {
  autoparser::generation_params vParams;
  vParams.vTools = fCommonChatToolsToJsonOaicompat(pInputs.lTools);
  const auto & vTmpl =
    vParams.vTools.is_array() && pTmpls->vTemplateToolUse ? *pTmpls->vTemplateToolUse : *pTmpls->vTemplateDefault;
  const auto &                         vSrc = vTmpl.mSource();
  const auto &                         caps = vTmpl.mOriginalCaps();
  std::vector<common_chat_msg>         lTrimmedMessages;
  const std::vector<common_chat_msg> * lMessagesToRender = &pInputs.lMessages;
  if (vSrc.find("You have access to the following functions in JSONSchema format") != std::string::npos) {
    // StepFun: trim message contents (including typed content parts) before rendering,
    // otherwise leftover whitespace drives the model into reasoning loops (issue #24181)
    lTrimmedMessages = pInputs.lMessages;
    workaround::fTrimAllContent(lTrimmedMessages);
    lMessagesToRender = &lTrimmedMessages;
  }
  vParams.vMessages            = fRenderMessageToJson(*lMessagesToRender, vTmpl.mOriginalCaps());
  vParams.vToolChoice          = pInputs.vToolChoice;
  vParams.vReasoningFormat     = pInputs.vReasoningFormat;
  vParams.vEnableThinking      = pInputs.vEnableThinking;
  vParams.vGrammar             = pInputs.vGrammar;
  vParams.vNow                 = pInputs.vNow;
  vParams.vAddGenerationPrompt = pInputs.vAddGenerationPrompt;
  vParams.vAddBos              = pTmpls->vAddBos;
  vParams.vAddEos              = pTmpls->vAddEos;

  vParams.vContinueFinalMessage = pInputs.vContinueFinalMessage;
  if (vParams.vContinueFinalMessage != cCommonChatContinuationNone) {
    vParams.vAddGenerationPrompt = false;

    if (!pInputs.lMessages.empty()) {
      // Render messages[:-1] and store continuation message separately
      vParams.vContinueMsg = pInputs.lMessages.back();
      vParams.vMessages.erase(vParams.vMessages.size() - 1);
    }

    if (vParams.vContinueFinalMessage == cCommonChatContinuationAuto && !pInputs.lMessages.empty()) {
      // Resolve based on message content
      vParams.vContinueFinalMessage = cCommonChatContinuationContent;
      if (!vParams.vContinueMsg.vReasoningContent.empty() && vParams.vContinueMsg.vContent.empty() &&
          vParams.vContinueMsg.lContentParts.empty()) {
        vParams.vContinueFinalMessage = cCommonChatContinuationReasoning;
      }
    }
  }

  if (vSrc.find("<|channel|>") == std::string::npos) {
    // map developer to system for all models except for GPT-OSS
    workaround::fMapDeveloperRoleToSystem(vParams.vMessages);
  }

  if (!vTmpl.mOriginalCaps().vSupportsSystemRole) {
    workaround::fSystemMessageNotSupported(vParams.vMessages);
  }

  if (vTmpl.mOriginalCaps().vSupportsToolCalls) {
    // some templates will require the content field in tool call messages
    // to still be non-null, this puts an empty string everywhere where the
    // content field is null
    workaround::fRequiresNonNullContent(vParams.vMessages);
  }

  if (vTmpl.mOriginalCaps().vSupportsObjectArguments) {
    workaround::fFuncArgsNotString(vParams.vMessages);
  }

  vParams.vExtraContext = fCommonChatExtraContext();
  for (auto vEl : pInputs.dChatTemplateKwargs) {
    vParams.vExtraContext[vEl.first] = json::mParse(vEl.second);
  }

  if (!pInputs.vJsonSchema.empty()) {
    vParams.vJsonSchema = json::mParse(pInputs.vJsonSchema);
  }

  vParams.vParallelToolCalls = pInputs.vParallelToolCalls;

  if (vParams.vTools.is_array()) {
    if (vParams.vToolChoice != cCommonChatToolChoiceNone && !vParams.vGrammar.empty()) {
      throw std::runtime_error("Cannot specify grammar with tools");
    }
    if (caps.vSupportsToolCalls && !caps.vSupportsTools) {
      LOG_WRN(
        "Template supports tool calls but does not natively describe tools. The fallback behaviour used may "
        "produce bad results, inspect prompt w/ --verbose & consider overriding the template.\n");
    }
  }

  if (pInputs.vForcePureContent) {
    LOG_WRN("Forcing pure content template, will not render reasoning or tools separately.");
    // Create the result structure
    common_chat_params data;
    auto               vParamsCopy = vParams;
    vParamsCopy.vReasoningFormat   = cCommonReasoningFormatNone;
    data.vPrompt                   = fCommonChatTemplateDirectApplyImpl(vTmpl, vParamsCopy);
    data.vGenerationPrompt         = fCommonChatTemplateGenerationPromptImpl(vTmpl, vParams);
    data.format                    = cCommonChatFormatPegNative;
    data.vParser                   = fBuildChatPegParser(
      [&data](common_chat_peg_builder & pP) { return pP.mLiteral(data.vGenerationPrompt) << pP.mContent(pP.mRest()); });
    return data;
  }

  if (auto vResult = fCommonChatTrySpecializedTemplate(vTmpl, vParams)) {
    return *vResult;
  }

  if (!vTmpl.analysis) {
    throw std::invalid_argument("Unable to generate parser for this template. Automatic parser generation failed: " +
                                vTmpl.analysis_error);
  }

  try {
    LOG_DBG("%s: using differential autoparser\n", __func__);
    const auto & vAutoparser = *vTmpl.analysis;
    auto         vAutoParams = autoparser::peg_generator::mGenerateParser(vTmpl, vParams, vAutoparser);

    common_chat_msg_delimiters vDelimiters;
    if (!vAutoparser.vAssistantStart.empty()) {
      vDelimiters.mAdd(cCommonChatRoleAssistant, vAutoparser.vAssistantStart);
    }
    if (!vAutoparser.vUserStart.empty()) {
      vDelimiters.mAdd(cCommonChatRoleUser, vAutoparser.vUserStart);
    }

    vAutoParams.vMessageDelimiters = std::move(vDelimiters);

    vAutoParams.vSupportsThinking = vAutoparser.vReasoning.vMode != autoparser::reasoning_mode::cNone;
    if (vAutoParams.vSupportsThinking) {
      vAutoParams.vThinkingStartTag = fTrimWhitespace(vAutoparser.vReasoning.vStart);
      auto vEndTag                  = fTrimWhitespace(vAutoparser.vReasoning.end);
      if (!vEndTag.empty()) {
        vAutoParams.lThinkingEndTags = { std::move(vEndTag) };
      }
    }
    const auto & vArena = vAutoParams.vParser;
    LOG_DBG("%s: generated parser:\n%s\n\nparser generation prompt: %s\n", __func__,
            vArena.mDump(vArena.mRoot()).c_str(), vAutoParams.vGenerationPrompt.c_str());
    return vAutoParams;
  } catch (const std::exception & vE) {
    throw std::invalid_argument(
      std::string("Unable to generate parser for this template. Automatic parser generation failed: ") + vE.what());
  }
}

// Legacy template route (adhoc C++ implementation of known templates), forward to dgs_chat_apply_template.

common_chat_params fCommonChatTemplatesApply(const struct common_chat_templates *        pTmpls,
                                             const struct common_chat_templates_inputs & pInputs) {
  GGML_ASSERT(pTmpls != nullptr);
  return fCommonChatTemplatesApplyJinja(pTmpls, pInputs);
}

void common_chat_input::mAppend(const std::string & pPiece, dgs_token token) {
  if (pPiece.empty()) {
    return;
  }
  lTokens.push_back(token);
  lTokens.resize(lTokens.size() + pPiece.size() - 1, DGS_TOKEN_NULL);
  vText += pPiece;
}

void common_chat_input::mAppend(const common_chat_input & pChunk) {
  lTokens.insert(lTokens.end(), pChunk.lTokens.begin(), pChunk.lTokens.end());
  vText += pChunk.vText;
}

void common_chat_input::mTruncate(size_t pPos) {
  if (pPos < vText.size()) {
    vText.erase(pPos);
    lTokens.resize(pPos);
  }
}

common_chat_input common_chat_input::substr(size_t pPos, size_t n) const {
  common_chat_input vOut;
  vOut.vText = vText.substr(pPos, n);
  vOut.lTokens.assign(lTokens.begin() + pPos, lTokens.begin() + pPos + vOut.size());
  return vOut;
}

void common_chat_input::mPrepend(const std::string & pPrefix) {
  lTokens.insert(lTokens.begin(), pPrefix.size(), DGS_TOKEN_NULL);
  vText = pPrefix + vText;
}

void common_chat_input::mPrepend(const common_chat_input & pPrefix) {
  lTokens.insert(lTokens.begin(), pPrefix.lTokens.begin(), pPrefix.lTokens.end());
  vText = pPrefix.vText + vText;
}

common_chat_input fCommonChatInputTokenize(const dgs_vocab * pVocab, const std::string & pText) {
  common_chat_input vInput;
  auto              lTokens = fCommonTokenize(pVocab, pText, false, true);
  for (size_t vI = 0; vI < lTokens.size(); vI++) {
    std::string vPiece = fCommonTokenToPiece(pVocab, lTokens[vI], true);
    if (vI == 0 && std::isspace(vPiece[0]) && !std::isspace(pText[0])) {
      // Some tokenizers will add a space before the first special token, need to exclude
      continue;
    }
    vInput.mAppend(vPiece, lTokens[vI]);
  }
  if (vInput.vText != pText) {
    // the pieces do not give back the same text, keep the text without tokens
    return common_chat_input(pText);
  }
  return vInput;
}

common_chat_msg fCommonChatParse(const common_chat_input &         pInput,
                                 bool                              pIsPartial,
                                 const common_chat_parser_params & pParams) {
  return fCommonChatPegParse(pParams.vParser, pInput, pIsPartial, pParams);
}

common_chat_msg fCommonChatPegParse(const common_peg_arena &          pSrcParser,
                                    const common_chat_input &         pInput,
                                    bool                              pIsPartial,
                                    const common_chat_parser_params & pParams) {
  // both branches must be lvalues, a temporary here would copy the arena on every call
  static const common_peg_arena cContentOnly =
    fBuildChatPegParser([](common_chat_peg_builder & pP) { return pP.mContent(pP.mRest()) + pP.end(); });
  const common_peg_arena & vParser = pSrcParser.empty() ? cContentOnly : pSrcParser;

  if (pSrcParser.empty()) {
    LOG_DBG("No parser definition detected, assuming pure content parser.");
  }

  common_chat_input vEffectiveInput = pInput;
  vEffectiveInput.mPrepend(pParams.vGenerationPrompt);

  //LOG_DBG("Parsing PEG input with format %s: %s\n", common_chat_format_name(params.format), effective_input.text.c_str());

  common_peg_parse_flags vFlags = cCommonPegParseFlagLenient;
  if (pParams.vDebug) {
    vFlags |= cCommonPegParseFlagDebug;
  }

  common_peg_parse_context vCtx(std::move(vEffectiveInput.vText), std::move(vEffectiveInput.lTokens), vFlags);
  auto                     vResult = vParser.mParse(vCtx);

  if (vResult.mFail()) {
    // During partial parsing, return partial results if any AST nodes were captured
    // This allows streaming to work correctly for formats like FUNC_MARKDOWN_CODE_BLOCK
    if (pIsPartial && vResult.end > 0) {
      // Try to extract any partial results from what was successfully parsed
      common_chat_msg vMsg;
      vMsg.vRole = "assistant";
      std::unique_ptr<common_chat_peg_mapper> vMapper;
      if (pParams.format == cCommonChatFormatPegGemma4) {
        vMapper = std::make_unique<common_chat_peg_gemma4_mapper>(vMsg);
      } else if (pParams.format == cCommonChatFormatPegMinimaxM3) {
        vMapper = std::make_unique<common_chat_peg_minimax_m3_mapper>(vMsg);
      } else {
        vMapper = std::make_unique<common_chat_peg_mapper>(vMsg);
      }
      vMapper->mFromAst(vCtx.vAst, vResult);

      if (vCtx.mIsDebug()) {
        fprintf(stderr, "\nAST for partial parse (fail):\n%s\n", vCtx.vAst.mDump().c_str());
        fflush(stderr);
      }
      return vMsg;
    }
    LOG_WRN("%s: unparsed %s output: %s\n", __func__, fCommonChatFormatName(pParams.format),
            vCtx.vInput.substr(vResult.end).c_str());
    LOG_DBG("%s: full %s output triggering error:\n=== BEGIN ===\n%s\n=== END ===\n", __func__,
            fCommonChatFormatName(pParams.format), vCtx.vInput.c_str());
    throw std::runtime_error(std::string("The model produced output that does not match the expected ") +
                             fCommonChatFormatName(pParams.format) + " format");
  }

  common_chat_msg vMsg;
  vMsg.vRole = "assistant";

  std::unique_ptr<common_chat_peg_mapper> vMapper;
  if (pParams.format == cCommonChatFormatPegGemma4) {
    vMapper = std::make_unique<common_chat_peg_gemma4_mapper>(vMsg);
  } else if (pParams.format == cCommonChatFormatPegMinimaxM3) {
    vMapper = std::make_unique<common_chat_peg_minimax_m3_mapper>(vMsg);
  } else {
    vMapper = std::make_unique<common_chat_peg_mapper>(vMsg);
  }
  vMapper->mFromAst(vCtx.vAst, vResult);

  if (vCtx.mIsDebug()) {
    fprintf(stderr, "\nAST for %s parse:\n%s\n", pIsPartial ? "partial" : "full", vCtx.vAst.mDump().c_str());
    fflush(stderr);
  }

  if (!pIsPartial) {
    LOG_DBG("Parsed message: %s\n", fCommonChatMsgsToJsonOaicompat({ vMsg }).at(0).mDump().c_str());
  }
  return vMsg;
}

common_chat_session::common_chat_session(const common_chat_templates *        pTmpls,
                                         const dgs_vocab *                    pVocab,
                                         const common_chat_templates_inputs & pInputs,
                                         const common_chat_session_params &   pParams) {
  auto vApplied = fCommonChatTemplatesApply(pTmpls, pInputs);

  vTemplated    = true;
  vPromptText   = std::move(vApplied.vPrompt);
  vResult.vRole = "assistant";

  vGrammarText          = std::move(vApplied.vGrammar);
  vGrammarLazy          = vApplied.vGrammarLazy;
  lStops                = std::move(vApplied.lAdditionalStops);
  vGenerationPromptText = vApplied.vGenerationPrompt;
  vThinkingStart        = std::move(vApplied.vThinkingStartTag);
  lThinkingEnds         = std::move(vApplied.lThinkingEndTags);

  vParserParams.format            = vApplied.format;
  vParserParams.vGenerationPrompt = pVocab ? fCommonChatInputTokenize(pVocab, vApplied.vGenerationPrompt) :
                                             common_chat_input(vApplied.vGenerationPrompt);
  vParserParams.vDebug            = pParams.vDebug;
  vParserParams.vParser           = std::move(vApplied.vParser);

  vDelimiters = std::move(vApplied.vMessageDelimiters);

  if (pVocab) {
    common_params_sampling vResolved;
    vResolved.vGrammarLazy = vApplied.vGrammarLazy;
    fCommonSamplingAddPreservedTokens(vResolved, pVocab, vApplied.lPreservedTokens);
    fCommonSamplingAddGrammarTriggers(vResolved, pVocab, std::move(vApplied.lGrammarTriggers));
    sPreservedTokens = std::move(vResolved.sPreservedTokens);
    lGrammarTriggers = std::move(vResolved.lGrammarTriggers);

    vDelimiters.mTokenize(pVocab);
  } else {
    lGrammarTriggers = std::move(vApplied.lGrammarTriggers);
  }

  if (pInputs.vContinueFinalMessage != cCommonChatContinuationNone && !pParams.vEcho) {
    // start from the prefill so it is not emitted as part of the first delta
    vResult = fCommonChatParse(vInput, true, vParserParams);
  }
}

void common_chat_session::mApplySampling(common_params_sampling & pSampling) const {
  if (!vTemplated) {
    return;
  }
  if (!vGrammarText.empty()) {
    pSampling.vGrammar = { cCommonGrammarTypeToolCalls, vGrammarText };
  }
  pSampling.vGrammarLazy      = vGrammarLazy;
  pSampling.vGenerationPrompt = vGenerationPromptText;
  pSampling.sPreservedTokens.insert(sPreservedTokens.begin(), sPreservedTokens.end());
  pSampling.lGrammarTriggers.insert(pSampling.lGrammarTriggers.end(), lGrammarTriggers.begin(), lGrammarTriggers.end());
}

const common_chat_msg & common_chat_session::mFeed(const common_chat_input & pChunk) {
  GGML_ASSERT(!vFinished && "feed() after finish()");
  vInput.mAppend(pChunk);
  auto vMsg = fCommonChatParse(vInput, true, vParserParams);
  if (!vMsg.empty()) {
    vResult = std::move(vMsg);
  }
  return vResult;
}

const common_chat_msg & common_chat_session::mFinish(const common_chat_input & pChunk) {
  GGML_ASSERT(!vFinished && "finish() called twice");
  vFinished = true;
  vInput.mAppend(pChunk);
  auto vMsg = fCommonChatParse(vInput, false, vParserParams);
  if (!vMsg.empty()) {
    vResult = std::move(vMsg);
  }
  return vResult;
}

std::map<std::string, bool> fCommonChatTemplatesGetCaps(const common_chat_templates * pChatTemplates) {
  GGML_ASSERT(pChatTemplates != nullptr);
  GGML_ASSERT(pChatTemplates->vTemplateDefault != nullptr);
  if (pChatTemplates->vTemplateToolUse != nullptr) {
    // take the more expressive template when available
    return pChatTemplates->vTemplateToolUse->caps.mToMap();
  }
  return pChatTemplates->vTemplateDefault->caps.mToMap();
}
