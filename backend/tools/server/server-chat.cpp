#include "server-chat.h"

#include "server-common.h"

#include <sstream>

json fServerChatConvertResponsesToChatcmpl(const json & pResponseBody) {
  if (!pResponseBody.contains("input")) {
    throw std::invalid_argument("'input' is required");
  }
  if (!fJsonValue(pResponseBody, "previous_response_id", std::string{}).empty()) {
    throw std::invalid_argument("dgs.cpp does not support 'previous_response_id'.");
  }

  const json cInputValue   = pResponseBody.at("input");
  json       vChatcmplBody = pResponseBody;
  vChatcmplBody.erase("input");
  std::vector<json> lChatcmplMessages;

  if (pResponseBody.contains("instructions")) {
    lChatcmplMessages.push_back({
      { "role", "system" },
      { "content", fJsonValue(pResponseBody, "instructions", std::string()) },
    });
    vChatcmplBody.erase("instructions");
  }

  if (cInputValue.is_string()) {
    // #responses_create-input-text_input
    lChatcmplMessages.push_back({
      { "role",    "user"      },
      { "content", cInputValue },
    });
  } else if (cInputValue.is_array()) {
    // #responses_create-input-input_item_list

    static auto vExistsAndIsArray = [](const json & pJ, const char * pKey) -> bool {
      return pJ.contains(pKey) && pJ.at(pKey).is_array();
    };
    static auto vExistsAndIsString = [](const json & pJ, const char * pKey) -> bool {
      return pJ.contains(pKey) && pJ.at(pKey).is_string();
    };

    for (json vItem : cInputValue) {
      bool vMergePrev = !lChatcmplMessages.empty() && lChatcmplMessages.back().value("role", "") == "assistant";

      if (vExistsAndIsString(vItem, "content")) {
        // #responses_create-input-input_item_list-input_message-content-text_input
        // Only "Input message" contains item["content"]::string
        // After converting item["content"]::string to item["content"]::array,
        // we can treat "Input message" as sum of "Item-Input message" and "Item-Output message"
        vItem["content"] = json::mArray({
          json{ { "text", vItem.at("content") }, { "type", "input_text" } }
        });
      }

      if (vExistsAndIsArray(vItem, "content") && vExistsAndIsString(vItem, "role") &&
          (vItem.at("role") == "user" || vItem.at("role") == "system" || vItem.at("role") == "developer")) {
        // #responses_create-input-input_item_list-item-input_message
        std::vector<json> lChatcmplContent;

        for (const json & vInputItem : vItem.at("content")) {
          const std::string type = fJsonValue(vInputItem, "type", std::string());

          if (type == "input_text") {
            if (!vInputItem.contains("text")) {
              throw std::invalid_argument("'Input text' requires 'text'");
            }
            lChatcmplContent.push_back({
              { "text", vInputItem.at("text") },
              { "type", "text"                },
            });
          } else if (type == "input_image") {
            // While `detail` is marked as required,
            // it has default value("auto") and can be omitted.

            if (!vInputItem.contains("image_url")) {
              throw std::invalid_argument("'image_url' is required");
            }
            lChatcmplContent.push_back({
              { "image_url", json{ { "url", vInputItem.at("image_url") } } },
              { "type",      "image_url"                                   },
            });
          } else if (type == "input_file") {
            throw std::invalid_argument("'input_file' is not supported by dgscpp at this moment");
          } else {
            throw std::invalid_argument("'type' must be one of 'input_text', 'input_image', or 'input_file'");
          }
        }

        if (vItem.contains("type")) {
          vItem.erase("type");
        }
        if (vItem.contains("status")) {
          vItem.erase("status");
        }
        vItem["content"] = lChatcmplContent;

        lChatcmplMessages.push_back(vItem);
      } else if (vExistsAndIsString(vItem, "role") && vItem.at("role") == "assistant" &&
                 vExistsAndIsString(vItem, "type") && vItem.at("type") == "message") {
        // #responses_create-input-input_item_list-item-output_message
        auto vChatcmplContent = json::mArray();

        // Handle both string content and array content
        if (vItem.contains("content") && vItem.at("content").is_string()) {
          // String content - convert to text content part
          vChatcmplContent.push_back({
            { "text", vItem.at("content") },
            { "type", "text"              },
          });
        } else if (vExistsAndIsArray(vItem, "content")) {
          // Array content - process each item
          for (const auto & vOutputText : vItem.at("content")) {
            const std::string type = fJsonValue(vOutputText, "type", std::string());
            if (type == "output_text" || type == "input_text") {
              // Accept both output_text and input_text (string content gets converted to input_text)
              if (!vExistsAndIsString(vOutputText, "text")) {
                throw std::invalid_argument("'Output text' requires 'text'");
              }
              vChatcmplContent.push_back({
                { "text", vOutputText.at("text") },
                { "type", "text"                 },
              });
            } else if (type == "refusal") {
              if (!vExistsAndIsString(vOutputText, "refusal")) {
                throw std::invalid_argument("'Refusal' requires 'refusal'");
              }
              vChatcmplContent.push_back({
                { "refusal", vOutputText.at("refusal") },
                { "type",    "refusal"                 },
              });
            } else {
              throw std::invalid_argument("'type' must be one of 'output_text' or 'refusal'");
            }
          }
        }

        if (vMergePrev) {
          auto & vPrevMsg = lChatcmplMessages.back();
          if (!vExistsAndIsArray(vPrevMsg, "content")) {
            vPrevMsg["content"] = json::mArray();
          }
          auto & vPrevContent = vPrevMsg["content"];
          vPrevContent.insert(vChatcmplContent);
        } else {
          vItem.erase("status");
          vItem.erase("type");
          vItem["content"] = vChatcmplContent;
          lChatcmplMessages.push_back(vItem);
        }
      } else if (vExistsAndIsString(vItem, "arguments") && vExistsAndIsString(vItem, "call_id") &&
                 vExistsAndIsString(vItem, "name") && vExistsAndIsString(vItem, "type") &&
                 vItem.at("type") == "function_call") {
        // #responses_create-input-input_item_list-item-function_tool_call
        json vToolCall = {
          { "function",
           json{
              { "arguments", vItem.at("arguments") },
              { "name", vItem.at("name") },
            }                               },
          { "id",       vItem.at("call_id") },
          { "type",     "function"          },
        };

        if (vMergePrev) {
          auto & vPrevMsg = lChatcmplMessages.back();
          if (!vExistsAndIsArray(vPrevMsg, "tool_calls")) {
            vPrevMsg["tool_calls"] = json::mArray();
          }
          vPrevMsg["tool_calls"].push_back(vToolCall);
        } else {
          lChatcmplMessages.push_back(json{
            { "role",       "assistant"                 },
            { "tool_calls", json::mArray({ vToolCall }) }
          });
        }
      } else if (vExistsAndIsString(vItem, "call_id") &&
                 (vExistsAndIsString(vItem, "output") || vExistsAndIsArray(vItem, "output")) &&
                 vExistsAndIsString(vItem, "type") && vItem.at("type") == "function_call_output") {
        // #responses_create-input-input_item_list-item-function_tool_call_output
        if (vItem.at("output").is_string()) {
          lChatcmplMessages.push_back(json{
            { "content",      vItem.at("output")  },
            { "role",         "tool"              },
            { "tool_call_id", vItem.at("call_id") },
          });
        } else {
          json vChatcmplOutputs = vItem.at("output");
          for (json & vChatcmplOutput : vChatcmplOutputs) {
            if (!vChatcmplOutput.contains("type")) {
              throw std::invalid_argument("Output of tool call missing 'type' field");
            }
            const auto type = vChatcmplOutput.at("type");
            if (type != "input_text" && type != "input_image") {
              throw std::invalid_argument("Output of tool call should be 'Input text' or 'Input image'");
            }
            if (type == "input_text") {
              vChatcmplOutput["type"] = "text";
            } else if (type == "input_image") {
              if (!vChatcmplOutput.contains("image_url")) {
                throw std::invalid_argument("'image_url' is required");
              }
              vChatcmplOutput["type"]      = "image_url";
              vChatcmplOutput["image_url"] = json{
                { "url", vChatcmplOutput.at("image_url") }
              };
            }
          }
          lChatcmplMessages.push_back(json{
            { "content",      vChatcmplOutputs    },
            { "role",         "tool"              },
            { "tool_call_id", vItem.at("call_id") },
          });
        }
      } else if (vExistsAndIsArray(vItem, "summary") && vExistsAndIsString(vItem, "type") &&
                 vItem.at("type") == "reasoning") {
        // #responses_create-input-input_item_list-item-reasoning

        if (!vExistsAndIsArray(vItem, "content")) {
          throw std::invalid_argument("item['content'] is not an array");
        }
        if (vItem.at("content").empty()) {
          throw std::invalid_argument("item['content'] is empty");
        }
        if (!vExistsAndIsString(vItem.at("content")[0], "text")) {
          throw std::invalid_argument("item['content']['text'] is not a string");
        }

        if (vMergePrev) {
          auto & vPrevMsg               = lChatcmplMessages.back();
          vPrevMsg["reasoning_content"] = vItem.at("content")[0].at("text");
        } else {
          lChatcmplMessages.push_back(json{
            { "role",              "assistant"                       },
            { "content",           json::mArray()                    },
            { "reasoning_content", vItem.at("content")[0].at("text") },
          });
        }
      } else {
        throw std::invalid_argument("Cannot determine type of 'item'");
      }
    }
  } else {
    throw std::invalid_argument("'input' must be a string or array of objects");
  }

  vChatcmplBody["messages"] = lChatcmplMessages;

  if (pResponseBody.contains("tools")) {
    if (!pResponseBody.at("tools").is_array()) {
      throw std::invalid_argument("'tools' must be an array of objects");
    }
    std::vector<json> lChatcmplTools;
    for (json vRespTool : pResponseBody.at("tools")) {
      json vChatcmplTool;

      const std::string type = fJsonValue(vRespTool, "type", std::string());
      if (type != "function") {
        // Non-function Responses tools have no Chat Completions equivalent.
        SRV_WRN("unsupported Responses tool type '%s' skipped\n", type.c_str());
        continue;
      }
      vRespTool.erase("type");
      vChatcmplTool["type"] = "function";

      if (!vRespTool.contains("strict")) {
        vRespTool["strict"] = true;
      }
      vChatcmplTool["function"] = vRespTool;
      lChatcmplTools.push_back(vChatcmplTool);
    }
    vChatcmplBody.erase("tools");
    if (!lChatcmplTools.empty()) {
      vChatcmplBody["tools"] = lChatcmplTools;
    }
  }

  if (pResponseBody.contains("max_output_tokens")) {
    vChatcmplBody.erase("max_output_tokens");
    vChatcmplBody["max_tokens"] = pResponseBody["max_output_tokens"];
  }

  if (pResponseBody.contains("reasoning")) {
    // Only "effort" is handled so far
    const json & vReasoning = pResponseBody.at("reasoning");
    if (vReasoning.contains("effort")) {
      vChatcmplBody["reasoning_effort"] = vReasoning.at("effort");
    }
    vChatcmplBody.erase("reasoning");
  }

  return vChatcmplBody;
}

// Edits the cch section of an "x-anthropic-billing-header" system prompt.
// Does nothing to any other prompt.
//
// This is a claude message with a "cch=ef01a" attribute that breaks prefix caching.
// The cch stamp is a whitebox end-to-end integrity hint. It's not meaningful as a
// system prompt data, particularly to dgs.cpp, but its presence means the prefix
// cache will not get past it: It changes on each request.
//
// Reference: 
// Example header:
// ```
// x-anthropic-billing-header: cc_version=2.1.101.e51; cc_entrypoint=cli; cch=a5145;You are Claude Code, Anthropic's official CLI for Claude.
//                                                                            ^^^^^
// ```
static void fNormalizeAnthropicBillingHeader(std::string & pSystemText) {
  if (pSystemText.rfind("x-anthropic-billing-header:", 0) != 0) {
    return;
  }

  const size_t cHeaderPrefixLength = strlen("x-anthropic-billing-header:");
  const size_t cCchLength          = 5;
  const size_t cIndexCch           = pSystemText.find("cch=", cHeaderPrefixLength);
  if (cIndexCch == std::string::npos) {
    return;
  }

  const size_t cIndexReplace = cIndexCch + 4;
  if (cIndexReplace + cCchLength < pSystemText.length() && pSystemText[cIndexReplace + cCchLength] == ';') {
    for (size_t vI = 0; vI < cCchLength; ++vI) {
      pSystemText[cIndexReplace + vI] = 'f';
    }
  } else {
    LOG_ERR("anthropic string not as expected: %s", pSystemText.c_str());
  }
}

json fServerChatConvertAnthropicToOai(const json & pBody) {
  json vOaiBody;

  // Convert system prompt
  json vOaiMessages = json::mArray();
  auto vSystemParam = fJsonValue(pBody, "system", json());
  if (!vSystemParam.is_null()) {
    std::string vSystemContent;

    if (vSystemParam.is_string()) {
      vSystemContent = vSystemParam.get<std::string>();
      fNormalizeAnthropicBillingHeader(vSystemContent);
    } else if (vSystemParam.is_array()) {
      for (const auto & vBlock : vSystemParam) {
        if (fJsonValue(vBlock, "type", std::string()) == "text") {
          auto vSystemText = fJsonValue(vBlock, "text", std::string());
          fNormalizeAnthropicBillingHeader(vSystemText);
          vSystemContent += vSystemText;
        }
      }
    }

    vOaiMessages.push_back({
      { "role",    "system"       },
      { "content", vSystemContent }
    });
  }

  // Convert messages
  if (!pBody.contains("messages")) {
    throw std::runtime_error("'messages' is required");
  }
  const json & vMessages = pBody.at("messages");
  if (vMessages.is_array()) {
    for (const auto & vMsg : vMessages) {
      std::string vRole = fJsonValue(vMsg, "role", std::string());

      if (!vMsg.contains("content")) {
        if (vRole == "assistant") {
          continue;
        }
        vOaiMessages.push_back(vMsg);
        continue;
      }

      const json & vContent = vMsg.at("content");

      if (vContent.is_string()) {
        vOaiMessages.push_back(vMsg);
        continue;
      }

      if (!vContent.is_array()) {
        vOaiMessages.push_back(vMsg);
        continue;
      }

      json        vToolCalls        = json::mArray();
      json        vConvertedContent = json::mArray();
      json        vToolResults      = json::mArray();
      std::string vReasoningContent;
      bool        vHasToolCalls = false;

      for (const auto & vBlock : vContent) {
        std::string type = fJsonValue(vBlock, "type", std::string());

        if (type == "text") {
          vConvertedContent.push_back(vBlock);
        } else if (type == "thinking") {
          vReasoningContent += fJsonValue(vBlock, "thinking", std::string());
        } else if (type == "image") {
          json        vSource     = fJsonValue(vBlock, "source", json::mObject());
          std::string vSourceType = fJsonValue(vSource, "type", std::string());

          if (vSourceType == "base64") {
            std::string        vMediaType = fJsonValue(vSource, "media_type", std::string("image/jpeg"));
            std::string        data       = fJsonValue(vSource, "data", std::string());
            std::ostringstream vSs;
            vSs << "data:" << vMediaType << ";base64," << data;

            vConvertedContent.push_back({
              { "type",      "image_url"              },
              { "image_url", { { "url", vSs.str() } } }
            });
          } else if (vSourceType == "url") {
            std::string vUrl = fJsonValue(vSource, "url", std::string());
            vConvertedContent.push_back({
              { "type",      "image_url"         },
              { "image_url", { { "url", vUrl } } }
            });
          }
        } else if (type == "tool_use") {
          vToolCalls.push_back({
            { "id", fJsonValue(vBlock, "id", std::string()) },
            { "type", "function" },
            { "function",
             { { "name", fJsonValue(vBlock, "name", std::string()) },
                { "arguments", fJsonValue(vBlock, "input", json::mObject()).mDump() } } }
          });
          vHasToolCalls = true;
        } else if (type == "tool_result") {
          std::string vToolUseId = fJsonValue(vBlock, "tool_use_id", std::string());

          auto vResultContent = fJsonValue(vBlock, "content", json());
          if (vResultContent.is_string()) {
            vToolResults.push_back({
              { "role",         "tool"                            },
              { "tool_call_id", vToolUseId                        },
              { "content",      vResultContent.get<std::string>() }
            });
          } else if (vResultContent.is_array()) {
            // Single-pass: build both text and content_parts, decide format at the end
            std::string vResultText;
            json        vContentParts = json::mArray();
            bool        vHasImages    = false;

            for (const auto & vC : vResultContent) {
              std::string vCType = fJsonValue(vC, "type", std::string());
              if (vCType == "text") {
                std::string vText = fJsonValue(vC, "text", std::string());
                vResultText += vText;
                vContentParts.push_back({
                  { "type", "text" },
                  { "text", vText  }
                });
              } else if (vCType == "image") {
                vHasImages              = true;
                json        vSource     = fJsonValue(vC, "source", json::mObject());
                std::string vSourceType = fJsonValue(vSource, "type", std::string());
                if (vSourceType == "base64") {
                  std::string vMediaType = fJsonValue(vSource, "media_type", std::string("image/jpeg"));
                  std::string data       = fJsonValue(vSource, "data", std::string());
                  std::string vUrl       = "data:" + vMediaType + ";base64," + data;
                  vContentParts.push_back({
                    { "type",      "image_url"         },
                    { "image_url", { { "url", vUrl } } }
                  });
                } else if (vSourceType == "url") {
                  vContentParts.push_back({
                    { "type",      "image_url"                                              },
                    { "image_url", { { "url", fJsonValue(vSource, "url", std::string()) } } }
                  });
                }
              }
            }

            if (!vHasImages) {
              // Text-only: collapse to a plain string for maximum compatibility
              vToolResults.push_back({
                { "role",         "tool"      },
                { "tool_call_id", vToolUseId  },
                { "content",      vResultText }
              });
            } else {
              // Mixed or image-only: use array content parts (OpenAI multimodal tool format)
              vToolResults.push_back({
                { "role",         "tool"        },
                { "tool_call_id", vToolUseId    },
                { "content",      vContentParts }
              });
            }
          } else {
            vToolResults.push_back({
              { "role",         "tool"     },
              { "tool_call_id", vToolUseId },
              { "content",      ""         }
            });
          }
        }
      }

      if (!vConvertedContent.empty() || vHasToolCalls || !vReasoningContent.empty()) {
        json vNewMsg = {
          { "role", vRole }
        };
        if (!vConvertedContent.empty()) {
          vNewMsg["content"] = vConvertedContent;
        } else if (vHasToolCalls || !vReasoningContent.empty()) {
          vNewMsg["content"] = "";
        }
        if (!vToolCalls.empty()) {
          vNewMsg["tool_calls"] = vToolCalls;
        }
        if (!vReasoningContent.empty()) {
          vNewMsg["reasoning_content"] = vReasoningContent;
        }
        vOaiMessages.push_back(vNewMsg);
      }

      for (const auto & vToolMsg : vToolResults) {
        vOaiMessages.push_back(vToolMsg);
      }
    }
  }

  vOaiBody["messages"] = vOaiMessages;

  // Convert tools
  if (pBody.contains("tools")) {
    const json & vTools = pBody.at("tools");
    if (vTools.is_array()) {
      json vOaiTools = json::mArray();
      for (const auto & vTool : vTools) {
        vOaiTools.push_back({
          { "type",     "function"                                                                            },
          { "function",
           { { "name", fJsonValue(vTool, "name", std::string()) },
              { "description", fJsonValue(vTool, "description", std::string()) },
              { "parameters", vTool.contains("input_schema") ? vTool.at("input_schema") : json::mObject() } } }
        });
      }
      vOaiBody["tools"] = vOaiTools;
    }
  }

  // Convert tool_choice
  if (pBody.contains("tool_choice")) {
    const json & vTc = pBody.at("tool_choice");
    if (vTc.is_object()) {
      std::string type = fJsonValue(vTc, "type", std::string());
      if (type == "auto") {
        vOaiBody["tool_choice"] = "auto";
      } else if (type == "any" || type == "tool") {
        vOaiBody["tool_choice"] = "required";
      }
    }
  }

  // Convert stop_sequences to stop
  if (pBody.contains("stop_sequences")) {
    vOaiBody["stop"] = pBody.at("stop_sequences");
  }

  // Handle max_tokens (required in Anthropic, but we're permissive)
  if (pBody.contains("max_tokens")) {
    vOaiBody["max_tokens"] = pBody.at("max_tokens");
  } else {
    vOaiBody["max_tokens"] = 4096;
  }

  // Pass through common params
  for (const auto & vKey : { "temperature", "top_p", "top_k", "stream", "chat_template_kwargs" }) {
    if (pBody.contains(vKey)) {
      vOaiBody[vKey] = pBody.at(vKey);
    }
  }

  // Handle Anthropic-specific thinking param
  if (pBody.contains("thinking")) {
    json        vThinking     = fJsonValue(pBody, "thinking", json::mObject());
    std::string vThinkingType = fJsonValue(vThinking, "type", std::string());
    if (vThinkingType == "enabled") {
      int vBudgetTokens                  = fJsonValue(vThinking, "budget_tokens", 10000);
      vOaiBody["thinking_budget_tokens"] = vBudgetTokens;
    }
  }

  // Handle Anthropic-specific metadata param
  if (pBody.contains("metadata")) {
    json        vMetadata = fJsonValue(pBody, "metadata", json::mObject());
    std::string vUserId   = fJsonValue(vMetadata, "user_id", std::string());
    if (!vUserId.empty()) {
      vOaiBody["__metadata_user_id"] = vUserId;
    }
  }

  return vOaiBody;
}

json fServerChatMsgDiffToJsonOaicompat(const common_chat_msg_diff & pDiff) {
  json vDelta = json::mObject();
  if (!pDiff.vReasoningContentDelta.empty()) {
    vDelta["reasoning_content"] = pDiff.vReasoningContentDelta;
  }
  if (!pDiff.vContentDelta.empty()) {
    vDelta["content"] = pDiff.vContentDelta;
  }
  if (pDiff.vToolCallIndex != std::string::npos) {
    json vToolCall;
    vToolCall["index"] = pDiff.vToolCallIndex;
    if (!pDiff.vToolCallDelta.vId.empty()) {
      vToolCall["id"]   = pDiff.vToolCallDelta.vId;
      vToolCall["type"] = "function";
    }
    if (!pDiff.vToolCallDelta.name.empty() || !pDiff.vToolCallDelta.vArguments.empty()) {
      json vFunction = json::mObject();
      if (!pDiff.vToolCallDelta.name.empty()) {
        vFunction["name"] = pDiff.vToolCallDelta.name;
      }
      if (!pDiff.vToolCallDelta.vArguments.empty()) {
        vFunction["arguments"] = pDiff.vToolCallDelta.vArguments;
      }
      vToolCall["function"] = vFunction;
    }
    vDelta["tool_calls"] = json::mArray({ vToolCall });
  }
  return vDelta;
}

json fConvertTranscriptionsToChatcmpl(const json &                                 pInpBody,
                                      const common_chat_templates *                pTmpls,
                                      const std::map<std::string, uploaded_file> & pInFiles,
                                      std::vector<raw_buffer> &                    pOutFiles) {
  // TODO @ngxson : this function may need to be improved in the future
  // handle input files
  pOutFiles.clear();
  auto vIt = pInFiles.find("file");
  if (vIt != pInFiles.end()) {
    pOutFiles.push_back(vIt->second.data);
  } else {
    throw std::invalid_argument("No input file found for transcription");
  }

  // handle input data
  std::string vPrompt         = fJsonValue(pInpBody, "prompt", std::string());
  std::string vLanguage       = fJsonValue(pInpBody, "language", std::string());
  std::string vResponseFormat = fJsonValue(pInpBody, "response_format", std::string("json"));
  if (vResponseFormat != "json") {
    throw std::invalid_argument("Only 'json' response_format is supported for transcription");
  }
  const common_chat_prompt_preset cPreset = fCommonChatGetAsrPrompt(pTmpls);
  if (vPrompt.empty()) {
    vPrompt = cPreset.vUser;
  }
  if (!vLanguage.empty()) {
    vPrompt += fStringFormat(" (language: %s)", vLanguage.c_str());
  }
  vPrompt += fGetMediaMarker();

  json vMessages = json::mArray();
  if (!cPreset.vSystem.empty()) {
    vMessages.push_back({
      { "role",    "system"        },
      { "content", cPreset.vSystem }
    });
  }
  vMessages.push_back({
    { "role",    "user"  },
    { "content", vPrompt }
  });

  json vChatcmplBody        = pInpBody;  // copy all fields
  vChatcmplBody["messages"] = vMessages;

  // because input from form-data, everything is string, we need to correct the types here
  std::string stream      = fJsonValue(pInpBody, "stream", std::string("false"));
  vChatcmplBody["stream"] = stream == "true";

  if (pInpBody.contains("max_tokens")) {
    std::string vInp            = pInpBody["max_tokens"].get<std::string>();
    vChatcmplBody["max_tokens"] = std::stoul(vInp);
  }

  if (pInpBody.contains("temperature")) {
    std::string vInp             = pInpBody["temperature"].get<std::string>();
    vChatcmplBody["temperature"] = std::stof(vInp);
  }

  return vChatcmplBody;
}
