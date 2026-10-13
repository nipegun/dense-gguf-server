#include "chat-auto-parser-helpers.h"
#include "chat-auto-parser.h"
#include "chat-peg-parser.h"
#include "chat.h"
#include "common.h"
#include "log.h"
#include "peg-parser.h"

#include <algorithm>
#include <cctype>
#include <numeric>
#include <ostream>
#include <sstream>

#define ANSI_RESET  "\033[0m"
#define ANSI_PURPLE "\033[1m\x1b[38;5;126m"
#define ANSI_ORANGE "\033[1m\x1b[38;5;214m"
#define ANSI_RED    "\033[1m\x1b[38;5;196m"

using json = common_json;

namespace autoparser {

static const std::string cFunFirst        = "FFF_FIRST_FUN_F";
static const std::string cFunSecond       = "SSS_SECOND_FUN_S";
static const std::string cArgFirst        = "AA_ARG_FST_AA";
static const std::string cArgSecond       = "BB_ARG_SND_BB";
static const std::string cUserMsg         = "U_USER_MSG Hello END_U";
static const std::string cUserMsgTwo      = "V_USER_MSG Hello END_V";
static const std::string cAssistantMsg    = "A_ASST_MSG I can help END_A";
static const std::string cThinkingContent = "REASON_PART I am thinking END_R";
static const std::string cCallId001       = "call00001";
static const std::string cCallId002       = "call00002";
static const std::string cCallId999       = "call99999";

static std::vector<std::function<void(const common_chat_template & pTmpl, autoparser &)>> lWorkarounds({
  // Old reasoning Qwen templates - they don't really display reasoning content, but we still want to
  // support reasoning on them
  [](const common_chat_template & pTmpl, autoparser & pAnalysis) -> void {
    if (pTmpl.src.find("content.split('</think>')") != std::string::npos &&
        pTmpl.src.find("reasoning_content") == std::string::npos &&
        pTmpl.src.find("<SPECIAL_12>") == std::string::npos && pAnalysis.vReasoning.vMode == reasoning_mode::cNone) {
      pAnalysis.vReasoning.vMode  = reasoning_mode::cTagBased;
      pAnalysis.vReasoning.vStart = "<think>";
      pAnalysis.vReasoning.end    = "</think>";
      pAnalysis.lPreservedTokens.push_back("<think>");
      pAnalysis.lPreservedTokens.push_back("</think>");
      LOG_DBG(ANSI_ORANGE "[Patch: old Qwen/Deepseek thinking template]\n" ANSI_RESET);
    }
  },
  // Granite 3.3, with separate reasoning and content markers
  [](const common_chat_template & pTmpl, autoparser & pAnalysis) -> void {
    if (pTmpl.src.find("Write your thoughts between <think></think> and write your response between "
                       "<response></response>") != std::string::npos) {
      pAnalysis.vReasoning.vMode  = reasoning_mode::cTagBased;
      pAnalysis.vReasoning.vStart = "<think>";
      pAnalysis.vReasoning.end    = "</think>";
      pAnalysis.lPreservedTokens.push_back("<think>");
      pAnalysis.lPreservedTokens.push_back("</think>");
      pAnalysis.vContent.vMode  = content_mode::cWrappedWithReasoning;
      pAnalysis.vContent.vStart = "<response>";
      pAnalysis.vContent.end    = "</response>";
      pAnalysis.lPreservedTokens.push_back("<response>");
      pAnalysis.lPreservedTokens.push_back("</response>");
      LOG_DBG(ANSI_ORANGE "[Patch: Granite 3.3]\n" ANSI_RESET);
    }
  },
  // Cohere Command R+ - content wrapped in <|CHATBOT_TOKEN|>...<|END_OF_TURN_TOKEN|>
  [](const common_chat_template & pTmpl, autoparser & pAnalysis) -> void {
    if (pTmpl.src.find("<|CHATBOT_TOKEN|>") != std::string::npos &&
        pTmpl.src.find("<|END_OF_TURN_TOKEN|>") != std::string::npos && pAnalysis.vContent.vStart.empty()) {
      pAnalysis.vContent.vMode  = content_mode::cAlwaysWrapped;
      pAnalysis.vContent.vStart = "<|CHATBOT_TOKEN|>";
      pAnalysis.vContent.end    = "<|END_OF_TURN_TOKEN|>";
      pAnalysis.lPreservedTokens.push_back("<|CHATBOT_TOKEN|>");
      pAnalysis.lPreservedTokens.push_back("<|END_OF_TURN_TOKEN|>");
      pAnalysis.vUserStart = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>";
      LOG_DBG(ANSI_ORANGE "[Patch: Cohere Command R+]\n" ANSI_RESET);
    }
  },
  // Functionary - no tool call section delimiter
  [](const common_chat_template & pTmpl, autoparser & pAnalysis) -> void {
    if (pTmpl.src.find("set has_code_interpreter = tools | selectattr(\"type\", \"equalto\", "
                       "\"code_interpreter\") | list | length > 0") != std::string::npos) {
      pAnalysis.vContent.vMode               = content_mode::cPlain;
      pAnalysis.vContent.end                 = "";
      pAnalysis.vTools.vFunction.vNamePrefix = "";
      pAnalysis.vTools.format.vSectionStart  = "";
      pAnalysis.vTools.format.vSectionEnd    = "";
      pAnalysis.vTools.format.vPerCallStart  = "<function=";
      pAnalysis.vTools.format.vPerCallEnd    = "</function>";
      pAnalysis.vTools.vFunction.vClose      = "";
      pAnalysis.lPreservedTokens.clear();
      pAnalysis.lPreservedTokens.push_back("<|eot_id|>");
      pAnalysis.lPreservedTokens.push_back("<|eom_id|>");
      pAnalysis.lPreservedTokens.push_back("<function=");
      pAnalysis.lPreservedTokens.push_back(">");
      pAnalysis.lPreservedTokens.push_back("</function>");
      LOG_DBG(ANSI_ORANGE "[Patch: Functionary 3.1]\n" ANSI_RESET);
    }
  },
  // DeepSeek-R1-Distill-Qwen
  [](const common_chat_template & pTmpl, autoparser & pAnalysis) -> void {
    if (pTmpl.src.find(
          "{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>'") !=
        std::string::npos) {
      pAnalysis.vTools.format.vSectionStart  = "<｜tool▁calls▁begin｜>";
      pAnalysis.vTools.format.vSectionEnd    = "<｜tool▁calls▁end｜>";
      pAnalysis.vTools.format.vPerCallStart  = "<｜tool▁call▁begin｜>function";
      pAnalysis.vTools.vFunction.vNamePrefix = "<｜tool▁sep｜>";
      pAnalysis.vTools.format.vPerCallEnd    = "<｜tool▁call▁end｜>";
      pAnalysis.vTools.vFunction.vClose      = "```";
      LOG_DBG(ANSI_ORANGE "[Patch: DeepSeek-R1-Distill-Qwen]\n" ANSI_RESET);
    }
  },
  // Nemotron Nano v2
  [](const common_chat_template & pTmpl, autoparser & pAnalysis) -> void {
    if (pTmpl.src.find("<SPECIAL_10>") != std::string::npos && pTmpl.src.find("<SPECIAL_11>") != std::string::npos &&
        pTmpl.src.find("<SPECIAL_12>") != std::string::npos && pTmpl.src.find("<TOOL_RESPONSE>") != std::string::npos) {
      pAnalysis.vTools.format.vMode              = tool_format::cJsonNative;
      pAnalysis.vTools.format.vSectionStart      = "";
      pAnalysis.vTools.format.vSectionEnd        = "";
      pAnalysis.vTools.format.vPerCallStart      = "<TOOLCALL>";
      pAnalysis.vTools.format.vPerCallEnd        = "</TOOLCALL>";
      pAnalysis.vTools.format.vToolsArrayWrapped = true;
      pAnalysis.vContent.vMode                   = content_mode::cPlain;
      pAnalysis.vContent.vStart                  = "";
      pAnalysis.vContent.end                     = "";
      pAnalysis.vReasoning.vMode                 = reasoning_mode::cTagBased;
      pAnalysis.vReasoning.vStart                = "<think>\n";
      pAnalysis.vReasoning.end                   = "</think>";
      pAnalysis.vAssistantStart                  = "<SPECIAL_11>Assistant";
      pAnalysis.vUserStart                       = "<SPECIAL_11>User";
      pAnalysis.lPreservedTokens.clear();
      pAnalysis.lPreservedTokens.push_back("<SPECIAL_11>");
      pAnalysis.lPreservedTokens.push_back("</think>");
      pAnalysis.lPreservedTokens.push_back("<TOOLCALL>");
      pAnalysis.lPreservedTokens.push_back("</TOOLCALL>");
      LOG_DBG(ANSI_ORANGE "[Patch: Nemotron Nano v2]\n" ANSI_RESET);
    }
  },
  // Fireworks
  [](const common_chat_template & pTmpl, autoparser & pAnalysis) -> void {
    if (pTmpl.src.find("{%- set system_prompt = '<|start_header_id|>' + 'system' + '<|end_header_id|>\\n\\n'"
                       " + message['content'] | trim + '\\n' + system_prompt_suffix + '<|eot_id|>' -%}") !=
        std::string::npos) {
      pAnalysis.vAssistantStart = "<|start_header_id|>assistant<|end_header_id|>";
      pAnalysis.vUserStart      = "<|start_header_id|>user<|end_header_id|>";
      LOG_DBG(ANSI_ORANGE "[Patch: Fireworks v2]\n" ANSI_RESET);
    }
  },
  // Solar Open
  [](const common_chat_template & pTmpl, autoparser & pAnalysis) -> void {
    if (pTmpl.src.find("<|begin|>assistant<|think|><|end|>") != std::string::npos) {
      pAnalysis.vAssistantStart = "<|begin|>assistant";
      LOG_DBG(ANSI_ORANGE "[Patch: Solar Open]\n" ANSI_RESET);
    }
  },
  // Apriel 1.6
  [](const common_chat_template & pTmpl, autoparser & pAnalysis) -> void {
    if (pTmpl.src.find("if not loop.last and '[BEGIN FINAL RESPONSE]' in asst_text") != std::string::npos) {
      pAnalysis.vUserStart      = "<|begin_user|>";
      pAnalysis.vAssistantStart = "<|begin_assistant|>";
      LOG_DBG(ANSI_ORANGE "[Patch: Apriel 1.6]\n" ANSI_RESET);
    }
  },
  // template uses the JSON {name, parameters} tool instruction, emits the OpenAI function wrapper
  [](const common_chat_template & pTmpl, autoparser & pAnalysis) -> void {
    if (pTmpl.src.find("Respond in the format {\"name\": function name") != std::string::npos &&
        pTmpl.src.find("Do not use variables.") != std::string::npos) {
      pAnalysis.vTools.format.vOpenaiWrapperTrigger = true;
      LOG_DBG(ANSI_ORANGE "[Patch: JSON name/parameters tool instruction]\n" ANSI_RESET);
    }
  },
  // Laguna (poolside) - the v4 chat template renders reasoning and tool-arg
  // delimiters with formatting whitespace ("<think>\n", "</arg_value>\n") that
  // the model does not emit, so the inferred delimiters carry a spurious
  // newline and never match the model output. Trim to the bare tag. (v8
  // renders without the whitespace, so this is a no-op there.)
  [](const common_chat_template & pTmpl, autoparser & pAnalysis) -> void {
    if (pTmpl.src.find("laguna_glm_thinking") != std::string::npos) {
      pAnalysis.vReasoning.vStart              = fTrimWhitespace(pAnalysis.vReasoning.vStart);
      pAnalysis.vReasoning.end                 = fTrimWhitespace(pAnalysis.vReasoning.end);
      pAnalysis.vTools.vArguments.vValuePrefix = fTrimWhitespace(pAnalysis.vTools.vArguments.vValuePrefix);
      pAnalysis.vTools.vArguments.vValueSuffix = fTrimWhitespace(pAnalysis.vTools.vArguments.vValueSuffix);
      pAnalysis.vTools.vArguments.vSeparator   = fTrimWhitespace(pAnalysis.vTools.vArguments.vSeparator);
      pAnalysis.vTools.vArguments.vTolerateIntertagWhitespace = true;
      // The CONTROL/eot </assistant> token only halts generation when emitted as the
      // single token; after tool calls the model can spell it out as text tokens.
      // A literal stop string catches it either way.
      pAnalysis.lAdditionalStops.push_back("</assistant>");
      LOG_DBG(ANSI_ORANGE "[Patch: Laguna]\n" ANSI_RESET);
    }
  },
  // Bailing V3
  [](const common_chat_template & pTmpl, autoparser & pAnalysis) -> void {
    if (pTmpl.src.find("Bailing V3 chat template") != std::string::npos) {
      pAnalysis.vTools.vArguments.vValueSuffix = fTrimWhitespace(pAnalysis.vTools.vArguments.vValueSuffix);
      pAnalysis.vTools.vArguments.vTolerateIntertagWhitespace = true;
      LOG_DBG(ANSI_ORANGE "[Patch: Bailing V3]\n" ANSI_RESET);
    }
  },

});

// Common JSON structures
static json vParamsSchema = {
  { "type",       "object"                                                             },
  { "properties",
   { { cArgFirst, { { "type", "string" }, { "description", "First argument" } } },
      { cArgSecond, { { "type", "string" }, { "description", "Second argument" } } } } },
  { "required",   json::mArray({})                                                     }
};

static json vTools = json::mArray({
  { { "type", "function" },
   { "function",
      json{ { "name", cFunFirst }, { "description", "Test function foo" }, { "parameters", vParamsSchema } } }  },
  { { "type", "function" },
   { "function",
      json{ { "name", cFunSecond }, { "description", "Test function bar" }, { "parameters", vParamsSchema } } } }
});

static json vUserMsg = json{
  { "role",    "user"   },
  { "content", cUserMsg }
};

static json fBuildToolCall(const std::string & name, const json & pArgs, const std::string & pId = cCallId001) {
  return json{
    { "id",       pId                                              },
    { "type",     "function"                                       },
    { "function", json{ { "name", name }, { "arguments", pArgs } } }
  };
}

static json vFirstToolCallZeroArgs       = fBuildToolCall(cFunFirst, json::mObject(), cCallId001);
static json vFirstToolCallOneArg         = fBuildToolCall(cFunFirst,
                                                          {
                                                            { cArgFirst, "XXXX" }
},
                                                          cCallId001);
static json vFirstToolCallOneArgOtherVal = fBuildToolCall(cFunFirst,
                                                          {
                                                            { cArgFirst, "YYYY" }
},
                                                          cCallId001);
static json vFirstToolCallOtherArg       = fBuildToolCall(cFunFirst,
                                                          {
                                                            { cArgSecond, "YYYY" }
},
                                                          cCallId001);

static json vFirstToolCall      = fBuildToolCall(cFunFirst,
                                                 json{
                                                   { cArgFirst,  "XXXX" },
                                                   { cArgSecond, "YYYY" }
},
                                                 cCallId001);
static json vSecondToolCall     = fBuildToolCall(cFunSecond,
                                                 json{
                                                   { cArgFirst,  "XXXX" },
                                                   { cArgSecond, "YYYY" }
},
                                                 cCallId002);
static json vFirstToolCallAltId = fBuildToolCall(cFunFirst,
                                                 json{
                                                   { cArgFirst,  "XXXX" },
                                                   { cArgSecond, "YYYY" }
},
                                                 cCallId999);

template <typename T> static std::string fModeToStr(T pMode) {
  std::ostringstream vOs;
  vOs << pMode;
  return vOs.str();
}

void autoparser::mAnalyzeTemplate(const common_chat_template & pTmpl) {
  vJinjaCaps = pTmpl.mOriginalCaps();
  vReasoning = analyze_reasoning(pTmpl, vJinjaCaps.vSupportsToolCalls);
  vContent   = analyze_content(pTmpl, vReasoning);
  vTools =
    analyze_tools(vJinjaCaps.vSupportsToolCalls ? analyze_tools(pTmpl, vJinjaCaps, vReasoning) : analyze_tools());
  vAssistantStart = mDetectAssistantStartMarker(pTmpl);
  vUserStart      = mDetectUserStartMarker(pTmpl);
  mCollectPreservedTokens();

  for (auto & vWorkaround : lWorkarounds) {
    vWorkaround(pTmpl, *this);
  }

  LOG_DBG("\n--- Reasoning & Content Structure ---\n");
  LOG_DBG("user_msg_start: %s\n", vUserStart.c_str());
  LOG_DBG("assistant_msg_start: %s\n", vAssistantStart.c_str());
  LOG_DBG("reasoning_mode: %s\n", fModeToStr(vReasoning.vMode).c_str());
  LOG_DBG("reasoning_start: '%s'\n", vReasoning.vStart.c_str());
  LOG_DBG("reasoning_end: '%s'\n", vReasoning.end.c_str());
  LOG_DBG("content_mode: %s\n", fModeToStr(vContent.vMode).c_str());
  LOG_DBG("content_start: '%s'\n", vContent.vStart.c_str());
  LOG_DBG("content_end: '%s'\n", vContent.end.c_str());

  LOG_DBG("\n--- Tool Call Structure ---\n");
  LOG_DBG("tool_mode: %s\n", fModeToStr(vTools.format.vMode).c_str());
  LOG_DBG("supports_tools: %s\n", vJinjaCaps.vSupportsTools ? "true" : "false");
  LOG_DBG("supports_parallel_calls: %s\n", vJinjaCaps.vSupportsParallelToolCalls ? "true" : "false");
  LOG_DBG("tool_section_start: '%s'\n", vTools.format.vSectionStart.c_str());
  LOG_DBG("tool_section_end: '%s'\n", vTools.format.vSectionEnd.c_str());
  LOG_DBG("per_call_start: '%s'\n", vTools.format.vPerCallStart.c_str());
  LOG_DBG("per_call_end: '%s'\n", vTools.format.vPerCallEnd.c_str());
  LOG_DBG("func_name_prefix: '%s'\n", vTools.vFunction.vNamePrefix.c_str());
  LOG_DBG("func_name_suffix: '%s'\n", vTools.vFunction.vNameSuffix.c_str());
  LOG_DBG("func_args_separator: '%s'\n", vTools.vFunction.vArgsSeparator.c_str());
  LOG_DBG("func_close: '%s'\n", vTools.vFunction.vClose.c_str());
  LOG_DBG("call_id_prefix: '%s'\n", vTools.vCallId.vPrefix.c_str());
  LOG_DBG("call_id_suffix: '%s'\n", vTools.vCallId.vSuffix.c_str());
  LOG_DBG("call_id_pos: '%s'\n", fModeToStr(vTools.vCallId.vPos).c_str());
  LOG_DBG("args_start: '%s'\n", vTools.vArguments.vStart.c_str());
  LOG_DBG("args_end: '%s'\n", vTools.vArguments.end.c_str());
  LOG_DBG("arg_name_prefix: '%s'\n", vTools.vArguments.vNamePrefix.c_str());
  LOG_DBG("arg_name_suffix: '%s'\n", vTools.vArguments.vNameSuffix.c_str());
  LOG_DBG("arg_value_prefix: '%s'\n", vTools.vArguments.vValuePrefix.c_str());
  LOG_DBG("arg_value_suffix: '%s'\n", vTools.vArguments.vValueSuffix.c_str());
  LOG_DBG("name_field: '%s'\n", vTools.format.vNameField.c_str());
  LOG_DBG("args_field: '%s'\n", vTools.format.vArgsField.c_str());
  LOG_DBG("id_field: '%s'\n", vTools.format.vIdField.c_str());
  LOG_DBG("gen_id_field: '%s'\n", vTools.format.vGenIdField.c_str());
  LOG_DBG(
    "parameter_order: '%s'\n",
    std::accumulate(vTools.format.lParameterOrder.begin(), vTools.format.lParameterOrder.end(), std::string(""),
                    [](const std::string & pA, const std::string & pB) { return pA.empty() ? pB : pA + ", " + pB; })
      .c_str());

  LOG_DBG(ANSI_PURPLE "=== Differential analysis complete ===\n" ANSI_RESET);
  vAnalysisComplete = true;
}

void autoparser::mCollectPreservedTokens() {
  auto vAddToken = [this](const std::string & pOrgToken) {
    std::string token = fTrimWhitespace(pOrgToken);
    if (!token.empty()) {
      // Avoid duplicates
      if (std::find(lPreservedTokens.begin(), lPreservedTokens.end(), token) == lPreservedTokens.end()) {
        lPreservedTokens.push_back(token);
      }
    }
  };

  vAddToken(vReasoning.vStart);
  vAddToken(vReasoning.end);
  vAddToken(vContent.vStart);
  vAddToken(vContent.end);
  vAddToken(vTools.format.vSectionStart);
  vAddToken(vTools.format.vSectionEnd);
  vAddToken(vTools.format.vPerCallStart);
  vAddToken(vTools.format.vPerCallEnd);
  vAddToken(vTools.vFunction.vNamePrefix);
  vAddToken(vTools.vFunction.vNameSuffix);
  vAddToken(vTools.vFunction.vArgsSeparator);
  vAddToken(vTools.vFunction.vClose);
  vAddToken(vTools.vArguments.vStart);
  vAddToken(vTools.vArguments.end);
  vAddToken(vTools.vArguments.vNamePrefix);
  vAddToken(vTools.vArguments.vNameSuffix);
  vAddToken(vTools.vArguments.vSeparator);
  vAddToken(vTools.vArguments.vValuePrefix);
  vAddToken(vTools.vArguments.vValueSuffix);
  vAddToken(vTools.vCallId.vPrefix);
  vAddToken(vTools.vCallId.vSuffix);
}

std::string autoparser::mDetectAssistantStartMarker(const common_chat_template & pTmpl) {
  json vUserMsg = json{
    { "role",    "user"   },
    { "content", cUserMsg }
  };

  json vAssistantNoReasoning = json{
    { "role",    "assistant"   },
    { "content", cAssistantMsg }
  };

  template_params vParams;
  vParams.vMessages            = json::mArray({ vUserMsg });
  vParams.vAddGenerationPrompt = false;
  vParams.vEnableThinking      = true;

  auto vComparison = fCompareVariants(
    pTmpl, vParams, [&](template_params & pP) { pP.vMessages = json::mArray({ vUserMsg, vAssistantNoReasoning }); });

  if (!vComparison) {
    LOG_DBG(ANSI_ORANGE "%s: Template application failed, skipping assistant start detection\n" ANSI_RESET, __func__);
    return "";
  }

  auto vUsermsg = vComparison->vDiff.vRight;
  if (vUsermsg.find(cAssistantMsg) == std::string::npos) {
    LOG_DBG(ANSI_ORANGE
            "%s: Did not find assistant message in assistant message block, skipping detection\n" ANSI_RESET,
            __func__);
  }

  auto vAstPrefix = vUsermsg.substr(0, vUsermsg.find(cAssistantMsg));
  if (!vReasoning.vStart.empty() && vAstPrefix.find(fTrimWhitespace(vReasoning.vStart)) != std::string::npos) {
    vAstPrefix = vAstPrefix.substr(0, vAstPrefix.find(fTrimWhitespace(vReasoning.vStart)));
  }
  if (!vReasoning.end.empty() && vAstPrefix.find(fTrimWhitespace(vReasoning.end)) != std::string::npos) {
    vAstPrefix = vAstPrefix.substr(0, vAstPrefix.find(fTrimWhitespace(vReasoning.end)));
  }
  return fTrimWhitespace(vAstPrefix);
}

std::string autoparser::mDetectUserStartMarker(const common_chat_template & pTmpl) {
  json vUserMsg = json{
    { "role",    "user"   },
    { "content", cUserMsg }
  };

  json vAssistant = json{
    { "role",    "assistant"   },
    { "content", cAssistantMsg }
  };

  json vUserMsgTwo = json{
    { "role",    "user"      },
    { "content", cUserMsgTwo }
  };

  template_params vParams;
  vParams.vMessages            = json::mArray({});
  vParams.vAddGenerationPrompt = false;
  vParams.vEnableThinking      = true;

  auto vComparison =
    fCompareVariants(pTmpl, vParams, [&](template_params & pP) { pP.vMessages = json::mArray({ vUserMsg }); });

  if (!vComparison) {
    LOG_DBG(ANSI_ORANGE
            "%s: Template application failed, unsupported empty messages? trying complex variant\n" ANSI_RESET,
            __func__);
    vParams.vMessages = json::mArray({ vUserMsgTwo, vAssistant });
    vComparison       = fCompareVariants(pTmpl, vParams, [&](template_params & pP) {
      pP.vMessages = json::mArray({ vUserMsgTwo, vAssistant, vUserMsg });
    });
    if (!vComparison) {
      LOG_DBG(ANSI_ORANGE "%s: Template application failed for reserve variant, aborting\n" ANSI_RESET, __func__);
      return "";
    }
  }

  auto vUsermsg = vComparison->vDiff.vRight;
  if (vUsermsg.find(cUserMsg) == std::string::npos) {
    LOG_DBG(ANSI_ORANGE "%s: Did not find user message in user message block, aborting detection\n" ANSI_RESET,
            __func__);
  }

  if (vUsermsg.find(cAssistantMsg) != std::string::npos) {
    vUsermsg = vUsermsg.substr(vUsermsg.find(cAssistantMsg) + cAssistantMsg.size());
  }

  auto              vCandidate      = vUsermsg.substr(0, vUsermsg.find(cUserMsg));
  auto              lCandidateSplit = fSegmentizeMarkers(vCandidate);
  std::stringstream vResult;
  bool              vEncounteredMarker = false;
  for (const auto & vMrk : lCandidateSplit) {
    std::string vLowerMrk = std::string(vMrk.value);
    std::transform(vLowerMrk.begin(), vLowerMrk.end(), vLowerMrk.begin(),
                   [](unsigned char pC) { return std::tolower(pC); });
    // heuristic to weed out potential end markers, but only at the start
    if (vMrk.type == segment_type::cMarker && !vEncounteredMarker &&
        (vLowerMrk.find("end") != std::string::npos || vLowerMrk.find("close") != std::string::npos)) {
      continue;
    }
    if (vMrk.type == segment_type::cText && !vEncounteredMarker && fTrimWhitespace(vMrk.value).empty()) {
      continue;
    }
    vEncounteredMarker |= vMrk.type == segment_type::cMarker;
    vResult << vMrk.value;
  }
  return fTrimWhitespace(vResult.str());
}

analyze_reasoning::analyze_reasoning(const common_chat_template & pTmpl, bool pSupportsTools) : analyze_base(pTmpl) {
  LOG_DBG(ANSI_PURPLE "=== Starting differential analysis ===\n" ANSI_RESET);
  LOG_DBG(ANSI_ORANGE "Phase 1: Reasoning analysis\n" ANSI_RESET);

  mCompareReasoningPresence();
  mCompareThinkingEnabled();
  if (pSupportsTools) {
    mCompareReasoningScope();
  }
}

void analyze_reasoning::mCompareReasoningPresence() {
  json vUserMsg = json{
    { "role",    "user"   },
    { "content", cUserMsg }
  };

  json vAssistantNoReasoning = json{
    { "role",    "assistant"   },
    { "content", cAssistantMsg }
  };

  json vAssistantWithReasoning = json{
    { "role",              "assistant"      },
    { "content",           cAssistantMsg    },
    { "reasoning_content", cThinkingContent }
  };

  template_params vParams;
  vParams.vMessages            = json::mArray({ vUserMsg, vAssistantNoReasoning });
  vParams.vAddGenerationPrompt = false;
  vParams.vEnableThinking      = true;

  auto vComparison = fCompareVariants(
    *vTmpl, vParams, [&](template_params & pP) { pP.vMessages = json::mArray({ vUserMsg, vAssistantWithReasoning }); });

  if (!vComparison) {
    LOG_DBG(ANSI_ORANGE "%s: Template application failed, skipping reasoning detection\n" ANSI_RESET, __func__);
    return;
  }

  const auto & vDiff = vComparison->vDiff;

  const std::string cReasoningContent = cThinkingContent;

  if (!vDiff.vRight.empty() && vDiff.vRight.find(cReasoningContent) != std::string::npos) {
    auto vParserDelimiter = fBuildTaggedPegParser([&](common_peg_parser_builder & pP) {
      return pP.mLiteral(cReasoningContent) + pP.mSpace() +
             pP.mOptional(pP.mTag("post", (pP.mMarker() + pP.mSpace())) + pP.mRest());
    });
    auto vParserWrapped   = fBuildTaggedPegParser([&](common_peg_parser_builder & pP) {
      return pP.mTag("pre", pP.mMarker() + pP.mSpace()) + pP.mLiteral(cReasoningContent) +
             pP.mTag("post", (pP.mSpace() + pP.mMarker() + pP.mSpace())) + pP.mRest();
    });
    // try the more aggressive parse first, if it fails, fall back to the delimiter one
    auto vResult          = vParserWrapped.mParseAnywhereAndExtract(vComparison->vOutputB);
    if (!vResult.vResult.mSuccess()) {
      vResult = vParserDelimiter.mParseAnywhereAndExtract(vComparison->vOutputB);
    }
    if (vResult.vResult.mSuccess()) {
      if (!vResult.dTags["pre"].empty() && !vResult.dTags["post"].empty()) {
        vMode  = reasoning_mode::cTagBased;
        vStart = vResult.dTags["pre"];
        end    = vResult.dTags["post"];
      } else if (!vResult.dTags["post"].empty()) {
        vMode = reasoning_mode::cTagBased;
        end   = vResult.dTags["post"];
      }
    }
  }
}

void analyze_reasoning::mCompareThinkingEnabled() {
  json vUserMsg = json{
    { "role",    "user"   },
    { "content", cUserMsg }
  };

  template_params vParams;
  vParams.vMessages            = json::mArray({ vUserMsg });
  vParams.vAddGenerationPrompt = true;
  vParams.vEnableThinking      = false;

  auto vComparison = fCompareVariants(*vTmpl, vParams, [&](template_params & pP) { pP.vEnableThinking = true; });

  if (!vComparison) {
    LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
    return;
  }

  const auto & vDiff = vComparison->vDiff;

  std::string vLeftTrimmed  = fTrimWhitespace(vDiff.vLeft);
  std::string vRightTrimmed = fTrimWhitespace(vDiff.vRight);

  if (vLeftTrimmed.empty() && !vDiff.vRight.empty()) {
    if (!vRightTrimmed.empty() && fStringEndsWith(vComparison->vOutputB, vRightTrimmed)) {
      if (vStart.empty()) {
        vStart = vDiff.vRight;
        vMode  = reasoning_mode::cTagBased;
      }
    }
  } else if (vRightTrimmed.empty() && !vDiff.vLeft.empty()) {
    if (!vLeftTrimmed.empty() && fStringEndsWith(vComparison->vOutputA, vLeftTrimmed)) {
      if (end.empty()) {
        auto lSeg = fPruneWhitespaceSegments(fSegmentizeMarkers(vComparison->vOutputA));
        if (lSeg.size() >= 2 && lSeg[lSeg.size() - 1].value == vLeftTrimmed &&
            lSeg[lSeg.size() - 2].type == segment_type::cMarker) {
          vStart = lSeg[lSeg.size() - 2].value;
        }
        end   = vDiff.vLeft;
        vMode = reasoning_mode::cTagBased;
      }
    }
  } else if (!vLeftTrimmed.empty() && !vRightTrimmed.empty()) {
    // Full-output diff is noisy (e.g., SmolLM3 changes the system message when enable_thinking flips).
    // Try to find reasoning markers by tail-anchoring:
    // one output's generation prompt tail may appear in the other with extra reasoning markers appended.
    const auto & vOutputA   = vComparison->vOutputA;
    const auto & vOutputB   = vComparison->vOutputB;
    const size_t cAnchorLen = 64;

    for (int vDir = 0; vDir < 2; vDir++) {
      const auto & vBase     = vDir == 0 ? vOutputB : vOutputA;
      const auto & vExtended = vDir == 0 ? vOutputA : vOutputB;

      size_t      vLen    = std::min(vBase.size(), cAnchorLen);
      std::string vAnchor = vBase.substr(vBase.size() - vLen);
      auto        vPos    = vExtended.rfind(vAnchor);
      if (vPos == std::string::npos || vPos + vLen >= vExtended.size()) {
        continue;
      }

      std::string vExtra = fTrimWhitespace(vExtended.substr(vPos + vLen));
      if (vExtra.empty()) {
        continue;
      }

      auto lSeg = fPruneWhitespaceSegments(fSegmentizeMarkers(vExtra));
      if (lSeg.size() == 2 && lSeg[0].type == segment_type::cMarker && lSeg[1].type == segment_type::cMarker) {
        if (vStart.empty()) {
          vStart = lSeg[0].value;
        }
        if (end.empty()) {
          end = lSeg[1].value;
        }
        vMode = reasoning_mode::cTagBased;
        break;
      }
    }
  }

  if (vMode == reasoning_mode::cNone && vStart.empty() && !end.empty()) {
    vMode = reasoning_mode::cTagBased;
  }
}

void analyze_reasoning::mCompareReasoningScope() {
  json vAssistantReasoningContent = json{
    { "role",              "assistant"      },
    { "content",           cAssistantMsg    },
    { "reasoning_content", cThinkingContent }
  };

  json vAssistantReasoningTools = json{
    { "role",              "assistant"                                                                  },
    { "content",           nullptr                                                                      },
    { "reasoning_content", cThinkingContent                                                             },
    { "tool_calls",
     json::mArray({ fBuildToolCall(cFunFirst, json{ { cArgFirst, "VVVV" }, { cArgSecond, "XXXX" } }) }) }
  };

  template_params vParams;
  vParams.vMessages            = json::mArray({ vUserMsg, vAssistantReasoningContent });
  vParams.vTools               = vTools;
  vParams.vAddGenerationPrompt = false;
  vParams.vEnableThinking      = true;

  auto vComparison = fCompareVariants(*vTmpl, vParams, [&](template_params & pP) {
    pP.vMessages = json::mArray({ vUserMsg, vAssistantReasoningTools });
  });

  if (!vComparison) {
    LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
    return;
  }

  std::string vReasoningContent = cThinkingContent;

  // Check if reasoning only appears in variant B (with tools)
  bool vReasoningInA = vComparison->vOutputA.find(vReasoningContent) != std::string::npos;
  bool vReasoningInB = vComparison->vOutputB.find(vReasoningContent) != std::string::npos;

  if (!vReasoningInA && vReasoningInB) {
    vMode = reasoning_mode::cToolsOnly;
    LOG_DBG(ANSI_ORANGE "%s: Detected TOOLS_ONLY reasoning mode\n" ANSI_RESET, __func__);

    auto vParserWrapped = fBuildTaggedPegParser([&](common_peg_parser_builder & pP) {
      return pP.mTag("pre", pP.mMarker() + pP.mSpace()) + pP.mLiteral(vReasoningContent) + pP.mSpace() +
             pP.mTag("post", (pP.mMarker() + pP.mSpace()));
    });
    auto vResult        = vParserWrapped.mParseAnywhereAndExtract(vComparison->vOutputB);
    if (vResult.vResult.mSuccess()) {
      vStart = vResult.dTags["pre"];
      end    = vResult.dTags["post"];
    } else {
      auto vParserDelimiter = fBuildTaggedPegParser([&](common_peg_parser_builder & pP) {
        return pP.mLiteral(vReasoningContent) + pP.mSpace() +
               pP.mOptional(pP.mTag("post", (pP.mMarker() + pP.mSpace())));
      });
      vResult               = vParserDelimiter.mParseAnywhereAndExtract(vComparison->vOutputB);
      if (vResult.vResult.mSuccess()) {
        end = vResult.dTags["post"];
      } else {
        LOG_DBG(ANSI_ORANGE "%s: Unable to extract reasoning markers, falling back to reasoning = NONE\n" ANSI_RESET,
                __func__);
        vMode = reasoning_mode::cNone;
      }
    }
  }
}

analyze_content::analyze_content(const common_chat_template & pTmpl, const analyze_reasoning & pReasoning) :
  analyze_base(pTmpl) {
  LOG_DBG(ANSI_ORANGE "Phase 2: Content analysis\n" ANSI_RESET);

  json vAssistantContentOnly = json{
    { "role",    "assistant"   },
    { "content", cAssistantMsg }
  };

  json vAssistantWithTools = json{
    { "role",       "assistant"                                                                 },
    { "content",    ""                                                                          },
    { "tool_calls", json::mArray({ fBuildToolCall("test_func", json{ { "arg1", "value1" } }) }) }
  };

  json vAssistantWithReasoning = json{
    { "role",              "assistant"      },
    { "content",           ""               },
    { "reasoning_content", cThinkingContent }
  };

  template_params vParamsContentOnly;
  vParamsContentOnly.vMessages            = json::mArray({ vUserMsg, vAssistantContentOnly });
  vParamsContentOnly.vAddGenerationPrompt = false;
  vParamsContentOnly.vEnableThinking      = true;
  vParamsContentOnly.vTools               = vTools;

  auto vComparisonWithTools = fCompareVariants(pTmpl, vParamsContentOnly, [&](template_params & pP) {
    pP.vMessages = json::mArray({ vUserMsg, vAssistantWithTools });
  });

  auto vComparisonWithReasoning = fCompareVariants(pTmpl, vParamsContentOnly, [&](template_params & pP) {
    pP.vMessages = json::mArray({ vUserMsg, vAssistantWithReasoning });
  });

  if (!vComparisonWithTools || !vComparisonWithReasoning) {
    LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
    return;
  }

  const auto & vDiffTools     = vComparisonWithTools->vDiff;
  const auto & vDiffReasoning = vComparisonWithReasoning->vDiff;

  std::string vResponse = cAssistantMsg;

  bool vFoundPlainContent = false;
  if (fTrimWhitespace(vDiffTools.vLeft) == vResponse) {
    auto vParser = fBuildTaggedPegParser([&](common_peg_parser_builder & pP) {
      return pP.mSpace() + vDiffReasoning.vLeft + pP.mSpace() + pP.mOptional(pP.mMarker()) + pP.mSpace() + pP.end();
    });
    if (vParser.mParseAndExtract(vDiffReasoning.vLeft).vResult.mSuccess()) {
      // We only have the content text in the diff (possibly with a stray EOG marker), so no markers
      vMode              = content_mode::cPlain;
      vFoundPlainContent = true;
    } else if (pReasoning.vMode != reasoning_mode::cNone && !pReasoning.end.empty()) {
      auto vPostReasoningParser = fBuildTaggedPegParser([&](common_peg_parser_builder & pP) {
        return pP.mLiteral(pReasoning.end) + pP.mSpace() + pP.mLiteral(vResponse);
      });
      if (vPostReasoningParser.mParseAnywhereAndExtract(vDiffReasoning.vLeft).vResult.mSuccess()) {
        vMode              = content_mode::cPlain;
        vFoundPlainContent = true;
      }
    }
  }
  if (!vFoundPlainContent) {
    std::string vRdiff = vDiffReasoning.vLeft;
    if (!pReasoning.end.empty() && vRdiff.find(pReasoning.end) != std::string::npos) {
      vRdiff = vRdiff.substr(vRdiff.find(pReasoning.end) + pReasoning.end.length());
    }
    // Take the more promising diff
    std::string vPureContent   = vRdiff.length() > vDiffTools.vLeft.length() ? vRdiff : vDiffTools.vLeft;
    auto        vParserWrapped = fBuildTaggedPegParser([&](common_peg_parser_builder & pP) {
      return pP.mTag("pre", pP.mMarker() + pP.mSpace()) + pP.mLiteral(vResponse) + pP.mSpace() +
             pP.mTag("post", (pP.mMarker() + pP.mSpace())) + pP.mRest();
    });
    auto        vResult        = vParserWrapped.mParseAnywhereAndExtract(vPureContent);
    vStart                     = vResult.dTags["pre"];
    end                        = vResult.dTags["post"];
    // TODO: WRAPPED_WITH_REASONING
  }

  // Determine content mode
  if (!vStart.empty() || !end.empty()) {
    vMode = content_mode::cAlwaysWrapped;
    // TODO: END_DELIMITED content mode - delimited at end but not at start?
  }
}

bool analyze_content::mIsAlwaysWrapped() const {
  return vMode == content_mode::cAlwaysWrapped && !vStart.empty() && !end.empty();
}

analyze_tools::analyze_tools(const common_chat_template & pTmpl,
                             const jinja::caps &          caps,
                             const analyze_reasoning &    pReasoning) :
  analyze_base(pTmpl) {
  LOG_DBG(ANSI_ORANGE "Phase 3: Tool call analysis\n" ANSI_RESET);

  mAnalyzeToolCalls(pReasoning, caps.vSupportsParallelToolCalls);

  if (format.vMode != tool_format::cNone && format.vMode != tool_format::cJsonNative) {
    if (caps.vSupportsParallelToolCalls) {
      mCheckPerCallMarkers();
    }
    LOG_DBG(ANSI_ORANGE "Phase 3a: Function call analysis\n" ANSI_RESET);
    mExtractFunctionMarkers();
    LOG_DBG(ANSI_ORANGE "Phase 3b: Argument analysis\n" ANSI_RESET);
    if (format.vMode == tool_format::cTagWithTagged) {
      mAnalyzeArguments();
    }
    mExtractArgumentSeparator();
    mExtractArgsMarkers();
    LOG_DBG(ANSI_ORANGE "Phase 3c: Call id analysis\n" ANSI_RESET);
    mExtractCallIdMarkers();
  }
}

void analyze_tools::mAnalyzeToolCalls(const analyze_reasoning & pReasoning, bool pSupportsParallelToolCalls) {
  json vAssistantNoTools = json{
    { "role",    "assistant"   },
    { "content", cAssistantMsg }
  };

  json vAssistantWithTools = json{
    { "role",       "assistant"                      },
    { "content",    ""                               },
    { "tool_calls", json::mArray({ vFirstToolCall }) }
  };

  template_params vParams;
  vParams.vMessages            = json::mArray({ vUserMsg, vAssistantNoTools });
  vParams.vTools               = vTools;
  vParams.vAddGenerationPrompt = false;
  vParams.vEnableThinking      = true;

  auto vComparison = fCompareVariants(
    *vTmpl, vParams, [&](template_params & pP) { pP.vMessages = json::mArray({ vUserMsg, vAssistantWithTools }); });

  if (!vComparison) {
    LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
    return;
  }

  const auto & vDiff = vComparison->vDiff;

  std::string vToolSection = vDiff.vRight;

  if (vToolSection.empty()) {
    return;
  }

  mAnalyzeToolCallFormat(vToolSection, cFunFirst, cArgFirst, pReasoning, pSupportsParallelToolCalls);
}

void analyze_tools::mAnalyzeToolCallFormat(const std::string &       pHaystack,
                                           const std::string &       pFunNameNeedle,
                                           const std::string &       pArgNameNeedle,
                                           const analyze_reasoning & pReasoning,
                                           bool                      pSupportsParallelToolCalls) {
  if (pFunNameNeedle.empty() || pArgNameNeedle.empty() || pHaystack.empty()) {
    return;
  }

  auto vInJsonHaystack = [&pHaystack](const std::string & pNeedle) -> bool {
    auto vParser = fBuildTaggedPegParser([&](common_peg_parser_builder & pP) {
      return pP.mChoice({ pP.mLiteral("{"), pP.mLiteral(":") })
             << pP.mChoice({ pP.mTag("dq", pP.mLiteral("\"") + pP.mLiteral(pNeedle) + pP.mLiteral("\"")) });
    });
    auto vResult = vParser.mParseAnywhereAndExtract(pHaystack);
    return vResult.vResult.mSuccess();
  };

  auto vFunQuote = vInJsonHaystack(pFunNameNeedle);
  auto vArgQuote = vInJsonHaystack(pArgNameNeedle);

  if (vFunQuote) {
    // no need to check further, we're in JSON land
    format.vMode = tool_format::cJsonNative;
  } else if (vArgQuote) {
    format.vMode = tool_format::cTagWithJson;
  } else {
    format.vMode = tool_format::cTagWithTagged;
  }

  // first, remove any reasoning markers
  std::string vCleanHaystack = pHaystack;
  if (!pReasoning.vStart.empty()) {
    auto vPos = pHaystack.find(pReasoning.vStart);
    if (vPos != std::string::npos) {
      vCleanHaystack = pHaystack.substr(0, vPos) + pHaystack.substr(vPos + pReasoning.vStart.length());
    }
  }
  if (!pReasoning.end.empty()) {
    auto vPos = vCleanHaystack.find(pReasoning.end);
    if (vPos != std::string::npos) {
      vCleanHaystack = vCleanHaystack.substr(0, vPos) + vCleanHaystack.substr(vPos + pReasoning.end.length());
    }
  }

  if (format.vMode == tool_format::cJsonNative) {
    mAnalyzeToolCallFormatJsonNative(vCleanHaystack, pFunNameNeedle, pArgNameNeedle);
    if (pSupportsParallelToolCalls) {
      mAnalyzeJsonNativeParallelCalls();
    }
  } else {
    mAnalyzeToolCallFormatNonJson(vCleanHaystack, pFunNameNeedle);
  }
  // always relax whitespace requirements on ending markers since they don't influence content
  format.vSectionEnd = fTrimWhitespace(format.vSectionEnd);
  format.vPerCallEnd = fTrimWhitespace(format.vPerCallEnd);
}

void analyze_tools::mAnalyzeJsonNativeParallelCalls() {
  json vAssistantOneTool = json{
    { "role",       "assistant"                      },
    { "content",    ""                               },
    { "tool_calls", json::mArray({ vFirstToolCall }) }
  };

  json vAssistantTwoTools = json{
    { "role",       "assistant"                                       },
    { "content",    ""                                                },
    { "tool_calls", json::mArray({ vFirstToolCall, vSecondToolCall }) }
  };

  template_params vParams;
  vParams.vMessages            = json::mArray({ vUserMsg, vAssistantOneTool });
  vParams.vTools               = vTools;
  vParams.vAddGenerationPrompt = false;
  vParams.vEnableThinking      = true;

  auto vComparison = fCompareVariants(
    *vTmpl, vParams, [&](template_params & pP) { pP.vMessages = json::mArray({ vUserMsg, vAssistantTwoTools }); });

  if (!vComparison) {
    LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
    return;
  }

  std::string & vSecondCall = vComparison->vDiff.vRight;
  if (!format.vSectionStart.empty() && vSecondCall.find(format.vSectionStart) != std::string::npos) {
    format.vPerCallStart = format.vSectionStart;
    format.vPerCallEnd   = format.vSectionEnd;
    format.vSectionStart.clear();
    format.vSectionEnd.clear();
  }
}

void analyze_tools::mAnalyzeToolCallFormatJsonNative(const std::string & pCleanHaystack,
                                                     const std::string & pFunNameNeedle,
                                                     const std::string & pArgNameNeedle) {
  // we might not have the typical OpenAI tool calling structure
  int         vJsonStart     = pCleanHaystack.find_first_of('{');
  int         vJsonEnd       = pCleanHaystack.find_last_of('}');
  std::string vCut           = pCleanHaystack.substr(vJsonStart, vJsonEnd - vJsonStart + 1);
  json        vCallStruct    = json::mParse(vCut);
  auto        vRegisterField = [&](const std::string & pPrefix, const common_json_entry & pSubel) {
    if (pSubel.value().is_string() && std::string(pSubel.value()).find("call0000") != std::string::npos) {
      format.vIdField = !pPrefix.empty() ? pPrefix + "." + pSubel.mKey() : pSubel.mKey();
    } else if (pSubel.value().is_string() && std::string(pSubel.value()) == pFunNameNeedle) {
      format.vNameField = !pPrefix.empty() ? pPrefix + "." + pSubel.mKey() : pSubel.mKey();
    } else if (pSubel.value().mDump().find(pArgNameNeedle) !=
               std::string::npos) {  // handle both string and JSON obj variants
      format.vArgsField = !pPrefix.empty() ? pPrefix + "." + pSubel.mKey() : pSubel.mKey();
    } else if (pSubel.mKey().find("id") != std::string::npos) {
      // heuristics for generated id field
      format.vGenIdField = !pPrefix.empty() ? pPrefix + "." + pSubel.mKey() : pSubel.mKey();
    }
  };
  for (const auto & vEl : vCallStruct.mItems()) {
    if (vEl.mKey() == pFunNameNeedle) {
      format.vFunNameIsKey = true;
      // When function name is the key, there's no name field and args are direct
      format.vNameField.clear();
      format.vArgsField.clear();
      // Don't register this element - the function name IS the key, not a field
    } else {
      if (vEl.value().is_object() &&
          vEl.value().mDump().find(pArgNameNeedle) == std::string::npos) {  // not the args object
        format.vFunctionField = vEl.mKey();
        for (const auto & vSubel : vEl.value().mItems()) {
          vRegisterField(vEl.mKey(), vSubel);
        }
      }
      // Register this element as a potential field
      vRegisterField("", vEl);
    }
  }
  auto vArrayParser = fBuildTaggedPegParser([&](common_peg_parser_builder & pP) {
    return pP.mTag("pre", pP.mLiteral("[") + pP.mSpace()) + pP.mLiteral(vCut) +
           pP.mTag("post", pP.mSpace() + pP.mLiteral("]"));
  });

  auto vArParseRes = vArrayParser.mParseAnywhereAndExtract(pCleanHaystack);
  if (vArParseRes.vResult.mSuccess()) {
    format.vToolsArrayWrapped = true;
    vJsonStart -= vArParseRes.dTags["pre"].length();
    vJsonEnd += vArParseRes.dTags["post"].length();
  }
  vJsonEnd++;  // we want to move past the closing char for end marker extraction

  std::vector<std::pair<size_t, std::string>> lLocatedParams;
  if (!format.vNameField.empty()) {
    lLocatedParams.push_back({ pCleanHaystack.find(format.vNameField), format.vNameField });
  }
  if (!format.vArgsField.empty()) {
    lLocatedParams.push_back({ pCleanHaystack.find(format.vArgsField), format.vArgsField });
  }
  if (!format.vIdField.empty()) {
    lLocatedParams.push_back({ pCleanHaystack.find(format.vIdField), format.vIdField });
  }
  if (!format.vGenIdField.empty()) {
    lLocatedParams.push_back({ pCleanHaystack.find(format.vGenIdField), format.vGenIdField });
  }
  std::sort(lLocatedParams.begin(), lLocatedParams.end());
  for (auto & vPair : lLocatedParams) {
    format.lParameterOrder.push_back(vPair.second);
  }
  // we can immediately extract tool calling markers too
  format.vSectionStart = fTrimLeadingWhitespace(pCleanHaystack.substr(0, vJsonStart));
  format.vSectionEnd   = fTrimWhitespace(pCleanHaystack.substr(vJsonEnd));
  // When tools_array_wrapped is true, the closing bracket is part of the array structure,
  // not a separate section end marker. Clear tool_section_end to avoid duplicate brackets.
  if (format.vToolsArrayWrapped && format.vSectionEnd == "]") {
    format.vSectionEnd.clear();
  }
}

void analyze_tools::mAnalyzeToolCallFormatNonJson(const std::string & pCleanHaystack,
                                                  const std::string & pFunNameNeedle) {
  // first, let's find out if the function is inside a tag or standalone
  auto        vFunMarkerParser = fBuildTaggedPegParser([&](common_peg_parser_builder & pP) {
    return pP.mTag(
      "fun_marker",
      pP.mChoice(
        { pP.mTag("fun_pre", pP.mLiteral("<") + pP.mUntilOneOf({ ">", pFunNameNeedle })) + pP.mLiteral(pFunNameNeedle) +
            pP.mTag("fun_post", pP.mNegate(pP.mSpace() + pP.mLiteral("<")) + pP.mUntil(">") + pP.mLiteral(">")) +
            pP.mSpace(),
          pP.mTag("fun_pre", pP.mLiteral("[") + pP.mUntilOneOf({ "]", pFunNameNeedle })) + pP.mLiteral(pFunNameNeedle) +
            pP.mTag("fun_post",
                    pP.mNegate(pP.mSpace() + pP.mLiteral("[") + pP.mUntil("]") + pP.mLiteral("]")) + pP.mSpace()) }));
  });
  auto        vFunRes          = vFunMarkerParser.mParseAnywhereAndExtract(pCleanHaystack);
  std::string vFunMarker       = pFunNameNeedle;
  if (vFunRes.vResult.mSuccess()) {
    vFunMarker = vFunRes.dTags["fun_marker"];
  }
  // now, consume up to two markers, then treat everything up to the function marker as function name prefix
  auto                vPerToolParser = fBuildTaggedPegParser([&](common_peg_parser_builder & pP) {
    return pP.mTag("sec_start", pP.mMarker() + pP.mSpace()) + pP.mTag("call_start", pP.mMarker() + pP.mSpace()) +
           pP.mTag("fun_pre", pP.mUntil(vFunMarker)) + vFunMarker + pP.mTag("rest", pP.mRest());
  });
  auto                vSectionParser = fBuildTaggedPegParser([&](common_peg_parser_builder & pP) {
    return pP.mTag("sec_start", pP.mMarker() + pP.mSpace()) + vFunMarker + pP.mTag("rest", pP.mRest());
  });
  auto                vResult        = vPerToolParser.mParseAnywhereAndExtract(pCleanHaystack);
  tagged_parse_result vResultEnd;
  if (vResult.vResult.mSuccess()) {
    auto vDoubleCloserParser = fBuildTaggedPegParser([&](common_peg_parser_builder & pP) {
      return pP.mTag("call_end", pP.mMarker() + pP.mSpace()) + pP.mTag("sec_end", pP.mMarker() + pP.mSpace()) +
             pP.end();
    });
    vResultEnd               = vDoubleCloserParser.mParseAnywhereAndExtract(vResult.dTags["rest"]);
    vFunction.vNamePrefix    = vFunRes.dTags["fun_pre"] + vFunction.vNamePrefix;
  } else {
    vResult                  = vSectionParser.mParseAnywhereAndExtract(pCleanHaystack);
    auto vSingleCloserParser = fBuildTaggedPegParser(
      [&](common_peg_parser_builder & pP) { return pP.mTag("sec_end", pP.mMarker() + pP.mSpace()) + pP.end(); });
    vResultEnd = vSingleCloserParser.mParseAnywhereAndExtract(vResult.dTags["rest"]);
  }
  format.vPerCallStart = vResult.dTags["call_start"];
  format.vPerCallEnd   = vResultEnd.dTags["call_end"];
  format.vSectionStart = vResult.dTags["sec_start"];
  format.vSectionEnd   = vResultEnd.dTags["sec_end"];
}

void analyze_tools::mCheckPerCallMarkers() {
  json vAssistantOneTool = json{
    { "role",       "assistant"                      },
    { "content",    ""                               },
    { "tool_calls", json::mArray({ vFirstToolCall }) }
  };

  json vAssistantTwoTools = json{
    { "role",       "assistant"                                       },
    { "content",    ""                                                },
    { "tool_calls", json::mArray({ vFirstToolCall, vSecondToolCall }) }
  };

  template_params vParams;
  vParams.vMessages            = json::mArray({ vUserMsg, vAssistantOneTool });
  vParams.vTools               = vTools;
  vParams.vAddGenerationPrompt = false;
  vParams.vEnableThinking      = true;

  auto vOneVsTwo = fCompareVariants(
    *vTmpl, vParams, [&](template_params & pP) { pP.vMessages = json::mArray({ vUserMsg, vAssistantTwoTools }); });

  if (!vOneVsTwo) {
    LOG_DBG(ANSI_ORANGE "%s: Generating double tool call comparison failed\n" ANSI_RESET, __func__);
    return;
  }

  diff_split vFilterCommonCallPart = fCalculateDiffSplit(vOneVsTwo->vDiff.vSuffix, vOneVsTwo->vDiff.vRight);

  std::string vSecondToolContent = fTrimLeadingWhitespace(vFilterCommonCallPart.vRight);
  if (!format.vSectionStart.empty() && vSecondToolContent.find(format.vSectionStart) == 0) {
    format.vPerCallStart = format.vSectionStart;
    format.vPerCallEnd   = format.vSectionEnd;
    format.vSectionStart.clear();
    format.vSectionEnd.clear();
  }

  if (!format.vPerCallEnd.empty()) {
    auto vCountOccurrences = [](const std::string & pHaystack, const std::string & pNeedle) {
      size_t count = 0;
      for (size_t vPos = pHaystack.find(pNeedle); vPos != std::string::npos;
           vPos        = pHaystack.find(pNeedle, vPos + pNeedle.size())) {
        count++;
      }
      return count;
    };
    size_t vCallsOne = vCountOccurrences(vOneVsTwo->vOutputA, format.vPerCallEnd);
    size_t vCallsTwo = vCountOccurrences(vOneVsTwo->vOutputB, format.vPerCallEnd);
    if (vCallsOne > 0 && vCallsOne == vCallsTwo) {
      format.vSectionEnd = format.vPerCallEnd;
      format.vPerCallEnd.clear();
    }
  }
}

void analyze_tools::mExtractFunctionMarkers() {
  json vAssistantNocall = json{
    { "role",    "assistant"   },
    { "content", cAssistantMsg },
  };

  json vAssistantFoofoo = json{
    { "role",       "assistant"                      },
    { "content",    ""                               },
    { "tool_calls", json::mArray({ vFirstToolCall }) }
  };

  json vAssistantBarbar = json{
    { "role",       "assistant"                       },
    { "content",    ""                                },
    { "tool_calls", json::mArray({ vSecondToolCall }) }
  };

  template_params vParams;
  vParams.vMessages            = json::mArray({ vUserMsg, vAssistantFoofoo });
  vParams.vTools               = vTools;
  vParams.vAddGenerationPrompt = false;
  vParams.vEnableThinking      = true;

  auto vComparison = fCompareVariants(
    *vTmpl, vParams, [&](template_params & pP) { pP.vMessages = json::mArray({ vUserMsg, vAssistantBarbar }); });

  if (!vComparison) {
    LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
    return;
  }

  const auto & vDiff = vComparison->vDiff;

  if (vDiff.vLeft.find(cFunFirst) != std::string::npos && vDiff.vRight.find(cFunSecond) != std::string::npos) {
    std::string vPrefixMarker;
    if (!format.vPerCallStart.empty()) {
      vPrefixMarker = format.vPerCallStart;
    } else {
      vPrefixMarker = format.vSectionStart;
    }
    if (!vPrefixMarker.empty() && vDiff.vPrefix.rfind(vPrefixMarker) != std::string::npos) {
      vFunction.vNamePrefix = vDiff.vPrefix.substr(vDiff.vPrefix.rfind(vPrefixMarker) + vPrefixMarker.size());
    }

    // Extract name prefix/suffix from diff.left (stop at the next marker boundary)
    auto vNameParser = fBuildTaggedPegParser([&](common_peg_parser_builder & pP) {
      return pP.mTag("pre", pP.mUntil(cFunFirst)) + pP.mLiteral(cFunFirst) +
             pP.mTag("post", pP.mZeroOrMore(pP.mNegate(pP.mMarker()) + pP.mAny()));
    });
    auto vNameResult = vNameParser.mParseAndExtract(vDiff.vLeft);
    if (vNameResult.vResult.mSuccess()) {
      vFunction.vNamePrefix += vNameResult.dTags["pre"];
      vFunction.vNameSuffix = vNameResult.dTags["post"];
    }

    // Extend name_suffix with content from diff.suffix before args begin
    if (format.vMode == tool_format::cTagWithJson) {
      // For JSON: name_suffix extends to the first non-marker { or [, including any
      // markers along the way. Only applies if there's at least one marker after
      // the JSON content (matching the original "stop < seg_suf.size() - 1" guard).
      auto vSuffixParser = fBuildTaggedPegParser([&](common_peg_parser_builder & pP) {
        auto vNonJson   = pP.mMarker() | (pP.mNegate(pP.mLiteral("{")) + pP.mNegate(pP.mLiteral("[")) + pP.mAny());
        auto vAfterJson = pP.mZeroOrMore(pP.mNegate(pP.mMarker()) + pP.mAny()) + pP.mMarker();
        return pP.mTag("ext", pP.mZeroOrMore(vNonJson)) + vAfterJson;
      });
      auto vSufResult    = vSuffixParser.mParseAndExtract(vDiff.vSuffix);
      if (vSufResult.vResult.mSuccess()) {
        vFunction.vNameSuffix += vSufResult.dTags["ext"];
      }
    } else {
      // For tagged: name_suffix extends to the first marker (arg marker)
      auto vSuffixParser = fBuildTaggedPegParser([&](common_peg_parser_builder & pP) {
        return pP.mTag("ext", pP.mZeroOrMore(pP.mNegate(pP.mMarker()) + pP.mAny()));
      });
      auto vSufResult    = vSuffixParser.mParseAndExtract(vDiff.vSuffix);
      if (vSufResult.vResult.mSuccess()) {
        vFunction.vNameSuffix += vSufResult.dTags["ext"];

        auto vArgStart = [&](common_peg_parser_builder & pP) {
          return pP.mMarker() + pP.mSpace() + pP.mChoice({ pP.mLiteral(cArgFirst), pP.mLiteral(cArgSecond) });
        };
        auto vSepParser = fBuildTaggedPegParser([&](common_peg_parser_builder & pP) {
          return pP.mTag("sep", pP.mZeroOrMore(pP.mNegate(vArgStart(pP)) + pP.mAny())) + vArgStart(pP);
        });
        auto vSepResult = vSepParser.mParseAndExtract(vDiff.vSuffix.substr(vSufResult.dTags["ext"].size()));
        if (vSepResult.vResult.mSuccess()) {
          vFunction.vArgsSeparator = fTrimWhitespace(vSepResult.dTags["sep"]);
        }
      }
    }

    // Extract the closer (between last arg and call/section end marker)
    std::string vSuffixMarker;
    if (!format.vPerCallEnd.empty()) {
      vSuffixMarker = format.vPerCallEnd;
    } else {
      vSuffixMarker = format.vSectionEnd;
    }
    std::string vCloserSuffix;
    if (vSuffixMarker.empty()) {
      // we'll have to rely on an extra diff with no-calls version
      auto vNotoolComp = fCompareVariants(
        *vTmpl, vParams, [&](template_params & pP) { pP.vMessages = json::mArray({ vUserMsg, vAssistantNocall }); });
      if (vNotoolComp) {
        auto vNtDiff  = vNotoolComp->vDiff;
        vCloserSuffix = vNtDiff.vLeft.substr(vNtDiff.vLeft.find("YYYY") + 4);
      }
    } else {
      vCloserSuffix = vDiff.vSuffix.substr(0, vDiff.vSuffix.find(vSuffixMarker));
    }
    if (!vCloserSuffix.empty()) {
      if (format.vMode == tool_format::cTagWithTagged) {
        // After last arg value, skip the closing arg marker, rest is closer
        auto vCloserParser = fBuildTaggedPegParser([&](common_peg_parser_builder & pP) {
          return pP.mUntil("YYYY") + pP.mLiteral("YYYY") + pP.mSpace() + pP.mMarker() + pP.mSpace() +
                 pP.mTag("close", pP.mRest());
        });
        auto vCloseResult  = vCloserParser.mParseAndExtract(vCloserSuffix);
        if (vCloseResult.vResult.mSuccess()) {
          vFunction.vClose = vCloseResult.dTags["close"];
        }
      } else if (format.vMode == tool_format::cTagWithJson) {
        // After last arg value, find end of JSON args, rest is closer
        auto vCloserParser = fBuildTaggedPegParser([&](common_peg_parser_builder & pP) {
          return pP.mUntil("YYYY") + pP.mLiteral("YYYY") + pP.mTag("post_val", pP.mRest());
        });
        auto vCloseResult  = vCloserParser.mParseAndExtract(vCloserSuffix);
        if (vCloseResult.vResult.mSuccess()) {
          const auto & vPost = vCloseResult.dTags["post_val"];
          size_t       vPos  = vPost.find_last_of("}]");
          if (vPos != std::string::npos && vPos < vPost.size() - 1) {
            vFunction.vClose = fTrimLeadingWhitespace(vPost.substr(vPos + 1));
          }
        }
      }
    }
    vFunction.vClose = fTrimLeadingWhitespace(vFunction.vClose);
  }
}

void analyze_tools::mAnalyzeArguments() {
  mExtractArgumentNameMarkers();
  mExtractArgumentValueMarkers();
}

void analyze_tools::mExtractArgumentNameMarkers() {
  json vAssistantFirstArg = json{
    { "role",       "assistant"                            },
    { "content",    ""                                     },
    { "tool_calls", json::mArray({ vFirstToolCallOneArg }) }
  };

  json vAssistantSecondArg = json{
    { "role",       "assistant"                              },
    { "content",    ""                                       },
    { "tool_calls", json::mArray({ vFirstToolCallOtherArg }) }
  };

  template_params vParams;
  vParams.vMessages            = json::mArray({ vUserMsg, vAssistantFirstArg });
  vParams.vTools               = vTools;
  vParams.vAddGenerationPrompt = false;
  vParams.vEnableThinking      = true;

  auto vComparison = fCompareVariants(
    *vTmpl, vParams, [&](template_params & pP) { pP.vMessages = json::mArray({ vUserMsg, vAssistantSecondArg }); });

  if (!vComparison) {
    LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
    return;
  }

  const auto & vDiff = vComparison->vDiff;

  if (!vDiff.vLeft.empty() && !vDiff.vRight.empty()) {
    // Parse both sides to find ARG_FIRST/ARG_SECOND and extract the surrounding structure
    auto vLeftParser  = fBuildTaggedPegParser([&](common_peg_parser_builder & pP) {
      return pP.mTag("pre", pP.mUntil(cArgFirst)) + pP.mLiteral(cArgFirst) +
             pP.mTag("suffix", pP.mUntilOneOf({ "\"", "X" }));
    });
    auto vRightParser = fBuildTaggedPegParser([&](common_peg_parser_builder & pP) {
      return pP.mTag("pre", pP.mUntil(cArgSecond)) + pP.mLiteral(cArgSecond) +
             pP.mTag("suffix", pP.mUntilOneOf({ "\"", "Y" }));
    });
    auto vLeftResult  = vLeftParser.mParseAnywhereAndExtract(vDiff.vLeft);
    auto vRightResult = vRightParser.mParseAnywhereAndExtract(vDiff.vRight);

    if (vLeftResult.vResult.mSuccess() && vRightResult.vResult.mSuccess() && !vLeftResult.dTags["pre"].empty() &&
        vLeftResult.dTags["pre"] == vRightResult.dTags["pre"] &&
        vLeftResult.dTags["suffix"] == vRightResult.dTags["suffix"]) {
      // Name is inside a structure (e.g., JSON key): prefix is the shared wrapper
      vArguments.vNamePrefix = vLeftResult.dTags["pre"];
      vArguments.vNameSuffix = vLeftResult.dTags["suffix"];
    } else if (vDiff.vLeft.substr(0, cArgFirst.length()) == cArgFirst &&
               vDiff.vRight.substr(0, cArgSecond.length()) == cArgSecond) {
      // Name is directly in the diff: prefix comes from last marker in diff.prefix
      auto vPreParser        = fBuildTaggedPegParser([&](common_peg_parser_builder & pP) {
        auto vLastMarker = pP.mMarker() + pP.mZeroOrMore(pP.mNegate(pP.mMarker()) + pP.mAny()) + pP.end();
        return pP.mZeroOrMore(pP.mNegate(vLastMarker) + pP.mAny()) + pP.mTag("name_prefix", vLastMarker);
      });
      auto vPreResult        = vPreParser.mParseAndExtract(vDiff.vPrefix);
      vArguments.vNamePrefix = vPreResult.vResult.mSuccess() ? vPreResult.dTags["name_prefix"] : vDiff.vPrefix;

      // Suffix extends from after ARG_FIRST to the first marker (+ optional whitespace).
      // The marker could be in diff.left itself or in diff.suffix, so we concatenate.
      std::string vAfterFirst   = vDiff.vLeft.substr(cArgFirst.length()) + vDiff.vSuffix;
      auto        vSuffixParser = fBuildTaggedPegParser([&](common_peg_parser_builder & pP) {
        return pP.mTag("suffix", pP.mZeroOrMore(pP.mNegate(pP.mMarker()) + pP.mAny()) + pP.mMarker() + pP.mSpace());
      });
      auto        vSufResult    = vSuffixParser.mParseAnywhereAndExtract(vAfterFirst);
      if (vSufResult.vResult.mSuccess()) {
        vArguments.vNameSuffix = vSufResult.dTags["suffix"];
      }
    }
  }
}

void analyze_tools::mExtractArgumentValueMarkers() {
  json vAssistantValX = json{
    { "role",       "assistant"                            },
    { "content",    ""                                     },
    { "tool_calls", json::mArray({ vFirstToolCallOneArg }) }
  };

  json vAssistantValY = json{
    { "role",       "assistant"                                    },
    { "content",    ""                                             },
    { "tool_calls", json::mArray({ vFirstToolCallOneArgOtherVal }) }
  };

  template_params vParams;
  vParams.vMessages            = json::mArray({ vUserMsg, vAssistantValX });
  vParams.vTools               = vTools;
  vParams.vAddGenerationPrompt = false;
  vParams.vEnableThinking      = true;

  auto vComparison = fCompareVariants(
    *vTmpl, vParams, [&](template_params & pP) { pP.vMessages = json::mArray({ vUserMsg, vAssistantValY }); });

  if (!vComparison) {
    LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
    return;
  }

  const auto & vDiff = vComparison->vDiff;

  if (vDiff.vLeft == "XXXX" && vDiff.vRight == "YYYY") {
    std::string vArgNameEnding = cArgFirst + vArguments.vNameSuffix;
    std::string vPrefix        = vDiff.vPrefix;
    if (vPrefix.rfind(vArgNameEnding) != std::string::npos) {
      vPrefix = vPrefix.substr(vPrefix.rfind(vArgNameEnding) + vArgNameEnding.size());
    }
    if (!vPrefix.empty()) {
      // Find the last marker + any trailing non-marker text to end
      auto vPrefixParser      = fBuildTaggedPegParser([&](common_peg_parser_builder & pP) {
        auto vLastMarker = pP.mMarker() + pP.mZeroOrMore(pP.mNegate(pP.mMarker()) + pP.mAny()) + pP.end();
        return pP.mZeroOrMore(pP.mNegate(vLastMarker) + pP.mAny()) + pP.mTag("val_prefix", vLastMarker);
      });
      auto vPreResult         = vPrefixParser.mParseAndExtract(vPrefix);
      vArguments.vValuePrefix = vPreResult.vResult.mSuccess() ? vPreResult.dTags["val_prefix"] : vPrefix;
    }

    std::string vValueSuffix = vDiff.vSuffix;
    if (!vFunction.vClose.empty()) {
      size_t vFuncClosePos = vValueSuffix.find(vFunction.vClose);
      if (vFuncClosePos != std::string::npos) {
        vValueSuffix = vValueSuffix.substr(0, vFuncClosePos);
      }
    } else if (!format.vPerCallEnd.empty() || !format.vSectionEnd.empty()) {
      std::string vEndMarker    = !format.vPerCallEnd.empty() ? format.vPerCallEnd : format.vSectionEnd;
      size_t      vEndMarkerPos = vValueSuffix.find(vEndMarker);
      if (vEndMarkerPos != std::string::npos) {
        vValueSuffix = vValueSuffix.substr(0, vEndMarkerPos);
      }
    }
    if (!fTrimWhitespace(vValueSuffix).empty()) {
      vArguments.vValueSuffix = vValueSuffix;
    }
  }
}

void analyze_tools::mExtractArgumentSeparator() {
  json vAssistantOneArg = json{
    { "role",       "assistant"                            },
    { "content",    ""                                     },
    { "tool_calls", json::mArray({ vFirstToolCallOneArg }) }
  };

  json vAssistantTwoArgs = json{
    { "role",       "assistant"                      },
    { "content",    ""                               },
    { "tool_calls", json::mArray({ vFirstToolCall }) }
  };

  template_params vParams;
  vParams.vMessages            = json::mArray({ vUserMsg, vAssistantOneArg });
  vParams.vTools               = vTools;
  vParams.vAddGenerationPrompt = false;
  vParams.vEnableThinking      = true;

  auto vComparison = fCompareVariants(
    *vTmpl, vParams, [&](template_params & pP) { pP.vMessages = json::mArray({ vUserMsg, vAssistantTwoArgs }); });

  if (!vComparison) {
    LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
    return;
  }

  const auto & vDiff = vComparison->vDiff;

  if (!vDiff.vRight.empty()) {
    std::string vSeparator = fUntilCommonPrefix(vDiff.vRight, cArgFirst, cArgSecond);
    vArguments.vSeparator  = vSeparator;
  }
}

void analyze_tools::mExtractArgsMarkers() {
  json vAssistantNoArgs = json{
    { "role",       "assistant"                              },
    { "content",    ""                                       },
    { "tool_calls", json::mArray({ vFirstToolCallZeroArgs }) }
  };

  json vAssistantWithArgs = json{
    { "role",       "assistant"                            },
    { "content",    ""                                     },
    { "tool_calls", json::mArray({ vFirstToolCallOneArg }) }
  };

  template_params vParams;
  vParams.vMessages            = json::mArray({ vUserMsg, vAssistantNoArgs });
  vParams.vTools               = vTools;
  vParams.vAddGenerationPrompt = false;
  vParams.vEnableThinking      = true;

  auto vComparison = fCompareVariants(
    *vTmpl, vParams, [&](template_params & pP) { pP.vMessages = json::mArray({ vUserMsg, vAssistantWithArgs }); });

  if (!vComparison) {
    LOG_DBG(ANSI_ORANGE "%s: Template application failed\n" ANSI_RESET, __func__);
    return;
  }

  const auto & vDiff = vComparison->vDiff;

  if (format.vMode == tool_format::cJsonNative) {
    std::string vPrefixMarker = !format.vSectionStart.empty() ? format.vSectionStart : format.vPerCallStart;
    std::string vSuffixMarker = !format.vSectionEnd.empty() ? format.vSectionEnd : format.vPerCallEnd;
    // these might happen earlier in the tools section as an example or somewhere else, so we need to find the closest ones
    size_t      vPrefixPos    = vPrefixMarker.empty() ? 0 : vDiff.vPrefix.rfind(vPrefixMarker);
    size_t      vSuffixPos    = vSuffixMarker.empty() ? vDiff.vSuffix.size() : vDiff.vSuffix.find(vSuffixMarker);
    if (vPrefixPos == std::string::npos) {
      vPrefixPos = 0;
    }
    if (vSuffixPos == std::string::npos) {
      vSuffixPos = vDiff.vSuffix.size();
    }
    std::string vPrefixCut = vDiff.vPrefix.substr(vPrefixPos + vPrefixMarker.size());
    std::string vSuffixCut = vDiff.vSuffix.substr(0, vSuffixPos);
    std::string vArgsStart = fUntilCommonPrefix(vPrefixCut, "{}", "{\"first\":");
    std::string vArgsEnd   = fAfterCommonSuffix(vSuffixCut, "{}", "\"XXXX\"}");

    if (!vArgsStart.empty() || !vArgsEnd.empty()) {
      size_t vFindFun = vArgsStart.find(cFunFirst);
      if (vFindFun != std::string::npos) {
        vArgsStart = vArgsStart.substr(vFindFun + cFunFirst.size(), vArgsStart.size() - vFindFun - cFunFirst.size());
      }
      size_t vFindCallId = vArgsStart.find(cCallId001);
      if (vFindCallId != std::string::npos) {
        vArgsStart =
          vArgsStart.substr(vFindCallId + cCallId001.size(), vArgsStart.size() - vFindCallId - cCallId001.size());
      }
      vArguments.vStart = vArgsStart;
      vArguments.end    = vArgsEnd;
    }
  }
}

void analyze_tools::mExtractCallIdMarkers() {
  json vAssistantId1 = json{
    { "role",       "assistant"                      },
    { "content",    ""                               },
    { "tool_calls", json::mArray({ vFirstToolCall }) }
  };

  json vAssistantId2 = json{
    { "role",       "assistant"                           },
    { "content",    ""                                    },
    { "tool_calls", json::mArray({ vFirstToolCallAltId }) }
  };

  template_params vParams;
  vParams.vMessages            = json::mArray({ vUserMsg, vAssistantId1 });
  vParams.vTools               = vTools;
  vParams.vAddGenerationPrompt = false;
  vParams.vEnableThinking      = true;

  auto vComparison = fCompareVariants(
    *vTmpl, vParams, [&](template_params & pP) { pP.vMessages = json::mArray({ vUserMsg, vAssistantId2 }); });

  if (!vComparison) {
    LOG_DBG(ANSI_ORANGE "%s: Template application failed for call_id detection\n" ANSI_RESET, __func__);
    return;
  }

  const auto & vDiff = vComparison->vDiff;

  if (vDiff.vLeft.empty() && vDiff.vRight.empty()) {
    return;
  }

  std::string vIdValue1 = cCallId001;
  std::string vIdValue2 = cCallId999;

  size_t vCommonIdPrefixLen = 0;
  for (size_t vI = 0; vI < std::min(vIdValue1.length(), vIdValue2.length()); vI++) {
    if (vIdValue1[vI] == vIdValue2[vI]) {
      vCommonIdPrefixLen++;
    } else {
      break;
    }
  }
  std::string vCommonIdPart = vIdValue1.substr(0, vCommonIdPrefixLen);

  // Check if the function name is in the prefix (normal case: BETWEEN_FUNC_AND_ARGS or POST_ARGS)
  // or in the suffix (call_id is PRE_FUNC_NAME)
  std::string vFuncName         = cFunFirst;
  size_t      vFuncNameInPrefix = vDiff.vPrefix.rfind(vFuncName);
  size_t      vFuncNameInSuffix = vDiff.vSuffix.find(vFuncName);

  // Helper: find the last marker in a string (returns just the marker, not trailing text)
  auto vFindLastMarker = [](const std::string & str) -> std::string {
    auto vParser = fBuildTaggedPegParser([&](common_peg_parser_builder & pP) {
      auto vLast = pP.mMarker() + pP.mZeroOrMore(pP.mNegate(pP.mMarker()) + pP.mAny()) + pP.end();
      return pP.mZeroOrMore(pP.mNegate(vLast) + pP.mAny()) + pP.mTag("m", pP.mMarker());
    });
    auto vRes    = vParser.mParseAnywhereAndExtract(str);
    return vRes.vResult.mSuccess() ? vRes.dTags["m"] : "";
  };

  // Helper: find the first marker in a string
  auto vFindFirstMarker = [](const std::string & str) -> std::string {
    auto vParser = fBuildTaggedPegParser([&](common_peg_parser_builder & pP) { return pP.mTag("m", pP.mMarker()); });
    auto vRes    = vParser.mParseAnywhereAndExtract(str);
    return vRes.vResult.mSuccess() ? vRes.dTags["m"] : "";
  };

  if (vFuncNameInPrefix != std::string::npos && vFuncNameInSuffix == std::string::npos) {
    // Function name is only in prefix - call_id is BETWEEN_FUNC_AND_ARGS or POST_ARGS
    // Check if args indicator "{" is in prefix or suffix
    size_t vArgsInPrefix = vDiff.vPrefix.find('{', vFuncNameInPrefix);
    size_t vArgsInSuffix = vDiff.vSuffix.find('{');

    if (vArgsInSuffix != std::string::npos &&
        (vArgsInPrefix == std::string::npos || vArgsInPrefix > vDiff.vPrefix.length())) {
      // Args are in suffix, so call_id is BETWEEN_FUNC_AND_ARGS
      vCallId.vPos = call_id_position::cBetweenFuncAndArgs;

      // Find call_id_prefix: marker immediately preceding common_id_part (no intervening markers)
      std::string vAfterFunc      = vDiff.vPrefix.substr(vFuncNameInPrefix + vFuncName.length());
      auto        vIdPrefixParser = fBuildTaggedPegParser([&](common_peg_parser_builder & pP) {
        return pP.mTag("prefix", pP.mMarker()) +
               pP.mZeroOrMore(pP.mNegate(pP.mMarker()) + pP.mNegate(pP.mLiteral(vCommonIdPart)) + pP.mAny()) +
               pP.mLiteral(vCommonIdPart);
      });
      auto        vIdRes          = vIdPrefixParser.mParseAnywhereAndExtract(vAfterFunc);
      if (vIdRes.vResult.mSuccess()) {
        vCallId.vPrefix = vIdRes.dTags["prefix"];
      } else {
        // Fallback: use the last marker in after_func
        vCallId.vPrefix = vFindLastMarker(vAfterFunc);
      }

      // Extract call_id_suffix: the first marker in the suffix before args "{"
      auto vSuffixParser = fBuildTaggedPegParser([&](common_peg_parser_builder & pP) {
        return pP.mZeroOrMore(pP.mNegate(pP.mMarker()) + pP.mNegate(pP.mLiteral("{")) + pP.mAny()) +
               pP.mTag("suffix", pP.mMarker());
      });
      auto vSufRes       = vSuffixParser.mParseAnywhereAndExtract(vDiff.vSuffix);
      if (vSufRes.vResult.mSuccess()) {
        vCallId.vSuffix = vSufRes.dTags["suffix"];
      }
    } else if (vArgsInPrefix != std::string::npos) {
      // Args are in prefix, so call_id is POST_ARGS
      vCallId.vPos = call_id_position::cPostArgs;

      // Extract last marker between args closing brace and the ID
      std::string vAfterArgs    = vDiff.vPrefix.substr(vArgsInPrefix);
      size_t      vClosingBrace = vAfterArgs.rfind('}');
      if (vClosingBrace != std::string::npos) {
        std::string vBetweenArgsAndId = vAfterArgs.substr(vClosingBrace + 1);
        vCallId.vPrefix               = vFindLastMarker(vBetweenArgsAndId);
      }

      // call_id_suffix: first marker in diff.suffix
      vCallId.vSuffix = vFindFirstMarker(vDiff.vSuffix);
    }
  } else if (vFuncNameInSuffix != std::string::npos && vFuncNameInPrefix == std::string::npos) {
    // Function name is only in suffix - call_id is PRE_FUNC_NAME
    vCallId.vPos = call_id_position::cPreFuncName;

    // call_id_prefix: last marker in diff.prefix
    vCallId.vPrefix = vFindLastMarker(vDiff.vPrefix);

    // call_id_suffix: first marker in the portion of diff.suffix before func_name
    std::string vBeforeFunc = vDiff.vSuffix.substr(0, vFuncNameInSuffix);
    vCallId.vSuffix         = vFindFirstMarker(vBeforeFunc);
  }

  if (vCallId.vPrefix == vArguments.end) {
    vCallId.vPrefix = "";
  }

  if (vCallId.vSuffix == vArguments.vStart) {
    vCallId.vSuffix = "";
  }

  // When call_id is detected, per_call_end may have been incorrectly set to include
  // the call_id_suffix and sample args. Clear it if it starts with call_id_suffix.
  if (vCallId.vPos != call_id_position::cNone && !vCallId.vSuffix.empty() &&
      format.vPerCallEnd.find(vCallId.vSuffix) == 0) {
    format.vPerCallEnd.clear();
  }
}

}  // namespace autoparser
