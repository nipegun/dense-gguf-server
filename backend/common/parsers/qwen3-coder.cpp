#include "parsers.h"

common_chat_params fCommonChatParamsInitQwen3Coder(const common_chat_template &          pTmpl,
                                                   const autoparser::generation_params & pInputs) {
  common_chat_params data;

  const std::string cGenPrefix = "<|im_start|>assistant\n";

  data.vPrompt           = fCommonChatTemplateDirectApplyImpl(pTmpl, pInputs);
  data.vGenerationPrompt = fCommonChatTemplateGenerationPromptImpl(pTmpl, pInputs);
  data.format            = cCommonChatFormatPegNative;

  auto vSupportsReasoning = pTmpl.mSource().find("<think>") != std::string::npos;

  data.vSupportsThinking = vSupportsReasoning;
  data.lPreservedTokens  = {
    "<tool_call>",
    "</tool_call>",
  };

  auto vIsQwen3Coder = !vSupportsReasoning;

  if (vSupportsReasoning) {
    data.vThinkingStartTag = "<think>";
    // Support both </think> and <tool_call> as reasoning end sequences.
    // The newline variant comes first so it is included in the forced message
    // <function= is omitted, as it is a workaround for Qwen3-Coder which is not a thinking model
    data.lThinkingEndTags  = { "\n</think>", "</think>", "<tool_call>" };
    data.lPreservedTokens.insert(data.lPreservedTokens.end(), { "<think>", "</think>" });
  }

  data.vMessageDelimiters = {
    { cCommonChatRoleAssistant, "<|im_start|>assistant"             },
    { cCommonChatRoleTool,      "<|im_start|>user\n<tool_response>" }, // Qwen3-Coder, Qwen3.5, Nemotron Nano 3
    { cCommonChatRoleTool,      "<|im_start|>tool_response"         }, // StepFun-3.5-Flash
    { cCommonChatRoleUser,      "<|im_start|>user"                  },
    { cCommonChatRoleSystem,    "<|im_start|>system"                },
  };

  auto vHasTools          = pInputs.vTools.is_array() && !pInputs.vTools.empty();
  auto vHasResponseFormat = pInputs.vJsonSchema.is_object() && !pInputs.vJsonSchema.empty();
  auto vExtractReasoning  = pInputs.vReasoningFormat != cCommonReasoningFormatNone;
  auto vIncludeGrammar    = vHasResponseFormat || (vHasTools && pInputs.vToolChoice != cCommonChatToolChoiceNone);

  if (pInputs.mHasContinuation()) {
    const auto & vMsg = pInputs.vContinueMsg;

    data.vGenerationPrompt = cGenPrefix;
    if (vSupportsReasoning) {
      data.vGenerationPrompt += "<think>\n" + vMsg.vReasoningContent;
      if (pInputs.vContinueFinalMessage == cCommonChatContinuationContent) {
        data.vGenerationPrompt += "\n</think>\n\n";
      }
    }
    if (pInputs.vContinueFinalMessage == cCommonChatContinuationContent) {
      data.vGenerationPrompt += vMsg.mRenderContent();
    }

    data.vPrompt += data.vGenerationPrompt;
  }

  std::vector<std::string> lToolCallStarts = { "<tool_call>" };

  if (vIsQwen3Coder) {
    // Match complete <function=name> opener for Qwen3-Coder models that occasionally omit the
    // starting <tool_call>. The model may hallucinate a tool name, but it is preferable over
    // constraining on <function which may occur in valid content generation, e.g. #include <functional>
    fForeachFunction(pInputs.vTools, [&](size_t, const json & pTool) {
      const std::string name = pTool.at("function").at("name");
      lToolCallStarts.push_back("<function=" + name + ">");
    });
  }

  data.vParser = fBuildChatPegParser([&](common_chat_peg_builder & pP) {
    auto vGenerationPrompt = pP.mLiteral(cGenPrefix);

    auto vReasoning = pP.mEps();
    if (vSupportsReasoning && vExtractReasoning) {
      vReasoning = pP.mOptional("<think>" + pP.mSpace() + pP.mReasoning(pP.mUntilOneOf({ "</think>", "<tool_call>" })) +
                                (pP.mLiteral("</think>") | pP.mPeek(pP.mLiteral("<tool_call>"))));
    }

    // Response format parser
    if (vHasResponseFormat) {
      return vGenerationPrompt +
             (vReasoning << pP.mContent(pP.mSchema(pP.mJson(), "response-format", pInputs.vJsonSchema)));
    }

    // Tool call parser
    if (vHasTools && pInputs.vToolChoice != cCommonChatToolChoiceNone) {
      auto vArgClose = pP.mToolArgClose(pP.mLiteral("\n</parameter>\n"));
      auto vArgString =
        pP.mRule("xml-arg-string",
                 pP.mAc(pP.mToolArgStringValue(pP.mUntil("\n</parameter>\n")) + vArgClose, "\n</parameter>\n"));

      auto vToolChoice = pP.mChoice();
      fForeachFunction(pInputs.vTools, [&](size_t pToolIndex, const json & pTool) {
        const auto & vFunction = pTool.at("function");
        std::string  name      = vFunction.at("name");

        std::vector<common_peg_parser> lRequiredArgs;
        std::vector<common_peg_parser> lOptionalArgs;

        fForeachParameter(vFunction, [&](size_t pParamIndex, const common_chat_schema_property & pParam,
                                         const common_chat_schema_document_ptr & pDoc) {
          auto vRuleName = "tool-" + std::to_string(pToolIndex) + "-arg-" + std::to_string(pParamIndex);

          auto vArgOpen = pP.mToolArgOpen("<parameter=" + pP.mToolArgName(pP.mLiteral(pParam.name)) + ">\n");

          auto vTypes = pParam.vSchema->mValueTypes();

          auto vArgValue = pP.mEps();
          if (!vTypes.has(common_chat_schema::cTypeString)) {
            vArgValue =
              pP.mToolArgJsonValue(pP.mSchema(pP.mJson(), vRuleName + "-schema", pDoc, *pParam.vSchema)) + vArgClose;
          } else if (vTypes.mIsOnly(common_chat_schema::cTypeString)) {
            vArgValue = vArgString;
          } else {
            // The string alternative accepts any text, so the grammar only keeps the raw string
            // rule. The parser still tries the JSON alternatives first to type the value.
            auto vJsonValue = pP.mChoice();
            if (vTypes.has(common_chat_schema::cTypeObject)) {
              vJsonValue |= pP.mJsonObject();
            }
            if (vTypes.has(common_chat_schema::cTypeArray)) {
              vJsonValue |= pP.mJsonArray();
            }
            if (vTypes.has(common_chat_schema::cTypeNumber) || vTypes.has(common_chat_schema::cTypeInteger)) {
              vJsonValue |= pP.mJsonNumber();
            }
            if (vTypes.has(common_chat_schema::cTypeBoolean)) {
              vJsonValue |= pP.mJsonBool();
            }
            if (vTypes.has(common_chat_schema::cTypeNull)) {
              vJsonValue |= pP.mJsonNull();
            }
            vArgValue =
              pP.mGbnf(pP.mAtomic(pP.mToolArgJsonValue(vJsonValue) + vArgClose) | vArgString, "xml-arg-string");
          }

          auto vArgRule = pP.mRule(vRuleName, pP.mToolArg(vArgOpen + vArgValue));

          (pParam.vRequired ? lRequiredArgs : lOptionalArgs).push_back(vArgRule);
        });

        // Accept required arguments in any order, as Qwen does not always adhere to the
        // order provided.
        auto vArgs = pP.mPermute("tool-" + std::to_string(pToolIndex) + "-args", lRequiredArgs);
        if (!lOptionalArgs.empty()) {
          vArgs = vArgs + pP.mZeroOrMore(pP.mChoice(lOptionalArgs));
        }

        auto vFunc = pP.mTool(pP.mToolOpen("<function=" + pP.mToolName(pP.mLiteral(name)) + ">\n") +
                              pP.mToolArgs(vArgs) + pP.mToolClose(pP.mLiteral("</function>\n")));

        vToolChoice |= pP.mRule("tool-" + std::to_string(pToolIndex), vFunc);
      });

      auto vMinCalls = pInputs.vToolChoice == cCommonChatToolChoiceRequired ? 1 : 0;

      auto vToolCallBody = vToolChoice + "</tool_call>" + pP.mSpace();
      auto vToolCall     = pP.mRule("tool-call", "<tool_call>\n" + vToolCallBody);

      // Qwen3-Coder models may occasionally omit the <tool_call> token.
      auto vToolCallFirst = vIsQwen3Coder ?
                              pP.mRule("tool-call-first", pP.mOptional(pP.mLiteral("<tool_call>\n")) + vToolCallBody) :
                              vToolCall;

      auto vCalls     = pInputs.vParallelToolCalls ? vToolCallFirst + pP.mZeroOrMore(vToolCall) : vToolCallFirst;
      auto vToolCalls = pP.mTriggerRule("tool-call-root", pP.mRepeat(vCalls, vMinCalls, 1));

      return vGenerationPrompt + (vReasoning << pP.mContent(pP.mUntilOneOf(lToolCallStarts)) << vToolCalls);
    }

    // Content only parser
    return vGenerationPrompt + (vReasoning << pP.mContent(pP.mRest()));
  });

  if (vIncludeGrammar) {
    data.vGrammarLazy = vHasTools && pInputs.vToolChoice == cCommonChatToolChoiceAuto;

    data.vGrammar = fBuildGrammar(
      [&](const common_grammar_builder & pBuilder) { data.vParser.mBuildGrammar(pBuilder, data.vGrammarLazy); });

    if (data.vGrammarLazy) {
      for (const auto & vStart : lToolCallStarts) {
        data.lGrammarTriggers.push_back({ cCommonGrammarTriggerTypeWord, vStart });
      }
    }
  }

  return data;
}
