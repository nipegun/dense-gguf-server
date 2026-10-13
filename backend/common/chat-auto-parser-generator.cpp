#include "chat-auto-parser-helpers.h"
#include "chat-auto-parser.h"
#include "chat-peg-parser.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "log.h"
#include "parsers/parsers.h"
#include "peg-parser.h"

#include <stdexcept>
#include <string>

using json = common_json;

namespace autoparser {

parser_build_context::parser_build_context(common_chat_peg_builder & pP, const generation_params & pInputs) :
  vP(pP),
  vInputs(pInputs),
  vReasoningParser(pP.mEps()) {}

common_chat_params peg_generator::mGenerateParser(const common_chat_template &     pTmpl,
                                                  const struct generation_params & pInputs) {
  // Run differential analysis to extract template structure
  struct autoparser vAutoparser;
  vAutoparser.mAnalyzeTemplate(pTmpl);
  return mGenerateParser(pTmpl, pInputs, vAutoparser);
}

common_chat_params peg_generator::mGenerateParser(const common_chat_template &     pTmpl,
                                                  const struct generation_params & pInputs,
                                                  const autoparser &               pAutoparser) {
  // Create the result structure
  common_chat_params data;
  data.vPrompt           = fCommonChatTemplateDirectApply(pTmpl, pInputs);
  data.vGenerationPrompt = fCommonChatTemplateGenerationPrompt(pTmpl, pInputs);
  data.format            = cCommonChatFormatPegNative;
  data.lPreservedTokens  = pAutoparser.lPreservedTokens;
  data.lAdditionalStops.insert(data.lAdditionalStops.end(), pAutoparser.lAdditionalStops.begin(),
                               pAutoparser.lAdditionalStops.end());

  std::string vParserGenerationPrompt = data.vGenerationPrompt;

  if (pInputs.vContinueFinalMessage != cCommonChatContinuationNone && !pInputs.vContinueMsg.empty()) {
    // Build up generation prompt manually
    const auto & vMsg = pInputs.vContinueMsg;

    if (!pAutoparser.vReasoning.vStart.empty()) {
      data.vGenerationPrompt =
        data.vGenerationPrompt.substr(0, data.vGenerationPrompt.find(pAutoparser.vReasoning.vStart));
      data.vGenerationPrompt += pAutoparser.vReasoning.vStart + vMsg.vReasoningContent;
      if (pInputs.vContinueFinalMessage == cCommonChatContinuationContent) {
        data.vGenerationPrompt += pAutoparser.vReasoning.end;
      }
    }

    if (pInputs.vContinueFinalMessage == cCommonChatContinuationContent) {
      data.vGenerationPrompt += vMsg.mRenderContent();
    }

    data.vPrompt += data.vGenerationPrompt;
  }

  data.vParser = pAutoparser.mBuildParser(pInputs, vParserGenerationPrompt);

  // Build grammar if tools are present
  bool vHasTools =
    pAutoparser.vTools.format.vMode != tool_format::cNone && pInputs.vTools.is_array() && !pInputs.vTools.empty();
  std::string vTriggerMarker = !pAutoparser.vTools.format.vSectionStart.empty() ?
                                 pAutoparser.vTools.format.vSectionStart :
                                 pAutoparser.vTools.format.vPerCallStart;

  bool vHasResponseFormat = !pInputs.vJsonSchema.empty() && pInputs.vJsonSchema.is_object();
  bool vIncludeGrammar = vHasResponseFormat ||
                         (vHasTools && ((pInputs.vToolChoice == cCommonChatToolChoiceAuto && !vTriggerMarker.empty()) ||
                                        pInputs.vToolChoice == cCommonChatToolChoiceRequired));

  if (vIncludeGrammar) {
    data.vGrammarLazy = !vHasResponseFormat && pInputs.vToolChoice == cCommonChatToolChoiceAuto;
    data.vGrammar     = fBuildGrammar(
      [&](const common_grammar_builder & pBuilder) { data.vParser.mBuildGrammar(pBuilder, data.vGrammarLazy); });

    // Set grammar triggers based on tool section markers (fall back to per-call markers)
    if (data.vGrammarLazy) {
      data.lGrammarTriggers = {
        { cCommonGrammarTriggerTypeWord, vTriggerMarker }
      };
      if (pAutoparser.vTools.format.vOpenaiWrapperTrigger) {
        // model emits the OpenAI function wrapper, trigger on it
        data.lGrammarTriggers.push_back({ cCommonGrammarTriggerTypeWord, "{\"type\": \"function\"," });
      }
    }
  }

  return data;
}

common_peg_arena autoparser::mBuildParser(const generation_params & pInputs,
                                          const std::string &       pGenerationPrompt) const {
  if (!vAnalysisComplete) {
    throw std::invalid_argument(
      "Cannot call build_parser on autoparser without performing analysis first, call analyze_template(...)");
  }
  return fBuildChatPegParser([&](common_chat_peg_builder & pP) {
    parser_build_context vCtx(pP, pInputs);
    bool                 vExtractReasoning = pInputs.vReasoningFormat != cCommonReasoningFormatNone;

    vCtx.vExtractingReasoning = vExtractReasoning && vReasoning.vMode != reasoning_mode::cNone;
    vCtx.vContent             = &vContent;
    vCtx.vReasoning           = &vReasoning;

    // Build reasoning parser
    vCtx.vReasoningParser = vReasoning.mBuildParser(vCtx);

    auto vParser = pP.mEps();

    bool vHasTools          = pInputs.vTools.is_array() && !pInputs.vTools.empty();
    bool vHasResponseFormat = pInputs.vJsonSchema.is_object() && !pInputs.vJsonSchema.empty();
    bool vPureContent       = vReasoning.vMode == reasoning_mode::cNone;

    if (vHasResponseFormat) {
      auto vResponseFormat =
        pP.mRule("response-format", pP.mContent(pP.mSchema(pP.mJson(), "response-format-schema", pInputs.vJsonSchema)));
      vParser = vCtx.vReasoningParser + pP.mSpace() +
                pP.mChoice({ pP.mLiteral("```json") + pP.mSpace() + vResponseFormat + pP.mSpace() + pP.mLiteral("```"),
                             pP.mSpace() + vResponseFormat + pP.mSpace() }) +
                pP.end();
      vPureContent = false;
    } else if (vHasTools && pInputs.vToolChoice != cCommonChatToolChoiceNone && vJinjaCaps.vSupportsToolCalls) {
      vParser      = vTools.mBuildParser(vCtx);
      vPureContent = false;
    } else {
      vParser = vContent.mBuildParser(vCtx);
    }
    const std::string cReasoningStart = fTrimWhitespace(vReasoning.vStart);
    return vPureContent ? pP.mPrefix(pGenerationPrompt, cReasoningStart) + vParser :
                          pP.mPrefix(pGenerationPrompt, cReasoningStart) << vParser;
  });
}

common_peg_parser analyze_reasoning::mBuildParser(parser_build_context & pCtx) const {
  auto & vP = pCtx.vP;

  if (!pCtx.vExtractingReasoning) {
    return vP.mEps();
  }

  if (vMode == reasoning_mode::cTagBased || vMode == reasoning_mode::cToolsOnly) {
    if (!end.empty()) {
      if (!vStart.empty()) {
        // Standard tag-based: optional(<think>reasoning</think>)
        return vP.mOptional(vP.mOptspace(vStart) + vP.mReasoning(vP.mUntil(fTrimWhitespace(end))) + vP.mOptspace(end));
      }
      // Delimiter-style (empty start)
      return vP.mOptional(vP.mReasoning(vP.mUntil(fTrimWhitespace(end))) + vP.mOptspace(end));
    }
  }

  return vP.mEps();
}

common_peg_parser analyze_content::mBuildParser(parser_build_context & pCtx) const {
  auto & vP = pCtx.vP;

  if (mIsAlwaysWrapped()) {
    if (pCtx.vExtractingReasoning) {
      return pCtx.vReasoningParser + vStart + vP.mContent(vP.mUntil(end)) + end + vP.end();
    }
    return vP.mContent(vP.mUntil(vStart)) + vStart + vP.mContent(vP.mUntil(end)) + end + vP.end();
  }
  return pCtx.vReasoningParser + vP.mContent(vP.mRest()) + vP.end();
}

common_peg_parser analyze_content::mBuildOptionalWrapped(parser_build_context & pCtx) const {
  auto & vP = pCtx.vP;

  if (mIsAlwaysWrapped()) {
    return vP.mOptional(vStart + vP.mContent(vP.mUntil(end)) + end);
  }
  return vP.mEps();
}

common_peg_parser analyze_tools::mBuildParser(parser_build_context & pCtx) const {
  switch (format.vMode) {
    case tool_format::cJsonNative:
      return mBuildToolParserJsonNative(pCtx);
    case tool_format::cTagWithJson:
      return mBuildToolParserTagJson(pCtx);
    case tool_format::cTagWithTagged:
      return mBuildToolParserTagTagged(pCtx);
    default:
      LOG_ERR(
        "[ERROR] Template seems to support tool calls, but failed to determine tool format. Tool calling will not work "
        "properly. "
        "Check for a fixed template for your model in the models/templates directory of your dgs.cpp installation or "
        "report an issue at doc/MANUAL.md\n");
      return pCtx.vP.mEps();
  }
}

common_peg_parser analyze_tools::mBuildToolParserJsonNative(parser_build_context & pCtx) const {
  auto &       vP      = pCtx.vP;
  const auto & vInputs = pCtx.vInputs;

  // Build effective field names with dot notation if function_field is set
  std::string vNameField = format.vNameField;
  std::string vArgsField = format.vArgsField;

  if (!format.vFunctionField.empty() && format.vFunctionField != "function" &&
      vNameField.find('.') == std::string::npos) {
    vNameField = format.vFunctionField + "." + vNameField;
    vArgsField = format.vFunctionField + "." + vArgsField;
  }

  auto vToolsParser = vP.mEps();
  if (format.vSectionStart.empty() && !format.vPerCallStart.empty()) {
    auto vSingleToolParser = vP.mStandardJsonTools(
      format.vPerCallStart, format.vPerCallEnd, vInputs.vTools, vInputs.vParallelToolCalls,
      vInputs.vToolChoice == cCommonChatToolChoiceRequired, vNameField, vArgsField, format.vToolsArrayWrapped,
      format.vFunNameIsKey, format.vIdField, format.vGenIdField, format.lParameterOrder, format.vOpenaiWrapperTrigger);
    vToolsParser = vP.mTriggerRule("tool-calls", vP.mOneOrMore(vSingleToolParser + vP.mSpace()));
  } else {
    vToolsParser = vP.mStandardJsonTools(
      format.vSectionStart, format.vSectionEnd, vInputs.vTools, vInputs.vParallelToolCalls,
      vInputs.vToolChoice == cCommonChatToolChoiceRequired, vNameField, vArgsField, format.vToolsArrayWrapped,
      format.vFunNameIsKey, format.vIdField, format.vGenIdField, format.lParameterOrder, format.vOpenaiWrapperTrigger);
  }

  // Handle content wrappers if present
  if (pCtx.vContent && pCtx.vContent->mIsAlwaysWrapped()) {
    auto vWrappedContent = pCtx.vContent->mBuildOptionalWrapped(pCtx);
    return pCtx.vReasoningParser + vWrappedContent + vToolsParser + vP.end();
  }

  std::string vToolStart = "{";
  if (!format.vSectionStart.empty()) {
    vToolStart = format.vSectionStart;
  } else if (!format.vPerCallStart.empty()) {
    vToolStart = format.vPerCallStart;
  }

  return pCtx.vReasoningParser + vP.mOptional(vP.mContent(vP.mUntil(vToolStart))) + vToolsParser + vP.end();
}

common_peg_parser analyze_tools::mBuildFuncParser(common_chat_peg_builder &        pP,
                                                  const std::string &              name,
                                                  const common_peg_parser &        pCallIdSection,
                                                  bool                             pHaveCallId,
                                                  const common_peg_parser &        pArgs,
                                                  std::optional<common_peg_parser> pAtomicPeek) const {
  auto vOpen          = pP.mToolOpen(vFunction.vNamePrefix + pP.mToolName(pP.mLiteral(name)) + vFunction.vNameSuffix);
  bool vMatchedAtomic = false;
  common_peg_parser vFuncParser = pP.mEps();

  if (!vFunction.vArgsSeparator.empty()) {
    vOpen = vOpen + pP.mSpace() + pP.mLiteral(vFunction.vArgsSeparator);
  }

  if (!vFunction.vNameSuffix.empty()) {
    vFuncParser    = vOpen + pCallIdSection + pP.mSpace() + pArgs;
    vMatchedAtomic = true;
  } else if (pHaveCallId) {
    vFuncParser    = pP.mAtomic(vOpen + pCallIdSection) + pP.mSpace() + pArgs;
    vMatchedAtomic = true;
  } else if (pAtomicPeek.has_value()) {
    vFuncParser    = pP.mAtomic(vOpen + pCallIdSection + pP.mSpace() + *pAtomicPeek) + pArgs;
    vMatchedAtomic = true;
  } else {
    vFuncParser = vOpen + pCallIdSection + pP.mSpace() + pArgs;
  }

  if (!vFunction.vClose.empty()) {
    vFuncParser = vFuncParser + pP.mSpace() + pP.mToolClose(pP.mLiteral(vFunction.vClose));
  } else if (!format.vPerCallEnd.empty()) {
    // When there's no func_close but there is a per_call_end marker, use peek() to ensure
    // we only emit tool_close when we can actually see the closing marker. This prevents
    // premature closing during partial parsing when we've seen e.g. "</" which could be
    // either "</tool_call>" (end) or "<arg_key>" prefix that failed to match.
    // Laguna (v4): the model may emit whitespace between the last </arg_value> and
    // </tool_call> even though the template renders them tight. Tolerate optional
    // leading space in the close lookahead so the tool call still closes.
    auto vClosePeek = vArguments.vTolerateIntertagWhitespace ? pP.mPeek(pP.mSpace() + pP.mLiteral(format.vPerCallEnd)) :
                                                               pP.mPeek(pP.mLiteral(format.vPerCallEnd));
    vFuncParser     = vFuncParser + pP.mToolClose(vClosePeek);
  } else {
    vFuncParser = vFuncParser + pP.mToolClose(pP.mSpace());  // force this to process tool closing callbacks in mapper
  }
  if (!vMatchedAtomic) {
    vFuncParser = pP.mAtomic(vFuncParser);
  }
  return vFuncParser;
}

common_peg_parser analyze_tools::mBuildToolParserTagJson(parser_build_context & pCtx) const {
  auto &       vP      = pCtx.vP;
  const auto & vInputs = pCtx.vInputs;

  common_peg_parser vToolChoice = vP.mChoice();

  fForeachFunction(vInputs.vTools, [&](size_t pToolIndex, const json & pTool) {
    const auto & vFunc   = pTool.at("function");
    std::string  name    = vFunc.at("name");
    const auto   cSchema = fCommonChatToolParameters(vFunc);

    // Build call_id parser based on position (if supported)
    bool              vHaveCallId    = false;
    common_peg_parser vCallIdSection = vP.mEps();
    if (vCallId.vPos == call_id_position::cBetweenFuncAndArgs && !vCallId.vPrefix.empty() &&
        (!vCallId.vSuffix.empty() || !vArguments.vStart.empty())) {
      if (!vCallId.vSuffix.empty()) {
        vCallIdSection = vP.mOptional(vCallId.vPrefix + vP.mToolId(vP.mUntil(vCallId.vSuffix))) + vCallId.vSuffix;
      } else {
        vCallIdSection = vP.mOptional(vCallId.vPrefix + vP.mToolId(vP.mUntil(vArguments.vStart)));
      }
      vHaveCallId = true;
    }
    auto vArgsParser = vP.mToolArgs(vP.mSchema(vP.mJson(), "tool-" + std::to_string(pToolIndex) + "-schema", cSchema));
    if (!vArguments.vStart.empty()) {
      vArgsParser = vP.mLiteral(vArguments.vStart) + vArgsParser;
    }
    if (!vArguments.end.empty()) {
      vArgsParser = vArgsParser + vP.mLiteral(vArguments.end);
    }

    auto vAtomicPeek =
      !vArguments.vStart.empty() ? std::optional(vP.mPeek(vP.mLiteral(vArguments.vStart))) : std::nullopt;
    auto vFuncParser = mBuildFuncParser(vP, name, vCallIdSection, vHaveCallId, vArgsParser, vAtomicPeek);
    vToolChoice |= vP.mRule("tool-" + std::to_string(pToolIndex), vFuncParser);
  });

  auto vRequireCalls = vInputs.vToolChoice == cCommonChatToolChoiceRequired;

  common_peg_parser vToolCalls = vP.mEps();

  if (!format.vPerCallStart.empty()) {
    auto vWrappedCall = format.vPerCallStart + vToolChoice + format.vPerCallEnd;
    if (vInputs.vParallelToolCalls) {
      vToolCalls = vP.mTriggerRule("tool-call", vWrappedCall + vP.mZeroOrMore(vP.mSpace() + vWrappedCall));
    } else {
      vToolCalls = vP.mTriggerRule("tool-call", vWrappedCall);
    }
    if (!format.vSectionStart.empty()) {
      vToolCalls =
        vP.mTriggerRule("tool-calls", vP.mLiteral(format.vSectionStart) + vP.mSpace() + vToolCalls + vP.mSpace() +
                                        (format.vSectionEnd.empty() ? vP.end() : vP.mLiteral(format.vSectionEnd)));
    }
  } else {
    std::string vSeparator = ", ";  // Default
    if (vInputs.vParallelToolCalls) {
      vToolCalls = vP.mTriggerRule("tool-call", format.vSectionStart + vToolChoice +
                                                  vP.mZeroOrMore(vSeparator + vToolChoice) + format.vSectionEnd);
    } else {
      vToolCalls = vP.mTriggerRule("tool-call", format.vSectionStart + vToolChoice + format.vSectionEnd);
    }
  }

  if (!vRequireCalls) {
    vToolCalls = vP.mOptional(vToolCalls);
  }

  std::string vTriggerMarker      = !format.vSectionStart.empty() ? format.vSectionStart : format.vPerCallStart;
  auto        vContentBeforeTools = vTriggerMarker.empty() ? vP.mEps() : vP.mUntil(vTriggerMarker);
  return pCtx.vReasoningParser + vP.mOptional(vP.mContent(vContentBeforeTools)) + vToolCalls + vP.end();
}

common_peg_parser analyze_tools::mBuildToolParserTagTagged(parser_build_context & pCtx) const {
  auto &       vP      = pCtx.vP;
  const auto & vInputs = pCtx.vInputs;

  auto vUntilSuffix = vP.mRule("until-suffix", vP.mUntil(vArguments.vValueSuffix));

  common_peg_parser vToolChoice = vP.mChoice();

  fForeachFunction(vInputs.vTools, [&](size_t pToolIndex, const json & pTool) {
    const auto & vFunc = pTool.at("function");
    std::string  name  = vFunc.at("name");

    // Build parser for each argument, separating required and optional
    std::vector<common_peg_parser> lRequiredParsers;
    std::vector<common_peg_parser> lOptionalParsers;
    fForeachParameter(vFunc, [&](size_t pParamIndex, const common_chat_schema_property & pParam,
                                 const common_chat_schema_document_ptr & pDoc) {
      auto vArg = vP.mToolArg(
        vP.mToolArgOpen(vArguments.vNamePrefix + vP.mToolArgName(vP.mLiteral(pParam.name)) + vArguments.vNameSuffix) +
        vArguments.vValuePrefix +
        (pParam.vSchema->mMayBeString() ?
           vP.mAc(vP.mToolArgStringValue(vUntilSuffix) + vP.mToolArgClose(vP.mLiteral(vArguments.vValueSuffix)),
                  vArguments.vValueSuffix) :
           (vP.mToolArgJsonValue(vP.mSchema(
              vP.mJson(), "tool-" + std::to_string(pToolIndex) + "-arg-" + std::to_string(pParamIndex) + "-schema",
              pDoc, *pParam.vSchema)) +
            vP.mToolArgClose(vP.mLiteral(vArguments.vValueSuffix)))));

      auto vNamedArg = vP.mRule("tool-" + std::to_string(pToolIndex) + "-arg-" + std::to_string(pParamIndex), vArg);
      if (pParam.vRequired) {
        lRequiredParsers.push_back(vNamedArg);
      } else {
        lOptionalParsers.push_back(vNamedArg);
      }
    });

    // Build required arg sequence in definition order
    common_peg_parser vArgsSeq = vP.mEps();
    for (size_t vI = 0; vI < lRequiredParsers.size(); vI++) {
      if (vI > 0) {
        vArgsSeq = vArgsSeq + vP.mSpace();
      }
      vArgsSeq = vArgsSeq + lRequiredParsers[vI];
    }

    // Build optional args with flexible ordering
    if (!lOptionalParsers.empty()) {
      common_peg_parser vAnyOpt = vP.mChoice();
      for (const auto & vOpt : lOptionalParsers) {
        vAnyOpt |= vOpt;
      }
      vArgsSeq = vArgsSeq + vP.mRepeat(vP.mSpace() + vAnyOpt, 0, -1);
    }

    if (!vArguments.vStart.empty()) {
      vArgsSeq = vP.mLiteral(vArguments.vStart) + vArgsSeq;
    }
    if (!vArguments.end.empty()) {
      vArgsSeq = vArgsSeq + vP.mLiteral(vArguments.end);
    }

    // Build call_id parser based on position (if supported)
    common_peg_parser vCallIdSection = vP.mEps();
    bool              vHaveCallId    = false;
    if (vCallId.vPos == call_id_position::cBetweenFuncAndArgs && !vCallId.vPrefix.empty() &&
        (!vCallId.vSuffix.empty() || !vArguments.vStart.empty())) {
      vHaveCallId = true;
      if (!vCallId.vSuffix.empty()) {
        vCallIdSection = vP.mOptional(vCallId.vPrefix + vP.mToolId(vP.mUntil(vCallId.vSuffix)) + vCallId.vSuffix);
      } else {
        vCallIdSection = vP.mOptional(vCallId.vPrefix + vP.mToolId(vP.mUntil(vArguments.vStart)));
      }
    }

    // Only peek for an arg tag when there are required args that must follow.
    // When all args are optional, the model may emit no arg tags at all (#20650).
    auto vAtomicPeek = (!vArguments.vNamePrefix.empty() && !lRequiredParsers.empty()) ?
                         std::optional(vP.mPeek(vP.mLiteral(vArguments.vNamePrefix))) :
                         std::nullopt;
    auto vFuncParser = mBuildFuncParser(vP, name, vCallIdSection, vHaveCallId, vArgsSeq, vAtomicPeek);
    vToolChoice |= vP.mRule("tool-" + std::to_string(pToolIndex), vFuncParser);
  });

  auto vRequireTools = vInputs.vToolChoice == cCommonChatToolChoiceRequired;

  common_peg_parser vToolCalls = vP.mEps();

  if (!format.vPerCallStart.empty()) {
    auto vWrappedCall = format.vPerCallStart + vP.mSpace() + vToolChoice + vP.mSpace() + format.vPerCallEnd;
    if (vInputs.vParallelToolCalls) {
      vToolCalls =
        vP.mTriggerRule("tool-call", vWrappedCall + vP.mZeroOrMore(vP.mSpace() + vWrappedCall) + vP.mSpace());
    } else {
      vToolCalls = vP.mTriggerRule("tool-call", vWrappedCall + vP.mSpace());
    }
    if (!format.vSectionStart.empty()) {
      vToolCalls = vP.mTriggerRule(
        "tool-calls", vP.mLiteral(format.vSectionStart) + vP.mSpace() + vToolCalls + vP.mSpace() +
                        (format.vSectionEnd.empty() ? vP.end() : vP.mLiteral(format.vSectionEnd) + vP.mSpace()));
    }
  } else {
    std::string vSeparator = ", ";  // Default

    if (vInputs.vParallelToolCalls) {
      vToolCalls =
        vP.mTriggerRule("tool-call", format.vSectionStart + vP.mSpace() + vToolChoice +
                                       vP.mZeroOrMore(vSeparator + vToolChoice) + vP.mSpace() + format.vSectionEnd);
    } else {
      vToolCalls = vP.mTriggerRule("tool-call",
                                   format.vSectionStart + vP.mSpace() + vToolChoice + vP.mSpace() + format.vSectionEnd);
    }
  }

  if (!vRequireTools) {
    vToolCalls = vP.mOptional(vToolCalls);
  }

  std::string vTriggerMarker      = !format.vSectionStart.empty() ? format.vSectionStart : format.vPerCallStart;
  auto        vContentBeforeTools = vTriggerMarker.empty() ? vP.mEps() : vP.mUntil(vTriggerMarker);
  return pCtx.vReasoningParser + vP.mOptional(vP.mContent(vContentBeforeTools)) + vToolCalls + vP.end();
}

}  // namespace autoparser
