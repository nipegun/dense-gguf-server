#include "chat-peg-parser.h"

#include "chat-auto-parser.h"
#include "ggml.h"
#include "peg-parser.h"

#include <cstdint>
#include <functional>

using ordered_json = common_json;

static std::string_view fTrimTrailingSpace(std::string_view pSv, int max = -1) {
  int count = 0;
  while (!pSv.empty() && std::isspace(static_cast<unsigned char>(pSv.back()))) {
    if (max != -1 && count >= max) {
      break;
    }
    pSv.remove_suffix(1);
    count++;
  }
  return pSv;
}

static std::string_view fTrimLeadingSpace(std::string_view pSv, int max = -1) {
  int count = 0;
  while (!pSv.empty() && std::isspace(static_cast<unsigned char>(pSv.front()))) {
    if (max != -1 && count >= max) {
      break;
    }
    pSv.remove_prefix(1);
    count++;
  }
  return pSv;
}

static std::string_view fTrim(std::string_view pSv) {
  return fTrimTrailingSpace(fTrimLeadingSpace(pSv, 1));
}

// Count the number of unclosed '{' braces in a JSON-like string,
// properly skipping braces inside quoted strings.
static int fJsonBraceDepth(const std::string & pS) {
  int  vDepth    = 0;
  bool vInString = false;
  bool vEscaped  = false;
  for (char vC : pS) {
    if (vEscaped) {
      vEscaped = false;
      continue;
    }
    if (vC == '\\' && vInString) {
      vEscaped = true;
      continue;
    }
    if (vC == '"') {
      vInString = !vInString;
      continue;
    }
    if (!vInString) {
      if (vC == '{') {
        vDepth++;
      } else if (vC == '}') {
        vDepth--;
      }
    }
  }
  return vDepth;
}

// JSON-escape a string and return the inner content (without surrounding quotes).
static std::string fEscapeJsonStringInner(const std::string & pS) {
  std::string vEscaped = ordered_json(pS).mDump();
  if (vEscaped.size() >= 2 && vEscaped.front() == '"' && vEscaped.back() == '"') {
    return vEscaped.substr(1, vEscaped.size() - 2);
  }
  return vEscaped;
}

// Convert Python-style single-quoted strings to JSON double-quoted strings
// Only converts outer string delimiters, properly handling escape sequences:
// - {'key': 'value'} -> {"key": "value"}
// - {'code': 'print(\'hello\')'} -> {"code": "print('hello')"}
// - {'msg': 'He said "hi"'} -> {"msg": "He said \"hi\""}
static std::string fNormalizeQuotesToJson(const std::string & pInput) {
  std::string vResult;
  vResult.reserve(pInput.size() + 16);  // May need extra space for escaping

  bool vInSingleQuoted = false;
  bool vInDoubleQuoted = false;

  auto vIsWordChar = [](char pCh) {
    return std::isalnum(static_cast<unsigned char>(pCh)) || pCh == '_';
  };

  for (size_t vI = 0; vI < pInput.size(); ++vI) {
    char vC = pInput[vI];

    // Handle escape sequences
    if (vC == '\\' && vI + 1 < pInput.size()) {
      char vNext = pInput[vI + 1];

      if (vInSingleQuoted) {
        // Inside a single-quoted string being converted to double quotes
        if (vNext == '\'') {
          // \' -> ' (escaped single quote becomes unescaped in double-quoted string)
          vResult += '\'';
          ++vI;
          continue;
        }
        if (vNext == '"') {
          // \" stays as \" (already escaped, works in double-quoted string)
          vResult += "\\\"";
          ++vI;
          continue;
        }
        // Other escapes (\n, \\, etc.): pass through both characters
        vResult += vC;
        vResult += vNext;
        ++vI;
        continue;
      }

      if (vInDoubleQuoted) {
        // Inside a double-quoted string - pass through escape sequences as-is
        vResult += vC;
        vResult += vNext;
        ++vI;
        continue;
      }

      // Outside any string - just pass through the backslash
      vResult += vC;
      continue;
    }

    // Handle quote characters
    if (vC == '"') {
      if (vInSingleQuoted) {
        // Unescaped double quote inside single-quoted string -> must escape for JSON
        vResult += "\\\"";
      } else {
        // Double quote as string delimiter or outside strings
        vInDoubleQuoted = !vInDoubleQuoted;
        vResult += vC;
      }
    } else if (vC == '\'') {
      if (vInDoubleQuoted) {
        // Single quote inside double-quoted string -> pass through
        vResult += vC;
      } else if (vInSingleQuoted) {
        // Closing single quote -> convert to double quote
        vInSingleQuoted = false;
        vResult += '"';
      } else {
        // Opening single quote -> convert to double quote
        vInSingleQuoted = true;
        vResult += '"';
      }
    } else if (!vInSingleQuoted && !vInDoubleQuoted && (vC == 'T' || vC == 'F' || vC == 'N') &&
               (vI == 0 || !vIsWordChar(pInput[vI - 1]))) {
      // Python literals -> JSON; prefix match keeps streamed partials monotonic.
      static constexpr std::pair<std::string_view, std::string_view> aLiterals[] = {
        { "True",  "true"  },
        { "False", "false" },
        { "None",  "null"  },
      };
      size_t n = 0;
      while (vI + n < pInput.size() && vIsWordChar(pInput[vI + n])) {
        ++n;
      }
      std::string_view token(pInput.data() + vI, n);
      bool             vMatched = false;
      for (const auto & [py, js] : aLiterals) {
        if (py.substr(0, n) == token) {
          vResult += js.substr(0, n);
          vI += n - 1;
          vMatched = true;
          break;
        }
      }
      if (!vMatched) {
        vResult += vC;
      }
    } else {
      vResult += vC;
    }
  }

  return vResult;
}

void tag_based_peg_mapper::mFromAst(const common_peg_ast_arena & pArena, const common_peg_parse_result & pResult) {
  pArena.mVisit(pResult, [this](const common_peg_ast_node & node) {
    if (!node.vTag.empty()) {
      dTags[node.vTag] = std::string(node.vText);
    }
  });
}

tagged_parse_result tagged_peg_parser::mParseAndExtract(const std::string &    pInput,
                                                        common_peg_parse_flags pExtraFlags) const {
  common_peg_parse_context vCtx(pInput, vFlags | pExtraFlags);
  auto                     vParseResult = vArena.mParse(vCtx);

  tag_based_peg_mapper vMapper;
  vMapper.mFromAst(vCtx.vAst, vParseResult);

  return { std::move(vParseResult), std::move(vMapper.dTags) };
}

tagged_parse_result tagged_peg_parser::mParseAnywhereAndExtract(const std::string & pInput) const {
  if (pInput.empty()) {
    return mParseAndExtract(pInput);
  }
  for (size_t vI = 0; vI < pInput.size(); vI++) {
    common_peg_parse_context vCtx(pInput, vFlags);
    auto                     vParseResult = vArena.mParse(vCtx, vI);
    if (vParseResult.mSuccess() || vI == pInput.size() - 1) {
      tag_based_peg_mapper vMapper;
      vMapper.mFromAst(vCtx.vAst, vParseResult);
      return { std::move(vParseResult), std::move(vMapper.dTags) };
    }
  }
  GGML_ABORT("Should not happen");
}

tagged_peg_parser fBuildTaggedPegParser(
  const std::function<common_peg_parser(common_peg_parser_builder & pBuilder)> & pFn) {
  common_peg_parser_builder vBuilder;
  vBuilder.mSetRoot(pFn(vBuilder));
  return { vBuilder.mBuild() };
}

common_peg_parser common_chat_peg_builder::mTagWithSafeContent(const std::string &       pTagName,
                                                               const std::string &       pMarker,
                                                               const common_peg_parser & pP) {
  if (pMarker.empty()) {
    return mZeroOrMore(mChoice({ pP, mRule(pTagName, mContent(mAny())) }));
  }
  auto vContentChunk = mRule(pTagName, mContent(mNegate(mLiteral(pMarker)) + mAny() + mUntil(pMarker)));
  return mZeroOrMore(mChoice({ pP, vContentChunk }));
}

common_peg_parser common_chat_peg_builder::mPermute(const std::string &                    pRulePrefix,
                                                    const std::vector<common_peg_parser> & pParsers) {
  if (pParsers.empty()) {
    return mEps();
  }

  if (pParsers.size() == 1 || pParsers.size() > cCommonChatMaxPermute) {
    return mSequence(pParsers);
  }

  std::map<uint32_t, common_peg_parser>      dRules;
  std::function<common_peg_parser(uint32_t)> vRemainingOf;

  vRemainingOf = [&](uint32_t pRemaining) -> common_peg_parser {
    if (pRemaining == 0) {
      return mEps();
    }

    auto vCached = dRules.find(pRemaining);
    if (vCached != dRules.end()) {
      return vCached->second;
    }

    auto vAlternatives = mChoice();
    for (size_t vI = 0; vI < pParsers.size(); vI++) {
      const uint32_t cBit = 1u << vI;
      if (pRemaining & cBit) {
        vAlternatives |= pParsers[vI] + vRemainingOf(pRemaining & ~cBit);
      }
    }

    return dRules.emplace(pRemaining, mRule(pRulePrefix + "-" + std::to_string(pRemaining), vAlternatives))
      .first->second;
  };

  return vRemainingOf((1u << pParsers.size()) - 1);
}

std::string & common_chat_peg_mapper::mArgsTarget() {
  return (vCurrentTool && !vCurrentTool->name.empty()) ? vCurrentTool->vArguments : vArgsBuffer;
}

std::string common_chat_peg_mapper::mNormalizeContainerValue(const std::string & pInput) {
  return fNormalizeQuotesToJson(pInput);
}

void common_chat_peg_mapper::mFromAst(const common_peg_ast_arena &    pArena,
                                      const common_peg_parse_result & pParseResultArg) {
  pArena.mVisit(pParseResultArg, [this](const common_peg_ast_node & node) { mMap(node); });
  // Flush any pending tool call that was started but never got a name
  // This happens during partial parsing when the tool call is incomplete
  if (vPendingToolCall.has_value() && !vPendingToolCall->name.empty()) {
    if (!vArgsBuffer.empty()) {
      vPendingToolCall->vArguments = vArgsBuffer;
    }
    if (vClosingQuotePending && !vPendingToolCall->vArguments.empty()) {
      vPendingToolCall->vArguments += "\"";
    }
    vResult.lToolCalls.push_back(vPendingToolCall.value());
    vPendingToolCall.reset();
  }

  // Discard whitespace-only reasoning content (e.g. from <think></think> prefill)
  if (!vResult.vReasoningContent.empty()) {
    bool vAllWhitespace = true;
    for (char vC : vResult.vReasoningContent) {
      if (vC != ' ' && vC != '\n' && vC != '\r' && vC != '\t') {
        vAllWhitespace = false;
        break;
      }
    }
    if (vAllWhitespace) {
      vResult.vReasoningContent.clear();
    }
  }
}

void common_chat_peg_mapper::mMap(const common_peg_ast_node & node) {
  // Handle reasoning/content tags
  bool vIsReasoning = node.vTag == common_chat_peg_builder::cReasoning;
  bool vIsContent   = node.vTag == common_chat_peg_builder::cContent;

  if (vIsReasoning) {  // GPT OSS can have more than 1 reasoning block, so concatenate here
    vResult.vReasoningContent += node.mSanitizedText();
  }

  if (vIsContent) {
    // Concatenate content from multiple content nodes (e.g., when reasoning markers
    // are preserved before content markers in reasoning_format=NONE mode)
    vResult.vContent += node.mSanitizedText();
  }

  // Handle tool-related tags (supporting both JSON and tagged formats)
  bool vIsToolOpen       = node.vTag == common_chat_peg_builder::cToolOpen;
  bool vIsToolClose      = node.vTag == common_chat_peg_builder::cToolClose;
  bool vIsToolName       = node.vTag == common_chat_peg_builder::cToolName;
  bool vIsToolId         = node.vTag == common_chat_peg_builder::cToolId;
  bool vIsToolArgs       = node.vTag == common_chat_peg_builder::cToolArgs;
  bool vIsArgOpen        = node.vTag == common_chat_peg_builder::cToolArgOpen;
  bool vIsArgClose       = node.vTag == common_chat_peg_builder::cToolArgClose;
  bool vIsArgName        = node.vTag == common_chat_peg_builder::cToolArgName;
  bool vIsArgValue       = node.vTag == common_chat_peg_builder::cToolArgValue;
  bool vIsArgStringValue = node.vTag == common_chat_peg_builder::cToolArgStringValue;

  if (vIsToolOpen) {
    vPendingToolCall = common_chat_tool_call();
    vCurrentTool     = &vPendingToolCall.value();
    vArgCount        = 0;
    vArgsBuffer.clear();
    vClosingQuotePending = false;
  }

  if (vIsToolId && vCurrentTool) {
    auto vText = fTrimTrailingSpace(node.vText);
    if (vText.size() >= 2 && vText.front() == '"' && vText.back() == '"') {
      vText = vText.substr(1, vText.size() - 2);
    }
    vCurrentTool->vId = std::string(vText);
  }

  if (vIsToolName && vCurrentTool) {
    vCurrentTool->name = std::string(fTrimTrailingSpace(node.vText));
    // Now that we have the name, populate the arguments from the buffer
    if (!vArgsBuffer.empty()) {
      vCurrentTool->vArguments = vArgsBuffer;
      vArgsBuffer.clear();
    } else if (vCurrentTool->vArguments.empty()) {
      vCurrentTool->vArguments = "{";
    }
    // Add the tool call to results so streaming can see it
    if (vPendingToolCall.has_value()) {
      vResult.lToolCalls.push_back(vPendingToolCall.value());
      vPendingToolCall.reset();
      vCurrentTool = &vResult.lToolCalls.back();
    }
  }

  if (vIsToolArgs && vCurrentTool) {
    // For JSON format: arguments come as a complete JSON object
    // For tagged format: built up from individual arg_name/arg_value nodes
    auto vText = fTrimTrailingSpace(node.vText);
    if (!vText.empty() && vText.front() == '{') {
      mArgsTarget() = std::string(vText);
    }
  }

  if (vIsArgOpen) {
    vClosingQuotePending = false;
  }

  if (vIsArgName && vCurrentTool) {
    std::string vArgEntry;
    if (vArgCount > 0) {
      vArgEntry = ",";
    }
    vArgEntry += ordered_json(fTrim(node.vText)).mDump() + ":";
    ++vArgCount;

    auto & vTarget = mArgsTarget();
    if (vTarget.empty()) {
      vTarget = "{";
    }
    vTarget += vArgEntry;
  }

  if ((vIsArgValue || vIsArgStringValue) && vCurrentTool) {
    std::string vValueContent = std::string(node.vText);

    std::string vValueToAdd;
    if (vValueContent.empty() && vIsArgStringValue) {
      // Empty string value - arg_close will add the closing quote
      vValueToAdd          = "\"";
      vClosingQuotePending = true;
    } else if (!vValueContent.empty() && vIsArgStringValue) {
      // Schema declares this as string type - always treat as literal string value
      if (!vClosingQuotePending) {
        vValueToAdd          = "\"";
        vClosingQuotePending = true;
      }
      vValueToAdd += fEscapeJsonStringInner(vValueContent);
    } else if (!vValueContent.empty()) {
      // Pythonic scalars/containers -> JSON.
      vValueToAdd += mNormalizeContainerValue(vValueContent);
    }

    mArgsTarget() += vValueToAdd;
  }

  if (vIsArgClose && vCurrentTool) {
    if (vClosingQuotePending) {
      mArgsTarget() += "\"";
      vClosingQuotePending = false;
    }
  }

  if (vIsToolClose && vCurrentTool) {
    // Flush buffer to arguments if tool name was never seen
    if (vCurrentTool->name.empty() && !vArgsBuffer.empty()) {
      vCurrentTool->vArguments = vArgsBuffer;
      vArgsBuffer.clear();
    }
    // Close any pending string quote
    if (vClosingQuotePending) {
      vCurrentTool->vArguments += "\"";
      vClosingQuotePending = false;
    }
    // Close any unclosed braces (accounts for nested objects)
    for (int vD = fJsonBraceDepth(vCurrentTool->vArguments); vD > 0; vD--) {
      vCurrentTool->vArguments += "}";
    }
    // Add tool call to results if named; otherwise discard
    if (vPendingToolCall.has_value()) {
      if (!vCurrentTool->name.empty()) {
        vResult.lToolCalls.push_back(vPendingToolCall.value());
      }
      vPendingToolCall.reset();
      vCurrentTool = nullptr;
    }
  }
}

common_peg_parser common_chat_peg_builder::mStandardConstructedTools(
  const std::map<std::string, std::string> & pMarkers,
  const ordered_json &                       pTools,
  bool                                       pParallelToolCalls,
  bool                                       pForceToolCalls) {
  if (!pTools.is_array() || pTools.empty()) {
    return mEps();
  }

  // Extract markers with defaults
  auto vGetMarker = [&pMarkers](const std::string & pKey, const std::string & pDefaultVal = "") -> std::string {
    auto vIt = pMarkers.find(pKey);
    return vIt != pMarkers.end() ? vIt->second : pDefaultVal;
  };

  std::string vSectionStart   = vGetMarker("tool_call_start_marker", "<tool_call>");
  std::string vSectionEnd     = vGetMarker("tool_call_end_marker", "</tool_call>");
  std::string vFuncOpener     = vGetMarker("function_opener", "<function=");
  std::string vFuncNameSuffix = vGetMarker("function_name_suffix", ">");
  std::string vFuncCloser     = vGetMarker("function_closer", "</function>");
  std::string vParamKeyPrefix = vGetMarker("parameter_key_prefix", "<param=");
  std::string vParamKeySuffix = vGetMarker("parameter_key_suffix", ">");
  std::string vParamCloser    = vGetMarker("parameter_closer", "</param>");

  // Build tool choices for tagged format
  auto vToolChoices = mChoice();

  for (size_t vI = 0; vI < pTools.size(); vI++) {
    const auto & vToolDef = pTools[vI];

    if (!vToolDef.contains("function")) {
      continue;
    }
    const auto & vFunction = vToolDef.at("function");
    std::string  name      = vFunction.at("name");
    ordered_json vParams   = fCommonChatToolParameters(vFunction);

    // Build argument parsers
    auto vArgs = mEps();
    if (vParams.contains("properties") && !vParams["properties"].empty()) {
      auto vArgChoice = mChoice();
      for (const auto & vEl : vParams["properties"].mItems()) {
        const std::string & vPropName = vEl.mKey();

        auto vArgNameParser =
          mChoice({ mLiteral(vPropName), mLiteral("\"" + vPropName + "\""), mLiteral("'" + vPropName + "'") });

        auto vArgRule =
          mToolArg(mToolArgOpen(mLiteral(vParamKeyPrefix)) + mToolArgName(vArgNameParser) + mLiteral(vParamKeySuffix) +
                   mToolArgValue(mUntil(vParamCloser)) + mToolArgClose(mLiteral(vParamCloser)));
        vArgChoice |= vArgRule;
      }
      vArgs = mZeroOrMore(vArgChoice + mSpace());
    }

    // Build function parser: <function=name>args</function>
    auto vToolParser = mTool(mToolOpen(mLiteral(vFuncOpener) + mToolName(mLiteral(name)) + mLiteral(vFuncNameSuffix)) +
                             mSpace() + mToolArgs(vArgs) + mSpace() + mToolClose(mLiteral(vFuncCloser)));

    vToolChoices |= mRule("tool-" + std::to_string(vI), vToolParser);
  }

  // Build the section with markers
  auto vSection =
    pParallelToolCalls ?
      mTriggerRule("tool-call",
                   mLiteral(vSectionStart) + mSpace() + mOneOrMore(vToolChoices + mSpace()) + mLiteral(vSectionEnd)) :
      mTriggerRule("tool-call", mLiteral(vSectionStart) + mSpace() + vToolChoices + mSpace() + mLiteral(vSectionEnd));

  return pForceToolCalls ? vSection : mOptional(vSection);
}

// Like python_value(), but the leaf also accepts JSON-cased true/false/null, used by LFM2/LFM2.5
common_peg_parser common_chat_peg_builder::mPythonOrJsonValue() {
  return mRule("python-or-json-value", [this]() {
    auto vWs   = mSpace();
    auto value = mPythonOrJsonValue();

    auto vMember  = mSequence({ mPythonString(), vWs, mLiteral(":"), vWs, value });
    auto vMembers = mSequence({ vMember, mZeroOrMore(mSequence({ vWs, mLiteral(","), vWs, vMember })) });
    auto vDict    = mRule("python-or-json-dict", [&]() {
      return mSequence(
        { mLiteral("{"), vWs, mChoice({ mLiteral("}"), mSequence({ vMembers, vWs, mLiteral("}") }) }), vWs });
    });

    auto vElements = mSequence({ value, mZeroOrMore(mSequence({ mLiteral(","), vWs, value })) });
    auto vArray    = mRule("python-or-json-array", [&]() {
      return mSequence(
        { mLiteral("["), vWs, mChoice({ mLiteral("]"), mSequence({ vElements, vWs, mLiteral("]") }) }), vWs });
    });

    return mChoice(
      { vDict, vArray, mPythonString(), mPythonNumber(), mPythonBool(), mPythonNull(), mJsonBool(), mJsonNull() });
  });
}

// Python-style tool calls: name(arg1="value1", arg2=123)
// Used only by LFM2 for now, so we don't merge it into autoparser
common_peg_parser common_chat_peg_builder::mPythonStyleToolCalls(const ordered_json & pTools,
                                                                 bool                 pParallelToolCalls,
                                                                 bool                 pAllowJsonLiterals) {
  if (!pTools.is_array() || pTools.empty()) {
    return mEps();
  }

  auto vToolChoices = mChoice();

  for (size_t vI = 0; vI < pTools.size(); vI++) {
    const auto & vToolDef = pTools[vI];
    if (!vToolDef.contains("function")) {
      continue;
    }
    const auto & vFunction = vToolDef.at("function");
    std::string  name      = vFunction.at("name");
    ordered_json vParams   = fCommonChatToolParameters(vFunction);

    auto vArgs = mEps();
    if (vParams.contains("properties") && !vParams["properties"].empty()) {
      auto vArgChoice = mChoice();
      for (const auto & vEl : vParams["properties"].mItems()) {
        const std::string & vPropName     = vEl.mKey();
        const auto &        vPropDef      = vEl.value();
        bool                vIsStringType = (vPropDef.contains("type") && vPropDef["type"] == "string");

        auto vArgNameParser = mLiteral(vPropName);

        common_peg_parser vArgValueParser = mEps();
        // Quoted literal as a value: normalize_quotes_to_json preserves escapes.
        auto vStringValueParser = mToolArgValue(mChoice({ mLiteral("\"") + mStringContent('"') + mLiteral("\""),
                                                          mLiteral("'") + mStringContent('\'') + mLiteral("'") }));

        if (vIsStringType) {
          vArgValueParser = vStringValueParser;
        } else {
          vArgValueParser = mToolArgValue(pAllowJsonLiterals ? mPythonOrJsonValue() : mPythonValue());
        }

        // Full argument: name="value" or name=value
        auto vArgRule = mToolArg(mToolArgOpen(mToolArgName(vArgNameParser) + mLiteral("=")) + vArgValueParser +
                                 mToolArgClose(mEps()));
        vArgChoice |= vArgRule;
      }

      vArgs = vArgChoice + mZeroOrMore("," + mSpace() + vArgChoice);
    }

    auto vToolParser = mTool(mToolOpen(mToolName(mLiteral(name)) + mLiteral("(")) + mSpace() + mToolArgs(vArgs) +
                             mSpace() + mToolClose(mLiteral(")")));

    vToolChoices |= mRule("tool-" + std::to_string(vI), vToolParser);
  }

  if (pParallelToolCalls) {
    return "[" + mSpace() + vToolChoices + mZeroOrMore("," + mSpace() + vToolChoices) + mSpace() + "]";
  }
  return "[" + mSpace() + vToolChoices + mSpace() + "]";
}

// Helper: Parse dot notation key into prefix and field name
static std::pair<std::string, std::string> fParseKeySpec(const std::string & pKey) {
  auto vDotPos = pKey.find('.');
  if (vDotPos == std::string::npos) {
    return { "", pKey };  // Top-level field
  }
  return { pKey.substr(0, vDotPos), pKey.substr(vDotPos + 1) };
}

// Mode 1: function_is_key — parse {"function_name": {...}}
common_peg_parser common_chat_peg_builder::mBuildJsonToolsFunctionIsKey(const ordered_json & pTools,
                                                                        const std::string &  pArgsKey,
                                                                        const std::string &  pEffectiveArgsKey,
                                                                        const std::string &  pCallIdKey,
                                                                        const std::string &  pGenCallIdKey) {
  auto vToolChoices = mChoice();

  for (size_t vI = 0; vI < pTools.size(); vI++) {
    const auto & vToolDef = pTools[vI];
    if (!vToolDef.contains("function")) {
      continue;
    }
    const auto & vFunction = vToolDef.at("function");
    std::string  name      = vFunction.at("name");
    ordered_json vParams   = fCommonChatToolParameters(vFunction);

    // Build inner object fields
    std::vector<common_peg_parser> lInnerFields;

    if (!pCallIdKey.empty()) {
      auto vIdParser = mAtomic(mLiteral("\"" + pCallIdKey + "\"") + mSpace() + mLiteral(":") + mSpace() +
                               mLiteral("\"") + mToolId(mStringContent('"')) + mLiteral("\""));
      lInnerFields.push_back(mOptional(vIdParser + mSpace() + mOptional(mLiteral(",") + mSpace())));
    }

    if (!pGenCallIdKey.empty()) {
      auto vGenIdParser =
        mAtomic(mLiteral("\"" + pGenCallIdKey + "\"") + mSpace() + mLiteral(":") + mSpace() +
                mChoice({ mLiteral("\"") + mToolId(mStringContent('"')) + mLiteral("\""), mToolId(mJsonNumber()) }));
      lInnerFields.push_back(mOptional(vGenIdParser + mSpace() + mOptional(mLiteral(",") + mSpace())));
    }

    // Arguments — either wrapped in args_key or parsed directly
    common_peg_parser vArgsParser = mEps();
    if (pArgsKey.empty()) {
      vArgsParser = mToolArgs(mSchema(mJson(), "tool-" + std::to_string(vI) + "-schema", vParams));
    } else {
      vArgsParser = mLiteral("\"" + pEffectiveArgsKey + "\"") + mSpace() + mLiteral(":") + mSpace() +
                    mToolArgs(mSchema(mJson(), "tool-" + std::to_string(vI) + "-schema", vParams));
    }
    lInnerFields.push_back(vArgsParser);

    // Build inner object parser
    common_peg_parser vInnerObject = mEps();
    if (pArgsKey.empty() && lInnerFields.size() == 1) {
      vInnerObject = lInnerFields[0];
    } else {
      vInnerObject = mLiteral("{") + mSpace();
      for (size_t vI = 0; vI < lInnerFields.size(); vI++) {
        vInnerObject = vInnerObject + lInnerFields[vI];
        if (vI < lInnerFields.size() - 1) {
          vInnerObject = vInnerObject + mSpace();
        }
      }
      vInnerObject = vInnerObject + mSpace() + mLiteral("}");
    }

    auto vToolParser =
      mTool(mToolOpen(mLiteral("{")) + mSpace() + mLiteral("\"") + mToolName(mLiteral(name)) + mLiteral("\"") +
            mSpace() + mLiteral(":") + mSpace() + vInnerObject + mSpace() + mToolClose(mLiteral("}")));

    vToolChoices |= mRule("tool-" + std::to_string(vI), vToolParser);
  }

  return vToolChoices;
}

// Mode 2: Nested keys (dot notation like "function.name")
common_peg_parser common_chat_peg_builder::mBuildJsonToolsNestedKeys(const ordered_json & pTools,
                                                                     const std::string &  pEffectiveNameKey,
                                                                     const std::string &  pEffectiveArgsKey,
                                                                     const std::string &  pCallIdKey,
                                                                     const std::string &  pGenCallIdKey) {
  auto vToolChoices = mChoice();

  auto vNameSpec = fParseKeySpec(pEffectiveNameKey);
  auto vArgsSpec = fParseKeySpec(pEffectiveArgsKey);

  std::string vNestedPrefix    = !vNameSpec.first.empty() ? vNameSpec.first : vArgsSpec.first;
  std::string vNestedNameField = !vNameSpec.first.empty() ? vNameSpec.second : pEffectiveNameKey;
  std::string vNestedArgsField = !vArgsSpec.first.empty() ? vArgsSpec.second : pEffectiveArgsKey;

  for (size_t vI = 0; vI < pTools.size(); vI++) {
    const auto & vToolDef = pTools[vI];
    if (!vToolDef.contains("function")) {
      continue;
    }
    const auto & vFunction = vToolDef.at("function");
    std::string  name      = vFunction.at("name");
    ordered_json vParams   = fCommonChatToolParameters(vFunction);

    auto vNestedName = mLiteral("\"" + vNestedNameField + "\"") + mSpace() + mLiteral(":") + mSpace() +
                       mAtomic(mLiteral("\"") + mToolName(mLiteral(name)) + mLiteral("\""));
    auto vNestedArgs = mLiteral("\"" + vNestedArgsField + "\"") + mSpace() + mLiteral(":") + mSpace() +
                       mToolArgs(mSchema(mJson(), "tool-" + std::to_string(vI) + "-schema", vParams));

    auto vNestedObject = mLiteral("{") + mSpace() + vNestedName + mSpace() + mLiteral(",") + mSpace() + vNestedArgs +
                         mSpace() + mLiteral("}");

    // Format: { id?, "function": {...} }
    auto vToolParserBody = mToolOpen(mLiteral("{")) + mSpace();

    if (!pCallIdKey.empty()) {
      auto vIdSpec = fParseKeySpec(pCallIdKey);
      if (vIdSpec.first.empty()) {
        auto vIdParser  = mAtomic(mLiteral("\"" + pCallIdKey + "\"") + mSpace() + mLiteral(":") + mSpace() +
                                  mLiteral("\"") + mToolId(mStringContent('"')) + mLiteral("\""));
        vToolParserBody = vToolParserBody + mOptional(vIdParser + mSpace() + mLiteral(",") + mSpace());
      }
    }

    if (!pGenCallIdKey.empty()) {
      auto vGenIdSpec = fParseKeySpec(pGenCallIdKey);
      if (vGenIdSpec.first.empty()) {
        auto vGenIdParser =
          mAtomic(mLiteral("\"" + pGenCallIdKey + "\"") + mSpace() + mLiteral(":") + mSpace() +
                  mChoice({ mLiteral("\"") + mToolId(mStringContent('"')) + mLiteral("\""), mToolId(mJsonNumber()) }));
        vToolParserBody = vToolParserBody + mOptional(vGenIdParser + mSpace() + mLiteral(",") + mSpace());
      }
    }

    auto vNestedField = mLiteral("\"" + vNestedPrefix + "\"") + mSpace() + mLiteral(":") + mSpace() + vNestedObject;
    vToolParserBody   = vToolParserBody + vNestedField + mSpace() + mToolClose(mLiteral("}"));

    vToolChoices |= mRule("tool-" + std::to_string(vI), mTool(vToolParserBody));
  }

  return vToolChoices;
}

// Mode 3: Flat keys with optional ID fields and parameter ordering
common_peg_parser common_chat_peg_builder::mBuildJsonToolsFlatKeys(const ordered_json &             pTools,
                                                                   const std::string &              pEffectiveNameKey,
                                                                   const std::string &              pEffectiveArgsKey,
                                                                   const std::string &              pCallIdKey,
                                                                   const std::string &              pGenCallIdKey,
                                                                   const std::vector<std::string> & pParametersOrder,
                                                                   bool pAcceptOpenaiWrapper) {
  auto vToolChoices   = mChoice();
  auto vNameKeyParser = mLiteral("\"" + pEffectiveNameKey + "\"");
  auto vArgsKeyParser = mLiteral("\"" + pEffectiveArgsKey + "\"");

  for (size_t vI = 0; vI < pTools.size(); vI++) {
    const auto & vToolDef = pTools[vI];
    if (!vToolDef.contains("function")) {
      continue;
    }
    const auto & vFunction = vToolDef.at("function");
    std::string  name      = vFunction.at("name");
    ordered_json vParams   = fCommonChatToolParameters(vFunction);

    auto vToolName = vNameKeyParser + mSpace() + mLiteral(":") + mSpace() +
                     mAtomic(mLiteral("\"") + mToolName(mLiteral(name)) + mLiteral("\""));
    auto vToolArgs = vArgsKeyParser + mSpace() + mLiteral(":") + mSpace() +
                     mToolArgs(mSchema(mJson(), "tool-" + std::to_string(vI) + "-schema", vParams));

    // Build ID parsers if keys are provided
    common_peg_parser vIdParser = mEps();
    if (!pCallIdKey.empty()) {
      vIdParser =
        mAtomic(mLiteral("\"" + pCallIdKey + "\"") + mSpace() + mLiteral(":") + mSpace() +
                mChoice({ mLiteral("\"") + mToolId(mStringContent('"')) + mLiteral("\""), mToolId(mJsonNumber()) }));
    }

    common_peg_parser vGenIdParser = mEps();
    if (!pGenCallIdKey.empty()) {
      vGenIdParser =
        mAtomic(mLiteral("\"" + pGenCallIdKey + "\"") + mSpace() + mLiteral(":") + mSpace() +
                mChoice({ mLiteral("\"") + mToolId(mStringContent('"')) + mLiteral("\""), mToolId(mJsonNumber()) }));
    }

    // Create (parser, key) pairs for all fields, then sort by parameters_order
    std::vector<std::pair<common_peg_parser, std::string>> lParserPairs;
    lParserPairs.emplace_back(vToolName, pEffectiveNameKey);
    lParserPairs.emplace_back(vToolArgs, pEffectiveArgsKey);
    if (!pCallIdKey.empty()) {
      lParserPairs.emplace_back(mOptional(vIdParser), pCallIdKey);
    }
    if (!pGenCallIdKey.empty()) {
      lParserPairs.emplace_back(mOptional(vGenIdParser), pGenCallIdKey);
    }

    std::sort(lParserPairs.begin(), lParserPairs.end(), [&pParametersOrder](const auto & pA, const auto & pB) {
      auto   vPosA = std::find(pParametersOrder.begin(), pParametersOrder.end(), pA.second);
      auto   vPosB = std::find(pParametersOrder.begin(), pParametersOrder.end(), pB.second);
      size_t vIdxA =
        (vPosA == pParametersOrder.end()) ? pParametersOrder.size() : std::distance(pParametersOrder.begin(), vPosA);
      size_t vIdxB =
        (vPosB == pParametersOrder.end()) ? pParametersOrder.size() : std::distance(pParametersOrder.begin(), vPosB);
      return vIdxA < vIdxB;
    });

    // accept an optional leading "type": "function" field when the model emits the OpenAI wrapper
    common_peg_parser vTypeField = mEps();
    if (pAcceptOpenaiWrapper) {
      vTypeField = mOptional(mLiteral("\"type\"") + mSpace() + mLiteral(":") + mSpace() + mLiteral("\"function\"") +
                             mSpace() + mLiteral(",") + mSpace());
    }
    auto vOrderedBody = mToolOpen(mLiteral("{")) + mSpace() + vTypeField;
    for (size_t vI = 0; vI < lParserPairs.size(); vI++) {
      vOrderedBody = vOrderedBody + lParserPairs[vI].first;
      if (vI < lParserPairs.size() - 1) {
        vOrderedBody = vOrderedBody + mSpace() + mLiteral(",") + mSpace();
      }
    }
    vOrderedBody = vOrderedBody + mSpace() + mToolClose(mLiteral("}"));

    vToolChoices |= mRule("tool-" + std::to_string(vI), mTool(vOrderedBody));
  }

  return vToolChoices;
}

common_peg_parser common_chat_peg_builder::mPrefix(const std::string & pS, const std::string & pDelimiter) {
  if (pS.empty()) {
    return mEps();
  }
  if (pDelimiter.empty()) {
    return mLiteral(pS);
  }
  return mLiteral(pS.substr(0, pS.find(pDelimiter)));
}

common_peg_parser common_chat_peg_builder::mOptspace(const std::string & pTag) {
  auto   vParser             = mEps();
  size_t vEndOfPrefixSpace   = pTag.size();
  size_t vStartOfSuffixSpace = pTag.size();
  for (size_t vI = 0; vI < pTag.size(); vI++) {
    if (!std::isspace(pTag[vI])) {
      vEndOfPrefixSpace = vI;
      break;
    }
  }
  for (size_t vI = pTag.size(); vI > 0; vI--) {
    if (!std::isspace(pTag[vI - 1])) {
      vStartOfSuffixSpace = vI;
      break;
    }
  }
  for (size_t vI = 0; vI < vEndOfPrefixSpace; vI++) {
    vParser += mOptional(mLiteral(std::string(1, pTag[vI])));
  }
  vParser += mLiteral(pTag.substr(vEndOfPrefixSpace, vStartOfSuffixSpace - vEndOfPrefixSpace));
  for (size_t vI = vStartOfSuffixSpace; vI < pTag.size(); vI++) {
    vParser += mOptional(mLiteral(std::string(1, pTag[vI])));
  }
  return vParser;
}

common_peg_parser common_chat_peg_builder::mStandardJsonTools(const std::string &              pSectionStart,
                                                              const std::string &              pSectionEnd,
                                                              const ordered_json &             pTools,
                                                              bool                             pParallelToolCalls,
                                                              bool                             pForceToolCalls,
                                                              const std::string &              pNameKey,
                                                              const std::string &              pArgsKey,
                                                              bool                             pArrayWrapped,
                                                              bool                             pFunctionIsKey,
                                                              const std::string &              pCallIdKey,
                                                              const std::string &              pGenCallIdKey,
                                                              const std::vector<std::string> & pParametersOrder,
                                                              bool                             pAcceptOpenaiWrapper) {
  if (!pTools.is_array() || pTools.empty()) {
    return mEps();
  }

  std::string vEffectiveNameKey = pNameKey.empty() ? "name" : pNameKey;
  std::string vEffectiveArgsKey = pArgsKey.empty() ? "arguments" : pArgsKey;

  // Dispatch to the appropriate builder based on the JSON layout mode
  common_peg_parser vToolChoices = mEps();
  if (pFunctionIsKey) {
    vToolChoices = mBuildJsonToolsFunctionIsKey(pTools, pArgsKey, vEffectiveArgsKey, pCallIdKey, pGenCallIdKey);
  } else {
    auto vNameSpec = fParseKeySpec(vEffectiveNameKey);
    auto vArgsSpec = fParseKeySpec(vEffectiveArgsKey);
    if (!vNameSpec.first.empty() || !vArgsSpec.first.empty()) {
      vToolChoices = mBuildJsonToolsNestedKeys(pTools, vEffectiveNameKey, vEffectiveArgsKey, pCallIdKey, pGenCallIdKey);
    } else {
      vToolChoices = mBuildJsonToolsFlatKeys(pTools, vEffectiveNameKey, vEffectiveArgsKey, pCallIdKey, pGenCallIdKey,
                                             pParametersOrder, pAcceptOpenaiWrapper);
    }
  }

  // Build the section with markers
  auto vToolCalls = vToolChoices;
  if (pParallelToolCalls) {
    vToolCalls = vToolCalls + mZeroOrMore(mSpace() + mLiteral(",") + mSpace() + vToolChoices);
  }

  if (pArrayWrapped) {
    vToolCalls = mLiteral("[") + mSpace() + vToolCalls + mSpace() + mLiteral("]");
  }

  auto vSection =
    mTriggerRule("tool-call", mLiteral(pSectionStart) + mSpace() + vToolCalls + mSpace() + mLiteral(pSectionEnd));

  return pForceToolCalls ? vSection : mOptional(vSection);
}

void common_chat_peg_gemma4_mapper::mFromAst(const common_peg_ast_arena &    pArena,
                                             const common_peg_parse_result & pResult) {
  for (const auto & node : pResult.lNodes) {
    mVisit(pArena, node);
  }
}

static std::string fGemma4ToJson(const common_peg_ast_arena & pArena, common_peg_ast_id pId) {
  const auto & node = pArena.get(pId);

  if (node.vText.empty()) {
    return "";
  }

  if (node.vRule == "gemma4-number" || node.vRule == "gemma4-bool" || node.vRule == "gemma4-null") {
    return std::string(node.vText);
  }

  if (node.vRule == "gemma4-string-content") {
    return fEscapeJsonStringInner(std::string(node.vText));
  }

  if (node.vRule == "gemma4-string") {
    std::string vResult = "\"";
    if (!node.lChildren.empty()) {
      vResult += fGemma4ToJson(pArena, node.lChildren[0]);
      if (!node.vIsPartial) {
        vResult += "\"";
      }
    }
    return vResult;
  }

  if (node.vRule == "gemma4-array") {
    std::string vResult = "[";

    bool vAddComma = false;
    for (auto vChildId : node.lChildren) {
      if (vAddComma) {
        vResult += ',';
      }
      vAddComma = true;
      vResult += fGemma4ToJson(pArena, vChildId);
    }

    if (!node.vIsPartial) {
      vResult += ']';
    }
    return vResult;
  }

  if (node.vRule == "gemma4-dict-key-name") {
    return std::string(node.vText);
  }

  if (node.vRule == "gemma4-dict-key") {
    std::string vResult = "\"";
    if (!node.lChildren.empty()) {
      vResult += fEscapeJsonStringInner(fGemma4ToJson(pArena, node.lChildren[0]));
    }
    if (!node.vIsPartial) {
      vResult += "\":";
    }
    return vResult;
  }

  if (node.vRule == "gemma4-dict-kv") {
    std::string vResult;
    for (auto vChildId : node.lChildren) {
      vResult += fGemma4ToJson(pArena, vChildId);
    }
    return vResult;
  }

  if (node.vRule == "gemma4-dict") {
    std::string vResult = "{";

    bool vAddComma = false;
    for (auto vChildId : node.lChildren) {
      if (vAddComma) {
        vResult += ',';
      }
      vAddComma = true;
      vResult += fGemma4ToJson(pArena, vChildId);
    }

    if (!node.vIsPartial) {
      vResult += '}';
    }
    return vResult;
  }

  if (node.vRule == "gemma4-value") {
    if (!node.lChildren.empty()) {
      return fGemma4ToJson(pArena, node.lChildren[0]);
    }
    return "";
  }

  return "";
}

void common_chat_peg_gemma4_mapper::mVisit(const common_peg_ast_arena & pArena, common_peg_ast_id pId) {
  const auto & node = pArena.get(pId);

  if (node.vTag == "reasoning") {
    vResult.vReasoningContent += node.mSanitizedText();
    return;
  }

  if (node.vTag == "content") {
    vResult.vContent += node.mSanitizedText();
    return;
  }

  if (node.vTag == "tool") {
    auto vNameId = pArena.mFindByTag(node, "tool-name");
    auto vArgsId = pArena.mFindByTag(node, "tool-args");

    if (vNameId != cCommonPegInvalidAstId && vArgsId != cCommonPegInvalidAstId) {
      const auto & vNameNode = pArena.get(vNameId);
      const auto & vArgsNode = pArena.get(vArgsId);

      if (!vNameNode.vIsPartial) {
        common_chat_tool_call vCall;
        vCall.name = std::string(vNameNode.vText);
        if (!vArgsNode.lChildren.empty()) {
          vCall.vArguments = fGemma4ToJson(pArena, vArgsNode.lChildren[0]);
        }
        vResult.lToolCalls.push_back(vCall);
      }
    }

    return;
  }

  for (auto vChildId : node.lChildren) {
    mVisit(pArena, vChildId);
  }
}

static void fMinimaxM3Collect(const common_peg_ast_arena &     pArena,
                              const common_peg_ast_node &      node,
                              const std::string &              pTag,
                              std::vector<common_peg_ast_id> & pOut) {
  for (auto vChildId : node.lChildren) {
    const auto & vChild = pArena.get(vChildId);
    if (vChild.vTag == pTag) {
      pOut.push_back(vChildId);
    } else {
      fMinimaxM3Collect(pArena, vChild, pTag, pOut);
    }
  }
}

static common_peg_ast_id fMinimaxM3ValueOf(const common_peg_ast_arena & pArena, const common_peg_ast_node & node) {
  for (auto vChildId : node.lChildren) {
    const auto & vTag = pArena.get(vChildId).vTag;
    if (vTag == common_chat_peg_builder::cToolArgValue || vTag == common_chat_peg_builder::cToolArgStringValue ||
        vTag == common_chat_peg_minimax_m3_mapper::cToolArgObject ||
        vTag == common_chat_peg_minimax_m3_mapper::cToolArgArray) {
      return vChildId;
    }
  }
  return cCommonPegInvalidAstId;
}

static std::string fMinimaxM3ValueToJson(const common_peg_ast_arena & pArena, common_peg_ast_id pId, bool pClosed);

static std::string fMinimaxM3MemberToJson(const common_peg_ast_arena & pArena, const common_peg_ast_node & node) {
  auto vNameId = pArena.mFindByTag(node, common_chat_peg_builder::cToolArgName);
  if (vNameId == cCommonPegInvalidAstId) {
    return "";
  }

  return ordered_json(pArena.get(vNameId).vText).mDump() + ":" +
         fMinimaxM3ValueToJson(pArena, fMinimaxM3ValueOf(pArena, node), !node.vIsPartial);
}

static std::string fMinimaxM3ContainerToJson(const common_peg_ast_arena & pArena,
                                             const common_peg_ast_node &  node,
                                             bool                         is_object,
                                             bool                         pClosed) {
  const std::string cTag =
    is_object ? common_chat_peg_builder::cToolArg : common_chat_peg_minimax_m3_mapper::cToolArgItem;

  std::vector<common_peg_ast_id> lEntries;
  fMinimaxM3Collect(pArena, node, cTag, lEntries);

  std::string vResult = is_object ? "{" : "[";

  bool vAddComma = false;
  for (auto vEntryId : lEntries) {
    const auto & vEntry = pArena.get(vEntryId);

    std::string vText;
    if (is_object) {
      vText = fMinimaxM3MemberToJson(pArena, vEntry);
    } else {
      vText = fMinimaxM3ValueToJson(pArena, fMinimaxM3ValueOf(pArena, vEntry), !vEntry.vIsPartial);
    }

    if (vText.empty()) {
      continue;
    }

    if (vAddComma) {
      vResult += ",";
    }
    vAddComma = true;
    vResult += vText;
  }

  if (pClosed) {
    vResult += is_object ? "}" : "]";
  }
  return vResult;
}

static std::string fMinimaxM3ValueToJson(const common_peg_ast_arena & pArena, common_peg_ast_id pId, bool pClosed) {
  if (pId == cCommonPegInvalidAstId) {
    return "";
  }

  const auto & node = pArena.get(pId);

  if (node.vTag == common_chat_peg_minimax_m3_mapper::cToolArgObject) {
    return fMinimaxM3ContainerToJson(pArena, node, /* is_object = */ true, pClosed);
  }

  if (node.vTag == common_chat_peg_minimax_m3_mapper::cToolArgArray) {
    return fMinimaxM3ContainerToJson(pArena, node, /* is_object = */ false, pClosed);
  }

  if (node.vTag == common_chat_peg_builder::cToolArgStringValue) {
    return "\"" + fEscapeJsonStringInner(std::string(node.vText)) + (pClosed ? "\"" : "");
  }

  // Numbers and booleans are written verbatim by the template
  return std::string(node.vText);
}

void common_chat_peg_minimax_m3_mapper::mFromAst(const common_peg_ast_arena &    pArena,
                                                 const common_peg_parse_result & pResult) {
  for (const auto & node : pResult.lNodes) {
    mVisit(pArena, node);
  }
}

void common_chat_peg_minimax_m3_mapper::mVisit(const common_peg_ast_arena & pArena, common_peg_ast_id pId) {
  const auto & node = pArena.get(pId);

  if (node.vTag == common_chat_peg_builder::cReasoning) {
    vResult.vReasoningContent += node.mSanitizedText();
    return;
  }

  if (node.vTag == common_chat_peg_builder::cContent) {
    vResult.vContent += node.mSanitizedText();
    return;
  }

  if (node.vTag == common_chat_peg_builder::cTool) {
    auto vNameId = pArena.mFindByTag(node, common_chat_peg_builder::cToolName);
    if (vNameId != cCommonPegInvalidAstId) {
      common_chat_tool_call vCall;
      vCall.name       = std::string(pArena.get(vNameId).vText);
      vCall.vArguments = fMinimaxM3ContainerToJson(pArena, node, /* is_object = */ true, !node.vIsPartial);
      vResult.lToolCalls.push_back(vCall);
    }
    return;
  }

  for (auto vChildId : node.lChildren) {
    mVisit(pArena, vChildId);
  }
}
