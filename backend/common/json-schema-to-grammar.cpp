#include "json-schema-to-grammar.h"

#include "common.h"
#include "trie.h"
#include "unicode.h"

#include <algorithm>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using json = common_json;

static std::string fBuildRepetition(const std::string & pItemRule,
                                    int                 pMinItems,
                                    int                 pMaxItems,
                                    const std::string & pSeparatorRule = "") {
  auto vHasMax = pMaxItems != std::numeric_limits<int>::max();

  if (pMaxItems == 0) {
    return "";
  }
  if (pMinItems == 0 && pMaxItems == 1) {
    return pItemRule + "?";
  }

  if (pSeparatorRule.empty()) {
    if (pMinItems == 1 && !vHasMax) {
      return pItemRule + "+";
    }
    if (pMinItems == 0 && !vHasMax) {
      return pItemRule + "*";
    }
    return pItemRule + "{" + std::to_string(pMinItems) + "," + (vHasMax ? std::to_string(pMaxItems) : "") + "}";
  }

  auto vResult = pItemRule + " " +
                 fBuildRepetition("(" + pSeparatorRule + " " + pItemRule + ")", pMinItems == 0 ? 0 : pMinItems - 1,
                                  vHasMax ? pMaxItems - 1 : pMaxItems);
  if (pMinItems == 0) {
    vResult = "(" + vResult + ")?";
  }
  return vResult;
}

static void fBuildMinMaxInt(int64_t             pMinValue,
                            int64_t             pMaxValue,
                            std::stringstream & pOut,
                            int                 pDecimalsLeft = 16,
                            bool                pTopLevel     = true) {
  auto vHasMin = pMinValue != std::numeric_limits<int64_t>::min();
  auto vHasMax = pMaxValue != std::numeric_limits<int64_t>::max();

  auto vDigitRange = [&](char pFrom, char pTo) {
    pOut << "[";
    if (pFrom == pTo) {
      pOut << pFrom;
    } else {
      pOut << pFrom << "-" << pTo;
    }
    pOut << "]";
  };
  auto vMoreDigits = [&](int pMinDigits, int pMaxDigits) {
    pOut << "[0-9]";
    if (pMinDigits == pMaxDigits && pMinDigits == 1) {
      return;
    }
    pOut << "{";
    pOut << pMinDigits;
    if (pMaxDigits != pMinDigits) {
      pOut << ",";
      if (pMaxDigits != std::numeric_limits<int>::max()) {
        pOut << pMaxDigits;
      }
    }
    pOut << "}";
  };
  std::function<void(const std::string_view &, const std::string_view &)> vUniformRange =
    [&](const std::string_view & pFrom, const std::string_view & pTo) {
      size_t vI = 0;
      while (vI < pFrom.length() && vI < pTo.length() && pFrom[vI] == pTo[vI]) {
        vI++;
      }
      if (vI > 0) {
        pOut << "\"" << pFrom.substr(0, vI) << "\"";
      }
      if (vI < pFrom.length() && vI < pTo.length()) {
        if (vI > 0) {
          pOut << " ";
        }
        auto vSubLen = pFrom.length() - vI - 1;
        if (vSubLen > 0) {
          auto vFromSub  = pFrom.substr(vI + 1);
          auto vToSub    = pTo.substr(vI + 1);
          auto vSubZeros = fStringRepeat("0", vSubLen);
          auto vSubNines = fStringRepeat("9", vSubLen);

          auto vToReached = false;
          pOut << "(";
          if (vFromSub == vSubZeros) {
            vDigitRange(pFrom[vI], pTo[vI] - 1);
            pOut << " ";
            vMoreDigits(vSubLen, vSubLen);
          } else {
            pOut << "[" << pFrom[vI] << "] ";
            pOut << "(";
            vUniformRange(vFromSub, vSubNines);
            pOut << ")";
            if (pFrom[vI] < pTo[vI] - 1) {
              pOut << " | ";
              if (vToSub == vSubNines) {
                vDigitRange(pFrom[vI] + 1, pTo[vI]);
                vToReached = true;
              } else {
                vDigitRange(pFrom[vI] + 1, pTo[vI] - 1);
              }
              pOut << " ";
              vMoreDigits(vSubLen, vSubLen);
            }
          }
          if (!vToReached) {
            pOut << " | ";
            vDigitRange(pTo[vI], pTo[vI]);
            pOut << " ";
            vUniformRange(vSubZeros, vToSub);
          }
          pOut << ")";
        } else {
          pOut << "[" << pFrom[vI] << "-" << pTo[vI] << "]";
        }
      }
    };

  if (vHasMin && vHasMax) {
    if (pMinValue < 0 && pMaxValue < 0) {
      pOut << "\"-\" (";
      fBuildMinMaxInt(-pMaxValue, -pMinValue, pOut, pDecimalsLeft, /* top_level= */ true);
      pOut << ")";
      return;
    }

    if (pMinValue < 0) {
      pOut << "\"-\" (";
      fBuildMinMaxInt(0, -pMinValue, pOut, pDecimalsLeft, /* top_level= */ true);
      pOut << ") | ";
      pMinValue = 0;
    }

    auto vMinS      = std::to_string(pMinValue);
    auto vMaxS      = std::to_string(pMaxValue);
    auto vMinDigits = vMinS.length();
    auto vMaxDigits = vMaxS.length();

    for (auto vDigits = vMinDigits; vDigits < vMaxDigits; vDigits++) {
      vUniformRange(vMinS, fStringRepeat("9", vDigits));
      vMinS = "1" + fStringRepeat("0", vDigits);
      pOut << " | ";
    }
    vUniformRange(vMinS, vMaxS);
    return;
  }

  auto vLessDecimals = std::max(pDecimalsLeft - 1, 1);

  if (vHasMin) {
    if (pMinValue < 0) {
      pOut << "\"-\" (";
      fBuildMinMaxInt(std::numeric_limits<int64_t>::min(), -pMinValue, pOut, pDecimalsLeft, /* top_level= */ false);
      pOut << ") | [0] | [1-9] ";
      vMoreDigits(0, pDecimalsLeft - 1);
    } else if (pMinValue == 0) {
      if (pTopLevel) {
        pOut << "[0] | [1-9] ";
        vMoreDigits(0, vLessDecimals);
      } else {
        vMoreDigits(1, pDecimalsLeft);
      }
    } else if (pMinValue <= 9) {
      char vC          = '0' + pMinValue;
      auto vRangeStart = pTopLevel ? '1' : '0';
      if (vC > vRangeStart) {
        vDigitRange(vRangeStart, vC - 1);
        pOut << " ";
        vMoreDigits(1, vLessDecimals);
        pOut << " | ";
      }
      vDigitRange(vC, '9');
      pOut << " ";
      vMoreDigits(0, vLessDecimals);
    } else {
      auto vMinS = std::to_string(pMinValue);
      auto vLen  = vMinS.length();
      auto vC    = vMinS[0];

      if (vC > '1') {
        vDigitRange(pTopLevel ? '1' : '0', vC - 1);
        pOut << " ";
        vMoreDigits(vLen, vLessDecimals);
        pOut << " | ";
      }
      vDigitRange(vC, vC);
      pOut << " (";
      fBuildMinMaxInt(std::stoll(vMinS.substr(1)), std::numeric_limits<int64_t>::max(), pOut, vLessDecimals,
                      /* top_level= */ false);
      pOut << ")";
      if (vC < '9') {
        pOut << " | ";
        vDigitRange(vC + 1, '9');
        pOut << " ";
        vMoreDigits(vLen - 1, vLessDecimals);
      }
    }
    return;
  }

  if (vHasMax) {
    if (pMaxValue >= 0) {
      if (pTopLevel) {
        pOut << "\"-\" [1-9] ";
        vMoreDigits(0, vLessDecimals);
        pOut << " | ";
      }
      fBuildMinMaxInt(0, pMaxValue, pOut, pDecimalsLeft, /* top_level= */ true);
    } else {
      pOut << "\"-\" (";
      fBuildMinMaxInt(-pMaxValue, std::numeric_limits<int64_t>::max(), pOut, pDecimalsLeft, /* top_level= */ false);
      pOut << ")";
    }
    return;
  }

  throw std::runtime_error("At least one of min_value or max_value must be set");
}

const std::string cSpaceRule = "| \" \" | \"\\n\"{1,2} [ \\t]{0,20}";

struct BuiltinRule {
  std::string              vContent;
  std::vector<std::string> lDeps;
};

static std::unordered_map<std::string, BuiltinRule> dPrimitiveRules = {
  { "boolean",       { "(\"true\" | \"false\")", {} }                                                    },
  { "decimal-part",  { "[0-9]{1,16}", {} }                                                               },
  { "integral-part", { "[0] | [1-9] [0-9]{0,15}", {} }                                                   },
  { "number",
   { "(\"-\"? integral-part) (\".\" decimal-part)? ([eE] [-+]? integral-part)?",
      { "integral-part", "decimal-part" } }                                                              },
  { "integer",       { "(\"-\"? integral-part)", { "integral-part" } }                                   },
  { "value",
   { "object | array | string | number | boolean | null",
      { "object", "array", "string", "number", "boolean", "null" } }                                     },
  { "object",
   { "\"{\" space ( string \":\" space value (\",\" space string \":\" space value)* )? space \"}\"",
      { "string", "value" } }                                                                            },
  { "array",         { "\"[\" space ( value (\",\" space value)* )? space \"]\"", { "value" } }          },
  { "uuid",
   { "\"\\\"\" [0-9a-fA-F]{8} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{4} \"-\" [0-9a-fA-F]{12} "
      "\"\\\"\"",
      {} }                                                                                               },
  { "char",          { "[^\"\\\\\\x7F\\x00-\\x1F] | [\\\\] ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4})", {} } },
  { "string",        { "\"\\\"\" char* \"\\\"\"", { "char" } }                                           },
  { "null",          { "\"null\"", {} }                                                                  },
};

static std::unordered_map<std::string, BuiltinRule> dStringFormatRules = {
  { "date",             { "[0-9]{4} \"-\" ( \"0\" [1-9] | \"1\" [0-2] ) \"-\" ( \"0\" [1-9] | [1-2] [0-9] | \"3\" [0-1] )", {} } },
  { "time",
   { "([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9] \":\" [0-5] [0-9] ( \".\" [0-9]{3} )? ( \"Z\" | ( \"+\" | \"-\" ) "
      "( [01] [0-9] | \"2\" [0-3] ) \":\" [0-5] [0-9] )",
      {} }                                                                                                                       },
  { "date-time",        { "date \"T\" time", { "date", "time" } }                                                                },
  { "date-string",      { "\"\\\"\" date \"\\\"\"", { "date" } }                                                                 },
  { "time-string",      { "\"\\\"\" time \"\\\"\"", { "time" } }                                                                 },
  { "date-time-string", { "\"\\\"\" date-time \"\\\"\"", { "date-time" } }                                                       }
};

static bool fIsReservedName(const std::string & name) {
  static const std::unordered_set<std::string> sReservedNames = [] {
    std::unordered_set<std::string> sS;
    sS.insert("root");
    for (const auto & vP : dPrimitiveRules) {
      sS.insert(vP.first);
    }
    for (const auto & vP : dStringFormatRules) {
      sS.insert(vP.first);
    }
    return sS;
  }();
  return sReservedNames.find(name) != sReservedNames.end();
}

static std::regex                            vInvalidRuleCharsRe("[^a-zA-Z0-9-]+");
static std::regex                            vGrammarLiteralEscapeRe("[\r\n\"\\\\]");
static std::regex                            vGrammarRangeLiteralEscapeRe("[\r\n\"\\]\\-\\\\]");
static std::unordered_map<char, std::string> dGrammarLiteralEscapes = {
  { '\r', "\\r"  },
  { '\n', "\\n"  },
  { '"',  "\\\"" },
  { '-',  "\\-"  },
  { ']',  "\\]"  },
  { '\\', "\\\\" }
};

static const int cMaxPatternDepth = 100;

static std::unordered_set<char> sNonLiteralSet = { '|', '.', '(', ')', '[', ']', '{', '}', '*', '+', '?', '^', '$' };
static std::unordered_set<char> sEscapedInRegexpsButNotInLiterals = { '^', '$', '.', '[', ']', '(', ')',
                                                                      '|', '{', '}', '*', '+', '?' };

static std::string fReplacePattern(const std::string &                                     pInput,
                                   const std::regex &                                      pRegex,
                                   const std::function<std::string(const std::smatch &)> & pReplacement) {
  std::smatch vMatch;
  std::string vResult;

  std::string::const_iterator vSearchStart(pInput.cbegin());
  std::string::const_iterator vSearchEnd(pInput.cend());

  while (std::regex_search(vSearchStart, vSearchEnd, vMatch, pRegex)) {
    vResult.append(vSearchStart, vSearchStart + vMatch.position());
    vResult.append(pReplacement(vMatch));
    vSearchStart = vMatch.suffix().first;
  }

  vResult.append(vSearchStart, vSearchEnd);

  return vResult;
}

static std::string fFormatLiteral(const std::string & pLiteral) {
  std::string vEscaped = fReplacePattern(pLiteral, vGrammarLiteralEscapeRe, [&](const std::smatch & pMatch) {
    char vC = pMatch.str()[0];
    return dGrammarLiteralEscapes.at(vC);
  });
  return "\"" + vEscaped + "\"";
}

std::string gbnf_format_literal(const std::string & pLiteral) {
  return fFormatLiteral(pLiteral);
}

static size_t fGbnfEscapeLength(const std::string & pPattern, size_t pPos) {
  if (pPos + 1 >= pPattern.length() || pPattern[pPos] != '\\') {
    return 0;
  }
  size_t vNHex = 0;
  switch (pPattern[pPos + 1]) {
    case 'x':
      vNHex = 2;
      break;
    case 'u':
      vNHex = 4;
      break;
    case 'U':
      vNHex = 8;
      break;
    // keep in sync with parse_char() in src/dgs-grammar.cpp
    case 't':
    case 'r':
    case 'n':
    case '\\':
    case '"':
    case '[':
    case ']':
    case '-':
      return 2;
    default:
      return 0;
  }
  if (pPos + 2 + vNHex > pPattern.length()) {
    return 0;
  }
  for (size_t vI = pPos + 2; vI < pPos + 2 + vNHex; vI++) {
    char vH = pPattern[vI];
    if (!((vH >= '0' && vH <= '9') || (vH >= 'a' && vH <= 'f') || (vH >= 'A' && vH <= 'F'))) {
      return 0;
    }
  }
  return 2 + vNHex;
}

class common_chat_schema_converter {
private:
  friend std::string                 fBuildGrammar(const std::function<void(const common_grammar_builder &)> & pCb,
                                                   const common_grammar_options &                              pOptions);
  bool                               vDotall;
  std::map<std::string, std::string> dRules;
  std::unordered_set<std::string>    sRefsBeingResolved;
  std::vector<std::string>           lErrors;
  std::vector<std::string>           lWarnings;

  template <typename T> static const T & mAs(const common_chat_schema & node) { return static_cast<const T &>(node); }

  std::string mAddRule(const std::string & name, const std::string & pRule) {
    std::string vEscName = regex_replace(name, vInvalidRuleCharsRe, "-");
    if (dRules.find(vEscName) == dRules.end() || dRules[vEscName] == pRule) {
      dRules[vEscName] = pRule;
      return vEscName;
    }
    int vI = 0;
    while (dRules.find(vEscName + std::to_string(vI)) != dRules.end() &&
           dRules[vEscName + std::to_string(vI)] != pRule) {
      vI++;
    }
    std::string vKey = vEscName + std::to_string(vI);
    dRules[vKey]     = pRule;
    return vKey;
  }

  std::string mGenerateUnionRule(const std::string & name, const std::vector<common_chat_schema_ptr> & pAltSchemas) {
    std::vector<std::string> lRules;
    lRules.reserve(pAltSchemas.size());
    for (size_t vI = 0; vI < pAltSchemas.size(); vI++) {
      lRules.push_back(mVisit(*pAltSchemas[vI], name + (name.empty() ? "alternative-" : "-") + std::to_string(vI)));
    }
    return string_join(lRules, " | ");
  }

  // thrown when the pattern is a valid regex with no grammar equivalent
  struct unsupported_pattern : public std::runtime_error {
    using std::runtime_error::runtime_error;
  };

  // thrown when the pattern is not a valid regex
  struct invalid_pattern : public std::runtime_error {
    using std::runtime_error::runtime_error;
  };

  std::string mVisitPattern(const std::string & pPattern, const std::string & name) {
    auto dRulesSnapshot = dRules;
    try {
      return mPatternToRule(pPattern, name);
    } catch (const unsupported_pattern & vErr) {
      // revert rules
      dRules = std::move(dRulesSnapshot);
      lWarnings.push_back("pattern " + pPattern + " is not supported (" + vErr.what() + "), accepting any string");
      return mAddRule(name, mAddPrimitive("string", dPrimitiveRules.at("string")));
    } catch (const invalid_pattern & vErr) {
      dRules = std::move(dRulesSnapshot);
      lErrors.push_back("Invalid pattern " + pPattern + ": " + vErr.what());
      return "";
    }
  }

  std::string mPatternToRule(const std::string & pPattern, const std::string & name) {
    if (pPattern.length() < 2 || pPattern.front() != '^' || pPattern.back() != '$') {
      throw unsupported_pattern("not anchored with '^' and '$'");
    }
    std::string                                  vSubPattern = pPattern.substr(1, pPattern.length() - 2);
    std::unordered_map<std::string, std::string> dSubRuleIds;

    size_t vI          = 0;
    size_t length      = vSubPattern.length();
    int    vParenDepth = 0;

    using literal_or_rule = std::pair<std::string, bool>;
    auto vToRule          = [&](const literal_or_rule & pLs) {
      auto vIsLiteral = pLs.second;
      auto vS         = pLs.first;
      return vIsLiteral ? "\"" + vS + "\"" : vS;
    };
    std::function<literal_or_rule()> transform = [&]() -> literal_or_rule {
      std::vector<literal_or_rule> lSeq;

      auto vGetDot = [&]() {
        std::string vRule;
        if (vDotall) {
          vRule = "[\\U00000000-\\U0010FFFF]";
        } else {
          vRule = "[^\\x0A\\x0D]";
        }
        return mAddRule("dot", vRule);
      };

      // Joins the sequence, merging consecutive literals together.
      auto vJoinSeq = [&]() {
        std::vector<literal_or_rule> lRet;

        std::string vLiteral;
        auto        vFlushLiteral = [&]() {
          if (vLiteral.empty()) {
            return false;
          }
          lRet.emplace_back(vLiteral, true);
          vLiteral.clear();
          return true;
        };

        for (const auto & vItem : lSeq) {
          auto vIsLiteral = vItem.second;
          if (vIsLiteral) {
            vLiteral += vItem.first;
          } else {
            vFlushLiteral();
            lRet.push_back(vItem);
          }
        }
        vFlushLiteral();

        std::vector<std::string> lResults;
        lResults.reserve(lRet.size());
        for (const auto & vItem : lRet) {
          lResults.push_back(vToRule(vItem));
        }
        return std::make_pair(string_join(lResults, " "), false);
      };

      while (vI < length) {
        char vC = vSubPattern[vI];
        if (vC == '.') {
          lSeq.emplace_back(vGetDot(), false);
          vI++;
        } else if (vC == '(') {
          vI++;
          if (vI < length && vSubPattern[vI] == '?') {
            if (vI + 1 < length && vSubPattern[vI + 1] == ':') {
              vI += 2;  // skip "?:" for non-capturing group, treat as regular group
            } else {
              // lookaround, named group, inline flags, ...
              throw unsupported_pattern("unsupported group syntax");
            }
          }
          vParenDepth++;
          if (vParenDepth > cMaxPatternDepth) {
            throw unsupported_pattern("pattern nesting too deep");
          }
          lSeq.emplace_back("(" + vToRule(transform()) + ")", false);
        } else if (vC == ')') {
          vI++;
          if (vParenDepth == 0) {
            throw invalid_pattern("unbalanced parentheses");
          }
          vParenDepth--;
          return vJoinSeq();
        } else if (vC == '^' || vC == '$') {
          throw unsupported_pattern("anchor inside the pattern");
        } else if (vC == '[') {
          std::string vSquareBrackets = std::string(1, vC);
          vI++;
          while (vI < length && vSubPattern[vI] != ']') {
            if (vSubPattern[vI] == '\\') {
              auto vEscapeLength = fGbnfEscapeLength(vSubPattern, vI);
              if (vEscapeLength == 0) {
                throw unsupported_pattern("unsupported escape in character class: " + vSubPattern.substr(vI, 2));
              }
              vSquareBrackets += vSubPattern.substr(vI, vEscapeLength);
              vI += vEscapeLength;
            } else {
              vSquareBrackets += vSubPattern[vI];
              vI++;
            }
          }
          if (vI >= length) {
            throw invalid_pattern("unterminated character class");
          }
          vSquareBrackets += ']';
          vI++;
          lSeq.emplace_back(vSquareBrackets, false);
        } else if (vC == '|') {
          lSeq.emplace_back("|", false);
          vI++;
        } else if (vC == '*' || vC == '+' || vC == '?') {
          if (lSeq.empty()) {
            throw invalid_pattern("nothing to repeat");
          }
          lSeq.back() = std::make_pair(vToRule(lSeq.back()) + vC, false);
          vI++;
        } else if (vC == '{') {
          std::string vCurlyBrackets = std::string(1, vC);
          vI++;
          while (vI < length && vSubPattern[vI] != '}') {
            vCurlyBrackets += vSubPattern[vI];
            vI++;
          }
          if (vI >= length) {
            throw unsupported_pattern("unterminated curly brackets");
          }
          vCurlyBrackets += '}';
          vI++;
          auto lNums     = fStringSplit(vCurlyBrackets.substr(1, vCurlyBrackets.length() - 2), ",");
          int  vMinTimes = 0;
          int  vMaxTimes = std::numeric_limits<int>::max();
          if (lNums.size() != 1 && lNums.size() != 2) {
            throw unsupported_pattern("wrong number of values in curly brackets");
          }
          try {
            if (lNums.size() == 1) {
              vMinTimes = vMaxTimes = std::stoi(lNums[0]);
            } else {
              if (!lNums[0].empty()) {
                vMinTimes = std::stoi(lNums[0]);
              }
              if (!lNums[1].empty()) {
                vMaxTimes = std::stoi(lNums[1]);
              }
            }
          } catch (const std::logic_error &) {
            throw unsupported_pattern("invalid number in curly brackets");
          }
          if (lSeq.empty()) {
            throw invalid_pattern("nothing to repeat");
          }
          auto & vLast         = lSeq.back();
          auto & vSub          = vLast.first;
          auto   vSubIsLiteral = vLast.second;

          if (!vSubIsLiteral) {
            std::string & vSubId = dSubRuleIds[vSub];
            if (vSubId.empty()) {
              vSubId = mAddRule(name + "-" + std::to_string(dSubRuleIds.size()), vSub);
            }
            vSub = vSubId;
          }
          lSeq.back().first  = fBuildRepetition(vSubIsLiteral ? "\"" + vSub + "\"" : vSub, vMinTimes, vMaxTimes, "");
          lSeq.back().second = false;
        } else {
          std::string vLiteral;
          auto        vIsNonLiteral = [&](char pC) {
            return sNonLiteralSet.find(pC) != sNonLiteralSet.end();
          };
          while (vI < length) {
            if (vSubPattern[vI] == '\\') {
              if (vI == length - 1) {
                throw invalid_pattern("trailing backslash");
              }
              char vNext = vSubPattern[vI + 1];
              if (sEscapedInRegexpsButNotInLiterals.find(vNext) != sEscapedInRegexpsButNotInLiterals.end()) {
                vI++;
                vLiteral += vSubPattern[vI];
                vI++;
              } else {
                auto vEscapeLength = fGbnfEscapeLength(vSubPattern, vI);
                if (vEscapeLength == 0) {
                  throw unsupported_pattern("unsupported escape: " + vSubPattern.substr(vI, 2));
                }
                vLiteral += vSubPattern.substr(vI, vEscapeLength);
                vI += vEscapeLength;
              }
            } else if (vSubPattern[vI] == '"') {
              vLiteral += "\\\"";
              vI++;
            } else if (!vIsNonLiteral(vSubPattern[vI]) &&
                       (vI == length - 1 || vLiteral.empty() || vSubPattern[vI + 1] == '.' ||
                        !vIsNonLiteral(vSubPattern[vI + 1]))) {
              vLiteral += vSubPattern[vI];
              vI++;
            } else {
              break;
            }
          }
          if (vLiteral.empty()) {  // nothing was consumed, ex. a stray ']' or '}'
            throw unsupported_pattern(std::string("unsupported character: ") + vC);
          }
          lSeq.emplace_back(vLiteral, true);
        }
      }
      return vJoinSeq();
    };

    auto vRule = vToRule(transform());
    if (vParenDepth != 0) {
      throw invalid_pattern("unbalanced parentheses");
    }

    return mAddRule(name, "\"\\\"\" (" + vRule + ") \"\\\"\"");
  }

  /*
        Returns a rule that matches a JSON string that is none of the provided strings

        not_strings({"a"})
            -> ["] ( [a] char+ | [^"a] char* )? ["]
        not_strings({"and", "also"})
            -> ["] ( [a] ([l] ([s] ([o] char+ | [^"o] char*) | [^"s] char*) | [n] ([d] char+ | [^"d] char*) | [^"ln] char*) | [^"a] char* )? ["]
    */
  std::string mNotStrings(const std::vector<std::string> & pStrings) {
    common_trie vTrie(pStrings);

    std::string        vCharRule = mAddPrimitive("char", dPrimitiveRules.at("char"));
    std::ostringstream vOut;
    vOut << "[\"] ( ";
    std::function<void(size_t)> vVisit = [&](size_t pIdx) {
      const auto & node = vTrie.lNodes[pIdx];
      std::string  vRejects;
      auto         vFirst = true;
      for (const auto & [cpt, child] : node.dChildren) {
        std::string vC = fCommonUnicodeCptToUtf8(cpt);
        vRejects += vC;
        if (vFirst) {
          vFirst = false;
        } else {
          vOut << " | ";
        }
        vOut << "[" << vC << "]";
        if (!vTrie.lNodes[child].dChildren.empty()) {
          vOut << " (";
          vVisit(child);
          vOut << ")";
        } else {
          vOut << " " << vCharRule << "+";
        }
      }
      if (!node.dChildren.empty()) {
        vOut << " | [^\"" << vRejects << "] " << vCharRule << "*";
      }
    };
    vVisit(0);

    vOut << " )";
    if (vTrie.lNodes[0].vPattern < 0) {
      vOut << "?";
    }
    vOut << " [\"]";
    return vOut.str();
  }

  std::string mResolveRef(const common_chat_schema_ref & pSchema) {
    auto                    vIt          = pSchema.vRef.find('#');
    std::string             vRefFragment = vIt != std::string::npos ? pSchema.vRef.substr(vIt + 1) : pSchema.vRef;
    static const std::regex cNonalphanumericRegex(R"([^a-zA-Z0-9-]+)");
    std::string             vRefName = "ref" + std::regex_replace(vRefFragment, cNonalphanumericRegex, "-");
    if (dRules.find(vRefName) == dRules.end() && sRefsBeingResolved.find(pSchema.vRef) == sRefsBeingResolved.end()) {
      if (!pSchema.vTarget) {
        lErrors.push_back("Unresolved $ref " + pSchema.vRef);
        return "";
      }
      sRefsBeingResolved.insert(pSchema.vRef);
      vRefName = mVisit(*pSchema.vTarget, vRefName);
      sRefsBeingResolved.erase(pSchema.vRef);
    }
    return vRefName;
  }

  std::string mBuildObjectRule(const std::vector<std::pair<std::string, const common_chat_schema *>> & pProperties,
                               const std::unordered_set<std::string> &                                 pRequired,
                               const std::string &                                                     name,
                               const common_chat_schema * pAdditionalProperties) {
    std::vector<std::string>                     lRequiredProps;
    std::vector<std::string>                     lOptionalProps;
    std::unordered_map<std::string, std::string> dPropKvRuleNames;
    std::vector<std::string>                     lPropNames;
    for (const auto & vKv : pProperties) {
      const auto & vPropName   = vKv.first;
      const auto & vPropSchema = vKv.second;

      std::string vPropRuleName = mVisit(*vPropSchema, name + (name.empty() ? "" : "-") + vPropName);
      dPropKvRuleNames[vPropName] =
        mAddRule(name + (name.empty() ? "" : "-") + vPropName + "-kv",
                 fFormatLiteral(json(vPropName).mDump()) + " space \":\" space " + vPropRuleName);
      if (pRequired.find(vPropName) != pRequired.end()) {
        lRequiredProps.push_back(vPropName);
      } else {
        lOptionalProps.push_back(vPropName);
      }
      lPropNames.push_back(vPropName);
    }
    if (pAdditionalProperties) {
      std::string vSubName   = name + (name.empty() ? "" : "-") + "additional";
      std::string vValueRule = pAdditionalProperties->mKind() != common_chat_schema::cKindAny ?
                                 mVisit(*pAdditionalProperties, vSubName + "-value") :
                                 mAddPrimitive("value", dPrimitiveRules.at("value"));

      auto        vKeyRule  = lPropNames.empty() ? mAddPrimitive("string", dPrimitiveRules.at("string")) :
                                                   mAddRule(vSubName + "-k", mNotStrings(lPropNames));
      std::string vKvRule   = mAddRule(vSubName + "-kv", vKeyRule + " \":\" space " + vValueRule);
      dPropKvRuleNames["*"] = vKvRule;
      lOptionalProps.push_back("*");
    }

    if (lRequiredProps.empty() && lOptionalProps.empty()) {
      return "\"{\" space \"}\"";
    }

    std::string vRule = "\"{\" space ";
    for (size_t vI = 0; vI < lRequiredProps.size(); vI++) {
      if (vI > 0) {
        vRule += " \",\" space ";
      }
      vRule += dPropKvRuleNames[lRequiredProps[vI]];
    }

    if (!lOptionalProps.empty()) {
      vRule += " (";
      if (!lRequiredProps.empty()) {
        vRule += " \",\" space ( ";
      }

      std::function<std::string(const std::vector<std::string> &, bool)> lGetRecursiveRefs =
        [&](const std::vector<std::string> & pKs, bool pFirstIsOptional) {
          std::string vRes;
          if (pKs.empty()) {
            return vRes;
          }
          const std::string & vK          = pKs[0];
          std::string         vKvRuleName = dPropKvRuleNames[vK];
          std::string         vCommaRef   = "( \",\" space " + vKvRuleName + " )";
          if (pFirstIsOptional) {
            vRes = vCommaRef + (vK == "*" ? "*" : "?");
          } else {
            vRes = vKvRuleName + (vK == "*" ? " " + vCommaRef + "*" : "");
          }
          if (pKs.size() > 1) {
            vRes += " " + mAddRule(name + (name.empty() ? "" : "-") + vK + "-rest",
                                   lGetRecursiveRefs(std::vector<std::string>(pKs.begin() + 1, pKs.end()), true));
          }
          return vRes;
        };

      for (size_t vI = 0; vI < lOptionalProps.size(); vI++) {
        if (vI > 0) {
          vRule += " | ";
        }
        vRule += lGetRecursiveRefs(std::vector<std::string>(lOptionalProps.begin() + vI, lOptionalProps.end()), false);
      }
      if (!lRequiredProps.empty()) {
        vRule += " )";
      }
      vRule += " )?";
    }

    vRule += " space \"}\"";

    return vRule;
  }

  std::string mAddPrimitive(const std::string & name, const BuiltinRule & pRule) {
    auto n = mAddRule(name, pRule.vContent);
    for (const auto & vDep : pRule.lDeps) {
      BuiltinRule vDepRule;
      auto        vIt = dPrimitiveRules.find(vDep);
      if (vIt == dPrimitiveRules.end()) {
        vIt = dStringFormatRules.find(vDep);
        if (vIt == dStringFormatRules.end()) {
          lErrors.push_back("Rule " + vDep + " not known");
          continue;
        }
      }
      if (dRules.find(vDep) == dRules.end()) {
        mAddPrimitive(vDep, vIt->second);
      }
    }
    return n;
  }

public:
  explicit common_chat_schema_converter(bool pDotall) : vDotall(pDotall) { dRules["space"] = cSpaceRule; }

  std::string mAddSchema(const std::string & name, const common_chat_schema & pSchema) { return mVisit(pSchema, name); }

  static std::string mGenerateConstantRule(const json & value) { return fFormatLiteral(value.mDump()); }

  std::string mVisitPrimitive(const std::string & pRuleName, const std::string & type) {
    return mAddPrimitive(pRuleName == "root" ? "root" : type, dPrimitiveRules.at(type));
  }

  std::string mVisitAllOf(const common_chat_schema_all_of & pSchema,
                          const std::string &               name,
                          const std::string &               pRuleName) {
    std::unordered_set<std::string>                                 sRequired;
    std::vector<std::pair<std::string, const common_chat_schema *>> lProperties;
    std::map<std::string, size_t>                                   dEnumValues;
    std::function<void(const common_chat_schema &, bool)> vAddComponent = [&](const common_chat_schema & pComp,
                                                                              bool                       pIsRequired) {
      if (pComp.mKind() == common_chat_schema::cKindRef) {
        if (const auto * vTarget = mAs<common_chat_schema_ref>(pComp).vTarget) {
          vAddComponent(*vTarget, pIsRequired);
        }
      } else if (pComp.mKind() == common_chat_schema::cKindObject) {
        for (const auto & vProp : mAs<common_chat_schema_object>(pComp).lProperties) {
          lProperties.emplace_back(vProp.name, vProp.vSchema.get());
          if (pIsRequired) {
            sRequired.insert(vProp.name);
          }
        }
      } else if (pComp.mKind() == common_chat_schema::cKindEnum) {
        for (const auto & vV : mAs<common_chat_schema_enum>(pComp).lValues) {
          dEnumValues[mGenerateConstantRule(vV)] += 1;
        }
      }
    };
    for (const auto & vChild : pSchema.lChildren) {
      if (vChild->mKind() == common_chat_schema::cKindAnyOf) {
        for (const auto & vAlt : mAs<common_chat_schema_any_of>(*vChild).lChildren) {
          vAddComponent(*vAlt, false);
        }
      } else {
        vAddComponent(*vChild, true);
      }
    }
    if (!dEnumValues.empty()) {
      std::vector<std::string> lEnumIntersection;
      for (const auto & vP : dEnumValues) {
        if (vP.second == pSchema.lChildren.size()) {
          lEnumIntersection.push_back(vP.first);
        }
      }
      if (!lEnumIntersection.empty()) {
        return mAddRule(pRuleName, "(" + string_join(lEnumIntersection, " | ") + ")");
      }
    }
    return mAddRule(pRuleName, mBuildObjectRule(lProperties, sRequired, name, nullptr));
  }

  std::string mVisit(const common_chat_schema & pSchema, const std::string & name) {
    std::string vRuleName = fIsReservedName(name) ? name + "-" : name.empty() ? "root" : name;
    std::string vSubName  = name + (name.empty() ? "" : "-");

    switch (pSchema.mKind()) {
      case common_chat_schema::cKindRef:
        return mAddRule(vRuleName, mResolveRef(mAs<common_chat_schema_ref>(pSchema)));
      case common_chat_schema::cKindAnyOf:
        return mAddRule(vRuleName, mGenerateUnionRule(name, mAs<common_chat_schema_any_of>(pSchema).lChildren));
      case common_chat_schema::cKindAllOf:
        return mVisitAllOf(mAs<common_chat_schema_all_of>(pSchema), name, vRuleName);
      case common_chat_schema::cKindConst:
        return mAddRule(vRuleName, mGenerateConstantRule(mAs<common_chat_schema_const>(pSchema).value));
      case common_chat_schema::cKindEnum:
        {
          std::vector<std::string> lEnumValues;
          for (const auto & vV : mAs<common_chat_schema_enum>(pSchema).lValues) {
            lEnumValues.push_back(mGenerateConstantRule(vV));
          }
          return mAddRule(vRuleName, "(" + string_join(lEnumValues, " | ") + ")");
        }
      case common_chat_schema::cKindObject:
        {
          const auto & vObj = mAs<common_chat_schema_object>(pSchema);
          if (vObj.lProperties.empty() && vObj.vAdditionalProperties &&
              vObj.vAdditionalProperties->mKind() == common_chat_schema::cKindAny) {
            return mAddRule(vRuleName, mAddPrimitive("object", dPrimitiveRules.at("object")));
          }
          std::vector<std::pair<std::string, const common_chat_schema *>> lProperties;
          std::unordered_set<std::string>                                 sRequired;
          for (const auto & vProp : vObj.lProperties) {
            lProperties.emplace_back(vProp.name, vProp.vSchema.get());
            if (vProp.vRequired) {
              sRequired.insert(vProp.name);
            }
          }
          return mAddRule(vRuleName, mBuildObjectRule(lProperties, sRequired, name, vObj.vAdditionalProperties.get()));
        }
      case common_chat_schema::cKindTuple:
        {
          const auto & lItems = mAs<common_chat_schema_tuple>(pSchema).lItems;
          std::string  vRule  = "\"[\" space ";
          for (size_t vI = 0; vI < lItems.size(); vI++) {
            if (vI > 0) {
              vRule += " \",\" space ";
            }
            vRule += mVisit(*lItems[vI], vSubName + "tuple-" + std::to_string(vI));
          }
          vRule += " space \"]\"";
          return mAddRule(vRuleName, vRule);
        }
      case common_chat_schema::cKindArray:
        {
          const auto & vArr = mAs<common_chat_schema_array>(pSchema);
          if (vArr.vItems->mKind() == common_chat_schema::cKindAny && vArr.vMinItems == 0 && vArr.vMaxItems < 0) {
            return mVisitPrimitive(vRuleName, "array");
          }
          std::string vItemRuleName = mVisit(*vArr.vItems, vSubName + "item");
          int         vMaxItems     = vArr.vMaxItems < 0 ? std::numeric_limits<int>::max() : vArr.vMaxItems;
          return mAddRule(vRuleName, "\"[\" space " +
                                       fBuildRepetition(vItemRuleName, vArr.vMinItems, vMaxItems, "\",\" space") +
                                       " space \"]\"");
        }
      case common_chat_schema::cKindString:
        {
          const auto & str = mAs<common_chat_schema_string>(pSchema);
          if (!str.vPattern.empty()) {
            return mVisitPattern(str.vPattern, vRuleName);
          }
          if (str.format == common_chat_schema::cFormatUuid) {
            return mVisitPrimitive(vRuleName, "uuid");
          }
          if (str.format != common_chat_schema::cFormatNone) {
            std::string vPrimName = std::string(str.format == common_chat_schema::cFormatDate ? "date" :
                                                str.format == common_chat_schema::cFormatTime ? "time" :
                                                                                                "date-time") +
                                    "-string";
            return mAddRule(vRuleName, mAddPrimitive(vPrimName, dStringFormatRules.at(vPrimName)));
          }
          if (str.vMinLength > 0 || str.vMaxLength >= 0) {
            std::string vCharRule = mAddPrimitive("char", dPrimitiveRules.at("char"));
            int         vMaxLen   = str.vMaxLength < 0 ? std::numeric_limits<int>::max() : str.vMaxLength;
            return mAddRule(vRuleName,
                            "\"\\\"\" " + fBuildRepetition(vCharRule, str.vMinLength, vMaxLen) + " \"\\\"\"");
          }
          return mVisitPrimitive(vRuleName, "string");
        }
      case common_chat_schema::cKindInteger:
        {
          const auto & vI = mAs<common_chat_schema_integer>(pSchema);
          if (vI.vMinimum == std::numeric_limits<int64_t>::min() &&
              vI.vMaximum == std::numeric_limits<int64_t>::max()) {
            return mVisitPrimitive(vRuleName, "integer");
          }
          std::stringstream vOut;
          vOut << "(";
          fBuildMinMaxInt(vI.vMinimum, vI.vMaximum, vOut);
          vOut << ")";
          return mAddRule(vRuleName, vOut.str());
        }
      case common_chat_schema::cKindNumber:
        return mVisitPrimitive(vRuleName, "number");
      case common_chat_schema::cKindBoolean:
        return mVisitPrimitive(vRuleName, "boolean");
      case common_chat_schema::cKindNull:
        return mVisitPrimitive(vRuleName, "null");
      case common_chat_schema::cKindAny:
        return mAddRule(vRuleName, mAddPrimitive("value", dPrimitiveRules.at("value")));
    }
    return "";
  }

  void mCheckErrors() {
    if (!lErrors.empty()) {
      throw std::invalid_argument("JSON schema conversion failed:\n" + string_join(lErrors, "\n"));
    }
    if (!lWarnings.empty()) {
      fprintf(stderr, "WARNING: JSON schema conversion was incomplete: %s\n", string_join(lWarnings, "; ").c_str());
    }
  }

  std::string mFormatGrammar() {
    std::stringstream vSs;
    for (const auto & vKv : dRules) {
      vSs << vKv.first << " ::= " << vKv.second << '\n';
    }
    return vSs.str();
  }
};

std::string fJsonSchemaToGrammar(const common_json & pSchema, bool pForceGbnf) {
#ifdef DGS_USE_LLGUIDANCE
  if (!force_gbnf) {
    return "%llguidance {}\nstart: %json " + schema.dump();
  }
#else
  (void) pForceGbnf;
#endif  // DGS_USE_LLGUIDANCE
  try {
    return fJsonSchemaToGrammar(fCommonChatSchemaFromJson(pSchema));
  } catch (const std::runtime_error & vE) {
    throw std::invalid_argument(std::string("JSON schema conversion failed:\n") + vE.what());
  }
}

std::string fJsonSchemaToGrammar(const common_chat_schema_document & pSchema) {
  common_chat_schema_converter vConverter(false);
  vConverter.mVisit(*pSchema.vRoot, "");
  vConverter.mCheckErrors();
  return vConverter.mFormatGrammar();
}

std::string fBuildGrammar(const std::function<void(const common_grammar_builder &)> & pCb,
                          const common_grammar_options &                              pOptions) {
  common_chat_schema_converter vConverter(pOptions.vDotall);
  common_grammar_builder       vBuilder{
    /* .add_rule = */ [&](const std::string & name, const std::string & pRule) {
      return vConverter.mAddRule(name, pRule);
    },
    /* .add_schema = */
    [&](const std::string & name, const common_chat_schema & pSchema) {
      return vConverter.mAddSchema(name == "root" ? "" : name, pSchema);
    },
  };
  pCb(vBuilder);
  vConverter.mCheckErrors();
  return vConverter.mFormatGrammar();
}
