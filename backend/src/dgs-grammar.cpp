#include "dgs-grammar.h"

#include "dgs-impl.h"
#include "dgs-sampler.h"
#include "dgs-vocab.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <stdexcept>

#define MAX_REPETITION_THRESHOLD 2000
//
// helpers
//

// NOTE: assumes valid utf8 (but checks for overrun)
static std::pair<uint32_t, const char *> fDecodeUtf8(const char * pSrc) {
  static const int aLookup[]  = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
  uint8_t          vFirstByte = static_cast<uint8_t>(*pSrc);
  uint8_t          vHighbits  = vFirstByte >> 4;
  int              vLen       = aLookup[vHighbits];
  uint8_t          vMask      = (1 << (8 - vLen)) - 1;
  uint32_t         value      = vFirstByte & vMask;
  const char *     end        = pSrc + vLen;  // may overrun!
  const char *     vPos       = pSrc + 1;
  for (; vPos < end && *vPos; vPos++) {
    value = (value << 6) + (static_cast<uint8_t>(*vPos) & 0x3F);
  }
  return std::make_pair(value, vPos);
}

static std::pair<std::vector<uint32_t>, dgs_partial_utf8> fDecodeUtf8(const std::string & pSrc,
                                                                      dgs_partial_utf8    pPartialStart) {
  static const int      aLookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4 };
  const char *          vPos      = pSrc.c_str();
  std::vector<uint32_t> lCodePoints;

  // common english strings have the same number of codepoints and bytes. `+ 1` for the terminating 0.
  lCodePoints.reserve(pSrc.size() + 1);
  uint32_t value    = pPartialStart.value;
  int      vNRemain = pPartialStart.vNRemain;

  // continue previous decode, if applicable
  while (*vPos != 0 && vNRemain > 0) {
    uint8_t vNextByte = static_cast<uint8_t>(*vPos);
    if ((vNextByte >> 6) != 2) {
      // invalid sequence, abort
      lCodePoints.push_back(0);
      return std::make_pair(std::move(lCodePoints), dgs_partial_utf8{ 0, -1 });
    }
    value = (value << 6) + (vNextByte & 0x3F);
    ++vPos;
    --vNRemain;
  }

  if (pPartialStart.vNRemain > 0 && vNRemain == 0) {
    lCodePoints.push_back(value);
  }

  // decode any subsequent utf-8 sequences, which may end in an incomplete one
  while (*vPos != 0) {
    uint8_t vFirstByte = static_cast<uint8_t>(*vPos);
    uint8_t vHighbits  = vFirstByte >> 4;
    vNRemain           = aLookup[vHighbits] - 1;

    if (vNRemain < 0) {
      // invalid sequence, abort
      lCodePoints.clear();
      lCodePoints.push_back(0);
      return std::make_pair(std::move(lCodePoints), dgs_partial_utf8{ 0, vNRemain });
    }

    uint8_t vMask = (1 << (7 - vNRemain)) - 1;
    value         = vFirstByte & vMask;

    ++vPos;
    while (*vPos != 0 && vNRemain > 0) {
      value = (value << 6) + (static_cast<uint8_t>(*vPos) & 0x3F);
      ++vPos;
      --vNRemain;
    }
    if (vNRemain == 0) {
      lCodePoints.push_back(value);
    }
  }
  lCodePoints.push_back(0);

  return std::make_pair(std::move(lCodePoints), dgs_partial_utf8{ value, vNRemain });
}

static bool fIsDigitChar(char pC) {
  return '0' <= pC && pC <= '9';
}

static bool fIsWordChar(char pC) {
  return ('a' <= pC && pC <= 'z') || ('A' <= pC && pC <= 'Z') || pC == '-' || fIsDigitChar(pC);
}

static std::pair<uint32_t, const char *> fParseHex(const char * pSrc, int size) {
  const char * vPos  = pSrc;
  const char * end   = pSrc + size;
  uint32_t     value = 0;
  for (; vPos < end && *vPos; vPos++) {
    value <<= 4;
    char vC = *vPos;
    if ('a' <= vC && vC <= 'f') {
      value += vC - 'a' + 10;
    } else if ('A' <= vC && vC <= 'F') {
      value += vC - 'A' + 10;
    } else if ('0' <= vC && vC <= '9') {
      value += vC - '0';
    } else {
      break;
    }
  }
  if (vPos != end) {
    throw std::runtime_error("expecting " + std::to_string(size) + " hex chars at " + pSrc);
  }
  return std::make_pair(value, vPos);
}

static const char * fParseSpace(const char * pSrc, bool pNewlineOk) {
  const char * vPos = pSrc;
  while (*vPos == ' ' || *vPos == '\t' || *vPos == '#' || (pNewlineOk && (*vPos == '\r' || *vPos == '\n'))) {
    if (*vPos == '#') {
      while (*vPos && *vPos != '\r' && *vPos != '\n') {
        vPos++;
      }
    } else {
      vPos++;
    }
  }
  return vPos;
}

static const char * fParseName(const char * pSrc) {
  const char * vPos = pSrc;
  while (fIsWordChar(*vPos)) {
    vPos++;
  }
  if (vPos == pSrc) {
    throw std::runtime_error(std::string("expecting name at ") + pSrc);
  }
  return vPos;
}

static const char * fParseInt(const char * pSrc) {
  const char * vPos = pSrc;
  while (fIsDigitChar(*vPos)) {
    vPos++;
  }
  if (vPos == pSrc) {
    throw std::runtime_error(std::string("expecting integer at ") + pSrc);
  }
  return vPos;
}

static std::pair<uint32_t, const char *> fParseChar(const char * pSrc) {
  if (*pSrc == '\\') {
    switch (pSrc[1]) {
      case 'x':
        return fParseHex(pSrc + 2, 2);
      case 'u':
        return fParseHex(pSrc + 2, 4);
      case 'U':
        return fParseHex(pSrc + 2, 8);
      case 't':
        return std::make_pair('\t', pSrc + 2);
      case 'r':
        return std::make_pair('\r', pSrc + 2);
      case 'n':
        return std::make_pair('\n', pSrc + 2);
      case '\\':
      case '"':
      case '[':
      case ']':
      case '-':
        return std::make_pair(pSrc[1], pSrc + 2);
      default:
        throw std::runtime_error(std::string("unknown escape at ") + pSrc);
    }
  } else if (*pSrc) {
    return fDecodeUtf8(pSrc);
  }
  throw std::runtime_error("unexpected end of input");
}

static std::pair<uint32_t, const char *> fParseToken(const dgs_vocab * pVocab, const char * pSrc) {
  const char * vPos = pSrc;
  if (*vPos != '<') {
    throw std::runtime_error(std::string("expecting '<' at ") + vPos);
  }
  vPos++;

  // Parse <[id]>
  if (*vPos == '[') {
    vPos++;
    const char *  vIntEnd = fParseInt(vPos);
    unsigned long vId     = std::stoul(std::string(vPos, vIntEnd - vPos));
    if (vId > std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error(std::string("parsed token id is too big at ") + vPos);
    }
    uint32_t vTokenId = static_cast<uint32_t>(vId);
    vPos              = vIntEnd;
    if (*vPos != ']') {
      throw std::runtime_error(std::string("expecting ']' at ") + vPos);
    }
    vPos++;
    if (*vPos != '>') {
      throw std::runtime_error(std::string("expecting '>' at ") + vPos);
    }
    vPos++;
    return std::make_pair(vTokenId, vPos);
  }

  if (pVocab == nullptr) {
    throw std::runtime_error(std::string("no vocab to parse token at ") + pSrc);
  }

  // Parse <token> and tokenize to obtain the token id
  while (*vPos != 0 && *vPos != '>') {
    vPos++;
  }
  if (*vPos != '>') {
    throw std::runtime_error(std::string("expecting '>' at ") + vPos);
  }
  vPos++;

  dgs_token aTokens[2];
  int32_t   vNTokens = pVocab->mTokenize(pSrc, static_cast<int32_t>(vPos - pSrc), aTokens, 2, false, true);
  if (vNTokens != 1) {
    // must tokenize to exactly 1 token
    throw std::runtime_error("invalid token '" + std::string(pSrc, vPos - pSrc) + "'");
  }
  return std::make_pair(aTokens[0], vPos);
}

static void fPrintGrammarChar(FILE * pFile, uint32_t pC) {
  if (0x20 <= pC && pC <= 0x7f) {
    fprintf(pFile, "%c", static_cast<char>(pC));
  } else {
    // cop out of encoding UTF-8
    fprintf(pFile, "<U+%04X>", pC);
  }
}

static bool fIsCharElement(dgs_grammar_element pElem) {
  switch (pElem.type) {
    case cDgsGretypeChar:
      return true;
    case cDgsGretypeCharNot:
      return true;
    case cDgsGretypeCharAlt:
      return true;
    case cDgsGretypeCharRngUpper:
      return true;
    case cDgsGretypeCharAny:
      return true;
    default:
      return false;
  }
}

static void fPrintRuleBinary(FILE * pFile, const dgs_grammar_rule & pRule) {
  for (auto vElem : pRule) {
    switch (vElem.type) {
      case cDgsGretypeEnd:
        fprintf(pFile, "END");
        break;
      case cDgsGretypeAlt:
        fprintf(pFile, "ALT");
        break;
      case cDgsGretypeRuleRef:
        fprintf(pFile, "RULE_REF");
        break;
      case cDgsGretypeChar:
        fprintf(pFile, "CHAR");
        break;
      case cDgsGretypeCharNot:
        fprintf(pFile, "CHAR_NOT");
        break;
      case cDgsGretypeCharRngUpper:
        fprintf(pFile, "CHAR_RNG_UPPER");
        break;
      case cDgsGretypeCharAlt:
        fprintf(pFile, "CHAR_ALT");
        break;
      case cDgsGretypeCharAny:
        fprintf(pFile, "CHAR_ANY");
        break;
      case cDgsGretypeToken:
        fprintf(pFile, "TOKEN");
        break;
      case cDgsGretypeTokenNot:
        fprintf(pFile, "TOKEN_NOT");
        break;
    }
    switch (vElem.type) {
      case cDgsGretypeEnd:
      case cDgsGretypeAlt:
      case cDgsGretypeRuleRef:
        fprintf(pFile, "(%u) ", vElem.value);
        break;
      case cDgsGretypeChar:
      case cDgsGretypeCharNot:
      case cDgsGretypeCharRngUpper:
      case cDgsGretypeCharAlt:
      case cDgsGretypeCharAny:
        fprintf(pFile, "(\"");
        fPrintGrammarChar(pFile, vElem.value);
        fprintf(pFile, "\") ");
        break;
      case cDgsGretypeToken:
        fprintf(pFile, "<[");
        fprintf(pFile, "%u", vElem.value);
        fprintf(pFile, "]> ");
        break;
      case cDgsGretypeTokenNot:
        fprintf(pFile, "!");
        fprintf(pFile, "<[");
        fprintf(pFile, "%u", vElem.value);
        fprintf(pFile, "]> ");
        break;
    }
  }
  fprintf(pFile, "\n");
}

static void fPrintRule(FILE *                                  pFile,
                       uint32_t                                pRuleId,
                       const dgs_grammar_rule &                pRule,
                       const std::map<uint32_t, std::string> & pSymbolIdNames) {
  if (pRule.empty() || pRule.back().type != cDgsGretypeEnd) {
    throw std::runtime_error("malformed rule, does not end with DGS_GRETYPE_END: " + std::to_string(pRuleId));
  }
  fprintf(pFile, "%s ::= ", pSymbolIdNames.at(pRuleId).c_str());
  for (size_t vI = 0, end = pRule.size() - 1; vI < end; vI++) {
    dgs_grammar_element vElem = pRule[vI];
    switch (vElem.type) {
      case cDgsGretypeEnd:
        throw std::runtime_error("unexpected end of rule: " + std::to_string(pRuleId) + "," + std::to_string(vI));
      case cDgsGretypeAlt:
        fprintf(pFile, "| ");
        break;
      case cDgsGretypeRuleRef:
        fprintf(pFile, "%s ", pSymbolIdNames.at(vElem.value).c_str());
        break;
      case cDgsGretypeChar:
        fprintf(pFile, "[");
        fPrintGrammarChar(pFile, vElem.value);
        break;
      case cDgsGretypeCharNot:
        fprintf(pFile, "[^");
        fPrintGrammarChar(pFile, vElem.value);
        break;
      case cDgsGretypeCharRngUpper:
        if (vI == 0 || !fIsCharElement(pRule[vI - 1])) {
          throw std::runtime_error("DGS_GRETYPE_CHAR_RNG_UPPER without preceding char: " + std::to_string(pRuleId) +
                                   "," + std::to_string(vI));
        }
        fprintf(pFile, "-");
        fPrintGrammarChar(pFile, vElem.value);
        break;
      case cDgsGretypeCharAlt:
        if (vI == 0 || !fIsCharElement(pRule[vI - 1])) {
          throw std::runtime_error("DGS_GRETYPE_CHAR_ALT without preceding char: " + std::to_string(pRuleId) + "," +
                                   std::to_string(vI));
        }
        fPrintGrammarChar(pFile, vElem.value);
        break;
      case cDgsGretypeCharAny:
        fprintf(pFile, ".");
        break;
      case cDgsGretypeToken:
        fprintf(pFile, "<[");
        fprintf(pFile, "%u", vElem.value);
        fprintf(pFile, "]> ");
        break;
      case cDgsGretypeTokenNot:
        fprintf(pFile, "!");
        fprintf(pFile, "<[");
        fprintf(pFile, "%u", vElem.value);
        fprintf(pFile, "]> ");
        break;
    }
    if (fIsCharElement(vElem)) {
      switch (pRule[vI + 1].type) {
        case cDgsGretypeCharAlt:
        case cDgsGretypeCharRngUpper:
        case cDgsGretypeCharAny:
          break;
        default:
          fprintf(pFile, "] ");
      }
    }
  }
  fprintf(pFile, "\n");
}

//
// Regex utilities
//

size_t dgs_grammar_trigger_pattern::find(const std::string & pInput) const {
  auto vFindStartPos = [](const std::smatch & pMatch) {
    // get from the first matched capturing group to the end of the string
    size_t vStart = std::string::npos;
    for (auto vI = 1u; vI < pMatch.size(); vI++) {
      if (pMatch.length(vI) > 0) {
        vStart = pMatch.position(vI);
        break;
      }
    }
    if (vStart == std::string::npos) {
      vStart = pMatch.position(0);
    }
    return vStart;
  };

  if (!vPattern.empty() && vPattern.front() == '^' && vPattern.back() == '$') {
    // match against the entire input
    std::smatch vMatch;
    if (std::regex_match(pInput, vMatch, vRegex)) {
      return vFindStartPos(vMatch);
    }
  }

  // search anywhere
  std::smatch vMatch;
  if (std::regex_search(pInput, vMatch, vRegex)) {
    return vFindStartPos(vMatch);
  }

  return std::string::npos;
}

//
// implementation
//

uint32_t dgs_grammar_parser::mGetSymbolId(const char * pSrc, size_t pLen) {
  uint32_t vNextId = static_cast<uint32_t>(dSymbolIds.size());
  auto     vResult = dSymbolIds.emplace(std::string(pSrc, pLen), vNextId);
  return vResult.first->second;
}

uint32_t dgs_grammar_parser::mGenerateSymbolId(const std::string & pBaseName) {
  uint32_t vNextId                                      = static_cast<uint32_t>(dSymbolIds.size());
  dSymbolIds[pBaseName + '_' + std::to_string(vNextId)] = vNextId;
  return vNextId;
}

void dgs_grammar_parser::mAddRule(uint32_t pRuleId, const dgs_grammar_rule & pRule) {
  if (lRules.size() <= pRuleId) {
    lRules.resize(pRuleId + 1);
  }
  lRules[pRuleId] = pRule;
}

const char * dgs_grammar_parser::mParseAlternates(const char *        pSrc,
                                                  const std::string & pRuleName,
                                                  uint32_t            pRuleId,
                                                  bool                pIsNested) {
  dgs_grammar_rule lRule;
  const char *     vPos = mParseSequence(pSrc, pRuleName, lRule, pIsNested);
  while (*vPos == '|') {
    lRule.push_back({ cDgsGretypeAlt, 0 });
    vPos = fParseSpace(vPos + 1, true);
    vPos = mParseSequence(vPos, pRuleName, lRule, pIsNested);
  }
  lRule.push_back({ cDgsGretypeEnd, 0 });
  mAddRule(pRuleId, lRule);
  return vPos;
}

const char * dgs_grammar_parser::mParseSequence(const char *        pSrc,
                                                const std::string & pRuleName,
                                                dgs_grammar_rule &  pRule,
                                                bool                pIsNested) {
  size_t       vLastSymStart = pRule.size();
  const char * vPos          = pSrc;
  uint64_t     vNPrevRules   = 1;

  // use UINT64_MAX as the empty value because we aligned to the proper uint64_t type so -1 can't be used
  // (though it's technically the same as -1 now)
  auto vHandleRepetitions = [&](uint64_t pMinTimes, uint64_t pMaxTimes) {
    bool vNoMax = pMaxTimes == UINT64_MAX;
    if (vLastSymStart == pRule.size()) {
      throw std::runtime_error(std::string("expecting preceding item to */+/?/{ at ") + vPos);
    }

    // apply transformation to previous symbol (last_sym_start to end) according to
    // the following rewrite rules:
    // S{m,n} --> S S S (m times) S'(n-m)
    //            S'(x)   ::= S S'(x-1) |
    //            (... n-m definitions of these S' rules ...)
    //            S'(1)   ::= S |
    // S{m,} -->  S S S (m times) S'
    //            S'     ::= S S' |
    // S*     --> S{0,}
    //        --> S'     ::= S S' |
    // S+     --> S{1,}
    //        --> S S'
    //            S'     ::= S S' |
    // S?     --> S{0,1}
    //        --> S'
    //            S'     ::= S |

    dgs_grammar_rule lPrevRule(pRule.begin() + vLastSymStart, pRule.end());
    // Calculate the total number of rules that will be generated by this repetition
    uint64_t         vTotalRules = 1;  // Start with 1 for the original rule
    if (!vNoMax && pMaxTimes > 0) {
      vTotalRules = pMaxTimes;
    } else if (pMinTimes > 0) {
      vTotalRules = pMinTimes;
    }

    if (vNPrevRules * vTotalRules > MAX_REPETITION_THRESHOLD) {
      throw std::runtime_error(
        "number of rules that are going to be repeated multiplied by the new repetition exceeds sane defaults, please "
        "reduce the number of repetitions or rule complexity");
    }

    if (pMinTimes == 0) {
      pRule.resize(vLastSymStart);
    } else {
      // Repeat the previous elements (min_times - 1) times
      for (uint64_t vI = 1; vI < pMinTimes; vI++) {
        pRule.insert(pRule.end(), lPrevRule.begin(), lPrevRule.end());
      }
    }

    uint32_t vLastRecRuleId = 0;
    auto     vNOpt          = vNoMax ? 1 : pMaxTimes - pMinTimes;

    dgs_grammar_rule lRecRule(lPrevRule);
    for (uint64_t vI = 0; vI < vNOpt; vI++) {
      lRecRule.resize(lPrevRule.size());
      uint32_t vRecRuleId = mGenerateSymbolId(pRuleName);
      if (vI > 0 || vNoMax) {
        lRecRule.push_back({ cDgsGretypeRuleRef, vNoMax ? vRecRuleId : vLastRecRuleId });
      }
      lRecRule.push_back({ cDgsGretypeAlt, 0 });
      lRecRule.push_back({ cDgsGretypeEnd, 0 });
      mAddRule(vRecRuleId, lRecRule);
      vLastRecRuleId = vRecRuleId;
    }
    if (vNOpt > 0) {
      pRule.push_back({ cDgsGretypeRuleRef, vLastRecRuleId });
    }
    vNPrevRules *= vTotalRules;
    GGML_ASSERT(vNPrevRules >= 1);
  };

  while (*vPos) {
    if (*vPos == '"') {  // literal string
      vPos++;
      vLastSymStart = pRule.size();
      vNPrevRules   = 1;
      while (*vPos != '"') {
        if (!*vPos) {
          throw std::runtime_error("unexpected end of input");
        }
        auto vCharPair = fParseChar(vPos);
        vPos           = vCharPair.second;
        pRule.push_back({ cDgsGretypeChar, vCharPair.first });
      }
      vPos = fParseSpace(vPos + 1, pIsNested);
    } else if (*vPos == '[') {  // char range(s)
      vPos++;
      enum dgs_gretype vStartType = cDgsGretypeChar;
      if (*vPos == '^') {
        vPos++;
        vStartType = cDgsGretypeCharNot;
      }
      vLastSymStart = pRule.size();
      vNPrevRules   = 1;
      while (*vPos != ']') {
        if (!*vPos) {
          throw std::runtime_error("unexpected end of input");
        }
        auto vCharPair        = fParseChar(vPos);
        vPos                  = vCharPair.second;
        enum dgs_gretype type = vLastSymStart < pRule.size() ? cDgsGretypeCharAlt : vStartType;

        pRule.push_back({ type, vCharPair.first });
        if (vPos[0] == '-' && vPos[1] != ']') {
          if (!vPos[1]) {
            throw std::runtime_error("unexpected end of input");
          }
          auto vEndcharPair = fParseChar(vPos + 1);
          vPos              = vEndcharPair.second;
          pRule.push_back({ cDgsGretypeCharRngUpper, vEndcharPair.first });
        }
      }
      vPos = fParseSpace(vPos + 1, pIsNested);
    } else if (*vPos == '<' || *vPos == '!') {  // token
      auto type = cDgsGretypeToken;
      if (*vPos == '!') {                       // token inverse
        type = cDgsGretypeTokenNot;
        vPos++;
      }
      auto         vTokenPair = fParseToken(vVocab, vPos);
      const char * vTokenEnd  = vTokenPair.second;
      vLastSymStart           = pRule.size();
      vNPrevRules             = 1;
      pRule.push_back({ type, vTokenPair.first });
      vPos = fParseSpace(vTokenEnd, pIsNested);
    } else if (fIsWordChar(*vPos)) {  // rule reference
      const char * vNameEnd   = fParseName(vPos);
      uint32_t     vRefRuleId = mGetSymbolId(vPos, vNameEnd - vPos);
      vPos                    = fParseSpace(vNameEnd, pIsNested);
      vLastSymStart           = pRule.size();
      vNPrevRules             = 1;
      pRule.push_back({ cDgsGretypeRuleRef, vRefRuleId });
    } else if (*vPos == '(') {  // grouping
      // parse nested alternates into synthesized rule
      vPos                   = fParseSpace(vPos + 1, true);
      uint32_t vNRulesBefore = dSymbolIds.size();
      uint32_t vSubRuleId    = mGenerateSymbolId(pRuleName);
      vPos                   = mParseAlternates(vPos, pRuleName, vSubRuleId, true);
      vNPrevRules            = std::max(1u, (uint32_t) dSymbolIds.size() - vNRulesBefore);
      vLastSymStart          = pRule.size();
      // output reference to synthesized rule
      pRule.push_back({ cDgsGretypeRuleRef, vSubRuleId });
      if (*vPos != ')') {
        throw std::runtime_error(std::string("expecting ')' at ") + vPos);
      }
      vPos = fParseSpace(vPos + 1, pIsNested);
    } else if (*vPos == '.') {  // any char
      vLastSymStart = pRule.size();
      vNPrevRules   = 1;
      pRule.push_back({ cDgsGretypeCharAny, 0 });
      vPos = fParseSpace(vPos + 1, pIsNested);
    } else if (*vPos == '*') {
      vPos = fParseSpace(vPos + 1, pIsNested);
      vHandleRepetitions(0, -1);
    } else if (*vPos == '+') {
      vPos = fParseSpace(vPos + 1, pIsNested);
      vHandleRepetitions(1, -1);
    } else if (*vPos == '?') {
      vPos = fParseSpace(vPos + 1, pIsNested);
      vHandleRepetitions(0, 1);
    } else if (*vPos == '{') {
      vPos = fParseSpace(vPos + 1, pIsNested);

      if (!fIsDigitChar(*vPos)) {
        throw std::runtime_error(std::string("expecting an int at ") + vPos);
      }
      const char * vIntEnd   = fParseInt(vPos);
      uint64_t     vMinTimes = std::stoull(std::string(vPos, vIntEnd - vPos));
      vPos                   = fParseSpace(vIntEnd, pIsNested);

      uint64_t vMaxTimes = UINT64_MAX;  // default: no max limit

      if (*vPos == '}') {
        vMaxTimes = vMinTimes;
        vPos      = fParseSpace(vPos + 1, pIsNested);
      } else if (*vPos == ',') {
        vPos = fParseSpace(vPos + 1, pIsNested);

        if (fIsDigitChar(*vPos)) {
          const char * vIntEnd = fParseInt(vPos);
          vMaxTimes            = std::stoull(std::string(vPos, vIntEnd - vPos));
          vPos                 = fParseSpace(vIntEnd, pIsNested);
        }

        if (*vPos != '}') {
          throw std::runtime_error(std::string("expecting '}' at ") + vPos);
        }
        vPos = fParseSpace(vPos + 1, pIsNested);
      } else {
        throw std::runtime_error(std::string("expecting ',' at ") + vPos);
      }
      if (vMinTimes > MAX_REPETITION_THRESHOLD) {
        throw std::runtime_error(
          std::string("number of repetitions exceeds sane defaults, please reduce the number of repetitions"));
      }
      if (vMaxTimes != UINT64_MAX && vMaxTimes > MAX_REPETITION_THRESHOLD) {
        vMaxTimes = UINT64_MAX;
      }
      vHandleRepetitions(vMinTimes, vMaxTimes);
    } else {
      break;
    }
  }
  return vPos;
}

const char * dgs_grammar_parser::mParseRule(const char * pSrc) {
  const char *      vNameEnd = fParseName(pSrc);
  const char *      vPos     = fParseSpace(vNameEnd, false);
  size_t            vNameLen = vNameEnd - pSrc;
  uint32_t          vRuleId  = mGetSymbolId(pSrc, vNameLen);
  const std::string name(pSrc, vNameLen);

  if (!(vPos[0] == ':' && vPos[1] == ':' && vPos[2] == '=')) {
    throw std::runtime_error(std::string("expecting ::= at ") + vPos);
  }
  vPos = fParseSpace(vPos + 3, true);

  vPos = mParseAlternates(vPos, name, vRuleId, false);

  if (*vPos == '\r') {
    vPos += vPos[1] == '\n' ? 2 : 1;
  } else if (*vPos == '\n') {
    vPos++;
  } else if (*vPos) {
    throw std::runtime_error(std::string("expecting newline or end at ") + vPos);
  }
  return fParseSpace(vPos, true);
}

bool dgs_grammar_parser::mParse(const char * pSrc) {
  try {
    const char * vPos = fParseSpace(pSrc, true);
    while (*vPos) {
      vPos = mParseRule(vPos);
    }
    // Validate the state to ensure that all rules are defined
    for (const auto & lRule : lRules) {
      if (lRule.empty()) {
        throw std::runtime_error("Undefined rule");
      }
      for (const auto & vElem : lRule) {
        if (vElem.type == cDgsGretypeRuleRef) {
          // Ensure that the rule at that location exists
          if (vElem.value >= lRules.size() || lRules[vElem.value].empty()) {
            // Get the name of the rule that is missing
            for (const auto & vKv : dSymbolIds) {
              if (vKv.second == vElem.value) {
                throw std::runtime_error("Undefined rule identifier '" + vKv.first + "'");
              }
            }
          }
        }
      }
    }
  } catch (const std::exception & vErr) {
    fprintf(stderr, "%s: error parsing grammar: %s\n\n%s\n", __func__, vErr.what(), pSrc);
    lRules.clear();
    return false;
  }

  return true;
}

void dgs_grammar_parser::mPrint(FILE * pFile) {
  try {
    std::map<uint32_t, std::string> dSymbolIdNames;
    for (const auto & vKv : dSymbolIds) {
      dSymbolIdNames[vKv.second] = vKv.first;
    }
    for (size_t vI = 0, end = lRules.size(); vI < end; vI++) {
      // fprintf(file, "%zu: ", i);
      // print_rule_binary(file, rules[i]);
      fPrintRule(pFile, uint32_t(vI), lRules[vI], dSymbolIdNames);
      // fprintf(file, "\n");
    }
  } catch (const std::exception & vErr) {
    fprintf(stderr, "\n%s: error printing grammar: %s\n", __func__, vErr.what());
  }
}

dgs_grammar_stack dgs_grammar_parser::mCRules() const {
  dgs_grammar_stack lRet;
  lRet.reserve(lRules.size());
  for (const auto & lRule : lRules) {
    lRet.push_back(lRule.data());
  }
  return lRet;
}

// returns true iff pos points to the end of one of the definitions of a rule
static bool fDgsGrammarIsEndOfSequence(const dgs_grammar_element * pPos) {
  switch (pPos->type) {
    case cDgsGretypeEnd:
      return true;  // NOLINT
    case cDgsGretypeAlt:
      return true;  // NOLINT
    default:
      return false;
  }
}

// returns true iff chr satisfies the char range at pos (regular or inverse range)
// asserts that pos is pointing to a char range element
static std::pair<bool, const dgs_grammar_element *> fDgsGrammarMatchChar(const dgs_grammar_element * pPos,
                                                                         const uint32_t              pChr) {
  bool vFound          = false;
  bool vIsPositiveChar = pPos->type == cDgsGretypeChar || pPos->type == cDgsGretypeCharAny;

  GGML_ASSERT(vIsPositiveChar || pPos->type == cDgsGretypeCharNot);  // NOLINT

  do {
    if (pPos[1].type == cDgsGretypeCharRngUpper) {
      // inclusive range, e.g. [a-z]
      vFound = vFound || (pPos->value <= pChr && pChr <= pPos[1].value);
      pPos += 2;
    } else if (pPos->type == cDgsGretypeCharAny) {
      // Any character matches "."
      vFound = true;
      pPos += 1;
    } else {
      // exact char match, e.g. [a] or "a"
      vFound = vFound || pPos->value == pChr;
      pPos += 1;
    }
  } while (pPos->type == cDgsGretypeCharAlt);

  return std::make_pair(vFound == vIsPositiveChar, pPos);
}

// returns true iff some continuation of the given partial UTF-8 sequence could satisfy the char
// range at pos (regular or inverse range)
// asserts that pos is pointing to a char range element
static bool fDgsGrammarMatchPartialChar(const dgs_grammar_element * pPos, const dgs_partial_utf8 pPartialUtf8) {
  bool vIsPositiveChar = pPos->type == cDgsGretypeChar || pPos->type == cDgsGretypeCharAny;
  GGML_ASSERT(vIsPositiveChar || pPos->type == cDgsGretypeCharNot);

  uint32_t vPartialValue = pPartialUtf8.value;
  int      vNRemain      = pPartialUtf8.vNRemain;

  // invalid sequence or 7-bit char split across 2 bytes (overlong)
  if (vNRemain < 0 || (vNRemain == 1 && vPartialValue < 2)) {
    return false;
  }

  // range of possible code points this partial UTF-8 sequence could complete to
  uint32_t vLow  = vPartialValue << (vNRemain * 6);
  uint32_t vHigh = vLow | ((1 << (vNRemain * 6)) - 1);

  if (vLow == 0) {
    if (vNRemain == 2) {
      vLow = 1 << 11;
    } else if (vNRemain == 3) {
      vLow = 1 << 16;
    }
  }

  do {
    if (pPos[1].type == cDgsGretypeCharRngUpper) {
      // inclusive range, e.g. [a-z]
      if (pPos->value <= vHigh && vLow <= pPos[1].value) {
        return vIsPositiveChar;
      }
      pPos += 2;
    } else if (pPos->type == cDgsGretypeCharAny) {
      // Any character matches "."
      return true;
    } else {
      // exact char match, e.g. [a] or "a"
      if (vLow <= pPos->value && pPos->value <= vHigh) {
        return vIsPositiveChar;
      }
      pPos += 1;
    }
  } while (pPos->type == cDgsGretypeCharAlt);

  return !vIsPositiveChar;
}

// returns true iff token matches the rule at pos (regular or inverse)
// asserts that pos is pointing to a token element
static bool fDgsGrammarMatchToken(const dgs_grammar_element * pPos, const dgs_token token) {
  GGML_ASSERT(pPos->type == cDgsGretypeToken || pPos->type == cDgsGretypeTokenNot);
  if (pPos->type == cDgsGretypeToken) {
    return pPos->value == static_cast<uint32_t>(token);
  }
  if (pPos->type == cDgsGretypeTokenNot) {
    return pPos->value != static_cast<uint32_t>(token);
  }
  return false;
}

// transforms a grammar pushdown stack into N possible stacks, all ending
// at a character range (terminal element)
static void fDgsGrammarAdvanceStack(const dgs_grammar_rules & pRules,
                                    const dgs_grammar_stack & pStack,
                                    dgs_grammar_stacks &      pNewStacks) {
  std::vector<dgs_grammar_stack> lTodo;
  lTodo.push_back(pStack);

  auto vStackCmp = [](const dgs_grammar_stack & pA, const dgs_grammar_stack & pB) {
    return std::lexicographical_compare(pA.begin(), pA.end(), pB.begin(), pB.end(),
                                        [](const dgs_grammar_element * pPa, const dgs_grammar_element * pPb) {
                                          return pPa < pPb;  // Compare pointer addresses
                                        });
  };

  std::set<dgs_grammar_stack, decltype(vStackCmp)> sSeen(vStackCmp);

  while (!lTodo.empty()) {
    dgs_grammar_stack lCurrStackCandidate = std::move(lTodo.back());
    lTodo.pop_back();

    auto [curr_stack_it, inserted] = sSeen.insert(std::move(lCurrStackCandidate));
    if (!inserted) {
      continue;
    }
    const dgs_grammar_stack & lCurrStack = *curr_stack_it;

    if (lCurrStack.empty()) {
      if (std::find(pNewStacks.begin(), pNewStacks.end(), lCurrStack) == pNewStacks.end()) {
        pNewStacks.emplace_back(lCurrStack);
      }
      continue;
    }

    const dgs_grammar_element * vPos = lCurrStack.back();

    switch (vPos->type) {
      case cDgsGretypeRuleRef:
        {
          const size_t                cRuleId = static_cast<size_t>(vPos->value);
          const dgs_grammar_element * vSubpos = pRules[cRuleId].data();
          do {
            // init new stack without the top (pos)
            dgs_grammar_stack lNextStack(lCurrStack.begin(), lCurrStack.end() - 1);
            if (!fDgsGrammarIsEndOfSequence(vPos + 1)) {
              // if this rule ref is followed by another element, add that to stack
              lNextStack.push_back(vPos + 1);
            }
            if (!fDgsGrammarIsEndOfSequence(vSubpos)) {
              // if alternate is nonempty, add to stack
              lNextStack.push_back(vSubpos);
            }
            lTodo.push_back(std::move(lNextStack));
            while (!fDgsGrammarIsEndOfSequence(vSubpos)) {
              // scan to end of alternate def
              vSubpos++;
            }
            if (vSubpos->type == cDgsGretypeAlt) {
              // there's another alternate def of this rule to process
              vSubpos++;
            } else {
              break;
            }
          } while (true);
          break;
        }
      case cDgsGretypeChar:
      case cDgsGretypeCharNot:
      case cDgsGretypeCharAny:
      case cDgsGretypeToken:
      case cDgsGretypeTokenNot:
        if (std::find(pNewStacks.begin(), pNewStacks.end(), lCurrStack) == pNewStacks.end()) {
          // only add the stack if it's not a duplicate of one we already have
          pNewStacks.emplace_back(lCurrStack);
        }
        break;
      default:
        // end of alternate (DGS_GRETYPE_END, DGS_GRETYPE_ALT) or middle of char range
        // (DGS_GRETYPE_CHAR_ALT, DGS_GRETYPE_CHAR_RNG_UPPER); stack should never be left on
        // those
        GGML_ABORT("fatal error");
    }
  }
}

static dgs_grammar_candidates fDgsGrammarRejectCandidates(const dgs_grammar_rules &      pRules,
                                                          const dgs_grammar_stacks &     pStacks,
                                                          const dgs_grammar_candidates & pCandidates) {
  GGML_ASSERT(!pStacks.empty());  // REVIEW

  if (pCandidates.empty()) {
    return {};
  }

  auto lRejects = fDgsGrammarRejectCandidatesForStack(pRules, pStacks.front(), pCandidates);

  for (size_t vI = 1, size = pStacks.size(); vI < size; ++vI) {
    lRejects = fDgsGrammarRejectCandidatesForStack(pRules, pStacks[vI], lRejects);
  }

  return lRejects;
}

static bool fDgsGrammarDetectLeftRecursion(const dgs_grammar_rules & pRules,
                                           size_t                    pRuleIndex,
                                           std::vector<bool> *       pRulesVisited,
                                           std::vector<bool> *       pRulesInProgress,
                                           std::vector<bool> *       pRulesMayBeEmpty) {
  if ((*pRulesInProgress)[pRuleIndex]) {
    return true;
  }

  (*pRulesInProgress)[pRuleIndex] = true;

  const dgs_grammar_rule & lRule = pRules[pRuleIndex];

  // First check if the rule might produce the empty string. This could be done combined with the second
  // step but it's more readable as two steps.
  bool vAtRuleStart = true;
  for (size_t vI = 0; vI < lRule.size(); vI++) {
    if (fDgsGrammarIsEndOfSequence(&lRule[vI])) {
      if (vAtRuleStart) {
        (*pRulesMayBeEmpty)[pRuleIndex] = true;
        break;
      }
      vAtRuleStart = true;
    } else {
      vAtRuleStart = false;
    }
  }

  // Second, recurse into leftmost nonterminals (or next-leftmost as long as the previous nonterminal may
  // be empty)
  bool vRecurseIntoNonterminal = true;
  for (size_t vI = 0; vI < lRule.size(); vI++) {
    if (lRule[vI].type == cDgsGretypeRuleRef && vRecurseIntoNonterminal) {
      if (fDgsGrammarDetectLeftRecursion(pRules, (size_t) lRule[vI].value, pRulesVisited, pRulesInProgress,
                                         pRulesMayBeEmpty)) {
        return true;
      }
      if (!((*pRulesMayBeEmpty)[(size_t) lRule[vI].value])) {
        vRecurseIntoNonterminal = false;
      }
    } else if (fDgsGrammarIsEndOfSequence(&lRule[vI])) {
      vRecurseIntoNonterminal = true;
    } else {
      vRecurseIntoNonterminal = false;
    }
  }

  (*pRulesInProgress)[pRuleIndex] = false;
  (*pRulesVisited)[pRuleIndex]    = true;

  return false;
}

const dgs_grammar_rules & fDgsGrammarGetRules(const struct dgs_grammar * pGrammar) {
  return pGrammar->lRules;
}

dgs_grammar_stacks & fDgsGrammarGetStacks(struct dgs_grammar * pGrammar) {
  return pGrammar->lStacks;
}

static void fDgsGrammarAcceptChr(struct dgs_grammar &      pGrammar,
                                 const dgs_grammar_stack & pStack,
                                 uint32_t                  pChr,
                                 dgs_grammar_stacks &      pNewStacks) {
  if (pStack.empty()) {
    return;
  }

  const dgs_grammar_element * vPos = pStack.back();

  // ignore if this turns into a token
  if (vPos->type == cDgsGretypeToken || vPos->type == cDgsGretypeTokenNot) {
    return;
  }

  auto vMatch = fDgsGrammarMatchChar(vPos, pChr);
  if (vMatch.first) {
    dgs_grammar_stack lNewStack(pStack.begin(), pStack.end() - 1);
    if (!fDgsGrammarIsEndOfSequence(vMatch.second)) {
      lNewStack.push_back(vMatch.second);
    }
    fDgsGrammarAdvanceStack(pGrammar.lRules, lNewStack, pNewStacks);
  }
}

void fDgsGrammarAccept(struct dgs_grammar * pGrammar, uint32_t pChr) {
  dgs_grammar_stacks lStacksNew;
  lStacksNew.reserve(pGrammar->lStacks.size());

  for (const auto & lStack : pGrammar->lStacks) {
    fDgsGrammarAcceptChr(*pGrammar, lStack, pChr, lStacksNew);
  }

  pGrammar->lStacks = std::move(lStacksNew);
}

dgs_grammar_candidates fDgsGrammarRejectCandidatesForStack(const dgs_grammar_rules &      pRules,
                                                           const dgs_grammar_stack &      pStack,
                                                           const dgs_grammar_candidates & pCandidates) {
  dgs_grammar_candidates lRejects;
  lRejects.reserve(pCandidates.size());

  if (pStack.empty()) {
    for (const auto & vTok : pCandidates) {
      if (*vTok.vCodePoints != 0 || vTok.vPartialUtf8.vNRemain != 0) {
        lRejects.push_back(vTok);
      }
    }
    return lRejects;
  }

  const dgs_grammar_element * vStackPos = pStack.back();

  // if the top of the stack is a token rule, then we only need to check the token id
  if (vStackPos->type == cDgsGretypeToken || vStackPos->type == cDgsGretypeTokenNot) {
    for (const auto & vTok : pCandidates) {
      if (*vTok.vCodePoints == 0) {
        // reached the end of a token consumed by char rules, reject iff it ended
        // in a partial response
        if (vTok.vPartialUtf8.vNRemain != 0) {
          lRejects.push_back(vTok);
        }
      } else if (!fDgsGrammarMatchToken(vStackPos, vTok.vId)) {
        lRejects.push_back(vTok);
      }
    }
    return lRejects;
  }

  dgs_grammar_candidates lNextCandidates;
  lNextCandidates.reserve(pCandidates.size());

  for (const auto & vTok : pCandidates) {
    if (*vTok.vCodePoints == 0) {
      // reached end of full codepoints in token, reject iff it ended in a partial sequence
      // that cannot satisfy this position in grammar
      if (vTok.vPartialUtf8.vNRemain != 0 && !fDgsGrammarMatchPartialChar(vStackPos, vTok.vPartialUtf8)) {
        lRejects.push_back(vTok);
      }
    } else if (fDgsGrammarMatchChar(vStackPos, *vTok.vCodePoints).first) {
      lNextCandidates.push_back({ vTok.vIndex, vTok.vCodePoints + 1, vTok.vPartialUtf8, vTok.vId });
    } else {
      lRejects.push_back(vTok);
    }
  }

  const auto * vStackPosAfter = fDgsGrammarMatchChar(vStackPos, 0).second;

  // update top of stack to next element, if any
  dgs_grammar_stack lStackAfter(pStack.begin(), pStack.end() - 1);
  if (!fDgsGrammarIsEndOfSequence(vStackPosAfter)) {
    lStackAfter.push_back(vStackPosAfter);
  }
  dgs_grammar_stacks lNextStacks;
  fDgsGrammarAdvanceStack(pRules, lStackAfter, lNextStacks);

  auto lNextRejects = fDgsGrammarRejectCandidates(pRules, lNextStacks, lNextCandidates);
  for (const auto & vTok : lNextRejects) {
    lRejects.push_back({ vTok.vIndex, vTok.vCodePoints - 1, vTok.vPartialUtf8, vTok.vId });
  }

  return lRejects;
}

////////////////////

struct dgs_grammar * fDgsGrammarInitImpl(const struct dgs_vocab *     pVocab,
                                         const dgs_grammar_element ** pRules,
                                         size_t                       pNRules,
                                         size_t                       pStartRuleIndex) {
  const dgs_grammar_element * vPos;

  // copy rule definitions into vectors
  dgs_grammar_rules lVecRules(pNRules);
  for (size_t vI = 0; vI < pNRules; vI++) {
    for (vPos = pRules[vI]; vPos->type != cDgsGretypeEnd; vPos++) {
      lVecRules[vI].push_back(*vPos);
    }
    lVecRules[vI].push_back({ cDgsGretypeEnd, 0 });
  }

  // Validate that all rule references point to valid rules
  for (size_t vI = 0; vI < pNRules; vI++) {
    for (const auto & vElem : lVecRules[vI]) {
      if (vElem.type == cDgsGretypeRuleRef) {
        if (vElem.value >= pNRules || lVecRules[vElem.value].empty()) {
          DGS_LOG_ERROR("invalid grammar: rule %zu references undefined rule %u\n", vI, vElem.value);
          return nullptr;
        }
      }
    }
  }

  // Check for left recursion
  std::vector<bool> lRulesVisited(pNRules);
  std::vector<bool> lRulesInProgress(pNRules);
  std::vector<bool> lRulesMayBeEmpty(pNRules);
  for (size_t vI = 0; vI < pNRules; vI++) {
    if (lRulesVisited[vI]) {
      continue;
    }
    if (fDgsGrammarDetectLeftRecursion(lVecRules, vI, &lRulesVisited, &lRulesInProgress, &lRulesMayBeEmpty)) {
      DGS_LOG_ERROR("unsupported grammar, left recursion detected for nonterminal at index %zu", vI);
      return nullptr;
    }
  }

  // loop over alternates of start rule to build initial stacks
  dgs_grammar_stacks lStacks;
  vPos = lVecRules[pStartRuleIndex].data();
  do {
    dgs_grammar_stack lStack;
    if (!fDgsGrammarIsEndOfSequence(vPos)) {
      // if alternate is nonempty, add to stack
      lStack.push_back(vPos);
    }
    fDgsGrammarAdvanceStack(lVecRules, lStack, lStacks);
    while (!fDgsGrammarIsEndOfSequence(vPos)) {
      // scan to end of alternate def
      vPos++;
    }
    if (vPos->type == cDgsGretypeAlt) {
      // there's another alternate def of this rule to process
      vPos++;
    } else {
      break;
    }
  } while (true);

  // Important: vec_rules has to be moved here, not copied, because stacks contains
  // pointers to elements of vec_rules. If vec_rules were copied into dgs_grammar
  // then the pointers would be invalidated when the local vec_rules goes out of scope.
  return new dgs_grammar{
    pVocab,
    std::move(lVecRules),
    std::move(lStacks),
    /* .partial_utf8 = */ {},
    /* .lazy = */ false,
    /* .awaiting_trigger = */ false,
    /* .trigger_buffer = */ "",
    /* .trigger_buffer_positions = */ {},
    /* .trigger_tokens = */ {},
    /* .trigger_patterns = */ {},
  };
}

struct dgs_grammar * fDgsGrammarInitImpl(const struct dgs_vocab * pVocab,
                                         const char *             pGrammarStr,
                                         const char *             pGrammarRoot,
                                         bool                     pLazy,
                                         const char **            pTriggerPatterns,
                                         size_t                   pNumTriggerPatterns,
                                         const dgs_token *        pTriggerTokens,
                                         size_t                   pNumTriggerTokens) {
  dgs_grammar_parser vParser(pVocab);

  // if there is a grammar, parse it
  // rules will be empty (default) if there are parse errors
  if (!vParser.mParse(pGrammarStr) || vParser.lRules.empty()) {
    DGS_LOG_ERROR("failed to parse grammar\n");
    return nullptr;
  }

  // Ensure that the grammar contains the start symbol
  if (vParser.dSymbolIds.find(pGrammarRoot) == vParser.dSymbolIds.end()) {
    DGS_LOG_ERROR("grammar does not contain a '%s' symbol\n", pGrammarRoot);
    return nullptr;
  }

  std::vector<const dgs_grammar_element *> lGrammarRules(vParser.mCRules());

  const size_t cNRules         = lGrammarRules.size();
  const size_t cStartRuleIndex = vParser.dSymbolIds.at(pGrammarRoot);

  const dgs_grammar_element * vPos;

  // copy rule definitions into vectors
  dgs_grammar_rules lVecRules(cNRules);
  for (size_t vI = 0; vI < cNRules; vI++) {
    for (vPos = lGrammarRules[vI]; vPos->type != cDgsGretypeEnd; vPos++) {
      lVecRules[vI].push_back(*vPos);
    }
    lVecRules[vI].push_back({ cDgsGretypeEnd, 0 });
  }

  // Check for left recursion
  std::vector<bool> lRulesVisited(cNRules);
  std::vector<bool> lRulesInProgress(cNRules);
  std::vector<bool> lRulesMayBeEmpty(cNRules);
  for (size_t vI = 0; vI < cNRules; vI++) {
    if (lRulesVisited[vI]) {
      continue;
    }
    if (fDgsGrammarDetectLeftRecursion(lVecRules, vI, &lRulesVisited, &lRulesInProgress, &lRulesMayBeEmpty)) {
      DGS_LOG_ERROR("unsupported grammar, left recursion detected for nonterminal at index %zu\n", vI);
      return nullptr;
    }
  }

  // loop over alternates of start rule to build initial stacks
  dgs_grammar_stacks lStacks;
  vPos = lVecRules[cStartRuleIndex].data();
  do {
    dgs_grammar_stack lStack;
    if (!fDgsGrammarIsEndOfSequence(vPos)) {
      // if alternate is nonempty, add to stack
      lStack.push_back(vPos);
    }
    fDgsGrammarAdvanceStack(lVecRules, lStack, lStacks);
    while (!fDgsGrammarIsEndOfSequence(vPos)) {
      // scan to end of alternate def
      vPos++;
    }
    if (vPos->type == cDgsGretypeAlt) {
      // there's another alternate def of this rule to process
      vPos++;
    } else {
      break;
    }
  } while (true);

  std::vector<dgs_token>                   lVecTriggerTokens;
  std::vector<dgs_grammar_trigger_pattern> lVecTriggerPatterns;
  for (size_t vI = 0; vI < pNumTriggerTokens; vI++) {
    GGML_ASSERT(pTriggerTokens != nullptr);
    lVecTriggerTokens.push_back(pTriggerTokens[vI]);
  }
  for (size_t vI = 0; vI < pNumTriggerPatterns; vI++) {
    GGML_ASSERT(pTriggerPatterns != nullptr);
    auto & vTrigger   = lVecTriggerPatterns.emplace_back();
    vTrigger.vPattern = pTriggerPatterns[vI];
    vTrigger.vRegex   = std::regex(vTrigger.vPattern);
  }

  // Important: vec_rules has to be moved here, not copied, because stacks contains
  // pointers to elements of vec_rules. If vec_rules were copied into dgs_grammar
  // then the pointers would be invalidated when the local vec_rules goes out of scope.
  return new dgs_grammar{
    pVocab,
    std::move(lVecRules),
    std::move(lStacks),
    /* .partial_utf8 = */ {},
    /* .lazy = */ pLazy,
    /* .awaiting_trigger = */ pLazy,
    /* .trigger_buffer = */ "",
    /* .trigger_buffer_positions = */ {},
    std::move(lVecTriggerTokens),
    std::move(lVecTriggerPatterns),
  };
}

void fDgsGrammarFreeImpl(struct dgs_grammar * pGrammar) {
  if (pGrammar == nullptr) {
    return;
  }

  delete pGrammar;
}

struct dgs_grammar * fDgsGrammarCloneImpl(const struct dgs_grammar & pGrammar) {
  auto * vResult = new dgs_grammar{
    pGrammar.vVocab,         pGrammar.lRules,           pGrammar.lStacks,        pGrammar.vPartialUtf8,
    pGrammar.vLazy,          pGrammar.vAwaitingTrigger, pGrammar.vTriggerBuffer, pGrammar.lTriggerBufferPositions,
    pGrammar.lTriggerTokens, pGrammar.lTriggerPatterns,
  };

  // redirect elements in stacks to point to new rules
  for (size_t vIs = 0; vIs < vResult->lStacks.size(); vIs++) {
    for (size_t vIe = 0; vIe < vResult->lStacks[vIs].size(); vIe++) {
      for (size_t vIr0 = 0; vIr0 < pGrammar.lRules.size(); vIr0++) {
        for (size_t vIr1 = 0; vIr1 < pGrammar.lRules[vIr0].size(); vIr1++) {
          if (pGrammar.lStacks[vIs][vIe] == &pGrammar.lRules[vIr0][vIr1]) {
            vResult->lStacks[vIs][vIe] = &vResult->lRules[vIr0][vIr1];
          }
        }
      }
    }
  }

  return vResult;
}

void fDgsGrammarApplyImpl(const struct dgs_grammar & pGrammar, dgs_token_data_array * pCurP) {
  GGML_ASSERT(pGrammar.vVocab != nullptr);

  if (pGrammar.vAwaitingTrigger) {
    return;
  }

  bool vAllowEog = false;
  for (const auto & lStack : pGrammar.lStacks) {
    if (lStack.empty()) {
      vAllowEog = true;
      break;
    }
  }

  std::vector<std::pair<std::vector<uint32_t>, dgs_partial_utf8>> lCandidatesDecoded;
  lCandidatesDecoded.reserve(pCurP->size);

  dgs_grammar_candidates lCandidatesGrammar;
  lCandidatesGrammar.reserve(pCurP->size);

  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    const dgs_token     cId    = pCurP->data[vI].vId;
    const std::string & vPiece = pGrammar.vVocab->mTokenToPiece(cId);

    if (pGrammar.vVocab->mIsEog(cId)) {
      if (!vAllowEog) {
        pCurP->data[vI].vLogit = -INFINITY;
      }
    } else if (vPiece.empty() || vPiece[0] == 0) {
      pCurP->data[vI].vLogit = -INFINITY;
    } else {
      lCandidatesDecoded.push_back(fDecodeUtf8(vPiece, pGrammar.vPartialUtf8));
      lCandidatesGrammar.push_back(
        { vI, lCandidatesDecoded.back().first.data(), lCandidatesDecoded.back().second, cId });
    }
  }

  const auto lRejects = fDgsGrammarRejectCandidates(pGrammar.lRules, pGrammar.lStacks, lCandidatesGrammar);
  for (const auto & vReject : lRejects) {
    pCurP->data[vReject.vIndex].vLogit = -INFINITY;
  }
}

void fDgsGrammarAcceptImpl(struct dgs_grammar & pGrammar, dgs_token token) {
  GGML_ASSERT(pGrammar.vVocab != nullptr);

  const auto & vPiece = pGrammar.vVocab->mTokenToPiece(token);

  if (pGrammar.vAwaitingTrigger) {
    if (std::find(pGrammar.lTriggerTokens.begin(), pGrammar.lTriggerTokens.end(), token) !=
        pGrammar.lTriggerTokens.end()) {
      pGrammar.vAwaitingTrigger = false;
      pGrammar.vTriggerBuffer.clear();
      fDgsGrammarAcceptToken(pGrammar, token, vPiece);
      DGS_LOG_DEBUG("Grammar triggered on token %u (`%s`)", token, vPiece.c_str());
      return;
    } else {
      auto vPosition = std::make_pair(pGrammar.vTriggerBuffer.size(), pGrammar.vTriggerBuffer.size() + vPiece.size());
      pGrammar.lTriggerBufferPositions.push_back(std::make_pair(token, vPosition));
      pGrammar.vTriggerBuffer += vPiece;

      for (const auto & vTriggerPattern : pGrammar.lTriggerPatterns) {
        auto vStart = vTriggerPattern.find(pGrammar.vTriggerBuffer);
        if (vStart != std::string::npos) {
          pGrammar.vAwaitingTrigger = false;

          // replay tokens that overlap with [start, end)
          for (const auto & [tok, tok_pos] : pGrammar.lTriggerBufferPositions) {
            auto [tok_start, tok_end] = tok_pos;
            if (tok_end <= vStart) {
              continue;
            }

            size_t vPieceStart = (tok_start < vStart) ? vStart : tok_start;  // allow for partial token pieces
            size_t vPieceLen   = tok_end - vPieceStart;
            auto   vTokPiece   = pGrammar.vTriggerBuffer.substr(vPieceStart, vPieceLen);
            fDgsGrammarAcceptToken(pGrammar, tok, vTokPiece);
          }

          auto vConstrainedStr = pGrammar.vTriggerBuffer.substr(vStart);
          pGrammar.vTriggerBuffer.clear();
          pGrammar.lTriggerBufferPositions.clear();
          DGS_LOG_DEBUG("Grammar triggered on regex: '%s'\n", vConstrainedStr.c_str());
          return;
        }
      }
      DGS_LOG_DEBUG("Grammar still awaiting trigger after token %d (`%s`)\n", token, vPiece.c_str());
      return;
    }
  }

  if (pGrammar.vVocab->mIsEog(token)) {
    for (const auto & lStack : pGrammar.lStacks) {
      if (lStack.empty()) {
        return;
      }
    }
    GGML_ABORT("fatal error");
  }

  fDgsGrammarAcceptToken(pGrammar, token, vPiece);
}

void fDgsGrammarAcceptStr(struct dgs_grammar & pGrammar, const std::string & pPiece) {
  // Note terminating 0 in decoded string
  const auto   lDecoded    = fDecodeUtf8(pPiece, pGrammar.vPartialUtf8);
  const auto & lCodePoints = lDecoded.first;

  for (auto lIt = lCodePoints.begin(), end = lCodePoints.end() - 1; lIt != end; ++lIt) {
    fDgsGrammarAccept(&pGrammar, *lIt);
  }

  pGrammar.vPartialUtf8 = lDecoded.second;
  if (pGrammar.lStacks.empty()) {
    throw std::runtime_error("Unexpected empty grammar stack after accepting piece: " + pPiece);
  }
}

void fDgsGrammarAcceptToken(struct dgs_grammar & pGrammar, dgs_token token, const std::string & pPiece) {
  // Note terminating 0 in decoded string
  const auto   lDecoded    = fDecodeUtf8(pPiece, pGrammar.vPartialUtf8);
  const auto & lCodePoints = lDecoded.first;

  dgs_grammar_stacks lStacksNew;
  lStacksNew.reserve(pGrammar.lStacks.size());

  for (const auto & lStack : pGrammar.lStacks) {
    if (lStack.empty()) {
      continue;
    }

    const dgs_grammar_element * vPos = lStack.back();

    if (vPos->type == cDgsGretypeToken || vPos->type == cDgsGretypeTokenNot) {
      if (fDgsGrammarMatchToken(vPos, token)) {
        dgs_grammar_stack lNewStack(lStack.begin(), lStack.end() - 1);
        if (!fDgsGrammarIsEndOfSequence(vPos + 1)) {
          lNewStack.push_back(vPos + 1);
        }
        fDgsGrammarAdvanceStack(pGrammar.lRules, lNewStack, lStacksNew);
      }
    } else {
      dgs_grammar_stacks lCurrentStacks = { lStack };

      for (auto lIt = lCodePoints.begin(), end = lCodePoints.end() - 1; lIt != end; ++lIt) {
        dgs_grammar_stacks lNextStacks;

        for (const auto & lCurStack : lCurrentStacks) {
          fDgsGrammarAcceptChr(pGrammar, lCurStack, *lIt, lNextStacks);
        }

        lCurrentStacks = std::move(lNextStacks);
        if (lCurrentStacks.empty()) {
          break;
        }
      }

      for (auto & lSurvivingStack : lCurrentStacks) {
        if (std::find(lStacksNew.begin(), lStacksNew.end(), lSurvivingStack) == lStacksNew.end()) {
          lStacksNew.emplace_back(lSurvivingStack);
        }
      }
    }
  }

  pGrammar.lStacks      = std::move(lStacksNew);
  pGrammar.vPartialUtf8 = lDecoded.second;

  if (pGrammar.lStacks.empty()) {
    throw std::runtime_error("Unexpected empty grammar stack after accepting piece: " + pPiece + " (" +
                             std::to_string(token) + ")");
  }
}
