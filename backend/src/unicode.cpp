#include "unicode.h"

#include "unicode-data.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

size_t fUnicodeLenUtf8(char pSrc) {
  const size_t aLookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
  uint8_t      vHighbits = static_cast<uint8_t>(pSrc) >> 4;
  return aLookup[vHighbits];
}

static std::string fUnicodeCptsToUtf8(const std::vector<uint32_t> & pCps) {
  std::string vResult;
  for (size_t vI = 0; vI < pCps.size(); ++vI) {
    vResult.append(fUnicodeCptToUtf8(pCps[vI]));
  }
  return vResult;
}

uint32_t fUnicodeCptFromUtf8(const std::string & pUtf8, size_t & pOffset) {
  assert(offset < utf8.size());
  if (!(pUtf8[pOffset + 0] & 0x80)) {
    auto vResult = pUtf8[pOffset + 0];
    pOffset += 1;
    return vResult;
  }
  if (!(pUtf8[pOffset + 0] & 0x40)) {
    throw std::invalid_argument("invalid character");
  }
  if (!(pUtf8[pOffset + 0] & 0x20)) {
    if (pOffset + 1 >= pUtf8.size() || !((pUtf8[pOffset + 1] & 0xc0) == 0x80)) {
      throw std::invalid_argument("invalid character");
    }
    auto vResult = ((pUtf8[pOffset + 0] & 0x1f) << 6) | (pUtf8[pOffset + 1] & 0x3f);
    pOffset += 2;
    return vResult;
  }
  if (!(pUtf8[pOffset + 0] & 0x10)) {
    if (pOffset + 2 >= pUtf8.size() || !((pUtf8[pOffset + 1] & 0xc0) == 0x80) ||
        !((pUtf8[pOffset + 2] & 0xc0) == 0x80)) {
      throw std::invalid_argument("invalid character");
    }
    auto vResult =
      ((pUtf8[pOffset + 0] & 0x0f) << 12) | ((pUtf8[pOffset + 1] & 0x3f) << 6) | (pUtf8[pOffset + 2] & 0x3f);
    pOffset += 3;
    return vResult;
  }
  if (!(pUtf8[pOffset + 0] & 0x08)) {
    if (pOffset + 3 >= pUtf8.size() || !((pUtf8[pOffset + 1] & 0xc0) == 0x80) ||
        !((pUtf8[pOffset + 2] & 0xc0) == 0x80) || !((pUtf8[pOffset + 3] & 0xc0) == 0x80)) {
      throw std::invalid_argument("invalid character");
    }
    auto vResult = ((pUtf8[pOffset + 0] & 0x07) << 18) | ((pUtf8[pOffset + 1] & 0x3f) << 12) |
                   ((pUtf8[pOffset + 2] & 0x3f) << 6) | (pUtf8[pOffset + 3] & 0x3f);
    pOffset += 4;
    return vResult;
  }
  throw std::invalid_argument("failed to convert utf8 to codepoint");
}

//static std::vector<uint16_t> unicode_cpt_to_utf16(uint32_t cpt) {
//    std::vector<uint16_t> result;
//    if (/* 0x0000 <= cpt && */ cpt <= 0xffff) {
//        result.emplace_back(cpt);
//        return result;
//    }
//    if (0x10000 <= cpt && cpt <= 0x10ffff) {
//        result.emplace_back(0xd800 | ((cpt - 0x10000) >> 10));
//        result.emplace_back(0xdc00 | ((cpt - 0x10000) & 0x03ff));
//        return result;
//    }
//    throw std::invalid_argument("failed to convert codepoint to utf16");
//}

//static std::vector<uint16_t> unicode_cpts_to_utf16(const std::vector<uint32_t> & cps) {
//    std::vector<uint16_t> result;
//    for (size_t i = 0; i < cps.size(); ++i) {
//        auto temp = unicode_cpt_to_utf16(cps[i]);
//        result.insert(result.end(), temp.begin(), temp.end());
//    }
//    return result;
//}

//static uint32_t unicode_cpt_from_utf16(const std::vector<uint16_t> & utf16, size_t & offset) {
//    assert(offset < utf16.size());
//    if (((utf16[0] >> 10) << 10) != 0xd800) {
//        auto result = utf16[offset + 0];
//        offset += 1;
//        return result;
//    }
//
//    if (offset + 1 >= utf16.size() || !((utf16[1] & 0xdc00) == 0xdc00)) {
//        throw std::invalid_argument("invalid character");
//    }
//
//    auto result = 0x10000 + (((utf16[0] & 0x03ff) << 10) | (utf16[1] & 0x03ff));
//    offset += 2;
//    return result;
//}

//static std::vector<uint32_t> unicode_cpts_from_utf16(const std::vector<uint16_t> & utf16) {
//    std::vector<uint32_t> result;
//    size_t offset = 0;
//    while (offset < utf16.size()) {
//        result.push_back(unicode_cpt_from_utf16(utf16, offset));
//    }
//    return result;
//}

static std::vector<unicode_cpt_flags> fUnicodeCptFlagsArray() {
  std::vector<unicode_cpt_flags> lCptFlags(cMaxCodepoints, unicode_cpt_flags::cUndefined);

  assert(unicode_ranges_flags.begin()[0].first == 0);
  assert(unicode_ranges_flags.begin()[unicode_ranges_flags.size() - 1].first == MAX_CODEPOINTS);
  for (size_t vI = 1; vI < cUnicodeRangesFlags.size(); ++vI) {
    const auto cRangeIni = cUnicodeRangesFlags.begin()[vI - 1];  // codepoint_ini, flags
    const auto cRangeEnd = cUnicodeRangesFlags.begin()[vI];      // codepoint_end, flags
    for (uint32_t vCpt = cRangeIni.first; vCpt < cRangeEnd.first; ++vCpt) {
      lCptFlags[vCpt] = cRangeIni.second;
    }
  }

  for (auto vCpt : sUnicodeSetWhitespace) {
    lCptFlags[vCpt].vIsWhitespace = true;
  }

  for (auto vP : cUnicodeMapLowercase) {
    lCptFlags[vP.second].vIsLowercase = true;
  }

  for (auto vP : cUnicodeMapUppercase) {
    lCptFlags[vP.second].vIsUppercase = true;
  }

  for (auto & vRange : cUnicodeRangesNfd) {  // start, last, nfd
    lCptFlags[vRange.vNfd].vIsNfd = true;
  }

  return lCptFlags;
}

static std::unordered_map<uint8_t, std::string> fUnicodeByteToUtf8Map() {
  std::unordered_map<uint8_t, std::string> dMap;
  for (int vCh = 0x21; vCh <= 0x7E; ++vCh) {  // u'!' to u'~'
    assert(0 <= ch && ch < 256);
    dMap[vCh] = fUnicodeCptToUtf8(vCh);
  }
  for (int vCh = 0xA1; vCh <= 0xAC; ++vCh) {  // u'¡' to u'¬'
    assert(0 <= ch && ch < 256);
    dMap[vCh] = fUnicodeCptToUtf8(vCh);
  }
  for (int vCh = 0xAE; vCh <= 0xFF; ++vCh) {  // u'®' to u'ÿ'
    assert(0 <= ch && ch < 256);
    dMap[vCh] = fUnicodeCptToUtf8(vCh);
  }
  auto n = 0;
  for (int vCh = 0; vCh < 256; ++vCh) {
    if (dMap.find(vCh) == dMap.end()) {
      dMap[vCh] = fUnicodeCptToUtf8(256 + n);
      ++n;
    }
  }
  return dMap;
}

static std::unordered_map<std::string, uint8_t> fUnicodeUtf8ToByteMap() {
  std::unordered_map<std::string, uint8_t> dMap;
  for (int vCh = 0x21; vCh <= 0x7E; ++vCh) {  // u'!' to u'~'
    assert(0 <= ch && ch < 256);
    dMap[fUnicodeCptToUtf8(vCh)] = vCh;
  }
  for (int vCh = 0xA1; vCh <= 0xAC; ++vCh) {  // u'¡' to u'¬'
    assert(0 <= ch && ch < 256);
    dMap[fUnicodeCptToUtf8(vCh)] = vCh;
  }
  for (int vCh = 0xAE; vCh <= 0xFF; ++vCh) {  // u'®' to u'ÿ'
    assert(0 <= ch && ch < 256);
    dMap[fUnicodeCptToUtf8(vCh)] = vCh;
  }
  auto n = 0;
  for (int vCh = 0; vCh < 256; ++vCh) {
    if (dMap.find(fUnicodeCptToUtf8(vCh)) == dMap.end()) {
      dMap[fUnicodeCptToUtf8(256 + n)] = vCh;
      ++n;
    }
  }
  return dMap;
}

static std::vector<std::string> fUnicodeByteEncodingProcess(const std::vector<std::string> & pBpeWords) {
  std::vector<std::string> lBpeEncodedWords;
  for (const auto & vWord : pBpeWords) {
    std::string vTextUtf;
    auto        lUtfWord = fUnicodeCptsFromUtf8(vWord);
    for (size_t vI = 0; vI < lUtfWord.size(); ++vI) {
      vTextUtf += fUnicodeCptToUtf8(lUtfWord[vI]);
    }

    std::string vEncodedToken;
    for (char & vC : vTextUtf) {
      vEncodedToken += fUnicodeByteToUtf8(vC);
    }
    lBpeEncodedWords.emplace_back(vEncodedToken);
  }
  return lBpeEncodedWords;
}

// GPT2 system regex:  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
static std::vector<size_t> fUnicodeRegexSplitCustomGpt2(const std::string &         pText,
                                                        const std::vector<size_t> & pOffsets) {
  std::vector<size_t> lBpeOffsets;       // store the offset of each word
  lBpeOffsets.reserve(pOffsets.size());  // Reserve memory for the approximate size

  const auto lCpts = fUnicodeCptsFromUtf8(pText);

  size_t vStart = 0;
  for (auto vOffset : pOffsets) {
    const size_t cOffsetIni = vStart;
    const size_t cOffsetEnd = vStart + vOffset;
    assert(offset_end <= cpts.size());
    vStart = cOffsetEnd;

    static const uint32_t cOutOfRange = 0xFFFFFFFF;
    auto                  vGetCpt     = [&](const size_t pPos) -> uint32_t {
      return (cOffsetIni <= pPos && pPos < cOffsetEnd) ? lCpts[pPos] : cOutOfRange;
    };

    auto vGetFlags = [&](const size_t pPos) -> unicode_cpt_flags {
      return (cOffsetIni <= pPos && pPos < cOffsetEnd) ? fUnicodeCptFlagsFromCpt(lCpts[pPos]) : unicode_cpt_flags{};
    };

    size_t vPrevEnd  = cOffsetIni;
    auto   vAddToken = [&](const size_t end) -> size_t {
      assert(_prev_end <= end && end <= offset_end);
      size_t vLen = end - vPrevEnd;
      if (vLen > 0) {
        lBpeOffsets.push_back(vLen);
      }
      vPrevEnd = end;
      //if (len > 0) {
      //    std::string s = "";
      //    for(size_t p = end-len; p < end; p++)
      //        s += unicode_cpt_to_utf8(cpts[p]);
      //    printf(">>> '%s'\n", s.c_str());
      //}
      return vLen;
    };

    for (size_t vPos = cOffsetIni; vPos < cOffsetEnd; /*pos++*/) {
      const uint32_t cCpt   = vGetCpt(vPos);
      const auto     cFlags = vGetFlags(vPos);

      // regex: 's|'t|'re|'ve|'m|'ll|'d
      if (cCpt == '\'' && vPos + 1 < cOffsetEnd) {
        uint32_t vCptNext = vGetCpt(vPos + 1);
        if (vCptNext == 's' || vCptNext == 't' || vCptNext == 'm' || vCptNext == 'd') {
          vPos += vAddToken(vPos + 2);
          continue;
        }
        if (vPos + 2 < cOffsetEnd) {
          uint32_t vCptNextNext = vGetCpt(vPos + 2);
          if ((vCptNext == 'r' && vCptNextNext == 'e') || (vCptNext == 'v' && vCptNextNext == 'e') ||
              (vCptNext == 'l' && vCptNextNext == 'l')) {
            vPos += vAddToken(vPos + 3);
            continue;
          }
        }
      }

      auto vFlags2 = (cCpt == ' ' ? vGetFlags(vPos + 1) : cFlags);
      // regex: <space>?\p{L}+
      if (vFlags2.vIsLetter) {
        vPos += (cCpt == ' ');
        while (vFlags2.vIsLetter) {
          vFlags2 = vGetFlags(++vPos);
        }
        vAddToken(vPos);
        continue;
      }
      // regex: <space>?\p{N}+
      if (vFlags2.is_number) {
        vPos += (cCpt == ' ');
        while (vFlags2.is_number) {
          vFlags2 = vGetFlags(++vPos);
        }
        vAddToken(vPos);
        continue;
      }
      // regex: <space>?[^\s\p{L}\p{N}]+
      if (!(vFlags2.vIsWhitespace | vFlags2.vIsLetter | vFlags2.is_number) && vFlags2.mAsUint()) {
        vPos += (cCpt == ' ');
        while (!(vFlags2.vIsWhitespace | vFlags2.vIsLetter | vFlags2.is_number) && vFlags2.mAsUint()) {
          vFlags2 = vGetFlags(++vPos);
        }
        vAddToken(vPos);
        continue;
      }

      size_t vNumWhitespaces = 0;
      while (vGetFlags(vPos + vNumWhitespaces).vIsWhitespace) {
        vNumWhitespaces++;
      }

      // regex: \s+(?!\S)
      if (vNumWhitespaces > 1 && vGetCpt(vPos + vNumWhitespaces) != cOutOfRange) {
        vPos += vNumWhitespaces - 1;
        vAddToken(vPos);
        continue;
      }

      // regex: \s+
      if (vNumWhitespaces > 0) {
        vPos += vNumWhitespaces;
        vAddToken(vPos);
        continue;
      }

      // no matches
      vAddToken(++vPos);
    }
  }

  return lBpeOffsets;
}

// DGS3 system regex: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
static std::vector<size_t> fUnicodeRegexSplitCustomDgs3(const std::string &         pText,
                                                        const std::vector<size_t> & pOffsets) {
  std::vector<size_t> lBpeOffsets;       // store the offset of each word
  lBpeOffsets.reserve(pOffsets.size());  // Reserve memory for the approximate size

  const auto lCpts = fUnicodeCptsFromUtf8(pText);

  size_t vStart = 0;
  for (auto vOffset : pOffsets) {
    const size_t cOffsetIni = vStart;
    const size_t cOffsetEnd = vStart + vOffset;
    assert(offset_end <= cpts.size());
    vStart = cOffsetEnd;

    static const uint32_t cOutOfRange = 0xFFFFFFFF;
    auto                  vGetCpt     = [&](const size_t pPos) -> uint32_t {
      return (cOffsetIni <= pPos && pPos < cOffsetEnd) ? lCpts[pPos] : cOutOfRange;
    };

    auto vGetFlags = [&](const size_t pPos) -> unicode_cpt_flags {
      return (cOffsetIni <= pPos && pPos < cOffsetEnd) ? fUnicodeCptFlagsFromCpt(lCpts[pPos]) : unicode_cpt_flags{};
    };

    size_t vPrevEnd  = cOffsetIni;
    auto   vAddToken = [&](const size_t end) -> size_t {
      assert(_prev_end <= end && end <= offset_end);
      size_t vLen = end - vPrevEnd;
      if (vLen > 0) {
        lBpeOffsets.push_back(vLen);
      }
      vPrevEnd = end;
      //if (len > 0) {
      //    std::string s = "";
      //    for(size_t p = end-len; p < end; p++)
      //        s += unicode_cpt_to_utf8(cpts[p]);
      //    printf(">>> '%s'\n", s.c_str());
      //}
      return vLen;
    };

    for (size_t vPos = cOffsetIni; vPos < cOffsetEnd; /*pos++*/) {
      const uint32_t cCpt   = vGetCpt(vPos);
      const auto     cFlags = vGetFlags(vPos);

      // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d) // case insensitive
      if (cCpt == '\'' && vPos + 1 < cOffsetEnd) {
        uint32_t vCptNext = fUnicodeTolower(vGetCpt(vPos + 1));
        if (vCptNext == 's' || vCptNext == 't' || vCptNext == 'm' || vCptNext == 'd') {
          vPos += vAddToken(vPos + 2);
          continue;
        }
        if (vPos + 2 < cOffsetEnd) {
          uint32_t vCptNextNext = fUnicodeTolower(vGetCpt(vPos + 2));
          if ((vCptNext == 'r' && vCptNextNext == 'e') || (vCptNext == 'v' && vCptNextNext == 'e') ||
              (vCptNext == 'l' && vCptNextNext == 'l')) {
            vPos += vAddToken(vPos + 3);
            continue;
          }
        }
      }

      // regex: [^\r\n\p{L}\p{N}]?\p{L}+
      if (!(cCpt == '\r' || cCpt == '\n' || cFlags.is_number)) {
        if (cFlags.vIsLetter || vGetFlags(vPos + 1).vIsLetter) {  // one or more letters
          vPos++;
          while (vGetFlags(vPos).vIsLetter) {
            vPos++;
          }
          vAddToken(vPos);
          continue;
        }
      }

      // regex: \p{N}{1,3}
      if (cFlags.is_number) {
        size_t vIni = vPos;
        while (vGetFlags(vPos).is_number) {
          if (++vPos - vIni >= 3) {
            vAddToken(vPos);
            vIni = vPos;
          }
        }
        vAddToken(vPos);
        continue;
      }

      // regex: <space>?[^\s\p{L}\p{N}]+[\r\n]*
      auto vFlags2 = (cCpt == ' ' ? vGetFlags(vPos + 1) : cFlags);
      if (!(vFlags2.vIsWhitespace | vFlags2.vIsLetter | vFlags2.is_number) && cFlags.mAsUint()) {
        vPos += (cCpt == ' ');
        while (!(vFlags2.vIsWhitespace | vFlags2.vIsLetter | vFlags2.is_number) && vFlags2.mAsUint()) {
          vFlags2 = vGetFlags(++vPos);
        }
        uint32_t vCpt2 = vGetCpt(vPos);
        while (vCpt2 == '\r' || vCpt2 == '\n') {
          vCpt2 = vGetCpt(++vPos);
        }
        vAddToken(vPos);
        continue;
      }

      size_t vNumWhitespaces = 0;
      size_t vLastEndROrN    = 0;
      while (vGetFlags(vPos + vNumWhitespaces).vIsWhitespace) {
        uint32_t vCpt2 = vGetCpt(vPos + vNumWhitespaces);
        if (vCpt2 == '\r' || vCpt2 == '\n') {
          vLastEndROrN = vPos + vNumWhitespaces + 1;
        }
        vNumWhitespaces++;
      }

      // regex: \s*[\r\n]+
      if (vLastEndROrN > 0) {
        vPos = vLastEndROrN;
        vAddToken(vPos);
        continue;
      }

      // regex: \s+(?!\S)
      if (vNumWhitespaces > 1 && vGetCpt(vPos + vNumWhitespaces) != cOutOfRange) {
        vPos += vNumWhitespaces - 1;
        vAddToken(vPos);
        continue;
      }

      // regex: \s+
      if (vNumWhitespaces > 0) {
        vPos += vNumWhitespaces;
        vAddToken(vPos);
        continue;
      }

      // no matches
      vAddToken(++vPos);
    }
  }

  return lBpeOffsets;
}

static std::vector<size_t> fUnicodeRegexSplitCustomK2Horizon(const std::string &         pText,
                                                             const std::vector<size_t> & pOffsets) {
  std::vector<size_t> lBpeOffsets;       // store the offset of each word
  lBpeOffsets.reserve(pOffsets.size());  // Reserve memory for the approximate size

  const auto lCpts = fUnicodeCptsFromUtf8(pText);

  size_t vStart = 0;
  for (auto vOffset : pOffsets) {
    const size_t cOffsetIni = vStart;
    const size_t cOffsetEnd = vStart + vOffset;
    assert(offset_end <= cpts.size());
    vStart = cOffsetEnd;

    static const uint32_t cOutOfRange = 0xFFFFFFFF;
    auto                  vGetCpt     = [&](const size_t pPos) -> uint32_t {
      return (cOffsetIni <= pPos && pPos < cOffsetEnd) ? lCpts[pPos] : cOutOfRange;
    };

    auto vGetFlags = [&](const size_t pPos) -> unicode_cpt_flags {
      return (cOffsetIni <= pPos && pPos < cOffsetEnd) ? fUnicodeCptFlagsFromCpt(lCpts[pPos]) : unicode_cpt_flags{};
    };

    // K2-Horizon: letter runs are (?:\p{L}|\p{M}|\u200C|\u200D)+
    auto vIsK2Letter = [&](const size_t pPos) -> bool {
      const uint32_t cC = vGetCpt(pPos);
      if (cC == 0x200C || cC == 0x200D) {
        return true;
      }
      const auto cF = vGetFlags(pPos);
      return cF.vIsLetter || cF.vIsAccentMark;
    };

    size_t vPrevEnd  = cOffsetIni;
    auto   vAddToken = [&](const size_t end) -> size_t {
      assert(_prev_end <= end && end <= offset_end);
      size_t vLen = end - vPrevEnd;
      if (vLen > 0) {
        lBpeOffsets.push_back(vLen);
      }
      vPrevEnd = end;
      return vLen;
    };

    for (size_t vPos = cOffsetIni; vPos < cOffsetEnd; /*pos++*/) {
      const uint32_t cCpt   = vGetCpt(vPos);
      const auto     cFlags = vGetFlags(vPos);

      // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d) // case insensitive
      if (cCpt == '\'' && vPos + 1 < cOffsetEnd) {
        uint32_t vCptNext = fUnicodeTolower(vGetCpt(vPos + 1));
        if (vCptNext == 0x017F) {
          vCptNext = 's';  // Unicode case-folding of long s
        }
        if (vCptNext == 's' || vCptNext == 't' || vCptNext == 'm' || vCptNext == 'd') {
          vPos += vAddToken(vPos + 2);
          continue;
        }
        if (vPos + 2 < cOffsetEnd) {
          uint32_t vCptNextNext = fUnicodeTolower(vGetCpt(vPos + 2));
          if ((vCptNext == 'r' && vCptNextNext == 'e') || (vCptNext == 'v' && vCptNextNext == 'e') ||
              (vCptNext == 'l' && vCptNextNext == 'l')) {
            vPos += vAddToken(vPos + 3);
            continue;
          }
        }
      }

      // regex: [^\r\n\p{L}\p{N}]?(?:\p{L}|\p{M}|\u200C|\u200D)+
      if (!(cCpt == '\r' || cCpt == '\n' || cFlags.is_number)) {
        if (vIsK2Letter(vPos) || vIsK2Letter(vPos + 1)) {  // one or more letters/marks/ZWNJ/ZWJ
          vPos++;
          while (vIsK2Letter(vPos)) {
            vPos++;
          }
          vAddToken(vPos);
          continue;
        }
      }

      // regex: \p{N}{1,3}
      if (cFlags.is_number) {
        size_t vIni = vPos;
        while (vGetFlags(vPos).is_number) {
          if (++vPos - vIni >= 3) {
            vAddToken(vPos);
            vIni = vPos;
          }
        }
        vAddToken(vPos);
        continue;
      }

      // regex: <space>?[^\s\p{L}\p{N}]+[\r\n]*
      auto vFlags2 = (cCpt == ' ' ? vGetFlags(vPos + 1) : cFlags);
      if (!(vFlags2.vIsWhitespace | vFlags2.vIsLetter | vFlags2.is_number) && cFlags.mAsUint()) {
        vPos += (cCpt == ' ');
        while (!(vFlags2.vIsWhitespace | vFlags2.vIsLetter | vFlags2.is_number) && vFlags2.mAsUint()) {
          vFlags2 = vGetFlags(++vPos);
        }
        uint32_t vCpt2 = vGetCpt(vPos);
        while (vCpt2 == '\r' || vCpt2 == '\n') {
          vCpt2 = vGetCpt(++vPos);
        }
        vAddToken(vPos);
        continue;
      }

      size_t vNumWhitespaces = 0;
      size_t vLastEndROrN    = 0;
      while (vGetFlags(vPos + vNumWhitespaces).vIsWhitespace) {
        uint32_t vCpt2 = vGetCpt(vPos + vNumWhitespaces);
        if (vCpt2 == '\r' || vCpt2 == '\n') {
          vLastEndROrN = vPos + vNumWhitespaces + 1;
        }
        vNumWhitespaces++;
      }

      // regex: \s*[\r\n]+
      if (vLastEndROrN > 0) {
        vPos = vLastEndROrN;
        vAddToken(vPos);
        continue;
      }

      // regex: \s+(?!\S)
      if (vNumWhitespaces > 1 && vGetCpt(vPos + vNumWhitespaces) != cOutOfRange) {
        vPos += vNumWhitespaces - 1;
        vAddToken(vPos);
        continue;
      }

      // regex: \s+
      if (vNumWhitespaces > 0) {
        vPos += vNumWhitespaces;
        vAddToken(vPos);
        continue;
      }

      // no matches
      vAddToken(++vPos);
    }
  }

  return lBpeOffsets;
}

// Qwen2 system regex: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
static std::vector<size_t> fUnicodeRegexSplitCustomQwen2(const std::string &         pText,
                                                         const std::vector<size_t> & pOffsets) {
  std::vector<size_t> lBpeOffsets;       // store the offset of each word
  lBpeOffsets.reserve(pOffsets.size());  // Reserve memory for the approximate size

  const auto lCpts = fUnicodeCptsFromUtf8(pText);

  size_t vStart = 0;
  for (auto vOffset : pOffsets) {
    const size_t cOffsetIni = vStart;
    const size_t cOffsetEnd = vStart + vOffset;
    assert(offset_end <= cpts.size());
    vStart = cOffsetEnd;

    static const uint32_t cOutOfRange = 0xFFFFFFFF;
    auto                  vGetCpt     = [&](const size_t pPos) -> uint32_t {
      return (cOffsetIni <= pPos && pPos < cOffsetEnd) ? lCpts[pPos] : cOutOfRange;
    };

    auto vGetFlags = [&](const size_t pPos) -> unicode_cpt_flags {
      return (cOffsetIni <= pPos && pPos < cOffsetEnd) ? fUnicodeCptFlagsFromCpt(lCpts[pPos]) : unicode_cpt_flags{};
    };

    size_t vPrevEnd  = cOffsetIni;
    auto   vAddToken = [&](const size_t end) -> size_t {
      assert(_prev_end <= end && end <= offset_end);
      size_t vLen = end - vPrevEnd;
      if (vLen > 0) {
        lBpeOffsets.push_back(vLen);
      }
      vPrevEnd = end;
      //if (len > 0) {
      //    std::string s = "";
      //    for(size_t p = end-len; p < end; p++)
      //        s += unicode_cpt_to_utf8(cpts[p]);
      //    printf(">>> '%s'\n", s.c_str());
      //}
      return vLen;
    };

    for (size_t vPos = cOffsetIni; vPos < cOffsetEnd; /*pos++*/) {
      const uint32_t cCpt   = vGetCpt(vPos);
      const auto     cFlags = vGetFlags(vPos);

      // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d) // case insensitive
      if (cCpt == '\'' && vPos + 1 < cOffsetEnd) {
        uint32_t vCptNext = fUnicodeTolower(vGetCpt(vPos + 1));
        if (vCptNext == 's' || vCptNext == 't' || vCptNext == 'm' || vCptNext == 'd') {
          vPos += vAddToken(vPos + 2);
          continue;
        }
        if (vPos + 2 < cOffsetEnd) {
          uint32_t vCptNextNext = fUnicodeTolower(vGetCpt(vPos + 2));
          if ((vCptNext == 'r' && vCptNextNext == 'e') || (vCptNext == 'v' && vCptNextNext == 'e') ||
              (vCptNext == 'l' && vCptNextNext == 'l')) {
            vPos += vAddToken(vPos + 3);
            continue;
          }
        }
      }

      // regex: [^\r\n\p{L}\p{N}]?\p{L}+
      if (!(cCpt == '\r' || cCpt == '\n' || cFlags.is_number)) {
        if (cFlags.vIsLetter || vGetFlags(vPos + 1).vIsLetter) {  // one or more letters
          vPos++;
          while (vGetFlags(vPos).vIsLetter) {
            vPos++;
          }
          vAddToken(vPos);
          continue;
        }
      }

      // regex: \p{N}
      if (cFlags.is_number) {
        vPos++;
        vAddToken(vPos);
        continue;
      }

      // regex: <space>?[^\s\p{L}\p{N}]+[\r\n]*
      auto vFlags2 = (cCpt == ' ' ? vGetFlags(vPos + 1) : cFlags);
      if (!(vFlags2.vIsWhitespace | vFlags2.vIsLetter | vFlags2.is_number) && cFlags.mAsUint()) {
        vPos += (cCpt == ' ');
        while (!(vFlags2.vIsWhitespace | vFlags2.vIsLetter | vFlags2.is_number) && vFlags2.mAsUint()) {
          vFlags2 = vGetFlags(++vPos);
        }
        uint32_t vCpt2 = vGetCpt(vPos);
        while (vCpt2 == '\r' || vCpt2 == '\n') {
          vCpt2 = vGetCpt(++vPos);
        }
        vAddToken(vPos);
        continue;
      }

      size_t vNumWhitespaces = 0;
      size_t vLastEndROrN    = 0;
      while (vGetFlags(vPos + vNumWhitespaces).vIsWhitespace) {
        uint32_t vCpt2 = vGetCpt(vPos + vNumWhitespaces);
        if (vCpt2 == '\r' || vCpt2 == '\n') {
          vLastEndROrN = vPos + vNumWhitespaces + 1;
        }
        vNumWhitespaces++;
      }

      // regex: \s*[\r\n]+
      if (vLastEndROrN > 0) {
        vPos = vLastEndROrN;
        vAddToken(vPos);
        continue;
      }

      // regex: \s+(?!\S)
      if (vNumWhitespaces > 1 && vGetCpt(vPos + vNumWhitespaces) != cOutOfRange) {
        vPos += vNumWhitespaces - 1;
        vAddToken(vPos);
        continue;
      }

      // regex: \s+
      if (vNumWhitespaces > 0) {
        vPos += vNumWhitespaces;
        vAddToken(vPos);
        continue;
      }

      // no matches
      vAddToken(++vPos);
    }
  }

  return lBpeOffsets;
}

// Qwen3.5 system regex: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
// Compared to Qwen2, letter-runs also consume Unicode combining marks (\p{M}): [\p{L}\p{M}]+ instead of \p{L}+
static std::vector<size_t> fUnicodeRegexSplitCustomQwen35(const std::string &         pText,
                                                          const std::vector<size_t> & pOffsets) {
  std::vector<size_t> lBpeOffsets;       // store the offset of each word
  lBpeOffsets.reserve(pOffsets.size());  // Reserve memory for the approximate size

  const auto lCpts = fUnicodeCptsFromUtf8(pText);

  size_t vStart = 0;
  for (auto vOffset : pOffsets) {
    const size_t cOffsetIni = vStart;
    const size_t cOffsetEnd = vStart + vOffset;
    assert(offset_end <= cpts.size());
    vStart = cOffsetEnd;

    static const uint32_t cOutOfRange = 0xFFFFFFFF;
    auto                  vGetCpt     = [&](const size_t pPos) -> uint32_t {
      return (cOffsetIni <= pPos && pPos < cOffsetEnd) ? lCpts[pPos] : cOutOfRange;
    };

    auto vGetFlags = [&](const size_t pPos) -> unicode_cpt_flags {
      return (cOffsetIni <= pPos && pPos < cOffsetEnd) ? fUnicodeCptFlagsFromCpt(lCpts[pPos]) : unicode_cpt_flags{};
    };

    size_t vPrevEnd  = cOffsetIni;
    auto   vAddToken = [&](const size_t end) -> size_t {
      assert(_prev_end <= end && end <= offset_end);
      size_t vLen = end - vPrevEnd;
      if (vLen > 0) {
        lBpeOffsets.push_back(vLen);
      }
      vPrevEnd = end;
      return vLen;
    };

    for (size_t vPos = cOffsetIni; vPos < cOffsetEnd; /*pos++*/) {
      const uint32_t cCpt   = vGetCpt(vPos);
      const auto     cFlags = vGetFlags(vPos);

      // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d) // case insensitive
      if (cCpt == '\'' && vPos + 1 < cOffsetEnd) {
        uint32_t vCptNext = fUnicodeTolower(vGetCpt(vPos + 1));
        if (vCptNext == 's' || vCptNext == 't' || vCptNext == 'm' || vCptNext == 'd') {
          vPos += vAddToken(vPos + 2);
          continue;
        }
        if (vPos + 2 < cOffsetEnd) {
          uint32_t vCptNextNext = fUnicodeTolower(vGetCpt(vPos + 2));
          if ((vCptNext == 'r' && vCptNextNext == 'e') || (vCptNext == 'v' && vCptNextNext == 'e') ||
              (vCptNext == 'l' && vCptNextNext == 'l')) {
            vPos += vAddToken(vPos + 3);
            continue;
          }
        }
      }

      // regex: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
      if (!(cCpt == '\r' || cCpt == '\n' || cFlags.is_number)) {
        if (cFlags.vIsLetter || cFlags.vIsAccentMark || vGetFlags(vPos + 1).vIsAccentMark ||
            vGetFlags(vPos + 1).vIsLetter) {
          vPos++;
          while (vGetFlags(vPos).vIsLetter || vGetFlags(vPos).vIsAccentMark) {
            vPos++;
          }
          vAddToken(vPos);
          continue;
        }
      }

      // regex: \p{N}
      if (cFlags.is_number) {
        vPos++;
        vAddToken(vPos);
        continue;
      }

      // regex: <space>?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
      auto vFlags2 = (cCpt == ' ' ? vGetFlags(vPos + 1) : cFlags);
      if (!(vFlags2.vIsWhitespace | vFlags2.vIsLetter | vFlags2.vIsAccentMark | vFlags2.is_number) &&
          cFlags.mAsUint()) {
        vPos += (cCpt == ' ');
        while (!(vFlags2.vIsWhitespace | vFlags2.vIsLetter | vFlags2.vIsAccentMark | vFlags2.is_number) &&
               vFlags2.mAsUint()) {
          vFlags2 = vGetFlags(++vPos);
        }
        uint32_t vCpt2 = vGetCpt(vPos);
        while (vCpt2 == '\r' || vCpt2 == '\n') {
          vCpt2 = vGetCpt(++vPos);
        }
        vAddToken(vPos);
        continue;
      }

      size_t vNumWhitespaces = 0;
      size_t vLastEndROrN    = 0;
      while (vGetFlags(vPos + vNumWhitespaces).vIsWhitespace) {
        uint32_t vCpt2 = vGetCpt(vPos + vNumWhitespaces);
        if (vCpt2 == '\r' || vCpt2 == '\n') {
          vLastEndROrN = vPos + vNumWhitespaces + 1;
        }
        vNumWhitespaces++;
      }

      // regex: \s*[\r\n]+
      if (vLastEndROrN > 0) {
        vPos = vLastEndROrN;
        vAddToken(vPos);
        continue;
      }

      // regex: \s+(?!\S)
      if (vNumWhitespaces > 1 && vGetCpt(vPos + vNumWhitespaces) != cOutOfRange) {
        vPos += vNumWhitespaces - 1;
        vAddToken(vPos);
        continue;
      }

      // regex: \s+
      if (vNumWhitespaces > 0) {
        vPos += vNumWhitespaces;
        vAddToken(vPos);
        continue;
      }

      // no matches
      vAddToken(++vPos);
    }
  }

  return lBpeOffsets;
}

template <typename CharT>
static std::vector<size_t> fUnicodeRegexSplitStl(const std::basic_string<CharT> & pText,
                                                 const std::basic_string<CharT> & pRegex,
                                                 const std::vector<size_t> &      pOffsets) {
  using BidirIt = typename std::basic_string<CharT>::const_iterator;
#ifdef _MSC_VER
  // Bypass bug in MSVC: 
  constexpr auto regex_flags = std::regex_constants::ECMAScript;
#else
  constexpr auto cRegexFlags = std::regex_constants::optimize | std::regex_constants::nosubs;
#endif
  std::basic_regex<CharT> vExpr(pRegex, cRegexFlags);
  std::vector<size_t>     lBpeOffsets;   // store the offset of each word
  lBpeOffsets.reserve(pOffsets.size());  // Reserve memory for the approximate size
  size_t vStart = 0;
  for (auto vOffset : pOffsets) {
    std::regex_iterator<BidirIt> vIt(pText.begin() + vStart, pText.begin() + vStart + vOffset, vExpr);
    std::regex_iterator<BidirIt> end;

    int64_t vStartIdx = 0;
    while (vIt != end) {
      std::match_results<BidirIt> vMatch = *vIt;
      if (vMatch.position() > vStartIdx) {
        lBpeOffsets.emplace_back(vMatch.position() - vStartIdx);
      }
      lBpeOffsets.emplace_back(vMatch.length());
      vStartIdx = vMatch.position() + vMatch.length();
      ++vIt;
    }

    if (vStartIdx < (int64_t) vOffset) {
      lBpeOffsets.emplace_back(vOffset - vStartIdx);
    }
    vStart += vOffset;
  }

  return lBpeOffsets;
}

// K2 system regex patterns (from tokenization_kimi.py):
// [\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
static std::vector<size_t> fUnicodeRegexSplitCustomKimiK2(const std::string &         pText,
                                                          const std::vector<size_t> & pOffsets) {
  std::vector<size_t> lBpeOffsets;
  lBpeOffsets.reserve(pOffsets.size());

  const auto lCpts = fUnicodeCptsFromUtf8(pText);

  size_t vStart = 0;
  for (auto vOffset : pOffsets) {
    const size_t cOffsetIni = vStart;
    const size_t cOffsetEnd = vStart + vOffset;
    assert(offset_end <= cpts.size());
    vStart = cOffsetEnd;

    static const uint32_t cOutOfRange = 0xFFFFFFFF;
    auto                  vGetCpt     = [&](const size_t pPos) -> uint32_t {
      return (cOffsetIni <= pPos && pPos < cOffsetEnd) ? lCpts[pPos] : cOutOfRange;
    };

    auto vGetFlags = [&](const size_t pPos) -> unicode_cpt_flags {
      return (cOffsetIni <= pPos && pPos < cOffsetEnd) ? fUnicodeCptFlagsFromCpt(lCpts[pPos]) : unicode_cpt_flags{};
    };

    size_t vPrevEnd  = cOffsetIni;
    auto   vAddToken = [&](const size_t end) -> size_t {
      assert(_prev_end <= end && end <= offset_end);
      size_t vLen = end - vPrevEnd;
      if (vLen > 0) {
        lBpeOffsets.push_back(vLen);
      }
      vPrevEnd = end;
      return vLen;
    };

    for (size_t vPos = cOffsetIni; vPos < cOffsetEnd; /*pos++*/) {
      const uint32_t cCpt   = vGetCpt(vPos);
      const auto     cFlags = vGetFlags(vPos);

      // Pattern 1: [\p{Han}]+ (Chinese characters)
      if (fUnicodeCptIsHan(cCpt)) {
        while (fUnicodeCptIsHan(vGetCpt(vPos))) {
          vPos++;
        }
        vAddToken(vPos);
        continue;
      }

      // Pattern 2 & 3: Letter words excluding Han characters with optional contractions
      // [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?:'s|'t|'re|'ve|'m|'ll|'d)?
      // [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?:'s|'t|'re|'ve|'m|'ll|'d)?
      // Check if current char is a letter OR if current char could be a leading char and next char is a letter
      bool vIsLetterPattern = (cFlags.vIsLetter && !fUnicodeCptIsHan(cCpt)) ||
                              (!(cCpt == '\r' || cCpt == '\n' || cFlags.vIsLetter || cFlags.is_number) &&
                               vGetFlags(vPos + 1).vIsLetter && !fUnicodeCptIsHan(vGetCpt(vPos + 1)));

      if (vIsLetterPattern) {
        // Handle optional leading non-letter/non-number character
        bool vHasLeadingChar = false;
        if (!(cCpt == '\r' || cCpt == '\n' || cFlags.vIsLetter || cFlags.is_number)) {
          vHasLeadingChar = true;
          vPos++;
        }

        // Match letter sequence (excluding Han characters)
        bool vHasLetters = false;
        while (vGetFlags(vPos).vIsLetter && !fUnicodeCptIsHan(vGetCpt(vPos))) {
          vHasLetters = true;
          vPos++;
        }

        // Only proceed if we found letters (after potentially skipping leading char)
        if (vHasLetters || (!vHasLeadingChar && vGetFlags(vPos).vIsLetter && !fUnicodeCptIsHan(vGetCpt(vPos)))) {
          if (!vHasLetters) {
            vPos++;  // consume the first letter if we didn't already
          }

          // Continue consuming letters
          while (vGetFlags(vPos).vIsLetter && !fUnicodeCptIsHan(vGetCpt(vPos))) {
            vPos++;
          }

          // Check for optional contractions (?:'s|'t|'re|'ve|'m|'ll|'d)
          if (vGetCpt(vPos) == '\'' && vPos + 1 < cOffsetEnd) {
            uint32_t vCptNext = fUnicodeTolower(vGetCpt(vPos + 1));
            if (vCptNext == 's' || vCptNext == 't' || vCptNext == 'm' || vCptNext == 'd') {
              vPos += 2;
            } else if (vPos + 2 < cOffsetEnd) {
              uint32_t vCptNextNext = fUnicodeTolower(vGetCpt(vPos + 2));
              if ((vCptNext == 'r' && vCptNextNext == 'e') || (vCptNext == 'v' && vCptNextNext == 'e') ||
                  (vCptNext == 'l' && vCptNextNext == 'l')) {
                vPos += 3;
              }
            }
          }

          vAddToken(vPos);
          continue;
        } else if (vHasLeadingChar) {
          // We consumed a leading char but found no letters, backtrack
          vPos--;
        }
      }

      // Pattern 4: \p{N}{1,3} (numbers 1-3 digits)
      if (cFlags.is_number) {
        size_t vIni = vPos;
        while (vGetFlags(vPos).is_number) {
          if (++vPos - vIni >= 3) {
            vAddToken(vPos);
            vIni = vPos;
          }
        }
        vAddToken(vPos);
        continue;
      }

      // Pattern 5:  ?[^\s\p{L}\p{N}]+[\r\n]* (optional space + non-word chars + optional newlines)
      auto vFlags2 = (cCpt == ' ' ? vGetFlags(vPos + 1) : cFlags);
      if (!(vFlags2.vIsWhitespace || vFlags2.vIsLetter || vFlags2.is_number) && vFlags2.mAsUint()) {
        vPos += (cCpt == ' ');
        while (!(vFlags2.vIsWhitespace || vFlags2.vIsLetter || vFlags2.is_number) && vFlags2.mAsUint()) {
          vFlags2 = vGetFlags(++vPos);
        }
        // Match optional [\r\n]*
        uint32_t vCpt2 = vGetCpt(vPos);
        while (vCpt2 == '\r' || vCpt2 == '\n') {
          vCpt2 = vGetCpt(++vPos);
        }
        vAddToken(vPos);
        continue;
      }

      // Count whitespace characters
      size_t vNumWhitespaces = 0;
      size_t vLastEndROrN    = 0;
      while (vGetFlags(vPos + vNumWhitespaces).vIsWhitespace) {
        uint32_t vCpt2 = vGetCpt(vPos + vNumWhitespaces);
        if (vCpt2 == '\r' || vCpt2 == '\n') {
          vLastEndROrN = vPos + vNumWhitespaces + 1;
        }
        vNumWhitespaces++;
      }

      // Pattern 6: \s*[\r\n]+ (whitespace with newlines)
      if (vLastEndROrN > 0) {
        vPos = vLastEndROrN;
        vAddToken(vPos);
        continue;
      }

      // Pattern 7: \s+(?!\S) (trailing whitespace)
      if (vNumWhitespaces > 1 && vGetCpt(vPos + vNumWhitespaces) != cOutOfRange) {
        vPos += vNumWhitespaces - 1;
        vAddToken(vPos);
        continue;
      }

      // Pattern 8: \s+ (general whitespace)
      if (vNumWhitespaces > 0) {
        vPos += vNumWhitespaces;
        vAddToken(vPos);
        continue;
      }

      // No matches - consume single character
      vAddToken(++vPos);
    }
  }

  return lBpeOffsets;
}

// AFMOE digit handling: splits digits with leading 1-2 based on total length modulo 3
static std::vector<size_t> fUnicodeRegexSplitCustomAfmoe(const std::string &         pText,
                                                         const std::vector<size_t> & pOffsets) {
  std::vector<size_t> lBpeOffsets;
  lBpeOffsets.reserve(pOffsets.size());

  const auto lCpts = fUnicodeCptsFromUtf8(pText);

  size_t vStart = 0;
  for (auto vOffset : pOffsets) {
    const size_t cOffsetIni = vStart;
    const size_t cOffsetEnd = vStart + vOffset;
    assert(offset_end <= cpts.size());
    vStart = cOffsetEnd;

    auto vGetFlags = [&](const size_t pPos) -> unicode_cpt_flags {
      return (cOffsetIni <= pPos && pPos < cOffsetEnd) ? fUnicodeCptFlagsFromCpt(lCpts[pPos]) : unicode_cpt_flags{};
    };

    size_t vPrevEnd  = cOffsetIni;
    auto   vAddToken = [&](const size_t end) -> size_t {
      assert(_prev_end <= end && end <= offset_end);
      size_t vLen = end - vPrevEnd;
      if (vLen > 0) {
        lBpeOffsets.push_back(vLen);
      }
      vPrevEnd = end;
      return vLen;
    };

    for (size_t vPos = cOffsetIni; vPos < cOffsetEnd;) {
      const auto cFlags = vGetFlags(vPos);

      // Handle digit sequences with special splitting logic
      if (cFlags.is_number) {
        size_t vDigitStart = vPos;
        size_t vDigitCount = 0;

        // Count consecutive digits
        while (vGetFlags(vPos).is_number && vPos < cOffsetEnd) {
          vDigitCount++;
          vPos++;
        }

        // Split based on total length modulo 3
        size_t vRemainder = vDigitCount % 3;
        size_t vCurrent   = vDigitStart;

        // Emit leading 1-2 digits if needed
        if (vRemainder > 0) {
          vAddToken(vCurrent + vRemainder);
          vCurrent += vRemainder;
        }

        // Emit groups of 3
        while (vCurrent < vDigitStart + vDigitCount) {
          vAddToken(vCurrent + 3);
          vCurrent += 3;
        }
        continue;
      }

      // For non-digits, just move forward
      vPos++;
    }

    // Add any remaining content
    if (vPrevEnd < cOffsetEnd) {
      vAddToken(cOffsetEnd);
    }
  }

  return lBpeOffsets;
}

// regex: [^\n]+|[\n]+
// splits text into runs of non-newline characters and runs of newline characters
static std::vector<size_t> fUnicodeRegexSplitCustomNewlines(const std::string &         pText,
                                                            const std::vector<size_t> & pOffsets) {
  std::vector<size_t> lBpeOffsets;
  lBpeOffsets.reserve(pOffsets.size());

  const auto lCpts = fUnicodeCptsFromUtf8(pText);

  size_t vStart = 0;
  for (auto vOffset : pOffsets) {
    const size_t cOffsetIni = vStart;
    const size_t cOffsetEnd = vStart + vOffset;
    assert(offset_end <= cpts.size());
    vStart = cOffsetEnd;

    size_t vPos = cOffsetIni;
    while (vPos < cOffsetEnd) {
      const bool   cIsNewline = (lCpts[vPos] == '\n');
      const size_t cRunStart  = vPos;
      while (vPos < cOffsetEnd && (lCpts[vPos] == '\n') == cIsNewline) {
        vPos++;
      }
      lBpeOffsets.push_back(vPos - cRunStart);
    }
  }

  return lBpeOffsets;
}

static std::vector<size_t> fUnicodeRegexSplitCustom(const std::string &         pText,
                                                    const std::string &         pRegexExpr,
                                                    const std::vector<size_t> & pOffsets) {
  std::vector<size_t> lBpeOffsets;

  if (pRegexExpr == "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)") {
    lBpeOffsets = fUnicodeRegexSplitCustomGpt2(pText, pOffsets);
  } else if (pRegexExpr ==
               "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| "
               "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" ||
             pRegexExpr ==
               "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| "
               "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {
    lBpeOffsets = fUnicodeRegexSplitCustomDgs3(pText, pOffsets);
  } else if (pRegexExpr ==
             "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| "
             "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {
    lBpeOffsets = fUnicodeRegexSplitCustomQwen2(pText, pOffsets);
  } else if (pRegexExpr ==
             "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| "
             "?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {
    lBpeOffsets = fUnicodeRegexSplitCustomQwen35(pText, pOffsets);
  } else if (pRegexExpr ==
             "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\p{M}|\\u200C|"
             "\\u200D)+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {
    // K2-Horizon: dgs3 splitter with marks + ZWNJ/ZWJ inside letter runs
    // (the generic std::regex fallback cannot parse \p{..} on MSVC)
    lBpeOffsets = fUnicodeRegexSplitCustomK2Horizon(pText, pOffsets);
  } else if (pRegexExpr == "\\p{Han}+") {
    // K2's first pattern - handle all K2 patterns together
    lBpeOffsets = fUnicodeRegexSplitCustomKimiK2(pText, pOffsets);
  } else if (pRegexExpr == "\\p{AFMoE_digits}") {
    // AFMOE digit pattern - use custom implementation for proper splitting
    lBpeOffsets = fUnicodeRegexSplitCustomAfmoe(pText, pOffsets);
  } else if (pRegexExpr == "[^\\n]+|[\\n]+") {
    lBpeOffsets = fUnicodeRegexSplitCustomNewlines(pText, pOffsets);
  } else if (pRegexExpr == "\\d{1,3}(?=(?:\\d{3})*\\b)") {
    // tiny_aya digit grouping pattern from tokenizer.json:
    //   {"type": "Split", "pattern": {"Regex": "\\d{1,3}(?=(?:\\d{3})*\\b)"}, "behavior": "Isolated"}
    // Splits digits into groups of 3 from the right (e.g., 1234567 -> 1, 234, 567)
    // TODO: Revisit this regex, in case there are any subtle tokenization differences with the original regex.
    lBpeOffsets = fUnicodeRegexSplitCustomAfmoe(pText, pOffsets);
  }

  return lBpeOffsets;
}

//
// interface
//

std::string fUnicodeCptToUtf8(uint32_t pCpt) {
  std::string vResult;

  if (/* 0x00 <= cpt && */ pCpt <= 0x7f) {
    vResult.push_back(pCpt);
    return vResult;
  }
  if (0x80 <= pCpt && pCpt <= 0x7ff) {
    vResult.push_back(0xc0 | ((pCpt >> 6) & 0x1f));
    vResult.push_back(0x80 | (pCpt & 0x3f));
    return vResult;
  }
  if (0x800 <= pCpt && pCpt <= 0xffff) {
    vResult.push_back(0xe0 | ((pCpt >> 12) & 0x0f));
    vResult.push_back(0x80 | ((pCpt >> 6) & 0x3f));
    vResult.push_back(0x80 | (pCpt & 0x3f));
    return vResult;
  }
  if (0x10000 <= pCpt && pCpt <= 0x10ffff) {
    vResult.push_back(0xf0 | ((pCpt >> 18) & 0x07));
    vResult.push_back(0x80 | ((pCpt >> 12) & 0x3f));
    vResult.push_back(0x80 | ((pCpt >> 6) & 0x3f));
    vResult.push_back(0x80 | (pCpt & 0x3f));
    return vResult;
  }

  throw std::invalid_argument("invalid codepoint");
}

std::vector<uint32_t> fUnicodeCptsNormalizeNfd(const std::vector<uint32_t> & pCpts) {
  auto vComp = [](const uint32_t pCpt, const range_nfd & pRange) {
    return pCpt < pRange.vFirst;
  };
  std::vector<uint32_t> lResult(pCpts.size());
  for (size_t vI = 0; vI < pCpts.size(); ++vI) {
    const uint32_t cCpt = pCpts[vI];
    auto           vIt  = std::upper_bound(cUnicodeRangesNfd.begin(), cUnicodeRangesNfd.end(), cCpt, vComp) - 1;
    lResult[vI]         = (vIt->vFirst <= cCpt && cCpt <= vIt->vLast) ? vIt->vNfd : cCpt;
  }
  return lResult;
}

std::vector<uint32_t> fUnicodeCptsFromUtf8(const std::string & pUtf8) {
  std::vector<uint32_t> lResult;
  lResult.reserve(pUtf8.size());
  size_t vOffset = 0;
  while (vOffset < pUtf8.size()) {
    try {
      lResult.push_back(fUnicodeCptFromUtf8(pUtf8, vOffset));
    } catch (const std::invalid_argument & /*ex*/) {
      // Silently ignore invalid UTF-8 input to avoid leaking the exception beyond dgs_tokenize
      ++vOffset;
      lResult.emplace_back(0xFFFD);  // replacement character
    }
  }
  return lResult;
}

unicode_cpt_flags fUnicodeCptFlagsFromCpt(const uint32_t pCpt) {
  static const unicode_cpt_flags cUndef(unicode_cpt_flags::cUndefined);
  static const auto              lCptFlags = fUnicodeCptFlagsArray();
  return pCpt < lCptFlags.size() ? lCptFlags[pCpt] : cUndef;
}

unicode_cpt_flags fUnicodeCptFlagsFromUtf8(const std::string & pUtf8) {
  static const unicode_cpt_flags cUndef(unicode_cpt_flags::cUndefined);
  if (pUtf8.empty()) {
    return cUndef;  // undefined
  }
  size_t vOffset = 0;
  return fUnicodeCptFlagsFromCpt(fUnicodeCptFromUtf8(pUtf8, vOffset));
}

std::string fUnicodeByteToUtf8(uint8_t pByte) {
  static std::unordered_map<uint8_t, std::string> dMap = fUnicodeByteToUtf8Map();
  return dMap.at(pByte);
}

uint8_t fUnicodeUtf8ToByte(const std::string & pUtf8) {
  static std::unordered_map<std::string, uint8_t> dMap = fUnicodeUtf8ToByteMap();
  return dMap.at(pUtf8);
}

uint32_t fUnicodeTolower(uint32_t pCpt) {
  // binary search
  auto vIt =
    std::lower_bound(cUnicodeMapLowercase.begin(), cUnicodeMapLowercase.end(), pCpt,
                     [](const std::pair<uint32_t, uint32_t> & pPair, uint32_t value) { return pPair.first < value; });
  if (vIt != cUnicodeMapLowercase.end() && vIt->first == pCpt) {
    return vIt->second;
  }
  return pCpt;  // Return the original code point if no lowercase mapping is found
}

bool fUnicodeCptIsHan(uint32_t pCpt) {
  // Han character ranges (Chinese/CJK characters)
  // CJK Unified Ideographs (most common)
  if (pCpt >= 0x4E00 && pCpt <= 0x9FFF) {
    return true;
  }

  // CJK Extension A
  if (pCpt >= 0x3400 && pCpt <= 0x4DBF) {
    return true;
  }

  // CJK Extension B
  if (pCpt >= 0x20000 && pCpt <= 0x2A6DF) {
    return true;
  }

  // CJK Extension C
  if (pCpt >= 0x2A700 && pCpt <= 0x2B73F) {
    return true;
  }

  // CJK Extension D
  if (pCpt >= 0x2B740 && pCpt <= 0x2B81F) {
    return true;
  }

  // CJK Extension E
  if (pCpt >= 0x2B820 && pCpt <= 0x2CEAF) {
    return true;
  }

  // CJK Extension F
  if (pCpt >= 0x2CEB0 && pCpt <= 0x2EBEF) {
    return true;
  }

  // CJK Compatibility Ideographs
  if (pCpt >= 0xF900 && pCpt <= 0xFAFF) {
    return true;
  }

  // CJK Compatibility Ideographs Supplement
  if (pCpt >= 0x2F800 && pCpt <= 0x2FA1F) {
    return true;
  }

  return false;
}

std::vector<std::string> fUnicodeRegexSplit(const std::string &              pText,
                                            const std::vector<std::string> & pRegexExprs,
                                            bool                             pByteEncode) {
  if (pText.empty()) {
    return {};
  }

  // unicode categories
  static const std::map<std::string, int> dKUcatEnum = {
    { "\\p{N}",  unicode_cpt_flags::cNumber      },
    { "\\p{L}",  unicode_cpt_flags::cLetter      },
    { "\\p{P}",  unicode_cpt_flags::cPunctuation },
    { "\\p{M}",  unicode_cpt_flags::cAccentMark  },
    { "\\p{S}",  unicode_cpt_flags::cSymbol      },
    { "\\p{Lu}", unicode_cpt_flags::cLetter      }, // Uppercase letter
    { "\\p{Ll}", unicode_cpt_flags::cLetter      }, // Lowercase letter
    { "\\p{Lt}", unicode_cpt_flags::cLetter      }, // Titlecase letter
    { "\\p{Lm}", unicode_cpt_flags::cLetter      }, // Modifier letter
    { "\\p{Lo}", unicode_cpt_flags::cLetter      }, // Other letter
  };

  static const std::map<int, int> dKUcatCpt = {
    { unicode_cpt_flags::cNumber,      0xD1 },
    { unicode_cpt_flags::cLetter,      0xD2 },
    { unicode_cpt_flags::cPunctuation, 0xD3 },
    { unicode_cpt_flags::cAccentMark,  0xD4 },
    { unicode_cpt_flags::cSymbol,      0xD5 },
  };

  static const std::map<int, std::string> dKUcatMap = {
    { unicode_cpt_flags::cNumber,      "\x30-\x39"                                 }, // 0-9
    { unicode_cpt_flags::cLetter,      "\x41-\x5A\x61-\x7A"                        }, // A-Za-z
    { unicode_cpt_flags::cPunctuation,
     "\x21-\x23\x25-\x2A\x2C-\x2F\x3A-\x3B\x3F-\x40\\\x5B-\\\x5D\x5F\\\x7B\\\x7D"  }, // !-#%-*,-/:-;?-@\[-\]_\{\}
    { unicode_cpt_flags::cAccentMark,  ""                                          }, // no sub-128 codepoints
    { unicode_cpt_flags::cSymbol,      "\\\x24\\\x2B\x3C-\x3E\x5E\x60\\\x7C\\\x7E" }, // $+<=>^`|~
  };

  // compute collapsed codepoints only if needed by at least one regex
  bool vNeedCollapse = false;
  for (const auto & vRegexExpr : pRegexExprs) {
    // search for unicode categories
    for (const auto & vUcat : dKUcatEnum) {
      if (std::string::npos != vRegexExpr.find(vUcat.first)) {
        vNeedCollapse = true;
        break;
      }
    }
  }

  const auto lCpts = fUnicodeCptsFromUtf8(pText);

  // generate a "collapsed" representation of the text, where all codepoints are replaced by a single byte
  std::string vTextCollapsed;
  if (vNeedCollapse) {
    // collapse all unicode categories
    vTextCollapsed.resize(lCpts.size());

    for (size_t vI = 0; vI < lCpts.size(); ++vI) {
      // keep single-byte codepoints as is
      if (lCpts[vI] < 128) {
        vTextCollapsed[vI] = lCpts[vI];
        continue;
      }

      const auto cFlags = fUnicodeCptFlagsFromCpt(lCpts[vI]);

      if (cFlags.vIsWhitespace) {
        //NOTE: C++ std::regex \s does not mach 0x85, Rust and Python regex does.
        //text_collapsed[i] = (char) 0x85;  // <Next Line> as whitespace fallback
        vTextCollapsed[vI] = (char) 0x0B;  // <vertical tab> as whitespace fallback
      } else if (dKUcatCpt.find(cFlags.mCategoryFlag()) != dKUcatCpt.end()) {
        vTextCollapsed[vI] = dKUcatCpt.at(cFlags.mCategoryFlag());
      } else {
        vTextCollapsed[vI] = (char) 0xD0;  // fallback
      }
    }
  }

  std::vector<size_t> lBpeOffsets = { lCpts.size() };

  for (const auto & vRegexExpr : pRegexExprs) {
    // first, see if we have an efficient custom regex implementation
    auto lTmp = fUnicodeRegexSplitCustom(pText, vRegexExpr, lBpeOffsets);

    if (!lTmp.empty()) {
      lBpeOffsets = std::move(lTmp);
      continue;
    }

    // fallback to general-purpose std::regex / std::wregex
    try {
      // if a unicode category is used in the regex, we use the collapsed text and replace the unicode category
      // with the corresponding collapsed representation
      bool vUseCollapsed = false;
      for (const auto & vUcat : dKUcatEnum) {
        if (std::string::npos != vRegexExpr.find(vUcat.first)) {
          vUseCollapsed = true;
          break;
        }
      }
      const auto lCptsRegex = fUnicodeCptsFromUtf8(vRegexExpr);

      if (vUseCollapsed) {
        // sanity-check that the original regex does not contain any non-ASCII characters
        for (size_t vI = 0; vI < lCptsRegex.size(); ++vI) {
          if (lCptsRegex[vI] >= 128) {
            throw std::runtime_error("Regex includes both unicode categories and non-ASCII characters - not supported");
          }
        }

        // generate a collapsed representation of the regex
        std::string vRegexExprCollapsed;

        // track if we are inside [], because nested [] are not allowed
        bool vInside = false;
        for (size_t vI = 0; vI < vRegexExpr.size(); ++vI) {
          if (vRegexExpr[vI] == '[' && (vI == 0 || vRegexExpr[vI - 1] != '\\')) {
            vRegexExprCollapsed += '[';
            vInside = true;
            continue;
          }

          if (vInside && vRegexExpr[vI] == ']' && vRegexExpr[vI - 1] != '\\') {
            vRegexExprCollapsed += ']';
            vInside = false;
            continue;
          }

          // Match \p{...} Unicode properties of varying lengths
          if (vRegexExpr[vI + 0] == '\\' && vI + 3 < vRegexExpr.size() && vRegexExpr[vI + 1] == 'p' &&
              vRegexExpr[vI + 2] == '{') {
            // Find the closing brace
            size_t vClosingBrace = vRegexExpr.find('}', vI + 3);
            if (vClosingBrace != std::string::npos && vClosingBrace <= vI + 10) {  // reasonable limit
              const std::string cPat = vRegexExpr.substr(vI, vClosingBrace - vI + 1);
              if (dKUcatEnum.find(cPat) != dKUcatEnum.end()) {
                if (!vInside) {
                  vRegexExprCollapsed += '[';
                }
                vRegexExprCollapsed += dKUcatCpt.at(dKUcatEnum.at(cPat));
                vRegexExprCollapsed += dKUcatMap.at(dKUcatEnum.at(cPat));
                if (!vInside) {
                  vRegexExprCollapsed += ']';
                }
                vI = vClosingBrace;
                continue;
              }
            }
          }

          vRegexExprCollapsed += vRegexExpr[vI];
        }

        //printf("text_collapsed: %s\n", text_collapsed.c_str());
        //printf("regex_expr_collapsed: %s\n", regex_expr_collapsed.c_str());
        lBpeOffsets = fUnicodeRegexSplitStl(vTextCollapsed, vRegexExprCollapsed, lBpeOffsets);
      } else {
        // no unicode category used, we can use std::wregex directly
        std::wstring vWregexExpr(lCptsRegex.begin(), lCptsRegex.end());

        // std::wregex \s does not mach non-ASCII whitespaces, using 0x0B as fallback
        std::wstring vWtext(lCpts.begin(), lCpts.end());
        for (size_t vI = 0; vI < vWtext.size(); ++vI) {
          if (vWtext[vI] > 0x7F && fUnicodeCptFlagsFromCpt(vWtext[vI]).vIsWhitespace) {
            vWtext[vI] = 0x0B;
          }
        }

        //printf("text: %s\n", text.c_str());
        //printf("regex_expr: %s\n", regex_expr.c_str());
        lBpeOffsets = fUnicodeRegexSplitStl(vWtext, vWregexExpr, lBpeOffsets);
      }
    } catch (std::regex_error & vE) {
      fprintf(stderr, "Failed to process regex: '%s'\n", vRegexExpr.c_str());
      fprintf(stderr, "Regex error: %s\n", vE.what());
      throw std::runtime_error("Failed to process regex");
    }
  }

  std::vector<std::string> lBpeWords;
  lBpeWords.reserve(lBpeOffsets.size());  // reserve memory for the approximate size

  size_t vStart = 0;
  for (size_t & vOffset : lBpeOffsets) {
    lBpeWords.emplace_back();
    for (size_t vI = vStart; vI < vStart + vOffset; ++vI) {
      lBpeWords.back() += fUnicodeCptToUtf8(lCpts[vI]);
    }
    vStart += vOffset;
  }

  if (pByteEncode) {
    return fUnicodeByteEncodingProcess(lBpeWords);
  }

  return lBpeWords;
}
