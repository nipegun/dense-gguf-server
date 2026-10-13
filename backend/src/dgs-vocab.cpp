#include "dgs-vocab.h"

#include "dgs-impl.h"
#include "dgs-model-loader.h"
#include "ggml.h"
#include "gguf.h"
#include "unicode.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <forward_list>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <unordered_map>

//
// helpers
//

struct naive_trie {
  naive_trie() : has_value(false), value(0) {}

  void insert(const char * pKey, size_t pLen, int32_t value = 0) {
    if (pLen == 0) {
      this->has_value = true;
      this->value     = value;
      return;
    }
    char vC   = pKey[0];
    auto vRes = dChildren.find(vC);
    if (vRes != dChildren.end()) {
      vRes->second.insert(pKey + 1, pLen - 1, value);
    } else {
      auto vRes = dChildren.insert(std::make_pair(vC, naive_trie()));
      vRes.first->second.insert(pKey + 1, pLen - 1, value);
    }
  }

  std::pair<const char *, size_t> mGetLongestPrefix(const char * pKey, size_t pLen, size_t pOffset = 0) const {
    if (pLen == 0 || pOffset == pLen) {
      return std::make_pair(pKey, pOffset);
    }
    char vC   = pKey[pOffset];
    auto vRes = dChildren.find(vC);
    if (vRes != dChildren.end()) {
      return vRes->second.mGetLongestPrefix(pKey, pLen, pOffset + 1);
    }

    return std::make_pair(pKey, pOffset);
  }

  const struct naive_trie * mTraverse(const char pC) const {
    auto vRes = dChildren.find(pC);
    if (vRes != dChildren.end()) {
      return &vRes->second;
    }

    return NULL;
  }
  std::map<char, struct naive_trie> dChildren;
  bool                              has_value;
  dgs_token                         value;
};

//
// tokenizers
//

struct llm_tokenizer {
  llm_tokenizer() {}

  virtual ~llm_tokenizer() = default;
};

struct llm_symbol {
  using index = int;
  index        vPrev;
  index        vNext;
  const char * vText;
  size_t       n;
};

static_assert(std::is_trivially_copyable<llm_symbol>::value, "llm_symbol is not trivially copyable");

//
// SPM tokenizer
// original implementation:
//

struct llm_bigram_spm {
  struct comparator {
    bool operator()(llm_bigram_spm & pL, llm_bigram_spm & pR) {
      return (pL.vScore < pR.vScore) || (pL.vScore == pR.vScore && pL.vLeft > pR.vLeft);
    }
  };

  using queue_storage = std::vector<llm_bigram_spm>;
  using queue         = std::priority_queue<llm_bigram_spm, queue_storage, comparator>;
  llm_symbol::index vLeft;
  llm_symbol::index vRight;
  float             vScore;
  size_t            size;
};

struct llm_tokenizer_spm : llm_tokenizer {
  llm_tokenizer_spm(const dgs_vocab & /*vocab*/) {}
};

struct llm_tokenizer_spm_session {
  llm_tokenizer_spm_session(const dgs_vocab & pVocab) : vVocab(pVocab) {}

  void mTokenize(const std::string & pText, std::vector<dgs_token> & pOutput) {
    // split string into utf8 chars
    int    vIndex = 0;
    size_t vOffs  = 0;
    while (vOffs < pText.size()) {
      llm_symbol vSym;
      size_t     vLen = fUnicodeLenUtf8(pText[vOffs]);
      vSym.vText      = pText.c_str() + vOffs;
      vSym.n          = std::min(vLen, pText.size() - vOffs);
      vOffs += vSym.n;
      vSym.vPrev = vIndex - 1;
      vSym.vNext = vOffs == pText.size() ? -1 : vIndex + 1;
      vIndex++;
      lSymbols.emplace_back(vSym);
    }

    // seed the work queue with all possible 2-character tokens.
    for (int vI = 1; vI < (int) lSymbols.size(); ++vI) {
      mTryAddBigram(vI - 1, vI);
    }

    // keep substituting the highest frequency pairs for as long as we can.
    while (!lWorkQueue.empty()) {
      auto vBigram = lWorkQueue.top();
      lWorkQueue.pop();

      auto & vLeftSym  = lSymbols[vBigram.vLeft];
      auto & vRightSym = lSymbols[vBigram.vRight];

      // if one of the symbols already got merged, skip it.
      if (vLeftSym.n == 0 || vRightSym.n == 0 || vLeftSym.n + vRightSym.n != vBigram.size) {
        continue;
      }

      // merge the right sym into the left one
      vLeftSym.n += vRightSym.n;
      vRightSym.n = 0;

      //DGS_LOG_INFO("left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text, bigram.size);

      // remove the right sym from the chain
      vLeftSym.vNext = vRightSym.vNext;
      if (vRightSym.vNext >= 0) {
        lSymbols[vRightSym.vNext].vPrev = vBigram.vLeft;
      }

      // find more substitutions
      mTryAddBigram(vLeftSym.vPrev, vBigram.vLeft);
      mTryAddBigram(vBigram.vLeft, vLeftSym.vNext);
    }

    for (int vI = 0; vI != -1; vI = lSymbols[vI].vNext) {
      auto & vSymbol = lSymbols[vI];
      mResegment(vSymbol, pOutput);
    }
  }

private:
  void mResegment(llm_symbol & pSymbol, std::vector<dgs_token> & pOutput) {
    auto vText = std::string(pSymbol.vText, pSymbol.n);
    auto token = vVocab.mTextToToken(vText);

    // Do we need to support is_unused?
    if (token != DGS_TOKEN_NULL) {
      pOutput.push_back(token);
      return;
    }

    const auto cP = dRevMerge.find(vText);

    if (cP == dRevMerge.end()) {
      // output any symbols that did not form tokens as bytes.
      pOutput.reserve(pOutput.size() + pSymbol.n);
      for (int vJ = 0; vJ < (int) pSymbol.n; ++vJ) {
        dgs_token vId = vVocab.mByteToToken(pSymbol.vText[vJ]);
        pOutput.push_back(vId);
      }
      return;
    }

    mResegment(lSymbols[cP->second.first], pOutput);
    mResegment(lSymbols[cP->second.second], pOutput);
  }

  void mTryAddBigram(int pLeft, int pRight) {
    if (pLeft == -1 || pRight == -1) {
      return;
    }
    const std::string cText = std::string(lSymbols[pLeft].vText, lSymbols[pLeft].n + lSymbols[pRight].n);
    auto              token = vVocab.mTextToToken(cText);

    if (token == DGS_TOKEN_NULL) {
      return;
    }

    if (static_cast<uint32_t>(token) >= vVocab.n_tokens()) {
      return;
    }

    const auto & vTokData = vVocab.mGetTokenData(token);

    llm_bigram_spm vBigram;
    vBigram.vLeft  = pLeft;
    vBigram.vRight = pRight;
    vBigram.vScore = vTokData.vScore;
    vBigram.size   = cText.size();

    lWorkQueue.push(vBigram);

    // Do we need to support is_unused?
    dRevMerge[cText] = std::make_pair(pLeft, pRight);
  }

  const dgs_vocab & vVocab;
  // currently unused
  // const llm_tokenizer_spm * spm_tokenizer;

  std::vector<llm_symbol>                    lSymbols;
  llm_bigram_spm::queue                      lWorkQueue;
  std::map<std::string, std::pair<int, int>> dRevMerge;
};

//
// BPE tokenizer
// adapted from https://github.com/cmp-nct/ggllm.cpp [MIT License]
// tried to simplify unicode stuff, so most likely does not work 100% correctly!
//

// TODO: there are a lot of common parts between spm and bpe tokenizers, should be refactored and reused

template <typename T, typename Container = std::vector<T>, typename Compare = std::less<typename Container::value_type>>
class dgs_priority_queue : public std::priority_queue<T, Container, Compare> {
public:
  using std::priority_queue<T, Container, Compare>::priority_queue;

  T mPopMove() {
    T vItem = std::move(this->c.front());
    std::pop_heap(this->c.begin(), this->c.end(), this->comp);
    this->c.pop_back();
    return vItem;
  }

  void mPop() = delete;
};

struct llm_bigram_bpe {
  struct comparator {
    bool operator()(const llm_bigram_bpe & pL, const llm_bigram_bpe & pR) const {
      return pL.vRank > pR.vRank || (pL.vRank == pR.vRank && pL.vLeft > pR.vLeft);
    }
  };

  using queue_storage = std::vector<llm_bigram_bpe>;
  using queue         = dgs_priority_queue<llm_bigram_bpe, queue_storage, comparator>;
  llm_symbol::index vLeft;
  llm_symbol::index vRight;
  std::string       vText;
  int               vRank;
  size_t            size;
};

struct llm_tokenizer_bpe : llm_tokenizer {
  llm_tokenizer_bpe(const dgs_vocab & pVocab) {
    GGML_ASSERT(pVocab.mGetType() == cDgsVocabTypeBpe);
    switch (pVocab.mGetPreType()) {
      case cDgsVocabPreTypeDgs3:
        lRegexExprs = {
          // original regex from tokenizer.json
          //"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",

          // adapted: https://github.com/ggml-org/llama.cpp/pull/6920#issuecomment-2080233989
          "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| "
          "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };
        break;
      case cDgsVocabPreTypeJais2:
        lRegexExprs = {
          // original regex from tokenizer.json
          //"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s{512}(?!\\S)|\\s{256}(?!\\S)|\\s{128}(?!\\S)|\\s{64}(?!\\S)|\\s{32}(?!\\S)|\\s{16}(?!\\S)|\\s{8}(?!\\S)|\\s{4}(?!\\S)|\\s{1,2}(?!\\S)|\\s{1}",

          // adapted: same as dgs3 but with cascading whitespace pattern
          "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| "
          "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s{512}(?!\\S)|\\s{256}(?!\\S)|\\s{128}(?!\\S)|\\s{64}(?!\\S)|"
          "\\s{32}(?!\\S)|\\s{16}(?!\\S)|\\s{8}(?!\\S)|\\s{4}(?!\\S)|\\s{1,2}(?!\\S)|\\s{1}",
        };
        break;
      case cDgsVocabPreTypeDbrx:
      case cDgsVocabPreTypeSmaug:
        lRegexExprs = {
          // same as dgs3
          "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| "
          "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };
        break;
      case cDgsVocabPreTypeDeepseekLlm:
        lRegexExprs = {
          "[\r\n]",
          "\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-"
          "ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-"
          "𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+",
          "\\s?[!-/:-~！-／：-～‘-‟　-。]+",
          "\\s+$",
          "[一-龥ࠀ-一가-퟿]+",
          "\\p{N}+",
        };
        break;
      case cDgsVocabPreTypeDeepseek3Llm:
      case cDgsVocabPreTypeHunyuanDense:
      case cDgsVocabPreTypeJoyaiLlm:
      case cDgsVocabPreTypeHyV4:
        lRegexExprs = {
          "\\p{N}{1,3}",
          "[一-龥぀-ゟ゠-ヿ]+",
          "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| "
          "?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
        };
        break;
      case cDgsVocabPreTypeSPARK2Point5:
        lRegexExprs = {
          "\\p{N}{1,3}",
          "[一-龥぀-ゟ゠-ヿ]+",
          "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| "
          "?[\\p{P}\\p{S}]+|[\r\n]|\\s+(?!\\S)|\\s+",
          "\\p{N}",
        };
        break;
      case cDgsVocabPreTypeYoutu:
        lRegexExprs = {
          "[가-힣ㄱ-ㆎ]+|[！…“”‘’—：；，、-〿︰-﹏]+|[ㄅ-ㄯ]+|[一-龥぀-ゟ゠-ヿ]+",
          "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'["
          "rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{"
          "Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}| "
          "?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };
        break;
      case cDgsVocabPreTypeDeepseekCoder:
        lRegexExprs = {
          "[\r\n]", "\\s?\\p{L}+", "\\s?\\p{P}+", "[一-龥ࠀ-一가-퟿]+", "\\p{N}",
        };
        break;
      case cDgsVocabPreTypeFalcon:
        lRegexExprs = {
          "[\\p{P}\\$\\+<=>\\^~\\|`]+",
          "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
          "[0-9][0-9][0-9]",
        };
        break;
      case cDgsVocabPreTypeStarcoder:
      case cDgsVocabPreTypeRefact:
      case cDgsVocabPreTypeCommandR:
      case cDgsVocabPreTypeSmollm:
      case cDgsVocabPreTypeCodeshell:
      case cDgsVocabPreTypeExaone:
      case cDgsVocabPreTypeMinerva:
      case cDgsVocabPreTypeMellum2:
        lRegexExprs = {
          "\\p{N}",
          "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
        };
        break;
      case cDgsVocabPreTypeGpt2:
      case cDgsVocabPreTypeMpt:
      case cDgsVocabPreTypeOlmo:
      case cDgsVocabPreTypeJais:
      case cDgsVocabPreTypeTrillion:
      case cDgsVocabPreTypeGraniteDocling:
        lRegexExprs = {
          "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
        };
        break;
      case cDgsVocabPreTypeStablelm2:
      case cDgsVocabPreTypeQwen2:
      case cDgsVocabPreTypeHunyuan:
      case cDgsVocabPreTypeSolarOpen:
        lRegexExprs = {
          // original regex from tokenizer.json
          // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
          "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| "
          "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };
        break;
      case cDgsVocabPreTypeQwen35:
        lRegexExprs = {
          // original regex from tokenizer.json
          // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
          "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| "
          "?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };
        break;
      case cDgsVocabPreTypePoro:
      case cDgsVocabPreTypeBloom:
      case cDgsVocabPreTypeGpt3Finnish:
        lRegexExprs = {
          " ?[^(\\s|.,!?…。，、।۔،)]+",
        };
        break;
      case cDgsVocabPreTypeChatglm4:
        lRegexExprs = {
          "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| "
          "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };
        break;
      case cDgsVocabPreTypeViking:
        lRegexExprs = {
          " ?[^(\\s|.,!?…。，、।۔،)]+",
          "\\p{N}",
        };
        break;
      case cDgsVocabPreTypeTekken:
        // original regex from tokenizer.json
        // "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
        lRegexExprs = {
          "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^"
          "a-z]))+((?=[\\p{L}])([^A-Z]))*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };
        break;
      case cDgsVocabPreTypeChameleon:
        // Note: in theory, the special token (sentinel and image token) regex_exprs below
        // are unnecessary, as they are split in `tokenizer_st_partition` anyway.
        // However, since the upstream pre-tokenizer uses them, they are also
        // included here (see https://huggingface.co/facebook/chameleon-7b).
        lRegexExprs = {
          "<sentinel:[0-9]+>",                    // Sentinel tokens
          "(IMGIMG)((A|B|C|D|E|F|G|H|I){1,4})Z",  // Image tokens
          "([\\t\\n]|    |  )",                   // directly from tokenizer.json
          "\\p{N}",                               // Individual digits
          "[\\p{P}!-/:-@\\[-`{-~]",               // Punctuation, Isolated
          "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
        };
        break;
      case cDgsVocabPreTypeGpt4o:
      case cDgsVocabPreTypeMinimaxM2:
        lRegexExprs = {
          // original regex from tokenizer.json
          // "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
          "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]"
          "|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR]["
          "eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| "
          "?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };
        break;
      case cDgsVocabPreTypeGraniteEmbMulti:
        // Same lookaheads as GPT4O but with \p{M} added so combining marks
        // (diacritics) attach to their base letters. Avoids excessive
        // backtracking on scripts that use them heavily (Bengali, Hindi,
        // Telugu, Thai, ...). See PR #22716 for benchmarks.
        lRegexExprs = {
          "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))*((?=[\\p{L}\\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'["
          "vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}\\p{M}])([^a-z]))+((?=[\\p{L}\\p{M}])([^A-"
          "Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| "
          "?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };
        break;
      case cDgsVocabPreTypeTinyAya:
        lRegexExprs = {
          // original regex from tokenizer.json: "\\d{1,3}(?=(?:\\d{3})*\\b)"
          "\\d{1,3}(?=(?:\\d{3})*\\b)",
          // original regex from tokenizer.json: "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
          "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?:'[sS]|'[tT]|'["
          "rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{"
          "Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| "
          "?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };
        break;
      case cDgsVocabPreTypeKimiK2:
        lRegexExprs = {
          // K2 trigger pattern - this will activate the custom K2 handler in unicode.cpp
          // The custom handler implements all K2 patterns with proper Han character exclusion
          "\\p{Han}+",
        };
        break;
      case cDgsVocabPreTypeSuperbpe:
        lRegexExprs = {
          "\\p{N}+",
          "(?=(\\d{3})+(?!\\d))",
        };
        break;
      case cDgsVocabPreTypeBailingmoe:
        lRegexExprs = {
          // original regex from tokenizer.json
          // "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+"
          // FIXME? Changed possessive quantifiers (?+ and ++) to greedy to avoid errors and imatrix hanging (tried atomic grouping but it's not supported?)
          "'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| "
          "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+",
        };
        break;
      case cDgsVocabPreTypeSeedCoder:
        lRegexExprs = {
          // original regex from tokenizer.json
          // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}\r\n]+|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"
          "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| "
          "?[^\\s\\p{L}\\p{N}\\r\\n]+|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };
        break;
      case cDgsVocabPreTypeUfakzeka:
        lRegexExprs = {
          // Qwen2 pattern without the English contraction group, so Turkish apostrophe suffixes stay attached
          "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };
        break;
      case cDgsVocabPreTypeGrok2:
        lRegexExprs = {
          // original regex from tokenizer.json
          // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
          "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| "
          "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };
        break;
      case cDgsVocabPreTypeAfmoe:
        lRegexExprs = {
          // Digit handling - uses custom implementation in unicode.cpp
          // Groups digits with leading 1-2 based on total length modulo 3
          "\\p{AFMoE_digits}",
          // CJK and Asian scripts (using direct Unicode literals)
          "[一-鿿㐀-䶿豈-﫿぀-ゟ゠-ヿ･-ﾟ⼀-⿟เ-๿຀-໿ក-៿က-႟ꩠ-ꩿꧠ-꧿가-힯ᄀ-ᇿ]"
          "+",
          // Main BPE pattern
          "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| "
          "?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };
        break;
      case cDgsVocabPreTypeLaguna:
        lRegexExprs = {
          "[^\\n]+|[\\n]+",
          "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| "
          "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };
        break;
      case cDgsVocabPreTypeExaoneMoe:
        lRegexExprs = {
          // original regex from tokenizer.json
          // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}\\p{M}*(?: \\p{L}\\p{M}*)*)+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]?|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+"
          "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}\\p{M}*(?: "
          "\\p{L}\\p{M}*)*)+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]?|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+",
        };
        break;
      case cDgsVocabPreTypeGemma4:
        // Gemma4 uses SPM-style BPE: spaces are replaced with ▁ by the
        // normalizer, then BPE merges run on the whole text without
        // word-level pre-splitting. We only need to split on newlines
        // since BPE merge lookup asserts no newlines in tokens.
        lRegexExprs = {
          "[^\\n]+|[\\n]+",
        };
        vByteEncode = false;  // uses raw UTF-8, not GPT-2 byte encoding
        break;
      case cDgsVocabPreTypeSarvamMoe:
        // Sarvam uses SPM-style BPE (same shape as Gemma4): spaces replaced with U+2581
        // by the normalizer, BPE merges over the whole text on raw UTF-8.
        lRegexExprs = {
          "[^\\n]+|[\\n]+",
        };
        vByteEncode = false;
        break;
      case cDgsVocabPreTypeMmbert:
        // same as Gemma4, the words are split in tokenize()
        lRegexExprs = {
          "[^\\n]+|[\\n]+",
        };
        vByteEncode = false;
        break;
      case cDgsVocabPreTypeMinicpm5:
        lRegexExprs = {
          // original regex from tokenizer.json (openbmb/MiniCPM5-1B)
          "\\p{N}{1,3}",
          // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}+| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
          "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}+| "
          "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };
        break;
      case cDgsVocabPreTypeK2Horizon:
        lRegexExprs = {
          "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\p{M}|\\u200C|"
          "\\u200D)+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };
        break;
      case cDgsVocabPreTypeWhitespace:
        // whitespace pre-tokenizer (jinaai/jina-embeddings-v2-base-zh)
        lRegexExprs = {
          "\\S+",
        };
        vByteEncode = false;
        break;
      default:
        // default regex for BPE tokenization pre-processing
        lRegexExprs = {
          "[\\p{P}\\$\\+<=>\\^~\\|]+",
          "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
          "\\p{N}+",
          "[0-9][0-9][0-9]",
        };
        break;
    }
  }

  std::vector<std::string> lRegexExprs;
  bool                     vByteEncode = true;  // GPT-2 byte encoding; false for SPM-style BPE (raw UTF-8)
};

struct llm_tokenizer_bpe_session {
  llm_tokenizer_bpe_session(const dgs_vocab & pVocab, const llm_tokenizer_bpe & pTokenizer) :
    vVocab(pVocab),
    vTokenizer(pTokenizer) {}

  virtual ~llm_tokenizer_bpe_session() = default;

  static void mAppend(const dgs_token pTokenId, std::vector<dgs_token> & pOutput) { pOutput.push_back(pTokenId); }

  bool mAppendBos(std::vector<dgs_token> & pOutput) const {
    if (vVocab.mGetAddBos()) {
      GGML_ASSERT(vVocab.mTokenBos() != DGS_TOKEN_NULL);
      pOutput.push_back(vVocab.mTokenBos());
      return true;
    }
    return false;
  }

  bool mAppendEos(std::vector<dgs_token> & pOutput) const {
    if (vVocab.mGetAddEos()) {
      GGML_ASSERT(vVocab.mTokenEos() != DGS_TOKEN_NULL);
      pOutput.push_back(vVocab.mTokenEos());
      return true;
    }
    return false;
  }

  void mCheckDoubleBosEos(const std::vector<dgs_token> & pOutput) const {
    if (vVocab.mGetAddBos() && pOutput.size() >= 2 && pOutput[1] == vVocab.mTokenBos()) {
      DGS_LOG_WARN(
        "%s: Added a BOS token to the prompt as specified by the model but the prompt "
        "also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. "
        "Are you sure this is what you want?\n",
        __FUNCTION__);
    }
    if (vVocab.mGetAddEos() && pOutput.size() >= 2 && *(pOutput.end() - 2) == vVocab.mTokenEos()) {
      DGS_LOG_WARN(
        "%s: Added a EOS token to the prompt as specified by the model but the prompt "
        "also ends with a EOS token. So now the final prompt ends with 2 EOS tokens. "
        "Are you sure this is what you want?\n",
        __FUNCTION__);
    }
  }

  virtual void mTokenize(const std::string & pText, std::vector<dgs_token> & pOutput) {
    int  vFinalPrevIndex = -1;
    auto lWordCollection = fUnicodeRegexSplit(pText, vTokenizer.lRegexExprs, vTokenizer.vByteEncode);

    lSymbolsFinal.clear();
    auto vTokPre = vVocab.mGetPreType();

    if (vTokPre == cDgsVocabPreTypeMmbert) {
      // Metaspace pre-tokenizer: a text starts with an escaped space, and each escaped space starts a new word
      static const std::string cSpace = "\xe2\x96\x81";
      std::vector<std::string> lWords;
      for (auto & vWord : lWordCollection) {
        if (vWord.find_first_not_of('\n') == std::string::npos) {
          lWords.push_back(vWord);
          continue;
        }
        if (vWord.compare(0, cSpace.size(), cSpace) != 0) {
          vWord = cSpace + vWord;
        }
        size_t vStart = 0;
        while (vStart < vWord.size()) {
          size_t end = vWord.find(cSpace, vStart + cSpace.size());
          end        = end == std::string::npos ? vWord.size() : end;
          lWords.push_back(vWord.substr(vStart, end - vStart));
          vStart = end;
        }
      }
      lWordCollection = std::move(lWords);
    }

    for (const auto & vWord : lWordCollection) {
      lWorkQueue = llm_bigram_bpe::queue();
      lSymbols.clear();

      int    vIndex  = 0;
      size_t vOffset = 0;

      //if (vocab.tokenizer_ignore_merges && vocab.token_to_id.find(word) != vocab.token_to_id.end()) {
      if (vVocab.mGetIgnoreMerges() && vVocab.mTextToToken(vWord) != DGS_TOKEN_NULL) {
        lSymbols.emplace_back(llm_symbol{ -1, -1, vWord.c_str(), vWord.size() });
        vOffset = vWord.size();
      } else if ((vTokPre == cDgsVocabPreTypeGemma4 || vTokPre == cDgsVocabPreTypeMmbert) &&
                 vWord.find_first_not_of('\n') == std::string::npos) {
        // fix for gemma 4, ref: 
        auto vTok = vVocab.mTextToToken(vWord);
        if (vTok != DGS_TOKEN_NULL) {
          lSymbols.emplace_back(llm_symbol{ -1, -1, vWord.c_str(), vWord.size() });
          vOffset = vWord.size();
        }
      }

      while (vOffset < vWord.size()) {
        llm_symbol vSym;
        size_t     vCharLen = std::min(vWord.size() - vOffset, (size_t) fUnicodeLenUtf8(vWord[vOffset]));
        vSym.vText          = vWord.c_str() + vOffset;
        vSym.n              = vCharLen;
        vOffset += vSym.n;
        vSym.vPrev = vIndex - 1;
        vSym.vNext = vOffset == vWord.size() ? -1 : vIndex + 1;
        vIndex++;
        lSymbols.emplace_back(vSym);
      }
      for (int vI = 1; vI < (int) lSymbols.size(); ++vI) {
        mAddNewBigram(vI - 1, vI);
      }

      // build token(s)
      while (!lWorkQueue.empty()) {
        auto vBigram = lWorkQueue.mPopMove();

        auto & vLeftSymbol  = lSymbols[vBigram.vLeft];
        auto & vRightSymbol = lSymbols[vBigram.vRight];

        if (vLeftSymbol.n == 0 || vRightSymbol.n == 0) {
          continue;
        }
        std::string vLeftToken  = std::string(vLeftSymbol.vText, vLeftSymbol.n);
        std::string vRightToken = std::string(vRightSymbol.vText, vRightSymbol.n);
        if (vLeftToken + vRightToken != vBigram.vText) {
          continue;  // Skip this bigram if it's outdated
        }

        // merge the right sym into the left one
        vLeftSymbol.n += vRightSymbol.n;
        vRightSymbol.n = 0;

        // remove the right sym from the chain
        vLeftSymbol.vNext = vRightSymbol.vNext;
        if (vRightSymbol.vNext >= 0) {
          lSymbols[vRightSymbol.vNext].vPrev = vBigram.vLeft;
        }

        mAddNewBigram(vLeftSymbol.vPrev, vBigram.vLeft);  // left side of current symbol
        mAddNewBigram(vBigram.vLeft, vLeftSymbol.vNext);  // right side of current symbol
      }

      // add the finished tokens to the final list keeping correct order for next and prev
      for (auto & vSym : lSymbols) {
        if (vSym.n > 0) {
          vSym.vPrev = vFinalPrevIndex;
          vSym.vNext = -1;
          if (vFinalPrevIndex != -1) {
            lSymbolsFinal[vFinalPrevIndex].vNext = lSymbolsFinal.size();
          }
          lSymbolsFinal.emplace_back(vSym);
          vFinalPrevIndex = lSymbolsFinal.size() - 1;
        }
      }
    }

    lSymbols = lSymbolsFinal;

    if (!lSymbols.empty()) {
      for (int vI = 0; vI != -1; vI = lSymbols[vI].vNext) {
        auto & vSymbol = lSymbols[vI];
        if (vSymbol.n == 0) {
          continue;
        }

        const std::string str   = std::string(vSymbol.vText, vSymbol.n);
        const auto        token = vVocab.mTextToToken(str);

        if (token == DGS_TOKEN_NULL) {
          for (auto vJ = str.begin(); vJ != str.end(); ++vJ) {
            dgs_token vTokenMultibyte = DGS_TOKEN_NULL;
            if (vTokenizer.vByteEncode) {
              std::string vByteStr(1, *vJ);
              vTokenMultibyte = vVocab.mTextToToken(vByteStr);
            } else {
              // For non-byte-encoded BPE (e.g. gemma-4), byte tokens use <0xXX> format
              static const char * vHex    = "0123456789ABCDEF";
              const uint8_t       cCh     = (uint8_t) *vJ;
              const char          aBuf[7] = { '<', '0', 'x', vHex[cCh >> 4], vHex[cCh & 15], '>', 0 };
              vTokenMultibyte             = vVocab.mTextToToken(aBuf);
            }
            if (vTokenMultibyte != DGS_TOKEN_NULL) {
              pOutput.push_back(vTokenMultibyte);
            }
          }
        } else {
          pOutput.push_back(token);
        }
      }
    }
  }

private:
  void mAddNewBigram(int pLeft, int pRight) {
    if (pLeft == -1 || pRight == -1) {
      return;
    }
    std::string vLeftToken  = std::string(lSymbols[pLeft].vText, lSymbols[pLeft].n);
    std::string vRightToken = std::string(lSymbols[pRight].vText, lSymbols[pRight].n);

    int vRankFound = -1;

    vRankFound = vVocab.mFindBpeRank(vLeftToken, vRightToken);

    if (vRankFound < 0) {
      return;
    }

    llm_bigram_bpe vBigram;

    vBigram.vLeft  = pLeft;
    vBigram.vRight = pRight;
    vBigram.vText  = vLeftToken + vRightToken;
    vBigram.size   = vLeftToken.size() + vRightToken.size();
    vBigram.vRank  = vRankFound;

    lWorkQueue.push(vBigram);
  }

  const dgs_vocab &         vVocab;
  const llm_tokenizer_bpe & vTokenizer;

  std::vector<llm_symbol> lSymbols;
  std::vector<llm_symbol> lSymbolsFinal;
  llm_bigram_bpe::queue   lWorkQueue;
};

//
// WPM tokenizer
//

struct llm_tokenizer_wpm : llm_tokenizer {
  llm_tokenizer_wpm(const dgs_vocab & /*vocab*/) {}
};

struct llm_tokenizer_wpm_session {
  llm_tokenizer_wpm_session(const dgs_vocab & pVocab) : vVocab(pVocab) {}

  void mTokenize(const std::string & pText, std::vector<dgs_token> & pOutput) {
    // normalize and split by whitespace
    std::vector<std::string> lWords = mPreprocess(pText, vVocab.mGetNormalizerOpts());
    // bos token prepended already

    // find the longest tokens that form the words
    for (const std::string & vWord : lWords) {
      // skip empty words
      if (vWord.size() == 0) {
        continue;
      }

      // prepend phantom space
      const std::string cWord1 = "\xe2\x96\x81" + vWord;
      const int         n      = cWord1.size();

      const size_t cCurrentTokens = pOutput.size();

      // we're at the start of a new word
      // move through character position in word
      for (int vI = 0; vI < n; ++vI) {
        // loop through possible match length
        bool vMatch = false;
        for (int vJ = std::min(n, vI + vVocab.mMaxTokenLen() + 1); vJ > vI; vJ--) {
          auto vId = vVocab.mTextToToken(cWord1.substr(vI, vJ - vI));
          if (vId != DGS_TOKEN_NULL) {
            pOutput.push_back(vId);
            vMatch = true;
            vI     = vJ - 1;
            break;
          }
        }

        if (!vMatch) {  // discard all
          pOutput.resize(cCurrentTokens);
          break;        // and discard next tokens
        }
      }

      // we didn't find any matches for this word
      if (cCurrentTokens == pOutput.size()) {
        pOutput.push_back(vVocab.mTokenUnk());
      }
    }
  }

  // TODO: reduce string copies by using cpts_offs array
  static std::vector<std::string> mPreprocess(const std::string &                   pText,
                                              const dgs_vocab::normalizer_options & pNormalizerOpts) {
    std::vector<uint32_t> lCpts = fUnicodeCptsFromUtf8(pText);
    if (pNormalizerOpts.vStripAccents) {
      lCpts = fUnicodeCptsNormalizeNfd(lCpts);
    }
    std::vector<std::string> lWords(1, "");

    for (const uint32_t cCpt : lCpts) {
      const auto cFlags = fUnicodeCptFlagsFromCpt(cCpt);

      if (cFlags.vIsWhitespace) {
        if (lWords.back().size()) {  // finish previous word if any
          lWords.emplace_back();
        }
        continue;
      }

      assert(!flags.is_separator);
      if (cCpt == 0 || cCpt == 0xFFFD || cFlags.vIsControl) {
        continue;
      }

      if (pNormalizerOpts.vStripAccents && cFlags.vIsAccentMark) {
        continue;
      }

      const std::string cS = fUnicodeCptToUtf8(pNormalizerOpts.vLowercase ? fUnicodeTolower(cCpt) : cCpt);
      if (cFlags.vIsPunctuation || (cCpt < 0x7F && cFlags.vIsSymbol) || mIsChineseChar(cCpt)) {
        if (lWords.back().size()) {  // finish previous word if any
          lWords.emplace_back();
        }
        lWords.back() = cS;     // single char word
        lWords.emplace_back();  // start a new word
      } else {
        lWords.back() += cS;    // append char to word
      }
    }

    if (!lWords.back().size()) {
      lWords.pop_back();
    }

    return lWords;
  }

  static bool mIsChineseChar(uint32_t pCpt) {
    return (pCpt >= 0x04E00 && pCpt <= 0x09FFF) || (pCpt >= 0x03400 && pCpt <= 0x04DBF) ||
           (pCpt >= 0x20000 && pCpt <= 0x2A6DF) || (pCpt >= 0x2A700 && pCpt <= 0x2B73F) ||
           (pCpt >= 0x2B740 && pCpt <= 0x2B81F) ||
           (pCpt >= 0x2B920 && pCpt <= 0x2CEAF) ||  // this should be 0x2B820 but in hf rust code it is 0x2B920
           (pCpt >= 0x0F900 && pCpt <= 0x0FAFF) || (pCpt >= 0x2F800 && pCpt <= 0x2FA1F);
    //(cpt >= 0x3000  && cpt <= 0x303F)  ||
    //(cpt >= 0xFF00  && cpt <= 0xFFEF);
  }

private:
  const dgs_vocab & vVocab;
  // currently unused
  // const llm_tokenizer_wpm * wpm_tokenizer;
};

//
// UGM tokenizer
//

struct llm_tokenizer_ugm : llm_tokenizer {
  llm_tokenizer_ugm(const dgs_vocab & pVocab, const std::vector<char> & pPrecompiledCharsmap) {
    if (pPrecompiledCharsmap.size() > 0) {
      size_t vCharsmapOffset = 0;

      // First four bytes of precompiled_charsmap contains length of binary
      // blob containing XOR-compressed compact double array (XCDA) entries
      uint32_t vXcdaBlobSize = *(const uint32_t *) &pPrecompiledCharsmap[0];
      vCharsmapOffset += sizeof(vXcdaBlobSize);

      // Next xcda_blob_size bytes contain entries of XOR-compressed compact
      // double array (XCDA). Each entry is bit-packed into a 32-bit integer.
      vXcdaArray     = (const uint32_t *) &pPrecompiledCharsmap[vCharsmapOffset];
      vXcdaArraySize = vXcdaBlobSize / sizeof(uint32_t);
      vCharsmapOffset += vXcdaBlobSize;

      // Remaining bytes of precompiled charsmap contain null-terminated
      // replacement strings for prefixes matched by the XCDA.
      vPrefixReplacements     = &pPrecompiledCharsmap[vCharsmapOffset];
      vPrefixReplacementsSize = pPrecompiledCharsmap.size() - vCharsmapOffset;
    }

    for (uint32_t vId = 0; vId < pVocab.n_tokens(); ++vId) {
      const auto & vTokenData = pVocab.mGetTokenData(vId);

      if (pVocab.mIsNormal(vId)) {
        vMinScore = std::min<float>(vMinScore, vTokenData.vScore);
        vMaxScore = std::max<float>(vMaxScore, vTokenData.vScore);
      }

      if (pVocab.mIsNormal(vId) || pVocab.mIsUserDefined(vId) || pVocab.mIsUnused(vId)) {
        vTokenMatcher.insert(vTokenData.vText.data(), vTokenData.vText.size(), vId);
      }

      if (pVocab.mIsUserDefined(vId)) {
        vUserDefinedTokenMatcher.insert(vTokenData.vText.data(), vTokenData.vText.size());
      }
    }

    vUnknownTokenScore = vMinScore - vUnknownTokenScorePenalty;
  }

  // escaped space symbol - U+2581 (Lower One Eighth Block)
  const std::string cEscapedSpace = "\xE2\x96\x81";

  const char * vPrefixReplacements     = NULL;
  size_t       vPrefixReplacementsSize = 0;

  const uint32_t * vXcdaArray     = NULL;
  size_t           vXcdaArraySize = 0;

  struct naive_trie vUserDefinedTokenMatcher;

  float vMinScore = FLT_MAX;
  float vMaxScore = -FLT_MAX;

  float vUnknownTokenScorePenalty = 10.0;
  float vUnknownTokenScore;

  struct naive_trie vTokenMatcher;
};

struct llm_tokenizer_ugm_session {
  llm_tokenizer_ugm_session(const dgs_vocab & pVocab, const llm_tokenizer_ugm & pTokenizer) :
    vVocab(pVocab),
    vTokenizer(pTokenizer) {}

  /* This implementation is based on SentencePiece optimized Viterbi algorithm for
     * unigram language models. The general idea is to:
     * - move along the input sequence in steps of one UTF code point,
     * - at each step find all possible tokenizations of the prefix by
     *   traversing the tokens trie,
     * - for each tokenization store the best one so far (by higher score)
     * - use the position in sequence after given token as an index to store
     *   results
     * - if there was no valid tokenization of the current UTF code point
     *   then use unknown token with additional score penalty
     * After processing the whole sequence we backtrack from the end to get
     * the best tokenization.
    */
  void mTokenize(const std::string & pText, std::vector<dgs_token> & pOutput) {
    // get current size of output (for reversal later)
    size_t vOutputSize = pOutput.size();

    // normalize the input first
    std::string vNormalized;
    mNormalize(pText, &vNormalized);
    size_t vInputLen = vNormalized.size();
    if (vInputLen == 0) {
      return;
    }

    // initialize score_sum to -FLT_MAX so it will be always lower than sums of token scores
    std::vector<struct best_tokenization> lTokenizationResults(vInputLen + 1, { vVocab.mTokenUnk(), 0, -DBL_MAX });
    // at the beginning tokenization score is zero
    lTokenizationResults[0] = { vVocab.mTokenUnk(), 0, 0 };

    for (size_t vInputOffset = 0; vInputOffset < vInputLen;) {
      size_t vPrefixOffset   = vInputOffset;
      // calculate how many code units are in the currently processed UTF code point
      size_t vNUtf8CodeUnits = std::min<size_t>(fUnicodeLenUtf8(vNormalized[vInputOffset]), vInputLen - vInputOffset);

      // traverse the token matcher trie to find a matching token
      bool                             vSingleCodepointTokenFound = false;
      const struct best_tokenization & vCurrentBest               = lTokenizationResults[vInputOffset];
      const struct naive_trie *        node = vTokenizer.vTokenMatcher.mTraverse(vNormalized[vPrefixOffset++]);

      while (vPrefixOffset <= vInputLen && node != NULL) {
        // check if we found valid token in prefix
        if (node->has_value) {
          // check if it corresponds to the whole UTF code point
          if (vPrefixOffset - vInputOffset == vNUtf8CodeUnits) {
            vSingleCodepointTokenFound = true;
          }
          dgs_token    vTokenId   = node->value;
          const auto & vTokenData = vVocab.mGetTokenData(vTokenId);

          // we set the user-defined token scores to 0 to make them more likely to be selected
          // (normal token scores are log probabilities, so they are negative)
          // score type is double here to make tokenization results exactly
          // the same as in the HF tokenizer using SentencePiece
          const double               cTokenScore      = vVocab.mIsUserDefined(vTokenId) ? 0.0 : vTokenData.vScore;
          const double               cChallengerScore = vCurrentBest.vScoreSum + cTokenScore;
          struct best_tokenization & vCurrentChamp    = lTokenizationResults[vPrefixOffset];
          if (cChallengerScore > vCurrentChamp.vScoreSum) {
            struct best_tokenization vChallenger = { vTokenId, vInputOffset, cChallengerScore };
            vCurrentChamp                        = vChallenger;
          }
        }
        node = node->mTraverse(vNormalized[vPrefixOffset++]);
      }

      // if we didn't find a valid token corresponding to the whole UTF code point
      // then use unknown token as the tokenization of this UTF code point
      if (!vSingleCodepointTokenFound) {
        const double cChallengerScore            = vCurrentBest.vScoreSum + vTokenizer.vUnknownTokenScore;
        vPrefixOffset                            = vInputOffset + vNUtf8CodeUnits;
        struct best_tokenization & vCurrentChamp = lTokenizationResults[vPrefixOffset];
        if (cChallengerScore > vCurrentChamp.vScoreSum) {
          struct best_tokenization vChallenger = { vVocab.mTokenUnk(), vInputOffset, cChallengerScore };
          vCurrentChamp                        = vChallenger;
        }
      }

      // move to the next UTF code point
      vInputOffset += vNUtf8CodeUnits;
    }

    // now backtrack from the end to gather token ids of the best tokenization
    // merge sequences of consecutive unknown tokens into single unknown tokens
    bool vIsPrevUnknown = false;
    for (struct best_tokenization & vTokenization = lTokenizationResults[vInputLen];;
         vTokenization                            = lTokenizationResults[vTokenization.vInputOffset]) {
      bool vIsUnknown = vTokenization.vTokenId == vVocab.mTokenUnk();
      if (!(vIsPrevUnknown && vIsUnknown)) {
        pOutput.push_back(vTokenization.vTokenId);
      }
      if (vTokenization.vInputOffset == 0) {
        break;
      }
      vIsPrevUnknown = vIsUnknown;
    }

    // reverse the output since we added tokens starting from the end of the input
    std::reverse(pOutput.begin() + vOutputSize, pOutput.end());
  }

private:
  // helper structure for returning normalization results
  struct normalization_result {
    const char * vNormalized;
    size_t       vNormalizedLen;
    size_t       vConsumedInput;
  };

  void mNormalize(const std::string & pInput, std::string * pNormalized) {
    pNormalized->clear();
    pNormalized->reserve(pInput.size() * 3);

    const std::string cSpace = vVocab.mGetEscapeWhitespaces() ? vTokenizer.cEscapedSpace : " ";

    const bool cShallPrependSpace = !vVocab.mGetTreatWhitespaceAsSuffix() && vVocab.mGetAddSpacePrefix();
    const bool cShallAppendSpace  = vVocab.mGetTreatWhitespaceAsSuffix() && vVocab.mGetAddSpacePrefix();
    const bool cShallMergeSpaces  = vVocab.mGetRemoveExtraWhitespaces();

    bool vIsSpacePrepended = false;
    bool vProcessingNonWs  = false;

    size_t vInputLen = pInput.size();

    for (size_t vInputOffset = 0; vInputOffset < vInputLen;) {
      auto vNormRes = mNormalizePrefix(pInput, vInputOffset);
      for (size_t vI = 0; vI < vNormRes.vNormalizedLen; vI++) {
        char vC = vNormRes.vNormalized[vI];
        if (vC != ' ') {
          if (!vProcessingNonWs) {
            vProcessingNonWs = true;
            if ((cShallPrependSpace && !vIsSpacePrepended) || cShallMergeSpaces) {
              pNormalized->append(cSpace);
              vIsSpacePrepended = true;
            }
          }
          pNormalized->push_back(vC);
        } else {
          if (vProcessingNonWs) {
            vProcessingNonWs = false;
          }
          if (!cShallMergeSpaces) {
            pNormalized->append(cSpace);
          }
        }
      }

      vInputOffset += vNormRes.vConsumedInput;
    }

    if (cShallAppendSpace) {
      pNormalized->append(cSpace);
    }
  }

  /*
     * This structure is a view wrapper for XOR-compressed double array (XCDA)
     * See Shunsuke Kanda (2018). Space- and Time-Efficient String Dictionaries.
     * Each bit-packed entry contains:
     * - BASE array value in bits 10-30
     * - LCHECK array value in bits 0-7
     * - LEAF array value in bit 9
     * Entries containing indexes of replacement sequences have set bit 31
     */
  struct xcda_array_view {
  public:
    xcda_array_view(const uint32_t * pXcdaArray, size_t pXcdaArraySize) :
      vXcdaArray(pXcdaArray),
      vXcdaArraySize(pXcdaArraySize) {}

    uint32_t mGetBase(size_t pIndex) {
      uint32_t vPackedNode = mGetNode(pIndex);
      return (vPackedNode >> 10) << ((vPackedNode & (1U << 9)) >> 6);
    }

    uint32_t mGetLcheck(size_t pIndex) {
      uint32_t vPackedNode = mGetNode(pIndex);
      return vPackedNode & ((1U << 31) | 0xff);
    }

    bool mGetLeaf(size_t pIndex) {
      uint32_t vPackedNode = mGetNode(pIndex);
      return (vPackedNode >> 8) & 1;
    }

    uint32_t mGetValue(size_t pIndex) {
      uint32_t vPackedNode = mGetNode(pIndex);
      return vPackedNode & ((1U << 31) - 1);
    }
  private:
    uint32_t mGetNode(size_t pIndex) {
      if (pIndex >= vXcdaArraySize) {
        throw std::runtime_error("Index out of array bounds in XCDA array!");
      }
      return vXcdaArray[pIndex];
    }

    const uint32_t * vXcdaArray;
    size_t           vXcdaArraySize;
  };

  // this structure stores the best tokenization so far at input_offset
  struct best_tokenization {
    dgs_token vTokenId;
    size_t    vInputOffset;
    double    vScoreSum;
  };

  struct normalization_result mNormalizePrefix(const std::string & pInput, size_t pInputOffset) {
    if (pInputOffset == pInput.size()) {
      return { &pInput[pInputOffset], 0, 0 };
    }

    // if input prefix matches some user-defined token return this token as normalization result
    auto vUserDefinedTokenMatch =
      vTokenizer.vUserDefinedTokenMatcher.mGetLongestPrefix(&pInput[pInputOffset], pInput.size() - pInputOffset);
    if (vUserDefinedTokenMatch.second > 0) {
      return { &pInput[pInputOffset], vUserDefinedTokenMatch.second, vUserDefinedTokenMatch.second };
    }

    size_t vLongestPrefixLength = 0;
    size_t vLongestPrefixOffset = 0;

    if (vTokenizer.vXcdaArraySize > 0) {
      struct xcda_array_view vXcdaView(vTokenizer.vXcdaArray, vTokenizer.vXcdaArraySize);

      // Find the longest normalized sequence matching the input prefix by walking
      // the XOR-compressed compact double array (XCDA) starting from the root node
      // We find the index of the next node by calculating BASE[s] ^ c where s is
      // the index of the previous node and c is a numerical character value
      uint32_t vNodeIndex = 0;
      // get BASE of the root node
      vNodeIndex          = vXcdaView.mGetBase(vNodeIndex);
      for (size_t vPrefixOffset = pInputOffset; vPrefixOffset < pInput.size(); vPrefixOffset++) {
        unsigned char vC = pInput[vPrefixOffset];
        if (vC == 0) {
          break;
        }
        vNodeIndex ^= vC;
        // if value of LCHECK is not c it means that this is not a child of
        // the previous node, so we stop matching
        if (vXcdaView.mGetLcheck(vNodeIndex) != vC) {
          break;
        }
        bool vIsLeaf = vXcdaView.mGetLeaf(vNodeIndex);
        // get BASE of the current node
        vNodeIndex ^= vXcdaView.mGetBase(vNodeIndex);
        // if LEAF of the current node is true, it means that its BASE points to the node
        // containing index of replacement sequence for currently matched input prefix
        if (vIsLeaf) {
          vLongestPrefixLength = vPrefixOffset - pInputOffset + 1;
          // get index of replacement sequence for currently matched input prefix
          vLongestPrefixOffset = vXcdaView.mGetValue(vNodeIndex);
        }
      }
    }

    if (vLongestPrefixLength > 0) {
      // we have a match, so return the replacement sequence
      if (vLongestPrefixOffset >= vTokenizer.vPrefixReplacementsSize) {
        throw std::runtime_error("Index out of array bounds in precompiled charsmap!");
      }
      const char * vPrefixReplacement = &(vTokenizer.vPrefixReplacements)[vLongestPrefixOffset];
      size_t       vMaxLen            = vTokenizer.vPrefixReplacementsSize - vLongestPrefixOffset;
      size_t       vReplLen           = 0;
      while (vReplLen < vMaxLen && vPrefixReplacement[vReplLen] != '\0') {
        vReplLen++;
      }
      if (vReplLen == vMaxLen) {
        throw std::runtime_error("Unterminated string in precompiled charsmap!");
      }
      return { vPrefixReplacement, vReplLen, vLongestPrefixLength };
    }

    // check if the input prefix contains a valid sequence of UTF-8 code units
    try {
      // if yes, return this sequence unmodified
      size_t vPrefixOffset = pInputOffset;
      fUnicodeCptFromUtf8(pInput, vPrefixOffset);
      return { &pInput[pInputOffset], vPrefixOffset - pInputOffset, vPrefixOffset - pInputOffset };
    } catch (std::invalid_argument & /*ex*/) {
      // if no, consume 1 byte and return U+FFFD - REPLACEMENT CHARACTER
      return { "\xEF\xBF\xBD", 3, 1 };
    }
  }

  const dgs_vocab &         vVocab;
  const llm_tokenizer_ugm & vTokenizer;
};

//
// RWKV tokenizer
//

static std::vector<uint8_t> fDgsUnescapeRwkvToken(const std::string & pEscaped) {
  std::vector<uint8_t> lOutput;
  lOutput.reserve(pEscaped.size());

  // Parser state
  bool    vEscaping     = false;
  uint8_t vHexRemaining = 0;
  uint8_t vHexAcc       = 0;

  // Step through characters, performing parsing
  for (const char & vC : pEscaped) {
    // If we're parsing a hex code, interpret the next character
    if (vHexRemaining != 0) {
      uint8_t value = (vC >= 'a') ? (vC - 'a' + 10) : (vC - '0');
      vHexAcc       = (vHexAcc << 4) + value;

      vHexRemaining -= 1;
      if (vHexRemaining == 0) {
        lOutput.push_back(vHexAcc);
        vHexAcc = 0;
      }

      continue;
    }

    // If we got an escape character, interpret it
    if (vEscaping) {
      if (vC == 't') {
        lOutput.push_back('\t');
      } else if (vC == 'n') {
        lOutput.push_back('\n');
      } else if (vC == 'r') {
        lOutput.push_back('\r');
      } else if (vC == 'x') {
        vHexRemaining = 2;
      } else {
        lOutput.push_back(vC);
      }

      vEscaping = false;
      continue;
    }

    if (vC == '\\') {
      vEscaping = true;
      continue;
    }

    lOutput.push_back(vC);
  }

  return lOutput;
}

struct llm_tokenizer_rwkv : llm_tokenizer {
  llm_tokenizer_rwkv(const dgs_vocab & pVocab) {
    // RWKV supports arbitrary byte tokens, but the vocab struct only supports string tokens.
    // For now, we decode the vocab here into the lookup we'll use for tokenization.

    // build trie
    for (uint32_t vId = 0; vId < pVocab.n_tokens(); ++vId) {
      const auto & data  = pVocab.mGetTokenData(vId);
      const auto   lText = fDgsUnescapeRwkvToken(data.vText);
      vTokenMatcher.insert((const char *) lText.data(), lText.size(), vId);
    }
  }

  struct naive_trie vTokenMatcher;
};

struct llm_tokenizer_rwkv_session {
  llm_tokenizer_rwkv_session(const dgs_vocab & pVocab, const llm_tokenizer_rwkv & pTokenizer) :
    vVocab(pVocab),
    vTokenizer(pTokenizer) {}

  void mTokenize(const std::string & pText, std::vector<dgs_token> & pOutput) {
    uint32_t vPosition = 0;
    while (vPosition < pText.size()) {
      const struct naive_trie * node = vTokenizer.vTokenMatcher.mTraverse(pText[vPosition]);
      if (node == NULL) {
        // no matching token found, add unknown token
        pOutput.push_back(vVocab.mTokenUnk());
        vPosition += 1;
        continue;
      }

      // traverse the trie to find the longest matching token
      uint32_t vTokenId     = 0;
      uint32_t vTokenLength = 0;
      while (node != NULL) {
        if (node->has_value) {
          vTokenId     = node->value;
          vTokenLength = vPosition + 1;
        }
        if (vPosition + 1 >= pText.size()) {
          break;
        }
        node = node->mTraverse(pText[++vPosition]);
      }

      // add the longest matching token
      pOutput.push_back(vTokenId);
      vPosition = vTokenLength;
    }
  }

private:
  const dgs_vocab &          vVocab;
  const llm_tokenizer_rwkv & vTokenizer;
};

struct llm_tokenizer_plamo2 : llm_tokenizer {
  llm_tokenizer_plamo2(const dgs_vocab & pVocab, bool pPreSegment) : cPreSegment(pPreSegment) { mBuild(pVocab); }

  void mBuild(const dgs_vocab & pVocab) {
    // Reset internal structures
    lTokens.clear();
    lBytes.assign(256, 0);
    dToSuffixId.clear();
    lTable.clear();

    // Build token list and byte mapping
    std::unordered_map<std::string, float>     dSuffixToScore;
    std::unordered_map<std::string, dgs_token> dTokenToId;

    for (size_t vTokenId = 0; vTokenId < pVocab.n_tokens(); ++vTokenId) {
      const auto & vEntry = pVocab.mGetTokenData(vTokenId);
      lTokens.push_back(vEntry.vText);
      dTokenToId[vEntry.vText] = static_cast<dgs_token>(vTokenId);

      // Handle byte tokens
      if (pVocab.mIsByte(vTokenId)) {
        if (vEntry.vText.length() == 6 && vEntry.vText.substr(0, 3) == "<0x" && vEntry.vText.back() == '>') {
          std::string vHexStr = vEntry.vText.substr(3, 2);
          if (std::isxdigit(static_cast<unsigned char>(vHexStr[0])) &&
              std::isxdigit(static_cast<unsigned char>(vHexStr[1]))) {
            int vByteVal     = std::stoi(vHexStr, nullptr, 16);
            lBytes[vByteVal] = static_cast<dgs_token>(vTokenId);
          }
        }
        continue;
      }

      // Add token and all its suffixes to suffix_to_score
      dSuffixToScore[vEntry.vText] = vEntry.vScore;

      // Extract suffixes character by character (UTF-8 aware)
      std::vector<uint32_t> lCpts = fUnicodeCptsFromUtf8(vEntry.vText);
      for (size_t vI = 1; vI < lCpts.size(); ++vI) {
        std::string vSuffix;
        for (size_t vJ = vI; vJ < lCpts.size(); ++vJ) {
          vSuffix += fUnicodeCptToUtf8(lCpts[vJ]);
        }
        if (dSuffixToScore.find(vSuffix) == dSuffixToScore.end()) {
          dSuffixToScore[vSuffix] = std::numeric_limits<float>::quiet_NaN();
        }
      }
    }

    // Check that all byte tokens are set
    for (int vI = 0; vI < 256; ++vI) {
      if (lBytes[vI] == 0) {
        throw std::runtime_error("Byte token for <0x" + std::to_string(vI) + "> is not set");
      }
    }

    // Build suffix list in lexicographical order of reversed strings
    std::vector<std::string> lSuffixes;
    lSuffixes.reserve(dSuffixToScore.size() + 1);
    for (const auto & vPair : dSuffixToScore) {
      lSuffixes.push_back(vPair.first);
    }
    lSuffixes.push_back("");  // Empty suffix

    std::sort(lSuffixes.begin(), lSuffixes.end(), [](const std::string & pA, const std::string & pB) {
      std::string vRevA(pA.rbegin(), pA.rend());
      std::string vRevB(pB.rbegin(), pB.rend());
      return vRevA < vRevB;
    });

    // Build suffix_to_id and to_suffix_id_
    std::unordered_map<std::string, int32_t> dSuffixToId;
    int32_t                                  vNumPieces = 0;

    for (const auto & vSuffix : lSuffixes) {
      dSuffixToId[vSuffix] = vNumPieces;
      if (!vSuffix.empty()) {
        std::vector<uint32_t> lCpts = fUnicodeCptsFromUtf8(vSuffix);

        std::string vRemaining;
        for (size_t vI = 1; vI < lCpts.size(); ++vI) {
          vRemaining += fUnicodeCptToUtf8(lCpts[vI]);
        }

        int64_t vPieceCode      = (static_cast<int64_t>(lCpts[0]) << 32) | dSuffixToId[vRemaining];
        dToSuffixId[vPieceCode] = vNumPieces;

        // Count number of pieces for this suffix
        int32_t vPiecesForSuffix = 1;  // sentinel row
        for (int32_t vPieceLength = static_cast<int32_t>(lCpts.size()); vPieceLength > 0; --vPieceLength) {
          std::string vPiece;
          for (int32_t vI = 0; vI < vPieceLength; ++vI) {
            vPiece += fUnicodeCptToUtf8(lCpts[vI]);
          }
          if (dSuffixToScore.find(vPiece) != dSuffixToScore.end()) {
            vPiecesForSuffix++;
          }
        }
        vNumPieces += vPiecesForSuffix;
      } else {
        vNumPieces++;  // Empty suffix contributes one piece (sentinel row)
      }
    }

    // Build flattened table
    lTable.resize(vNumPieces, std::vector<int32_t>(4, 0));
    int32_t vTableIdx = 0;

    for (const auto & vSuffix : lSuffixes) {
      // Add all prefixes of the suffix to the table (in decreasing order of length)
      std::vector<uint32_t> lCpts = fUnicodeCptsFromUtf8(vSuffix);
      for (int32_t vPieceLength = static_cast<int32_t>(lCpts.size()); vPieceLength > 0; --vPieceLength) {
        std::string vPiece;
        for (int32_t vI = 0; vI < vPieceLength; ++vI) {
          vPiece += fUnicodeCptToUtf8(lCpts[vI]);
        }

        auto vScoreIt = dSuffixToScore.find(vPiece);
        if (vScoreIt == dSuffixToScore.end()) {
          continue;
        }

        lTable[vTableIdx][cTablePieceLength] = vPieceLength;
        auto vTokenIt                        = dTokenToId.find(vPiece);
        lTable[vTableIdx][cTableTokenId]     = (vTokenIt != dTokenToId.end()) ? vTokenIt->second : -1;

        float vScore = vScoreIt->second;
        lTable[vTableIdx][cTableScore] =
          std::isfinite(vScore) ? static_cast<int32_t>(std::round(vScore * 1e4)) : cInvalidScore;
        lTable[vTableIdx][cTablePieceId] = dSuffixToId[vPiece];

        vTableIdx++;
      }

      // Add sentinel row
      lTable[vTableIdx][cTablePieceLength] = 1;
      lTable[vTableIdx][cTableTokenId]     = -1;
      lTable[vTableIdx][cTableScore]       = cUnknownScore;
      vTableIdx++;
    }
  }

  std::vector<dgs_token> mEncode(const std::string & pText) const {
    std::vector<uint32_t> lUnicodeData = fUnicodeCptsFromUtf8(pText);
    // The PLaMo-3 tokenizer keeps a leading U+FEFF in the input.
    if (!cPreSegment) {
      if (!lUnicodeData.empty() && lUnicodeData[0] == 0xFEFF) {
        lUnicodeData.erase(lUnicodeData.begin());
      }
      return mEncodeCpts(lUnicodeData);
    }

    const size_t      n = lUnicodeData.size();
    std::vector<bool> lCut(n + 1, false);

    // pass 1: <|plamo:...|>
    {
      static constexpr uint32_t aPrefix[]  = { '<', '|', 'p', 'l', 'a', 'm', 'o', ':' };
      const size_t              cPrefixLen = std::size(aPrefix);
      size_t                    vI         = 0;
      while (vI + cPrefixLen <= n) {
        if (!std::equal(aPrefix, aPrefix + cPrefixLen, lUnicodeData.begin() + vI)) {
          vI++;
          continue;
        }
        // An empty body is valid.
        size_t vJ = vI + cPrefixLen;
        // Treat U+001C..U+001F as whitespace (equivalent to Python \s).
        while (vJ < n && vJ - (vI + cPrefixLen) < 64 && lUnicodeData[vJ] != '|' &&
               (lUnicodeData[vJ] < 0x1C || lUnicodeData[vJ] > 0x1F) &&
               !fUnicodeCptFlagsFromCpt(lUnicodeData[vJ]).vIsWhitespace) {
          vJ++;
        }
        if (vJ + 1 < n && lUnicodeData[vJ] == '|' && lUnicodeData[vJ + 1] == '>') {
          lCut[vI]     = true;
          lCut[vJ + 2] = true;
          vI           = vJ + 2;
        } else {
          vI++;
        }
      }
    }

    // pass 2: runs of repeated characters / spaces (a run never crosses a boundary from pass 1)
    {
      size_t vI = 0;
      while (vI < n) {
        const uint32_t cC   = lUnicodeData[vI];
        size_t         vRun = 1;
        while (vI + vRun < n && lUnicodeData[vI + vRun] == cC && !lCut[vI + vRun]) {
          vRun++;
        }

        const bool cIsRepeatedChars = cC != '\n' && vRun >= 4;
        const bool cIsSpaces        = cC == ' ' && vRun >= 2;
        if (cIsRepeatedChars || cIsSpaces) {
          lCut[vI]        = true;
          lCut[vI + vRun] = true;
        }

        // a run that does not match cannot match at any later position either (it only gets shorter)
        vI += vRun;
      }
    }

    std::vector<dgs_token> lOutput;
    size_t                 vSegStart = 0;
    for (size_t vSegEnd = 0; vSegEnd <= n; ++vSegEnd) {
      // U+EE00 is the tokenizer's private-use boundary marker; literal occurrences split segments and are not emitted.
      const bool cIsBoundary = vSegEnd < n && lUnicodeData[vSegEnd] == 0xEE00;
      if (vSegEnd == n || lCut[vSegEnd] || cIsBoundary) {
        if (vSegStart < vSegEnd) {
          const std::vector<uint32_t>  lSegment(lUnicodeData.begin() + vSegStart, lUnicodeData.begin() + vSegEnd);
          const std::vector<dgs_token> lTokens = mEncodeCpts(lSegment);
          lOutput.insert(lOutput.end(), lTokens.begin(), lTokens.end());
        }
        vSegStart = vSegEnd + cIsBoundary;
      }
    }

    return lOutput;
  }

  std::vector<dgs_token> mEncodeCpts(const std::vector<uint32_t> & pUnicodeData) const {
    if (pUnicodeData.empty()) {
      return {};
    }

    const size_t cDataLen = pUnicodeData.size();

    // Initialize scores array (dynamic programming)
    std::vector<int64_t> lScores(cDataLen + 1, static_cast<int64_t>(1) << 60);
    lScores[cDataLen] = 0;

    // Path array to track best tokenization
    std::vector<std::vector<int32_t>> lPath(cDataLen + 1, std::vector<int32_t>(3, 0));

    int32_t vSuffixId = 0;

    // Process from end to beginning
    for (int vI = static_cast<int>(cDataLen) - 1; vI >= 0; --vI) {
      uint32_t vC = pUnicodeData[vI];

      // Find next suffix ID
      for (size_t vP = vSuffixId; vP < lTable.size(); ++vP) {
        int64_t vPieceCode = (static_cast<int64_t>(vC) << 32) | lTable[vP][cTablePieceId];
        auto    vIt        = dToSuffixId.find(vPieceCode);
        vSuffixId          = (vIt != dToSuffixId.end()) ? vIt->second : 0;

        if (vSuffixId > 0 || lTable[vP][cTableScore] == cUnknownScore) {
          break;
        }
      }

      // Update best path
      for (size_t vP = vSuffixId; vP < lTable.size(); ++vP) {
        int32_t vScore = lTable[vP][cTableScore];
        if (vScore > cInvalidScore) {
          int32_t vPieceLength = lTable[vP][cTablePieceLength];
          int64_t vS           = lScores[vI + vPieceLength] - vScore;

          if (vS < lScores[vI]) {
            lScores[vI]                 = vS;
            lPath[vI][cPathTokenLength] = vPieceLength;
            lPath[vI][cPathTokenId]     = lTable[vP][cTableTokenId];
            lPath[vI][cPathNumTokens]   = lPath[vI + vPieceLength][cPathNumTokens] + 1;

            if (vScore == cUnknownScore) {
              // Add UTF-8 byte count
              lPath[vI][cPathNumTokens] += (vC >= 0x80) + (vC >= 0x800) + (vC >= 0x10000);
            }
          }
        }

        if (vScore == cUnknownScore) {
          break;
        }
      }
    }

    // Decode the best path
    std::vector<dgs_token> lTokenIds;
    lTokenIds.reserve(lPath[0][cPathNumTokens]);

    int vPos = 0;
    while (vPos < static_cast<int>(cDataLen)) {
      if (lPath[vPos][cPathTokenId] >= 0) {
        lTokenIds.push_back(lPath[vPos][cPathTokenId]);
      } else {
        // Fall back to byte tokens
        uint32_t vC = pUnicodeData[vPos];
        int      vS = 1 + (vC >= 0x80) + (vC >= 0x800) + (vC >= 0x10000);

        for (int vI = 0; vI < vS; ++vI) {
          uint8_t vB;
          if (vS == 1) {
            vB = vC;
          } else {
            if (vI == 0) {
              vB = (0xF00 >> vS) & 0xFF;
            } else {
              vB = 0x80;
            }
          }
          lTokenIds.push_back(lBytes[vB | ((vC >> ((vS - vI - 1) * 6)) & 0x3F)]);
        }
      }

      assert(path[pos][PATH_TOKEN_LENGTH] > 0);
      vPos += lPath[vPos][cPathTokenLength];
    }

    return lTokenIds;
  }
private:
  // Constants for table structure
  static constexpr int32_t cTablePieceLength = 0;
  static constexpr int32_t cTableTokenId     = 1;
  static constexpr int32_t cTableScore       = 2;
  static constexpr int32_t cTablePieceId     = 3;

  // Constants for path array
  static constexpr int32_t cPathTokenLength = 0;
  static constexpr int32_t cPathTokenId     = 1;
  static constexpr int32_t cPathNumTokens   = 2;

  // Score constants
  static constexpr int32_t cInvalidScore = -20000000;
  static constexpr int32_t cUnknownScore = -10000000;

  // List of tokens in the vocabulary
  std::vector<std::string> lTokens;

  // Mapping from byte code point to token ID (for byte fallback)
  std::vector<dgs_token> lBytes;

  // Mapping from piece code to suffix ID
  std::unordered_map<int64_t, int32_t> dToSuffixId;

  // Flattened table representing the Trie structure
  // Each row contains: [piece_length, token_id, score, piece_id]
  std::vector<std::vector<int32_t>> lTable;

  const bool cPreSegment;
};

struct llm_tokenizer_plamo2_session {
  llm_tokenizer_plamo2_session(const llm_tokenizer_plamo2 & pTokenizer) : vTokenizer(pTokenizer) {}

  void mTokenize(const std::string & pText, std::vector<dgs_token> & pOutput) {
    std::vector<dgs_token> lTokens = vTokenizer.mEncode(pText);
    pOutput.insert(pOutput.end(), lTokens.begin(), lTokens.end());
  }

private:
  const llm_tokenizer_plamo2 & vTokenizer;
};

// reserved suffix (U+E000) that keeps DNA k-mers distinct from identical
// base-vocab BPE tokens (e.g. CCCCCC) in token_to_id; erased from id_to_token
// text at load
static const std::string cDnaKmerMarker = "\xee\x80\x80";

struct llm_tokenizer_hybriddna_session : llm_tokenizer_bpe_session {
  llm_tokenizer_hybriddna_session(const dgs_vocab & pVocab, const llm_tokenizer_bpe & pTokenizer) :
    llm_tokenizer_bpe_session{ pVocab, pTokenizer },
    vVocab{ pVocab } {}

  void mTokenize(const std::string & pText, std::vector<dgs_token> & pOutput) override {
    static const std::string cOpenTag  = "<dna>";
    static const std::string cCloseTag = "</dna>";

    const auto cDnaBeginId = vVocab.mTextToToken(cOpenTag);
    const auto cDnaEndId   = vVocab.mTextToToken(cCloseTag);
    const auto cDnaOovId   = vVocab.mTextToToken("<oov>");

    // Fall back to plain BPE if the DNA pieces aren't in the vocab.
    if (cDnaBeginId == DGS_TOKEN_NULL || cDnaEndId == DGS_TOKEN_NULL || cDnaOovId == DGS_TOKEN_NULL) {
      llm_tokenizer_bpe_session::mTokenize(pText, pOutput);
      return;
    }

    const size_t cK   = 6;
    size_t       vPos = 0;

    while (vPos < pText.size()) {
      const size_t cStart = pText.find(cOpenTag, vPos);
      if (cStart == std::string::npos) {
        if (vPos < pText.size()) {
          llm_tokenizer_bpe_session::mTokenize(pText.substr(vPos), pOutput);
        }
        break;
      }
      if (cStart > vPos) {
        llm_tokenizer_bpe_session::mTokenize(pText.substr(vPos, cStart - vPos), pOutput);
      }
      pOutput.push_back(cDnaBeginId);

      const size_t cContentStart = cStart + cOpenTag.size();
      const size_t end           = pText.find(cCloseTag, cContentStart);
      const size_t cContentEnd   = (end == std::string::npos) ? pText.size() : end;

      mEmitDnaKmers(pText.substr(cContentStart, cContentEnd - cContentStart), cK, cDnaOovId, pOutput);

      if (end == std::string::npos) {
        break;
      }
      pOutput.push_back(cDnaEndId);
      vPos = end + cCloseTag.size();
    }
  }

private:
  void mEmitDnaKmers(const std::string & pRaw, size_t pK, dgs_token pOovId, std::vector<dgs_token> & pOutput) {
    std::string vSeq = pRaw;
    for (char & vC : vSeq) {
      if (vC >= 'a' && vC <= 'z') {
        vC = char(vC - 32);
      }
    }

    // k-mers carry the reserved marker suffix; a non-ACGT k-mer simply
    // isn't in the vocab and falls back to <oov>
    auto vKmerToken = [&](const std::string & pKmer) {
      const auto cTok = vVocab.mTextToToken(pKmer + cDnaKmerMarker);
      return cTok != DGS_TOKEN_NULL ? cTok : pOovId;
    };

    size_t vI = 0;
    for (; vI + pK <= vSeq.size(); vI += pK) {
      pOutput.push_back(vKmerToken(vSeq.substr(vI, pK)));
    }
    if (vI < vSeq.size()) {
      std::string vKmer = vSeq.substr(vI);
      vKmer.append(pK - vKmer.size(), 'A');
      pOutput.push_back(vKmerToken(vKmer));
    }
  }

  const dgs_vocab & vVocab;
};

struct llm_tokenizer_whitespace_session : llm_tokenizer_bpe_session {
  llm_tokenizer_whitespace_session(const dgs_vocab & pVocab, const llm_tokenizer_bpe & pTokenizer) :
    llm_tokenizer_bpe_session{ pVocab, pTokenizer },
    vVocab{ pVocab } {}

  void mTokenize(const std::string & pText, std::vector<dgs_token> & pOutput) override {
    const bool cLowercase = vVocab.mGetNormalizerOpts().vLowercase;

    std::string vSegment;
    auto        vFlush = [&]() {
      if (!vSegment.empty()) {
        llm_tokenizer_bpe_session::mTokenize(vSegment, pOutput);
        vSegment.clear();
      }
    };

    for (uint32_t vCpt : fUnicodeCptsFromUtf8(pText)) {
      // drop whitespace
      if (fUnicodeCptFlagsFromCpt(vCpt).vIsWhitespace) {
        vFlush();
      } else {
        vSegment += fUnicodeCptToUtf8(cLowercase ? fUnicodeTolower(vCpt) : vCpt);
      }
    }
    vFlush();
  }

private:
  const dgs_vocab & vVocab;
};

//
// impl
//

typedef enum FRAGMENT_BUFFER_VARIANT_TYPE {
  cFragmentBufferVariantTypeToken,
  cFragmentBufferVariantTypeRawText
} FRAGMENT_BUFFER_VARIANT_TYPE;

struct fragment_buffer_variant {
  fragment_buffer_variant(dgs_token pToken) :
    type(cFragmentBufferVariantTypeToken),
    token(pToken),
    vRawText(cDummy),
    cOffset(0),
    length(0) {}

  fragment_buffer_variant(const std::string & pRawText, int64_t pOffset, int64_t pLength) :
    type(cFragmentBufferVariantTypeRawText),
    token((dgs_token) -1),
    vRawText(pRawText),
    cOffset(pOffset),
    length(pLength) {
    GGML_ASSERT(pOffset >= 0);
    GGML_ASSERT(pLength >= 1);
    GGML_ASSERT(cOffset + length <= vRawText.length());
  }

  const FRAGMENT_BUFFER_VARIANT_TYPE type;
  const dgs_token                    token;
  const std::string                  cDummy;
  const std::string &                vRawText;
  const uint64_t                     cOffset;
  const uint64_t                     length;
};

struct dgs_vocab::impl {
  uint32_t vNTokenTypes = 0;  // for BERT-style token types

  std::string vTokenizerModel;
  std::string vTokenizerPre;

  enum fDgsVocabType      type     = cDgsVocabTypeSpm;
  enum dgs_vocab_pre_type vPreType = cDgsVocabPreTypeDefault;

  int vMaxTokenLen = 0;  // used for optimizing longest token search

  // default LLaMA special tokens
  // TODO: should we set all of these to DGS_TOKEN_NULL?
  dgs_token vSpecialBosId  = 1;
  dgs_token vSpecialEosId  = 2;
  dgs_token vSpecialEotId  = DGS_TOKEN_NULL;
  dgs_token vSpecialEomId  = DGS_TOKEN_NULL;
  dgs_token vSpecialUnkId  = 0;
  dgs_token vSpecialSepId  = DGS_TOKEN_NULL;
  dgs_token vSpecialPadId  = DGS_TOKEN_NULL;
  dgs_token vSpecialMaskId = DGS_TOKEN_NULL;

  dgs_token vLinefeedId = 13;

  // fim tokens
  dgs_token vSpecialFimPreId = DGS_TOKEN_NULL;
  dgs_token vSpecialFimSufId = DGS_TOKEN_NULL;
  dgs_token vSpecialFimMidId = DGS_TOKEN_NULL;
  dgs_token vSpecialFimPadId = DGS_TOKEN_NULL;
  dgs_token vSpecialFimRepId = DGS_TOKEN_NULL;  // repo
  dgs_token vSpecialFimSepId = DGS_TOKEN_NULL;  // file separator

  // tokenizer flags
  bool vAddSpacePrefix          = false;
  bool vAddBos                  = false;
  bool vAddEos                  = false;
  bool vAddSep                  = false;
  bool vIgnoreMerges            = false;
  bool vCleanSpaces             = false;  // clean_up_tokenization_spaces
  bool vRemoveExtraWhitespaces  = false;
  bool vEscapeWhitespaces       = true;
  bool vTreatWhitespaceAsSuffix = false;

  // BertNormalizer options
  dgs_vocab::normalizer_options vNormalizerOpts;

  std::unordered_map<std::string, dgs_token> dTokenToId;
  std::vector<token_data>                    lIdToToken;

  std::vector<dgs_token>   lCacheSpecialTokens;
  std::vector<std::string> lCacheTokenToPiece;  // dgs_token_to_piece(special = true);

  struct pair_hash {
    size_t operator()(const std::pair<std::string, std::string> & pP) const {
      return std::hash<std::string>{}(pP.first) ^  //create some hash for pair
             (std::hash<std::string>{}(pP.second) << 1);
    }
  };

  std::unordered_map<std::pair<std::string, std::string>, int, pair_hash> dBpeRanks;

  // set of all tokens that cause "end of generation"
  std::set<dgs_token> sSpecialEogIds;

  std::vector<dgs_token> lSuppressTokens;

  std::unique_ptr<llm_tokenizer> vTokenizer;

  std::vector<char> lPrecompiledCharsmap;

  impl(const dgs_vocab & pVocab) : vVocab(pVocab) {}

  ~impl() = default;

  void mLoad(dgs_model_loader & pMl, const LLM_KV & pKv);

  enum fDgsVocabType mGetType() const;

  std::string mTypeName() const;

  bool mIsNormal(dgs_token pId) const;
  bool mIsUnknown(dgs_token pId) const;
  bool mIsControl(dgs_token pId) const;
  bool mIsByte(dgs_token pId) const;
  bool mIsUserDefined(dgs_token pId) const;
  bool mIsUnused(dgs_token pId) const;
  bool mIsEog(dgs_token pId) const;

  uint8_t mTokenToByte(dgs_token pId) const;

  dgs_token_attr mTokenGetAttr(dgs_token pId) const;

  void mInitTokenizer(enum fDgsVocabType type);

  void mTokenizerStPartition(std::forward_list<fragment_buffer_variant> & pBuffer, bool pParseSpecial) const;

  std::string mTokenToPieceForCache(dgs_token token, bool pSpecial) const;

  std::vector<dgs_token> mTokenize(const std::string & pRawText, bool pAddSpecial, bool pParseSpecial = false) const;

  int32_t mTokenize(const char * pText,
                    int32_t      pTextLen,
                    dgs_token *  pTokens,
                    int32_t      pNTokensMax,
                    bool         pAddSpecial,
                    bool         pParseSpecial) const;

  // does not write null-terminator to buf
  int32_t mTokenToPiece(dgs_token token, char * pBuf, int32_t length, int32_t pLstrip, bool pSpecial) const;

  // use cached data
  const std::string & mTokenToPiece(dgs_token token) const;

  int32_t mDetokenize(const dgs_token * pTokens,
                      int32_t           pNTokens,
                      char *            pText,
                      int32_t           pTextLenMax,
                      bool              pRemoveSpecial,
                      bool              pUnparseSpecial) const;

  std::string mDetokenize(const std::vector<dgs_token> & pTokens, bool pSpecial) const;

  void mPrintInfo() const;

private:
  const dgs_vocab & vVocab;
};

void dgs_vocab::impl::mLoad(dgs_model_loader & pMl, const LLM_KV & pKv) {
  struct gguf_context * vCtx = pMl.vMetadata;

  // determine vocab type
  {
    pMl.get_key(cLlmKvTokenizerModel, vTokenizerModel);
    pMl.get_key(cLlmKvTokenizerPre, vTokenizerPre, false);

    pMl.get_key(cLlmKvTokenizerTokenTypeCount, vNTokenTypes, false);

    if (vTokenizerModel == "no_vocab" || vTokenizerModel == "none") {
      type = cDgsVocabTypeNone;

      // default special tokens
      vSpecialBosId  = DGS_TOKEN_NULL;
      vSpecialEosId  = DGS_TOKEN_NULL;
      vSpecialUnkId  = DGS_TOKEN_NULL;
      vSpecialSepId  = DGS_TOKEN_NULL;
      vSpecialPadId  = DGS_TOKEN_NULL;
      vSpecialMaskId = DGS_TOKEN_NULL;
      vLinefeedId    = DGS_TOKEN_NULL;

      // read vocab size from metadata
      uint32_t vNTokens = 0;
      if (pMl.get_key(cLlmKvVocabSize, vNTokens, false)) {
        DGS_LOG_WARN("%s: adding %u dummy tokens\n", __func__, vNTokens);
        lIdToToken.resize(vNTokens);
      }

      return;
    }

    if (vTokenizerModel == "llama") {
      type = cDgsVocabTypeSpm;

      // default special tokens
      vSpecialBosId  = 1;
      vSpecialEosId  = 2;
      vSpecialUnkId  = 0;
      vSpecialSepId  = DGS_TOKEN_NULL;
      vSpecialPadId  = DGS_TOKEN_NULL;
      vSpecialMaskId = DGS_TOKEN_NULL;
    } else if (vTokenizerModel == "bert") {
      type = cDgsVocabTypeWpm;

      // default special tokens
      vSpecialBosId  = 101;
      vSpecialEosId  = DGS_TOKEN_NULL;
      vSpecialUnkId  = 100;
      vSpecialSepId  = 102;
      vSpecialPadId  = 0;
      vSpecialMaskId = 103;

      vAddSep = true;
    } else if (vTokenizerModel == "gpt2" || vTokenizerModel == "hybriddna" || vTokenizerModel == "whitespace") {
      type = cDgsVocabTypeBpe;

      // read bpe merges and populate bpe ranks
      const int  cMergesKeyidx = gguf_find_key(vCtx, pKv(cLlmKvTokenizerMerges).c_str());
      // Kimi-K2 uses custom tokenization without traditional BPE merges
      const bool cIsKimiK2     = (vTokenizerPre == "kimi-k2");

      if (cMergesKeyidx == -1) {
        if (!cIsKimiK2) {
          throw std::runtime_error("cannot find tokenizer merges in model file\n");
        }
        // Kimi-K2 doesn't need merges, skip
        DGS_LOG_INFO("%s: Kimi-K2 tokenizer detected, skipping BPE merges\n", __func__);
      } else {
        if (gguf_get_kv_type(vCtx, cMergesKeyidx) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(vCtx, cMergesKeyidx) != GGUF_TYPE_STRING) {
          throw std::runtime_error(format("invalid gguf type for %s", pKv(cLlmKvTokenizerMerges).c_str()));
        }
        const int cNMerges = gguf_get_arr_n(vCtx, cMergesKeyidx);
        for (int vI = 0; vI < cNMerges; vI++) {
          const std::string cWord = gguf_get_arr_str(vCtx, cMergesKeyidx, vI);
          //GGML_ASSERT(unicode_cpts_from_utf8(word).size() > 0);

          std::string vFirst;
          std::string vSecond;

          const size_t cPos = cWord.find(' ', 1);

          if (cPos != std::string::npos) {
            vFirst  = cWord.substr(0, cPos);
            vSecond = cWord.substr(cPos + 1);
          }

          dBpeRanks.emplace(std::make_pair(vFirst, vSecond), vI);
        }
      }

      // default special tokens
      vSpecialBosId  = 11;
      vSpecialEosId  = 11;
      vSpecialUnkId  = DGS_TOKEN_NULL;
      vSpecialSepId  = DGS_TOKEN_NULL;
      vSpecialPadId  = DGS_TOKEN_NULL;
      vSpecialMaskId = DGS_TOKEN_NULL;
    } else if (vTokenizerModel == "t5") {
      type = cDgsVocabTypeUgm;

      // default special tokens
      vSpecialBosId  = DGS_TOKEN_NULL;
      vSpecialEosId  = 1;
      vSpecialUnkId  = 2;
      vSpecialSepId  = DGS_TOKEN_NULL;
      vSpecialPadId  = 0;
      vSpecialMaskId = DGS_TOKEN_NULL;

      const int cPrecompiledCharsmapKeyidx = gguf_find_key(vCtx, pKv(cLlmKvTokenizerPrecompiledCharsmap).c_str());
      if (cPrecompiledCharsmapKeyidx != -1) {
        if (gguf_get_kv_type(vCtx, cPrecompiledCharsmapKeyidx) != GGUF_TYPE_ARRAY) {
          throw std::runtime_error(format("invalid gguf type for %s", pKv(cLlmKvTokenizerPrecompiledCharsmap).c_str()));
        }
        const gguf_type cPcType = gguf_get_arr_type(vCtx, cPrecompiledCharsmapKeyidx);
        if (cPcType != GGUF_TYPE_INT8 && cPcType != GGUF_TYPE_UINT8) {
          throw std::runtime_error(format("invalid gguf type for %s", pKv(cLlmKvTokenizerPrecompiledCharsmap).c_str()));
        }

        const size_t cNPrecompiledCharsmap = gguf_get_arr_n(vCtx, cPrecompiledCharsmapKeyidx);
        const char * vPc                   = (const char *) gguf_get_arr_data(vCtx, cPrecompiledCharsmapKeyidx);
        lPrecompiledCharsmap.assign(vPc, vPc + cNPrecompiledCharsmap);
        if (lPrecompiledCharsmap.size() < sizeof(uint32_t)) {
          throw std::runtime_error("precompiled_charsmap too small for xcda_blob_size header!");
        }
        uint32_t * vXcdaBlobSize = (uint32_t *) &lPrecompiledCharsmap[0];
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        *xcda_blob_size = __builtin_bswap32(*xcda_blob_size);
#endif
        if (*vXcdaBlobSize + sizeof(uint32_t) >= lPrecompiledCharsmap.size()) {
          throw std::runtime_error("Index out of array bounds in precompiled charsmap!");
        }
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        // correct endianness of data in precompiled_charsmap binary blob
        size_t     xcda_array_size = *xcda_blob_size / sizeof(uint32_t);
        uint32_t * xcda_array      = (uint32_t *) &precompiled_charsmap[sizeof(uint32_t)];
        for (size_t i = 0; i < xcda_array_size; ++i) {
          xcda_array[i] = __builtin_bswap32(xcda_array[i]);
        }
#endif
      }
    } else if (vTokenizerModel == "rwkv") {
      type = cDgsVocabTypeRwkv;

      // default special tokens
      vSpecialBosId = DGS_TOKEN_NULL;
      vSpecialEosId = DGS_TOKEN_NULL;
      vSpecialUnkId = DGS_TOKEN_NULL;
      vSpecialSepId = DGS_TOKEN_NULL;
      vSpecialPadId = DGS_TOKEN_NULL;
    } else if (vTokenizerModel == "test") {
      type = cDgsVocabTypeTest;

      // default special tokens
      vSpecialBosId  = DGS_TOKEN_NULL;
      vSpecialEosId  = DGS_TOKEN_NULL;
      vSpecialUnkId  = DGS_TOKEN_NULL;
      vSpecialSepId  = DGS_TOKEN_NULL;
      vSpecialPadId  = DGS_TOKEN_NULL;
      vSpecialMaskId = DGS_TOKEN_NULL;
    } else if (vTokenizerModel == "plamo2" || vTokenizerModel == "plamo3") {
      type = vTokenizerModel == "plamo2" ? cDgsVocabTypePlamo2 : cDgsVocabTypePlamo3;

      // PLaMo default special tokens (these will be overridden by model config)
      vSpecialBosId  = 1;  // <|plamo:bos|>
      vSpecialEosId  = 2;  // <|plamo:eos|>
      vSpecialUnkId  = 0;  // <|plamo:unk|>
      vSpecialSepId  = DGS_TOKEN_NULL;
      vSpecialPadId  = 3;  // <|plamo:pad|>
      vSpecialMaskId = DGS_TOKEN_NULL;
    } else if (vTokenizerModel == "gemma4") {
      type = cDgsVocabTypeBpe;

      // read bpe merges and populate bpe ranks
      const int cMergesKeyidx = gguf_find_key(vCtx, pKv(cLlmKvTokenizerMerges).c_str());
      if (cMergesKeyidx == -1) {
        throw std::runtime_error("cannot find tokenizer merges in model file\n");
      }
      {
        if (gguf_get_kv_type(vCtx, cMergesKeyidx) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(vCtx, cMergesKeyidx) != GGUF_TYPE_STRING) {
          throw std::runtime_error(format("invalid gguf type for %s", pKv(cLlmKvTokenizerMerges).c_str()));
        }
        const int cNMerges = gguf_get_arr_n(vCtx, cMergesKeyidx);
        for (int vI = 0; vI < cNMerges; vI++) {
          const std::string cWord = gguf_get_arr_str(vCtx, cMergesKeyidx, vI);

          std::string vFirst;
          std::string vSecond;

          const size_t cPos = cWord.find(' ', 1);

          if (cPos != std::string::npos) {
            vFirst  = cWord.substr(0, cPos);
            vSecond = cWord.substr(cPos + 1);
          }

          dBpeRanks.emplace(std::make_pair(vFirst, vSecond), vI);
        }
      }

      // default special tokens (to be read from GGUF)
      vSpecialBosId  = DGS_TOKEN_NULL;
      vSpecialEosId  = DGS_TOKEN_NULL;
      vSpecialUnkId  = DGS_TOKEN_NULL;
      vSpecialSepId  = DGS_TOKEN_NULL;
      vSpecialPadId  = DGS_TOKEN_NULL;
      vSpecialMaskId = DGS_TOKEN_NULL;

      vTokenizerPre = "gemma4";
    } else {
      throw std::runtime_error(format("unknown tokenizer: '%s'", vTokenizerModel.c_str()));
    }

    // for now, only BPE models have pre-tokenizers
    if (type == cDgsVocabTypeBpe) {
      vAddSpacePrefix    = false;
      vEscapeWhitespaces = false;
      vCleanSpaces       = true;
      if (vTokenizerPre.empty()) {
        DGS_LOG_WARN("%s: missing pre-tokenizer type, using: 'default'\n", __func__);
        DGS_LOG_WARN("%s:                                             \n", __func__);
        DGS_LOG_WARN("%s: ************************************        \n", __func__);
        DGS_LOG_WARN("%s: GENERATION QUALITY WILL BE DEGRADED!        \n", __func__);
        DGS_LOG_WARN("%s: CONSIDER REGENERATING THE MODEL             \n", __func__);
        DGS_LOG_WARN("%s: ************************************        \n", __func__);
        DGS_LOG_WARN("%s:                                             \n", __func__);
        vPreType = cDgsVocabPreTypeDefault;
      } else if (vTokenizerPre == "default") {
        vPreType = cDgsVocabPreTypeDefault;
      } else if (vTokenizerPre == "minicpm5") {
        vPreType      = cDgsVocabPreTypeMinicpm5;
        vIgnoreMerges = true;
      } else if (vTokenizerPre == "llama3" || vTokenizerPre == "llama-v3" || vTokenizerPre == "llama-bpe" ||
                 vTokenizerPre == "falcon3" || vTokenizerPre == "falcon-h1" || vTokenizerPre == "pixtral" ||
                 vTokenizerPre == "midm-2.0" || vTokenizerPre == "lfm2" || vTokenizerPre == "jina-v5-nano") {
        vPreType      = cDgsVocabPreTypeDgs3;
        vIgnoreMerges = true;
        vAddBos       = true;
      } else if (vTokenizerPre == "deepseek-llm") {
        vPreType     = cDgsVocabPreTypeDeepseekLlm;
        vCleanSpaces = false;
      } else if (vTokenizerPre == "deepseek-coder") {
        vPreType     = cDgsVocabPreTypeDeepseekCoder;
        vCleanSpaces = false;
      } else if (vTokenizerPre == "deepseek-v3") {
        vPreType     = cDgsVocabPreTypeDeepseek3Llm;
        vCleanSpaces = false;
      } else if (vTokenizerPre == "spark2_5") {
        vPreType     = cDgsVocabPreTypeSPARK2Point5;
        vCleanSpaces = false;
      } else if (vTokenizerPre == "youtu") {
        vPreType      = cDgsVocabPreTypeYoutu;
        vCleanSpaces  = false;
        vIgnoreMerges = true;
      } else if (vTokenizerPre == "falcon") {
        vPreType = cDgsVocabPreTypeFalcon;
      } else if (vTokenizerPre == "mpt") {
        vPreType = cDgsVocabPreTypeMpt;
      } else if (vTokenizerPre == "starcoder") {
        vPreType = cDgsVocabPreTypeStarcoder;
      } else if (vTokenizerPre == "gpt-2" || vTokenizerPre == "phi-2" || vTokenizerPre == "jina-es" ||
                 vTokenizerPre == "jina-de" || vTokenizerPre == "gigachat" || vTokenizerPre == "jina-v2-es" ||
                 vTokenizerPre == "jina-v2-de" || vTokenizerPre == "a.x-4.0" || vTokenizerPre == "mellum" ||
                 vTokenizerPre == "modern-bert") {
        vPreType = cDgsVocabPreTypeGpt2;
      } else if (vTokenizerPre == "jais-2") {
        vPreType = cDgsVocabPreTypeJais2;
      } else if (vTokenizerPre == "gemma4" || vTokenizerPre == "granite-embed-multi-311m") {
        vPreType           = cDgsVocabPreTypeGemma4;
        vEscapeWhitespaces = true;
      } else if (vTokenizerPre == "mmbert") {
        vPreType           = cDgsVocabPreTypeMmbert;
        vEscapeWhitespaces = true;
      } else if (vTokenizerPre == "sarvam-moe") {
        vPreType           = cDgsVocabPreTypeSarvamMoe;
        vEscapeWhitespaces = true;
        vCleanSpaces       = false;
      } else if (vTokenizerPre == "jina-v1-en" || vTokenizerPre == "jina-v2-code" || vTokenizerPre == "roberta-bpe") {
        vPreType = cDgsVocabPreTypeGpt2;
        vAddSep  = true;
      } else if (vTokenizerPre == "whitespace") {
        vPreType                   = cDgsVocabPreTypeWhitespace;
        vNormalizerOpts.vLowercase = false;
      } else if (vTokenizerPre == "refact") {
        vPreType = cDgsVocabPreTypeRefact;
      } else if (vTokenizerPre == "command-r") {
        vPreType     = cDgsVocabPreTypeCommandR;
        vCleanSpaces = false;
      } else if (vTokenizerPre == "qwen2" || vTokenizerPre == "deepseek-r1-qwen" || vTokenizerPre == "kormo" ||
                 vTokenizerPre == "f2llmv2") {
        vPreType     = cDgsVocabPreTypeQwen2;
        vCleanSpaces = false;
      } else if (vTokenizerPre == "qwen35") {
        vPreType     = cDgsVocabPreTypeQwen35;
        vCleanSpaces = false;
      } else if (vTokenizerPre == "stablelm2") {
        vPreType = cDgsVocabPreTypeStablelm2;
      } else if (vTokenizerPre == "olmo") {
        vPreType = cDgsVocabPreTypeOlmo;
      } else if (vTokenizerPre == "dbrx") {
        vPreType = cDgsVocabPreTypeDbrx;
      } else if (vTokenizerPre == "smaug-bpe") {
        vPreType = cDgsVocabPreTypeSmaug;
      } else if (vTokenizerPre == "poro-chat") {
        vPreType     = cDgsVocabPreTypePoro;
        vCleanSpaces = false;
      } else if (vTokenizerPre == "chatglm-bpe") {
        vPreType      = cDgsVocabPreTypeChatglm4;
        vSpecialBosId = DGS_TOKEN_NULL;
      } else if (vTokenizerPre == "glm4" || vTokenizerPre == "glm5") {
        vPreType      = cDgsVocabPreTypeChatglm4;
        vSpecialBosId = DGS_TOKEN_NULL;
        vIgnoreMerges = true;
      } else if (vTokenizerPre == "viking") {
        vPreType     = cDgsVocabPreTypeViking;
        vCleanSpaces = false;
      } else if (vTokenizerPre == "jais") {
        vPreType = cDgsVocabPreTypeJais;
      } else if (vTokenizerPre == "tekken") {
        vPreType      = cDgsVocabPreTypeTekken;
        vCleanSpaces  = false;
        vIgnoreMerges = true;
        vAddBos       = true;
      } else if (vTokenizerPre == "smollm") {
        vPreType     = cDgsVocabPreTypeSmollm;
        vCleanSpaces = false;
      } else if (vTokenizerPre == "codeshell") {
        vPreType = cDgsVocabPreTypeCodeshell;
      } else if (vTokenizerPre == "bloom") {
        vPreType = cDgsVocabPreTypeBloom;
      } else if (vTokenizerPre == "gpt3-finnish") {
        vPreType = cDgsVocabPreTypeGpt3Finnish;
      } else if (vTokenizerPre == "exaone") {
        vPreType = cDgsVocabPreTypeExaone;
      } else if (vTokenizerPre == "exaone4") {
        vPreType = cDgsVocabPreTypeGpt2;
      } else if (vTokenizerPre == "exaone-moe") {
        vPreType = cDgsVocabPreTypeExaoneMoe;
      } else if (vTokenizerPre == "chameleon") {
        vPreType     = cDgsVocabPreTypeChameleon;
        vAddBos      = true;
        vCleanSpaces = false;
      } else if (vTokenizerPre == "minerva-7b") {
        vPreType = cDgsVocabPreTypeMinerva;
      } else if (vTokenizerPre == "megrez") {
        vPreType = cDgsVocabPreTypeQwen2;
      } else if (vTokenizerPre == "gpt-4o" || vTokenizerPre == "llama4" || vTokenizerPre == "kanana2" ||
                 vTokenizerPre == "talkie") {
        vPreType     = cDgsVocabPreTypeGpt4o;
        vCleanSpaces = false;
      } else if (vTokenizerPre == "granite-embed-multi-97m") {
        vPreType      = cDgsVocabPreTypeGraniteEmbMulti;
        vCleanSpaces  = false;
        vIgnoreMerges = true;
      } else if (vTokenizerPre == "tiny_aya" || vTokenizerPre == "cohere2moe") {
        vPreType     = cDgsVocabPreTypeTinyAya;
        vCleanSpaces = false;
      } else if (vTokenizerPre == "superbpe") {
        vPreType     = cDgsVocabPreTypeSuperbpe;
        vCleanSpaces = false;
      } else if (vTokenizerPre == "trillion") {
        vPreType     = cDgsVocabPreTypeTrillion;
        vCleanSpaces = false;
      } else if (vTokenizerPre == "granite-docling") {
        vPreType     = cDgsVocabPreTypeGraniteDocling;
        vCleanSpaces = false;
      } else if (vTokenizerPre == "bailingmoe" || vTokenizerPre == "bailingmoe2" || vTokenizerPre == "llada-moe") {
        vPreType     = cDgsVocabPreTypeBailingmoe;
        vCleanSpaces = false;
      } else if (vTokenizerPre == "seed-coder") {
        vPreType     = cDgsVocabPreTypeSeedCoder;
        vCleanSpaces = false;
      } else if (vTokenizerPre == "hunyuan") {
        vPreType     = cDgsVocabPreTypeHunyuan;
        vCleanSpaces = false;
      } else if (vTokenizerPre == "hunyuan-dense") {
        vPreType     = cDgsVocabPreTypeHunyuanDense;
        vCleanSpaces = false;
      } else if (vTokenizerPre == "hy_v4") {
        vPreType     = cDgsVocabPreTypeHyV4;
        vCleanSpaces = false;
      } else if (vTokenizerPre == "joyai-llm") {
        vPreType     = cDgsVocabPreTypeJoyaiLlm;
        vCleanSpaces = false;
      } else if (vTokenizerPre == "kimi-k2") {
        vPreType     = cDgsVocabPreTypeKimiK2;
        vCleanSpaces = false;
      } else if (vTokenizerPre == "ufakzeka") {
        vPreType     = cDgsVocabPreTypeUfakzeka;
        vCleanSpaces = false;
      } else if (vTokenizerPre == "grok-2") {
        vPreType     = cDgsVocabPreTypeGrok2;
        vCleanSpaces = false;
      } else if (vTokenizerPre == "afmoe") {
        vPreType     = cDgsVocabPreTypeAfmoe;
        vCleanSpaces = false;
      } else if (vTokenizerPre == "laguna") {
        vPreType     = cDgsVocabPreTypeLaguna;
        vCleanSpaces = false;
      } else if (vTokenizerPre == "minimax-m2") {
        vPreType     = cDgsVocabPreTypeMinimaxM2;
        vCleanSpaces = false;
      } else if (vTokenizerPre == "solar-open") {
        vPreType     = cDgsVocabPreTypeSolarOpen;
        vCleanSpaces = false;
      } else if (vTokenizerPre == "mellum2") {
        vPreType = cDgsVocabPreTypeMellum2;
      } else if (vTokenizerPre == "k2-horizon") {
        vPreType     = cDgsVocabPreTypeK2Horizon;
        vCleanSpaces = false;
      } else {
        throw std::runtime_error(format("unknown pre-tokenizer type: '%s'", vTokenizerPre.c_str()));
      }
    } else if (type == cDgsVocabTypeSpm) {
      vPreType        = cDgsVocabPreTypeDefault;
      vAddSpacePrefix = true;
      vCleanSpaces    = false;
      vAddBos         = true;
      vAddEos         = false;
    } else if (type == cDgsVocabTypeWpm) {
      vPreType        = cDgsVocabPreTypeDefault;
      vAddSpacePrefix = false;
      vCleanSpaces    = true;
      vAddBos         = true;
      vAddEos         = false;
      vAddSep         = true;
    } else if (type == cDgsVocabTypeUgm) {
      vPreType = cDgsVocabPreTypeDefault;
      vAddBos  = false;
      vAddEos  = true;
    } else if (type == cDgsVocabTypeRwkv) {
      vPreType        = cDgsVocabPreTypeDefault;
      vAddSpacePrefix = false;
      vCleanSpaces    = false;
      vAddBos         = false;
      vAddEos         = false;
    } else {
      vPreType = cDgsVocabPreTypeDefault;
    }

    pMl.get_key(cLlmKvTokenizerAddPrefix, vAddSpacePrefix, false);
    pMl.get_key(cLlmKvTokenizerRemoveExtraWs, vRemoveExtraWhitespaces, false);
  }

  const int cTokenIdx = gguf_find_key(vCtx, pKv(cLlmKvTokenizerList).c_str());
  if (cTokenIdx == -1) {
    throw std::runtime_error("cannot find tokenizer vocab in model file\n");
  }

  if (gguf_get_kv_type(vCtx, cTokenIdx) != GGUF_TYPE_ARRAY || gguf_get_arr_type(vCtx, cTokenIdx) != GGUF_TYPE_STRING) {
    throw std::runtime_error(format("invalid gguf type for %s", pKv(cLlmKvTokenizerList).c_str()));
  }

  const uint32_t cNTokens = gguf_get_arr_n(vCtx, cTokenIdx);

  const float * vScores   = nullptr;
  const int *   vIscores  = nullptr;
  const int     cScoreIdx = gguf_find_key(vCtx, pKv(cLlmKvTokenizerScores).c_str());
  if (cScoreIdx != -1) {
    const gguf_type cKvType  = gguf_get_kv_type(vCtx, cScoreIdx);
    const gguf_type cArrType = cKvType == GGUF_TYPE_ARRAY ? gguf_get_arr_type(vCtx, cScoreIdx) : GGUF_TYPE_COUNT;
    if (cArrType != GGUF_TYPE_INT32 && cArrType != GGUF_TYPE_FLOAT32) {
      throw std::runtime_error(format("invalid gguf type for %s", pKv(cLlmKvTokenizerScores).c_str()));
    }
    const uint32_t cNScores = gguf_get_arr_n(vCtx, cScoreIdx);
    if (cNScores < cNTokens) {
      throw std::runtime_error("Index out of array bounds for scores (" + std::to_string(cNScores) + " < " +
                               std::to_string(cNTokens) + ")\n");
    }
    if (cArrType == GGUF_TYPE_INT32) {
      vIscores = (const int *) gguf_get_arr_data(vCtx, cScoreIdx);
    } else {
      vScores = (const float *) gguf_get_arr_data(vCtx, cScoreIdx);
    }
  }

  const int * vToktypes   = nullptr;
  const int   cToktypeIdx = gguf_find_key(vCtx, pKv(cLlmKvTokenizerTokenType).c_str());
  if (cToktypeIdx != -1) {
    if (gguf_get_kv_type(vCtx, cToktypeIdx) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(vCtx, cToktypeIdx) != GGUF_TYPE_INT32) {
      throw std::runtime_error(format("invalid gguf type for %s", pKv(cLlmKvTokenizerTokenType).c_str()));
    }
    const uint32_t cNToktypes = gguf_get_arr_n(vCtx, cToktypeIdx);
    if (cNToktypes < cNTokens) {
      throw std::runtime_error("Index out of array bounds for toktypes (" + std::to_string(cNToktypes) + " < " +
                               std::to_string(cNTokens) + ")\n");
    }
    vToktypes = (const int *) gguf_get_arr_data(vCtx, cToktypeIdx);
  }

  lIdToToken.resize(cNTokens);

  for (uint32_t vI = 0; vI < cNTokens; vI++) {
    std::string vWord = gguf_get_arr_str(vCtx, cTokenIdx, vI);
    if (vWord.empty()) {
      DGS_LOG_WARN("%s: empty token at index %u\n", __func__, vI);
      vWord = "[EMPTY_" + std::to_string(vI) + "]";
    }

    dTokenToId[vWord] = vI;
    vMaxTokenLen      = std::max(vMaxTokenLen, (int) vWord.size());

    auto & vTokenData = lIdToToken[vI];
    vTokenData.vText  = std::move(vWord);
    if (vScores) {
      vTokenData.vScore = vScores[vI];
    } else if (vIscores) {
      vTokenData.vScore = static_cast<float>(vIscores[vI]);
    } else {
      vTokenData.vScore = 0.0f;
    }
    vTokenData.vAttr = cDgsTokenAttrNormal;

    if (vToktypes) {  //TODO: remove, required until per token attributes are available from GGUF file
      switch (vToktypes[vI]) {
        case cDgsTokenTypeUnknown:
          vTokenData.vAttr = cDgsTokenAttrUnknown;
          break;
        case cDgsTokenTypeUnused:
          vTokenData.vAttr = cDgsTokenAttrUnused;
          break;
        case cDgsTokenTypeNormal:
          vTokenData.vAttr = cDgsTokenAttrNormal;
          break;
        case cDgsTokenTypeControl:
          vTokenData.vAttr = cDgsTokenAttrControl;
          break;
        case cDgsTokenTypeUserDefined:
          vTokenData.vAttr = cDgsTokenAttrUserDefined;
          break;
        case cDgsTokenTypeByte:
          vTokenData.vAttr = cDgsTokenAttrByte;
          break;
        case cDgsTokenTypeUndefined:
          vTokenData.vAttr = cDgsTokenAttrUndefined;
          break;
        default:
          vTokenData.vAttr = cDgsTokenAttrUndefined;
          break;
      }
    }
  }
  GGML_ASSERT(lIdToToken.size() == dTokenToId.size());

  // hybriddna: the marker suffix kept k-mer ids distinct in token_to_id; erase
  // it from id_to_token so the k-mers detokenize to the bare DNA sequence. The
  // k-mers are the block right after <oov>, so only scan from there.
  if (vTokenizerModel == "hybriddna") {
    const auto cIdx = dTokenToId.find("<oov>");
    if (cIdx != dTokenToId.end()) {
      auto lIt = lIdToToken.begin() + cIdx->second + 1;
      for (; lIt != lIdToToken.end(); ++lIt) {
        std::string & vText = lIt->vText;
        if (vText.size() > cDnaKmerMarker.size() &&
            vText.compare(vText.size() - cDnaKmerMarker.size(), cDnaKmerMarker.size(), cDnaKmerMarker) == 0) {
          vText.erase(vText.size() - cDnaKmerMarker.size());
        }
      }
    }
  }

  mInitTokenizer(type);

  // determine the newline token: LLaMA "<0x0A>" == 10 == '\n', Falcon 193 == '\n'
  if (type == cDgsVocabTypeSpm) {
    try {
      vLinefeedId = vVocab.mByteToToken('\n');
    } catch (const std::exception & vE) {
      DGS_LOG_WARN("%s: SPM vocabulary, but newline token not found: %s! Using special_pad_id instead.", __func__,
                   vE.what());
      vLinefeedId = vSpecialPadId;
    }
  } else if (type == cDgsVocabTypeWpm) {
    vLinefeedId = vSpecialPadId;
  } else if (type == cDgsVocabTypeRwkv) {
    const std::vector<int> lIds = mTokenize("\n", false);
    GGML_ASSERT(!lIds.empty() && "model vocab missing newline token");
    vLinefeedId = lIds[0];
  } else {
    const std::vector<int> lIds = mTokenize("\n", false);

    //GGML_ASSERT(!ids.empty() && "model vocab missing newline token");
    if (lIds.empty()) {
      DGS_LOG_WARN("%s: model vocab missing newline token, using special_pad_id instead\n", __func__);
      vLinefeedId = vSpecialPadId;
    } else {
      vLinefeedId = lIds[0];
    }
  }

  // special tokens
  {
    const std::vector<std::pair<enum llm_kv, int32_t &>> lSpecialTokenTypes = {
      { cLlmKvTokenizerBosId,    vSpecialBosId    },
      { cLlmKvTokenizerEosId,    vSpecialEosId    },
      { cLlmKvTokenizerEotId,    vSpecialEotId    },
      { cLlmKvTokenizerEomId,    vSpecialEomId    },
      { cLlmKvTokenizerUnkId,    vSpecialUnkId    },
      { cLlmKvTokenizerSepId,    vSpecialSepId    },
      { cLlmKvTokenizerPadId,    vSpecialPadId    },
      { cLlmKvTokenizerMaskId,   vSpecialMaskId   },
      { cLlmKvTokenizerFimPreId, vSpecialFimPreId },
      { cLlmKvTokenizerFimSufId, vSpecialFimSufId },
      { cLlmKvTokenizerFimMidId, vSpecialFimMidId },
      { cLlmKvTokenizerFimPadId, vSpecialFimPadId },
      { cLlmKvTokenizerFimRepId, vSpecialFimRepId },
      { cLlmKvTokenizerFimSepId, vSpecialFimSepId },
    };

    for (const auto & vIt : lSpecialTokenTypes) {
      const std::string & vKey = pKv(std::get<0>(vIt));
      int32_t &           vId  = std::get<1>(vIt);

      if (vId >= 0 && static_cast<size_t>(vId) >= lIdToToken.size()) {
        DGS_LOG_WARN("%s: default special token '%s' = %d out of vocab range, disabling\n", __func__, vKey.c_str(),
                     vId);
        vId = DGS_TOKEN_NULL;
      }

      uint32_t vNewId;
      if (!pMl.get_key(std::get<0>(vIt), vNewId, false)) {
        continue;
      }
      if (vNewId >= lIdToToken.size()) {
        DGS_LOG_WARN("%s: bad special token: '%s' = %u, using default id %d\n", __func__, vKey.c_str(), vNewId, vId);
      } else {
        vId = vNewId;
      }
    }

    // Handle add_bos, add_eos and add_sep
    {
      bool vTemp = true;

      if (pMl.get_key(cLlmKvTokenizerAddBos, vTemp, false)) {
        vAddBos = vTemp;
      }
      if (pMl.get_key(cLlmKvTokenizerAddEos, vTemp, false)) {
        vAddEos = vTemp;
      }
      if (pMl.get_key(cLlmKvTokenizerAddSep, vTemp, false)) {
        vAddSep = vTemp;
      }

      // workaround for Gemma 4
      if (vPreType == cDgsVocabPreTypeGemma4 && !vAddBos) {
        vAddBos = true;

        DGS_LOG_WARN("%s: override '%s' to 'true' for Gemma4\n", __func__, pKv(cLlmKvTokenizerAddBos).c_str());
      }
    }

    // BertNormalizer options
    pMl.get_key(cLlmKvTokenizerNormalizerLowercase, vNormalizerOpts.vLowercase, false);
    vNormalizerOpts.vStripAccents = vNormalizerOpts.vLowercase;
    pMl.get_key(cLlmKvTokenizerNormalizerStripAccents, vNormalizerOpts.vStripAccents, false);

    // suppress tokens
    {
      const int cSuppressIdx = gguf_find_key(vCtx, pKv(cLlmKvTokenizerSuppressTokens).c_str());
      if (cSuppressIdx != -1) {
        if (gguf_get_kv_type(vCtx, cSuppressIdx) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(vCtx, cSuppressIdx) != GGUF_TYPE_INT32) {
          throw std::runtime_error(format("invalid gguf type for %s", pKv(cLlmKvTokenizerSuppressTokens).c_str()));
        }
        const int       n    = gguf_get_arr_n(vCtx, cSuppressIdx);
        const int32_t * data = (const int32_t *) gguf_get_arr_data(vCtx, cSuppressIdx);
        // drop out-of-range ids
        lSuppressTokens.reserve(n);
        for (int vI = 0; vI < n; ++vI) {
          const int32_t cId = data[vI];
          if (cId >= 0 && cId < (int) lIdToToken.size()) {
            lSuppressTokens.push_back(cId);
          }
        }
      }
    }

    // auto-detect special tokens by text
    // TODO: convert scripts should provide these tokens through the KV metadata LLM_KV_TOKENIZER_...
    //       for now, we apply this workaround to find the tokens based on their text

    for (const auto & vT : dTokenToId) {
      auto & vAttr = lIdToToken[vT.second].vAttr;

      // find EOT token: "<|eot_id|>", "<|im_end|>", "<end_of_turn>", etc.
      if (vSpecialEotId == DGS_TOKEN_NULL) {
        if (false || vT.first == "<|eot_id|>" || vT.first == "<|im_end|>" || vT.first == "<|end|>" ||
            vT.first == "<end_of_turn>" || vT.first == "<|endoftext|>" || vT.first == "<|end_of_text|>"  // granite
            || vT.first == "<EOT>" || vT.first == "_<EOT>" || vT.first == "[EOT]"                        // Kimi-K2
            || vT.first == "<｜end▁of▁sentence｜>"                                                       // DeepSeek
            || vT.first == "<end_of_utterance>"                                                          // smoldocling
        ) {
          vSpecialEotId = vT.second;
          if ((vAttr & cDgsTokenAttrControl) == 0) {
            DGS_LOG_WARN(
              "%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type "
              "will be overridden\n",
              __func__, vT.second, vT.first.c_str());
            vAttr = (dgs_token_attr) (vAttr | cDgsTokenAttrControl);
          }
        }
      }

      // find EOM token: "<|eom_id|>"
      if (vSpecialEomId == DGS_TOKEN_NULL) {
        if (false || vT.first == "<|eom_id|>") {
          vSpecialEomId = vT.second;
          if ((vAttr & cDgsTokenAttrControl) == 0) {
            DGS_LOG_WARN(
              "%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type "
              "will be overridden\n",
              __func__, vT.second, vT.first.c_str());
            vAttr = (dgs_token_attr) (vAttr | cDgsTokenAttrControl);
          }
        }
      }

      // find FIM_PRE token: "<|fim_prefix|>", "<fim-prefix>", "<PRE>", etc.
      if (vSpecialFimPreId == DGS_TOKEN_NULL) {
        if (false || vT.first == "<|fim_prefix|>"                        // Qwen
            || vT.first == "<fim-prefix>" || vT.first == "<fim_prefix>"  // Granite
            || vT.first == "<｜fim▁begin｜>"                             // DeepSeek
            || vT.first == "<PRE>" || vT.first == "▁<PRE>"               // CodeLlama
            || vT.first == "<|code_prefix|>"                             // GLM-4.5
            || vT.first == "<|prefix|>"                                  // Falcon-H1-Tiny-Coder
            || vT.first == "<|plamo:fim_prefix|>"                        // PLaMo-3
        ) {
          vSpecialFimPreId = vT.second;
          if ((vAttr & cDgsTokenAttrControl) == 0) {
            DGS_LOG_WARN(
              "%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type "
              "will be overridden\n",
              __func__, vT.second, vT.first.c_str());
            vAttr = (dgs_token_attr) (vAttr | cDgsTokenAttrControl);
          }
        }
      }

      // find FIM_SUF token: "<|fim_suffix|>", "<fim-suffix>", "<SUF>", etc.
      if (vSpecialFimSufId == DGS_TOKEN_NULL) {
        if (false || vT.first == "<|fim_suffix|>"                        // Qwen
            || vT.first == "<fim-suffix>" || vT.first == "<fim_suffix>"  // Granite
            || vT.first == "<｜fim▁hole｜>"                              // DeepSeek
            || vT.first == "<SUF>" || vT.first == "▁<SUF>"               // CodeLlama
            || vT.first == "<|code_suffix|>"                             // GLM-4.5
            || vT.first == "<|suffix|>"                                  // Falcon-H1-Tiny-Coder
            || vT.first == "<|plamo:fim_suffix|>"                        // PLaMo-3
        ) {
          vSpecialFimSufId = vT.second;
          if ((vAttr & cDgsTokenAttrControl) == 0) {
            DGS_LOG_WARN(
              "%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type "
              "will be overridden\n",
              __func__, vT.second, vT.first.c_str());
            vAttr = (dgs_token_attr) (vAttr | cDgsTokenAttrControl);
          }
        }
      }

      // find FIM_MID token: "<|fim_middle|>", "<fim-middle>", "<MID>", etc.
      if (vSpecialFimMidId == DGS_TOKEN_NULL) {
        if (false || vT.first == "<|fim_middle|>"                        // Qwen
            || vT.first == "<fim-middle>" || vT.first == "<fim_middle>"  // Granite
            || vT.first == "<｜fim▁end｜>"                               // DeepSeek
            || vT.first == "<MID>" || vT.first == "▁<MID>"               // CodeLlama
            || vT.first == "<|code_middle|>"                             // GLM-4.5
            || vT.first == "<|middle|>"                                  // Falcon-H1-Tiny-Coder
            || vT.first == "<|plamo:fim_middle|>"                        // PLaMo-3
        ) {
          vSpecialFimMidId = vT.second;
          if ((vAttr & cDgsTokenAttrControl) == 0) {
            DGS_LOG_WARN(
              "%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type "
              "will be overridden\n",
              __func__, vT.second, vT.first.c_str());
            vAttr = (dgs_token_attr) (vAttr | cDgsTokenAttrControl);
          }
        }
      }

      // find FIM_PAD token: "<|fim_pad|>", "<fim-pad>", "<PAD>", etc.
      if (vSpecialFimPadId == DGS_TOKEN_NULL) {
        if (false || vT.first == "<|fim_pad|>"                     // Qwen
            || vT.first == "<fim-pad>" || vT.first == "<fim_pad>"  // Granite
            || vT.first == "<PAD>" || vT.first == "[PAD]"          // Kimi-K2
        ) {
          vSpecialFimPadId = vT.second;
          if ((vAttr & cDgsTokenAttrControl) == 0) {
            DGS_LOG_WARN(
              "%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type "
              "will be overridden\n",
              __func__, vT.second, vT.first.c_str());
            vAttr = (dgs_token_attr) (vAttr | cDgsTokenAttrControl);
          }
        }
      }

      // find FIM_REP token: "<|fim_repo|>", "<fim-repo>", "<REP>", etc.
      if (vSpecialFimRepId == DGS_TOKEN_NULL) {
        if (false || vT.first == "<|fim_repo|>"  // Qwen
            || vT.first == "<|repo_name|>" || vT.first == "<fim-repo>" || vT.first == "<REPO>" ||
            vT.first == "<reponame>"             // Granite
        ) {
          vSpecialFimRepId = vT.second;
          if ((vAttr & cDgsTokenAttrControl) == 0) {
            DGS_LOG_WARN(
              "%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type "
              "will be overridden\n",
              __func__, vT.second, vT.first.c_str());
            vAttr = (dgs_token_attr) (vAttr | cDgsTokenAttrControl);
          }
        }
      }

      // find FIM_SEP token: "<|file_sep|>"
      if (vSpecialFimSepId == DGS_TOKEN_NULL) {
        if (false || vT.first == "<|file_sep|>"        // Qwen
            || vT.first == "<|plamo:file_separator|>"  // PLaMo-2/3
        ) {
          vSpecialFimSepId = vT.second;
          if ((vAttr & cDgsTokenAttrControl) == 0) {
            DGS_LOG_WARN(
              "%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type "
              "will be overridden\n",
              __func__, vT.second, vT.first.c_str());
            vAttr = (dgs_token_attr) (vAttr | cDgsTokenAttrControl);
          }
        }
      }
    }

    // auto-detect unused tokens: e.g. control tokens with the word "unused"
    // ideally, these tokens should be marked as unused during conversion
    {
      uint32_t vNUnused = 0;

      for (const auto & vT : dTokenToId) {
        auto & vAttr = lIdToToken[vT.second].vAttr;

        if ((vAttr & cDgsTokenAttrControl) == 0) {
          continue;
        }

        if ((vAttr & cDgsTokenAttrUnused) == 0) {
          if (strstr(vT.first.c_str(), "unused") != NULL) {
            vAttr = (dgs_token_attr) (vAttr | cDgsTokenAttrUnused);
          }
        }

        if (vAttr & cDgsTokenAttrUnused) {
          vNUnused++;
        }
      }

      DGS_LOG_INFO("%s: %u unused tokens\n", __func__, vNUnused);
    }

    // maintain a list of tokens that cause end-of-generation
    // this is currently determined based on the token text, which is obviously not ideal
    sSpecialEogIds.clear();

    if (vSpecialFimPadId != DGS_TOKEN_NULL && sSpecialEogIds.count(vSpecialFimPadId) == 0) {
      sSpecialEogIds.insert(vSpecialFimPadId);
    }

    if (vSpecialFimRepId != DGS_TOKEN_NULL && sSpecialEogIds.count(vSpecialFimRepId) == 0) {
      sSpecialEogIds.insert(vSpecialFimRepId);
    }

    if (vSpecialFimSepId != DGS_TOKEN_NULL && sSpecialEogIds.count(vSpecialFimSepId) == 0) {
      sSpecialEogIds.insert(vSpecialFimSepId);
    }

    for (const auto & vT : dTokenToId) {
      auto & vAttr = lIdToToken[vT.second].vAttr;

      if (false || vT.first == "<|eot_id|>" || vT.first == "<|im_end|>" || vT.first == "<|end|>" ||
          vT.first == "<|return|>"                                                             // o200k_harmony
          || vT.first == "<|call|>"                                                            // o200k_harmony
          || vT.first == "<|flush|>"                                                           // solar-open
          || vT.first == "<|calls|>"                                                           // solar-open
          || vT.first == "<end_of_turn>" || vT.first == "<|endoftext|>" || vT.first == "</s>"  // paddleocr
          || vT.first == "<|eom_id|>" || vT.first == "<EOT>" || vT.first == "_<EOT>" || vT.first == "[EOT]"  // Kimi-K2
          || vT.first == "[EOS]"                                                                             // Kimi-K2
          || vT.first == "<|end_of_text|>" || vT.first == "<end_of_utterance>"  // smoldocling
          || vT.first == "<eos>"                                                // gemma4
          || vT.first == "<turn|>"                                              // gemma4
          || vT.first == "<|tool_response>"                                     // gemma4
          || vT.first == "<｜end▁of▁sentence｜>"                                // deepseek-ocr
          || vT.first == "[e~["                                                 // minimax-m2/m3
          || vT.first == "<|ifm|im_end|>"                                       // k2-horizon
      ) {
        sSpecialEogIds.insert(vT.second);
        if ((vAttr & cDgsTokenAttrControl) == 0) {
          DGS_LOG_WARN(
            "%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type "
            "will be overridden\n",
            __func__, vT.second, vT.first.c_str());
          vAttr = (dgs_token_attr) (vAttr | cDgsTokenAttrControl);
        }
      } else {
        if (vAttr & cDgsTokenAttrControl && !(vAttr & cDgsTokenAttrUnused)) {
          // token is control, but not marked as EOG -> print a debug log
          if (sSpecialEogIds.count(vT.second) == 0) {
            DGS_LOG_DEBUG("%s: control token: %6d '%s' is not marked as EOG\n", __func__, vT.second, vT.first.c_str());
          }
        }
      }
    }

    // @ngxson : quick hack for gpt-oss, always render these tokens
    for (const auto & vT : dTokenToId) {
      auto & vAttr = lIdToToken[vT.second].vAttr;

      if (vT.first == "<|channel|>" || vT.first == "<|message|>" || vT.first == "<|start|>" ||
          vT.first == "<|constrain|>") {
        DGS_LOG_WARN("%s: setting token '%s' (%d) attribute to USER_DEFINED (%u), old attributes: %u\n", __func__,
                     vT.first.c_str(), vT.second, cDgsTokenAttrUserDefined, vAttr);

        vAttr = cDgsTokenAttrUserDefined;
      }
    }

    // sanity checks
    if (vSpecialEosId != DGS_TOKEN_NULL && sSpecialEogIds.count(vSpecialEosId) == 0) {
      sSpecialEogIds.insert(vSpecialEosId);
      DGS_LOG_WARN("%s: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect\n", __func__);
    }

    if (vSpecialEotId != DGS_TOKEN_NULL && sSpecialEogIds.count(vSpecialEotId) == 0) {
      sSpecialEogIds.insert(vSpecialEotId);
      DGS_LOG_WARN("%s: special_eot_id is not in special_eog_ids - the tokenizer config may be incorrect\n", __func__);
    }

    if (vSpecialEomId != DGS_TOKEN_NULL && sSpecialEogIds.count(vSpecialEomId) == 0) {
      sSpecialEogIds.insert(vSpecialEomId);
      DGS_LOG_WARN("%s: special_eom_id is not in special_eog_ids - the tokenizer config may be incorrect\n", __func__);
    }

    // TODO: workaround for o200k_harmony and solar-open tokenizer: the "<|end|>" token should not be EOG
    //       we don't have a good way to detect this, so for now, if we have "<|return|>" and "<|call|>" tokens ("<|calls|>" and "<|flush|>" for solar-open),
    //       we remove the "<|end|>" token from the EOG list
    {
      bool vHasReturn = false;
      bool vHasCall   = false;
      bool vHasEnd    = false;
      bool vHasFlush  = false;

      dgs_token vEndId = DGS_TOKEN_NULL;

      DGS_LOG_INFO("%s: printing all EOG tokens:\n", __func__);
      for (auto vTid : sSpecialEogIds) {
        if (vTid < 0 || vTid >= (dgs_token) lIdToToken.size()) {
          DGS_LOG_WARN("%s: EOG token id %d is out of range (vocab size %zu), skipping\n", __func__, vTid,
                       lIdToToken.size());
          continue;
        }
        auto & vText = lIdToToken[vTid].vText;

        DGS_LOG_INFO("%s:   - %d ('%s')\n", __func__, vTid, vText.c_str());

        if (vText == "<|return|>") {
          vHasReturn = true;
        } else if (vText == "<|call|>" || vText == "<|calls|>") {
          vHasCall = true;
        } else if (vText == "<|flush|>") {
          vHasFlush = true;
        } else if (vText == "<|end|>") {
          vHasEnd = true;
          vEndId  = vTid;
        }
      }

      if ((vHasReturn && vHasCall && vHasEnd) || (vHasCall && vHasFlush && vHasEnd)) {
        sSpecialEogIds.erase(vEndId);

        auto & vAttr = lIdToToken[vEndId].vAttr;
        vAttr        = cDgsTokenAttrUserDefined;

        DGS_LOG_WARN(
          "%s: special_eog_ids contains both '<|return|>' and '<|call|>', or '<|calls|>' and '<|flush|>' tokens, "
          "removing '<|end|>' token from EOG list\n",
          __func__);
      }
    }

    // gemma4 and plamo have a normal </s> token, unlike paddleocr
    {
      bool vHasNormalSMarker = false;
      bool vHasS             = false;

      dgs_token vSId = DGS_TOKEN_NULL;

      for (auto vTid : sSpecialEogIds) {
        if (vTid < 0 || vTid >= (dgs_token) lIdToToken.size()) {
          continue;
        }
        const auto & vText = lIdToToken[vTid].vText;
        if (vText == "<|tool_response>" || vText == "<|plamo:eos|>") {
          vHasNormalSMarker = true;
        } else if (vText == "</s>") {
          vHasS = true;
          vSId  = vTid;
        }
      }

      if (vHasNormalSMarker && vHasS) {
        sSpecialEogIds.erase(vSId);

        auto & vAttr = lIdToToken[vSId].vAttr;
        vAttr        = cDgsTokenAttrNormal;

        DGS_LOG_WARN("%s: '</s>' is a normal token here, removing it from EOG list\n", __func__);
      }
    }
  }

  // build special tokens cache
  {
    for (dgs_token vId = 0; vId < (dgs_token) cNTokens; ++vId) {
      if (lIdToToken[vId].vAttr & (cDgsTokenAttrControl | cDgsTokenAttrUserDefined | cDgsTokenAttrUnknown)) {
        lCacheSpecialTokens.push_back(vId);
      }
    }

    std::sort(lCacheSpecialTokens.begin(), lCacheSpecialTokens.end(), [&](const dgs_token pA, const dgs_token pB) {
      return lIdToToken[pA].vText.size() > lIdToToken[pB].vText.size();
    });

    DGS_LOG_INFO("%s: special tokens cache size = %u\n", __func__, (uint32_t) lCacheSpecialTokens.size());
  }

  // build token to piece cache
  {
    size_t vSizeCache = 0;

    std::vector<std::string> lCache(cNTokens);

    for (uint32_t vId = 0; vId < cNTokens; ++vId) {
      lCache[vId] = mTokenToPieceForCache(vId, true);

      vSizeCache += lCache[vId].size();
    }

    std::swap(lCacheTokenToPiece, lCache);

    DGS_LOG_INFO("%s: token to piece cache size = %.4f MB\n", __func__, vSizeCache / 1024.0 / 1024.0);
  }

  // Handle per token attributes
  //NOTE: Each model customizes per token attributes.
  //NOTE: Per token attributes are missing from the GGUF file.
  //TODO: Extract attributes from GGUF file.
  {
    auto vContainsAny = [](const std::string & str, const std::vector<std::string_view> & pSubstrs) -> bool {
      for (const auto & substr : pSubstrs) {
        if (str.find(substr) != std::string::npos) {
          return true;
        }
      }
      return false;
    };

    auto vSetTokenidAttr = [&](const dgs_token pId, dgs_token_attr pAttr, bool value) {
      uint32_t vCurrent     = lIdToToken.at(pId).vAttr;
      vCurrent              = value ? (vCurrent | pAttr) : (vCurrent & ~pAttr);
      lIdToToken[pId].vAttr = (dgs_token_attr) vCurrent;
    };

    auto vSetTokenAttr = [&](const std::string & token, dgs_token_attr pAttr, bool value) {
      vSetTokenidAttr(dTokenToId.at(token), pAttr, value);
    };

    std::string vModelName;
    std::string vTokenizerPre;
    std::string vGeneralArch;

    pMl.get_key(cLlmKvGeneralName, vModelName, false);
    pMl.get_key(cLlmKvTokenizerPre, vTokenizerPre, false);
    pMl.get_key(cLlmKvGeneralArchitecture, vGeneralArch, false);

    // model name to lowercase
    std::transform(vModelName.begin(), vModelName.end(), vModelName.begin(),
                   [](const std::string::value_type pX) { return std::tolower(pX); });

    // set attributes by model/tokenizer/architecture name
    if (false || vContainsAny(vTokenizerPre, { "jina-v2-de", "jina-v2-es", "jina-v2-code", "mmbert" }) ||
        vContainsAny(vGeneralArch, { "nomic-bert-moe", "jina-bert-v3" })) {
      if (dTokenToId.count("<mask>") == 0) {
        DGS_LOG_WARN("%s: Mask token is missing in vocab, please reconvert model!\n", __func__);
      } else {
        vSetTokenAttr("<mask>", cDgsTokenAttrLstrip, true);
      }
    } else if (vContainsAny(vModelName, { "phi-3", "phi3" })) {
      for (auto vId : lCacheSpecialTokens) {
        vSetTokenidAttr(vId, cDgsTokenAttrRstrip, true);
      }
      for (const auto * token : { "</s>" }) {
        vSetTokenAttr(token, cDgsTokenAttrRstrip, true);
      }
      for (const auto * token : { "<unk>", "<s>", "<|endoftext|>" }) {
        vSetTokenAttr(token, cDgsTokenAttrRstrip, false);
      }
    } else if (vContainsAny(vModelName, { "modern-bert" })) {
      if (dTokenToId.count("[MASK]") == 0) {
        DGS_LOG_WARN("%s: Mask token missing in vocab!\n", __func__);
      } else {
        vSetTokenAttr("[MASK]", cDgsTokenAttrLstrip, true);
      }
    }
  }
}

enum fDgsVocabType dgs_vocab::impl::mGetType() const {
  return type;
}

std::string dgs_vocab::impl::mTypeName() const {
  switch (type) {
    case cDgsVocabTypeNone:
      return "no vocab";
    case cDgsVocabTypeSpm:
      return "SPM";
    case cDgsVocabTypeBpe:
      return "BPE";
    case cDgsVocabTypeWpm:
      return "WPM";
    case cDgsVocabTypeUgm:
      return "UGM";
    case cDgsVocabTypeRwkv:
      return "RWKV";
    case cDgsVocabTypePlamo2:
      return "PLaMo2";
    case cDgsVocabTypePlamo3:
      return "PLaMo3";
    case cDgsVocabTypeTest:
      return "TEST";
    default:
      return "unknown";
  }
}

bool dgs_vocab::impl::mIsNormal(dgs_token pId) const {
  GGML_ASSERT(type != cDgsVocabTypeNone);
  return lIdToToken[pId].vAttr & cDgsTokenAttrNormal;
}

bool dgs_vocab::impl::mIsUnknown(dgs_token pId) const {
  GGML_ASSERT(type != cDgsVocabTypeNone);
  return lIdToToken[pId].vAttr & cDgsTokenAttrUnknown;
}

bool dgs_vocab::impl::mIsControl(dgs_token pId) const {
  GGML_ASSERT(type != cDgsVocabTypeNone);
  return lIdToToken[pId].vAttr & cDgsTokenAttrControl;
}

bool dgs_vocab::impl::mIsByte(dgs_token pId) const {
  GGML_ASSERT(type != cDgsVocabTypeNone);
  return lIdToToken[pId].vAttr & cDgsTokenAttrByte;
}

bool dgs_vocab::impl::mIsUserDefined(dgs_token pId) const {
  GGML_ASSERT(type != cDgsVocabTypeNone);
  return lIdToToken[pId].vAttr & cDgsTokenAttrUserDefined;
}

bool dgs_vocab::impl::mIsUnused(dgs_token pId) const {
  GGML_ASSERT(type != cDgsVocabTypeNone);
  return lIdToToken[pId].vAttr & cDgsTokenAttrUnused;
}

bool dgs_vocab::impl::mIsEog(dgs_token pId) const {
  return pId != DGS_TOKEN_NULL && sSpecialEogIds.count(pId) > 0;
}

uint8_t dgs_vocab::impl::mTokenToByte(dgs_token pId) const {
  GGML_ASSERT(mGetType() != cDgsVocabTypeNone);
  GGML_ASSERT(mIsByte(pId));
  const auto & vTokenData = lIdToToken.at(pId);
  switch (mGetType()) {
    case cDgsVocabTypeSpm:
    case cDgsVocabTypeUgm:
      {
        auto vBuf = vTokenData.vText.substr(3, 2);
        return strtol(vBuf.c_str(), NULL, 16);
      }
    case cDgsVocabTypeBpe:
      {
        // Gemma4 uses BPE with SPM-style byte fallback tokens (<0xXX>)
        auto vBuf = vTokenData.vText.substr(3, 2);
        return strtol(vBuf.c_str(), NULL, 16);
      }
    case cDgsVocabTypeWpm:
      {
        GGML_ABORT("fatal error");
      }
    default:
      GGML_ABORT("fatal error");
  }
}

dgs_token_attr dgs_vocab::impl::mTokenGetAttr(dgs_token pId) const {
  GGML_ASSERT(type != cDgsVocabTypeNone);
  return lIdToToken.at(pId).vAttr;
}

void dgs_vocab::impl::mInitTokenizer(enum fDgsVocabType type) {
  DGS_LOG_DEBUG("%s: initializing tokenizer for type %d\n", __func__, type);

  switch (type) {
    case cDgsVocabTypeSpm:
      vTokenizer = std::make_unique<llm_tokenizer_spm>(vVocab);
      break;
    case cDgsVocabTypeBpe:
      vTokenizer = std::make_unique<llm_tokenizer_bpe>(vVocab);
      break;
    case cDgsVocabTypeWpm:
      vTokenizer = std::make_unique<llm_tokenizer_wpm>(vVocab);
      break;
    case cDgsVocabTypeUgm:
      vTokenizer = std::make_unique<llm_tokenizer_ugm>(vVocab, lPrecompiledCharsmap);
      break;
    case cDgsVocabTypeRwkv:
      vTokenizer = std::make_unique<llm_tokenizer_rwkv>(vVocab);
      break;
    case cDgsVocabTypePlamo2:
      vTokenizer = std::make_unique<llm_tokenizer_plamo2>(vVocab, false);
      break;
    case cDgsVocabTypePlamo3:
      vTokenizer = std::make_unique<llm_tokenizer_plamo2>(vVocab, true);
      break;
    case cDgsVocabTypeTest:
      vTokenizer = std::make_unique<llm_tokenizer>();
      break;
    default:
      GGML_ABORT("unsupported vocab type");
  }
}

//
// (de-) tokenize
//

// #define PRETOKENIZERDEBUG

void dgs_vocab::impl::mTokenizerStPartition(std::forward_list<fragment_buffer_variant> & pBuffer,
                                            bool                                         pParseSpecial) const {
  // for each special token
  for (const dgs_token cSpecialId : lCacheSpecialTokens) {
    const auto & data  = vVocab.mGetTokenData(cSpecialId);
    const auto & vText = data.vText;

    if (!pParseSpecial && (data.vAttr & (cDgsTokenAttrControl | cDgsTokenAttrUnknown))) {
      // Ignore control and unknown tokens when parse_special == false
      continue;
      // User-defined tokens are still pre-tokenized before everything else
      // ref: https://github.com/huggingface/tokenizers/blob/fdd26ba9a3f0c133427aab0423888cbde91362d7/tokenizers/src/tokenizer/mod.rs#L726
      // This is mostly relevant for neox-style tokenizers (mpt, olmo, stablelm, etc.)
    }

    // for each text fragment
    std::forward_list<fragment_buffer_variant>::iterator vIt = pBuffer.begin();
    while (vIt != pBuffer.end()) {
      auto & vFragment = (*vIt);

      // if a fragment is text ( not yet processed )
      if (vFragment.type == cFragmentBufferVariantTypeRawText) {
        const auto & vRawText = vFragment.vRawText;

        auto vRawTextBaseOffset = vFragment.cOffset;
        auto vRawTextBaseLength = vFragment.length;

        // loop over the text
        while (true) {
          // find the first occurrence of a given special token in this fragment
          //  passing offset argument only limit the "search area" but match coordinates
          //  are still relative to the source full raw_text
          //  string_view begins at pos 0 for the same reason
          auto vMatch =
            std::string_view(vRawText.data(), vRawTextBaseOffset + vRawTextBaseLength).find(vText, vRawTextBaseOffset);

          // no occurrences found, stop processing this fragment for a given special token
          if (vMatch == std::string::npos) {
            break;
          }

#ifdef PRETOKENIZERDEBUG
          DGS_LOG_WARN("FF: (%ld %ld %ld) '%s'\n", raw_text->length(), raw_text_base_offset, raw_text_base_length,
                       raw_text->substr(raw_text_base_offset, raw_text_base_length).c_str());
#endif
          auto vSource = std::distance(pBuffer.begin(), vIt);

          // if match is further than base offset
          //  then we have some text to the left of it
          if (vMatch > vRawTextBaseOffset) {
            // left
            const int64_t cLeftReminderOffset = vRawTextBaseOffset + 0;
            int64_t       vLeftReminderLength = vMatch - vRawTextBaseOffset;

            if (data.vAttr & cDgsTokenAttrLstrip) {
              while (vLeftReminderLength > 0 && isspace(vRawText[cLeftReminderOffset + vLeftReminderLength - 1])) {
                vLeftReminderLength--;
              }
            }

            if (vLeftReminderLength > 0) {
              pBuffer.emplace_after(vIt, vRawText, cLeftReminderOffset, vLeftReminderLength);
              vIt++;
            }

#ifdef PRETOKENIZERDEBUG
            DGS_LOG_WARN("FL: (%ld %ld) '%s'\n", left_reminder_offset, left_reminder_length,
                         raw_text->substr(left_reminder_offset, left_reminder_length).c_str());
#endif
          }

          // special token
          pBuffer.emplace_after(vIt, cSpecialId);
          vIt++;

          // right
          if (vMatch + vText.length() < vRawTextBaseOffset + vRawTextBaseLength) {
            int64_t vRightReminderOffset = vMatch + vText.length();
            int64_t vRightReminderLength = vRawTextBaseLength - ((vMatch - vRawTextBaseOffset) + vText.length());

            if (data.vAttr & cDgsTokenAttrRstrip) {
              while (vRightReminderLength > 0 && isspace(vRawText[vRightReminderOffset])) {
                vRightReminderOffset++;
                vRightReminderLength--;
              }
            }

            if (vRightReminderLength > 0) {
              pBuffer.emplace_after(vIt, vRawText, vRightReminderOffset, vRightReminderLength);
              vIt++;
            }

#ifdef PRETOKENIZERDEBUG
            DGS_LOG_WARN("FR: (%ld %ld) '%s'\n", right_reminder_offset, right_reminder_length,
                         raw_text->substr(right_reminder_offset, right_reminder_length).c_str());
#endif

            if (vSource == 0) {
              pBuffer.erase_after(pBuffer.before_begin());
            } else {
              pBuffer.erase_after(std::next(pBuffer.begin(), (vSource - 1)));
            }

            // repeat for the right side
            vRawTextBaseOffset = vRightReminderOffset;
            vRawTextBaseLength = vRightReminderLength;

#ifdef PRETOKENIZERDEBUG
            DGS_LOG_WARN("RR: (%ld %ld) '%s'\n", raw_text_base_offset, raw_text_base_length,
                         raw_text->substr(raw_text_base_offset, raw_text_base_length).c_str());
#endif
          } else {
            if (vSource == 0) {
              pBuffer.erase_after(pBuffer.before_begin());
            } else {
              pBuffer.erase_after(std::next(pBuffer.begin(), (vSource - 1)));
            }
            break;
          }
        }
      }
      vIt++;
    }
  }
}

// NOTE: avoid ever using this except for building the token_to_piece caches
std::string dgs_vocab::impl::mTokenToPieceForCache(dgs_token token, bool pSpecial) const {
  std::string vPiece;
  vPiece.resize(vPiece.capacity());  // using string internal cache
  const int cNChars = vVocab.mTokenToPiece(token, &vPiece[0], vPiece.size(), 0, pSpecial);
  if (cNChars < 0) {
    vPiece.resize(-cNChars);
    int vCheck = vVocab.mTokenToPiece(token, &vPiece[0], vPiece.size(), 0, pSpecial);
    GGML_ASSERT(vCheck == -cNChars);
  } else {
    vPiece.resize(cNChars);
  }

  return vPiece;
}

static void fDgsEscapeWhitespace(std::string & pText) {
  fReplaceAll(pText, " ", "\xe2\x96\x81");
}

static void fDgsUnescapeWhitespace(std::string & pWord) {
  fReplaceAll(pWord, "\xe2\x96\x81", " ");
}

static std::string fDgsDecodeText(const std::string & pText) {
  std::string vDecodedText;

  const auto lCpts = fUnicodeCptsFromUtf8(pText);
  for (const auto cCpt : lCpts) {
    const auto cUtf8 = fUnicodeCptToUtf8(cCpt);
    try {
      vDecodedText += fUnicodeUtf8ToByte(cUtf8);
    } catch (const std::out_of_range & /*e*/) {
      vDecodedText += "[UNK_BYTE_0x";
      for (const auto cC : cUtf8) {
        vDecodedText += format("%02x", (uint8_t) cC);
      }
      vDecodedText += pText + "]";
    }
  }

  return vDecodedText;
}

std::vector<dgs_token> dgs_vocab::impl::mTokenize(const std::string & pRawText,
                                                  bool                pAddSpecial,
                                                  bool                pParseSpecial) const {
  GGML_ASSERT(vTokenizer && "Tokenizer not initialized. Call dgs_vocab::init_tokenizer() first.");

  std::vector<dgs_token>                     lOutput;
  std::forward_list<fragment_buffer_variant> vFragmentBuffer;

  if (!pRawText.empty()) {
    vFragmentBuffer.emplace_front(pRawText, 0, pRawText.length());
    mTokenizerStPartition(vFragmentBuffer, pParseSpecial);
  }

  switch (mGetType()) {
    case cDgsVocabTypeSpm:
      {
        // OG tokenizer behavior:
        //
        // tokenizer.encode('', add_special_tokens=True)  returns [1]
        // tokenizer.encode('', add_special_tokens=False) returns []

        bool vIsPrevSpecial = true;  // prefix with space if first token

        if (pAddSpecial && vAddBos) {
          GGML_ASSERT(vSpecialBosId != DGS_TOKEN_NULL);
          lOutput.push_back(vSpecialBosId);
          vIsPrevSpecial = true;
        }

        for (const auto & vFragment : vFragmentBuffer) {
          if (vFragment.type == cFragmentBufferVariantTypeRawText) {
            std::string vText;

            // prefix with space if previous is special
            if (vAddSpacePrefix && vIsPrevSpecial) {
              vText = ' ';
            }

            vText += vFragment.vRawText.substr(vFragment.cOffset, vFragment.length);

#ifdef PRETOKENIZERDEBUG
            DGS_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", text.length(), fragment.offset, fragment.length, text.c_str());
#endif
            fDgsEscapeWhitespace(vText);
            llm_tokenizer_spm_session vSession(vVocab);
            vSession.mTokenize(vText, lOutput);
            vIsPrevSpecial = false;
          } else {  // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
            lOutput.push_back(vFragment.token);
            vIsPrevSpecial = true;
          }
        }

        if (pAddSpecial && vAddBos && lOutput.size() >= 2 && lOutput[1] == vSpecialBosId) {
          DGS_LOG_WARN(
            "%s: Added a BOS token to the prompt as specified by the model but the prompt "
            "also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. "
            "Are you sure this is what you want?\n",
            __FUNCTION__);
        }

        if (pAddSpecial && vAddEos) {
          GGML_ASSERT(vSpecialEosId != DGS_TOKEN_NULL);
          lOutput.push_back(vSpecialEosId);
        }
      }
      break;
    case cDgsVocabTypeBpe:
      {
        // it calls some other methods that are not exist in llm_tokenizer,
        // here just cast it to bpe tokenizer object
        const llm_tokenizer_bpe * vTokBpe = static_cast<const llm_tokenizer_bpe *>(vTokenizer.get());

        std::unique_ptr<llm_tokenizer_bpe_session> vSession;
        if (vVocab.mGetTokenizerModel() == "hybriddna") {
          vSession = std::make_unique<llm_tokenizer_hybriddna_session>(vVocab, *vTokBpe);
        } else if (vVocab.mGetTokenizerModel() == "whitespace") {
          vSession = std::make_unique<llm_tokenizer_whitespace_session>(vVocab, *vTokBpe);
        } else {
          vSession = std::make_unique<llm_tokenizer_bpe_session>(vVocab, *vTokBpe);
        }

        if (pAddSpecial) {
          vSession->mAppendBos(lOutput);
        }
        for (const auto & vFragment : vFragmentBuffer) {
          if (vFragment.type == cFragmentBufferVariantTypeRawText) {
            std::string vText = vFragment.vRawText.substr(vFragment.cOffset, vFragment.length);

            if (vEscapeWhitespaces) {
              fDgsEscapeWhitespace(vText);
            }

#ifdef PRETOKENIZERDEBUG
            DGS_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", text.length(), fragment.offset, fragment.length, text.c_str());
#endif
            vSession->mTokenize(vText, lOutput);
          } else {  // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
            vSession->mAppend(vFragment.token, lOutput);
          }
        }

        if (pAddSpecial) {
          vSession->mAppendEos(lOutput);
          vSession->mCheckDoubleBosEos(lOutput);
        }
      }
      break;
    case cDgsVocabTypeWpm:
      {
        if (pAddSpecial) {
          GGML_ASSERT(vSpecialBosId != DGS_TOKEN_NULL);
          lOutput.push_back(vSpecialBosId);
        }

        llm_tokenizer_wpm_session vSession(vVocab);

        for (const auto & vFragment : vFragmentBuffer) {
          if (vFragment.type == cFragmentBufferVariantTypeRawText) {
            std::string vText = vFragment.vRawText.substr(vFragment.cOffset, vFragment.length);

#ifdef PRETOKENIZERDEBUG
            DGS_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", text.length(), fragment.offset, fragment.length, text.c_str());
#endif
            vSession.mTokenize(vText, lOutput);
          } else {  // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
            lOutput.push_back(vFragment.token);
          }
        }

        if (pAddSpecial) {
          GGML_ASSERT(vSpecialSepId != DGS_TOKEN_NULL);
          lOutput.push_back(vSpecialSepId);
        }
      }
      break;
    case cDgsVocabTypeUgm:
      {
        if (pAddSpecial && vAddBos) {
          GGML_ASSERT(vSpecialBosId != DGS_TOKEN_NULL);
          lOutput.push_back(vSpecialBosId);
        }
        llm_tokenizer_ugm_session vSession(vVocab, *static_cast<const llm_tokenizer_ugm *>(vTokenizer.get()));

        for (const auto & vFragment : vFragmentBuffer) {
          if (vFragment.type == cFragmentBufferVariantTypeRawText) {
            std::string vText = vFragment.vRawText.substr(vFragment.cOffset, vFragment.length);
#ifdef PRETOKENIZERDEBUG
            DGS_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", text.length(), fragment.offset, fragment.length, text.c_str());
#endif
            vSession.mTokenize(vText, lOutput);
          } else {  // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
            lOutput.push_back(vFragment.token);
          }
        }

        if (pAddSpecial && vAddBos && lOutput.size() >= 2 && lOutput[1] == vSpecialBosId) {
          DGS_LOG_WARN(
            "%s: Added a BOS token to the prompt as specified by the model but the prompt "
            "also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. "
            "Are you sure this is what you want?\n",
            __FUNCTION__);
        }

        if (pAddSpecial && vAddEos) {
          GGML_ASSERT(vSpecialEosId != DGS_TOKEN_NULL);
          lOutput.push_back(vSpecialEosId);
        }
      }
      break;
    case cDgsVocabTypeRwkv:
      {
        llm_tokenizer_rwkv_session vSession(vVocab, *static_cast<const llm_tokenizer_rwkv *>(vTokenizer.get()));
        for (const auto & vFragment : vFragmentBuffer) {
          if (vFragment.type == cFragmentBufferVariantTypeRawText) {
            std::string vText = vFragment.vRawText.substr(vFragment.cOffset, vFragment.length);

#ifdef PRETOKENIZERDEBUG
            DGS_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", text.length(), fragment.offset, fragment.length, text.c_str());
#endif

            vSession.mTokenize(vText, lOutput);
          } else {  // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
            lOutput.push_back(vFragment.token);
          }
        }
      }
      break;
    case cDgsVocabTypePlamo2:
    case cDgsVocabTypePlamo3:
      {
        if (pAddSpecial && vAddBos) {
          GGML_ASSERT(vSpecialBosId != DGS_TOKEN_NULL);
          lOutput.push_back(vSpecialBosId);
        }
        llm_tokenizer_plamo2_session vSession(*static_cast<const llm_tokenizer_plamo2 *>(vTokenizer.get()));
        for (const auto & vFragment : vFragmentBuffer) {
          if (vFragment.type == cFragmentBufferVariantTypeRawText) {
            std::string vText = vFragment.vRawText.substr(vFragment.cOffset, vFragment.length);

#ifdef PRETOKENIZERDEBUG
            DGS_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", text.length(), fragment.offset, fragment.length, text.c_str());
#endif

            vSession.mTokenize(vText, lOutput);
          } else {  // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
            lOutput.push_back(vFragment.token);
          }
        }

        if (pAddSpecial && vAddBos && lOutput.size() >= 2 && lOutput[1] == vSpecialBosId) {
          DGS_LOG_WARN(
            "%s: Added a BOS token to the prompt as specified by the model but the prompt "
            "also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. "
            "Are you sure this is what you want?\n",
            __FUNCTION__);
        }

        if (pAddSpecial && vAddEos) {
          GGML_ASSERT(vSpecialEosId != DGS_TOKEN_NULL);
          lOutput.push_back(vSpecialEosId);
        }
      }
      break;
    case cDgsVocabTypeTest:
      {
        const uint32_t   cNVocab    = vVocab.n_tokens();
        constexpr size_t cChunkSize = 5;

        // reserve output to avoid repeated reallocations
        size_t vNTokens = 0;
        for (const auto & vFragment : vFragmentBuffer) {
          if (vFragment.type == cFragmentBufferVariantTypeRawText) {
            vNTokens += (vFragment.length + cChunkSize - 1) / cChunkSize;
          } else {
            ++vNTokens;
          }
        }
        lOutput.reserve(lOutput.size() + vNTokens);

        for (const auto & vFragment : vFragmentBuffer) {
          if (vFragment.type == cFragmentBufferVariantTypeRawText) {
            const auto & vText = vFragment.vRawText;
            const size_t begin = vFragment.cOffset;
            const size_t end   = begin + vFragment.length;
            size_t       vPos  = begin;
            while (vPos < end) {
              const size_t n    = std::min(cChunkSize, end - vPos);
              uint64_t     hash = 0;
              for (size_t vI = 0; vI < n; ++vI) {
                hash = hash * 31 + (uint8_t) vText[vPos + vI];
              }
              lOutput.push_back((dgs_token) (hash % cNVocab));
              vPos += n;
            }
          } else {  // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
            lOutput.push_back(vFragment.token);
          }
        }
      }
      break;
    case cDgsVocabTypeNone:
      GGML_ABORT("fatal error");
  }

  return lOutput;
}

int32_t dgs_vocab::impl::mTokenToPiece(dgs_token token,
                                       char *    buf,
                                       int32_t   length,
                                       int32_t   lstrip,
                                       bool      pSpecial) const {
  static const int     cAttrSpecial = cDgsTokenAttrUnknown | cDgsTokenAttrControl;
  const dgs_token_attr cAttr        = mTokenGetAttr(token);
  if (!pSpecial && (cAttr & cAttrSpecial)) {
    return 0;
  }

  // copy piece chars to output text buffer
  // skip up to 'lstrip' leading spaces before copying
  auto vTryCopy = [=](const char * token, size_t size) -> int32_t {
    if (size >= static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
      GGML_ABORT("invalid token size: %zu exceeds int32_t limit", size);
    }

    for (int32_t vI = 0; vI < lstrip && size && *token == ' '; ++vI) {
      token++;
      size--;
    }
    if (length < (int32_t) size) {
      return -(int32_t) size;
    }
    memcpy(buf, token, size);
    return (int32_t) size;
  };

  // if we have a cache - use it
  {
    const auto & lCache = lCacheTokenToPiece;

    if (!lCache.empty()) {
      const auto & vResult = lCache.at(token);
      return vTryCopy(vResult.data(), vResult.size());
    }
  }

  if (0 <= token && token < (int32_t) lIdToToken.size()) {
    const std::string & vTokenText = lIdToToken[token].vText;
    switch (mGetType()) {
      case cDgsVocabTypeWpm:
      case cDgsVocabTypeSpm:
      case cDgsVocabTypeUgm:
        {
          // NOTE: we accept all unsupported token types,
          // suppressing them like CONTROL tokens.
          if (cAttr & (cAttrSpecial | cDgsTokenAttrUserDefined)) {
            return vTryCopy(vTokenText.data(), vTokenText.size());
          }
          if (cAttr & cDgsTokenAttrNormal) {
            std::string vResult = vTokenText;
            fDgsUnescapeWhitespace(vResult);
            return vTryCopy(vResult.data(), vResult.size());
          }
          if (cAttr & cDgsTokenAttrByte) {
            char vByte = (char) mTokenToByte(token);
            return vTryCopy((char *) &vByte, 1);
          }
          break;
        }
      case cDgsVocabTypeBpe:
        {
          // NOTE: we accept all unsupported token types,
          // suppressing them like CONTROL tokens.
          if (cAttr & (cAttrSpecial | cDgsTokenAttrUserDefined)) {
            return vTryCopy(vTokenText.data(), vTokenText.size());
          }
          if (cAttr & cDgsTokenAttrNormal) {
            if (vEscapeWhitespaces) {
              // SPM-style BPE: tokens contain ▁ for spaces
              std::string vResult = vTokenText;
              fDgsUnescapeWhitespace(vResult);
              return vTryCopy(vResult.data(), vResult.size());
            }
            std::string vResult = fDgsDecodeText(vTokenText);
            return vTryCopy(vResult.data(), vResult.size());
          }
          if (cAttr & cDgsTokenAttrByte) {
            char vByte = (char) mTokenToByte(token);
            return vTryCopy((char *) &vByte, 1);
          }
          break;
        }
      case cDgsVocabTypeRwkv:
        {
          std::vector<uint8_t> lResult = fDgsUnescapeRwkvToken(vTokenText);

          // If we don't have enough space, return an error
          if (lResult.size() > (size_t) length) {
            return -(int) lResult.size();
          }

          memcpy(buf, lResult.data(), lResult.size());
          return (int) lResult.size();
        }
      case cDgsVocabTypeTest:
        {
          // tokens -> text: simply stringify the token id in hex
          std::string vResult = format("%x", token);
          return vTryCopy(vResult.data(), vResult.size());
        }
      case cDgsVocabTypePlamo2:
      case cDgsVocabTypePlamo3:
        {
          // PLaMo uses similar token handling as BPE/SPM
          if (vVocab.mIsByte(token)) {
            // Handle byte tokens like <0xXX>
            if (vTokenText.length() == 6 && vTokenText.substr(0, 3) == "<0x" && vTokenText.back() == '>') {
              std::string vHexStr = vTokenText.substr(3, 2);
              if (std::isxdigit(static_cast<unsigned char>(vHexStr[0])) &&
                  std::isxdigit(static_cast<unsigned char>(vHexStr[1]))) {
                int vHexVal = std::stoi(vHexStr, nullptr, 16);
                if (length < 1) {
                  return -1;
                }
                buf[0] = static_cast<char>(vHexVal);
                return 1;
              }
            }
          }

          // Normal token - just copy the text
          std::string vResult = vTokenText;
          return vTryCopy(vResult.data(), vResult.size());
        }
      default:
        GGML_ABORT("fatal error");
    }
  }

  return 0;
}

const std::string & dgs_vocab::impl::mTokenToPiece(dgs_token token) const {
  return lCacheTokenToPiece.at(token);
}

int32_t dgs_vocab::impl::mDetokenize(const dgs_token * pTokens,
                                     int32_t           pNTokens,
                                     char *            pText,
                                     int32_t           pTextLenMax,
                                     bool              pRemoveSpecial,
                                     bool              pUnparseSpecial) const {
  if (type == cDgsVocabTypeNone) {
    return 0;
  }

  GGML_ASSERT(vTokenizer && "Tokenizer not initialized. Call dgs_vocab::init_tokenizer() first.");

  int32_t vAvail = pTextLenMax;
  int32_t vTotal = 0;

  // remove the leading space
  bool vRemoveSpace = vAddSpacePrefix;

  if (pRemoveSpecial && vAddBos) {
    if (pNTokens > 0 && pTokens[0] == vSpecialBosId) {
      vRemoveSpace = false;
      pNTokens--;
      pTokens++;
    }
  }

  if (pRemoveSpecial && vAddEos) {
    if (pNTokens > 0 && pTokens[pNTokens - 1] == vSpecialEosId) {
      pNTokens--;
    }
  }

  for (int32_t vI = 0; vI < pNTokens; ++vI) {
    GGML_ASSERT(vAvail >= 0);
    int32_t vNChars = mTokenToPiece(pTokens[vI], pText, vAvail, vRemoveSpace, pUnparseSpecial);
    vRemoveSpace    = false;
    if (vNChars < 0) {
      vAvail = 0;
      vTotal -= vNChars;
    } else if (vNChars > 0) {
      vAvail -= vNChars;
      pText += vNChars;
      vTotal += vNChars;
    }
  }

  if (vTotal > pTextLenMax) {
    return -vTotal;
  }

  if (vCleanSpaces) {
    pText -= vTotal;  // restart text

    // first pass: characters ?!.,  //TODO: where do these characters come from?
    const int32_t cTotal1 = vTotal;
    vTotal                = vTotal ? 1 : 0;
    for (int32_t vI = 1; vI < cTotal1; ++vI) {
      const char cX = pText[vI];
      if (pText[vI - 1] == ' ') {
        if (cX == '?' || cX == '!' || cX == '.' || cX == ',') {  // " ?", " !", " .", " ,"
          vTotal--;                                              // remove space
        }
      }
      pText[vTotal++] = cX;
    }

    // second pass: strip single apostrophe between spaces
    const int32_t cTotal2 = vTotal;
    vTotal                = vTotal ? 1 : 0;
    for (int32_t vI = 1; vI < cTotal2; ++vI) {
      const char cX = pText[vI];
      if (cX == '\'' && vI + 1 < cTotal2 && pText[vI - 1] == ' ' && pText[vI + 1] == ' ') {  // " ' "
        vTotal--;                                                                            // remove prev space
        pText[++vI] = '\0';                                                                  // remove next space
      }
      pText[vTotal++] = cX;
    }

    // third pass: apostrophe contractions  //NOTE: this makes sense?
    const int32_t cTotal3 = vTotal;
    vTotal                = vTotal ? 1 : 0;
    for (int32_t vI = 1; vI < cTotal3; ++vI) {
      const char cX = pText[vI];
      if (pText[vI - 1] == ' ') {
        if (cX == '\'' && vI + 1 < cTotal3) {
          const char cX1 = pText[vI + 1];
          if (cX1 == 't' || cX1 == 'd') {         // " 't", " 'd"
                                                  //total--;  // remove space
          } else if (cX1 == 's' || cX1 == 'm') {  // " 's", " 'm"
            vTotal--;                             // remove space
          } else if (vI + 2 < cTotal3) {
            const char cX2 = pText[vI + 2];
            if ((cX1 == 'l' && cX2 == 'l')) {                                       // " 'll"
                                                                                    //total--;  // remove space
            } else if ((cX1 == 'r' && cX2 == 'e') || (cX1 == 'v' && cX2 == 'e')) {  // " 're", " 've"
              vTotal--;                                                             // remove space
            } else {
              //total--;  // remove space
            }
          } else {
            //total--;  // remove space
          }
        }
      }
      pText[vTotal++] = cX;
    }
  }

  return vTotal <= pTextLenMax ? vTotal : -vTotal;
}

void dgs_vocab::impl::mPrintInfo() const {
  DGS_LOG_INFO("%s: vocab type            = %s\n", __func__, mTypeName().c_str());
  DGS_LOG_INFO("%s: n_vocab               = %u\n", __func__, vVocab.n_tokens());
  DGS_LOG_INFO("%s: n_merges              = %u\n", __func__, (uint32_t) dBpeRanks.size());

  // special tokens
  if (vSpecialBosId != DGS_TOKEN_NULL) {
    DGS_LOG_INFO("%s: BOS token             = %d '%s'\n", __func__, vSpecialBosId,
                 lIdToToken.at(vSpecialBosId).vText.c_str());
  }
  if (vSpecialEosId != DGS_TOKEN_NULL) {
    DGS_LOG_INFO("%s: EOS token             = %d '%s'\n", __func__, vSpecialEosId,
                 lIdToToken.at(vSpecialEosId).vText.c_str());
  }
  if (vSpecialEotId != DGS_TOKEN_NULL) {
    DGS_LOG_INFO("%s: EOT token             = %d '%s'\n", __func__, vSpecialEotId,
                 lIdToToken.at(vSpecialEotId).vText.c_str());
  }
  if (vSpecialEomId != DGS_TOKEN_NULL) {
    DGS_LOG_INFO("%s: EOM token             = %d '%s'\n", __func__, vSpecialEomId,
                 lIdToToken.at(vSpecialEomId).vText.c_str());
  }
  if (vSpecialUnkId != DGS_TOKEN_NULL) {
    DGS_LOG_INFO("%s: UNK token             = %d '%s'\n", __func__, vSpecialUnkId,
                 lIdToToken.at(vSpecialUnkId).vText.c_str());
  }
  if (vSpecialSepId != DGS_TOKEN_NULL) {
    DGS_LOG_INFO("%s: SEP token             = %d '%s'\n", __func__, vSpecialSepId,
                 lIdToToken.at(vSpecialSepId).vText.c_str());
  }
  if (vSpecialPadId != DGS_TOKEN_NULL) {
    DGS_LOG_INFO("%s: PAD token             = %d '%s'\n", __func__, vSpecialPadId,
                 lIdToToken.at(vSpecialPadId).vText.c_str());
  }
  if (vSpecialMaskId != DGS_TOKEN_NULL) {
    DGS_LOG_INFO("%s: MASK token            = %d '%s'\n", __func__, vSpecialMaskId,
                 lIdToToken.at(vSpecialMaskId).vText.c_str());
  }

  if (vLinefeedId != DGS_TOKEN_NULL) {
    DGS_LOG_INFO("%s: LF token              = %d '%s'\n", __func__, vLinefeedId,
                 lIdToToken.at(vLinefeedId).vText.c_str());
  }

  if (vSpecialFimPreId != DGS_TOKEN_NULL) {
    DGS_LOG_INFO("%s: FIM PRE token         = %d '%s'\n", __func__, vSpecialFimPreId,
                 lIdToToken.at(vSpecialFimPreId).vText.c_str());
  }
  if (vSpecialFimSufId != DGS_TOKEN_NULL) {
    DGS_LOG_INFO("%s: FIM SUF token         = %d '%s'\n", __func__, vSpecialFimSufId,
                 lIdToToken.at(vSpecialFimSufId).vText.c_str());
  }
  if (vSpecialFimMidId != DGS_TOKEN_NULL) {
    DGS_LOG_INFO("%s: FIM MID token         = %d '%s'\n", __func__, vSpecialFimMidId,
                 lIdToToken.at(vSpecialFimMidId).vText.c_str());
  }
  if (vSpecialFimPadId != DGS_TOKEN_NULL) {
    DGS_LOG_INFO("%s: FIM PAD token         = %d '%s'\n", __func__, vSpecialFimPadId,
                 lIdToToken.at(vSpecialFimPadId).vText.c_str());
  }
  if (vSpecialFimRepId != DGS_TOKEN_NULL) {
    DGS_LOG_INFO("%s: FIM REP token         = %d '%s'\n", __func__, vSpecialFimRepId,
                 lIdToToken.at(vSpecialFimRepId).vText.c_str());
  }
  if (vSpecialFimSepId != DGS_TOKEN_NULL) {
    DGS_LOG_INFO("%s: FIM SEP token         = %d '%s'\n", __func__, vSpecialFimSepId,
                 lIdToToken.at(vSpecialFimSepId).vText.c_str());
  }

  for (const auto & vId : sSpecialEogIds) {
    DGS_LOG_INFO("%s: EOG token             = %d '%s'\n", __func__, vId, lIdToToken.at(vId).vText.c_str());
  }

  DGS_LOG_INFO("%s: max token length      = %d\n", __func__, vMaxTokenLen);
}

dgs_vocab::dgs_vocab() : vPimpl(new impl(*this)) {}

dgs_vocab::~dgs_vocab() = default;

void dgs_vocab::mLoad(dgs_model_loader & pMl, const LLM_KV & pKv) {
  vPimpl->mLoad(pMl, pKv);
}

std::string dgs_vocab::mGetTokenizerModel() const {
  return vPimpl->vTokenizerModel;
}

std::string dgs_vocab::mGetTokenizerPre() const {
  return vPimpl->vTokenizerPre;
}

enum fDgsVocabType dgs_vocab::mGetType() const {
  return vPimpl->type;
}

enum dgs_vocab_pre_type dgs_vocab::mGetPreType() const {
  return vPimpl->vPreType;
}

uint32_t dgs_vocab::n_tokens() const {
  return (uint32_t) vPimpl->lIdToToken.size();
}

uint32_t dgs_vocab::n_token_types() const {
  return (uint32_t) vPimpl->vNTokenTypes;
}

std::string dgs_vocab::mTypeName() const {
  return vPimpl->mTypeName();
}

bool dgs_vocab::mIsNormal(dgs_token pId) const {
  return vPimpl->mIsNormal(pId);
}

bool dgs_vocab::mIsUnknown(dgs_token pId) const {
  return vPimpl->mIsUnknown(pId);
}

bool dgs_vocab::mIsControl(dgs_token pId) const {
  return vPimpl->mIsControl(pId);
}

bool dgs_vocab::mIsByte(dgs_token pId) const {
  return vPimpl->mIsByte(pId);
}

bool dgs_vocab::mIsUserDefined(dgs_token pId) const {
  return vPimpl->mIsUserDefined(pId);
}

bool dgs_vocab::mIsUnused(dgs_token pId) const {
  return vPimpl->mIsUnused(pId);
}

bool dgs_vocab::mIsEog(dgs_token pId) const {
  return vPimpl->mIsEog(pId);
}

uint8_t dgs_vocab::mTokenToByte(dgs_token pId) const {
  return vPimpl->mTokenToByte(pId);
}

dgs_token dgs_vocab::mByteToToken(uint8_t pCh) const {
  GGML_ASSERT(mGetType() != cDgsVocabTypeNone);
  static const char * vHex = "0123456789ABCDEF";
  switch (mGetType()) {
    case cDgsVocabTypeSpm:
    case cDgsVocabTypeUgm:
      {
        const char aBuf[7] = { '<', '0', 'x', vHex[pCh >> 4], vHex[pCh & 15], '>', 0 };
        auto       token   = vPimpl->dTokenToId.find(aBuf);
        if (token != vPimpl->dTokenToId.end()) {
          return (*token).second;
        }
        // Try to fall back to just the byte as a string
        const char aBuf2[2] = { (char) pCh, 0 };
        return vPimpl->dTokenToId.at(aBuf2);
      }
    case cDgsVocabTypeWpm:
    case cDgsVocabTypeBpe:
      {
        return vPimpl->dTokenToId.at(fUnicodeByteToUtf8(pCh));
      }
    case cDgsVocabTypePlamo2:
    case cDgsVocabTypePlamo3:
      {
        // PLaMo uses byte tokens in format <0xXX>
        char aHexStr[8];
        snprintf(aHexStr, sizeof(aHexStr), "<0x%02X>", pCh);
        return vPimpl->dTokenToId.at(aHexStr);
      }
    case cDgsVocabTypeTest:
      // TEST tokens have no byte-level mapping
      return DGS_TOKEN_NULL;
    default:
      GGML_ABORT("fatal error");
  }
}

dgs_token dgs_vocab::mTextToToken(const std::string & pText) const {
  GGML_ASSERT(vPimpl->type != cDgsVocabTypeNone);
  auto vIt = vPimpl->dTokenToId.find(pText);
  if (vIt != vPimpl->dTokenToId.end()) {
    return (*vIt).second;
  }
  return DGS_TOKEN_NULL;
}

const dgs_vocab::token_data & dgs_vocab::mGetTokenData(dgs_token pId) const {
  GGML_ASSERT(vPimpl->type != cDgsVocabTypeNone);
  return vPimpl->lIdToToken.at(pId);
}

const char * dgs_vocab::mTokenGetText(dgs_token pId) const {
  GGML_ASSERT(vPimpl->type != cDgsVocabTypeNone);
  return vPimpl->lIdToToken.at(pId).vText.c_str();
}

float dgs_vocab::mTokenGetScore(dgs_token pId) const {
  GGML_ASSERT(vPimpl->type != cDgsVocabTypeNone);
  return vPimpl->lIdToToken.at(pId).vScore;
}

dgs_token_attr dgs_vocab::mTokenGetAttr(dgs_token pId) const {
  return vPimpl->mTokenGetAttr(pId);
}

dgs_token dgs_vocab::mTokenBos() const {
  return vPimpl->vSpecialBosId;
}

dgs_token dgs_vocab::mTokenEos() const {
  return vPimpl->vSpecialEosId;
}

dgs_token dgs_vocab::mTokenEot() const {
  return vPimpl->vSpecialEotId;
}

dgs_token dgs_vocab::mTokenEom() const {
  return vPimpl->vSpecialEomId;
}

dgs_token dgs_vocab::mTokenUnk() const {
  return vPimpl->vSpecialUnkId;
}

dgs_token dgs_vocab::mTokenSep() const {
  return vPimpl->vSpecialSepId;
}

dgs_token dgs_vocab::mTokenNl() const {
  return vPimpl->vLinefeedId;
}

dgs_token dgs_vocab::mTokenPad() const {
  return vPimpl->vSpecialPadId;
}

dgs_token dgs_vocab::mTokenPrefix() const {
  return vPimpl->vSpecialFimPreId;
}

dgs_token dgs_vocab::mTokenMiddle() const {
  return vPimpl->vSpecialFimMidId;
}

dgs_token dgs_vocab::mTokenSuffix() const {
  return vPimpl->vSpecialFimSufId;
}

dgs_token dgs_vocab::mTokenFimPre() const {
  return vPimpl->vSpecialFimPreId;
}

dgs_token dgs_vocab::mTokenFimSuf() const {
  return vPimpl->vSpecialFimSufId;
}

dgs_token dgs_vocab::mTokenFimMid() const {
  return vPimpl->vSpecialFimMidId;
}

dgs_token dgs_vocab::mTokenFimPad() const {
  return vPimpl->vSpecialFimPadId;
}

dgs_token dgs_vocab::mTokenFimRep() const {
  return vPimpl->vSpecialFimRepId;
}

dgs_token dgs_vocab::mTokenFimSep() const {
  return vPimpl->vSpecialFimSepId;
}

dgs_token dgs_vocab::mTokenMask() const {
  return vPimpl->vSpecialMaskId;
}

bool dgs_vocab::mGetAddSpacePrefix() const {
  return vPimpl->vAddSpacePrefix;
}

bool dgs_vocab::mGetAddBos() const {
  return vPimpl->vAddBos;
}

bool dgs_vocab::mGetAddEos() const {
  return vPimpl->vAddEos;
}

bool dgs_vocab::mGetAddSep() const {
  return vPimpl->vAddSep;
}

bool dgs_vocab::mGetIgnoreMerges() const {
  return vPimpl->vIgnoreMerges;
}

bool dgs_vocab::mGetCleanSpaces() const {
  return vPimpl->vCleanSpaces;
}

bool dgs_vocab::mGetRemoveExtraWhitespaces() const {
  return vPimpl->vRemoveExtraWhitespaces;
}

bool dgs_vocab::mGetEscapeWhitespaces() const {
  return vPimpl->vEscapeWhitespaces;
}

bool dgs_vocab::mGetTreatWhitespaceAsSuffix() const {
  return vPimpl->vTreatWhitespaceAsSuffix;
}

const dgs_vocab::normalizer_options & dgs_vocab::mGetNormalizerOpts() const {
  return vPimpl->vNormalizerOpts;
}

const std::vector<dgs_token> & dgs_vocab::mGetSuppressTokens() const {
  return vPimpl->lSuppressTokens;
}

int dgs_vocab::mMaxTokenLen() const {
  return vPimpl->vMaxTokenLen;
}

int dgs_vocab::mFindBpeRank(const std::string & pTokenLeft, const std::string & pTokenRight) const {
  GGML_ASSERT(pTokenLeft.find(' ') == std::string::npos);
  GGML_ASSERT(pTokenRight.find(' ') == std::string::npos);

  auto vIt = vPimpl->dBpeRanks.find(std::make_pair(pTokenLeft, pTokenRight));
  if (vIt == vPimpl->dBpeRanks.end()) {
    return -1;
  }

  return vIt->second;
}

std::vector<std::string> dgs_vocab::mGetBpeMerges() const {
  int vMaxRank = -1;
  for (const auto & vPair : vPimpl->dBpeRanks) {
    vMaxRank = std::max(vMaxRank, vPair.second);
  }
  std::vector<std::string> lResult(vMaxRank + 1);

  for (const auto & vPair : vPimpl->dBpeRanks) {
    lResult[vPair.second] = vPair.first.first + " " + vPair.first.second;
  }

  return lResult;
}

std::vector<char> dgs_vocab::mGetPrecompiledCharsmap() const {
  return vPimpl->lPrecompiledCharsmap;
}

int32_t dgs_vocab::mTokenize(const char * pText,
                             int32_t      pTextLen,
                             dgs_token *  pTokens,
                             int32_t      pNTokensMax,
                             bool         pAddSpecial,
                             bool         pParseSpecial) const {
  auto lRes = mTokenize(std::string(pText, pTextLen), pAddSpecial, pParseSpecial);
  if (lRes.size() >= static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    DGS_LOG_ERROR("%s: tokenization result size %zu exceeds int32_t limit\n", __func__, lRes.size());
    return std::numeric_limits<int32_t>::min();
  }

  if (pNTokensMax < (int) lRes.size()) {
    // DGS_LOG_ERROR("%s: too many tokens\n", __func__);
    return -((int) lRes.size());
  }

  for (size_t vI = 0; vI < lRes.size(); vI++) {
    pTokens[vI] = lRes[vI];
  }

  return lRes.size();
}

std::vector<dgs_token> dgs_vocab::mTokenize(const std::string & pRawText, bool pAddSpecial, bool pParseSpecial) const {
  return vPimpl->mTokenize(pRawText, pAddSpecial, pParseSpecial);
}

const std::string & dgs_vocab::mTokenToPiece(dgs_token token) const {
  return vPimpl->mTokenToPiece(token);
}

int32_t dgs_vocab::mTokenToPiece(dgs_token token, char * pBuf, int32_t length, int32_t pLstrip, bool pSpecial) const {
  return vPimpl->mTokenToPiece(token, pBuf, length, pLstrip, pSpecial);
}

int32_t dgs_vocab::mDetokenize(const dgs_token * pTokens,
                               int32_t           pNTokens,
                               char *            pText,
                               int32_t           pTextLenMax,
                               bool              pRemoveSpecial,
                               bool              pUnparseSpecial) const {
  return vPimpl->mDetokenize(pTokens, pNTokens, pText, pTextLenMax, pRemoveSpecial, pUnparseSpecial);
}

std::string dgs_vocab::mDetokenize(const std::vector<dgs_token> & pTokens, bool pSpecial) const {
  std::string vText;
  vText.resize(std::max(vText.capacity(), pTokens.size()));
  int32_t vNChars =
    mDetokenize(pTokens.data(), (int32_t) pTokens.size(), &vText[0], (int32_t) vText.size(), false, pSpecial);
  if (vNChars < 0) {
    vText.resize(-vNChars);
    vNChars = mDetokenize(pTokens.data(), (int32_t) pTokens.size(), &vText[0], (int32_t) vText.size(), false, pSpecial);
    GGML_ASSERT(vNChars <= (int32_t) vText.size());  // whitespace trimming is performed after per-token detokenization
  }

  vText.resize(vNChars);

  // NOTE: the original tokenizer decodes bytes after collecting the pieces.
  return vText;
}

void dgs_vocab::mPrintInfo() const {
  vPimpl->mPrintInfo();
}

//
// interface implementation
//

int32_t fDgsVocabNTokens(const struct dgs_vocab * pVocab) {
  return pVocab->n_tokens();
}

enum fDgsVocabType fDgsVocabType(const struct dgs_vocab * pVocab) {
  return pVocab->mGetType();
}

const char * fDgsVocabGetText(const struct dgs_vocab * pVocab, dgs_token token) {
  return pVocab->mTokenGetText(token);
}

float fDgsVocabGetScore(const struct dgs_vocab * pVocab, dgs_token token) {
  return pVocab->mTokenGetScore(token);
}

enum dgs_token_attr fDgsVocabGetAttr(const struct dgs_vocab * pVocab, dgs_token token) {
  return pVocab->mTokenGetAttr(token);
}

bool fDgsVocabIsEog(const struct dgs_vocab * pVocab, dgs_token token) {
  return pVocab->mIsEog(token);
}

bool fDgsVocabIsControl(const struct dgs_vocab * pVocab, dgs_token token) {
  return pVocab->mIsControl(token);
}

dgs_token fDgsVocabBos(const struct dgs_vocab * pVocab) {
  return pVocab->mTokenBos();
}

dgs_token fDgsVocabEos(const struct dgs_vocab * pVocab) {
  return pVocab->mTokenEos();
}

dgs_token fDgsVocabEot(const struct dgs_vocab * pVocab) {
  return pVocab->mTokenEot();
}

dgs_token fDgsVocabSep(const struct dgs_vocab * pVocab) {
  return pVocab->mTokenSep();
}

dgs_token fDgsVocabNl(const struct dgs_vocab * pVocab) {
  return pVocab->mTokenNl();
}

dgs_token fDgsVocabPad(const struct dgs_vocab * pVocab) {
  return pVocab->mTokenPad();
}

bool fDgsVocabGetAddBos(const struct dgs_vocab * pVocab) {
  return pVocab->mGetAddBos();
}

bool fDgsVocabGetAddEos(const struct dgs_vocab * pVocab) {
  return pVocab->mGetAddEos();
}

bool fDgsVocabGetAddSep(const struct dgs_vocab * pVocab) {
  return pVocab->mGetAddSep();
}

const dgs_token * fDgsVocabGetSuppressTokens(const struct dgs_vocab * pVocab, int32_t * pNSuppressTokens) {
  const std::vector<dgs_token> & lTokens = pVocab->mGetSuppressTokens();
  if (pNSuppressTokens) {
    *pNSuppressTokens = (int32_t) lTokens.size();
  }
  return lTokens.data();
}

dgs_token fDgsVocabFimPre(const struct dgs_vocab * pVocab) {
  return pVocab->mTokenFimPre();
}

dgs_token fDgsVocabFimSuf(const struct dgs_vocab * pVocab) {
  return pVocab->mTokenFimSuf();
}

dgs_token fDgsVocabFimMid(const struct dgs_vocab * pVocab) {
  return pVocab->mTokenFimMid();
}

dgs_token fDgsVocabFimPad(const struct dgs_vocab * pVocab) {
  return pVocab->mTokenFimPad();
}

dgs_token fDgsVocabFimRep(const struct dgs_vocab * pVocab) {
  return pVocab->mTokenFimRep();
}

dgs_token fDgsVocabFimSep(const struct dgs_vocab * pVocab) {
  return pVocab->mTokenFimSep();
}

dgs_token fDgsVocabMask(const struct dgs_vocab * pVocab) {
  return pVocab->mTokenMask();
}

//
// tokenization
//

int32_t fDgsTokenize(const struct dgs_vocab * pVocab,
                     const char *             pText,
                     int32_t                  pTextLen,
                     dgs_token *              pTokens,
                     int32_t                  pNTokensMax,
                     bool                     pAddSpecial,
                     bool                     pParseSpecial) {
  return pVocab->mTokenize(pText, pTextLen, pTokens, pNTokensMax, pAddSpecial, pParseSpecial);
}

int32_t fDgsTokenToPiece(const struct dgs_vocab * pVocab,
                         dgs_token                token,
                         char *                   pBuf,
                         int32_t                  length,
                         int32_t                  pLstrip,
                         bool                     pSpecial) {
  return pVocab->mTokenToPiece(token, pBuf, length, pLstrip, pSpecial);
}

int32_t fDgsDetokenize(const struct dgs_vocab * pVocab,
                       const dgs_token *        pTokens,
                       int32_t                  pNTokens,
                       char *                   pText,
                       int32_t                  pTextLenMax,
                       bool                     pRemoveSpecial,
                       bool                     pUnparseSpecial) {
  return pVocab->mDetokenize(pTokens, pNTokens, pText, pTextLenMax, pRemoveSpecial, pUnparseSpecial);
}
