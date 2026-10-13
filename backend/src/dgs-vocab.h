#pragma once

#include "dgs.h"

#include <memory>
#include <string>
#include <vector>

// pre-tokenization types
enum dgs_vocab_pre_type {
  cDgsVocabPreTypeDefault         = 0,
  cDgsVocabPreTypeDgs3            = 1,
  cDgsVocabPreTypeDeepseekLlm     = 2,
  cDgsVocabPreTypeDeepseekCoder   = 3,
  cDgsVocabPreTypeFalcon          = 4,
  cDgsVocabPreTypeMpt             = 5,
  cDgsVocabPreTypeStarcoder       = 6,
  cDgsVocabPreTypeGpt2            = 7,
  cDgsVocabPreTypeRefact          = 8,
  cDgsVocabPreTypeCommandR        = 9,
  cDgsVocabPreTypeStablelm2       = 10,
  cDgsVocabPreTypeQwen2           = 11,
  cDgsVocabPreTypeOlmo            = 12,
  cDgsVocabPreTypeDbrx            = 13,
  cDgsVocabPreTypeSmaug           = 14,
  cDgsVocabPreTypePoro            = 15,
  cDgsVocabPreTypeChatglm3        = 16,
  cDgsVocabPreTypeChatglm4        = 17,
  cDgsVocabPreTypeViking          = 18,
  cDgsVocabPreTypeJais            = 19,
  cDgsVocabPreTypeTekken          = 20,
  cDgsVocabPreTypeSmollm          = 21,
  cDgsVocabPreTypeCodeshell       = 22,
  cDgsVocabPreTypeBloom           = 23,
  cDgsVocabPreTypeGpt3Finnish     = 24,
  cDgsVocabPreTypeExaone          = 25,
  cDgsVocabPreTypeChameleon       = 26,
  cDgsVocabPreTypeMinerva         = 27,
  cDgsVocabPreTypeDeepseek3Llm    = 28,
  cDgsVocabPreTypeGpt4o           = 29,
  cDgsVocabPreTypeSuperbpe        = 30,
  cDgsVocabPreTypeTrillion        = 31,
  cDgsVocabPreTypeBailingmoe      = 32,
  cDgsVocabPreTypeDgs4            = 33,
  cDgsVocabPreTypePixtral         = 34,
  cDgsVocabPreTypeSeedCoder       = 35,
  cDgsVocabPreTypeHunyuan         = 36,
  cDgsVocabPreTypeKimiK2          = 37,
  cDgsVocabPreTypeHunyuanDense    = 38,
  cDgsVocabPreTypeGrok2           = 39,
  cDgsVocabPreTypeGraniteDocling  = 40,
  cDgsVocabPreTypeMinimaxM2       = 41,
  cDgsVocabPreTypeAfmoe           = 42,
  cDgsVocabPreTypeSolarOpen       = 43,
  cDgsVocabPreTypeYoutu           = 44,
  cDgsVocabPreTypeExaoneMoe       = 45,
  cDgsVocabPreTypeQwen35          = 46,
  cDgsVocabPreTypeTinyAya         = 47,
  cDgsVocabPreTypeJoyaiLlm        = 48,
  cDgsVocabPreTypeJais2           = 49,
  cDgsVocabPreTypeGemma4          = 50,
  cDgsVocabPreTypeSarvamMoe       = 51,
  cDgsVocabPreTypeMinicpm5        = 52,
  cDgsVocabPreTypeWhitespace      = 53,
  cDgsVocabPreTypeGraniteEmbMulti = 54,
  cDgsVocabPreTypeMellum2         = 55,
  cDgsVocabPreTypeLaguna          = 56,
  cDgsVocabPreTypeHyV4            = 57,
  cDgsVocabPreTypeSPARK2Point5    = 58,
  cDgsVocabPreTypeUfakzeka        = 59,
  cDgsVocabPreTypeMmbert          = 60,
  cDgsVocabPreTypeK2Horizon       = 61,
};

struct LLM_KV;
struct dgs_model_loader;

struct dgs_vocab {
  struct token_data {
    std::string    vText;
    float          vScore;
    dgs_token_attr vAttr;
  };

  struct normalizer_options {
    bool vLowercase    = true;
    bool vStripAccents = true;
    // TODO: clean_text, handle_chinese_chars
  };

  dgs_vocab();
  ~dgs_vocab();

  void mLoad(dgs_model_loader & pMl, const LLM_KV & pKv);

  std::string mGetTokenizerModel() const;
  std::string mGetTokenizerPre() const;

  enum fDgsVocabType      mGetType() const;
  enum dgs_vocab_pre_type mGetPreType() const;

  uint32_t n_tokens() const;
  uint32_t n_token_types() const;

  std::string mTypeName() const;

  bool mIsNormal(dgs_token pId) const;
  bool mIsUnknown(dgs_token pId) const;
  bool mIsControl(dgs_token pId) const;
  bool mIsByte(dgs_token pId) const;
  bool mIsUserDefined(dgs_token pId) const;
  bool mIsUnused(dgs_token pId) const;
  bool mIsEog(dgs_token pId) const;

  uint8_t   mTokenToByte(dgs_token pId) const;
  dgs_token mByteToToken(uint8_t pCh) const;

  dgs_token mTextToToken(const std::string & pText) const;

  const token_data & mGetTokenData(dgs_token pId) const;

  const char *   mTokenGetText(dgs_token pId) const;
  float          mTokenGetScore(dgs_token pId) const;
  dgs_token_attr mTokenGetAttr(dgs_token pId) const;

  dgs_token mTokenBos() const;
  dgs_token mTokenEos() const;
  dgs_token mTokenEot() const;
  dgs_token mTokenEom() const;
  dgs_token mTokenUnk() const;
  dgs_token mTokenSep() const;
  dgs_token mTokenNl() const;
  dgs_token mTokenPad() const;
  dgs_token mTokenMask() const;

  dgs_token mTokenPrefix() const;
  dgs_token mTokenMiddle() const;
  dgs_token mTokenSuffix() const;

  dgs_token mTokenFimPre() const;
  dgs_token mTokenFimSuf() const;
  dgs_token mTokenFimMid() const;
  dgs_token mTokenFimPad() const;
  dgs_token mTokenFimRep() const;
  dgs_token mTokenFimSep() const;

  bool                       mGetAddSpacePrefix() const;
  bool                       mGetAddBos() const;
  bool                       mGetAddEos() const;
  bool                       mGetAddSep() const;
  bool                       mGetIgnoreMerges() const;
  bool                       mGetCleanSpaces() const;
  bool                       mGetRemoveExtraWhitespaces() const;
  bool                       mGetEscapeWhitespaces() const;
  bool                       mGetTreatWhitespaceAsSuffix() const;
  const normalizer_options & mGetNormalizerOpts() const;

  const std::vector<dgs_token> & mGetSuppressTokens() const;

  int mMaxTokenLen() const;

  int                      mFindBpeRank(const std::string & pTokenLeft, const std::string & pTokenRight) const;
  std::vector<std::string> mGetBpeMerges() const;

  std::vector<char> mGetPrecompiledCharsmap() const;

  int32_t mTokenize(const char * pText,
                    int32_t      pTextLen,
                    dgs_token *  pTokens,
                    int32_t      pNTokensMax,
                    bool         pAddSpecial,
                    bool         pParseSpecial) const;

  std::vector<dgs_token> mTokenize(const std::string & pRawText, bool pAddSpecial, bool pParseSpecial = false) const;

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
  struct impl;
  std::unique_ptr<impl> vPimpl;
};
