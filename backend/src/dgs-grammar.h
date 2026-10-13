#pragma once

#include "dgs.h"

#include <map>
#include <regex>
#include <string>
#include <vector>

struct dgs_vocab;

// grammar element type
enum dgs_gretype {
  // end of rule definition
  cDgsGretypeEnd = 0,

  // start of alternate definition for rule
  cDgsGretypeAlt = 1,

  // non-terminal element: reference to rule
  cDgsGretypeRuleRef = 2,

  // terminal element: character (code point)
  cDgsGretypeChar = 3,

  // inverse char(s) ([^a], [^a-b] [^abc])
  cDgsGretypeCharNot = 4,

  // modifies a preceding DGS_GRETYPE_CHAR or DGS_GRETYPE_CHAR_ALT to
  // be an inclusive range ([a-z])
  cDgsGretypeCharRngUpper = 5,

  // modifies a preceding DGS_GRETYPE_CHAR or
  // DGS_GRETYPE_CHAR_RNG_UPPER to add an alternate char to match ([ab], [a-zA])
  cDgsGretypeCharAlt = 6,

  // any character (.)
  cDgsGretypeCharAny = 7,

  // terminal element: token (<[token-id]>)
  cDgsGretypeToken = 8,

  // inverse token (!<[token-id]>)
  cDgsGretypeTokenNot = 9,
};

typedef struct dgs_grammar_element {
  enum dgs_gretype type;
  uint32_t         value;  // Unicode code point, rule ID, or token ID
} dgs_grammar_element;

struct dgs_partial_utf8 {
  uint32_t value;     // bit value so far (unshifted)
  int      vNRemain;  // num bytes remaining; -1 indicates invalid sequence
};

struct dgs_grammar_candidate {
  size_t           vIndex;
  const uint32_t * vCodePoints;
  dgs_partial_utf8 vPartialUtf8;
  dgs_token        vId;
};

using dgs_grammar_rule  = std::vector<dgs_grammar_element>;
using dgs_grammar_stack = std::vector<const dgs_grammar_element *>;

using dgs_grammar_rules      = std::vector<dgs_grammar_rule>;
using dgs_grammar_stacks     = std::vector<dgs_grammar_stack>;
using dgs_grammar_candidates = std::vector<dgs_grammar_candidate>;

// TODO: remove, needed for tests atm
const dgs_grammar_rules & fDgsGrammarGetRules(const struct dgs_grammar * pGrammar);
dgs_grammar_stacks &      fDgsGrammarGetStacks(struct dgs_grammar * pGrammar);

// takes a set of possible pushdown stacks on a grammar, which are required to
// be positioned at a character range (see `dgs_grammar_advance_stack`), and
// produces the N possible stacks if the given char is accepted at those
// positions
void fDgsGrammarAccept(struct dgs_grammar * pGrammar, uint32_t pChr);

std::vector<dgs_grammar_candidate> fDgsGrammarRejectCandidatesForStack(const dgs_grammar_rules &      pRules,
                                                                       const dgs_grammar_stack &      pStack,
                                                                       const dgs_grammar_candidates & pCandidates);

struct dgs_grammar_parser {
  const dgs_vocab *               vVocab;
  std::map<std::string, uint32_t> dSymbolIds;

  dgs_grammar_rules lRules;

  dgs_grammar_parser(const struct dgs_vocab * pVocab = nullptr) : vVocab(pVocab) {}

  dgs_grammar_stack mCRules() const;

  uint32_t mGetSymbolId(const char * pSrc, size_t pLen);
  uint32_t mGenerateSymbolId(const std::string & pBaseName);

  void mAddRule(uint32_t pRuleId, const dgs_grammar_rule & pRule);

  const char * mParseAlternates(const char * pSrc, const std::string & pRuleName, uint32_t pRuleId, bool pIsNested);

  const char * mParseSequence(const char *        pSrc,
                              const std::string & pRuleName,
                              dgs_grammar_rule &  pRule,
                              bool                pIsNested);

  const char * mParseRule(const char * pSrc);

  bool mParse(const char * pSrc);
  void mPrint(FILE * pFile);
};

struct dgs_grammar_trigger_pattern {
  std::string vPattern;
  std::regex  vRegex;

  size_t find(const std::string & pInput) const;
};

struct dgs_grammar {
  // maintain a list of dgs_tokens and their positions in the trigger_buffer
  using token_pos = std::pair<dgs_token, std::pair<size_t, size_t>>;

  // note: allow null vocab for testing (not great)
  const dgs_vocab * vVocab;

  const dgs_grammar_rules lRules;  // TODO: shared ptr
  dgs_grammar_stacks      lStacks;

  // buffer for partially generated UTF-8 sequence from accepted tokens
  dgs_partial_utf8 vPartialUtf8;

  // lazy grammars wait for trigger words or tokens before constraining the sampling.
  // we still have trigger_tokens for non-lazy grammars to force printing of special trigger tokens.
  // (useful e.g. for tool_choice=required)
  bool        vLazy            = false;
  bool        vAwaitingTrigger = false;  // Initialized to true for lazy grammars only
  std::string vTriggerBuffer;            // Output buffered by lazy grammar. Will be cleared once trigger is found.
  std::vector<token_pos>
    lTriggerBufferPositions;             // Tokens buffered by lazy grammar. Used to replay when a trigger is found.
  std::vector<dgs_token>
    lTriggerTokens;    // Tokens that trigger a lazy grammar, or tokens to force printing of (even if special).
  std::vector<dgs_grammar_trigger_pattern>
    lTriggerPatterns;  // Regular expressions that trigger a lazy grammar. Must be a full match of the entire generated
                       // string, and the grammar will be given the string from the first match group onwards.
};

//
// internal API
//

// note: needed for tests (not great)
struct dgs_grammar * fDgsGrammarInitImpl(const struct dgs_vocab *     pVocab,
                                         const dgs_grammar_element ** pRules,
                                         size_t                       pNRules,
                                         size_t                       pStartRuleIndex);

struct dgs_grammar * fDgsGrammarInitImpl(const struct dgs_vocab * pVocab,
                                         const char *             pGrammarStr,
                                         const char *             pGrammarRoot,
                                         bool                     pLazy,
                                         const char **            pTriggerPatterns,
                                         size_t                   pNumTriggerPatterns,
                                         const dgs_token *        pTriggerTokens,
                                         size_t                   pNumTriggerTokens);

void fDgsGrammarFreeImpl(struct dgs_grammar * pGrammar);

struct dgs_grammar * fDgsGrammarCloneImpl(const struct dgs_grammar & pGrammar);

// TODO: move the API below as member functions of dgs_grammar
void fDgsGrammarApplyImpl(const struct dgs_grammar & pGrammar, dgs_token_data_array * pCurP);

void fDgsGrammarAcceptImpl(struct dgs_grammar & pGrammar, dgs_token token);

void fDgsGrammarAcceptStr(struct dgs_grammar & pGrammar, const std::string & pPiece);

void fDgsGrammarAcceptToken(struct dgs_grammar & pGrammar, dgs_token token, const std::string & pPiece);
