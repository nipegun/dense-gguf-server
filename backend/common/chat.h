// Chat support (incl. tool call grammar constraining & output parsing) w/ generic & custom template handlers.

#pragma once

#include "common.h"
#include "jinja/caps.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "json.h"
#include "peg-parser.h"

#include <chrono>
#include <functional>
#include <map>
#include <string>
#include <vector>

using chat_template_caps = jinja::caps;

struct common_chat_templates;

namespace autoparser {
struct generation_params;
struct autoparser;
}  // namespace autoparser

struct common_chat_params;
struct common_chat_template;

// Builds the prompt and parser for a template that has a dedicated handler (see common/parsers)
using common_chat_params_init_fn = common_chat_params (*)(const common_chat_template &          pTmpl,
                                                          const autoparser::generation_params & pInputs);

struct common_chat_tool_call {
  std::string name;
  std::string vArguments;
  std::string vId;

  bool operator==(const common_chat_tool_call & pOther) const {
    return name == pOther.name && vArguments == pOther.vArguments && vId == pOther.vId;
  }
};

struct common_chat_msg_content_part {
  std::string type;
  std::string vText;

  // TODO @ngxson : no known chat templates support reasoning_content in content parts yet
  //                this can be useful for models with interleaved thinking (like Kimi-K2)
  //                if you see any templates explicitly support this, please ping me
  // std::string reasoning_content;

  bool operator==(const common_chat_msg_content_part & pOther) const {
    return type == pOther.type && vText == pOther.vText;
  }
};

struct common_chat_template {
  jinja::program             prog;
  std::string                bos_tok;
  std::string                eos_tok;
  std::string                src;
  chat_template_caps         caps;
  // Dedicated handler picked once from the source, null when the differential autoparser is used
  common_chat_params_init_fn params_init = nullptr;

  // Differential analysis, run once here when there is no dedicated handler. Null when there
  // is one, or when the analysis failed, in which case analysis_error says why.
  std::unique_ptr<autoparser::autoparser> analysis;
  std::string                             analysis_error;

  common_chat_template(const std::string & pSrc, const std::string & pBosToken, const std::string & pEosToken);

  // autoparser is incomplete here, so these are defined where it is complete
  ~common_chat_template();
  common_chat_template(common_chat_template &&);
  common_chat_template & operator=(common_chat_template &&);

  const std::string & mSource() const { return src; }

  const std::string & mBosToken() const { return bos_tok; }

  const std::string & mEosToken() const { return eos_tok; }

  chat_template_caps mOriginalCaps() const { return caps; }
};

struct common_chat_msg {
  std::string                               vRole;
  std::string                               vContent;
  std::vector<common_chat_msg_content_part> lContentParts;
  std::vector<common_chat_tool_call>        lToolCalls;
  std::string                               vReasoningContent;
  std::string                               vToolName;
  std::string                               vToolCallId;

  common_json mToJsonOaicompat(bool pConcatTypedText = false) const;

  std::string mRenderContent(const std::string & pDelimiter = "\n\n") const;

  bool empty() const {
    return vContent.empty() && lContentParts.empty() && lToolCalls.empty() && vReasoningContent.empty() &&
           vToolName.empty() && vToolCallId.empty();
  }

  bool mContainsMedia() const {
    for (const auto & vPart : lContentParts) {
      if (vPart.type == "media_marker") {
        return true;
      }
    }
    return false;
  }

  void mSetToolCallIds(std::vector<std::string> & pIdsCache, const std::function<std::string()> & pGenToolCallId) {
    for (auto vI = 0u; vI < lToolCalls.size(); vI++) {
      if (pIdsCache.size() <= vI) {
        auto vId = lToolCalls[vI].vId;
        if (vId.empty()) {
          vId = pGenToolCallId();
        }
        pIdsCache.push_back(vId);
      }
      lToolCalls[vI].vId = pIdsCache[vI];
    }
  }

  bool operator==(const common_chat_msg & pOther) const {
    return vRole == pOther.vRole && vContent == pOther.vContent && lContentParts == pOther.lContentParts &&
           lToolCalls == pOther.lToolCalls && vReasoningContent == pOther.vReasoningContent &&
           vToolName == pOther.vToolName && vToolCallId == pOther.vToolCallId;
  }

  bool operator!=(const common_chat_msg & pOther) const { return !(*this == pOther); }
};

struct common_chat_msg_diff {
  std::string           vReasoningContentDelta;
  std::string           vContentDelta;
  size_t                vToolCallIndex = std::string::npos;
  common_chat_tool_call vToolCallDelta;

  static std::vector<common_chat_msg_diff> mComputeDiffs(const common_chat_msg & pMsgPrv,
                                                         const common_chat_msg & pMsgNew);

  bool operator==(const common_chat_msg_diff & pOther) const {
    return vContentDelta == pOther.vContentDelta && vToolCallIndex == pOther.vToolCallIndex &&
           vToolCallDelta == pOther.vToolCallDelta;
  }
};

enum common_chat_role {
  cCommonChatRoleUnknown,
  cCommonChatRoleSystem,
  cCommonChatRoleAssistant,
  cCommonChatRoleUser,
  cCommonChatRoleTool
};

common_chat_role fCommonChatRoleFromString(const std::string & pRole);
const char *     fCommonChatRoleToString(common_chat_role pRole);

struct common_chat_msg_span {
  common_chat_role vRole = cCommonChatRoleUnknown;
  std::size_t      vPos  = 0;
  std::size_t      vLen  = 0;

  bool mValid() const { return vRole != cCommonChatRoleUnknown; }
};

struct common_chat_msg_spans {
  std::vector<common_chat_msg_span> lSpans;

  void mAdd(common_chat_role pRole, size_t pPos, size_t pLen) { lSpans.push_back({ pRole, pPos, pLen }); }

  bool mIsUserStart(int32_t pPos) const {
    for (auto lIt = lSpans.begin(); lIt != lSpans.end(); ++lIt) {
      if (lIt->vRole == cCommonChatRoleUser && pPos == (int32_t) lIt->vPos) {
        return true;
      }
    }
    return false;
  }

  int32_t mLastUserMessagePos() const {
    for (auto lIt = lSpans.rbegin(); lIt != lSpans.rend(); ++lIt) {
      if (lIt->vRole == cCommonChatRoleUser) {
        return (int32_t) lIt->vPos;
      }
    }
    return -1;
  }
};

struct common_chat_msg_delimiter {
  common_chat_role vRole = cCommonChatRoleUnknown;
  std::string      vDelimiter;
  dgs_tokens       lTokens = {};
};

struct common_chat_msg_delimiters {
  std::vector<common_chat_msg_delimiter> lDelimiters;

  common_chat_msg_delimiters() = default;

  common_chat_msg_delimiters(std::initializer_list<common_chat_msg_delimiter> pDelims) : lDelimiters(pDelims) {}

  void mAdd(common_chat_role pRole, const std::string & pDelimiter) { lDelimiters.push_back({ pRole, pDelimiter }); }

  void mTokenize(const dgs_vocab * pVocab);

  // split tokens into message spans. skips maps a start index to a length of a region to jump over without matching
  common_chat_msg_spans mSplit(const dgs_tokens & pTokens, const std::map<size_t, size_t> & pSkips = {}) const;
};

struct common_chat_tool {
  std::string name;
  std::string vDescription;
  std::string vParameters;
};

enum common_chat_tool_choice {
  cCommonChatToolChoiceAuto,
  cCommonChatToolChoiceRequired,
  cCommonChatToolChoiceNone,
};

enum common_chat_format {
  cCommonChatFormatContentOnly,

  // These are intended to be parsed by the PEG parser
  cCommonChatFormatPegSimple,
  cCommonChatFormatPegNative,
  cCommonChatFormatPegGemma4,
  cCommonChatFormatPegMinimaxM3,

  cCommonChatFormatCount,  // Not a format, just the # formats
};

// Continuation method provided via `continue_final_message`
enum common_chat_continuation {
  cCommonChatContinuationNone,
  cCommonChatContinuationAuto,
  cCommonChatContinuationReasoning,
  cCommonChatContinuationContent,
};

struct common_chat_templates_inputs {
  std::vector<common_chat_msg>  lMessages;
  std::string                   vGrammar;
  std::string                   vJsonSchema;
  bool                          vAddGenerationPrompt  = true;
  common_chat_continuation      vContinueFinalMessage = cCommonChatContinuationNone;
  bool                          vUseJinja             = true;
  // Parameters below only supported when use_jinja is true
  std::vector<common_chat_tool> lTools;
  common_chat_tool_choice       vToolChoice        = cCommonChatToolChoiceAuto;
  bool                          vParallelToolCalls = false;
  common_reasoning_format       vReasoningFormat =
    cCommonReasoningFormatNone;  // TODO: refactor this to "bool enable_thinking"
  bool                                  vEnableThinking = true;
  std::chrono::system_clock::time_point vNow            = std::chrono::system_clock::now();
  std::map<std::string, std::string>    dChatTemplateKwargs;
  bool                                  vAddBos           = false;
  bool                                  vAddEos           = false;
  bool                                  vForcePureContent = false;
};

struct common_chat_params {
  common_chat_format                  format = cCommonChatFormatContentOnly;
  std::string                         vPrompt;
  std::string                         vGrammar;
  bool                                vGrammarLazy = false;
  std::string                         vGenerationPrompt;
  bool                                vSupportsThinking = false;
  std::string                         vThinkingStartTag;  // e.g., "<think>"
  std::vector<std::string>            lThinkingEndTags;   // e.g., "</think>"
  std::vector<common_grammar_trigger> lGrammarTriggers;
  std::vector<std::string>            lPreservedTokens;
  std::vector<std::string>            lAdditionalStops;
  common_peg_arena                    vParser;
  common_chat_msg_delimiters          vMessageDelimiters;
};

struct common_chat_input {
  std::string            vText;
  std::vector<dgs_token> lTokens;

  common_chat_input() = default;

  // plain text, with no tokens
  explicit common_chat_input(std::string pText) :
    vText(std::move(pText)),
    lTokens(this->vText.size(), DGS_TOKEN_NULL) {}

  size_t size() const { return vText.size(); }

  bool empty() const { return vText.empty(); }

  void mAppend(const std::string & pPiece, dgs_token token);
  void mAppend(const common_chat_input & pChunk);

  void mPrepend(const std::string & pPrefix);
  void mPrepend(const common_chat_input & pPrefix);

  void mTruncate(size_t pPos);

  common_chat_input substr(size_t pPos, size_t n = std::string::npos) const;
};

common_chat_input fCommonChatInputTokenize(const dgs_vocab * pVocab, const std::string & pText);

// per-message parsing syntax
// should be derived from common_chat_params
struct common_chat_parser_params {
  common_chat_format format = cCommonChatFormatContentOnly;
  common_chat_input  vGenerationPrompt;
  bool               vDebug   = false;  // Enable debug output for PEG parser
  common_peg_arena   vParser  = {};
  common_chat_parser_params() = default;

  common_chat_parser_params(const common_chat_params & pChatParams) {
    format            = pChatParams.format;
    vGenerationPrompt = common_chat_input(pChatParams.vGenerationPrompt);
  }
};

// Check if the template supplied via "--chat-template" is supported or not. Returns true if it's valid
bool fCommonChatVerifyTemplate(const std::string & pTmpl, bool pUseJinja);

void fCommonChatTemplatesFree(struct common_chat_templates * pTmpls);

struct common_chat_templates_deleter {
  void operator()(common_chat_templates * pTmpls) { fCommonChatTemplatesFree(pTmpls); }
};

typedef std::unique_ptr<struct common_chat_templates, common_chat_templates_deleter> common_chat_templates_ptr;

common_chat_templates_ptr fCommonChatTemplatesInit(const struct dgs_model * pModel,
                                                   const std::string &      pChatTemplateOverride,
                                                   const std::string &      pBosTokenOverride = "",
                                                   const std::string &      pEosTokenOverride = "");

bool        fCommonChatTemplatesWasExplicit(const struct common_chat_templates * pTmpls);
std::string fCommonChatTemplatesSource(const struct common_chat_templates * pTmpls, const std::string & pVariant = "");

struct common_chat_params fCommonChatTemplatesApply(const struct common_chat_templates *        pTmpls,
                                                    const struct common_chat_templates_inputs & pInputs);

// Format single message, while taking into account the position of that message in chat history
std::string fCommonChatFormatSingle(const struct common_chat_templates * pTmpls,
                                    const std::vector<common_chat_msg> & pPastMsg,
                                    const common_chat_msg &              pNewMsg,
                                    bool                                 pAddAss,
                                    bool                                 pUseJinja);

// Returns an example of formatted chat
std::string fCommonChatFormatExample(const struct common_chat_templates *       pTmpls,
                                     bool                                       pUseJinja,
                                     const std::map<std::string, std::string> & pChatTemplateKwargs);

const char *    fCommonChatFormatName(common_chat_format format);
common_chat_msg fCommonChatParse(const common_chat_input &         pInput,
                                 bool                              pIsPartial,
                                 const common_chat_parser_params & pParams);
common_chat_msg fCommonChatPegParse(const common_peg_arena &          pSrcParser,
                                    const common_chat_input &         pInput,
                                    bool                              pIsPartial,
                                    const common_chat_parser_params & pParams);

struct common_chat_session_params {
  bool vEcho  = false;  // include the assistant prefill in the output when continuing a message
  bool vDebug = false;  // enable debug output for the PEG parser
};

class common_chat_session {
public:
  common_chat_session() { vResult.vRole = "assistant"; }

  common_chat_session(const common_chat_templates *        pTmpls,
                      const dgs_vocab *                    pVocab,
                      const common_chat_templates_inputs & pInputs,
                      const common_chat_session_params &   pParams = {});

  const std::string & mPrompt() const { return vPromptText; }

  common_chat_format format() const { return vParserParams.format; }

  const common_chat_msg & mMsg() const { return vResult; }

  const common_peg_arena & mParser() const { return vParserParams.vParser; }

  const std::string & mGrammar() const { return vGrammarText; }

  const std::string & mGenerationPrompt() const { return vGenerationPromptText; }

  const std::string & mThinkingStartTag() const { return vThinkingStart; }

  const std::vector<std::string> & mThinkingEndTags() const { return lThinkingEnds; }

  const std::vector<std::string> & mAdditionalStops() const { return lStops; }

  const common_chat_msg_delimiters & mMessageDelimiters() const { return vDelimiters; }

  void mApplySampling(common_params_sampling & pSampling) const;

  bool mHasTemplate() const { return vTemplated; }

  const common_chat_msg & mFeed(const common_chat_input & pChunk);

  const common_chat_msg & mFinish(const common_chat_input & pChunk = {});

private:
  std::string                         vPromptText;
  std::string                         vGrammarText;
  bool                                vGrammarLazy = false;
  std::vector<common_grammar_trigger> lGrammarTriggers;
  std::set<dgs_token>                 sPreservedTokens;
  std::vector<std::string>            lStops;
  std::string                         vGenerationPromptText;
  std::string                         vThinkingStart;
  std::vector<std::string>            lThinkingEnds;

  common_chat_parser_params  vParserParams;
  common_chat_msg_delimiters vDelimiters;
  common_chat_input          vInput;
  common_chat_msg            vResult;
  bool                       vTemplated = false;
  bool                       vFinished  = false;
};

// used by arg and server
const char *            fCommonReasoningFormatName(common_reasoning_format format);
common_reasoning_format fCommonReasoningFormatFromName(const std::string & format);

common_chat_tool_choice fCommonChatToolChoiceParseOaicompat(const std::string & pToolChoice);

bool fCommonChatTemplatesSupportEnableThinking(const common_chat_templates * pChatTemplates);

// Parses a JSON array of messages in OpenAI's chat completion API format.
std::vector<common_chat_msg> fCommonChatMsgsParseOaicompat(const common_json & pMessages);

std::vector<common_chat_tool> fCommonChatToolsParseOaicompat(const common_json & pTools);

common_chat_continuation fCommonChatContinuationParse(const common_json & value);

// DEPRECATED: only used in tests
common_json fCommonChatMsgsToJsonOaicompat(const std::vector<common_chat_msg> & pMsgs, bool pConcatTypedText = false);

common_json fCommonChatToolsToJsonOaicompat(const std::vector<common_chat_tool> & pTools);

// The parameters schema of a function tool. A tool without parameters, or with an empty {}, takes zero arguments.
common_json fCommonChatToolParameters(const common_json & pFunction);

// get template caps, useful for reporting to server /props endpoint
std::map<std::string, bool> fCommonChatTemplatesGetCaps(const common_chat_templates * pChatTemplates);

std::string fCommonChatTemplateDirectApply(const common_chat_template &          pTmpl,
                                           const autoparser::generation_params & pInputs);

std::string fCommonChatTemplateGenerationPrompt(const common_chat_template &          pTmpl,
                                                const autoparser::generation_params & pInputs);

std::optional<common_chat_params> fCommonChatTrySpecializedTemplate(const common_chat_template &          pTmpl,
                                                                    const autoparser::generation_params & pParams);

// specialized per-task preset
struct common_chat_prompt_preset {
  std::string vSystem;
  std::string vUser;
};

common_chat_prompt_preset fCommonChatGetAsrPrompt(const common_chat_templates * pChatTemplates);
