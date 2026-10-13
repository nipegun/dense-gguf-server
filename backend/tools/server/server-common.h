#pragma once

#include "chat.h"
#include "common.h"
#include "dgs.h"
#include "json.h"
#include "log.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "subproc.h"

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

using json = common_json;

#define SLT_DBG(slot, fmt, ...)                                                                                      \
  LOG_DBG("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).vId, ((slot).vTask ? (slot).vTask->vId : -1), \
          __VA_ARGS__)
#define SLT_TRC(slot, fmt, ...)                                                                                      \
  LOG_TRC("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).vId, ((slot).vTask ? (slot).vTask->vId : -1), \
          __VA_ARGS__)
#define SLT_INF(slot, fmt, ...)                                                                                      \
  LOG_INF("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).vId, ((slot).vTask ? (slot).vTask->vId : -1), \
          __VA_ARGS__)
#define SLT_WRN(slot, fmt, ...)                                                                                      \
  LOG_WRN("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).vId, ((slot).vTask ? (slot).vTask->vId : -1), \
          __VA_ARGS__)
#define SLT_ERR(slot, fmt, ...)                                                                                      \
  LOG_ERR("slot %12.*s: id %2d | task %d | " fmt, 12, __func__, (slot).vId, ((slot).vTask ? (slot).vTask->vId : -1), \
          __VA_ARGS__)
#define SLT_CNT(slot, fmt, ...) LOG_CNT("" fmt, __VA_ARGS__)

#define SRV_DBG(fmt, ...) LOG_DBG("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_TRC(fmt, ...) LOG_TRC("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_INF(fmt, ...) LOG_INF("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_WRN(fmt, ...) LOG_WRN("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_ERR(fmt, ...) LOG_ERR("srv  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SRV_CNT(fmt, ...) LOG_CNT("" fmt, __VA_ARGS__)

using raw_buffer = std::vector<uint8_t>;

template <typename T> static T fJsonValue(const json & pBody, const std::string & pKey, const T & pDefaultValue) {
  // Fallback null to default value
  if (pBody.contains(pKey) && !pBody.at(pKey).is_null()) {
    try {
      return pBody.at(pKey).get<T>();
    } catch (const common_json_error & vErr) {
      LOG_WRN("Wrong type supplied for parameter '%s', using default value: %s\n", pKey.c_str(), vErr.what());
      return pDefaultValue;
    }
  } else {
    return pDefaultValue;
  }
}

// https://community.openai.com/t/openai-chat-list-of-error-codes-and-types/357791/11
enum error_type {
  cErrorTypeInvalidRequest,
  cErrorTypeAuthentication,
  cErrorTypeServer,
  cErrorTypeNotFound,
  cErrorTypePermission,
  cErrorTypeUnavailable,        // custom error
  cErrorTypeNotSupported,       // custom error
  cErrorTypeExceedContextSize,  // custom error
};

// thin wrapper around common_grammar_trigger with (de)serialization functions
struct server_grammar_trigger {
  common_grammar_trigger value;

  server_grammar_trigger() = default;

  server_grammar_trigger(const common_grammar_trigger & value) : value(value) {}

  server_grammar_trigger(const json & pIn) {
    value.type  = (common_grammar_trigger_type) pIn.at("type").get<int>();
    value.value = pIn.at("value").get<std::string>();
    if (value.type == cCommonGrammarTriggerTypeToken) {
      value.token = (dgs_token) pIn.at("token").get<int>();
    }
  }

  json to_json() const {
    json vOut{
      { "type",  (int) value.type },
      { "value", value.value      },
    };
    if (value.type == cCommonGrammarTriggerTypeToken) {
      vOut["token"] = (int) value.token;
    }
    return vOut;
  }
};

json fFormatErrorResponse(const std::string & pMessage, const enum error_type type);

//
// random string / id
//

std::string fRandomString();
std::string fGenChatcmplid();
std::string fGenToolCallId();

// get a random marker; note: each time the server restarts, the marker will be different
const char * fGetMediaMarker();

//
// model output modalities
//

// output modalities for architecture.output_modalities in GET /models
std::vector<std::string> fServerModelOutputModalities(common_decision_type pDecisionType);

// architecture object of GET /models; shared by the direct server and the router
json fServerModelArchitectureJson(bool                             pInpImage,
                                  bool                             pInpAudio,
                                  bool                             pInpVideo,
                                  const std::vector<std::string> & pOutputModalities);

//
// lora utils
//

// check whether the given lora set has only aloras activated (empty => false)
bool fLoraAllAlora(const std::vector<common_adapter_lora_info> & pLoras);

// if the two sets of loras are different, they require a cache clear unless the
// change is only from aloras to aloras.
bool fLoraShouldClearCache(const std::vector<common_adapter_lora_info> & pCurrent,
                           const std::vector<common_adapter_lora_info> & pNext);

std::map<int, float> fParseLoraRequest(const json & data);

bool fAreLoraEqual(const std::vector<common_adapter_lora_info> & pL1,
                   const std::vector<common_adapter_lora_info> & pL2);

// get the ids of all enabled loras
std::vector<size_t> fLoraGetEnabledIds(const std::vector<common_adapter_lora_info> & pLoras);

//
// server_tokens
//

/**
 * server_tokens is a helper to manage the input tokens and image for the server.
 * it is made this way to simplify the logic of KV cache management.
 */
struct server_tokens {
  bool has_mtmd = false;

private:  // disallow accessing these members directly, risking out-of-sync
  // map a **start** index in tokens to the image chunk
  // note: the order need to be in-sync with tokens
  std::map<size_t, mtmd::input_chunk_ptr> map_idx_to_media;

  // list of tokens
  //   if the token is DGS_TOKEN_NULL, it indicates that this position is occupied by media chunk
  //   otherwise, it is a normal text token
  // note: a non-text chunk can occupy multiple tokens (aka memory cells) in the token list
  // note(2): for M-RoPE, an image can occupy different number of pos; do not assume 1-to-1 mapping tokens <-> pos
  dgs_tokens tokens;

  // for ex. with input of 5 text tokens and 2 images (each image occupies 3 tokens and 2 pos):
  //      [0] [1] [2] [3] [4] [img0] [img0] [img0] [img1] [img1] [img1]
  // idx  0   1   2   3   4   5      6      7      8      9      10
  // pos  0   1   2   3   4   5      5      5      7      7      7
  // map_idx_to_media will contain: {5, img0}, {8, img1}

public:
  server_tokens()  = default;
  ~server_tokens() = default;

  // Prevent copying
  // TODO: server_tokens should be copyable - remove this:
  server_tokens(const server_tokens &)             = delete;
  server_tokens & operator=(const server_tokens &) = delete;

  // Allow moving (usually implicitly generated if members are movable)
  server_tokens(server_tokens &&)             = default;
  server_tokens & operator=(server_tokens &&) = default;

  // Allow accessing elements using [] operator
  dgs_token operator[](size_t pIndex) { return tokens[pIndex]; }

  const dgs_token & operator[](size_t pIndex) const { return tokens[pIndex]; }

  server_tokens(mtmd::input_chunks & pMtmdChunks, bool pHasMtmd);
  server_tokens(const dgs_tokens & pTokens, bool pHasMtmd);

  // for debugging
  std::string str() const;

  // the next position after n_tokens. if n_tokens < 0, return the next position after all tokens.
  dgs_pos mPosNext(int64_t pNTokens = -1) const;

  // number of tokens with position < max_pos
  size_t mSizeUpToPos(dgs_pos pMaxPos) const;

  const mtmd::input_chunk_ptr & mFindChunk(size_t pIdx) const;

  // find next media chunk after idx
  // returns a pair of pointer to the chunk (nullptr if not found) and its start index in tokens
  std::pair<const mtmd::input_chunk_ptr *, size_t> mFindNextMediaChunk(size_t pIdx) const;

  void push_back(dgs_token pTok);

  // will create a copy of the chunk if it contains non-text data
  void push_back(const mtmd_input_chunk * pChunk);

  // same as push_back, but media chunks are stored as placeholders (no image/audio data)
  // only use this if the chunk will never be encoded again (e.g. it is already in the KV cache)
  void mPushBackPlaceholder(const mtmd_input_chunk * pChunk);

  // appends server tokens, updates the media map. copies media chunks.
  void push_back(server_tokens & pTokens);

  // for compatibility with context shift and prompt truncation
  void insert(const dgs_tokens & pInpTokens);

  // for compatibility with speculative decoding, ctx shift
  const dgs_tokens & mGetTokens() const;

  dgs_tokens mGetTextTokens() const;

  std::vector<char>    serialize() const;
  static server_tokens deserialize(const dgs_tokens & pPacked, bool pHasMtmd);

  // for compatibility with speculative decoding
  void mSetToken(dgs_pos pPos, dgs_token pId);

  size_t size() const { return tokens.size(); }

  bool empty() const { return tokens.empty(); }

  void clear() {
    map_idx_to_media.clear();
    tokens.clear();
  }

  void mKeepFirst(size_t n);

  std::string mDetokenize(const dgs_context * pCtx, bool pSpecial) const;

  size_t mGetCommonPrefix(const server_tokens & pB) const;

  // split the tokens into message spans, skipping over media chunks
  common_chat_msg_spans mFindMessageSpans(const common_chat_msg_delimiters & pDelims) const;

  // check text token IDs and the mapping between media chunks and token ranges
  bool mValidate(const struct dgs_context * pCtx) const;

  server_tokens clone() const;
};

//
// tokenizer and input processing utils
//

bool fJsonIsArrayOfNumbers(const json & data);

// is array having BOTH numbers & strings?
bool fJsonIsArrayOfMixedNumbersStrings(const json & data);

// does array have any individual integers/tokens?
bool fJsonIsArrayAndContainsNumbers(const json & data);

// get value by path(key1 / key2)
json fJsonGetNestedValues(const std::vector<std::string> & pPaths, const json & pJs);

/**
 * this handles 2 cases:
 * - only string, example: "string"
 * - mixed string and tokens, example: [12, 34, "string", 56, 78]
 */
dgs_tokens fTokenizeMixed(const dgs_vocab * pVocab, const json & pJsonPrompt, bool pAddSpecial, bool pParseSpecial);

// return the last index of character that can form a valid string
// if the last character is potentially cut in half, return the index before the cut
// if validate_utf8(text) == text.size(), then the whole text is valid utf8
size_t fValidateUtf8(const std::string & pText);

// load a media file from an URL (http, file, data) or from raw base64 data
void fHandleMedia(std::vector<raw_buffer> & pOutFiles, const std::string & pUrl, const std::string & pMediaPath);

// process mtmd prompt, return the server_tokens containing both text tokens and media chunks
// if is_placeholder is true, the media chunk will be treated as placeholder for counting tokens; the output tokens are not usable for actual inference (e.g. for submitting a task to server_queue)
server_tokens fProcessMtmdPrompt(mtmd_context *                  pMctx,
                                 const std::string &             pPrompt,
                                 const std::vector<raw_buffer> & pFiles,
                                 const mtmd_helper_init_opt &    pInitOpt,
                                 bool                            pIsPlaceholder = false);

/**
 * break the input "prompt" object into multiple prompt if needed, then tokenize them
 * this supports these cases:
 * - "prompt": "string"
 * - "prompt": [12, 34, 56]
 * - "prompt": [12, 34, "string", 56, 78]
 * - "prompt": { "prompt_string": "string", "multimodal_data": [ "base64" ] }
 * and multiple prompts (multi-tasks):
 * - "prompt": ["string1", "string2"]
 * - "prompt": ["string1", [12, 34, 56]]
 * - "prompt": [[12, 34, 56], [78, 90, 12]]
 * - "prompt": [[12, 34, "string", 56, 78], [12, 34, 56], { "prompt_string": "string", "multimodal_data": [ "base64" ]}]
 */
std::vector<server_tokens> fTokenizeInputPrompts(const dgs_vocab *            pVocab,
                                                 mtmd_context *               pMctx,
                                                 const json &                 pJsonPrompt,
                                                 bool                         pAddSpecial,
                                                 bool                         pParseSpecial,
                                                 const mtmd_helper_init_opt & pInitOpt);

// tokenize a single prompt, see tokenize_input_prompts() for the supported shapes
server_tokens fTokenizeInputSubprompt(const dgs_vocab *            pVocab,
                                      mtmd_context *               pMctx,
                                      const json &                 pJsonPrompt,
                                      bool                         pAddSpecial,
                                      bool                         pParseSpecial,
                                      const mtmd_helper_init_opt & pInitOpt);

//
// OAI utils
//

// global server parameters for chat formatting / parsing
struct server_chat_params {
  bool                               vUseJinja;
  bool                               vPrefillAssistant;
  common_reasoning_format            vReasoningFormat;
  std::map<std::string, std::string> dChatTemplateKwargs;  // mapping key --> json value
  common_chat_templates_ptr          vTmpls;
  bool                               vAllowImage;
  bool                               vAllowAudio;
  bool                               vAllowVideo;
  bool                               vEnableThinking  = true;
  int                                vReasoningBudget = -1;
  std::string                        vReasoningBudgetMessage;
  std::string                        vMediaPath;
  bool                               vForcePureContent = false;
};

// used by /completions endpoint
json fOaicompatCompletionParamsParse(const json & pBody);

// used by /chat/completions endpoint
json fOaicompatChatParamsParse(const dgs_vocab *          pVocab,
                               json &                     pBody, /* openai api json semantics */
                               const server_chat_params & pOpt,
                               std::vector<raw_buffer> &  pOutFiles,
                               common_chat_session &      pOutSession);

// used by /embeddings endpoint, content has the same format as a chat message content array
server_tokens fTokenizeOaiContentArray(const dgs_vocab *            pVocab,
                                       mtmd_context *               pMctx,
                                       const server_chat_params &   pOpt,
                                       json                         pContent,
                                       bool                         pAddSpecial,
                                       bool                         pParseSpecial,
                                       const mtmd_helper_init_opt & pInitOpt);

// TODO: move it to server-task.cpp
json fFormatEmbeddingsResponseOaicompat(const json &        pRequest,
                                        const std::string & pModelName,
                                        const json &        pEmbeddings,
                                        bool                pUseBase64 = false);

// TODO: move it to server-task.cpp
json fFormatResponseRerank(const json &               pRequest,
                           const std::string &        pModelName,
                           const json &               pRanks,
                           bool                       pIsTeiFormat,
                           std::vector<std::string> & pTexts,
                           int                        pTopN);

//
// stats and metrics
//

// shared between server_slot and server_task_result_*
struct server_slot_stats {
  uint64_t vNPromptCached    = 0;
  uint64_t vNPromptProcessed = 0;
  uint64_t vNGen             = 0;

  // speculative decoding stats
  // note: the per-position breakdown lives in server_slot, it is not needed in a task result
  uint64_t vNDraftTokens     = 0;
  uint64_t vNDraftAccepted   = 0;
  uint64_t vNDraftVerifSteps = 0;

  // these are absolute timestamps (in us)
  // note: must be signed - they are subtracted before the later ones are set
  int64_t vTStart      = 0;
  int64_t vTPromptLast = 0;
  int64_t vTGenLast    = 0;

  // can only move one direction: start -> prompt -> gen
  void mUpdatePromptStart() {
    GGML_ASSERT(vTStart == 0);
    vTStart = ggml_time_us();
  }

  void mSetPromptLast(int64_t pTUs) {
    GGML_ASSERT(vTStart > 0);
    vTPromptLast = pTUs;
  }

  void mUpdatePromptLast() { mSetPromptLast(ggml_time_us()); }

  void mUpdateGenLast() {
    GGML_ASSERT(vTPromptLast > 0);
    vTGenLast = ggml_time_us();
  }

  // these are time durations
  int64_t mTElapsedUs() const { return ggml_time_us() - vTStart; }

  double mTPromptMs() const {
    if (vTPromptLast == 0) {
      return 0.0;  // the prompt is not processed yet
    }
    return (vTPromptLast - vTStart) / 1000.0;
  }

  int64_t mTGenUs() const {
    if (vTGenLast == 0) {
      return 0;  // the generation is not started yet
    }
    // clamp to 1 us, the first token can land in the same us as t_prompt_last
    return std::max<int64_t>(1, vTGenLast - vTPromptLast);
  }

  double mTGenMs() const { return mTGenUs() / 1000.0; }

  // number of decode steps spent on generation
  // the first token is free, it comes from the logits of the last prompt batch
  uint64_t mNGenSteps() const { return vNGen > 0 ? vNGen - 1 : 0; }

  // other derived metrics
  // note: all of them return 0.0 if the divisor is not known yet
  double mTPromptPerTokenMs() const { return vNPromptProcessed > 0 ? mTPromptMs() / vNPromptProcessed : 0.0; }

  double mTGenPerTokenMs() const { return mNGenSteps() > 0 ? mTGenMs() / mNGenSteps() : 0.0; }

  double mNPromptTps() const {
    const double cTMs = mTPromptMs();
    return cTMs > 0.0 ? 1e3 / cTMs * vNPromptProcessed : 0.0;
  }

  double mNGenTps() const {
    const double cTMs = mTGenMs();
    return cTMs > 0.0 ? 1e3 / cTMs * mNGenSteps() : 0.0;
  }

  // false if the slot never started, i.e. the task result carries no stats
  bool mIsSet() const { return vTStart > 0; }

  json to_json() const;
};

// shared between server_context_impl and server_task_result_*
// unlike server_slot_stats, server_metrics is server-global and cumulative, not tied to a slot
struct server_metrics {
  int64_t vTStart = 0;

  struct bucket {
    uint64_t count  = 0;  // number of tokens
    uint64_t vSteps = 0;  // number of decode steps,
                          // this excludes first generated token (logits from prompt batch)
    uint64_t vTime  = 0;  // in microseconds

    // the rate uses the decode steps, so that "free" tokens do not inflate it
    double mNPerSecond() const { return vTime > 0 ? (double) vSteps / (double) vTime * 1e6 : 0.0; }

    void mAdd(uint64_t n, uint64_t pNSteps, uint64_t pTUs) {
      count += n;
      vSteps += pNSteps;
      vTime += pTUs;
    }
  };

  // these are reset by reset_bucket(), only the rate is read from them
  bucket vPromptBucket;
  bucket vPredictBucket;

  // metrics below are cumulative since the server started
  bucket vPrompt;  // only processed tokens, cached ones are counted separately below
  bucket vPredict;

  // tokens reused from the cache need no decode, so they only have a count
  uint64_t vNPromptCached = 0;

  uint64_t vNTokensMax = 0;

  uint64_t vNDecode    = 0;
  uint64_t vNBusySlots = 0;

  uint64_t              vNDraftTokens     = 0;  // Total draft tokens generated
  uint64_t              vNDraftAccepted   = 0;  // Draft tokens actually accepted
  uint64_t              vNDraftVerifSteps = 0;  // Total draft token verification steps by the target model
  std::vector<uint64_t> lNAcceptedPerPos;       // Accepted tokens per draft position

  void init() { vTStart = ggml_time_us(); }

  void mResetBucket() {
    vPromptBucket  = {};
    vPredictBucket = {};
  }

  void mAddPrompt(uint64_t pNTokens, uint64_t pTUs) {
    vPrompt.mAdd(pNTokens, pNTokens, pTUs);
    vPromptBucket.mAdd(pNTokens, pNTokens, pTUs);
  }

  void mAddPromptCached(uint64_t pNTokens) { vNPromptCached += pNTokens; }
};

//
// other utils
//

std::vector<dgs_token_data> fGetTokenProbabilities(dgs_context * pCtx, int pIdx, size_t pNTop);

std::string fSafeJsonToStr(const json & data);

std::string fTokensToStr(dgs_context * pCtx, const dgs_tokens & pTokens);
std::string fTokensToStr(const dgs_vocab * pVocab, const dgs_tokens & pTokens);

// format incomplete utf-8 multibyte character for output
std::string fTokensToOutputFormattedString(const dgs_context * pCtx, const dgs_token token);

// format server-sent event (SSE), return the formatted string to send
// note: if data is a json array, it will be sent as multiple events, one per item
std::string fFormatOaiSse(const json & data);

std::string fFormatOaiRespSse(const json & data);

// format Anthropic-style SSE with event types
std::string fFormatAnthropicSse(const json & data);

bool fIsValidUtf8(const std::string & str);

//
// formatting output responses
// TODO: move these to server-task.cpp
//

dgs_tokens fFormatPromptInfill(const dgs_vocab *  pVocab,
                               const json &       pInputPrefix,
                               const json &       pInputSuffix,
                               const json &       pInputExtra,
                               const int          pNBatch,
                               const int          pNPredict,
                               const int          pNCtx,
                               const bool         pSpmInfill,
                               const dgs_tokens & pTokensPrompt);

// format rerank task: [BOS]query[EOS][SEP]doc[EOS].
server_tokens fFormatPromptRerank(const struct dgs_model *     pModel,
                                  const struct dgs_vocab *     pVocab,
                                  mtmd_context *               pMctx,
                                  const std::string &          pQuery,
                                  const std::string &          pDoc,
                                  const mtmd_helper_init_opt & pInitOpt);

// simple implementation of a pipe
// used for streaming data between threads
template <typename T> struct server_pipe {
  std::mutex              vMutex;
  std::condition_variable vCv;
  std::queue<T>           lQueue;
  std::atomic<bool>       vWriterClosed{ false };
  std::atomic<bool>       vReaderClosed{ false };

  // 0 = unbounded (default)
  // > 0, write() drops the oldest item once the queue is full
  size_t vMaxSize = 0;

  void mCloseWrite() {
    vWriterClosed.store(true, std::memory_order_relaxed);
    vCv.notify_all();
  }

  void mCloseRead() {
    vReaderClosed.store(true, std::memory_order_relaxed);
    vCv.notify_all();
  }

  // close_on_stop = true: should_stop means the reader is gone for good, so the writer is told the pipe is broken.
  // close_on_stop = false: should_stop is a per-read deadline and further reads still come, so the pipe stays usable.
  bool read(T & pOutput, const std::function<bool()> & pShouldStop, bool pCloseOnStop = true) {
    std::unique_lock<std::mutex> vLk(vMutex);
    constexpr auto               cPollInterval = std::chrono::milliseconds(500);
    while (true) {
      if (!lQueue.empty()) {
        pOutput = std::move(lQueue.front());
        lQueue.pop();
        return true;
      }
      if (vWriterClosed.load()) {
        return false;                      // clean EOF
      }
      if (pShouldStop && pShouldStop()) {  // a null should_stop means "never stop"
        if (pCloseOnStop) {
          mCloseRead();                    // signal broken pipe to writer
        }
        return false;                      // cancelled / deadline reached
      }
      vCv.wait_for(vLk, cPollInterval);
    }
  }

  bool write(T && data) {
    std::lock_guard<std::mutex> vLk(vMutex);
    if (vReaderClosed.load()) {
      return false;  // broken pipe
    }
    if (vMaxSize > 0) {
      while (lQueue.size() >= vMaxSize) {
        lQueue.pop();  // drop oldest to stay bounded
      }
    }
    lQueue.push(std::move(data));
    vCv.notify_one();
    return true;
  }
};

// a child server writes its state commands to stdout and its logs to stderr
enum server_subproc_stream {
  cServerSubprocStdout,
  cServerSubprocStderr,
  cServerSubprocStreams,
};

// gives the current stdout to the caller as a stream of its own, and sends everything else written to stdout to stderr
FILE * fServerReserveStdout();

// wrapper around common_subproc to manage a child server process
// mainly used by router mode
struct server_subproc {
  common_subproc    vSproc;
  std::atomic<bool> vStopped{ false };  // set by the monitor once the process exited and was reaped

  bool mIsAlive() { return vSproc.mAlive(); }

  void mTerminate() { vSproc.mTerminate(); }

  int mJoin() { return vSproc.mJoin(); }

  // true if both output pipes of the child are available (call after create())
  bool mHasOutput();

  // true once both output pipes are closed
  bool mOutputClosed() const;

  // non-blocking read from one output pipe
  // returns the number of bytes read, 0 when nothing is available, -1 when the pipe is closed or broken
  int mReadOutput(server_subproc_stream stream, char * pBuf, size_t pLen);

  // wait until one of a set of children has output, wake() is called, or a timeout passes
  struct waiter {
    waiter();
    ~waiter();

    // thread-safe; on Windows this is a no-op, wait() returns within 50 ms anyway
    void mWake();

    // timeout_ms < 0 waits until data or wake(); ready[i] is set for each child with data on an open pipe (or a broken pipe)
    void mWait(const std::vector<server_subproc *> & pProcs, std::vector<bool> & pReady, int64_t pTimeoutMs);

  private:
#ifndef _WIN32
    intptr_t aWakeFd[2] = { -1, -1 };  // POSIX self-pipe
#endif
  };

private:
  intptr_t aOutHandles[cServerSubprocStreams] = { -1, -1 };  // fd on POSIX, HANDLE on Windows; taken lazily from sproc
  bool     aOutClosed[cServerSubprocStreams]  = { false, false };
};
