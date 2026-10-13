#pragma once

#include "common.h"
#include "dgs.h"

#include <list>
#include <map>
#include <string>
#include <unordered_set>

// TODO: prevent including the whole server-common.h as we only use server_tokens
#include "server-common.h"

enum server_task_type {
  cServerTaskTypeCompletion,
  cServerTaskTypeEmbedding,
  cServerTaskTypeRerank,
  cServerTaskTypeDecision,
  cServerTaskTypeInfill,
  cServerTaskTypeCancel,
  cServerTaskTypeControl,
  cServerTaskTypeNextResponse,
  cServerTaskTypeMetrics,
  cServerTaskTypeSlotGet,
  cServerTaskTypeSlotSave,
  cServerTaskTypeSlotRestore,
  cServerTaskTypeSlotErase,
  cServerTaskTypeGetLora,
  cServerTaskTypeSetLora,
};

// TODO: change this to more generic "response_format" to replace the "format_response_*" in server-common
enum task_response_type {
  cTaskResponseTypeNone,  // dgs.cpp native format
  cTaskResponseTypeOaiChat,
  cTaskResponseTypeOaiCmpl,
  cTaskResponseTypeOaiResp,
  cTaskResponseTypeOaiAsr,  // transcriptions API
  cTaskResponseTypeOaiEmbd,
  cTaskResponseTypeAnthropic,
};

enum stop_type {
  cStopTypeNone,
  cStopTypeEos,
  cStopTypeWord,
  cStopTypeLimit,
};

struct task_params {
  bool vFromWebui      = false;
  bool vQueueNotice    = false;
  bool stream          = false;
  bool vIncludeUsage   = false;
  bool vCachePrompt    = true;  // remember the prompt to avoid reprocessing all prompt
  bool vReturnTokens   = false;
  bool vReturnProgress = false;

  int32_t vSsePingInterval = 30;  // seconds between SSE comment pings while the stream stays silent, -1 disables

  int32_t vNKeep = 0;             // number of tokens to keep from initial prompt
  int32_t vNDiscard =
    0;  // number of tokens after n_keep that may be discarded when shifting context, 0 defaults to half
  int32_t vNPredict = -1;       // new tokens to predict
  int32_t vNIndent  = 0;        // minimum line indentation for the generated text in number of whitespace characters
  int32_t vNCmpl    = 1;        // number of completions to generate from this prompt

  int32_t vNCacheReuse = 0;     // min chunk size to attempt reusing from the cache via KV shifting (0 = disabled)

  int64_t vTMaxPromptMs  = -1;  // TODO: implement
  int64_t vTMaxPredictMs = -1;  // if positive, limit the generation phase to this time limit

  std::map<int, float> lora;    // mapping adapter ID -> scale

  std::vector<std::string> lAntiprompt;
  std::vector<std::string> lResponseFields;

  bool vTimingsPerToken   = false;
  bool vPostSamplingProbs = false;

  struct common_params_sampling    vSampling;
  struct common_params_speculative vSpeculative;

  // response formatting
  bool               vVerbose = false;
  task_response_type vResType = cTaskResponseTypeNone;
  std::string        vOaicompatModel;
  std::string        vOaicompatCmplId;

  // realtime control (SERVER_TASK_TYPE_CONTROL)
  std::string vControlAction;
  std::string vControlCmplId;

  // reported in generation_settings, parsing itself is owned by the chat session
  common_chat_format      vChatFormat      = cCommonChatFormatContentOnly;
  common_reasoning_format vReasoningFormat = cCommonReasoningFormatNone;

  // message spans for checkpointing
  common_chat_msg_spans vMessageSpans;

  // Embeddings
  int32_t vEmbdNormalize = 2;  // (-1=none, 0=max absolute int16, 1=taxicab, 2=Euclidean/L2, >2=p-norm)

  json mFormatLogitBias(const std::vector<dgs_logit_bias> & pLogitBias) const;
  json to_json(bool pOnlyMetrics = false) const;
};

// struct for tracking the state of a task (e.g., for streaming)
struct task_result_state {
  // tracking diffs for partial tool calls
  std::vector<common_chat_msg_diff> lDiffs;
  common_chat_session               vChatSession;  // owns all parsing for this generation
  common_chat_msg                   vChatMsg;
  std::vector<std::string>          lGeneratedToolCallIds;
  std::unordered_set<size_t>        sSentToolCallNames;

  // for OpenAI Responses and Anthropic streaming API:
  // track output item / content block state across chunks
  bool vThinkingBlockStarted = false;
  bool vTextBlockStarted     = false;

  // for OpenAI Responses streaming API
  bool              vOaiRespCreated = false;
  const std::string cOaiRespId;
  const std::string cOaiRespReasoningId;
  const std::string cOaiRespMessageId;
  std::string       vOaiRespFcId;  // function call ID for current args delta

  task_result_state(common_chat_session pSession = {});

  // parse partial tool calls and update the internal state
  common_chat_msg mUpdateChatMsg(const common_chat_input &           pAdded,
                                 bool                                pIsPartial,
                                 std::vector<common_chat_msg_diff> & pDiffs,
                                 bool                                pFilterToolCalls = false);
};

struct server_task {
  int vId = -1;  // to be filled by server_queue

  // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
  size_t vIndex = 0;  // used when there are multiple prompts (batch request)

  // used by SERVER_TASK_TYPE_CANCEL
  int vIdTarget = -1;
  int vIdSlot   = -1;

  // used by parallel sampling (multiple completions from same prompt)
  int                      vIdParent  = -1;
  // temporary store of child tasks for scheduling
  // note: accessing to elements is invalid after the task is moved to server_slot
  bool                     vWasQueued = false;
  std::vector<server_task> lChildTasks;
  // if set on a parent, the children have their own prompt and only share its first n_tokens_shared tokens
  int32_t                  vNTokensShared = 0;

  // used by SERVER_TASK_TYPE_INFERENCE
  task_params   vParams;
  server_tokens vTokens;

  // only used by CLI, this allow tokenizing CLI inputs on server side
  // we need this because mtmd_context and vocab are not accessible outside of server_context
  bool                    vCli = false;
  std::string             vCliPrompt;
  std::vector<raw_buffer> lCliFiles;

  server_task_type type;

  // used by SERVER_TASK_TYPE_SLOT_SAVE, SERVER_TASK_TYPE_SLOT_RESTORE, SERVER_TASK_TYPE_SLOT_ERASE
  struct slot_action {
    int         vIdSlot;
    std::string vFilename;
    std::string vFilepath;
  };

  slot_action vSlotAction;

  // used by SERVER_TASK_TYPE_METRICS
  bool vMetricsResetBucket = false;

  // used by SERVER_TASK_TYPE_DECISION
  // where to read the model output of each option, exactly one of the two lists is used
  struct decision {
    std::vector<dgs_token> lLabels;       // logits of these tokens, at the last prompt token
    std::vector<int32_t>   lLabelGroups;  // if set, number of labels per output, the output is their max
    std::vector<int32_t>   lMarkers;      // embeddings[column] at these prompt positions
    int32_t                vColumn = 0;
    // if set, embeddings is [q | k], and the output is instead the scaled dot product of q[pointer] and k[marker]
    int32_t                pointer = -1;

    // first prompt position that is read, -1 if none
    int32_t mPosFirst() const {
      int32_t vPos = pointer;
      for (const int32_t cMarker : lMarkers) {
        vPos = vPos < 0 ? cMarker : std::min(vPos, cMarker);
      }
      for (size_t vI = 0; vI < lOrder.size(); vI++) {
        if (lOrder[vI] != 0) {
          return vPos < 0 ? (int32_t) vI : std::min(vPos, (int32_t) vI);
        }
      }
      return vPos;
    }

    // for a joint head: one value per prompt token, see dgs_batch_ext_set_decision_order()
    // the scores are the first n_scores rows of the embeddings
    std::vector<int32_t> lOrder;
    int32_t              vNScores = 0;
  };

  decision decision;

  // used by SERVER_TASK_TYPE_SET_LORA
  std::map<int, float> dSetLora;  // mapping adapter ID -> scale

  server_task() = default;

  server_task(server_task_type type) : type(type) {}

  int32_t mNTokens() const { return vTokens.size(); }

  bool mNeedEmbd() const {
    switch (type) {
      case cServerTaskTypeEmbedding:
      case cServerTaskTypeRerank:
        return true;
      case cServerTaskTypeDecision:
        return !decision.lMarkers.empty() || !decision.lOrder.empty();
      default:
        return false;
    }
  }

  bool mNeedLogits() const {
    switch (type) {
      case cServerTaskTypeCompletion:
      case cServerTaskTypeInfill:
        return true;
      case cServerTaskTypeDecision:
        return !decision.lLabels.empty();
      default:
        return false;
    }
  }

  bool mNeedSampling() const {
    switch (type) {
      case cServerTaskTypeCompletion:
      case cServerTaskTypeInfill:
        return true;
      default:
        return false;
    }
  }

  // utility function
  static std::unordered_set<int> mGetListId(const std::vector<server_task> & pTasks) {
    std::unordered_set<int> sIds(pTasks.size());
    for (size_t vI = 0; vI < pTasks.size(); vI++) {
      sIds.insert(pTasks[vI].vId);
      for (auto & vChild : pTasks[vI].lChildTasks) {
        sIds.insert(vChild.vId);
      }
    }
    return sIds;
  }

  void mApplyChatSession(const common_chat_session & pSession) {
    if (!pSession.mHasTemplate()) {
      return;
    }

    pSession.mApplySampling(vParams.vSampling);
    vParams.vChatFormat = pSession.format();
    vParams.lAntiprompt.insert(vParams.lAntiprompt.end(), pSession.mAdditionalStops().begin(),
                               pSession.mAdditionalStops().end());
    vParams.vMessageSpans = vTokens.mFindMessageSpans(pSession.mMessageDelimiters());
  }

  void mAddChild(int pIdParent, int pIdChild) {
    server_task copy;

    copy.vId       = pIdChild;
    copy.vIdParent = pIdParent;
    copy.vParams   = vParams;
    copy.type      = type;
    copy.vTokens   = vTokens.clone();
    copy.vIdSlot   = -1;  // child tasks cannot specify slot

    // use different sampling seed for each child
    // note: 
    if (copy.vParams.vSampling.vSeed != DGS_DEFAULT_SEED) {
      copy.vParams.vSampling.vSeed += (uint32_t) lChildTasks.size() + 1;
    }

    lChildTasks.push_back(std::move(copy));
  }

  bool mIsParent() const { return lChildTasks.size() > 0; }

  bool mIsChild() const { return vIdParent != -1; }
};

struct result_prompt_progress {
  int32_t vTotal     = 0;
  int32_t vCache     = 0;
  int32_t vProcessed = 0;
  int64_t vTimeMs    = 0;

  json to_json() const;
};

struct server_task_result {
  int vId     = -1;
  int vIdSlot = -1;

  // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
  size_t vIndex = 0;  // to be used for batched tasks

  virtual bool mIsError() {
    // only used by server_task_result_error
    return false;
  }

  virtual bool mIsStop() {
    // only used by server_task_result_cmpl_*
    return true;
  }

  virtual void mUpdate(task_result_state &) {
    // only used by server_task_result_cmpl_*
  }

  virtual json to_json()        = 0;
  virtual ~server_task_result() = default;

  virtual server_task_result * clone() const { GGML_ABORT("not implemented for this task type"); }
};

// using shared_ptr for polymorphism of server_task_result
using server_task_result_ptr = std::unique_ptr<server_task_result>;

struct completion_token_output {
  dgs_token   vTok;
  float       vProb;
  std::string vTextToSend;

  struct prob_info {
    dgs_token   vTok;
    std::string vTxt;
    float       vProb;
  };

  std::vector<prob_info> lProbs;

  json to_json(bool pPostSamplingProbs) const;

  static json mProbsVectorToJson(const std::vector<completion_token_output> & pProbs, bool pPostSamplingProbs);

  static float mLogarithm(float pX);

  static std::vector<unsigned char> mStrToBytes(const std::string & str);
};

struct server_task_result_cmpl_final : server_task_result {
  common_chat_input vContent;
  dgs_tokens        lTokens;

  bool              stream;
  bool              vIncludeUsage;
  server_slot_stats vStats;
  std::string       vPrompt;

  bool        vTruncated;
  int32_t     vNDecoded;
  int32_t     vNPromptTokens;
  int32_t     vNPromptTokensCache;
  int32_t     vNTokensCached;
  bool        vHasNewLine;
  std::string vStoppingWord;
  stop_type   vStop = cStopTypeNone;

  bool                                 vPostSamplingProbs;
  std::vector<completion_token_output> lProbsOutput;
  std::vector<std::string>             lResponseFields;

  task_params vGenerationParams;

  // response formatting
  bool               vVerbose = false;
  task_response_type vResType = cTaskResponseTypeNone;
  std::string        vOaicompatModel;
  std::string        vOaicompatCmplId;
  common_chat_msg    vOaicompatMsg;                      // to be populated by update()

  std::vector<common_chat_msg_diff> lOaicompatMsgDiffs;  // to be populated by update()
  bool                              vIsUpdated = false;

  // for OpenAI Responses API
  std::string vOaiRespId;
  std::string vOaiRespReasoningId;
  std::string vOaiRespMessageId;

  virtual bool mIsStop() override {
    return true;  // in stream mode, final responses are considered stop
  }

  virtual json to_json() override;

  virtual void mUpdate(task_result_state & pState) override {
    vIsUpdated    = true;
    vOaicompatMsg = pState.mUpdateChatMsg(vContent, false, lOaicompatMsgDiffs);

    vOaiRespId          = pState.cOaiRespId;
    vOaiRespReasoningId = pState.cOaiRespReasoningId;
    vOaiRespMessageId   = pState.cOaiRespMessageId;
  }

  json mToJsonNonOaicompat();

  json mUsageJsonOaicompat();

  json mToJsonOaicompat();

  json mToJsonOaicompatChat();

  json mToJsonOaicompatChatStream();

  json mToJsonOaicompatResp();

  json mToJsonOaicompatRespStream();

  json mToJsonOaicompatAsr();

  json mToJsonAnthropic();

  json mToJsonAnthropicStream();
};

struct server_task_result_cmpl_partial : server_task_result {
  common_chat_input vContent;
  dgs_tokens        lTokens;

  int32_t vNDecoded;
  int32_t vNPromptTokens;
  int32_t vNPromptTokensCache;

  bool                    vPostSamplingProbs;
  bool                    vIsProgress = false;
  bool                    vIsBegin    = false;  // whether to send 200 status to HTTP client (begin of SSE stream)
  completion_token_output vProbOutput;
  server_slot_stats       vStats;
  result_prompt_progress  vProgress;

  // response formatting
  bool                              vVerbose = false;
  task_response_type                vResType = cTaskResponseTypeNone;
  std::string                       vOaicompatModel;
  std::string                       vOaicompatCmplId;
  std::vector<common_chat_msg_diff> lOaicompatMsgDiffs;  // to be populated by update()
  bool                              vIsUpdated = false;

  // Streaming state copied from task_result_state for this chunk
  bool vThinkingBlockStarted = false;
  bool vTextBlockStarted     = false;

  // for OpenAI Responses API
  bool        vOaiRespCreated = false;
  std::string vOaiRespId;
  std::string vOaiRespReasoningId;
  std::string vOaiRespMessageId;
  std::string vOaiRespFcId;

  // for Anthropic API: track if any reasoning content has been generated
  bool vAnthropicHasReasoning = false;

  virtual bool mIsStop() override {
    return false;  // in stream mode, partial responses are not considered stop
  }

  virtual void mUpdate(task_result_state & pState) override;

  virtual json to_json() override;

  json mToJsonNonOaicompat();

  json mToJsonOaicompat();

  json mToJsonOaicompatChat();

  json mToJsonOaicompatResp();

  json mToJsonOaicompatAsr();

  json mToJsonAnthropic();
};

// Queue status of a task that waits for a free slot, streamed as {"dgs_queue": {...}} to the Web UI requests that ask
// for it: sent when the task is queued, when the requests ahead of it change and, with every count at 0, when it
// starts. API clients never receive it.
struct server_task_result_queue : server_task_result {
  int32_t vAhead      = 0;  // requests resolved before this one, running or waiting
  int32_t vAheadApi   = 0;  // of them, sent directly to the API
  int32_t vAheadWebui = 0;  // of them, sent by the Web UI

  virtual bool mIsStop() override {
    return false;  // the task keeps going
  }

  virtual json to_json() override;
};

struct server_task_result_embd : server_task_result {
  std::vector<std::vector<float>> lEmbedding;

  int32_t vNTokens;

  // response formatting
  task_response_type vResType = cTaskResponseTypeNone;

  virtual json to_json() override;

  json mToJsonNonOaicompat();

  json mToJsonOaicompat();
};

struct server_task_result_rerank : server_task_result {
  float vScore = -1e6;

  int32_t vNTokens;

  virtual json to_json() override;
};

struct server_task_result_decision : server_task_result {
  std::vector<float> lScores;  // one raw model output per option

  int32_t vNTokens;

  virtual json to_json() override;
};

struct server_task_result_error : server_task_result {
  error_type  vErrType = cErrorTypeServer;
  std::string vErrMsg;

  // for ERROR_TYPE_EXCEED_CONTEXT_SIZE
  int32_t vNPromptTokens = 0;
  int32_t vNCtx          = 0;

  virtual bool mIsError() override { return true; }

  virtual json to_json() override;
};

// used by /metrics API
struct server_task_result_metrics : server_task_result {
  // these are immediate stats, not accumulated (server_metrics is cumulative)
  int vNProcessingSlots = 0;
  int vNTasksDeferred   = 0;

  server_metrics vMetrics;

  virtual json to_json() override;

  struct metric_item {
    std::string name;
    std::string vDescription;
    double      value;  // prometheus values are always float64
  };

  std::string mToMetrics();
};

// used by /slots API
struct server_task_result_slots : server_task_result {
  int vNIdleSlots = 0;

  // while we can also use std::vector<server_slot> this requires copying the slot object which can be quite messy
  // therefore, we use json to temporarily store the slot.to_json() result
  json vSlotsData = json::mArray();

  virtual json to_json() override;
};

struct server_task_result_slot_save_load : server_task_result {
  std::string vFilename;
  bool        vIsSave;  // true = save, false = load

  size_t vNTokens;
  size_t vNBytes;
  double vTMs;

  virtual json to_json() override;
};

struct server_task_result_slot_erase : server_task_result {
  size_t vNErased;

  virtual json to_json() override;
};

struct server_task_result_control : server_task_result {
  bool        vSuccess = false;
  std::string vMessage;  // optional detail when success is false

  virtual json to_json() override {
    json vOut = json{
      { "success", vSuccess }
    };
    if (!vMessage.empty()) {
      vOut["message"] = vMessage;
    }
    return vOut;
  }
};

struct server_task_result_get_lora : server_task_result {
  struct lora {
    common_adapter_lora_info info;
    std::string              vAloraInvocationString;
    dgs_tokens               lAloraInvocationTokens;
  };

  std::vector<lora> lLoras;

  virtual json to_json() override;
};

struct server_task_result_apply_lora : server_task_result {
  virtual json to_json() override;
};

struct server_prompt {
  server_tokens vTokens;

  std::list<common_prompt_checkpoint> lCheckpoints;

  void clear() {
    vTokens.clear();
    lCheckpoints.clear();
  }

  int mNTokens() const { return vTokens.size(); }

  server_prompt clone() const {
    return server_prompt{
      vTokens.clone(),
      lCheckpoints,
    };
  }
};

struct server_prompt_data {
  std::vector<uint8_t> main;
  std::vector<uint8_t> lDrft;

  size_t size() const { return main.size() + lDrft.size(); }
};

struct server_prompt_cache_state {
  server_prompt      vPrompt;
  server_prompt_data data;

  size_t size() const {
    size_t vRes = data.size();

    for (const auto & vCkpt : vPrompt.lCheckpoints) {
      vRes += vCkpt.size();
    }

    return vRes;
  }
};

struct server_prompt_cache {
  server_prompt_cache(int32_t pLimitSizeMib, size_t pLimitTokens) {
    this->vLimitSize   = 1024ull * 1024ull * (pLimitSizeMib < 0 ? 0 : pLimitSizeMib);
    this->vLimitTokens = pLimitTokens;
  }

  std::list<server_prompt_cache_state> lStates;

  // in bytes, 0 = no limit
  size_t vLimitSize = 0;

  // in tokens, 0 = no limit
  size_t vLimitTokens = 0;

  size_t size() const;

  size_t mNTokens() const;

  server_prompt_cache_state * mAlloc(const server_prompt & pPrompt, size_t pStateSizeMain, size_t pStateSizeDrft);

  bool mLoad(server_prompt &       pPrompt,
             const server_tokens & pTokensNew,
             dgs_context *         pCtxTgt,
             dgs_context *         pCtxDft,
             int32_t               pIdSlot);

  void mUpdate();
};

// used exclusively by router mode
struct server_task_result_router : server_task_result {
  json data;

  virtual json to_json() override { return data; }

  virtual server_task_result * clone() const override { return new server_task_result_router(*this); }
};
