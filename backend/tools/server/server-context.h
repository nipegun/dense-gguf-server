#pragma once

#include "json.h"
#include "server-http.h"
#include "server-queue.h"
#include "server-task.h"

#include <cstddef>
#include <memory>
#include <mutex>
#include <set>

struct server_context_impl;  // private implementation

struct server_context_meta {
  std::string              vBuildInfo;
  std::string              vModelName;
  std::set<std::string>    sModelAliases;
  std::set<std::string>    sModelTags;
  std::string              vModelPath;
  std::vector<std::string> lModelOutputModalities;  // output modalities for GET /models
  bool                     vHasMtmd;
  bool                     vHasInpImage;
  bool                     vHasInpAudio;
  bool                     vHasInpVideo;
  json                     vJsonUiSettings;
  int                      vSlotNCtx;
  enum fDgsPoolingType     vPoolingType;

  // chat params
  server_chat_params &        vChatParams;
  std::map<std::string, bool> dChatTemplateCaps;

  // tokens
  std::string vBosTokenStr;
  std::string vEosTokenStr;
  dgs_token   vFimPreToken;
  dgs_token   vFimSubToken;
  dgs_token   vFimMidToken;
  dgs_token   vFimPadToken;
  dgs_token   vFimRepToken;
  dgs_token   vFimSepToken;

  // sampling
  std::vector<dgs_logit_bias> lLogitBiasEog;

  // model meta
  enum fDgsVocabType vModelVocabType;
  int32_t            vModelVocabNTokens;
  int32_t            vModelNCtxTrain;
  int32_t            vModelNEmbdInp;
  uint64_t           vModelNParams;
  uint64_t           vModelSize;
  std::string        vModelFtype;
};

enum server_state {
  cServerStateDownloading,
  cServerStateLoading,
  cServerStateReady,
  cServerStateSleeping,
};

static std::string fServerStateToStr(server_state pState) {
  switch (pState) {
    case cServerStateDownloading:
      return "downloading";
    case cServerStateLoading:
      return "loading";
    case cServerStateReady:
      return "ready";
    case cServerStateSleeping:
      return "sleeping";
    default:
      GGML_ASSERT(false && "invalid server_state");
  }
}

static server_state fServerStateFromStr(const std::string & str) {
  if (str == "downloading") {
    return cServerStateDownloading;
  }
  if (str == "loading") {
    return cServerStateLoading;
  }
  if (str == "ready") {
    return cServerStateReady;
  }
  if (str == "sleeping") {
    return cServerStateSleeping;
  }
  GGML_ASSERT(false && "invalid server_state string");
}

using server_state_callback_t = std::function<void(server_state, json /* payload */)>;

struct server_context {
  std::unique_ptr<server_context_impl> impl;

  server_context();
  ~server_context();

  // load the model and initialize dgs_context
  // returns true on success
  bool mLoadModel(common_params & pParams);

  // this function will block main thread until termination
  void mStartLoop();

  // terminate main loop (will unblock start_loop)
  json mGetMonitorSnapshot() const;
  void mTerminate();

  // get the underlaying dgs_context, can return nullptr if sleeping
  // not thread-safe, should only be used from the main thread
  dgs_context * mGetDgsContext() const;

  // get a new response reader, used by CLI application
  server_response_reader mGetResponseReader();

  // get server metadata (read-only), can only be called after load_model()
  // not thread-safe, should only be used from the main thread
  server_context_meta mGetMeta() const;

  // note: must be set before load_model() is called
  void mSetStateCallback(server_state_callback_t pCallback);
};

// forward declarations
struct server_res_generator;

struct server_routes {
  server_routes(const common_params & pParams, server_context & pCtxServer);

  void mInitRoutes();

  // note: this is not thread-safe and can only when ctx_http.is_ready is false
  void mUpdateMeta(const server_context & pCtxServer) {
    this->vMeta = std::make_unique<server_context_meta>(pCtxServer.mGetMeta());
  }

  // handlers using lambda function, so that they can capture `this` without `std::bind`
  // they won't be called until ctx_http.is_ready is set to true
  server_http_context::handler_t vGetHealth;
  server_http_context::handler_t vGetMonitor;
  server_http_context::handler_t vGetMonitorMemory;
  server_http_context::handler_t vGetMonitorUsage;
  server_http_context::handler_t vGetMetrics;
  server_http_context::handler_t vGetSlots;
  server_http_context::handler_t vPostSlots;
  server_http_context::handler_t vGetProps;
  server_http_context::handler_t vPostProps;
  server_http_context::handler_t vPostInfill;
  server_http_context::handler_t vPostCompletions;
  server_http_context::handler_t vPostCompletionsOai;
  server_http_context::handler_t vPostChatCompletions;
  server_http_context::handler_t vPostChatCompletionsTok;
  server_http_context::handler_t vPostControl;
  server_http_context::handler_t vPostResponsesOai;
  server_http_context::handler_t vPostResponsesTokOai;
  server_http_context::handler_t vPostTranscriptionsOai;
  server_http_context::handler_t vPostAnthropicMessages;
  server_http_context::handler_t vPostAnthropicCountTokens;
  server_http_context::handler_t vPostApplyTemplate;
  server_http_context::handler_t vGetModels;
  server_http_context::handler_t vPostTokenize;
  server_http_context::handler_t vPostDetokenize;
  server_http_context::handler_t vPostEmbeddings;
  server_http_context::handler_t vPostEmbeddingsOai;
  server_http_context::handler_t vPostRerank;
  server_http_context::handler_t vPostSystemone;
  server_http_context::handler_t vGetLoraAdapters;
  server_http_context::handler_t vPostLoraAdapters;

  // to be used in router mode
  json mGetModelInfo() const;

private:
  std::unique_ptr<server_res_generator> mHandleCompletionsImpl(const server_http_req &         pReq,
                                                               server_task_type                type,
                                                               const json &                    data,
                                                               const std::vector<raw_buffer> & pFiles,
                                                               task_response_type              pResType,
                                                               const common_chat_session &     pChatSession = {});
  std::unique_ptr<server_res_generator> mHandleSlotsSave(const server_http_req & pReq, int pIdSlot);
  std::unique_ptr<server_res_generator> mHandleSlotsRestore(const server_http_req & pReq, int pIdSlot);
  std::unique_ptr<server_res_generator> mHandleSlotsErase(const server_http_req &, int pIdSlot);
  std::unique_ptr<server_res_generator> mHandleEmbeddingsImpl(const server_http_req & pReq,
                                                              task_response_type      pResType);
  std::unique_ptr<server_res_generator> mHandleCountTokens(const server_http_req & pReq, task_response_type pResType);

  // using unique_ptr to allow late initialization of const
  std::unique_ptr<const server_context_meta> vMeta;

  const common_params & vParams;
  server_context_impl & vCtxServer;

  server_queue &                        vQueueTasks;
  server_response &                     vQueueResults;
  std::unique_ptr<server_res_generator> mCreateResponse(bool pBypassSleep = false);

  // cached responses, to be used during sleep
  std::mutex     vMutexCache;
  json           vCachedModels = nullptr;
  json           vCachedProps  = nullptr;
  server_metrics vCachedMetrics;
  // set when a scrape during sleep already reported the throughput buckets
  bool           vShouldResetBuckets = false;
  // call right before sleep to update the cached responses
  void           mUpdateCachedResponses(bool pIsSleeping);
};
