#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "dgs.h"
#include "fit.h"
#include "log.h"
#include "server-context.h"
#include "server-cors-proxy.h"
#include "server-http.h"
#include "server-models.h"
#include "server-shell-mcp.h"
#include "server-stream.h"
#include "server-tools.h"
#include "server-ui-settings.h"

#include <signal.h>

#include <atomic>
#include <clocale>
#include <exception>
#include <thread>  // for std::thread::hardware_concurrency

#if defined(_WIN32)
#  include <windows.h>
#endif

static std::function<void(int)> vShutdownHandler;
static std::atomic_flag         vIsTerminating = ATOMIC_FLAG_INIT;

static inline void fSignalHandler(int pSignal) {
  if (vIsTerminating.test_and_set()) {
    // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
    // this is for better developer experience, we can remove when the server is stable enough
    fprintf(stderr, "Received second interrupt, terminating immediately.\n");
    exit(1);
  }

  vShutdownHandler(pSignal);
}

// satisfies -Wmissing-declarations (used by dgs command)
int fDgsServer(int pArgc, char ** pArgv);

// to be used via CLI (argc / argv are used by router mode only)
int  fDgsServer(common_params & pParams, int pArgc, char ** pArgv);
int  fDgsServer(common_params & pParams, int pArgc, char ** pArgv, server_child & pChild);
void fDgsServerTerminate();

void fDgsServerTerminate() {
  if (vShutdownHandler) {
    vShutdownHandler(0);
  }
}

// wrapper function that handles exceptions and logs errors
// this is to make sure handler_t never throws exceptions; instead, it returns an error response
static server_http_context::handler_t fExWrapper(server_http_context::handler_t pFunc) {
  return [vFunc = std::move(pFunc)](const server_http_req & pReq) -> server_http_res_ptr {
    std::string vMessage;
    error_type  vError;
    try {
      return vFunc(pReq);
    } catch (const std::invalid_argument & vE) {
      // treat invalid_argument as invalid request (400)
      vError   = cErrorTypeInvalidRequest;
      vMessage = vE.what();
    } catch (const common_json_error & vE) {
      // JSON parse and type errors are invalid requests (400)
      vError   = cErrorTypeInvalidRequest;
      vMessage = vE.what();
    } catch (const std::exception & vE) {
      // treat other exceptions as server error (500)
      vError   = cErrorTypeServer;
      vMessage = vE.what();
    } catch (...) {
      vError   = cErrorTypeServer;
      vMessage = "unknown error";
    }

    auto vRes     = std::make_unique<server_http_res>();
    vRes->vStatus = 500;
    try {
      json vErrorData = fFormatErrorResponse(vMessage, vError);
      vRes->vStatus   = fJsonValue(vErrorData, "code", 500);
      vRes->data      = fSafeJsonToStr({
        { "error", vErrorData }
      });
      SRV_WRN("got exception: %s\n", vRes->data.c_str());
    } catch (const std::exception & vE) {
      SRV_ERR("got another exception: %s | while handling exception: %s\n", vE.what(), vMessage.c_str());
      vRes->data = "Internal Server Error";
    }
    return vRes;
  };
}

int fDgsServer(int pArgc, char ** pArgv) {
  const int cAskpassStatus = fServerShellMcpAskpassMain(pArgc, pArgv);
  if (cAskpassStatus >= 0) {
    return cAskpassStatus;
  }
  server_child vChild;

  std::setlocale(LC_NUMERIC, "C");

#ifndef _WIN32
  // Ignore SIGPIPE so the server does not crash if a child (MCP server, tools runtime) exits while we are writing to its stdin
  signal(SIGPIPE, SIG_IGN);
#endif

  // own arguments required by this example
  common_params vParams;

  fCommonInit();

  // start the stream session manager GC right after common init, before any HTTP route can
  // touch it. lifecycle is symmetric, stop_gc() runs in clean_up() before backend free
  fServerStreamSessionManagerStart();

  if (!fCommonParamsParse(pArgc, pArgv, vParams, cDgsExampleServer)) {
    return 1;
  }

  SRV_INF("%s", "initializing ...\n");

  fDgsBackendInit();
  fDgsNumaInit(vParams.vNuma);

  const int cResult = fDgsServer(vParams, pArgc, pArgv, vChild);
  fCommonLogFlush(fCommonLogMain());
  return cResult;
}

int fDgsServer(common_params & pParams, int pArgc, char ** pArgv) {
  server_child vChild;
  return fDgsServer(pParams, pArgc, pArgv, vChild);
}

int fDgsServer(common_params & pParams, int pArgc, char ** pArgv, server_child & pChild) {
  bool vIsRunByCli = (pArgv == nullptr);

  common_models_handler vModelsHandler;

  // note: router mode also accepts -hf remote-preset, so we need to check that first
  if (!vIsRunByCli && !pParams.vModel.vHfRepo.empty()) {
    try {
      vModelsHandler = fCommonModelsHandlerInit(pParams, cDgsExampleServer);
      if (fCommonModelsHandlerIsPresetRepo(vModelsHandler)) {
        // apply the preset and start the server in router mode
        fCommonModelsHandlerApply(vModelsHandler, pParams);
      }
    } catch (const std::exception & vE) {
      SRV_ERR("failed to fetch model metadata: %s\n", vE.what());
      return 1;
    }
  }

  // router server never loads a model and must not touch the GPU
  const bool cIsRouterServer =
    pParams.vModel.vPath.empty() && pParams.vModel.vHfRepo.empty() && pParams.vModel.vDockerRepo.empty();

  // skip device enumeration so the CUDA primary context stays uncreated
  fCommonParamsPrintInfo(pParams, !cIsRouterServer);

  if (!cIsRouterServer) {
    // validate batch size for embeddings
    // embeddings require all tokens to be processed in a single ubatch
    if (pParams.vEmbedding && pParams.vNBatch > pParams.vNUbatch) {
      SRV_WRN("embeddings enabled with n_batch (%d) > n_ubatch (%d)\n", pParams.vNBatch, pParams.vNUbatch);
      SRV_WRN("setting n_batch = n_ubatch = %d to avoid assertion failure\n", pParams.vNUbatch);
      pParams.vNBatch = pParams.vNUbatch;
    }

    if (pParams.vNParallel < 0) {
      SRV_TRC("%s", "n_parallel is set to auto, using n_parallel = 4 and kv_unified = true\n");

      pParams.vNParallel = 4;
      pParams.vKvUnified = true;
    }
  }

  // size the KV pool from --kv-unified-per-slot, unless the user pinned it with -c
  // or with -c 0 for max context
  const bool cCtxPoolAutoSized =
    pParams.vKvUnifiedPerSlot > 0 && pParams.vNCtx == 0 && (uint32_t) pParams.vFitParamsMinCtx != UINT32_MAX;

  if (cCtxPoolAutoSized) {
    pParams.vNCtx = pParams.vNParallel * pParams.vKvUnifiedPerSlot;
    SRV_INF("--kv-unified-per-slot: sizing KV pool to n_parallel * kv_unified_per_slot = %d * %d = %d\n",
            pParams.vNParallel, pParams.vKvUnifiedPerSlot, pParams.vNCtx);
  }

  // for consistency between server router mode and single-model mode, we set the same model name as alias
  auto vModelName = pParams.vModel.mGetName();
  if (pParams.sModelAlias.empty() && !vModelName.empty()) {
    pParams.sModelAlias.insert(vModelName);
  }

  // note: this is guaranteed to out-live ctx_http and tools
  server_shell_mcp vShellMcp;
  server_mcp       vMcpMgr;

  // struct that contains dgs context and inference
  server_context vCtxServer;

  //
  // Router
  //

  // register API routes
  server_routes vRoutes(pParams, vCtxServer);
  server_tools  vTools;

  std::optional<server_models_routes> vModelsRoutes{};

  server_http_context vCtxHttp;
  if (!vCtxHttp.init(pParams)) {
    SRV_ERR("%s", "failed to initialize HTTP server\n");
    return 1;
  }

  if (cIsRouterServer) {
    // setup server instances manager
    try {
      vModelsRoutes.emplace(pParams, pArgc, pArgv);
    } catch (const std::exception & vE) {
      SRV_ERR("failed to initialize router models: %s\n", vE.what());
      return 1;
    }

    // proxy handlers
    // note: routes.get_health stays the same
    vRoutes.vGetMetrics               = vModelsRoutes->vProxyGet;
    vRoutes.vPostProps                = vModelsRoutes->vProxyPost;
    vRoutes.vPostCompletions          = vModelsRoutes->vProxyPost;
    vRoutes.vPostCompletionsOai       = vModelsRoutes->vProxyPost;
    vRoutes.vPostChatCompletions      = vModelsRoutes->vProxyPost;
    vRoutes.vPostControl              = vModelsRoutes->vProxyPost;
    vRoutes.vPostResponsesOai         = vModelsRoutes->vProxyPost;
    vRoutes.vPostTranscriptionsOai    = vModelsRoutes->vProxyPost;
    vRoutes.vPostAnthropicMessages    = vModelsRoutes->vProxyPost;
    vRoutes.vPostAnthropicCountTokens = vModelsRoutes->vProxyPost;
    vRoutes.vPostInfill               = vModelsRoutes->vProxyPost;
    vRoutes.vPostEmbeddings           = vModelsRoutes->vProxyPost;
    vRoutes.vPostEmbeddingsOai        = vModelsRoutes->vProxyPost;
    vRoutes.vPostRerank               = vModelsRoutes->vProxyPost;
    vRoutes.vPostSystemone            = vModelsRoutes->vProxyPost;
    vRoutes.vPostTokenize             = vModelsRoutes->vProxyPost;
    vRoutes.vPostDetokenize           = vModelsRoutes->vProxyPost;
    vRoutes.vPostApplyTemplate        = vModelsRoutes->vProxyPost;
    vRoutes.vPostChatCompletionsTok   = vModelsRoutes->vProxyPost;
    vRoutes.vPostResponsesTokOai      = vModelsRoutes->vProxyPost;
    vRoutes.vGetLoraAdapters          = vModelsRoutes->vProxyGet;
    vRoutes.vPostLoraAdapters         = vModelsRoutes->vProxyPost;
    vRoutes.vGetSlots                 = vModelsRoutes->vProxyGet;
    vRoutes.vPostSlots                = vModelsRoutes->vProxyPost;

    vRoutes.vGetMonitor       = vModelsRoutes->vGetRouterMonitor;
    vRoutes.vGetMonitorMemory = vModelsRoutes->vGetRouterMonitorMemory;

    // custom routes for router
    vRoutes.vGetProps  = vModelsRoutes->vGetRouterProps;
    vRoutes.vGetModels = vModelsRoutes->vGetRouterModels;

    vCtxHttp.mPost("/models", fExWrapper(vModelsRoutes->vPostRouterModels));
    vCtxHttp.mPost("/models/load", fExWrapper(vModelsRoutes->vPostRouterModelsLoad));
    vCtxHttp.mPost("/models/unload", fExWrapper(vModelsRoutes->vPostRouterModelsUnload));
    vCtxHttp.get("/models/sse", fExWrapper(vModelsRoutes->vGetRouterModelsSse));
    vCtxHttp.mDel("/models", fExWrapper(vModelsRoutes->vDelRouterModels));
  }

  vCtxHttp.get("/health", fExWrapper(vRoutes.vGetHealth));     // public endpoint (no API key check)
  vCtxHttp.get("/v1/health", fExWrapper(vRoutes.vGetHealth));  // public endpoint (no API key check)
  vCtxHttp.get("/ui/settings", fExWrapper(fUiSettingsMakeGetHandler()));
  vCtxHttp.mPost("/ui/settings", fExWrapper(fUiSettingsMakePostHandler()));
  vCtxHttp.get("/monitor", fExWrapper(vRoutes.vGetMonitor));
  vCtxHttp.get("/monitor/memory", fExWrapper(vRoutes.vGetMonitorMemory));
  vCtxHttp.get("/monitor/usage", fExWrapper(vRoutes.vGetMonitorUsage));
  vCtxHttp.get("/metrics", fExWrapper(vRoutes.vGetMetrics));
  vCtxHttp.get("/props", fExWrapper(vRoutes.vGetProps));
  vCtxHttp.mPost("/props", fExWrapper(vRoutes.vPostProps));
  vCtxHttp.get("/models", fExWrapper(vRoutes.vGetModels));
  vCtxHttp.get("/v1/models", fExWrapper(vRoutes.vGetModels));
  vCtxHttp.mPost("/completions", fExWrapper(vRoutes.vPostCompletions));
  vCtxHttp.mPost("/v1/completions", fExWrapper(vRoutes.vPostCompletionsOai));
  vCtxHttp.mPost("/chat/completions", fExWrapper(vRoutes.vPostChatCompletions));
  vCtxHttp.mPost("/v1/chat/completions", fExWrapper(vRoutes.vPostChatCompletions));
  vCtxHttp.mPost("/v1/chat/completions/control", fExWrapper(vRoutes.vPostControl));
  vCtxHttp.mPost("/v1/responses", fExWrapper(vRoutes.vPostResponsesOai));
  vCtxHttp.mPost("/responses", fExWrapper(vRoutes.vPostResponsesOai));
  vCtxHttp.mPost("/v1/audio/transcriptions", fExWrapper(vRoutes.vPostTranscriptionsOai));
  vCtxHttp.mPost("/audio/transcriptions", fExWrapper(vRoutes.vPostTranscriptionsOai));
  vCtxHttp.mPost("/v1/messages", fExWrapper(vRoutes.vPostAnthropicMessages));  // anthropic messages API
  vCtxHttp.mPost("/infill", fExWrapper(vRoutes.vPostInfill));
  vCtxHttp.mPost("/embeddings", fExWrapper(vRoutes.vPostEmbeddings));
  vCtxHttp.mPost("/v1/embeddings", fExWrapper(vRoutes.vPostEmbeddingsOai));
  vCtxHttp.mPost("/rerank", fExWrapper(vRoutes.vPostRerank));
  vCtxHttp.mPost("/reranking", fExWrapper(vRoutes.vPostRerank));
  vCtxHttp.mPost("/v1/rerank", fExWrapper(vRoutes.vPostRerank));
  vCtxHttp.mPost("/v1/reranking", fExWrapper(vRoutes.vPostRerank));
  vCtxHttp.mPost("/v1/systemone", fExWrapper(vRoutes.vPostSystemone));
  vCtxHttp.mPost("/tokenize", fExWrapper(vRoutes.vPostTokenize));
  vCtxHttp.mPost("/detokenize", fExWrapper(vRoutes.vPostDetokenize));
  vCtxHttp.mPost("/apply-template", fExWrapper(vRoutes.vPostApplyTemplate));
  // token counting
  vCtxHttp.mPost("/chat/completions/input_tokens", fExWrapper(vRoutes.vPostChatCompletionsTok));
  vCtxHttp.mPost("/v1/chat/completions/input_tokens", fExWrapper(vRoutes.vPostChatCompletionsTok));
  vCtxHttp.mPost("/responses/input_tokens", fExWrapper(vRoutes.vPostResponsesTokOai));
  vCtxHttp.mPost("/v1/responses/input_tokens", fExWrapper(vRoutes.vPostResponsesTokOai));
  vCtxHttp.mPost("/v1/messages/count_tokens",
                 fExWrapper(vRoutes.vPostAnthropicCountTokens));  // anthropic token counting
  // LoRA adapters hotswap
  vCtxHttp.get("/lora-adapters", fExWrapper(vRoutes.vGetLoraAdapters));
  vCtxHttp.mPost("/lora-adapters", fExWrapper(vRoutes.vPostLoraAdapters));
  // Save & load slots
  vCtxHttp.get("/slots", fExWrapper(vRoutes.vGetSlots));
  vCtxHttp.mPost("/slots/:id_slot", fExWrapper(vRoutes.vPostSlots));

  // resumable streaming: a child binds the local session factories, the router binds
  // proxies that resolve the owning child, see server-stream.h
  server_http_context::handler_t vStreamGetH;
  server_http_context::handler_t vStreamsLookupH;
  server_http_context::handler_t vStreamDeleteH;
  if (cIsRouterServer) {
    vStreamGetH     = vModelsRoutes->vRouterStreamGet;
    vStreamsLookupH = vModelsRoutes->vRouterStreamsLookup;
    vStreamDeleteH  = vModelsRoutes->vRouterStreamDelete;
  } else {
    vStreamGetH     = fServerStreamMakeGetHandler();
    vStreamsLookupH = fServerStreamMakeLookupHandler();
    vStreamDeleteH  = fServerStreamMakeDeleteHandler();
  }
  vCtxHttp.get("/v1/stream", fExWrapper(vStreamGetH));
  vCtxHttp.mPost("/v1/streams/lookup", fExWrapper(vStreamsLookupH));
  vCtxHttp.mDel("/v1/stream", fExWrapper(vStreamDeleteH));

  // Google Cloud Platform (Vertex AI) compat
  vCtxHttp.mRegisterGcpCompat();

  // return 403 for disabled features
  server_http_context::handler_t vRes403 = [](const server_http_req &) {
    auto vRes     = std::make_unique<server_http_res>();
    vRes->vStatus = 403;
    vRes->data    = fSafeJsonToStr({
      { "error",
       {
          { "message", "this feature is disabled" },
          { "type", "feature_disabled" },
        } }
    });
    return vRes;
  };

  if (pParams.vCorsOrigins == "*" && pParams.lApiKeys.empty()) {
    SRV_WRN("%s",
            "security: no API key is set and CORS allows all origins (see "
            "doc/MANUAL.md)\n");
  }

  // CORS proxy (EXPERIMENTAL, only used by the Web UI for MCP)
  std::vector<std::string> lWarnNames;
  if (cIsRouterServer) {
    lWarnNames.push_back("router mode");
  }

  if (pParams.vUiMcpProxy) {
    vCtxHttp.get("/cors-proxy", fExWrapper(vProxyHandlerGet));
    vCtxHttp.mPost("/cors-proxy", fExWrapper(vProxyHandlerPost));
    lWarnNames.push_back("MCP proxy (experimental)");
  } else {
    vCtxHttp.get("/cors-proxy", fExWrapper(vRes403));
    vCtxHttp.mPost("/cors-proxy", fExWrapper(vRes403));
  }

  // Local shell MCP server (EXPERIMENTAL, added to the Web UI by default)
  if (pParams.vShellMcpServer) {
    vShellMcp.mStart();
    vCtxHttp.mPost("/mcp/shell", fExWrapper([&vShellMcp](const server_http_req & pRequest) {
                     return vShellMcp.mHandlePost(pRequest);
                   }));
    const server_http_context::handler_t vUnsupported = [&vShellMcp](const server_http_req & pRequest) {
      return vShellMcp.mHandleUnsupported(pRequest);
    };
    vCtxHttp.get("/mcp/shell", fExWrapper(vUnsupported));
    vCtxHttp.mDel("/mcp/shell", fExWrapper(vUnsupported));
    vCtxHttp.get("/mcp/shell/askpass", fExWrapper([&vShellMcp](const server_http_req & pRequest) {
                   return vShellMcp.mHandleAskpassGet(pRequest);
                 }));
    vCtxHttp.mPost("/mcp/shell/askpass", fExWrapper([&vShellMcp](const server_http_req & pRequest) {
                     return vShellMcp.mHandleAskpassPost(pRequest);
                   }));
    lWarnNames.push_back(
      "local shell MCP server (experimental): anyone who can use the API can run commands as this user");
  } else {
    vCtxHttp.get("/mcp/shell", fExWrapper(vRes403));
    vCtxHttp.mPost("/mcp/shell", fExWrapper(vRes403));
    vCtxHttp.mDel("/mcp/shell", fExWrapper(vRes403));
    vCtxHttp.get("/mcp/shell/askpass", fExWrapper(vRes403));
    vCtxHttp.mPost("/mcp/shell/askpass", fExWrapper(vRes403));
  }

  try {
    vMcpMgr.mStart(pParams);
  } catch (const std::exception & vE) {
    SRV_ERR("MCP starting failed: %s\n", vE.what());
    return 1;
  }

  if (!pParams.lServerTools.empty() || !vMcpMgr.empty()) {
    try {
      vTools.mSetup(pParams.lServerTools, vMcpMgr, pParams.vServerToolsRuntime);
    } catch (const std::exception & vE) {
      SRV_ERR("tools setup failed: %s\n", vE.what());
      return 1;
    }
    vCtxHttp.get("/tools", fExWrapper(vTools.vHandleGet));
    vCtxHttp.mPost("/tools", fExWrapper(vTools.vHandlePost));
    if (!pParams.lServerTools.empty()) {
      lWarnNames.push_back("server tools (experimental)");
    }
    if (!pParams.vServerToolsRuntime.empty()) {
      lWarnNames.push_back("tools runtime (experimental)");
    }
    if (!vMcpMgr.empty()) {
      lWarnNames.push_back("MCP servers (experimental)");
    }
  } else {
    vCtxHttp.get("/tools", fExWrapper(vRes403));
    vCtxHttp.mPost("/tools", fExWrapper(vRes403));
  }

  if (!lWarnNames.empty()) {
    std::string vFeatures;
    for (const auto & name : lWarnNames) {
      if (!vFeatures.empty()) {
        vFeatures += ", ";
      }
      vFeatures += name;
    }
    SRV_WRN("security: %s enabled - do not expose to untrusted environments\n", vFeatures.c_str());
  }

  //
  // Handle downloading model
  //

  if (pChild.mIsChild() && pChild.mGetMode() == cServerChildModeDownload) {
    return pChild.mRunDownload(pParams);
  } else if (!cIsRouterServer && !vIsRunByCli) {
    // single-model mode (NOT spawned by router)
    // if this is invoked by CLI, model downloading should be already handled
    try {
      fCommonModelsHandlerApply(vModelsHandler, pParams);
    } catch (const std::exception & vE) {
      SRV_ERR("failed to download model: %s\n", vE.what());
      return 1;
    }
  }

  //
  // Start the server
  //

  std::function<void()> vCleanUp;

  if (cIsRouterServer) {
    SRV_INF("%s", "starting server in router mode. models will be automatically loaded on-demand\n");

    vCleanUp = [&vModelsRoutes, &vMcpMgr]() {
      SRV_INF("%s: cleaning up before exit...\n", __func__);
      // stop the session GC first, it finalizes live sessions and wakes pending readers
      fServerStreamSessionManagerStop();
      if (vModelsRoutes.has_value()) {
        vModelsRoutes->vStopping.store(true);  // maybe redundant, but just to be safe
        vModelsRoutes->vModels.mUnloadAll();
      }
      vMcpMgr.mShutdown();
      fDgsBackendFree();
    };

    if (!vCtxHttp.mStart()) {
      vCleanUp();
      SRV_ERR("%s", "exiting due to HTTP server error\n");
      return 1;
    }
    vCtxHttp.vIsReady.store(true);

    vShutdownHandler = [&](int) {
      if (vModelsRoutes.has_value()) {
        // important to disconnect any SSE clients
        vModelsRoutes->vStopping.store(true);
      }
      vMcpMgr.mShutdown();
      vCtxHttp.mStop();
    };

    try {
      vModelsRoutes->vModels.mLoadStartupModels();
    } catch (const std::exception & vE) {
      SRV_ERR("failed to load models on startup: %s\n", vE.what());
      vCtxHttp.mStop();
      vCtxHttp.mJoin();
      vCleanUp();
      return 1;
    }

  } else {
    // setup clean up function, to be called before exit
    vCleanUp = [&vCtxHttp, &vCtxServer, &vMcpMgr]() {
      SRV_INF("%s: cleaning up before exit...\n", __func__);
      // stop the session GC first, it finalizes live sessions and wakes pending readers
      fServerStreamSessionManagerStop();
      vCtxHttp.mStop();
      vCtxServer.mTerminate();
      vMcpMgr.mShutdown();
      fDgsBackendFree();
    };

    // start the HTTP server before loading the model to be able to serve /health requests
    if (!vCtxHttp.mStart()) {
      vCleanUp();
      SRV_ERR("%s", "exiting due to HTTP server error\n");
      return 1;
    }

    // setup communication child --> router if necessary
    if (pChild.mIsChild()) {
      vCtxServer.mSetStateCallback(
        [&](server_state pState, json pPayload) { pChild.mNotifyToRouter(fServerStateToStr(pState), pPayload); });
    }

    if (!vCtxServer.mLoadModel(pParams)) {
      vCleanUp();
      vCtxHttp.mJoin();
      SRV_ERR("%s", "exiting due to model loading error\n");
      return 1;
    }

    vRoutes.mUpdateMeta(vCtxServer);
    vCtxHttp.vIsReady.store(true);

    SRV_INF("%s", "model loaded\n");

    vShutdownHandler = [&](int) {
      vMcpMgr.mShutdown();
      // this will unblock start_loop()
      vCtxServer.mTerminate();
    };
  }

  // register signal handler if not running by CLI
  if (!vIsRunByCli) {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    struct sigaction vSigintAction;
    vSigintAction.sa_handler = fSignalHandler;
    sigemptyset(&vSigintAction.sa_mask);
    vSigintAction.sa_flags = 0;
    sigaction(SIGINT, &vSigintAction, NULL);
    sigaction(SIGTERM, &vSigintAction, NULL);
#elif defined(_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
      return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
  }

  for (const auto & vAddress : vCtxHttp.lListeningAddresses) {
    SRV_INF("listening on %s\n", vAddress.c_str());
  }

  if (cIsRouterServer) {
    if (!pParams.vModelsPresetHf.empty()) {
      SRV_WRN("NOTE: using preset.ini from HF repo '%s'\n", pParams.vModelsPresetHf.c_str());
      SRV_WRN("%s", "      please only use presets that you can trust! Unknown presets may be unsafe\n");
    }

    vCtxHttp.mJoin();  // keep the main thread alive

    // when the HTTP server stops, clean up and exit
    vCleanUp();
  } else {
    // optionally, notify router server that this instance is ready
    std::thread vMonitorThread;
    if (pChild.mIsChild()) {
      vMonitorThread = pChild.mSetup(vShutdownHandler);
      pChild.mNotifyToRouter(fServerStateToStr(cServerStateReady), vRoutes.mGetModelInfo());
    }

    // this call blocks the main thread until queue_tasks.terminate() is called
    vCtxServer.mStartLoop();

    vCleanUp();
    vCtxHttp.mJoin();
    if (vMonitorThread.joinable()) {
      vMonitorThread.join();
    }

    auto * vLlCtx = vCtxServer.mGetDgsContext();
    if (vLlCtx != nullptr) {
      fCommonMemoryBreakdownPrint(vLlCtx);
    }
  }

  return 0;
}
