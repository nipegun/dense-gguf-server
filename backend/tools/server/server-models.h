#pragma once

#include "common.h"
#include "download.h"
#include "preset.h"
#include "server-common.h"
#include "server-http.h"
#include "server-queue.h"

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>

/**
 * state diagram:
 *
 * DOWNLOADING ──► DOWNLOADED ──► (replaced by new instance)
 *
 * UNLOADED ──► LOADING ──► LOADED ◄──── SLEEPING
 *  ▲            │            │               ▲
 *  └───failed───┘            │               │
 *  ▲                         └──sleeping─────┘
 *  └────────unloaded─────────┘
 */
enum server_model_status {
  // TODO: also add downloading state when the logic is added
  cServerModelStatusDownloading,
  cServerModelStatusDownloaded,
  cServerModelStatusUnloaded,
  cServerModelStatusLoading,
  cServerModelStatusLoaded,
  cServerModelStatusSleeping
};

enum server_model_source {
  cServerModelSourcePreset,
  cServerModelSourceModelsDir,
  cServerModelSourceCache,
};

enum server_child_mode {
  cServerChildModeNormal,    // load the model and run normally
  cServerChildModeDownload,  // download the model and exit
};

static std::string fServerModelStatusToString(server_model_status pStatus) {
  switch (pStatus) {
    case cServerModelStatusDownloading:
      return "downloading";
    case cServerModelStatusDownloaded:
      return "downloaded";
    case cServerModelStatusUnloaded:
      return "unloaded";
    case cServerModelStatusLoading:
      return "loading";
    case cServerModelStatusLoaded:
      return "loaded";
    case cServerModelStatusSleeping:
      return "sleeping";
    default:
      return "unknown";
  }
}

static std::string fServerModelSourceToString(server_model_source pSource) {
  switch (pSource) {
    case cServerModelSourcePreset:
      return "preset";
    case cServerModelSourceModelsDir:
      return "models_dir";
    case cServerModelSourceCache:
      return "cache";
    default:
      return "unknown";
  }
}

struct server_model_meta {
  server_model_source      vSource = cServerModelSourceCache;
  common_preset            vPreset;
  std::string              name;
  std::set<std::string>    sAliases;  // additional names that resolve to this model
  std::set<std::string>    sTags;     // informational tags, not used for routing
  int                      vPort     = 0;
  server_model_status      vStatus   = cServerModelStatusUnloaded;
  int64_t                  vLastUsed = 0;  // for LRU unloading
  std::vector<std::string> lArgs;          // args passed to the model instance, will be populated by render_args()
  json
    vLoadedInfo;  // info to be reflected via /v1/models endpoint ; if in DOWNLOADING state, it should contain download progress info
  json     vProgress;              // reflect load or download progress info, if any
  int      vExitCode     = 0;      // exit code of the model instance process (only valid if status == FAILED)
  int      vStopTimeout  = 0;      // seconds to wait before force-killing the model instance during shutdown
  bool     vHidden       = false;  // hidden from GET /models, but still accept if requested
  json     vArchitecture = fServerModelArchitectureJson(false, false, false, { "text" });
  uint32_t vNCtxTrain    = 0;      // trained context, read from the GGUF metadata; 0 when unknown

  bool mIsReady() const { return vStatus == cServerModelStatusLoaded; }

  bool mIsRunning() const {
    return vStatus == cServerModelStatusLoaded || vStatus == cServerModelStatusLoading ||
           vStatus == cServerModelStatusSleeping;
  }

  bool mIsReadyOrSleep() const { return vStatus == cServerModelStatusLoaded || vStatus == cServerModelStatusSleeping; }

  bool mIsFailed() const { return vStatus == cServerModelStatusUnloaded && vExitCode != 0; }

  void mUpdateArgs(common_preset_context & pCtxPresets, std::string pBinPath);
  void mUpdateCaps(const common_params & pBase);
};

struct server_models_routes;
struct server_lru_sched;  // defined in server-models.cpp
struct server_monitor;    // defined in server-models.cpp

struct server_models {
  friend struct server_models_routes;
  friend struct server_lru_sched;
  friend struct server_monitor;

private:
  struct instance_t {
    std::shared_ptr<server_subproc> vSubproc;       // shared with the monitor thread
    server_model_meta               vMeta;
    int                             vReqCount = 0;  // number of active proxy requests

    // ask the child to exit (it handles the command on its stdin, see server_child::setup)
    void mRequestExit() const;
  };

  std::mutex                        vMutex;
  std::condition_variable           vCv;
  std::map<std::string, instance_t> dMapping;

  // models asked to stop, still counted as running until the monitor records their exit
  std::set<std::string> sStoppingModels;

  // set to true while load_models() is executing a reload; load() will wait until clear
  bool vIsReloading = false;

  // if true, the next get_meta() will trigger a reload of model list
  bool vNeedReload = false;

  // models marked with load-on-startup, unset once load_startup_models() drains it
  // no value means the startup phase is over, so a reload must not queue anything
  std::optional<std::vector<std::string>> lStartupModels{ std::in_place };

  // conv_id -> model name that currently serves its stream session, lets the resumable stream
  // routes go straight to the owning child instead of polling every one. populated when
  // proxy_request forwards a POST carrying an X-Conversation-Id. best effort: a stale entry just
  // makes the child answer not found and the client recovers. owns its lock, one mutex per struct
  struct conv_model_tracker {
    // returns the ticket of this registration, 0 when nothing was registered. erasing or
    // replacing the entry invalidates the ticket, which is how a stop cancels a request
    // parked in the model load wait
    uint64_t mRemember(const std::string & pConvId, const std::string & pModel) {
      if (pConvId.empty() || pModel.empty()) {
        return 0;
      }
      std::lock_guard<std::mutex> vLock(vMu);
      uint64_t                    vTicket = vNextTicket++;
      dMap[pConvId]                       = { pModel, vTicket };
      return vTicket;
    }

    // false means a stop erased the entry or a newer request replaced it
    bool mAlive(const std::string & pConvId, uint64_t pTicket) {
      std::lock_guard<std::mutex> vLock(vMu);
      auto                        vIt = dMap.find(pConvId);
      return vIt != dMap.end() && vIt->second.vTicket == pTicket;
    }

    std::optional<std::string> mLookup(const std::string & pConvId) {
      if (pConvId.empty()) {
        return std::nullopt;
      }
      std::lock_guard<std::mutex> vLock(vMu);
      auto                        vIt = dMap.find(pConvId);
      if (vIt == dMap.end()) {
        return std::nullopt;
      }
      return vIt->second.vModel;
    }

    void mForget(const std::string & pConvId) {
      if (pConvId.empty()) {
        return;
      }
      std::lock_guard<std::mutex> vLock(vMu);
      dMap.erase(pConvId);
    }

  private:
    struct entry_t {
      std::string vModel;
      uint64_t    vTicket;
    };

    std::mutex                               vMu;
    uint64_t                                 vNextTicket = 1;
    std::unordered_map<std::string, entry_t> dMap;
  };

  common_preset_context vCtxPreset;

  common_params            vBaseParams;
  std::string              vBinPath;
  std::vector<std::string> lBaseEnv;
  common_preset            vBasePreset;  // base preset from dgs-server CLI args

  // queue of requests waiting for a models_max slot
  std::unique_ptr<server_lru_sched> vSched;

  // if true, add some delay to simulate works (useful for testing)
  bool vDebugFakeTiming = false;

  void mUpdateMeta(const std::string & name, const server_model_meta & pMeta);

  // unload least recently used models if the limit is reached
  void mUnloadLru();

  // not thread-safe, caller must hold mutex
  void mAddModel(server_model_meta && pMeta);

  // ask the monitor to stop a running instance; send_exit is false for a child that was already force-killed
  // not thread-safe, caller must hold mutex
  void mRequestStop(const std::string & name, bool pSendExit = true);

  // called by the monitor once a child exited and was reaped
  void mOnChildExit(const std::string &                     name,
                    const std::shared_ptr<server_subproc> & pProc,
                    server_child_mode                       pMode,
                    int                                     pExitCode);

  // notify SSE clients
  void mNotifySse(const std::string & pEvent, const std::string & pModelId, const json & data = nullptr);

public:
  // conv_id -> model tracker for the resumable stream routes, owns its lock
  conv_model_tracker vConvModels;

  server_models(const common_params & pParams, int pArgc, char ** pArgv);
  ~server_models();

  server_response vSse;  // for real-time updates via SSE endpoint

  // (re-)load the list of models from various sources and prepare the metadata mapping
  // - if this is called the first time, simply populate the metadata
  // - if this is called subsequently (e.g. when refreshing from disk):
  //   - if a model is running but updated or removed from the source, it will be unloaded
  //   - if a model is not running, it will be added or updated according to the source
  void mLoadModels();

  // lazy-load startup_models, to be called after main() setup phase
  void mLoadStartupModels();

  // check if a model instance exists (thread-safe)
  bool mHasModel(const std::string & name);

  // return a copy of model metadata (thread-safe)
  std::optional<server_model_meta> mGetMeta(const std::string & name);

  // return a copy of all model metadata (thread-safe)
  std::vector<server_model_meta> mGetAllMeta();

  struct load_options {
    server_child_mode                vMode       = cServerChildModeNormal;
    // used for spawning a downloading child process
    std::optional<server_model_meta> vCustomMeta = std::nullopt;
  };

  // load and unload model instances
  // these functions are thread-safe
  void mLoad(const std::string & name);
  void mLoad(const std::string & name, const load_options & pOpts);
  void mUnload(const std::string & name);
  void mUnloadAll();

  struct update_status_args {
    server_model_status vStatus;
    int                 vExitCode   = 0;  // only valid if status == UNLOADED
    json                vLoadedInfo = nullptr;
    json                vProgress   = nullptr;
  };

  // update the status of a model instance (thread-safe)
  // also send SSE notification to /models/sse endpoint
  void mUpdateStatus(const std::string & name, const update_status_args & pArgs);
  void mUpdateDownloadProgress(const std::string &              name,
                               const common_download_progress & pProgress,
                               bool                             pDone,
                               bool                             pOk = true);

  // remove a cache model from disk and update the list (thread-safe)
  // note: only cache models can be removed; returns false if the model doesn't exist or is not a cache model
  bool mRemove(const std::string & name);

  // wait until the model instance is fully loaded (thread-safe)
  // note: predicate is called while holding the lock
  // return when the model no longer in "loading" state
  void mWait(const std::string & name, std::function<bool(const server_model_meta &)> pPredicate);
  void mWait(std::unique_lock<std::mutex> &                 pLk,
             const std::string &                            name,
             std::function<bool(const server_model_meta &)> pPredicate);

  // ensure the model is in ready state (thread-safe)
  // return false if model is ready
  // otherwise, load the model and blocking wait until it's ready, then return true (meta may need to be refreshed)
  // if models_max is reached, the request waits in a queue until a slot frees up
  // throws if the load fails, or if should_stop fires while waiting
  bool mEnsureModelReady(const std::string & name, const std::function<bool()> & pShouldStop = nullptr);

  // proxy an HTTP request to the model instance
  server_http_res_ptr mProxyRequest(const server_http_req & pReq,
                                    const std::string &     pMethod,
                                    const std::string &     name,
                                    bool                    pUpdateLastUsed,
                                    bool                    pDetached = false);

  // handle message sent from server_child::notify_to_router()
  // raw input must starts with CMD_CHILD_TO_ROUTER_STATE, followed by a JSON string
  // called from the monitor thread
  // payload per state:
  //     state = loading     -> payload = {} (TODO: add progress info)
  //     state = ready       -> payload = model_info (json), or {} if wakeup from sleeping
  //     state = sleeping    -> payload = {}
  void mHandleChildState(const std::string & name, const std::string & pRawInput);

private:
  // one thread watching every child; keep last, the destructor joins the thread
  std::unique_ptr<server_monitor> vMonitor;
};

struct server_child {
  // serializes the notify_to_router writes
  std::mutex        vMtxStdout;
  std::atomic<bool> vIsFinishedDownloading = false;  // set by run_download

  // in a child, keeps stdout for the commands to the router, so it is created before anything is written;
  // everything else written to stdout goes to stderr with the logs
  server_child();
  ~server_child();

  // return true if the current process is a child server instance
  bool              mIsChild();
  server_child_mode mGetMode();
  int               mRunDownload(common_params & pParams);

  // register the shutdown_handler to be called by the router
  // return the monitoring thread (to be joined by the caller)
  std::thread mSetup(const std::function<void(int)> & pShutdownHandler);

  // notify router server for status changes (e.g. loading, downloading, sleeping, etc.)
  // message will be handled by server_models::handle_child_state() on the router side
  void mNotifyToRouter(const std::string & pStateName, const json & pPayload);

private:
  FILE * vCmdOut = nullptr;  // the stdout the router reads the commands from
};

struct server_models_routes {
  common_params     vParams;
  json              vUiSettings = json::mObject();  // Primary: new name
  std::atomic<bool> vStopping   = false;            // for graceful disconnecting SSE clients during shutdown
  server_models     vModels;

  server_models_routes(const common_params & pParams, int pArgc, char ** pArgv) :
    vParams(pParams),
    vModels(pParams, pArgc, pArgv) {
    const std::string & vCfg = this->vParams.vUiConfigJson;
    if (!vCfg.empty()) {
      try {
        json vJsonSettings = json::mParse(vCfg);
        vUiSettings        = vJsonSettings;
      } catch (const std::exception & vE) {
        LOG_ERR("%s: failed to parse UI config: %s\n", __func__, vE.what());
        throw;
      }
    }
    mInitRoutes();
  }

  void                           mInitRoutes();
  // handlers using lambda function, so that they can capture `this` without `std::bind`
  server_http_context::handler_t vGetRouterMonitor;
  server_http_context::handler_t vGetRouterMonitorMemory;
  server_http_context::handler_t vGetRouterProps;
  server_http_context::handler_t vProxyGet;
  server_http_context::handler_t vProxyPost;
  server_http_context::handler_t vGetRouterModels;
  server_http_context::handler_t vPostRouterModelsLoad;
  server_http_context::handler_t vPostRouterModelsUnload;
  // management API
  server_http_context::handler_t vGetRouterModelsSse;
  server_http_context::handler_t vPostRouterModels;
  server_http_context::handler_t vDelRouterModels;

  // router side handlers for the resumable streaming routes. each resolves the child that owns
  // a conversation through the conv_id -> model map, no probing or fan out
  server_http_context::handler_t vRouterStreamGet;
  server_http_context::handler_t vRouterStreamsLookup;
  server_http_context::handler_t vRouterStreamDelete;
};

/**
 * A simple HTTP proxy that forwards requests to another server
 * and relays the responses back.
 */
struct server_http_proxy : server_http_res {
  std::function<void()> vCleanup = nullptr;
  server_http_proxy(const std::string &                          pMethod,
                    const std::string &                          pScheme,
                    const std::string &                          pHost,
                    int                                          pPort,
                    const std::string &                          pPath,
                    const std::map<std::string, std::string> &   pHeaders,
                    const std::string &                          pBody,
                    const std::map<std::string, uploaded_file> & pFiles,
                    const std::function<bool()>                  pShouldStop,
                    int32_t                                      pTimeoutRead,
                    int32_t                                      pTimeoutWrite);

  ~server_http_proxy() {
    if (vCleanupPipes) {
      vCleanupPipes();
    }
    if (vCleanup) {
      vCleanup();
    }
  }
private:
  std::function<void()> vCleanupPipes = nullptr;
  std::thread           vThread;

  struct msg_t {
    std::map<std::string, std::string> dHeaders;
    int                                vStatus = 0;
    std::string                        data;
    std::string                        vContentType;
  };
};
