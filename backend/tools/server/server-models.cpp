#include "server-models.h"

#include "build-info.h"
#include "download.h"
#include "hf-cache.h"
#include "http.h"
#include "preset.h"
#include "server-common.h"
#include "server-context.h"
#include "server-monitor.h"
#include "server-stream.h"
#include "subproc.h"

#include <cpp-httplib/httplib.h>  // TODO: remove this once we use HTTP client from download.h

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <queue>
#include <random>
#include <sstream>
#include <thread>

#ifndef _WIN32
extern char ** environ;
#endif

#if defined(__APPLE__) && defined(__MACH__)
// macOS: use _NSGetExecutablePath to get the executable path
#  include <limits.h>
#  include <mach-o/dyld.h>
#endif

#define DEFAULT_STOP_TIMEOUT 10  // seconds

#define CMD_ROUTER_TO_CHILD_EXIT  "cmd_router_to_child:exit"
#define CMD_CHILD_TO_ROUTER_STATE "cmd_child_to_router:state:"  // followed by json string

// note: SIGPIPE is ignored by the server
static void fRequestChildExit(server_subproc & pProc) {
  FILE * vStdinFile = pProc.vSproc.mStdinFile();
  if (vStdinFile) {
    fprintf(vStdinFile, "%s\n", CMD_ROUTER_TO_CHILD_EXIT);
    fflush(vStdinFile);
  }
}

// address for child process, this is needed because router may run on 0.0.0.0
#define CHILD_ADDR "127.0.0.1"

// single-threaded, watching all child processes at once
struct server_monitor {
  server_monitor(server_models & pModels) : vModels(pModels) {
    vTh = std::thread([this]() { mRun(); });
  }

  ~server_monitor() {
    mPush({ cmd_t::cQuit, {}, "", 0, false });
    vTh.join();
  }

  // thread-safe
  void mWatch(const std::string & name, std::shared_ptr<server_subproc> pProc, server_child_mode pMode, int pPort) {
    child_t vC;
    vC.name  = name;
    vC.vProc = std::move(pProc);
    vC.vMode = pMode;
    vC.vPort = pPort;
    if (!vC.vProc->mHasOutput()) {
      SRV_ERR("failed to get stdout/stderr of child process for name=%s\n", name.c_str());
      vC.vEof = true;
    }
    mPush({ cmd_t::cWatch, std::move(vC), "", 0, false });
  }

  // thread-safe
  void mStop(const std::string & name, int pStopTimeout, bool pSendExit) {
    mPush({ cmd_t::cStop, {}, name, pStopTimeout, pSendExit });
  }

private:
  struct child_t {
    std::string                     name;
    std::shared_ptr<server_subproc> vProc;
    server_child_mode               vMode = cServerChildModeNormal;
    int                             vPort = 0;
    std::string                     aBuf[cServerSubprocStreams];  // partial line of each pipe
    bool                            vEof      = false;            // output closed, waiting for the process to be reaped
    int64_t                         vDeadline = 0;                // force-kill time in ms, 0 when no stop is pending
  };

  struct cmd_t {
    enum { cWatch, cStop, cQuit } type;

    child_t     vChild;
    std::string name;
    int         vStopTimeout;
    bool        vSendExit;
  };

  void mPush(cmd_t && pCmd) {
    {
      std::lock_guard<std::mutex> vLk(vMu);
      lCmds.push_back(std::move(pCmd));
    }
    vWaiter.mWake();
  }

  // returns true if the loop should exit
  bool mHandleCommands() {
    std::deque<cmd_t> lBatch;
    {
      std::lock_guard<std::mutex> vLk(vMu);
      lBatch.swap(lCmds);
    }
    for (auto & vCmd : lBatch) {
      switch (vCmd.type) {
        case cmd_t::cWatch:
          lChildren.push_back(std::move(vCmd.vChild));
          break;
        case cmd_t::cStop:
          // the newest child with this name is the one the registry knows
          for (auto lIt = lChildren.rbegin(); lIt != lChildren.rend(); ++lIt) {
            if (lIt->name != vCmd.name) {
              continue;
            }
            if (vCmd.vSendExit && !lIt->vEof) {
              fRequestChildExit(*lIt->vProc);
            }
            lIt->vDeadline = ggml_time_ms() + (int64_t) vCmd.vStopTimeout * 1000;
            break;
          }
          break;
        case cmd_t::cQuit:
          return true;
      }
    }
    return false;
  }

  // read what the child wrote, handle its commands and forward its logs, line by line
  void mReadOutput(child_t & pC) {
    for (int vI = 0; vI < cServerSubprocStreams; vI++) {
      mReadStream(pC, (server_subproc_stream) vI);
    }
    pC.vEof = pC.vProc->mOutputClosed();
  }

  void mReadStream(child_t & pC, server_subproc_stream stream) {
    char          aChunk[4096];
    std::string & vBuf    = pC.aBuf[stream];
    bool          vClosed = false;
    while (true) {
      int n = pC.vProc->mReadOutput(stream, aChunk, sizeof(aChunk));
      if (n < 0) {
        vClosed = true;
        break;
      }
      if (n == 0) {
        break;
      }
      vBuf.append(aChunk, (size_t) n);
      size_t vStart = 0;
      while (true) {
        size_t vNl = vBuf.find('\n', vStart);
        if (vNl == std::string::npos) {
          break;
        }
        mOnLine(pC, stream, vBuf.substr(vStart, vNl + 1 - vStart));
        vStart = vNl + 1;
      }
      vBuf.erase(0, vStart);
      if (vBuf.size() > cMaxLine) {
        vBuf.clear();  // a child that never writes a newline must not grow this without bound
      }
    }
    if (vClosed && !vBuf.empty()) {
      mOnLine(pC, stream, vBuf);
      vBuf.clear();
    }
  }

  void mOnLine(child_t & pC, server_subproc_stream stream, const std::string & pLine) {
    if (stream == cServerSubprocStderr) {
      LOG("[%5d] %s", pC.vPort, pLine.c_str());      // forward log
    } else if (fStringStartsWith(pLine, CMD_CHILD_TO_ROUTER_STATE)) {
      LOG_DBG("[%5d] %s", pC.vPort, pLine.c_str());  // prevent spamming the log
      vModels.mHandleChildState(pC.name, pLine);
    } else {
      SRV_WRN("[%5d] unexpected output on the command pipe: %s", pC.vPort, pLine.c_str());
    }
  }

  void mRun() {
    while (true) {
      if (mHandleCommands()) {
        return;
      }

      // wait for output, a wakeup, or the next deadline;
      // a child whose output closed is polled for its exit every 50 ms
      int64_t vNow     = ggml_time_ms();
      int64_t vTimeout = -1;
      for (const auto & vC : lChildren) {
        if (vC.vEof) {
          vTimeout = vTimeout < 0 ? 50 : std::min<int64_t>(vTimeout, 50);
        }
        if (vC.vDeadline) {
          int64_t vD = std::max<int64_t>(0, vC.vDeadline - vNow);
          vTimeout   = vTimeout < 0 ? vD : std::min(vTimeout, vD);
        }
      }
      std::vector<server_subproc *> lProcs;
      std::vector<child_t *>        lOwners;
      for (auto & vC : lChildren) {
        if (!vC.vEof) {
          lProcs.push_back(vC.vProc.get());
          lOwners.push_back(&vC);
        }
      }
      std::vector<bool> lReady;
      vWaiter.mWait(lProcs, lReady, vTimeout);
      for (size_t vI = 0; vI < lOwners.size(); vI++) {
        if (lReady[vI]) {
          mReadOutput(*lOwners[vI]);
        }
      }

      // deadlines and exits
      vNow = ggml_time_ms();
      for (auto lIt = lChildren.begin(); lIt != lChildren.end();) {
        if (lIt->vDeadline && vNow >= lIt->vDeadline && !lIt->vProc->vStopped.load(std::memory_order_acquire)) {
          SRV_WRN("force-killing model instance name=%s after timeout\n", lIt->name.c_str());
          lIt->vProc->mTerminate();
          lIt->vDeadline = 0;
        }
        if (lIt->vEof && !lIt->vProc->mIsAlive()) {
          int vExitCode = lIt->vProc->mJoin();
          lIt->vProc->vStopped.store(true, std::memory_order_release);
          vModels.mOnChildExit(lIt->name, lIt->vProc, lIt->vMode, vExitCode);
          SRV_INF("instance name=%s exited with status %d\n", lIt->name.c_str(), vExitCode);
          lIt = lChildren.erase(lIt);
        } else {
          ++lIt;
        }
      }
    }
  }

  static constexpr size_t cMaxLine = 1024 * 1024;

  server_models &        vModels;
  std::mutex             vMu;
  std::deque<cmd_t>      lCmds;
  std::vector<child_t>   lChildren;  // monitor thread only
  server_subproc::waiter vWaiter;
  std::thread            vTh;
};

struct server_lru_sched {
  server_lru_sched(server_models & pModels) : vModels(pModels) {}

  bool mHasCapacity(std::unique_lock<std::mutex> & pLk) {
    mCheckLock(pLk);
    return vModels.vBaseParams.vModelsMax <= 0 || mCountRunning() < (size_t) vModels.vBaseParams.vModelsMax;
  }

  // returns "" if no model can be given up
  std::string mPickVictim(std::unique_lock<std::mutex> & pLk) {
    mCheckLock(pLk);
    std::string vVictim;
    int64_t     vVictimLastUsed = 0;
    for (const auto & vM : vModels.dMapping) {
      // a busy model is mid-request, one still coming up has no request to finish
      if (vM.second.vReqCount != 0 || !vM.second.vMeta.mIsReadyOrSleep()) {
        continue;
      }
      // already on its way out, or a queued request wants it
      if (vModels.sStoppingModels.count(vM.first) || find(vM.first)) {
        continue;
      }
      if (vVictim.empty() || vM.second.vMeta.vLastUsed < vVictimLastUsed) {
        vVictim         = vM.first;
        vVictimLastUsed = vM.second.vMeta.vLastUsed;
      }
    }
    return vVictim;
  }

  // requests wanting the same model share one entry, so they all need only one slot
  // and all get unblocked by the single load that entry performs
  void mJoin(std::unique_lock<std::mutex> & pLk, const std::string & pModelId) {
    mCheckLock(pLk);
    if (entry_t * vE = find(pModelId)) {
      vE->vNWaiters++;
      SRV_INF("request for name=%s joined the queue, %d waiting\n", pModelId.c_str(), vE->vNWaiters);
      return;
    }
    lQueue.push_back({ pModelId, 1, false });
    SRV_INF("request for name=%s queued at position %zu\n", pModelId.c_str(), lQueue.size());
  }

  void mLeave(std::unique_lock<std::mutex> & pLk, const std::string & pModelId) {
    mCheckLock(pLk);
    for (auto vIt = lQueue.begin(); vIt != lQueue.end(); ++vIt) {
      if (vIt->vModelId == pModelId) {
        if (--vIt->vNWaiters <= 0) {
          lQueue.erase(vIt);  // last one waiting for this model went away
        }
        return;
      }
    }
  }

  bool mQueueEmpty(std::unique_lock<std::mutex> & pLk) {
    mCheckLock(pLk);
    return lQueue.empty();
  }

  // true if it is this model's turn to load, and nobody is loading it yet
  bool mTryClaim(std::unique_lock<std::mutex> & pLk, const std::string & pModelId) {
    mCheckLock(pLk);
    if (lQueue.empty() || lQueue.front().vModelId != pModelId || lQueue.front().vLoading) {
      return false;
    }
    if (!mHasCapacity(pLk)) {
      return false;
    }
    lQueue.front().vLoading = true;
    return true;
  }

  // on failure the entry is back in line; on success it stays until its waiters leave,
  // so the model coming up is never picked as a victim before they use it
  void mClaimDone(std::unique_lock<std::mutex> & pLk, const std::string & pModelId, bool pOk) {
    mCheckLock(pLk);
    if (pOk) {
      return;
    }
    for (auto vIt = lQueue.begin(); vIt != lQueue.end(); ++vIt) {
      if (vIt->vModelId == pModelId) {
        vIt->vLoading = false;
        return;
      }
    }
  }

  // evict idle models while queued requests outnumber the slots that are free or being freed
  // caller must hold models.mutex; never blocks, so it is safe from any thread
  void mTick(std::unique_lock<std::mutex> & pLk) {
    mCheckLock(pLk);
    if (vModels.vBaseParams.vModelsMax <= 0 || lQueue.empty()) {
      return;
    }
    int vNRunning  = 0;
    int vNStopping = 0;
    for (const auto & vM : vModels.dMapping) {
      if (vM.second.vMeta.mIsRunning()) {
        vNRunning++;
        if (vModels.sStoppingModels.count(vM.first)) {
          vNStopping++;
        }
      }
    }
    int vNNeeded  = 0;
    int vNClaimed = 0;  // claimed the slot, but load() has not spawned yet
    for (const auto & vE : lQueue) {
      if (!vE.vLoading) {
        vNNeeded++;
        continue;
      }
      auto vIt = vModels.dMapping.find(vE.vModelId);
      if (vIt != vModels.dMapping.end() && !vIt->second.vMeta.mIsRunning()) {
        vNClaimed++;
      }
    }
    int vNFree = vModels.vBaseParams.vModelsMax - vNRunning + vNStopping - vNClaimed;
    while (vNFree < vNNeeded) {
      std::string vVictim = mPickVictim(pLk);
      if (vVictim.empty()) {
        return;  // all remaining models are busy, wait for a request to end
      }
      SRV_INF("evicting idle LRU name=%s for a queued request\n", vVictim.c_str());
      vModels.mRequestStop(vVictim);
      vNFree++;
    }
  }

private:
  struct entry_t {
    std::string vModelId;
    int         vNWaiters;  // requests waiting for this model
    bool        vLoading;   // one of the waiters is doing the load right now
  };

  entry_t * find(const std::string & pModelId) {
    for (auto & vE : lQueue) {
      if (vE.vModelId == pModelId) {
        return &vE;
      }
    }
    return nullptr;
  }

  void mCheckLock(std::unique_lock<std::mutex> & pLk) {
    GGML_ASSERT(pLk.owns_lock() && pLk.mutex() == &vModels.vMutex);
  }

  size_t mCountRunning() {
    size_t count = 0;
    for (const auto & vM : vModels.dMapping) {
      if (vM.second.vMeta.mIsRunning()) {
        count++;
      }
    }
    return count;
  }

  server_models &     vModels;
  std::deque<entry_t> lQueue;
};

// short loopback budget for the resumable stream router to child JSON calls (probe, lookup,
// delete). distinct from params.timeout_read/write which only applies to the generation proxy
static constexpr int cStreamLookupTimeoutMs = 250;

static std::filesystem::path fGetServerExecPath() {
#if defined(_WIN32)
  wchar_t buf[32768] = { 0 };  // Large buffer to handle long paths
  DWORD   len        = GetModuleFileNameW(nullptr, buf, _countof(buf));
  if (len == 0 || len >= _countof(buf)) {
    throw std::runtime_error("GetModuleFileNameW failed or path too long");
  }
  return std::filesystem::path(buf);
#elif defined(__APPLE__) && defined(__MACH__)
  char     small_path[PATH_MAX];
  uint32_t size = sizeof(small_path);

  if (_NSGetExecutablePath(small_path, &size) == 0) {
    // resolve any symlinks to get absolute path
    try {
      return std::filesystem::canonical(std::filesystem::path(small_path));
    } catch (...) {
      return std::filesystem::path(small_path);
    }
  } else {
    // buffer was too small, allocate required size and call again
    std::vector<char> buf(size);
    if (_NSGetExecutablePath(buf.data(), &size) == 0) {
      try {
        return std::filesystem::canonical(std::filesystem::path(buf.data()));
      } catch (...) {
        return std::filesystem::path(buf.data());
      }
    }
    throw std::runtime_error("_NSGetExecutablePath failed after buffer resize");
  }
#else
  char    aPath[FILENAME_MAX];
  ssize_t count = readlink("/proc/self/exe", aPath, FILENAME_MAX);
  if (count <= 0) {
    throw std::runtime_error("failed to resolve /proc/self/exe");
  }
  return std::filesystem::path(std::string(aPath, count));
#endif
}

static void fUnsetReservedArgs(common_preset & pPreset, bool pUnsetModelArgs) {
  pPreset.mUnsetOption("DGS_ARG_LOCAL_SHELL_MCP_SERVER");
  pPreset.mUnsetOption("DGS_ARG_API_PREFIX");
  pPreset.mUnsetOption("DGS_ARG_SSL_KEY_FILE");
  pPreset.mUnsetOption("DGS_ARG_SSL_CERT_FILE");
  pPreset.mUnsetOption("DGS_API_KEY");
  pPreset.mUnsetOption("DGS_ARG_API_KEY_FILE");
  pPreset.mUnsetOption("DGS_ARG_MODELS_DIR");
  pPreset.mUnsetOption("DGS_ARG_MODELS_MAX");
  pPreset.mUnsetOption("DGS_ARG_MODELS_PRESET");
  pPreset.mUnsetOption("DGS_ARG_MODELS_AUTOLOAD");
  if (pUnsetModelArgs) {
    pPreset.mUnsetOption("DGS_ARG_MODEL");
    pPreset.mUnsetOption("DGS_ARG_MMPROJ");
    pPreset.mUnsetOption("DGS_ARG_ALIAS");
    pPreset.mUnsetOption("DGS_ARG_HF_REPO");
  }
}

static std::vector<std::string> fGetEnvironment() {
  std::vector<std::string> lEnv;

#ifdef _WIN32
  LPWCH env_block = GetEnvironmentStringsW();
  if (!env_block) {
    return env;
  }
  for (LPWCH e = env_block; *e; e += wcslen(e) + 1) {
    env.emplace_back(wstring_to_utf8(e));
  }
  FreeEnvironmentStringsW(env_block);
#else
  if (environ == nullptr) {
    return lEnv;
  }
  for (char ** vE = environ; *vE != nullptr; vE++) {
    lEnv.emplace_back(*vE);
  }
#endif

  return lEnv;
}

void server_model_meta::mUpdateArgs(common_preset_context & pCtxPreset, std::string pBinPath) {
  // update params
  fUnsetReservedArgs(vPreset, false);
  vPreset.mSetOption(pCtxPreset, "DGS_ARG_HOST", CHILD_ADDR);
  vPreset.mSetOption(pCtxPreset, "DGS_ARG_PORT", std::to_string(vPort));
  vPreset.mSetOption(pCtxPreset, "DGS_ARG_ALIAS", name);
  // the child output goes through the router to its terminal, so it follows the router colors
  vPreset.mSetOption(pCtxPreset, "DGS_ARG_LOG_COLORS", fCommonLogGetColors(fCommonLogMain()) ? "on" : "off");
  // TODO: maybe validate preset before rendering ?
  // render args
  lArgs = vPreset.mToArgs(pBinPath);

  // unified binary dispatches by subcommand, re-inject it right after the
  // binary path so the child starts as 'dgs serve ...' not 'dgs ...'
  const char * vAppCmd = std::getenv("DGS_APP_CMD");
  if (vAppCmd != nullptr && vAppCmd[0] != '\0' && !pBinPath.empty()) {
    lArgs.insert(lArgs.begin() + 1, vAppCmd);
  }
}

void server_model_meta::mUpdateCaps(const common_params & pBase) {
  // reset to the default so a failed refresh cannot keep old values
  vArchitecture = fServerModelArchitectureJson(false, false, false, { "text" });
  vNCtxTrain    = 0;

  // resolve the model file offline; do not download
  common_params vParams;
  vParams.vModel    = pBase.vModel;
  // --no-mmproj applies to child models and blocks auto-attached projectors
  vParams.vNoMmproj = pBase.vNoMmproj;
  try {
    vPreset.mApplyToParams(vParams, {
                                      "DGS_ARG_MODEL",
                                      "DGS_ARG_MODEL_URL",
                                      "DGS_ARG_MMPROJ",
                                      "DGS_ARG_MMPROJ_URL",
                                      "DGS_ARG_MMPROJ_AUTO",
                                      "DGS_ARG_HF_REPO",
                                      "DGS_ARG_HF_FILE",
                                    });
    vParams.vOffline               = true;
    common_models_handler vHandler = fCommonModelsHandlerInit(vParams, cDgsExampleServer);
    fCommonModelsHandlerApply(vHandler, vParams);
  } catch (const std::exception & vE) {
    LOG_WRN("failed to resolve the model of '%s': %s\n", name.c_str(), vE.what());
    return;
  }

  // read the output modalities and the trained context from the GGUF metadata
  std::vector<std::string> lOutputModalities = { "text" };
  if (!vParams.vModel.vPath.empty()) {
    const common_gguf_info info = fCommonGetGgufInfo(vParams.vModel.vPath);
    lOutputModalities           = fServerModelOutputModalities(info.vDecisionType);
    vNCtxTrain                  = info.vNCtxTrain;
  }

  bool vInpImage = false;
  bool vInpAudio = false;
  try {
    if (!vParams.vNoMmproj && !vParams.vMmproj.vPath.empty()) {
      mtmd_caps caps = fMtmdGetCapFromFile(vParams.vMmproj.vPath.c_str());
      vInpImage      = caps.vInpVision;
      vInpAudio      = caps.vInpAudio;
    }
  } catch (const std::exception & vE) {
    LOG_WRN("failed to read the multimodal capabilities of '%s': %s\n", name.c_str(), vE.what());
    // keep the output modalities from the GGUF metadata
  }

  // offline discovery cannot see video; a loaded model reports it
  vArchitecture = fServerModelArchitectureJson(vInpImage, vInpAudio, false, lOutputModalities);
}

//
// server_models
//

server_models::server_models(const common_params & pParams, int pArgc, char ** pArgv) :
  vCtxPreset(cDgsExampleServer),
  vBaseParams(pParams),
  lBaseEnv(fGetEnvironment()),
  vBasePreset(vCtxPreset.mLoadFromArgs(pArgc, pArgv)),
  vSched(std::make_unique<server_lru_sched>(*this)),
  vMonitor(std::make_unique<server_monitor>(*this)) {
  // propagate base params to child
  fUnsetReservedArgs(vBasePreset, true);

  // The public listener and its shell tools belong to the router, not its private model processes.
  lBaseEnv.erase(std::remove_if(lBaseEnv.begin(), lBaseEnv.end(),
                                [](const std::string & pEntry) {
                                  const std::string vName = pEntry.substr(0, pEntry.find('='));
                                  return vName == "DGS_ARG_LOCAL_SHELL_MCP_SERVER" || vName == "DGS_ARG_API_PREFIX" ||
                                         vName == "DGS_API_KEY" || vName == "DGS_ARG_API_KEY_FILE" ||
                                         vName == "DGS_ARG_SSL_KEY_FILE" || vName == "DGS_ARG_SSL_CERT_FILE" ||
                                         vName == "AIP_MODE" || vName == "AIP_HTTP_PORT";
                                }),
                 lBaseEnv.end());

  // do not propagate these options, but allow preset to explicitly set them
  vBasePreset.mUnsetOption("DGS_ARG_LOG_FILE");

  // set binary path
  try {
    vBinPath = fFsPathToUtf8(fGetServerExecPath());
  } catch (const std::exception & vE) {
    vBinPath = pArgv[0];
    LOG_WRN("failed to get server executable path: %s\n", vE.what());
    LOG_WRN("using original argv[0] as fallback: %s\n", pArgv[0]);
  }
  mLoadModels();
  vDebugFakeTiming = !fCommonGetEnv("DGS_SERVER_DEBUG_FAKE_TIMING").empty();
}

server_models::~server_models() = default;

void server_models::instance_t::mRequestExit() const {
  fRequestChildExit(*vSubproc);
}

void server_models::mAddModel(server_model_meta && pMeta) {
  if (dMapping.find(pMeta.name) != dMapping.end()) {
    throw std::runtime_error(fStringFormat("model '%s' appears multiple times", pMeta.name.c_str()));
  }

  // check model name does not conflict with existing aliases
  for (const auto & [key, inst] : dMapping) {
    if (inst.vMeta.sAliases.count(pMeta.name)) {
      throw std::runtime_error(
        fStringFormat("model name '%s' conflicts with alias of model '%s'", pMeta.name.c_str(), key.c_str()));
    }
  }

  // parse aliases from preset's --alias option (comma-separated)
  std::string vAliasStr;
  if (pMeta.vPreset.mGetOption("DGS_ARG_ALIAS", vAliasStr) && !vAliasStr.empty()) {
    for (auto & vAlias : fStringSplit<std::string>(vAliasStr, ',')) {
      vAlias = fStringStrip(vAlias);
      if (!vAlias.empty()) {
        pMeta.sAliases.insert(vAlias);
      }
    }
  }

  // parse tags from preset's --tags option (comma-separated)
  std::string vTagsStr;
  if (pMeta.vPreset.mGetOption("DGS_ARG_TAGS", vTagsStr) && !vTagsStr.empty()) {
    for (auto & vTag : fStringSplit<std::string>(vTagsStr, ',')) {
      vTag = fStringStrip(vTag);
      if (!vTag.empty()) {
        pMeta.sTags.insert(vTag);
      }
    }
  }

  // validate aliases do not conflict with existing names or aliases
  for (const auto & vAlias : pMeta.sAliases) {
    if (dMapping.find(vAlias) != dMapping.end()) {
      throw std::runtime_error(fStringFormat("alias '%s' for model '%s' conflicts with existing model name",
                                             vAlias.c_str(), pMeta.name.c_str()));
    }
    for (const auto & [key, inst] : dMapping) {
      if (inst.vMeta.sAliases.count(vAlias)) {
        throw std::runtime_error(fStringFormat("alias '%s' for model '%s' conflicts with alias of model '%s'",
                                               vAlias.c_str(), pMeta.name.c_str(), key.c_str()));
      }
    }
  }

  pMeta.mUpdateArgs(vCtxPreset, vBinPath);  // render args
  pMeta.mUpdateCaps(vBaseParams);
  std::string name = pMeta.name;
  dMapping[name]   = instance_t{ /* subproc */ std::make_shared<server_subproc>(),
                               /* meta    */ std::move(pMeta) };
}

void server_models::mNotifySse(const std::string & pEvent, const std::string & pModelId, const json & data) {
  std::unique_ptr<server_task_result_router> vResult = std::make_unique<server_task_result_router>();
  vResult->data                                      = {
    { "model", pModelId },
    { "event", pEvent   },
  };
  if (!data.is_null()) {
    vResult->data["data"] = data;
  }
  SRV_DBG("notifying SSE clients about event '%s' for model '%s': %s\n", pEvent.c_str(), pModelId.c_str(),
          fSafeJsonToStr(vResult->data).c_str());
  vSse.mBroadcast(std::move(vResult));
}

void server_models::mLoadModels() {
  // Phase 1: load presets from all sources - pure I/O, no lock needed
  // 1. cached models
  common_presets dCachedModels = vCtxPreset.mLoadFromCache();
  SRV_TRC("Loaded %zu cached model presets from %s\n", dCachedModels.size(), hf_cache::fGetCachePath().c_str());
  // 2. local models from --models-dir
  common_presets dLocalModels;
  if (!vBaseParams.vModelsDir.empty()) {
    dLocalModels = vCtxPreset.mLoadFromModelsDir(vBaseParams.vModelsDir);
    SRV_TRC("Loaded %zu local model presets from %s\n", dLocalModels.size(), vBaseParams.vModelsDir.c_str());
  }
  // 3. custom-path models from presets
  common_preset  vGlobal        = {};
  common_presets dCustomPresets = {};
  if (!vBaseParams.vModelsPreset.empty()) {
    dCustomPresets = vCtxPreset.mLoadFromIni(vBaseParams.vModelsPreset, vGlobal);
    SRV_TRC("Loaded %zu custom model presets from %s\n", dCustomPresets.size(), vBaseParams.vModelsPreset.c_str());
  }

  // cascade, apply global preset first
  dCachedModels  = vCtxPreset.mCascade(vGlobal, dCachedModels);
  dLocalModels   = vCtxPreset.mCascade(vGlobal, dLocalModels);
  dCustomPresets = vCtxPreset.mCascade(vGlobal, dCustomPresets);

  // note: if a model exists in both cached and local, local takes precedence
  common_presets                                       dFinalPresets;
  std::unordered_map<std::string, server_model_source> dSourceMap;
  for (const auto & [name, preset] : dCachedModels) {
    dFinalPresets[name] = preset;
    dSourceMap[name]    = cServerModelSourceCache;
  }
  for (const auto & [name, preset] : dLocalModels) {
    dFinalPresets[name] = preset;
    dSourceMap[name]    = cServerModelSourceModelsDir;
  }
  for (const auto & [name, custom] : dCustomPresets) {
    if (dFinalPresets.find(name) != dFinalPresets.end()) {
      dFinalPresets[name].mMerge(custom);
    } else {
      dFinalPresets[name] = custom;
    }
    dSourceMap[name] = cServerModelSourcePreset;
  }

  // overlay router's own CLI args on top of every model preset so that
  // e.g. `dgs-server --temp 0` is honoured by all child processes
  for (auto & [name, preset] : dFinalPresets) {
    preset.mMerge(vBasePreset);
  }

  auto vGetSource = [&](const std::string & name) {
    return dSourceMap.count(name) ? dSourceMap.at(name) : cServerModelSourcePreset;
  };

  // hide cache models whose resolved file is already used by a preset with dedup-cache-models enabled
  std::set<std::string> sHiddenModels;
  {
    std::set<std::string> sPresetPaths;
    auto vAddHfPath = [&sPresetPaths](const common_preset & pPreset, const char * pRepoKey, const char * pFileKey) {
      std::string vHfRepo;
      if (!pPreset.mGetOption(pRepoKey, vHfRepo) || vHfRepo.empty()) {
        return;
      }
      std::string hf_file;
      pPreset.mGetOption(pFileKey, hf_file);
      std::string vPath = fCommonDownloadResolvePath(vHfRepo, hf_file);
      if (!vPath.empty()) {
        sPresetPaths.insert(vPath);
      }
    };
    for (const auto & [name, preset] : dCustomPresets) {
      std::string vVal;
      if (!preset.mGetOption(COMMON_ARG_PRESET_DEDUP_CACHE_MODELS, vVal) || !common_arg_utils::fIsTruthy(vVal)) {
        continue;
      }
      vAddHfPath(preset, "DGS_ARG_HF_REPO", "DGS_ARG_HF_FILE");
      vAddHfPath(preset, "DGS_ARG_SPEC_DRAFT_HF_REPO", "DGS_ARG_SPEC_DRAFT_MODEL");
    }
    if (!sPresetPaths.empty()) {
      for (const auto & [name, preset] : dCachedModels) {
        if (vGetSource(name) != cServerModelSourceCache) {
          continue;  // merged with another source, not a pure cache entry
        }
        std::string vPath = fCommonDownloadResolvePath(name);
        if (!vPath.empty() && sPresetPaths.count(vPath)) {
          SRV_INF("hiding cache model name=%s (deduplicated by a preset)\n", name.c_str());
          sHiddenModels.insert(name);
        }
      }
    }
  }

  // Helpers that read `mapping` - must be called while holding the lock.
  auto vJoinSet = [](const std::set<std::string> & pS) {
    std::string vResult;
    for (const auto & vV : pS) {
      if (!vResult.empty()) {
        vResult += ", ";
      }
      vResult += vV;
    }
    return vResult;
  };
  auto vLogAvailableModels = [&]() {
    SRV_INF("Available models (%zu):\n", dMapping.size());
    if (dMapping.empty()) {
      SRV_INF("%s", "  no models found on the system (use the Models manager in the DGS web interface)\n");
    } else {
      for (const auto & [name, inst] : dMapping) {
        const std::string cSource = fServerModelSourceToString(inst.vMeta.vSource);

        std::string info;
        if (!inst.vMeta.sAliases.empty()) {
          info += " (aliases: " + vJoinSet(inst.vMeta.sAliases) + ")";
        }
        if (!inst.vMeta.sTags.empty()) {
          info += " [tags: " + vJoinSet(inst.vMeta.sTags) + "]";
        }

        SRV_INF("  [%10s] %s%s\n", cSource.c_str(), name.c_str(), info.c_str());
      }
    }
  };
  auto vApplyStopTimeout = [&]() {
    for (auto & [name, inst] : dMapping) {
      std::string vVal;
      if (inst.vMeta.vPreset.mGetOption(COMMON_ARG_PRESET_STOP_TIMEOUT, vVal)) {
        try {
          inst.vMeta.vStopTimeout = std::stoi(vVal);
        } catch (...) {
          SRV_WRN("invalid stop-timeout value '%s' for model '%s', using default %d seconds\n", vVal.c_str(),
                  name.c_str(), DEFAULT_STOP_TIMEOUT);
          inst.vMeta.vStopTimeout = DEFAULT_STOP_TIMEOUT;
        }
      }
    }
  };
  auto vApplyHidden = [&]() {
    for (auto & [name, inst] : dMapping) {
      inst.vMeta.vHidden = sHiddenModels.count(name) > 0;
    }
  };
  // update_args() injects HOST/PORT/ALIAS/LOG_COLORS, so strip them before comparing presets
  auto vPresetOptionsForCompare = [](common_preset pP) {
    pP.mUnsetOption("DGS_ARG_HOST");
    pP.mUnsetOption("DGS_ARG_PORT");
    pP.mUnsetOption("DGS_ARG_ALIAS");
    pP.mUnsetOption("DGS_ARG_LOG_COLORS");
    return pP.dOptions;
  };

  // Phase 2: acquire the lock once for all mapping mutations.
  // We temporarily release it only when calling functions that acquire it internally (unload)
  std::unique_lock<std::mutex> vLk(vMutex);

  vNeedReload       = false;
  bool vIsFirstLoad = dMapping.empty();

  if (vIsFirstLoad) {
    // FIRST LOAD: add all models, then unlock for autoloading
    for (const auto & [name, preset] : dFinalPresets) {
      server_model_meta vMeta{
        /* source        */ vGetSource(name),
        /* preset        */ preset,
        /* name          */ name,
        /* aliases       */ {},
        /* tags          */ {},
        /* port          */ 0,
        /* status        */ cServerModelStatusUnloaded,
        /* last_used     */ 0,
        /* args          */ std::vector<std::string>(),
        /* loaded_info   */ {},
        /* progress      */ {},
        /* exit_code     */ 0,
        /* stop_timeout  */ DEFAULT_STOP_TIMEOUT,
        // /* need_download */ false,
      };
      mAddModel(std::move(vMeta));
    }
    vApplyStopTimeout();
    vApplyHidden();
    vLogAvailableModels();

    // skipped on reload, see startup_models
    if (lStartupModels.has_value()) {
      std::vector<std::string> lModelsToLoad;
      for (const auto & [name, inst] : dMapping) {
        std::string vVal;
        if (inst.vMeta.vPreset.mGetOption(COMMON_ARG_PRESET_LOAD_ON_STARTUP, vVal) &&
            common_arg_utils::fIsTruthy(vVal)) {
          lModelsToLoad.push_back(name);
        }
      }
      if ((int) lModelsToLoad.size() > vBaseParams.vModelsMax) {
        throw std::runtime_error(fStringFormat("number of models to load on startup (%zu) exceeds models_max (%d)",
                                               lModelsToLoad.size(), vBaseParams.vModelsMax));
      }

      // to be lazy-loaded after main() setup phase is completed
      lStartupModels = std::move(lModelsToLoad);
    }

    vLk.unlock();
  } else {
    // RELOAD: diff the new preset list against the current mapping and reconcile
    vIsReloading = true;

    // find running models whose source was removed or whose preset changed
    std::vector<std::string> lToUnload;
    for (const auto & [name, inst] : dMapping) {
      if (!inst.vMeta.mIsRunning()) {
        continue;
      }
      auto vIt = dFinalPresets.find(name);
      if (vIt == dFinalPresets.end()) {
        lToUnload.push_back(name);  // removed from source
      } else if (vPresetOptionsForCompare(inst.vMeta.vPreset) != vPresetOptionsForCompare(vIt->second)) {
        lToUnload.push_back(name);  // preset changed
      }
    }

    // unload() acquires the lock internally, so release before each call
    for (const auto & name : lToUnload) {
      SRV_INF("(reload) unloading model name=%s (source updated or removed)\n", name.c_str());
      vLk.unlock();
      mUnload(name);
      vLk.lock();
    }

    // wait for all targeted models to reach UNLOADED; cv.wait handles unlock/relock
    vCv.wait(vLk, [&]() {
      for (const auto & name : lToUnload) {
        auto vIt = dMapping.find(name);
        if (vIt != dMapping.end() && vIt->second.vMeta.mIsRunning()) {
          return false;
        }
      }
      return true;
    });

    // erase models no longer in any source
    for (auto vIt = dMapping.begin(); vIt != dMapping.end();) {
      if (vIt->second.vMeta.vStatus == cServerModelStatusDownloading) {
        ++vIt;  // download thread is still busy, skip
      } else if (vIt->second.vMeta.vStatus == cServerModelStatusDownloaded) {
        // download finished, safe to erase
        vIt = dMapping.erase(vIt);
      } else if (dFinalPresets.find(vIt->first) == dFinalPresets.end()) {
        SRV_INF("(reload) removing model name=%s (no longer in source)\n", vIt->first.c_str());
        vIt = dMapping.erase(vIt);
      } else {
        ++vIt;
      }
    }

    // update presets for non-running models still in source
    for (auto & [name, inst] : dMapping) {
      if (inst.vMeta.mIsRunning()) {
        continue;
      }
      auto vIt = dFinalPresets.find(name);
      if (vIt == dFinalPresets.end()) {
        continue;  // erased above
      }

      inst.vMeta.vPreset = vIt->second;

      // re-parse aliases, then validate against other models
      std::set<std::string> sNewAliases;
      std::string           vAliasStr;
      if (inst.vMeta.vPreset.mGetOption("DGS_ARG_ALIAS", vAliasStr) && !vAliasStr.empty()) {
        for (auto & vAlias : fStringSplit<std::string>(vAliasStr, ',')) {
          vAlias = fStringStrip(vAlias);
          if (!vAlias.empty()) {
            sNewAliases.insert(vAlias);
          }
        }
      }
      inst.vMeta.sAliases.clear();
      for (const auto & vAlias : sNewAliases) {
        bool vConflict = false;
        for (const auto & [other_name, other_inst] : dMapping) {
          if (other_name == name) {
            continue;
          }
          if (other_name == vAlias || other_inst.vMeta.sAliases.count(vAlias)) {
            SRV_WRN("(reload) alias '%s' for model '%s' conflicts with model '%s', skipping\n", vAlias.c_str(),
                    name.c_str(), other_name.c_str());
            vConflict = true;
            break;
          }
        }
        if (!vConflict) {
          inst.vMeta.sAliases.insert(vAlias);
        }
      }

      // re-parse tags
      inst.vMeta.sTags.clear();
      std::string vTagsStr;
      if (inst.vMeta.vPreset.mGetOption("DGS_ARG_TAGS", vTagsStr) && !vTagsStr.empty()) {
        for (auto & vTag : fStringSplit<std::string>(vTagsStr, ',')) {
          vTag = fStringStrip(vTag);
          if (!vTag.empty()) {
            inst.vMeta.sTags.insert(vTag);
          }
        }
      }

      inst.vMeta.vExitCode = 0;  // clear failed state so the model can be reloaded
      inst.vMeta.mUpdateArgs(vCtxPreset, vBinPath);
      inst.vMeta.mUpdateCaps(vBaseParams);
    }

    // add models that are new in this reload, load-on-startup is not honored here since a
    // reload never spawns an instance
    for (const auto & [name, preset] : dFinalPresets) {
      if (dMapping.find(name) == dMapping.end()) {
        server_model_meta vMeta{
          /* source        */ vGetSource(name),
          /* preset        */ preset,
          /* name          */ name,
          /* aliases       */ {},
          /* tags          */ {},
          /* port          */ 0,
          /* status        */ cServerModelStatusUnloaded,
          /* last_used     */ 0,
          /* args          */ std::vector<std::string>(),
          /* loaded_info   */ {},
          /* progress      */ {},
          /* exit_code     */ 0,
          /* stop_timeout  */ DEFAULT_STOP_TIMEOUT,
          // /* need_download */ false,
        };
        mAddModel(std::move(vMeta));
      }
    }

    vApplyStopTimeout();
    vApplyHidden();

    // clear reload flag under the lock, this releases the load() calls waiting on !is_reloading
    vIsReloading = false;
    vCv.notify_all();

    vLogAvailableModels();

    vLk.unlock();

    mNotifySse("models_reload", "*");
  }
}

void server_models::mLoadStartupModels() {
  std::vector<std::string> lToLoad;
  {
    std::lock_guard<std::mutex> vLk(vMutex);
    if (!lStartupModels.has_value()) {
      return;  // already drained
    }
    lToLoad = std::move(*lStartupModels);
    lStartupModels.reset();
  }
  for (const auto & name : lToLoad) {
    SRV_INF("(startup) loading model %s\n", name.c_str());
    mLoad(name);
  }
}

void server_models::mUpdateMeta(const std::string & name, const server_model_meta & pMeta) {
  std::lock_guard<std::mutex> vLk(vMutex);
  auto                        vIt = dMapping.find(name);
  if (vIt != dMapping.end()) {
    vIt->second.vMeta = pMeta;
  }
  vCv.notify_all();  // notify wait_until_loading_finished
}

bool server_models::mHasModel(const std::string & name) {
  std::lock_guard<std::mutex> vLk(vMutex);
  if (dMapping.find(name) != dMapping.end()) {
    return true;
  }
  for (const auto & [key, inst] : dMapping) {
    if (inst.vMeta.sAliases.count(name)) {
      return true;
    }
  }
  return false;
}

std::optional<server_model_meta> server_models::mGetMeta(const std::string & name) {
  std::unique_lock<std::mutex> vLk(vMutex);
  if (vNeedReload) {
    vLk.unlock();
    mLoadModels();
    vLk.lock();
  }

  auto vIt = dMapping.find(name);
  if (vIt != dMapping.end()) {
    return vIt->second.vMeta;
  }
  for (const auto & [key, inst] : dMapping) {
    if (inst.vMeta.sAliases.count(name)) {
      return inst.vMeta;
    }
  }
  return std::nullopt;
}

std::vector<server_model_meta> server_models::mGetAllMeta() {
  std::unique_lock<std::mutex> vLk(vMutex);
  if (vNeedReload) {
    vLk.unlock();
    mLoadModels();
    vLk.lock();
  }

  std::vector<server_model_meta> lResult;
  lResult.reserve(dMapping.size());
  for (const auto & [name, inst] : dMapping) {
    lResult.push_back(inst.vMeta);
  }
  return lResult;
}

void server_models::mUnloadLru() {
  if (vBaseParams.vModelsMax <= 0) {
    return;  // no limit
  }
  // remove one of the servers if we passed the models_max (least recently used - LRU)
  std::string vLruModelName;
  {
    std::unique_lock<std::mutex> vLk(vMutex);
    if (vSched->mHasCapacity(vLk)) {
      return;
    }
    vLruModelName = vSched->mPickVictim(vLk);
  }
  if (!vLruModelName.empty()) {
    SRV_INF("models_max limit reached, removing LRU name=%s\n", vLruModelName.c_str());
    mUnload(vLruModelName);
    // wait for unload to complete
    {
      std::unique_lock<std::mutex> vLk(vMutex);
      vCv.wait(
        vLk, [this, &vLruModelName]() { return dMapping[vLruModelName].vMeta.vStatus == cServerModelStatusUnloaded; });
    }
  }
}

void server_models::mLoad(const std::string & name) {
  mLoad(name, load_options{});
}

void server_models::mLoad(const std::string & name, const load_options & pOpts) {
  if (vDebugFakeTiming) {
    // do not hold the mutex here, other requests must keep making progress
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }

  if (!pOpts.vCustomMeta.has_value()) {
    if (!mHasModel(name)) {
      throw std::runtime_error("model name=" + name + " is not found");
    }
    mUnloadLru();
  }

  std::unique_lock<std::mutex> vLk(vMutex);
  // edge case: block until any in-progress reload has finished so we always load
  // against the freshest preset and a consistent mapping state
  vCv.wait(vLk, [this]() { return !vIsReloading; });

  auto vMeta = pOpts.vCustomMeta.has_value() ? *pOpts.vCustomMeta : dMapping[name].vMeta;
  if (vMeta.vStatus != cServerModelStatusUnloaded) {
    SRV_INF("model %s is not ready\n", name.c_str());
    return;
  }

  // Re-check capacity under the lock to prevent concurrent loads from
  // exceeding models_max. Without this, the window between unload_lru()
  // releasing its lock and this lock_guard acquiring allows multiple
  // threads to each observe capacity and all proceed to load.
  // Download workers do not use models_max slots.
  if (pOpts.vMode == cServerChildModeNormal && vBaseParams.vModelsMax > 0) {
    size_t vCountActive = 0;
    for (const auto & vM : dMapping) {
      if (vM.second.vMeta.mIsRunning()) {
        vCountActive++;
      }
    }
    if (vCountActive >= (size_t) vBaseParams.vModelsMax) {
      throw std::runtime_error("model limit reached, try again later");
    }
  }

  // prepare new instance info
  instance_t vInst;
  vInst.vMeta             = vMeta;
  vInst.vMeta.vPort       = fCommonHttpGetFreePort();
  vInst.vMeta.vStatus     = cServerModelStatusLoading;
  vInst.vMeta.vLoadedInfo = json{};
  vInst.vMeta.vLastUsed   = ggml_time_ms();

  if (vInst.vMeta.vPort <= 0) {
    throw std::runtime_error("failed to get a port number");
  }

  vInst.vSubproc = std::make_shared<server_subproc>();
  {
    SRV_INF("spawning server instance with name=%s on port %d\n", vInst.vMeta.name.c_str(), vInst.vMeta.vPort);

    vInst.vMeta.mUpdateArgs(vCtxPreset, vBinPath);            // render args

    std::vector<std::string> lChildArgs = vInst.vMeta.lArgs;  // copy
    std::vector<std::string> lChildEnv  = lBaseEnv;           // copy
    lChildEnv.push_back("DGS_SERVER_ROUTER_PORT=" + std::to_string(vBaseParams.vPort));

    if (pOpts.vMode == cServerChildModeDownload) {
      vInst.vMeta.vStatus = cServerModelStatusDownloading;
      lChildEnv.push_back("DGS_SERVER_CHILD_MODE=download");
      lChildEnv.push_back("DGS_ARG_HF_REPO=" + name);
    }

    SRV_INF("%s", "spawning server instance with args:\n");
    for (const auto & vArg : lChildArgs) {
      SRV_INF("  %s\n", vArg.c_str());
    }
    vInst.vMeta.lArgs = lChildArgs;  // save for debugging

    // the child writes its commands to stdout and its logs to stderr
    int vOptions = subprocess_option_no_window;
    if (!vInst.vSubproc->vSproc.mCreate(lChildArgs, vOptions, lChildEnv)) {
      throw std::runtime_error("failed to spawn server instance");
    }
  }

  // old process should have exited already, but just in case, we clean it up here
  {
    auto vIt = dMapping.find(name);
    if (vIt != dMapping.end() && vIt->second.vSubproc && vIt->second.vSubproc->mIsAlive()) {
      SRV_WRN("old process for model name=%s is still alive, this is unexpected\n", name.c_str());
      vIt->second.vSubproc->mTerminate();  // force kill
    }
  }

  mNotifySse("model_status", name,
             {
               { "status", fServerModelStatusToString(vInst.vMeta.vStatus) },
  });

  auto vProc     = vInst.vSubproc;
  int  vPort     = vInst.vMeta.vPort;
  dMapping[name] = std::move(vInst);
  vMonitor->mWatch(name, vProc, pOpts.vMode, vPort);
  vCv.notify_all();
}

void server_models::mRequestStop(const std::string & name, bool pSendExit) {
  auto vIt = dMapping.find(name);
  if (vIt == dMapping.end() || sStoppingModels.count(name)) {
    return;
  }
  sStoppingModels.insert(name);
  vMonitor->mStop(name, vIt->second.vMeta.vStopTimeout, pSendExit);
}

void server_models::mOnChildExit(const std::string &                     name,
                                 const std::shared_ptr<server_subproc> & pProc,
                                 server_child_mode                       pMode,
                                 int                                     pExitCode) {
  {
    std::lock_guard<std::mutex> vLk(vMutex);
    auto                        vIt = dMapping.find(name);
    if (vIt == dMapping.end() || vIt->second.vSubproc != pProc) {
      sStoppingModels.erase(name);
      return;  // entry erased, or a newer instance took the name
    }
  }
  if (pMode == cServerChildModeDownload) {
    // instance will be cleaned up on next load_models() call
    std::lock_guard<std::mutex> vLk(vMutex);
    sStoppingModels.erase(name);
    vCv.notify_all();
  } else {
    mUpdateStatus(name, { cServerModelStatusUnloaded, pExitCode });
  }
}

void server_models::mUnload(const std::string & name) {
  std::unique_lock<std::mutex> vLk(vMutex);
  auto                         vIt = dMapping.find(name);
  if (vIt != dMapping.end()) {
    if (vIt->second.vMeta.vStatus == cServerModelStatusDownloading) {
      SRV_INF("cancelling download for model name=%s\n", name.c_str());
      vIt->second.mRequestExit();
      // for convenience, we wait the status change here
      mWait(vLk, name,
            [](const server_model_meta & pNewMeta) { return pNewMeta.vStatus != cServerModelStatusDownloading; });
    } else if (vIt->second.vMeta.mIsRunning()) {
      SRV_INF("stopping model instance name=%s\n", name.c_str());
      bool vLoading = vIt->second.vMeta.vStatus == cServerModelStatusLoading;
      if (vLoading) {
        // special case: if model is in loading state, unloading means force-killing it
        SRV_WRN("model name=%s is still loading, force-killing\n", name.c_str());
        vIt->second.vSubproc->mTerminate();
      }
      mRequestStop(name, !vLoading);
      // status change will be handled by the monitor
    } else {
      SRV_WRN("model instance name=%s is not running\n", name.c_str());
    }
  }
}

void server_models::mUnloadAll() {
  std::unique_lock<std::mutex> vLk(vMutex);
  for (auto & [name, inst] : dMapping) {
    if (inst.vMeta.vStatus == cServerModelStatusDownloading) {
      SRV_INF("cancelling download for model name=%s\n", name.c_str());
      inst.mRequestExit();
    } else if (inst.vMeta.mIsRunning()) {
      SRV_INF("stopping model instance name=%s\n", name.c_str());
      bool vLoading = inst.vMeta.vStatus == cServerModelStatusLoading;
      if (vLoading) {
        inst.vSubproc->mTerminate();
      }
      mRequestStop(name, !vLoading);
    }
  }
  // wait for every child to exit, the monitor force-kills the ones that ignore the exit command
  vCv.wait(vLk, [this]() {
    for (const auto & [name, inst] : dMapping) {
      if (inst.vMeta.mIsRunning() || inst.vMeta.vStatus == cServerModelStatusDownloading) {
        return false;
      }
    }
    return true;
  });
}

void server_models::mUpdateStatus(const std::string & name, const update_status_args & pArgs) {
  std::unique_lock<std::mutex> vLk(vMutex);
  auto                         vIt = dMapping.find(name);
  if (vIt != dMapping.end()) {
    auto & vMeta    = vIt->second.vMeta;
    vMeta.vStatus   = pArgs.vStatus;
    vMeta.vExitCode = pArgs.vExitCode;
    if (pArgs.vStatus == cServerModelStatusUnloaded) {
      sStoppingModels.erase(name);
    }
    if (!pArgs.vLoadedInfo.is_null()) {
      vMeta.vLoadedInfo = pArgs.vLoadedInfo;
      // the child replaces both arrays in full; a bad or missing value changes nothing
      if (pArgs.vLoadedInfo.contains("architecture") && pArgs.vLoadedInfo.at("architecture").is_object()) {
        const json & vChildArch = pArgs.vLoadedInfo.at("architecture");
        for (const char * vKey : { "input_modalities", "output_modalities" }) {
          if (!vChildArch.contains(vKey) || !vChildArch.at(vKey).is_array()) {
            continue;
          }
          std::vector<std::string> lModalities;
          bool                     vValid = true;
          for (const auto & vM : vChildArch.at(vKey)) {
            if (!vM.is_string()) {
              vValid = false;
              break;
            }
            lModalities.push_back(vM.get<std::string>());
          }
          if (vValid) {
            vMeta.vArchitecture[vKey] = std::move(lModalities);
          }
        }
      }
    }
    if (!pArgs.vProgress.is_null()) {
      vMeta.vProgress = pArgs.vProgress;
    }
    // a model that comes up idle or goes down changes the slot count for queued requests
    vSched->mTick(vLk);
  }
  // broadcast status change to SSE
  {
    json data = {
      { "status", fServerModelStatusToString(pArgs.vStatus) },
    };
    if (pArgs.vStatus == cServerModelStatusUnloaded) {
      data["exit_code"] = pArgs.vExitCode;
    }
    if (!pArgs.vLoadedInfo.is_null()) {
      data["info"] = pArgs.vLoadedInfo;
    }
    if (!pArgs.vProgress.is_null()) {
      data["progress"] = pArgs.vProgress;
    }
    // note: notify_sse doesn't acquire the lock, so no deadlock here
    mNotifySse("status_change", name, data);
  }
  vCv.notify_all();
}

void server_models::mUpdateDownloadProgress(const std::string &              name,
                                            const common_download_progress & pProgress,
                                            bool                             pDone,
                                            bool                             pOk) {
  json vCurr;
  {
    std::lock_guard<std::mutex> vLk(vMutex);
    auto                        vIt = dMapping.find(name);
    if (vIt != dMapping.end()) {
      if (pDone) {
        // mark the instance to be erased on next load_models() call
        vIt->second.vMeta.vStatus = cServerModelStatusDownloaded;
        vNeedReload               = true;
      } else {
        json & info = vIt->second.vMeta.vLoadedInfo;
        if (!info.contains("progress")) {
          info["progress"] = json{};
        }
        info["progress"][pProgress.vUrl] = {
          { "done",  pProgress.vDownloaded },
          { "total", pProgress.vTotal      },
        };
        vCurr = vIt->second.vMeta.vLoadedInfo;  // copy
      }
    }
  }
  if (pDone) {
    vCv.notify_all();  // notify in case unload() is waiting for download to be cancelled
    mNotifySse(pOk ? "download_finished" : "download_failed", name, {});
  } else {
    mNotifySse("download_progress", name, vCurr);
  }
}

bool server_models::mRemove(const std::string & name) {
  // do everything under one lock acquisition; avoid get_meta() /
  // unload() because they can trigger load_models() which erases
  // transient DOWNLOADING / DOWNLOADED entries as a side-effect
  std::unique_lock<std::mutex> vLk(vMutex);

  auto vIt = dMapping.find(name);
  if (vIt == dMapping.end()) {
    throw std::runtime_error("model name=" + name + " is not found");
  }
  if (vIt->second.vMeta.vSource != cServerModelSourceCache) {
    throw std::runtime_error("model name=" + name + " is not removable (not from cache)");
  }

  if (vIt->second.vMeta.vStatus == cServerModelStatusDownloading) {
    // cancel in-flight download
    SRV_INF("cancelling download for model name=%s\n", name.c_str());
    vIt->second.mRequestExit();
  } else if (vIt->second.vMeta.mIsRunning()) {
    // stop running instance
    SRV_INF("stopping model instance name=%s\n", name.c_str());
    bool vLoading = vIt->second.vMeta.vStatus == cServerModelStatusLoading;
    if (vLoading) {
      vIt->second.vSubproc->mTerminate();
    }
    mRequestStop(name, !vLoading);
  }

  // wait until the child is gone
  mWait(vLk, name, [](const server_model_meta & pMeta) {
    return pMeta.vStatus == cServerModelStatusUnloaded || pMeta.vStatus == cServerModelStatusDownloaded;
  });

  // re-find after wait - load_models() may have erased the entry during the wait
  vIt = dMapping.find(name);
  if (vIt == dMapping.end()) {
    // load_models() already erased the entry; we just need to clean up the cached files on disk
    vLk.unlock();
    bool vOk = fCommonDownloadRemove(name);
    SRV_INF("removing model name=%s from cache (%s)\n", name.c_str(), vOk ? "succeeded" : "partial");
    mNotifySse("model_remove", name, {});
    return true;
  }

  // remove from disk (best-effort: cancelled downloads may have no cached files)
  bool vOk = fCommonDownloadRemove(name);
  dMapping.erase(name);
  if (!vOk) {
    SRV_WRN("removing model name=%s from disk returned false (no cached files?)\n", name.c_str());
  }
  SRV_INF("removing model name=%s from cache (%s)\n", name.c_str(), vOk ? "succeeded" : "partial");
  mNotifySse("model_remove", name, {});
  return true;
}

void server_models::mWait(const std::string & name, std::function<bool(const server_model_meta &)> pPredicate) {
  std::unique_lock<std::mutex> vLk(vMutex);
  mWait(vLk, name, pPredicate);
}

void server_models::mWait(std::unique_lock<std::mutex> &                 pLk,
                          const std::string &                            name,
                          std::function<bool(const server_model_meta &)> pPredicate) {
  vCv.wait(pLk, [this, &name, &pPredicate]() {
    auto vIt = dMapping.find(name);
    if (vIt != dMapping.end()) {
      return pPredicate(vIt->second.vMeta);
    }
    // model was removed from mapping by another code path (e.g. load_models()).
    // nothing left to wait for - tell the caller to proceed.
    return true;
  });
}

bool server_models::mEnsureModelReady(const std::string & name, const std::function<bool()> & pShouldStop) {
  auto vMeta = mGetMeta(name);
  if (!vMeta.has_value()) {
    throw std::runtime_error("model name=" + name + " is not found");
  }
  bool vStopping;
  {
    std::lock_guard<std::mutex> vLk(vMutex);
    vStopping = sStoppingModels.count(name) > 0;
  }
  if (!vStopping && vMeta->mIsReady()) {
    return false;  // ready for taking requests
  }
  if (!vStopping && vMeta->vStatus == cServerModelStatusSleeping) {
    return false;  // child is sleeping but still running; new request will wake it up
  }

  bool vQueued  = false;
  bool vDidLoad = false;
  {
    std::unique_lock<std::mutex> vLk(vMutex);
    auto                         vIt = dMapping.find(name);
    if (vIt != dMapping.end() && vIt->second.vMeta.vStatus == cServerModelStatusUnloaded) {
      // the queue entry protects the model from eviction until its waiters leave
      vSched->mJoin(vLk, name);
      vSched->mTick(vLk);
      vQueued = true;
    }
  }

  // while queued, this is also where the load happens: the head of the queue does it
  SRV_INF("waiting until model name=%s is fully loaded...\n", name.c_str());
  std::unique_lock<std::mutex> vLk(vMutex);
  auto                         vLeaveQueue = [this, &vQueued, &vLk, &name]() {
    if (vQueued) {
      vSched->mLeave(vLk, name);
      vQueued = false;
    }
  };

  try {
    bool vSawLoading = false;
    while (true) {
      auto vIt = dMapping.find(name);
      if (vIt == dMapping.end()) {
        break;  // removed by another code path, nothing to wait for
      }
      if (sStoppingModels.count(name)) {
        // a stopping instance takes no new request, the next instance serves it
        if (!vQueued) {
          vSched->mJoin(vLk, name);
          vSched->mTick(vLk);
          vQueued = true;
        }
        if (pShouldStop && pShouldStop()) {
          throw std::runtime_error("request cancelled while waiting for model name=" + name);
        }
        vCv.wait_for(vLk, std::chrono::milliseconds(200));
        continue;
      }
      const server_model_status cStatus = vIt->second.vMeta.vStatus;

      if (cStatus == cServerModelStatusLoaded || cStatus == cServerModelStatusSleeping) {
        break;
      }
      if (cStatus == cServerModelStatusDownloading || cStatus == cServerModelStatusDownloaded) {
        break;  // do not wait on a download child
      }
      if (cStatus == cServerModelStatusLoading) {
        vSawLoading = true;
      } else if (cStatus == cServerModelStatusUnloaded) {
        if (vDidLoad || vSawLoading) {
          // a spawn happened and the instance came back down
          if (vIt->second.vMeta.mIsFailed()) {
            throw std::runtime_error("model name=" + name + " failed to load");
          }
          break;  // unloaded by another code path, caller reports "not running"
        }
        if (!vQueued) {
          break;  // not queued, and the load someone else started fell over
        }
      }

      if (pShouldStop && pShouldStop()) {
        // if a model was evicted for us, the free slot goes to the next waiter
        throw std::runtime_error("request cancelled while waiting for model name=" + name);
      }

      // our turn: our model is at the head, and a slot really did free up
      if (cStatus == cServerModelStatusUnloaded && vSched->mTryClaim(vLk, name)) {
        vLk.unlock();
        bool vOk = true;
        try {
          SRV_INF("slot available, loading queued model name=%s\n", name.c_str());
          mLoad(name);
          vDidLoad = true;
        } catch (const std::exception & vE) {
          // lost a race for the slot, stay in line and retry
          SRV_WRN("queued load of name=%s did not go through: %s\n", name.c_str(), vE.what());
          vOk = false;
        }
        vLk.lock();
        vSched->mClaimDone(vLk, name, vOk);
        vSched->mTick(vLk);
        continue;
      }

      vCv.wait_for(vLk, std::chrono::milliseconds(200));
    }
  } catch (...) {
    vLeaveQueue();
    vSched->mTick(vLk);  // a slot freed for this waiter goes to the next one
    throw;
  }
  vLeaveQueue();

  return true;
}

server_http_res_ptr server_models::mProxyRequest(const server_http_req & pReq,
                                                 const std::string &     pMethod,
                                                 const std::string &     name,
                                                 bool                    pUpdateLastUsed,
                                                 bool                    pDetached) {
  auto vMeta = mGetMeta(name);
  if (!vMeta.has_value()) {
    throw std::runtime_error("model name=" + name + " is not found");
  }
  if (!vMeta->mIsRunning()) {
    throw std::invalid_argument("model name=" + name + " is not running");
  }
  {
    std::unique_lock<std::mutex> vLk(vMutex);
    if (pUpdateLastUsed) {
      dMapping[name].vMeta.vLastUsed = ggml_time_ms();
    }
    dMapping[name].vReqCount++;
  }
  if (vDebugFakeTiming) {
    // sleep after req_count++, so the model counts as busy while we wait here
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }
  SRV_INF("proxying request to model %s on port %d\n", name.c_str(), vMeta->vPort);
  std::string vProxyPath = "/api" + pReq.vPath;
  if (!pReq.vQueryString.empty()) {
    vProxyPath += '?' + pReq.vQueryString;
  }
  auto vProxy = std::make_unique<server_http_proxy>(
    pMethod, "http", CHILD_ADDR, vMeta->vPort, vProxyPath, pReq.dHeaders, pReq.vBody, pReq.dFiles,
    // a detached request belongs to a replay session
    pDetached ? std::function<bool()>([]() { return false; }) : pReq.vShouldStop, vBaseParams.vTimeoutRead,
    vBaseParams.vTimeoutWrite);

  vProxy->vCleanup = [this, name]() {
    std::unique_lock<std::mutex> vLk(vMutex);
    auto                         vIt = dMapping.find(name);
    if (vIt != dMapping.end() && vIt->second.vReqCount > 0) {
      vIt->second.vReqCount--;
      if (vIt->second.vReqCount == 0) {
        vSched->mTick(vLk);
      }
    }
  };

  return vProxy;
}

void server_models::mHandleChildState(const std::string & name, const std::string & pRawInput) {
  server_state vState;
  json         vPayload;

  try {
    json data = json::mParse(pRawInput.substr(strlen(CMD_CHILD_TO_ROUTER_STATE)));
    vState    = fServerStateFromStr(fJsonValue(data, "state", std::string()));
    vPayload  = fJsonValue(data, "payload", json{});
  } catch (const std::exception & vE) {
    SRV_ERR("failed to parse child state update for name=%s: %s\n", name.c_str(), vE.what());
    return;
  }

  switch (vState) {
    case cServerStateDownloading:
      {
        std::string vResult      = fJsonValue(vPayload, "result", std::string());
        std::string vUrl         = fJsonValue(vPayload, "url", std::string());
        auto        vRequestExit = [&]() {
          std::lock_guard<std::mutex> vLk(vMutex);
          auto                        vIt = dMapping.find(name);
          if (vIt != dMapping.end()) {
            return vIt->second.mRequestExit();
          }
        };
        if (vResult == "download_finished") {
          mUpdateDownloadProgress(name, {}, true, true);
          vRequestExit();
        } else if (vResult == "download_failed") {
          mUpdateDownloadProgress(name, {}, true, false);
          vRequestExit();
        } else if (!vUrl.empty()) {
          common_download_progress vP;
          vP.vUrl        = vUrl;
          vP.vDownloaded = fJsonValue(vPayload, "downloaded", (size_t) 0);
          vP.vTotal      = fJsonValue(vPayload, "total", (size_t) 0);
          mUpdateDownloadProgress(name, vP, false);
        }
      }
      break;
    case cServerStateLoading:
      {
        mUpdateStatus(name, {
                              cServerModelStatusLoading,
                              0,
                              nullptr,  // no loaded_info yet
                              vPayload,
                            });
      }
      break;
    case cServerStateReady:
      {
        mUpdateStatus(name, {
                              cServerModelStatusLoaded,
                              0,
                              // note: payload can be empty if this is a wakeup from sleep
                              vPayload.size() > 0 ? vPayload : nullptr,
                              {},  // reset progress info
                            });
      }
      break;
    case cServerStateSleeping:
      {
        mUpdateStatus(name, { cServerModelStatusSleeping });
      }
      break;
    default:
      // should never happen, but just in case
      GGML_ASSERT(false && "unexpected state from child server");
  }
}

//
// server_child
//

server_child::server_child() {
  if (mIsChild()) {
    vCmdOut = fServerReserveStdout();
  }
}

server_child::~server_child() {
  if (vCmdOut) {
    fclose(vCmdOut);
  }
}

bool server_child::mIsChild() {
  const char * vRouterPort = std::getenv("DGS_SERVER_ROUTER_PORT");
  return vRouterPort != nullptr;
}

server_child_mode server_child::mGetMode() {
  const char * vMode = std::getenv("DGS_SERVER_CHILD_MODE");
  std::string  vModeStr(vMode ? vMode : "");
  if (vModeStr == "download") {
    return cServerChildModeDownload;
  } else {
    return cServerChildModeNormal;
  }
}

struct server_download_state : public common_download_callback {
  server_child *        vSelf;
  std::function<bool()> vShouldStop;
  std::atomic<int64_t>  vLastProgressTime{ 0 };  // multiple files downloading in different threads
  bool                  vIsOk = false;

  server_download_state(server_child * pS) : vSelf(pS) {}

  bool mRun(common_params & pParams) {
    try {
      common_models_handler vHandler = fCommonModelsHandlerInit(pParams, cDgsExampleServer);
      fCommonModelsHandlerApply(vHandler, pParams, this);
      vIsOk = true;
    } catch (const std::exception & vE) {
      auto vModelName = pParams.vModel.mGetName();
      SRV_ERR("download failed for model name=%s: %s\n", vModelName.c_str(), vE.what());
      vIsOk = false;
    }
    return vIsOk;
  }

  void mOnProgress(const common_download_progress & pP) {
    json data = {
      { "url",        pP.vUrl        },
      { "downloaded", pP.vDownloaded },
      { "total",      pP.vTotal      },
    };
    vSelf->mNotifyToRouter(fServerStateToStr(cServerStateDownloading), data);
  }

  void mOnStart(const common_download_progress & pP) override { mOnProgress(pP); }

  void mOnUpdate(const common_download_progress & pP) override {
    int64_t vNow = ggml_time_ms();
    // throttle progress updates to avoid flooding logs
    if (vNow - vLastProgressTime.load(std::memory_order_relaxed) >= 100) {
      mOnProgress(pP);
      vLastProgressTime.store(vNow, std::memory_order_relaxed);
    }
  }

  void mOnDone(const common_download_progress & pP, bool) override { mOnProgress(pP); }

  bool mIsCancelled() const override { return vShouldStop ? vShouldStop() : false; }
};

int server_child::mRunDownload(common_params & pParams) {
  auto vCancelled = std::make_shared<std::atomic<bool>>(false);

  // monitor stdin for cancellation command from the router
  std::thread vSignalThread = mSetup([vCancelled](int) { vCancelled->store(true, std::memory_order_relaxed); });

  server_download_state vDl(this);
  vDl.vShouldStop = [vCancelled]() {
    return vCancelled->load(std::memory_order_relaxed);
  };

  bool vOk = vDl.mRun(pParams);

  mNotifyToRouter(fServerStateToStr(cServerStateDownloading),
                  {
                    { "result", vOk ? "download_finished" : "download_failed" },
  });

  // router should send CMD_ROUTER_TO_CHILD_EXIT after receiving the result
  if (vSignalThread.joinable()) {
    vSignalThread.join();
  }

  SRV_INF("download completed %s\n", vOk ? "successfully" : "with errors");
  return 0;
}

std::thread server_child::mSetup(const std::function<void(int)> & pShutdownHandler) {
  // setup thread for monitoring stdin
  return std::thread([pShutdownHandler]() {
    // wait for EOF on stdin
    SRV_INF("%s", "child server monitoring thread started, waiting for EOF on stdin...\n");
    bool vEof = false;
    while (true) {
      std::string vLine;
      if (!std::getline(std::cin, vLine)) {
        // EOF detected, that means the router server is unexpectedly exit or killed
        vEof = true;
        break;
      }
      if (vLine.find(CMD_ROUTER_TO_CHILD_EXIT) != std::string::npos) {
        SRV_INF("%s", "exit command received, exiting...\n");
        pShutdownHandler(0);
        break;
      }
    }
    if (vEof) {
      SRV_INF("%s", "EOF on stdin detected, forcing shutdown...\n");
      exit(1);
    }
  });
}

void server_child::mNotifyToRouter(const std::string & pState, const json & pPayload) {
  json data = {
    { "state",   pState   },
    { "payload", pPayload },
  };
  std::lock_guard<std::mutex> vLk(vMtxStdout);
  fprintf(vCmdOut, "%s%s\n", CMD_CHILD_TO_ROUTER_STATE, fSafeJsonToStr(data).c_str());
  fflush(vCmdOut);
}

//
// server_models_routes
//

// RAII wrapper similar to server_response_reader, but doesn't use server_queue
static std::atomic<int> vSseClientIdCounter = 0;

struct server_models_sse_client {
  server_response & vQueueResults;
  int               vClientId;

  server_models_sse_client(server_response & pQ) :
    vQueueResults(pQ),
    vClientId(vSseClientIdCounter.fetch_add(1, std::memory_order_relaxed)) {
    SRV_DBG("new SSE client connected, assigned client_id=%d\n", vClientId);
    vQueueResults.mAddWaitingTaskId(vClientId);
  }

  ~server_models_sse_client() {
    SRV_DBG("SSE client disconnected, removing client_id=%d\n", vClientId);
    vQueueResults.mRemoveWaitingTaskId(vClientId);
  }

  // return nullptr if should_stop() is true before receiving a result
  // note: if one error is received, it will stop further processing and return error result
  server_task_result_ptr mNext(const std::function<bool()> & pShouldStop) {
    while (true) {
      static const int       cHttpPollingSeconds = 1;  // check should_stop every 1 second
      server_task_result_ptr vResult             = vQueueResults.mRecvWithTimeout({ vClientId }, cHttpPollingSeconds);
      if (vResult == nullptr) {
        // timeout, check stop condition
        if (pShouldStop()) {
          return nullptr;
        }
        // continue waiting otherwise
      } else {
        SRV_DBG("recv result for client_id=%d: %s\n", vClientId, fSafeJsonToStr(vResult->to_json()).c_str());
        return vResult;
      }
    }
    // should not reach here
  }
};

static void fResOk(std::unique_ptr<server_http_res> & pRes, const json & pResponseData) {
  pRes->vStatus = 200;
  pRes->data    = fSafeJsonToStr(pResponseData);
}

static void fResErr(std::unique_ptr<server_http_res> & pRes, const json & pErrorData) {
  pRes->vStatus = fJsonValue(pErrorData, "code", 500);
  pRes->data    = fSafeJsonToStr({
    { "error", pErrorData }
  });
}

static bool fRouterValidateModel(std::string &                      name,
                                 server_models &                    pModels,
                                 bool                               pModelsAutoload,
                                 std::unique_ptr<server_http_res> & pRes) {
  if (name.empty()) {
    fResErr(pRes, fFormatErrorResponse("model name is missing from the request", cErrorTypeInvalidRequest));
    return false;
  }
  auto vMeta = pModels.mGetMeta(name);
  if (!vMeta.has_value()) {
    fResErr(pRes, fFormatErrorResponse(fStringFormat("model '%s' not found", name.c_str()), cErrorTypeInvalidRequest));
    return false;
  }
  // resolve alias to canonical model name
  name = vMeta->name;
  if (!pModelsAutoload && !vMeta->mIsRunning()) {
    fResErr(pRes, fFormatErrorResponse("model is not loaded", cErrorTypeInvalidRequest));
    return false;
  }
  return true;
}

static bool fIsAutoload(const common_params & pParams, const server_http_req & pReq) {
  std::string vAutoload = pReq.mGetParam("autoload");
  if (vAutoload.empty()) {
    return pParams.vModelsAutoload;
  } else {
    return vAutoload == "true" || vAutoload == "1";
  }
}

// percent encode one query or path component, covers reserved chars without pulling in
// httplib::detail. used by the stream routes to forward conversation_id to children safely
static std::string fEncodeQs(const std::string & pIn) {
  std::string vOut;
  vOut.reserve(pIn.size() * 3);
  for (unsigned char vC : pIn) {
    bool vSafe = (vC >= 'A' && vC <= 'Z') || (vC >= 'a' && vC <= 'z') || (vC >= '0' && vC <= '9') || vC == '-' ||
                 vC == '_' || vC == '.' || vC == '~';
    if (vSafe) {
      vOut.push_back(char(vC));
    } else {
      char aBuf[4];
      std::snprintf(aBuf, sizeof(aBuf), "%%%02X", vC);
      vOut.append(aBuf, 3);
    }
  }
  return vOut;
}

// resolve the child that owns a conversation's stream session via the conv_id -> model map
// populated when the POST was routed. single map lookup then a meta lookup, no polling, no
// parsing of the conv id. returns nullopt when nothing maps, the caller answers not found and
// the client recovers
static std::optional<server_model_meta> fResolveChildForConv(server_models &     pModels,
                                                             const std::string & pConversationId) {
  if (pConversationId.empty()) {
    return std::nullopt;
  }
  auto vTracked = pModels.vConvModels.mLookup(pConversationId);
  if (!vTracked.has_value()) {
    return std::nullopt;
  }
  auto vMeta = pModels.mGetMeta(*vTracked);
  if (vMeta.has_value() && vMeta->mIsReady()) {
    return vMeta;
  }
  return std::nullopt;
}

// Memory snapshots of the running children. Loading and sleeping children only report their state,
// so the monitor never loads or wakes a model.
static std::vector<json> fCollectMonitorSnapshots(server_models & pModels) {
  std::vector<json> lLSnapshots;
  for (const server_model_meta & vMeta : pModels.mGetAllMeta()) {
    if (!vMeta.mIsRunning()) {
      continue;
    }
    if (!vMeta.mIsReady()) {
      lLSnapshots.push_back({
        { "state", fServerModelStatusToString(vMeta.vStatus) },
        { "model", vMeta.name                                }
      });
      continue;
    }
    try {
      httplib::Client vClient(CHILD_ADDR, vMeta.vPort);
      vClient.set_connection_timeout(0, 500 * 1000);
      vClient.set_read_timeout(0, 500 * 1000);
      vClient.set_write_timeout(0, 500 * 1000);
      auto vResponse = vClient.Get("/api/monitor/memory");
      if (!vResponse || vResponse->status != 200) {
        lLSnapshots.push_back({
          { "state", "unavailable" },
          { "model", vMeta.name    }
        });
        continue;
      }
      json dBody = json::mParse(vResponse->body);
      for (json dSnapshot : dBody.value("models", json::mArray())) {
        dSnapshot["model"] = vMeta.name;
        lLSnapshots.push_back(std::move(dSnapshot));
      }
    } catch (const std::exception &) {
      lLSnapshots.push_back({
        { "state", "unavailable" },
        { "model", vMeta.name    }
      });
    }
  }
  return lLSnapshots;
}

void server_models_routes::mInitRoutes() {
  vGetRouterMonitor = [this](const server_http_req &) {
    auto vResult  = std::make_unique<server_http_res>();
    vResult->data = fSafeJsonToStr(fMonitorReport(fCollectMonitorSnapshots(vModels)));
    return vResult;
  };
  vGetRouterMonitorMemory = [this](const server_http_req &) {
    auto vResult = std::make_unique<server_http_res>();
    json aModels = json::mArray();
    for (const auto & dSnapshot : fCollectMonitorSnapshots(vModels)) {
      aModels.push_back(dSnapshot);
    }
    vResult->data = fSafeJsonToStr(json{
      { "models", aModels }
    });
    return vResult;
  };

  if (!common_subproc::mIsSupported()) {
    throw std::runtime_error("subprocess is not enabled on this build");
  }

  this->vGetRouterProps = [this](const server_http_req & pReq) {
    std::string name = pReq.mGetParam("model");
    if (name.empty()) {
      // main instance
      auto vRes = std::make_unique<server_http_res>();
      fResOk(vRes, {
                     // TODO: add support for this on web UI
                     { "role",                        "router"                     },
                     { "max_instances",               vParams.vModelsMax           },
                     { "models_autoload",             vParams.vModelsAutoload      },
                     // this is a dummy response to make sure the UI doesn't break
                     { "model_alias",                 "dgs-server"                 },
                     { "model_path",                  "none"                       },
                     { "default_generation_settings",
                      {
                         { "params", json{} },
                         { "n_ctx", 0 },
                       }                                                           },
                     // New key
                     { "ui_settings",                 vUiSettings                  },
                     { "build_info",                  std::string(fDgsBuildInfo()) },
                     { "shell_mcp_server",            vParams.vShellMcpServer      },
                     { "cors_proxy_enabled",          vParams.vUiMcpProxy          },
      });
      return vRes;
    }
    return vProxyGet(pReq);
  };

  this->vProxyGet = [this](const server_http_req & pReq) {
    std::string vMethod   = "GET";
    std::string name      = pReq.mGetParam("model");
    bool        vAutoload = fIsAutoload(vParams, pReq);
    auto        vErrorRes = std::make_unique<server_http_res>();
    if (!fRouterValidateModel(name, vModels, vAutoload, vErrorRes)) {
      return vErrorRes;
    }
    if (vAutoload) {
      vModels.mEnsureModelReady(name, pReq.vShouldStop);
    }
    return vModels.mProxyRequest(pReq, vMethod, name, false);
  };

  this->vProxyPost = [this](const server_http_req & pReq) {
    std::string vMethod   = "POST";
    json        vBody     = json::mParse(pReq.vBody);
    std::string name      = fJsonValue(vBody, "model", std::string());
    bool        vAutoload = fIsAutoload(vParams, pReq);
    auto        vErrorRes = std::make_unique<server_http_res>();
    if (!fRouterValidateModel(name, vModels, vAutoload, vErrorRes)) {
      return vErrorRes;
    }
    // remember which child serves this conversation so the stream routes can route straight
    // to it without polling, keyed on the exact conv id from the header. registered before
    // the load wait so a stop issued while the model loads can erase the entry and cancel
    // this request instead of leaving an orphan generation
    std::string vConvId     = fServerStreamConvIdFromHeaders(pReq.dHeaders);
    uint64_t    vTicket     = vModels.vConvModels.mRemember(vConvId, name);
    // a dead socket must not cancel a session request, only a stop does (checked right below)
    auto        vShouldStop = vTicket == 0 ? pReq.vShouldStop : nullptr;
    bool        vWaited     = vAutoload && vModels.mEnsureModelReady(name, vShouldStop);
    if (vTicket != 0 && !vModels.vConvModels.mAlive(vConvId, vTicket)) {
      SRV_INF("request for conv_id=%s cancelled while model name=%s was loading\n", vConvId.c_str(), name.c_str());
      fResErr(vErrorRes, fFormatErrorResponse("request cancelled by a stop while the model was loading",
                                              cErrorTypeInvalidRequest));
      return vErrorRes;
    }
    // a session request that waited for a load detaches from the client socket: the
    // client may have dropped during the wait (page reload) and the session buffer must
    // still receive the generation for a later resume
    return vModels.mProxyRequest(pReq, vMethod, name, true,
                                 vWaited && vTicket != 0);  // update last usage for POST request only
  };

  this->vPostRouterModelsLoad = [this](const server_http_req & pReq) {
    auto        vRes  = std::make_unique<server_http_res>();
    json        vBody = json::mParse(pReq.vBody);
    std::string name  = fJsonValue(vBody, "model", std::string());
    auto        vMeta = vModels.mGetMeta(name);
    if (!vMeta.has_value()) {
      fResErr(vRes, fFormatErrorResponse("model is not found", cErrorTypeNotFound));
      return vRes;
    }
    if (vMeta->mIsRunning()) {
      fResErr(vRes, fFormatErrorResponse("model is already running", cErrorTypeInvalidRequest));
      return vRes;
    }
    vModels.mLoad(vMeta->name);
    fResOk(vRes, {
                   { "success", true }
    });
    return vRes;
  };

  this->vGetRouterModels = [this](const server_http_req & pReq) {
    bool vReload = !pReq.mGetParam("reload", "").empty();
    if (vReload) {
      vModels.mLoadModels();
    }
    auto        vRes        = std::make_unique<server_http_res>();
    json        vModelsJson = json::mArray();
    auto        lAllModels  = vModels.mGetAllMeta();
    std::time_t vT          = std::time(0);
    for (const auto & vMeta : lAllModels) {
      if (vMeta.vHidden) {
        continue;  // cache model deduplicated by a preset
      }
      json vStatus{
        { "value", fServerModelStatusToString(vMeta.vStatus) },
        { "args",  vMeta.lArgs                               },
      };
      if (!vMeta.vPreset.name.empty()) {
        common_preset vPresetCopy = vMeta.vPreset;
        fUnsetReservedArgs(vPresetCopy, false);
        vPresetCopy.mUnsetOption("DGS_ARG_HOST");
        vPresetCopy.mUnsetOption("DGS_ARG_PORT");
        vPresetCopy.mUnsetOption("DGS_ARG_ALIAS");
        vPresetCopy.mUnsetOption("DGS_ARG_TAGS");
        vStatus["preset"] = vPresetCopy.mToIni();
      }
      if (vMeta.mIsFailed()) {
        vStatus["exit_code"] = vMeta.vExitCode;
        vStatus["failed"]    = true;
      }

      json vModelInfo = json{
        { "id",           vMeta.name                                },
        { "aliases",      vMeta.sAliases                            },
        { "tags",         vMeta.sTags                               },
        { "object",       "model"                                   }, // for OAI-compat
        { "owned_by",     "dgscpp"                                  }, // for OAI-compat
        { "created",      vT                                        }, // for OAI-compat
        { "status",       vStatus                                   },
        { "architecture", vMeta.vArchitecture                       },
        { "source",       fServerModelSourceToString(vMeta.vSource) },
        { "can_remove",   vMeta.vSource == cServerModelSourceCache  },
        // {"need_download", meta.need_download},
        // TODO: add other fields from the GGUF metadata
      };

      if (vMeta.vNCtxTrain > 0) {
        vModelInfo["context_length"] = vMeta.vNCtxTrain;
      }

      // merge with loaded_info from the child process if available
      if (vMeta.mIsRunning()) {
        for (auto vIt = vMeta.vLoadedInfo.begin(); vIt != vMeta.vLoadedInfo.end(); ++vIt) {
          if (!vModelInfo.contains(vIt.mKey())) {
            vModelInfo[vIt.mKey()] = vIt.value();
          }
        }
      }
      vModelsJson.push_back(vModelInfo);
    }
    fResOk(vRes, {
                   { "data",   vModelsJson },
                   { "object", "list"      },
    });
    return vRes;
  };

  this->vPostRouterModelsUnload = [this](const server_http_req & pReq) {
    auto        vRes   = std::make_unique<server_http_res>();
    json        vBody  = json::mParse(pReq.vBody);
    std::string name   = fJsonValue(vBody, "model", std::string());
    auto        vModel = vModels.mGetMeta(name);
    if (!vModel.has_value()) {
      fResErr(vRes, fFormatErrorResponse("model is not found", cErrorTypeInvalidRequest));
      return vRes;
    }
    if (!vModel->mIsRunning() && vModel->vStatus != cServerModelStatusDownloading) {
      fResErr(vRes, fFormatErrorResponse("model is not running", cErrorTypeInvalidRequest));
      return vRes;
    }
    vModels.mUnload(vModel->name);
    fResOk(vRes, {
                   { "success", true }
    });
    return vRes;
  };

  this->vGetRouterModelsSse = [this](const server_http_req & pReq) {
    auto vRes          = std::make_unique<server_http_res>();
    vRes->vStatus      = 200;
    vRes->vContentType = "text/event-stream";
    auto vSseClient    = std::make_shared<server_models_sse_client>(vModels.vSse);
    vRes->vNext        = [this, vSseClient, &pReq](std::string & pOutput) -> bool {
      auto vResult =
        vSseClient->mNext([&]() { return vStopping.load(std::memory_order_relaxed) || pReq.vShouldStop(); });
      if (vResult == nullptr) {
        return false;  // client disconnected or should_stop
      }
      pOutput = "data: " + fSafeJsonToStr(vResult->to_json()) + "\n\n";
      return true;  // listen for the next event
    };
    return vRes;
  };

  this->vPostRouterModels = [this](const server_http_req & pReq) {
    auto vRes = std::make_unique<server_http_res>();

    json        vBody = json::mParse(pReq.vBody);
    std::string name  = fJsonValue(vBody, "model", std::string());
    if (name.empty()) {
      throw std::invalid_argument("model must be a non-empty string");
    }

    common_params vP;
    vP.vModel.vHfRepo = name;
    vP.vHfToken       = vParams.vHfToken;

    // validate by fetching metadata
    bool vOk = false;
    try {
      fCommonModelsHandlerInit(vP, cDgsExampleServer);
      vOk = true;
    } catch (...) {
      SRV_ERR("unknown error while validating model '%s'\n", name.c_str());
      // other exceptions will be handled by the outer ex_wrapper()
      throw;
    }

    if (!vOk) {
      throw std::invalid_argument("model validation failed, unable to download");
    }

    // reject if model already exists
    if (vModels.mHasModel(name)) {
      throw std::invalid_argument("model '" + name + "' already exists");
    }

    // then, proceed with the actual download
    SRV_INF("starting download for model '%s'\n", name.c_str());
    {
      server_models::load_options vLoadOpts;
      vLoadOpts.vMode                = cServerChildModeDownload;
      vLoadOpts.vCustomMeta          = server_model_meta{};
      vLoadOpts.vCustomMeta->vSource = cServerModelSourceCache;
      vLoadOpts.vCustomMeta->name    = name;
      vModels.mLoad(name, vLoadOpts);
    }

    fResOk(vRes, {
                   { "success", true }
    });
    return vRes;
  };

  this->vDelRouterModels = [this](const server_http_req & pReq) {
    auto vRes = std::make_unique<server_http_res>();

    std::string name = pReq.mGetParam("model");
    if (name.empty()) {
      throw std::invalid_argument("model must be a non-empty string");
    }

    vModels.mRemove(name);  // throws on error

    fResOk(vRes, {
                   { "success", true }
    });
    return vRes;
  };

  this->vRouterStreamGet = [this](const server_http_req & pReq) {
    // GET /v1/stream?conv_id=<id>&from=N. resolve the owning child from the conv_id -> model
    // map, 404 when nothing maps
    auto        vRes    = std::make_unique<server_http_res>();
    std::string vConvId = pReq.mGetParam("conv_id");
    if (vConvId.empty()) {
      fResErr(vRes, fFormatErrorResponse("Missing conversation id in path", cErrorTypeInvalidRequest));
      return vRes;
    }
    std::optional<server_model_meta> vOwner = fResolveChildForConv(vModels, vConvId);
    if (!vOwner.has_value()) {
      // a registered conv whose model is still loading earns a retry: the session appears
      // once the load ends and the pending request reaches the child
      auto vTracked   = vModels.vConvModels.mLookup(vConvId);
      auto vMeta      = vTracked.has_value() ? vModels.mGetMeta(*vTracked) : std::nullopt;
      bool vTransient = vMeta.has_value() && (vMeta->vStatus == cServerModelStatusLoading ||
                                              vMeta->vStatus == cServerModelStatusDownloading ||
                                              vMeta->vStatus == cServerModelStatusDownloaded);
      if (vTransient) {
        fResErr(vRes, fFormatErrorResponse("Stream owner model is loading, retry later", cErrorTypeUnavailable));
      } else {
        fResErr(vRes, fFormatErrorResponse("Stream not found or expired", cErrorTypeNotFound));
      }
      return vRes;
    }
    std::string vFrom      = pReq.mGetParam("from");
    std::string vChildPath = "/v1/stream?conv_id=" + fEncodeQs(vConvId);
    if (!vFrom.empty()) {
      vChildPath += "&from=" + vFrom;
    }
    SRV_TRC("proxying stream resume to model %s on port %d, path=%s\n", vOwner->name.c_str(), vOwner->vPort,
            vChildPath.c_str());
    auto vProxy = std::make_unique<server_http_proxy>("GET", "http", CHILD_ADDR, vOwner->vPort, "/api" + vChildPath,
                                                      pReq.dHeaders, pReq.vBody, pReq.dFiles, pReq.vShouldStop,
                                                      vParams.vTimeoutRead, vParams.vTimeoutWrite);
    return std::unique_ptr<server_http_res>(std::move(vProxy));
  };

  this->vRouterStreamsLookup = [this](const server_http_req & pReq) {
    // POST /v1/streams/lookup. resolve each requested conv id to its owning child via the
    // map, group the ids per child, and query only the children that actually own some of
    // them instead of fanning out to every ready child. a child only answers for the ids
    // it owns, never lists anything else
    auto                     vRes = std::make_unique<server_http_res>();
    std::vector<std::string> lRequested;
    try {
      json vBody = json::mParse(pReq.vBody);
      if (vBody.contains("conversation_ids") && vBody["conversation_ids"].is_array()) {
        for (const auto & vV : vBody["conversation_ids"]) {
          if (vV.is_string() && !vV.get<std::string>().empty()) {
            lRequested.push_back(vV.get<std::string>());
          }
        }
      }
    } catch (const std::exception &) {
      fResOk(vRes, json::mArray());
      return vRes;
    }

    // group requested ids by the child port that owns them, drop ids that map to nothing
    std::unordered_map<int, json> dPerChild;
    for (const auto & vCid : lRequested) {
      auto vOwner = fResolveChildForConv(vModels, vCid);
      if (!vOwner.has_value()) {
        continue;
      }
      dPerChild[vOwner->vPort].push_back(vCid);
    }

    json vAggregated = json::mArray();
    for (auto & [port, ids] : dPerChild) {
      json vChildBody = {
        { "conversation_ids", ids }
      };
      httplib::Client vCli(CHILD_ADDR, port);
      vCli.set_connection_timeout(0, cStreamLookupTimeoutMs * 1000);
      vCli.set_read_timeout(0, cStreamLookupTimeoutMs * 1000);
      vCli.set_write_timeout(0, cStreamLookupTimeoutMs * 1000);
      auto vResp = vCli.Post("/api/v1/streams/lookup", vChildBody.mDump(), "application/json");
      if (!vResp || vResp->status != 200) {
        continue;
      }
      try {
        json vChildArr = json::mParse(vResp->body);
        if (!vChildArr.is_array()) {
          continue;
        }
        for (auto & vEntry : vChildArr) {
          if (vEntry.is_object()) {
            vAggregated.push_back(vEntry);
          }
        }
      } catch (const std::exception &) {
        continue;
      }
    }
    fResOk(vRes, vAggregated);
    return vRes;
  };

  this->vRouterStreamDelete = [this](const server_http_req & pReq) {
    // DELETE /v1/stream?conv_id=<id>. resolve the owning child via the map and forward only to
    // it, evict_and_cancel is idempotent on the child
    auto        vRes    = std::make_unique<server_http_res>();
    std::string vConvId = pReq.mGetParam("conv_id");
    if (vConvId.empty()) {
      fResErr(vRes, fFormatErrorResponse("Missing conversation id in path", cErrorTypeInvalidRequest));
      return vRes;
    }
    std::string vChildPath = "/v1/stream?conv_id=" + fEncodeQs(vConvId);
    auto        vOwner     = fResolveChildForConv(vModels, vConvId);
    if (vOwner.has_value()) {
      httplib::Client vCli(CHILD_ADDR, vOwner->vPort);
      vCli.set_connection_timeout(0, cStreamLookupTimeoutMs * 1000);
      vCli.set_read_timeout(0, cStreamLookupTimeoutMs * 1000);
      vCli.set_write_timeout(0, cStreamLookupTimeoutMs * 1000);
      auto vResp = vCli.Delete(("/api" + vChildPath).c_str());
      (void) vResp;  // the child logs its own miss when the session is unknown there
    } else if (auto vTracked = vModels.vConvModels.mLookup(vConvId); vTracked.has_value()) {
      // the entry exists but its model is still loading: the forget below erases it,
      // which cancels the request parked in proxy_post before the generation starts
      SRV_INF("router stop for conv_id=%s while model name=%s is loading, cancelling the pending request\n",
              vConvId.c_str(), vTracked->c_str());
    } else {
      SRV_WRN("router stop for unknown conv_id=%s, no owning child in the conv map\n", vConvId.c_str());
    }
    // drop the tracking entry, the session is being torn down
    vModels.vConvModels.mForget(vConvId);
    vRes->vStatus      = 204;
    vRes->vContentType = "application/json";
    return vRes;
  };
}

//
// server_http_proxy
//

static std::string fToLowerCopy(const std::string & value) {
  std::string vLowered(value.size(), '\0');
  std::transform(value.begin(), value.end(), vLowered.begin(), [](unsigned char pC) { return std::tolower(pC); });
  return vLowered;
}

static bool fShouldStripProxyHeader(const std::string & pHeaderName) {
  // Headers that get duplicated when router forwards child responses
  if (pHeaderName == "server" || pHeaderName == "transfer-encoding" ||
      pHeaderName == "content-length" ||  // quick fix for doc/MANUAL.md
      pHeaderName == "keep-alive") {
    return true;
  }

  // Router injects CORS, child also sends them: duplicate
  if (pHeaderName.rfind("access-control-", 0) == 0) {
    return true;
  }

  return false;
}

static std::string fGenerateMultipartBoundary() {
  thread_local std::mt19937       vGen(std::random_device{}());
  static const char               aChars[] = "0123456789abcdefghijklmnopqrstuvwxyz";
  std::uniform_int_distribution<> vDis(0, sizeof(aChars) - 2);
  std::string                     vBoundary = "----dgs-cpp-proxy-";
  for (int vI = 0; vI < 16; vI++) {
    vBoundary += aChars[vDis(vGen)];
  }
  return vBoundary;
}

static std::string fBuildMultipartBody(const json &                                 pFormFields,
                                       const std::map<std::string, uploaded_file> & pFiles,
                                       const std::string &                          pBoundary) {
  static auto vSanitizeField = [](const std::string & pText) {
    std::string vResult;
    vResult.reserve(pText.size());
    for (char vC : pText) {
      if (vC != '\n' && vC != '\r' && vC != '"') {
        vResult += vC;
      }
    }
    return vResult;
  };

  std::ostringstream vBody;

  for (const auto & [key, value] : pFormFields.mItems()) {
    if (value.is_array()) {
      for (const auto & vItem : value) {
        vBody << "--" << pBoundary << "\r\n";
        vBody << "Content-Disposition: form-data; name=\"" << vSanitizeField(key) << "\"\r\n";
        vBody << "\r\n";
        if (!vItem.is_string()) {
          throw std::invalid_argument("expected string");
        }
        vBody << vItem.get<std::string>() << "\r\n";
      }
    } else {
      vBody << "--" << pBoundary << "\r\n";
      vBody << "Content-Disposition: form-data; name=\"" << vSanitizeField(key) << "\"\r\n";
      vBody << "\r\n";
      if (!value.is_string()) {
        throw std::invalid_argument("expected string");
      }
      vBody << value.get<std::string>() << "\r\n";
    }
  }

  for (const auto & [key, file] : pFiles) {
    vBody << "--" << pBoundary << "\r\n";
    vBody << "Content-Disposition: form-data; name=\"" << vSanitizeField(key) << "\"";
    if (!file.vFilename.empty()) {
      vBody << "; filename=\"" << vSanitizeField(file.vFilename) << "\"";
    }
    vBody << "\r\n";
    if (!file.vContentType.empty()) {
      vBody << "Content-Type: " << vSanitizeField(file.vContentType) << "\r\n";
    } else {
      vBody << "Content-Type: application/octet-stream\r\n";
    }
    vBody << "\r\n";
    vBody.write(reinterpret_cast<const char *>(file.data.data()), file.data.size());
    vBody << "\r\n";
  }

  vBody << "--" << pBoundary << "--\r\n";
  return vBody.str();
}

server_http_proxy::server_http_proxy(const std::string &                          pMethod,
                                     const std::string &                          pScheme,
                                     const std::string &                          pHost,
                                     int                                          pPort,
                                     const std::string &                          pPath,
                                     const std::map<std::string, std::string> &   pHeaders,
                                     const std::string &                          pBody,
                                     const std::map<std::string, uploaded_file> & pFiles,
                                     const std::function<bool()>                  pShouldStop,
                                     int32_t                                      pTimeoutRead,
                                     int32_t                                      pTimeoutWrite) {
  // shared between reader and writer threads
  auto vCli  = std::make_shared<httplib::ClientImpl>(pHost, pPort);
  auto vPipe = std::make_shared<server_pipe<msg_t>>();

  if (pScheme == "https") {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    vCli.reset(new httplib::SSLClient(pHost, pPort));
#else
    throw std::runtime_error("HTTPS requested but CPPHTTPLIB_OPENSSL_SUPPORT is not defined");
#endif
  }

  // setup Client
  vCli->set_follow_location(true);
  vCli->set_connection_timeout(pTimeoutRead, 0);  // use --timeout value instead of hardcoded 5 s
  vCli->set_write_timeout(pTimeoutRead, 0);       // reversed for cli (client) vs srv (server)
  vCli->set_read_timeout(pTimeoutWrite, 0);
  this->vStatus       = 500;                      // to be overwritten upon response
  this->vCleanupPipes = [vPipe]() {
    vPipe->mCloseRead();
    vPipe->mCloseWrite();
  };

  // wire up the receive end of the pipe
  this->vNext = [vPipe, pShouldStop](std::string & pOut) -> bool {
    msg_t vMsg;
    bool  vHasNext = vPipe->read(vMsg, pShouldStop);
    if (!vMsg.data.empty()) {
      pOut = std::move(vMsg.data);
    }
    return vHasNext;  // false if EOF or pipe broken
  };

  // build the header message forwarded to the reader thread, stripping internal proxy headers
  auto vMakeHeaderMsg = [](const httplib::Response & pResponse) {
    msg_t vMsg;
    vMsg.vStatus = pResponse.status;
    for (const auto & [key, value] : pResponse.headers) {
      const auto cLowered = fToLowerCopy(key);
      if (fShouldStripProxyHeader(cLowered)) {
        continue;
      }
      if (cLowered == "content-type") {
        vMsg.vContentType = value;
        continue;
      }
      vMsg.dHeaders[key] = value;
    }
    return vMsg;
  };

  // true once response_handler has already forwarded the headers
  auto vHeadersSent = std::make_shared<std::atomic<bool>>(false);

  // wire up the HTTP client
  // note: do NOT capture `this` pointer, as it may be destroyed before the thread ends
  httplib::ResponseHandler vResponseHandler = [vPipe, vHeadersSent,
                                               vMakeHeaderMsg](const httplib::Response & pResponse) {
    vHeadersSent->store(true);
    return vPipe->write(vMakeHeaderMsg(pResponse));  // send headers first
  };
  httplib::ContentReceiverWithProgress vContentReceiver = [vPipe](const char * data, size_t pDataLength, size_t,
                                                                  size_t) {
    // send data chunks
    // returns false if pipe is closed / broken (signal to stop receiving)
    return vPipe->write({ {}, 0, std::string(data, pDataLength), "" });
  };

  // when files are present, the body was converted from multipart form data to JSON
  // we need to reconstruct the multipart body for the downstream server
  std::string vEffectiveBody = pBody;
  std::string vOverrideContentType;
  bool        vHasFiles = !pFiles.empty();

  if (vHasFiles) {
    json vFormFields = json::mParseNoThrow(pBody);
    if (!vFormFields.mIsDiscarded()) {
      auto vBoundary       = fGenerateMultipartBoundary();
      vEffectiveBody       = fBuildMultipartBody(vFormFields, pFiles, vBoundary);
      vOverrideContentType = "multipart/form-data; boundary=" + vBoundary;
    } else {
      throw std::runtime_error("failed to parse multipart form fields JSON");
    }
  }

  // prepare the request to destination server
  httplib::Request vReq;
  {
    vReq.method = pMethod;
    vReq.path   = pPath;
    for (const auto & [key, value] : pHeaders) {
      const auto cLowered = fToLowerCopy(key);
      if (cLowered == "accept-encoding") {
        // disable Accept-Encoding to avoid compressed responses
        continue;
      }
      if (cLowered == "transfer-encoding") {
        // the body is already decoded
        continue;
      }
      if (cLowered == "content-length") {
        // let httplib calculate Content-Length from the actual body
        continue;
      }
      if (cLowered == "content-type") {
        if (vHasFiles) {
          // we set our own Content-Type with the new boundary
          continue;
        }
        // when no files but the original request was multipart,
        // the body is now JSON, so correct the Content-Type
        if (value.find("multipart/form-data") != std::string::npos) {
          vOverrideContentType = "application/json; charset=utf-8";
          continue;
        }
      }
      if (cLowered == "host") {
        bool              vIsDefaultPort = (pScheme == "https" && pPort == 443) || (pScheme == "http" && pPort == 80);
        const std::string cUrlHost       = fCommonHttpFormatHost(pHost);
        vReq.set_header(key, vIsDefaultPort ? cUrlHost : cUrlHost + ":" + std::to_string(pPort));
      } else {
        vReq.set_header(key, value);
      }
    }
    vReq.body = vEffectiveBody;
    if (!vOverrideContentType.empty()) {
      vReq.set_header("Content-Type", vOverrideContentType);
    }
    vReq.response_handler = vResponseHandler;
    vReq.content_receiver = vContentReceiver;
  }

  // start the proxy thread
  SRV_DBG("start proxy thread %s %s\n", vReq.method.c_str(), vReq.path.c_str());
  this->vThread = std::thread([vCli, vPipe, vReq, vHeadersSent, vMakeHeaderMsg]() {
    auto vResult = vCli->send(std::move(vReq));
    if (vResult.error() != httplib::Error::Success) {
      auto vErrStr = httplib::to_string(vResult.error());
      SRV_ERR("http client error: %s\n", vErrStr.c_str());
      vPipe->write({ {}, 500, "", "" });                       // header
      vPipe->write({ {}, 0, "proxy error: " + vErrStr, "" });  // body
    } else if (!vHeadersSent->load()) {
      // httplib skips response_handler for bodyless statuses like 204, send headers here instead
      vPipe->write(vMakeHeaderMsg(*vResult));
    }
    vPipe->mCloseWrite();  // signal EOF to reader
    SRV_DBG("%s", "client request thread ended\n");
  });
  this->vThread.detach();

  // wait for the first chunk (headers)
  {
    msg_t vHeader;
    if (vPipe->read(vHeader, pShouldStop)) {
      SRV_DBG("%s", "received response headers\n");
      this->vStatus  = vHeader.vStatus;
      this->dHeaders = std::move(vHeader.dHeaders);
      if (!vHeader.vContentType.empty()) {
        this->vContentType = std::move(vHeader.vContentType);
      }
    } else {
      SRV_DBG("%s", "no response headers received (request cancelled?)\n");
    }
  }
}
