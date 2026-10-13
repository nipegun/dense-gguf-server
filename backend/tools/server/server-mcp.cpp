#include "server-mcp.h"

#include "common.h"
#include "subproc.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <functional>
#include <sstream>
#include <thread>

#if defined(_WIN32)
#  include <io.h>
#  include <windows.h>
#else
#  include <errno.h>
#  include <fcntl.h>
#  include <poll.h>
#  include <unistd.h>
extern char ** environ;
#endif

// read NDJSON lines from a child pipe, calling on_line per line until `running` clears, EOF/error, or on_line returns false.
// polled, not blocking: a grandchild can inherit the pipe's write end and hold it open (terminate() kills only the direct child), so a blocking read would hang teardown on an EOF that never comes.
static void fMcpPumpNdjson(FILE *                                      pF,
                           std::atomic<bool> &                         pRunning,
                           const std::function<bool(std::string &&)> & pOnLine) {
  if (!pF) {
    return;
  }
  const int    cPollMs = 50;
  const size_t cMaxLine =
    8 * 1024 *
    1024;  // drop any single NDJSON line larger than this, so a child that never emits '\n' can't grow buf without bound
#if defined(_WIN32)
  HANDLE h = (HANDLE) _get_osfhandle(_fileno(f));
#else
  int vFd = fileno(pF);
  int vFl = fcntl(vFd, F_GETFL, 0);
  if (vFl >= 0) {
    fcntl(vFd, F_SETFL, vFl | O_NONBLOCK);
  }
#endif
  std::string vBuf;
  bool        vSkipping = false;  // discarding an over-long line until its terminating newline
  char        aChunk[4096];
  while (pRunning.load()) {
    size_t n = 0;
#if defined(_WIN32)
    DWORD avail = 0;
    if (!PeekNamedPipe(h, NULL, 0, NULL, &avail, NULL)) {
      break;  // pipe broken / child gone
    }
    if (avail == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
      continue;
    }
    DWORD to_read = avail < (DWORD) sizeof(chunk) ? avail : (DWORD) sizeof(chunk);
    DWORD got     = 0;
    if (!ReadFile(h, chunk, to_read, &got, NULL) || got == 0) {
      break;
    }
    n = (size_t) got;
#else
    struct pollfd vPfd;
    vPfd.fd      = vFd;
    vPfd.events  = POLLIN;
    vPfd.revents = 0;
    int vPr      = poll(&vPfd, 1, cPollMs);
    if (vPr < 0) {
      if (errno == EINTR) {
        continue;
      }
      break;
    }
    if (vPr == 0) {
      continue;  // timeout -> re-check running
    }
    if (vPfd.revents & (POLLERR | POLLNVAL)) {
      break;
    }
    ssize_t vR = read(vFd, aChunk, sizeof(aChunk));
    if (vR < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
        continue;
      }
      break;
    }
    if (vR == 0) {
      break;  // EOF: child (and any pipe writers) closed the stream
    }
    n = (size_t) vR;
#endif
    vBuf.append(aChunk, n);

    // resync after an over-long, unterminated line: discard bytes until the next newline
    if (vSkipping) {
      size_t vNl = vBuf.find('\n');
      if (vNl == std::string::npos) {
        if (vBuf.size() > cMaxLine) {
          vBuf.clear();  // stay bounded while waiting for a terminator
        }
        continue;
      }
      vBuf.erase(0, vNl + 1);
      vSkipping = false;
    }

    size_t vPos;
    while ((vPos = vBuf.find('\n')) != std::string::npos) {
      std::string vLine = vBuf.substr(0, vPos);
      vBuf.erase(0, vPos + 1);
      if (!vLine.empty() && vLine.back() == '\r') {
        vLine.pop_back();
      }
      if (vLine.empty()) {
        continue;
      }
      if (!pOnLine(std::move(vLine))) {
        return;
      }
    }

    // a partial line already larger than the cap and still no newline: drop it to avoid unbounded growth
    if (vBuf.size() > cMaxLine) {
      SRV_WRN("MCP: dropping oversized line (> %zu bytes) from child pipe\n", cMaxLine);
      vBuf.clear();
      vSkipping = true;
    }
  }
}

//
// server_mcp_server_config
//

std::vector<server_mcp_server_config> server_mcp_server_config::mParseFromJson(const std::string & pJsonStr) {
  return mParseCursorFormat(json::mParse(pJsonStr));
}

std::vector<server_mcp_server_config> server_mcp_server_config::mParseCursorFormat(const json & pJ) {
  std::vector<server_mcp_server_config> lResult;

  if (!pJ.contains("mcpServers") || !pJ.at("mcpServers").is_object()) {
    return lResult;
  }

  for (const auto & [name, cfg] : pJ.at("mcpServers").mItems()) {
    server_mcp_server_config vSc;
    vSc.name       = name;
    vSc.vCommand   = cfg.value("command", std::string());
    vSc.vCwd       = cfg.value("cwd", std::string());
    vSc.vTimeoutMs = cfg.value("timeout_ms", vSc.vTimeoutMs);

    if (cfg.contains("args") && cfg.at("args").is_array()) {
      for (const auto & vA : cfg.at("args")) {
        vSc.lArgs.push_back(vA.get<std::string>());
      }
    }
    if (cfg.contains("env") && cfg.at("env").is_object()) {
      for (const auto & [k, v] : cfg.at("env").mItems()) {
        vSc.dEnv[k] = v.get<std::string>();
      }
    }

    if (vSc.vCommand.empty()) {
      SRV_WRN("MCP server '%s' has no command, skipping\n", name.c_str());
      continue;
    }
    lResult.push_back(std::move(vSc));
  }

  return lResult;
}

//
// server_mcp_transport
//

static constexpr const char * cMcpProtocolVersion = "2024-11-05";

static std::string fRpcErrorMessage(const json & pResp) {
  if (pResp.contains("error")) {
    const json & vE = pResp.at("error");
    if (vE.is_object()) {
      return vE.value("message", "unknown error");
    }
    if (vE.is_string()) {
      return vE.get<std::string>();
    }
  }
  return "unknown error";
}

// normalize an MCP tools/call result to the /tools contract (see README-dev.md):
// concat text parts of result.content[], and surface an isError result
static json fMcpResultToResponse(const json & pResult) {
  std::string vText;
  if (pResult.contains("content") && pResult.at("content").is_array()) {
    for (const auto & vPart : pResult.at("content")) {
      if (vPart.is_object() && vPart.value("type", "") == "text") {
        if (!vText.empty()) {
          vText += "\n";
        }
        vText += vPart.value("text", "");
      }
    }
  }
  if (pResult.is_object() && pResult.value("isError", false)) {
    return {
      { "error", vText.empty() ? "MCP tool returned an error" : vText }
    };
  }
  return {
    { "plain_text_response", vText }
  };
}

json server_mcp_transport::mSendRpc(const json & pRequest, const std::function<bool()> & pShouldStop) {
  if (!vToServer.write(pRequest.mDump())) {
    return {
      { "error", { { "code", -32603 }, { "message", "transport closed" } } }
    };
  }

  const bool cHasId    = pRequest.contains("id");
  const auto cDeadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(vTimeoutMs);
  auto       vStop     = [&]() {
    return (pShouldStop && pShouldStop()) || std::chrono::steady_clock::now() >= cDeadline;
  };

  std::string vFrame;
  while (vFromServer.read(vFrame, vStop, false)) {
    json vReply;
    try {
      vReply = json::mParse(vFrame);
    } catch (...) {
      if (std::chrono::steady_clock::now() >= cDeadline) {
        break;
      }
      continue;  // skip malformed frame
    }
    // no id: a notification. mismatched id: a stale reply from a timed-out request (ids are monotonic, never a future one)
    if (!cHasId || (vReply.contains("id") && vReply.at("id") == pRequest.at("id"))) {
      return vReply;
    }
    if (std::chrono::steady_clock::now() >= cDeadline) {
      break;  // a flood of notifications must not outrun the deadline
    }
  }

  if (pShouldStop && pShouldStop()) {
    return {
      { "error", { { "code", -32603 }, { "message", "cancelled" } } }
    };
  }
  if (std::chrono::steady_clock::now() >= cDeadline) {
    return {
      { "error", { { "code", -32603 }, { "message", "request timed out" } } }
    };
  }
  return {
    { "error", { { "code", -32603 }, { "message", "transport closed" } } }
  };
}

bool server_mcp_transport::mEnsureInit(const std::function<bool()> & pShouldStop) {
  if (vInitialized) {
    return true;
  }

  json vInitReq = {
    { "jsonrpc", "2.0"        },
    { "id",      vNextId++    },
    { "method",  "initialize" },
    { "params",
     {
        { "protocolVersion", cMcpProtocolVersion },
        { "capabilities", json::mObject() },
        { "clientInfo", { { "name", "dgs.cpp" }, { "version", "1.0" } } },
      }                       },
  };
  json vResp = mSendRpc(vInitReq, pShouldStop);
  if (!vResp.contains("result")) {
    vLastError = "initialize failed: " + fRpcErrorMessage(vResp);
    return false;
  }

  // notifications/initialized: no id, no reply expected
  json vNotif = {
    { "jsonrpc", "2.0"                       },
    { "method",  "notifications/initialized" }
  };
  vToServer.write(vNotif.mDump());

  vInitialized = true;
  return true;
}

std::vector<server_mcp_tool_def> server_mcp_transport::mListTools(const std::function<bool()> & pShouldStop) {
  std::lock_guard<std::mutex> vLock(vRpcMutex);
  if (!mEnsureInit(pShouldStop)) {
    return {};
  }
  if (!lTools.empty()) {
    return lTools;
  }

  json vReq = {
    { "jsonrpc", "2.0"        },
    { "id",      vNextId++    },
    { "method",  "tools/list" }
  };
  json vResp = mSendRpc(vReq, pShouldStop);
  if (!vResp.contains("result")) {
    vLastError = "tools/list failed: " + fRpcErrorMessage(vResp);
    return {};
  }

  const json & vResult = vResp.at("result");
  if (vResult.contains("tools") && vResult.at("tools").is_array()) {
    for (const auto & vT : vResult.at("tools")) {
      server_mcp_tool_def vDef;
      vDef.vServerName  = name;
      vDef.name         = vT.value("name", "");
      vDef.vDescription = vT.value("description", "");
      if (vT.contains("inputSchema")) {
        vDef.vInputSchema = vT.at("inputSchema");
      }
      lTools.push_back(std::move(vDef));
    }
  }
  return lTools;
}

json server_mcp_transport::mCallTool(const std::string &           pToolName,
                                     const json &                  pArguments,
                                     const std::function<bool()> & pShouldStop) {
  std::lock_guard<std::mutex> vLock(vRpcMutex);
  if (!mEnsureInit(pShouldStop)) {
    return {
      { "error", vLastError }
    };
  }

  json vReq = {
    { "jsonrpc", "2.0"                                                  },
    { "id",      vNextId++                                              },
    { "method",  "tools/call"                                           },
    { "params",  { { "name", pToolName }, { "arguments", pArguments } } },
  };
  json vResp = mSendRpc(vReq, pShouldStop);
  if (vResp.contains("error")) {
    return {
      { "error", fRpcErrorMessage(vResp) }
    };
  }
  if (vResp.contains("result")) {
    return fMcpResultToResponse(vResp.at("result"));
  }
  return {
    { "error", "invalid response from MCP server" }
  };
}

//
// server_mcp_stdio
//

struct server_mcp_stdio::process_handle {
  common_subproc vSp;
  FILE *         vIn  = nullptr;  // child stdin
  FILE *         vOut = nullptr;  // child stdout
  FILE *         vErr = nullptr;  // child stderr
};

#if defined(_WIN32)
// config strings are UTF-8 (from JSON) and subprocess.h converts them with CP_UTF8, so inputs must be UTF-8, not the active code page
static std::string wide_to_utf8(const wchar_t * s, int len /* -1 for NUL-terminated */) {
  return wstring_to_utf8(len == -1 ? std::wstring(s) : std::wstring(s, s + len));
}
#endif

static std::string fMcpResolveCommand(const std::string & pCommand) {
#if defined(_WIN32)
  // For Windows: make sure we handle ".exe" correctly, as well as UTF-8
  std::wstring wcmd = utf8_to_wstring(command);
  wchar_t      buf[MAX_PATH * 4];
  const DWORD  cap = (DWORD) (sizeof(buf) / sizeof(buf[0]));

  auto search = [&](const wchar_t * ext) -> std::string {
    DWORD n = SearchPathW(NULL, wcmd.c_str(), ext, cap, buf, NULL);
    return (n > 0 && n < cap) ? wide_to_utf8(buf, (int) n) : std::string();
  };

  std::string found = search(NULL);  // exact path / already-extensioned / .exe on PATH
  if (!found.empty()) {
    return found;
  }

  std::wstring pathext;
  DWORD        need = GetEnvironmentVariableW(L"PATHEXT", NULL, 0);
  if (need > 0) {
    pathext.resize(need);
    DWORD got = GetEnvironmentVariableW(L"PATHEXT", &pathext[0], need);
    pathext.resize(got);
  }
  if (pathext.empty()) {
    pathext = L".COM;.EXE;.BAT;.CMD";
  }
  for (size_t start = 0; start <= pathext.size();) {
    size_t       sep = pathext.find(L';', start);
    std::wstring ext = pathext.substr(start, sep == std::wstring::npos ? std::wstring::npos : sep - start);
    if (!ext.empty()) {
      found = search(ext.c_str());
      if (!found.empty()) {
        return found;
      }
    }
    if (sep == std::wstring::npos) {
      break;
    }
    start = sep + 1;
  }
  return command;  // give up and let subprocess.h report the spawn error
#else
  return pCommand;
#endif             // _WIN32
}

static std::vector<std::string> fMcpParentEnv() {
  std::vector<std::string> lEnv;
#if defined(_WIN32)
  LPWCH block = GetEnvironmentStringsW();
  if (block) {
    for (LPWCH e = block; *e; e += wcslen(e) + 1) {
      env.emplace_back(wide_to_utf8(e, -1));
    }
    FreeEnvironmentStringsW(block);
  }
#else
  if (environ) {
    for (char ** vE = environ; *vE; ++vE) {
      lEnv.emplace_back(*vE);
    }
  }
#endif
  return lEnv;
}

// parent env with the config overrides applied, in "KEY=VALUE" form
static std::vector<std::string> fMcpBuildEnv(const std::map<std::string, std::string> & pOverrides) {
  std::vector<std::string> lEnv;
  for (auto & vE : fMcpParentEnv()) {
    size_t      vEq  = vE.find('=');
    std::string vKey = vEq == std::string::npos ? vE : vE.substr(0, vEq);
    if (pOverrides.find(vKey) == pOverrides.end()) {
      lEnv.push_back(vE);
    }
  }
  for (auto & [k, v] : pOverrides) {
    lEnv.push_back(k + "=" + v);
  }
  return lEnv;
}

server_mcp_stdio::server_mcp_stdio(const server_mcp_server_config & pConfig) : vConfig(pConfig) {
  name                 = pConfig.name;
  vTimeoutMs           = pConfig.vTimeoutMs;
  // bound the reply queue: send_rpc only drains during a call, so unsolicited notifications would otherwise grow it without limit
  vFromServer.vMaxSize = 65536;
}

server_mcp_stdio::~server_mcp_stdio() {
  mJoinPumps();
}

bool server_mcp_stdio::mStart() {
  std::vector<std::string> lArgvS;
  lArgvS.push_back(fMcpResolveCommand(vConfig.vCommand));
  lArgvS.insert(lArgvS.end(), vConfig.lArgs.begin(), vConfig.lArgs.end());

  int                      vOptions = subprocess_option_no_window | subprocess_option_search_user_path;
  std::vector<std::string> lEnvpS;
  if (vConfig.dEnv.empty()) {
    vOptions |= subprocess_option_inherit_environment;
  } else {
    lEnvpS = fMcpBuildEnv(vConfig.dEnv);
  }

  auto vHandle = std::make_unique<process_handle>();
  bool vOk     = vHandle->vSp.mCreate(lArgvS, vOptions, lEnvpS, vConfig.vCwd.empty() ? nullptr : vConfig.vCwd.c_str());
  if (!vOk) {
    SRV_WRN("MCP '%s': failed to spawn '%s'\n", vConfig.name.c_str(), vConfig.vCommand.c_str());
    return false;
  }
  vHandle->vIn  = vHandle->vSp.mStdinFile();
  vHandle->vOut = vHandle->vSp.mStdoutFile();
  vHandle->vErr = vHandle->vSp.mStderrFile();

  vProc = std::move(vHandle);
  vRunning.store(true);
  vReader = std::thread([this] { mReaderLoop(); });
  vWriter = std::thread([this] { mWriterLoop(); });
  vErrlog = std::thread([this] { mErrlogLoop(); });
  return true;
}

void server_mcp_stdio::mClose() {
  mJoinPumps();
}

bool server_mcp_stdio::mIsAlive() const {
  return vRunning.load();
}

std::string server_mcp_stdio::mDiagnostics() {
  std::string vOut;
  {
    std::lock_guard<std::mutex> vLock(vRpcMutex);  // last_error is written by send_rpc's callers
    vOut = vLastError;
  }
  std::lock_guard<std::mutex> vLk(vErrMu);
  if (!vErrTail.empty()) {
    if (!vOut.empty()) {
      vOut += "; ";
    }
    vOut += "last stderr: " + vErrTail;
  }
  return vOut;
}

void server_mcp_stdio::mReaderLoop() {
  fMcpPumpNdjson(vProc->vOut, vRunning, [this](std::string && pLine) {
    return vFromServer.write(std::move(pLine));  // false => consumer gone, stop
  });
  vRunning.store(false);
  vToServer.mCloseWrite();    // stop the writer
  vFromServer.mCloseWrite();  // EOF to any waiting caller
}

// write all of `data` to child stdin, non-blocking and polled so teardown never hangs (a grandchild can hold the read end of a full pipe open). returns false on error/close/shutdown.
static bool fMcpWriteAll(FILE * pF, const std::string & data, std::atomic<bool> & pRunning) {
  if (!pF) {
    return false;
  }
  size_t vTotal = 0;
#if defined(_WIN32)
  HANDLE h      = (HANDLE) _get_osfhandle(_fileno(f));
  DWORD  nowait = PIPE_NOWAIT;
  SetNamedPipeHandleState(h, &nowait, NULL, NULL);
  while (total < data.size() && running.load()) {
    DWORD written = 0;
    BOOL  ok      = WriteFile(h, data.data() + total, (DWORD) (data.size() - total), &written, NULL);
    if (ok && written > 0) {
      total += written;
      continue;
    }
    if (!ok) {
      DWORD err = GetLastError();
      if (err != ERROR_NO_DATA && err != ERROR_PIPE_BUSY) {
        return false;
      }
    }
    // backpressure (pipe full) is rare for small JSON-RPC frames; sleep rather than spin.
    // no writable-wait exists for a PIPE_NOWAIT anonymous pipe, so this polls like the POSIX poll() path.
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
#else
  int vFd = fileno(pF);
  int vFl = fcntl(vFd, F_GETFL, 0);
  if (vFl >= 0) {
    fcntl(vFd, F_SETFL, vFl | O_NONBLOCK);
  }
  while (vTotal < data.size() && pRunning.load()) {
    ssize_t n = write(vFd, data.data() + vTotal, data.size() - vTotal);
    if (n > 0) {
      vTotal += (size_t) n;
      continue;
    }
    if (n == 0) {
      return false;
    }
    if (errno == EINTR) {
      continue;
    }
    if (errno != EAGAIN && errno != EWOULDBLOCK) {
      return false;
    }
    struct pollfd vPfd;
    vPfd.fd      = vFd;
    vPfd.events  = POLLOUT;
    vPfd.revents = 0;
    int vPr      = poll(&vPfd, 1, 50);
    if (vPr < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    if (vPfd.revents & (POLLERR | POLLNVAL | POLLHUP)) {
      return false;
    }
  }
#endif
  return vTotal == data.size();
}

void server_mcp_stdio::mWriterLoop() {
  auto vShouldStop = [this] {
    return !vRunning.load();
  };
  std::string vMsg;
  while (vToServer.read(vMsg, vShouldStop)) {
    vMsg.push_back('\n');
    if (!fMcpWriteAll(vProc->vIn, vMsg, vRunning)) {
      break;  // child gone or shutting down
    }
  }
  vRunning.store(false);
  vToServer.mCloseRead();     // fail fast on any further send_rpc write
  vFromServer.mCloseWrite();  // wake any caller waiting for a reply
}

void server_mcp_stdio::mErrlogLoop() {
  static constexpr size_t cErrTailMax = 4096;
  // drain stderr (an undrained pipe blocks the child):
  // log it, and keep a bounded tail for reporting when the server dies
  fMcpPumpNdjson(vProc->vErr, vRunning, [this](std::string && pLine) {
    SRV_DBG("MCP '%s' stderr: %s\n", name.c_str(), pLine.c_str());
    std::lock_guard<std::mutex> vLk(vErrMu);
    vErrTail += pLine;
    vErrTail += '\n';
    if (vErrTail.size() > cErrTailMax) {
      vErrTail.erase(0, vErrTail.size() - cErrTailMax);
    }
    return true;
  });
}

void server_mcp_stdio::mJoinPumps() {
  if (!vProc) {
    return;
  }
  vRunning.store(false);
  vToServer.mCloseWrite();    // wake the writer if it waits for a message
  vFromServer.mCloseWrite();  // wake any caller waiting for a reply

  vProc->vSp.mTerminate();    // child death unblocks the blocked fread/fwrite

  if (vWriter.joinable()) {
    vWriter.join();
  }
  if (vReader.joinable()) {
    vReader.join();
  }
  if (vErrlog.joinable()) {
    vErrlog.join();
  }

  vProc->vSp.mJoin();  // reap the child: never waiting would leave the pid a zombie for the process lifetime
  vProc.reset();
}

//
// server_mcp
//

static constexpr int cMcpCooldownSeconds      = 5;
static constexpr int cMcpWarmupTimeoutSeconds = 10;  // cap per-server tool discovery at startup

server_mcp::~server_mcp() {
  mShutdown();

  std::vector<std::shared_ptr<server_mcp_transport>> lToClose;
  {
    std::lock_guard<std::mutex> vLock(vMutex);
    for (auto & [name, t] : dTransports) {
      lToClose.push_back(std::move(t));
    }
    dTransports.clear();
  }
  for (auto & vT : lToClose) {
    vT->mClose();
  }
}

std::shared_ptr<server_mcp_transport> server_mcp::mCreateTransport(const server_mcp_server_config & pCfg) {
  return std::make_shared<server_mcp_stdio>(pCfg);
}

void server_mcp::mShutdown() {
  vStopping.store(true);
}

const server_mcp_server_config * server_mcp::mFindConfig(const std::string & name) const {
  for (const auto & vC : lConfigs) {
    if (vC.name == name) {
      return &vC;
    }
  }
  return nullptr;
}

void server_mcp::mStart(const common_params & pParams) {
  auto vAppend = [this](const std::string & pJsonStr) {
    try {
      auto lParsed = server_mcp_server_config::mParseFromJson(pJsonStr);
      if (lParsed.empty()) {
        SRV_WRN("%s", "MCP config: no servers found in JSON\n");
      }
      for (auto & vP : lParsed) {
        // names must be unique across both config sources: get_or_create / find_config key on the name
        if (mFindConfig(vP.name)) {
          SRV_WRN("MCP config: duplicate server name '%s', skipping\n", vP.name.c_str());
          continue;
        }
        lConfigs.push_back(std::move(vP));
      }
    } catch (const std::exception & vE) {
      throw std::runtime_error(std::string("failed to parse MCP config JSON: ") + vE.what());
    }
  };
  if (!pParams.vMcpServersConfig.empty()) {
    std::ifstream vF(std::filesystem::u8path(pParams.vMcpServersConfig), std::ios::in);
    if (!vF) {
      throw std::runtime_error("failed to open MCP config file: " + pParams.vMcpServersConfig);
    }
    std::stringstream vSs;
    vSs << vF.rdbuf();
    vAppend(vSs.str());
  }
  if (!pParams.vMcpServersJson.empty()) {
    vAppend(pParams.vMcpServersJson);
  }

  if (lConfigs.empty()) {
    return;
  }

  std::vector<server_mcp_tool_def> lDiscovered;
  for (const auto & vCfg : lConfigs) {
    auto vT = mCreateTransport(vCfg);
    if (!vT->mStart()) {
      SRV_WRN("MCP warmup: failed to spawn '%s': %s\n", vCfg.name.c_str(), vT->mDiagnostics().c_str());
      continue;
    }
    // bound warmup per server so an unresponsive one can't stall startup for the full per-call timeout
    const auto cDeadline   = std::chrono::steady_clock::now() + std::chrono::seconds(cMcpWarmupTimeoutSeconds);
    auto       vShouldStop = [this, cDeadline]() {
      return vStopping.load() || std::chrono::steady_clock::now() >= cDeadline;
    };
    auto lTools = vT->mListTools(vShouldStop);
    SRV_INF("MCP warmup: '%s' discovered %zu tools\n", vCfg.name.c_str(), lTools.size());
    lDiscovered.insert(lDiscovered.end(), lTools.begin(), lTools.end());
    vT->mClose();
  }

  std::lock_guard<std::mutex> vLock(vMutex);
  lRegistry.swap(lDiscovered);
}

std::vector<server_mcp_tool_def> server_mcp::mListTools() const {
  std::lock_guard<std::mutex> vLock(vMutex);
  return lRegistry;
}

json server_mcp::mCallTool(const std::string &           pServerName,
                           const std::string &           pToolName,
                           const json &                  pArguments,
                           const std::function<bool()> & pShouldStop) {
  auto vTransport = mGetOrCreate(pServerName);
  if (!vTransport) {
    return {
      { "error", "MCP server unavailable: " + pServerName }
    };
  }

  auto vStop = [this, &pShouldStop]() {
    return vStopping.load() || (pShouldStop && pShouldStop());
  };
  return vTransport->mCallTool(pToolName, pArguments, vStop);
}

std::shared_ptr<server_mcp_transport> server_mcp::mGetOrCreate(const std::string & name) {
  std::vector<std::shared_ptr<server_mcp_transport>> lToClose;  // closed after unlock
  std::shared_ptr<server_mcp_transport>              vResult;

  {
    std::lock_guard<std::mutex> vLock(vMutex);
    if (vStopping.load()) {
      return nullptr;
    }

    auto vNow    = std::chrono::steady_clock::now();
    auto vDeadIt = dDeadServers.find(name);
    if (vDeadIt != dDeadServers.end()) {
      if (vNow < vDeadIt->second) {
        return nullptr;
      }
      dDeadServers.erase(vDeadIt);
    }

    auto vIt = dTransports.find(name);
    if (vIt != dTransports.end()) {
      if (vIt->second->mIsAlive()) {
        return vIt->second;
      }
      SRV_WRN("MCP '%s' is no longer alive: %s\n", name.c_str(), vIt->second->mDiagnostics().c_str());
      lToClose.push_back(std::move(vIt->second));
      dTransports.erase(vIt);
    }

    const server_mcp_server_config * vCfg = mFindConfig(name);
    if (vCfg) {
      auto vFresh = mCreateTransport(*vCfg);
      if (vFresh->mStart() && vFresh->mIsAlive()) {
        dTransports[name] = vFresh;
        vResult           = vFresh;
      } else {
        SRV_WRN("MCP '%s': failed to start: %s\n", name.c_str(), vFresh->mDiagnostics().c_str());
        lToClose.push_back(std::move(vFresh));
        dDeadServers[name] = vNow + std::chrono::seconds(cMcpCooldownSeconds);
      }
    }
  }

  for (auto & vT : lToClose) {
    vT->mClose();  // blocking call, no leaks
  }

  return vResult;
}
