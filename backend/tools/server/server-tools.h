#pragma once

#include "server-common.h"
#include "server-http.h"
#include "server-mcp.h"
#include "server-queue.h"

#include <atomic>
#include <functional>
#include <memory>

struct server_tool {
  std::string name;
  std::string vDisplayName;
  bool        vPermissionWrite = false;
  bool        vSupportStream   = false;  // if true, output can be streamed
  bool        vUsesCwd         = false;  // if true, the tool resolves paths and runs against the working directory

  virtual ~server_tool()              = default;
  virtual json mGetDefinition() const = 0;

  virtual std::string type() const { return "server"; }

  struct stream {
    server_response &     vQr;
    int                   vId;
    std::function<bool()> vAlive;
    void                  mPush(const std::string & pChunk);
  };

  virtual json mInvoke(json pParams, stream * pSt = nullptr) const = 0;

  json to_json() const;
};

struct server_tools_runtime;  // impl detail, defined in server-tools.cpp

struct server_tools {
  std::vector<std::unique_ptr<server_tool>> lTools;

  // for streaming
  server_response  vQueueRes;
  std::atomic<int> vResId{ 0 };

  // set when --tools-runtime is configured; routes every tool call through an isolate
  std::unique_ptr<server_tools_runtime> vRuntime;

  void mSetup(const std::vector<std::string> & pEnabledTools, server_mcp & pMcpMgr, const std::string & pToolsRuntime);

  server_http_context::handler_t vHandleGet;
  server_http_context::handler_t vHandlePost;

  server_tools();
  ~server_tools();
};
