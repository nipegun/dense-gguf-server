#include "server-http.h"

#include "common.h"
#include "http.h"
#include "server-api-doc.h"
#include "server-common.h"
#include "server-ui-settings.h"
#include "ui.h"

#include <cpp-httplib/httplib.h>

#include <functional>
#include <future>
#include <memory>
#include <string>
#include <thread>

//
// HTTP implementation using cpp-httplib
//

class server_http_context::Impl {
public:
  std::vector<std::unique_ptr<httplib::Server>> lServers;
  std::vector<std::string>                      lHosts;
  std::vector<std::thread>                      lThreads;  // one thread per listener
  std::unique_ptr<httplib::ThreadPool>          vPool;     // single pool shared among all listeners
  int                                           vNThreadsHttp = 0;
};

class server_http_task_queue : public httplib::TaskQueue {
  httplib::ThreadPool & vPool;
public:
  explicit server_http_task_queue(httplib::ThreadPool & pPool) : vPool(pPool) {}

  bool enqueue(std::function<void()> pFn) override { return vPool.enqueue(std::move(pFn)); }

  // note: must call join() to drain the pool
  void shutdown() override { /* no-op */ }
};

server_http_context::server_http_context() : vPimpl(std::make_unique<Impl>()) {}

server_http_context::~server_http_context() {
  // just in case any exit paths that forget to call join()
  try {
    mStop();
    mJoin();
  } catch (const std::exception & vE) {
    SRV_ERR("failed to stop HTTP server: %s\n", vE.what());
  } catch (...) {
    SRV_ERR("%s", "failed to stop HTTP server\n");
  }
}

static void fLogServerRequest(const httplib::Request & pReq, const httplib::Response & pRes) {
  // skip logging requests that are regularly sent, to avoid log spam
  if (pReq.path == "/health" || pReq.path == "/v1/health" || pReq.path == "/models" || pReq.path == "/v1/models" ||
      pReq.path == "/props" || pReq.path == "/metrics") {
    return;
  }

  // reminder: this function is not covered by httplib's exception handler; if someone does more complicated stuff, think about wrapping it in try-catch

  SRV_TRC("done request: %s %s %s %d\n", pReq.method.c_str(), pReq.path.c_str(), pReq.remote_addr.c_str(), pRes.status);

  SRV_DBG("request:  %s\n", pReq.body.c_str());
  SRV_DBG("response: %s\n", pRes.body.c_str());
}

// returns true if the Origin header value's host is localhost / 127.0.0.1 / ::1 (any port)
static bool fOriginIsLocalhost(const std::string & pOrigin) {
  try {
    const std::string cHost = fCommonHttpParseUrl(pOrigin).vHost;
    return cHost == "localhost" || cHost == "127.0.0.1" || cHost == "::1";
  } catch (const std::exception &) {
    return false;
  }
}

// For Google Cloud Platform deployment compatibility
struct gcp_params {
  bool        vEnabled;
  std::string vPathHealth;
  std::string vPathPredict;
  int         vPort;

  // Ref: https://docs.cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#aip-variables
  gcp_params() {
    vEnabled     = mGetenv("AIP_MODE", "") == "PREDICTION";
    vPathHealth  = mGetenv("AIP_HEALTH_ROUTE", "", true);  // default: using the route defined in server.cpp
    vPathPredict = mGetenv("AIP_PREDICT_ROUTE", "/predict", true);
    vPort        = std::stoi(mGetenv("AIP_HTTP_PORT", "8080"));
  }

  static std::string mGetenv(const char * name, const std::string & pDefaultValue, bool pEnsureLeadingSlash = false) {
    const auto * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
      return pDefaultValue;
    }
    std::string vVal = value;
    if (pEnsureLeadingSlash && !vVal.empty() && vVal[0] != '/') {
      vVal.insert(vVal.begin(), '/');
    }
    return vVal;
  }
};

bool server_http_context::init(const common_params & pParams) {
  const gcp_params cGcp;

  vPathPrefix = pParams.vApiPrefix;
  if ((vPathPrefix != "/api" && vPathPrefix.rfind("/api/", 0) != 0) || vPathPrefix == "/api/doc" ||
      vPathPrefix.rfind("/api/doc/", 0) == 0 || vPathPrefix.back() == '/' ||
      vPathPrefix.find("..") != std::string::npos ||
      vPathPrefix.find_first_not_of("/abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_") !=
        std::string::npos) {
    SRV_ERR("%s", "API prefix must be /api or a path below /api, without a trailing slash\n");
    return false;
  }
  vPort = pParams.vPort;

  if (cGcp.vEnabled) {
    SRV_TRC("Google Cloud Platform compat: health route = %s, predict route = %s, port = %d\n",
            cGcp.vPathHealth.c_str(), cGcp.vPathPredict.c_str(), cGcp.vPort);

    if (vPort != cGcp.vPort) {
      SRV_WRN("Google Cloud Platform compat: overriding server port %d with AIP_HTTP_PORT %d\n", vPort, cGcp.vPort);
    }

    vPort = cGcp.vPort;
  }

  vPimpl->lHosts    = pParams.lHostnames;
  size_t vNTcpHosts = 0;
  for (const auto & vHost : vPimpl->lHosts) {
    if (!fStringEndsWith(vHost, ".sock")) {
      vNTcpHosts++;
    }
  }
  if (vPort == 0 && vNTcpHosts > 1) {
    SRV_ERR("%s", "--port 0 is not supported with multiple TCP addresses\n");
    return false;
  }
  for (size_t vI = 0; vI < vPimpl->lHosts.size(); ++vI) {
    vPimpl->lServers.emplace_back();
    if (!mInitListener(pParams)) {
      return false;
    }
    // with multiple TCP addresses, [::] must not also claim 0.0.0.0
    if (vNTcpHosts > 1) {
      vPimpl->lServers.back()->set_ipv6_v6only(true);
    }
  }

  vPimpl->vNThreadsHttp = pParams.vNThreadsHttp;
  if (vPimpl->vNThreadsHttp < 1) {
    // +4 threads for monitoring, health and MCP.
    vPimpl->vNThreadsHttp =
      std::max(pParams.vNParallel + 4, static_cast<int32_t>(std::thread::hardware_concurrency() - 1));
  }
  SRV_TRC("using %d threads for HTTP server\n", vPimpl->vNThreadsHttp);
  return true;
}

bool server_http_context::mInitListener(const common_params & pParams) {
  auto & vSrv = vPimpl->lServers.back();

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  if (!pParams.vSslFileKey.empty() && !pParams.vSslFileCert.empty()) {
    SRV_TRC("running with SSL: key = %s, cert = %s\n", pParams.vSslFileKey.c_str(), pParams.vSslFileCert.c_str());
    vSrv   = std::make_unique<httplib::SSLServer>(pParams.vSslFileCert.c_str(), pParams.vSslFileKey.c_str());
    vIsSsl = true;
  } else {
    SRV_TRC("%s", "running without SSL\n");
    vSrv = std::make_unique<httplib::Server>();
  }
#else
  if (pParams.vSslFileKey != "" && pParams.vSslFileCert != "") {
    SRV_ERR("%s", "the server is built without SSL support\n");
    return false;
  }
  vSrv.reset(new httplib::Server());
#endif

  vSrv->set_default_headers({
    { "Server", "dgs-server" }
  });
  // vSrv->set_logger(log_server_request); // TODO @ngxson : this is too spamy, no very useful; improve it in the future
  vSrv->set_exception_handler([](const httplib::Request &, httplib::Response & pRes, const std::exception_ptr & pEp) {
    // this is fail-safe; exceptions should already handled by `ex_wrapper`

    std::string vMessage;
    try {
      std::rethrow_exception(pEp);
    } catch (const std::exception & vE) {
      vMessage = vE.what();
    } catch (...) {
      vMessage = "Unknown Exception";
    }

    pRes.status = 500;
    pRes.set_content(vMessage, "text/plain");
    SRV_ERR("got exception: %s\n", vMessage.c_str());
  });

  vSrv->set_error_handler([](const httplib::Request &, httplib::Response & pRes) {
    if (pRes.status == 404) {
      pRes.set_content(
        fSafeJsonToStr(json{
          { "error", { { "message", "File Not Found" }, { "type", "not_found_error" }, { "code", 404 } } }
      }),
        "application/json; charset=utf-8");
    }
    // for other error codes, we skip processing here because it's already done by res->error()
  });

  // set timeouts and change hostname and port
  vSrv->set_read_timeout(pParams.vTimeoutRead);
  vSrv->set_write_timeout(pParams.vTimeoutWrite);
  vSrv->set_socket_options([vReusePort = pParams.vReusePort](const socket_t pSock) {
    httplib::set_socket_opt(pSock, SOL_SOCKET, SO_REUSEADDR, 1);
    if (vReusePort) {
#ifdef SO_REUSEPORT
      httplib::set_socket_opt(pSock, SOL_SOCKET, SO_REUSEPORT, 1);
#else
      SRV_WRN("%s", "SO_REUSEPORT is not supported\n");
#endif
    }
  });

  if (pParams.lApiKeys.size() == 1) {
    const auto        cKey   = pParams.lApiKeys[0];
    const std::string substr = cKey.substr(std::max(static_cast<int>(cKey.length() - 4), 0));
    SRV_TRC("api_keys: ****%s\n", substr.c_str());
  } else if (pParams.lApiKeys.size() > 1) {
    SRV_TRC("api_keys: %zu keys loaded\n", pParams.lApiKeys.size());
  }

  //
  // Middlewares
  //

  // Frontend paths - all embedded UI assets
  static const std::unordered_set<std::string> sFrontendPaths = []() {
    std::unordered_set<std::string> sPaths{ "/" };
    for (const dgs_ui_asset & vA : fDgsUiGetAssets()) {
      sPaths.insert("/" + vA.vName);
    }
    return sPaths;
  }();

  // Public endpoints - API routes plus all embedded UI assets
  const std::unordered_set<std::string> sGetPublicEndpoints = [&pParams]() {
    std::unordered_set<std::string> sEndpoints{
      pParams.vApiPrefix + "/health", pParams.vApiPrefix + "/v1/health", "/api/config.js", "/api/doc",
      "/api/doc/swagger-ui.css",      "/api/doc/swagger-ui-bundle.js",   "/api/doc/",      "/api/doc/openapi.json",
    };
    sEndpoints.insert(sFrontendPaths.begin(), sFrontendPaths.end());
    return sEndpoints;
  }();

  auto vMiddlewareValidateApiKey = [lApiKeys = pParams.lApiKeys, sGetPublicEndpoints](const httplib::Request & pReq,
                                                                                      httplib::Response &      pRes) {
    // If API key is not set, skip validation
    if (lApiKeys.empty()) {
      return true;
    }

    // If path is public or a UI asset, skip validation
    if ((pReq.method == "GET" || pReq.method == "HEAD") && sGetPublicEndpoints.count(pReq.path)) {
      return true;
    }

    // Check for API key in the Authorization header
    std::string vReqApiKey = pReq.get_header_value("Authorization");
    if (vReqApiKey.empty()) {
      // retry with anthropic header
      vReqApiKey = pReq.get_header_value("X-Api-Key");
    }

    // remove the "Bearer " prefix if needed
    static std::string vPrefix = "Bearer ";
    if (vReqApiKey.substr(0, vPrefix.size()) == vPrefix) {
      vReqApiKey = vReqApiKey.substr(vPrefix.size());
    }

    // validate the API key
    if (std::find(lApiKeys.begin(), lApiKeys.end(), vReqApiKey) != lApiKeys.end()) {
      return true;  // API key is valid
    }

    // API key is invalid or not provided
    pRes.status = 401;
    pRes.set_content(
      fSafeJsonToStr(json{
        { "error", { { "message", "Invalid API Key" }, { "type", "authentication_error" }, { "code", 401 } } }
    }),
      "application/json; charset=utf-8");

    SRV_WRN("%s", "unauthorized: Invalid API Key\n");

    return false;
  };

  auto vMiddlewareServerState = [this, vPrefix = pParams.vApiPrefix](const httplib::Request & pReq,
                                                                     httplib::Response &      pRes) {
    if (!vIsReady.load()) {
      if (sFrontendPaths.count(pReq.path) || pReq.path == "/api/config.js" ||
          (pReq.path == "/api/doc" || pReq.path.rfind("/api/doc/", 0) == 0) ||
          pReq.path == vPrefix + "/monitor/usage" || pReq.path == vPrefix + "/mcp/shell" ||
          pReq.path == vPrefix + "/mcp/shell/askpass" || pReq.path == vPrefix + "/ui/settings") {
        return true;  // frontend asset, allow it to load and show "loading"
      }
      // no endpoints are allowed to be accessed when the server is not ready
      // this is to prevent any data races or inconsistent states
      pRes.status = 503;
      pRes.set_content(
        fSafeJsonToStr(json{
          { "error", { { "message", "Loading model" }, { "type", "unavailable_error" }, { "code", 503 } } }
      }),
        "application/json; charset=utf-8");
      return false;
    }
    return true;
  };

  // register server middlewares
  vSrv->set_pre_routing_handler([&pParams, vMiddlewareValidateApiKey, vMiddlewareServerState](
                                  const httplib::Request & pReq, httplib::Response & pRes) {
    if (pParams.vCorsCredentials && pParams.vCorsOrigins == "*") {
      // special case: echo back the Origin header to allow any origin to access the server with credentials
      pRes.set_header("Access-Control-Allow-Origin", pReq.get_header_value("Origin"));
    } else if (pParams.vCorsOrigins == "localhost") {
      // special case: only reflect the Origin header if it is a localhost origin
      std::string vOrigin = pReq.get_header_value("Origin");
      if (!vOrigin.empty() && fOriginIsLocalhost(vOrigin)) {
        pRes.set_header("Access-Control-Allow-Origin", vOrigin);
      } else if (!vOrigin.empty()) {
        SRV_WRN("(CORS) skip non-localhost origin: %s\n", vOrigin.c_str());
      }
    } else {
      pRes.set_header("Access-Control-Allow-Origin", pParams.vCorsOrigins);
    }
    // If this is OPTIONS request, skip validation because browsers don't include Authorization header
    if (pReq.method == "OPTIONS") {
      pRes.set_header("Access-Control-Allow-Credentials", pParams.vCorsCredentials ? "true" : "false");
      pRes.set_header("Access-Control-Allow-Methods", pParams.vCorsMethods);
      pRes.set_header("Access-Control-Allow-Headers", pParams.vCorsHeaders);
      pRes.set_content("", "text/html");                 // blank response, no data
      return httplib::Server::HandlerResponse::Handled;  // skip further processing
    }
    if (!vMiddlewareServerState(pReq, pRes)) {
      return httplib::Server::HandlerResponse::Handled;
    }
    if (!vMiddlewareValidateApiKey(pReq, pRes)) {
      return httplib::Server::HandlerResponse::Handled;
    }
    return httplib::Server::HandlerResponse::Unhandled;
  });

  vSrv->new_task_queue = [this] {
    return new server_http_task_queue(*vPimpl->vPool);
  };

  //
  // Web UI setup
  //

  fRegisterApiDocumentation(*vSrv, *this);
  vSrv->Get("/api/config.js", [vPrefix = vPathPrefix](const httplib::Request &, httplib::Response & pResponse) {
    const std::string cLanguage = fUiSettingsLanguage();
    pResponse.set_header("Cache-Control", "no-store");
    pResponse.set_content("globalThis.cApiPrefix = " + json(vPrefix).mDump() + ";\nglobalThis.cUiLanguage = " +
                            (cLanguage.empty() ? std::string("null") : json(cLanguage).mDump()) + ";\n",
                          "application/javascript; charset=utf-8");
  });

  // Serve the configured current web application.
  if (!pParams.vUi) {
    SRV_INF("%s", "The UI is disabled\n");
    SRV_INF("%s", "Use --ui/--no-ui to enable/disable\n");
  } else {
    // register static assets routes
    if (!pParams.vPublicPath.empty()) {
      // Set the base directory for serving static files
      if (const auto cIsFound = vSrv->set_mount_point("/", pParams.vPublicPath); !cIsFound) {
        SRV_ERR("static assets path not found: %s\n", pParams.vPublicPath.c_str());
        return false;
      }
    } else {
#if defined(DGS_UI_HAS_ASSETS)
      static auto handle_gzip_header = [](const httplib::Request & req, httplib::Response & res) {
        if (!fDgsUiUseGzip()) {
          // no gzip build, skip
          return true;
        }
        if (req.get_header_value("Accept-Encoding").find("gzip") == std::string::npos) {
          res.status = 415;  // unsupported media type
          res.set_content("Error: gzip is not supported by this browser", "text/plain");
          return false;
        } else {
          res.set_header("Content-Encoding", "gzip");
        }
        return true;
      };

      // Hashed assets never change under a given name, so they can be cached forever.
      // `index.html` is the exception: its name is stable while its contents change on
      // every build, and it is what names the hashed asset versions the UI loads.
      static constexpr auto cache_immutable  = "public, max-age=31536000, immutable";
      static constexpr auto cache_revalidate = "no-cache";

      // Serves an asset with ETag/304 handling, under the given caching policy.
      auto serve_asset_cached = [](const std::string & name, bool isolation, const char * cache_control) {
        return [name, isolation, cache_control](const httplib::Request & req, httplib::Response & res) {
          if (!handle_gzip_header(req, res)) {
            return true;  // returns error message
          }
          const dgs_ui_asset * a = fDgsUiFindAsset(name);
          if (!a) {
            res.status = 404;
            return false;
          }
          res.set_header("ETag", a->vEtag);
          if (const std::string & inm = req.get_header_value("If-None-Match");
              !inm.empty() && (inm == a->vEtag || inm == std::string("W/") + a->vEtag)) {
            res.status = 304;
            return false;
          }
          if (isolation) {
            res.set_header("Cross-Origin-Embedder-Policy", "require-corp");
            res.set_header("Cross-Origin-Opener-Policy", "same-origin");
          }
          res.set_header("Cache-Control", cache_control);
          res.set_content(reinterpret_cast<const char *>(a->vData), a->vSize, a->vType.c_str());
          return false;
        };
      };

      auto serve_asset_nocache = [](const std::string & name) {
        return [name](const httplib::Request & req, httplib::Response & res) {
          if (!handle_gzip_header(req, res)) {
            return true;  // returns error message
          }
          const dgs_ui_asset * a = fDgsUiFindAsset(name);
          if (!a) {
            res.status = 404;
            return false;
          }
          res.set_header("Cache-Control", "no-cache");
          res.set_content(reinterpret_cast<const char *>(a->vData), a->vSize, a->vType.c_str());
          return false;
        };
      };

      // main index file -- revalidated, so a new build is picked up on the next load
      vSrv->Get("/", serve_asset_cached("index.html", true, cache_revalidate));
      vSrv->Get("/index.html", serve_asset_cached("index.html", true, cache_revalidate));

      // All remaining assets registered directly from the embedded asset table.
      // PWA revalidation files (sw.js, manifest, version.json) use no-cache;
      // everything else is immutable.
      static const std::unordered_set<std::string> no_cache_names = { "sw.js", "manifest.webmanifest",
                                                                      "_app/version.json", "build.json" };

      for (const auto & a : fDgsUiGetAssets()) {
        if (a.vName == "index.html") {
          continue;  // served at "/" and "/index.html" above
        }
        if (no_cache_names.count(a.vName)) {
          SRV_DBG("serve nocache for %s\n", a.vName.c_str());
          vSrv->Get("/" + a.vName, serve_asset_nocache(a.vName));
        } else {
          vSrv->Get("/" + a.vName, serve_asset_cached(a.vName, false, cache_immutable));
        }
      }

#endif
    }
  }

  // a browser that used the built-in UI keeps its service worker, so it shows the old UI even after the UI is replaced or disabled
  // answer the worker's update check with a worker that removes itself; a sw.js in public_path is served first
  if (!pParams.vUi || !pParams.vPublicPath.empty()) {
    vSrv->Get("/sw.js", [](const httplib::Request &, httplib::Response & pRes) {
      static constexpr const char * cSwRemoveJs = R"(
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil((async () => {
    await self.registration.unregister();
    for (const key of await caches.keys()) await caches.delete(key);
    for (const c of await self.clients.matchAll({ type: 'window' })) c.navigate(c.url);
})()));
)";
      pRes.set_header("Cache-Control", "no-cache");
      pRes.set_content(cSwRemoveJs, "application/javascript");
    });
  }
  return true;
}

bool server_http_context::mStart() {
  // Bind and listen

  lListeningAddresses.clear();
  for (size_t vI = 0; vI < vPimpl->lServers.size(); ++vI) {
    const auto & vSrv    = vPimpl->lServers[vI];
    const auto & vHost   = vPimpl->lHosts[vI];
    const bool   cIsSock = fStringEndsWith(vHost, ".sock");
    bool         vWasBound;
    if (cIsSock) {
      SRV_TRC("%s", "setting address family to AF_UNIX\n");
      vSrv->set_address_family(AF_UNIX);
      // AF_UNIX ignores the port, but bind_to_port requires a nonzero value.
      vWasBound = vSrv->bind_to_port(vHost, 8080);
    } else if (vPort == 0) {
      const auto cBoundPort = vSrv->bind_to_any_port(vHost);
      vWasBound             = cBoundPort >= 0;
      if (vWasBound) {
        vPort = cBoundPort;
      }
    } else {
      vWasBound = vSrv->bind_to_port(vHost, vPort);
    }
    if (!vWasBound) {
      SRV_ERR("couldn't bind HTTP server socket, hostname: %s, port: %d\n", vHost.c_str(), vPort);
      mStop();
      lListeningAddresses.clear();
      return false;
    }
    lListeningAddresses.push_back(
      cIsSock ? fStringFormat("unix://%s", vHost.c_str()) :
                fStringFormat("%s://%s:%d", vIsSsl ? "https" : "http", fCommonHttpFormatHost(vHost).c_str(), vPort));
  }

  // n_threads_http fixed threads (always alive), plus up to 1024 dynamic threads destroyed after each request
  // ref: https://github.com/yhirose/cpp-httplib/pull/2368
  vPimpl->vPool = std::make_unique<httplib::ThreadPool>(vPimpl->vNThreadsHttp, vPimpl->vNThreadsHttp + 1024);
  for (size_t vI = 0; vI < vPimpl->lServers.size(); ++vI) {
    const auto & vSrv = vPimpl->lServers[vI];
    vPimpl->lThreads.emplace_back([vSrv = vSrv.get(), vAddr = lListeningAddresses[vI]] {
      if (!vSrv->listen_after_bind()) {
        SRV_ERR("listener on %s stopped unexpectedly\n", vAddr.c_str());
      }
    });
    vSrv->wait_until_ready();
    if (!vSrv->is_running()) {
      SRV_ERR("couldn't start HTTP listener on %s\n", lListeningAddresses[vI].c_str());
      mStop();
      mJoin();
      lListeningAddresses.clear();
      return false;
    }
  }
  return true;
}

void server_http_context::mStop() const {
  for (const auto & vSrv : vPimpl->lServers) {
    if (vSrv) {
      vSrv->stop();
    }
  }
}

void server_http_context::mJoin() {
  for (auto & vThread : vPimpl->lThreads) {
    if (vThread.joinable()) {
      vThread.join();
    }
  }
  // Queued requests still refer to their servers until the workers finish.
  if (vPimpl->vPool) {
    vPimpl->vPool->shutdown();
    vPimpl->vPool.reset();
  }
}

static void fSetHeaders(httplib::Response & pRes, const std::map<std::string, std::string> & pHeaders) {
  for (const auto & [key, value] : pHeaders) {
    pRes.set_header(key, value);
  }
}

// percent-decode a path component (%XX). path params arrive raw from httplib, unlike query
// params, so a conv id like "conv::model" sent as "conv%3A%3Amodel" must be decoded here to
// match the value the client put in the X-Conversation-Id header
static std::string fDecodePathComponent(const std::string & pIn) {
  std::string vOut;
  vOut.reserve(pIn.size());
  for (size_t vI = 0; vI < pIn.size(); vI++) {
    if (pIn[vI] == '%' && vI + 2 < pIn.size()) {
      auto vHex = [](char pC) -> int {
        if (pC >= '0' && pC <= '9') {
          return pC - '0';
        }
        if (pC >= 'a' && pC <= 'f') {
          return pC - 'a' + 10;
        }
        if (pC >= 'A' && pC <= 'F') {
          return pC - 'A' + 10;
        }
        return -1;
      };
      int vHi = vHex(pIn[vI + 1]);
      int vLo = vHex(pIn[vI + 2]);
      if (vHi >= 0 && vLo >= 0) {
        vOut.push_back(char((vHi << 4) | vLo));
        vI += 2;
        continue;
      }
    }
    vOut.push_back(pIn[vI]);
  }
  return vOut;
}

static std::map<std::string, std::string> fGetParams(const httplib::Request & pReq) {
  std::map<std::string, std::string> dParams;
  for (const auto & [key, value] : pReq.params) {
    dParams[key] = value;
  }
  for (const auto & [key, value] : pReq.path_params) {
    dParams[key] = fDecodePathComponent(value);
  }
  return dParams;
}

static std::map<std::string, std::string> fGetHeaders(const httplib::Request & pReq) {
  std::map<std::string, std::string> dHeaders;
  for (const auto & [key, value] : pReq.headers) {
    dHeaders[key] = value;
  }
  return dHeaders;
}

static std::string fBuildQueryString(const httplib::Request & pReq) {
  std::string vQs;
  for (const auto & [key, value] : pReq.params) {
    if (!vQs.empty()) {
      vQs += '&';
    }
    vQs += httplib::encode_query_component(key) + "=" + httplib::encode_query_component(value);
  }
  return vQs;
}

// using unique_ptr for request to allow safe capturing in lambdas
using server_http_req_ptr = std::unique_ptr<server_http_req>;

static void fProcessHandlerResponse(server_http_req_ptr && pRequest,
                                    server_http_res_ptr &  pResponse,
                                    httplib::Response &    pRes) {
  if (pResponse->mIsStream()) {
    pRes.status                              = pResponse->vStatus;
    // Tell Nginx to not buffer any streamed response
    pResponse->dHeaders["X-Accel-Buffering"] = "no";
    fSetHeaders(pRes, pResponse->dHeaders);
    const std::string                cContentType = pResponse->vContentType;
    // convert to shared_ptr as both chunked_content_provider() and on_complete() need to use it
    std::shared_ptr<server_http_req> vQPtr        = std::move(pRequest);
    std::shared_ptr<server_http_res> vRPtr        = std::move(pResponse);

    const auto cChunkedContentProvider = [vResponse = vRPtr](size_t, httplib::DataSink & pSink) -> bool {
      std::string vChunk;
      const bool  cHasNext = vResponse->vNext(vChunk);
      if (!vChunk.empty()) {
        if (!pSink.write(vChunk.data(), vChunk.size())) {
          return false;
        }
        SRV_DBG("http: streamed chunk: %s\n", vChunk.c_str());
      }
      if (!cHasNext) {
        pSink.done();
        SRV_DBG("%s", "http: stream ended\n");
      }
      return cHasNext;
    };
    const auto cOnComplete = [vRequest = vQPtr, vResponse = vRPtr](bool) mutable {
      vResponse->mOnComplete();
      vResponse.reset();
      vRequest.reset();
    };
    pRes.set_chunked_content_provider(cContentType, cChunkedContentProvider, cOnComplete);
  } else {
    pRes.status = pResponse->vStatus;
    fSetHeaders(pRes, pResponse->dHeaders);
    pRes.set_content(pResponse->data, pResponse->vContentType);
    pResponse->mOnComplete();
  }
}

void server_http_context::get(const std::string & pPath, const server_http_context::handler_t & pHandler) const {
  dApiMethods[pPath].insert("get");
  dHandlers.emplace(pPath, pHandler);
  auto vCallback = [pHandler, vPrefix = vPathPrefix](const httplib::Request & pReq, httplib::Response & pRes) {
    server_http_req_ptr vRequest  = std::make_unique<server_http_req>(server_http_req{ fGetParams(pReq),
                                                                                      fGetHeaders(pReq),
                                                                                      pReq.path.substr(vPrefix.size()),
                                                                                      fBuildQueryString(pReq),
                                                                                      pReq.body,
                                                                                       {},
                                                                                      pReq.is_connection_closed });
    server_http_res_ptr vResponse = pHandler(*vRequest);
    fProcessHandlerResponse(std::move(vRequest), vResponse, pRes);
  };
  const std::string cFullPath = vPathPrefix + pPath;
  for (const auto & vSrv : vPimpl->lServers) {
    vSrv->Get(cFullPath, vCallback);
  }
}

void server_http_context::mPost(const std::string & pPath, const server_http_context::handler_t & pHandler) const {
  dApiMethods[pPath].insert("post");
  dHandlers.emplace(pPath, pHandler);
  auto vCallback = [pHandler, vPrefix = vPathPrefix](const httplib::Request & pReq, httplib::Response & pRes) {
    std::string                          vBody = pReq.body;
    std::map<std::string, uploaded_file> dFiles;

    if (pReq.is_multipart_form_data()) {
      // translate text fields to a JSON object and use it as the body
      json vFormJson = json::mObject();
      for (const auto & [key, field] : pReq.form.fields) {
        if (vFormJson.contains(key)) {
          // if the key already exists, convert it to an array
          if (!vFormJson[key].is_array()) {
            json vExistingValue = vFormJson[key];
            vFormJson[key]      = json::mArray({ vExistingValue });
          }
          vFormJson[key].push_back(field.content);
        } else {
          vFormJson[key] = field.content;
        }
      }
      vBody = vFormJson.mDump();

      // populate files from multipart form
      for (const auto & [key, file] : pReq.form.files) {
        dFiles[key] = uploaded_file{
          raw_buffer(file.content.begin(), file.content.end()),
          file.filename,
          file.content_type,
        };
      }
    }

    server_http_req_ptr vRequest = std::make_unique<server_http_req>(
      server_http_req{ fGetParams(pReq), fGetHeaders(pReq), pReq.path.substr(vPrefix.size()), fBuildQueryString(pReq),
                       vBody, std::move(dFiles), pReq.is_connection_closed });
    server_http_res_ptr vResponse = pHandler(*vRequest);
    fProcessHandlerResponse(std::move(vRequest), vResponse, pRes);
  };
  const std::string cFullPath = vPathPrefix + pPath;
  for (const auto & vSrv : vPimpl->lServers) {
    vSrv->Post(cFullPath, vCallback);
  }
}

void server_http_context::mDel(const std::string & pPath, const server_http_context::handler_t & pHandler) const {
  dApiMethods[pPath].insert("delete");
  dHandlers.emplace(pPath, pHandler);
  auto vCallback = [pHandler, vPrefix = vPathPrefix](const httplib::Request & pReq, httplib::Response & pRes) {
    server_http_req_ptr vRequest  = std::make_unique<server_http_req>(server_http_req{ fGetParams(pReq),
                                                                                      fGetHeaders(pReq),
                                                                                      pReq.path.substr(vPrefix.size()),
                                                                                      fBuildQueryString(pReq),
                                                                                      pReq.body,
                                                                                       {},
                                                                                      pReq.is_connection_closed });
    server_http_res_ptr vResponse = pHandler(*vRequest);
    fProcessHandlerResponse(std::move(vRequest), vResponse, pRes);
  };
  const std::string cFullPath = vPathPrefix + pPath;
  for (const auto & vSrv : vPimpl->lServers) {
    vSrv->Delete(cFullPath, vCallback);
  }
}

//
// Vertex AI Prediction protocol (AIP_PREDICT_ROUTE)
// https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements
//

// Derives the camelCase @requestFormat alias for a registered path.
// e.g. "/v1/chat/completions" -> "chatCompletions", "/apply-template" -> "applyTemplate"
static std::string fPathToGcpFormat(const std::string & pPath) {
  std::string vS = pPath;
  if (vS.size() > 3 && vS[0] == '/' && vS[1] == 'v' && vS[2] == '1') {
    vS = vS.substr(3);
  }
  if (!vS.empty() && vS[0] == '/') {
    vS = vS.substr(1);
  }
  std::string vResult;
  bool        vCap = false;
  for (unsigned char vC : vS) {
    if (vC == ':') {
      break;  // stop before path parameters
    }
    if (vC == '/' || vC == '-' || vC == '_') {
      vCap = true;
    } else {
      vResult += static_cast<char>(vCap ? std::toupper(vC) : vC);
      vCap = false;
    }
  }
  return vResult;
}

static json fParseGcpPredictResponse(const server_http_res_ptr & pRes) {
  if (pRes == nullptr) {
    throw std::runtime_error("empty response from internal handler");
  }
  if (pRes->mIsStream()) {
    throw std::invalid_argument("predict route does not support streaming responses");
  }
  if (pRes->data.empty()) {
    return nullptr;
  }
  try {
    return json::mParse(pRes->data);
  } catch (...) {
    return pRes->data;
  }
}

void server_http_context::mRegisterGcpCompat() const {
  const gcp_params cGcp;

  if (!cGcp.vEnabled) {
    // do nothing
    return;
  }

  if (dHandlers.count(cGcp.vPathPredict)) {
    SRV_ERR("AIP_PREDICT_ROUTE=%s conflicts with an existing dgs-server route\n", cGcp.vPathPredict.c_str());
    exit(1);
  }

  // camelCase alias -> canonical path (first registration wins on collision)
  // e.g. "chatCompletions" -> "/v1/chat/completions"
  std::unordered_map<std::string, std::string> dAliasToPath;
  for (const auto & [path, _] : dHandlers) {
    dAliasToPath.emplace(fPathToGcpFormat(path), path);
  }

  if (!cGcp.vPathHealth.empty()) {
    const auto cHealthHandler = dHandlers.find("/health");
    GGML_ASSERT(cHealthHandler != dHandlers.end());
    get(cGcp.vPathHealth, cHealthHandler->second);
  }

  mPost(
    cGcp.vPathPredict,
    [this, dAliasToPath = std::move(dAliasToPath)](const server_http_req & pReq) -> server_http_res_ptr {
      static const auto cBuildError = [](const std::string & pMessage, error_type type) -> json {
        return json{
          { "error", fFormatErrorResponse(pMessage, type) }
        };
      };

      json data;
      try {
        data = json::mParse(pReq.vBody);
      } catch (const std::exception & vE) {
        auto vRes     = std::make_unique<server_http_res>();
        vRes->vStatus = 400;
        vRes->data    = fSafeJsonToStr({
          { "error", fFormatErrorResponse(vE.what(), cErrorTypeInvalidRequest) }
        });
        return vRes;
      }
      if (!data.is_object()) {
        auto vRes     = std::make_unique<server_http_res>();
        vRes->vStatus = 400;
        vRes->data    = fSafeJsonToStr({
          { "error", fFormatErrorResponse("request body must be a JSON object", cErrorTypeInvalidRequest) }
        });
        return vRes;
      }
      if (!data.contains("instances") || !data.at("instances").is_array()) {
        auto vRes     = std::make_unique<server_http_res>();
        vRes->vStatus = 400;
        vRes->data    = fSafeJsonToStr({
          { "error", fFormatErrorResponse("request body must include an array field named instances",
           cErrorTypeInvalidRequest) }
        });
        return vRes;
      }

      const json &        vInstances    = data.at("instances");
      static const size_t cMaxInstances = 128;
      if (vInstances.size() > cMaxInstances) {
        auto vRes     = std::make_unique<server_http_res>();
        vRes->vStatus = 400;
        vRes->data    = fSafeJsonToStr({
          { "error", fFormatErrorResponse("instances array exceeds maximum size of " + std::to_string(cMaxInstances),
           cErrorTypeInvalidRequest) }
        });
        return vRes;
      }

      std::vector<std::future<json>> lFutures;
      lFutures.reserve(vInstances.size());

      for (const auto & vInstance : vInstances) {
        lFutures.push_back(std::async(std::launch::async, [this, &pReq, &dAliasToPath, vInstance]() -> json {
          if (!vInstance.is_object()) {
            return cBuildError("each instance must be a JSON object", cErrorTypeInvalidRequest);
          }
          if (!vInstance.contains("@requestFormat") || !vInstance.at("@requestFormat").is_string()) {
            return cBuildError("each instance must include a string @requestFormat", cErrorTypeInvalidRequest);
          }

          try {
            json              vPayload = vInstance;
            const std::string format   = vPayload.at("@requestFormat").get<std::string>();
            vPayload.erase("@requestFormat");

            if (vPayload.contains("stream")) {
              SRV_WRN(
                "%s",
                "ignoring client-provided stream field in instance, streaming is not supported in predict route\n");
              vPayload["stream"] = false;
            }

            // accept both camelCase aliases (e.g. "chatCompletions") and direct paths
            std::string vDispatchPath;
            auto        vItAlias = dAliasToPath.find(format);
            if (vItAlias != dAliasToPath.end()) {
              vDispatchPath = vItAlias->second;
            } else if (dHandlers.count(format)) {
              vDispatchPath = format;
            } else {
              return cBuildError("no handler registered for @requestFormat: " + format, cErrorTypeInvalidRequest);
            }

            const server_http_req cInternalReq{
              pReq.dParams, pReq.dHeaders,    vPathPrefix + vDispatchPath, pReq.vQueryString, vPayload.mDump(),
              {},           pReq.vShouldStop,
            };

            server_http_res_ptr vInternalRes = dHandlers.at(vDispatchPath)(cInternalReq);
            return fParseGcpPredictResponse(vInternalRes);
          } catch (const std::invalid_argument & vE) {
            return cBuildError(vE.what(), cErrorTypeInvalidRequest);
          } catch (const std::exception & vE) {
            return cBuildError(vE.what(), cErrorTypeServer);
          } catch (...) {
            return cBuildError("unknown error", cErrorTypeServer);
          }
        }));
      }

      json vPredictions = json::mArray();
      for (auto & vFuture : lFutures) {
        vPredictions.push_back(vFuture.get());
      }

      auto vRes  = std::make_unique<server_http_res>();
      vRes->data = fSafeJsonToStr({
        { "predictions", vPredictions }
      });
      return vRes;
    });
}
