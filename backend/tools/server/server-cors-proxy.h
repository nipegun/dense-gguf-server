#pragma once

#include "common.h"
#include "http.h"
#include "server-http.h"
#include "server-models.h"

#include <algorithm>
#include <cctype>
#include <list>
#include <map>
#include <string>
#include <unordered_set>

static std::string fProxyHeaderToLower(std::string pHeader) {
  std::transform(pHeader.begin(), pHeader.end(), pHeader.begin(), [](unsigned char pC) { return std::tolower(pC); });
  return pHeader;
}

static server_http_res_ptr fProxyRequest(const server_http_req & pReq, std::string pMethod) {
  std::string     vTargetUrl = pReq.mGetParam("url");
  common_http_url vParsedUrl = fCommonHttpParseUrl(vTargetUrl);

  if (vParsedUrl.vHost.empty()) {
    throw std::runtime_error("invalid target URL: missing host");
  }

  if (vParsedUrl.vPath.empty()) {
    vParsedUrl.vPath = "/";
  }

  if (!vParsedUrl.vPassword.empty()) {
    throw std::runtime_error("authentication in target URL is not supported");
  }

  if (vParsedUrl.vScheme != "http" && vParsedUrl.vScheme != "https") {
    throw std::runtime_error("unsupported URL scheme in target URL: " + vParsedUrl.vScheme);
  }

  SRV_INF("proxying %s request to %s://%s:%i%s\n", pMethod.c_str(), vParsedUrl.vScheme.c_str(),
          fCommonHttpFormatHost(vParsedUrl.vHost).c_str(), vParsedUrl.vPort, vParsedUrl.vPath.c_str());

  std::map<std::string, std::string> dHeaders;
  const std::string                  cProxyHeaderPrefix = "x-dgs-server-proxy-header-";
  for (auto [key, value] : pReq.dHeaders) {
    const std::string cLoweredKey = fProxyHeaderToLower(key);
    if (!fStringStartsWith(cLoweredKey, cProxyHeaderPrefix)) {
      continue;
    }

    auto vNewKey = key.substr(cProxyHeaderPrefix.size());
    if (vNewKey.empty()) {
      continue;
    }

    dHeaders[vNewKey] = value;
  }

  auto vProxy =
    std::make_unique<server_http_proxy>(pMethod, vParsedUrl.vScheme, vParsedUrl.vHost, vParsedUrl.vPort,
                                        vParsedUrl.vPath, dHeaders, pReq.vBody, pReq.dFiles, pReq.vShouldStop,
                                        600,  // timeout_read (default to 10 minutes)
                                        600   // timeout_write (default to 10 minutes)
    );

  return vProxy;
}

static server_http_context::handler_t vProxyHandlerPost = [](const server_http_req & pReq) -> server_http_res_ptr {
  return fProxyRequest(pReq, "POST");
};

static server_http_context::handler_t vProxyHandlerGet = [](const server_http_req & pReq) -> server_http_res_ptr {
  return fProxyRequest(pReq, "GET");
};
