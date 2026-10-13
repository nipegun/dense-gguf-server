#pragma once

#include "hf-cache.h"

#include <functional>
#include <string>
#include <vector>

struct common_params_model;

using common_header      = std::pair<std::string, std::string>;
using common_header_list = std::vector<common_header>;

struct common_download_progress {
  std::string vUrl;
  size_t      vDownloaded = 0;
  size_t      vTotal      = 0;
  bool        vCached     = false;
};

class common_download_callback {
public:
  virtual ~common_download_callback()                                 = default;
  virtual void mOnStart(const common_download_progress & pP)          = 0;
  virtual void mOnUpdate(const common_download_progress & pP)         = 0;
  virtual void mOnDone(const common_download_progress & pP, bool pOk) = 0;

  virtual bool mIsCancelled() const { return false; }
};

struct common_remote_params {
  common_header_list lHeaders;
  long               vTimeout = 0;  // in seconds, 0 means no timeout
  long               vMaxSize = 0;  // unlimited if 0
};

// get remote file content, returns <http_code, raw_response_body>
std::pair<long, std::vector<char>> fCommonRemoteGetContent(const std::string &          pUrl,
                                                           const common_remote_params & pParams);

// split HF repo with tag into <repo, tag>, for example:
// - "ggml-org/models:F16" -> <"ggml-org/models", "F16">
// tag is optional and can be empty
std::pair<std::string, std::string> fCommonDownloadSplitRepoTag(const std::string & pHfRepoWithTag);

// Result of common_list_cached_models
struct common_cached_model_info {
  std::string vRepo;
  std::string vTag;

  std::string to_string() const { return vRepo + ":" + vTag; }
};

// Options for common_download_file_single
struct common_download_opts {
  std::string                vBearerToken;
  common_header_list         lHeaders;
  bool                       vOffline        = false;
  bool                       vDownloadMmproj = false;
  bool                       vDownloadMtp    = false;
  bool                       vDownloadEagle3 = false;
  bool                       vDownloadDflash = false;
  bool                       vDownloadDspark = false;
  common_download_callback * vCallback       = nullptr;
};

struct common_download_task {
  common_download_opts  vOpts;
  std::string           vUrl;
  std::string           vLocalPath;
  std::function<void()> vOnDone;
  bool                  vIsHf = false;

  common_download_task() = default;

  common_download_task(hf_cache::hf_file            pF,
                       const common_download_opts & pOpts,
                       std::function<void()>        pOnDone = nullptr) :
    vOpts(pOpts),
    vUrl(pF.vUrl),
    vLocalPath(pF.vLocalPath),
    vOnDone(pOnDone),
    vIsHf(true) {}
};

void fCommonDownloadRunTasks(const std::vector<common_download_task> & pTasks);

// if url is a multi-part GGUF file, returns all parts, otherwise returns the single file
std::vector<std::string> fCommonDownloadGetAllParts(const std::string & pUrl);

// returns list of cached models
std::vector<common_cached_model_info> fCommonListCachedModels();

// resolve the local cached file path for a HF repo without network access (hf_file, if given, must match exactly)
// returns an empty string if the model is not present in the cache
std::string fCommonDownloadResolvePath(const std::string & pHfRepoWithTag, const std::string & hf_file = "");

// download single file from url to local path
// returns status code or -1 on error
// skip_etag: if true, don't read/write .etag files (for HF cache where filename is the hash)
int fCommonDownloadFileSingle(const std::string &          pUrl,
                              const std::string &          pPath,
                              const common_download_opts & pOpts     = {},
                              bool                         pSkipEtag = false);

// resolve and download model from Docker registry
// return local path to downloaded model file
std::string fCommonDockerResolveModel(const std::string & pDocker);

// Remove a cached model from disk
// input format: "user/model" or "user/model:tag"
// - if tag is omitted, removes the entire repo cache directory
// - if tag is present, removes only files matching that tag (and orphaned blobs)
// returns true if anything was removed
bool fCommonDownloadRemove(const std::string & pHfRepoWithTag);

struct common_download_hf_plan {
  hf_cache::hf_file  vPrimary;
  hf_cache::hf_files lModelFiles;
  hf_cache::hf_file  vMmproj;
  hf_cache::hf_file  vMtp;
  hf_cache::hf_file  vEagle3;
  hf_cache::hf_file  vDflash;
  hf_cache::hf_file  vDspark;
  hf_cache::hf_file  vPreset;  // if set, only this file is downloaded
};

common_download_hf_plan fCommonDownloadGetHfPlan(const common_params_model &  pModel,
                                                 const common_download_opts & pOpts);
