#include "download.h"

#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "hf-cache.h"
#include "http.h"
#include "json.h"
#include "log.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <future>
#include <map>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#ifndef __EMSCRIPTEN__
#  ifdef __linux__
#    include <linux/limits.h>
#  elif defined(_WIN32)
#    if !defined(PATH_MAX)
#      define PATH_MAX MAX_PATH
#    endif
#  elif defined(_AIX)
#    include <sys/limits.h>
#  else
#    include <sys/syslimits.h>
#  endif
#endif

//
// downloader
//

static void fWriteEtag(const std::string & pPath, const std::string & pEtag) {
  const std::string cEtagPath = pPath + ".etag";
  fFsWriteAtomic(std::filesystem::u8path(cEtagPath), pEtag);
  LOG_DBG("%s: file etag saved: %s\n", __func__, cEtagPath.c_str());
}

static std::string fReadEtag(const std::string & pPath) {
  const std::string cEtagPath = pPath + ".etag";
  if (!std::filesystem::exists(cEtagPath)) {
    return {};
  }
  std::ifstream vEtagIn(cEtagPath);
  if (!vEtagIn) {
    LOG_ERR("%s: could not open .etag file for reading: %s\n", __func__, cEtagPath.c_str());
    return {};
  }
  std::string vEtag;
  std::getline(vEtagIn, vEtag);
  return vEtag;
}

static bool fIsHttpStatusOk(int pStatus) {
  return pStatus >= 200 && pStatus < 400;
}

std::pair<std::string, std::string> fCommonDownloadSplitRepoTag(const std::string & pHfRepoWithTag) {
  auto        lParts  = fStringSplit<std::string>(pHfRepoWithTag, ':');
  std::string vTag    = lParts.size() > 1 ? lParts.back() : "";
  std::string vHfRepo = lParts[0];
  if (fStringSplit<std::string>(vHfRepo, '/').size() != 2) {
    throw std::invalid_argument("error: invalid HF repo format, expected <user>/<model>[:quant]\n");
  }
  return { vHfRepo, vTag };
}

class ProgressBar : public common_download_callback {
  static inline std::mutex                         vMutex;
  static inline std::map<const ProgressBar *, int> dLines;
  static inline int                                vMaxLine = 0;

  std::string vFilename;
  size_t      vLen = 0;

  static void mCleanup(const ProgressBar * pLine) {
    dLines.erase(pLine);
    if (dLines.empty()) {
      vMaxLine = 0;
    }
  }

  static bool mIsOutputATty() { return fCommonIsTty(stdout); }

public:
  ProgressBar() = default;

  void mOnStart(const common_download_progress & pP) override {
    vFilename = pP.vUrl;

    if (auto vPos = vFilename.rfind('/'); vPos != std::string::npos) {
      vFilename = vFilename.substr(vPos + 1);
    }
    if (auto vPos = vFilename.find('?'); vPos != std::string::npos) {
      vFilename = vFilename.substr(0, vPos);
    }
    for (size_t vI = 0; vI < vFilename.size(); ++vI) {
      if ((vFilename[vI] & 0xC0) != 0x80) {
        if (vLen++ == 39) {
          vFilename.resize(vI);
          vFilename += "…";
          break;
        }
      }
    }
  }

  void mOnDone(const common_download_progress &, bool) override {
    std::lock_guard<std::mutex> vLock(vMutex);
    mCleanup(this);
  }

  void mOnUpdate(const common_download_progress & pP) override {
    if (!pP.vTotal || !mIsOutputATty()) {
      return;
    }

    std::lock_guard<std::mutex> vLock(vMutex);

    if (dLines.find(this) == dLines.end()) {
      dLines[this] = vMaxLine++;
      std::cout << "\n";
    }
    int vLinesUp = vMaxLine - dLines[this];

    size_t vBar = (55 - vLen) * 2;
    size_t vPct = (100 * pP.vDownloaded) / pP.vTotal;
    size_t vPos = (vBar * pP.vDownloaded) / pP.vTotal;

    if (vLinesUp > 0) {
      std::cout << "\033[" << vLinesUp << "A";
    }
    std::cout << '\r' << "Downloading " << vFilename << " ";

    for (size_t vI = 0; vI < vBar; vI += 2) {
      std::cout << (vI + 1 < vPos ? "─" : (vI < vPos ? "╴" : " "));
    }
    std::cout << std::setw(4) << vPct << "%\033[K";

    if (vLinesUp > 0) {
      std::cout << "\033[" << vLinesUp << "B";
    }
    std::cout << '\r' << std::flush;

    if (pP.vDownloaded == pP.vTotal) {
      mCleanup(this);
    }
  }

  ProgressBar(const ProgressBar &)             = delete;
  ProgressBar & operator=(const ProgressBar &) = delete;
};

static bool fCommonPullFile(httplib::Client &          pCli,
                            const std::string &        pResolvePath,
                            const std::string &        pPathTmp,
                            bool                       pSupportsRanges,
                            common_download_progress & pP,
                            common_download_callback * pCallback) {
  std::ofstream vOfs(pPathTmp, std::ios::binary | std::ios::app);
  if (!vOfs.is_open()) {
    LOG_ERR("%s: error opening local file for writing: %s\n", __func__, pPathTmp.c_str());
    return false;
  }

  httplib::Headers vHeaders;
  if (pSupportsRanges && pP.vDownloaded > 0) {
    vHeaders.emplace("Range", "bytes=" + std::to_string(pP.vDownloaded) + "-");
  }

  const char * vFunc         = __func__;  // avoid __func__ inside a lambda
  size_t       vProgressStep = 0;

  auto vRes = pCli.Get(
    pResolvePath, vHeaders,
    [&](const httplib::Response & pResponse) {
      if (pP.vDownloaded > 0 && pResponse.status != 206) {
        LOG_WRN("%s: server did not respond with 206 Partial Content for a resume request. Status: %d\n", vFunc,
                pResponse.status);
        return false;
      }
      if (pP.vDownloaded == 0 && pResponse.status != 200) {
        LOG_WRN("%s: download received non-successful status code: %d\n", vFunc, pResponse.status);
        return false;
      }
      if (pP.vTotal == 0 && pResponse.has_header("Content-Length")) {
        try {
          size_t vContentLength = std::stoull(pResponse.get_header_value("Content-Length"));
          pP.vTotal             = pP.vDownloaded + vContentLength;
        } catch (const std::exception & vE) {
          LOG_WRN("%s: invalid Content-Length header: %s\n", vFunc, vE.what());
        }
      }
      return true;
    },
    [&](const char * data, size_t pLen) {
      vOfs.write(data, pLen);
      if (!vOfs) {
        LOG_ERR("%s: error writing to file: %s\n", vFunc, pPathTmp.c_str());
        return false;
      }
      pP.vDownloaded += pLen;
      vProgressStep += pLen;

      if (vProgressStep >= pP.vTotal / 1000 || pP.vDownloaded == pP.vTotal) {
        if (pCallback) {
          pCallback->mOnUpdate(pP);
          if (pCallback->mIsCancelled()) {
            return false;
          }
        }
        vProgressStep = 0;
      }
      return true;
    },
    nullptr);

  if (!vRes) {
    LOG_ERR("%s: download failed: %s (status: %d)\n", __func__, httplib::to_string(vRes.error()).c_str(),
            vRes ? vRes->status : -1);
    return false;
  }

  vOfs.close();
  if (!vOfs) {
    LOG_ERR("%s: error closing file: %s\n", __func__, pPathTmp.c_str());
    return false;
  }

  return true;
}

// download one single file from remote URL to local path
// returns status code or -1 on error
static int fCommonDownloadFileSingleOnline(const std::string &          pUrl,
                                           const std::string &          pPath,
                                           const common_download_opts & pOpts,
                                           bool                         pSkipEtag) {
  static const int cMaxAttempts       = 3;
  static const int cRetryDelaySeconds = 2;

  const bool cFileExists = std::filesystem::exists(std::filesystem::u8path(pPath));

  if (cFileExists && pSkipEtag) {
    LOG_DBG("%s: using cached file: %s\n", __func__, pPath.c_str());
    return 304;  // 304 Not Modified - fake cached response
  }

  auto [cli, parts] = fCommonHttpClient(pUrl);

  httplib::Headers vHeaders;
  for (const auto & vH : pOpts.lHeaders) {
    vHeaders.emplace(vH.first, vH.second);
  }
  if (vHeaders.find("User-Agent") == vHeaders.end()) {
    vHeaders.emplace("User-Agent", "dgs-cpp/" + std::string(fDgsBuildInfo()));
  }
  if (!pOpts.vBearerToken.empty()) {
    vHeaders.emplace("Authorization", "Bearer " + pOpts.vBearerToken);
  }
  cli.set_default_headers(vHeaders);

  std::string vLastEtag;
  if (cFileExists) {
    vLastEtag = fReadEtag(pPath);
  } else {
    LOG_DBG("%s: no previous model file found %s\n", __func__, pPath.c_str());
  }

  auto vHead = cli.Head(parts.vPath);
  if (!vHead || vHead->status < 200 || vHead->status >= 300) {
    LOG_TRC("%s: HEAD failed, status: %d\n", __func__, vHead ? vHead->status : -1);
    if (cFileExists) {
      LOG_TRC("%s: using cached file (HEAD failed): %s\n", __func__, pPath.c_str());
      return 304;  // 304 Not Modified - fake cached response
    }
    return vHead ? vHead->status : -1;
  }

  std::string vEtag;
  if (vHead->has_header("ETag")) {
    vEtag = vHead->get_header_value("ETag");
  }

  common_download_progress vP;
  vP.vUrl = pUrl;
  if (vHead->has_header("Content-Length")) {
    try {
      vP.vTotal = std::stoull(vHead->get_header_value("Content-Length"));
    } catch (const std::exception & vE) {
      LOG_WRN("%s: invalid Content-Length in HEAD response: %s\n", __func__, vE.what());
    }
  }

  bool vSupportsRanges = false;
  if (vHead->has_header("Accept-Ranges")) {
    vSupportsRanges = vHead->get_header_value("Accept-Ranges") != "none";
  }

  if (cFileExists) {
    if (vEtag.empty()) {
      LOG_DBG("%s: using cached file (no server etag): %s\n", __func__, pPath.c_str());
      return 304;  // 304 Not Modified - fake cached response
    }
    if (!vLastEtag.empty() && vLastEtag == vEtag) {
      LOG_DBG("%s: using cached file (same etag): %s\n", __func__, pPath.c_str());
      return 304;  // 304 Not Modified - fake cached response
    }
    // pass this point, the file exists but is different from the server version, so we need to redownload it
    if (remove(pPath.c_str()) != 0) {
      LOG_ERR("%s: unable to delete file: %s\n", __func__, pPath.c_str());
      return -1;
    }
  }

  {  // silent
    std::error_code vEc;
    std::filesystem::create_directories(std::filesystem::path(pPath).parent_path(), vEc);
  }

  bool              vSuccess       = false;
  const std::string cPathTemporary = pPath + ".downloadInProgress";
  int               vDelay         = cRetryDelaySeconds;

  if (pOpts.vCallback) {
    pOpts.vCallback->mOnStart(vP);
  }

  for (int vI = 0; vI < cMaxAttempts; ++vI) {
    if (pOpts.vCallback && pOpts.vCallback->mIsCancelled()) {
      break;
    }
    if (vI) {
      LOG_WRN("%s: retrying after %d seconds...\n", __func__, vDelay);
      std::this_thread::sleep_for(std::chrono::seconds(vDelay));
      vDelay *= cRetryDelaySeconds;
    }

    size_t vExistingSize = 0;

    if (std::filesystem::exists(cPathTemporary)) {
      if (vSupportsRanges) {
        vExistingSize = std::filesystem::file_size(cPathTemporary);
      } else if (remove(cPathTemporary.c_str()) != 0) {
        LOG_ERR("%s: unable to delete file: %s\n", __func__, cPathTemporary.c_str());
        break;
      }
    }

    vP.vDownloaded = vExistingSize;

    LOG_DBG("%s: downloading from %s to %s (etag:%s)...\n", __func__, fCommonHttpShowMaskedUrl(parts).c_str(),
            cPathTemporary.c_str(), vEtag.c_str());

    if (fCommonPullFile(cli, parts.vPath, cPathTemporary, vSupportsRanges, vP, pOpts.vCallback)) {
      if (std::rename(cPathTemporary.c_str(), pPath.c_str()) != 0) {
        LOG_ERR("%s: unable to rename file: %s to %s\n", __func__, cPathTemporary.c_str(), pPath.c_str());
        break;
      }
      if (!vEtag.empty() && !pSkipEtag) {
        fWriteEtag(pPath, vEtag);
      }
      vSuccess = true;
      break;
    }
  }

  if (pOpts.vCallback) {
    pOpts.vCallback->mOnDone(vP, vSuccess);
  }
  if (pOpts.vCallback && pOpts.vCallback->mIsCancelled() && std::filesystem::exists(cPathTemporary)) {
    if (remove(cPathTemporary.c_str()) != 0) {
      LOG_ERR("%s: unable to delete temporary file: %s\n", __func__, cPathTemporary.c_str());
    }
  }
  if (!vSuccess) {
    LOG_ERR("%s: download failed after %d attempts\n", __func__, cMaxAttempts);
    return -1;  // max attempts reached
  }

  return vHead->status;
}

std::pair<long, std::vector<char>> fCommonRemoteGetContent(const std::string &          pUrl,
                                                           const common_remote_params & pParams) {
  auto [cli, parts] = fCommonHttpClient(pUrl);

  httplib::Headers vHeaders;
  for (const auto & vH : pParams.lHeaders) {
    vHeaders.emplace(vH.first, vH.second);
  }
  if (vHeaders.find("User-Agent") == vHeaders.end()) {
    vHeaders.emplace("User-Agent", "dgs-cpp/" + std::string(fDgsBuildInfo()));
  }

  if (pParams.vTimeout > 0) {
    cli.set_read_timeout(pParams.vTimeout, 0);
    cli.set_write_timeout(pParams.vTimeout, 0);
  }

  std::vector<char> lBuf;
  auto              vRes = cli.Get(
    parts.vPath, vHeaders,
    [&](const char * data, size_t pLen) {
      lBuf.insert(lBuf.end(), data, data + pLen);
      return pParams.vMaxSize == 0 || lBuf.size() <= static_cast<size_t>(pParams.vMaxSize);
    },
    nullptr);

  if (!vRes) {
    throw std::runtime_error("error: cannot make GET request");
  }

  return { vRes->status, std::move(lBuf) };
}

int fCommonDownloadFileSingle(const std::string &          pUrl,
                              const std::string &          pPath,
                              const common_download_opts & pOpts,
                              bool                         pSkipEtag) {
  if (!pOpts.vOffline) {
    ProgressBar          vTtyCb;
    common_download_opts vOnlineOpts = pOpts;
    if (!vOnlineOpts.vCallback) {
      vOnlineOpts.vCallback = &vTtyCb;
    }
    return fCommonDownloadFileSingleOnline(pUrl, pPath, vOnlineOpts, pSkipEtag);
  }

  if (!std::filesystem::exists(std::filesystem::u8path(pPath))) {
    LOG_ERR("%s: required file is not available in cache (offline mode): %s\n", __func__, pPath.c_str());
    return -1;
  }

  LOG_DBG("%s: using cached file (offline mode): %s\n", __func__, pPath.c_str());

  // notify the callback that the file was cached
  if (pOpts.vCallback) {
    common_download_progress vP;
    vP.vUrl    = pUrl;
    vP.vCached = true;
    pOpts.vCallback->mOnStart(vP);
    pOpts.vCallback->mOnDone(vP, true);
  }

  return 304;  // Not Modified - fake cached response
}

struct gguf_split_info {
  std::string vPrefix;  // tag included
  std::string vTag;
  int         vIndex;
  int         count;
};

static gguf_split_info fGetGgufSplitInfo(const std::string & pPath) {
  static const std::regex cReSplit("^(.+)-([0-9]{5})-of-([0-9]{5})$", std::regex::icase);
  static const std::regex cReTag("[-.]([A-Z0-9_]+)$", std::regex::icase);
  std::smatch             vM;

  std::string vPrefix = pPath;
  if (!fStringRemoveSuffix(vPrefix, ".gguf")) {
    return {};
  }

  int vIndex = 1;
  int count  = 1;

  if (std::regex_match(vPrefix, vM, cReSplit)) {
    vIndex  = std::stoi(vM[2].str());
    count   = std::stoi(vM[3].str());
    vPrefix = vM[1].str();
  }

  std::string vTag;
  if (std::regex_search(vPrefix, vM, cReTag)) {
    vTag = vM[1].str();
    for (char & vC : vTag) {
      vC = std::toupper((unsigned char) vC);
    }
  }

  return { std::move(vPrefix), std::move(vTag), vIndex, count };
}

// Q4_0 -> 4, F16 -> 16, NVFP4 -> 4, Q8_K_M -> 8, etc
static int fExtractQuantBits(const std::string & pFilename) {
  auto vSplit = fGetGgufSplitInfo(pFilename);

  auto vPos = vSplit.vTag.find_first_of("0123456789");
  if (vPos == std::string::npos) {
    return 0;
  }

  return std::stoi(vSplit.vTag.substr(vPos));
}

static hf_cache::hf_files fGetSplitFiles(const hf_cache::hf_files & pFiles, const hf_cache::hf_file & pFile) {
  auto vSplit = fGetGgufSplitInfo(pFile.vPath);

  if (vSplit.count <= 1) {
    return { pFile };
  }
  hf_cache::hf_files lResult;

  for (const auto & vF : pFiles) {
    auto vSplitF = fGetGgufSplitInfo(vF.vPath);
    if (vSplitF.count == vSplit.count && vSplitF.vPrefix == vSplit.vPrefix) {
      lResult.push_back(vF);
    }
  }
  return lResult;
}

// pick the best sibling GGUF whose filename contains `keyword` (e.g. "mmproj" / "mtp"),
// preferring deeper shared directory prefix with the model, then exact `tag` match,
// then closest quantization to the tag when given, or to the model otherwise
static hf_cache::hf_file fFindBestSibling(const hf_cache::hf_files & pFiles,
                                          const std::string &        pModel,
                                          const std::string &        pKeyword,
                                          const std::string &        pTag = "") {
  hf_cache::hf_file vBest;
  size_t            vBestDepth = 0;
  int               vBestDiff  = 0;
  bool              vBestExact = false;
  bool              vFound     = false;

  std::string vTagUpper = pTag;
  for (char & vC : vTagUpper) {
    vC = (char) std::toupper((unsigned char) vC);
  }

  int vModelBits = 0;
  if (!vTagUpper.empty()) {
    auto vPos  = vTagUpper.find_first_of("0123456789");
    vModelBits = vPos == std::string::npos ? 0 : std::stoi(vTagUpper.substr(vPos));
  } else {
    vModelBits = fExtractQuantBits(pModel);
  }
  auto lModelParts = fStringSplit<std::string>(pModel, '/');
  auto lModelDir   = lModelParts.end() - 1;

  for (const auto & vF : pFiles) {
    if (!fStringEndsWith(vF.vPath, ".gguf") || vF.vPath.find(pKeyword) == std::string::npos) {
      continue;
    }

    auto lSibParts = fStringSplit<std::string>(vF.vPath, '/');
    auto lSibDir   = lSibParts.end() - 1;

    auto [_, dir] = std::mismatch(lModelParts.begin(), lModelDir, lSibParts.begin(), lSibDir);
    if (dir != lSibDir) {
      continue;
    }

    size_t vDepth = dir - lSibParts.begin();
    auto   vBits  = fExtractQuantBits(vF.vPath);
    auto   vDiff  = std::abs(vBits - vModelBits);

    std::string vPathUpper = vF.vPath;
    for (char & vC : vPathUpper) {
      vC = (char) std::toupper((unsigned char) vC);
    }
    bool vExact = !vTagUpper.empty() && vPathUpper.find("-" + vTagUpper + ".") != std::string::npos;

    if (!vFound || vDepth > vBestDepth || (vDepth == vBestDepth && vExact && !vBestExact) ||
        (vDepth == vBestDepth && vExact == vBestExact && vDiff < vBestDiff)) {
      vBest      = vF;
      vBestDepth = vDepth;
      vBestDiff  = vDiff;
      vBestExact = vExact;
      vFound     = true;
    }
  }
  return vBest;
}

static hf_cache::hf_file fFindBestMmproj(const hf_cache::hf_files & pFiles, const std::string & pModel) {
  return fFindBestSibling(pFiles, pModel, "mmproj");
}

static hf_cache::hf_file fFindBestMtp(const hf_cache::hf_files & pFiles,
                                      const std::string &        pModel,
                                      const std::string &        pTag = "") {
  return fFindBestSibling(pFiles, pModel, "mtp-", pTag);
}

static hf_cache::hf_file fFindBestEagle3(const hf_cache::hf_files & pFiles,
                                         const std::string &        pModel,
                                         const std::string &        pTag = "") {
  return fFindBestSibling(pFiles, pModel, "eagle3-", pTag);
}

static hf_cache::hf_file fFindBestDflash(const hf_cache::hf_files & pFiles,
                                         const std::string &        pModel,
                                         const std::string &        pTag = "") {
  return fFindBestSibling(pFiles, pModel, "dflash-", pTag);
}

static hf_cache::hf_file fFindBestDspark(const hf_cache::hf_files & pFiles,
                                         const std::string &        pModel,
                                         const std::string &        pTag = "") {
  return fFindBestSibling(pFiles, pModel, "dspark-", pTag);
}

static bool fGgufFilenameIsModel(const std::string & pFilepath) {
  if (!fStringEndsWith(pFilepath, ".gguf")) {
    return false;
  }

  std::string vFilename = pFilepath;
  if (auto vPos = vFilename.rfind('/'); vPos != std::string::npos) {
    vFilename = vFilename.substr(vPos + 1);
  }

  return vFilename.find("mmproj") == std::string::npos && vFilename.find("imatrix") == std::string::npos &&
         vFilename.find("mtp-") == std::string::npos && vFilename.find("eagle3-") == std::string::npos &&
         vFilename.find("dflash-") == std::string::npos && vFilename.find("dspark-") == std::string::npos;
}

static hf_cache::hf_file fFindBestModel(const hf_cache::hf_files & pFiles, const std::string & pTag) {
  std::vector<std::string> lTags;

  if (!pTag.empty()) {
    lTags.push_back(pTag);
  } else {
    lTags = { "Q4_K_M", "Q8_0" };
  }

  for (const auto & vT : lTags) {
    std::regex vPattern(vT + "[.-]", std::regex::icase);
    for (const auto & vF : pFiles) {
      if (fGgufFilenameIsModel(vF.vPath) && std::regex_search(vF.vPath, vPattern)) {
        auto vSplit = fGetGgufSplitInfo(vF.vPath);
        if (vSplit.count > 1 && vSplit.vIndex != 1) {
          continue;
        }
        return vF;
      }
    }
  }

  // fallback to first available model only if tag is empty
  if (pTag.empty()) {
    for (const auto & vF : pFiles) {
      if (fGgufFilenameIsModel(vF.vPath)) {
        auto vSplit = fGetGgufSplitInfo(vF.vPath);
        if (vSplit.count > 1 && vSplit.vIndex != 1) {
          continue;
        }
        return vF;
      }
    }
  }

  return {};
}

static void fListAvailableGgufFiles(const hf_cache::hf_files & pFiles) {
  LOG_INF("Available GGUF files:\n");
  for (const auto & vF : pFiles) {
    if (fStringEndsWith(vF.vPath, ".gguf")) {
      LOG_INF(" - %s\n", vF.vPath.c_str());
    }
  }
}

common_download_hf_plan fCommonDownloadGetHfPlan(const common_params_model &  pModel,
                                                 const common_download_opts & pOpts) {
  common_download_hf_plan vPlan;
  hf_cache::hf_files      lAll;

  auto [repo, tag] = fCommonDownloadSplitRepoTag(pModel.vHfRepo);

  if (!pOpts.vOffline) {
    lAll = hf_cache::fGetRepoFiles(repo, pOpts.vBearerToken);
  }
  if (lAll.empty()) {
    lAll = hf_cache::fGetCachedFiles(repo);
  }
  if (lAll.empty()) {
    return vPlan;
  }

  // if preset.ini exists in the repo root, download only that file
  for (const auto & vF : lAll) {
    if (vF.vPath == "preset.ini") {
      vPlan.vPreset = vF;
      return vPlan;
    }
  }

  hf_cache::hf_file vPrimary;

  if (!pModel.hf_file.empty()) {
    for (const auto & vF : lAll) {
      if (vF.vPath == pModel.hf_file) {
        vPrimary = vF;
        break;
      }
    }
    if (vPrimary.vPath.empty()) {
      LOG_ERR("%s: file '%s' not found in repository\n", __func__, pModel.hf_file.c_str());
      fListAvailableGgufFiles(lAll);
      return vPlan;
    }
  } else {
    vPrimary = fFindBestModel(lAll, tag);
    // a requested sidecar can resolve on its own, without a full model of the same tag
    if (vPrimary.vPath.empty() && !pOpts.vDownloadMtp && !pOpts.vDownloadDflash && !pOpts.vDownloadEagle3 &&
        !pOpts.vDownloadDspark) {
      LOG_ERR("%s: no GGUF files found in repository %s\n", __func__, repo.c_str());
      fListAvailableGgufFiles(lAll);
      return vPlan;
    }
  }

  if (!vPrimary.vPath.empty()) {
    vPlan.vPrimary    = vPrimary;
    vPlan.lModelFiles = fGetSplitFiles(lAll, vPrimary);
  }

  if (pOpts.vDownloadMmproj && !vPrimary.vPath.empty()) {
    vPlan.vMmproj = fFindBestMmproj(lAll, vPrimary.vPath);
  }
  if (pOpts.vDownloadMtp) {
    vPlan.vMtp = fFindBestMtp(lAll, vPrimary.vPath, tag);
  }
  if (pOpts.vDownloadDflash) {
    vPlan.vDflash = fFindBestDflash(lAll, vPrimary.vPath, tag);
  }
  if (pOpts.vDownloadEagle3) {
    vPlan.vEagle3 = fFindBestEagle3(lAll, vPrimary.vPath, tag);
  }
  if (pOpts.vDownloadDspark) {
    vPlan.vDspark = fFindBestDspark(lAll, vPrimary.vPath, tag);
  }

  if (vPrimary.vPath.empty() && vPlan.vMtp.vLocalPath.empty() && vPlan.vDflash.vLocalPath.empty() &&
      vPlan.vEagle3.vLocalPath.empty() && vPlan.vDspark.vLocalPath.empty()) {
    LOG_ERR("%s: no GGUF files found in repository %s\n", __func__, repo.c_str());
    fListAvailableGgufFiles(lAll);
  }

  return vPlan;
}

void fCommonDownloadRunTasks(const std::vector<common_download_task> & pTasks) {
  std::vector<std::future<int>> lFutures;
  for (const auto & vTask : pTasks) {
    lFutures.push_back(std::async(std::launch::async, [&vTask]() {
      return fCommonDownloadFileSingle(vTask.vUrl, vTask.vLocalPath, vTask.vOpts, vTask.vIsHf);
    }));
  }

  for (size_t vI = 0; vI < lFutures.size(); ++vI) {
    std::string vUrl    = pTasks[vI].vUrl;
    int         vStatus = lFutures[vI].get();
    bool        vIsOk   = fIsHttpStatusOk(vStatus);
    if (!vIsOk) {
      throw std::runtime_error(fStringFormat("Download '%s' failed with status code: %d", vUrl.c_str(), vStatus));
    }
  }
}

std::vector<std::string> fCommonDownloadGetAllParts(const std::string & pUrl) {
  auto vSplit = fGetGgufSplitInfo(pUrl);

  if (vSplit.count <= 1) {
    return { pUrl };
  }

  std::vector<std::string> lParts;
  for (int vI = 1; vI <= vSplit.count; vI++) {
    auto vSuffix = fStringFormat("-%05d-of-%05d.gguf", vI, vSplit.count);
    lParts.push_back(vSplit.vPrefix + vSuffix);
  }
  return lParts;
}

//
// Docker registry functions
//

static std::string fCommonDockerGetToken(const std::string & pRepo) {
  std::string vUrl = "https://auth.docker.io/token?service=registry.docker.io&scope=repository:" + pRepo + ":pull";

  common_remote_params vParams;
  auto                 lRes = fCommonRemoteGetContent(vUrl, vParams);

  if (lRes.first != 200) {
    throw std::runtime_error("Failed to get Docker registry token, HTTP code: " + std::to_string(lRes.first));
  }

  std::string vResponseStr(lRes.second.begin(), lRes.second.end());
  common_json vResponse = common_json::mParse(vResponseStr);

  if (!vResponse.contains("token")) {
    throw std::runtime_error("Docker registry token response missing 'token' field");
  }

  return vResponse["token"].get<std::string>();
}

std::string fCommonDockerResolveModel(const std::string & pDocker) {
  // Parse ai/smollm2:135M-Q4_0
  size_t      vColonPos = pDocker.find(':');
  std::string vRepo, vTag;
  if (vColonPos != std::string::npos) {
    vRepo = pDocker.substr(0, vColonPos);
    vTag  = pDocker.substr(vColonPos + 1);
  } else {
    vRepo = pDocker;
    vTag  = "latest";
  }

  // ai/ is the default
  size_t vSlashPos = pDocker.find('/');
  if (vSlashPos == std::string::npos) {
    vRepo.insert(0, "ai/");
  }

  LOG_INF("%s: Downloading Docker Model: %s:%s\n", __func__, vRepo.c_str(), vTag.c_str());
  try {
    // --- helper: digest validation ---
    auto vValidateOciDigest = [](const std::string & pDigest) -> std::string {
      // Expected: algo:hex ; start with sha256 (64 hex chars)
      // You can extend this map if supporting other algorithms in future.
      static const std::regex cRe("^sha256:([a-fA-F0-9]{64})$");
      std::smatch             vM;
      if (!std::regex_match(pDigest, vM, cRe)) {
        throw std::runtime_error("Invalid OCI digest format received in manifest: " + pDigest);
      }
      // normalize hex to lowercase
      std::string vNormalized = pDigest;
      std::transform(vNormalized.begin() + 7, vNormalized.end(), vNormalized.begin() + 7,
                     [](unsigned char pC) { return std::tolower(pC); });
      return vNormalized;
    };

    std::string token = fCommonDockerGetToken(vRepo);  // Get authentication token

    // Get manifest
    // TODO: cache the manifest response so that it appears in the model list
    const std::string    cUrlPrefix   = "https://registry-1.docker.io/v2/" + vRepo;
    std::string          vManifestUrl = cUrlPrefix + "/manifests/" + vTag;
    common_remote_params vManifestParams;
    vManifestParams.lHeaders.push_back({ "Authorization", "Bearer " + token });
    vManifestParams.lHeaders.push_back(
      { "Accept", "application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json" });
    auto lManifestRes = fCommonRemoteGetContent(vManifestUrl, vManifestParams);
    if (lManifestRes.first != 200) {
      throw std::runtime_error("Failed to get Docker manifest, HTTP code: " + std::to_string(lManifestRes.first));
    }

    std::string vManifestStr(lManifestRes.second.begin(), lManifestRes.second.end());
    common_json vManifest = common_json::mParse(vManifestStr);
    std::string vGgufDigest;  // Find the GGUF layer
    if (vManifest.contains("layers")) {
      for (const auto & layer : vManifest["layers"]) {
        if (layer.contains("mediaType")) {
          std::string vMediaType = layer["mediaType"].get<std::string>();
          if (vMediaType == "application/vnd.docker.ai.gguf.v3" || vMediaType.find("gguf") != std::string::npos) {
            vGgufDigest = layer["digest"].get<std::string>();
            break;
          }
        }
      }
    }

    if (vGgufDigest.empty()) {
      throw std::runtime_error("No GGUF layer found in Docker manifest");
    }

    // Validate & normalize digest
    vGgufDigest = vValidateOciDigest(vGgufDigest);
    LOG_DBG("%s: Using validated digest: %s\n", __func__, vGgufDigest.c_str());

    // Prepare local filename
    std::string vModelFilename = vRepo;
    std::replace(vModelFilename.begin(), vModelFilename.end(), '/', '_');
    vModelFilename += "_" + vTag + ".gguf";
    std::string vLocalPath = fFsPathToUtf8(fFsGetCacheFile(vModelFilename));

    const std::string    cBlobUrl = cUrlPrefix + "/blobs/" + vGgufDigest;
    common_download_opts vOpts;
    vOpts.vBearerToken    = token;
    const int cHttpStatus = fCommonDownloadFileSingle(cBlobUrl, vLocalPath, vOpts);
    if (!fIsHttpStatusOk(cHttpStatus)) {
      throw std::runtime_error("Failed to download Docker Model");
    }

    LOG_INF("%s: Downloaded Docker Model to: %s\n", __func__, vLocalPath.c_str());
    return vLocalPath;
  } catch (const std::exception & vE) {
    LOG_ERR("%s: Docker Model download failed: %s\n", __func__, vE.what());
    throw;
  }
}

std::vector<common_cached_model_info> fCommonListCachedModels() {
  std::unordered_set<std::string>       sSeen;
  std::vector<common_cached_model_info> lResult;

  auto lFiles = hf_cache::fGetCachedFiles();

  for (const auto & vF : lFiles) {
    auto vSplit = fGetGgufSplitInfo(vF.vPath);
    if (vSplit.vIndex != 1 || vSplit.vTag.empty() || vSplit.vPrefix.find("mmproj") != std::string::npos ||
        vSplit.vPrefix.find("mtp-") != std::string::npos || vSplit.vPrefix.find("eagle3-") != std::string::npos ||
        vSplit.vPrefix.find("dflash-") != std::string::npos || vSplit.vPrefix.find("dspark-") != std::string::npos) {
      continue;
    }
    if (sSeen.insert(vF.vRepoId + ":" + vSplit.vTag).second) {
      lResult.push_back({ vF.vRepoId, vSplit.vTag });
    }
  }

  return lResult;
}

std::string fCommonDownloadResolvePath(const std::string & pHfRepoWithTag, const std::string & hf_file) {
  auto [repo, tag] = fCommonDownloadSplitRepoTag(pHfRepoWithTag);

  auto lFiles = hf_cache::fGetCachedFiles(repo);
  if (lFiles.empty()) {
    return "";
  }

  if (!hf_file.empty()) {
    for (const auto & vF : lFiles) {
      if (vF.vPath == hf_file) {
        return vF.vLocalPath;
      }
    }
    return "";
  }

  return fFindBestModel(lFiles, tag).vLocalPath;
}

bool fCommonDownloadRemove(const std::string & pHfRepoWithTag) {
  namespace fs = std::filesystem;

  auto [repo_id, tag] = fCommonDownloadSplitRepoTag(pHfRepoWithTag);

  if (tag.empty()) {
    return hf_cache::fRemoveCachedRepo(repo_id);
  }

  std::string vTagUpper = tag;
  for (char & vC : vTagUpper) {
    vC = (char) std::toupper((unsigned char) vC);
  }

  auto lFiles = hf_cache::fGetCachedFiles(repo_id);
  if (lFiles.empty()) {
    return false;
  }

  // collect snapshot entries whose tag matches
  std::vector<fs::path> lToRemove;
  for (const auto & vF : lFiles) {
    auto vSplit = fGetGgufSplitInfo(vF.vPath);
    if (vSplit.vTag == vTagUpper) {
      lToRemove.emplace_back(vF.vLocalPath);
    }
  }

  if (lToRemove.empty()) {
    return false;
  }

  // resolve blob paths from symlinks before deleting snapshot entries
  std::vector<fs::path> lBlobsToCheck;
  for (const auto & vP : lToRemove) {
    std::error_code vEc;
    if (fs::is_symlink(vP, vEc)) {
      auto vTarget = fs::read_symlink(vP, vEc);
      if (!vEc) {
        lBlobsToCheck.push_back((vP.parent_path() / vTarget).lexically_normal());
      }
    }
  }

  // remove snapshot entries
  for (const auto & vP : lToRemove) {
    std::error_code vEc;
    fs::remove(vP, vEc);
    if (vEc) {
      LOG_WRN("%s: failed to remove %s: %s\n", __func__, vP.string().c_str(), vEc.message().c_str());
    }
  }

  if (lBlobsToCheck.empty()) {
    return true;
  }

  // collect blobs still referenced by remaining snapshot entries
  std::unordered_set<std::string> sStillReferenced;
  for (const auto & vF : hf_cache::fGetCachedFiles(repo_id)) {
    fs::path        vP(vF.vLocalPath);
    std::error_code vEc;
    if (fs::is_symlink(vP, vEc)) {
      auto vTarget = fs::read_symlink(vP, vEc);
      if (!vEc) {
        sStillReferenced.insert((vP.parent_path() / vTarget).lexically_normal().string());
      }
    }
  }

  // remove orphaned blobs
  for (const auto & vBlob : lBlobsToCheck) {
    if (sStillReferenced.find(vBlob.string()) == sStillReferenced.end()) {
      std::error_code vEc;
      fs::remove(vBlob, vEc);
      if (vEc) {
        LOG_WRN("%s: failed to remove blob %s: %s\n", __func__, vBlob.string().c_str(), vEc.message().c_str());
      }
    }
  }

  return true;
}
