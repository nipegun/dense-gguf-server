#include "hf-cache.h"

#include "build-info.h"
#include "common.h"
#include "http.h"
#include "json.h"
#include "log.h"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  define HOME_DIR "USERPROFILE"
#  include <windows.h>
#else
#  define HOME_DIR "HOME"
#  include <pwd.h>
#  include <unistd.h>
#endif

namespace hf_cache {

namespace fs = std::filesystem;

static fs::path fGetCacheDirectory() {
  static const fs::path cCache = []() {
    struct {
      const char * vVar;
      fs::path     vPath;
    } aEntries[] = {
      { "DGS_CACHE",             fs::path()                                 },
      { "HF_HUB_CACHE",          fs::path()                                 },
      { "HUGGINGFACE_HUB_CACHE", fs::path()                                 },
      { "HF_HOME",               fs::path("hub")                            },
      { "XDG_CACHE_HOME",        fs::path("huggingface") / "hub"            },
      { HOME_DIR,                fs::path(".cache") / "huggingface" / "hub" }
    };
    for (const auto & vEntry : aEntries) {
      if (fs::path vBase = fCommonGetPathFromEnv(vEntry.vVar); !vBase.empty()) {
        return vEntry.vPath.empty() ? vBase : vBase / vEntry.vPath;
      }
    }
#ifndef _WIN32
    const struct passwd * vPw = getpwuid(getuid());

    if (vPw && vPw->pw_dir && *vPw->pw_dir) {
      return fs::path(vPw->pw_dir) / ".cache" / "huggingface" / "hub";
    }
#endif
    throw std::runtime_error("Failed to determine HF cache directory");
  }();

  return cCache;
}

std::string fGetCachePath() {
  return fFsPathToUtf8(fGetCacheDirectory());
}

static std::string fFolderNameToRepo(const std::string & pFolder) {
  constexpr std::string_view cPrefix = "models--";
  if (pFolder.rfind(cPrefix, 0)) {
    return {};
  }
  std::string vResult = pFolder.substr(cPrefix.length());
  fStringReplaceAll(vResult, "--", "/");
  return vResult;
}

static std::string fRepoToFolderName(const std::string & pRepoId) {
  constexpr std::string_view cPrefix = "models--";
  std::string                vResult = std::string(cPrefix) + pRepoId;
  fStringReplaceAll(vResult, "/", "--");
  return vResult;
}

static fs::path fGetRepoPath(const std::string & pRepoId) {
  return fGetCacheDirectory() / fRepoToFolderName(pRepoId);
}

static bool fIsHexChar(const char pC) {
  return (pC >= 'A' && pC <= 'F') || (pC >= 'a' && pC <= 'f') || (pC >= '0' && pC <= '9');
}

static bool fIsHexString(const std::string & pS, size_t pExpectedLen) {
  if (pS.length() != pExpectedLen) {
    return false;
  }
  for (const char cC : pS) {
    if (!fIsHexChar(cC)) {
      return false;
    }
  }
  return true;
}

static bool fIsAlphanum(const char pC) {
  return (pC >= 'A' && pC <= 'Z') || (pC >= 'a' && pC <= 'z') || (pC >= '0' && pC <= '9');
}

static bool fIsSpecialChar(char pC) {
  return pC == '/' || pC == '.' || pC == '-';
}

// base chars [A-Za-z0-9_] are always valid
// special chars [/.-] must be surrounded by base chars
// exactly one '/' required
static bool fIsValidRepoId(const std::string & pRepoId) {
  if (pRepoId.empty() || pRepoId.length() > 256) {
    return false;
  }
  int  vSlash   = 0;
  bool vSpecial = true;

  for (const char cC : pRepoId) {
    if (fIsAlphanum(cC) || cC == '_') {
      vSpecial = false;
    } else if (fIsSpecialChar(cC)) {
      if (vSpecial) {
        return false;
      }
      vSlash += (cC == '/');
      vSpecial = true;
    } else {
      return false;
    }
  }
  return !vSpecial && vSlash == 1;
}

static bool fIsValidHfToken(const std::string & token) {
  if (token.length() < 37 || token.length() > 256 || !fStringStartsWith(token, "hf_")) {
    return false;
  }
  for (size_t vI = 3; vI < token.length(); ++vI) {
    if (!fIsAlphanum(token[vI])) {
      return false;
    }
  }
  return true;
}

static bool fIsValidCommit(const std::string & hash) {
  return fIsHexString(hash, 40);
}

static bool fIsValidOid(const std::string & pOid) {
  return fIsHexString(pOid, 40) || fIsHexString(pOid, 64);
}

static bool fIsValidSubpath(const fs::path & pPath, const fs::path & pSubpath) {
  if (pSubpath.is_absolute()) {
    return false;  // never do a / b with b absolute
  }
  auto vB         = fs::absolute(pPath).lexically_normal();
  auto vT         = (vB / pSubpath).lexically_normal();
  auto [b_end, _] = std::mismatch(vB.begin(), vB.end(), vT.begin(), vT.end());

  return b_end == vB.end();
}

static common_json fApiGet(const std::string & pUrl, const std::string & token) {
  auto [cli, parts] = fCommonHttpClient(pUrl);

  httplib::Headers vHeaders = {
    { "User-Agent", "dgs-cpp/" + std::string(fDgsBuildInfo()) },
    { "Accept",     "application/json"                        }
  };

  if (fIsValidHfToken(token)) {
    vHeaders.emplace("Authorization", "Bearer " + token);
  } else if (!token.empty()) {
    LOG_WRN("%s: invalid token, authentication disabled\n", __func__);
  }

  if (auto vRes = cli.Get(parts.vPath, vHeaders)) {
    auto vBody = vRes->body;

    if (vRes->status == 200) {
      return common_json::mParse(vRes->body);
    }
    try {
      vBody = common_json::mParse(vRes->body)["error"].get<std::string>();
    } catch (...) {
    }

    throw std::runtime_error("GET failed (" + std::to_string(vRes->status) + "): " + vBody);
  } else {
    throw std::runtime_error("HTTPLIB failed: " + httplib::to_string(vRes.error()));
  }
}

static std::string fGetRepoCommit(const std::string & pRepoId, const std::string & token) {
  try {
    auto vEndpoint = fCommonGetModelEndpoint();
    auto vJson     = fApiGet(vEndpoint + "api/models/" + pRepoId + "/refs", token);

    if (!vJson.is_object() || !vJson.contains("branches") || !vJson["branches"].is_array()) {
      LOG_WRN("%s: missing 'branches' for '%s'\n", __func__, pRepoId.c_str());
      return {};
    }

    fs::path    vRefsPath = fGetRepoPath(pRepoId) / "refs";
    std::string name;
    std::string vCommit;
    fs::path    vNamePath;

    for (const auto & vBranch : vJson["branches"]) {
      if (!vBranch.is_object() || !vBranch.contains("name") || !vBranch["name"].is_string() ||
          !vBranch.contains("targetCommit") || !vBranch["targetCommit"].is_string()) {
        continue;
      }
      std::string vName   = vBranch["name"].get<std::string>();
      std::string vCommit = vBranch["targetCommit"].get<std::string>();

      if (!fIsValidCommit(vCommit)) {
        LOG_WRN("%s: skip invalid commit: %s\n", __func__, vCommit.c_str());
        continue;
      }
      const fs::path cCandidate = fs::u8path(vName);

      if (!fIsValidSubpath(vRefsPath, cCandidate)) {
        LOG_WRN("%s: skip invalid branch: %s\n", __func__, vName.c_str());
        continue;
      }

      if (vName == "main") {
        name      = vName;
        vCommit   = vCommit;
        vNamePath = cCandidate;
        break;
      }

      if (name.empty() || vCommit.empty()) {
        name      = vName;
        vCommit   = vCommit;
        vNamePath = cCandidate;
      }
    }

    if (name.empty() || vCommit.empty()) {
      LOG_WRN("%s: no valid branch for '%s'\n", __func__, pRepoId.c_str());
      return {};
    }

    fFsWriteAtomic(vRefsPath / vNamePath, vCommit);
    return vCommit;

  } catch (const common_json_error & vE) {
    LOG_ERR("%s: JSON error: %s\n", __func__, vE.what());
  } catch (const std::exception & vE) {
    LOG_ERR("%s: error: %s\n", __func__, vE.what());
  }
  return {};
}

hf_files fGetRepoFiles(const std::string & pRepoId, const std::string & token) {
  if (!fIsValidRepoId(pRepoId)) {
    LOG_WRN("%s: invalid repository: %s\n", __func__, pRepoId.c_str());
    return {};
  }

  std::string vCommit = fGetRepoCommit(pRepoId, token);
  if (vCommit.empty()) {
    LOG_WRN("%s: failed to resolve commit for %s\n", __func__, pRepoId.c_str());
    return {};
  }

  fs::path vBlobsPath  = fGetRepoPath(pRepoId) / "blobs";
  fs::path vCommitPath = fGetRepoPath(pRepoId) / "snapshots" / vCommit;

  hf_files lFiles;

  try {
    auto vEndpoint = fCommonGetModelEndpoint();
    auto vJson     = fApiGet(vEndpoint + "api/models/" + pRepoId + "/tree/" + vCommit + "?recursive=true", token);

    if (!vJson.is_array()) {
      LOG_WRN("%s: response is not an array for '%s'\n", __func__, pRepoId.c_str());
      return {};
    }

    for (const auto & vItem : vJson) {
      if (!vItem.is_object() || !vItem.contains("type") || !vItem["type"].is_string() || vItem["type"] != "file" ||
          !vItem.contains("path") || !vItem["path"].is_string()) {
        continue;
      }

      hf_file vFile;
      vFile.vRepoId = pRepoId;
      vFile.vPath   = vItem["path"].get<std::string>();

      const fs::path cSubpath = fs::u8path(vFile.vPath);

      if (!fIsValidSubpath(vCommitPath, cSubpath)) {
        LOG_WRN("%s: skip invalid path: %s\n", __func__, vFile.vPath.c_str());
        continue;
      }

      if (vItem.contains("lfs") && vItem["lfs"].is_object()) {
        if (vItem["lfs"].contains("oid") && vItem["lfs"]["oid"].is_string()) {
          vFile.vOid = vItem["lfs"]["oid"].get<std::string>();
        }
      } else if (vItem.contains("oid") && vItem["oid"].is_string()) {
        vFile.vOid = vItem["oid"].get<std::string>();
      }

      if (!vFile.vOid.empty() && !fIsValidOid(vFile.vOid)) {
        LOG_WRN("%s: skip invalid oid: %s\n", __func__, vFile.vOid.c_str());
        continue;
      }

      vFile.vUrl = vEndpoint + pRepoId + "/resolve/" + vCommit + "/" + vFile.vPath;

      fs::path vFinalPath = vCommitPath / cSubpath;
      vFile.vFinalPath    = fFsPathToUtf8(vFinalPath);

      if (!vFile.vOid.empty() && !fs::exists(vFinalPath)) {
        fs::path vLocalPath = vBlobsPath / vFile.vOid;
        vFile.vLocalPath    = fFsPathToUtf8(vLocalPath);
      } else {
        vFile.vLocalPath = vFile.vFinalPath;
      }

      lFiles.push_back(vFile);
    }
  } catch (const common_json_error & vE) {
    LOG_ERR("%s: JSON error: %s\n", __func__, vE.what());
  } catch (const std::exception & vE) {
    LOG_ERR("%s: error: %s\n", __func__, vE.what());
  }
  return lFiles;
}

static std::string fGetCachedRef(const fs::path & pRepoPath) {
  fs::path vRefsPath = pRepoPath / "refs";
  if (!fs::is_directory(vRefsPath)) {
    return {};
  }
  std::string vFallback;

  for (const auto & vEntry : fs::directory_iterator(vRefsPath)) {
    if (!vEntry.is_regular_file()) {
      continue;
    }
    std::ifstream vF(vEntry.path());
    std::string   vCommit;
    if (!vF || !std::getline(vF, vCommit) || vCommit.empty()) {
      continue;
    }
    if (!fIsValidCommit(vCommit)) {
      LOG_WRN("%s: skip invalid commit: %s\n", __func__, vCommit.c_str());
      continue;
    }
    if (vEntry.path().filename() == "main") {
      return vCommit;
    }
    if (vFallback.empty()) {
      vFallback = vCommit;
    }
  }
  return vFallback;
}

hf_files fGetCachedFiles(const std::string & pRepoId) {
  const fs::path cCachePath = fGetCacheDirectory();
  if (!fs::exists(cCachePath)) {
    return {};
  }

  if (!pRepoId.empty() && !fIsValidRepoId(pRepoId)) {
    LOG_WRN("%s: invalid repository: %s\n", __func__, pRepoId.c_str());
    return {};
  }

  hf_files lFiles;

  for (const auto & vRepo : fs::directory_iterator(cCachePath)) {
    if (!vRepo.is_directory()) {
      continue;
    }
    fs::path vSnapshotsPath = vRepo.path() / "snapshots";

    if (!fs::exists(vSnapshotsPath)) {
      continue;
    }
    std::string vRepoId = fFolderNameToRepo(fFsPathToUtf8(vRepo.path().filename()));

    if (!fIsValidRepoId(vRepoId)) {
      continue;
    }
    if (!pRepoId.empty() && vRepoId != pRepoId) {
      continue;
    }
    std::string vCommit     = fGetCachedRef(vRepo.path());
    fs::path    vCommitPath = vSnapshotsPath / vCommit;

    if (vCommit.empty() || !fs::is_directory(vCommitPath)) {
      continue;
    }
    for (const auto & vEntry : fs::recursive_directory_iterator(vCommitPath)) {
      if (!vEntry.is_regular_file() && !vEntry.is_symlink()) {
        continue;
      }
      fs::path vPath = vEntry.path().lexically_relative(vCommitPath);

      if (!vPath.empty()) {
        hf_file vFile;
        vFile.vRepoId           = vRepoId;
        const auto cGenericPath = vPath.generic_u8string();
        vFile.vPath             = std::string(cGenericPath.begin(), cGenericPath.end());
        vFile.vLocalPath        = fFsPathToUtf8(vEntry.path());
        vFile.vFinalPath        = vFile.vLocalPath;
        lFiles.push_back(std::move(vFile));
      }
    }
  }

  return lFiles;
}

std::string fFinalizeFile(const hf_file & pFile) {
  static std::atomic<bool> vSymlinksDisabled{ false };

  std::error_code vEc;
  fs::path        vLocalPath = fs::u8path(pFile.vLocalPath);
  fs::path        vFinalPath = fs::u8path(pFile.vFinalPath);

  if (vLocalPath == vFinalPath || fs::exists(vFinalPath, vEc)) {
    return pFile.vFinalPath;
  }

  if (!fs::exists(vLocalPath, vEc)) {
    return pFile.vFinalPath;
  }

  fs::create_directories(vFinalPath.parent_path(), vEc);

  if (!vSymlinksDisabled) {
    fs::path vTarget = fs::relative(vLocalPath, vFinalPath.parent_path(), vEc);
    if (!vEc) {
      fs::create_symlink(vTarget, vFinalPath, vEc);
    }
    if (!vEc) {
      return pFile.vFinalPath;
    }
  }

  if (!vSymlinksDisabled.exchange(true)) {
    LOG_WRN("%s: failed to create symlink: %s\n", __func__, vEc.message().c_str());
    LOG_WRN("%s: switching to degraded mode\n", __func__);
  }

  fs::rename(vLocalPath, vFinalPath, vEc);
  if (vEc) {
    LOG_WRN("%s: failed to move file to snapshots: %s\n", __func__, vEc.message().c_str());
    fs::copy(vLocalPath, vFinalPath, vEc);
    if (vEc) {
      LOG_ERR("%s: failed to copy file to snapshots: %s\n", __func__, vEc.message().c_str());
    }
  }
  return pFile.vFinalPath;
}

bool fRemoveCachedRepo(const std::string & pRepoId) {
  if (!fIsValidRepoId(pRepoId)) {
    LOG_WRN("%s: invalid repository: %s\n", __func__, pRepoId.c_str());
    return false;
  }
  fs::path        vRepoPath = fGetRepoPath(pRepoId);
  std::error_code vEc;
  auto            vRemoved = fs::remove_all(vRepoPath, vEc);
  if (vEc) {
    LOG_ERR("%s: failed to remove repo cache %s: %s\n", __func__, fFsPathToUtf8(vRepoPath).c_str(),
            vEc.message().c_str());
    return false;
  }
  return vRemoved > 0;
}

}  // namespace hf_cache
