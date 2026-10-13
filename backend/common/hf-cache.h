#pragma once

#include <string>
#include <vector>

// Ref: https://huggingface.co/docs/hub/local-cache.md

namespace hf_cache {

struct hf_file {
  std::string vPath;
  std::string vUrl;
  std::string vLocalPath;
  std::string vFinalPath;
  std::string vOid;
  std::string vRepoId;
};

using hf_files = std::vector<hf_file>;

// Get files from HF API
hf_files fGetRepoFiles(const std::string & pRepoId, const std::string & token);

hf_files fGetCachedFiles(const std::string & pRepoId = {});

// Create snapshot path (link or move/copy) and return it
std::string fFinalizeFile(const hf_file & pFile);

// Remove the entire cached directory for a repo, returns true if removed
bool fRemoveCachedRepo(const std::string & pRepoId);

// Returns the HuggingFace hub cache path
std::string fGetCachePath();

}  // namespace hf_cache
