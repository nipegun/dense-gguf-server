#include "server-tools.h"

#include "base64.hpp"
#include "subproc.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <mutex>
#include <regex>
#include <thread>
#include <tuple>
#include <unordered_set>

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <fcntl.h>
#  include <io.h>
#  include <windows.h>
#else
#  include <unistd.h>

#  include <cerrno>
#endif

namespace fs = std::filesystem;

//
// internal helpers
//

// a child process writes in the OEM code page, so accented output would reach
// the JSON layer as invalid bytes. run() spawns without a console, so the
// console code page never applies
static std::string fConsoleOutputToUtf8(const std::string & pText) {
#if defined(_WIN32)
  // a chunk can end mid sequence, so the incomplete tail is dropped first
  if (text.empty() || is_valid_utf8(text.substr(0, validate_utf8(text)))) {
    // never decode twice a child that already emits UTF-8
    return text;
  }

  const UINT cp = GetOEMCP();

  // fail rather than emit replacement characters when the code page is wrong
  const int wide_len = MultiByteToWideChar(cp, MB_ERR_INVALID_CHARS, text.data(), (int) text.size(), nullptr, 0);
  if (wide_len <= 0) {
    return text;
  }
  std::wstring wide(wide_len, L'\0');
  MultiByteToWideChar(cp, MB_ERR_INVALID_CHARS, text.data(), (int) text.size(), wide.data(), wide_len);

  const int utf8_len = WideCharToMultiByte(CP_UTF8, 0, wide.data(), wide_len, nullptr, 0, nullptr, nullptr);
  if (utf8_len <= 0) {
    return text;
  }
  std::string utf8(utf8_len, '\0');
  WideCharToMultiByte(CP_UTF8, 0, wide.data(), wide_len, utf8.data(), utf8_len, nullptr, nullptr);
  return utf8;
#else
  return pText;
#endif
}

json server_tool::to_json() const {
  return {
    { "display_name", vDisplayName                          },
    { "tool",         name                                  },
    { "type",         type()                                },
    { "permissions",  json{ { "write", vPermissionWrite } } },
    { "uses_cwd",     vUsesCwd                              },
    { "definition",   mGetDefinition()                      },
  };
}

static constexpr size_t cServerToolGitLsFilesMaxOutput = 8 * 1024 * 1024;  // 8 MB
// budget for one listing call, shared by the git and walker paths
static constexpr int    cServerToolListEntriesTimeout  = 15;  // seconds

// entry kinds a directory listing may return
enum class list_kind {
  cFiles,  // regular files only
  cDirs,   // directories only
  cAll,    // both
};

// a narrow path uses the active code page on Windows, so every crossing between
// a std::string (always UTF-8 here) and fs::path is converted explicitly
static fs::path fPathFromUtf8(const std::string & pS) {
  return fs::u8path(pS);
}

// '/' separators on every platform: Windows accepts them, the web UI needs them
static std::string fPathToUtf8(const fs::path & pP) {
  const auto cS = pP.generic_u8string();
  return std::string(cS.begin(), cS.end());
}

// home directory, read once at first use (getenv is not thread safe against setenv)
static const std::string & fHomeDir() {
  static const std::string cHome = [] {
#ifdef _WIN32
    // the narrow getenv would return the profile path in the active code page
    const wchar_t * w = _wgetenv(L"HOME");
    if (w == nullptr)
      w = _wgetenv(L"USERPROFILE");
    return w ? path_to_utf8(fs::path(w)) : std::string();
#else
    const char * vH = getenv("HOME");
    return vH ? std::string(vH) : std::string();
#endif
  }();
  return cHome;
}

static std::string fExpandHome(const std::string & pPath) {
  if (pPath.empty() || pPath[0] != '~') {
    return pPath;
  }
  if (pPath.size() > 1 && pPath[1] != '/' && pPath[1] != '\\') {
    return pPath;
  }
  const std::string & vHome = fHomeDir();
  if (vHome.empty()) {
    return pPath;
  }
  return vHome + pPath.substr(1);
}

// depth of a '/'-separated relative path: "a/b/c" is 3
static int fEntryDepth(const std::string & pRel) {
  return 1 + (int) std::count(pRel.begin(), pRel.end(), '/');
}

// directories that a listing reports but never descends into: they can be enormous
// lowercase only, the local walker case-folds a name before the lookup
static const char * const aServerToolJunkDirNames[] = {
  ".git", ".svn",  ".hg",    "node_modules", "__pycache__", ".venv",   "venv",
  "dist", "build", "target", ".cache",       ".idea",       ".vscode",
};

class tools_io {
public:
  struct exec_result {
    std::string vOutput;
    int         vExitCode = -1;
    bool        vTimedOut = false;
  };

  virtual ~tools_io() = default;

  virtual bool        mIsDirectory(const std::string & pPath) const                             = 0;
  virtual bool        mIsRegularFile(const std::string & pPath) const                           = 0;
  virtual bool        mFileSize(const std::string & pPath, uintmax_t & pOutSize) const          = 0;
  virtual bool        mReadFile(const std::string & pPath, std::string & pOut) const            = 0;
  virtual bool        mWriteFile(const std::string & pPath, const std::string & pContent) const = 0;
  // resolve `path` against the IO's working directory; absolute paths are returned unchanged
  virtual std::string mResolve(const std::string & pPath) const                                 = 0;

  struct list_entry {
    std::string vRel;  // '/'-separated, relative to `base`
    bool        vIsDir = false;
  };

  struct list_result {
    std::vector<list_entry> lEntries;
    std::string             vErr;                // set when `base` is not a directory
    bool                    vTruncated = false;  // set when the walk could not see everything
  };

  // entries relative to `base`, which must already be resolved (absolute)
  // max_depth == 0 means unlimited, 1 means direct children of `base` only
  virtual list_result mListEntries(const std::string & pBase, int pMaxDepth, list_kind pKind) const   = 0;
  // on_chunk, if set, is called with each chunk of output as it is read (before truncation cuts in);
  // returning false terminates the process early (e.g. the client disconnected)
  virtual exec_result mRun(const std::vector<std::string> &                 pArgs,
                           size_t                                           pMaxOutput,
                           int                                              pTimeoutSecs,
                           const std::function<bool(const std::string &)> & pOnChunk = nullptr) const = 0;
};

// shared subprocess execution helper, used by both the local and the isolate-backed tools_io implementations.
// combine_stderr=false when the raw stdout bytes must not be tainted by stderr, e.g. reading file contents.
static tools_io::exec_result fRunSubprocess(const std::vector<std::string> &                 pArgs,
                                            size_t                                           pMaxOutput,
                                            int                                              pTimeoutSecs,
                                            const std::function<bool(const std::string &)> & pOnChunk,
                                            bool                                             pCombineStderr,
                                            const std::string &                              pCwd       = "",
                                            const std::string *                              pStdinData = nullptr) {
  tools_io::exec_result vRes;

  common_subproc vProc;

  int vOptions =
    subprocess_option_no_window | subprocess_option_inherit_environment | subprocess_option_search_user_path;
  if (pCombineStderr) {
    vOptions |= subprocess_option_combined_stdout_stderr;
  }

  if (!vProc.mCreate(pArgs, vOptions, {}, pCwd.empty() ? nullptr : pCwd.c_str())) {
    vRes.vOutput = "failed to spawn process";
    return vRes;
  }

  std::atomic<bool> vDone{ false };
  std::atomic<bool> vTimedOut{ false };

  std::thread vTimeoutThread([&]() {
    auto vDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(pTimeoutSecs);
    while (!vDone.load()) {
      if (std::chrono::steady_clock::now() >= vDeadline) {
        vTimedOut.store(true);
        vProc.mTerminate();
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  });

  // write stdin before reading stdout, the child drains stdin as it goes
  // always close stdin, a transport client waits forever if its stdin pipe stays open
  if (FILE * vIn = vProc.mStdinFile()) {
    if (pStdinData != nullptr && !pStdinData->empty()) {
#if defined(_WIN32)
      // pipe fds default to CRT text mode: binary keeps the bytes untranslated
      _setmode(_fileno(in), _O_BINARY);
#endif
      // a short write is not an error by itself, the exit code below decides
      fwrite(pStdinData->data(), 1, pStdinData->size(), vIn);
    }
    fflush(vIn);
  }
  vProc.mCloseStdin();

  FILE *      vF = vProc.mStdoutFile();
  std::string vOutput;
  bool        vTruncated = false;
  if (vF) {
#if defined(_WIN32)
    // pipe fds default to CRT text mode: binary keeps the bytes untranslated
    _setmode(_fileno(f), _O_BINARY);
#endif
    // read raw bytes, not lines: the output can hold NUL and must arrive as soon as it is ready
    // keep draining past the size cap, else the child blocks on a full pipe
    char aBuf[4096];
    for (;;) {
#if defined(_WIN32)
      const int n = _read(_fileno(f), buf, (unsigned) sizeof(buf));
#else
      ssize_t n = read(fileno(vF), aBuf, sizeof(aBuf));
      while (n < 0 && errno == EINTR) {
        n = read(fileno(vF), aBuf, sizeof(aBuf));
      }
#endif
      if (n <= 0) {
        break;
      }
      if (vTruncated) {
        continue;
      }
      const size_t cLen = (size_t) n;
      if (vOutput.size() + cLen <= pMaxOutput) {
        vOutput.append(aBuf, cLen);
        if (pOnChunk && !pOnChunk(fConsoleOutputToUtf8(std::string(aBuf, cLen)))) {
          vProc.mTerminate();
          break;
        }
      } else {
        size_t vRemaining = pMaxOutput - vOutput.size();
        vOutput.append(aBuf, vRemaining);
        if (pOnChunk && vRemaining > 0) {
          pOnChunk(fConsoleOutputToUtf8(std::string(aBuf, vRemaining)));
        }
        vTruncated = true;
      }
    }
  }

  vDone.store(true);
  if (vTimeoutThread.joinable()) {
    vTimeoutThread.join();
  }

  vRes.vExitCode = vProc.mJoin();

  vRes.vOutput   = fConsoleOutputToUtf8(vOutput);
  vRes.vTimedOut = vTimedOut.load();
  if (vTruncated) {
    vRes.vOutput += "\n[output truncated]";
  }
  return vRes;
}

class tools_io_basic : public tools_io {
public:
  // cwd, if non-empty, is used to resolve relative paths and as the working directory for run()
  explicit tools_io_basic(std::string pCwd = "") : vCwd(std::move(pCwd)) {}

  // expands a leading `~`, then resolves `path` against `cwd` (or the server
  // working directory when `cwd` is unset); the result is always absolute
  std::string mResolve(const std::string & pPath) const override {
    const std::string cP = fExpandHome(pPath);

    fs::path vFull = fPathFromUtf8(cP);
    if (!vFull.is_absolute()) {
      if (vCwd.empty()) {
        std::error_code vEc;
        const fs::path  cCur = fs::current_path(vEc);
        if (vEc) {
          return cP;
        }
        vFull = cCur / vFull;
      } else {
        vFull = fPathFromUtf8(vCwd) / vFull;
      }
    }

    // drop "." and ".." so they never reach git or the client
    vFull = vFull.lexically_normal();
    // a trailing ".." normalizes to a path that ends with a separator
    if (!vFull.has_filename() && vFull != vFull.root_path()) {
      vFull = vFull.parent_path();
    }
    return fPathToUtf8(vFull);
  }

  bool mIsDirectory(const std::string & pPath) const override {
    std::error_code vEc;
    return fs::is_directory(fPathFromUtf8(mResolve(pPath)), vEc) && !vEc;
  }

  bool mIsRegularFile(const std::string & pPath) const override {
    std::error_code vEc;
    return fs::is_regular_file(fPathFromUtf8(mResolve(pPath)), vEc) && !vEc;
  }

  bool mFileSize(const std::string & pPath, uintmax_t & pOutSize) const override {
    std::error_code vEc;
    pOutSize = fs::file_size(fPathFromUtf8(mResolve(pPath)), vEc);
    return !vEc;
  }

  bool mReadFile(const std::string & pPath, std::string & pOut) const override {
    std::ifstream vF(fPathFromUtf8(mResolve(pPath)), std::ios::binary);
    if (!vF) {
      return false;
    }
    std::ostringstream vSs;
    vSs << vF.rdbuf();
    pOut = vSs.str();
    return true;
  }

  bool mWriteFile(const std::string & pPath, const std::string & pContent) const override {
    std::error_code vEc;
    fs::path        vFpath = fPathFromUtf8(mResolve(pPath));
    if (vFpath.has_parent_path()) {
      fs::create_directories(vFpath.parent_path(), vEc);
      if (vEc) {
        return false;
      }
    }
    std::ofstream vF(vFpath, std::ios::binary);
    if (!vF) {
      return false;
    }
    vF << pContent;
    return (bool) vF;
  }

  list_result mListEntries(const std::string & pBase, int pMaxDepth, list_kind pKind) const override {
    list_result vOut;

    std::error_code vEc;
    if (!fs::is_directory(pBase, vEc) || vEc) {
      vOut.vErr = "path does not exist or is not a directory";
      return vOut;
    }

    const auto cDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(cServerToolListEntriesTimeout);

    // git ls-files cannot list directories; use the walker when they are requested
    if (pKind == list_kind::cFiles) {
      auto vRes = mRun({ "git", "-C", pBase, "ls-files", "--cached", "--others", "--exclude-standard" },
                       cServerToolGitLsFilesMaxOutput, cServerToolListEntriesTimeout);

      if (vRes.vExitCode == 0 && !vRes.vTimedOut) {
        std::istringstream vIss(vRes.vOutput);
        std::string        vLine;
        while (std::getline(vIss, vLine)) {
          if (!vLine.empty() && vLine.back() == '\r') {
            vLine.pop_back();
          }
          if (vLine.empty()) {
            continue;
          }
          std::replace(vLine.begin(), vLine.end(), '\\', '/');
          if (pMaxDepth > 0 && fEntryDepth(vLine) > pMaxDepth) {
            continue;
          }
          if (mIsRegularFile(fPathToUtf8(fPathFromUtf8(pBase) / fPathFromUtf8(vLine)))) {
            vOut.lEntries.push_back({ vLine, false });
          }
        }
        return vOut;
      }
    }

    vOut.lEntries = mListEntriesFallback(pBase, pMaxDepth, pKind, cDeadline, vOut.vTruncated);
    return vOut;
  }

  exec_result mRun(const std::vector<std::string> &                 pArgs,
                   size_t                                           pMaxOutput,
                   int                                              pTimeoutSecs,
                   const std::function<bool(const std::string &)> & pOnChunk = nullptr) const override {
    return fRunSubprocess(pArgs, pMaxOutput, pTimeoutSecs, pOnChunk, /*combine_stderr=*/true, vCwd);
  }

private:
  std::string vCwd;

  // a link can point back to an ancestor and loop forever, so it is never walked
  static bool mIsLink(const fs::directory_entry & pEntry) {
    std::error_code vEc;
    if (pEntry.is_symlink(vEc) || vEc) {
      return true;
    }
#if defined(_WIN32)
    // a junction looks like a plain directory to std::filesystem, so read the reparse tag
    WIN32_FIND_DATAW data;
    const HANDLE     h = FindFirstFileW(entry.path().c_str(), &data);
    if (h == INVALID_HANDLE_VALUE) {
      return false;
    }
    FindClose(h);
    if ((data.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) == 0) {
      return false;
    }
    // other reparse points (cloud placeholder, dedup stub) are real directories
    return data.dwReserved0 == IO_REPARSE_TAG_SYMLINK || data.dwReserved0 == IO_REPARSE_TAG_MOUNT_POINT;
#else
    return false;
#endif
  }

  // NTFS is case insensitive, so Build and build are the same directory
  static std::string mGetEffectiveName(const std::string & pFname) {
#if defined(_WIN32)
    std::string lowered = fname;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return (char) std::tolower(c); });
    return lowered;
#else
    return pFname;
#endif
  }

  static const std::unordered_set<std::string> & mJunkDirNames() {
    static const std::unordered_set<std::string> sNames(std::begin(aServerToolJunkDirNames),
                                                        std::end(aServerToolJunkDirNames));
    return sNames;
  }

  std::vector<list_entry> mListEntriesFallback(const std::string &                   pBase,
                                               int                                   pMaxDepth,
                                               list_kind                             pKind,
                                               std::chrono::steady_clock::time_point pDeadline,
                                               bool &                                pTruncated) const {
    std::vector<list_entry> lResult;

    std::vector<std::tuple<fs::path, fs::path, int>> lStack;
    lStack.emplace_back(fPathFromUtf8(pBase), fs::path(), 0);

    while (!lStack.empty()) {
      if (std::chrono::steady_clock::now() >= pDeadline) {
        pTruncated = true;
        return lResult;
      }

      auto [dir, rel_dir, depth] = std::move(lStack.back());
      lStack.pop_back();

      std::error_code        vEc;
      // step the iterator by hand: the throwing increment escapes on a directory that goes away
      fs::directory_iterator vIt(dir, fs::directory_options::skip_permission_denied, vEc);
      // permission errors are skipped above, so this is a subtree the caller never sees
      if (vEc) {
        pTruncated = true;
        continue;
      }
      for (const fs::directory_iterator end; vIt != end; vIt.increment(vEc)) {
        if (vEc) {
          pTruncated = true;
          break;
        }
        if (std::chrono::steady_clock::now() >= pDeadline) {
          pTruncated = true;
          return lResult;
        }
        const fs::directory_entry & vEntry = *vIt;
        const fs::path              cFname = vEntry.path().filename();
        std::error_code             vTec;
        const bool                  cIsDir = vEntry.is_directory(vTec);
        if (vTec) {
          continue;
        }
        if (cIsDir) {
          if (pKind == list_kind::cDirs || pKind == list_kind::cAll) {
            lResult.push_back({ fPathToUtf8(rel_dir / cFname), true });
          }
          // junk directories stay selectable but are never walked: they can be enormous
          if (mJunkDirNames().count(mGetEffectiveName(fPathToUtf8(cFname))) > 0) {
            continue;
          }
          if (!mIsLink(vEntry) && (pMaxDepth == 0 || depth + 1 < pMaxDepth)) {
            lStack.emplace_back(vEntry.path(), rel_dir / cFname, depth + 1);
          }
        } else if (vEntry.is_regular_file(vTec)) {
          if (pKind == list_kind::cFiles || pKind == list_kind::cAll) {
            lResult.push_back({ fPathToUtf8(rel_dir / cFname), false });
          }
        }
      }
    }

    return lResult;
  }
};

// timeout for auxiliary isolate calls (stat/mkdir/ls helpers); exec_shell_command uses its own
// caller-controlled timeout instead, enforced separately in run()
static constexpr int    cServerToolIsolateExecTimeout     = 15;                // seconds
static constexpr size_t cServerToolIsolateReadFileMaxSize = 64 * 1024 * 1024;  // 64 MB

// runs every tools_io operation as a command inside an isolate: a container, a remote host, ...
// the isolate is created, mounted, and torn down externally by the caller
// it must provide a POSIX environment: sh, cat, wc, mkdir, dirname, find, timeout
class tools_io_isolate : public tools_io {
public:
  // cwd, if non-empty, is used to resolve relative paths and as the working directory for run()
  explicit tools_io_isolate(std::string pCwd = "") : vCwd(std::move(pCwd)) {}

  // resolves `path` against `cwd` if `path` is relative and `cwd` is set; otherwise returns `path` unchanged.
  // isolate paths are always POSIX-style ('/'), regardless of host OS.
  std::string mResolve(const std::string & pPath) const override {
    if (vCwd.empty() || (!pPath.empty() && pPath[0] == '/')) {
      return pPath;
    }
    return vCwd + "/" + pPath;
  }

  bool mIsDirectory(const std::string & pPath) const override { return mShellTest("-d", mResolve(pPath)); }

  bool mIsRegularFile(const std::string & pPath) const override { return mShellTest("-f", mResolve(pPath)); }

  bool mFileSize(const std::string & pPath, uintmax_t & pOutSize) const override {
    auto vRes = mExec({ "sh", "-c", "wc -c < \"$1\"", "_", mResolve(pPath) }, 64, true);
    if (vRes.vExitCode != 0 || vRes.vTimedOut) {
      return false;
    }
    try {
      size_t vPos;
      pOutSize = (uintmax_t) std::stoull(vRes.vOutput, &vPos);
    } catch (...) {
      return false;
    }
    return true;
  }

  bool mReadFile(const std::string & pPath, std::string & pOut) const override {
    // combine_stderr=false: stderr must not be spliced into raw file bytes
    auto vRes = mExec({ "cat", "--", mResolve(pPath) }, cServerToolIsolateReadFileMaxSize, false);
    if (vRes.vExitCode != 0 || vRes.vTimedOut) {
      return false;
    }
    pOut = vRes.vOutput;
    return true;
  }

  bool mWriteFile(const std::string & pPath, const std::string & pContent) const override {
    // the content travels on stdin: no argv for the far side to re-parse, no temp file on the host
    auto vRes =
      fRunSubprocess(mBuildArgv({ "sh", "-c", "mkdir -p \"$(dirname \"$1\")\" && cat > \"$1\"", "_", mResolve(pPath) },
                                /*needs_stdin=*/true),
                     4096, cServerToolIsolateExecTimeout, nullptr, true, "", &pContent);
    return vRes.vExitCode == 0 && !vRes.vTimedOut;
  }

  list_result mListEntries(const std::string & pBase, int pMaxDepth, list_kind pKind) const override {
    list_result vOut;

    const std::string cAbsBase = mResolve(pBase);
    if (!mIsDirectory(pBase)) {
      vOut.vErr = "path does not exist or is not a directory";
      return vOut;
    }

    // git ls-files cannot list directories; use the walker when they are requested
    if (pKind == list_kind::cFiles) {
      auto vRes = mExec({ "sh", "-c", "cd \"$1\" && git ls-files --cached --others --exclude-standard", "_", cAbsBase },
                        cServerToolGitLsFilesMaxOutput, true);

      if (vRes.vExitCode == 0 && !vRes.vTimedOut) {
        for (const auto & vRel : mSplitLines(vRes.vOutput, /*strip_dot_slash=*/false)) {
          if (pMaxDepth > 0 && fEntryDepth(vRel) > pMaxDepth) {
            continue;
          }
          vOut.lEntries.push_back({ vRel, false });
        }
        return vOut;
      }
    }

    if (pKind == list_kind::cDirs || pKind == list_kind::cAll) {
      for (auto & vRel : mFindEntries(cAbsBase, pMaxDepth, /*dirs=*/true, vOut.vTruncated)) {
        vOut.lEntries.push_back({ std::move(vRel), true });
      }
    }
    if (pKind == list_kind::cFiles || pKind == list_kind::cAll) {
      for (auto & vRel : mFindEntries(cAbsBase, pMaxDepth, /*dirs=*/false, vOut.vTruncated)) {
        vOut.lEntries.push_back({ std::move(vRel), false });
      }
    }

    return vOut;
  }

  // wraps the command with an in-isolate `timeout`, since killing the host-side client
  // does not kill the process tree running inside the isolate
  exec_result mRun(const std::vector<std::string> &                 pArgs,
                   size_t                                           pMaxOutput,
                   int                                              pTimeoutSecs,
                   const std::function<bool(const std::string &)> & pOnChunk = nullptr) const override {
    std::vector<std::string> lInner = { "timeout", std::to_string(pTimeoutSecs) + "s" };
    lInner.insert(lInner.end(), pArgs.begin(), pArgs.end());
    // small buffer over timeout_secs so the in-isolate `timeout` has a chance to exit cleanly
    // before the host-side supervisory timeout forcibly kills the client
    return fRunSubprocess(mBuildArgv(mWithCwd(lInner), /*needs_stdin=*/true), pMaxOutput, pTimeoutSecs + 5, pOnChunk,
                          true);
  }

protected:
  // wrap `inner` (a complete POSIX argv) into the host-side argv that runs it in the isolate
  // a transport that re-parses its args in a remote shell (ssh) must join `inner` with shell_quote_join()
  virtual std::vector<std::string> mBuildArgv(const std::vector<std::string> & pInner, bool pNeedsStdin) const = 0;

  // quote `argv` into a single string that a POSIX shell re-parses into exactly `argv`
  static std::string mShellQuoteJoin(const std::vector<std::string> & pArgv) {
    std::string vOut;
    for (const auto & vArg : pArgv) {
      if (!vOut.empty()) {
        vOut += ' ';
      }
      vOut += '\'';
      for (const char cC : vArg) {
        // a single quote cannot be escaped inside single quotes: close, escape, reopen
        if (cC == '\'') {
          vOut += "'\\''";
        } else {
          vOut += cC;
        }
      }
      vOut += '\'';
    }
    return vOut;
  }

private:
  std::string vCwd;

  // set the working directory in the command itself, no `-w` equivalent exists on every transport
  // auxiliary calls do not need this, they use the absolute paths from resolve()
  std::vector<std::string> mWithCwd(const std::vector<std::string> & pInner) const {
    if (vCwd.empty()) {
      return pInner;
    }
    // 127 is what a shell reports for a command it could not run
    std::vector<std::string> lOut = { "sh", "-c", "cd \"$1\" || exit 127; shift; exec \"$@\"", "_", vCwd };
    lOut.insert(lOut.end(), pInner.begin(), pInner.end());
    return lOut;
  }

  exec_result mExec(const std::vector<std::string> & pInner, size_t pMaxOutput, bool pCombineStderr) const {
    return fRunSubprocess(mBuildArgv(pInner, /*needs_stdin=*/false), pMaxOutput, cServerToolIsolateExecTimeout, nullptr,
                          pCombineStderr);
  }

  bool mShellRun(const std::vector<std::string> & pInner) const {
    auto vRes = mExec(pInner, 4096, true);
    return vRes.vExitCode == 0 && !vRes.vTimedOut;
  }

  bool mShellTest(const char * pFlag, const std::string & pPath) const {
    return mShellRun({ "sh", "-c", std::string("[ ") + pFlag + " \"$1\" ]", "_", pPath });
  }

  static std::vector<std::string> mSplitLines(const std::string & pText, bool pStripDotSlash) {
    std::vector<std::string> lResult;
    std::istringstream       vIss(pText);
    std::string              vLine;
    while (std::getline(vIss, vLine)) {
      if (!vLine.empty() && vLine.back() == '\r') {
        vLine.pop_back();
      }
      if (vLine.empty()) {
        continue;
      }
      if (pStripDotSlash && vLine.rfind("./", 0) == 0) {
        vLine = vLine.substr(2);
      }
      std::replace(vLine.begin(), vLine.end(), '\\', '/');
      lResult.push_back(vLine);
    }
    return lResult;
  }

  // one `find` pass in the isolate. junk directories stay selectable but are never descended into,
  // and -mindepth/-maxdepth keep a busybox image working as well as a GNU one
  std::vector<std::string> mFindEntries(const std::string & pAbsBase,
                                        int                 pMaxDepth,
                                        bool                pDirs,
                                        bool &              pTruncated) const {
    std::string vPruneExpr;
    for (const char * n : aServerToolJunkDirNames) {
      if (!vPruneExpr.empty()) {
        vPruneExpr += " -o ";
      }
      vPruneExpr += std::string("-name ") + n;
    }

    std::string vCmd = "cd \"$1\" && find . -mindepth 1";
    if (pMaxDepth > 0) {
      vCmd += " -maxdepth " + std::to_string(pMaxDepth);
    }
    vCmd += " \\( " + vPruneExpr + " \\) -prune";
    vCmd += pDirs ? " -print -o -type d -print" : " -o -type f -print";

    auto vRes  = mExec({ "sh", "-c", vCmd, "_", pAbsBase }, cServerToolGitLsFilesMaxOutput, true);
    pTruncated = pTruncated || vRes.vTimedOut;
    return mSplitLines(vRes.vOutput, /*strip_dot_slash=*/true);
  }
};

// an already-running container, driven through `<engine> exec`
// docker and podman take the same verbs and the same argument order, so one class drives both
class tools_io_container : public tools_io_isolate {
public:
  tools_io_container(std::string pBin, std::string pContainerId, std::string pCwd = "") :
    tools_io_isolate(std::move(pCwd)),
    vBin(std::move(pBin)),
    vContainerId(std::move(pContainerId)) {}

protected:
  std::vector<std::string> mBuildArgv(const std::vector<std::string> & pInner, bool pNeedsStdin) const override {
    std::vector<std::string> lArgv = { vBin, "exec" };
    if (pNeedsStdin) {
      lArgv.push_back("-i");
    }
    lArgv.push_back(vContainerId);
    lArgv.insert(lArgv.end(), pInner.begin(), pInner.end());
    return lArgv;
  }

private:
  std::string vBin;
  std::string vContainerId;
};

// a remote host reached over ssh
// this is remoting, not isolation: the tools can do anything the target account can do
class tools_io_ssh : public tools_io_isolate {
public:
  tools_io_ssh(std::string pTarget, std::string pCwd = "") :
    tools_io_isolate(std::move(pCwd)),
    vTarget(std::move(pTarget)) {}

  // the target can come from a client header, and ssh reads options from its argv
  // a target starting with '-' would become one, e.g. -oProxyCommand=<anything> runs on the host
  static bool mIsValidTarget(const std::string & pTarget) {
    if (pTarget.empty() || pTarget[0] == '-') {
      return false;
    }
    return std::all_of(pTarget.begin(), pTarget.end(), [](unsigned char pC) {
      return std::isalnum(pC) || pC == '.' || pC == '-' || pC == '_' || pC == '@';
    });
  }

protected:
  std::vector<std::string> mBuildArgv(const std::vector<std::string> & pInner, bool pNeedsStdin) const override {
    // the remote shell re-parses the command line, so `inner` travels as one quoted word
    std::vector<std::string> lArgv = mSshArgv();
    if (!pNeedsStdin) {
      lArgv.push_back("-n");
    }
    lArgv.push_back(vTarget);
    lArgv.push_back(mShellQuoteJoin(pInner));
    return lArgv;
  }

private:
  std::string vTarget;

  // there is no console here, so a prompt would hang the tool call
  // key-based auth only, and the admin must trust the host key beforehand
  static std::vector<std::string> mSshArgv() {
    return {
      "ssh",
      "-o",
      "BatchMode=yes",
      "-o",
      "PasswordAuthentication=no",
      "-o",
      "KbdInteractiveAuthentication=no",
      "-o",
      "StrictHostKeyChecking=yes",
    };
  }
};

// "<engine>:<image>" spawns a container and owns it, "<engine>-container:<id>" attaches to one
struct container_runtime_spec {
  std::string vBin;
  std::string vArg;  // image name when spawning, container id when attaching
  bool        vAttach = false;

  static bool mParse(const std::string & pSpec, container_runtime_spec & pOut) {
    // docker and podman take the same verbs, hence a single implementation
    static const char * aEngines[] = { "docker", "podman" };
    for (const char * vBin : aEngines) {
      const std::string cAttachPrefix = std::string(vBin) + "-container:";
      if (pSpec.rfind(cAttachPrefix, 0) == 0) {
        pOut = { vBin, pSpec.substr(cAttachPrefix.size()), true };
        return true;
      }
      const std::string cSpawnPrefix = std::string(vBin) + ":";
      if (pSpec.rfind(cSpawnPrefix, 0) == 0) {
        pOut = { vBin, pSpec.substr(cSpawnPrefix.size()), false };
        return true;
      }
    }
    return false;
  }

  // same risk as the ssh target: an id starting with '-' would become an engine option,
  // e.g. --privileged
  static bool mIsValidId(const std::string & pId) {
    if (pId.empty() || !std::isalnum((unsigned char) pId[0])) {
      return false;
    }
    return std::all_of(pId.begin(), pId.end(),
                       [](unsigned char pC) { return std::isalnum(pC) || pC == '.' || pC == '-' || pC == '_'; });
  }
};

static std::unique_ptr<tools_io> fMakeToolsIo(const json & pParams) {
  std::string vCwd     = fJsonValue(pParams, "cwd", std::string());
  std::string vRuntime = fJsonValue(pParams, "runtime", std::string());
  if (vRuntime.empty()) {
    // an empty runtime runs the tools on the host
    return std::make_unique<tools_io_basic>(vCwd);
  }
  container_runtime_spec vContainer;
  if (container_runtime_spec::mParse(vRuntime, vContainer)) {
    // spawning belongs to the runtime that owns the container, a tool call only attaches
    if (!vContainer.vAttach) {
      throw std::runtime_error("tool runtime must name a running container: " + vRuntime);
    }
    if (!container_runtime_spec::mIsValidId(vContainer.vArg)) {
      throw std::runtime_error("invalid container id: " + vContainer.vArg);
    }
    return std::make_unique<tools_io_container>(vContainer.vBin, vContainer.vArg, vCwd);
  }
  const std::string cSshPrefix = "ssh:";
  if (vRuntime.rfind(cSshPrefix, 0) == 0) {
    std::string vTarget = vRuntime.substr(cSshPrefix.size());
    if (!tools_io_ssh::mIsValidTarget(vTarget)) {
      throw std::runtime_error("invalid ssh target: " + vTarget);
    }
    return std::make_unique<tools_io_ssh>(vTarget, vCwd);
  }
  // do not fall back to the host, the caller asked for an isolate
  throw std::runtime_error("unknown tool runtime: " + vRuntime);
}

// no '/' in pattern -> match basename at any depth; else match full relative path
static bool fPathGlobMatch(const std::string & pPattern, const std::string & pRelPath) {
  if (pPattern.find('/') == std::string::npos) {
    return fGlobMatch(pPattern, fPathToUtf8(fPathFromUtf8(pRelPath).filename()));
  }
  if (pPattern == "**" || pPattern.rfind("**/", 0) == 0 || pPattern.rfind('/', 0) == 0) {
    return fGlobMatch(pPattern, pRelPath);
  }
  return fGlobMatch("**/" + pPattern, pRelPath);
}

//
// read_file: read a file with optional line range and line-number prefix
//

static constexpr size_t cServerToolReadFileMaxSize       = 16 * 1024;         // 16 KB
static constexpr size_t cServerToolReadFileMaxSizeBase64 = 32 * 1024 * 1024;  // 32 MB

struct server_tool_read_file : server_tool {
  server_tool_read_file() {
    name             = "read_file";
    vDisplayName     = "Read file";
    vUsesCwd         = true;
    vPermissionWrite = false;
  }

  json mGetDefinition() const override {
    return {
      { "type",     "function" },
      { "function",
       {
          { "name", name },
          { "description",
            "Read the contents of a file. Optionally specify a 1-based line range. "
            "If append_loc is true, each line is prefixed with its line number (e.g. \"1\u2192...\")." },
          { "parameters",
            {
              { "type", "object" },
              { "properties",
                {
                  { "path", { { "type", "string" }, { "description", "Path to the file" } } },
                  { "start_line",
                    { { "type", "integer" }, { "description", "First line to read, 1-based (default: 1)" } } },
                  { "end_line",
                    { { "type", "integer" },
                      { "description", "Last line to read, 1-based inclusive (default: end of file)" } } },
                  { "append_loc",
                    { { "type", "boolean" }, { "description", "Prefix each line with its line number" } } },
                } },
              { "required", json::mArray({ "path" }) },
            } },
        }                      },
    };
  }

  json mInvoke(json pParams, server_tool::stream *) const override {
    std::string vPath      = pParams.at("path").get<std::string>();
    int         vStartLine = fJsonValue(pParams, "start_line", 1);
    int         vEndLine   = fJsonValue(pParams, "end_line", -1);  // -1 = no limit
    bool        vAppendLoc = fJsonValue(pParams, "append_loc", false);
    // comes from the x-resp-type header, the model cannot ask for it
    bool        vAsBase64  = fJsonValue(pParams, "resp_type", std::string()) == "base64";

    auto vIo = fMakeToolsIo(pParams);

    uintmax_t vFileSize = 0;
    if (!vIo->mFileSize(vPath, vFileSize)) {
      return {
        { "error", "cannot stat file: " + vPath }
      };
    }

    if (vAsBase64) {
      if (vFileSize > cServerToolReadFileMaxSizeBase64) {
        return {
          { "error",
           fStringFormat("file too large (%zu bytes, max %zu)", (size_t) vFileSize, cServerToolReadFileMaxSizeBase64) }
        };
      }
      std::string vContent;
      if (!vIo->mReadFile(vPath, vContent)) {
        return {
          { "error", "failed to open file: " + vPath }
        };
      }
      return {
        { "base64", base64::mEncode(vContent.data(), vContent.size()) },
        { "size_bytes", (size_t) vContent.size() },
      };
    }

    if (vFileSize > cServerToolReadFileMaxSize && vEndLine == -1) {
      return {
        { "error", fStringFormat("file too large (%zu bytes, max %zu). Use start_line/end_line to read a portion.",
         (size_t) vFileSize, cServerToolReadFileMaxSize) }
      };
    }

    std::string vContent;
    if (!vIo->mReadFile(vPath, vContent)) {
      return {
        { "error", "failed to open file: " + vPath }
      };
    }

    std::istringstream vF(vContent);
    std::string        vResult;
    std::string        vLine;
    int                vLineno = 0;

    while (std::getline(vF, vLine)) {
      vLineno++;
      if (vLineno < vStartLine) {
        continue;
      }
      if (vEndLine != -1 && vLineno > vEndLine) {
        break;
      }

      std::string vOutLine;
      if (vAppendLoc) {
        vOutLine = std::to_string(vLineno) + "\u2192" + vLine + "\n";
      } else {
        vOutLine = vLine + "\n";
      }

      if (vResult.size() + vOutLine.size() > cServerToolReadFileMaxSize) {
        vResult += "[output truncated]";
        break;
      }
      vResult += vOutLine;
    }

    return {
      { "plain_text_response", vResult }
    };
  }
};

//
// file_glob_search: find files matching a glob pattern under a base directory
//

static constexpr int          cServerToolFileSearchMaxResults = 100;
static constexpr const char * cServerToolFileSearchTypeFile   = "file";
static constexpr const char * cServerToolFileSearchTypeDir    = "dir";
static constexpr const char * cServerToolFileSearchTypeAll    = "all";

struct server_tool_file_glob_search : server_tool {
  server_tool_file_glob_search() {
    name             = "file_glob_search";
    vDisplayName     = "File search";
    vUsesCwd         = true;
    vPermissionWrite = false;
  }

  json mGetDefinition() const override {
    return {
      { "type",     "function" },
      { "function",
       {
          { "name", name },
          { "description",
            "Recursively search for files matching a glob pattern under a directory. "
            "Automatically skips files ignored by .gitignore (when the directory is inside a git repo) "
            "and common junk directories (.git, node_modules, build, dist, etc.) otherwise. "
            "A pattern with no '/' (e.g. \"*.cpp\") matches the file's basename at any depth. "
            "A pattern containing '/' matches the full relative path; unless already anchored with "
            "\"**/\" or a leading '/', it is automatically prefixed with \"**/\". "
            "Use type=\"dir\" or \"all\" to also list directories; directory entries are suffixed with '/' in the "
            "output. "
            "Note: directory listings do not apply .gitignore filtering." },
          { "parameters",
            {
              { "type", "object" },
              { "properties",
                {
                  { "path", { { "type", "string" }, { "description", "Base directory to search in" } } },
                  { "include",
                    { { "type", "string" },
                      { "description",
                        "Glob pattern for files to include (e.g. \"*.cpp\" or \"src/**/*.cpp\"). Default: **" } } },
                  { "exclude", { { "type", "string" }, { "description", "Glob pattern for files to exclude" } } },
                  { "type",
                    { { "type", "string" },
                      { "description", "Entry type to return: \"file\" (default), \"dir\" or \"all\"" } } },
                  { "max_depth",
                    { { "type", "integer" },
                      { "description",
                        "Maximum depth to descend into subdirectories (default: 0 = unlimited; 1 = direct children "
                        "only)" } } },
                  { "limit",
                    { { "type", "integer" },
                      { "description",
                        fStringFormat("Maximum number of results to return, capped at %d (default %d)",
                                      cServerToolFileSearchMaxResults, cServerToolFileSearchMaxResults) } } },
                } },
              { "required", json::mArray({ "path" }) },
            } },
        }                      },
    };
  }

  json mInvoke(json pParams, server_tool::stream *) const override {
    auto vIo = fMakeToolsIo(pParams);

    const std::string cPath = pParams.at("path").get<std::string>();

    std::string vBase     = vIo->mResolve(cPath);
    std::string vInclude  = fJsonValue(pParams, "include", std::string("**"));
    std::string vExclude  = fJsonValue(pParams, "exclude", std::string(""));
    std::string type      = fJsonValue(pParams, "type", std::string("file"));
    int         vMaxDepth = std::max(0, fJsonValue(pParams, "max_depth", 0));
    const int   cLimitReq = fJsonValue(pParams, "limit", cServerToolFileSearchMaxResults);
    if (cLimitReq < 1) {
      return {
        { "error", "invalid limit: " + std::to_string(cLimitReq) + " (expected 1 or more)" }
      };
    }
    const int cLimit = std::min(cLimitReq, cServerToolFileSearchMaxResults);

    list_kind vKind;
    if (type == cServerToolFileSearchTypeFile) {
      vKind = list_kind::cFiles;
    } else if (type == cServerToolFileSearchTypeDir) {
      vKind = list_kind::cDirs;
    } else if (type == cServerToolFileSearchTypeAll) {
      vKind = list_kind::cAll;
    } else {
      return {
        { "error", "invalid type: " + type + " (expected \"file\", \"dir\" or \"all\")" }
      };
    }

    const auto cListing = vIo->mListEntries(vBase, vMaxDepth, vKind);
    if (!cListing.vErr.empty()) {
      return {
        { "error", cListing.vErr + ": " + cPath }
      };
    }

    std::vector<tools_io::list_entry> lMatches;
    for (const auto & vEntry : cListing.lEntries) {
      if (!fPathGlobMatch(vInclude, vEntry.vRel)) {
        continue;
      }
      if (!vExclude.empty() && fPathGlobMatch(vExclude, vEntry.vRel)) {
        continue;
      }
      lMatches.push_back(vEntry);
    }

    size_t vTotal = lMatches.size();
    size_t vShown = std::min(vTotal, (size_t) cLimit);

    std::ostringstream vOutputText;
    json               vEntriesJson = json::mArray();
    for (size_t vI = 0; vI < vShown; vI++) {
      vOutputText << lMatches[vI].vRel << (lMatches[vI].vIsDir ? "/" : "") << "\n";
      vEntriesJson.push_back({
        { "path", lMatches[vI].vRel                    },
        { "type", lMatches[vI].vIsDir ? "dir" : "file" },
      });
    }

    vOutputText << "\n---\nTotal matches: " << vTotal << "\n";
    if (vTotal > vShown) {
      vOutputText << fStringFormat(
        "[%zu results limit reached (%zu total matches). Refine the glob pattern to narrow the search.]\n", vShown,
        vTotal);
    }
    if (cListing.vTruncated) {
      vOutputText << "[results truncated: time budget or unreadable directory]\n";
    }

    // `base` is always absolute (resolve falls back to the server cwd), so
    // API clients (e.g. the web UI picker) can join the relative entries
    // into absolute paths. `plain_text_response` is what the model sees;
    // `entries` is the same data as structured JSON for the UI picker,
    // which reads `entries`/`base` instead of re-parsing the text.
    return {
      { "plain_text_response", vOutputText.str() },
      { "entries",             vEntriesJson      },
      { "base",                vBase             }
    };
  }
};

//
// grep_search: search for a regex pattern in files
//

static constexpr size_t cServerToolGrepSearchMaxResults = 100;

struct server_tool_grep_search : server_tool {
  server_tool_grep_search() {
    name             = "grep_search";
    vDisplayName     = "Grep search";
    vUsesCwd         = true;
    vPermissionWrite = false;
  }

  json mGetDefinition() const override {
    return {
      { "type",     "function" },
      { "function",
       {
          { "name", name },
          { "description",
            "Search for a pattern in files under a path. Returns matching lines with file paths "
            "(and, unless searching a single file, paths relative to the given directory). "
            "Automatically skips files ignored by .gitignore (when the directory is inside a git repo) "
            "and common junk directories (.git, node_modules, build, dist, etc.) otherwise. "
            "include/exclude: a pattern with no '/' matches the basename at any depth; a pattern "
            "containing '/' matches the full relative path (auto-anchored with \"**/\" unless already anchored)." },
          { "parameters",
            {
              { "type", "object" },
              { "properties",
                {
                  { "path", { { "type", "string" }, { "description", "File or directory to search in" } } },
                  { "pattern",
                    { { "type", "string" },
                      { "description", "Pattern to search for (regular expression unless literal is true)" } } },
                  { "include",
                    { { "type", "string" }, { "description", "Glob pattern to filter files (default: **)" } } },
                  { "exclude", { { "type", "string" }, { "description", "Glob pattern to exclude files" } } },
                  { "return_line_numbers",
                    { { "type", "boolean" }, { "description", "If true, include line numbers in results" } } },
                  { "literal",
                    { { "type", "boolean" },
                      { "description",
                        "Treat pattern as a literal string instead of a regular expression (default: false)" } } },
                  { "ignore_case",
                    { { "type", "boolean" }, { "description", "Case-insensitive search (default: false)" } } },
                  { "context_lines",
                    { { "type", "integer" },
                      { "description",
                        "Number of lines of context to show before and after each match (default: 0)" } } },
                } },
              { "required", json::mArray({ "path", "pattern" }) },
            } },
        }                      },
    };
  }

  json mInvoke(json pParams, server_tool::stream *) const override {
    std::string vPath       = pParams.at("path").get<std::string>();
    std::string vPatStr     = pParams.at("pattern").get<std::string>();
    std::string vInclude    = fJsonValue(pParams, "include", std::string("**"));
    std::string vExclude    = fJsonValue(pParams, "exclude", std::string(""));
    bool        vShowLineno = fJsonValue(pParams, "return_line_numbers", false);
    bool        vLiteral    = fJsonValue(pParams, "literal", false);
    bool        vIgnoreCase = fJsonValue(pParams, "ignore_case", false);
    int         vCtxLines   = std::max(0, fJsonValue(pParams, "context_lines", 0));

    std::string vPatternSrc = vPatStr;
    if (vLiteral) {
      static const std::string cSpecials = "\\^$.|?*+()[]{}";
      std::string              vEscaped;
      vEscaped.reserve(vPatStr.size() * 2);
      for (char vC : vPatStr) {
        if (cSpecials.find(vC) != std::string::npos) {
          vEscaped += '\\';
        }
        vEscaped += vC;
      }
      vPatternSrc = vEscaped;
    }

    std::regex vPattern;
    try {
      auto vFlags = std::regex::ECMAScript;
      if (vIgnoreCase) {
        vFlags |= std::regex::icase;
      }
      vPattern = std::regex(vPatternSrc, vFlags);
    } catch (const std::regex_error & vE) {
      return {
        { "error", std::string("invalid regex: ") + vE.what() }
      };
    }

    auto vIo = fMakeToolsIo(pParams);

    // collect (absolute_path, display_path) pairs to search
    std::vector<std::pair<std::string, std::string>> lFiles;

    const std::string cAbsPath = vIo->mResolve(vPath);
    if (vIo->mIsRegularFile(cAbsPath)) {
      lFiles.emplace_back(cAbsPath, vPath);
    } else if (vIo->mIsDirectory(cAbsPath)) {
      const auto cListing = vIo->mListEntries(cAbsPath, 0, list_kind::cFiles);
      if (!cListing.vErr.empty()) {
        return {
          { "error", cListing.vErr + ": " + vPath }
        };
      }
      for (const auto & vEntry : cListing.lEntries) {
        if (!fPathGlobMatch(vInclude, vEntry.vRel)) {
          continue;
        }
        if (!vExclude.empty() && fPathGlobMatch(vExclude, vEntry.vRel)) {
          continue;
        }
        lFiles.emplace_back(fPathToUtf8(fPathFromUtf8(cAbsPath) / fPathFromUtf8(vEntry.vRel)), vEntry.vRel);
      }
    } else {
      return {
        { "error", "path does not exist: " + vPath }
      };
    }

    std::ostringstream vOutputText;
    size_t             vTotal        = 0;
    bool               vLimitReached = false;
    bool               vShowNum      = vShowLineno || vCtxLines > 0;

    for (const auto & vFileEntry : lFiles) {
      if (vLimitReached) {
        break;
      }
      const std::string & vFpath       = vFileEntry.first;
      const std::string & vDisplayPath = vFileEntry.second;

      std::string vContent;
      if (!vIo->mReadFile(vFpath, vContent)) {
        continue;
      }
      std::vector<std::string> lLines;
      {
        std::istringstream vF(vContent);
        std::string        vLine;
        while (std::getline(vF, vLine)) {
          lLines.push_back(vLine);
        }
      }

      for (size_t vI = 0; vI < lLines.size(); vI++) {
        if (vTotal >= cServerToolGrepSearchMaxResults) {
          vLimitReached = true;
          break;
        }
        if (!std::regex_search(lLines[vI], vPattern)) {
          continue;
        }

        long vCtxStart = vCtxLines > 0 ? std::max<long>(0, (long) vI - vCtxLines) : (long) vI;
        long vCtxEnd   = vCtxLines > 0 ? std::min<long>((long) lLines.size() - 1, (long) vI + vCtxLines) : (long) vI;

        for (long vJ = vCtxStart; vJ <= vCtxEnd; vJ++) {
          bool vIsMatch = (vJ == (long) vI);
          vOutputText << vDisplayPath << (vIsMatch ? ':' : '-');
          if (vShowNum) {
            vOutputText << (vJ + 1) << (vIsMatch ? ':' : '-');
          }
          vOutputText << lLines[vJ] << "\n";
        }
        if (vCtxLines > 0) {
          vOutputText << "--\n";
        }
        vTotal++;
      }
    }

    vOutputText << "\n---\nTotal matches: " << vTotal << "\n";
    if (vLimitReached) {
      vOutputText << fStringFormat("[%zu matches limit reached. Narrow the path/pattern/include to see more.]\n",
                                   cServerToolGrepSearchMaxResults);
    }

    return {
      { "plain_text_response", vOutputText.str() }
    };
  }
};

//
// exec_shell_command: run an arbitrary shell command
//

static constexpr size_t cServerToolExecShellCommandMaxOutputSize = 16 * 1024;  // 16 KB
static constexpr int    cServerToolExecShellCommandMaxTimeout    = 60;         // seconds

struct server_tool_exec_shell_command : server_tool {
  server_tool_exec_shell_command() {
    name             = "exec_shell_command";
    vDisplayName     = "Execute shell command";
    vUsesCwd         = true;
    vPermissionWrite = true;
    vSupportStream   = true;
  }

  json mGetDefinition() const override {
    return {
      { "type",     "function" },
      { "function",
       {
          { "name", name },
          { "description", "Execute a shell command and return its output (stdout and stderr combined)." },
          { "parameters",
            {
              { "type", "object" },
              { "properties",
                {
                  { "command", { { "type", "string" }, { "description", "Shell command to execute" } } },
                  { "timeout",
                    { { "type", "integer" },
                      { "description", fStringFormat("Timeout in seconds (default 10, max %d)",
                                                     cServerToolExecShellCommandMaxTimeout) } } },
                  { "max_output_size",
                    { { "type", "integer" },
                      { "description", fStringFormat("Maximum output size in bytes (default %zu)",
                                                     cServerToolExecShellCommandMaxOutputSize) } } },
                } },
              { "required", json::mArray({ "command" }) },
            } },
        }                      },
    };
  }

  json mInvoke(json pParams, server_tool::stream * pSt) const override {
    std::string vCommand = pParams.at("command").get<std::string>();
    int         vTimeout = fJsonValue(pParams, "timeout", 10);
    size_t vMaxOutput = (size_t) fJsonValue(pParams, "max_output_size", (int) cServerToolExecShellCommandMaxOutputSize);

    vTimeout   = std::min(vTimeout, cServerToolExecShellCommandMaxTimeout);
    vMaxOutput = std::min(vMaxOutput, cServerToolExecShellCommandMaxOutputSize);

    // an isolate is always POSIX regardless of host OS, so it always gets `sh -c`
#ifdef _WIN32
    std::vector<std::string> args = !json_value(params, "runtime", std::string()).empty() ?
                                      std::vector<std::string>{ "sh", "-c", command } :
                                      std::vector<std::string>{ "cmd", "/c", command };
#else
    std::vector<std::string> lArgs = { "sh", "-c", vCommand };
#endif

    auto vIo = fMakeToolsIo(pParams);

    if (pSt) {
      auto vRes = vIo->mRun(lArgs, vMaxOutput, vTimeout, [pSt](const std::string & pChunk) {
        pSt->mPush(pChunk);
        return !pSt->vAlive || pSt->vAlive();
      });
      if (pSt->vAlive && !pSt->vAlive()) {
        return json();
      }
      std::string vTail = fStringFormat("\n[exit code: %d]", vRes.vExitCode);
      if (vRes.vTimedOut) {
        vTail += " [exit due to timed out]";
      }
      pSt->mPush(vTail);
      return json();
    }

    auto vRes = vIo->mRun(lArgs, vMaxOutput, vTimeout);

    std::string vTextOutput = vRes.vOutput;
    vTextOutput += fStringFormat("\n[exit code: %d]", vRes.vExitCode);
    if (vRes.vTimedOut) {
      vTextOutput += " [exit due to timed out]";
    }

    return {
      { "plain_text_response", vTextOutput }
    };
  }
};

//
// write_file: create or overwrite a file
//

struct server_tool_write_file : server_tool {
  server_tool_write_file() {
    name             = "write_file";
    vDisplayName     = "Write file";
    vUsesCwd         = true;
    vPermissionWrite = true;
  }

  json mGetDefinition() const override {
    return {
      { "type",     "function" },
      { "function",
       {
          { "name", name },
          { "description",
            "Write content to a file, creating it (including parent directories) if it does not exist. May use with "
            "edit_file for more complex edits." },
          { "parameters",
            {
              { "type", "object" },
              { "properties",
                {
                  { "path", { { "type", "string" }, { "description", "Path of the file to write" } } },
                  { "content", { { "type", "string" }, { "description", "Content to write" } } },
                } },
              { "required", json::mArray({ "path", "content" }) },
            } },
        }                      },
    };
  }

  json mInvoke(json pParams, server_tool::stream *) const override {
    std::string vPath    = pParams.at("path").get<std::string>();
    std::string vContent = pParams.at("content").get<std::string>();

    auto vIo = fMakeToolsIo(pParams);
    if (!vIo->mWriteFile(vPath, vContent)) {
      return {
        { "error", "failed to write file: " + vPath }
      };
    }

    return {
      { "result", "file written successfully" },
      { "path",   vPath                       },
      { "bytes",  vContent.size()             }
    };
  }
};

//
// edit_file: exact text replacement, one or more edits per call
//

struct server_tool_edit_file : server_tool {
  server_tool_edit_file() {
    name             = "edit_file";
    vDisplayName     = "Edit file";
    vUsesCwd         = true;
    vPermissionWrite = true;
  }

  json mGetDefinition() const override {
    return {
      { "type",     "function" },
      { "function",
       {
          { "name", name },
          { "description",
            "Edit a file using exact text replacement. Each edits[].old_text must be unique in the file "
            "and is matched against the original content, not incrementally. Merge nearby changes into "
            "one edit instead of overlapping edits. Use write_file to replace the whole file." },
          { "parameters",
            {
              { "type", "object" },
              { "properties",
                {
                  { "path", { { "type", "string" }, { "description", "Path to the file to edit" } } },
                  { "edits",
                    {
                      { "type", "array" },
                      { "description", "One or more exact text replacements to apply" },
                      { "items",
                        {
                          { "type", "object" },
                          { "properties",
                            {
                              { "old_text",
                                { { "type", "string" },
                                  { "description",
                                    "Exact text to find; must be unique in the file and must not overlap with other "
                                    "edits" } } },
                              { "new_text",
                                { { "type", "string" }, { "description", "Text to replace old_text with" } } },
                            } },
                          { "required", json::mArray({ "old_text", "new_text" }) },
                        } },
                    } },
                } },
              { "required", json::mArray({ "path", "edits" }) },
            } },
        }                      },
    };
  }

  json mInvoke(json pParams, server_tool::stream *) const override {
    std::string  vPath      = pParams.at("path").get<std::string>();
    const json & vEditsJson = pParams.at("edits");

    if (!vEditsJson.is_array() || vEditsJson.empty()) {
      return {
        { "error", "\"edits\" must be a non-empty array" }
      };
    }

    struct edit_req {
      std::string vOldText;
      std::string vNewText;
    };

    std::vector<edit_req> lEdits;
    lEdits.reserve(vEditsJson.size());
    for (const auto & vE : vEditsJson) {
      edit_req vEr;
      vEr.vOldText = vE.at("old_text").get<std::string>();
      vEr.vNewText = vE.at("new_text").get<std::string>();
      if (vEr.vOldText.empty()) {
        return {
          { "error", fStringFormat("edits[%zu].old_text must not be empty", lEdits.size()) }
        };
      }
      lEdits.push_back(std::move(vEr));
    }

    auto        vIo = fMakeToolsIo(pParams);
    std::string vOriginalContent;
    if (!vIo->mReadFile(vPath, vOriginalContent)) {
      return {
        { "error", "failed to open file: " + vPath }
      };
    }

    // does any old_text need fuzzy matching (no exact match found)?
    bool vAnyFuzzy = false;
    for (size_t vI = 0; vI < lEdits.size(); vI++) {
      if (vOriginalContent.find(lEdits[vI].vOldText) != std::string::npos) {
        continue;
      }
      std::string vFuzzyContent = mNormalizeForFuzzyMatch(vOriginalContent);
      std::string vFuzzyOld     = mNormalizeForFuzzyMatch(lEdits[vI].vOldText);
      if (vFuzzyContent.find(vFuzzyOld) == std::string::npos) {
        return {
          { "error",
           fStringFormat("could not find edits[%zu].old_text in %s, it must match the file's current content exactly",
           vI, vPath.c_str()) }
        };
      }
      vAnyFuzzy = true;
    }

    std::string vBaseContent = vAnyFuzzy ? mNormalizeForFuzzyMatch(vOriginalContent) : vOriginalContent;

    // uniqueness check always uses fuzzy-normalized text, so a whitespace-only duplicate still counts
    std::vector<matched_edit> lMatched;
    lMatched.reserve(lEdits.size());
    for (size_t vI = 0; vI < lEdits.size(); vI++) {
      std::string vNeedle = vAnyFuzzy ? mNormalizeForFuzzyMatch(lEdits[vI].vOldText) : lEdits[vI].vOldText;
      size_t      vOccurrences =
        mCountOccurrences(mNormalizeForFuzzyMatch(vOriginalContent), mNormalizeForFuzzyMatch(lEdits[vI].vOldText));
      if (vOccurrences > 1) {
        return {
          { "error", fStringFormat("found %zu occurrences of edits[%zu].old_text in %s, it must be unique",
           vOccurrences, vI, vPath.c_str()) }
        };
      }
      size_t vIdx = vBaseContent.find(vNeedle);
      lMatched.push_back({ vI, vIdx, vNeedle.size(), lEdits[vI].vNewText });
    }

    std::sort(lMatched.begin(), lMatched.end(),
              [](const matched_edit & pA, const matched_edit & pB) { return pA.vMatchIndex < pB.vMatchIndex; });
    for (size_t vI = 1; vI < lMatched.size(); vI++) {
      if (lMatched[vI - 1].vMatchIndex + lMatched[vI - 1].vMatchLength > lMatched[vI].vMatchIndex) {
        return {
          { "error", fStringFormat(
                       "edits[%zu] and edits[%zu] overlap in %s; merge them into one edit or target disjoint regions", lMatched[vI - 1].vEditIndex, lMatched[vI].vEditIndex, vPath.c_str()) }
        };
      }
    }

    std::string vNewContent = vAnyFuzzy ?
                                mApplyReplacementsPreservingUnchangedLines(vOriginalContent, vBaseContent, lMatched) :
                                mApplyReplacements(vBaseContent, lMatched, 0);

    if (vNewContent == vOriginalContent) {
      return {
        { "error", "no changes made: the replacement(s) produced identical content" }
      };
    }

    if (!vIo->mWriteFile(vPath, vNewContent)) {
      return {
        { "error", "failed to write file: " + vPath }
      };
    }

    return {
      { "result",        "file edited successfully" },
      { "path",          vPath                      },
      { "edits_applied", (int) lMatched.size()      }
    };
  }

private:
  // strip trailing whitespace, normalize smart quotes/dashes/spaces to ASCII
  static std::string mNormalizeLineForFuzzyMatch(const std::string & pLine) {
    size_t end = pLine.size();
    while (end > 0 && (pLine[end - 1] == ' ' || pLine[end - 1] == '\t' || pLine[end - 1] == '\r')) {
      end--;
    }
    std::string vS = pLine.substr(0, end);

    auto vReplaceAll = [](std::string & str, const std::string & pFrom, const std::string & pTo) {
      if (pFrom.empty()) {
        return;
      }
      size_t vPos = 0;
      while ((vPos = str.find(pFrom, vPos)) != std::string::npos) {
        str.replace(vPos, pFrom.size(), pTo);
        vPos += pTo.size();
      }
    };

    // smart single quotes -> '
    for (unsigned char vB : { 0x98, 0x99, 0x9A, 0x9B }) {
      vReplaceAll(vS, std::string("\xE2\x80") + (char) vB, "'");
    }
    // smart double quotes -> "
    for (unsigned char vB : { 0x9C, 0x9D, 0x9E, 0x9F }) {
      vReplaceAll(vS, std::string("\xE2\x80") + (char) vB, "\"");
    }
    // various dashes -> -
    for (unsigned char vB = 0x90; vB <= 0x95; vB++) {
      vReplaceAll(vS, std::string("\xE2\x80") + (char) vB, "-");
    }
    vReplaceAll(vS, "\xE2\x88\x92", "-");  // minus sign
    // special spaces -> ' '
    vReplaceAll(vS, "\xC2\xA0", " ");  // no-break space
    for (unsigned char vB = 0x82; vB <= 0x8A; vB++) {
      vReplaceAll(vS, std::string("\xE2\x80") + (char) vB, " ");
    }
    vReplaceAll(vS, "\xE2\x80\xAF", " ");  // narrow no-break space
    vReplaceAll(vS, "\xE2\x81\x9F", " ");  // medium mathematical space
    vReplaceAll(vS, "\xE3\x80\x80", " ");  // ideographic space

    return vS;
  }

  // applies the per-line transform above to every line; preserves line count/positions
  static std::string mNormalizeForFuzzyMatch(const std::string & pContent) {
    std::string vResult;
    vResult.reserve(pContent.size());
    size_t vStart = 0;
    while (true) {
      size_t      vNl     = pContent.find('\n', vStart);
      bool        vIsLast = vNl == std::string::npos;
      std::string vLine   = vIsLast ? pContent.substr(vStart) : pContent.substr(vStart, vNl - vStart);
      vResult += mNormalizeLineForFuzzyMatch(vLine);
      if (vIsLast) {
        break;
      }
      vResult += '\n';
      vStart = vNl + 1;
    }
    return vResult;
  }

  // lines with trailing '\n' kept, so untouched ones can be reconstructed verbatim
  static std::vector<std::string> mSplitLinesWithEndings(const std::string & pContent) {
    std::vector<std::string> lLines;
    size_t                   vStart = 0;
    while (vStart < pContent.size()) {
      size_t vNl = pContent.find('\n', vStart);
      if (vNl == std::string::npos) {
        lLines.push_back(pContent.substr(vStart));
        break;
      }
      lLines.push_back(pContent.substr(vStart, vNl - vStart + 1));
      vStart = vNl + 1;
    }
    return lLines;
  }

  struct line_span {
    size_t vStart;
    size_t end;
  };

  static std::vector<line_span> mGetLineSpans(const std::string & pContent) {
    std::vector<line_span> lSpans;
    size_t                 vOffset = 0;
    for (const auto & vLine : mSplitLinesWithEndings(pContent)) {
      lSpans.push_back({ vOffset, vOffset + vLine.size() });
      vOffset += vLine.size();
    }
    return lSpans;
  }

  // count non-overlapping occurrences of `needle` in `content`
  static size_t mCountOccurrences(const std::string & pContent, const std::string & pNeedle) {
    if (pNeedle.empty()) {
      return 0;
    }
    size_t count = 0, vPos = 0;
    while ((vPos = pContent.find(pNeedle, vPos)) != std::string::npos) {
      count++;
      vPos += pNeedle.size();
    }
    return count;
  }

  struct matched_edit {
    size_t      vEditIndex;
    size_t      vMatchIndex;  // offset into the "base content" (see below)
    size_t      vMatchLength;
    std::string vNewText;
  };

  // replacements must be sorted ascending by match_index and non-overlapping
  static std::string mApplyReplacements(const std::string &               pContent,
                                        const std::vector<matched_edit> & pReplacements,
                                        size_t                            pOffset) {
    std::string vResult = pContent;
    for (auto lIt = pReplacements.rbegin(); lIt != pReplacements.rend(); ++lIt) {
      size_t vLocalIndex = lIt->vMatchIndex - pOffset;
      vResult = vResult.substr(0, vLocalIndex) + lIt->vNewText + vResult.substr(vLocalIndex + lIt->vMatchLength);
    }
    return vResult;
  }

  // widen a replacement's byte range to the line(s) of `lines` it touches
  static bool mGetReplacementLineRange(const std::vector<line_span> & pLines,
                                       size_t                         pMatchIndex,
                                       size_t                         pMatchLength,
                                       size_t &                       pOutStartLine,
                                       size_t &                       pOutEndLine /* exclusive */) {
    size_t vReplacementStart = pMatchIndex;
    size_t vReplacementEnd   = pMatchIndex + pMatchLength;

    size_t vStartLine = (size_t) -1;
    for (size_t vI = 0; vI < pLines.size(); vI++) {
      if (vReplacementStart >= pLines[vI].vStart && vReplacementStart < pLines[vI].end) {
        vStartLine = vI;
        break;
      }
    }
    if (vStartLine == (size_t) -1) {
      return false;
    }

    size_t vEndLine = vStartLine;
    while (vEndLine < pLines.size() && pLines[vEndLine].end < vReplacementEnd) {
      vEndLine++;
    }
    if (vEndLine >= pLines.size()) {
      return false;
    }

    pOutStartLine = vStartLine;
    pOutEndLine   = vEndLine + 1;
    return true;
  }

  // like apply_replacements, but untouched lines come from `original_content`
  static std::string mApplyReplacementsPreservingUnchangedLines(
    const std::string &               pOriginalContent,
    const std::string &               pBaseContent,
    const std::vector<matched_edit> & pReplacements /* ascending, non-overlapping */) {
    auto lOriginalLines = mSplitLinesWithEndings(pOriginalContent);
    auto lBaseLines     = mGetLineSpans(pBaseContent);

    struct group {
      size_t                    vStartLine;
      size_t                    vEndLine;  // exclusive
      std::vector<matched_edit> lReps;
    };

    std::vector<group> lGroups;

    for (const auto & vRep : pReplacements) {
      size_t vStartLine = 0, vEndLine = 0;
      mGetReplacementLineRange(lBaseLines, vRep.vMatchIndex, vRep.vMatchLength, vStartLine, vEndLine);
      if (!lGroups.empty() && vStartLine < lGroups.back().vEndLine) {
        lGroups.back().vEndLine = std::max(lGroups.back().vEndLine, vEndLine);
        lGroups.back().lReps.push_back(vRep);
      } else {
        lGroups.push_back({ vStartLine, vEndLine, { vRep } });
      }
    }

    size_t      vOriginalLineIndex = 0;
    std::string vResult;
    for (auto & vG : lGroups) {
      for (size_t vI = vOriginalLineIndex; vI < vG.vStartLine; vI++) {
        vResult += lOriginalLines[vI];
      }

      size_t      vGroupStartOffset = lBaseLines[vG.vStartLine].vStart;
      size_t      vGroupEndOffset   = lBaseLines[vG.vEndLine - 1].end;
      std::string vSlice            = pBaseContent.substr(vGroupStartOffset, vGroupEndOffset - vGroupStartOffset);
      vResult += mApplyReplacements(vSlice, vG.lReps, vGroupStartOffset);

      vOriginalLineIndex = vG.vEndLine;
    }
    for (size_t vI = vOriginalLineIndex; vI < lOriginalLines.size(); vI++) {
      vResult += lOriginalLines[vI];
    }

    return vResult;
  }
};

//
// get_info: returns runtime info (OS name/version and cwd)
//

static constexpr size_t cServerToolGetInfoMaxOutput = 4096;
static constexpr int    cServerToolGetInfoTimeout   = 5;  // seconds

struct server_tool_get_info : server_tool {
  server_tool_get_info() {
    name             = "get_info";
    vDisplayName     = "Get Runtime Info";
    vUsesCwd         = true;
    vPermissionWrite = false;
  }

  json mGetDefinition() const override {
    return {
      { "type",     "function" },
      { "function",
       {
          { "name", name },
          { "description", "Returns runtime info: the OS name/version and the current working directory" },
          { "parameters",
            {
              { "type", "object" },
              { "properties", json::mObject() },
            } },
        }                      },
    };
  }

  json mInvoke(json pParams, server_tool::stream *) const override {
    auto vIo = fMakeToolsIo(pParams);

    // inside an isolate, we always use the linux command
#ifdef _WIN32
    std::vector<std::string> args = !json_value(params, "runtime", std::string()).empty() ?
                                      std::vector<std::string>{ "uname", "-a" } :
                                      std::vector<std::string>{ "cmd", "/c", "ver" };
#else
    std::vector<std::string> lArgs = { "uname", "-a" };
#endif

    auto        vRes    = vIo->mRun(lArgs, cServerToolGetInfoMaxOutput, cServerToolGetInfoTimeout);
    // "ver" prints a blank line before the version, so the output is stripped on both ends;
    // a failed spawn or a timeout leaves a diagnostic in res.output, which is not an OS name
    std::string vOsInfo = vRes.vExitCode == 0 && !vRes.vTimedOut ? fStringStrip(vRes.vOutput) : "unknown";

    std::string vCwd = fJsonValue(pParams, "cwd", std::string());
    if (vCwd.empty()) {
      if (fJsonValue(pParams, "runtime", std::string()).empty()) {
        std::error_code vEc;
        vCwd = fPathToUtf8(fs::current_path(vEc));
      } else {
        auto vPwd = vIo->mRun({ "pwd" }, cServerToolGetInfoMaxOutput, cServerToolGetInfoTimeout);
        vCwd      = vPwd.vExitCode == 0 && !vPwd.vTimedOut ? fStringStrip(vPwd.vOutput) : "unknown";
      }
    }

    return {
      { "os",  vOsInfo },
      { "cwd", vCwd    },
    };
  }
};

struct server_tool_stream_result : server_task_result {
  std::string vChunk;
  bool        vDone = false;
  std::string vErrorMsg;

  json to_json() override {
    if (!vDone) {
      return {
        { "chunk", vChunk }
      };
    } else {
      json vResult = {
        { "done", true }
      };
      if (!vErrorMsg.empty()) {
        vResult["error"] = vErrorMsg;
      }
      return vResult;
    }
  }
};

void server_tool::stream::mPush(const std::string & pChunk) {
  if (pChunk.empty()) {
    return;
  }
  auto vR    = std::make_unique<server_tool_stream_result>();
  vR->vId    = vId;
  vR->vChunk = pChunk;
  vQr.mSend(std::move(vR));
}

struct server_tools_res : server_http_res {
  std::thread       vWorker;
  server_response * vQr = nullptr;  // set only for streaming responses
  int               vId = -1;

  ~server_tools_res() override {
    if (vWorker.joinable()) {
      vWorker.join();
    }
    if (vQr) {
      vQr->mRemoveWaitingTaskId(vId);
    }
  }
};

//
// server_mcp_tool: exposes one tool from a running MCP server as a server_tool.
//
struct server_mcp_tool : server_tool {
  std::string         vServerName;
  std::string         vToolName;
  server_mcp_tool_def vDef;
  server_mcp &        vMcpMgr;

  server_mcp_tool(server_mcp_tool_def pD, server_mcp & pMgr) :
    vServerName(pD.vServerName),
    vToolName(pD.name),
    vDef(std::move(pD)),
    vMcpMgr(pMgr) {
    name             = vServerName + "_" + vToolName;
    vDisplayName     = name;
    vPermissionWrite = false;
    vSupportStream   = false;
  }

  std::string type() const override { return "mcp"; }

  json mGetDefinition() const override {
    json vSchema = vDef.vInputSchema;
    if (vSchema.is_null() || !vSchema.is_object()) {
      vSchema = json::mObject();
    }
    return {
      { "type",     "function" },
      { "function",
       {
          { "name", name },
          { "description", vDef.vDescription },
          { "parameters", vSchema },
        }                      },
    };
  }

  json mInvoke(json pParams, server_tool::stream *) const override {
    return vMcpMgr.mCallTool(vServerName, vToolName, pParams);
  }
};

// resolves --tools-runtime into the isolate that every tool call runs through
// spec() returns the runtime string make_tools_io() takes, and runs once per tool call
struct server_tools_runtime {
  virtual ~server_tools_runtime() = default;
  virtual std::string mSpec()     = 0;
};

// a target that already exists and needs no lifecycle
// the spec is validated once at startup, then passed straight through
struct server_tools_static_runtime : server_tools_runtime {
  explicit server_tools_static_runtime(std::string pSpec) : vRuntimeSpec(std::move(pSpec)) {}

  std::string mSpec() override { return vRuntimeSpec; }

private:
  std::string vRuntimeSpec;
};

// owns the container the tools run in, as set by --tools-runtime "<engine>:<image>"
// it is spawned here and stopped when the server exits
struct server_tools_container_runtime : server_tools_runtime {
  server_tools_container_runtime(const server_tools_container_runtime &) = delete;

  explicit server_tools_container_runtime(const std::string & pSpec) {
    container_runtime_spec vParsed;
    if (!container_runtime_spec::mParse(pSpec, vParsed)) {
      throw std::runtime_error("unknown --tools-runtime option: " + pSpec);
    }

    vBin   = vParsed.vBin;
    vImage = vParsed.vArg;
    if (vImage.empty()) {
      throw std::runtime_error("--tools-runtime " + vBin + ":<image> requires an image name");
    }
    mSpawn();
  }

  ~server_tools_container_runtime() override {
    // closing stdin signals the container's shell (its pid 1) to exit; --rm then removes it
    vProc.mCloseStdin();
    vProc.mJoin();
  }

  // respawns a container that died on its own, so the returned spec always names a running one
  std::string mSpec() override {
    std::lock_guard<std::mutex> vLock(vMutex);
    if (!vProc.mAlive()) {
      SRV_WRN("%s tools runtime container \"%s\" died, respawning\n", vBin.c_str(), vContainerId.c_str());
      mSpawn();
    }
    return vBin + "-container:" + vContainerId;
  }

private:
  std::string    vBin;
  std::string    vImage;
  std::string    vContainerId;
  common_subproc vProc;  // `<engine> run` client that keeps the container alive
  std::mutex     vMutex;

  // spawns "<engine> run --rm -i <image> sh" and keeps its stdin open; the shell blocks reading stdin,
  // so the container stays alive until we close it (see destructor) or it is killed from the outside
  void mSpawn() {
    // create() writes over the handle it is given, so the previous one is released first
    vProc.mJoin();

    std::error_code vEc;
    fs::path        vCidfile =
      fs::temp_directory_path(vEc) /
      fStringFormat("dgs-tools-runtime-cid-%zu.tmp", std::hash<std::thread::id>{}(std::this_thread::get_id()));
    fs::remove(vCidfile, vEc);

    std::vector<std::string> lArgs = { vBin, "run", "--rm", "-i", "--cidfile", fPathToUtf8(vCidfile), vImage, "sh" };
    int                      vOptions =
      subprocess_option_no_window | subprocess_option_inherit_environment | subprocess_option_search_user_path;
    if (!vProc.mCreate(lArgs, vOptions)) {
      throw std::runtime_error("failed to spawn " + vBin + " container for tools runtime (image: " + vImage + ")");
    }

    std::string vCid;
    for (int vI = 0; vI < 100 && vCid.empty(); vI++) {
      std::ifstream vF(vCidfile);
      if (vF) {
        std::getline(vF, vCid);
      }
      if (vCid.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }
    fs::remove(vCidfile, vEc);
    if (vCid.empty()) {
      vProc.mTerminate();
      throw std::runtime_error("timed out waiting for " + vBin + " container to start (image: " + vImage + ")");
    }
    vContainerId = vCid;
  }
};

static server_tool & fFindTool(std::vector<std::unique_ptr<server_tool>> & pTools,
                               const std::string &                         name,
                               bool                                        pRequireStream) {
  for (auto & vT : pTools) {
    if (vT->name == name) {
      if (pRequireStream && !vT->vSupportStream) {
        throw std::invalid_argument(fStringFormat("tool \"%s\" does not support stream = true", name.c_str()));
      }
      return *vT;
    }
  }
  throw std::invalid_argument(fStringFormat("unknown tool \"%s\"", name.c_str()));
}

//
// public API
//

static std::vector<std::unique_ptr<server_tool>> fBuildTools() {
  // IMPORTANT: for contributors, please keep this array of tools as minimal as possible
  //            we only accept minimal i/o and shell command tools here
  //            for example, do not add: web search, get date time, etc.
  //            high-level functionality should be added either via MCP or web UI
  std::vector<std::unique_ptr<server_tool>> lTools;
  lTools.push_back(std::make_unique<server_tool_read_file>());
  lTools.push_back(std::make_unique<server_tool_file_glob_search>());
  lTools.push_back(std::make_unique<server_tool_grep_search>());
  lTools.push_back(std::make_unique<server_tool_exec_shell_command>());
  lTools.push_back(std::make_unique<server_tool_write_file>());
  lTools.push_back(std::make_unique<server_tool_edit_file>());
  lTools.push_back(std::make_unique<server_tool_get_info>());
  return lTools;
}

static std::string fStrToLower(const std::string & value) {
  std::string vLowered(value.size(), '\0');
  std::transform(value.begin(), value.end(), vLowered.begin(), [](unsigned char pC) { return std::tolower(pC); });
  return vLowered;
}

static std::string fGetHeader(const std::map<std::string, std::string> & pHeaders,
                              const std::string &                        pKey,
                              std::string                                pDefaultValue = "") {
  const auto cLoweredKey = fStrToLower(pKey);
  for (const auto & vH : pHeaders) {
    if (fStrToLower(vH.first) == cLoweredKey) {
      return vH.second;
    }
  }
  return pDefaultValue;
}

server_tools::server_tools()  = default;
server_tools::~server_tools() = default;

// the "<engine>:<image>" form owns a container lifecycle
// anything else names an existing target, so only its spec is validated here at startup
static std::unique_ptr<server_tools_runtime> fMakeToolsRuntime(const std::string & pSpec) {
  container_runtime_spec vParsed;
  if (container_runtime_spec::mParse(pSpec, vParsed) && !vParsed.vAttach) {
    return std::make_unique<server_tools_container_runtime>(pSpec);
  }
  fMakeToolsIo({
    { "runtime", pSpec }
  });  // nothing to own, just reject a bad spec now
  return std::make_unique<server_tools_static_runtime>(pSpec);
}

void server_tools::mSetup(const std::vector<std::string> & pEnabledTools,
                          server_mcp &                     pMcpMgr,
                          const std::string &              pToolsRuntime) {
  if (!pToolsRuntime.empty()) {
    vRuntime = fMakeToolsRuntime(pToolsRuntime);
  }

  if (!pEnabledTools.empty()) {
    if (!common_subproc::mIsSupported()) {
      throw std::runtime_error("subprocess is not enabled on this build");
    }

    std::unordered_set<std::string> sEnabledSet(pEnabledTools.begin(), pEnabledTools.end());
    auto                            lAllTools = fBuildTools();

    // collect all known tool names for validation
    std::vector<std::string> lKnownNames;
    lKnownNames.reserve(lAllTools.size());
    for (const auto & vT : lAllTools) {
      lKnownNames.push_back(vT->name);
    }

    // validate that every requested tool is known
    for (const auto & name : pEnabledTools) {
      if (name == "all") {
        continue;
      }
      if (std::find(lKnownNames.begin(), lKnownNames.end(), name) == lKnownNames.end()) {
        throw std::runtime_error(fStringFormat("unknown tool \"%s\". available tools: %s", name.c_str(),
                                               string_join(lKnownNames, ", ").c_str()));
      }
    }

    lTools.clear();
    for (auto & vT : lAllTools) {
      if (sEnabledSet.count(vT->name) > 0 || sEnabledSet.count("all") > 0) {
        lTools.push_back(std::move(vT));
      }
    }
  }

  // append MCP tools, skipping any that collide with a server tool or another MCP tool of the same "<server>_<tool>" name
  if (!pMcpMgr.empty()) {
    std::unordered_set<std::string> sSeenNames;
    for (auto & vT : lTools) {
      sSeenNames.insert(vT->name);
    }
    size_t vNAdded = 0;
    for (const auto & vDef : pMcpMgr.mListTools()) {
      std::string vMcpName = vDef.vServerName + "_" + vDef.name;
      if (sSeenNames.count(vMcpName)) {
        SRV_WRN("MCP tool \"%s\" from server \"%s\" collides with an existing tool, skipping\n", vMcpName.c_str(),
                vDef.vServerName.c_str());
        continue;
      }
      sSeenNames.insert(vMcpName);
      lTools.push_back(std::make_unique<server_mcp_tool>(vDef, pMcpMgr));
      vNAdded++;
    }
    if (vNAdded > 0) {
      SRV_INF("Added %zu MCP tools\n", vNAdded);
    }
  }

  vHandleGet = [this](const server_http_req &) -> server_http_res_ptr {
    auto vRes = std::make_unique<server_http_res>();
    try {
      json vResult = json::mArray();
      for (const auto & vT : lTools) {
        vResult.push_back(vT->to_json());
      }
      vRes->data = fSafeJsonToStr(vResult);
    } catch (const std::exception & vE) {
      SRV_ERR("got exception: %s\n", vE.what());
      vRes->vStatus = 500;
      vRes->data    = fSafeJsonToStr(fFormatErrorResponse(vE.what(), cErrorTypeServer));
    }
    return vRes;
  };

  vHandlePost = [this](const server_http_req & pReq) -> server_http_res_ptr {
    auto vRes = std::make_unique<server_tools_res>();
    try {
      json        vBody     = json::mParse(pReq.vBody);
      std::string vToolName = vBody.at("tool").get<std::string>();
      json        vParams   = vBody.value("params", json::mObject());
      bool        stream    = vBody.value("stream", false);

      // accept x-tool-cwd header to override of the process
      if (vParams.contains("cwd")) {
        vParams.erase("cwd");
      }
      auto vCwd = fGetHeader(pReq.dHeaders, "x-tool-cwd");
      if (!vCwd.empty()) {
        vParams["cwd"] = vCwd;
      }

      // accept x-tool-runtime header to route tool I/O through an isolate, e.g. "docker-container:<id>";
      // falls back to the --tools-runtime isolate, if configured
      if (vParams.contains("runtime")) {
        vParams.erase("runtime");
      }
      auto vRuntimeHeader = fGetHeader(pReq.dHeaders, "x-tool-runtime");
      if (!vRuntimeHeader.empty()) {
        vParams["runtime"] = vRuntimeHeader;
      } else if (vRuntime) {
        vParams["runtime"] = vRuntime->mSpec();
      }

      // x-resp-type header is only used by read_file for now
      if (vParams.contains("resp_type")) {
        vParams.erase("resp_type");
      }
      auto vRespType = fGetHeader(pReq.dHeaders, "x-resp-type");
      if (!vRespType.empty()) {
        vParams["resp_type"] = vRespType;
      }

      server_tool & vTool = fFindTool(lTools, vToolName, stream);

      if (stream) {
        int vId = vResId.fetch_add(1);
        vQueueRes.mAddWaitingTaskId(vId);
        vRes->vQr = &vQueueRes;
        vRes->vId = vId;

        vRes->vWorker = std::thread([this, vId, &pReq, &vTool, vParams]() mutable {
          server_tool::stream vSt{ vQueueRes, vId, [&pReq]() { return !pReq.vShouldStop(); } };

          auto vDone = std::make_unique<server_tool_stream_result>();
          try {
            vTool.mInvoke(vParams, &vSt);
          } catch (const std::exception & vE) {
            vDone->vErrorMsg = vE.what();
          } catch (...) {
            vDone->vErrorMsg = "An unknown error occurred";
          }
          vDone->vId   = vSt.vId;
          vDone->vDone = true;
          vSt.vQr.mSend(std::move(vDone));
        });

        vRes->vContentType = "text/event-stream";
        vRes->vStatus      = 200;
        vRes->vNext        = [this, vId](std::string & pOutput) -> bool {
          auto   vResult = vQueueRes.mRecv(vId);
          auto * vR      = dynamic_cast<server_tool_stream_result *>(vResult.get());
          GGML_ASSERT(vR != nullptr);
          pOutput = "data: " + fSafeJsonToStr(vR->to_json()) + "\n\n";
          if (vR->vDone) {
            vQueueRes.mRemoveWaitingTaskId(vId);
            return false;
          }
          return true;
        };
      } else {
        json vResult  = vTool.mInvoke(vParams, nullptr);
        vRes->vStatus = 200;
        vRes->data    = fSafeJsonToStr(vResult);
      }
    } catch (const common_json_error & vE) {
      vRes->vStatus = 400;
      vRes->data    = fSafeJsonToStr(fFormatErrorResponse(vE.what(), cErrorTypeInvalidRequest));
    } catch (const std::invalid_argument & vE) {
      vRes->vStatus = 404;
      vRes->data    = fSafeJsonToStr(fFormatErrorResponse(vE.what(), cErrorTypeInvalidRequest));
    } catch (const std::exception & vE) {
      SRV_ERR("got exception: %s\n", vE.what());
      vRes->vStatus = 500;
      vRes->data    = fSafeJsonToStr(fFormatErrorResponse(vE.what(), cErrorTypeServer));
    }
    return vRes;
  };
}
