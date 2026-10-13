#include "subproc.h"

bool common_subproc::mIsSupported() {
#ifdef DGS_SUBPROCESS
  return true;
#else
  return false;
#endif
}

#ifdef DGS_SUBPROCESS

static std::vector<char *> fToCstrVec(const std::vector<std::string> & pV) {
  std::vector<char *> lR;
  lR.reserve(pV.size() + 1);
  for (const auto & vS : pV) {
    lR.push_back(const_cast<char *>(vS.c_str()));
  }
  lR.push_back(nullptr);
  return lR;
}

common_subproc::~common_subproc() {
  if (vIsCreated) {
    subprocess_destroy(&vProc);
    vIsCreated = false;
  }
}

bool common_subproc::mCreate(const std::vector<std::string> & pArgs,
                             int                              pOptions,
                             const std::vector<std::string> & pEnv,
                             const char *                     pCwd) {
  auto lArgv = fToCstrVec(pArgs);

  int vResult;
  if (pEnv.empty() && pCwd == nullptr) {
    vResult = subprocess_create(lArgv.data(), pOptions, &vProc);
  } else {
    auto lEnvp = fToCstrVec(pEnv);
    vResult    = subprocess_create_ex(lArgv.data(), pOptions, pEnv.empty() ? nullptr : lEnvp.data(), pCwd, &vProc);
  }

  vIsCreated = vResult == 0;
  return vIsCreated;
}

bool common_subproc::mHasHandle() const {
  if (!vIsCreated) {
    return false;
  }
#  if defined(_WIN32)
  return proc.hProcess != nullptr;
#  else
  return vProc.child > 0;
#  endif
}

bool common_subproc::mAlive() {
  return vIsCreated && subprocess_alive(&vProc);
}

FILE * common_subproc::mStdinFile() {
  return vIsCreated ? subprocess_stdin(&vProc) : nullptr;
}

FILE * common_subproc::mStdoutFile() {
  return vIsCreated ? subprocess_stdout(&vProc) : nullptr;
}

FILE * common_subproc::mStderrFile() {
  return vIsCreated ? subprocess_stderr(&vProc) : nullptr;
}

void common_subproc::mCloseStdin() {
  if (vIsCreated && vProc.stdin_file) {
    fclose(vProc.stdin_file);
    vProc.stdin_file = nullptr;
  }
}

void common_subproc::mTerminate() {
  if (mHasHandle()) {
    subprocess_terminate(&vProc);
  }
}

int common_subproc::mJoin() {
  int vExitCode = -1;
  if (vIsCreated) {
    subprocess_join(&vProc, &vExitCode);
    subprocess_destroy(&vProc);
    vIsCreated = false;
  }
  return vExitCode;
}

#else   // !DGS_SUBPROCESS

common_subproc::~common_subproc() = default;

bool common_subproc::create(const std::vector<std::string> &, int, const std::vector<std::string> &, const char *) {
  (void) (proc);
  (void) (is_created);
  return false;
}

bool common_subproc::has_handle() const {
  return false;
}

bool common_subproc::alive() {
  return false;
}

FILE * common_subproc::stdin_file() {
  return nullptr;
}

FILE * common_subproc::stdout_file() {
  return nullptr;
}

FILE * common_subproc::stderr_file() {
  return nullptr;
}

void common_subproc::close_stdin() {}

void common_subproc::terminate() {}

int common_subproc::join() {
  return -1;
}

#endif  // DGS_SUBPROCESS
