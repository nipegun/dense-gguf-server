#include "common.h"

#include "../src/dgs-ext.h"
#include "build-info.h"
#include "dgs.h"
#include "fit.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "gguf.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "unicode.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#if defined(__APPLE__) && defined(__MACH__)
#  include <sys/sysctl.h>
#  include <sys/types.h>
#endif

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <fcntl.h>
#  include <io.h>
#  include <string.h>
#  include <windows.h>

#  include <locale>
#else
#  include <sys/ioctl.h>
#  include <unistd.h>
#endif

#if !defined(_WIN32)
#  include <pwd.h>
#  include <sys/types.h>
#endif

#if defined(_AIX)
#  include <sys/systemcfg.h>
#endif

#if defined(_MSC_VER)
#  pragma warning(disable : 4244 4267)  // possible loss of data
#endif

common_time_meas::common_time_meas(int64_t & pTAcc, bool pDisable) :
  cTStartUs(pDisable ? -1 : ggml_time_us()),
  vTAcc(pTAcc) {}

common_time_meas::~common_time_meas() {
  if (cTStartUs >= 0) {
    vTAcc += ggml_time_us() - cTStartUs;
  }
}

//
// CPU utils
//

int32_t fCommonCpuGetNumPhysicalCores() {
#if defined(_AIX)
  int32_t logical_cpus = _system_configuration.ncpus;
  int32_t smt_threads  = _system_configuration.smt_threads;
  if (smt_threads > 0) {
    return static_cast<int32_t>(logical_cpus / smt_threads);
  }
  if (logical_cpus > 0) {
    return static_cast<int32_t>(logical_cpus);
  }
#elif defined(__linux__)
  // enumerate the set of thread siblings, num entries is num cores
  std::unordered_set<std::string> sSiblings;
  for (uint32_t vCpu = 0; vCpu < UINT32_MAX; ++vCpu) {
    std::ifstream vThreadSiblings("/sys/devices/system/cpu/cpu" + std::to_string(vCpu) + "/topology/thread_siblings");
    if (!vThreadSiblings.is_open()) {
      break;  // no more cpus
    }
    std::string vLine;
    if (std::getline(vThreadSiblings, vLine)) {
      sSiblings.insert(vLine);
    }
  }
  if (!sSiblings.empty()) {
    return static_cast<int32_t>(sSiblings.size());
  }
#elif defined(__APPLE__) && defined(__MACH__)
  int32_t num_physical_cores;
  size_t  len    = sizeof(num_physical_cores);
  int     result = sysctlbyname("hw.perflevel0.physicalcpu", &num_physical_cores, &len, NULL, 0);
  if (result == 0) {
    return num_physical_cores;
  }
  result = sysctlbyname("hw.physicalcpu", &num_physical_cores, &len, NULL, 0);
  if (result == 0) {
    return num_physical_cores;
  }
#elif defined(_WIN32) && (_WIN32_WINNT >= 0x0601) && !defined(__MINGW64__)  // windows 7 and later
  // TODO: windows + arm64 + mingw64
  unsigned int n_threads_win   = std::thread::hardware_concurrency();
  unsigned int default_threads = n_threads_win > 0 ? (n_threads_win <= 4 ? n_threads_win : n_threads_win / 2) : 4;

  DWORD buffer_size = 0;
  if (!GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &buffer_size)) {
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
      return default_threads;
    }
  }

  std::vector<char> buffer(buffer_size);
  if (!GetLogicalProcessorInformationEx(RelationProcessorCore,
                                        reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()),
                                        &buffer_size)) {
    return default_threads;
  }

  int32_t                                  num_physical_cores = 0;
  PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info =
    reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
  while (buffer_size > 0) {
    if (info->Relationship == RelationProcessorCore) {
      num_physical_cores += info->Processor.GroupCount;
    }
    buffer_size -= info->Size;
    info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(reinterpret_cast<char *>(info) + info->Size);
  }

  return num_physical_cores > 0 ? num_physical_cores : default_threads;
#endif
  unsigned int vNThreads = std::thread::hardware_concurrency();
  return vNThreads > 0 ? (vNThreads <= 4 ? vNThreads : vNThreads / 2) : 4;
}

#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
#  include <pthread.h>

static void fCpuid(unsigned   pLeaf,
                   unsigned   pSubleaf,
                   unsigned * pEax,
                   unsigned * pEbx,
                   unsigned * pEcx,
                   unsigned * pEdx) {
  __asm__(
    "movq\t%%rbx,%%rsi\n\t"
    "cpuid\n\t"
    "xchgq\t%%rbx,%%rsi"
    : "=a"(*pEax), "=S"(*pEbx), "=c"(*pEcx), "=d"(*pEdx)
    : "0"(pLeaf), "2"(pSubleaf));
}

static int fPinCpu(int pCpu) {
  cpu_set_t vMask;
  CPU_ZERO(&vMask);
  CPU_SET(pCpu, &vMask);
  return pthread_setaffinity_np(pthread_self(), sizeof(vMask), &vMask);
}

static bool fIsHybridCpu(void) {
  unsigned vEax, vEbx, vEcx, vEdx;
  fCpuid(7, 0, &vEax, &vEbx, &vEcx, &vEdx);
  return !!(vEdx & (1u << 15));
}

static bool fIsRunningOnEfficiencyCore(void) {
  unsigned vEax, vEbx, vEcx, vEdx;
  fCpuid(0x1a, 0, &vEax, &vEbx, &vEcx, &vEdx);
  int vIntelAtom = 0x20;
  int vCoreType  = (vEax & 0xff000000u) >> 24;
  return vCoreType == vIntelAtom;
}

static int fCpuCountMathCpus(int pNCpu) {
  int vResult = 0;
  for (int vCpu = 0; vCpu < pNCpu; ++vCpu) {
    if (fPinCpu(vCpu)) {
      return -1;
    }
    if (fIsRunningOnEfficiencyCore()) {
      continue;  // efficiency cores harm lockstep threading
    }
    ++vCpu;      // hyperthreading isn't useful for linear algebra
    ++vResult;
  }
  return vResult;
}

#endif  // __x86_64__ && __linux__

/**
 * Returns number of CPUs on system that are useful for math.
 */
int32_t fCommonCpuGetNumMath() {
#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
  int vNCpu = sysconf(_SC_NPROCESSORS_ONLN);
  if (vNCpu < 1) {
    return fCommonCpuGetNumPhysicalCores();
  }
  if (fIsHybridCpu()) {
    cpu_set_t vAffinity;
    if (!pthread_getaffinity_np(pthread_self(), sizeof(vAffinity), &vAffinity)) {
      int vResult = fCpuCountMathCpus(vNCpu);
      pthread_setaffinity_np(pthread_self(), sizeof(vAffinity), &vAffinity);
      if (vResult > 0) {
        return vResult;
      }
    }
  }
#elif defined(__powerpc64__) || defined(__powerpc__)
  int32_t smt_factor   = 1;
  int     phy_cpus     = fCommonCpuGetNumPhysicalCores();
  int     logical_cpus = sysconf(_SC_NPROCESSORS_ONLN);
  if (phy_cpus > 0 && logical_cpus > phy_cpus) {
    smt_factor = logical_cpus / phy_cpus;
  }
  return phy_cpus * std::min(smt_factor, 2);
#endif
  return fCommonCpuGetNumPhysicalCores();
}

// Helper for setting process priority

#if defined(_WIN32)

bool set_process_priority(enum ggml_sched_priority prio) {
  if (prio == GGML_SCHED_PRIO_NORMAL) {
    return true;
  }

  DWORD p = NORMAL_PRIORITY_CLASS;
  switch (prio) {
    case GGML_SCHED_PRIO_LOW:
      p = BELOW_NORMAL_PRIORITY_CLASS;
      break;
    case GGML_SCHED_PRIO_NORMAL:
      p = NORMAL_PRIORITY_CLASS;
      break;
    case GGML_SCHED_PRIO_MEDIUM:
      p = ABOVE_NORMAL_PRIORITY_CLASS;
      break;
    case GGML_SCHED_PRIO_HIGH:
      p = HIGH_PRIORITY_CLASS;
      break;
    case GGML_SCHED_PRIO_REALTIME:
      p = REALTIME_PRIORITY_CLASS;
      break;
  }

  if (!SetPriorityClass(GetCurrentProcess(), p)) {
    COM_WRN("failed to set process priority class %d : (%d)\n", prio, (int) GetLastError());
    return false;
  }

  return true;
}

#else  // MacOS and POSIX
#  include <sys/resource.h>
#  include <sys/types.h>

bool fSetProcessPriority(enum ggml_sched_priority pPrio) {
  if (pPrio == GGML_SCHED_PRIO_NORMAL) {
    return true;
  }

  int vP = 0;
  switch (pPrio) {
    case GGML_SCHED_PRIO_LOW:
      vP = 5;
      break;
    case GGML_SCHED_PRIO_NORMAL:
      vP = 0;
      break;
    case GGML_SCHED_PRIO_MEDIUM:
      vP = -5;
      break;
    case GGML_SCHED_PRIO_HIGH:
      vP = -10;
      break;
    case GGML_SCHED_PRIO_REALTIME:
      vP = -20;
      break;
  }

  if (setpriority(PRIO_PROCESS, 0, vP) != 0) {
    COM_WRN("failed to set process priority %d : %s (%d)\n", pPrio, strerror(errno), errno);
    return false;
  }
  return true;
}

#endif

//
// CLI argument parsing
//

void fPostprocessCpuParams(common_cpu_params & pCpuparams, const common_cpu_params * pRoleModel) {
  int32_t vNSet = 0;

  if (pCpuparams.vNThreads < 0) {
    // Assuming everything about cpuparams is invalid
    if (pRoleModel != nullptr) {
      pCpuparams = *pRoleModel;
    } else {
      pCpuparams.vNThreads = fCommonCpuGetNumMath();
    }
  }

  for (int32_t vI = 0; vI < GGML_MAX_N_THREADS; vI++) {
    if (pCpuparams.aCpumask[vI]) {
      vNSet++;
    }
  }

  if (vNSet && vNSet < pCpuparams.vNThreads) {
    // Not enough set bits, may experience performance issues.
    COM_WRN("Not enough set bits in CPU mask (%d) to satisfy requested thread count: %d\n", vNSet,
            pCpuparams.vNThreads);
  }
}

bool fParseCpuRange(const std::string & pRange, bool (&pBoolmask)[GGML_MAX_N_THREADS]) {
  size_t vDashLoc = pRange.find('-');
  if (vDashLoc == std::string::npos) {
    COM_ERR("%s", "Format of CPU range is invalid! Expected [<start>]-[<end>].\n");
    return false;
  }

  size_t vStartI;
  size_t vEndI;

  if (vDashLoc == 0) {
    vStartI = 0;
  } else {
    vStartI = std::stoull(pRange.substr(0, vDashLoc));
    if (vStartI >= GGML_MAX_N_THREADS) {
      COM_ERR("%s", "Start index out of bounds!\n");
      return false;
    }
  }

  if (vDashLoc == pRange.length() - 1) {
    vEndI = GGML_MAX_N_THREADS - 1;
  } else {
    vEndI = std::stoull(pRange.substr(vDashLoc + 1));
    if (vEndI >= GGML_MAX_N_THREADS) {
      COM_ERR("%s", "End index out of bounds!\n");
      return false;
    }
  }

  for (size_t vI = vStartI; vI <= vEndI; vI++) {
    pBoolmask[vI] = true;
  }

  return true;
}

bool fParseCpuMask(const std::string & pMask, bool (&pBoolmask)[GGML_MAX_N_THREADS]) {
  // Discard potential 0x prefix
  size_t vStartI = 0;
  if (pMask.length() >= 2 && pMask.substr(0, 2) == "0x") {
    vStartI = 2;
  }

  size_t vNumDigits = pMask.length() - vStartI;
  vNumDigits        = std::min<size_t>(vNumDigits, 128);

  size_t vEndI = vNumDigits + vStartI;

  for (size_t vI = vStartI, n = (vNumDigits * 4 - 1); vI < vEndI; vI++, n -= 4) {
    char   vC  = pMask.at(vI);
    int8_t vId = vC;

    if ((vC >= '0' && vC <= '9')) {
      vId -= '0';
    } else if (vC >= 'a' && vC <= 'f') {
      vId -= 'a' - 10;
    } else if (vC >= 'A' && vC <= 'F') {
      vId -= 'A' - 10;
    } else {
      COM_ERR("Invalid hex character '%c' at position %d\n", vC, int32_t(vI));
      return false;
    }

    pBoolmask[n]     = pBoolmask[n] || ((vId & 8) != 0);
    pBoolmask[n - 1] = pBoolmask[n - 1] || ((vId & 4) != 0);
    pBoolmask[n - 2] = pBoolmask[n - 2] || ((vId & 2) != 0);
    pBoolmask[n - 3] = pBoolmask[n - 3] || ((vId & 1) != 0);
  }

  return true;
}

void fCommonInit() {
#if defined(_WIN32)
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
#endif

  fCommonLogSetPrefix(fCommonLogMain(), true);
  fCommonLogSetTimestamps(fCommonLogMain(), true);

  fDgsLogSet(fCommonLogDefaultCallback, NULL);
}

void fCommonParamsPrintInfo(const common_params & pParams, bool pPrintDevices) {
#ifdef NDEBUG
  const char * vBuildType = "";
#else
  const char * build_type = " (debug)";
#endif
  COM_TRC("%s: build %d (%s) with %s for %s%s\n", __func__, fDgsBuildNumber(), fDgsCommit(), fDgsCompiler(),
          fDgsBuildTarget(), vBuildType);

  const int cVerbosity = fCommonLogGetVerbosityThold();
  COM_INF("%s: verbosity = %d (adjust with the `-lv N` CLI arg)\n", __func__, cVerbosity);

  // device enumeration creates a primary context on CUDA backends, skip it when the caller does not own any device
  if (pPrintDevices && cVerbosity >= LOG_LEVEL_TRACE) {
    COM_TRC("%s", "device_info:\n");
    for (size_t vI = 0; vI < ggml_backend_dev_count(); ++vI) {
      auto * vDev = ggml_backend_dev_get(vI);
      size_t free, vTotal;
      ggml_backend_dev_memory(vDev, &free, &vTotal);
      COM_TRC("  - %-8s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(vDev), ggml_backend_dev_description(vDev),
              vTotal / 1024 / 1024, free / 1024 / 1024);
    }
  }
  COM_TRC("%s\n", fCommonParamsGetSystemInfo(pParams).c_str());
}

std::string fCommonParamsGetSystemInfo(const common_params & pParams) {
  std::ostringstream vOs;

  vOs << "system_info: n_threads = " << pParams.vCpuparams.vNThreads;
  if (pParams.vCpuparamsBatch.vNThreads != -1) {
    vOs << " (n_threads_batch = " << pParams.vCpuparamsBatch.vNThreads << ")";
  }
#if defined(_WIN32) && (_WIN32_WINNT >= 0x0601) && !defined(__MINGW64__)  // windows 7 and later
  // TODO: windows + arm64 + mingw64
  DWORD logicalProcessorCount = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
  os << " / " << logicalProcessorCount << " | " << fDgsPrintSystemInfo();
#else
  vOs << " / " << std::thread::hardware_concurrency() << " | " << fDgsPrintSystemInfo();
#endif

  return vOs.str();
}

//
// String utils
//

std::string fStringFormat(const char * pFmt, ...) {
  va_list vAp;
  va_list vAp2;
  va_start(vAp, pFmt);
  va_copy(vAp2, vAp);
  int size = vsnprintf(NULL, 0, pFmt, vAp);
  GGML_ASSERT(size >= 0 && size < INT_MAX);  // NOLINT
  std::vector<char> lBuf(size + 1);
  int               vSize2 = vsnprintf(lBuf.data(), size + 1, pFmt, vAp2);
  GGML_ASSERT(vSize2 == size);
  va_end(vAp2);
  va_end(vAp);
  return std::string(lBuf.data(), size);
}

std::string fStringStrip(const std::string & str) {
  size_t vStart = 0;
  size_t end    = str.size();
  while (vStart < end && std::isspace(str[vStart])) {
    vStart++;
  }
  while (end > vStart && std::isspace(str[end - 1])) {
    end--;
  }
  return str.substr(vStart, end - vStart);
}

std::string fStringLcs(std::string_view pA, std::string_view pB) {
  if (pA.empty() || pB.empty()) {
    return {};
  }

  std::vector<std::vector<size_t>> lDp(pA.size() + 1, std::vector<size_t>(pB.size() + 1, 0));
  size_t                           vBestLen  = 0;
  size_t                           vBestEndA = 0;

  for (size_t vI = 1; vI <= pA.size(); ++vI) {
    for (size_t vJ = 1; vJ <= pB.size(); ++vJ) {
      if (pA[vI - 1] == pB[vJ - 1]) {
        lDp[vI][vJ] = lDp[vI - 1][vJ - 1] + 1;
        if (lDp[vI][vJ] > vBestLen) {
          vBestLen  = lDp[vI][vJ];
          vBestEndA = vI;
        }
      }
    }
  }
  return std::string(pA.substr(vBestEndA - vBestLen, vBestLen));
}

std::string fStringGetSortableTimestamp() {
  using clock = std::chrono::system_clock;

  const clock::time_point cCurrentTime = clock::now();
  const time_t            cAsTimeT     = clock::to_time_t(cCurrentTime);
  char                    aTimestampNoNs[100];
  std::strftime(aTimestampNoNs, 100, "%Y_%m_%d-%H_%M_%S", std::localtime(&cAsTimeT));

  const int64_t cNs =
    std::chrono::duration_cast<std::chrono::nanoseconds>(cCurrentTime.time_since_epoch() % 1000000000).count();
  char aTimestampNs[11];
  snprintf(aTimestampNs, 11, "%09" PRId64, cNs);

  return std::string(aTimestampNoNs) + "." + std::string(aTimestampNs);
}

void fStringReplaceAll(std::string & pS, const std::string & pSearch, const std::string & pReplace) {
  if (pSearch.empty()) {
    return;
  }
  std::string vBuilder;
  vBuilder.reserve(pS.length());
  size_t vPos     = 0;
  size_t vLastPos = 0;
  while ((vPos = pS.find(pSearch, vLastPos)) != std::string::npos) {
    vBuilder.append(pS, vLastPos, vPos - vLastPos);
    vBuilder.append(pReplace);
    vLastPos = vPos + pSearch.length();
  }
  vBuilder.append(pS, vLastPos, std::string::npos);
  pS = std::move(vBuilder);
}

std::string fRegexEscape(const std::string & pS) {
  static const std::regex cSpecialChars("[.^$|()*+?\\[\\]{}\\\\]");
  return std::regex_replace(pS, cSpecialChars, "\\$&");
}

std::string string_join(const std::vector<std::string> & pValues, const std::string & pSeparator) {
  std::ostringstream vResult;
  for (size_t vI = 0; vI < pValues.size(); ++vI) {
    if (vI > 0) {
      vResult << pSeparator;
    }
    vResult << pValues[vI];
  }
  return vResult.str();
}

std::vector<std::string> fStringSplit(const std::string & str, const std::string & pDelimiter) {
  std::vector<std::string> lParts;
  size_t                   vStart = 0;
  size_t                   end    = str.find(pDelimiter);

  while (end != std::string::npos) {
    lParts.push_back(str.substr(vStart, end - vStart));
    vStart = end + pDelimiter.length();
    end    = str.find(pDelimiter, vStart);
  }

  lParts.push_back(str.substr(vStart));

  return lParts;
}

std::string fStringRepeat(const std::string & str, size_t n) {
  if (n == 0) {
    return "";
  }

  std::string vResult;
  vResult.reserve(str.length() * n);

  for (size_t vI = 0; vI < n; ++vI) {
    vResult += str;
  }

  return vResult;
}

std::string fStringFrom(bool value) {
  return value ? "true" : "false";
}

std::string fStringFrom(const std::vector<int> & pValues) {
  std::stringstream vBuf;

  vBuf << "[ ";
  bool vFirst = true;
  for (auto vE : pValues) {
    if (vFirst) {
      vFirst = false;
    } else {
      vBuf << ", ";
    }
    vBuf << std::to_string(vE);
  }
  vBuf << " ]";

  return vBuf.str();
}

std::string fStringFrom(const struct dgs_context * pCtx, const std::vector<dgs_token> & pTokens) {
  std::stringstream vBuf;

  vBuf << "[ ";

  bool vFirst = true;
  for (const auto & token : pTokens) {
    if (!vFirst) {
      vBuf << ", ";
    } else {
      vFirst = false;
    }

    auto vDetokenized = fCommonTokenToPiece(pCtx, token);

    vBuf << "'" << vDetokenized << "'"
         << ":" << std::to_string(token);
  }

  vBuf << " ]";

  return vBuf.str();
}

void fStringProcessEscapes(std::string & pInput) {
  std::size_t vInputLen  = pInput.length();
  std::size_t vOutputIdx = 0;

  for (std::size_t vInputIdx = 0; vInputIdx < vInputLen; ++vInputIdx) {
    if (pInput[vInputIdx] == '\\' && vInputIdx + 1 < vInputLen) {
      switch (pInput[++vInputIdx]) {
        case 'n':
          pInput[vOutputIdx++] = '\n';
          break;
        case 'r':
          pInput[vOutputIdx++] = '\r';
          break;
        case 't':
          pInput[vOutputIdx++] = '\t';
          break;
        case '\'':
          pInput[vOutputIdx++] = '\'';
          break;
        case '\"':
          pInput[vOutputIdx++] = '\"';
          break;
        case '\\':
          pInput[vOutputIdx++] = '\\';
          break;
        case 'x':
          // Handle \x12, etc
          if (vInputIdx + 2 < vInputLen) {
            const char aX[3] = { pInput[vInputIdx + 1], pInput[vInputIdx + 2], 0 };
            char *     vErrP = nullptr;
            const long cVal  = std::strtol(aX, &vErrP, 16);
            if (vErrP == aX + 2) {
              vInputIdx += 2;
              pInput[vOutputIdx++] = char(cVal);
              break;
            }
          }
          // fall through
        default:
          pInput[vOutputIdx++] = '\\';
          pInput[vOutputIdx++] = pInput[vInputIdx];
          break;
      }
    } else {
      pInput[vOutputIdx++] = pInput[vInputIdx];
    }
  }

  pInput.resize(vOutputIdx);
}

bool fStringParseKvOverride(const char * data, std::vector<dgs_model_kv_override> & pOverrides) {
  const char * vSep = strchr(data, '=');
  if (vSep == nullptr || vSep - data >= 128) {
    COM_ERR("%s: malformed KV override '%s'\n", __func__, data);
    return false;
  }
  dgs_model_kv_override vKvo;
  std::strncpy(vKvo.aKey, data, vSep - data);
  vKvo.aKey[vSep - data] = 0;
  vSep++;
  if (strncmp(vSep, "int:", 4) == 0) {
    vSep += 4;
    vKvo.vTag    = cDgsKvOverrideTypeInt;
    vKvo.vValI64 = std::atol(vSep);
  } else if (strncmp(vSep, "float:", 6) == 0) {
    vSep += 6;
    vKvo.vTag    = cDgsKvOverrideTypeFloat;
    vKvo.vValF64 = std::atof(vSep);
  } else if (strncmp(vSep, "bool:", 5) == 0) {
    vSep += 5;
    vKvo.vTag = cDgsKvOverrideTypeBool;
    if (std::strcmp(vSep, "true") == 0) {
      vKvo.vValBool = true;
    } else if (std::strcmp(vSep, "false") == 0) {
      vKvo.vValBool = false;
    } else {
      COM_ERR("%s: invalid boolean value for KV override '%s'\n", __func__, data);
      return false;
    }
  } else if (strncmp(vSep, "str:", 4) == 0) {
    vSep += 4;
    vKvo.vTag = cDgsKvOverrideTypeStr;
    if (strlen(vSep) > 127) {
      COM_ERR("%s: malformed KV override '%s', value cannot exceed 127 chars\n", __func__, data);
      return false;
    }
    strncpy(vKvo.aValStr, vSep, 127);
    vKvo.aValStr[127] = '\0';
  } else {
    COM_ERR("%s: invalid type for KV override '%s'\n", __func__, data);
    return false;
  }
  pOverrides.emplace_back(std::move(vKvo));
  return true;
}

static inline bool fGlobClassMatch(const char pC, const char * pPattern, const char * pClassEnd) {
  const char * vClassStart = pPattern;
  bool         vNegated    = false;

  if (*vClassStart == '!') {
    vNegated = true;
    vClassStart++;
  }

  // If first character after negation is ']' or '-', treat it as literal
  if (*vClassStart == ']' || *vClassStart == '-') {
    if (vClassStart < pClassEnd && *vClassStart == pC) {
      return !vNegated;
    }
    vClassStart++;
  }

  bool vMatched = false;

  while (vClassStart < pClassEnd) {
    if (vClassStart + 2 < pClassEnd && vClassStart[1] == '-' && vClassStart[2] != ']') {
      char vStartChar = *vClassStart;
      char vEndChar   = vClassStart[2];
      if (pC >= vStartChar && pC <= vEndChar) {
        vMatched = true;
        break;
      }
      vClassStart += 3;
    } else {
      if (*vClassStart == pC) {
        vMatched = true;
        break;
      }
      vClassStart++;
    }
  }

  return vNegated ? !vMatched : vMatched;
}

// simple glob: * matches non-/ chars, ** matches anything including /, [] matches character class
static inline bool fGlobMatch(const char * pPattern, const char * str) {
  if (*pPattern == '\0') {
    return *str == '\0';
  }
  if (pPattern[0] == '*' && pPattern[1] == '*') {
    const char * vP = pPattern + 2;
    if (fGlobMatch(vP, str)) {
      return true;
    }
    if (*str != '\0') {
      return fGlobMatch(pPattern, str + 1);
    }
    return false;
  }
  if (*pPattern == '*') {
    const char * vP = pPattern + 1;
    for (; *str != '\0' && *str != '/'; str++) {
      if (fGlobMatch(vP, str)) {
        return true;
      }
    }
    return fGlobMatch(vP, str);
  }
  if (*pPattern == '?' && *str != '\0' && *str != '/') {
    return fGlobMatch(pPattern + 1, str + 1);
  }
  if (*pPattern == '[') {
    const char * vClassEnd = pPattern + 1;
    // If first character after '[' is ']' or '-', treat it as literal
    if (*vClassEnd == ']' || *vClassEnd == '-') {
      vClassEnd++;
    }
    while (*vClassEnd != '\0' && *vClassEnd != ']') {
      vClassEnd++;
    }
    if (*vClassEnd == ']') {
      if (*str == '\0') {
        return false;
      }
      bool vMatched = fGlobClassMatch(*str, pPattern + 1, vClassEnd);
      return vMatched && fGlobMatch(vClassEnd + 1, str + 1);
    } else {
      if (*str == '[') {
        return fGlobMatch(pPattern + 1, str + 1);
      }
      return false;
    }
  }
  if (*pPattern == *str) {
    return fGlobMatch(pPattern + 1, str + 1);
  }
  return false;
}

bool fGlobMatch(const std::string & pPattern, const std::string & str) {
  return fGlobMatch(pPattern.c_str(), str.c_str());
}

//
// Filesystem utils
//

// Validate if a filename is safe to use
// To validate a full path, split the path by the OS-specific path separator, and validate each part with this function
bool fFsValidateFilename(const std::string & pFilename, bool pAllowSubdirs) {
  if (!pFilename.length()) {
    // Empty filename invalid
    return false;
  }
  if (pFilename.length() > 255) {
    // Limit at common largest possible filename on Linux filesystems
    // to avoid unnecessary further validation
    // (On systems with smaller limits it will be caught by the OS)
    return false;
  }

  size_t vOffset = 0;
  while (vOffset < pFilename.size()) {
    utf8_parse_result vResult = fCommonParseUtf8Codepoint(pFilename, vOffset);

    if (vResult.vStatus != utf8_parse_result::cSuccess) {
      return false;
    }
    uint32_t vC = vResult.vCodepoint;

    if ((vResult.vBytesConsumed == 2 && vC < 0x80) || (vResult.vBytesConsumed == 3 && vC < 0x800) ||
        (vResult.vBytesConsumed == 4 && vC < 0x10000)) {
      return false;
    }

    // Check for forbidden codepoints:
    // - Control characters
    // - Unicode equivalents of illegal characters
    // - UTF-16 surrogate pairs
    // - UTF-8 replacement character
    // - Byte order mark (BOM)
    // - Illegal characters: / \ : * ? " < > |
    if (vC <= 0x1F                         // Control characters (C0)
        || vC == 0x7F                      // Control characters (DEL)
        || (vC >= 0x80 && vC <= 0x9F)      // Control characters (C1)
        || vC == 0xFF0E                    // Fullwidth Full Stop (period equivalent)
        || vC == 0x2215                    // Division Slash (forward slash equivalent)
        || vC == 0x2216                    // Set Minus (backslash equivalent)
        || (vC >= 0xD800 && vC <= 0xDFFF)  // UTF-16 surrogate pairs
        || vC > 0x10FFFF                   // Max Unicode limit
        || vC == 0xFFFD                    // Replacement Character (UTF-8)
        || vC == 0xFEFF                    // Byte Order Mark (BOM)
        || vC == ':' || vC == '*'          // Illegal characters
        || vC == '?' || vC == '"' || vC == '<' || vC == '>' || vC == '|') {
      return false;
    }
    if (!pAllowSubdirs && (vC == '/' || vC == '\\')) {
      // Subdirectories not allowed, reject path separators
      return false;
    }
    vOffset += vResult.vBytesConsumed;
  }

  // Reject any leading or trailing ' ', or any trailing '.', these are stripped on Windows and will cause a different filename
  // Unicode and other whitespace is not affected, only 0x20 space
  if (pFilename.front() == ' ' || pFilename.back() == ' ' || pFilename.back() == '.') {
    return false;
  }

  // Reject any ".." (currently stricter than necessary, it should be fine to just check for == ".." instead)
  if (pFilename.find("..") != std::string::npos) {
    return false;
  }

  // Reject "."
  if (pFilename == ".") {
    return false;
  }

  return true;
}

#include <iostream>

#ifdef _WIN32
std::wstring utf8_to_wstring(const std::string & str) {
  if (str.empty()) {
    return std::wstring();
  }

  int size = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int) str.size(), NULL, 0);

  if (size <= 0) {
    return std::wstring();
  }

  std::wstring wstr(size, 0);
  MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int) str.size(), &wstr[0], size);

  return wstr;
}

std::string wstring_to_utf8(const std::wstring & str) {
  if (str.empty()) {
    return std::string();
  }

  int size = WideCharToMultiByte(CP_UTF8, 0, str.c_str(), (int) str.size(), NULL, 0, NULL, NULL);

  if (size <= 0) {
    return std::string();
  }

  std::string utf8(size, 0);
  WideCharToMultiByte(CP_UTF8, 0, str.c_str(), (int) str.size(), &utf8[0], size, NULL, NULL);

  return utf8;
}
#endif

// returns the path as a UTF-8 string, preserving its separators
std::string fFsPathToUtf8(const std::filesystem::path & pPath) {
  const auto value = pPath.u8string();
  return std::string(value.begin(), value.end());
}

void fFsWriteAtomic(const std::filesystem::path & pPath, const std::string & data) {
  std::error_code       vEc;
  std::filesystem::path vPathTmp = pPath;
  vPathTmp += ".tmp";

  if (pPath.has_parent_path()) {
    std::filesystem::create_directories(pPath.parent_path(), vEc);
  }

  std::ofstream vFile(vPathTmp, std::ios::binary);
  vFile << data;
  vFile.close();

  if (!vFile.fail()) {
    std::filesystem::rename(vPathTmp, pPath, vEc);
  }

  if (vFile.fail() || vEc) {
    std::filesystem::remove(vPathTmp, vEc);
    throw std::runtime_error("failed to write file: " + fFsPathToUtf8(pPath));
  }
}

bool fFsIsDirectory(const std::string & pPath) {
  std::filesystem::path vDir(pPath);
  return std::filesystem::exists(vDir) && std::filesystem::is_directory(vDir);
}

std::string fCommonGetEnv(const std::string & name) {
  const char * value = std::getenv(name.c_str());
  return value == nullptr ? "" : value;
}

void fCommonSetEnv(const std::string & name, const std::string & value) {
#if defined(_WIN32)
  _putenv_s(name.c_str(), value.c_str());
#else
  if (value.empty()) {
    unsetenv(name.c_str());
  } else {
    setenv(name.c_str(), value.c_str(), 1);
  }
#endif
}

std::filesystem::path fCommonGetPathFromEnv(const std::string & name) {
#if defined(_WIN32)
  const std::wstring wname  = utf8_to_wstring(name);
  const wchar_t *    wvalue = _wgetenv(wname.c_str());
  return wvalue ? std::filesystem::path(wvalue) : std::filesystem::path();
#else
  const char * value = std::getenv(name.c_str());
  return value ? std::filesystem::path(value) : std::filesystem::path();
#endif
}

#if !defined(_WIN32)
static std::filesystem::path fGetHomeDirectory() {
  std::filesystem::path vHome = fCommonGetPathFromEnv("HOME");
  if (!vHome.empty()) {
    return vHome;
  }
  const struct passwd * vPw = getpwuid(getuid());
  if (!vPw || !vPw->pw_dir || !*vPw->pw_dir) {
    throw std::runtime_error("Failed to find $HOME directory");
  }
  return vPw->pw_dir;
}
#endif

std::filesystem::path fFsGetCacheDirectory() {
  std::filesystem::path vCacheDirectory = fCommonGetPathFromEnv("DGS_CACHE");
  if (!vCacheDirectory.empty()) {
    return vCacheDirectory;
  }
#if defined(_WIN32)
  cache_directory = fCommonGetPathFromEnv("LOCALAPPDATA");
  if (cache_directory.empty()) {
    throw std::runtime_error("Failed to find %LOCALAPPDATA% directory");
  }
#elif defined(__APPLE__)
  cache_directory = get_home_directory() / "Library/Caches";
#else
  vCacheDirectory = fCommonGetPathFromEnv("XDG_CACHE_HOME");
  if (vCacheDirectory.empty()) {
    vCacheDirectory = fGetHomeDirectory() / ".cache";
  }
#endif
  return vCacheDirectory / "dgs.cpp";
}

std::filesystem::path fFsGetConfigDirectory() {
  std::filesystem::path vConfigDirectory;
#if defined(_WIN32)
  config_directory = fCommonGetPathFromEnv("APPDATA");
  if (config_directory.empty()) {
    throw std::runtime_error("Failed to find %APPDATA% directory");
  }
#else
  vConfigDirectory = fCommonGetPathFromEnv("XDG_CONFIG_HOME");
  if (vConfigDirectory.empty()) {
    vConfigDirectory = fGetHomeDirectory() / ".config";
  }
#endif
  return vConfigDirectory / "dgs.cpp";
}

std::filesystem::path fFsGetCacheFile(const std::string & pFilename) {
  GGML_ASSERT(pFilename.find(DIRECTORY_SEPARATOR) == std::string::npos);
  const std::filesystem::path cCacheDirectory = fFsGetCacheDirectory();
  std::error_code             vEc;
  fCommonCreateDirectories(cCacheDirectory, vEc);
  if (vEc) {
    throw std::runtime_error("failed to create cache directory: " + fFsPathToUtf8(cCacheDirectory));
  }
  return cCacheDirectory / std::filesystem::u8path(pFilename);
}

//
// TTY utils
//

bool fCommonIsTty(FILE * pFile) {
#if defined(_WIN32)
  return _isatty(_fileno(file));
#else
  return isatty(fileno(pFile));
#endif
}

bool fTtyCanUseColors() {
  // Check NO_COLOR environment variable (https://no-color.org/)
  if (const char * vNoColor = std::getenv("NO_COLOR")) {
    if (vNoColor[0] != '\0') {
      return false;
    }
  }

  // Check TERM environment variable
  if (const char * vTerm = std::getenv("TERM")) {
    if (std::strcmp(vTerm, "dumb") == 0) {
      return false;
    }
  }

  // Check if stdout and stderr are connected to a terminal
  // We check both because log messages can go to either
  return fCommonIsTty(stdout) || fCommonIsTty(stderr);
}

bool fTtyEnableAnsi() {
#if defined(_WIN32)
  // a Windows console renders ANSI sequences only in virtual terminal mode, pipes and files take them as is
  for (DWORD id : { STD_OUTPUT_HANDLE, STD_ERROR_HANDLE }) {
    HANDLE h    = GetStdHandle(id);
    DWORD  mode = 0;
    if (GetConsoleMode(h, &mode) && !SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
      return false;
    }
  }
#endif
  return true;
}

//
// Model utils
//

// TODO: move to common/sampling
static void fCommonInitSamplerFromModel(const dgs_model * pModel, common_params_sampling & pSparams) {
  const uint64_t cConfig = pSparams.vUserSamplingConfig;

  auto vGetInt32 = [&](const char * pKey, int32_t & pDst, uint64_t pUserConfig) {
    if (cConfig & pUserConfig) {
      return;
    }

    char aBuf[64] = { 0 };
    if (fDgsModelMetaValStr(pModel, pKey, aBuf, sizeof(aBuf)) > 0) {
      char *  end = nullptr;
      int32_t vV  = strtol(aBuf, &end, 10);
      if (end && end != aBuf) {
        pDst = vV;
      }
    }
  };

  auto vGetFloat = [&](const char * pKey, float & pDst, uint64_t pUserConfig) {
    if (cConfig & pUserConfig) {
      return;
    }

    char aBuf[128] = { 0 };
    if (fDgsModelMetaValStr(pModel, pKey, aBuf, sizeof(aBuf)) > 0) {
      char * end = nullptr;
      float  vV  = strtof(aBuf, &end);
      if (end && end != aBuf) {
        pDst = vV;
      }
    }
  };

  // Sampling sequence
  if (!(cConfig & common_params_sampling_config::cCommonParamsSamplingConfigSamplers)) {
    char aBuf[512] = { 0 };
    if (fDgsModelMetaValStr(pModel, fDgsModelMetaKeyStr(cDgsModelMetaKeySamplingSequence), aBuf, sizeof(aBuf)) > 0) {
      const std::vector<std::string> lSamplerNames = fStringSplit<std::string>(std::string(aBuf), ';');
      if (!lSamplerNames.empty()) {
        pSparams.lSamplers = fCommonSamplerTypesFromNames(lSamplerNames);
      }
    }
  }

  vGetInt32(fDgsModelMetaKeyStr(cDgsModelMetaKeySamplingTopK), pSparams.vTopK,
            common_params_sampling_config::cCommonParamsSamplingConfigTopK);
  vGetFloat(fDgsModelMetaKeyStr(cDgsModelMetaKeySamplingTopP), pSparams.vTopP,
            common_params_sampling_config::cCommonParamsSamplingConfigTopP);
  vGetFloat(fDgsModelMetaKeyStr(cDgsModelMetaKeySamplingMinP), pSparams.vMinP,
            common_params_sampling_config::cCommonParamsSamplingConfigMinP);
  vGetFloat(fDgsModelMetaKeyStr(cDgsModelMetaKeySamplingXtcProbability), pSparams.vXtcProbability,
            common_params_sampling_config::cCommonParamsSamplingConfigXtcProbability);
  vGetFloat(fDgsModelMetaKeyStr(cDgsModelMetaKeySamplingXtcThreshold), pSparams.vXtcThreshold,
            common_params_sampling_config::cCommonParamsSamplingConfigXtcThreshold);
  vGetFloat(fDgsModelMetaKeyStr(cDgsModelMetaKeySamplingTemp), pSparams.vTemp,
            common_params_sampling_config::cCommonParamsSamplingConfigTemp);
  vGetInt32(fDgsModelMetaKeyStr(cDgsModelMetaKeySamplingPenaltyLastN), pSparams.vPenaltyLastN,
            common_params_sampling_config::cCommonParamsSamplingConfigPenaltyLastN);
  vGetFloat(fDgsModelMetaKeyStr(cDgsModelMetaKeySamplingPenaltyRepeat), pSparams.vPenaltyRepeat,
            common_params_sampling_config::cCommonParamsSamplingConfigPenaltyRepeat);
  vGetInt32(fDgsModelMetaKeyStr(cDgsModelMetaKeySamplingMirostat), pSparams.vMirostat,
            common_params_sampling_config::cCommonParamsSamplingConfigMirostat);
  vGetFloat(fDgsModelMetaKeyStr(cDgsModelMetaKeySamplingMirostatTau), pSparams.vMirostatTau,
            common_params_sampling_config::cCommonParamsSamplingConfigMirostatTau);
  vGetFloat(fDgsModelMetaKeyStr(cDgsModelMetaKeySamplingMirostatEta), pSparams.vMirostatEta,
            common_params_sampling_config::cCommonParamsSamplingConfigMirostatEta);
}

struct common_init_result::impl {
  impl()  = default;
  ~impl() = default;

  // note: the order in which model, context, etc. are declared matters because their destructors will be called bottom-to-top

  common_threadpools vThreadpools;

  dgs_model_ptr   vModel;
  dgs_context_ptr vContext;

  std::vector<dgs_adapter_lora_ptr> lora;

  std::vector<common_sampler_ptr>     lSamplers;
  std::vector<dgs_sampler_seq_config> lSamplersSeqConfig;
};

static const std::map<common_decision_type, std::string> dCommonDecisionTypeNames = {
  { cCommonDecisionTypeOpenjev,       "openjev"        },
  { cCommonDecisionTypeLev,           "lev"            },
  { cCommonDecisionTypeKev,           "kev"            },
  { cCommonDecisionTypeNimble,        "nimble"         },
  { cCommonDecisionTypeLaya,          "laya"           },
  { cCommonDecisionTypeClef,          "clef"           },
  { cCommonDecisionTypePplxDecider,   "pplx-decider"   },
  { cCommonDecisionTypeVllmDecision2, "vllm-decision2" },
  { cCommonDecisionTypeVllmD3,        "vllm-d3"        },
  { cCommonDecisionTypeLfm2D1,        "lfm2-d1"        },
  { cCommonDecisionTypeLfm2D1Omni,    "lfm2-d1-omni"   },
  // for future models, ALWAYS add org name prefix to avoid name collisions, e.g. "orgname-modelname"
};

static common_decision_type fCommonDecisionTypeFromString(const std::string & str) {
  for (const auto & vPair : dCommonDecisionTypeNames) {
    if (vPair.second == str) {
      return vPair.first;
    }
  }
  return cCommonDecisionTypeUnknown;
}

common_decision_type fCommonGetDecisionType(const struct dgs_model * pModel) {
  char aBuf[64];
  if (fDgsModelMetaValStr(pModel, "general.architecture", aBuf, sizeof(aBuf)) < 0) {
    return cCommonDecisionTypeNone;
  }
  const std::string cKey = std::string(aBuf) + ".decision.type";
  if (fDgsModelMetaValStr(pModel, cKey.c_str(), aBuf, sizeof(aBuf)) < 0) {
    return cCommonDecisionTypeNone;
  }
  return fCommonDecisionTypeFromString(aBuf);
}

common_gguf_info fCommonGetGgufInfo(const std::string & pFname) {
  common_gguf_info info;

  struct gguf_init_params vGgufParams = {
    /* .no_alloc = */ true,
    /* .ctx      = */ nullptr,
  };

  gguf_context_ptr vGgufCtx(gguf_init_from_file(pFname.c_str(), vGgufParams));
  if (!vGgufCtx) {
    return info;  // missing or unreadable file
  }

  const int64_t cArchId = gguf_find_key(vGgufCtx.get(), "general.architecture");
  if (cArchId < 0 || gguf_get_kv_type(vGgufCtx.get(), cArchId) != GGUF_TYPE_STRING) {
    return info;  // no architecture in the metadata
  }
  const std::string cArch = gguf_get_val_str(vGgufCtx.get(), cArchId);
  if (cArch.empty()) {
    return info;
  }

  const int64_t cTypeId = gguf_find_key(vGgufCtx.get(), (cArch + ".decision.type").c_str());
  if (cTypeId < 0) {
    info.vDecisionType = cCommonDecisionTypeNone;
  } else if (gguf_get_kv_type(vGgufCtx.get(), cTypeId) == GGUF_TYPE_STRING) {
    info.vDecisionType = fCommonDecisionTypeFromString(gguf_get_val_str(vGgufCtx.get(), cTypeId));
  }

  // same key and type as the model loader
  const int64_t cCtxId = gguf_find_key(vGgufCtx.get(), (cArch + ".context_length").c_str());
  if (cCtxId >= 0 && gguf_get_kv_type(vGgufCtx.get(), cCtxId) == GGUF_TYPE_UINT32) {
    info.vNCtxTrain = gguf_get_val_u32(vGgufCtx.get(), cCtxId);
  }

  return info;
}

common_init_result::common_init_result(common_params & pParams, bool pModelOnly) : vPimpl(new impl{}) {
  auto vMparams = fCommonModelParamsToDgs(pParams);
  auto vCparams = fCommonContextParamsToDgs(pParams);

  if (pParams.vFitParams) {
    COM_TRC("%s", "fitting params to device memory ...\n");
    COM_TRC("%s",
            "(for bugs during this step try to reproduce them with -fit off, or provide --verbose logs if the bug only "
            "occurs with -fit on)\n");

    // the draft context is created from the same base params and follows the main context, fit both together
    const bool cHasDraft = pParams.vSpeculative.mHasDft();
    const bool cSpecMtp  = std::find(pParams.vSpeculative.lTypes.begin(), pParams.vSpeculative.lTypes.end(),
                                     cCommonSpeculativeTypeDraftMtp) != pParams.vSpeculative.lTypes.end();

    common_params vParamsDft = fCommonBaseParamsToSpeculative(pParams);

    auto vMparamsDft = fCommonModelParamsToDgs(vParamsDft);
    auto vCparamsDft = fCommonContextParamsToDgs(vParamsDft);
    if (cSpecMtp) {
      vCparamsDft.vCtxType = cDgsContextTypeMtp;
    }
    vCparamsDft.vNRsSeq = 0;

    const common_fit_extra_model cExtra = {
      /*.path_model   =*/vParamsDft.vModel.vPath.c_str(),
      /*.mparams      =*/&vMparamsDft,
      /*.cparams      =*/&vCparamsDft,
      /*.shares_model =*/!cHasDraft,  // an MTP context runs on the weights of the main model
    };

    fCommonFitParams(pParams.vModel.vPath.c_str(), &vMparams, &vCparams, pParams.aTensorSplit,
                     pParams.lTensorBuftOverrides.data(), pParams.lFitParamsTarget.data(), pParams.vFitParamsMinCtx,
                     cHasDraft || cSpecMtp ? &cExtra : nullptr,
                     pParams.vVerbosity >= LOG_LEVEL_DEBUG ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR);
  }

  dgs_model * vModel = fDgsModelLoadFromFile(pParams.vModel.vPath.c_str(), vMparams);
  if (vModel == NULL) {
    return;
  }

  vPimpl->vModel.reset(vModel);

  if (pModelOnly) {
    return;
  }

  const dgs_vocab * vVocab = fDgsModelGetVocab(vModel);

  // these decision models return a score for each token via the embeddings output
  // TODO: maybe improve this in the future
  const auto cDecisionType = fCommonGetDecisionType(vModel);
  if (cDecisionType == cCommonDecisionTypeLaya || cDecisionType == cCommonDecisionTypeKev ||
      cDecisionType == cCommonDecisionTypeClef || cDecisionType == cCommonDecisionTypeLfm2D1Omni ||
      cDecisionType == cCommonDecisionTypeVllmDecision2) {
    pParams.vEmbedding   = true;
    pParams.vPoolingType = cDgsPoolingTypeNone;

    vCparams.vEmbeddings        = true;
    vCparams.vPoolingType       = cDgsPoolingTypeNone;
    vCparams.vNOutputsMax       = vCparams.vNBatch;
    vCparams.vNOutputsMaxPerSeq = 1;

    LOG_INF("%s", "decision model reads the embeddings output, enabling embedding mode\n");
  }

  // embeddings need the whole batch in one ubatch, so n_batch must not be larger than n_ubatch
  // (server.cpp does this check for --embedding, but before the model is loaded)
  if (vCparams.vEmbeddings && vCparams.vNBatch > vCparams.vNUbatch) {
    LOG_WRN("embeddings enabled: setting n_batch = n_ubatch = %u\n", vCparams.vNUbatch);
    vCparams.vNBatch = vCparams.vNUbatch;
    pParams.vNBatch  = pParams.vNUbatch;
  }

  // load and optionally apply lora adapters
  for (auto & vLa : pParams.lLoraAdapters) {
    dgs_adapter_lora_ptr lora;
    lora.reset(fDgsAdapterLoraInit(vModel, vLa.vPath.c_str()));
    if (lora == nullptr) {
      COM_ERR("failed to load lora adapter '%s'\n", vLa.vPath.c_str());
      return;
    }

    char aBuf[1024];
    vLa.vPtr = lora.get();
    fDgsAdapterMetaValStr(vLa.vPtr, "adapter.lora.task_name", aBuf, sizeof(aBuf));
    vLa.vTaskName = aBuf;
    fDgsAdapterMetaValStr(vLa.vPtr, "adapter.lora.prompt_prefix", aBuf, sizeof(aBuf));
    vLa.vPromptPrefix = aBuf;
    vPimpl->lora.emplace_back(std::move(lora));  // copy to list of loaded adapters
  }

  // updates params.sampling
  // TODO: fix naming
  fCommonInitSamplerFromModel(vModel, pParams.vSampling);

  if (pParams.vSampling.vIgnoreEos && fDgsVocabEos(vVocab) == DGS_TOKEN_NULL) {
    COM_WRN("%s", "vocab does not have an EOS token, ignoring --ignore-eos\n");
    pParams.vSampling.vIgnoreEos = false;
  }

  // initialize once
  for (dgs_token vI = 0; vI < fDgsVocabNTokens(vVocab); vI++) {
    if (fDgsVocabIsEog(vVocab, vI)) {
      COM_TRC("added %s logit bias = %f\n", fCommonTokenToPiece(vVocab, vI).c_str(), -INFINITY);
      pParams.vSampling.lLogitBiasEog.push_back({ vI, -INFINITY });
    }
  }

  if (pParams.vSampling.vIgnoreEos) {
    // add EOG biases to the active set of logit biases
    pParams.vSampling.lLogitBias.insert(pParams.vSampling.lLogitBias.end(), pParams.vSampling.lLogitBiasEog.begin(),
                                        pParams.vSampling.lLogitBiasEog.end());
  }

  // init the backend samplers as part of the context creation
  vPimpl->lSamplers.resize(vCparams.vNSeqMax);
  vPimpl->lSamplersSeqConfig.resize(vCparams.vNSeqMax);

  for (int vI = 0; vI < (int) vCparams.vNSeqMax; ++vI) {
    vPimpl->lSamplers[vI].reset(fCommonSamplerInit(vModel, pParams.vSampling));
    vPimpl->lSamplersSeqConfig[vI] = { vI, fCommonSamplerGet(vPimpl->lSamplers[vI].get()) };
  }

  if (pParams.vSampling.vBackendSampling) {
    vCparams.vSamplers  = vPimpl->lSamplersSeqConfig.data();
    vCparams.vNSamplers = vPimpl->lSamplersSeqConfig.size();
  }

  dgs_context * vLctx = fDgsInitFromModel(vModel, vCparams);
  if (vLctx == NULL) {
    COM_ERR("failed to create context with model '%s'\n", pParams.vModel.vPath.c_str());
    return;
  }

  vPimpl->vContext.reset(vLctx);

  fSetProcessPriority(pParams.vCpuparams.vPriority);

  vPimpl->vThreadpools.init(vLctx, pParams);
}

dgs_model * common_init_result::mModel() {
  return vPimpl->vModel.get();
}

dgs_context * common_init_result::mContext() {
  return vPimpl->vContext.get();
}

common_sampler * common_init_result::mSampler(dgs_seq_id pSeqId) {
  if (pSeqId < 0 || pSeqId >= (int) vPimpl->lSamplers.size()) {
    return nullptr;
  }
  return vPimpl->lSamplers[pSeqId].get();
}

void common_init_result::mResetSamplers() {
  for (int vI = 0; vI < (int) vPimpl->lSamplers.size(); ++vI) {
    fDgsSamplerReset(fCommonSamplerGet(vPimpl->lSamplers[vI].get()));
  }
}

std::vector<dgs_adapter_lora_ptr> & common_init_result::lora() {
  return vPimpl->lora;
}

// only for warmup and probe decodes, fill zeros as dummy input
static void fCommonBatchSetZeroState(common_batch & pBatch, const dgs_model * pModel, std::vector<float> & pZeros) {
  pZeros.assign(fDgsModelNEmbdOut(pModel), 0.0f);
  for (int32_t vI = 0; vI < pBatch.size(); ++vI) {
    pBatch.mSetEmbdState(vI, { pZeros.data(), 1, pZeros.size() });
  }
}

common_init_result_ptr fCommonInitFromParams(common_params & pParams, bool pModelOnly) {
  common_init_result_ptr vRes(new common_init_result(pParams, pModelOnly));

  dgs_model * vModel = vRes->mModel();
  if (vModel == NULL) {
    COM_ERR("failed to load model '%s'\n", pParams.vModel.vPath.c_str());
    return vRes;
  }

  if (pModelOnly) {
    return vRes;
  }

  dgs_context * vLctx = vRes->mContext();
  if (vLctx == NULL) {
    COM_ERR("failed to create context with model '%s'\n", pParams.vModel.vPath.c_str());
    return vRes;
  }

  const dgs_vocab * vVocab = fDgsModelGetVocab(vModel);

  if (pParams.vCtxShift && !fDgsMemoryCanShift(fDgsGetMemory(vLctx))) {
    COM_WRN("%s", "KV cache shifting is not supported for this context, disabling KV cache shifting\n");
    pParams.vCtxShift = false;
  }

  if (!pParams.lControlVectors.empty()) {
    if (pParams.vControlVectorLayerStart <= 0) {
      pParams.vControlVectorLayerStart = 1;
    }
    if (pParams.vControlVectorLayerEnd <= 0) {
      pParams.vControlVectorLayerEnd = fDgsModelNLayer(vModel);
    }

    const auto cCvec = fCommonControlVectorLoad(pParams.lControlVectors);
    if (cCvec.vNEmbd == -1) {
      return vRes;
    }

    int vErr = fDgsSetAdapterCvec(vLctx, cCvec.data.data(), cCvec.data.size(), cCvec.vNEmbd,
                                  pParams.vControlVectorLayerStart, pParams.vControlVectorLayerEnd);
    if (vErr) {
      return vRes;
    }
  }

  if (fDgsPoolingType(vLctx) == cDgsPoolingTypeRank) {
    bool vOk = true;

    if (fDgsVocabBos(vVocab) == DGS_TOKEN_NULL) {
      COM_WRN("%s", "vocab does not have a  BOS token, reranking will not work\n");
      vOk = false;
    }

    bool vHasEos          = fDgsVocabEos(vVocab) != DGS_TOKEN_NULL;
    bool vHasSep          = fDgsVocabSep(vVocab) != DGS_TOKEN_NULL;
    bool vHasRerankPrompt = fDgsModelChatTemplate(vModel, "rerank") != NULL;

    if (!vHasEos && !vHasSep && !vHasRerankPrompt) {
      COM_WRN("%s", "vocab does not have an EOS token, SEP token, or rerank prompt. Reranking will not work\n");
      vOk = false;
    } else if (!vHasEos) {
      COM_WRN("%s", "vocab does not have an EOS token, using SEP token as fallback\n");
    }

    if (!vOk) {
      return vRes;
    }
  }

  if (!pParams.vLoraInitWithoutApply) {
    fCommonSetAdapterLora(vLctx, pParams.lLoraAdapters);
  }

  if (pParams.vWarmup) {
    COM_TRC("%s", "warming up the model with an empty run - please wait ... (--no-warmup to disable)\n");

    std::vector<dgs_token> lTmp;
    dgs_token              vBos = fDgsVocabBos(vVocab);
    dgs_token              vEos = fDgsVocabEos(vVocab);

    // some models (e.g. T5) don't have a BOS token
    if (vBos != DGS_TOKEN_NULL) {
      lTmp.push_back(vBos);
    }
    if (vEos != DGS_TOKEN_NULL) {
      lTmp.push_back(vEos);
    }
    if (lTmp.empty()) {
      lTmp.push_back(0);
    }

    if (fDgsModelHasEncoder(vModel)) {
      common_batch vBatch = fCommonBatchGetOne(vLctx, lTmp);
      fDgsProcess(vLctx, cDgsProcessTypeEncode, vBatch.get());
      dgs_token vDecoderStartTokenId = fDgsModelDecoderStartToken(vModel);
      if (vDecoderStartTokenId == DGS_TOKEN_NULL) {
        vDecoderStartTokenId = vBos;
      }
      lTmp.clear();
      lTmp.push_back(vDecoderStartTokenId);
    }
    if (fDgsModelHasDecoder(vModel)) {
      lTmp.resize(std::min(lTmp.size(), (size_t) pParams.vNBatch));
      common_batch       vBatch = fCommonBatchGetOne(vLctx, lTmp);
      std::vector<float> lZeros;
      fCommonBatchSetZeroState(vBatch, vModel, lZeros);
      fDgsProcess(vLctx, cDgsProcessTypeDecode, vBatch.get());
    }
    fDgsMemoryClear(fDgsGetMemory(vLctx), true);
    fDgsSynchronize(vLctx);
    fDgsPerfContextReset(vLctx);

    // reset samplers to reset RNG state after warmup to the seeded state
    vRes->mResetSamplers();
  }

  return vRes;
}

common_init_result::~common_init_result() = default;

std::string fCommonGetModelEndpoint() {
  std::string vEndpoint = fCommonGetEnv("MODEL_ENDPOINT");

  if (vEndpoint.empty()) {
    return "https://huggingface.co/";
  }
  if (vEndpoint.back() != '/') {
    vEndpoint += '/';
  }
  return vEndpoint;
}

char * fCommonGetModelOrExit(int pArgc, char * pArgv[]) {
  if (pArgc > 1) {
    return pArgv[1];
  }

  char * vPath = getenv("DGSCPP_TEST_MODELFILE");
  if (!vPath || strlen(vPath) == 0) {
    fprintf(stderr,
            "\033[33mWARNING: No model file provided. Skipping this test. Set DGSCPP_TEST_MODELFILE=<gguf_model_path> "
            "to silence this warning and run this test.\n\033[0m");
    exit(EXIT_SUCCESS);
  }

  return vPath;
}

common_context_seq_rm_type fCommonContextCanSeqRm(dgs_context * pCtx) {
  auto * vMem = fDgsGetMemory(pCtx);
  if (vMem == nullptr) {
    return cCommonContextSeqRmTypeNo;
  }

  if (fDgsNRsSeq(pCtx) > 0) {
    COM_TRC("%s", "the context supports bounded partial sequence removal\n");
    return cCommonContextSeqRmTypeRs;
  }

  common_context_seq_rm_type vRes = cCommonContextSeqRmTypePart;

  fDgsMemoryClear(vMem, true);

  // eval 2 tokens to check if the context is compatible
  std::vector<dgs_token> lTmp;
  lTmp.push_back(0);
  lTmp.push_back(0);

  int vRet;
  {
    common_batch       vBatch = fCommonBatchGetOne(pCtx, lTmp);
    std::vector<float> lZeros;
    fCommonBatchSetZeroState(vBatch, fDgsGetModel(pCtx), lZeros);
    vRet = fDgsProcess(pCtx, cDgsProcessTypeDecode, vBatch.get());
  }
  if (vRet != 0) {
    COM_ERR("dgs_process() failed: %d\n", vRet);
    vRes = cCommonContextSeqRmTypeNo;
    goto done;
  }

  // try to remove the last tokens
  if (!fDgsMemorySeqRm(vMem, 0, 1, -1)) {
    COM_TRC("%s", "the context does not support partial sequence removal\n");
    vRes = cCommonContextSeqRmTypeFull;
    goto done;
  }

done:
  fDgsMemoryClear(vMem, true);
  fDgsSynchronize(pCtx);

  return vRes;
}

static void fCommonContextSeqRm(dgs_context * pCtx, dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1) {
  auto * vMem = fDgsGetMemory(pCtx);
  if (!fDgsMemorySeqRm(vMem, pSeqId, pP0, pP1)) {
    GGML_ABORT("%s", fStringFormat("failed to remove sequence %d with p0=%d, p1=%d\n", pSeqId, pP0, pP1).c_str());
  }
}

static void fCommonContextSeqCp(dgs_context * pCtx,
                                dgs_seq_id    pSeqIdSrc,
                                dgs_seq_id    pSeqIdDst,
                                dgs_pos       pP0,
                                dgs_pos       pP1) {
  auto * vMem = fDgsGetMemory(pCtx);
  fDgsMemorySeqCp(vMem, pSeqIdSrc, pSeqIdDst, pP0, pP1);
}

static void fCommonContextSeqAdd(dgs_context * pCtx, dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1, dgs_pos pDelta) {
  auto * vMem = fDgsGetMemory(pCtx);
  fDgsMemorySeqAdd(vMem, pSeqId, pP0, pP1, pDelta);
}

void common_memory::init(dgs_context * pCtxTgt, dgs_context * pCtxDft) {
  this->vCtxTgt = pCtxTgt;
  this->vCtxDft = pCtxDft;
}

void common_memory::mSeqRm(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1) const {
  fCommonContextSeqRm(vCtxTgt, pSeqId, pP0, pP1);
  if (vCtxDft) {
    fCommonContextSeqRm(vCtxDft, pSeqId, pP0, pP1);
  }
}

void common_memory::mSeqCp(dgs_seq_id pSeqIdSrc, dgs_seq_id pSeqIdDst, dgs_pos pP0, dgs_pos pP1) const {
  fCommonContextSeqCp(vCtxTgt, pSeqIdSrc, pSeqIdDst, pP0, pP1);
  if (vCtxDft) {
    fCommonContextSeqCp(vCtxDft, pSeqIdSrc, pSeqIdDst, pP0, pP1);
  }
}

void common_memory::mSeqAdd(dgs_seq_id pSeqId, dgs_pos pP0, dgs_pos pP1, dgs_pos pDelta) const {
  fCommonContextSeqAdd(vCtxTgt, pSeqId, pP0, pP1, pDelta);
  if (vCtxDft) {
    fCommonContextSeqAdd(vCtxDft, pSeqId, pP0, pP1, pDelta);
  }
}

void fCommonSetAdapterLora(struct dgs_context * pCtx, std::vector<common_adapter_lora_info> & lora) {
  std::vector<dgs_adapter_lora *> lLoras;
  std::vector<float>              lScales;

  for (auto & vLa : lora) {
    lLoras.push_back(vLa.vPtr);
    lScales.push_back(vLa.vScale);
  }

  fDgsSetAdaptersLora(pCtx, lLoras.data(), lLoras.size(), lScales.data());
}

struct dgs_model_params fCommonModelParamsToDgs(common_params & pParams) {
  auto vMparams = fDgsModelDefaultParams();

  if (!pParams.lDevices.empty()) {
    vMparams.vDevices = pParams.lDevices.data();
  }

  vMparams.vNGpuLayers    = pParams.vNGpuLayers;
  vMparams.vMainGpu       = pParams.vMainGpu;
  vMparams.vSplitMode     = pParams.vSplitMode;
  vMparams.vLoadMode      = pParams.vLoadMode;
  vMparams.vLazyMode      = pParams.vLazyMode;
  vMparams.vTensorSplit   = pParams.aTensorSplit;
  vMparams.vCheckTensors  = pParams.vCheckTensors;
  vMparams.vUseExtraBufts = !pParams.vNoExtraBufts;
  vMparams.vNoHost        = pParams.vNoHost;

  if (pParams.lKvOverrides.empty()) {
    vMparams.vKvOverrides = NULL;
  } else {
    GGML_ASSERT(pParams.lKvOverrides.back().aKey[0] == 0 && "KV overrides not terminated with empty key");
    vMparams.vKvOverrides = pParams.lKvOverrides.data();
  }

  if (pParams.lTensorBuftOverrides.empty()) {
    vMparams.vTensorBuftOverrides = NULL;
  } else {
    GGML_ASSERT(pParams.lTensorBuftOverrides.back().vPattern == nullptr &&
                "Tensor buffer overrides not terminated with empty pattern");
    vMparams.vTensorBuftOverrides = pParams.lTensorBuftOverrides.data();
  }

  vMparams.vProgressCallback         = pParams.vLoadProgressCallback;
  vMparams.vProgressCallbackUserData = pParams.vLoadProgressCallbackUserData;
  vMparams.vNoAlloc                  = pParams.vNoAlloc;
  vMparams.vLoadMtp =
    pParams.vLoadMtp || std::find(pParams.vSpeculative.lTypes.begin(), pParams.vSpeculative.lTypes.end(),
                                  cCommonSpeculativeTypeDraftMtp) != pParams.vSpeculative.lTypes.end();

  return vMparams;
}

struct dgs_context_params fCommonContextParamsToDgs(const common_params & pParams) {
  auto vCparams = fDgsContextDefaultParams();

  vCparams.vNCtx              = pParams.vNCtx;
  vCparams.vNSeqMax           = pParams.vNParallel;
  vCparams.vNRsSeq            = pParams.vSpeculative.mNeedNRsSeq();
  vCparams.vNOutputsMax       = std::max(pParams.vNOutputsMax, 0);
  vCparams.vNOutputsMaxPerSeq = std::max(pParams.vNOutputsMaxPerSeq, 0);
  vCparams.vNBatch            = pParams.vNBatch;
  vCparams.vNUbatch           = pParams.vNUbatch;
  vCparams.vNThreads          = pParams.vCpuparams.vNThreads;
  vCparams.vNThreadsBatch =
    pParams.vCpuparamsBatch.vNThreads == -1 ? pParams.vCpuparams.vNThreads : pParams.vCpuparamsBatch.vNThreads;
  vCparams.vEmbeddings      = pParams.vEmbedding;
  vCparams.vRopeScalingType = pParams.vRopeScalingType;
  vCparams.vRopeFreqBase    = pParams.vRopeFreqBase;
  vCparams.vRopeFreqScale   = pParams.vRopeFreqScale;
  vCparams.vYarnExtFactor   = pParams.vYarnExtFactor;
  vCparams.vYarnAttnFactor  = pParams.vYarnAttnFactor;
  vCparams.vYarnBetaFast    = pParams.vYarnBetaFast;
  vCparams.vYarnBetaSlow    = pParams.vYarnBetaSlow;
  vCparams.vYarnOrigCtx     = pParams.vYarnOrigCtx;
  vCparams.vPoolingType     = pParams.vPoolingType;
  vCparams.vAttentionType   = pParams.vAttentionType;
  vCparams.vFlashAttnType   = pParams.vFlashAttnType;
  vCparams.vCbEval          = pParams.vCbEval;
  vCparams.vCbEvalUserData  = pParams.vCbEvalUserData;
  vCparams.vOffloadKqv      = !pParams.vNoKvOffload;
  vCparams.vNoPerf          = pParams.vNoPerf;
  vCparams.vOpOffload       = !pParams.vNoOpOffload;
  vCparams.vSwaFull         = pParams.vSwaFull;
  vCparams.vKvUnified       = pParams.vKvUnified;

  vCparams.vTypeK = pParams.vCacheTypeK;
  vCparams.vTypeV = pParams.vCacheTypeV;

  return vCparams;
}

//
// Threadpool utils
//

struct ggml_threadpool_params fGgmlThreadpoolParamsFromCpuParams(const common_cpu_params & pParams) {
  struct ggml_threadpool_params vTpp;

  ggml_threadpool_params_init(&vTpp, pParams.vNThreads);  // setup the defaults

  if (pParams.vMaskValid) {
    std::memcpy(&vTpp.cpumask, &pParams.aCpumask, GGML_MAX_N_THREADS);
  }

  vTpp.prio       = pParams.vPriority;
  vTpp.poll       = pParams.vPoll;
  vTpp.strict_cpu = pParams.vStrictCpu;

  return vTpp;
}

common_threadpools::~common_threadpools() {
  if (!vFreeFn) {
    return;
  }
  vFreeFn(vThreadpool);
  vFreeFn(vThreadpoolBatch);
}

void common_threadpools::init(dgs_context * pCtx, const common_params & pParams) {
  GGML_ASSERT(!vThreadpool);
  GGML_ASSERT(!vThreadpoolBatch);

  COM_INF("dgs threadpool init, n_threads = %d\n", (int) pParams.vCpuparams.vNThreads);

  auto * vCpuDev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
  if (!vCpuDev) {
    COM_WRN("%s", "no CPU backend found\n");
    return;
  }
  auto * vReg = ggml_backend_dev_backend_reg(vCpuDev);
  auto * vGgmlThreadpoolNewFn =
    (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(vReg, "ggml_threadpool_new");
  vFreeFn = (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(vReg, "ggml_threadpool_free");

  struct ggml_threadpool_params vTppBatch = fGgmlThreadpoolParamsFromCpuParams(pParams.vCpuparamsBatch);
  struct ggml_threadpool_params vTpp      = fGgmlThreadpoolParamsFromCpuParams(pParams.vCpuparams);

  // each pool needs to match the respective n_threads exactly
  if (!ggml_threadpool_params_match(&vTpp, &vTppBatch)) {
    vThreadpoolBatch = vGgmlThreadpoolNewFn(&vTppBatch);
    if (!vThreadpoolBatch) {
      COM_WRN("batch threadpool create failed : n_threads %d\n", vTppBatch.n_threads);
      return;
    }

    // start the non-batch threadpool in the paused state
    vTpp.paused = true;
  }

  vThreadpool = vGgmlThreadpoolNewFn(&vTpp);
  if (!vThreadpool) {
    COM_WRN("threadpool create failed : n_threads %d\n", vTpp.n_threads);
    vFreeFn(vThreadpoolBatch);
    vThreadpoolBatch = nullptr;
    return;
  }

  fDgsAttachThreadpool(pCtx, vThreadpool, vThreadpoolBatch);
}

//
// Vocab utils
//

std::vector<dgs_token> fCommonTokenize(const struct dgs_context * pCtx,
                                       const std::string &        pText,
                                       bool                       pAddSpecial,
                                       bool                       pParseSpecial) {
  const dgs_model * vModel = fDgsGetModel(pCtx);
  const dgs_vocab * vVocab = fDgsModelGetVocab(vModel);
  return fCommonTokenize(vVocab, pText, pAddSpecial, pParseSpecial);
}

std::vector<dgs_token> fCommonTokenize(const struct dgs_vocab * pVocab,
                                       const std::string &      pText,
                                       bool                     pAddSpecial,
                                       bool                     pParseSpecial) {
  // upper limit for the number of tokens
  int                    vNTokens = pText.length() + 2 * pAddSpecial;
  std::vector<dgs_token> lResult(vNTokens);
  vNTokens =
    fDgsTokenize(pVocab, pText.data(), pText.length(), lResult.data(), lResult.size(), pAddSpecial, pParseSpecial);
  if (vNTokens == std::numeric_limits<int32_t>::min()) {
    throw std::runtime_error("Tokenization failed: input text too large, tokenization result exceeds int32_t limit");
  }
  if (vNTokens < 0) {
    lResult.resize(-vNTokens);
    int vCheck =
      fDgsTokenize(pVocab, pText.data(), pText.length(), lResult.data(), lResult.size(), pAddSpecial, pParseSpecial);
    GGML_ASSERT(vCheck == -vNTokens);
  } else {
    lResult.resize(vNTokens);
  }
  return lResult;
}

std::string fCommonTokenToPiece(const struct dgs_context * pCtx, dgs_token token, bool pSpecial) {
  const dgs_model * vModel = fDgsGetModel(pCtx);
  const dgs_vocab * vVocab = fDgsModelGetVocab(vModel);
  return fCommonTokenToPiece(vVocab, token, pSpecial);
}

std::string fCommonTokenToPiece(const struct dgs_vocab * pVocab, dgs_token token, bool pSpecial) {
  std::string vPiece;
  vPiece.resize(vPiece.capacity());  // using string internal cache, 15 bytes + '\n'
  const int cNChars = fDgsTokenToPiece(pVocab, token, &vPiece[0], vPiece.size(), 0, pSpecial);
  if (cNChars < 0) {
    vPiece.resize(-cNChars);
    int vCheck = fDgsTokenToPiece(pVocab, token, &vPiece[0], vPiece.size(), 0, pSpecial);
    GGML_ASSERT(vCheck == -cNChars);
  } else {
    vPiece.resize(cNChars);
  }

  return vPiece;
}

std::string fCommonDetokenize(const struct dgs_context * pCtx, const std::vector<dgs_token> & pTokens, bool pSpecial) {
  const dgs_model * vModel = fDgsGetModel(pCtx);
  const dgs_vocab * vVocab = fDgsModelGetVocab(vModel);
  return fCommonDetokenize(vVocab, pTokens, pSpecial);
}

std::string fCommonDetokenize(const struct dgs_vocab * pVocab, const std::vector<dgs_token> & pTokens, bool pSpecial) {
  std::string vText;
  vText.resize(std::max(vText.capacity(), pTokens.size()));
  int32_t vNChars = fDgsDetokenize(pVocab, pTokens.data(), (int32_t) pTokens.size(), &vText[0], (int32_t) vText.size(),
                                   false, pSpecial);
  if (vNChars < 0) {
    vText.resize(-vNChars);
    vNChars = fDgsDetokenize(pVocab, pTokens.data(), (int32_t) pTokens.size(), &vText[0], (int32_t) vText.size(), false,
                             pSpecial);
    GGML_ASSERT(vNChars <= (int32_t) vText.size());  // whitespace trimming is performed after per-token detokenization
  }

  vText.resize(vNChars);

  // NOTE: the original tokenizer decodes bytes after collecting the pieces.
  return vText;
}

//
// Embedding utils
//

void fCommonEmbdNormalize(const float * pInp, float * pOut, int n, int pEmbdNorm) {
  double vSum = 0.0;

  switch (pEmbdNorm) {
    case -1:  // no normalisation
      vSum = 1.0;
      break;
    case 0:  // max absolute
      for (int vI = 0; vI < n; vI++) {
        if (vSum < std::abs(pInp[vI])) {
          vSum = std::abs(pInp[vI]);
        }
      }
      vSum /= 32760.0;  // make an int16 range
      break;
    case 2:             // euclidean
      for (int vI = 0; vI < n; vI++) {
        vSum += pInp[vI] * pInp[vI];
      }
      vSum = std::sqrt(vSum);
      break;
    default:  // p-norm (euclidean is p-norm p=2)
      for (int vI = 0; vI < n; vI++) {
        vSum += std::pow(std::abs(pInp[vI]), pEmbdNorm);
      }
      vSum = std::pow(vSum, 1.0 / pEmbdNorm);
      break;
  }

  const float cNorm = vSum > 0.0 ? 1.0 / vSum : 0.0f;

  for (int vI = 0; vI < n; vI++) {
    pOut[vI] = pInp[vI] * cNorm;
  }
}

float fCommonEmbdSimilarityCos(const float * pEmbd1, const float * pEmbd2, int n) {
  double vSum  = 0.0;
  double vSum1 = 0.0;
  double vSum2 = 0.0;

  for (int vI = 0; vI < n; vI++) {
    vSum += pEmbd1[vI] * pEmbd2[vI];
    vSum1 += pEmbd1[vI] * pEmbd1[vI];
    vSum2 += pEmbd2[vI] * pEmbd2[vI];
  }

  // Handle the case where one or both vectors are zero vectors
  if (vSum1 == 0.0 || vSum2 == 0.0) {
    if (vSum1 == 0.0 && vSum2 == 0.0) {
      return 1.0f;  // two zero vectors are similar
    }
    return 0.0f;
  }

  return vSum / (sqrt(vSum1) * sqrt(vSum2));
}

//
// Control vector utils
//

static common_control_vector_data fCommonControlVectorLoadOne(const common_control_vector_load_info & pLoadInfo) {
  common_control_vector_data vResult = { -1, {} };

  ggml_context *          vCtx            = nullptr;
  struct gguf_init_params vMetaGgufParams = {
    /* .no_alloc = */ false,
    /* .ctx      = */ &vCtx,
  };
  struct gguf_context * vCtxGguf = gguf_init_from_file(pLoadInfo.vFname.c_str(), vMetaGgufParams);
  if (!vCtxGguf) {
    COM_ERR("failed to load control vector file from %s\n", pLoadInfo.vFname.c_str());
    return vResult;
  }

  int32_t vNTensors = gguf_get_n_tensors(vCtxGguf);
  if (vNTensors == 0) {
    COM_WRN("no direction tensors found in %s\n", pLoadInfo.vFname.c_str());
  }

  for (int vI = 0; vI < vNTensors; vI++) {
    std::string name = gguf_get_tensor_name(vCtxGguf, vI);

    int vLayerIdx = -1;

    // split on '.'
    size_t vDotpos = name.find('.');
    if (vDotpos != std::string::npos && name.substr(0, vDotpos) == "direction") {
      try {
        vLayerIdx = std::stoi(name.substr(vDotpos + 1));
      } catch (...) {
        vLayerIdx = -1;
      }
    }
    if (vLayerIdx < 0) {
      COM_ERR("invalid/unparsable direction tensor layer index in %s\n", pLoadInfo.vFname.c_str());
      vResult.vNEmbd = -1;
      break;
    } else if (vLayerIdx == 0) {
      COM_ERR("invalid (zero) direction tensor layer index in %s\n", pLoadInfo.vFname.c_str());
      vResult.vNEmbd = -1;
      break;
    }

    struct ggml_tensor * vTensor = ggml_get_tensor(vCtx, name.c_str());
    if (vTensor->type != GGML_TYPE_F32) {
      COM_ERR("invalid (non-F32) direction tensor type in %s\n", pLoadInfo.vFname.c_str());
      vResult.vNEmbd = -1;
      break;
    }
    if (ggml_n_dims(vTensor) != 1) {
      COM_ERR("invalid (non-1D) direction tensor shape in %s\n", pLoadInfo.vFname.c_str());
      vResult.vNEmbd = -1;
      break;
    }

    if (vResult.vNEmbd == -1) {
      vResult.vNEmbd = ggml_nelements(vTensor);
    } else if (ggml_nelements(vTensor) != vResult.vNEmbd) {
      COM_ERR("direction tensor in %s does not match previous dimensions\n", pLoadInfo.vFname.c_str());
      vResult.vNEmbd = -1;
      break;
    }

    // extend if necessary - do not store data for layer 0 (it's not used)
    vResult.data.resize(std::max(vResult.data.size(), static_cast<size_t>(vResult.vNEmbd * vLayerIdx)), 0.0f);

    const float * vSrc = (const float *) vTensor->data;
    float *       vDst = vResult.data.data() + vResult.vNEmbd * (vLayerIdx - 1);  // layer 1 at [0]
    for (int vJ = 0; vJ < vResult.vNEmbd; vJ++) {
      vDst[vJ] += vSrc[vJ] * pLoadInfo.vStrength;  // allows multiple directions for same layer in same file
    }
  }

  if (vResult.vNEmbd == -1) {
    COM_WRN("skipping %s due to invalid direction tensors\n", pLoadInfo.vFname.c_str());
    vResult.data.clear();
  }

  gguf_free(vCtxGguf);
  ggml_free(vCtx);

  return vResult;
}

common_control_vector_data fCommonControlVectorLoad(const std::vector<common_control_vector_load_info> & pLoadInfos) {
  common_control_vector_data vResult = { -1, {} };

  for (const auto & info : pLoadInfos) {
    auto vCur = fCommonControlVectorLoadOne(info);

    if (vCur.vNEmbd == -1) {
      vResult.vNEmbd = -1;
      break;
    }
    if (vResult.vNEmbd != -1 && vResult.vNEmbd != vCur.vNEmbd) {
      COM_ERR("control vectors in %s does not match previous dimensions\n", info.vFname.c_str());
      vResult.vNEmbd = -1;
      break;
    }

    if (vResult.vNEmbd == -1) {
      vResult = std::move(vCur);
    } else {
      vResult.data.resize(std::max(vResult.data.size(), vCur.data.size()), 0.0f);  // extend if necessary
      for (size_t vI = 0; vI < vCur.data.size(); vI++) {
        vResult.data[vI] += vCur.data[vI];
      }
    }
  }

  if (vResult.vNEmbd == -1) {
    COM_ERR("%s", "no valid control vector files passed\n");
    vResult.data.clear();
  }

  return vResult;
}

// TODO make all command line args case-insensitive
static inline bool fEqCaseInsensitive(const char * pA, const char * pB) {
  return !
#if defined(_MSC_VER)
    _stricmp
#else
    strcasecmp
#endif  // defined(_MSC_VER)
    (pA, pB);
}

// TODO simplify to use just log and exp
static const float cKLog2 = std::log(2.f);

bool fCommonReplayLastToken(struct dgs_context * pCtx, dgs_token pLastToken, int32_t pPos) {
  common_batch vBatch(pCtx);
  vBatch.mAdd(pLastToken, pPos, 0, true);

  if (fDgsProcess(pCtx, cDgsProcessTypeDecode, vBatch.get())) {
    LOG_ERR("%s: failed to replay last token\n", __func__);
    return false;
  }
  return true;
}

common_batch::common_batch(dgs_context * pCtx) : vBatch(fDgsBatchExtInit(pCtx)) {
  const auto cRopeType = fDgsModelRopeType(fDgsGetModel(pCtx));
  vNPos                = cRopeType == cDgsRopeTypeMrope || cRopeType == cDgsRopeTypeImrope ? GGML_MROPE_SECTIONS : 1;
}

void common_batch::clear() {
  lTokens.clear();
}

int32_t common_batch::mAdd(dgs_token pId, dgs_pos pPos, dgs_seq_id pSeqId, bool pOutput) {
  lTokens.push_back({
    pId, { pPos, 0, 0, 0 },
     pSeqId, pOutput, { nullptr, 0, 0 },
     { nullptr, 0, 0 },
     {}
  });
  return size() - 1;
}

int32_t common_batch::mAdd(dgs_token pId, dgs_pos pPos, const std::vector<dgs_seq_id> & pSeqIds, bool pOutput) {
  GGML_ASSERT(!pSeqIds.empty());

  const int32_t cIdx = mAdd(pId, pPos, pSeqIds[0], pOutput);
  for (size_t vS = 1; vS < pSeqIds.size(); ++vS) {
    mAddSeq(cIdx, pSeqIds[vS]);
  }
  return cIdx;
}

bool common_batch::mAddSeq(int32_t pIdx, dgs_seq_id pSeqId) {
  if (pIdx < 0 || pIdx >= size()) {
    return false;
  }
  lTokens[pIdx].lSeqIdsExtra.push_back(pSeqId);
  return true;
}

bool common_batch::mSetOutput(int32_t pIdx, bool value) {
  if (pIdx < 0 || pIdx >= size()) {
    return false;
  }
  lTokens[pIdx].vOutput = value;
  return true;
}

bool common_batch::mSetEmbd(int32_t pIdx, dgs_embd pEmbd) {
  if (pIdx < 0 || pIdx >= size() || lTokens[pIdx].vEmbd.data != nullptr) {
    return false;
  }
  lTokens[pIdx].vEmbd = pEmbd;
  return true;
}

bool common_batch::mSetEmbdState(int32_t pIdx, dgs_embd pState) {
  if (pIdx < 0 || pIdx >= size() || lTokens[pIdx].vState.data != nullptr) {
    return false;
  }
  lTokens[pIdx].vState = pState;
  return true;
}

int32_t common_batch::mAddEmbd(dgs_embd pEmbd, const dgs_pos * pPos, dgs_seq_id pSeqId, bool pOutput) {
  token vT = {
    DGS_TOKEN_NULL, { 0, 0, 0, 0 },
     pSeqId, pOutput, pEmbd, { nullptr, 0, 0 },
     {}
  };
  for (int32_t vJ = 0; vJ < vNPos; ++vJ) {
    vT.aPos[vJ] = pPos[vJ];
  }
  lTokens.push_back(vT);
  return size() - 1;
}

dgs_batch_ext * common_batch::mGetSubBatch(int32_t pOff, int32_t n) {
  GGML_ASSERT(vBatch && "common_batch was not initialized with a context");
  GGML_ASSERT(pOff >= 0 && n >= 0 && pOff + n <= size());

  dgs_batch_ext * vRes = vBatch.get();
  fDgsBatchExtClear(vRes);

  for (int32_t vI = pOff; vI < pOff + n; ++vI) {
    const token & vT = lTokens[vI];

    int32_t vIdx;
    if (vT.vId != DGS_TOKEN_NULL) {
      vIdx = fDgsBatchExtAddToken(vRes, vT.vSeqId, vT.vId);
      if (vIdx < 0) {
        GGML_ABORT("%s: failed to add token %d at index %d (error %d, n = %d)\n", __func__, vT.vId, vI, vIdx, n);
      }
      fDgsBatchExtSetPos(vRes, vIdx, vT.aPos.data());
      if (vT.vEmbd.data && !fDgsBatchExtSetEmbdToken(vRes, vIdx, vT.vEmbd)) {
        GGML_ABORT("%s: failed to set the embedding of token %d at index %d\n", __func__, vT.vId, vI);
      }
    } else {
      vIdx = fDgsBatchExtAddEmbd(vRes, vT.vSeqId, vT.vEmbd);
      if (vIdx < 0) {
        GGML_ABORT("%s: failed to add embedding at index %d (error %d, n = %d)\n", __func__, vI, vIdx, n);
      }
      fDgsBatchExtSetPos(vRes, vIdx, vT.aPos.data());
    }
    GGML_ASSERT(vIdx == vI - pOff);

    for (const dgs_seq_id cSeqId : vT.lSeqIdsExtra) {
      if (!fDgsBatchExtAddSeq(vRes, vIdx, cSeqId)) {
        GGML_ABORT("%s: failed to add seq %d to the entry at index %d\n", __func__, cSeqId, vI);
      }
    }
    if (vT.vOutput) {
      fDgsBatchExtSetOutputLogits(vRes, vIdx, true);
    }
    if (vT.vState.data) {
      fDgsBatchExtSetEmbdState(vRes, vIdx, vT.vState);  // contexts without a state input ignore it
    }
    if (vT.vDecisionOrder != 0) {
      fDgsBatchExtSetDecisionOrder(vRes, vIdx, (dgs_decision_order) vT.vDecisionOrder);
    }
  }

  return vRes;
}

common_batch fCommonBatchGetOne(dgs_context * pCtx, const dgs_token * pTokens, int32_t pNTokens) {
  common_batch vBatch(pCtx);

  auto    vMem = fDgsGetMemory(pCtx);
  dgs_pos vPos = fDgsMemorySeqPosMax(vMem, 0) + 1;  // -1 + 1 == 0 when the memory is empty

  for (int32_t vI = 0; vI < pNTokens; ++vI) {
    const bool cOutput = vI == pNTokens - 1;
    vBatch.mAdd(pTokens[vI], vPos, 0, cOutput);
    vPos++;
  }

  return vBatch;
}

common_batch fCommonBatchGetOne(dgs_context * pCtx, const dgs_tokens & pTokens) {
  return fCommonBatchGetOne(pCtx, pTokens.data(), (int32_t) pTokens.size());
}

bool fCommonPromptBatchDecode(struct dgs_context * pCtx,
                              const dgs_tokens &   pAllTokens,
                              int                  pNNew,
                              int &                pNPast,
                              int                  pNBatch,
                              std::string_view     pStatePath,
                              bool                 pSaveState) {
  if (pNNew == 0) {
    return true;
  }
  const int cOffset = pAllTokens.size() - pNNew;

  if (pSaveState && pNNew > 1) {
    const int cNTokensBeforeLast = pNNew - 1;

    GGML_ASSERT(pNNew <= pNBatch);

    // Decode all but the last token so we can save the memory state before decoding the last token.
    // This is done so we can restore the session state later and replay the last token.
    // Memory implementations in recurrent/hybrid models don't support removing tokens from their
    // memory, so we can't just remove the last token from the memory and replay the last token which
    // is the reason for this logic.
    dgs_tokens   lPrefixTokens(pAllTokens.begin() + cOffset, pAllTokens.begin() + cOffset + cNTokensBeforeLast);
    common_batch vBatchPrefix = fCommonBatchGetOne(pCtx, lPrefixTokens);
    if (fDgsProcess(pCtx, cDgsProcessTypeDecode, vBatchPrefix.get())) {
      COM_ERR("%s", "failed to eval\n");
      return false;
    }
    pNPast += cNTokensBeforeLast;

    fDgsStateSaveFile(pCtx, pStatePath.data(), pAllTokens.data(), pAllTokens.size());
    COM_INF("saved session before last token to %s, n_new = %zu\n", pStatePath.data(), pAllTokens.size());

    common_batch vBatchLast(pCtx);
    vBatchLast.mAdd(pAllTokens.back(), pNPast, 0, true);

    if (fDgsProcess(pCtx, cDgsProcessTypeDecode, vBatchLast.get())) {
      COM_ERR("%s", "failed to eval last token\n");
      return false;
    }
    pNPast++;
  } else {
    dgs_tokens   lNewTokens(pAllTokens.begin() + cOffset, pAllTokens.begin() + cOffset + pNNew);
    common_batch vBatch = fCommonBatchGetOne(pCtx, lNewTokens);
    if (fDgsProcess(pCtx, cDgsProcessTypeDecode, vBatch.get())) {
      COM_ERR("%s", "failed to eval\n");
      return false;
    }
    pNPast += pNNew;
  }

  return true;
}

size_t common_prompt_checkpoint::size() const {
  return lDataTgt.size() + lDataDft.size() + lDataSpec.size();
}

bool common_prompt_checkpoint::empty() const {
  return lDataTgt.empty();
}

void common_prompt_checkpoint::clear() {
  vNTokens = 0;

  vPosMin = 0;
  vPosMax = 0;

  lDataTgt.clear();
  lDataDft.clear();
  lDataSpec.clear();
}

void common_prompt_checkpoint::mUpdatePos(int64_t pNTokens, dgs_pos pPosMin, dgs_pos pPosMax) {
  this->vNTokens = pNTokens;
  this->vPosMin  = pPosMin;
  this->vPosMax  = pPosMax;
}

void common_prompt_checkpoint::mUpdateTgt(dgs_context * pCtx, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags) {
  if (pCtx == nullptr) {
    return;
  }

  const size_t cCkptSize = fDgsStateSeqGetSizeExt(pCtx, pSeqId, pFlags);

  lDataTgt.resize(cCkptSize);

  const size_t n = fDgsStateSeqGetDataExt(pCtx, lDataTgt.data(), cCkptSize, pSeqId, pFlags);
  if (n != cCkptSize) {
    GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", cCkptSize, n);
  }
}

void common_prompt_checkpoint::mUpdateDft(dgs_context * pCtx, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags) {
  if (pCtx == nullptr) {
    return;
  }

  const size_t cCkptSize = fDgsStateSeqGetSizeExt(pCtx, pSeqId, pFlags);

  lDataDft.resize(cCkptSize);

  const size_t n = fDgsStateSeqGetDataExt(pCtx, lDataDft.data(), cCkptSize, pSeqId, pFlags);
  if (n != cCkptSize) {
    GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", cCkptSize, n);
  }
}

bool common_prompt_checkpoint::mLoadTgt(dgs_context * pCtx, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags) const {
  if (pCtx == nullptr) {
    return true;
  }

  if (lDataTgt.empty()) {
    return true;
  }

  const size_t n = fDgsStateSeqSetDataExt(pCtx, lDataTgt.data(), lDataTgt.size(), pSeqId, pFlags);
  return n == lDataTgt.size();
}

bool common_prompt_checkpoint::mLoadDft(dgs_context * pCtx, dgs_seq_id pSeqId, dgs_state_seq_flags pFlags) const {
  if (pCtx == nullptr) {
    return true;
  }

  if (lDataDft.empty()) {
    return true;
  }

  const size_t n = fDgsStateSeqSetDataExt(pCtx, lDataDft.data(), lDataDft.size(), pSeqId, pFlags);
  return n == lDataDft.size();
}

void common_prompt_checkpoint::mClearTgt() {
  lDataTgt.clear();
}

void common_prompt_checkpoint::mClearDft() {
  lDataDft.clear();
  lDataSpec.clear();
}
