#include "arg.h"

#include "build-info.h"
#include "chat.h"
#include "common.h"
#include "dgs.h"
#include "download.h"
#include "json-schema-to-grammar.h"
#include "json.h"
#include "log.h"
#include "preset.h"
#include "sampling.h"
#include "speculative.h"

// fix problem with std::min and std::max
#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <shellapi.h>
#  include <windows.h>
#endif

#include <algorithm>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <filesystem>
#include <fstream>
#include <list>
#include <numeric>
#include <regex>
#include <set>
#include <string>
#include <system_error>
#include <thread>  // for hardware_concurrency
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

#define DGS_MAX_URL_LENGTH 2084  // Maximum URL Length in Chrome: 2083

using json = common_json;
using namespace common_arg_utils;

static std::initializer_list<enum dgs_example> vMmprojExamples = {
  cDgsExampleMtmd,
  cDgsExampleServer,
  cDgsExampleCli,
  cDgsExampleTts,
};

static std::string fReadFile(const std::string & pFname) {
  std::ifstream vFile(pFname);
  if (!vFile) {
    throw std::runtime_error(fStringFormat("error: failed to open file '%s'\n", pFname.c_str()));
  }
  std::string vContent((std::istreambuf_iterator<char>(vFile)), std::istreambuf_iterator<char>());
  vFile.close();
  return vContent;
}

static const std::vector<common_arg> & fGetCommonArgDefs() {
  static const std::vector<common_arg> lOptions = [] {
    common_params vParams;
    auto          vCtx = fCommonParamsParserInit(vParams, cDgsExampleServer, nullptr);
    return vCtx.lOptions;
  }();
  return lOptions;
}

common_arg & common_arg::mSetExamples(std::initializer_list<enum dgs_example> pExamples) {
  this->sExamples = pExamples;
  return *this;
}

common_arg & common_arg::mSetExcludes(std::initializer_list<enum dgs_example> pExcludes) {
  this->sExcludes = pExcludes;
  return *this;
}

common_arg & common_arg::mSetEnv(const char * pEnv) {
  vHelp      = vHelp + "\n(env: " + pEnv + ")";
  this->vEnv = pEnv;
  return *this;
}

common_arg & common_arg::mSetSampling() {
  vIsSampling = true;
  return *this;
}

common_arg & common_arg::mSetSpec() {
  vIsSpec = true;
  return *this;
}

common_arg & common_arg::mSetPresetOnly() {
  vIsPresetOnly = true;
  return *this;
}

bool common_arg::mInExample(enum dgs_example pEx) {
  return sExamples.find(pEx) != sExamples.end();
}

bool common_arg::mIsExclude(enum dgs_example pEx) {
  return sExcludes.find(pEx) != sExcludes.end();
}

bool common_arg::mGetValueFromEnv(std::string & pOutput) const {
  if (vEnv == nullptr) {
    return false;
  }
  if (!lArgsNeg.empty()) {
    // for compatibility, we need to check DGS_ARG_NO_ env as well
    std::string vNegEnv = vEnv;
    fStringReplaceAll(vNegEnv, "DGS_ARG_", "DGS_ARG_NO_");
    char * vNegValue = std::getenv(vNegEnv.c_str());
    if (vNegValue) {
      pOutput = "0";  // falsey
      return true;
    }
  }
  char * value = std::getenv(vEnv);
  if (value) {
    pOutput = value;
    return true;
  }
  return false;
}

bool common_arg::mHasValueFromEnv() const {
  if (vEnv != nullptr && !lArgsNeg.empty()) {
    // for compatibility, we need to check DGS_ARG_NO_ env as well
    std::string vNegEnv = vEnv;
    fStringReplaceAll(vNegEnv, "DGS_ARG_", "DGS_ARG_NO_");
    if (std::getenv(vNegEnv.c_str())) {
      return true;
    }
  }
  return vEnv != nullptr && std::getenv(vEnv);
}

static std::vector<std::string> fBreakStrIntoLines(std::string pInput, size_t pMaxCharPerLine) {
  std::vector<std::string> lResult;
  std::istringstream       vIss(pInput);
  std::string              vLine;
  auto                     vAddLine = [&](const std::string & pL) {
    if (pL.length() <= pMaxCharPerLine) {
      lResult.push_back(pL);
    } else {
      std::istringstream vLineStream(pL);
      std::string        vWord, vCurrentLine;
      while (vLineStream >> vWord) {
        if (vCurrentLine.length() + !vCurrentLine.empty() + vWord.length() > pMaxCharPerLine) {
          if (!vCurrentLine.empty()) {
            lResult.push_back(vCurrentLine);
          }
          vCurrentLine = vWord;
        } else {
          vCurrentLine += (!vCurrentLine.empty() ? " " : "") + vWord;
        }
      }
      if (!vCurrentLine.empty()) {
        lResult.push_back(vCurrentLine);
      }
    }
  };
  while (std::getline(vIss, vLine)) {
    vAddLine(vLine);
  }
  return lResult;
}

std::string common_arg::to_string() const {
  // params for printing to console
  const static int cNLeadingSpaces   = 40;
  const static int cNCharPerLineHelp = 70;  // TODO: detect this based on current console
  std::string      vLeadingSpaces(cNLeadingSpaces, ' ');

  std::ostringstream vSs;
  auto               lAllArgs = mGetArgs();  // also contains args_neg
  for (const auto & vArg : lAllArgs) {
    if (vArg == lAllArgs.front()) {
      if (lAllArgs.size() == 1) {
        vSs << vArg;
      } else {
        // first arg is usually abbreviation, we need padding to make it more beautiful
        auto vTmp    = std::string(vArg) + ", ";
        auto vSpaces = std::string(std::max(0, 7 - (int) vTmp.size()), ' ');
        vSs << vTmp << vSpaces;
      }
    } else {
      vSs << vArg << (vArg != lAllArgs.back() ? ", " : "");
    }
  }
  if (vValueHint) {
    vSs << " " << vValueHint;
  }
  if (vValueHint2) {
    vSs << " " << vValueHint2;
  }
  if (vSs.tellp() > cNLeadingSpaces - 3) {
    // current line is too long, add new line
    vSs << "\n" << vLeadingSpaces;
  } else {
    // padding between arg and help, same line
    vSs << std::string(vLeadingSpaces.size() - vSs.tellp(), ' ');
  }
  const auto lHelpLines = fBreakStrIntoLines(vHelp, cNCharPerLineHelp);
  for (const auto & vLine : lHelpLines) {
    vSs << (&vLine == &lHelpLines.front() ? "" : vLeadingSpaces) << vLine << "\n";
  }
  return vSs.str();
}

std::vector<std::string> common_arg::mGetArgs() const {
  std::vector<std::string> lResult;
  for (const auto & vArg : lArgs) {
    lResult.push_back(std::string(vArg));
  }
  for (const auto & vArg : lArgsNeg) {
    lResult.push_back(std::string(vArg));
  }
  return lResult;
}

std::vector<std::string> common_arg::mGetEnv() const {
  std::vector<std::string> lResult;
  if (vEnv) {
    lResult.push_back(std::string(vEnv));
  }
  if (!lArgsNeg.empty() && vEnv) {
    // for compatibility, we need to add DGS_ARG_NO_ variant
    std::string vNegEnv = vEnv;
    fStringReplaceAll(vNegEnv, "DGS_ARG_", "DGS_ARG_NO_");
    lResult.push_back(vNegEnv);
  }
  return lResult;
}

//
// utils
//

// Helper function to parse tensor buffer override strings
static void fParseTensorBufferOverrides(const std::string &                           value,
                                        std::vector<dgs_model_tensor_buft_override> & pOverrides) {
  ggml_backend_load_all();

  std::map<std::string, ggml_backend_buffer_type_t> dBuftList;
  for (size_t vI = 0; vI < ggml_backend_dev_count(); ++vI) {
    auto * vDev  = ggml_backend_dev_get(vI);
    auto * vBuft = ggml_backend_dev_buffer_type(vDev);
    if (vBuft) {
      dBuftList[ggml_backend_buft_name(vBuft)] = vBuft;
    }
  }

  for (const auto & vOverride : fStringSplit<std::string>(value, ',')) {
    std::string::size_type vPos = vOverride.find('=');
    if (vPos == std::string::npos) {
      throw std::invalid_argument("invalid value");
    }
    std::string vTensorName = vOverride.substr(0, vPos);
    std::string vBufferType = vOverride.substr(vPos + 1);

    if (dBuftList.find(vBufferType) == dBuftList.end()) {
      printf("Available buffer types:\n");
      for (const auto & vIt : dBuftList) {
        printf("  %s\n", ggml_backend_buft_name(vIt.second));
      }
      throw std::invalid_argument("unknown buffer type");
    }
    // keep strings alive and avoid leaking memory by storing them in a static vector
    static std::list<std::string> lBuftOverrides;
    lBuftOverrides.push_back(vTensorName);
    pOverrides.push_back({ lBuftOverrides.back().c_str(), dBuftList.at(vBufferType) });
  }
}

static std::string fCleanFileName(const std::string & pFname) {
  std::string vCleanFname = pFname;
  fStringReplaceAll(vCleanFname, "\\", "_");
  fStringReplaceAll(vCleanFname, "/", "_");
  return vCleanFname;
}

struct handle_model_result {
  bool                vFoundMmproj = false;
  common_params_model vMmproj;

  bool                vFoundMtp = false;
  common_params_model vMtp;

  bool        vFoundPreset = false;
  std::string vPresetPath;
};

const std::vector<ggml_type> lKvCacheTypes = {
  GGML_TYPE_F32,  GGML_TYPE_F16,    GGML_TYPE_BF16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0,
  GGML_TYPE_Q4_1, GGML_TYPE_IQ4_NL, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
};

static ggml_type fKvCacheTypeFromStr(const std::string & pS) {
  for (const auto & type : lKvCacheTypes) {
    if (ggml_type_name(type) == pS) {
      return type;
    }
  }
  throw std::runtime_error("Unsupported cache type: " + pS);
}

static std::string fGetAllKvCacheTypes() {
  std::ostringstream vMsg;
  for (const auto & type : lKvCacheTypes) {
    vMsg << ggml_type_name(type) << (&type == &lKvCacheTypes.back() ? "" : ", ");
  }
  return vMsg.str();
}

static bool fParseBoolValue(const std::string & value) {
  if (fIsTruthy(value)) {
    return true;
  } else if (fIsFalsey(value)) {
    return false;
  } else {
    throw std::invalid_argument("invalid boolean value");
  }
}

[[noreturn]] static void fArgRemoved(const std::string & pMsg) {
  throw std::invalid_argument("the argument has been removed. " + pMsg);
}

//
// common_models_handler
//

static std::string fGetDefaultLocalPath(const std::string & pUrl) {
  auto vF = fStringSplit<std::string>(pUrl, '#').front();
  vF      = fStringSplit<std::string>(vF, '?').front();
  return fFsPathToUtf8(fFsGetCacheFile(fStringSplit<std::string>(vF, '/').back()));
}

static bool fSpecTypesIsDefault(const common_params & pParams) {
  return pParams.vSpeculative.lTypes == std::vector<enum common_speculative_type>{ cCommonSpeculativeTypeNone };
}

common_models_handler fCommonModelsHandlerInit(const common_params & pParams, dgs_example pCurrEx) {
  common_download_hf_plan vPlan;
  common_download_hf_plan vPlanSpec;
  common_download_opts    vOpts;

  const bool cSpecTypeDraftMtp = std::find(pParams.vSpeculative.lTypes.begin(), pParams.vSpeculative.lTypes.end(),
                                           cCommonSpeculativeTypeDraftMtp) != pParams.vSpeculative.lTypes.end();

  const bool cSpecTypeDraftDflash = std::find(pParams.vSpeculative.lTypes.begin(), pParams.vSpeculative.lTypes.end(),
                                              cCommonSpeculativeTypeDraftDflash) != pParams.vSpeculative.lTypes.end();

  const bool cSpecTypeDraftEagle3 = std::find(pParams.vSpeculative.lTypes.begin(), pParams.vSpeculative.lTypes.end(),
                                              cCommonSpeculativeTypeDraftEagle3) != pParams.vSpeculative.lTypes.end();

  const bool cSpecTypeDraftDspark = std::find(pParams.vSpeculative.lTypes.begin(), pParams.vSpeculative.lTypes.end(),
                                              cCommonSpeculativeTypeDraftDspark) != pParams.vSpeculative.lTypes.end();

  // only download mmproj if the current example is using it
  bool vUseMmproj = false;
  for (const auto & vEx : vMmprojExamples) {
    if (pCurrEx == vEx) {
      vUseMmproj = true;
      break;
    }
  }
  if (pCurrEx == cDgsExampleDownload) {
    vUseMmproj = true;
  }

  vOpts.vBearerToken    = pParams.vHfToken;
  vOpts.vOffline        = pParams.vOffline;
  vOpts.vDownloadMtp    = cSpecTypeDraftMtp;
  vOpts.vDownloadEagle3 = cSpecTypeDraftEagle3;
  vOpts.vDownloadDflash = cSpecTypeDraftDflash;
  vOpts.vDownloadDspark = cSpecTypeDraftDspark;
  vOpts.vDownloadMmproj =
    vUseMmproj && !pParams.vNoMmproj && pParams.vMmproj.vPath.empty() && pParams.vMmproj.vUrl.empty();

  if (!pParams.vModel.vHfRepo.empty()) {
    vPlan = fCommonDownloadGetHfPlan(pParams.vModel, vOpts);
  }

  if (!pParams.vSpeculative.vDraft.vMparams.vHfRepo.empty()) {
    // without a requested type, discover every sidecar the draft repo ships to infer the type later
    auto vOptsSpec = vOpts;
    if (fSpecTypesIsDefault(pParams)) {
      vOptsSpec.vDownloadMtp    = true;
      vOptsSpec.vDownloadDflash = true;
      vOptsSpec.vDownloadEagle3 = true;
      vOptsSpec.vDownloadDspark = true;
    }
    vPlanSpec = fCommonDownloadGetHfPlan(pParams.vSpeculative.vDraft.vMparams, vOptsSpec);
  }

  return common_models_handler{ vPlan, vPlanSpec, vOpts };
}

bool fCommonModelsHandlerIsPresetRepo(const common_models_handler & pHandler) {
  return !pHandler.vPlan.vPreset.vUrl.empty();
}

static std::vector<common_download_task> fBuildUrlTasks(const common_params_model & pModel,
                                                        common_download_opts        pOpts) {
  auto                              lParts = fCommonDownloadGetAllParts(pModel.vUrl);
  std::vector<common_download_task> lTasks;

  // single-part: download straight to model.path if the user gave one (-m), else the cache default
  if (lParts.size() == 1) {
    common_download_task vTask;
    vTask.vUrl       = lParts[0];
    vTask.vLocalPath = pModel.vPath.empty() ? fGetDefaultLocalPath(lParts[0]) : pModel.vPath;
    vTask.vOpts      = pOpts;
    lTasks.push_back(std::move(vTask));
    return lTasks;
  }

  // multi-part: place each part under the user's -m directory (if given), else the cache default
  std::string vBaseDir;
  if (!pModel.vPath.empty()) {
    auto vPos = pModel.vPath.rfind('/');
    vBaseDir  = vPos == std::string::npos ? std::string(".") : pModel.vPath.substr(0, vPos);
  }

  for (const auto & vPart : lParts) {
    common_download_task vTask;
    vTask.vUrl  = vPart;
    vTask.vOpts = pOpts;

    std::string vLocal = fGetDefaultLocalPath(vPart);
    if (!vBaseDir.empty()) {
      auto        vPos = vLocal.rfind('/');
      std::string name = vPos == std::string::npos ? vLocal : vLocal.substr(vPos + 1);
      vLocal           = vBaseDir + "/" + name;
    }
    vTask.vLocalPath = vLocal;
    lTasks.push_back(std::move(vTask));
  }
  return lTasks;
}

void fCommonModelsHandlerApply(common_models_handler &    pHandler,
                               common_params &            pParams,
                               common_download_callback * pCallback) {
  std::vector<common_download_task> lTasks;

  auto & vPlan     = pHandler.vPlan;
  auto & vPlanSpec = pHandler.vPlanSpec;

  auto vOpts      = pHandler.vOpts;  // copy
  vOpts.vCallback = pCallback;

  // handle plain "url" if needed
  auto vHandleUrl = [&](common_params_model & pModel) {
    if (!pModel.vUrl.empty()) {
      if (pModel.vPath.empty()) {
        pModel.vPath = fGetDefaultLocalPath(pModel.vUrl);
      }
    }
  };
  vHandleUrl(pParams.vModel);
  vHandleUrl(pParams.vMmproj);
  vHandleUrl(pParams.vSpeculative.vDraft.vMparams);

  // optionally, if docker repo is set, resolve it
  if (!pParams.vModel.vDockerRepo.empty()) {
    pParams.vModel.vUrl  = fCommonDockerResolveModel(pParams.vModel.vDockerRepo);
    pParams.vModel.vPath = fGetDefaultLocalPath(pParams.vModel.vUrl);
  }

  // handle plain "url" tasks (non-hf)
  if (!pParams.vModel.vUrl.empty()) {
    auto lUrlTasks = fBuildUrlTasks(pParams.vModel, vOpts);
    // the first part is what gets loaded, so point params.model.path at it
    if (!lUrlTasks.empty()) {
      std::string vFirstPath    = lUrlTasks.front().vLocalPath;
      lUrlTasks.front().vOnDone = [&, vFirstPath]() {
        pParams.vModel.vPath = vFirstPath;
      };
    }
    for (auto & vTask : lUrlTasks) {
      lTasks.push_back(std::move(vTask));
    }
  }
  if (!pParams.vMmproj.vUrl.empty()) {
    common_download_task vTask;
    vTask.vUrl       = pParams.vMmproj.vUrl;
    vTask.vLocalPath = pParams.vMmproj.vPath;
    vTask.vOpts      = vOpts;
    lTasks.push_back(vTask);
  }
  bool vHadSpecUrl = false;
  if (!pParams.vSpeculative.vDraft.vMparams.vUrl.empty()) {
    common_download_task vTask;
    vTask.vUrl       = pParams.vSpeculative.vDraft.vMparams.vUrl;
    vTask.vLocalPath = pParams.vSpeculative.vDraft.vMparams.vPath;
    vTask.vOpts      = vOpts;
    lTasks.push_back(vTask);
    vHadSpecUrl = true;
  }

  // handle hf_plan tasks
  auto vAddTasks = [&vOpts, &lTasks](const hf_cache::hf_files & pModelFiles, const hf_cache::hf_file & pPrimary,
                                     common_params_model & pModel) {
    for (size_t vI = 0; vI < pModelFiles.size(); ++vI) {
      auto & vModelFile = pModelFiles[vI];
      bool   vIsPrimary = (vModelFile.vPath == pPrimary.vPath);
      lTasks.emplace_back(vModelFile, vOpts, [&, vIsPrimary]() {
        if (vIsPrimary) {
          // the primary file is the first split (00001-of), use it as model path
          pModel.vPath = hf_cache::fFinalizeFile(vModelFile);
        } else {
          hf_cache::fFinalizeFile(vModelFile);
        }
      });
    }
  };

  // an explicit draft file selection (e.g. -md with -hfd) disables the sidecar resolution of the draft repo
  if (!pParams.vSpeculative.vDraft.vMparams.hf_file.empty()) {
    vPlanSpec.vMtp    = {};
    vPlanSpec.vDflash = {};
    vPlanSpec.vEagle3 = {};
    vPlanSpec.vDspark = {};
  }

  // infer the speculative type from the sidecar shipped by the draft repo when none is requested
  if (fSpecTypesIsDefault(pParams)) {
    if (!vPlanSpec.vMtp.vLocalPath.empty()) {
      pParams.vSpeculative.lTypes = { cCommonSpeculativeTypeDraftMtp };
      vPlanSpec.vDspark           = {};
      vPlanSpec.vDflash           = {};
      vPlanSpec.vEagle3           = {};
    } else if (!vPlanSpec.vDspark.vLocalPath.empty()) {
      // dspark outranks dflash, its sidecar carries the extra Markov head
      pParams.vSpeculative.lTypes = { cCommonSpeculativeTypeDraftDspark };
      vPlanSpec.vDflash           = {};
      vPlanSpec.vEagle3           = {};
    } else if (!vPlanSpec.vDflash.vLocalPath.empty()) {
      pParams.vSpeculative.lTypes = { cCommonSpeculativeTypeDraftDflash };
      vPlanSpec.vEagle3           = {};
    } else if (!vPlanSpec.vEagle3.vLocalPath.empty()) {
      pParams.vSpeculative.lTypes = { cCommonSpeculativeTypeDraftEagle3 };
    }
  }

  // infer the speculative type from the draft GGUF metadata when none is requested
  // note: reads only the first split - sharded drafts need an explicit --spec-type
  if (fSpecTypesIsDefault(pParams) && !pParams.vSpeculative.vDraft.vMparams.vPath.empty()) {
    const auto lTypesGguf = fCommonSpeculativeTypesFromGguf(pParams.vSpeculative.vDraft.vMparams.vPath);
    if (!lTypesGguf.empty()) {
      pParams.vSpeculative.lTypes = lTypesGguf;
    }
  }

  // when a sidecar type is requested, the draft repo resolves to its sidecar instead of a full model
  const bool cSpecSidecarFound = !vPlanSpec.vMtp.vLocalPath.empty() || !vPlanSpec.vDflash.vLocalPath.empty() ||
                                 !vPlanSpec.vEagle3.vLocalPath.empty() || !vPlanSpec.vDspark.vLocalPath.empty();
  if (!vPlanSpec.vMtp.vLocalPath.empty() && !vHadSpecUrl) {
    lTasks.emplace_back(vPlanSpec.vMtp, vOpts, [&]() {
      // only use the discovered MTP head when no draft path is set yet
      if (pParams.vSpeculative.vDraft.vMparams.vPath.empty()) {
        pParams.vSpeculative.vDraft.vMparams.vPath = hf_cache::fFinalizeFile(vPlanSpec.vMtp);
      } else {
        hf_cache::fFinalizeFile(vPlanSpec.vMtp);
      }
    });
  }
  if (!vPlanSpec.vDflash.vLocalPath.empty() && !vHadSpecUrl) {
    lTasks.emplace_back(vPlanSpec.vDflash, vOpts, [&]() {
      // only use the discovered DFlash sidecar when no draft path is set yet
      if (pParams.vSpeculative.vDraft.vMparams.vPath.empty()) {
        pParams.vSpeculative.vDraft.vMparams.vPath = hf_cache::fFinalizeFile(vPlanSpec.vDflash);
      } else {
        hf_cache::fFinalizeFile(vPlanSpec.vDflash);
      }
    });
  }
  if (!vPlanSpec.vEagle3.vLocalPath.empty() && !vHadSpecUrl) {
    lTasks.emplace_back(vPlanSpec.vEagle3, vOpts, [&]() {
      // only use the discovered Eagle3 sidecar when no draft path is set yet
      if (pParams.vSpeculative.vDraft.vMparams.vPath.empty()) {
        pParams.vSpeculative.vDraft.vMparams.vPath = hf_cache::fFinalizeFile(vPlanSpec.vEagle3);
      } else {
        hf_cache::fFinalizeFile(vPlanSpec.vEagle3);
      }
    });
  }
  if (!vPlanSpec.vDspark.vLocalPath.empty() && !vHadSpecUrl) {
    lTasks.emplace_back(vPlanSpec.vDspark, vOpts, [&]() {
      // only use the discovered DSpark sidecar when no draft path is set yet
      if (pParams.vSpeculative.vDraft.vMparams.vPath.empty()) {
        pParams.vSpeculative.vDraft.vMparams.vPath = hf_cache::fFinalizeFile(vPlanSpec.vDspark);
      } else {
        hf_cache::fFinalizeFile(vPlanSpec.vDspark);
      }
    });
  }

  // a wired draft sidecar counts as an explicit draft for the main plan fallback below
  if (cSpecSidecarFound) {
    vHadSpecUrl = true;
  }

  // handle plan_spec (e.g. --spec-draft-hf)
  if (!vPlanSpec.lModelFiles.empty() && !vHadSpecUrl && !cSpecSidecarFound) {
    vAddTasks(vPlanSpec.lModelFiles, vPlanSpec.vPrimary, pParams.vSpeculative.vDraft.vMparams);
    vHadSpecUrl = true;
  }

  if (!vPlan.lModelFiles.empty()) {
    vAddTasks(vPlan.lModelFiles, vPlan.vPrimary, pParams.vModel);
  }
  if (!vPlan.vMmproj.vLocalPath.empty()) {
    lTasks.emplace_back(vPlan.vMmproj, vOpts,
                        [&]() { pParams.vMmproj.vPath = hf_cache::fFinalizeFile(vPlan.vMmproj); });
  }
  if (!vPlan.vMtp.vLocalPath.empty() && !vHadSpecUrl) {
    lTasks.emplace_back(vPlan.vMtp, vOpts, [&]() {
      // only fall back to the discovered MTP head when no draft was explicitly provided
      if (pParams.vSpeculative.vDraft.vMparams.empty()) {
        pParams.vSpeculative.vDraft.vMparams.vPath = hf_cache::fFinalizeFile(vPlan.vMtp);
      } else {
        hf_cache::fFinalizeFile(vPlan.vMtp);
      }
    });
  }
  if (!vPlan.vDflash.vLocalPath.empty() && !vHadSpecUrl) {
    lTasks.emplace_back(vPlan.vDflash, vOpts, [&]() {
      // only fall back to the discovered DFlash sidecar when no draft was explicitly provided
      if (pParams.vSpeculative.vDraft.vMparams.empty()) {
        pParams.vSpeculative.vDraft.vMparams.vPath = hf_cache::fFinalizeFile(vPlan.vDflash);
      } else {
        hf_cache::fFinalizeFile(vPlan.vDflash);
      }
    });
  }
  if (!vPlan.vEagle3.vLocalPath.empty() && !vHadSpecUrl) {
    lTasks.emplace_back(vPlan.vEagle3, vOpts, [&]() {
      // only fall back to the discovered Eagle3 sidecar when no draft was explicitly provided
      if (pParams.vSpeculative.vDraft.vMparams.empty()) {
        pParams.vSpeculative.vDraft.vMparams.vPath = hf_cache::fFinalizeFile(vPlan.vEagle3);
      } else {
        hf_cache::fFinalizeFile(vPlan.vEagle3);
      }
    });
  }
  if (!vPlan.vDspark.vLocalPath.empty() && !vHadSpecUrl) {
    lTasks.emplace_back(vPlan.vDspark, vOpts, [&]() {
      // only fall back to the discovered DSpark sidecar when no draft was explicitly provided
      if (pParams.vSpeculative.vDraft.vMparams.empty()) {
        pParams.vSpeculative.vDraft.vMparams.vPath = hf_cache::fFinalizeFile(vPlan.vDspark);
      } else {
        hf_cache::fFinalizeFile(vPlan.vDspark);
      }
    });
  }
  if (!vPlan.vPreset.vLocalPath.empty()) {
    lTasks.emplace_back(vPlan.vPreset, vOpts, [&]() {
      // if HF repo is a preset repo, we simply run server in router mode with the preset.ini file
      pParams.vModelsPresetHf = pParams.vModel.vHfRepo;  // only for showing a warning
      pParams.vModelsPreset   = hf_cache::fFinalizeFile(vPlan.vPreset);
      // clear the model so the server starts in router mode
      pParams.vModel.vPath.clear();
      pParams.vModel.vHfRepo.clear();
      pParams.vModel.vDockerRepo.clear();
    });
  }

  // run all tasks in parallel
  if (!pParams.vOffline) {
    // if duplicated files are found, only download once (but still call on_done for each task)
    std::unordered_map<std::string, common_download_task *> dUniqueTasks;
    for (auto & vTask : lTasks) {
      auto vIt = dUniqueTasks.find(vTask.vLocalPath);
      if (vIt == dUniqueTasks.end()) {
        dUniqueTasks[vTask.vLocalPath] = &vTask;
      }
    }
    std::vector<common_download_task> lUniqueTasksVec;
    for (auto & vPair : dUniqueTasks) {
      LOG_DBG("download task: %s -> %s\n", vPair.second->vUrl.c_str(), vPair.second->vLocalPath.c_str());
      lUniqueTasksVec.push_back(*vPair.second);
    }
    fCommonDownloadRunTasks(lUniqueTasksVec);
  }

  // download successful, update params with the downloaded paths
  for (const auto & vTask : lTasks) {
    if (vTask.vOnDone) {
      vTask.vOnDone();
    }
  }
}

//
// CLI argument parsing functions
//

// apply config files (if present), a later file overrides an earlier one:
// 1. system-wide: /etc/dgs.cpp/config.ini (%PROGRAMDATA%\dgs.cpp\config.ini on windows)
// 2. user-level: ${XDG_CONFIG_HOME:-~/.config}/dgs.cpp/config.ini (%APPDATA%\dgs.cpp\config.ini on windows)
static void fCommonParamsApplySystemConfig(common_params & pParams, dgs_example pEx) {
  std::vector<std::filesystem::path> lPaths;

#if defined(_WIN32)
  const std::filesystem::path program_data = fCommonGetPathFromEnv("PROGRAMDATA");
  if (!program_data.empty()) {
    paths.push_back(program_data / "dgs.cpp" / "config.ini");
  }
#else
  lPaths.push_back("/etc/dgs.cpp/config.ini");
#endif

  try {
    lPaths.push_back(fFsGetConfigDirectory() / "config.ini");
  } catch (const std::exception & vE) {
    LOG_DBG("cannot read user-level config file, skipping: %s\n", vE.what());
  }

  std::vector<std::filesystem::path> lFound;
  for (const auto & vPath : lPaths) {
    std::error_code vEc;
    if (std::filesystem::exists(vPath, vEc)) {
      lFound.push_back(vPath);
    }
  }
  if (lFound.empty()) {
    return;
  }

  common_preset_context vCtx(pEx);
  vCtx.vIgnoreUnknownKeys = true;  // the same config file is shared by all programs
  for (const auto & vPath : lFound) {
    LOG_INF("using config file: %s\n", fFsPathToUtf8(vPath).c_str());
    common_preset  vGlobal;
    common_presets dPresets = vCtx.mLoadFromIni(vPath, vGlobal);
    vGlobal.mApplyToParams(pParams);
    auto vIt = dPresets.find(cCommonPresetDefaultName);
    if (vIt != dPresets.end()) {
      vIt->second.mApplyToParams(pParams);
    }
  }
}

static bool fCommonParamsParseEx(int pArgc, char ** pArgv, common_params_context & pCtxArg) {
  common_params & vParams = pCtxArg.vParams;

  // setup log directly from params.verbosity: see tools/cli/cli.cpp
  fCommonLogSetVerbosityThold(vParams.vVerbosity);

  // config file applies first, so env variables and CLI arguments override it
  fCommonParamsApplySystemConfig(vParams, pCtxArg.vEx);

  std::unordered_map<std::string, std::pair<common_arg *, bool>> dArgToOptions;
  for (auto & vOpt : pCtxArg.lOptions) {
    for (const auto & vArg : vOpt.lArgs) {
      dArgToOptions[vArg] = { &vOpt, /* is_positive */ true };
    }
    for (const auto & vArg : vOpt.lArgsNeg) {
      dArgToOptions[vArg] = { &vOpt, /* is_positive */ false };
    }
  }

  // handle environment variables
  for (auto & vOpt : pCtxArg.lOptions) {
    std::string value;
    if (vOpt.mGetValueFromEnv(value)) {
      try {
        if (vOpt.vHandlerVoid && fIsTruthy(value)) {
          vOpt.vHandlerVoid(vParams);
        }
        if (vOpt.vHandlerInt) {
          vOpt.vHandlerInt(vParams, std::stoi(value));
        }
        if (vOpt.vHandlerBool) {
          vOpt.vHandlerBool(vParams, fParseBoolValue(value));
        }
        if (vOpt.vHandlerString) {
          vOpt.vHandlerString(vParams, value);
          continue;
        }
      } catch (std::exception & vE) {
        throw std::invalid_argument(
          fStringFormat("error while handling environment variable \"%s\": %s\n\n", vOpt.vEnv, vE.what()));
      }
    }
  }

  // handle command line arguments
  auto vCheckArg = [&](int pI) {
    if (pI + 1 >= pArgc) {
      throw std::invalid_argument("expected value for argument");
    }
  };

  auto vParseCliArgs = [&]() {
    std::set<std::string> sSeenArgs;

    for (int vI = 1; vI < pArgc; vI++) {
      std::string vArg = pArgv[vI];
      if (dArgToOptions.find(vArg) == dArgToOptions.end()) {
        throw std::invalid_argument(fStringFormat("error: invalid argument: %s", vArg.c_str()));
      }
      if (!sSeenArgs.insert(vArg).second) {
        const bool cSkip = (vArg == "--spec-type");

        if (!cSkip) {
          throw std::invalid_argument("argument specified multiple times: " + vArg);
        }
      }
      auto & vTmp        = dArgToOptions[vArg];
      auto   vOpt        = *vTmp.first;
      bool   vIsPositive = vTmp.second;
      if (vOpt.mHasValueFromEnv()) {
        fprintf(stderr, "warn: %s environment variable is set, but will be overwritten by command line argument %s\n",
                vOpt.vEnv, vArg.c_str());
      }
      try {
        if (vOpt.vHandlerVoid) {
          vOpt.vHandlerVoid(vParams);
          continue;
        }
        if (vOpt.vHandlerBool) {
          vOpt.vHandlerBool(vParams, vIsPositive);
          continue;
        }

        // arg with single value
        vCheckArg(vI);
        std::string vVal = pArgv[++vI];
        if (vOpt.vHandlerInt) {
          vOpt.vHandlerInt(vParams, std::stoi(vVal));
          continue;
        }
        if (vOpt.vHandlerString) {
          vOpt.vHandlerString(vParams, vVal);
          continue;
        }

        // arg with 2 values
        vCheckArg(vI);
        std::string vVal2 = pArgv[++vI];
        if (vOpt.vHandlerStrStr) {
          vOpt.vHandlerStrStr(vParams, vVal, vVal2);
          continue;
        }
      } catch (std::exception & vE) {
        throw std::invalid_argument(
          fStringFormat("error while handling argument \"%s\": %s\n\n"
                        "usage:\n%s\n\nto show complete usage, run with -h",
                        vArg.c_str(), vE.what(), vOpt.to_string().c_str()));
      }
    }
  };

  // parse all CLI args now, so that -hf is available below for remote preset resolution
  vParseCliArgs();

  fPostprocessCpuParams(vParams.vCpuparams, nullptr);
  fPostprocessCpuParams(vParams.vCpuparamsBatch, &vParams.vCpuparams);

  fPostprocessCpuParams(vParams.vSpeculative.vDraft.vCpuparams, &vParams.vCpuparams);
  fPostprocessCpuParams(vParams.vSpeculative.vDraft.vCpuparamsBatch, &vParams.vCpuparamsBatch);

  // default the mmproj device to the global device selection if not set explicitly with -mmdev
  if (vParams.vMmprojUseGpu && vParams.vMmprojDevice == nullptr && !vParams.lDevices.empty()) {
    vParams.vMmprojDevice = vParams.lDevices.front();
    vParams.vMmprojUseGpu = vParams.vMmprojDevice != nullptr;
  }

  if (vParams.vPromptCacheAll && (vParams.vInteractive || vParams.vInteractiveFirst)) {
    throw std::invalid_argument("error: --prompt-cache-all not supported in interactive mode yet\n");
  }

  const bool cSkipModelDownload =
    // server will call common_params_handle_models() later, so we skip it here
    pCtxArg.vEx == cDgsExampleServer ||
    // download calls common_params_handle_models() itself and prints the paths
    pCtxArg.vEx == cDgsExampleDownload ||
    // export_graph_ops loads only metadata
    pCtxArg.vEx == cDgsExampleExportGraphOps;

  if (!cSkipModelDownload) {
    // handle model and download
    common_models_handler vHandler = fCommonModelsHandlerInit(vParams, pCtxArg.vEx);
    fCommonModelsHandlerApply(vHandler, vParams);

    // model is required (except for server)
    // TODO @ngxson : maybe show a list of available models in CLI in this case
    bool vCanSkipModel = vParams.vUsage || vParams.vCompletion || !vParams.vServerBase.empty();
    if (!vCanSkipModel && vParams.vModel.vPath.empty()) {
      throw std::invalid_argument("error: --model is required\n");
    }
  }

  if (vParams.vEscape) {
    fStringProcessEscapes(vParams.vPrompt);
    fStringProcessEscapes(vParams.vInputPrefix);
    fStringProcessEscapes(vParams.vInputSuffix);
    for (auto & vAntiprompt : vParams.lAntiprompt) {
      fStringProcessEscapes(vAntiprompt);
    }
    for (auto & vSeqBreaker : vParams.vSampling.lDrySequenceBreakers) {
      fStringProcessEscapes(vSeqBreaker);
    }
  }

  if (!vParams.lKvOverrides.empty()) {
    vParams.lKvOverrides.emplace_back();
    vParams.lKvOverrides.back().aKey[0] = 0;
  }

  const bool cMcpEnabled = !vParams.vMcpServersConfig.empty() || !vParams.vMcpServersJson.empty();
  if ((!vParams.lServerTools.empty() || cMcpEnabled || vParams.vShellMcpServer) && !vParams.vCorsOriginsExplicit) {
    LOG_WRN(
      "server tools or MCP servers are enabled, using localhost as default CORS origin (change via --cors-origins)\n");
    vParams.vCorsOrigins = "localhost";
  }

  // pad tensor_buft_overrides for dgs_params_fit:
  const size_t cNtbo = fDgsMaxTensorBuftOverrides();
  while (vParams.lTensorBuftOverrides.size() < cNtbo) {
    vParams.lTensorBuftOverrides.push_back({ nullptr, nullptr });
  }

  if (!vParams.vSpeculative.vDraft.lTensorBuftOverrides.empty()) {
    vParams.vSpeculative.vDraft.lTensorBuftOverrides.push_back({ nullptr, nullptr });
  }

  if (!vParams.vChatTemplate.empty() && !fCommonChatVerifyTemplate(vParams.vChatTemplate, vParams.vUseJinja)) {
    throw std::runtime_error(
      fStringFormat("error: the supplied chat template is not supported: %s%s\n", vParams.vChatTemplate.c_str(), ""));
  }

  // if the preserve_reasoning kwarg was not specified explicitly, enable it by default
  if (!vParams.dDefaultTemplateKwargs.count("preserve_reasoning")) {
    vParams.dDefaultTemplateKwargs["preserve_reasoning"] = "true";
  }

  return true;
}

static void fCommonParamsPrintUsage(common_params_context & pCtxArg) {
  auto vPrintOptions = [](std::vector<common_arg *> & pOptions) {
    for (common_arg * vOpt : pOptions) {
      printf("%s", vOpt->to_string().c_str());
    }
  };

  std::vector<common_arg *> lCommonOptions;
  std::vector<common_arg *> lSamplingOptions;
  std::vector<common_arg *> lSpecOptions;
  std::vector<common_arg *> lSpecificOptions;
  for (auto & vOpt : pCtxArg.lOptions) {
    // in case multiple DGS_EXAMPLE_* are set, we prioritize the DGS_EXAMPLE_* matching current example
    if (vOpt.vIsSampling) {
      lSamplingOptions.push_back(&vOpt);
    } else if (vOpt.vIsSpec) {
      lSpecOptions.push_back(&vOpt);
    } else if (vOpt.mInExample(pCtxArg.vEx)) {
      lSpecificOptions.push_back(&vOpt);
    } else {
      lCommonOptions.push_back(&vOpt);
    }
  }
  bool vFirst        = true;
  auto vPrintSection = [&](const char * pHeader, std::vector<common_arg *> & pOptions) {
    if (pOptions.empty()) {
      return;
    }
    printf("%s----- %s -----\n\n", vFirst ? "" : "\n\n", pHeader);
    vFirst = false;
    vPrintOptions(pOptions);
  };
  vPrintSection("common params", lCommonOptions);
  vPrintSection("sampling params", lSamplingOptions);
  vPrintSection("speculative params", lSpecOptions);
  vPrintSection("example-specific params", lSpecificOptions);
}

static void fCommonParamsPrintCompletion(common_params_context & pCtxArg) {
  std::vector<common_arg *> lCommonOptions;
  std::vector<common_arg *> lSamplingOptions;
  std::vector<common_arg *> lSpecOptions;
  std::vector<common_arg *> lSpecificOptions;

  for (auto & vOpt : pCtxArg.lOptions) {
    if (vOpt.vIsSampling) {
      lSamplingOptions.push_back(&vOpt);
    } else if (vOpt.vIsSpec) {
      lSpecOptions.push_back(&vOpt);
    } else if (vOpt.mInExample(pCtxArg.vEx)) {
      lSpecificOptions.push_back(&vOpt);
    } else {
      lCommonOptions.push_back(&vOpt);
    }
  }

  printf("_dgs_completions() {\n");
  printf("    local cur prev opts\n");
  printf("    COMPREPLY=()\n");
  printf("    cur=\"${COMP_WORDS[COMP_CWORD]}\"\n");
  printf("    prev=\"${COMP_WORDS[COMP_CWORD-1]}\"\n\n");

  printf("    opts=\"");
  auto vPrintOptions = [](const std::vector<common_arg *> & pOptions) {
    for (const common_arg * vOpt : pOptions) {
      for (const char * vArg : vOpt->lArgs) {
        printf("%s ", vArg);
      }
    }
  };

  vPrintOptions(lCommonOptions);
  vPrintOptions(lSamplingOptions);
  vPrintOptions(lSpecOptions);
  vPrintOptions(lSpecificOptions);
  printf("\"\n\n");

  printf("    case \"$prev\" in\n");
  printf("        --model|-m)\n");
  printf("            COMPREPLY=( $(compgen -f -X '!*.gguf' -- \"$cur\") $(compgen -d -- \"$cur\") )\n");
  printf("            return 0\n");
  printf("            ;;\n");
  printf("        --grammar-file)\n");
  printf("            COMPREPLY=( $(compgen -f -X '!*.gbnf' -- \"$cur\") $(compgen -d -- \"$cur\") )\n");
  printf("            return 0\n");
  printf("            ;;\n");
  printf("        --chat-template-file)\n");
  printf("            COMPREPLY=( $(compgen -f -X '!*.jinja' -- \"$cur\") $(compgen -d -- \"$cur\") )\n");
  printf("            return 0\n");
  printf("            ;;\n");
  printf("        *)\n");
  printf("            COMPREPLY=( $(compgen -W \"${opts}\" -- \"$cur\") )\n");
  printf("            return 0\n");
  printf("            ;;\n");
  printf("    esac\n");
  printf("}\n\n");

  const std::set<std::string> sExecutables = { "dgs-server" };

  for (const auto & vExe : sExecutables) {
    printf("complete -F _dgs_completions %s\n", vExe.c_str());
  }
}

static std::vector<ggml_backend_dev_t> fParseDeviceList(const std::string & value) {
  std::vector<ggml_backend_dev_t> lDevices;
  auto                            lDevNames = fStringSplit<std::string>(value, ',');
  if (lDevNames.empty()) {
    throw std::invalid_argument("no devices specified");
  }
  if (lDevNames.size() == 1 && lDevNames[0] == "none") {
    lDevices.push_back(nullptr);
  } else {
    ggml_backend_load_all();
    for (const auto & vDevice : lDevNames) {
      auto * vDev = ggml_backend_dev_by_name(vDevice.c_str());
      if (!vDev || ggml_backend_dev_type(vDev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        throw std::invalid_argument(fStringFormat("invalid device: %s", vDevice.c_str()));
      }
      lDevices.push_back(vDev);
    }
    lDevices.push_back(nullptr);
  }
  return lDevices;
}

void fCommonPrintAvailableDevices() {
  constexpr size_t                cMiB = 1024 * 1024;
  std::vector<ggml_backend_dev_t> lDevices;

  ggml_backend_load_all();

  for (size_t vI = 0; vI < ggml_backend_dev_count(); ++vI) {
    auto * vDev = ggml_backend_dev_get(vI);
    if (ggml_backend_dev_type(vDev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
      lDevices.push_back(vDev);
    }
  }
  printf("Available devices:\n");

  if (lDevices.empty()) {
    printf("  (none)\n");
    return;
  }
  for (auto * vDev : lDevices) {
    size_t free, vTotal;
    ggml_backend_dev_memory(vDev, &free, &vTotal);
    printf("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(vDev), ggml_backend_dev_description(vDev),
           vTotal / cMiB, free / cMiB);
  }
}

static void fAddRpcDevices(const std::string & pServers) {
  auto lRpcServers = fStringSplit<std::string>(pServers, ',');
  if (lRpcServers.empty()) {
    throw std::invalid_argument("no RPC servers specified");
  }
  ggml_backend_load_all();
  ggml_backend_reg_t vRpcReg = ggml_backend_reg_by_name("RPC");
  if (!vRpcReg) {
    throw std::invalid_argument("failed to find RPC backend");
  }
  typedef ggml_backend_reg_t (*ggml_backend_rpc_add_server_t)(const char * pEndpoint);
  ggml_backend_rpc_add_server_t vGgmlBackendRpcAddServerFn =
    (ggml_backend_rpc_add_server_t) ggml_backend_reg_get_proc_address(vRpcReg, "ggml_backend_rpc_add_server");
  if (!vGgmlBackendRpcAddServerFn) {
    throw std::invalid_argument("failed to find RPC add server function");
  }
  for (const auto & vServer : lRpcServers) {
    auto vReg = vGgmlBackendRpcAddServerFn(vServer.c_str());
    ggml_backend_register(vReg);
  }
}

bool fCommonParamsToMap(int pArgc, char ** pArgv, dgs_example pEx, std::map<common_arg, std::string> & pOutMap) {
  common_params         vDummyParams;
  common_params_context vCtxArg = fCommonParamsParserInit(vDummyParams, pEx, nullptr);

  std::unordered_map<std::string, common_arg *> dArgToOptions;
  for (auto & vOpt : vCtxArg.lOptions) {
    for (const auto & vArg : vOpt.lArgs) {
      dArgToOptions[vArg] = &vOpt;
    }
    for (const auto & vArg : vOpt.lArgsNeg) {
      dArgToOptions[vArg] = &vOpt;
    }
  }

  // TODO @ngxson : find a way to deduplicate this code

  // handle command line arguments
  auto vCheckArg = [&](int pI) {
    if (pI + 1 >= pArgc) {
      throw std::invalid_argument("expected value for argument");
    }
  };

  std::set<std::string> sSeenArgs;

  for (int vI = 1; vI < pArgc; vI++) {
    std::string vArg = pArgv[vI];
    if (dArgToOptions.find(vArg) == dArgToOptions.end()) {
      throw std::invalid_argument(fStringFormat("error: invalid argument: %s", vArg.c_str()));
    }
    if (!sSeenArgs.insert(vArg).second) {
      const bool cSkip = (vArg == "--spec-type");

      if (!cSkip) {
        throw std::invalid_argument("argument specified multiple times: " + vArg);
      }
    }
    auto        vOpt = *dArgToOptions[vArg];
    std::string vVal;
    if (vOpt.vValueHint == nullptr && vOpt.vValueHint2 == nullptr) {
      // bool arg (need to reverse the meaning for negative args)
      bool vIsNeg = std::find(vOpt.lArgsNeg.begin(), vOpt.lArgsNeg.end(), vArg) != vOpt.lArgsNeg.end();
      vVal        = vIsNeg ? "0" : "1";
    }
    if (vOpt.vValueHint != nullptr) {
      // arg with single value
      vCheckArg(vI);
      vVal = pArgv[++vI];
    }
    if (vOpt.vValueHint2 != nullptr) {
      // TODO: support arg with 2 values
      throw std::invalid_argument("error: argument with 2 values is not yet supported\n");
    }
    pOutMap[vOpt] = vVal;
  }

  return true;
}

#ifdef _WIN32
struct utf8_argv {
  std::vector<std::string> buf;
  std::vector<char *>      ptrs;
};

static utf8_argv make_utf8_argv() {
  utf8_argv out;
  int       wargc = 0;
  LPWSTR *  wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
  if (!wargv) {
    return out;
  }

  out.buf.reserve(wargc);
  for (int i = 0; i < wargc; ++i) {
    int n = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wargv[i], -1, nullptr, 0, nullptr, nullptr);
    if (n <= 0) {
      out.buf.emplace_back();
      continue;
    }
    auto & s = out.buf.emplace_back();
    s.resize(static_cast<size_t>(n - 1));
    (void) WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, s.data(), n, nullptr, nullptr);
  }
  LocalFree(wargv);

  out.ptrs.reserve(out.buf.size() + 1);
  for (auto & s : out.buf) {
    out.ptrs.push_back(s.data());
  }
  out.ptrs.push_back(nullptr);
  return out;
}
#endif

bool fCommonParamsParse(int             pArgc,
                        char **         pArgv,
                        common_params & pParams,
                        dgs_example     pEx,
                        void (*pPrintUsage)(int, char **)) {
#ifdef _WIN32
  auto utf8 = make_utf8_argv();
  // repair argv only when it matches the process command line
  if (static_cast<int>(utf8.buf.size()) == argc) {
    argv = utf8.ptrs.data();
  }
#endif

  auto                vCtxArg    = fCommonParamsParserInit(pParams, pEx, pPrintUsage);
  const common_params cParamsOrg = vCtxArg.vParams;  // the example can modify the default params

  try {
    if (!fCommonParamsParseEx(pArgc, pArgv, vCtxArg)) {
      vCtxArg.vParams = cParamsOrg;
      return false;
    }
    if (vCtxArg.vParams.vUsage) {
      fCommonParamsPrintUsage(vCtxArg);
      if (vCtxArg.vPrintUsage) {
        vCtxArg.vPrintUsage(pArgc, pArgv);
      }
      fCommonLogFlush(fCommonLogMain());
      exit(0);
    }
    if (vCtxArg.vParams.vCompletion) {
      fCommonParamsPrintCompletion(vCtxArg);
      exit(0);
    }
  } catch (const std::invalid_argument & vEx) {
    fprintf(stderr, "%s\n", vEx.what());
    vCtxArg.vParams = cParamsOrg;
    return false;
  } catch (std::exception & vEx) {
    fprintf(stderr, "%s\n", vEx.what());
    exit(1);  // for other exceptions, we exit with status code 1
  }

  return true;
}

bool common_arg_utils::fIsTruthy(const std::string & value) {
  return value == "on" || value == "enabled" || value == "true" || value == "1";
}

bool common_arg_utils::fIsFalsey(const std::string & value) {
  return value == "off" || value == "disabled" || value == "false" || value == "0";
}

bool common_arg_utils::fIsAutoy(const std::string & value) {
  return value == "auto" || value == "-1";
}

// Simple CSV parser that handles quoted fields and escaped quotes
// example:
//    input:  value1,"value, with, commas","value with ""escaped"" quotes",value4
//    output: [value1] [value, with, commas] [value with "escaped" quotes] [value4]
static std::vector<std::string> fParseCsvRow(const std::string & pInput) {
  std::vector<std::string> lFields;
  std::string              vField;
  bool                     vInQuotes = false;

  for (size_t vI = 0; vI < pInput.length(); ++vI) {
    char vCh = pInput[vI];

    if (vCh == '"') {
      if (!vInQuotes) {
        // start of quoted field (only valid if at beginning of field)
        if (!vField.empty()) {
          // quote appeared in middle of unquoted field, treat as literal
          vField += '"';
        } else {
          vInQuotes = true;  // start
        }
      } else {
        if (vI + 1 < pInput.length() && pInput[vI + 1] == '"') {
          // escaped quote: ""
          vField += '"';
          ++vI;               // skip the next quote
        } else {
          vInQuotes = false;  // end
        }
      }
    } else if (vCh == ',') {
      if (vInQuotes) {
        vField += ',';
      } else {
        lFields.push_back(std::move(vField));
        vField.clear();
      }
    } else {
      vField += vCh;
    }
  }

  // Add the last field
  lFields.push_back(std::move(vField));

  return lFields;
}

common_params_context fCommonParamsParserInit(common_params & pParams,
                                              dgs_example     pEx,
                                              void (*pPrintUsage)(int, char **)) {
  // per-example default params
  // we define here to make sure it's included in dgs-gen-docs
  if (pEx == cDgsExampleCompletion) {
  } else if (pEx == cDgsExampleMtmd) {
    pParams.vSampling.vTemp = 0.2;  // lower temp by default for better quality
  } else if (pEx == cDgsExampleServer) {
    pParams.vNParallel = -1;        // auto by default
  } else if (pEx == cDgsExampleTokenize) {
    pParams.vParseSpecial = true;   // parse special tokens by default, like the old tokenize tool
  } else if (pEx == cDgsExampleTts) {
    pParams.vOutFile                 = "output.wav";
    pParams.vSampling.vPenaltyRepeat = 1.05f;
    pParams.vSampling.vPenaltyLastN  = -1;
  }

  pParams.vUseColor = fTtyCanUseColors();

  common_params_context vCtxArg(pParams);
  vCtxArg.vPrintUsage = pPrintUsage;
  vCtxArg.vEx         = pEx;

  std::string vSamplerTypeChars;
  std::string vSamplerTypeNames;
  for (const auto & vSampler : pParams.vSampling.lSamplers) {
    vSamplerTypeChars += fCommonSamplerTypeToChr(vSampler);
    vSamplerTypeNames += fCommonSamplerTypeToStr(vSampler) + ";";
  }
  if (!vSamplerTypeNames.empty()) {
    vSamplerTypeNames.pop_back();  // remove last semicolon
  }

  /**
     * filter options by example
     * rules:
     * - all examples inherit options from DGS_EXAMPLE_COMMON
     * - if DGS_EXAMPLE_* is set (other than COMMON), we only show the option in the corresponding example
     * - if both {DGS_EXAMPLE_COMMON, DGS_EXAMPLE_*,} are set, we will prioritize the DGS_EXAMPLE_* matching current example
     */
  auto vAddOpt = [&](common_arg pArg) {
    // download only exposes the handful of args explicitly tagged for it
    const bool cInheritCommon = pEx != cDgsExampleDownload;
    if ((pArg.mInExample(pEx) || (cInheritCommon && pArg.mInExample(cDgsExampleCommon))) && !pArg.mIsExclude(pEx)) {
      vCtxArg.lOptions.push_back(std::move(pArg));
    }
  };

  vAddOpt(common_arg({ "-h", "--help" }, "print usage and exit", [](common_params & pParams) {
            pParams.vUsage = true;
          }).mSetExamples({ cDgsExampleCommon, cDgsExampleDownload }));
  vAddOpt(common_arg({ "--version" }, "show version and build info", [](common_params &) {
    fDgsPrintBuildInfo(fDgsVersion());
    exit(0);
  }));
  vAddOpt(common_arg({ "-cl", "--cache-list" }, "show list of models in cache", [](common_params &) {
    auto lModels = fCommonListCachedModels();
    printf("number of models in cache: %zu\n", lModels.size());
    for (size_t vI = 0; vI < lModels.size(); vI++) {
      printf("%4zu. %s\n", vI + 1, lModels[vI].to_string().c_str());
    }
    exit(0);
  }));
  vAddOpt(common_arg({ "--completion-bash" }, "print source-able bash completion script for dgs.cpp",
                     [](common_params & pParams) { pParams.vCompletion = true; }));

  vAddOpt(common_arg(
            { "-t", "--threads" }, "N",
            fStringFormat("number of CPU threads to use during generation (default: %d)", pParams.vCpuparams.vNThreads),
            [](common_params & pParams, int value) {
              pParams.vCpuparams.vNThreads = value;
              if (pParams.vCpuparams.vNThreads <= 0) {
                pParams.vCpuparams.vNThreads = std::thread::hardware_concurrency();
              }
            })
            .mSetEnv("DGS_ARG_THREADS"));
  vAddOpt(common_arg({ "-tb", "--threads-batch" }, "N",
                     "number of threads to use during batch and prompt processing (default: same as --threads)",
                     [](common_params & pParams, int value) {
                       pParams.vCpuparamsBatch.vNThreads = value;
                       if (pParams.vCpuparamsBatch.vNThreads <= 0) {
                         pParams.vCpuparamsBatch.vNThreads = std::thread::hardware_concurrency();
                       }
                     }));
  vAddOpt(common_arg({ "-C", "--cpu-mask" }, "M",
                     "CPU affinity mask: arbitrarily long hex. Complements cpu-range (default: \"\")",
                     [](common_params & pParams, const std::string & pMask) {
                       pParams.vCpuparams.vMaskValid = true;
                       if (!fParseCpuMask(pMask, pParams.vCpuparams.aCpumask)) {
                         throw std::invalid_argument("invalid cpumask");
                       }
                     }));
  vAddOpt(common_arg({ "-Cr", "--cpu-range" }, "lo-hi", "range of CPUs for affinity. Complements --cpu-mask",
                     [](common_params & pParams, const std::string & pRange) {
                       pParams.vCpuparams.vMaskValid = true;
                       if (!fParseCpuRange(pRange, pParams.vCpuparams.aCpumask)) {
                         throw std::invalid_argument("invalid range");
                       }
                     }));
  vAddOpt(common_arg(
    { "--cpu-strict" }, "<0|1>",
    fStringFormat("use strict CPU placement (default: %u)\n", (unsigned) pParams.vCpuparams.vStrictCpu),
    [](common_params & pParams, const std::string & value) { pParams.vCpuparams.vStrictCpu = std::stoul(value); }));
  vAddOpt(common_arg(
    { "--prio" }, "N",
    fStringFormat("set process/thread priority : low(-1), normal(0), medium(1), high(2), realtime(3) (default: %d)\n",
                  pParams.vCpuparams.vPriority),
    [](common_params & pParams, int pPrio) {
      if (pPrio < GGML_SCHED_PRIO_LOW || pPrio > GGML_SCHED_PRIO_REALTIME) {
        throw std::invalid_argument("invalid value");
      }
      pParams.vCpuparams.vPriority = (enum ggml_sched_priority) pPrio;
    }));
  vAddOpt(common_arg(
    { "--poll" }, "<0...100>",
    fStringFormat("use polling level to wait for work (0 - no polling, default: %u)\n",
                  (unsigned) pParams.vCpuparams.vPoll),
    [](common_params & pParams, const std::string & value) { pParams.vCpuparams.vPoll = std::stoul(value); }));
  vAddOpt(
    common_arg({ "-Cb", "--cpu-mask-batch" }, "M",
               "CPU affinity mask: arbitrarily long hex. Complements cpu-range-batch (default: same as --cpu-mask)",
               [](common_params & pParams, const std::string & pMask) {
                 pParams.vCpuparamsBatch.vMaskValid = true;
                 if (!fParseCpuMask(pMask, pParams.vCpuparamsBatch.aCpumask)) {
                   throw std::invalid_argument("invalid cpumask");
                 }
               }));
  vAddOpt(common_arg({ "-Crb", "--cpu-range-batch" }, "lo-hi",
                     "ranges of CPUs for affinity. Complements --cpu-mask-batch",
                     [](common_params & pParams, const std::string & pRange) {
                       pParams.vCpuparamsBatch.vMaskValid = true;
                       if (!fParseCpuRange(pRange, pParams.vCpuparamsBatch.aCpumask)) {
                         throw std::invalid_argument("invalid range");
                       }
                     }));
  vAddOpt(common_arg({ "--cpu-strict-batch" }, "<0|1>", "use strict CPU placement (default: same as --cpu-strict)",
                     [](common_params & pParams, int value) { pParams.vCpuparamsBatch.vStrictCpu = value; }));
  vAddOpt(
    common_arg({ "--prio-batch" }, "N",
               fStringFormat("set process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n",
                             pParams.vCpuparamsBatch.vPriority),
               [](common_params & pParams, int pPrio) {
                 if (pPrio < 0 || pPrio > 3) {
                   throw std::invalid_argument("invalid value");
                 }
                 pParams.vCpuparamsBatch.vPriority = (enum ggml_sched_priority) pPrio;
               }));
  vAddOpt(common_arg({ "--poll-batch" }, "<0|1>", "use polling to wait for work (default: same as --poll)",
                     [](common_params & pParams, int value) { pParams.vCpuparamsBatch.vPoll = value; }));
  vAddOpt(common_arg({ "-lcs", "--lookup-cache-static" }, "FNAME",
                     "path to static lookup cache to use for lookup decoding (not updated by generation)",
                     [](common_params & pParams, const std::string & value) {
                       pParams.vSpeculative.vNgramCache.vLookupCacheStatic = value;
                     })
            .mSetExamples({ cDgsExampleLookup, cDgsExampleServer }));
  vAddOpt(common_arg({ "-lcd", "--lookup-cache-dynamic" }, "FNAME",
                     "path to dynamic lookup cache to use for lookup decoding (updated by generation)",
                     [](common_params & pParams, const std::string & value) {
                       pParams.vSpeculative.vNgramCache.vLookupCacheDynamic = value;
                     })
            .mSetExamples({ cDgsExampleLookup, cDgsExampleServer }));
  vAddOpt(common_arg({ "-c", "--ctx-size" }, "N",
                     fStringFormat("size of the prompt context (default: %d, 0 = loaded from model)", pParams.vNCtx),
                     [](common_params & pParams, int value) {
                       pParams.vNCtx = value;
                       if (value == 0) {
                         // disable context reduction in dgs_params_fit if the user explicitly requests the full context size:
                         pParams.vFitParamsMinCtx = UINT32_MAX;
                       }
                     })
            .mSetEnv("DGS_ARG_CTX_SIZE"));
  vAddOpt(common_arg({ "--kv-unified-per-slot" }, "N",
                     "context limit per parallel slot (default: unset, behavior unchanged).\n"
                     "when set without -c/--ctx-size, the shared KV pool is sized to n_parallel*N",
                     [](common_params & pParams, int value) { pParams.vKvUnifiedPerSlot = value; })
            .mSetEnv("DGS_ARG_KV_UNIFIED_PER_SLOT")
            .mSetExamples({ cDgsExampleServer }));
  vAddOpt(
    common_arg({ "-n", "--predict" }, "N",
               fStringFormat(pEx == cDgsExampleCompletion ?
                               "number of tokens to predict (default: %d, -1 = infinity, -2 = until context filled)" :
                               "number of tokens to predict (default: %d, -1 = infinity)",
                             pParams.vNPredict),
               [](common_params & pParams, int value) { pParams.vNPredict = value; })
      .mSetEnv("DGS_ARG_N_PREDICT"));
  vAddOpt(common_arg({ "-b", "--batch-size" }, "N",
                     fStringFormat("logical maximum batch size (default: %d)", pParams.vNBatch),
                     [](common_params & pParams, int value) { pParams.vNBatch = value; })
            .mSetEnv("DGS_ARG_BATCH"));
  vAddOpt(common_arg({ "-ub", "--ubatch-size" }, "N",
                     fStringFormat("physical maximum batch size (default: %d)", pParams.vNUbatch),
                     [](common_params & pParams, int value) { pParams.vNUbatch = value; })
            .mSetEnv("DGS_ARG_UBATCH"));
  vAddOpt(common_arg(
    { "--keep" }, "N",
    fStringFormat("number of tokens to keep from the initial prompt (default: %d, -1 = all)", pParams.vNKeep),
    [](common_params & pParams, int value) { pParams.vNKeep = value; }));
  vAddOpt(
    common_arg({ "--swa-full" },
               fStringFormat("use full-size SWA cache (default: %s)\n"
                             "[(more info)](doc/MANUAL.md)",
                             pParams.vSwaFull ? "true" : "false"),
               [](common_params & pParams) { pParams.vSwaFull = true; })
      .mSetEnv("DGS_ARG_SWA_FULL"));
  vAddOpt(common_arg({ "-ctxcp", "--ctx-checkpoints" }, "N",
                     fStringFormat("max number of context checkpoints to create per slot (default: %d)"
                                   "[(more info)](doc/MANUAL.md)",
                                   pParams.vNCtxCheckpoints),
                     [](common_params & pParams, int value) { pParams.vNCtxCheckpoints = value; })
            .mSetEnv("DGS_ARG_CTX_CHECKPOINTS")
            .mSetExamples({ cDgsExampleServer, cDgsExampleCli }));
  vAddOpt(
    common_arg({ "-cms", "--checkpoint-min-step" }, "N",
               fStringFormat("minimum spacing between context checkpoints in tokens (default: %d, 0 = no minimum)",
                             pParams.vCheckpointMinStep),
               [](common_params & pParams, int value) {
                 if (value < 0) {
                   throw std::invalid_argument("checkpoint-min-step must be non-negative");
                 }
                 pParams.vCheckpointMinStep = value;
               })
      .mSetEnv("DGS_ARG_CHECKPOINT_MIN_SPACING_NT")
      .mSetExamples({ cDgsExampleServer }));
  vAddOpt(common_arg({ "-cram", "--cache-ram" }, "N",
                     fStringFormat("set the maximum cache size in MiB (default: %d, -1 - no limit, 0 - disable)"
                                   "[(more info)](doc/MANUAL.md)",
                                   pParams.vCacheRamMib),
                     [](common_params & pParams, int value) { pParams.vCacheRamMib = value; })
            .mSetEnv("DGS_ARG_CACHE_RAM")
            .mSetExamples({ cDgsExampleServer, cDgsExampleCli }));
  vAddOpt(
    common_arg({ "-kvu", "--kv-unified" }, { "-no-kvu", "--no-kv-unified" },
               "use single unified KV buffer shared across all sequences (default: enabled if number of slots is auto)",
               [](common_params & pParams, bool value) { pParams.vKvUnified = value; })
      .mSetEnv("DGS_ARG_KV_UNIFIED")
      .mSetExamples({ cDgsExampleServer, cDgsExamplePerplexity, cDgsExampleBatched, cDgsExampleBench,
                      cDgsExampleParallel, cDgsExampleSpeculative }));
  vAddOpt(common_arg({ "--cache-idle-slots" }, { "--no-cache-idle-slots" },
                     "save idle slots to the prompt cache on new task, and clear them when using unified KV (default: "
                     "enabled, requires cache-ram)",
                     [](common_params & pParams, bool value) { pParams.vCacheIdleSlots = value; })
            .mSetEnv("DGS_ARG_CACHE_IDLE_SLOTS")
            .mSetExamples({ cDgsExampleServer }));
  vAddOpt(common_arg({ "--context-shift" }, { "--no-context-shift" },
                     fStringFormat("whether to use context shift on infinite text generation (default: %s)",
                                   pParams.vCtxShift ? "enabled" : "disabled"),
                     [](common_params & pParams, bool value) { pParams.vCtxShift = value; })
            .mSetExamples(
              { cDgsExampleCompletion, cDgsExampleCli, cDgsExampleServer, cDgsExampleImatrix, cDgsExamplePerplexity })
            .mSetEnv("DGS_ARG_CONTEXT_SHIFT"));

  vAddOpt(common_arg({ "-fa", "--flash-attn" }, "[on|off|auto]",
                     fStringFormat("set Flash Attention use ('on', 'off', or 'auto', default: '%s')",
                                   fDgsFlashAttnTypeName(pParams.vFlashAttnType)),
                     [](common_params & pParams, const std::string & value) {
                       if (fIsTruthy(value)) {
                         pParams.vFlashAttnType = cDgsFlashAttnTypeEnabled;
                       } else if (fIsFalsey(value)) {
                         pParams.vFlashAttnType = cDgsFlashAttnTypeDisabled;
                       } else if (fIsAutoy(value)) {
                         pParams.vFlashAttnType = cDgsFlashAttnTypeAuto;
                       } else {
                         throw std::runtime_error(
                           fStringFormat("error: unknown value for --flash-attn: '%s'\n", value.c_str()));
                       }
                     })
            .mSetEnv("DGS_ARG_FLASH_ATTN"));
  vAddOpt(common_arg({ "-p", "--prompt" }, "PROMPT", "prompt to start generation with; for system message, use -sys",
                     [](common_params & pParams, const std::string & value) { pParams.vPrompt = value; })
            .mSetExcludes({ cDgsExampleServer }));

  vAddOpt(common_arg({ "--perf" }, { "--no-perf" },
                     fStringFormat("whether to enable internal libdgs performance timings (default: %s)",
                                   pParams.vNoPerf ? "true" : "false"),
                     [](common_params & pParams, bool value) {
                       pParams.vNoPerf           = !value;
                       pParams.vSampling.vNoPerf = !value;
                     })
            .mSetEnv("DGS_ARG_PERF"));

  vAddOpt(common_arg({ "-f", "--file" }, "FNAME", "a file containing the prompt (default: none)",
                     [](common_params & pParams, const std::string & value) {
                       pParams.vPrompt     = fReadFile(value);
                       // store the external file name in params
                       pParams.vPromptFile = value;
                       if (!pParams.vPrompt.empty() && pParams.vPrompt.back() == '\n') {
                         pParams.vPrompt.pop_back();
                       }
                     })
            .mSetExcludes({ cDgsExampleServer }));

  vAddOpt(common_arg({ "-bf", "--binary-file" }, "FNAME", "binary file containing the prompt (default: none)",
                     [](common_params & pParams, const std::string & value) {
                       std::ifstream vFile(value, std::ios::binary);
                       if (!vFile) {
                         throw std::runtime_error(fStringFormat("error: failed to open file '%s'\n", value.c_str()));
                       }
                       // store the external file name in params
                       pParams.vPromptFile = value;
                       std::ostringstream vSs;
                       vSs << vFile.rdbuf();
                       pParams.vPrompt = vSs.str();
                       fprintf(stderr, "Read %zu bytes from binary file %s\n", pParams.vPrompt.size(), value.c_str());
                     })
            .mSetExcludes({ cDgsExampleServer }));
  vAddOpt(
    common_arg({ "-e", "--escape" }, { "--no-escape" },
               fStringFormat("whether to process escapes sequences (\\n, \\r, \\t, \\', \\\", \\\\) (default: %s)",
                             pParams.vEscape ? "true" : "false"),
               [](common_params & pParams, bool value) { pParams.vEscape = value; }));

  vAddOpt(common_arg(
            { "-r", "--reverse-prompt" }, "PROMPT", "halt generation at PROMPT, return control in interactive mode\n",
            [](common_params & pParams, const std::string & value) { pParams.lAntiprompt.emplace_back(value); })
            .mSetExamples({ cDgsExampleCompletion, cDgsExampleCli, cDgsExampleServer }));
  vAddOpt(common_arg({ "-sp", "--special" },
                     fStringFormat("special tokens output enabled (default: %s)", pParams.vSpecial ? "true" : "false"),
                     [](common_params & pParams) { pParams.vSpecial = true; })
            .mSetExamples({ cDgsExampleCompletion, cDgsExampleCli, cDgsExampleServer }));

  vAddOpt(common_arg({ "--warmup" }, { "--no-warmup" },
                     fStringFormat("whether to perform warmup with an empty run (default: %s)",
                                   pParams.vWarmup ? "enabled" : "disabled"),
                     [](common_params & pParams, bool value) { pParams.vWarmup = value; })
            .mSetExamples({ cDgsExampleCompletion, cDgsExampleCli, cDgsExampleServer, cDgsExampleMtmd,
                            cDgsExampleEmbedding, cDgsExampleRetrieval, cDgsExamplePerplexity, cDgsExampleDebug }));
  vAddOpt(common_arg({ "--spm-infill" },
                     fStringFormat("use Suffix/Prefix/Middle pattern for infill (instead of Prefix/Suffix/Middle) as "
                                   "some models prefer this. (default: %s)",
                                   pParams.vSpmInfill ? "enabled" : "disabled"),
                     [](common_params & pParams) { pParams.vSpmInfill = true; })
            .mSetExamples({ cDgsExampleServer }));
  vAddOpt(common_arg(
            { "--samplers" }, "SAMPLERS",
            fStringFormat("samplers that will be used for generation in the order, separated by \';\'\n(default: %s)",
                          vSamplerTypeNames.c_str()),
            [](common_params & pParams, const std::string & value) {
              const auto lSamplerNames    = fStringSplit<std::string>(value, ';');
              pParams.vSampling.lSamplers = fCommonSamplerTypesFromNames(lSamplerNames);
              pParams.vSampling.vUserSamplingConfig |=
                common_params_sampling_config::cCommonParamsSamplingConfigSamplers;
            })
            .mSetSampling());
  vAddOpt(common_arg(
            { "-s", "--seed" }, "SEED",
            fStringFormat("RNG seed (default: %d, use random seed for %d)", pParams.vSampling.vSeed, DGS_DEFAULT_SEED),
            [](common_params & pParams, const std::string & value) { pParams.vSampling.vSeed = std::stoul(value); })
            .mSetSampling());
  vAddOpt(common_arg({ "--sampler-seq" }, "SEQUENCE",
                     fStringFormat("simplified sequence for samplers that will be used (default: %s)",
                                   vSamplerTypeChars.c_str()),
                     [](common_params & pParams, const std::string & value) {
                       pParams.vSampling.lSamplers = fCommonSamplerTypesFromChars(value);
                     })
            .mSetSampling());
  vAddOpt(common_arg({ "--ignore-eos" },
                     "ignore end of stream token and continue generating (implies --logit-bias EOS-inf)",
                     [](common_params & pParams) { pParams.vSampling.vIgnoreEos = true; })
            .mSetSampling());
  vAddOpt(common_arg({ "--temp" }, "N", fStringFormat("temperature (default: %.2f)", (double) pParams.vSampling.vTemp),
                     [](common_params & pParams, const std::string & value) {
                       pParams.vSampling.vTemp = std::stof(value);
                       pParams.vSampling.vTemp = std::max(pParams.vSampling.vTemp, 0.0f);
                       pParams.vSampling.vUserSamplingConfig |=
                         common_params_sampling_config::cCommonParamsSamplingConfigTemp;
                     })
            .mSetSampling()
            .mSetEnv("DGS_ARG_TEMPERATURE"));
  vAddOpt(common_arg(
            { "--top-k" }, "N", fStringFormat("top-k sampling (default: %d, 0 = disabled)", pParams.vSampling.vTopK),
            [](common_params & pParams, int value) {
              pParams.vSampling.vTopK = value;
              pParams.vSampling.vUserSamplingConfig |= common_params_sampling_config::cCommonParamsSamplingConfigTopK;
            })
            .mSetSampling()
            .mSetEnv("DGS_ARG_TOP_K"));
  vAddOpt(common_arg({ "--top-p" }, "N",
                     fStringFormat("top-p sampling (default: %.2f, 1.0 = disabled)", (double) pParams.vSampling.vTopP),
                     [](common_params & pParams, const std::string & value) {
                       pParams.vSampling.vTopP = std::stof(value);
                       pParams.vSampling.vUserSamplingConfig |=
                         common_params_sampling_config::cCommonParamsSamplingConfigTopP;
                     })
            .mSetSampling()
            .mSetEnv("DGS_ARG_TOP_P"));
  vAddOpt(common_arg({ "--min-p" }, "N",
                     fStringFormat("min-p sampling (default: %.2f, 0.0 = disabled)", (double) pParams.vSampling.vMinP),
                     [](common_params & pParams, const std::string & value) {
                       pParams.vSampling.vMinP = std::stof(value);
                       pParams.vSampling.vUserSamplingConfig |=
                         common_params_sampling_config::cCommonParamsSamplingConfigMinP;
                     })
            .mSetSampling()
            .mSetEnv("DGS_ARG_MIN_P"));
  vAddOpt(common_arg(
            { "--top-nsigma" }, "N",
            fStringFormat("top-n-sigma sampling (default: %.2f, -1.0 = disabled)", pParams.vSampling.vTopNSigma),
            [](common_params & pParams, const std::string & value) { pParams.vSampling.vTopNSigma = std::stof(value); })
            .mSetSampling());
  vAddOpt(common_arg({ "--xtc-probability" }, "N",
                     fStringFormat("xtc probability (default: %.2f, 0.0 = disabled)",
                                   (double) pParams.vSampling.vXtcProbability),
                     [](common_params & pParams, const std::string & value) {
                       pParams.vSampling.vXtcProbability = std::stof(value);
                       pParams.vSampling.vUserSamplingConfig |=
                         common_params_sampling_config::cCommonParamsSamplingConfigXtcProbability;
                     })
            .mSetSampling());
  vAddOpt(
    common_arg({ "--xtc-threshold" }, "N",
               fStringFormat("xtc threshold (default: %.2f, 1.0 = disabled)", (double) pParams.vSampling.vXtcThreshold),
               [](common_params & pParams, const std::string & value) {
                 pParams.vSampling.vXtcThreshold = std::stof(value);
                 pParams.vSampling.vUserSamplingConfig |=
                   common_params_sampling_config::cCommonParamsSamplingConfigXtcThreshold;
               })
      .mSetSampling());
  vAddOpt(
    common_arg({ "--typical" }, "N",
               fStringFormat("locally typical sampling, parameter p (default: %.2f, 1.0 = disabled)",
                             (double) pParams.vSampling.vTypP),
               [](common_params & pParams, const std::string & value) { pParams.vSampling.vTypP = std::stof(value); })
      .mSetSampling());
  vAddOpt(common_arg({ "--repeat-last-n" }, "N",
                     fStringFormat("last n tokens to consider for penalize (default: %d, 0 = disabled)",
                                   pParams.vSampling.vPenaltyLastN),
                     [](common_params & pParams, int value) {
                       if (value < 0) {
                         throw std::runtime_error(fStringFormat("error: invalid repeat-last-n = %d\n", value));
                       }
                       pParams.vSampling.vPenaltyLastN = value;
                       pParams.vSampling.vNPrev = std::max(pParams.vSampling.vNPrev, pParams.vSampling.vPenaltyLastN);
                       pParams.vSampling.vUserSamplingConfig |=
                         common_params_sampling_config::cCommonParamsSamplingConfigPenaltyLastN;
                     })
            .mSetSampling());
  vAddOpt(common_arg({ "--repeat-penalty" }, "N",
                     fStringFormat("penalize repeat sequence of tokens (default: %.2f, 1.0 = disabled)",
                                   (double) pParams.vSampling.vPenaltyRepeat),
                     [](common_params & pParams, const std::string & value) {
                       const float cPenaltyRepeat = std::stof(value);
                       if (!std::isfinite(cPenaltyRepeat) || cPenaltyRepeat <= 0.0f ||
                           !std::isfinite(1.0f / cPenaltyRepeat)) {
                         throw std::runtime_error("error: repeat-penalty must be finite and greater than 0\n");
                       }
                       pParams.vSampling.vPenaltyRepeat = cPenaltyRepeat;
                       pParams.vSampling.vUserSamplingConfig |=
                         common_params_sampling_config::cCommonParamsSamplingConfigPenaltyRepeat;
                     })
            .mSetSampling()
            .mSetEnv("DGS_ARG_REPEAT_PENALTY"));
  vAddOpt(common_arg({ "--presence-penalty" }, "N",
                     fStringFormat("repeat alpha presence penalty (default: %.2f, 0.0 = disabled)",
                                   (double) pParams.vSampling.vPenaltyPresent),
                     [](common_params & pParams, const std::string & value) {
                       const float cPenaltyPresent = std::stof(value);
                       if (!std::isfinite(cPenaltyPresent)) {
                         throw std::runtime_error("error: presence-penalty must be finite\n");
                       }
                       pParams.vSampling.vPenaltyPresent = cPenaltyPresent;
                     })
            .mSetSampling()
            .mSetEnv("DGS_ARG_PRESENCE_PENALTY"));
  vAddOpt(common_arg({ "--frequency-penalty" }, "N",
                     fStringFormat("repeat alpha frequency penalty (default: %.2f, 0.0 = disabled)",
                                   (double) pParams.vSampling.vPenaltyFreq),
                     [](common_params & pParams, const std::string & value) {
                       const float cPenaltyFreq = std::stof(value);
                       if (!std::isfinite(cPenaltyFreq)) {
                         throw std::runtime_error("error: frequency-penalty must be finite\n");
                       }
                       pParams.vSampling.vPenaltyFreq = cPenaltyFreq;
                     })
            .mSetSampling()
            .mSetEnv("DGS_ARG_FREQUENCY_PENALTY"));
  vAddOpt(common_arg({ "--dry-multiplier" }, "N",
                     fStringFormat("set DRY sampling multiplier (default: %.2f, 0.0 = disabled)",
                                   (double) pParams.vSampling.vDryMultiplier),
                     [](common_params & pParams, const std::string & value) {
                       pParams.vSampling.vDryMultiplier = std::stof(value);
                     })
            .mSetSampling());
  vAddOpt(common_arg({ "--dry-base" }, "N",
                     fStringFormat("set DRY sampling base value (default: %.2f)", (double) pParams.vSampling.vDryBase),
                     [](common_params & pParams, const std::string & value) {
                       float vPotentialBase = std::stof(value);
                       if (vPotentialBase >= 1.0f) {
                         pParams.vSampling.vDryBase = vPotentialBase;
                       }
                     })
            .mSetSampling());
  vAddOpt(
    common_arg({ "--dry-allowed-length" }, "N",
               fStringFormat("set allowed length for DRY sampling (default: %d)", pParams.vSampling.vDryAllowedLength),
               [](common_params & pParams, int value) { pParams.vSampling.vDryAllowedLength = value; })
      .mSetSampling());
  vAddOpt(common_arg({ "--dry-penalty-last-n" }, "N",
                     fStringFormat("set DRY penalty for the last n tokens (default: %d, 0 = disable)",
                                   pParams.vSampling.vDryPenaltyLastN),
                     [](common_params & pParams, int value) {
                       if (value < 0) {
                         throw std::runtime_error(fStringFormat("error: invalid dry-penalty-last-n = %d\n", value));
                       }
                       pParams.vSampling.vDryPenaltyLastN = value;
                     })
            .mSetSampling());
  vAddOpt(common_arg({ "--dry-sequence-breaker" }, "STRING",
                     fStringFormat("add sequence breaker for DRY sampling, clearing out default breakers (%s) in the "
                                   "process; use \"none\" to not use any sequence breakers\n",
                                   pParams.vSampling.lDrySequenceBreakers.empty() ?
                                     "none" :
                                     std::accumulate(std::next(pParams.vSampling.lDrySequenceBreakers.begin()),
                                                     pParams.vSampling.lDrySequenceBreakers.end(),
                                                     std::string("'") +
                                                       (pParams.vSampling.lDrySequenceBreakers[0] == "\n" ?
                                                          "\\n" :
                                                          pParams.vSampling.lDrySequenceBreakers[0]) +
                                                       "'",
                                                     [](const std::string & pA, const std::string & pB) {
                                                       std::string vFormattedB = (pB == "\n") ? "\\n" : pB;
                                                       return pA + ", '" + vFormattedB + "'";
                                                     })
                                       .c_str()),
                     [](common_params & pParams, const std::string & value) {
                       static bool vDefaultsCleared = false;

                       if (!vDefaultsCleared) {
                         pParams.vSampling.lDrySequenceBreakers.clear();
                         vDefaultsCleared = true;
                       }

                       if (value == "none") {
                         pParams.vSampling.lDrySequenceBreakers.clear();
                       } else {
                         pParams.vSampling.lDrySequenceBreakers.emplace_back(value);
                       }
                     })
            .mSetSampling());
  vAddOpt(common_arg({ "--adaptive-target" }, "N",
                     fStringFormat("adaptive-p: select tokens near this probability (valid range 0.0 "
                                   "to 1.0; negative = disabled) (default: %.2f)\n"
                                   "[(more info)](doc/MANUAL.md)",
                                   (double) pParams.vSampling.vAdaptiveTarget),
                     [](common_params & pParams, const std::string & value) {
                       pParams.vSampling.vAdaptiveTarget = std::stof(value);
                     })
            .mSetSampling());
  vAddOpt(common_arg({ "--adaptive-decay" }, "N",
                     fStringFormat("adaptive-p: decay rate for target adaptation over time. lower values "
                                   "are more reactive, higher values are more stable.\n"
                                   "(valid range 0.0 to 0.99) (default: %.2f)",
                                   (double) pParams.vSampling.vAdaptiveDecay),
                     [](common_params & pParams, const std::string & value) {
                       pParams.vSampling.vAdaptiveDecay = std::stof(value);
                     })
            .mSetSampling());
  vAddOpt(common_arg({ "--dynatemp-range" }, "N",
                     fStringFormat("dynamic temperature range (default: %.2f, 0.0 = disabled)",
                                   (double) pParams.vSampling.vDynatempRange),
                     [](common_params & pParams, const std::string & value) {
                       pParams.vSampling.vDynatempRange = std::stof(value);
                     })
            .mSetSampling());
  vAddOpt(common_arg(
            { "--dynatemp-exp" }, "N",
            fStringFormat("dynamic temperature exponent (default: %.2f)", (double) pParams.vSampling.vDynatempExponent),
            [](common_params & pParams, const std::string & value) {
              pParams.vSampling.vDynatempExponent = std::stof(value);
            })
            .mSetSampling());
  vAddOpt(common_arg(
            { "--mirostat" }, "N",
            fStringFormat("use Mirostat sampling.\nTop K, Nucleus and Locally Typical samplers are ignored if used.\n"
                          "(default: %d, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)",
                          pParams.vSampling.vMirostat),
            [](common_params & pParams, int value) {
              pParams.vSampling.vMirostat = value;
              pParams.vSampling.vUserSamplingConfig |=
                common_params_sampling_config::cCommonParamsSamplingConfigMirostat;
            })
            .mSetSampling());
  vAddOpt(common_arg({ "--mirostat-lr" }, "N",
                     fStringFormat("Mirostat learning rate, parameter eta (default: %.2f)",
                                   (double) pParams.vSampling.vMirostatEta),
                     [](common_params & pParams, const std::string & value) {
                       pParams.vSampling.vMirostatEta = std::stof(value);
                       pParams.vSampling.vUserSamplingConfig |=
                         common_params_sampling_config::cCommonParamsSamplingConfigMirostatEta;
                     })
            .mSetSampling());
  vAddOpt(common_arg({ "--mirostat-ent" }, "N",
                     fStringFormat("Mirostat target entropy, parameter tau (default: %.2f)",
                                   (double) pParams.vSampling.vMirostatTau),
                     [](common_params & pParams, const std::string & value) {
                       pParams.vSampling.vMirostatTau = std::stof(value);
                       pParams.vSampling.vUserSamplingConfig |=
                         common_params_sampling_config::cCommonParamsSamplingConfigMirostatTau;
                     })
            .mSetSampling());
  vAddOpt(common_arg({ "-l", "--logit-bias" }, "TOKEN_ID(+/-)BIAS",
                     "modifies the likelihood of token appearing in the completion,\n"
                     "i.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',\n"
                     "or `--logit-bias 15043-1` to decrease likelihood of token ' Hello'",
                     [](common_params & pParams, const std::string & value) {
                       std::stringstream vSs(value);
                       dgs_token         vKey;
                       char              vSign;
                       std::string       vValueStr;
                       try {
                         if (vSs >> vKey && vSs >> vSign && std::getline(vSs, vValueStr) &&
                             (vSign == '+' || vSign == '-')) {
                           const float cBias = std::stof(vValueStr) * ((vSign == '-') ? -1.0f : 1.0f);
                           pParams.vSampling.lLogitBias.push_back({ vKey, cBias });
                         } else {
                           throw std::invalid_argument("invalid input format");
                         }
                       } catch (const std::exception &) {
                         throw std::invalid_argument("invalid input format");
                       }
                     })
            .mSetSampling());
  vAddOpt(common_arg({ "--grammar" }, "GRAMMAR",
                     "BNF-like grammar to constrain generations (see samples in grammars/ dir)",
                     [](common_params & pParams, const std::string & value) {
                       pParams.vSampling.vGrammar = { cCommonGrammarTypeUser, value };
                     })
            .mSetSampling());
  vAddOpt(common_arg({ "--grammar-file" }, "FNAME", "file to read grammar from",
                     [](common_params & pParams, const std::string & value) {
                       pParams.vSampling.vGrammar = { cCommonGrammarTypeUser, fReadFile(value) };
                     })
            .mSetSampling());
  vAddOpt(common_arg({ "-j", "--json-schema" }, "SCHEMA",
                     "JSON schema to constrain generations (https://json-schema.org/), e.g. `{\"type\": \"object\"}` "
                     "for any JSON object",
                     [](common_params & pParams, const std::string & value) {
                       pParams.vSampling.vGrammar = { cCommonGrammarTypeOutputFormat,
                                                      fJsonSchemaToGrammar(json::mParse(value)) };
                     })
            .mSetSampling());
  vAddOpt(common_arg({ "-jf", "--json-schema-file" }, "FILE",
                     "File containing a JSON schema to constrain generations (https://json-schema.org/), e.g. "
                     "`{\"type\": \"object\"}` for any JSON object",
                     [](common_params & pParams, const std::string & value) {
                       std::ifstream vFile(value);
                       if (!vFile) {
                         throw std::runtime_error(fStringFormat("error: failed to open file '%s'\n", value.c_str()));
                       }
                       std::string vSchema;
                       std::copy(std::istreambuf_iterator<char>(vFile), std::istreambuf_iterator<char>(),
                                 std::back_inserter(vSchema));
                       pParams.vSampling.vGrammar = { cCommonGrammarTypeOutputFormat,
                                                      fJsonSchemaToGrammar(json::mParse(vSchema)) };
                     })
            .mSetSampling());
  vAddOpt(common_arg({ "-bs", "--backend-sampling" }, "enable backend sampling (experimental) (default: disabled)",
                     [](common_params & pParams) { pParams.vSampling.vBackendSampling = true; })
            .mSetSampling()
            .mSetEnv("DGS_ARG_BACKEND_SAMPLING"));
  vAddOpt(common_arg({ "--pooling" }, "{none,mean,cls,last,rank}",
                     "pooling type for embeddings, use model default if unspecified",
                     [](common_params & pParams, const std::string & value) {
                       /**/ if (value == "none") {
                         pParams.vPoolingType = cDgsPoolingTypeNone;
                       } else if (value == "mean") {
                         pParams.vPoolingType = cDgsPoolingTypeMean;
                       } else if (value == "cls") {
                         pParams.vPoolingType = cDgsPoolingTypeCls;
                       } else if (value == "last") {
                         pParams.vPoolingType = cDgsPoolingTypeLast;
                       } else if (value == "rank") {
                         pParams.vPoolingType = cDgsPoolingTypeRank;
                       } else {
                         throw std::invalid_argument("invalid value");
                       }
                     })
            .mSetExamples({ cDgsExampleEmbedding, cDgsExampleRetrieval, cDgsExampleServer, cDgsExampleDebug })
            .mSetEnv("DGS_ARG_POOLING"));

  vAddOpt(common_arg({ "--rope-scaling" }, "{none,linear,yarn}",
                     "RoPE frequency scaling method, defaults to linear unless specified by the model",
                     [](common_params & pParams, const std::string & value) {
                       /**/ if (value == "none") {
                         pParams.vRopeScalingType = cDgsRopeScalingTypeNone;
                       } else if (value == "linear") {
                         pParams.vRopeScalingType = cDgsRopeScalingTypeLinear;
                       } else if (value == "yarn") {
                         pParams.vRopeScalingType = cDgsRopeScalingTypeYarn;
                       } else {
                         throw std::invalid_argument("invalid value");
                       }
                     })
            .mSetEnv("DGS_ARG_ROPE_SCALING_TYPE"));
  vAddOpt(common_arg({ "--rope-scale" }, "N", "RoPE context scaling factor, expands context by a factor of N",
                     [](common_params & pParams, const std::string & value) {
                       pParams.vRopeFreqScale = 1.0f / std::stof(value);
                     })
            .mSetEnv("DGS_ARG_ROPE_SCALE"));
  vAddOpt(common_arg(
            { "--rope-freq-base" }, "N", "RoPE base frequency, used by NTK-aware scaling (default: loaded from model)",
            [](common_params & pParams, const std::string & value) { pParams.vRopeFreqBase = std::stof(value); })
            .mSetEnv("DGS_ARG_ROPE_FREQ_BASE"));
  vAddOpt(
    common_arg({ "--rope-freq-scale" }, "N", "RoPE frequency scaling factor, expands context by a factor of 1/N",
               [](common_params & pParams, const std::string & value) { pParams.vRopeFreqScale = std::stof(value); })
      .mSetEnv("DGS_ARG_ROPE_FREQ_SCALE"));
  vAddOpt(common_arg({ "--yarn-orig-ctx" }, "N",
                     fStringFormat("YaRN: original context size of model (default: %d = model training context size)",
                                   pParams.vYarnOrigCtx),
                     [](common_params & pParams, int value) { pParams.vYarnOrigCtx = value; })
            .mSetEnv("DGS_ARG_YARN_ORIG_CTX"));
  vAddOpt(
    common_arg({ "--yarn-ext-factor" }, "N",
               fStringFormat("YaRN: extrapolation mix factor (default: %.2f, 0.0 = full interpolation)",
                             (double) pParams.vYarnExtFactor),
               [](common_params & pParams, const std::string & value) { pParams.vYarnExtFactor = std::stof(value); })
      .mSetEnv("DGS_ARG_YARN_EXT_FACTOR"));
  vAddOpt(
    common_arg(
      { "--yarn-attn-factor" }, "N",
      fStringFormat("YaRN: scale sqrt(t) or attention magnitude (default: %.2f)", (double) pParams.vYarnAttnFactor),
      [](common_params & pParams, const std::string & value) { pParams.vYarnAttnFactor = std::stof(value); })
      .mSetEnv("DGS_ARG_YARN_ATTN_FACTOR"));
  vAddOpt(
    common_arg({ "--yarn-beta-slow" }, "N",
               fStringFormat("YaRN: high correction dim or alpha (default: %.2f)", (double) pParams.vYarnBetaSlow),
               [](common_params & pParams, const std::string & value) { pParams.vYarnBetaSlow = std::stof(value); })
      .mSetEnv("DGS_ARG_YARN_BETA_SLOW"));
  vAddOpt(
    common_arg({ "--yarn-beta-fast" }, "N",
               fStringFormat("YaRN: low correction dim or beta (default: %.2f)", (double) pParams.vYarnBetaFast),
               [](common_params & pParams, const std::string & value) { pParams.vYarnBetaFast = std::stof(value); })
      .mSetEnv("DGS_ARG_YARN_BETA_FAST"));

  vAddOpt(common_arg({ "-kvo", "--kv-offload" }, { "-nkvo", "--no-kv-offload" },
                     fStringFormat("whether to enable KV cache offloading (default: %s)",
                                   pParams.vNoKvOffload ? "disabled" : "enabled"),
                     [](common_params & pParams, bool value) { pParams.vNoKvOffload = !value; })
            .mSetEnv("DGS_ARG_KV_OFFLOAD"));
  vAddOpt(common_arg({ "--repack" }, { "-nr", "--no-repack" },
                     fStringFormat("whether to enable weight repacking (default: %s)",
                                   pParams.vNoExtraBufts ? "disabled" : "enabled"),
                     [](common_params & pParams, bool value) { pParams.vNoExtraBufts = !value; })
            .mSetEnv("DGS_ARG_REPACK"));
  vAddOpt(
    common_arg({ "--no-host" }, "bypass host buffer allowing extra buffers to be used", [](common_params & pParams) {
      pParams.vNoHost = true;
    }).mSetEnv("DGS_ARG_NO_HOST"));
  vAddOpt(common_arg({ "-ctk", "--cache-type-k" }, "TYPE",
                     fStringFormat("KV cache data type for K\n"
                                   "allowed values: %s\n"
                                   "(default: %s)",
                                   fGetAllKvCacheTypes().c_str(), ggml_type_name(pParams.vCacheTypeK)),
                     [](common_params & pParams, const std::string & value) {
                       pParams.vCacheTypeK = fKvCacheTypeFromStr(value);
                     })
            .mSetEnv("DGS_ARG_CACHE_TYPE_K"));
  vAddOpt(common_arg({ "-ctv", "--cache-type-v" }, "TYPE",
                     fStringFormat("KV cache data type for V\n"
                                   "allowed values: %s\n"
                                   "(default: %s)",
                                   fGetAllKvCacheTypes().c_str(), ggml_type_name(pParams.vCacheTypeV)),
                     [](common_params & pParams, const std::string & value) {
                       pParams.vCacheTypeV = fKvCacheTypeFromStr(value);
                     })
            .mSetEnv("DGS_ARG_CACHE_TYPE_V"));

  if (pEx == cDgsExampleServer) {
    // this is to make sure this option appears in the server-specific section of the help message
    vAddOpt(common_arg({ "-np", "--parallel" }, "N",
                       fStringFormat("number of server slots (default: %d, -1 = auto)", pParams.vNParallel),
                       [](common_params & pParams, int value) {
                         if (value == 0) {
                           throw std::invalid_argument("error: invalid value for n_parallel\n");
                         }
                         pParams.vNParallel = value;
                       })
              .mSetEnv("DGS_ARG_N_PARALLEL")
              .mSetExamples({ cDgsExampleServer }));
  } else {
    vAddOpt(common_arg({ "-np", "--parallel" }, "N",
                       fStringFormat("number of parallel sequences to decode (default: %d)", pParams.vNParallel),
                       [](common_params & pParams, int value) { pParams.vNParallel = value; })
              .mSetEnv("DGS_ARG_N_PARALLEL"));
  }

  vAddOpt(common_arg({ "-cb", "--cont-batching" }, { "-nocb", "--no-cont-batching" },
                     fStringFormat("whether to enable continuous batching (a.k.a dynamic batching) (default: %s)",
                                   pParams.vContBatching ? "enabled" : "disabled"),
                     [](common_params & pParams, bool value) { pParams.vContBatching = value; })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_CONT_BATCHING"));

  vAddOpt(common_arg({ "--mmproj-auto" }, { "--no-mmproj" },
                     fStringFormat(
                       "whether to use multimodal projector file (if available), useful when using -hf (default: %s)",
                       pParams.vNoMmproj ? "disabled" : "enabled"),
                     [](common_params & pParams, bool value) { pParams.vNoMmproj = !value; })
            .mSetExamples({ cDgsExampleMtmd, cDgsExampleServer, cDgsExampleCli, cDgsExampleDownload })
            .mSetEnv("DGS_ARG_MMPROJ_AUTO"));

  vAddOpt(common_arg({ "--mtmd-batch-max-tokens" }, "N",
                     fStringFormat("maximum number of image tokens per batch when encoding images (default: %d)",
                                   pParams.vMtmdBatchMaxTokens),
                     [](common_params & pParams, int value) { pParams.vMtmdBatchMaxTokens = value; })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_MTMD_BATCH_MAX_TOKENS"));

  vAddOpt(common_arg({ "--rpc" }, "SERVERS", "comma-separated list of RPC servers (host:port)",
                     [](common_params & pParams, const std::string & value) {
                       if (!fDgsSupportsRpc()) {
                         throw std::invalid_argument("RPC not supported in this build");
                       }
                       fAddRpcDevices(value);
                       GGML_UNUSED(pParams);
                     })
            .mSetEnv("DGS_ARG_RPC"));
  vAddOpt(
    common_arg({ "-lm", "--load-mode" }, "MODE",
               "model loading mode (default: auto)\n"
               "- auto: mmap, unless a device does not support it\n"
               "- none: no special loading mode\n"
               "- mmap: memory-map model (if mmap disabled, slower load but may reduce pageouts if not using mlock)\n"
               "- mlock: force system to keep model in RAM rather than swapping or compressing\n"
               "- mmap+mlock: mmap + force system to keep model in RAM rather than swapping or compressing\n"
               "- dio: use DirectIO if available\n",
               [](common_params & pParams, const std::string & value) {
                 /**/ if (value == "auto") {
                   pParams.vLoadMode = cDgsLoadModeAuto;
                 } else if (value == "none") {
                   pParams.vLoadMode = cDgsLoadModeNone;
                 } else if (value == "mmap") {
                   pParams.vLoadMode = cDgsLoadModeMmap;
                 } else if (value == "mlock") {
                   pParams.vLoadMode = cDgsLoadModeMlock;
                 } else if (value == "mmap+mlock") {
                   pParams.vLoadMode = cDgsLoadModeMmapMlock;
                 } else if (value == "dio") {
                   pParams.vLoadMode = cDgsLoadModeDirectIo;
                 } else {
                   throw std::invalid_argument("invalid value");
                 }
               })
      .mSetEnv("DGS_ARG_LOAD_MODE"));
  vAddOpt(common_arg(
            { "-lzm", "--lazy-mode" }, "MODE",
            "on-demand reading of certain tensors, for example per-layer embeddings (default: auto)\n"
            "- on: read the rows of such tensors from disk on demand instead of keeping them resident (requires mmap)\n"
            "- auto: on, but only for tensors larger than 4 GiB\n"
            "- off: always keep them resident",
            [](common_params & pParams, const std::string & value) {
              /**/ if (value == "on") {
                pParams.vLazyMode = cDgsLazyModeOn;
              } else if (value == "auto") {
                pParams.vLazyMode = cDgsLazyModeAuto;
              } else if (value == "off") {
                pParams.vLazyMode = cDgsLazyModeOff;
              } else {
                throw std::invalid_argument("invalid value");
              }
            })
            .mSetEnv("DGS_ARG_LAZY_MODE"));
  vAddOpt(
    common_arg({ "--numa" }, "TYPE",
               "attempt optimizations that help on some NUMA systems\n"
               "- distribute: spread execution evenly over all nodes\n"
               "- isolate: only spawn threads on CPUs on the node that execution started on\n"
               "- numactl: use the CPU map provided by numactl\n"
               "if run without this previously, it is recommended to drop the system page cache before using this\n"
               "see doc/MANUAL.md",
               [](common_params & pParams, const std::string & value) {
                 /**/ if (value == "distribute" || value == "") {
                   pParams.vNuma = GGML_NUMA_STRATEGY_DISTRIBUTE;
                 } else if (value == "isolate") {
                   pParams.vNuma = GGML_NUMA_STRATEGY_ISOLATE;
                 } else if (value == "numactl") {
                   pParams.vNuma = GGML_NUMA_STRATEGY_NUMACTL;
                 } else {
                   throw std::invalid_argument("invalid value");
                 }
               })
      .mSetEnv("DGS_ARG_NUMA"));
  vAddOpt(
    common_arg({ "-dev", "--device" }, "<dev1,dev2,..>",
               "comma-separated list of devices to use for offloading (none = don't offload)\n"
               "use --list-devices to see a list of available devices",
               [](common_params & pParams, const std::string & value) { pParams.lDevices = fParseDeviceList(value); })
      .mSetEnv("DGS_ARG_DEVICE"));
  vAddOpt(common_arg({ "--list-devices" }, "print list of available devices and exit", [](common_params &) {
    fCommonPrintAvailableDevices();
    exit(0);
  }));
  vAddOpt(common_arg({ "-ot", "--override-tensor" }, "<tensor name pattern>=<buffer type>,...",
                     "override tensor buffer type",
                     [](common_params & pParams, const std::string & value) {
                       fParseTensorBufferOverrides(value, pParams.lTensorBuftOverrides);
                     })
            .mSetEnv("DGS_ARG_OVERRIDE_TENSOR"));

  vAddOpt(common_arg({ "-ncffn", "--n-cpu-ffn" }, "N",
                     "keep the dense FFN weights of the first N layers in the CPU\n"
                     "(dense feed-forward weights)",
                     [](common_params & pParams, int value) {
                       if (value < 0) {
                         throw std::invalid_argument("invalid value");
                       }
                       fLlmAddNCpuFfnOverrides(value, cLlmFfnDenseRegex, pParams.lTensorBuftOverrides);
                     })
            .mSetEnv("DGS_ARG_N_CPU_FFN"));
  GGML_ASSERT(pParams.vNGpuLayers < 0);  // string_format would need to be extended for a default >= 0
  vAddOpt(
    common_arg(
      { "-ngl", "--gpu-layers" }, "N",
      fStringFormat("max. number of layers to store in VRAM, either an exact number, 'auto', or 'all' (default: %s)",
                    pParams.vNGpuLayers == -1 ? "auto" : "all"),
      [](common_params & pParams, const std::string & value) {
        if (value == "auto") {
          pParams.vNGpuLayers = -1;
        } else if (value == "all") {
          pParams.vNGpuLayers = -2;
        } else {
          pParams.vNGpuLayers = std::stoi(value);
        }
        if (!fDgsSupportsGpuOffload()) {
          fprintf(stderr, "warning: no usable GPU found, --gpu-layers option will be ignored\n");
          fprintf(stderr, "warning: one possible reason is that dgs.cpp was compiled without GPU support\n");
          fprintf(stderr, "warning: consult docs/build.md for compilation instructions\n");
        }
      })
      .mSetEnv("DGS_ARG_N_GPU_LAYERS"));
  vAddOpt(
    common_arg(
      { "-sm", "--split-mode" }, "{none,layer,row,tensor}",
      "how to split the model across multiple GPUs, one of:\n"
      "- none: use one GPU only\n"
      "- layer (default): split layers and KV across GPUs (pipelined)\n"
      "- row: split weight across GPUs by rows (parallelized)\n"
      "- tensor: split weights and KV across GPUs (parallelized, EXPERIMENTAL)",
      [](common_params & pParams, const std::string & value) {
        if (value == "none") {
          pParams.vSplitMode = cDgsSplitModeNone;
        } else if (value == "layer") {
          pParams.vSplitMode = cDgsSplitModeLayer;
        } else if (value == "row") {
          pParams.vSplitMode = cDgsSplitModeRow;
        } else if (value == "tensor") {
          pParams.vSplitMode = cDgsSplitModeTensor;
        } else {
          throw std::invalid_argument("invalid value");
        }
        if (!fDgsSupportsGpuOffload()) {
          fprintf(
            stderr,
            "warning: dgs.cpp was compiled without support for GPU offload. Setting the split mode has no effect.\n");
        }
      })
      .mSetEnv("DGS_ARG_SPLIT_MODE"));
  vAddOpt(
    common_arg(
      { "-ts", "--tensor-split" }, "N0,N1,N2,...",
      "fraction of the model to offload to each GPU, comma-separated list of proportions, e.g. 3,1",
      [](common_params & pParams, const std::string & value) {
        std::string vArgNext = value;

        // split string by , and /
        const std::regex           cRegex{ R"([,/]+)" };
        std::sregex_token_iterator vIt{ vArgNext.begin(), vArgNext.end(), cRegex, -1 };
        std::vector<std::string>   lSplitArg{ vIt, {} };
        if (lSplitArg.size() >= fDgsMaxDevices()) {
          throw std::invalid_argument(fStringFormat("got %zu input configs, but system only has %zu devices",
                                                    lSplitArg.size(), fDgsMaxDevices()));
        }
        for (size_t vI = 0; vI < fDgsMaxDevices(); ++vI) {
          if (vI < lSplitArg.size()) {
            pParams.aTensorSplit[vI] = std::stof(lSplitArg[vI]);
          } else {
            pParams.aTensorSplit[vI] = 0.0f;
          }
        }
        if (!fDgsSupportsGpuOffload()) {
          fprintf(
            stderr,
            "warning: dgs.cpp was compiled without support for GPU offload. Setting a tensor split has no effect.\n");
        }
      })
      .mSetEnv("DGS_ARG_TENSOR_SPLIT"));
  vAddOpt(
    common_arg(
      { "-mg", "--main-gpu" }, "INDEX",
      fStringFormat("the GPU to use for the model (with split-mode = none), or for intermediate results and KV (with "
                    "split-mode = row) (default: %d)",
                    pParams.vMainGpu),
      [](common_params & pParams, int value) {
        pParams.vMainGpu = value;
        if (!fDgsSupportsGpuOffload()) {
          fprintf(
            stderr,
            "warning: dgs.cpp was compiled without support for GPU offload. Setting the main GPU has no effect.\n");
        }
      })
      .mSetEnv("DGS_ARG_MAIN_GPU"));
  vAddOpt(
    common_arg({ "-fit", "--fit" }, "[on|off]",
               fStringFormat("whether to adjust unset arguments to fit in device memory ('on' or 'off', default: '%s')",
                             pParams.vFitParams ? "on" : "off"),
               [](common_params & pParams, const std::string & value) {
                 if (fIsTruthy(value)) {
                   pParams.vFitParams = true;
                 } else if (fIsFalsey(value)) {
                   pParams.vFitParams = false;
                 } else {
                   throw std::runtime_error(fStringFormat("error: unknown value for --fit: '%s'\n", value.c_str()));
                 }
               })
      .mSetEnv("DGS_ARG_FIT"));

  vAddOpt(common_arg({ "-fitt", "--fit-target" }, "MiB0,MiB1,MiB2,...",
                     fStringFormat("target margin per device for --fit, comma-separated list of values, "
                                   "single value is broadcast across all devices, default: %zu",
                                   pParams.lFitParamsTarget[0] / (1024 * 1024)),
                     [](common_params & pParams, const std::string & value) {
                       std::string vArgNext = value;

                       // split string by , and /
                       const std::regex           cRegex{ R"([,/]+)" };
                       std::sregex_token_iterator vIt{ vArgNext.begin(), vArgNext.end(), cRegex, -1 };
                       std::vector<std::string>   lSplitArg{ vIt, {} };
                       if (lSplitArg.size() >= fDgsMaxDevices()) {
                         throw std::invalid_argument(
                           fStringFormat("got %zu input configs, but system only has %zu devices", lSplitArg.size(),
                                         fDgsMaxDevices()));
                       }
                       if (lSplitArg.size() == 1) {
                         std::fill(pParams.lFitParamsTarget.begin(), pParams.lFitParamsTarget.end(),
                                   std::stoull(lSplitArg[0]) * 1024 * 1024);
                         return;
                       }
                       for (size_t vI = 0; vI < lSplitArg.size(); vI++) {
                         pParams.lFitParamsTarget[vI] = std::stoull(lSplitArg[vI]) * 1024 * 1024;
                       }
                     })
            .mSetEnv("DGS_ARG_FIT_TARGET"));
  vAddOpt(common_arg({ "-fitc", "--fit-ctx" }, "N",
                     fStringFormat("minimum ctx size that can be set by --fit option, default: %" PRIu32,
                                   pParams.vFitParamsMinCtx),
                     [](common_params & pParams, int value) { pParams.vFitParamsMinCtx = value; })
            .mSetEnv("DGS_ARG_FIT_CTX"));
  vAddOpt(common_arg(
    { "--check-tensors" },
    fStringFormat("check model tensor data for invalid values (default: %s)", pParams.vCheckTensors ? "true" : "false"),
    [](common_params & pParams) { pParams.vCheckTensors = true; }));
  vAddOpt(common_arg({ "--override-kv" }, "KEY=TYPE:VALUE,...",
                     "advanced option to override model metadata by key. to specify multiple overrides, either use "
                     "comma-separated values.\n"
                     "types: int, float, bool, str. example: --override-kv "
                     "tokenizer.ggml.add_bos_token=bool:false,tokenizer.ggml.add_eos_token=bool:false",
                     [](common_params & pParams, const std::string & value) {
                       for (const auto & vItem : fParseCsvRow(value)) {
                         if (!fStringParseKvOverride(vItem.c_str(), pParams.lKvOverrides)) {
                           throw std::runtime_error(
                             fStringFormat("error: Invalid type for KV override: %s\n", vItem.c_str()));
                         }
                       }
                     }));
  vAddOpt(common_arg({ "--op-offload" }, { "--no-op-offload" },
                     fStringFormat("whether to offload host tensor operations to device (default: %s)",
                                   pParams.vNoOpOffload ? "false" : "true"),
                     [](common_params & pParams, bool value) { pParams.vNoOpOffload = !value; }));
  vAddOpt(
    common_arg(
      { "--lora" }, "FNAME", "path to LoRA adapter (use comma-separated values to load multiple adapters)",
      [](common_params & pParams, const std::string & value) {
        for (const auto & vItem : fParseCsvRow(value)) {
          pParams.lLoraAdapters.push_back({ vItem, 1.0, "", "", nullptr });
        }
      }
      // we define this arg on both COMMON and EXPORT_LORA, so when showing help message of export-lora, it will be categorized as "example-specific" arg
      )
      .mSetExamples({ cDgsExampleCommon, cDgsExampleExportLora }));
  vAddOpt(
    common_arg(
      { "--lora-scaled" }, "FNAME:SCALE,...",
      "path to LoRA adapter with user defined scaling (format: FNAME:SCALE,...)\n"
      "note: use comma-separated values",
      [](common_params & pParams, const std::string & value) {
        for (const auto & vItem : fParseCsvRow(value)) {
          auto lParts = fStringSplit<std::string>(vItem, ':');
          if (lParts.size() != 2) {
            throw std::invalid_argument("lora-scaled format: FNAME:SCALE");
          }
          pParams.lLoraAdapters.push_back({ lParts[0], std::stof(lParts[1]), "", "", nullptr });
        }
      }
      // we define this arg on both COMMON and EXPORT_LORA, so when showing help message of export-lora, it will be categorized as "example-specific" arg
      )
      .mSetExamples({ cDgsExampleCommon, cDgsExampleExportLora }));
  vAddOpt(common_arg({ "--control-vector" }, "FNAME",
                     "add a control vector\nnote: use comma-separated values to add multiple control vectors",
                     [](common_params & pParams, const std::string & value) {
                       for (const auto & vItem : fParseCsvRow(value)) {
                         pParams.lControlVectors.push_back({
                           1.0f,
                           vItem,
                         });
                       }
                     }));
  vAddOpt(common_arg({ "--control-vector-scaled" }, "FNAME:SCALE,...",
                     "add a control vector with user defined scaling SCALE\n"
                     "note: use comma-separated values (format: FNAME:SCALE,...)",
                     [](common_params & pParams, const std::string & value) {
                       for (const auto & vItem : fParseCsvRow(value)) {
                         auto lParts = fStringSplit<std::string>(vItem, ':');
                         if (lParts.size() != 2) {
                           throw std::invalid_argument("control-vector-scaled format: FNAME:SCALE");
                         }
                         pParams.lControlVectors.push_back({ std::stof(lParts[1]), lParts[0] });
                       }
                     }));
  vAddOpt(common_arg({ "--control-vector-layer-range" }, "START", "END",
                     "layer range to apply the control vector(s) to, start and end inclusive",
                     [](common_params & pParams, const std::string & pStart, const std::string & end) {
                       pParams.vControlVectorLayerStart = std::stoi(pStart);
                       pParams.vControlVectorLayerEnd   = std::stoi(end);
                     }));
  vAddOpt(common_arg({ "-a", "--alias" }, "STRING", "set model name aliases, comma-separated (to be used by API)",
                     [](common_params & pParams, const std::string & value) {
                       for (auto & vAlias : fStringSplit<std::string>(value, ',')) {
                         vAlias = fStringStrip(vAlias);
                         if (!vAlias.empty()) {
                           pParams.sModelAlias.insert(vAlias);
                         }
                       }
                     })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_ALIAS"));
  vAddOpt(common_arg({ "--tags" }, "STRING", "set model tags, comma-separated (informational, not used for routing)",
                     [](common_params & pParams, const std::string & value) {
                       for (auto & vTag : fStringSplit<std::string>(value, ',')) {
                         vTag = fStringStrip(vTag);
                         if (!vTag.empty()) {
                           pParams.sModelTags.insert(vTag);
                         }
                       }
                     })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_TAGS"));
  vAddOpt(common_arg({ "-m", "--model" }, "FNAME",
                     pEx == cDgsExampleExportLora ? "model path from which to load base model" : "model path to load",
                     [](common_params & pParams, const std::string & value) { pParams.vModel.vPath = value; })
            .mSetExamples({ cDgsExampleCommon, cDgsExampleExportLora, cDgsExampleDownload, cDgsExampleTokenize })
            .mSetEnv("DGS_ARG_MODEL"));
  vAddOpt(common_arg({ "-mu", "--model-url" }, "MODEL_URL", "model download url (default: unused)",
                     [](common_params & pParams, const std::string & value) { pParams.vModel.vUrl = value; })
            .mSetExamples({ cDgsExampleCommon, cDgsExampleDownload, cDgsExampleTokenize })
            .mSetEnv("DGS_ARG_MODEL_URL"));
  vAddOpt(
    common_arg({ "-dr", "--docker-repo" }, "[<repo>/]<model>[:quant]",
               "Docker Hub model repository. repo is optional, default to ai/. quant is optional, default to :latest.\n"
               "example: <publisher>/<selected-qwen-model>\n"
               "(default: unused)",
               [](common_params & pParams, const std::string & value) { pParams.vModel.vDockerRepo = value; })
      .mSetExamples({ cDgsExampleCommon, cDgsExampleDownload, cDgsExampleTokenize })
      .mSetEnv("DGS_ARG_DOCKER_REPO"));
  vAddOpt(common_arg({ "-hf", "-hfr", "--hf-repo" }, "<user>/<model>[:quant]",
                     "Hugging Face model repository; quant is optional, case-insensitive, default to Q4_K_M, or falls "
                     "back to the first file in the repo if Q4_K_M doesn't exist.\n"
                     "mmproj is also downloaded automatically if available. to disable, add --no-mmproj\n"
                     "(default: unused)",
                     [](common_params & pParams, const std::string & value) { pParams.vModel.vHfRepo = value; })
            .mSetExamples({ cDgsExampleCommon, cDgsExampleDownload, cDgsExampleTokenize })
            .mSetEnv("DGS_ARG_HF_REPO"));
  vAddOpt(common_arg({ "-hff", "--hf-file" }, "FILE",
                     "Hugging Face model file. If specified, it will override the quant in --hf-repo (default: unused)",
                     [](common_params & pParams, const std::string & value) { pParams.vModel.hf_file = value; })
            .mSetExamples({ cDgsExampleCommon, cDgsExampleDownload, cDgsExampleTokenize })
            .mSetEnv("DGS_ARG_HF_FILE"));
  vAddOpt(common_arg({ "-hft", "--hf-token" }, "TOKEN",
                     "Hugging Face access token (default: value from HF_TOKEN environment variable)",
                     [](common_params & pParams, const std::string & value) { pParams.vHfToken = value; })
            .mSetExamples({ cDgsExampleCommon, cDgsExampleDownload, cDgsExampleTokenize })
            .mSetEnv("HF_TOKEN"));

  vAddOpt(
    common_arg(
      { "--embd-normalize" }, "N",
      fStringFormat(
        "normalisation for embeddings (default: %d) (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)",
        pParams.vEmbdNormalize),
      [](common_params & pParams, int value) { pParams.vEmbdNormalize = value; })
      .mSetExamples({ cDgsExampleEmbedding, cDgsExampleServer, cDgsExampleDebug }));

  vAddOpt(
    common_arg({ "--host" }, "HOST",
               fStringFormat(
                 "IP addresses to listen on, comma-separated, or UNIX socket paths ending in .sock; with multiple TCP "
                 "addresses, :: binds IPv6 only; overlapping addresses result in undefined behavior (default: %s)",
                 pParams.lHostnames[0].c_str()),
               [](common_params & pParams, const std::string & value) {
                 pParams.lHostnames.clear();
                 for (auto & vHost : fParseCsvRow(value)) {
                   vHost = fStringStrip(vHost);
                   if (!vHost.empty()) {
                     pParams.lHostnames.push_back(vHost);
                   }
                 }
                 if (pParams.lHostnames.empty()) {
                   throw std::invalid_argument("--host requires at least one address");
                 }
               })
      .mSetExamples({ cDgsExampleServer })
      .mSetEnv("DGS_ARG_HOST"));
  vAddOpt(common_arg({ "--port" }, "PORT", fStringFormat("port to listen (default: %d)", pParams.vPort),
                     [](common_params & pParams, int value) { pParams.vPort = value; })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_PORT"));
  vAddOpt(common_arg({ "--reuse-port" },
                     fStringFormat("allow multiple sockets to bind to the same port (default: %s)",
                                   pParams.vReusePort ? "enabled" : "disabled"),
                     [](common_params & pParams) { pParams.vReusePort = true; })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_REUSE_PORT"));
  vAddOpt(common_arg({ "--path" }, "PATH",
                     fStringFormat("path to serve static files from (default: %s)", pParams.vPublicPath.c_str()),
                     [](common_params & pParams, const std::string & value) { pParams.vPublicPath = value; })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_STATIC_PATH"));
  vAddOpt(
    common_arg({ "--cors-origins" }, "ORIGINS",
               fStringFormat("comma-separated list of allowed origins for CORS (default: %s)\n"
                             "if set to special value 'localhost', reflect the Origin header only if it is localhost",
                             pParams.vCorsOrigins.c_str()),
               [](common_params & pParams, const std::string & value) {
                 pParams.vCorsOrigins         = value;
                 pParams.vCorsOriginsExplicit = true;
               })
      .mSetExamples({ cDgsExampleServer })
      .mSetEnv("DGS_ARG_CORS_ORIGINS"));
  vAddOpt(common_arg({ "--cors-methods" }, "METHODS",
                     fStringFormat("comma-separated list of allowed methods for CORS (default: %s)",
                                   pParams.vCorsMethods.c_str()),
                     [](common_params & pParams, const std::string & value) { pParams.vCorsMethods = value; })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_CORS_METHODS"));
  vAddOpt(common_arg({ "--cors-headers" }, "HEADERS",
                     fStringFormat("comma-separated list of allowed headers for CORS (default: %s)",
                                   pParams.vCorsHeaders.c_str()),
                     [](common_params & pParams, const std::string & value) { pParams.vCorsHeaders = value; })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_CORS_HEADERS"));
  vAddOpt(common_arg({ "--cors-credentials" }, { "--no-cors-credentials" },
                     fStringFormat("whether to allow credentials for CORS (default: %s)\n"
                                   "note: if this is enabled and --cors-origins is set to * (default), the Origin "
                                   "header will be echoed back, and credentials will always be allowed",
                                   pParams.vCorsCredentials ? "enabled" : "disabled"),
                     [](common_params & pParams, bool value) { pParams.vCorsCredentials = value; })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_CORS_CREDENTIALS"));
  vAddOpt(common_arg({ "--api-prefix" }, "PREFIX",
                     fStringFormat("prefix path the server serves from, without the trailing slash (default: %s)",
                                   pParams.vApiPrefix.c_str()),
                     [](common_params & pParams, const std::string & value) { pParams.vApiPrefix = value; })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_API_PREFIX"));
  vAddOpt(common_arg({ "--ui-config" }, "JSON", "JSON that provides default UI settings (overrides UI defaults)",
                     [](common_params & pParams, const std::string & value) { pParams.vUiConfigJson = value; })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_UI_CONFIG"));
  vAddOpt(
    common_arg({ "--ui-config-file" }, "PATH", "JSON file that provides default UI settings (overrides UI defaults)",
               [](common_params & pParams, const std::string & value) { pParams.vUiConfigJson = fReadFile(value); })
      .mSetExamples({ cDgsExampleServer })
      .mSetEnv("DGS_ARG_UI_CONFIG_FILE"));
  vAddOpt(
    common_arg(
      { "--ui-mcp-proxy" }, { "--no-ui-mcp-proxy" },
      "experimental: whether to enable MCP CORS proxy - do not enable in untrusted environments (default: disabled)",
      [](common_params & pParams, bool value) { pParams.vUiMcpProxy = value; })
      .mSetExamples({ cDgsExampleServer })
      .mSetEnv("DGS_ARG_UI_MCP_PROXY"));
  vAddOpt(
    common_arg(
      { "--tools" }, "TOOL1,TOOL2,...",
      "experimental: whether to enable built-in tools for AI agents - do not enable in untrusted environments "
      "(default: no tools)\n"
      "specify \"all\" to enable all tools\n"
      "available tools: read_file, file_glob_search, grep_search, exec_shell_command, write_file, edit_file, get_info\n"
      "note: for security reasons, this will limit --cors-origins to localhost by default",
      [](common_params & pParams, const std::string & value) { pParams.lServerTools = fParseCsvRow(value); })
      .mSetExamples({ cDgsExampleServer })
      .mSetEnv("DGS_ARG_TOOLS"));
  vAddOpt(
    common_arg(
      { "--tools-runtime" }, "OPTION",
      "experimental: run tools in a separate runtime environment (default: none, use host environment)\n"
      "available options:\n"
      "  'docker:<image>', 'podman:<image>': spin up a new container and reuse it for all invocations, clean up on "
      "server exit\n"
      "  'docker-container:<id>', 'podman-container:<id>': use an existing container by ID, won't stop on server exit\n"
      "  'ssh:<target>': run tools on a remote POSIX host over SSH, key-based auth and a trusted host key are "
      "required\n",
      [](common_params & pParams, const std::string & value) { pParams.vServerToolsRuntime = value; })
      .mSetExamples({ cDgsExampleServer })
      .mSetEnv("DGS_ARG_TOOLS_RUNTIME"));
  vAddOpt(common_arg({ "--local-shell-mcp-server" },
                     "enable the local shell MCP server (bash.run, zsh.run, sh.run, psh.run)",
                     [](common_params & pParams) { pParams.vShellMcpServer = true; })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_LOCAL_SHELL_MCP_SERVER"));

  vAddOpt(common_arg({ "--mcp-servers-config" }, "PATH",
                     "experimental: path to JSON file with MCP server definitions (Cursor-compatible format) - do not "
                     "enable in untrusted environments (default: none)\n"
                     "note: for security reasons, this will limit --cors-origins to localhost by default",
                     [](common_params & pParams, const std::string & value) { pParams.vMcpServersConfig = value; })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_MCP_SERVERS_CONFIG"));
  vAddOpt(common_arg({ "--mcp-servers-json" }, "JSON",
                     "experimental: inline JSON with MCP server definitions (Cursor-compatible format) - do not enable "
                     "in untrusted environments (default: none)\n"
                     "note: for security reasons, this will limit --cors-origins to localhost by default",
                     [](common_params & pParams, const std::string & value) { pParams.vMcpServersJson = value; })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_MCP_SERVERS_JSON"));
  vAddOpt(common_arg({ "-ag", "--agent" }, { "-no-ag", "--no-agent" },
                     "whether to enable CORS proxy and all built-in tools - do not enable in untrusted environments "
                     "(default: disabled)\n"
                     "note: for security reasons, this will limit --cors-origins to localhost by default",
                     [](common_params & pParams, bool value) {
                       if (value) {
                         pParams.lServerTools = { "all" };
                         pParams.vUiMcpProxy  = true;
                       } else {
                         pParams.lServerTools.clear();
                         pParams.vUiMcpProxy = false;
                       }
                       // note: do not modify cors_origins here, as the options are not evaluated in order (user may explicitly set --cors-origins before --agent)
                     })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_AGENT"));
  vAddOpt(common_arg({ "--ui" }, { "--no-ui" },
                     fStringFormat("whether to enable the Web UI (default: %s)", pParams.vUi ? "enabled" : "disabled"),
                     [](common_params & pParams, bool value) { pParams.vUi = value; })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_UI"));
  vAddOpt(
    common_arg({ "--embedding" },
               fStringFormat(
                 "restrict to only support embedding use case; use only with dedicated embedding models (default: %s)",
                 pParams.vEmbedding ? "enabled" : "disabled"),
               [](common_params & pParams) { pParams.vEmbedding = true; })
      .mSetExamples({ cDgsExampleServer, cDgsExampleDebug })
      .mSetEnv("DGS_ARG_EMBEDDINGS"));
  vAddOpt(common_arg({ "--rerank" }, fStringFormat("enable reranking endpoint on server (default: %s)", "disabled"),
                     [](common_params & pParams) {
                       pParams.vEmbedding   = true;
                       pParams.vPoolingType = cDgsPoolingTypeRank;
                     })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_RERANKING"));
  vAddOpt(
    common_arg(
      { "--api-key" }, "KEY",
      "API key to use for authentication, multiple keys can be provided as a comma-separated list (default: none)",
      [](common_params & pParams, const std::string & value) {
        for (const auto & vKey : fParseCsvRow(value)) {
          if (!vKey.empty()) {
            pParams.lApiKeys.push_back(vKey);
          }
        }
      })
      .mSetExamples({ cDgsExampleServer })
      .mSetEnv("DGS_API_KEY"));
  vAddOpt(common_arg({ "--api-key-file" }, "FNAME",
                     "path to file containing API keys, one per line; lines starting with a hash are treated as "
                     "comments (default: none)",
                     [](common_params & pParams, const std::string & value) {
                       std::ifstream vKeyFile(value);
                       if (!vKeyFile) {
                         throw std::runtime_error(fStringFormat("error: failed to open file '%s'\n", value.c_str()));
                       }
                       std::string vKey;
                       while (std::getline(vKeyFile, vKey)) {
                         if (!vKey.empty() && vKey[0] != '#') {
                           pParams.lApiKeys.push_back(vKey);
                         }
                       }
                       vKeyFile.close();
                     })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_API_KEY_FILE"));
  vAddOpt(common_arg({ "--ssl-key-file" }, "FNAME", "path to file a PEM-encoded SSL private key",
                     [](common_params & pParams, const std::string & value) { pParams.vSslFileKey = value; })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_SSL_KEY_FILE"));
  vAddOpt(common_arg({ "--ssl-cert-file" }, "FNAME", "path to file a PEM-encoded SSL certificate",
                     [](common_params & pParams, const std::string & value) { pParams.vSslFileCert = value; })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_SSL_CERT_FILE"));
  vAddOpt(common_arg({ "--chat-template-kwargs" }, "STRING",
                     "sets additional params for the json template parser, must be a valid json object string, e.g. "
                     "'{\"key1\":\"value1\",\"key2\":\"value2\"}'",
                     [](common_params & pParams, const std::string & value) {
                       auto vParsed = json::mParse(value);
                       for (const auto & vItem : vParsed.mItems()) {
                         if (vItem.mKey() == "enable_thinking") {
                           throw std::invalid_argument(
                             "Use the current reasoning options instead of chat-template-kwargs.enable_thinking");
                         }
                         if (vItem.mKey() == "preserve_reasoning") {
                           throw std::invalid_argument(
                             "Use the current reasoning options instead of chat-template-kwargs.preserve_reasoning");
                         }
                         pParams.dDefaultTemplateKwargs[vItem.mKey()] = vItem.value().mDump();
                       }
                     })
            .mSetExamples({ cDgsExampleServer, cDgsExampleCli })
            .mSetEnv("DGS_ARG_CHAT_TEMPLATE_KWARGS"));
  vAddOpt(common_arg({ "-to", "--timeout" }, "N",
                     fStringFormat("server read/write timeout in seconds (default: %d)", pParams.vTimeoutRead),
                     [](common_params & pParams, int value) {
                       pParams.vTimeoutRead  = value;
                       pParams.vTimeoutWrite = value;
                     })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_TIMEOUT"));
  vAddOpt(common_arg(
            { "--sse-ping-interval" }, "N",
            fStringFormat("server SSE ping interval in seconds (-1 = disabled, default: %d)", pParams.vSsePingInterval),
            [](common_params & pParams, int value) { pParams.vSsePingInterval = value; })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_SSE_PING_INTERVAL"));
  vAddOpt(
    common_arg({ "--threads-http" }, "N",
               fStringFormat("number of threads used to process HTTP requests (default: %d)", pParams.vNThreadsHttp),
               [](common_params & pParams, int value) { pParams.vNThreadsHttp = value; })
      .mSetExamples({ cDgsExampleServer })
      .mSetEnv("DGS_ARG_THREADS_HTTP"));
  vAddOpt(common_arg({ "--cache-prompt" }, { "--no-cache-prompt" },
                     fStringFormat("whether to enable prompt caching (default: %s)",
                                   pParams.vCachePrompt ? "enabled" : "disabled"),
                     [](common_params & pParams, bool value) { pParams.vCachePrompt = value; })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_CACHE_PROMPT"));
  vAddOpt(common_arg({ "--cache-reuse" }, "N",
                     fStringFormat("min chunk size to attempt reusing from the cache via KV shifting, requires prompt "
                                   "caching to be enabled (default: %d)\n"
                                   "[(card)](https://ggml.ai/f0.png)",
                                   pParams.vNCacheReuse),
                     [](common_params & pParams, int value) { pParams.vNCacheReuse = value; })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_CACHE_REUSE"));
  vAddOpt(common_arg({ "--metrics" },
                     fStringFormat("enable prometheus compatible metrics endpoint (default: %s)",
                                   pParams.vEndpointMetrics ? "enabled" : "disabled"),
                     [](common_params & pParams) { pParams.vEndpointMetrics = true; })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_ENDPOINT_METRICS"));
  vAddOpt(common_arg({ "--props" },
                     fStringFormat("enable changing global properties via POST /props (default: %s)",
                                   pParams.vEndpointProps ? "enabled" : "disabled"),
                     [](common_params & pParams) { pParams.vEndpointProps = true; })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_ENDPOINT_PROPS"));
  vAddOpt(common_arg({ "--slots" }, { "--no-slots" },
                     fStringFormat("expose slots monitoring endpoint (default: %s)",
                                   pParams.vEndpointSlots ? "enabled" : "disabled"),
                     [](common_params & pParams, bool value) { pParams.vEndpointSlots = value; })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_ENDPOINT_SLOTS"));
  vAddOpt(common_arg({ "--slot-save-path" }, "PATH", "path to save slot kv cache (default: disabled)",
                     [](common_params & pParams, const std::string & value) {
                       pParams.vSlotSavePath = value;
                       if (!fFsIsDirectory(pParams.vSlotSavePath)) {
                         throw std::invalid_argument("not a directory: " + value);
                       }
                       // if doesn't end with DIRECTORY_SEPARATOR, add it
                       if (!pParams.vSlotSavePath.empty() &&
                           pParams.vSlotSavePath[pParams.vSlotSavePath.size() - 1] != DIRECTORY_SEPARATOR) {
                         pParams.vSlotSavePath += DIRECTORY_SEPARATOR;
                       }
                     })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_SLOT_SAVE_PATH"));
  vAddOpt(common_arg({ "--media-path" }, "PATH",
                     "directory for loading local media files; files can be accessed via file:// URLs using relative "
                     "paths (default: disabled)",
                     [](common_params & pParams, const std::string & value) {
                       pParams.vMediaPath = value;
                       if (!fFsIsDirectory(pParams.vMediaPath)) {
                         throw std::invalid_argument("not a directory: " + value);
                       }
                       // if doesn't end with DIRECTORY_SEPARATOR, add it
                       if (!pParams.vMediaPath.empty() &&
                           pParams.vMediaPath[pParams.vMediaPath.size() - 1] != DIRECTORY_SEPARATOR) {
                         pParams.vMediaPath += DIRECTORY_SEPARATOR;
                       }
                     })
            .mSetExamples({ cDgsExampleServer }));
  vAddOpt(common_arg({ "--models-dir" }, "PATH",
                     "directory containing models for the router server (default: disabled)",
                     [](common_params & pParams, const std::string & value) { pParams.vModelsDir = value; })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_MODELS_DIR"));
  vAddOpt(common_arg({ "--models-preset" }, "PATH",
                     "path to INI file containing model presets for the router server (default: disabled)",
                     [](common_params & pParams, const std::string & value) { pParams.vModelsPreset = value; })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_MODELS_PRESET"));
  vAddOpt(
    common_arg(
      { "--models-max" }, "N",
      fStringFormat("for router server, maximum number of models to load simultaneously (default: %d, 0 = unlimited)",
                    pParams.vModelsMax),
      [](common_params & pParams, int value) { pParams.vModelsMax = value; })
      .mSetExamples({ cDgsExampleServer })
      .mSetEnv("DGS_ARG_MODELS_MAX"));
  vAddOpt(common_arg({ "--models-autoload" }, { "--no-models-autoload" },
                     fStringFormat("for router server, whether to automatically load models (default: %s)",
                                   pParams.vModelsAutoload ? "enabled" : "disabled"),
                     [](common_params & pParams, bool value) { pParams.vModelsAutoload = value; })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_MODELS_AUTOLOAD"));

  vAddOpt(common_arg(
            { "--reasoning-format" }, "FORMAT",
            "controls whether thought tags are allowed and/or extracted from the response, and in which format they're "
            "returned; one of:\n"
            "- none: leaves thoughts unparsed in `message.content`\n"
            "- deepseek: puts thoughts in `message.reasoning_content`\n"
            "(default: auto)",
            [](common_params & pParams, const std::string & value) {
              pParams.vReasoningFormat = fCommonReasoningFormatFromName(value);
            })
            .mSetExamples({ cDgsExampleServer, cDgsExampleCompletion, cDgsExampleCli })
            .mSetEnv("DGS_ARG_THINK"));
  vAddOpt(
    common_arg({ "-rea", "--reasoning" }, "[on|off|auto]",
               "Use reasoning/thinking in the chat ('on', 'off', or 'auto', default: 'auto' (detect from template))",
               [](common_params & pParams, const std::string & value) {
                 if (fIsTruthy(value)) {
                   pParams.vEnableReasoning                          = 1;
                   pParams.dDefaultTemplateKwargs["enable_thinking"] = "true";
                 } else if (fIsFalsey(value)) {
                   pParams.vEnableReasoning                          = 0;
                   pParams.dDefaultTemplateKwargs["enable_thinking"] = "false";
                 } else if (fIsAutoy(value)) {
                   pParams.vEnableReasoning = -1;
                 } else {
                   throw std::invalid_argument(
                     fStringFormat("error: unknown value for --reasoning: '%s'\n", value.c_str()));
                 }
               })
      .mSetExamples({ cDgsExampleServer, cDgsExampleCompletion, cDgsExampleCli })
      .mSetEnv("DGS_ARG_REASONING"));
  vAddOpt(common_arg({ "--reasoning-effort" }, "LEVEL",
                     "reasoning effort level given to the chat template: 'default' to keep the template default,\n"
                     "or a level such as 'minimal', 'low', 'medium', 'high', 'xhigh' or 'max' (default: default)",
                     [](common_params & pParams, const std::string & value) {
                       if (value == "default") {
                         pParams.dDefaultTemplateKwargs.erase("reasoning_effort");
                       } else {
                         pParams.dDefaultTemplateKwargs["reasoning_effort"] = json(value).mDump();
                       }
                     })
            .mSetExamples({ cDgsExampleServer, cDgsExampleCompletion, cDgsExampleCli })
            .mSetEnv("DGS_ARG_REASONING_EFFORT"));
  vAddOpt(common_arg(
            { "--reasoning-budget" }, "N",
            "token budget for thinking: -1 for unrestricted, 0 for immediate end, N>0 for token budget (default: -1)",
            [](common_params & pParams, int value) {
              if (value < -1) {
                throw std::invalid_argument("invalid value");
              }
              pParams.vSampling.vReasoningBudgetTokens = value;
            })
            .mSetExamples({ cDgsExampleServer, cDgsExampleCompletion, cDgsExampleCli })
            .mSetEnv("DGS_ARG_THINK_BUDGET"));
  vAddOpt(
    common_arg(
      { "--reasoning-budget-message" }, "MESSAGE",
      "message injected before the end-of-thinking tag when reasoning budget is exhausted (default: none)",
      [](common_params & pParams, const std::string & value) { pParams.vSampling.vReasoningBudgetMessage = value; })
      .mSetExamples({ cDgsExampleServer, cDgsExampleCompletion, cDgsExampleCli })
      .mSetEnv("DGS_ARG_THINK_BUDGET_MESSAGE"));
  vAddOpt(
    common_arg({ "--reasoning-preserve" }, { "--no-reasoning-preserve" },
               "preserve reasoning trace in the full history, not just the last assistant message (default: enabled)\n"
               "compatible with certain templates having 'supports_preserve_reasoning' capability\n"
               "example: https://docs.z.ai/guides/capabilities/thinking-mode#preserved-thinking",
               [](common_params & pParams, bool value) {
                 if (value) {
                   pParams.dDefaultTemplateKwargs["preserve_reasoning"] = "true";
                 } else {
                   pParams.dDefaultTemplateKwargs["preserve_reasoning"] = "false";
                 }
                 pParams.vPreserveReasoningSpecified = true;
               })
      .mSetExamples({ cDgsExampleServer, cDgsExampleCompletion, cDgsExampleCli })
      .mSetEnv("DGS_ARG_REASONING_PRESERVE"));
  vAddOpt(common_arg({ "--chat-template" }, "JINJA_TEMPLATE",
                     "Use a complete Jinja template; the default is read from the selected Qwen GGUF",
                     [](common_params & pParams, const std::string & value) { pParams.vChatTemplate = value; })
            .mSetExamples({ cDgsExampleCompletion, cDgsExampleCli, cDgsExampleServer, cDgsExampleMtmd })
            .mSetEnv("DGS_ARG_CHAT_TEMPLATE"));
  vAddOpt(
    common_arg({ "--chat-template-file" }, "JINJA_TEMPLATE_FILE",
               "Use a complete Jinja template; the default is read from the selected Qwen GGUF",
               [](common_params & pParams, const std::string & value) { pParams.vChatTemplate = fReadFile(value); })
      .mSetExamples({ cDgsExampleCompletion, cDgsExampleCli, cDgsExampleServer })
      .mSetEnv("DGS_ARG_CHAT_TEMPLATE_FILE"));
  vAddOpt(
    common_arg(
      { "--skip-chat-parsing" }, { "--no-skip-chat-parsing" },
      fStringFormat("force a pure content parser, even if a Jinja template is specified; model will output everything "
                    "in the content section, including any reasoning and/or tool calls (default: disabled)"),
      [](common_params & pParams, bool value) { pParams.vForcePureContentParser = value; })
      .mSetExamples({ cDgsExampleCompletion, cDgsExampleCli, cDgsExampleServer })
      .mSetEnv("DGS_ARG_SKIP_CHAT_PARSING"));
  vAddOpt(common_arg({ "--prefill-assistant" }, { "--no-prefill-assistant" },
                     fStringFormat("whether to prefill the assistant's response if the last message is an assistant "
                                   "message (default: prefill enabled)\n"
                                   "when this flag is set, if the last message is an assistant message then it will be "
                                   "treated as a full message and not prefilled\n"),
                     [](common_params & pParams, bool value) { pParams.vPrefillAssistant = value; })
            .mSetExamples({ cDgsExampleServer })
            .mSetEnv("DGS_ARG_PREFILL_ASSISTANT"));
  vAddOpt(common_arg({ "-sps", "--slot-prompt-similarity" }, "SIMILARITY",
                     fStringFormat("how much the prompt of a request must match the prompt of a slot in order to use "
                                   "that slot (default: %.2f, 0.0 = disabled)\n",
                                   pParams.vSlotPromptSimilarity),
                     [](common_params & pParams, const std::string & value) {
                       pParams.vSlotPromptSimilarity = std::stof(value);
                     })
            .mSetExamples({ cDgsExampleServer }));
  vAddOpt(common_arg({ "--lora-init-without-apply" },
                     fStringFormat(
                       "load LoRA adapters without applying them (apply later via POST /lora-adapters) (default: %s)",
                       pParams.vLoraInitWithoutApply ? "enabled" : "disabled"),
                     [](common_params & pParams) { pParams.vLoraInitWithoutApply = true; })
            .mSetExamples({ cDgsExampleServer }));
  vAddOpt(common_arg({ "--sleep-idle-seconds" }, "SECONDS",
                     fStringFormat(
                       "number of seconds of idleness after which the server will sleep (default: %d; -1 = disabled)",
                       pParams.vSleepIdleSeconds),
                     [](common_params & pParams, int value) {
                       if (value == 0 || value < -1) {
                         throw std::invalid_argument("invalid value: cannot be 0 or less than -1");
                       }
                       pParams.vSleepIdleSeconds = value;
                     })
            .mSetExamples({ cDgsExampleServer }));

  vAddOpt(common_arg({ "--log-disable" }, "Log disable", [](common_params &) { fCommonLogPause(fCommonLogMain()); }));
  vAddOpt(common_arg({ "--log-file" }, "FNAME", "Log to file", [](common_params &, const std::string & value) {
            fCommonLogSetFile(fCommonLogMain(), value.c_str());
          }).mSetEnv("DGS_ARG_LOG_FILE"));
  vAddOpt(common_arg(
            { "--log-jsonl" }, { "--no-log-jsonl" },
            "Log as JSONL (one JSON object per line) to stdout, this also disables colored logging (default: disabled)",
            [](common_params &, bool value) { fCommonLogSetJsonl(value); })
            .mSetEnv("DGS_ARG_LOG_JSONL"));
  vAddOpt(
    common_arg({ "--log-prompts-dir" }, "PATH",
               "Log prompts to directory (auto-created if not present; only used for debugging, default: disabled)",
               [](common_params & pParams, const std::string & value) {
                 pParams.vPathPromptsLogDir = value;
                 std::error_code vEc;
                 std::filesystem::create_directories(value, vEc);
                 if (vEc) {
                   fprintf(stderr, "warning: failed to create prompts-log-dir '%s': %s\n", value.c_str(),
                           vEc.message().c_str());
                 }
               })
      .mSetExamples({ cDgsExampleServer, cDgsExampleCli }));
  vAddOpt(common_arg({ "--log-colors" }, "[on|off|auto]",
                     "Set colored logging ('on', 'off', or 'auto', default: 'auto')\n"
                     "'auto' enables colors when output is to a terminal",
                     [](common_params &, const std::string & value) {
                       if (fIsTruthy(value)) {
                         fCommonLogSetColors(fCommonLogMain(), cLogColorsEnabled);
                       } else if (fIsFalsey(value)) {
                         fCommonLogSetColors(fCommonLogMain(), cLogColorsDisabled);
                       } else if (fIsAutoy(value)) {
                         fCommonLogSetColors(fCommonLogMain(), cLogColorsAuto);
                       } else {
                         throw std::invalid_argument(
                           fStringFormat("error: unknown value for --log-colors: '%s'\n", value.c_str()));
                       }
                     })
            .mSetEnv("DGS_ARG_LOG_COLORS"));
  vAddOpt(common_arg({ "-v", "--verbose" },
                     "Set verbosity level to infinity (i.e. log all messages, useful for debugging)",
                     [](common_params & pParams) {
                       pParams.vVerbosity = INT_MAX;
                       fCommonLogSetVerbosityThold(INT_MAX);
                     }));
  vAddOpt(common_arg({ "--offline" }, "Offline mode: forces use of cache, prevents network access",
                     [](common_params & pParams) { pParams.vOffline = true; })
            .mSetExamples({ cDgsExampleCommon, cDgsExampleDownload, cDgsExampleTokenize })
            .mSetEnv("DGS_ARG_OFFLINE"));
  vAddOpt(
    common_arg({ "-lv", "--verbosity" }, "N",
               fStringFormat("Set the verbosity threshold. Messages with a higher verbosity will be ignored. Values:\n"
                             " - 0: generic output\n"
                             " - 1: error\n"
                             " - 2: warning\n"
                             " - 3: info\n"
                             " - 4: trace (more info)\n"
                             " - 5: debug\n"
                             "(default: %d)\n",
                             pParams.vVerbosity),
               [](common_params & pParams, int value) {
                 pParams.vVerbosity = value;
                 fCommonLogSetVerbosityThold(value);
               })
      .mSetEnv("DGS_ARG_LOG_VERBOSITY"));
  vAddOpt(common_arg({ "--log-prefix" }, { "--no-log-prefix" }, "Enable prefix in log messages",
                     [](common_params &, bool value) { fCommonLogSetPrefix(fCommonLogMain(), value); })
            .mSetEnv("DGS_ARG_LOG_PREFIX"));
  vAddOpt(common_arg({ "--log-timestamps" }, { "--no-log-timestamps" }, "Enable timestamps in log messages",
                     [](common_params &, bool value) { fCommonLogSetTimestamps(fCommonLogMain(), value); })
            .mSetEnv("DGS_ARG_LOG_TIMESTAMPS"));

  //
  // speculative parameters
  //

  vAddOpt(
    common_arg(
      { "--spec-draft-hf" }, "<user>/<model>[:quant]", "Same as --hf-repo, but for the draft model (default: unused)",
      [](common_params & pParams, const std::string & value) { pParams.vSpeculative.vDraft.vMparams.vHfRepo = value; })
      .mSetSpec()
      .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli })
      .mSetEnv("DGS_ARG_SPEC_DRAFT_HF_REPO"));
  vAddOpt(common_arg({ "--spec-draft-threads" }, "N",
                     "number of threads to use during generation (default: same as --threads)",
                     [](common_params & pParams, int value) {
                       pParams.vSpeculative.vDraft.vCpuparams.vNThreads = value;
                       if (pParams.vSpeculative.vDraft.vCpuparams.vNThreads <= 0) {
                         pParams.vSpeculative.vDraft.vCpuparams.vNThreads = std::thread::hardware_concurrency();
                       }
                     })
            .mSetSpec()
            .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli }));
  vAddOpt(common_arg({ "--spec-draft-threads-batch" }, "N",
                     "number of threads to use during batch and prompt processing (default: same as --threads-draft)",
                     [](common_params & pParams, int value) {
                       pParams.vSpeculative.vDraft.vCpuparamsBatch.vNThreads = value;
                       if (pParams.vSpeculative.vDraft.vCpuparamsBatch.vNThreads <= 0) {
                         pParams.vSpeculative.vDraft.vCpuparamsBatch.vNThreads = std::thread::hardware_concurrency();
                       }
                     })
            .mSetSpec()
            .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli }));
  vAddOpt(common_arg({ "--spec-draft-cpu-mask" }, "M",
                     "Draft model CPU affinity mask. Complements cpu-range-draft (default: same as --cpu-mask)",
                     [](common_params & pParams, const std::string & pMask) {
                       pParams.vSpeculative.vDraft.vCpuparams.vMaskValid = true;
                       if (!fParseCpuMask(pMask, pParams.vSpeculative.vDraft.vCpuparams.aCpumask)) {
                         throw std::invalid_argument("invalid cpumask");
                       }
                     })
            .mSetSpec()
            .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli }));
  vAddOpt(common_arg({ "--spec-draft-cpu-range" }, "lo-hi", "Ranges of CPUs for affinity. Complements --cpu-mask-draft",
                     [](common_params & pParams, const std::string & pRange) {
                       pParams.vSpeculative.vDraft.vCpuparams.vMaskValid = true;
                       if (!fParseCpuRange(pRange, pParams.vSpeculative.vDraft.vCpuparams.aCpumask)) {
                         throw std::invalid_argument("invalid range");
                       }
                     })
            .mSetSpec()
            .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli }));
  vAddOpt(
    common_arg({ "--spec-draft-cpu-strict" }, "<0|1>",
               "Use strict CPU placement for draft model (default: same as --cpu-strict)",
               [](common_params & pParams, int value) { pParams.vSpeculative.vDraft.vCpuparams.vStrictCpu = value; })
      .mSetSpec()
      .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli }));
  vAddOpt(common_arg(
            { "--spec-draft-prio" }, "N",
            fStringFormat("set draft process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n",
                          pParams.vSpeculative.vDraft.vCpuparams.vPriority),
            [](common_params & pParams, int pPrio) {
              if (pPrio < 0 || pPrio > 3) {
                throw std::invalid_argument("invalid value");
              }
              pParams.vSpeculative.vDraft.vCpuparams.vPriority = (enum ggml_sched_priority) pPrio;
            })
            .mSetSpec()
            .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli }));
  vAddOpt(common_arg({ "--spec-draft-poll" }, "<0|1>",
                     "Use polling to wait for draft model work (default: same as --poll)",
                     [](common_params & pParams, int value) { pParams.vSpeculative.vDraft.vCpuparams.vPoll = value; })
            .mSetSpec()
            .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli }));
  vAddOpt(common_arg({ "--spec-draft-cpu-mask-batch" }, "M",
                     "Draft model CPU affinity mask. Complements cpu-range-draft (default: same as --cpu-mask)",
                     [](common_params & pParams, const std::string & pMask) {
                       pParams.vSpeculative.vDraft.vCpuparamsBatch.vMaskValid = true;
                       if (!fParseCpuMask(pMask, pParams.vSpeculative.vDraft.vCpuparamsBatch.aCpumask)) {
                         throw std::invalid_argument("invalid cpumask");
                       }
                     })
            .mSetSpec()
            .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli }));

  vAddOpt(common_arg(
            { "--spec-draft-cpu-strict-batch" }, "<0|1>",
            "Use strict CPU placement for draft model (default: --cpu-strict-draft)",
            [](common_params & pParams, int value) { pParams.vSpeculative.vDraft.vCpuparamsBatch.vStrictCpu = value; })
            .mSetSpec()
            .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli }));
  vAddOpt(common_arg(
            { "--spec-draft-prio-batch" }, "N",
            fStringFormat("set draft process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n",
                          pParams.vSpeculative.vDraft.vCpuparamsBatch.vPriority),
            [](common_params & pParams, int pPrio) {
              if (pPrio < 0 || pPrio > 3) {
                throw std::invalid_argument("invalid value");
              }
              pParams.vSpeculative.vDraft.vCpuparamsBatch.vPriority = (enum ggml_sched_priority) pPrio;
            })
            .mSetSpec()
            .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli }));
  vAddOpt(common_arg(
            { "--spec-draft-poll-batch" }, "<0|1>", "Use polling to wait for draft model work (default: --poll-draft)",
            [](common_params & pParams, int value) { pParams.vSpeculative.vDraft.vCpuparamsBatch.vPoll = value; })
            .mSetSpec()
            .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli }));
  vAddOpt(
    common_arg({ "--spec-draft-type-k" }, "TYPE",
               fStringFormat("KV cache data type for K for the draft model\n"
                             "allowed values: %s\n"
                             "(default: %s)",
                             fGetAllKvCacheTypes().c_str(), ggml_type_name(pParams.vSpeculative.vDraft.vCacheTypeK)),
               [](common_params & pParams, const std::string & value) {
                 pParams.vSpeculative.vDraft.vCacheTypeK = fKvCacheTypeFromStr(value);
               })
      .mSetEnv("DGS_ARG_SPEC_DRAFT_CACHE_TYPE_K"));
  vAddOpt(
    common_arg({ "--spec-draft-type-v" }, "TYPE",
               fStringFormat("KV cache data type for V for the draft model\n"
                             "allowed values: %s\n"
                             "(default: %s)",
                             fGetAllKvCacheTypes().c_str(), ggml_type_name(pParams.vSpeculative.vDraft.vCacheTypeV)),
               [](common_params & pParams, const std::string & value) {
                 pParams.vSpeculative.vDraft.vCacheTypeV = fKvCacheTypeFromStr(value);
               })
      .mSetEnv("DGS_ARG_SPEC_DRAFT_CACHE_TYPE_V"));
  vAddOpt(common_arg({ "--spec-draft-override-tensor" }, "<tensor name pattern>=<buffer type>,...",
                     "override tensor buffer type for draft model",
                     [](common_params & pParams, const std::string & value) {
                       fParseTensorBufferOverrides(value, pParams.vSpeculative.vDraft.lTensorBuftOverrides);
                     })
            .mSetSpec()
            .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli }));

  vAddOpt(common_arg({ "--spec-draft-n-max" }, "N",
                     fStringFormat("number of tokens to draft for speculative decoding (default: %d)",
                                   pParams.vSpeculative.vDraft.vNMax),
                     [](common_params & pParams, int value) {
                       if (value < 0) {
                         throw std::invalid_argument("invalid value");
                       }
                       pParams.vSpeculative.vDraft.vNMax = value;
                     })
            .mSetSpec()
            .mSetExamples({ cDgsExampleSpeculative, cDgsExampleLookup, cDgsExampleServer, cDgsExampleCli })
            .mSetEnv("DGS_ARG_SPEC_DRAFT_N_MAX"));
  vAddOpt(common_arg({ "--spec-draft-n-min" }, "N",
                     fStringFormat("minimum number of draft tokens to use for speculative decoding (default: %d)",
                                   pParams.vSpeculative.vDraft.vNMin),
                     [](common_params & pParams, int value) { pParams.vSpeculative.vDraft.vNMin = value; })
            .mSetSpec()
            .mSetExamples({ cDgsExampleSpeculative, cDgsExampleLookup, cDgsExampleServer, cDgsExampleCli })
            .mSetEnv("DGS_ARG_SPEC_DRAFT_N_MIN"));
  vAddOpt(common_arg({ "--spec-synth-len" }, "L",
                     "target mean synthetic acceptance length, including the target token (benchmarking only)",
                     [](common_params & pParams, const std::string & value) {
                       const std::string cText  = fStringStrip(value);
                       size_t            vPos   = 0;
                       const double      length = std::stod(cText, &vPos);
                       if (vPos != cText.size() || length == -1.0) {
                         throw std::invalid_argument("invalid value");
                       }
                       pParams.vSpeculative.vSynthLen = length;
                     })
            .mSetSpec()
            .mSetExamples({ cDgsExampleServer, cDgsExampleCli })
            .mSetEnv("DGS_ARG_SPEC_SYNTH_LEN"));
  vAddOpt(
    common_arg({ "--spec-synth-rates" }, "P0,P1,...",
               "comma-separated unconditional per-position synthetic acceptance probabilities (benchmarking only)",
               [](common_params & pParams, const std::string & value) {
                 const auto          lValues = fStringSplit<std::string>(value, ',');
                 std::vector<double> lRates;
                 lRates.reserve(lValues.size());
                 for (const auto & vRaw : lValues) {
                   const std::string cText = fStringStrip(vRaw);
                   size_t            vPos  = 0;
                   const double      cRate = std::stod(cText, &vPos);
                   if (vPos != cText.size()) {
                     throw std::invalid_argument("invalid value");
                   }
                   lRates.push_back(cRate);
                 }
                 pParams.vSpeculative.lSynthRates = std::move(lRates);
               })
      .mSetSpec()
      .mSetExamples({ cDgsExampleServer, cDgsExampleCli })
      .mSetEnv("DGS_ARG_SPEC_SYNTH_RATES"));

  vAddOpt(common_arg({ "--spec-draft-p-split" }, "P",
                     fStringFormat("speculative decoding split probability (default: %.2f)",
                                   (double) pParams.vSpeculative.vDraft.vPSplit),
                     [](common_params & pParams, const std::string & value) {
                       pParams.vSpeculative.vDraft.vPSplit = std::stof(value);
                     })
            .mSetSpec()
            .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli })
            .mSetEnv("DGS_ARG_SPEC_DRAFT_P_SPLIT"));
  vAddOpt(common_arg({ "--spec-draft-p-min" }, "P",
                     fStringFormat("minimum speculative decoding probability (greedy) (default: %.2f)",
                                   (double) pParams.vSpeculative.vDraft.vPMin),
                     [](common_params & pParams, const std::string & value) {
                       pParams.vSpeculative.vDraft.vPMin = std::stof(value);
                     })
            .mSetSpec()
            .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli })
            .mSetEnv("DGS_ARG_SPEC_DRAFT_P_MIN"));
  vAddOpt(common_arg({ "--spec-draft-backend-sampling" }, { "--no-spec-draft-backend-sampling" },
                     fStringFormat("offload draft sampling to the backend (default: %s)",
                                   pParams.vSpeculative.vDraft.vBackendSampling ? "enabled" : "disabled"),
                     [](common_params & pParams, bool value) { pParams.vSpeculative.vDraft.vBackendSampling = value; })
            .mSetSpec()
            .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli })
            .mSetEnv("DGS_ARG_SPEC_DRAFT_BACKEND_SAMPLING"));
  vAddOpt(
    common_arg({ "--spec-draft-sampling" }, "{greedy,probabilistic}",
               fStringFormat("how the draft is sampled: greedy takes its argmax, probabilistic samples it and has "
                             "the target verify by rejection sampling (default: %s)",
                             pParams.vSpeculative.vDraft.vProbabilistic ? "probabilistic" : "greedy"),
               [](common_params & pParams, const std::string & value) {
                 if (value == "greedy") {
                   pParams.vSpeculative.vDraft.vProbabilistic = false;
                 } else if (value == "probabilistic") {
                   pParams.vSpeculative.vDraft.vProbabilistic = true;
                 } else {
                   throw std::invalid_argument("invalid value, must be one of: greedy, probabilistic");
                 }
               })
      .mSetSpec()
      .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli })
      .mSetEnv("DGS_ARG_SPEC_DRAFT_SAMPLING"));
  vAddOpt(common_arg({ "--spec-draft-device" }, "<dev1,dev2,..>",
                     "comma-separated list of devices to use for offloading the draft model (none = don't offload, "
                     "default: follows --device)\n"
                     "use --list-devices to see a list of available devices",
                     [](common_params & pParams, const std::string & value) {
                       pParams.vSpeculative.vDraft.lDevices = fParseDeviceList(value);
                     })
            .mSetSpec()
            .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli }));
  GGML_ASSERT(pParams.vSpeculative.vDraft.vNGpuLayers <
              0);  // string_format would need to be extended for a default >= 0
  vAddOpt(
    common_arg(
      { "--spec-draft-ngl" }, "N",
      fStringFormat(
        "max. number of draft model layers to store in VRAM, either an exact number, 'auto', or 'all' (default: %s)",
        pParams.vSpeculative.vDraft.vNGpuLayers == -1 ? "auto" : "all"),
      [](common_params & pParams, const std::string & value) {
        if (value == "auto") {
          pParams.vSpeculative.vDraft.vNGpuLayers = -1;
        } else if (value == "all") {
          pParams.vSpeculative.vDraft.vNGpuLayers = -2;
        } else {
          pParams.vSpeculative.vDraft.vNGpuLayers = std::stoi(value);
        }
        if (!fDgsSupportsGpuOffload()) {
          fprintf(stderr, "warning: no usable GPU found, --gpu-layers-draft option will be ignored\n");
          fprintf(stderr, "warning: one possible reason is that dgs.cpp was compiled without GPU support\n");
          fprintf(stderr, "warning: consult docs/build.md for compilation instructions\n");
        }
      })
      .mSetSpec()
      .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli })
      .mSetEnv("DGS_ARG_N_GPU_LAYERS_DRAFT"));
  vAddOpt(common_arg({ "--spec-draft-model" }, "FNAME", "draft model for speculative decoding (default: unused)",
                     [](common_params & pParams, const std::string & value) {
                       pParams.vSpeculative.vDraft.vMparams.vPath   = value;
                       pParams.vSpeculative.vDraft.vMparams.hf_file = value;  // will be used if --spec-draft-hf is set
                     })
            .mSetSpec()
            .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli, cDgsExampleImatrix })
            .mSetEnv("DGS_ARG_SPEC_DRAFT_MODEL"));
  vAddOpt(common_arg({ "--spec-type" }, fCommonSpeculativeAllTypesStr(),
                     fStringFormat("comma-separated list of types of speculative decoding to use (default: %s)\n",
                                   fCommonSpeculativeTypeNameStr(pParams.vSpeculative.lTypes).c_str()),
                     [](common_params & pParams, const std::string & value) {
                       const auto lTypesStr = fStringSplit<std::string>(value, ',');
                       auto       lTypes    = fCommonSpeculativeTypesFromNames(lTypesStr);
                       pParams.vSpeculative.lTypes.insert(pParams.vSpeculative.lTypes.end(), lTypes.begin(),
                                                          lTypes.end());
                     })
            .mSetSpec()
            .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli })
            .mSetEnv("DGS_ARG_SPEC_TYPE"));
  vAddOpt(
    common_arg({ "--spec-ngram-mod-n-min" }, "N",
               fStringFormat("minimum number of ngram tokens to use for ngram-based speculative decoding (default: %d)",
                             pParams.vSpeculative.vNgramMod.vNMin),
               [](common_params & pParams, int value) {
                 if (value < 0 || value > 1024) {
                   throw std::invalid_argument("ngram n-min must be between 0 and 1024 inclusive");
                 }
                 pParams.vSpeculative.vNgramMod.vNMin = value;
               })
      .mSetSpec()
      .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli }));
  vAddOpt(
    common_arg({ "--spec-ngram-mod-n-max" }, "N",
               fStringFormat("maximum number of ngram tokens to use for ngram-based speculative decoding (default: %d)",
                             pParams.vSpeculative.vNgramMod.vNMax),
               [](common_params & pParams, int value) {
                 if (value < 0 || value > 1024) {
                   throw std::invalid_argument("ngram n-max must be between 0 and 1024 inclusive");
                 }
                 pParams.vSpeculative.vNgramMod.vNMax = value;
               })
      .mSetSpec()
      .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli }));
  vAddOpt(common_arg({ "--spec-ngram-mod-n-match" }, "N",
                     fStringFormat("ngram-mod lookup length (default: %d)", pParams.vSpeculative.vNgramMod.vNMatch),
                     [](common_params & pParams, int value) {
                       if (value < 1 || value > 1024) {
                         throw std::invalid_argument("ngram size N must be between 1 and 1024 inclusive");
                       }
                       pParams.vSpeculative.vNgramMod.vNMatch = value;
                     })
            .mSetSpec()
            .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli }));

  vAddOpt(common_arg(
            { "--spec-ngram-simple-size-n" }, "N",
            fStringFormat("ngram size N for ngram-simple speculative decoding, length of lookup n-gram (default: %d)",
                          pParams.vSpeculative.vNgramSimple.vSizeN),
            [](common_params & pParams, int value) {
              if (value < 1 || value > 1024) {
                throw std::invalid_argument("ngram size N must be between 1 and 1024 inclusive");
              }
              pParams.vSpeculative.vNgramSimple.vSizeN = value;
            })
            .mSetSpec()
            .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli }));
  vAddOpt(
    common_arg({ "--spec-ngram-simple-size-m" }, "N",
               fStringFormat("ngram size M for ngram-simple speculative decoding, length of draft m-gram (default: %d)",
                             pParams.vSpeculative.vNgramSimple.vSizeM),
               [](common_params & pParams, int value) {
                 if (value < 1 || value > 1024) {
                   throw std::invalid_argument("ngram size M must be between 1 and 1024 inclusive");
                 }
                 pParams.vSpeculative.vNgramSimple.vSizeM = value;
               })
      .mSetSpec()
      .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli }));
  vAddOpt(common_arg({ "--spec-ngram-simple-min-hits" }, "N",
                     fStringFormat("minimum hits for ngram-simple speculative decoding (default: %d)",
                                   pParams.vSpeculative.vNgramSimple.vMinHits),
                     [](common_params & pParams, int value) {
                       if (value < 1) {
                         throw std::invalid_argument("ngram min hits must be at least 1");
                       }
                       pParams.vSpeculative.vNgramSimple.vMinHits = value;
                     })
            .mSetSpec()
            .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli }));

  vAddOpt(
    common_arg({ "--spec-ngram-map-k-size-n" }, "N",
               fStringFormat("ngram size N for ngram-map-k speculative decoding, length of lookup n-gram (default: %d)",
                             pParams.vSpeculative.vNgramMapK.vSizeN),
               [](common_params & pParams, int value) {
                 if (value < 1 || value > 1024) {
                   throw std::invalid_argument("ngram size N must be between 1 and 1024 inclusive");
                 }
                 pParams.vSpeculative.vNgramMapK.vSizeN = value;
               })
      .mSetSpec()
      .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli }));
  vAddOpt(
    common_arg({ "--spec-ngram-map-k-size-m" }, "N",
               fStringFormat("ngram size M for ngram-map-k speculative decoding, length of draft m-gram (default: %d)",
                             pParams.vSpeculative.vNgramMapK.vSizeM),
               [](common_params & pParams, int value) {
                 if (value < 1 || value > 1024) {
                   throw std::invalid_argument("ngram size M must be between 1 and 1024 inclusive");
                 }
                 pParams.vSpeculative.vNgramMapK.vSizeM = value;
               })
      .mSetSpec()
      .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli }));
  vAddOpt(common_arg({ "--spec-ngram-map-k-min-hits" }, "N",
                     fStringFormat("minimum hits for ngram-map-k speculative decoding (default: %d)",
                                   pParams.vSpeculative.vNgramMapK.vMinHits),
                     [](common_params & pParams, int value) {
                       if (value < 1) {
                         throw std::invalid_argument("ngram min hits must be at least 1");
                       }
                       pParams.vSpeculative.vNgramMapK.vMinHits = value;
                     })
            .mSetSpec()
            .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli }));

  vAddOpt(common_arg(
            { "--spec-ngram-map-k4v-size-n" }, "N",
            fStringFormat("ngram size N for ngram-map-k4v speculative decoding, length of lookup n-gram (default: %d)",
                          pParams.vSpeculative.vNgramMapK4v.vSizeN),
            [](common_params & pParams, int value) {
              if (value < 1 || value > 1024) {
                throw std::invalid_argument("ngram size N must be between 1 and 1024 inclusive");
              }
              pParams.vSpeculative.vNgramMapK4v.vSizeN = value;
            })
            .mSetSpec()
            .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli }));
  vAddOpt(common_arg(
            { "--spec-ngram-map-k4v-size-m" }, "N",
            fStringFormat("ngram size M for ngram-map-k4v speculative decoding, length of draft m-gram (default: %d)",
                          pParams.vSpeculative.vNgramMapK4v.vSizeM),
            [](common_params & pParams, int value) {
              if (value < 1 || value > 1024) {
                throw std::invalid_argument("ngram size M must be between 1 and 1024 inclusive");
              }
              pParams.vSpeculative.vNgramMapK4v.vSizeM = value;
            })
            .mSetSpec()
            .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli }));
  vAddOpt(common_arg({ "--spec-ngram-map-k4v-min-hits" }, "N",
                     fStringFormat("minimum hits for ngram-map-k4v speculative decoding (default: %d)",
                                   pParams.vSpeculative.vNgramMapK4v.vMinHits),
                     [](common_params & pParams, int value) {
                       if (value < 1) {
                         throw std::invalid_argument("ngram min hits must be at least 1");
                       }
                       pParams.vSpeculative.vNgramMapK4v.vMinHits = value;
                     })
            .mSetSpec()
            .mSetExamples({ cDgsExampleSpeculative, cDgsExampleServer, cDgsExampleCli }));

  //
  // removed params
  //

  vAddOpt(common_arg({ "--draft" }, "N",
                     "the argument has been removed. use --spec-draft-n-max or --spec-ngram-mod-n-max",
                     [](common_params & /*params*/, int /*value*/) {
                       fArgRemoved("use --spec-draft-n-max or --spec-ngram-mod-n-max");
                     })
            .mSetSpec()
            .mSetExamples({ cDgsExampleSpeculative, cDgsExampleLookup, cDgsExampleServer, cDgsExampleCli })
            .mSetEnv("DGS_ARG_DRAFT_MAX"));
  vAddOpt(common_arg({ "--draft-min" }, "N",
                     "the argument has been removed. use --spec-draft-n-min or --spec-ngram-mod-n-min",
                     [](common_params & /*params*/, int /*value*/) {
                       fArgRemoved("use --spec-draft-n-min or --spec-ngram-mod-n-min");
                     })
            .mSetSpec()
            .mSetExamples({ cDgsExampleSpeculative, cDgsExampleLookup, cDgsExampleServer, cDgsExampleCli })
            .mSetEnv("DGS_ARG_DRAFT_MIN"));
  vAddOpt(common_arg(
            { "--spec-ngram-size-n" }, "N",
            "the argument has been removed. use the respective --spec-ngram-*-size-n or --spec-ngram-mod-n-match",
            [](common_params & /*params*/, int /*value*/) { fArgRemoved("use the respective --spec-ngram-*-size-n"); })
            .mSetSpec()
            .mSetExamples({ cDgsExampleServer }));
  vAddOpt(common_arg(
            { "--spec-ngram-size-m" }, "N", "the argument has been removed. use the respective --spec-ngram-*-size-m",
            [](common_params & /*params*/, int /*value*/) { fArgRemoved("use the respective --spec-ngram-*-size-m"); })
            .mSetSpec()
            .mSetExamples({ cDgsExampleServer }));
  vAddOpt(
    common_arg(
      { "--spec-ngram-min-hits" }, "N", "the argument has been removed. use the respective --spec-ngram-*-min-hits",
      [](common_params & /*params*/, int /*value*/) { fArgRemoved("use the respective --spec-ngram-*-min-hits"); })
      .mSetSpec()
      .mSetExamples({ cDgsExampleServer }));

  //
  // TTS params
  //

  //
  // diffusion params
  //

  // presets

  vAddOpt(common_arg({ "--spec-default" }, fStringFormat("enable default speculative decoding config"),
                     [](common_params & pParams) {
                       pParams.vSpeculative.lTypes.push_back(cCommonSpeculativeTypeNgramMod);
                       pParams.vSpeculative.vNgramMod.vNMatch = 24;
                       pParams.vSpeculative.vNgramMod.vNMin   = 48;
                       pParams.vSpeculative.vNgramMod.vNMax   = 64;

                       // TODO: not sure if this is a good config - explore more settings and potentially enable it
                       //params.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
                       //params.speculative.ngram_map_k4v.size_n = 8;
                       //params.speculative.ngram_map_k4v.size_m = 24;
                       //params.speculative.ngram_map_k4v.min_hits = 2;
                     })
            .mSetExamples({ cDgsExampleServer, cDgsExampleCli }));

  return vCtxArg;
}

void fCommonParamsAddPresetOptions(std::vector<common_arg> & pArgs) {
  // arguments below won't be treated as CLI args, only preset options
  pArgs.push_back(common_arg({ "load-on-startup" }, "NAME", "in server router mode, autoload this model on startup",
                             [](common_params &, const std::string &) { /* unused */ })
                    .mSetEnv(COMMON_ARG_PRESET_LOAD_ON_STARTUP)
                    .mSetPresetOnly());

  pArgs.push_back(
    common_arg({ "stop-timeout" }, "SECONDS",
               "in server router mode, force-kill model instance after this many seconds of graceful shutdown",
               [](common_params &, int) { /* unused */ })
      .mSetEnv(COMMON_ARG_PRESET_STOP_TIMEOUT)
      .mSetPresetOnly());

  pArgs.push_back(
    common_arg(
      { "dedup-cache-models" }, "0|1",
      "in server router mode, hide a cached model from the model list when this preset resolves to the same model file",
      [](common_params &, const std::string &) { /* unused */ })
      .mSetEnv(COMMON_ARG_PRESET_DEDUP_CACHE_MODELS)
      .mSetPresetOnly());

  // args.push_back(common_arg(
  //     {"pin"},
  //     "in server router mode, do not unload this model if models_max is exceeded",
  //     [](common_params &) { /* unused */ }
  // ).set_preset_only());
}
