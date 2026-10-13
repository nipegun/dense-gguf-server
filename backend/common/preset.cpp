#include "preset.h"

#include "arg.h"
#include "download.h"
#include "log.h"
#include "peg-parser.h"

#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>

static std::string fRmLeadingDashes(const std::string & str) {
  size_t vPos = 0;
  while (vPos < str.size() && str[vPos] == '-') {
    ++vPos;
  }
  return str.substr(vPos);
}

static std::string fCanonicalTag(const std::string & pTag) {
  static const std::regex cReTag("[-.]([A-Z0-9_]+)$", std::regex::icase);
  std::smatch             vM;
  if (std::regex_search(pTag, vM, cReTag)) {
    std::string vCanon = vM[1].str();
    for (char & vC : vCanon) {
      vC = (char) std::toupper((unsigned char) vC);
    }
    return vCanon;
  }
  std::string vUpper = pTag;
  for (char & vC : vUpper) {
    vC = (char) std::toupper((unsigned char) vC);
  }
  return vUpper;
}

std::vector<std::string> common_preset::mToArgs(const std::string & pBinPath) const {
  std::vector<std::string> lArgs;

  if (!pBinPath.empty()) {
    lArgs.push_back(pBinPath);
  }

  for (const auto & [opt, value] : dOptions) {
    if (opt.vIsPresetOnly) {
      continue;  // skip preset-only options (they are not CLI args)
    }

    // use the last arg as the main arg (i.e. --long-form)
    lArgs.push_back(opt.lArgs.back());

    // handle value(s)
    if (opt.vValueHint == nullptr && opt.vValueHint2 == nullptr) {
      // flag option, no value
      if (common_arg_utils::fIsFalsey(value)) {
        // use negative arg if available
        if (!opt.lArgsNeg.empty()) {
          lArgs.back() = opt.lArgsNeg.back();
        } else {
          // otherwise, skip the flag
          // TODO: maybe throw an error instead?
          lArgs.pop_back();
        }
      }
    }
    if (opt.vValueHint != nullptr) {
      // single value
      lArgs.push_back(value);
    }
    if (opt.vValueHint != nullptr && opt.vValueHint2 != nullptr) {
      throw std::runtime_error(fStringFormat(
        "common_preset::to_args(): option '%s' has two values, which is not supported yet", opt.lArgs.back()));
    }
  }

  return lArgs;
}

std::string common_preset::mToIni() const {
  std::ostringstream vSs;

  vSs << "[" << name << "]\n";
  for (const auto & [opt, value] : dOptions) {
    auto vEspacedValue = value;
    fStringReplaceAll(vEspacedValue, "\n", "\\\n");
    vSs << fRmLeadingDashes(opt.lArgs.back()) << " = ";
    vSs << vEspacedValue << "\n";
  }
  vSs << "\n";

  return vSs.str();
}

void common_preset::mSetOption(const common_preset_context & pCtx,
                               const std::string &           pEnv,
                               const std::string &           value) {
  // try if option exists, update it
  for (auto & [opt, val] : dOptions) {
    if (opt.vEnv && pEnv == opt.vEnv) {
      val = value;
      return;
    }
  }
  // if option does not exist, we need to add it
  if (pCtx.dKeyToOpt.find(pEnv) == pCtx.dKeyToOpt.end()) {
    throw std::runtime_error(fStringFormat("%s: option with env '%s' not found in ctx_params", __func__, pEnv.c_str()));
  }
  dOptions[pCtx.dKeyToOpt.at(pEnv)] = value;
}

void common_preset::mUnsetOption(const std::string & pEnv) {
  for (auto vIt = dOptions.begin(); vIt != dOptions.end();) {
    const common_arg & vOpt = vIt->first;
    if (vOpt.vEnv && pEnv == vOpt.vEnv) {
      vIt = dOptions.erase(vIt);
      return;
    } else {
      ++vIt;
    }
  }
}

bool common_preset::mGetOption(const std::string & pEnv, std::string & value) const {
  for (const auto & [opt, val] : dOptions) {
    if (opt.vEnv && pEnv == opt.vEnv) {
      value = val;
      return true;
    }
  }
  return false;
}

void common_preset::mMerge(const common_preset & pOther) {
  for (const auto & [opt, val] : pOther.dOptions) {
    dOptions[opt] = val;  // overwrite existing options
  }
}

void common_preset::mApplyToParams(common_params & pParams, const std::set<std::string> & pHandledKeys) const {
  for (const auto & [opt, val] : dOptions) {
    if (!pHandledKeys.empty()) {
      if (!opt.vEnv || pHandledKeys.find(opt.vEnv) == pHandledKeys.end()) {
        continue;
      }
    }
    // apply each option to params
    if (opt.vHandlerString) {
      opt.vHandlerString(pParams, val);
    } else if (opt.vHandlerInt) {
      opt.vHandlerInt(pParams, std::stoi(val));
    } else if (opt.vHandlerBool) {
      opt.vHandlerBool(pParams, common_arg_utils::fIsTruthy(val));
    } else if (opt.vHandlerStrStr) {
      // not supported yet
      throw std::runtime_error(fStringFormat("%s: option with two values is not supported yet", __func__));
    } else if (opt.vHandlerVoid) {
      opt.vHandlerVoid(pParams);
    } else {
      GGML_ABORT("unknown handler type");
    }
  }
}

static std::map<std::string, std::map<std::string, std::string>> fParseIniFromFile(
  const std::filesystem::path & pPath) {
  std::map<std::string, std::map<std::string, std::string>> dParsed;

  if (!std::filesystem::exists(pPath)) {
    throw std::runtime_error("preset file does not exist: " + fFsPathToUtf8(pPath));
  }

  std::ifstream vFile(pPath);
  if (!vFile.good()) {
    throw std::runtime_error("failed to open server preset file: " + fFsPathToUtf8(pPath));
  }

  std::string vContents((std::istreambuf_iterator<char>(vFile)), std::istreambuf_iterator<char>());

  static const auto cParser = fBuildPegParser([](auto & pP) {
    // newline ::= "\r\n" / "\n" / "\r"
    auto vNewline = pP.mRule("newline", pP.mLiteral("\r\n") | pP.mLiteral("\n") | pP.mLiteral("\r"));

    // ws ::= [ \t]*
    auto vWs = pP.mRule("ws", pP.mChars("[ \t]", 0, -1));

    // comment ::= [;#] (!newline .)*
    auto vComment = pP.mRule("comment", pP.mChars("[;#]", 1, 1) + pP.mZeroOrMore(pP.mNegate(vNewline) + pP.mAny()));

    // eol ::= ws comment? (newline / EOF)
    auto vEol = pP.mRule("eol", vWs + pP.mOptional(vComment) + (vNewline | pP.end()));

    // ident ::= [a-zA-Z_] [a-zA-Z0-9_.-]*
    auto vIdent = pP.mRule("ident", pP.mChars("[a-zA-Z_]", 1, 1) + pP.mChars("[a-zA-Z0-9_.-]", 0, -1));

    // value ::= (!eol-start .)*
    auto vEolStart = pP.mRule("eol-start", vWs + (pP.mChars("[;#]", 1, 1) | vNewline | pP.end()));
    auto value     = pP.mRule("value", pP.mZeroOrMore(pP.mNegate(vEolStart) + pP.mAny()));

    // header-line ::= "[" ws ident ws "]" eol
    auto vHeaderLine =
      pP.mRule("header-line", "[" + vWs + pP.mTag("section-name", pP.mChars("[^]]")) + vWs + "]" + vEol);

    // kv-line ::= ident ws "=" ws value eol
    auto vKvLine = pP.mRule("kv-line", pP.mTag("key", vIdent) + vWs + "=" + vWs + pP.mTag("value", value) + vEol);

    // comment-line ::= ws comment (newline / EOF)
    auto vCommentLine = pP.mRule("comment-line", vWs + vComment + (vNewline | pP.end()));

    // blank-line ::= ws (newline / EOF)
    auto vBlankLine = pP.mRule("blank-line", vWs + (vNewline | pP.end()));

    // line ::= header-line / kv-line / comment-line / blank-line
    auto vLine = pP.mRule("line", vHeaderLine | vKvLine | vCommentLine | vBlankLine);

    // ini ::= line* EOF
    auto vIni = pP.mRule("ini", pP.mZeroOrMore(vLine) + pP.end());

    return vIni;
  });

  common_peg_parse_context vCtx(vContents);
  const auto               cResult = cParser.mParse(vCtx);
  if (!cResult.mSuccess()) {
    throw std::runtime_error("failed to parse server config file: " + fFsPathToUtf8(pPath));
  }

  std::string vCurrentSection = cCommonPresetDefaultName;
  std::string vCurrentKey;

  vCtx.vAst.mVisit(cResult, [&](const auto & node) {
    if (node.vTag == "section-name") {
      const std::string cSection = std::string(node.vText);
      vCurrentSection            = cSection;
      dParsed[vCurrentSection]   = {};
    } else if (node.vTag == "key") {
      const std::string cKey = std::string(node.vText);
      vCurrentKey            = cKey;
    } else if (node.vTag == "value" && !vCurrentKey.empty() && !vCurrentSection.empty()) {
      dParsed[vCurrentSection][vCurrentKey] = std::string(node.vText);
      vCurrentKey.clear();
    }
  });

  return dParsed;
}

static std::map<std::string, common_arg> fGetMapKeyOpt(common_params_context & pCtxParams) {
  std::map<std::string, common_arg> dMapping;
  for (const auto & vOpt : pCtxParams.lOptions) {
    for (const auto & vEnv : vOpt.mGetEnv()) {
      dMapping[vEnv] = vOpt;
    }
    for (const auto & vArg : vOpt.mGetArgs()) {
      dMapping[fRmLeadingDashes(vArg)] = vOpt;
    }
  }
  return dMapping;
}

static bool fIsBoolArg(const common_arg & pArg) {
  return !pArg.lArgsNeg.empty();
}

static std::string fParseBoolArg(const common_arg & pArg, const std::string & pKey, const std::string & value) {
  // if this is a negated arg, we need to reverse the value
  for (const auto & vNegArg : pArg.lArgsNeg) {
    if (fRmLeadingDashes(vNegArg) == pKey) {
      return common_arg_utils::fIsTruthy(value) ? "false" : "true";
    }
  }
  // otherwise, not negated
  return value;
}

common_preset_context::common_preset_context(dgs_example pEx) :
  vCtxParams(fCommonParamsParserInit(vDefaultParams, pEx)) {
  fCommonParamsAddPresetOptions(vCtxParams.lOptions);
  dKeyToOpt = fGetMapKeyOpt(vCtxParams);
}

common_presets common_preset_context::mLoadFromIni(const std::filesystem::path & pPath, common_preset & pGlobal) const {
  common_presets dOut;
  auto           dIniData = fParseIniFromFile(pPath);

  for (auto dSection : dIniData) {
    common_preset vPreset;
    std::string   vSectionName = dSection.first.empty() ? std::string(cCommonPresetDefaultName) : dSection.first;
    if (vSectionName != "*" && vSectionName != cCommonPresetDefaultName) {
      auto vColonIdx = vSectionName.rfind(':');
      if (vColonIdx != std::string::npos) {
        std::string vTag      = vSectionName.substr(vColonIdx + 1);
        std::string vCanonTag = fCanonicalTag(vTag);
        if (vCanonTag != vTag) {
          vSectionName = vSectionName.substr(0, vColonIdx + 1) + vCanonTag;
        }
      }
    }
    vPreset.name = vSectionName;
    LOG_DBG("loading preset: %s\n", vPreset.name.c_str());
    for (const auto & [key, value] : dSection.second) {
      if (key == "version") {
        // skip version key (reserved for future use)
        continue;
      }

      LOG_DBG("option: %s = %s\n", key.c_str(), value.c_str());
      if (vFilterAllowedKeys && sAllowedKeys.find(key) == sAllowedKeys.end()) {
        throw std::runtime_error(fStringFormat("option '%s' is not allowed in remote presets", key.c_str()));
      }
      if (dKeyToOpt.find(key) != dKeyToOpt.end()) {
        const auto & vOpt = dKeyToOpt.at(key);
        if (fIsBoolArg(vOpt)) {
          vPreset.dOptions[vOpt] = fParseBoolArg(vOpt, key, value);
        } else {
          vPreset.dOptions[vOpt] = value;
        }
        LOG_DBG("accepted option: %s = %s\n", key.c_str(), vPreset.dOptions[vOpt].c_str());
      } else if (vIgnoreUnknownKeys) {
        LOG_WRN("ignoring option '%s' from %s: not supported by this program\n", key.c_str(),
                fFsPathToUtf8(pPath).c_str());
      } else {
        throw std::runtime_error(
          fStringFormat("option '%s' not recognized in preset '%s'", key.c_str(), vPreset.name.c_str()));
      }
    }

    if (vPreset.name == cCommonPresetDefaultName && vPreset.dOptions.empty()) {
      continue;
    }

    if (vPreset.name == "*") {
      // handle global preset
      pGlobal = vPreset;
    } else {
      dOut[vPreset.name] = vPreset;
    }
  }

  return dOut;
}

common_presets common_preset_context::mLoadFromCache() const {
  common_presets dOut;

  auto lCachedModels = fCommonListCachedModels();
  for (const auto & vModel : lCachedModels) {
    common_preset vPreset;
    vPreset.name = vModel.to_string();
    vPreset.mSetOption(*this, "DGS_ARG_HF_REPO", vModel.to_string());
    dOut[vPreset.name] = vPreset;
  }

  return dOut;
}

struct local_model {
  std::string name;
  std::string vPath;
  std::string vPathMmproj;
  std::string vPathDraft;
};

// TODO @ngxson: handle "eagle3-" when it's supported by common_speculative_types_from_gguf()
static const char * aDraftPrefixes[] = { "mtp-", "dspark-", "dflash-" };

static bool fIsMmprojFile(const std::string & pFname) {
  return pFname.find("mmproj") != std::string::npos;
}

static bool fIsDraftFile(const std::string & pFname) {
  for (const auto & vPrefix : aDraftPrefixes) {
    if (pFname.rfind(vPrefix, 0) == 0) {
      return true;
    }
  }
  return false;
}

common_presets common_preset_context::mLoadFromModelsDir(const std::string & pModelsDir) const {
  const std::filesystem::path cDir = std::filesystem::u8path(pModelsDir);
  if (!std::filesystem::exists(cDir) || !std::filesystem::is_directory(cDir)) {
    throw std::runtime_error(fStringFormat("error: '%s' does not exist or is not a directory\n", pModelsDir.c_str()));
  }

  std::vector<local_model> lModels;
  auto vScanSubdir = [&lModels](const std::filesystem::path & pSubdirPath, const std::string & name) {
    std::filesystem::path vModelFile;
    std::filesystem::path vFirstShardFile;
    std::filesystem::path vMmprojFile;
    std::filesystem::path vDraftFile;
    std::error_code       vEc;
    for (const auto & vEntry : std::filesystem::directory_iterator(pSubdirPath)) {
      if (!vEntry.is_regular_file(vEc)) {
        continue;
      }
      const std::string cFname = fFsPathToUtf8(vEntry.path().filename());
      if (!fStringEndsWith(cFname, ".gguf")) {
        continue;
      }
      if (fIsMmprojFile(cFname)) {
        vMmprojFile = vEntry.path();
      } else if (fIsDraftFile(cFname)) {
        if (vDraftFile.empty()) {
          vDraftFile = vEntry.path();  // first sidecar found wins
        }
      } else if (cFname.find("-00001-of-") != std::string::npos) {
        vFirstShardFile = vEntry.path();
      } else {
        vModelFile = vEntry.path();
      }
    }
    // single file model
    const std::filesystem::path & vPath = vFirstShardFile.empty() ? vModelFile : vFirstShardFile;
    if (!vPath.empty()) {
      lModels.push_back({
        /* name        */ name,
        /* path        */ fFsPathToUtf8(vPath),
        /* path_mmproj */ fFsPathToUtf8(vMmprojFile),  // can be empty
        /* path_draft  */ fFsPathToUtf8(vDraftFile)    // can be empty
      });
    }
  };

  for (const auto & vEntry : std::filesystem::directory_iterator(cDir)) {
    std::error_code vEc;
    if (vEntry.is_directory(vEc)) {
      vScanSubdir(vEntry.path(), fFsPathToUtf8(vEntry.path().filename()));
      continue;
    }
    if (!vEntry.is_regular_file(vEc)) {
      continue;
    }
    const std::string cFname = fFsPathToUtf8(vEntry.path().filename());
    if (!fStringEndsWith(cFname, ".gguf")) {
      continue;
    }
    if (fIsMmprojFile(cFname) || fIsDraftFile(cFname)) {
      continue;  // companion file, cannot be loaded as a model on its own
    }
    // single file model
    std::string name = cFname;
    fStringReplaceAll(name, ".gguf", "");
    lModels.push_back({ /* name        */ name,
                        /* path        */ fFsPathToUtf8(vEntry.path()),
                        /* path_mmproj */ "",
                        /* path_draft  */ "" });
  }

  // convert local models to presets
  common_presets dOut;
  for (const auto & vModel : lModels) {
    common_preset vPreset;
    vPreset.name = vModel.name;
    vPreset.mSetOption(*this, "DGS_ARG_MODEL", vModel.vPath);
    if (!vModel.vPathMmproj.empty()) {
      vPreset.mSetOption(*this, "DGS_ARG_MMPROJ", vModel.vPathMmproj);
    }
    if (!vModel.vPathDraft.empty()) {
      vPreset.mSetOption(*this, "DGS_ARG_SPEC_DRAFT_MODEL", vModel.vPathDraft);
    }
    dOut[vPreset.name] = vPreset;
  }

  return dOut;
}

common_preset common_preset_context::mLoadFromArgs(int pArgc, char ** pArgv) const {
  common_preset vPreset;
  vPreset.name = cCommonPresetDefaultName;

  bool vOk = fCommonParamsToMap(pArgc, pArgv, vCtxParams.vEx, vPreset.dOptions);
  if (!vOk) {
    throw std::runtime_error("failed to parse CLI arguments into preset");
  }

  return vPreset;
}

common_presets common_preset_context::mCascade(const common_presets & pBase, const common_presets & pAdded) const {
  common_presets dOut = pBase;  // copy
  for (const auto & [name, preset_added] : pAdded) {
    if (dOut.find(name) != dOut.end()) {
      // if exists, merge
      common_preset & vTarget = dOut[name];
      vTarget.mMerge(preset_added);
    } else {
      // otherwise, add directly
      dOut[name] = preset_added;
    }
  }
  return dOut;
}

common_presets common_preset_context::mCascade(const common_preset & pBase, const common_presets & pPresets) const {
  common_presets dOut;
  for (const auto & [name, preset] : pPresets) {
    common_preset vTmp = pBase;  // copy
    vTmp.name          = name;
    vTmp.mMerge(preset);
    dOut[name] = std::move(vTmp);
  }
  return dOut;
}
