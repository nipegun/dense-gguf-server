#include "value.h"

#include "runtime.h"
#include "unicode.h"

// for converting from JSON to jinja values
#include "json.h"

#include <algorithm>
#include <cctype>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#define FILENAME "jinja-value"

namespace jinja {

// func_args method implementations

value func_args::mGetKwarg(const std::string & pKey, value pDefaultVal) const {
  for (const auto & vArg : lArgs) {
    if (is_val<value_kwarg>(vArg)) {
      auto * vKwarg = fCastVal<value_kwarg>(vArg);
      if (vKwarg->vKey == pKey) {
        return vKwarg->vVal;
      }
    }
  }
  return pDefaultVal;
}

value func_args::mGetKwargOrPos(const std::string & pKey, size_t pPos) const {
  value vVal = mGetKwarg(pKey, mk_val<value_undefined>());

  if (vVal->mIsUndefined() && pPos < count() && !is_val<value_kwarg>(lArgs[pPos])) {
    return lArgs[pPos];
  }

  return vVal;
}

value func_args::mGetPos(size_t pPos) const {
  if (count() > pPos) {
    return lArgs[pPos];
  }
  throw raised_exception("Function '" + vFuncName + "' expected at least " + std::to_string(pPos + 1) +
                         " arguments, got " + std::to_string(count()));
}

value func_args::mGetPos(size_t pPos, value pDefaultVal) const {
  if (count() > pPos) {
    return lArgs[pPos];
  }
  return pDefaultVal;
}

void func_args::push_back(const value & pVal) {
  lArgs.push_back(pVal);
}

void func_args::mPushFront(const value & pVal) {
  lArgs.insert(lArgs.begin(), pVal);
}

const std::vector<value> & func_args::mGetArgs() const {
  return lArgs;
}

/**
 * Function that mimics Python's array slicing.
 */
template <typename T> static T fSlice(const T & pArray, int64_t pStart, int64_t pStop, int64_t pStep = 1) {
  int64_t vLen       = static_cast<int64_t>(pArray.size());
  int64_t vDirection = (pStep > 0) ? 1 : ((pStep < 0) ? -1 : 0);
  int64_t vStartVal  = 0;
  int64_t vStopVal   = 0;
  if (vDirection >= 0) {
    vStartVal = pStart;
    if (vStartVal < 0) {
      vStartVal = std::max(vLen + vStartVal, (int64_t) 0);
    } else {
      vStartVal = std::min(vStartVal, vLen);
    }

    vStopVal = pStop;
    if (vStopVal < 0) {
      vStopVal = std::max(vLen + vStopVal, (int64_t) 0);
    } else {
      vStopVal = std::min(vStopVal, vLen);
    }
  } else {
    vStartVal = pStart;
    if (vStartVal < 0) {
      vStartVal = std::max(vLen + vStartVal, (int64_t) 0);
    } else {
      vStartVal = std::min(vStartVal, vLen - 1);
    }

    vStopVal = pStop;
    if (vStopVal < -1) {
      vStopVal = std::max(vLen + vStopVal, (int64_t) -1);
    } else {
      vStopVal = std::min(vStopVal, vLen - 1);
    }
  }
  T vResult;
  if (vDirection == 0) {
    return vResult;
  }
  for (int64_t vI = vStartVal; vDirection * vI < vDirection * vStopVal; vI += pStep) {
    if (vI >= 0 && vI < vLen) {
      vResult.push_back(pArray[static_cast<size_t>(vI)]);
    }
  }
  return vResult;
}

template <typename T> static value empty_value_fn(const func_args &) {
  if constexpr (std::is_same_v<T, value_int>) {
    return mk_val<T>(0);
  } else if constexpr (std::is_same_v<T, value_float>) {
    return mk_val<T>(0.0);
  } else if constexpr (std::is_same_v<T, value_bool>) {
    return mk_val<T>(false);
  } else {
    return mk_val<T>();
  }
}

template <typename T> static value test_type_fn(const func_args & pArgs) {
  pArgs.mEnsureCount(1);
  bool vIsType = is_val<T>(pArgs.mGetPos(0));
  JJ_DEBUG("test_type_fn: type=%s result=%d", typeid(T).name(), vIsType ? 1 : 0);
  return mk_val<value_bool>(vIsType);
}

template <typename T, typename U> static value test_type_fn(const func_args & pArgs) {
  pArgs.mEnsureCount(1);
  bool vIsType = is_val<T>(pArgs.mGetPos(0)) || is_val<U>(pArgs.mGetPos(0));
  JJ_DEBUG("test_type_fn: type=%s or %s result=%d", typeid(T).name(), typeid(U).name(), vIsType ? 1 : 0);
  return mk_val<value_bool>(vIsType);
}

template <typename T, typename U, typename V> static value test_type_fn(const func_args & pArgs) {
  pArgs.mEnsureCount(1);
  bool vIsType = is_val<T>(pArgs.mGetPos(0)) || is_val<U>(pArgs.mGetPos(0)) || is_val<V>(pArgs.mGetPos(0));
  JJ_DEBUG("test_type_fn: type=%s, %s or %s result=%d", typeid(T).name(), typeid(U).name(), typeid(V).name(),
           vIsType ? 1 : 0);
  return mk_val<value_bool>(vIsType);
}

template <typename T, typename U, typename V, typename W> static value test_type_fn(const func_args & pArgs) {
  pArgs.mEnsureCount(1);
  bool vIsType = is_val<T>(pArgs.mGetPos(0)) || is_val<U>(pArgs.mGetPos(0)) || is_val<V>(pArgs.mGetPos(0)) ||
                 is_val<W>(pArgs.mGetPos(0));
  JJ_DEBUG("test_type_fn: type=%s, %s, %s or %s result=%d", typeid(T).name(), typeid(U).name(), typeid(V).name(),
           typeid(W).name(), vIsType ? 1 : 0);
  return mk_val<value_bool>(vIsType);
}

template <value_compare_op op> static value test_compare_fn(const func_args & pArgs) {
  pArgs.mEnsureCount(2, 2);
  return mk_val<value_bool>(value_compare(pArgs.mGetPos(0), pArgs.mGetPos(1), op));
}

static void fAppendCodepointAsAsciiJsonEscape(std::string & pOut, uint32_t pCodepoint) {
  auto vAppendU16 = [&pOut](uint32_t value) {
    char aBuf[8];
    snprintf(aBuf, sizeof(aBuf), "\\u%04x", static_cast<unsigned int>(value));
    pOut += aBuf;
  };

  if (pCodepoint <= 0xFFFF) {
    vAppendU16(pCodepoint);
    return;
  }

  pCodepoint -= 0x10000;
  vAppendU16(0xD800 + ((pCodepoint >> 10) & 0x3FF));
  vAppendU16(0xDC00 + (pCodepoint & 0x3FF));
}

static std::string fJsonEnsureAsciiPreservingFormat(const std::string & pJsonStr) {
  std::string vOutput;
  vOutput.reserve(pJsonStr.size());

  bool vInString = false;
  bool vEscaped  = false;

  for (size_t vPos = 0; vPos < pJsonStr.size();) {
    const char cCh = pJsonStr[vPos];
    if (!vInString) {
      vOutput.push_back(cCh);
      if (cCh == '"') {
        vInString = true;
      }
      ++vPos;
      continue;
    }

    if (vEscaped) {
      vOutput.push_back(cCh);
      vEscaped = false;
      ++vPos;
      continue;
    }

    if (cCh == '\\') {
      vOutput.push_back(cCh);
      vEscaped = true;
      ++vPos;
      continue;
    }

    if (cCh == '"') {
      vOutput.push_back(cCh);
      vInString = false;
      ++vPos;
      continue;
    }

    const unsigned char cUch = static_cast<unsigned char>(cCh);
    if (cUch < 0x80) {
      vOutput.push_back(cCh);
      ++vPos;
      continue;
    }

    auto vParsed = fCommonParseUtf8Codepoint(pJsonStr, vPos);
    if (vParsed.vStatus != utf8_parse_result::cSuccess) {
      vOutput += "\\ufffd";
      ++vPos;
      continue;
    }

    fAppendCodepointAsAsciiJsonEscape(vOutput, vParsed.vCodepoint);
    vPos += vParsed.vBytesConsumed;
  }

  return vOutput;
}

static value fTojson(const func_args & pArgs) {
  pArgs.mEnsureCount(1, 5);
  value vValAscii      = pArgs.mGetKwargOrPos("ensure_ascii", 1);
  value vValIndent     = pArgs.mGetKwargOrPos("indent", 2);
  value vValSeparators = pArgs.mGetKwargOrPos("separators", 3);
  value vValSort       = pArgs.mGetKwargOrPos("sort_keys", 4);
  int   vIndent        = -1;
  if (pArgs.vCtx.vIsGetStats) {
    // mark as used (recursively) for stats
    auto vValInput = pArgs.mGetPos(0);
    value_t::stats_t::mMarkUsed(const_cast<value &>(vValInput), true);
  }
  if (is_val<value_int>(vValIndent)) {
    vIndent = static_cast<int>(vValIndent->mAsInt());
  }
  if (vValSort->mAsBool()) {  // undefined == false
    throw not_implemented_exception("tojson sort_keys=true not implemented");
  }
  const bool  cEnsureAscii = vValAscii->mAsBool();  // undefined == false
  auto        lSeparators  = (is_val<value_array>(vValSeparators) ? vValSeparators : mk_val<value_array>())->mAsArray();
  std::string vItemSep     = lSeparators.size() > 0 ? lSeparators[0]->mAsString().str() : (vIndent < 0 ? ", " : ",");
  std::string vKeySep      = lSeparators.size() > 1 ? lSeparators[1]->mAsString().str() : ": ";
  std::string vJsonStr     = fValueToJson(pArgs.mGetPos(0), vIndent, vItemSep, vKeySep);
  if (cEnsureAscii) {
    vJsonStr = fJsonEnsureAsciiPreservingFormat(vJsonStr);
  }
  return mk_val<value_string>(vJsonStr);
}

static value & fGetAttribute(const value & pVal, const value & pAttr, value & pDefaultVal) {
  if (!pAttr->mIsUndefined()) {
    if (is_val<value_array>(pVal)) {
      value vIdx = pAttr;

      if (is_val<value_string>(pAttr)) {
        const std::string cS = pAttr->mAsString().str();
        if (!cS.empty() && std::all_of(cS.begin(), cS.end(), [](unsigned char pC) { return std::isdigit(pC); })) {
          try {
            vIdx = mk_val<value_int>(std::stoll(cS));
          } catch (...) {
            vIdx = mk_val<value_undefined>();
          }
        }
      }

      return pVal->at(vIdx, pDefaultVal);
    } else if (is_val<value_object>(pVal)) {
      return pVal->at(pAttr, pDefaultVal);
    }
  }
  return pDefaultVal;
}

template <bool is_reject> static value selectattr(const func_args & pArgs) {
  pArgs.mEnsureCount(2, 4);
  pArgs.mEnsureVals<value_array, value_string, value_string, value_string>(true, true, false, false);

  auto  lArr        = pArgs.mGetPos(0)->mAsArray();
  auto  vAttribute  = pArgs.mGetPos(1);
  auto  vOut        = mk_val<value_array>();
  value vValDefault = mk_val<value_undefined>();

  if (pArgs.count() == 2) {
    // example: array | selectattr("active")
    for (const auto & vItem : lArr) {
      value vAttrVal    = fGetAttribute(vItem, vAttribute, vValDefault);
      bool  vIsSelected = vAttrVal->mAsBool();
      if constexpr (is_reject) {
        vIsSelected = !vIsSelected;
      }
      if (vIsSelected) {
        vOut->push_back(vItem);
      }
    }
    return vOut;

  } else if (pArgs.count() == 3) {
    // example: array | selectattr("equalto", "text")
    // translated to: test_is_equalto(item, "text")
    std::string vTestName = pArgs.mGetPos(1)->mAsString().str();
    value       vTestVal  = pArgs.mGetPos(2);
    auto &      dBuiltins = fGlobalBuiltins();
    auto        vIt       = dBuiltins.find("test_is_" + vTestName);
    if (vIt == dBuiltins.end()) {
      throw raised_exception("selectattr: unknown test '" + vTestName + "'");
    }
    auto vTestFn = vIt->second;
    for (const auto & vItem : lArr) {
      func_args vTestArgs(pArgs.vCtx);
      vTestArgs.push_back(vItem);     // current object
      vTestArgs.push_back(vTestVal);  // extra argument
      value vTestResult = vTestFn(vTestArgs);
      bool  vIsSelected = vTestResult->mAsBool();
      if constexpr (is_reject) {
        vIsSelected = !vIsSelected;
      }
      if (vIsSelected) {
        vOut->push_back(vItem);
      }
    }
    return vOut;

  } else if (pArgs.count() == 4) {
    // example: array | selectattr("status", "equalto", "active")
    // translated to: test_is_equalto(item.status, "active")
    std::string vTestName = pArgs.mGetPos(2)->mAsString().str();
    auto        vExtraArg = pArgs.mGetPos(3);
    auto &      dBuiltins = fGlobalBuiltins();
    auto        vIt       = dBuiltins.find("test_is_" + vTestName);
    if (vIt == dBuiltins.end()) {
      throw raised_exception("selectattr: unknown test '" + vTestName + "'");
    }
    auto vTestFn = vIt->second;
    for (const auto & vItem : lArr) {
      value     vAttrVal = fGetAttribute(vItem, vAttribute, vValDefault);
      func_args vTestArgs(pArgs.vCtx);
      vTestArgs.push_back(vAttrVal);   // attribute value
      vTestArgs.push_back(vExtraArg);  // extra argument
      value vTestResult = vTestFn(vTestArgs);
      bool  vIsSelected = vTestResult->mAsBool();
      if constexpr (is_reject) {
        vIsSelected = !vIsSelected;
      }
      if (vIsSelected) {
        vOut->push_back(vItem);
      }
    }
    return vOut;
  } else {
    throw raised_exception("selectattr: invalid number of arguments");
  }

  return vOut;
}

static value fDefaultValue(const func_args & pArgs) {
  pArgs.mEnsureCount(2, 3);
  value vValCheck  = pArgs.mGetKwargOrPos("boolean", 2);
  bool  vCheckBool = vValCheck->mAsBool();  // undefined == false
  bool  vNoValue =
    vCheckBool ? (!pArgs.mGetPos(0)->mAsBool()) : (pArgs.mGetPos(0)->mIsUndefined() || pArgs.mGetPos(0)->mIsNone());
  return vNoValue ? pArgs.mGetPos(1) : pArgs.mGetPos(0);
}

static value fToobject(const func_args & pArgs) {
  auto  vOut       = mk_val<value_object>();
  value vIter      = pArgs.mGetPos(0, mk_val<value_undefined>());
  bool  vIterFirst = false;
  if (is_val<value_array>(vIter)) {
    vIterFirst = true;
    for (const auto & vIt : vIter->mAsArray()) {
      if (is_val<value_array>(vIt) && vIt->mAsArray().size() == 2) {
        auto lTuple = vIt->mAsArray();
        auto vKey   = lTuple[0];
        auto vVal   = lTuple[1];
        JJ_DEBUG("namespace/dict: adding key '%s'", vKey->mAsString().str().c_str());
        vOut->insert(vKey, vVal);
      } else {
        throw raised_exception("namespace/dict() iterable argument must consist of tuples, not " + vIt->type());
      }
    }
  } else if (is_val<value_object>(vIter)) {
    vIterFirst = true;
    for (const auto & vPair : vIter->mAsOrderedObject()) {
      JJ_DEBUG("namespace/dict: adding key '%s'", vPair.first->mAsString().str().c_str());
      vOut->insert(vPair.first, vPair.second);
    }
  }
  for (const auto & vArg : pArgs.mGetArgs()) {
    if (is_val<value_kwarg>(vArg)) {
      auto vKwarg = fCastVal<value_kwarg>(vArg);
      JJ_DEBUG("namespace/dict: adding key '%s'", vKwarg->vKey.c_str());
      vOut->insert(vKwarg->vKey, vKwarg->vVal);
    } else if (!vIterFirst) {
      throw raised_exception("namespace/dict() arguments must be kwargs, dict and/or iterable of tuples, not " +
                             vArg->type());
    }
    vIterFirst = false;
  }
  return vOut;
}

const func_builtins & fGlobalBuiltins() {
  static const func_builtins dBuiltins = {
    { "raise_exception",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_string>();
        std::string vMsg = pArgs.mGetPos(0)->mAsString().str();
        throw raised_exception("Jinja Exception: " + vMsg);
      } },
    { "dict", fToobject },
    { "namespace", fToobject },
    { "strftime_now",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_string>();
        std::string format = pArgs.mGetPos(0)->mAsString().str();
        // get current time
        // TODO: make sure this is the same behavior as Python's strftime
        char        aBuf[100];
        if (std::strftime(aBuf, sizeof(aBuf), format.c_str(), std::localtime(&pArgs.vCtx.vCurrentTime))) {
          return mk_val<value_string>(std::string(aBuf));
        } else {
          throw raised_exception("strftime_now: failed to format time");
        }
      } },
    { "range",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureCount(1, 3);
        pArgs.mEnsureVals<value_int, value_int, value_int>(true, false, false);

        auto vArg0 = pArgs.mGetPos(0);
        auto vArg1 = pArgs.mGetPos(1, mk_val<value_undefined>());
        auto vArg2 = pArgs.mGetPos(2, mk_val<value_undefined>());

        int64_t vStart, vStop, vStep;
        if (pArgs.count() == 1) {
          vStart = 0;
          vStop  = vArg0->mAsInt();
          vStep  = 1;
        } else if (pArgs.count() == 2) {
          vStart = vArg0->mAsInt();
          vStop  = vArg1->mAsInt();
          vStep  = 1;
        } else {
          vStart = vArg0->mAsInt();
          vStop  = vArg1->mAsInt();
          vStep  = vArg2->mAsInt();
        }

        auto vOut = mk_val<value_array>();
        if (vStep == 0) {
          throw raised_exception("range() step argument must not be zero");
        }
        if (vStep > 0) {
          for (int64_t vI = vStart; vI < vStop; vI += vStep) {
            vOut->push_back(mk_val<value_int>(vI));
          }
        } else {
          for (int64_t vI = vStart; vI > vStop; vI += vStep) {
            vOut->push_back(mk_val<value_int>(vI));
          }
        }
        return vOut;
      } },
    { "tojson", fTojson },

    // tests
    { "test_is_boolean", test_type_fn<value_bool> },
    { "test_is_callable", test_type_fn<value_func> },
    { "test_is_odd",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_int>();
        int64_t vVal = pArgs.mGetPos(0)->mAsInt();
        return mk_val<value_bool>(vVal % 2 != 0);
      } },
    { "test_is_even",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_int>();
        int64_t vVal = pArgs.mGetPos(0)->mAsInt();
        return mk_val<value_bool>(vVal % 2 == 0);
      } },
    { "test_is_false",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureCount(1);
        bool vVal = is_val<value_bool>(pArgs.mGetPos(0)) && !pArgs.mGetPos(0)->mAsBool();
        return mk_val<value_bool>(vVal);
      } },
    { "test_is_true",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureCount(1);
        bool vVal = is_val<value_bool>(pArgs.mGetPos(0)) && pArgs.mGetPos(0)->mAsBool();
        return mk_val<value_bool>(vVal);
      } },
    { "test_is_divisibleby",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_int, value_int>();
        bool vRes = pArgs.mGetPos(0)->vValInt % pArgs.mGetPos(1)->vValInt == 0;
        return mk_val<value_bool>(vRes);
      } },
    { "test_is_string", test_type_fn<value_string> },
    { "test_is_integer", test_type_fn<value_int> },
    { "test_is_float", test_type_fn<value_float> },
    { "test_is_number", test_type_fn<value_int, value_float> },
    { "test_is_iterable", test_type_fn<value_object, value_array, value_string, value_undefined> },
    { "test_is_sequence", test_type_fn<value_object, value_array, value_string, value_undefined> },
    { "test_is_mapping", test_type_fn<value_object> },
    { "test_is_lower",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_string>();
        return mk_val<value_bool>(pArgs.mGetPos(0)->vValStr.mIsLowercase());
      } },
    { "test_is_upper",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_string>();
        return mk_val<value_bool>(pArgs.mGetPos(0)->vValStr.mIsUppercase());
      } },
    { "test_is_none", test_type_fn<value_none> },
    { "test_is_defined",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureCount(1);
        bool vRes = !pArgs.mGetPos(0)->mIsUndefined();
        JJ_DEBUG("test_is_defined: result=%d", vRes ? 1 : 0);
        return mk_val<value_bool>(vRes);
      } },
    { "test_is_undefined", test_type_fn<value_undefined> },
    { "test_is_eq", test_compare_fn<value_compare_op::cEq> },
    { "test_is_equalto", test_compare_fn<value_compare_op::cEq> },
    { "test_is_ge", test_compare_fn<value_compare_op::cGe> },
    { "test_is_gt", test_compare_fn<value_compare_op::cGt> },
    { "test_is_greaterthan", test_compare_fn<value_compare_op::cGt> },
    { "test_is_lt", test_compare_fn<value_compare_op::cLt> },
    { "test_is_lessthan", test_compare_fn<value_compare_op::cLt> },
    { "test_is_ne", test_compare_fn<value_compare_op::cNe> },
    { "test_is_in",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureCount(2);
        auto vNeedle   = pArgs.mGetPos(0);
        auto vHaystack = pArgs.mGetPos(1);
        if (is_val<value_undefined>(vHaystack)) {
          return mk_val<value_bool>(false);
        }
        if (is_val<value_array>(vHaystack)) {
          for (const auto & vItem : vHaystack->mAsArray()) {
            if (*vNeedle == *vItem) {
              return mk_val<value_bool>(true);
            }
          }
          return mk_val<value_bool>(false);
        }
        if (is_val<value_string>(vHaystack)) {
          if (!is_val<value_string>(vNeedle)) {
            throw raised_exception("'in' test expects args[1] as string when args[0] is string, got args[1] as " +
                                   vNeedle->type());
          }
          return mk_val<value_bool>(vHaystack->mAsString().str().find(vNeedle->mAsString().str()) != std::string::npos);
        }
        if (is_val<value_object>(vHaystack)) {
          return mk_val<value_bool>(vHaystack->mHasKey(vNeedle));
        }
        throw raised_exception("'in' test expects iterable as first argument, got " + vHaystack->type());
      } },
    { "test_is_test",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_string>();
        auto &      dBuiltins = fGlobalBuiltins();
        std::string vTestName = pArgs.mGetPos(0)->vValStr.str();
        auto        vIt       = dBuiltins.find("test_is_" + vTestName);
        bool        vRes      = vIt != dBuiltins.end();
        return mk_val<value_bool>(vRes);
      } },
    { "test_is_sameas",
     [](const func_args & pArgs) -> value {
        // Check if an object points to the same memory address as another object
        pArgs.mEnsureCount(2);
        auto vA   = pArgs.mGetPos(0);
        auto vB   = pArgs.mGetPos(1);
        bool vRes = false;
        if (!is_val<value_undefined>(vA) && !is_val<value_undefined>(vB)) {
          if (is_val<value_none>(vA) && is_val<value_none>(vB)) {
            vRes = true;
          } else if (is_val<value_bool>(vA) && is_val<value_bool>(vB)) {
            if (vA->mAsBool() == vB->mAsBool()) {
              vRes = true;
            }
          } else if (is_val<value_int>(vA) && is_val<value_int>(vB)) {
            const int64_t cX = vA->mAsInt();
            // Allow comparison within small-int cache range
            if (cX >= -5 && cX <= 256 && cX == vB->mAsInt()) {
              vRes = true;
            }
          } else if (vA == vB) {
            vRes = true;
          }
        }
        return mk_val<value_bool>(vRes);
      } },
    { "test_is_escaped",
     [](const func_args & pArgs) -> value {
        (void) pArgs;
        throw not_implemented_exception("escaped test not implemented");
      } },
    { "test_is_filter",
     [](const func_args & pArgs) -> value {
        (void) pArgs;
        throw not_implemented_exception("filter test not implemented");
      } },
  };
  return dBuiltins;
}

const func_builtins & value_int_t::mGetBuiltins() const {
  static const func_builtins dBuiltins = {
    { "default", fDefaultValue },
    { "abs",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_int>();
        int64_t vVal = pArgs.mGetPos(0)->mAsInt();
        return mk_val<value_int>(vVal < 0 ? -vVal : vVal);
      }                        },
    { "int",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_int>();
        return mk_val<value_int>(pArgs.mGetPos(0)->mAsInt());
      }                        },
    { "float",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_int>();
        double vVal = static_cast<double>(pArgs.mGetPos(0)->mAsInt());
        return mk_val<value_float>(vVal);
      }                        },
    { "safe",    fTojson       },
    { "string",  fTojson       },
    { "tojson",  fTojson       },
  };
  return dBuiltins;
}

const func_builtins & value_float_t::mGetBuiltins() const {
  static const func_builtins dBuiltins = {
    { "default", fDefaultValue },
    { "abs",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_float>();
        double vVal = pArgs.mGetPos(0)->mAsFloat();
        return mk_val<value_float>(vVal < 0.0 ? -vVal : vVal);
      }                        },
    { "int",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_float>();
        int64_t vVal = static_cast<int64_t>(pArgs.mGetPos(0)->mAsFloat());
        return mk_val<value_int>(vVal);
      }                        },
    { "float",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_float>();
        return mk_val<value_float>(pArgs.mGetPos(0)->mAsFloat());
      }                        },
    { "safe",    fTojson       },
    { "string",  fTojson       },
    { "tojson",  fTojson       },
  };
  return dBuiltins;
}

static bool fStringStartswith(const std::string & str, const std::string & pPrefix) {
  if (str.length() < pPrefix.length()) {
    return false;
  }
  return str.compare(0, pPrefix.length(), pPrefix) == 0;
}

static bool fStringEndswith(const std::string & str, const std::string & pSuffix) {
  if (str.length() < pSuffix.length()) {
    return false;
  }
  return str.compare(str.length() - pSuffix.length(), pSuffix.length(), pSuffix) == 0;
}

[[noreturn]] static value fStringJoinNotImplemented(const func_args &) {
  throw not_implemented_exception("String join builtin not implemented");
}

const func_builtins & value_string_t::mGetBuiltins() const {
  static const func_builtins dBuiltins = {
    { "default",    fDefaultValue             },
    { "upper",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_string>();
        jinja::string str = pArgs.mGetPos(0)->mAsString().mUppercase();
        return mk_val<value_string>(str);
      }                                       },
    { "lower",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_string>();
        jinja::string str = pArgs.mGetPos(0)->mAsString().mLowercase();
        return mk_val<value_string>(str);
      }                                       },
    { "strip",
     [](const func_args & pArgs) -> value {
        value vValInput = pArgs.mGetPos(0);
        if (!is_val<value_string>(vValInput)) {
          throw raised_exception("strip() first argument must be a string");
        }
        value vValChars = pArgs.mGetKwargOrPos("chars", 1);
        if (vValChars->mIsUndefined()) {
          return mk_val<value_string>(pArgs.mGetPos(0)->mAsString().mStrip(true, true));
        } else {
          return mk_val<value_string>(pArgs.mGetPos(0)->mAsString().mStrip(true, true, vValChars->mAsString().str()));
        }
      }                                       },
    { "rstrip",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_string>();
        value vValChars = pArgs.mGetKwargOrPos("chars", 1);
        if (vValChars->mIsUndefined()) {
          return mk_val<value_string>(pArgs.mGetPos(0)->mAsString().mStrip(false, true));
        } else {
          return mk_val<value_string>(pArgs.mGetPos(0)->mAsString().mStrip(false, true, vValChars->mAsString().str()));
        }
      }                                       },
    { "lstrip",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_string>();
        value vValChars = pArgs.mGetKwargOrPos("chars", 1);
        if (vValChars->mIsUndefined()) {
          return mk_val<value_string>(pArgs.mGetPos(0)->mAsString().mStrip(true, false));
        } else {
          return mk_val<value_string>(pArgs.mGetPos(0)->mAsString().mStrip(true, false, vValChars->mAsString().str()));
        }
      }                                       },
    { "title",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_string>();
        jinja::string str = pArgs.mGetPos(0)->mAsString().mTitlecase();
        return mk_val<value_string>(str);
      }                                       },
    { "capitalize",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_string>();
        jinja::string str = pArgs.mGetPos(0)->mAsString().mCapitalize();
        return mk_val<value_string>(str);
      }                                       },
    { "length",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_string>();
        jinja::string str = pArgs.mGetPos(0)->mAsString();
        return mk_val<value_int>(str.length());
      }                                       },
    { "startswith",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_string, value_string>();
        std::string str     = pArgs.mGetPos(0)->mAsString().str();
        std::string vPrefix = pArgs.mGetPos(1)->mAsString().str();
        return mk_val<value_bool>(fStringStartswith(str, vPrefix));
      }                                       },
    { "endswith",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_string, value_string>();
        std::string str     = pArgs.mGetPos(0)->mAsString().str();
        std::string vSuffix = pArgs.mGetPos(1)->mAsString().str();
        return mk_val<value_bool>(fStringEndswith(str, vSuffix));
      }                                       },
    { "split",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureCount(1, 3);
        value vValInput = pArgs.mGetPos(0);
        if (!is_val<value_string>(vValInput)) {
          throw raised_exception("split() first argument must be a string");
        }
        std::string str    = vValInput->mAsString().str();
        // FIXME: Support non-specified delimiter (split on consecutive (no leading or trailing) whitespace)
        std::string vDelim = (pArgs.count() > 1) ? pArgs.mGetPos(1)->mAsString().str() : " ";
        if (vDelim.empty()) {
          throw raised_exception("empty separator");
        }
        int64_t     vMaxsplit = (pArgs.count() > 2) ? pArgs.mGetPos(2)->mAsInt() : -1;
        auto        vResult   = mk_val<value_array>();
        size_t      vPos      = 0;
        std::string token;
        while ((vPos = str.find(vDelim)) != std::string::npos && vMaxsplit != 0) {
          token = str.substr(0, vPos);
          vResult->push_back(mk_val<value_string>(token));
          str.erase(0, vPos + vDelim.length());
          --vMaxsplit;
        }
        auto vRes = mk_val<value_string>(str);
        vRes->vValStr.mMarkInputBasedOn(pArgs.mGetPos(0)->vValStr);
        vResult->push_back(std::move(vRes));
        return vResult;
      }                                       },
    { "rsplit",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureCount(1, 3);
        value vValInput = pArgs.mGetPos(0);
        if (!is_val<value_string>(vValInput)) {
          throw raised_exception("rsplit() first argument must be a string");
        }
        std::string str    = vValInput->mAsString().str();
        // FIXME: Support non-specified delimiter (split on consecutive (no leading or trailing) whitespace)
        std::string vDelim = (pArgs.count() > 1) ? pArgs.mGetPos(1)->mAsString().str() : " ";
        if (vDelim.empty()) {
          throw raised_exception("empty separator");
        }
        int64_t     vMaxsplit = (pArgs.count() > 2) ? pArgs.mGetPos(2)->mAsInt() : -1;
        auto        vResult   = mk_val<value_array>();
        size_t      vPos      = 0;
        std::string token;
        while ((vPos = str.rfind(vDelim)) != std::string::npos && vMaxsplit != 0) {
          token = str.substr(vPos + vDelim.length());
          vResult->push_back(mk_val<value_string>(token));
          str.erase(vPos);
          --vMaxsplit;
        }
        auto vRes = mk_val<value_string>(str);
        vRes->vValStr.mMarkInputBasedOn(pArgs.mGetPos(0)->vValStr);
        vResult->push_back(std::move(vRes));
        vResult->mReverse();
        return vResult;
      }                                       },
    { "replace",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_string, value_string, value_string, value_int>(true, true, true, false);
        std::string str     = pArgs.mGetPos(0)->mAsString().str();
        std::string vOldStr = pArgs.mGetPos(1)->mAsString().str();
        std::string vNewStr = pArgs.mGetPos(2)->mAsString().str();
        int64_t     count   = pArgs.count() > 3 ? pArgs.mGetPos(3)->mAsInt() : -1;
        if (count > 0) {
          throw not_implemented_exception("String replace with count argument not implemented");
        }
        if (vOldStr != vNewStr) {
          size_t vPos = 0;
          if (vOldStr.empty()) {
            std::string vNewRes;
            vNewRes.reserve(str.length() + vNewStr.length() * (str.length() + 1));
            vNewRes += vNewStr;
            for (const char cC : str) {
              vNewRes.push_back(cC);
              vNewRes += vNewStr;
            }
            str = vNewRes;
          } else {
            while ((vPos = str.find(vOldStr, vPos)) != std::string::npos) {
              str.replace(vPos, vOldStr.length(), vNewStr);
              vPos += vNewStr.length();
            }
          }
        }
        auto vRes = mk_val<value_string>(str);
        vRes->vValStr.mMarkInputBasedOn(pArgs.mGetPos(0)->vValStr);
        return vRes;
      }                                       },
    { "format",
     [](const func_args & pArgs) -> value {
        value vValInput = pArgs.mGetPos(0);
        if (!is_val<value_string>(vValInput)) {
          throw raised_exception("format() first argument must be a string");
        }
        const jinja::string & vFmt        = vValInput->mAsString();
        const bool            cFmtIsInput = vFmt.mAllPartsAreInput();

        const std::string str = vFmt.str();
        jinja::string     vResult;
        std::string       vLiteral;
        auto              vFlushLiteral = [&]() {
          if (!vLiteral.empty()) {
            vResult.lParts.push_back({ cFmtIsInput, vLiteral });
            vLiteral.clear();
          }
        };

        size_t vArgIdx = 1;  // positional args follow the format string
        for (size_t vI = 0; vI < str.size(); ++vI) {
          if (str[vI] != '{') {
            vLiteral += str[vI];
            continue;
          }
          if (vI + 1 >= str.size() || str[vI + 1] != '}') {
            throw not_implemented_exception("format() only supports simple '{}' placeholders");
          }
          ++vI;
          vFlushLiteral();
          const jinja::string cArgStr = pArgs.mGetPos(vArgIdx++)->mAsString();
          vResult.lParts.insert(vResult.lParts.end(), cArgStr.lParts.begin(), cArgStr.lParts.end());
        }
        vFlushLiteral();
        return mk_val<value_string>(vResult);
      }                                       },
    { "int",
     [](const func_args & pArgs) -> value {
        value     vValInput   = pArgs.mGetPos(0);
        value     vValDefault = pArgs.mGetKwargOrPos("default", 1);
        value     vValBase    = pArgs.mGetKwargOrPos("base", 2);
        const int cBase       = vValBase->mIsUndefined() ? 10 : vValBase->mAsInt();
        if (cBase != 0 && (cBase < 2 || cBase > 36)) {
          // an out-of-range base makes std::stoi fail fast on the MSVC CRT instead of throwing
          throw raised_exception("int() base must be 0 or between 2 and 36");
        }
        if (is_val<value_string>(vValInput) == false) {
          throw raised_exception("int() first argument must be a string");
        }
        std::string str = vValInput->mAsString().str();
        try {
          return mk_val<value_int>(std::stoi(str, nullptr, cBase));
        } catch (...) {
          return mk_val<value_int>(vValDefault->mIsUndefined() ? 0 : vValDefault->mAsInt());
        }
      }                                       },
    { "float",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_string>();
        value       vValDefault = pArgs.mGetKwargOrPos("default", 1);
        std::string str         = pArgs.mGetPos(0)->mAsString().str();
        try {
          return mk_val<value_float>(std::stod(str));
        } catch (...) {
          return mk_val<value_float>(vValDefault->mIsUndefined() ? 0.0 : vValDefault->mAsFloat());
        }
      }                                       },
    { "string",
     [](const func_args & pArgs) -> value {
        // no-op
        pArgs.mEnsureVals<value_string>();
        return mk_val<value_string>(pArgs.mGetPos(0)->mAsString());
      }                                       },
    { "default",
     [](const func_args & pArgs) -> value {
        value vInput = pArgs.mGetPos(0);
        if (!is_val<value_string>(vInput)) {
          throw raised_exception("default() first argument must be a string");
        }
        value vDefaultVal = mk_val<value_string>("");
        if (pArgs.count() > 1 && !pArgs.mGetPos(1)->mIsUndefined()) {
          vDefaultVal = pArgs.mGetPos(1);
        }
        value vBooleanVal = pArgs.mGetKwargOrPos("boolean", 2);  // undefined == false
        if (vInput->mIsUndefined() || (vBooleanVal->mAsBool() && !vInput->mAsBool())) {
          return vDefaultVal;
        } else {
          return vInput;
        }
      }                                       },
    { "slice",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureCount(1, 4);
        pArgs.mEnsureVals<value_string, value_int, value_int, value_int>(true, true, false, false);

        auto vArg0 = pArgs.mGetPos(1);
        auto vArg1 = pArgs.mGetPos(2, mk_val<value_undefined>());
        auto vArg2 = pArgs.mGetPos(3, mk_val<value_undefined>());

        int64_t vStart, vStop, vStep;
        if (pArgs.count() == 1) {
          vStart = 0;
          vStop  = vArg0->mAsInt();
          vStep  = 1;
        } else if (pArgs.count() == 2) {
          vStart = vArg0->mAsInt();
          vStop  = vArg1->mAsInt();
          vStep  = 1;
        } else {
          vStart = vArg0->mAsInt();
          vStop  = vArg1->mAsInt();
          vStep  = vArg2->mAsInt();
        }
        if (vStep == 0) {
          throw raised_exception("slice step cannot be zero");
        }
        auto vInput  = pArgs.mGetPos(0);
        auto vSliced = fSlice(vInput->mAsString().str(), vStart, vStop, vStep);
        auto vRes    = mk_val<value_string>(vSliced);
        vRes->vValStr.mMarkInputBasedOn(vInput->mAsString());
        return vRes;
      }                                       },
    { "safe",
     [](const func_args & pArgs) -> value {
        // no-op for now
        pArgs.mEnsureVals<value_string>();
        return pArgs.mGetPos(0);
      }                                       },
    { "tojson",     fTojson                   },
    { "indent",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureCount(1, 4);
        value      vValInput = pArgs.mGetPos(0);
        value      vValWidth = pArgs.mGetKwargOrPos("width", 1);
        const bool cFirst    = pArgs.mGetKwargOrPos("first", 2)->mAsBool();  // undefined == false
        const bool cBlank    = pArgs.mGetKwargOrPos("blank", 3)->mAsBool();  // undefined == false
        if (!is_val<value_string>(vValInput)) {
          throw raised_exception("indent() first argument must be a string");
        }
        std::string vIndent;
        if (is_val<value_int>(vValWidth)) {
          vIndent.assign(vValWidth->mAsInt(), ' ');
        } else if (is_val<value_string>(vValWidth)) {
          vIndent = vValWidth->mAsString().str();
        } else {
          vIndent = "    ";
        }
        std::string        vIndented;
        std::string        vInput = vValInput->mAsString().str();
        std::istringstream vIss   = std::istringstream(vInput);
        std::string        vLine;
        while (std::getline(vIss, vLine)) {
          if (!vIndented.empty()) {
            vIndented.push_back('\n');
          }
          if ((vIndented.empty() ? cFirst : (!vLine.empty() || cBlank))) {
            vIndented += vIndent;
          }
          vIndented += vLine;
        }
        if (!vInput.empty() && vInput.back() == '\n') {
          vIndented.push_back('\n');
          if (cBlank) {
            vIndented += vIndent;
          }
        }

        auto vRes = mk_val<value_string>(vIndented);
        vRes->vValStr.mMarkInputBasedOn(vValInput->mAsString());
        return vRes;
      }                                       },
    { "join",       fStringJoinNotImplemented },
  };
  return dBuiltins;
}

const func_builtins & value_bool_t::mGetBuiltins() const {
  static const func_handler cTostring = [](const func_args & pArgs) -> value {
    pArgs.mEnsureVals<value_bool>();
    bool vVal = pArgs.mGetPos(0)->mAsBool();
    return mk_val<value_string>(vVal ? "True" : "False");
  };
  static const func_builtins dBuiltins = {
    { "default", fDefaultValue },
    { "int",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_bool>();
        bool vVal = pArgs.mGetPos(0)->mAsBool();
        return mk_val<value_int>(vVal ? 1 : 0);
      }                        },
    { "float",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_bool>();
        bool vVal = pArgs.mGetPos(0)->mAsBool();
        return mk_val<value_float>(vVal ? 1.0 : 0.0);
      }                        },
    { "safe",    cTostring     },
    { "string",  cTostring     },
    { "tojson",  fTojson       },
  };
  return dBuiltins;
}

[[noreturn]] static value fArrayUniqueNotImplemented(const func_args &) {
  throw not_implemented_exception("Array unique builtin not implemented");
}

const func_builtins & value_array_t::mGetBuiltins() const {
  static const func_builtins dBuiltins = {
    { "default",    fDefaultValue              },
    { "list",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_array>();
        const auto & lArr    = pArgs.mGetPos(0)->mAsArray();
        auto         vResult = mk_val<value_array>();
        for (const auto & vV : lArr) {
          vResult->push_back(vV);
        }
        return vResult;
      }                                        },
    { "first",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_array>();
        const auto & lArr = pArgs.mGetPos(0)->mAsArray();
        if (lArr.empty()) {
          return mk_val<value_undefined>();
        }
        return lArr[0];
      }                                        },
    { "last",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_array>();
        const auto & lArr = pArgs.mGetPos(0)->mAsArray();
        if (lArr.empty()) {
          return mk_val<value_undefined>();
        }
        return lArr[lArr.size() - 1];
      }                                        },
    { "length",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_array>();
        const auto & lArr = pArgs.mGetPos(0)->mAsArray();
        return mk_val<value_int>(static_cast<int64_t>(lArr.size()));
      }                                        },
    { "slice",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureCount(1, 4);
        pArgs.mEnsureVals<value_array, value_int, value_int, value_int>(true, true, false, false);

        auto vVal  = pArgs.mGetPos(0);
        auto vArg0 = pArgs.mGetPos(1);
        auto vArg1 = pArgs.mGetPos(2, mk_val<value_undefined>());
        auto vArg2 = pArgs.mGetPos(3, mk_val<value_undefined>());

        int64_t vStart, vStop, vStep;
        if (pArgs.count() == 1) {
          vStart = 0;
          vStop  = vArg0->mAsInt();
          vStep  = 1;
        } else if (pArgs.count() == 2) {
          vStart = vArg0->mAsInt();
          vStop  = vArg1->mAsInt();
          vStep  = 1;
        } else {
          vStart = vArg0->mAsInt();
          vStop  = vArg1->mAsInt();
          vStep  = vArg2->mAsInt();
        }
        if (vStep == 0) {
          throw raised_exception("slice step cannot be zero");
        }
        auto lArr = fSlice(vVal->mAsArray(), vStart, vStop, vStep);
        return is_val<value_tuple>(vVal) ? mk_val<value_tuple>(std::move(lArr)) : mk_val<value_array>(std::move(lArr));
      }                                        },
    { "selectattr", selectattr<false>          },
    { "select",     selectattr<false>          },
    { "rejectattr", selectattr<true>           },
    { "reject",     selectattr<true>           },
    { "join",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureCount(1, 3);
        if (!is_val<value_array>(pArgs.mGetPos(0))) {
          throw raised_exception("join() first argument must be an array");
        }
        value             vValDelim  = pArgs.mGetKwargOrPos("d", 1);
        value             vAttribute = pArgs.mGetKwargOrPos("attribute", 2);
        value             vUndef     = mk_val<value_undefined>();
        const auto &      lArr       = pArgs.mGetPos(0)->mAsArray();
        const std::string cDelim     = vValDelim->mIsUndefined() ? "" : vValDelim->mAsString().str();
        std::string       vResult;
        for (size_t vI = 0; vI < lArr.size(); ++vI) {
          value vValArr = lArr[vI];
          if (!vAttribute->mIsUndefined()) {
            vValArr = fGetAttribute(vValArr, vAttribute, vUndef);
          }
          if (!is_val<value_string>(vValArr) && !is_val<value_int>(vValArr) && !is_val<value_float>(vValArr)) {
            throw raised_exception("join() can only join arrays of strings or numerics");
          }
          vResult += vValArr->mAsString().str();
          if (vI < lArr.size() - 1) {
            vResult += cDelim;
          }
        }
        return mk_val<value_string>(vResult);
      }                                        },
    { "string",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_array>();
        if (pArgs.vCtx.vIsGetStats) {
          // mark as used (recursively) for stats
          auto vValInput = pArgs.mGetPos(0);
          value_t::stats_t::mMarkUsed(const_cast<value &>(vValInput), true);
        }
        return mk_val<value_string>(pArgs.mGetPos(0)->mAsString());
      }                                        },
    { "tojson",     fTojson                    },
    { "map",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureCount(2);
        if (!is_val<value_array>(pArgs.mGetPos(0))) {
          throw raised_exception("map: first argument must be an array");
        }
        if (!is_val<value_kwarg>(pArgs.mGetArgs().at(1))) {
          throw not_implemented_exception("map: filter-mapping not implemented");
        }
        value vVal        = pArgs.mGetPos(0);
        value vAttribute  = pArgs.mGetKwargOrPos("attribute", 1);
        value vDefaultVal = pArgs.mGetKwarg("default", mk_val<value_undefined>());
        auto  vOut        = mk_val<value_array>();
        auto  lArr        = vVal->mAsArray();
        for (const auto & vItem : lArr) {
          value vAttrVal = fGetAttribute(vItem, vAttribute, vDefaultVal);
          vOut->push_back(vAttrVal);
        }
        return is_val<value_tuple>(vVal) ? mk_val<value_tuple>(std::move(vOut->mAsArray())) : vOut;
      }                                        },
    { "append",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureCount(2);
        if (!is_val<value_array>(pArgs.mGetPos(0))) {
          throw raised_exception("append: first argument must be an array");
        }
        const value_array_t * vArr         = fCastVal<value_array>(pArgs.mGetPos(0));
        // need to use const_cast here to modify the array
        value_array_t *       vArrEditable = const_cast<value_array_t *>(vArr);
        vArrEditable->push_back(pArgs.mGetPos(1));
        return pArgs.mGetPos(0);
      }                                        },
    { "pop",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureCount(1, 2);
        pArgs.mEnsureVals<value_array, value_int>(true, false);
        int64_t               vIndex       = pArgs.count() == 2 ? pArgs.mGetPos(1)->mAsInt() : -1;
        const value_array_t * vArr         = fCastVal<value_array>(pArgs.mGetPos(0));
        // need to use const_cast here to modify the array
        value_array_t *       vArrEditable = const_cast<value_array_t *>(vArr);
        return vArrEditable->mPopAt(vIndex);
      }                                        },
    { "sort",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureCount(1, 4);
        if (!is_val<value_array>(pArgs.mGetPos(0))) {
          throw raised_exception("sort: first argument must be an array");
        }
        value              vVal        = pArgs.mGetPos(0);
        value              vValReverse = pArgs.mGetKwargOrPos("reverse", 1);
        value              vValCase    = pArgs.mGetKwargOrPos("case_sensitive", 2);
        value              vAttribute  = pArgs.mGetKwargOrPos("attribute", 3);
        // FIXME: sorting is currently always case sensitive
        //const bool case_sensitive = val_case->as_bool(); // undefined == false
        const bool         cReverse    = vValReverse->mAsBool();  // undefined == false
        value              vUndef      = mk_val<value_undefined>();
        std::vector<value> lArr        = vVal->mAsArray();        // copy
        std::sort(lArr.begin(), lArr.end(), [&](const value & pA, const value & pB) {
          value vValA = pA;
          value vValB = pB;
          if (!vAttribute->mIsUndefined()) {
            vValA = fGetAttribute(pA, vAttribute, vUndef);
            vValB = fGetAttribute(pB, vAttribute, vUndef);
          }
          return value_compare(vValA, vValB, cReverse ? value_compare_op::cGt : value_compare_op::cLt);
        });
        return is_val<value_tuple>(vVal) ? mk_val<value_tuple>(std::move(lArr)) : mk_val<value_array>(std::move(lArr));
      }                                        },
    { "reverse",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_array>();
        value              vVal = pArgs.mGetPos(0);
        std::vector<value> lArr = vVal->mAsArray();  // copy
        std::reverse(lArr.begin(), lArr.end());
        return is_val<value_tuple>(vVal) ? mk_val<value_tuple>(std::move(lArr)) : mk_val<value_array>(std::move(lArr));
      }                                        },
    { "min",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureCount(1, 4);
        pArgs.mEnsureVals<value_array>();
        value vValCase   = pArgs.mGetKwargOrPos("case_sensitive", 1);
        value vAttribute = pArgs.mGetKwargOrPos("attribute", 2);
        // FIXME: min is currently always case sensitive
        (void) vValCase;
        value        vUndef = mk_val<value_undefined>();
        const auto & lArr   = pArgs.mGetPos(0)->mAsArray();
        if (lArr.empty()) {
          return vUndef;
        }
        value vResult = lArr[0];
        for (const auto & vItem : lArr) {
          value vValArr = vItem;
          value vValCmp = vResult;
          if (!vAttribute->mIsUndefined()) {
            vValArr = fGetAttribute(vValArr, vAttribute, vUndef);
            vValCmp = fGetAttribute(vValCmp, vAttribute, vUndef);
          }
          if (value_compare(vValArr, vValCmp, value_compare_op::cLt)) {
            vResult = vItem;
          }
        }
        return vResult;
      }                                        },
    { "max",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureCount(1, 4);
        pArgs.mEnsureVals<value_array>();
        value vValCase   = pArgs.mGetKwargOrPos("case_sensitive", 1);
        value vAttribute = pArgs.mGetKwargOrPos("attribute", 2);
        // FIXME: max is currently always case sensitive
        (void) vValCase;
        value        vUndef = mk_val<value_undefined>();
        const auto & lArr   = pArgs.mGetPos(0)->mAsArray();
        if (lArr.empty()) {
          return vUndef;
        }
        value vResult = lArr[0];
        for (const auto & vItem : lArr) {
          value vValArr = vItem;
          value vValCmp = vResult;
          if (!vAttribute->mIsUndefined()) {
            vValArr = fGetAttribute(vValArr, vAttribute, vUndef);
            vValCmp = fGetAttribute(vValCmp, vAttribute, vUndef);
          }
          if (value_compare(vValArr, vValCmp, value_compare_op::cGt)) {
            vResult = vItem;
          }
        }
        return vResult;
      }                                        },
    { "unique",     fArrayUniqueNotImplemented },
  };
  return dBuiltins;
}

[[noreturn]] static value fObjectJoinNotImplemented(const func_args &) {
  throw not_implemented_exception("object join not implemented");
}

const func_builtins & value_object_t::mGetBuiltins() const {
  if (!vHasBuiltins) {
    static const func_builtins dNoBuiltins = {};
    return dNoBuiltins;
  }

  static const func_builtins dBuiltins = {
    // {"default", default_value}, // cause issue with gpt-oss
    { "get",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureCount(2, 3);
        if (!is_val<value_object>(pArgs.mGetPos(0))) {
          throw raised_exception("get: first argument must be an object");
        }
        if (!is_val<value_string>(pArgs.mGetPos(1))) {
          throw raised_exception("get: second argument must be a string (key)");
        }
        value vDefaultVal = mk_val<value_none>();
        if (pArgs.count() == 3) {
          vDefaultVal = pArgs.mGetPos(2);
        }
        const value cObj = pArgs.mGetPos(0);
        const value cKey = pArgs.mGetPos(1);
        return cObj->at(cKey, vDefaultVal);
      }                                     },
    { "keys",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_object>();
        const auto & lObj    = pArgs.mGetPos(0)->mAsOrderedObject();
        auto         vResult = mk_val<value_array>();
        for (const auto & vPair : lObj) {
          vResult->push_back(vPair.first);
        }
        return vResult;
      }                                     },
    { "values",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_object>();
        const auto & lObj    = pArgs.mGetPos(0)->mAsOrderedObject();
        auto         vResult = mk_val<value_array>();
        for (const auto & vPair : lObj) {
          vResult->push_back(vPair.second);
        }
        return vResult;
      }                                     },
    { "items",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_object>();
        const auto & lObj    = pArgs.mGetPos(0)->mAsOrderedObject();
        auto         vResult = mk_val<value_array>();
        for (const auto & vPair : lObj) {
          auto vItem = mk_val<value_tuple>(vPair);
          vResult->push_back(std::move(vItem));
        }
        return vResult;
      }                                     },
    { "tojson",   fTojson                   },
    { "string",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_object>();
        if (pArgs.vCtx.vIsGetStats) {
          // mark as used (recursively) for stats
          auto vValInput = pArgs.mGetPos(0);
          value_t::stats_t::mMarkUsed(const_cast<value &>(vValInput), true);
        }
        return mk_val<value_string>(pArgs.mGetPos(0)->mAsString());
      }                                     },
    { "length",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_object>();
        const auto & lObj = pArgs.mGetPos(0)->mAsOrderedObject();
        return mk_val<value_int>(static_cast<int64_t>(lObj.size()));
      }                                     },
    { "tojson",
     [](const func_args & pArgs) -> value {
        pArgs.mEnsureVals<value_object>();
        // use global to_json
        return fGlobalBuiltins().at("tojson")(pArgs);
      }                                     },
    { "dictsort",
     [](const func_args & pArgs) -> value {
        value      vValInput   = pArgs.mGetPos(0);
        value      vValCase    = pArgs.mGetKwargOrPos("case_sensitive", 1);
        value      vValBy      = pArgs.mGetKwargOrPos("by", 2);
        value      vValReverse = pArgs.mGetKwargOrPos("reverse", 3);
        // FIXME: sorting is currently always case sensitive
        //const bool case_sensitive = val_case->as_bool(); // undefined == false
        const bool cReverse    = vValReverse->mAsBool();           // undefined == false
        const bool cByValue    = is_val<value_string>(vValBy) && vValBy->mAsString().str() == "value" ? true : false;
        auto       vResult     = mk_val<value_object>(vValInput);  // copy
        std::sort(vResult->lValObj.begin(), vResult->lValObj.end(), [&](const auto & pA, const auto & pB) {
          if (cByValue) {
            return value_compare(pA.second, pB.second, cReverse ? value_compare_op::cGt : value_compare_op::cLt);
          } else {
            return value_compare(pA.first, pB.first, cReverse ? value_compare_op::cGt : value_compare_op::cLt);
          }
        });
        return vResult;
      }                                     },
    { "join",     fObjectJoinNotImplemented },
  };
  return dBuiltins;
}

const func_builtins & value_none_t::mGetBuiltins() const {
  static const func_handler cTostring = [](const func_args &) -> value {
    return mk_val<value_string>("None");
  };
  static const func_builtins dBuiltins = {
    { "default",    fDefaultValue               },
    { "tojson",     fTojson                     },
    { "string",     cTostring                   },
    { "safe",       cTostring                   },
    { "items",      empty_value_fn<value_array> },
    { "map",        empty_value_fn<value_array> },
    { "reject",     empty_value_fn<value_array> },
    { "rejectattr", empty_value_fn<value_array> },
    { "select",     empty_value_fn<value_array> },
    { "selectattr", empty_value_fn<value_array> },
    { "unique",     empty_value_fn<value_array> },
  };
  return dBuiltins;
}

const func_builtins & value_undefined_t::mGetBuiltins() const {
  static const func_builtins dBuiltins = {
    { "default",    fDefaultValue                   },
    { "capitalize", empty_value_fn<value_string>    },
    { "first",      empty_value_fn<value_undefined> },
    { "items",      empty_value_fn<value_array>     },
    { "join",       empty_value_fn<value_string>    },
    { "last",       empty_value_fn<value_undefined> },
    { "length",     empty_value_fn<value_int>       },
    { "list",       empty_value_fn<value_array>     },
    { "lower",      empty_value_fn<value_string>    },
    { "map",        empty_value_fn<value_array>     },
    { "max",        empty_value_fn<value_undefined> },
    { "min",        empty_value_fn<value_undefined> },
    { "reject",     empty_value_fn<value_array>     },
    { "rejectattr", empty_value_fn<value_array>     },
    { "replace",    empty_value_fn<value_string>    },
    { "reverse",    empty_value_fn<value_array>     },
    { "safe",       empty_value_fn<value_string>    },
    { "select",     empty_value_fn<value_array>     },
    { "selectattr", empty_value_fn<value_array>     },
    { "sort",       empty_value_fn<value_array>     },
    { "string",     empty_value_fn<value_string>    },
    { "strip",      empty_value_fn<value_string>    },
    { "sum",        empty_value_fn<value_int>       },
    { "title",      empty_value_fn<value_string>    },
    { "truncate",   empty_value_fn<value_string>    },
    { "unique",     empty_value_fn<value_array>     },
    { "upper",      empty_value_fn<value_string>    },
    { "wordcount",  empty_value_fn<value_int>       },
  };
  return dBuiltins;
}

//////////////////////////////////

static value from_json(const common_json & pJ, bool pMarkInput) {
  if (pJ.is_null()) {
    return mk_val<value_none>();
  } else if (pJ.is_boolean()) {
    return mk_val<value_bool>(pJ.get<bool>());
  } else if (pJ.mIsNumberInteger()) {
    return mk_val<value_int>(pJ.get<int64_t>());
  } else if (pJ.mIsNumberFloat()) {
    return mk_val<value_float>(pJ.get<double>());
  } else if (pJ.is_string()) {
    auto str = mk_val<value_string>(pJ.get<std::string>());
    if (pMarkInput) {
      str->mMarkInput();
    }
    return str;
  } else if (pJ.is_array()) {
    auto vArr = mk_val<value_array>();
    for (const auto & vItem : pJ) {
      vArr->push_back(from_json(vItem, pMarkInput));
    }
    return vArr;
  } else if (pJ.is_object()) {
    auto vObj = mk_val<value_object>();
    for (auto vIt = pJ.begin(); vIt != pJ.end(); ++vIt) {
      vObj->insert(vIt.mKey(), from_json(vIt.value(), pMarkInput));
    }
    return vObj;
  } else {
    throw std::runtime_error("Unsupported JSON value type");
  }
}

// compare operator for value_t
bool value_compare(const value & pA, const value & pB, value_compare_op pOp) {
  auto vCmp = [&]() {
    // compare numeric types
    if ((is_val<value_int>(pA) || is_val<value_float>(pA)) && (is_val<value_int>(pB) || is_val<value_float>(pB))) {
      try {
        if (pOp == value_compare_op::cEq) {
          return pA->mAsFloat() == pB->mAsFloat();
        } else if (pOp == value_compare_op::cGe) {
          return pA->mAsFloat() >= pB->mAsFloat();
        } else if (pOp == value_compare_op::cGt) {
          return pA->mAsFloat() > pB->mAsFloat();
        } else if (pOp == value_compare_op::cLt) {
          return pA->mAsFloat() < pB->mAsFloat();
        } else if (pOp == value_compare_op::cNe) {
          return pA->mAsFloat() != pB->mAsFloat();
        } else {
          throw std::runtime_error("Unsupported comparison operator for numeric types");
        }
      } catch (...) {
      }
    }
    // compare string and number
    // TODO: not sure if this is the right behavior
    if ((is_val<value_string>(pB) && (is_val<value_int>(pA) || is_val<value_float>(pA))) ||
        (is_val<value_string>(pA) && (is_val<value_int>(pB) || is_val<value_float>(pB))) ||
        (is_val<value_string>(pA) && is_val<value_string>(pB))) {
      try {
        if (pOp == value_compare_op::cEq) {
          return pA->mAsString().str() == pB->mAsString().str();
        } else if (pOp == value_compare_op::cGe) {
          return pA->mAsString().str() >= pB->mAsString().str();
        } else if (pOp == value_compare_op::cGt) {
          return pA->mAsString().str() > pB->mAsString().str();
        } else if (pOp == value_compare_op::cLt) {
          return pA->mAsString().str() < pB->mAsString().str();
        } else if (pOp == value_compare_op::cNe) {
          return pA->mAsString().str() != pB->mAsString().str();
        } else {
          throw std::runtime_error("Unsupported comparison operator for string/number types");
        }
      } catch (...) {
      }
    }
    // compare boolean simple
    if (is_val<value_bool>(pA) && is_val<value_bool>(pB)) {
      if (pOp == value_compare_op::cEq) {
        return pA->mAsBool() == pB->mAsBool();
      } else if (pOp == value_compare_op::cNe) {
        return pA->mAsBool() != pB->mAsBool();
      } else {
        throw std::runtime_error("Unsupported comparison operator for bool type");
      }
    }
    // compare by type
    if (pA->type() != pB->type()) {
      return false;
    }
    return false;
  };
  auto vResult = vCmp();
  JJ_DEBUG("Comparing types: %s and %s result=%d", pA->type().c_str(), pB->type().c_str(), vResult);
  return vResult;
}

template <> void fGlobalFromJson(context & pCtx, const common_json & pJsonObj, bool pMarkInput) {
  // printf("global_from_json: %s\n" , json_obj.dump(2).c_str());
  if (pJsonObj.is_null() || !pJsonObj.is_object()) {
    throw std::runtime_error("global_from_json: input JSON value must be an object");
  }
  for (auto vIt = pJsonObj.begin(); vIt != pJsonObj.end(); ++vIt) {
    JJ_DEBUG("global_from_json: setting key '%s'", vIt.mKey().c_str());
    pCtx.mSetVal(vIt.mKey(), from_json(vIt.value(), pMarkInput));
  }
}

// recursively convert value to JSON string
// TODO: avoid circular references
static void fValueToJsonInternal(std::ostringstream &   pOss,
                                 const value &          pVal,
                                 int                    pCurrLvl,
                                 int                    pIndent,
                                 const std::string_view pItemSep,
                                 const std::string_view pKeySep) {
  auto vIndentStr = [pIndent, pCurrLvl]() -> std::string {
    return (pIndent > 0) ? std::string(pCurrLvl * pIndent, ' ') : "";
  };
  auto vNewline = [pIndent]() -> std::string {
    return (pIndent >= 0) ? "\n" : "";
  };

  if (is_val<value_none>(pVal) || pVal->mIsUndefined()) {
    pOss << "null";
  } else if (is_val<value_bool>(pVal)) {
    pOss << (pVal->mAsBool() ? "true" : "false");
  } else if (is_val<value_int>(pVal)) {
    pOss << pVal->mAsInt();
  } else if (is_val<value_float>(pVal)) {
    pOss << pVal->mAsFloat();
  } else if (is_val<value_string>(pVal)) {
    pOss << "\"";
    for (char vC : pVal->mAsString().str()) {
      switch (vC) {
        case '"':
          pOss << "\\\"";
          break;
        case '\\':
          pOss << "\\\\";
          break;
        case '\b':
          pOss << "\\b";
          break;
        case '\f':
          pOss << "\\f";
          break;
        case '\n':
          pOss << "\\n";
          break;
        case '\r':
          pOss << "\\r";
          break;
        case '\t':
          pOss << "\\t";
          break;
        default:
          if (static_cast<unsigned char>(vC) < 0x20) {
            char aBuf[7];
            snprintf(aBuf, sizeof(aBuf), "\\u%04x", static_cast<unsigned char>(vC));
            pOss << aBuf;
          } else {
            pOss << vC;
          }
      }
    }
    pOss << "\"";
  } else if (is_val<value_array>(pVal)) {
    const auto & lArr = pVal->mAsArray();
    pOss << "[";
    if (!lArr.empty()) {
      pOss << vNewline();
      for (size_t vI = 0; vI < lArr.size(); ++vI) {
        pOss << vIndentStr() << (pIndent > 0 ? std::string(pIndent, ' ') : "");
        fValueToJsonInternal(pOss, lArr[vI], pCurrLvl + 1, pIndent, pItemSep, pKeySep);
        if (vI < lArr.size() - 1) {
          pOss << pItemSep;
        }
        pOss << vNewline();
      }
      pOss << vIndentStr();
    }
    pOss << "]";
  } else if (is_val<value_object>(pVal)) {
    const auto & lObj = pVal->mAsOrderedObject();  // IMPORTANT: need to keep exact order
    pOss << "{";
    if (!lObj.empty()) {
      pOss << vNewline();
      size_t vI = 0;
      for (const auto & vPair : lObj) {
        pOss << vIndentStr() << (pIndent > 0 ? std::string(pIndent, ' ') : "");
        fValueToJsonInternal(pOss, mk_val<value_string>(vPair.first->mAsString().str()), pCurrLvl + 1, pIndent,
                             pItemSep, pKeySep);
        pOss << pKeySep;
        fValueToJsonInternal(pOss, vPair.second, pCurrLvl + 1, pIndent, pItemSep, pKeySep);
        if (vI < lObj.size() - 1) {
          pOss << pItemSep;
        }
        pOss << vNewline();
        ++vI;
      }
      pOss << vIndentStr();
    }
    pOss << "}";
  } else {
    pOss << "null";
  }
}

std::string fValueToJson(const value &          pVal,
                         int                    pIndent,
                         const std::string_view pItemSep,
                         const std::string_view pKeySep) {
  std::ostringstream vOss;
  fValueToJsonInternal(vOss, pVal, 0, pIndent, pItemSep, pKeySep);
  JJ_DEBUG("value_to_json: result=%s", vOss.str().c_str());
  return vOss.str();
}

// TODO: avoid circular references
std::string fValueToStringRepr(const value & pVal) {
  if (is_val<value_string>(pVal)) {
    const std::string cValStr = pVal->mAsString().str();

    if (cValStr.find('\'') != std::string::npos) {
      return fValueToJson(pVal);
    } else {
      return "'" + cValStr + "'";
    }
  } else {
    return pVal->mAsRepr();
  }
}

// stats utility
void value_t::stats_t::mMarkUsed(value & pVal, bool pDeep) {
  pVal->vStats.vUsed = true;
  if (pDeep) {
    if (is_val<value_array>(pVal)) {
      for (auto & vItem : pVal->lValArr) {
        mMarkUsed(vItem, pDeep);
      }
    } else if (is_val<value_object>(pVal)) {
      for (auto & vPair : pVal->lValObj) {
        mMarkUsed(vPair.first, pDeep);
        mMarkUsed(vPair.second, pDeep);
      }
    }
  }
}

}  // namespace jinja
