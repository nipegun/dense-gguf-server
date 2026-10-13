#include "runtime.h"

#include "lexer.h"
#include "utils.h"
#include "value.h"

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#define FILENAME "jinja-runtime"

bool g_jinja_debug = false;

namespace jinja {

void fEnableDebug(bool pEnable) {
  g_jinja_debug = pEnable;
}

static value_string fExecStatements(const statements & pStmts, context & pCtx) {
  auto vResult = mk_val<value_array>();
  for (const auto & vStmt : pStmts) {
    JJ_DEBUG("Executing statement of type %s", vStmt->type().c_str());
    vResult->push_back(vStmt->mExecute(pCtx));
  }
  // convert to string parts
  value_string str = mk_val<value_string>();
  fGatherStringPartsRecursive(vResult, str);
  return str;
}

static std::string fGetLineCol(const std::string & pSource, size_t pPos) {
  size_t vLine = 1;
  size_t vCol  = 1;
  for (size_t vI = 0; vI < pPos && vI < pSource.size(); vI++) {
    if (pSource[vI] == '\n') {
      vLine++;
      vCol = 1;
    } else {
      vCol++;
    }
  }
  return "line " + std::to_string(vLine) + ", column " + std::to_string(vCol);
}

static void fEnsureKeyTypeAllowed(const value & pVal) {
  if (!pVal->mIsHashable()) {
    throw std::runtime_error("Type: " + pVal->type() + " is not allowed as object key");
  }
}

// execute with error handling
value statement::mExecute(context & pCtx) const {
  try {
    return mExecuteImpl(pCtx);
  } catch (const continue_statement::signal & /* ex */) {
    throw;
  } catch (const break_statement::signal & /* ex */) {
    throw;
  } catch (const rethrown_exception & /* ex */) {
    throw;
  } catch (const not_implemented_exception & /* ex */) {
    throw;
  } catch (const std::exception & vE) {
    const std::string & vSource = *pCtx.vSrc;
    if (vSource.empty()) {
      std::ostringstream vOss;
      vOss << "\nError executing " << type() << " at position " << vPos << ": " << vE.what();
      throw rethrown_exception(vOss.str());
    } else {
      std::ostringstream vOss;
      vOss << "\n------------\n";
      vOss << "While executing " << type() << " at " << fGetLineCol(vSource, vPos) << " in source:\n";
      vOss << fPeakSource(vSource, vPos) << "\n";
      vOss << "Error: " << vE.what();
      // throw as another exception to avoid repeated formatting
      throw rethrown_exception(vOss.str());
    }
  }
}

value identifier::mExecuteImpl(context & pCtx) const {
  auto vIt       = pCtx.mGetVal(vVal);
  auto dBuiltins = fGlobalBuiltins();
  if (!vIt->mIsUndefined()) {
    if (pCtx.vIsGetStats) {
      value_t::stats_t::mMarkUsed(vIt);
    }
    JJ_DEBUG("Identifier '%s' found, type = %s", vVal.c_str(), vIt->type().c_str());
    return vIt;
  } else if (dBuiltins.find(vVal) != dBuiltins.end()) {
    JJ_DEBUG("Identifier '%s' found in builtins", vVal.c_str());
    return mk_val<value_func>(vVal, dBuiltins.at(vVal));
  } else {
    JJ_DEBUG("Identifier '%s' not found, returning undefined", vVal.c_str());
    return mk_val<value_undefined>(vVal);
  }
}

value object_literal::mExecuteImpl(context & pCtx) const {
  auto vObj = mk_val<value_object>();
  for (const auto & vPair : lVal) {
    value vKey = vPair.first->mExecute(pCtx);
    value vVal = vPair.second->mExecute(pCtx);
    JJ_DEBUG("Object literal: setting key '%s' with value type %s", vKey->mAsString().str().c_str(),
             vVal->type().c_str());
    vObj->insert(vKey, vVal);
  }
  return vObj;
}

value binary_expression::mExecuteImpl(context & pCtx) const {
  value vLeftVal = vLeft->mExecute(pCtx);

  // Logical operators
  if (vOp.value == "and") {
    JJ_DEBUG("Executing logical test: %s AND %s", vLeft->type().c_str(), vRight->type().c_str());
    return vLeftVal->mAsBool() ? vRight->mExecute(pCtx) : std::move(vLeftVal);
  } else if (vOp.value == "or") {
    JJ_DEBUG("Executing logical test: %s OR %s", vLeft->type().c_str(), vRight->type().c_str());
    return vLeftVal->mAsBool() ? std::move(vLeftVal) : vRight->mExecute(pCtx);
  }

  // Equality operators
  value vRightVal = vRight->mExecute(pCtx);
  JJ_DEBUG("Executing binary expression %s '%s' %s", vLeftVal->type().c_str(), vOp.value.c_str(),
           vRightVal->type().c_str());
  if (vOp.value == "==") {
    return mk_val<value_bool>(*vLeftVal == *vRightVal);
  } else if (vOp.value == "!=") {
    return mk_val<value_bool>(!(*vLeftVal == *vRightVal));
  }

  auto vWorkaroundConcatNullWithStr = [&](value & pRes) -> bool {
    bool vIsLeftNull  = vLeftVal->mIsNone() || vLeftVal->mIsUndefined();
    bool vIsRightNull = vRightVal->mIsNone() || vRightVal->mIsUndefined();
    bool vIsLeftStr   = is_val<value_string>(vLeftVal);
    bool vIsRightStr  = is_val<value_string>(vRightVal);
    if ((vIsLeftNull && vIsRightStr) || (vIsRightNull && vIsLeftStr)) {
      JJ_DEBUG("%s", "Workaround: treating null/undefined as empty string for string concatenation");
      string vLeftStr  = vIsLeftNull ? string() : vLeftVal->mAsString();
      string vRightStr = vIsRightNull ? string() : vRightVal->mAsString();
      auto   vOutput   = vLeftStr.mAppend(vRightStr);
      pRes             = mk_val<value_string>(std::move(vOutput));
      return true;
    }
    return false;
  };

  auto vTestIsIn = [&]() -> bool {
    func_args vArgs(pCtx);
    vArgs.push_back(vLeftVal);
    vArgs.push_back(vRightVal);
    return fGlobalBuiltins().at("test_is_in")(vArgs)->mAsBool();
  };

  // Handle undefined and null values
  if (is_val<value_undefined>(vLeftVal) || is_val<value_undefined>(vRightVal)) {
    if (is_val<value_undefined>(vRightVal) && (vOp.value == "in" || vOp.value == "not in")) {
      // Special case: `anything in undefined` is `false` and `anything not in undefined` is `true`
      return mk_val<value_bool>(vOp.value == "not in");
    }
    if (vOp.value == "+" || vOp.value == "~") {
      value vRes = mk_val<value_undefined>();
      if (vWorkaroundConcatNullWithStr(vRes)) {
        return vRes;
      }
    }
    throw std::runtime_error("Cannot perform operation " + vOp.value + " on undefined values");
  } else if (is_val<value_none>(vLeftVal) || is_val<value_none>(vRightVal)) {
    if (!is_val<value_none>(vRightVal) && (vOp.value == "in" || vOp.value == "not in")) {
      // case: none in {'low': 1}
      // A null left operand is looked up like any other value.
      bool vMember = vTestIsIn();
      return mk_val<value_bool>(vOp.value == "in" ? vMember : !vMember);
    }
    if (vOp.value == "+" || vOp.value == "~") {
      value vRes = mk_val<value_undefined>();
      if (vWorkaroundConcatNullWithStr(vRes)) {
        return vRes;
      }
    }
    throw std::runtime_error("Cannot perform operation on null values");
  }

  // Float operations
  if ((is_val<value_int>(vLeftVal) || is_val<value_float>(vLeftVal)) &&
      (is_val<value_int>(vRightVal) || is_val<value_float>(vRightVal))) {
    double vA = vLeftVal->mAsFloat();
    double vB = vRightVal->mAsFloat();
    if (vOp.value == "+" || vOp.value == "-" || vOp.value == "*") {
      double vRes = (vOp.value == "+") ? vA + vB : (vOp.value == "-") ? vA - vB : vA * vB;
      JJ_DEBUG("Arithmetic operation: %f %s %f = %f", vA, vOp.value.c_str(), vB, vRes);
      bool vIsFloat = is_val<value_float>(vLeftVal) || is_val<value_float>(vRightVal);
      if (vIsFloat) {
        return mk_val<value_float>(vRes);
      } else {
        return mk_val<value_int>(static_cast<int64_t>(vRes));
      }
    } else if (vOp.value == "/") {
      JJ_DEBUG("Division operation: %f / %f", vA, vB);
      return mk_val<value_float>(vA / vB);
    } else if (vOp.value == "%") {
      double vRem = std::fmod(vA, vB);
      JJ_DEBUG("Modulo operation: %f %% %f = %f", vA, vB, vRem);
      bool vIsFloat = is_val<value_float>(vLeftVal) || is_val<value_float>(vRightVal);
      if (vIsFloat) {
        return mk_val<value_float>(vRem);
      } else {
        return mk_val<value_int>(static_cast<int64_t>(vRem));
      }
    } else if (vOp.value == "<") {
      JJ_DEBUG("Comparison operation: %f < %f is %d", vA, vB, vA < vB);
      return mk_val<value_bool>(vA < vB);
    } else if (vOp.value == ">") {
      JJ_DEBUG("Comparison operation: %f > %f is %d", vA, vB, vA > vB);
      return mk_val<value_bool>(vA > vB);
    } else if (vOp.value == ">=") {
      JJ_DEBUG("Comparison operation: %f >= %f is %d", vA, vB, vA >= vB);
      return mk_val<value_bool>(vA >= vB);
    } else if (vOp.value == "<=") {
      JJ_DEBUG("Comparison operation: %f <= %f is %d", vA, vB, vA <= vB);
      return mk_val<value_bool>(vA <= vB);
    }
  }

  // Array operations
  if (is_val<value_array>(vLeftVal) && is_val<value_array>(vRightVal)) {
    if (vOp.value == "+") {
      auto & lLeftArr  = vLeftVal->mAsArray();
      auto & lRightArr = vRightVal->mAsArray();
      auto   vResult   = mk_val<value_array>();
      for (const auto & vItem : lLeftArr) {
        vResult->push_back(vItem);
      }
      for (const auto & vItem : lRightArr) {
        vResult->push_back(vItem);
      }
      return vResult;
    }
  } else if (is_val<value_array>(vRightVal)) {
    // case: 1 in [0, 1, 2]
    bool vMember = vTestIsIn();
    if (vOp.value == "in") {
      return mk_val<value_bool>(vMember);
    } else if (vOp.value == "not in") {
      return mk_val<value_bool>(!vMember);
    }
  }

  // String concatenation with ~ and +
  if ((is_val<value_string>(vLeftVal) || is_val<value_string>(vRightVal)) && (vOp.value == "~" || vOp.value == "+")) {
    JJ_DEBUG("String concatenation with %s operator", vOp.value.c_str());
    auto vOutput  = vLeftVal->mAsString().mAppend(vRightVal->mAsString());
    auto vRes     = mk_val<value_string>();
    vRes->vValStr = std::move(vOutput);
    return vRes;
  }

  // Python-style string repetition
  // TODO: support array/tuple repetition (e.g., [1, 2] * 3 → [1, 2, 1, 2, 1, 2])
  if (vOp.value == "*" && ((is_val<value_string>(vLeftVal) && is_val<value_int>(vRightVal)) ||
                           (is_val<value_int>(vLeftVal) && is_val<value_string>(vRightVal)))) {
    const auto &  str     = is_val<value_string>(vLeftVal) ? vLeftVal->mAsString() : vRightVal->mAsString();
    const int64_t cRepeat = is_val<value_int>(vRightVal) ? vRightVal->mAsInt() : vLeftVal->mAsInt();
    auto          vRes    = mk_val<value_string>();
    if (cRepeat <= 0) {
      return vRes;
    }
    for (int64_t vI = 0; vI < cRepeat; ++vI) {
      vRes->vValStr.mAppend(str);
    }
    return vRes;
  }

  // String membership
  if (is_val<value_string>(vLeftVal) && is_val<value_string>(vRightVal)) {
    // case: "a" in "abc"
    bool vMember = vTestIsIn();
    if (vOp.value == "in") {
      return mk_val<value_bool>(vMember);
    } else if (vOp.value == "not in") {
      return mk_val<value_bool>(!vMember);
    }
  }

  // Value key in object
  if (is_val<value_object>(vRightVal)) {
    // case: key in {key: value}
    bool vMember = vTestIsIn();
    if (vOp.value == "in") {
      return mk_val<value_bool>(vMember);
    } else if (vOp.value == "not in") {
      return mk_val<value_bool>(!vMember);
    }
  }

  throw std::runtime_error("Unknown operator \"" + vOp.value + "\" between " + vLeftVal->type() + " and " +
                           vRightVal->type());
}

static value fTryBuiltinFunc(context & pCtx, const std::string & name, value & pInput, bool pUndefOnMissing = false) {
  JJ_DEBUG("Trying built-in function '%s' for type %s", name.c_str(), pInput->type().c_str());
  if (pCtx.vIsGetStats) {
    value_t::stats_t::mMarkUsed(pInput);
    pInput->vStats.sOps.insert(name);
  }
  auto dBuiltins = pInput->mGetBuiltins();
  auto vIt       = dBuiltins.find(name);
  if (vIt != dBuiltins.end()) {
    JJ_DEBUG("Binding built-in '%s'", name.c_str());
    return mk_val<value_func>(name, vIt->second, pInput);
  }
  if (pUndefOnMissing) {
    return mk_val<value_undefined>(name);
  }
  throw std::runtime_error("Unknown (built-in) filter '" + name + "' for type " + pInput->type());
}

static value fApplyFilter(context & pCtx, const statement_ptr & pFilter, value pInput) {
  JJ_DEBUG("Applying filter to %s", pInput->type().c_str());

  auto vSetFilterAlias = [](auto & pFilterId) {
    if (pFilterId == "count") {
      pFilterId = "length";
    } else if (pFilterId == "d") {
      pFilterId = "default";
    } else if (pFilterId == "e") {
      pFilterId = "escape";
    } else if (pFilterId == "trim") {
      pFilterId = "strip";
    }
  };

  if (fIsStmt<identifier>(pFilter)) {
    auto vFilterId = fCastStmt<identifier>(pFilter)->vVal;

    vSetFilterAlias(vFilterId);
    JJ_DEBUG("Applying filter '%s' to %s", vFilterId.c_str(), pInput->type().c_str());
    // TODO: Refactor filters so this coercion can be done automatically
    if (!pInput->mIsUndefined() && !is_val<value_string>(pInput) &&
        (vFilterId == "capitalize" || vFilterId == "lower" || vFilterId == "replace" || vFilterId == "strip" ||
         vFilterId == "title" || vFilterId == "upper" || vFilterId == "wordcount")) {
      JJ_DEBUG("Coercing %s to String for '%s' filter", pInput->type().c_str(), vFilterId.c_str());
      pInput = mk_val<value_string>(pInput->mAsString());
    }
    return fTryBuiltinFunc(pCtx, vFilterId, pInput)->mInvoke(func_args(pCtx));

  } else if (fIsStmt<call_expression>(pFilter)) {
    auto vCall = fCastStmt<call_expression>(pFilter);
    if (!fIsStmt<identifier>(vCall->vCallee)) {
      throw std::runtime_error("Filter callee must be an identifier");
    }
    auto vFilterId = fCastStmt<identifier>(vCall->vCallee)->vVal;

    vSetFilterAlias(vFilterId);
    JJ_DEBUG("Applying filter '%s' with arguments to %s", vFilterId.c_str(), pInput->type().c_str());
    func_args vArgs(pCtx);
    for (const auto & vArgExpr : vCall->lArgs) {
      vArgs.push_back(vArgExpr->mExecute(pCtx));
    }

    return fTryBuiltinFunc(pCtx, vFilterId, pInput)->mInvoke(vArgs);

  } else {
    throw std::runtime_error("Invalid filter expression");
  }
}

value filter_expression::mExecuteImpl(context & pCtx) const {
  return fApplyFilter(pCtx, vFilter, vOperand->mExecute(pCtx));
}

value filter_statement::mExecuteImpl(context & pCtx) const {
  // eval body as string, then apply filter
  auto         vBodyVal = fExecStatements(lBody, pCtx);
  value_string vParts   = mk_val<value_string>();
  fGatherStringPartsRecursive(vBodyVal, vParts);

  JJ_DEBUG("FilterStatement: applying filter to body string of length %zu", vParts->vValStr.length());
  return fApplyFilter(pCtx, vFilter, vParts);
}

value test_expression::mExecuteImpl(context & pCtx) const {
  // NOTE: "value is something" translates to function call "test_is_something(value)"
  const auto & dBuiltins = fGlobalBuiltins();

  std::string vTestId;
  value       vInput = vOperand->mExecute(pCtx);

  func_args vArgs(pCtx);
  vArgs.push_back(vInput);

  if (fIsStmt<identifier>(vTest)) {
    vTestId = fCastStmt<identifier>(vTest)->vVal;
  } else if (fIsStmt<call_expression>(vTest)) {
    auto vCall = fCastStmt<call_expression>(vTest);
    if (!fIsStmt<identifier>(vCall->vCallee)) {
      throw std::runtime_error("Test callee must be an identifier");
    }
    vTestId = fCastStmt<identifier>(vCall->vCallee)->vVal;

    JJ_DEBUG("Applying test '%s' with arguments to %s", vTestId.c_str(), vInput->type().c_str());
    for (const auto & vArgExpr : vCall->lArgs) {
      vArgs.push_back(vArgExpr->mExecute(pCtx));
    }

  } else {
    throw std::runtime_error("Invalid test expression");
  }

  const std::string cTestName = "test_is_" + vTestId;
  auto              vIt       = dBuiltins.find(cTestName);
  JJ_DEBUG("Test expression %s '%s' %s (using function '%s')", vOperand->type().c_str(), vTestId.c_str(),
           vNegate ? "(negate)" : "", cTestName.c_str());
  if (vIt == dBuiltins.end()) {
    throw std::runtime_error("Unknown test '" + vTestId + "'");
  }

  if (pCtx.vIsGetStats) {
    value_t::stats_t::mMarkUsed(vInput);
    vInput->vStats.sOps.insert(cTestName);
  }

  auto vRes = vIt->second(vArgs);

  if (vNegate) {
    return mk_val<value_bool>(!vRes->mAsBool());
  } else {
    return vRes;
  }
}

value unary_expression::mExecuteImpl(context & pCtx) const {
  value vOperandVal = vArgument->mExecute(pCtx);
  JJ_DEBUG("Executing unary expression with operator '%s'", vOp.value.c_str());

  if (vOp.value == "not") {
    return mk_val<value_bool>(!vOperandVal->mAsBool());
  } else if (vOp.value == "-") {
    if (is_val<value_int>(vOperandVal)) {
      return mk_val<value_int>(-vOperandVal->mAsInt());
    } else if (is_val<value_float>(vOperandVal)) {
      return mk_val<value_float>(-vOperandVal->mAsFloat());
    } else {
      throw std::runtime_error("Unary - operator requires numeric operand");
    }
  } else if (vOp.value == "+") {
    if (is_val<value_int>(vOperandVal) || is_val<value_float>(vOperandVal)) {
      return vOperandVal;
    }
    throw std::runtime_error("Unary + operator requires numeric operand");
  }

  throw std::runtime_error("Unknown unary operator '" + vOp.value + "'");
}

value if_statement::mExecuteImpl(context & pCtx) const {
  value vTestVal = vTest->mExecute(pCtx);

  auto vOut = mk_val<value_array>();
  if (vTestVal->mAsBool()) {
    for (auto & vStmt : lBody) {
      JJ_DEBUG("IF --> Executing THEN body, current block: %s", vStmt->type().c_str());
      vOut->push_back(vStmt->mExecute(pCtx));
    }
  } else {
    for (auto & vStmt : lAlternate) {
      JJ_DEBUG("IF --> Executing ELSE body, current block: %s", vStmt->type().c_str());
      vOut->push_back(vStmt->mExecute(pCtx));
    }
  }
  // convert to string parts
  value_string str = mk_val<value_string>();
  fGatherStringPartsRecursive(vOut, str);
  return str;
}

value for_statement::mExecuteImpl(context & pCtx) const {
  context vScope(pCtx);  // new scope for loop variables

  const jinja::select_expression * vSelectExpr = fCastStmt<select_expression>(vIterable);
  statement_ptr                    vTestExprNullptr;

  const statement_ptr & vIterExpr = vSelectExpr ? vSelectExpr->vLhs : vIterable;
  const statement_ptr & vTestExpr = vSelectExpr ? vSelectExpr->vTest : vTestExprNullptr;

  JJ_DEBUG("Executing for statement, iterable type: %s", vIterExpr->type().c_str());

  value vIterableVal = vIterExpr->mExecute(vScope);

  // mark the variable being iterated as used for stats
  if (pCtx.vIsGetStats) {
    value_t::stats_t::mMarkUsed(vIterableVal);
    vIterableVal->vStats.sOps.insert("array_access");
  }

  if (vIterableVal->mIsUndefined()) {
    JJ_DEBUG("%s", "For loop iterable is undefined, skipping loop");
    vIterableVal = mk_val<value_array>();
  }

  if (!is_val<value_array>(vIterableVal) && !is_val<value_object>(vIterableVal)) {
    throw std::runtime_error("Expected iterable or object type in for loop: got " + vIterableVal->type());
  }

  std::vector<value> lItems;
  if (is_val<value_object>(vIterableVal)) {
    JJ_DEBUG("%s", "For loop over object keys");
    auto & lObj = vIterableVal->mAsOrderedObject();
    for (auto & vP : lObj) {
      auto vTuple = mk_val<value_tuple>(vP);
      lItems.push_back(std::move(vTuple));
    }
    if (pCtx.vIsGetStats) {
      value_t::stats_t::mMarkUsed(vIterableVal);
      vIterableVal->vStats.sOps.insert("object_access");
    }
  } else {
    JJ_DEBUG("%s", "For loop over array items");
    auto & lArr = vIterableVal->mAsArray();
    for (const auto & vItem : lArr) {
      lItems.push_back(vItem);
    }
    if (pCtx.vIsGetStats) {
      value_t::stats_t::mMarkUsed(vIterableVal);
      vIterableVal->vStats.sOps.insert("array_access");
    }
  }

  std::vector<std::function<void(context &)>> lScopeUpdateFns;

  std::vector<value> lFilteredItems;
  for (size_t vI = 0; vI < lItems.size(); ++vI) {
    value vCurrent = lItems[vI];

    std::function<void(context &)> vScopeUpdateFn = [](context &) { /* no-op */ };
    if (fIsStmt<identifier>(vLoopvar)) {
      auto vId = fCastStmt<identifier>(vLoopvar)->vVal;

      if (is_val<value_object>(vIterableVal)) {
        // case example: {% for key in dict %}
        vCurrent       = lItems[vI]->mAsArray()[0];
        vScopeUpdateFn = [vId, &lItems, vI](context & pCtx) {
          pCtx.mSetVal(vId, lItems[vI]->mAsArray()[0]);
        };
      } else {
        // case example: {% for item in list %}
        vScopeUpdateFn = [vId, &lItems, vI](context & pCtx) {
          pCtx.mSetVal(vId, lItems[vI]);
        };
      }

    } else if (fIsStmt<tuple_literal>(vLoopvar)) {
      // case example: {% for key, value in dict %}
      auto vTuple = fCastStmt<tuple_literal>(vLoopvar);
      if (!is_val<value_array>(vCurrent)) {
        throw std::runtime_error("Cannot unpack non-iterable type: " + vCurrent->type());
      }
      auto & lCArr = vCurrent->mAsArray();
      if (vTuple->lVal.size() != lCArr.size()) {
        throw std::runtime_error(std::string("Too ") + (vTuple->lVal.size() > lCArr.size() ? "few" : "many") +
                                 " items to unpack");
      }
      vScopeUpdateFn = [vTuple, &lItems, vI](context & pCtx) {
        auto & lCArr = lItems[vI]->mAsArray();
        for (size_t vJ = 0; vJ < vTuple->lVal.size(); ++vJ) {
          if (!fIsStmt<identifier>(vTuple->lVal[vJ])) {
            throw std::runtime_error("Cannot unpack non-identifier type: " + vTuple->lVal[vJ]->type());
          }
          auto vId = fCastStmt<identifier>(vTuple->lVal[vJ])->vVal;
          pCtx.mSetVal(vId, lCArr[vJ]);
        }
      };

    } else {
      throw std::runtime_error("Invalid loop variable(s): " + vLoopvar->type());
    }

    if (vSelectExpr && vTestExpr) {
      context vLoopScope(vScope);
      vScopeUpdateFn(vLoopScope);
      value vTestVal = vTestExpr->mExecute(vLoopScope);
      if (!vTestVal->mAsBool()) {
        continue;
      }
    }
    JJ_DEBUG("For loop: adding item type %s at index %zu", vCurrent->type().c_str(), vI);
    lFilteredItems.push_back(vCurrent);
    lScopeUpdateFns.push_back(vScopeUpdateFn);
  }
  JJ_DEBUG("For loop: %zu items after filtering", lFilteredItems.size());

  auto vResult = mk_val<value_array>();

  bool vNoIteration = true;
  for (size_t vI = 0; vI < lFilteredItems.size(); vI++) {
    JJ_DEBUG("For loop iteration %zu/%zu", vI + 1, lFilteredItems.size());
    value_object vLoopObj  = mk_val<value_object>();
    vLoopObj->vHasBuiltins = false;  // loop object has no builtins
    vLoopObj->insert("index", mk_val<value_int>(vI + 1));
    vLoopObj->insert("index0", mk_val<value_int>(vI));
    vLoopObj->insert("revindex", mk_val<value_int>(lFilteredItems.size() - vI));
    vLoopObj->insert("revindex0", mk_val<value_int>(lFilteredItems.size() - vI - 1));
    vLoopObj->insert("first", mk_val<value_bool>(vI == 0));
    vLoopObj->insert("last", mk_val<value_bool>(vI == lFilteredItems.size() - 1));
    vLoopObj->insert("length", mk_val<value_int>(lFilteredItems.size()));
    vLoopObj->insert("previtem", vI > 0 ? lFilteredItems[vI - 1] : mk_val<value_undefined>("previtem"));
    vLoopObj->insert("nextitem",
                     vI < lFilteredItems.size() - 1 ? lFilteredItems[vI + 1] : mk_val<value_undefined>("nextitem"));
    vScope.mSetVal("loop", vLoopObj);
    lScopeUpdateFns[vI](vScope);
    try {
      for (auto & vStmt : lBody) {
        value vVal = vStmt->mExecute(vScope);
        vResult->push_back(vVal);
      }
    } catch (const continue_statement::signal &) {
      continue;
    } catch (const break_statement::signal &) {
      break;
    }
    vNoIteration = false;
  }

  JJ_DEBUG("For loop complete, total iterations: %zu", lFilteredItems.size());
  if (vNoIteration) {
    for (auto & vStmt : lDefaultBlock) {
      value vVal = vStmt->mExecute(pCtx);
      vResult->push_back(vVal);
    }
  }

  // convert to string parts
  value_string str = mk_val<value_string>();
  fGatherStringPartsRecursive(vResult, str);
  return str;
}

value set_statement::mExecuteImpl(context & pCtx) const {
  auto vRhs = vVal ? vVal->mExecute(pCtx) : fExecStatements(lBody, pCtx);

  if (fIsStmt<identifier>(vAssignee)) {
    // case: {% set my_var = value %}
    auto vVarName = fCastStmt<identifier>(vAssignee)->vVal;
    JJ_DEBUG("Setting global variable '%s' with value type %s", vVarName.c_str(), vRhs->type().c_str());
    pCtx.mSetVal(vVarName, vRhs);

  } else if (fIsStmt<tuple_literal>(vAssignee)) {
    // case: {% set a, b = value %}
    auto vTuple = fCastStmt<tuple_literal>(vAssignee);
    if (!is_val<value_array>(vRhs)) {
      throw std::runtime_error("Cannot unpack non-iterable type in set: " + vRhs->type());
    }
    auto & lArr = vRhs->mAsArray();
    if (lArr.size() != vTuple->lVal.size()) {
      throw std::runtime_error(std::string("Too ") + (vTuple->lVal.size() > lArr.size() ? "few" : "many") +
                               " items to unpack in set");
    }
    for (size_t vI = 0; vI < vTuple->lVal.size(); ++vI) {
      auto & vElem = vTuple->lVal[vI];
      if (!fIsStmt<identifier>(vElem)) {
        throw std::runtime_error("Cannot unpack to non-identifier in set: " + vElem->type());
      }
      auto vVarName = fCastStmt<identifier>(vElem)->vVal;
      pCtx.mSetVal(vVarName, lArr[vI]);
    }

  } else if (fIsStmt<member_expression>(vAssignee)) {
    // case: {% set ns.my_var = value %}
    auto vMember = fCastStmt<member_expression>(vAssignee);
    if (vMember->vComputed) {
      throw std::runtime_error("Cannot assign to computed member");
    }
    if (!fIsStmt<identifier>(vMember->vProperty)) {
      throw std::runtime_error("Cannot assign to member with non-identifier property");
    }
    auto vPropName = fCastStmt<identifier>(vMember->vProperty)->vVal;

    value vObject = vMember->vObject->mExecute(pCtx);
    if (!is_val<value_object>(vObject)) {
      throw std::runtime_error("Cannot assign to member of non-object");
    }
    auto vObjPtr = fCastVal<value_object>(vObject);
    JJ_DEBUG("Setting object property '%s' with value type %s", vPropName.c_str(), vRhs->type().c_str());
    vObjPtr->insert(vPropName, vRhs);

  } else {
    throw std::runtime_error("Invalid LHS inside assignment expression: " + vAssignee->type());
  }
  return mk_val<value_undefined>();
}

static inline void fBindParameters(const std::string & name,
                                   const statements &  pThisArgs,
                                   const func_args &   pArgs,
                                   context &           pCtx) {
  const size_t cExpectedCount = pThisArgs.size();
  const size_t cInputCount    = pArgs.count();

  JJ_DEBUG("Invoking '%s' with %zu input arguments (expected %zu)", name.c_str(), cInputCount, cExpectedCount);
  for (size_t vI = 0; vI < cExpectedCount; ++vI) {
    if (vI < cInputCount) {
      if (fIsStmt<identifier>(pThisArgs[vI])) {
        // normal parameter
        std::string vParamName  = fCastStmt<identifier>(pThisArgs[vI])->vVal;
        value       vParamValue = pArgs.mGetKwargOrPos(vParamName, vI);
        JJ_DEBUG("  Binding parameter '%s' to argument of type %s", vParamName.c_str(), vParamValue->type().c_str());
        pCtx.mSetVal(vParamName, vParamValue);
      } else if (fIsStmt<keyword_argument_expression>(pThisArgs[vI])) {
        // default argument used as normal parameter
        auto vKwarg = fCastStmt<keyword_argument_expression>(pThisArgs[vI]);
        if (!fIsStmt<identifier>(vKwarg->vKey)) {
          throw std::runtime_error("Keyword argument key must be an identifier in '" + name + "'");
        }
        std::string vParamName  = fCastStmt<identifier>(vKwarg->vKey)->vVal;
        value       vParamValue = pArgs.mGetKwargOrPos(vParamName, vI);
        JJ_DEBUG("  Binding parameter '%s' to argument of type %s", vParamName.c_str(), vParamValue->type().c_str());
        pCtx.mSetVal(vParamName, vParamValue);
      } else {
        throw std::runtime_error("Invalid parameter type in '" + name + "'");
      }
    } else {
      auto & vDefaultArg = pThisArgs[vI];
      if (fIsStmt<keyword_argument_expression>(vDefaultArg)) {
        auto vKwarg = fCastStmt<keyword_argument_expression>(vDefaultArg);
        if (!fIsStmt<identifier>(vKwarg->vKey)) {
          throw std::runtime_error("Keyword argument key must be an identifier in '" + name + "'");
        }
        std::string vParamName = fCastStmt<identifier>(vKwarg->vKey)->vVal;
        JJ_DEBUG("  Binding parameter '%s' to default argument of type %s", vParamName.c_str(),
                 vKwarg->vVal->type().c_str());
        pCtx.mSetVal(vParamName, vKwarg->vVal->mExecute(pArgs.vCtx));
      } else {
        throw std::runtime_error("Not enough arguments provided to '" + name + "'");
      }
      //std::string param_name = cast_stmt<identifier>(default_args[i])->val;
      //JJ_DEBUG("  Binding parameter '%s' to default", param_name.c_str());
      //ctx.var[param_name] = default_args[i]->execute(ctx);
    }
  }
}

value macro_statement::mExecuteImpl(context & pCtx) const {
  if (!fIsStmt<identifier>(this->name)) {
    throw std::runtime_error("Macro name must be an identifier");
  }
  std::string name = fCastStmt<identifier>(this->name)->vVal;

  const func_handler cFunc = [this, name](const func_args & pArgs) -> value {
    context vMacroCtx(pArgs.vCtx);  // new scope for macro execution

    fBindParameters(name, this->lArgs, pArgs, vMacroCtx);

    // execute macro body
    JJ_DEBUG("Executing macro '%s' body with %zu statements", name.c_str(), this->lBody.size());
    auto vRes = fExecStatements(this->lBody, vMacroCtx);
    JJ_DEBUG("Macro '%s' execution complete, result: %s", name.c_str(), vRes->vValStr.str().c_str());
    return vRes;
  };

  JJ_DEBUG("Defining macro '%s' with %zu parameters", name.c_str(), lArgs.size());
  pCtx.mSetVal(name, mk_val<value_func>(name, cFunc));
  return mk_val<value_undefined>();
}

value call_statement::mExecuteImpl(context & pCtx) const {
  auto vCallExpr = fCastStmt<call_expression>(this->vCall);
  if (!vCallExpr) {
    throw std::runtime_error("Call statement requires a valid call expression");
  }

  value vCalleeVal = vCallExpr->vCallee->mExecute(pCtx);
  if (!is_val<value_func>(vCalleeVal)) {
    throw std::runtime_error("Callee is not a function: got " + vCalleeVal->type());
  }
  auto * vCalleeFunc = fCastVal<value_func>(vCalleeVal);

  context vCallerCtx(pCtx);  // new scope for caller execution

  const func_handler cFunc = [this, vCallerCtx = std::move(vCallerCtx)](const func_args & pArgs) -> value {
    context vBlockCtx(vCallerCtx);  // new scope for block execution

    fBindParameters("caller", this->lCallerArgs, pArgs, vBlockCtx);

    JJ_DEBUG("Executing call body with %zu statements", this->lBody.size());
    auto vRes = fExecStatements(this->lBody, vBlockCtx);
    JJ_DEBUG("Call body execution complete, result: %s", vRes->vValStr.str().c_str());
    return vRes;
  };

  context vCallCtx(pCtx);
  vCallCtx.mSetVal("caller", mk_val<value_func>("caller", cFunc));

  func_args vArgs(vCallCtx);

  for (const auto & vArgExpr : vCallExpr->lArgs) {
    auto vArgVal = vArgExpr->mExecute(pCtx);
    JJ_DEBUG("  Argument type: %s", vArgVal->type().c_str());
    vArgs.push_back(vArgVal);
  }

  JJ_DEBUG("Calling macro '%s' with %zu arguments", vCalleeFunc->name.c_str(), vArgs.count());
  return vCalleeFunc->mInvoke(vArgs);
}

value member_expression::mExecuteImpl(context & pCtx) const {
  value vObject = this->vObject->mExecute(pCtx);

  value vProperty;
  if (this->vComputed) {
    // syntax: obj[expr]
    JJ_DEBUG("Member expression, computing property type %s", this->vProperty->type().c_str());

    int64_t vArrSize = 0;
    if (is_val<value_array>(vObject)) {
      vArrSize = vObject->mAsArray().size();
    } else if (is_val<value_string>(vObject)) {
      vArrSize = vObject->mAsString().length();
    }

    if (fIsStmt<slice_expression>(this->vProperty)) {
      auto  vS        = fCastStmt<slice_expression>(this->vProperty);
      value vStepVal  = vS->vStepExpr ? vS->vStepExpr->mExecute(pCtx) : mk_val<value_int>(1);
      value vStartVal = vS->vStartExpr ?
                          vS->vStartExpr->mExecute(pCtx) :
                          (vStepVal->mAsInt() < 0 ? mk_val<value_int>(vArrSize - 1) : mk_val<value_int>(0));
      value vStopVal  = vS->vStopExpr ? vS->vStopExpr->mExecute(pCtx) :
                                        (vStepVal->mAsInt() < 0 ? mk_val<value_int>(-1) : mk_val<value_int>(vArrSize));

      // translate to function call: obj.slice(start, stop, step)
      JJ_DEBUG("Member expression is a slice: start %s, stop %s, step %s", vStartVal->mAsRepr().c_str(),
               vStopVal->mAsRepr().c_str(), vStepVal->mAsRepr().c_str());
      auto      vSliceFunc = fTryBuiltinFunc(pCtx, "slice", vObject);
      func_args vArgs(pCtx);
      vArgs.push_back(vStartVal);
      vArgs.push_back(vStopVal);
      vArgs.push_back(vStepVal);
      return vSliceFunc->mInvoke(vArgs);
    } else {
      vProperty = this->vProperty->mExecute(pCtx);
    }
  } else if (fIsStmt<integer_literal>(this->vProperty)) {
    // syntax: obj.index
    vProperty = mk_val<value_int>(fCastStmt<integer_literal>(this->vProperty)->vVal);
    if (vProperty->mAsInt() < 0) {
      throw std::runtime_error("Static member property cannot be negative");
    }
  } else {
    // syntax: obj.prop
    if (!fIsStmt<identifier>(this->vProperty)) {
      throw std::runtime_error("Static member property must be an identifier");
    }
    vProperty         = mk_val<value_string>(fCastStmt<identifier>(this->vProperty)->vVal);
    std::string vProp = vProperty->mAsString().str();
    JJ_DEBUG("Member expression, object type %s, static property '%s'", vObject->type().c_str(), vProp.c_str());

    // behavior of jinja2: obj having prop as a built-in function AND 'prop', as an object key,
    // then obj.prop returns the built-in function, not the property value.
    // while obj['prop'] returns the property value.
    // example: {"obj": {"items": 123}} -> obj.items is the built-in function, obj['items'] is 123

    value vVal = fTryBuiltinFunc(pCtx, vProp, vObject, true);
    if (!is_val<value_undefined>(vVal)) {
      return vVal;
    }
    // else, fallthrough to normal property access below
  }

  JJ_DEBUG("Member expression on object type %s, property type %s", vObject->type().c_str(), vProperty->type().c_str());
  value vVal = mk_val<value_undefined>("object_property");

  if (vProperty->mIsUndefined()) {
    JJ_DEBUG("%s", "Member expression property is undefined, returning undefined");
    return vVal;
  }

  fEnsureKeyTypeAllowed(vProperty);

  if (is_val<value_undefined>(vObject)) {
    JJ_DEBUG("%s", "Accessing property on undefined object, returning undefined");
    return vVal;

  } else if (is_val<value_object>(vObject)) {
    auto vKey = vProperty->mAsString().str();
    vVal      = vObject->at(vProperty, vVal);
    if (is_val<value_undefined>(vVal)) {
      vVal = fTryBuiltinFunc(pCtx, vKey, vObject, true);
    }
    JJ_DEBUG("Accessed property '%s' value, got type: %s", vKey.c_str(), vVal->type().c_str());

  } else if (is_val<value_array>(vObject) || is_val<value_string>(vObject)) {
    if (is_val<value_int>(vProperty) || is_val<value_bool>(vProperty)) {
      int64_t vIndex = vProperty->mAsInt();
      JJ_DEBUG("Accessing %s index %d", vObject->type().c_str(), (int) vIndex);
      if (is_val<value_array>(vObject)) {
        auto & lArr = vObject->mAsArray();
        if (vIndex < 0) {
          vIndex += static_cast<int64_t>(lArr.size());
        }
        if (vIndex >= 0 && vIndex < static_cast<int64_t>(lArr.size())) {
          vVal = lArr[vIndex];
        }
      } else {  // value_string
        auto str = vObject->mAsString().str();
        if (vIndex >= 0 && vIndex < static_cast<int64_t>(str.size())) {
          vVal = mk_val<value_string>(std::string(1, str[vIndex]));
        }
      }

    } else if (is_val<value_string>(vProperty)) {
      auto vKey = vProperty->mAsString().str();
      JJ_DEBUG("Accessing %s built-in '%s'", is_val<value_array>(vObject) ? "array" : "string", vKey.c_str());
      vVal = fTryBuiltinFunc(pCtx, vKey, vObject, true);
    }
  } else {
    if (!is_val<value_string>(vProperty)) {
      throw std::runtime_error("Cannot access property with non-string: got " + vProperty->type());
    }
    auto vKey = vProperty->mAsString().str();
    vVal      = fTryBuiltinFunc(pCtx, vKey, vObject, true);
  }

  if (pCtx.vIsGetStats && vVal && vObject && vProperty) {
    value_t::stats_t::mMarkUsed(vVal);
    value_t::stats_t::mMarkUsed(vObject);
    value_t::stats_t::mMarkUsed(vProperty);
    if (is_val<value_object>(vObject) || is_val<value_string>(vProperty) || is_val<value_float>(vProperty) ||
        is_val<value_array>(vProperty) || is_val<value_none>(vProperty)) {
      vObject->vStats.sOps.insert("object_access");
    } else if (is_val<value_int>(vProperty) || is_val<value_bool>(vProperty)) {
      vObject->vStats.sOps.insert("array_access");
    }
  }

  return vVal;
}

value call_expression::mExecuteImpl(context & pCtx) const {
  // gather arguments
  func_args vArgs(pCtx);
  for (auto & vArgStmt : this->lArgs) {
    auto vArgVal = vArgStmt->mExecute(pCtx);
    JJ_DEBUG("  Argument type: %s", vArgVal->type().c_str());
    vArgs.push_back(vArgVal);
  }
  // execute callee
  value vCalleeVal = vCallee->mExecute(pCtx);
  if (!is_val<value_func>(vCalleeVal)) {
    throw std::runtime_error("Callee is not a function: got " + vCalleeVal->type());
  }
  auto * vCalleeFunc = fCastVal<value_func>(vCalleeVal);
  JJ_DEBUG("Calling function '%s' with %zu arguments", vCalleeFunc->name.c_str(), vArgs.count());
  return vCalleeFunc->mInvoke(vArgs);
}

value keyword_argument_expression::mExecuteImpl(context & pCtx) const {
  if (!fIsStmt<identifier>(vKey)) {
    throw std::runtime_error("Keyword argument key must be identifiers");
  }

  std::string vK = fCastStmt<identifier>(vKey)->vVal;
  JJ_DEBUG("Keyword argument expression key: %s, value: %s", vK.c_str(), vVal->type().c_str());

  value vV = vVal->mExecute(pCtx);
  JJ_DEBUG("Keyword argument value executed, type: %s", vV->type().c_str());

  return mk_val<value_kwarg>(vK, vV);
}

std::string runtime::mDebugDumpProgram(const program & pProg, const std::string & pSrc) {
  std::ostringstream vOss;
  size_t             vLvl = 0;
  context            vCtx;
  vCtx.vSrc.reset(new std::string(pSrc));

  auto vIndent = [](size_t pLvl) -> std::string {
    return std::string(pLvl * 2, ' ');
  };

  vCtx.lVisitor = [&](bool pIsLeaf, const statement * node, std::vector<visitor_pair> pChildren) {
    vOss << vIndent(vLvl) << node->type() << ":\n";
    vLvl++;
    if (pIsLeaf) {
      const auto & vPos = node->vPos;
      vOss << vIndent(vLvl) << "(leaf) at " << fGetLineCol(pSrc, vPos) << " in source:\n";
      std::string vSnippet = fPeakSource(pSrc, vPos);
      fStringReplaceAll(vSnippet, "\n", "\n" + vIndent(vLvl));
      vOss << vIndent(vLvl) << vSnippet << "\n";
    } else {
      for (auto & [label, children_vec] : pChildren) {
        vOss << vIndent(vLvl) << label << ":\n";
        vLvl++;
        if (children_vec.empty()) {
          vOss << vIndent(vLvl) << "<empty>\n\n";
        } else {
          for (auto * vChild : children_vec) {
            if (!vChild) {
              continue;
            }
            vChild->mVisit(vCtx);
          }
        }
        vLvl--;
      }
    }
    vLvl--;
  };

  for (const auto & vStmt : pProg.lBody) {
    vStmt->mVisit(vCtx);
  }

  return vOss.str();
}

}  // namespace jinja
