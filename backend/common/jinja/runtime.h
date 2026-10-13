#pragma once

#include "lexer.h"
#include "value.h"

#include <cassert>
#include <ctime>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#define JJ_DEBUG(msg, ...)                                            \
  do {                                                                \
    if (g_jinja_debug)                                                \
      printf("%s:%-3d : " msg "\n", FILENAME, __LINE__, __VA_ARGS__); \
  } while (0)

extern bool g_jinja_debug;

namespace jinja {

struct statement;
using statement_ptr = std::unique_ptr<statement>;
using statements    = std::vector<statement_ptr>;

// Helpers for dynamic casting and type checking
template <typename T> struct extract_pointee_unique {
  using type = T;
};

template <typename U> struct extract_pointee_unique<std::unique_ptr<U>> {
  using type = U;
};

template <typename T> bool fIsStmt(const statement_ptr & pPtr) {
  return dynamic_cast<const T *>(pPtr.get()) != nullptr;
}

template <typename T> T * fCastStmt(statement_ptr & pPtr) {
  return dynamic_cast<T *>(pPtr.get());
}

template <typename T> const T * fCastStmt(const statement_ptr & pPtr) {
  return dynamic_cast<const T *>(pPtr.get());
}

// End Helpers

// not thread-safe
void fEnableDebug(bool pEnable);

// for visiting AST nodes
// function signature: void(bool is_leaf, const statement * node, pair of <label, children>)
using visitor_pair = std::pair<std::string, std::vector<const statement *>>;
using visitor_fn   = std::function<void(bool, const statement *, std::vector<visitor_pair>)>;

struct context {
  std::shared_ptr<std::string> vSrc;          // for debugging; use shared_ptr to avoid copying on scope creation
  std::time_t                  vCurrentTime;  // for functions that need current time

  bool vIsGetStats = false;                   // whether to collect stats

  visitor_fn lVisitor;

  // src is optional, used for error reporting
  context(std::string pSrc = "") : vSrc(std::make_shared<std::string>(std::move(pSrc))) {
    vEnv               = mk_val<value_object>();
    vEnv->vHasBuiltins = false;  // context object has no builtins
    vEnv->insert("true", mk_val<value_bool>(true));
    vEnv->insert("True", mk_val<value_bool>(true));
    vEnv->insert("false", mk_val<value_bool>(false));
    vEnv->insert("False", mk_val<value_bool>(false));
    vEnv->insert("none", mk_val<value_none>());
    vEnv->insert("None", mk_val<value_none>());
    vCurrentTime = std::time(nullptr);
  }

  ~context() = default;

  context(const context & pParent) : context() {
    // inherit variables (for example, when entering a new scope)
    auto & lPvar = pParent.vEnv->mAsOrderedObject();
    for (const auto & vPair : lPvar) {
      mSetVal(vPair.first, vPair.second);
    }
    vCurrentTime = pParent.vCurrentTime;
    vIsGetStats  = pParent.vIsGetStats;
    vSrc         = pParent.vSrc;
  }

  value mGetVal(const std::string & name) {
    value vDefaultVal = mk_val<value_undefined>(name);
    return vEnv->at(name, vDefaultVal);
  }

  void mSetVal(const std::string & name, const value & pVal) { vEnv->insert(name, pVal); }

  void mSetVal(const value & name, const value & pVal) { vEnv->insert(name, pVal); }

  void mPrintVars() const { printf("Context Variables:\n%s\n", fValueToJson(vEnv, 2).c_str()); }

private:
  value_object vEnv;
};

// utils for visiting AST nodes
static std::vector<const statement *> fStmtsToPtr(const statements & pStmts) {
  std::vector<const statement *> lChildren;
  for (const auto & vStmt : pStmts) {
    lChildren.push_back(vStmt.get());
  }
  return lChildren;
}

/**
 * Base class for all nodes in the AST.
 * The AST is shared between threads, so visit and execute must be const.
 */
struct statement {
  size_t vPos;  // position in source, for debugging
  virtual ~statement() = default;

  virtual std::string type() const { return "Statement"; }

  virtual void mVisit(context & pCtx) const { pCtx.lVisitor(true, this, {}); }

  // execute_impl must be overridden by derived classes
  virtual value mExecuteImpl(context &) const { mThrowExecError(); }

  // execute is the public method to execute a statement with error handling
  value mExecute(context &) const;

private:
  [[noreturn]] void mThrowExecError() const { throw std::runtime_error("cannot exec " + type()); }
};

// Type Checking Utilities

template <typename T> static void fChkType(const statement_ptr & pPtr) {
  if (!pPtr) {
    return;  // Allow null for optional fields
  }
  assert(dynamic_cast<T *>(ptr.get()) != nullptr);
}

template <typename T, typename U> static void fChkType(const statement_ptr & pPtr) {
  if (!pPtr) {
    return;
  }
  assert(dynamic_cast<T *>(ptr.get()) != nullptr || dynamic_cast<U *>(ptr.get()) != nullptr);
}

// Base Types

/**
 * Expressions will result in a value at runtime (unlike statements).
 */
struct expression : public statement {
  std::string type() const override { return "Expression"; }
};

// Statements

struct program : public statement {
  statements lBody;

  program() = default;

  explicit program(statements && pBody) : lBody(std::move(pBody)) {}

  std::string type() const override { return "Program"; }

  [[noreturn]] value mExecuteImpl(context &) const override {
    throw std::runtime_error("Cannot execute program directly, use jinja::runtime instead");
  }
};

struct if_statement : public statement {
  statement_ptr vTest;
  statements    lBody;
  statements    lAlternate;

  if_statement(statement_ptr && pTest, statements && pBody, statements && pAlternate) :
    vTest(std::move(pTest)),
    lBody(std::move(pBody)),
    lAlternate(std::move(pAlternate)) {
    fChkType<expression>(this->vTest);
  }

  std::string type() const override { return "If"; }

  value mExecuteImpl(context & pCtx) const override;

  void mVisit(context & pCtx) const override {
    pCtx.lVisitor(
      false, this,
      {
        { "test",      { vTest.get() }         },
        { "body",      fStmtsToPtr(lBody)      },
        { "alternate", fStmtsToPtr(lAlternate) }
    });
  }
};

struct identifier;
struct tuple_literal;

/**
 * Loop over each item in a sequence
 * https://jinja.palletsprojects.com/en/3.0.x/templates/#for
 */
struct for_statement : public statement {
  statement_ptr vLoopvar;  // Identifier | TupleLiteral
  statement_ptr vIterable;
  statements    lBody;
  statements    lDefaultBlock;  // if no iteration took place

  for_statement(statement_ptr && pLoopvar,
                statement_ptr && pIterable,
                statements &&    pBody,
                statements &&    pDefaultBlock) :
    vLoopvar(std::move(pLoopvar)),
    vIterable(std::move(pIterable)),
    lBody(std::move(pBody)),
    lDefaultBlock(std::move(pDefaultBlock)) {
    fChkType<identifier, tuple_literal>(this->vLoopvar);
    fChkType<expression>(this->vIterable);
  }

  std::string type() const override { return "For"; }

  value mExecuteImpl(context & pCtx) const override;

  void mVisit(context & pCtx) const override {
    pCtx.lVisitor(false, this,
                  {
                    { "loopvar",       { vLoopvar.get() }         },
                    { "iterable",      { vIterable.get() }        },
                    { "body",          fStmtsToPtr(lBody)         },
                    { "default_block", fStmtsToPtr(lDefaultBlock) }
    });
  }
};

struct break_statement : public statement {
  std::string type() const override { return "Break"; }

  struct signal : public std::exception {
    const char * what() const noexcept override { return "Break statement executed"; }
  };

  [[noreturn]] value mExecuteImpl(context &) const override { throw break_statement::signal(); }
};

struct continue_statement : public statement {
  std::string type() const override { return "Continue"; }

  struct signal : public std::exception {
    const char * what() const noexcept override { return "Continue statement executed"; }
  };

  [[noreturn]] value mExecuteImpl(context &) const override { throw continue_statement::signal(); }
};

// do nothing
struct noop_statement : public statement {
  std::string type() const override { return "Noop"; }

  value mExecuteImpl(context &) const override { return mk_val<value_undefined>(); }
};

struct set_statement : public statement {
  statement_ptr vAssignee;
  statement_ptr vVal;
  statements    lBody;

  set_statement(statement_ptr && pAssignee, statement_ptr && value, statements && pBody) :
    vAssignee(std::move(pAssignee)),
    vVal(std::move(value)),
    lBody(std::move(pBody)) {
    fChkType<expression>(this->vAssignee);
    fChkType<expression>(this->vVal);
  }

  std::string type() const override { return "Set"; }

  value mExecuteImpl(context & pCtx) const override;

  void mVisit(context & pCtx) const override {
    pCtx.lVisitor(false, this,
                  {
                    { "assignee", { vAssignee.get() } },
                    { "value",    { vVal.get() }      },
                    { "body",     fStmtsToPtr(lBody)  }
    });
  }
};

struct macro_statement : public statement {
  statement_ptr name;
  statements    lArgs;
  statements    lBody;

  macro_statement(statement_ptr && name, statements && pArgs, statements && pBody) :
    name(std::move(name)),
    lArgs(std::move(pArgs)),
    lBody(std::move(pBody)) {
    fChkType<identifier>(this->name);
    for (const auto & vArg : this->lArgs) {
      fChkType<expression>(vArg);
    }
  }

  std::string type() const override { return "Macro"; }

  value mExecuteImpl(context & pCtx) const override;

  void mVisit(context & pCtx) const override {
    pCtx.lVisitor(false, this,
                  {
                    { "name", { name.get() }     },
                    { "args", fStmtsToPtr(lArgs) },
                    { "body", fStmtsToPtr(lBody) }
    });
  }
};

struct comment_statement : public statement {
  std::string vVal;

  explicit comment_statement(const std::string & pV) : vVal(pV) {}

  std::string type() const override { return "Comment"; }

  value mExecuteImpl(context &) const override { return mk_val<value_undefined>(); }
};

// Expressions

// Represents an omitted expression in a computed member, e.g. `a[]`.
struct blank_expression : public expression {
  std::string type() const override { return "BlankExpression"; }

  value mExecuteImpl(context &) const override { return mk_val<value_undefined>(); }
};

struct member_expression : public expression {
  statement_ptr vObject;
  statement_ptr vProperty;
  bool          vComputed;  // true if obj[expr] and false if obj.prop

  member_expression(statement_ptr && pObject, statement_ptr && pProperty, bool pComputed) :
    vObject(std::move(pObject)),
    vProperty(std::move(pProperty)),
    vComputed(pComputed) {
    fChkType<expression>(this->vObject);
    fChkType<expression>(this->vProperty);
  }

  std::string type() const override { return "MemberExpression"; }

  value mExecuteImpl(context & pCtx) const override;

  void mVisit(context & pCtx) const override {
    pCtx.lVisitor(false, this,
                  {
                    { "object",   { vObject.get() }   },
                    { "property", { vProperty.get() } }
    });
  }
};

struct call_expression : public expression {
  statement_ptr vCallee;
  statements    lArgs;

  call_expression(statement_ptr && pCallee, statements && pArgs) :
    vCallee(std::move(pCallee)),
    lArgs(std::move(pArgs)) {
    fChkType<expression>(this->vCallee);
    for (const auto & vArg : this->lArgs) {
      fChkType<expression>(vArg);
    }
  }

  std::string type() const override { return "CallExpression"; }

  value mExecuteImpl(context & pCtx) const override;

  void mVisit(context & pCtx) const override {
    pCtx.lVisitor(false, this,
                  {
                    { "callee", { vCallee.get() }  },
                    { "args",   fStmtsToPtr(lArgs) }
    });
  }
};

/**
 * Represents a user-defined variable or symbol in the template.
 */
struct identifier : public expression {
  std::string vVal;

  explicit identifier(const std::string & pVal) : vVal(pVal) {}

  std::string type() const override { return "Identifier"; }

  value mExecuteImpl(context & pCtx) const override;
};

// Literals

struct integer_literal : public expression {
  int64_t vVal;

  explicit integer_literal(int64_t pVal) : vVal(pVal) {}

  std::string type() const override { return "IntegerLiteral"; }

  value mExecuteImpl(context &) const override { return mk_val<value_int>(vVal); }
};

struct float_literal : public expression {
  double vVal;

  explicit float_literal(double pVal) : vVal(pVal) {}

  std::string type() const override { return "FloatLiteral"; }

  value mExecuteImpl(context &) const override { return mk_val<value_float>(vVal); }
};

struct string_literal : public expression {
  std::string vVal;

  explicit string_literal(const std::string & pVal) : vVal(pVal) {}

  std::string type() const override { return "StringLiteral"; }

  value mExecuteImpl(context &) const override { return mk_val<value_string>(vVal); }
};

struct array_literal : public expression {
  statements lVal;

  explicit array_literal(statements && pVal) : lVal(std::move(pVal)) {
    for (const auto & vItem : this->lVal) {
      fChkType<expression>(vItem);
    }
  }

  std::string type() const override { return "ArrayLiteral"; }

  value mExecuteImpl(context & pCtx) const override {
    auto vArr = mk_val<value_array>();
    for (const auto & vItemStmt : lVal) {
      vArr->push_back(vItemStmt->mExecute(pCtx));
    }
    return vArr;
  }
};

struct tuple_literal : public expression {
  statements lVal;

  explicit tuple_literal(statements && pVal) : lVal(std::move(pVal)) {
    for (const auto & vItem : this->lVal) {
      fChkType<expression>(vItem);
    }
  }

  std::string type() const override { return "TupleLiteral"; }

  value mExecuteImpl(context & pCtx) const override {
    auto vArr = mk_val<value_array>();
    for (const auto & vItemStmt : lVal) {
      vArr->push_back(vItemStmt->mExecute(pCtx));
    }
    return mk_val<value_tuple>(std::move(vArr->mAsArray()));
  }
};

struct object_literal : public expression {
  std::vector<std::pair<statement_ptr, statement_ptr>> lVal;

  explicit object_literal(std::vector<std::pair<statement_ptr, statement_ptr>> && pVal) : lVal(std::move(pVal)) {
    for (const auto & vPair : this->lVal) {
      fChkType<expression>(vPair.first);
      fChkType<expression>(vPair.second);
    }
  }

  std::string type() const override { return "ObjectLiteral"; }

  value mExecuteImpl(context & pCtx) const override;
};

// Complex Expressions

/**
 * An operation with two sides, separated by an operator.
 * Note: Either side can be a Complex Expression, with order
 * of operations being determined by the operator.
 */
struct binary_expression : public expression {
  token         vOp;
  statement_ptr vLeft;
  statement_ptr vRight;

  binary_expression(token pOp, statement_ptr && pLeft, statement_ptr && pRight) :
    vOp(std::move(pOp)),
    vLeft(std::move(pLeft)),
    vRight(std::move(pRight)) {
    fChkType<expression>(this->vLeft);
    fChkType<expression>(this->vRight);
  }

  std::string type() const override { return "BinaryExpression"; }

  value mExecuteImpl(context & pCtx) const override;

  void mVisit(context & pCtx) const override {
    pCtx.lVisitor(false, this,
                  {
                    { "left",  { vLeft.get() }  },
                    { "right", { vRight.get() } }
    });
  }
};

/**
 * An operation with two sides, separated by the | operator.
 * Operator precedence: https://github.com/pallets/jinja/issues/379#issuecomment-168076202
 */
struct filter_expression : public expression {
  statement_ptr vOperand;
  statement_ptr vFilter;

  filter_expression(statement_ptr && pOperand, statement_ptr && pFilter) :
    vOperand(std::move(pOperand)),
    vFilter(std::move(pFilter)) {
    fChkType<expression>(this->vOperand);
    fChkType<identifier, call_expression>(this->vFilter);
  }

  std::string type() const override { return "FilterExpression"; }

  value mExecuteImpl(context & pCtx) const override;

  void mVisit(context & pCtx) const override {
    pCtx.lVisitor(false, this,
                  {
                    { "operand", { vOperand.get() } },
                    { "filter",  { vFilter.get() }  }
    });
  }
};

struct filter_statement : public statement {
  statement_ptr vFilter;
  statements    lBody;

  filter_statement(statement_ptr && pFilter, statements && pBody) :
    vFilter(std::move(pFilter)),
    lBody(std::move(pBody)) {
    fChkType<identifier, call_expression>(this->vFilter);
  }

  std::string type() const override { return "FilterStatement"; }

  value mExecuteImpl(context & pCtx) const override;

  void mVisit(context & pCtx) const override {
    pCtx.lVisitor(false, this,
                  {
                    { "filter", { vFilter.get() }  },
                    { "body",   fStmtsToPtr(lBody) }
    });
  }
};

/**
 * An operation which filters a sequence of objects by applying a test to each object,
 * and only selecting the objects with the test succeeding.
 *
 * It may also be used as a shortcut for a ternary operator.
 */
struct select_expression : public expression {
  statement_ptr vLhs;
  statement_ptr vTest;

  select_expression(statement_ptr && pLhs, statement_ptr && pTest) : vLhs(std::move(pLhs)), vTest(std::move(pTest)) {
    fChkType<expression>(this->vLhs);
    fChkType<expression>(this->vTest);
  }

  std::string type() const override { return "SelectExpression"; }

  value mExecuteImpl(context & pCtx) const override {
    auto vPredicate = vTest->mExecuteImpl(pCtx);
    if (!vPredicate->mAsBool()) {
      return mk_val<value_undefined>();
    }
    return vLhs->mExecuteImpl(pCtx);
  }

  void mVisit(context & pCtx) const override {
    pCtx.lVisitor(false, this,
                  {
                    { "lhs",  { vLhs.get() }  },
                    { "test", { vTest.get() } }
    });
  }
};

/**
 * An operation with two sides, separated by the "is" operator.
 * NOTE: "value is something" translates to function call "test_is_something(value)"
 */
struct test_expression : public expression {
  statement_ptr vOperand;
  bool          vNegate;
  statement_ptr vTest;

  test_expression(statement_ptr && pOperand, bool pNegate, statement_ptr && pTest) :
    vOperand(std::move(pOperand)),
    vNegate(pNegate),
    vTest(std::move(pTest)) {
    fChkType<expression>(this->vOperand);
    fChkType<identifier, call_expression>(this->vTest);
  }

  std::string type() const override { return "TestExpression"; }

  value mExecuteImpl(context & pCtx) const override;

  void mVisit(context & pCtx) const override {
    pCtx.lVisitor(false, this,
                  {
                    { "operand", { vOperand.get() } },
                    { "test",    { vTest.get() }    }
    });
  }
};

/**
 * An operation with one side (operator on the left).
 */
struct unary_expression : public expression {
  token         vOp;
  statement_ptr vArgument;

  unary_expression(token pOp, statement_ptr && pArgument) : vOp(std::move(pOp)), vArgument(std::move(pArgument)) {
    fChkType<expression>(this->vArgument);
  }

  std::string type() const override { return "UnaryExpression"; }

  value mExecuteImpl(context & pCtx) const override;

  void mVisit(context & pCtx) const override {
    pCtx.lVisitor(false, this,
                  {
                    { "argument", { vArgument.get() } }
    });
  }
};

struct slice_expression : public expression {
  statement_ptr vStartExpr;
  statement_ptr vStopExpr;
  statement_ptr vStepExpr;

  slice_expression(statement_ptr && pStartExpr, statement_ptr && pStopExpr, statement_ptr && pStepExpr) :
    vStartExpr(std::move(pStartExpr)),
    vStopExpr(std::move(pStopExpr)),
    vStepExpr(std::move(pStepExpr)) {
    fChkType<expression>(this->vStartExpr);
    fChkType<expression>(this->vStopExpr);
    fChkType<expression>(this->vStepExpr);
  }

  std::string type() const override { return "SliceExpression"; }

  [[noreturn]] value mExecuteImpl(context &) const override {
    throw std::runtime_error("must be handled by MemberExpression");
  }

  void mVisit(context & pCtx) const override {
    pCtx.lVisitor(false, this,
                  {
                    { "start_expr", { vStartExpr.get() } },
                    { "stop_expr",  { vStopExpr.get() }  },
                    { "step_expr",  { vStepExpr.get() }  }
    });
  }
};

struct keyword_argument_expression : public expression {
  statement_ptr vKey;
  statement_ptr vVal;

  keyword_argument_expression(statement_ptr && pKey, statement_ptr && pVal) :
    vKey(std::move(pKey)),
    vVal(std::move(pVal)) {
    fChkType<identifier>(this->vKey);
    fChkType<expression>(this->vVal);
  }

  std::string type() const override { return "KeywordArgumentExpression"; }

  value mExecuteImpl(context & pCtx) const override;

  void mVisit(context & pCtx) const override {
    pCtx.lVisitor(false, this,
                  {
                    { "key", { vKey.get() } },
                    { "val", { vVal.get() } }
    });
  }
};

struct spread_expression : public expression {
  statement_ptr vArgument;

  explicit spread_expression(statement_ptr && pArgument) : vArgument(std::move(pArgument)) {
    fChkType<expression>(this->vArgument);
  }

  std::string type() const override { return "SpreadExpression"; }

  void mVisit(context & pCtx) const override {
    pCtx.lVisitor(false, this,
                  {
                    { "argument", { vArgument.get() } }
    });
  }
};

struct call_statement : public statement {
  statement_ptr vCall;
  statements    lCallerArgs;
  statements    lBody;

  call_statement(statement_ptr && pCall, statements && pCallerArgs, statements && pBody) :
    vCall(std::move(pCall)),
    lCallerArgs(std::move(pCallerArgs)),
    lBody(std::move(pBody)) {
    fChkType<call_expression>(this->vCall);
    for (const auto & vArg : this->lCallerArgs) {
      fChkType<expression>(vArg);
    }
  }

  std::string type() const override { return "CallStatement"; }

  value mExecuteImpl(context & pCtx) const override;

  void mVisit(context & pCtx) const override {
    pCtx.lVisitor(
      false, this,
      {
        { "call",        { vCall.get() }          },
        { "caller_args", fStmtsToPtr(lCallerArgs) },
        { "body",        fStmtsToPtr(lBody)       }
    });
  }
};

struct ternary_expression : public expression {
  statement_ptr vCondition;
  statement_ptr vTrueExpr;
  statement_ptr vFalseExpr;

  ternary_expression(statement_ptr && pCondition, statement_ptr && pTrueExpr, statement_ptr && pFalseExpr) :
    vCondition(std::move(pCondition)),
    vTrueExpr(std::move(pTrueExpr)),
    vFalseExpr(std::move(pFalseExpr)) {
    fChkType<expression>(this->vCondition);
    fChkType<expression>(this->vTrueExpr);
    fChkType<expression>(this->vFalseExpr);
  }

  std::string type() const override { return "Ternary"; }

  value mExecuteImpl(context & pCtx) const override {
    value vCondVal = vCondition->mExecute(pCtx);
    if (vCondVal->mAsBool()) {
      return vTrueExpr->mExecute(pCtx);
    } else {
      return vFalseExpr->mExecute(pCtx);
    }
  }

  void mVisit(context & pCtx) const override {
    pCtx.lVisitor(false, this,
                  {
                    { "condition",  { vCondition.get() } },
                    { "true_expr",  { vTrueExpr.get() }  },
                    { "false_expr", { vFalseExpr.get() } }
    });
  }
};

struct raised_exception : public std::exception {
  std::string vMessage;

  raised_exception(const std::string & pMsg) : vMessage(pMsg) {}

  const char * what() const noexcept override { return vMessage.c_str(); }
};

// Used to rethrow exceptions with modified messages
struct rethrown_exception : public std::exception {
  std::string vMessage;

  rethrown_exception(const std::string & pMsg) : vMessage(pMsg) {}

  const char * what() const noexcept override { return vMessage.c_str(); }
};

//////////////////////

static void fGatherStringPartsRecursive(const value & pVal, value_string & pParts) {
  // TODO: probably allow print value_none as "None" string? currently this breaks some templates
  if (is_val<value_string>(pVal)) {
    const auto & vStrVal = fCastVal<value_string>(pVal)->vValStr;
    pParts->vValStr.mAppend(vStrVal);
  } else if (is_val<value_int>(pVal) || is_val<value_float>(pVal) || is_val<value_bool>(pVal)) {
    std::string vStrVal = pVal->mAsString().str();
    pParts->vValStr.mAppend(vStrVal);
  } else if (is_val<value_array>(pVal)) {
    auto lItems = fCastVal<value_array>(pVal)->mAsArray();
    for (const auto & vItem : lItems) {
      fGatherStringPartsRecursive(vItem, pParts);
    }
  }
}

static std::string fRenderStringParts(const value_string & pParts) {
  std::ostringstream vOss;
  for (const auto & vPart : pParts->vValStr.lParts) {
    vOss << vPart.vVal;
  }
  return vOss.str();
}

struct runtime {
  context & vCtx;

  explicit runtime(context & pCtx) : vCtx(pCtx) {}

  value_array mExecute(const program & pProg) {
    value_array vResults = mk_val<value_array>();
    for (const auto & vStmt : pProg.lBody) {
      value vRes = vStmt->mExecute(vCtx);
      vResults->push_back(std::move(vRes));
    }
    return vResults;
  }

  static value_string mGatherStringParts(const value & pVal) {
    value_string vParts = mk_val<value_string>();
    fGatherStringPartsRecursive(pVal, vParts);
    // join consecutive parts with the same type
    auto & lP = vParts->vValStr.lParts;
    if (lP.empty()) {
      return vParts;
    }
    size_t vW = 0;
    for (size_t vR = 1; vR < lP.size(); vR++) {
      if (lP[vW].vIsInput == lP[vR].vIsInput) {
        lP[vW].vVal += lP[vR].vVal;
      } else {
        vW++;
        if (vW != vR) {
          // the guard is needed, self-move leaves the string in an unspecified state
          lP[vW] = std::move(lP[vR]);
        }
      }
    }
    lP.resize(vW + 1);
    return vParts;
  }

  static std::string mDebugDumpProgram(const program & pProg, const std::string & pSrc);
};

}  // namespace jinja
