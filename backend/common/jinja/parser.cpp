#include "parser.h"

#include "lexer.h"
#include "runtime.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define FILENAME "jinja-parser"

namespace jinja {

// Helper to check type without asserting (useful for logic)
template <typename T> static bool fIsType(const statement_ptr & pPtr) {
  return dynamic_cast<const T *>(pPtr.get()) != nullptr;
}

class parser {
  const std::vector<token> & lTokens;
  size_t                     vCurrent = 0;

  std::string vSource;  // for error reporting

public:
  parser(const std::vector<token> & pT, const std::string & pSrc) : lTokens(pT), vSource(pSrc) {}

  program mParse() {
    statements lBody;
    while (vCurrent < lTokens.size()) {
      lBody.push_back(mParseAny());
    }
    return program(std::move(lBody));
  }

  // NOTE: start_pos is the token index, used for error reporting
  template <typename T, typename... Args> std::unique_ptr<T> mMkStmt(size_t pStartPos, Args &&... pArgs) {
    auto vPtr = std::make_unique<T>(std::forward<Args>(pArgs)...);
    assert(start_pos < tokens.size());
    vPtr->vPos = lTokens[pStartPos].vPos;
    return vPtr;
  }

private:
  const token & mPeek(size_t pOffset = 0) const {
    if (vCurrent + pOffset >= lTokens.size()) {
      static const token cEndToken{ token::cEof, "", 0 };
      return cEndToken;
    }
    return lTokens[vCurrent + pOffset];
  }

  const token & mNext() {
    if (vCurrent >= lTokens.size()) {
      throw parser_exception("Parser Error: Unexpected EOF", vSource, lTokens.empty() ? 0 : lTokens.back().vPos);
    }
    return lTokens[vCurrent++];
  }

  token mExpect(token::type type, const std::string & pError) {
    const auto & vT = mPeek();
    if (vT.vT != type) {
      throw parser_exception("Parser Error: " + pError + " (Got " + vT.value + ")", vSource, vT.vPos);
    }
    vCurrent++;
    return vT;
  }

  void mExpectIdentifier(const std::string & name) {
    const auto & vT = mPeek();
    if (vT.vT != token::cIdentifier || vT.value != name) {
      throw parser_exception("Expected identifier: " + name, vSource, vT.vPos);
    }
    vCurrent++;
  }

  bool mIs(token::type type) const { return mPeek().vT == type; }

  bool mIsIdentifier(const std::string & name) const {
    return mPeek().vT == token::cIdentifier && mPeek().value == name;
  }

  bool mIsStatement(const std::vector<std::string> & pNames) const {
    if (mPeek(0).vT != token::cOpenStatement || mPeek(1).vT != token::cIdentifier) {
      return false;
    }
    std::string vVal = mPeek(1).value;
    return std::find(pNames.begin(), pNames.end(), vVal) != pNames.end();
  }

  statement_ptr mParseAny() {
    size_t vStartPos = vCurrent;
    switch (mPeek().vT) {
      case token::cComment:
        return mMkStmt<comment_statement>(vStartPos, mNext().value);
      case token::cText:
        return mMkStmt<string_literal>(vStartPos, mNext().value);
      case token::cOpenStatement:
        return mParseJinjaStatement();
      case token::cOpenExpression:
        return mParseJinjaExpression();
      default:
        throw std::runtime_error("Unexpected token type");
    }
  }

  statement_ptr mParseJinjaExpression() {
    // Consume {{ }} tokens
    mExpect(token::cOpenExpression, "Expected {{");
    auto vResult = mParseExpression();
    mExpect(token::cCloseExpression, "Expected }}");
    return vResult;
  }

  statement_ptr mParseJinjaStatement() {
    // Consume {% token
    mExpect(token::cOpenStatement, "Expected {%");

    if (mPeek().vT != token::cIdentifier) {
      throw std::runtime_error("Unknown statement");
    }

    size_t      vStartPos = vCurrent;
    std::string name      = mNext().value;

    statement_ptr vResult;
    if (name == "set") {
      vResult = mParseSetStatement(vStartPos);

    } else if (name == "if") {
      vResult = mParseIfStatement(vStartPos);
      // expect {% endif %}
      mExpect(token::cOpenStatement, "Expected {%");
      mExpectIdentifier("endif");
      mExpect(token::cCloseStatement, "Expected %}");

    } else if (name == "macro") {
      vResult = mParseMacroStatement(vStartPos);
      // expect {% endmacro %}
      mExpect(token::cOpenStatement, "Expected {%");
      mExpectIdentifier("endmacro");
      mExpect(token::cCloseStatement, "Expected %}");

    } else if (name == "for") {
      vResult = mParseForStatement(vStartPos);
      // expect {% endfor %}
      mExpect(token::cOpenStatement, "Expected {%");
      mExpectIdentifier("endfor");
      mExpect(token::cCloseStatement, "Expected %}");

    } else if (name == "break") {
      mExpect(token::cCloseStatement, "Expected %}");
      vResult = mMkStmt<break_statement>(vStartPos);

    } else if (name == "continue") {
      mExpect(token::cCloseStatement, "Expected %}");
      vResult = mMkStmt<continue_statement>(vStartPos);

    } else if (name == "call") {
      statements lCallerArgs;
      // bool has_caller_args = false;
      if (mIs(token::cOpenParen)) {
        // Optional caller arguments, e.g. {% call(user) dump_users(...) %}
        lCallerArgs = mParseArgs();
        // has_caller_args = true;
      }
      auto vCallee = mParsePrimaryExpression();
      if (!fIsType<identifier>(vCallee)) {
        throw std::runtime_error("Expected identifier");
      }

      auto lCallArgs = mParseArgs();
      mExpect(token::cCloseStatement, "Expected %}");

      statements lBody;
      while (!mIsStatement({ "endcall" })) {
        lBody.push_back(mParseAny());
      }

      mExpect(token::cOpenStatement, "Expected {%");
      mExpectIdentifier("endcall");
      mExpect(token::cCloseStatement, "Expected %}");

      auto vCallExpr = mMkStmt<call_expression>(vStartPos, std::move(vCallee), std::move(lCallArgs));
      vResult = mMkStmt<call_statement>(vStartPos, std::move(vCallExpr), std::move(lCallerArgs), std::move(lBody));

    } else if (name == "filter") {
      auto vFilterNode = mParsePrimaryExpression();
      if (fIsType<identifier>(vFilterNode) && mIs(token::cOpenParen)) {
        vFilterNode = mParseCallExpression(std::move(vFilterNode));
      }
      mExpect(token::cCloseStatement, "Expected %}");

      statements lBody;
      while (!mIsStatement({ "endfilter" })) {
        lBody.push_back(mParseAny());
      }

      mExpect(token::cOpenStatement, "Expected {%");
      mExpectIdentifier("endfilter");
      mExpect(token::cCloseStatement, "Expected %}");
      vResult = mMkStmt<filter_statement>(vStartPos, std::move(vFilterNode), std::move(lBody));

    } else if (name == "generation" || name == "endgeneration") {
      // Ignore generation blocks (transformers-specific)
      // See https://github.com/huggingface/transformers/pull/30650 for more information.
      vResult = mMkStmt<noop_statement>(vStartPos);
      ++vCurrent;

    } else {
      throw std::runtime_error("Unknown statement: " + name);
    }
    return vResult;
  }

  statement_ptr mParseSetStatement(size_t pStartPos) {
    // NOTE: `set` acts as both declaration statement and assignment expression
    auto          vLeft = mParseExpressionSequence();
    statement_ptr value = nullptr;
    statements    lBody;

    if (mIs(token::cEquals)) {
      ++vCurrent;
      value = mParseExpressionSequence();
    } else {
      // parsing multiline set here
      mExpect(token::cCloseStatement, "Expected %}");
      while (!mIsStatement({ "endset" })) {
        lBody.push_back(mParseAny());
      }
      mExpect(token::cOpenStatement, "Expected {%");
      mExpectIdentifier("endset");
    }
    mExpect(token::cCloseStatement, "Expected %}");
    return mMkStmt<set_statement>(pStartPos, std::move(vLeft), std::move(value), std::move(lBody));
  }

  statement_ptr mParseIfStatement(size_t pStartPos) {
    auto vTest = mParseExpression();
    mExpect(token::cCloseStatement, "Expected %}");

    statements lBody;
    statements lAlternate;

    // Keep parsing 'if' body until we reach the first {% elif %} or {% else %} or {% endif %}
    while (!mIsStatement({ "elif", "else", "endif" })) {
      lBody.push_back(mParseAny());
    }

    if (mIsStatement({ "elif" })) {
      size_t vPos0 = vCurrent;
      ++vCurrent;                                      // consume {%
      ++vCurrent;                                      // consume 'elif'
      lAlternate.push_back(mParseIfStatement(vPos0));  // nested If
    } else if (mIsStatement({ "else" })) {
      ++vCurrent;                                      // consume {%
      ++vCurrent;                                      // consume 'else'
      mExpect(token::cCloseStatement, "Expected %}");

      // keep going until we hit {% endif %}
      while (!mIsStatement({ "endif" })) {
        lAlternate.push_back(mParseAny());
      }
    }
    return mMkStmt<if_statement>(pStartPos, std::move(vTest), std::move(lBody), std::move(lAlternate));
  }

  statement_ptr mParseMacroStatement(size_t pStartPos) {
    auto name  = mParsePrimaryExpression();
    auto lArgs = mParseArgs();
    mExpect(token::cCloseStatement, "Expected %}");
    statements lBody;
    // Keep going until we hit {% endmacro
    while (!mIsStatement({ "endmacro" })) {
      lBody.push_back(mParseAny());
    }
    return mMkStmt<macro_statement>(pStartPos, std::move(name), std::move(lArgs), std::move(lBody));
  }

  statement_ptr mParseExpressionSequence(bool pPrimary = false) {
    size_t     vStartPos = vCurrent;
    statements lExprs;
    lExprs.push_back(pPrimary ? mParsePrimaryExpression() : mParseExpression());
    bool vIsTuple = mIs(token::cComma);
    while (mIs(token::cComma)) {
      ++vCurrent;  // consume comma
      lExprs.push_back(pPrimary ? mParsePrimaryExpression() : mParseExpression());
    }
    return vIsTuple ? mMkStmt<tuple_literal>(vStartPos, std::move(lExprs)) : std::move(lExprs[0]);
  }

  statement_ptr mParseForStatement(size_t pStartPos) {
    // e.g., `message` in `for message in messages`
    auto vLoopVar = mParseExpressionSequence(true);  // should be an identifier/tuple
    if (!mIsIdentifier("in")) {
      throw std::runtime_error("Expected 'in'");
    }
    ++vCurrent;  // consume 'in'

    // `messages` in `for message in messages`
    auto vIterable = mParseExpression();
    mExpect(token::cCloseStatement, "Expected %}");

    statements lBody;
    statements lAlternate;

    // Keep going until we hit {% endfor or {% else
    while (!mIsStatement({ "endfor", "else" })) {
      lBody.push_back(mParseAny());
    }

    if (mIsStatement({ "else" })) {
      ++vCurrent;  // consume {%
      ++vCurrent;  // consume 'else'
      mExpect(token::cCloseStatement, "Expected %}");
      while (!mIsStatement({ "endfor" })) {
        lAlternate.push_back(mParseAny());
      }
    }
    return mMkStmt<for_statement>(pStartPos, std::move(vLoopVar), std::move(vIterable), std::move(lBody),
                                  std::move(lAlternate));
  }

  statement_ptr mParseExpression() {
    // Choose parse function with lowest precedence
    return mParseIfExpression();
  }

  statement_ptr mParseIfExpression() {
    auto vA = mParseLogicalOrExpression();
    if (mIsIdentifier("if")) {
      // Ternary expression
      size_t vStartPos = vCurrent;
      ++vCurrent;  // consume 'if'
      auto vTest = mParseLogicalOrExpression();
      if (mIsIdentifier("else")) {
        // Ternary expression with else
        size_t vPos0 = vCurrent;
        ++vCurrent;                              // consume 'else'
        auto vFalseExpr = mParseIfExpression();  // recurse to support chained ternaries
        return mMkStmt<ternary_expression>(vPos0, std::move(vTest), std::move(vA), std::move(vFalseExpr));
      } else {
        // Select expression on iterable
        return mMkStmt<select_expression>(vStartPos, std::move(vA), std::move(vTest));
      }
    }
    return vA;
  }

  statement_ptr mParseLogicalOrExpression() {
    auto vLeft = mParseLogicalAndExpression();
    while (mIsIdentifier("or")) {
      size_t vStartPos = vCurrent;
      token  vOp       = mNext();
      vLeft            = mMkStmt<binary_expression>(vStartPos, vOp, std::move(vLeft), mParseLogicalAndExpression());
    }
    return vLeft;
  }

  statement_ptr mParseLogicalAndExpression() {
    auto vLeft = mParseLogicalNegationExpression();
    while (mIsIdentifier("and")) {
      size_t vStartPos = vCurrent;
      auto   vOp       = mNext();
      vLeft = mMkStmt<binary_expression>(vStartPos, vOp, std::move(vLeft), mParseLogicalNegationExpression());
    }
    return vLeft;
  }

  statement_ptr mParseLogicalNegationExpression() {
    // Try parse unary operators
    if (mIsIdentifier("not")) {
      size_t vStartPos = vCurrent;
      auto   vOp       = mNext();
      return mMkStmt<unary_expression>(vStartPos, vOp, mParseLogicalNegationExpression());
    }
    return mParseComparisonExpression();
  }

  statement_ptr mParseComparisonExpression() {
    // NOTE: membership has same precedence as comparison
    // e.g., ('a' in 'apple' == 'b' in 'banana') evaluates as ('a' in ('apple' == ('b' in 'banana')))
    auto vLeft = mParseAdditiveExpression();
    while (true) {
      token  vOp;
      size_t vStartPos = vCurrent;
      if (mIsIdentifier("not") && mPeek(1).vT == token::cIdentifier && mPeek(1).value == "in") {
        vOp = { token::cIdentifier, "not in", lTokens[vCurrent].vPos };
        ++vCurrent;  // consume 'not'
        ++vCurrent;  // consume 'in'
      } else if (mIsIdentifier("in")) {
        vOp = mNext();
      } else if (mIs(token::cComparisonBinaryOperator)) {
        vOp = mNext();
      } else {
        break;
      }
      vLeft = mMkStmt<binary_expression>(vStartPos, vOp, std::move(vLeft), mParseAdditiveExpression());
    }
    return vLeft;
  }

  statement_ptr mParseAdditiveExpression() {
    auto vLeft = mParseMultiplicativeExpression();
    while (mIs(token::cAdditiveBinaryOperator)) {
      size_t vStartPos = vCurrent;
      auto   vOp       = mNext();
      vLeft            = mMkStmt<binary_expression>(vStartPos, vOp, std::move(vLeft), mParseMultiplicativeExpression());
    }
    return vLeft;
  }

  statement_ptr mParseMultiplicativeExpression() {
    auto vLeft = mParseTestExpression();
    while (mIs(token::cMultiplicativeBinaryOperator)) {
      size_t vStartPos = vCurrent;
      auto   vOp       = mNext();
      vLeft            = mMkStmt<binary_expression>(vStartPos, vOp, std::move(vLeft), mParseTestExpression());
    }
    return vLeft;
  }

  statement_ptr mParseTestExpression() {
    auto vOperand = mParseFilterExpression();
    while (mIsIdentifier("is")) {
      size_t vStartPos = vCurrent;
      ++vCurrent;  // consume 'is'
      bool vNegate = false;
      if (mIsIdentifier("not")) {
        ++vCurrent;
        vNegate = true;
      }
      auto vTestId = mParsePrimaryExpression();
      if (mIs(token::cOpenParen)) {
        vTestId = mParseCallExpression(std::move(vTestId));
      } else if (mIs(token::cNumericLiteral) || mIs(token::cStringLiteral) || mIs(token::cOpenCurlyBracket) ||
                 mIs(token::cOpenSquareBracket) ||
                 (mIs(token::cIdentifier) && !mIsIdentifier("and") && !mIsIdentifier("or") && !mIsIdentifier("else"))) {
        size_t     vCallPos = vCurrent;
        statements lArgs;
        lArgs.push_back(mParseUnaryExpression());
        vTestId = mMkStmt<call_expression>(vCallPos, std::move(vTestId), std::move(lArgs));
      }
      vOperand = mMkStmt<test_expression>(vStartPos, std::move(vOperand), vNegate, std::move(vTestId));
    }
    return vOperand;
  }

  statement_ptr mParseFilterExpression() {
    // Filters/tests bind outside unary so -n|abs is (-n)|abs, not -(n|abs).
    auto vOperand = mParseUnaryExpression();
    while (mIs(token::cPipe)) {
      size_t vStartPos = vCurrent;
      ++vCurrent;  // consume pipe
      auto vFilter = mParsePrimaryExpression();
      if (mIs(token::cOpenParen)) {
        vFilter = mParseCallExpression(std::move(vFilter));
      }
      vOperand = mMkStmt<filter_expression>(vStartPos, std::move(vOperand), std::move(vFilter));
    }
    return vOperand;
  }

  statement_ptr mParseUnaryExpression() {
    if (mIs(token::cUnaryOperator)) {
      size_t vStartPos = vCurrent;
      auto   vOp       = mNext();
      return mMkStmt<unary_expression>(vStartPos, vOp, mParseUnaryExpression());
    }
    return mParseCallMemberExpression();
  }

  statement_ptr mParseCallMemberExpression() {
    // Handle member expressions recursively
    auto vMember = mParseMemberExpression(mParsePrimaryExpression());
    return mIs(token::cOpenParen) ? mParseCallExpression(std::move(vMember))  // foo.x()
                                    :
                                    std::move(vMember);
  }

  statement_ptr mParseCallExpression(statement_ptr pCallee) {
    size_t vStartPos = vCurrent;
    auto   vExpr     = mMkStmt<call_expression>(vStartPos, std::move(pCallee), mParseArgs());
    auto   vMember   = mParseMemberExpression(std::move(vExpr));              // foo.x().y
    return mIs(token::cOpenParen) ? mParseCallExpression(std::move(vMember))  // foo.x()()
                                    :
                                    std::move(vMember);
  }

  statements mParseArgs() {
    // comma-separated arguments list
    mExpect(token::cOpenParen, "Expected (");
    statements lArgs;
    while (!mIs(token::cCloseParen)) {
      statement_ptr vArg;
      // unpacking: *expr
      if (mPeek().vT == token::cMultiplicativeBinaryOperator && mPeek().value == "*") {
        size_t vStartPos = vCurrent;
        ++vCurrent;  // consume *
        vArg = mMkStmt<spread_expression>(vStartPos, mParseExpression());
      } else {
        vArg = mParseExpression();
        if (mIs(token::cEquals)) {
          // keyword argument
          // e.g., func(x = 5, y = a or b)
          size_t vStartPos = vCurrent;
          ++vCurrent;  // consume equals
          vArg = mMkStmt<keyword_argument_expression>(vStartPos, std::move(vArg), mParseExpression());
        }
      }
      lArgs.push_back(std::move(vArg));
      if (mIs(token::cComma)) {
        ++vCurrent;  // consume comma
      }
    }
    mExpect(token::cCloseParen, "Expected )");
    return lArgs;
  }

  statement_ptr mParseMemberExpression(statement_ptr pObject) {
    size_t vStartPos = vCurrent;
    while (mIs(token::cDot) || mIs(token::cOpenSquareBracket)) {
      auto          vOp       = mNext();
      bool          vComputed = vOp.vT == token::cOpenSquareBracket;
      statement_ptr vProp;
      if (vComputed) {
        vProp = mParseMemberExpressionArguments();
        mExpect(token::cCloseSquareBracket, "Expected ]");
      } else {
        vProp = mParsePrimaryExpression();
      }
      pObject = mMkStmt<member_expression>(vStartPos, std::move(pObject), std::move(vProp), vComputed);
    }
    return pObject;
  }

  statement_ptr mParseMemberExpressionArguments() {
    // NOTE: This also handles slice expressions colon-separated arguments list
    // e.g., ['test'], [0], [:2], [1:], [1:2], [1:2:3]
    statements lSlices;
    bool       vIsSlice  = false;
    size_t     vStartPos = vCurrent;
    while (!mIs(token::cCloseSquareBracket)) {
      if (mIs(token::cColon)) {
        // A case where a default is used
        // e.g., [:2] will be parsed as [undefined, 2]
        lSlices.push_back(nullptr);
        ++vCurrent;  // consume colon
        vIsSlice = true;
      } else {
        lSlices.push_back(mParseExpression());
        if (mIs(token::cColon)) {
          ++vCurrent;  // consume colon after expression, if it exists
          vIsSlice = true;
        }
      }
    }
    if (vIsSlice) {
      statement_ptr vStart = lSlices.size() > 0 ? std::move(lSlices[0]) : nullptr;
      statement_ptr vStop  = lSlices.size() > 1 ? std::move(lSlices[1]) : nullptr;
      statement_ptr vStep  = lSlices.size() > 2 ? std::move(lSlices[2]) : nullptr;
      return mMkStmt<slice_expression>(vStartPos, std::move(vStart), std::move(vStop), std::move(vStep));
    }
    if (lSlices.empty()) {
      return mMkStmt<blank_expression>(vStartPos);
    }
    return std::move(lSlices[0]);
  }

  statement_ptr mParsePrimaryExpression() {
    size_t vStartPos = vCurrent;
    auto   vT        = mNext();
    switch (vT.vT) {
      case token::cNumericLiteral:
        if (vT.value.find('.') != std::string::npos) {
          return mMkStmt<float_literal>(vStartPos, std::stod(vT.value));
        } else {
          return mMkStmt<integer_literal>(vStartPos, std::stoll(vT.value));
        }
      case token::cStringLiteral:
        {
          std::string vVal = vT.value;
          while (mIs(token::cStringLiteral)) {
            vVal += mNext().value;
          }
          return mMkStmt<string_literal>(vStartPos, vVal);
        }
      case token::cIdentifier:
        return mMkStmt<identifier>(vStartPos, vT.value);
      case token::cOpenParen:
        {
          auto vExpr = mParseExpressionSequence();
          mExpect(token::cCloseParen, "Expected )");
          return vExpr;
        }
      case token::cOpenSquareBracket:
        {
          statements lVals;
          while (!mIs(token::cCloseSquareBracket)) {
            lVals.push_back(mParseExpression());
            if (mIs(token::cComma)) {
              ++vCurrent;
            }
          }
          ++vCurrent;
          return mMkStmt<array_literal>(vStartPos, std::move(lVals));
        }
      case token::cOpenCurlyBracket:
        {
          std::vector<std::pair<statement_ptr, statement_ptr>> lPairs;
          while (!mIs(token::cCloseCurlyBracket)) {
            auto vKey = mParseExpression();
            mExpect(token::cColon, "Expected :");
            lPairs.push_back({ std::move(vKey), mParseExpression() });
            if (mIs(token::cComma)) {
              ++vCurrent;
            }
          }
          ++vCurrent;
          return mMkStmt<object_literal>(vStartPos, std::move(lPairs));
        }
      default:
        throw std::runtime_error("Unexpected token: " + vT.value + " of type " + std::to_string(vT.vT));
    }
  }
};

program fParseFromTokens(const lexer_result & pLexerRes) {
  return parser(pLexerRes.lTokens, pLexerRes.vSource).mParse();
}

}  // namespace jinja
