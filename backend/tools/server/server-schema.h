#pragma once

#include "sampling.h"
#include "server-common.h"
#include "server-task.h"
#include "speculative.h"

#include <climits>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace server_schema {

struct field_eval_context {
  task_params &                       vParams;
  const dgs_vocab *                   vVocab        = nullptr;
  const std::vector<dgs_logit_bias> * lLogitBiasEog = nullptr;

  field_eval_context(task_params & pParams) : vParams(pParams) {}
};

using field_handler = std::function<void(field_eval_context &, const json &)>;

struct field {
  std::vector<const char *> name;
  const char *              vDesc = "";
  field_handler             vCustomHandler;
  field() = default;

  field(const char * n) : name({ n }) {}

  virtual ~field() = default;

  field * mSetDesc(const char * pS) {
    vDesc = pS;
    return this;
  }

  // if 'name' is present, use it, otherwise look for aliases following the order they were added
  field * mAddAlias(const char * n) {
    name.push_back(n);
    return this;
  }

  field * mSetHandler(field_handler pH) {
    this->vCustomHandler = pH;
    return this;
  }

  virtual void mEval(field_eval_context & pCtx, const json & data) = 0;
};

template <typename T = int32_t> struct field_num : public field {
  T &  vVal;
  T    min          = std::numeric_limits<T>::lowest();
  T    max          = std::numeric_limits<T>::max();
  bool vIsHardLimit = false;  // if true, throw error if the value is invalid

  field_num(const char * n, T & pVal) : field(n), vVal(pVal) {}

  // limits are inclusive, min <= value <= max
  field_num * mSetLimits(T min, T max) {
    this->min = min;
    this->max = max;
    return this;
  }

  field_num * mSetHardLimits(T min, T max) {
    mSetLimits(min, max);
    vIsHardLimit = true;
    return this;
  }

  virtual void mEval(field_eval_context & pCtx, const json & data) override;
};

struct field_str : public field {
  field_str(const char * n) : field(n) {}

  virtual void mEval(field_eval_context & pCtx, const json & data) override;
};

struct field_bool : public field {
  bool & vVal;

  field_bool(const char * n, bool & pVal) : field(n), vVal(pVal) {}

  virtual void mEval(field_eval_context & pCtx, const json & data) override;
};

struct field_json : public field {
  field_json(const char * n) : field(n) {}

  virtual void mEval(field_eval_context & pCtx, const json & data) override;
};

struct field_nested : public field {
  std::vector<std::unique_ptr<field>> lSubfields;

  field_nested(const char * n) : field(n) {}

  field_nested * mAddSubfield(field * pF) {
    lSubfields.emplace_back(std::unique_ptr<field>(pF));
    return this;
  }

  virtual void mEval(field_eval_context & pCtx, const json & data) override;
};

std::vector<std::unique_ptr<field>> fMakeDgsCmplSchema(const common_params & pParamsBase, task_params & pParams);

task_params fEvalDgsCmplSchema(const dgs_vocab *                   pVocab,
                               const common_params &               pParamsBase,
                               const std::vector<dgs_logit_bias> & pLogitBiasEog,
                               const json &                        data);

}  // namespace server_schema
