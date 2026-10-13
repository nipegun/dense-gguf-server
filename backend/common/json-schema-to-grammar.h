#pragma once

#include "json-schema.h"
#include "json.h"

#include <functional>
#include <string>

std::string fJsonSchemaToGrammar(const common_json & pSchema, bool pForceGbnf = false);
std::string fJsonSchemaToGrammar(const common_chat_schema_document & pSchema);

struct common_grammar_builder {
  std::function<std::string(const std::string &, const std::string &)>        vAddRule;
  std::function<std::string(const std::string &, const common_chat_schema &)> vAddSchema;
};

struct common_grammar_options {
  bool vDotall = false;
};

std::string gbnf_format_literal(const std::string & pLiteral);

std::string fBuildGrammar(const std::function<void(const common_grammar_builder &)> & pCb,
                          const common_grammar_options &                              pOptions = {});
