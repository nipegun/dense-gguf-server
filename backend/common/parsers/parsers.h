#pragma once

#include "chat-auto-parser-helpers.h"
#include "chat-auto-parser.h"
#include "chat-peg-parser.h"
#include "chat.h"
#include "common.h"
#include "ggml.h"
#include "json-schema-to-grammar.h"
#include "json.h"

#include <functional>
#include <optional>
#include <set>
#include <string>
#include <vector>

using json = common_json;

// iterate over the function tools of an OpenAI-style tools array, passing each tool with its index in the array
void fForeachFunction(const json & pTools, const std::function<void(size_t, const json &)> & pFn);

// iterate over the parameters of a function tool, passing each parameter with its index and the document that owns it
void fForeachParameter(
  const json & pFunction,
  const std::function<void(size_t, const common_chat_schema_property &, const common_chat_schema_document_ptr &)> &
    pFn);

// render a template; the override arguments let a parser feed in messages, tools or context it has rewritten
std::string fCommonChatTemplateDirectApplyImpl(const common_chat_template &          pTmpl,
                                               const autoparser::generation_params & pInputs,
                                               const std::optional<json> &           pMessagesOverride  = std::nullopt,
                                               const std::optional<json> &           pToolsOverride     = std::nullopt,
                                               const std::optional<json> &           pAdditionalContext = std::nullopt);

// the suffix a template appends when add_generation_prompt is set
std::string fCommonChatTemplateGenerationPromptImpl(const common_chat_template &          pTmpl,
                                                    const autoparser::generation_params & pInputs,
                                                    const std::optional<json> & pMessagesOverride  = std::nullopt,
                                                    const std::optional<json> & pToolsOverride     = std::nullopt,
                                                    const std::optional<json> & pAdditionalContext = std::nullopt);

common_chat_params fCommonChatParamsInitQwen3Coder(const common_chat_template &          pTmpl,
                                                   const autoparser::generation_params & pInputs);
