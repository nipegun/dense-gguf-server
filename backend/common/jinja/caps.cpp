#include "caps.h"

#include "runtime.h"
#include "value.h"

// note: the json dependency is only for defining input in a convenient way
// we can remove it in the future when we figure out a better way to define inputs using jinja::value
#include "json.h"

#include <functional>
#include <sstream>

#define FILENAME "jinja-caps"

using json = common_json;

namespace jinja {

using caps_json_fn    = std::function<json()>;
using caps_ctx_fn     = std::function<void(context &)>;
using caps_analyze_fn = std::function<void(context &, bool, value &, value &, const std::string &)>;

void fCapsApplyPreserveReasoning(jinja::context & pCtx, bool pEnabled) {
  pCtx.mSetVal("preserve_thinking", mk_val<value_bool>(pEnabled));
  pCtx.mSetVal("clear_thinking", mk_val<value_bool>(!pEnabled));
  pCtx.mSetVal("truncate_history_thinking", mk_val<value_bool>(!pEnabled));
  pCtx.mSetVal("drop_thinking", mk_val<value_bool>(!pEnabled));
}

void fCapsApplyReasoningEffort(jinja::context & pCtx, const std::string & pEffort) {
  value vVar = mk_val<value_string>(pEffort);  // bind to the same value for stats
  pCtx.mSetVal("reasoning_effort", vVar);
  pCtx.mSetVal("reasoning_strength", vVar);
}

static void fCapsTryExecute(jinja::program &        pProg,
                            const caps_json_fn &    pMessagesFn,
                            const caps_ctx_fn &     pCtxFn,
                            const caps_json_fn &    pToolsFn,
                            const caps_analyze_fn & pAnalyzeFn) {
  json vMsgs = pMessagesFn();
  for (int vAttempt = 0; vAttempt < 2; vAttempt++) {
    context vCtx;
    vCtx.vIsGetStats = true;
    jinja::fGlobalFromJson(vCtx,
                           json{
                             { "messages",              vMsgs                                  },
                             { "tools",                 pToolsFn ? pToolsFn() : json::mArray() },
                             { "bos_token",             ""                                     },
                             { "eos_token",             ""                                     },
                             { "add_generation_prompt", true                                   }
    },
                           true);

    if (pCtxFn) {
      pCtxFn(vCtx);
    }

    auto vMessages = vCtx.mGetVal("messages");
    auto vTools    = vCtx.mGetVal("tools");

    bool        vSuccess = false;
    std::string vResult;
    try {
      jinja::runtime vRuntime(vCtx);
      auto           vResults = vRuntime.mExecute(pProg);
      auto           vParts   = jinja::runtime::mGatherStringParts(vResults);
      vResult                 = vParts->mAsString().str();
      vSuccess                = true;
    } catch (const std::exception & vE) {
      JJ_DEBUG("Exception during execution: %s", vE.what());
      vResult = "";
      // ignore exceptions during capability analysis
    }

    // some templates require a thinking field on every assistant turn (e.g. K2 Horizon):
    // retry once with an empty reasoning_content on the assistant turns that lack one
    if (!vSuccess && vAttempt == 0) {
      bool vAdded = false;
      for (auto & vMsg : vMsgs) {
        if (vMsg.is_object() && vMsg.value("role", "") == "assistant" && !vMsg.contains("reasoning_content")) {
          vMsg["reasoning_content"] = "";
          vAdded                    = true;
        }
      }
      if (vAdded) {
        continue;
      }
    }

    pAnalyzeFn(vCtx, vSuccess, vMessages, vTools, vResult);
    return;
  }
}

// for debugging only
static void fCapsPrintStats(value & pV, const std::string & pPath) {
  std::string vOps;
  for (const auto & name : pV->vStats.sOps) {
    vOps += name + " ";
  }
  JJ_DEBUG("Value %s, type: %s %s, ops: %s", pPath.c_str(), pV->type().c_str(), pV->vStats.vUsed ? "(used)" : "",
           vOps.c_str());
}

std::map<std::string, bool> caps::mToMap() const {
  return {
    { "supports_string_content",      vSupportsStringContent     },
    { "supports_typed_content",       vSupportsTypedContent      },
    { "supports_tools",               vSupportsTools             },
    { "supports_tool_calls",          vSupportsToolCalls         },
    { "supports_parallel_tool_calls", vSupportsParallelToolCalls },
    { "supports_system_role",         vSupportsSystemRole        },
    { "supports_preserve_reasoning",  vSupportsPreserveReasoning },
    { "supports_reasoning_effort",    vSupportsReasoningEffort   },
    { "supports_object_arguments",    vSupportsObjectArguments   },
  };
}

std::string caps::to_string() const {
  std::ostringstream vSs;
  vSs << "Caps(\n";
  for (const auto & [key, value] : mToMap()) {
    vSs << "  " << key << "=" << (value ? "true" : "false") << "\n";
  }
  vSs << ")";
  return vSs.str();
}

caps fCapsGet(jinja::program & pProg) {
  caps vResult;

  static const auto cHasOp = [](value & pV, const std::string & pOpName) {
    return pV->vStats.sOps.find(pOpName) != pV->vStats.sOps.end();
  };

  JJ_DEBUG("%s\n", ">>> Running capability check: typed content");

  bool                     vChecksForString = false;
  static const std::string cContentMarker   = "STRING_MARKER";

  // case: typed content support
  fCapsTryExecute(
    pProg,
    [&]() {
      // messages
      return json::mArray({
        { { "role", "user" }, { "content", cContentMarker } }
      });
    },
    nullptr,  // ctx_fn
    nullptr,  // tools_fn
    [&](context &, bool pSuccess, value & pMessages, value &, const std::string & pRendered) {
      auto & vContent = pMessages->at(0)->at("content");
      fCapsPrintStats(vContent, "messages[0].content");
      if (cHasOp(vContent, "test_is_string")) {
        // checked if content is string
        vChecksForString = true;
      }
      bool vUsedAsArray = cHasOp(vContent, "selectattr") || cHasOp(vContent, "array_access");
      if (vUsedAsArray) {
        // accessed as an array
        vResult.vSupportsTypedContent = true;
      }
      if (!pSuccess) {
        // failed to execute with content as string
        vResult.vSupportsStringContent = false;
      } else if (vUsedAsArray && pRendered.find(cContentMarker) == std::string::npos) {
        // edge case: string may be accessed for checking, but does not appear in the output
        vResult.vSupportsStringContent = false;
      }
    });

  if (vChecksForString) {
    fCapsTryExecute(
      pProg,
      [&]() {
        // messages
        return json::mArray({
          { { "role", "user" }, { "content", json::mArray({}) } }
        });
      },
      nullptr,  // ctx_fn
      nullptr,  // tools_fn
      [&](context &, bool pSuccess, value & pMessages, value &, const std::string &) {
        auto & vContent = pMessages->at(0)->at("content");
        fCapsPrintStats(vContent, "messages[0].content");
        bool vUsedAsArray = cHasOp(vContent, "selectattr") || cHasOp(vContent, "array_access");
        if (vUsedAsArray && pSuccess) {
          // accessed as an array
          vResult.vSupportsTypedContent = true;
        }
      });
  }

  JJ_DEBUG("%s\n", ">>> Running capability check: system prompt");

  // case: system prompt support
  fCapsTryExecute(
    pProg,
    [&]() {
      // messages
      return json::mArray({
        { { "role", "system" }, { "content", "System message" } },
        { { "role", "user" },   { "content", "User message" }   },
      });
    },
    nullptr,  // ctx_fn
    nullptr,  // tools_fn
    [&](context &, bool, value & pMessages, value &, const std::string &) {
      auto & vContent = pMessages->at(0)->at("content");
      fCapsPrintStats(vContent, "messages[0].content");
      if (!vContent->vStats.vUsed) {
        vResult.vSupportsSystemRole = false;
      }
    });

  JJ_DEBUG("%s\n", ">>> Running capability check: single tool with object arguments support");

  // case: tools support: single call with object arguments
  fCapsTryExecute(
    pProg,
    [&]() {
      // messages
      return json::mArray({
        {
         { "role", "user" },
         { "content", "User message" },
         },
        { { "role", "assistant" },
         { "content", "" },  // Some templates expect content to be empty with tool calls
          { "tool_calls",
            json::mArray({ { { "id", "call00001" },
                             { "type", "function" },
                             { "function", { { "name", "tool1" }, { "arguments", { { "arg", "value" } } } } } } }) } },
        { { "role", "tool" }, { "content", "Tool response" }, { "tool_call_id", "call00001" } },
        { { "role", "assistant" }, { "content", "The tool response was 'tool response'" } },
        {
         { "role", "user" },
         { "content", "User message" },
         },
      });
    },
    nullptr,  // ctx_fn
    [&]() {
      // tools
      return json::mArray({
        {
         { "name", "tool" },
         { "type", "function" },
         { "function",
            {
              { "name", "tool1" },
              { "description", "Tool description" },
              { "parameters",
                {
                  { "type", "object" },
                  { "properties",
                    {
                      { "arg",
                        {
                          { "type", "string" },
                          { "description", "Arg description" },
                        } },
                    } },
                  { "required", json::mArray({ "arg" }) },
                } },
            } },
         },
      });
    },
    [&](context &, bool pSuccess, value & pMessages, value & pTools, const std::string &) {
      if (!pSuccess) {
        return;  // Nothing can be inferred
      }

      auto & vToolName = pTools->at(0)->at("function")->at("name");
      fCapsPrintStats(vToolName, "tools[0].function.name");
      fCapsPrintStats(pTools, "tools");
      if (!vToolName->vStats.vUsed) {
        vResult.vSupportsTools = false;
      }

      auto & vToolCalls = pMessages->at(1)->at("tool_calls");
      ;
      fCapsPrintStats(vToolCalls, "messages[1].tool_calls");
      if (!vToolCalls->vStats.vUsed) {
        vResult.vSupportsToolCalls = false;
        return;
      }

      auto & vToolArg = vToolCalls->at(0)->at("function")->at("arguments")->at("arg");
      fCapsPrintStats(vToolArg, "messages[1].tool_calls[0].function.arguments.arg");
      if (vToolArg->vStats.vUsed) {
        vResult.vSupportsObjectArguments = true;
      }
    });

  if (!vResult.vSupportsObjectArguments) {
    JJ_DEBUG("%s\n", ">>> Running capability check: single tool with string arguments support");

    // case: tools support: single call with string arguments
    fCapsTryExecute(
      pProg,
      [&]() {
        // messages
        return json::mArray({
          {
           { "role", "user" },
           { "content", "User message" },
           },
          { { "role", "assistant" },
           { "content", "" },  // Some templates expect content to be empty with tool calls
            { "tool_calls",
              json::mArray({ { { "id", "call00001" },
                               { "type", "function" },
                               { "function", { { "name", "tool1" }, { "arguments", R"({"arg": "value"})" } } } } }) } },
          { { "role", "tool" }, { "content", "Tool response" }, { "tool_call_id", "call00001" } },
          { { "role", "assistant" }, { "content", "The tool response was 'tool response'" } },
          {
           { "role", "user" },
           { "content", "User message" },
           },
        });
      },
      nullptr,  // ctx_fn
      [&]() {
        // tools
        return json::mArray({
          {
           { "name", "tool" },
           { "type", "function" },
           { "function",
              {
                { "name", "tool1" },
                { "description", "Tool description" },
                { "parameters",
                  {
                    { "type", "object" },
                    { "properties",
                      {
                        { "arg",
                          {
                            { "type", "string" },
                            { "description", "Arg description" },
                          } },
                      } },
                    { "required", json::mArray({ "arg" }) },
                  } },
              } },
           },
        });
      },
      [&](context &, bool pSuccess, value & pMessages, value & pTools, const std::string &) {
        if (!pSuccess) {
          vResult.vSupportsToolCalls = false;
          vResult.vSupportsTools     = false;
          return;
        }

        auto & vToolName = pTools->at(0)->at("function")->at("name");
        fCapsPrintStats(vToolName, "tools[0].function.name");
        fCapsPrintStats(pTools, "tools");
        if (!vToolName->vStats.vUsed) {
          vResult.vSupportsTools = false;
        }

        auto & vToolCalls = pMessages->at(1)->at("tool_calls");
        fCapsPrintStats(vToolCalls, "messages[1].tool_calls");
        if (!vToolCalls->vStats.vUsed) {
          vResult.vSupportsToolCalls = false;
          return;
        }
      });
  }

  JJ_DEBUG("%s\n", ">>> Running capability check: parallel tool support");

  // case: tools support: parallel calls
  fCapsTryExecute(
    pProg,
    [&]() {
      json vArgs = json(R"({"arg": "value"})");
      if (vResult.vSupportsObjectArguments) {
        vArgs = json{
          { "arg", "value" }
        };
      }

      // messages
      return json::mArray({
        {
         { "role", "user" },
         { "content", "User message" },
         },
        { { "role", "assistant" },
         { "content", "" },  // Some templates expect content to be empty with tool calls
          { "tool_calls", json::mArray({ { { "id", "call00001" },
                                           { "type", "function" },
                                           { "function", { { "name", "tool1" }, { "arguments", vArgs } } } },
                                         { { "id", "call00002" },
                                           { "type", "function" },
                                           { "function", { { "name", "tool1" }, { "arguments", vArgs } } } } }) } },
        { { "role", "tool" }, { "content", "Tool response" }, { "tool_call_id", "call00001" } },
        { { "role", "assistant" }, { "content", "The tool response was 'tool response'" } },
        {
         { "role", "user" },
         { "content", "User message" },
         },
      });
    },
    nullptr,  // ctx_fn
    [&]() {
      // tools
      return json::mArray({
        {
         { "name", "tool" },
         { "type", "function" },
         { "function",
            {
              { "name", "tool1" },
              { "description", "Tool description" },
              { "parameters",
                {
                  { "type", "object" },
                  { "properties",
                    {
                      { "arg",
                        {
                          { "type", "string" },
                          { "description", "Arg description" },
                        } },
                    } },
                  { "required", json::mArray({ "arg" }) },
                } },
            } },
         },
      });
    },
    [&](context &, bool pSuccess, value & pMessages, value &, const std::string &) {
      if (!pSuccess) {
        vResult.vSupportsParallelToolCalls = false;
        return;
      }

      auto & vToolCalls = pMessages->at(1)->at("tool_calls");
      fCapsPrintStats(vToolCalls, "messages[1].tool_calls");

      // check for second tool call usage
      auto & vToolCall1 = vToolCalls->at(1)->at("function");
      fCapsPrintStats(vToolCall1, "messages[1].tool_calls[1].function");
      if (!vToolCall1->vStats.vUsed) {
        vResult.vSupportsParallelToolCalls = false;
      }
    });

  JJ_DEBUG("%s\n", ">>> Running capability check: preserve reasoning");

  // case: preserve reasoning content in chat history
  const std::string cReasoningPlaceholder = "<REASONING_CONTENT_PLACEHOLDER>";
  fCapsTryExecute(
    pProg,
    [&]() {
      // messages
      return json::mArray({
        { { "role", "user" }, { "content", "User message" } },
        { { "role", "assistant" },
         { "content", "Assistant message" },
         // check of reasoning_content deeper in the history, not just the last assistant message
          { "reasoning_content", cReasoningPlaceholder } },
        { { "role", "user" }, { "content", "User message" } },
        { { "role", "assistant" }, { "content", "Assistant message" }, { "reasoning_content", "Reasoning content" } },
        { { "role", "user" }, { "content", "User message" } },
      });
    },
    [&](context & pCtx) {
      pCtx.mSetVal("enable_thinking", mk_val<value_bool>(true));
      fCapsApplyPreserveReasoning(pCtx, true);
    },
    nullptr,  // tools_fn
    [&](context &, bool, value &, value &, const std::string & pOutput) {
      // note: we cannot use stats here because the reasoning_content may be used for "if" condition test, but not actually outputted in the final result
      if (pOutput.find(cReasoningPlaceholder) != std::string::npos) {
        vResult.vSupportsPreserveReasoning = true;
      }
    });

  JJ_DEBUG("%s\n", ">>> Running capability check: reasoning effort");

  // case: reasoning effort level
  fCapsTryExecute(
    pProg,
    [&]() {
      // messages
      return json::mArray({
        { { "role", "user" }, { "content", "User message" } },
      });
    },
    [&](context & pCtx) {
      pCtx.mSetVal("enable_thinking", mk_val<value_bool>(true));
      fCapsApplyReasoningEffort(pCtx, "low");
    },
    nullptr,  // tools_fn
    [&](context & pCtx, bool, value &, value &, const std::string &) {
      value vEffort = pCtx.mGetVal("reasoning_effort");
      fCapsPrintStats(vEffort, "reasoning_effort");
      vResult.vSupportsReasoningEffort = vEffort->vStats.vUsed;
    });

  JJ_DEBUG("%s\n", vResult.to_string().c_str());

  return vResult;
}

}  // namespace jinja
