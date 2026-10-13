#include "server-schema.h"

#include "json-schema-to-grammar.h"

namespace server_schema {

//
// dgs.cpp-specific completion schema
//

std::vector<std::unique_ptr<field>> fMakeDgsCmplSchema(const common_params & pParamsBase, task_params & pParams) {
  std::vector<std::unique_ptr<field>> lFields;
  auto                                vAdd = [&](field * pF) {
    lFields.emplace_back(pF);
  };

  vAdd((new field_str("dgs_client"))
         ->mSetDesc("Origin of the request. The bundled web interface sends \"webui\"; its streamed OpenAI-style "
                    "requests then receive {\"dgs_queue\": {\"ahead\", \"ahead_api\", \"ahead_webui\"}} events "
                    "while they wait for a free slot (all zero when the request starts). Other clients omit it "
                    "and never receive those events")
         ->mSetHandler([&](field_eval_context & pCtx, const json & pData) {
           pCtx.vParams.vFromWebui = pData.at("dgs_client").get<std::string>() == "webui";
         }));

  vAdd((new field_bool("verbose", pParams.vVerbose))
         ->mSetDesc("Include __verbose field in the response with additional debug information"));

  vAdd((new field_bool("timings_per_token", pParams.vTimingsPerToken))
         ->mSetDesc("Include prompt processing and text generation speed information in each response"));

  vAdd(
    (new field_bool("stream", pParams.stream))
      ->mSetDesc("Allows receiving each predicted token in real-time instead of waiting for the completion to finish"));

  vAdd((new field_nested("stream_options"))
         ->mAddSubfield((new field_bool("include_usage", pParams.vIncludeUsage))
                          ->mSetDesc("Whether to include usage information in the stream"))
         ->mSetDesc("Additional options for streaming responses"));

  vAdd((new field_bool("cache_prompt", pParams.vCachePrompt))
         ->mSetDesc("Re-use KV cache from a previous request if possible. This way the common prefix does not have to "
                    "be re-processed, only the suffix that differs between the requests"));

  vAdd((new field_bool("return_tokens", pParams.vReturnTokens))
         ->mSetDesc("Return the raw generated token ids in the `tokens` field"));

  vAdd((new field_bool("return_progress", pParams.vReturnProgress))
         ->mSetDesc("Include prompt processing progress events in stream mode"));

  vAdd((new field_num("sse_ping_interval", pParams.vSsePingInterval))
         ->mSetHardLimits(-1, INT32_MAX)
         ->mSetDesc(
           "Interval in seconds between SSE comment pings emitted while the stream stays silent, -1 disables pings"));

  vAdd((new field_num("n_predict", pParams.vNPredict))
         ->mSetHardLimits(-1, INT32_MAX)
         ->mAddAlias("max_completion_tokens")
         ->mAddAlias("max_tokens")
         ->mSetDesc("Set the maximum number of tokens to predict. When 0, no tokens will be generated but the prompt "
                    "is evaluated into the cache"));

  vAdd((new field_num("n_indent", pParams.vNIndent))
         ->mSetHardLimits(0, INT32_MAX)
         ->mSetDesc("Specify the minimum line indentation for the generated text in number of whitespace characters. "
                    "Useful for code completion tasks"));

  vAdd((new field_num("n_keep", pParams.vNKeep))
         ->mSetHardLimits(-1, INT32_MAX)
         ->mSetDesc("Specify the number of tokens from the initial prompt to retain when context size is exceeded. Use "
                    "-1 to retain all tokens from the prompt"));

  vAdd((new field_num("n_discard", pParams.vNDiscard))
         ->mSetHardLimits(0, INT32_MAX)
         ->mSetDesc("Number of tokens after n_keep that may be discarded when shifting context (0 = half context)"));

  vAdd((new field_num("n_cmpl", pParams.vNCmpl))
         ->mSetHardLimits(1, pParamsBase.vNParallel)
         ->mAddAlias("n")  // alias "n" as fallback (OpenAI completions API)
         ->mSetDesc("Number of completions to generate. If the input has multiple prompts, total outputs will be N "
                    "prompts times n_cmpl"));

  vAdd((new field_num("n_cache_reuse", pParams.vNCacheReuse))
         ->mSetHardLimits(0, INT32_MAX)
         ->mSetDesc("Min chunk size to attempt reusing from the cache via KV shifting. See --cache-reuse arg"));

  // TODO: implement t_max_prompt_ms
  // add((new field_num("t_max_prompt_ms", params.t_max_prompt_ms))

  vAdd((new field_num("t_max_predict_ms", pParams.vTMaxPredictMs))
         ->mSetHardLimits(-1, std::numeric_limits<int64_t>::max())
         ->mSetDesc(
           "Set a time limit in milliseconds for the prediction phase. The timeout triggers if generation exceeds this "
           "time (measured since the first token) and a newline has been generated. Useful for FIM applications"));

  vAdd((new field_json("response_fields"))
         ->mSetDesc("A list of response fields to return. Missing fields are omitted without error. Fields with a "
                    "slash are unnested (e.g. generation_settings/n_predict moves n_predict to the root)")
         ->mSetHandler([&](field_eval_context & pCtx, const json & data) {
           pCtx.vParams.lResponseFields = fJsonValue(data, "response_fields", std::vector<std::string>());
         }));

  //
  // Sampling params
  //

  vAdd((new field_num("top_k", pParams.vSampling.vTopK))
         ->mSetLimits(0, INT32_MAX)
         ->mSetDesc("Limit the next token selection to the K most probable tokens (0 = disabled)"));

  vAdd((new field_num("top_p", pParams.vSampling.vTopP))
         ->mSetLimits(0.0f, 1.0f)
         ->mSetDesc("Limit the next token selection to a subset of tokens with cumulative probability above threshold "
                    "P (1.0 = disabled)"));

  vAdd((new field_num("min_p", pParams.vSampling.vMinP))
         ->mSetLimits(0.0f, 1.0f)
         ->mSetDesc("The minimum probability for a token to be considered, relative to the probability of the most "
                    "likely token (0 = disabled)"));

  vAdd((new field_num("top_n_sigma", pParams.vSampling.vTopNSigma))
         ->mSetDesc("Keep tokens within n standard deviations of the top token logit (< 0 = disabled)"));

  vAdd((new field_num("xtc_probability", pParams.vSampling.vXtcProbability))
         ->mSetLimits(0.0f, 1.0f)
         ->mSetDesc("Set the chance for token removal via XTC sampler (0 = disabled)"));

  vAdd(
    (new field_num("xtc_threshold", pParams.vSampling.vXtcThreshold))
      ->mSetLimits(0.0f, 1.0f)
      ->mSetDesc("Set a minimum probability threshold for tokens to be removed via XTC sampler (> 0.5 disables XTC)"));

  vAdd((new field_num("typical_p", pParams.vSampling.vTypP))
         // ->set_limits(0.0f, 1.0f) // what's the valid range?
         ->mSetDesc("Enable locally typical sampling with parameter p (1.0 = disabled)"));

  vAdd((new field_num("temperature", pParams.vSampling.vTemp))
         ->mSetLimits(0.0f, std::numeric_limits<float>::infinity())
         ->mSetDesc("Adjust the randomness of the generated text (0 = greedy)"));

  vAdd((new field_num("dynatemp_range", pParams.vSampling.vDynatempRange))
         ->mSetDesc("Dynamic temperature range. The final temperature will be in [temperature - range, temperature + "
                    "range] (0 = disabled)"));

  vAdd((new field_num("dynatemp_exponent", pParams.vSampling.vDynatempExponent))
         ->mSetDesc("Dynamic temperature exponent, controls how entropy maps to temperature"));

  vAdd((new field_num("repeat_last_n", pParams.vSampling.vPenaltyLastN))
         ->mSetHardLimits(0, INT32_MAX)
         ->mSetDesc("Last n tokens to consider for penalizing repetition (0 = disabled)"));

  vAdd((new field_num("repeat_penalty", pParams.vSampling.vPenaltyRepeat))
         ->mSetDesc("Control the repetition of token sequences in the generated text (1.0 = disabled)"));

  vAdd((new field_num("frequency_penalty", pParams.vSampling.vPenaltyFreq))
         ->mSetDesc("Repeat alpha frequency penalty (0 = disabled)"));

  vAdd((new field_num("presence_penalty", pParams.vSampling.vPenaltyPresent))
         ->mSetDesc("Repeat alpha presence penalty (0 = disabled)"));

  vAdd((new field_num("dry_multiplier", pParams.vSampling.vDryMultiplier))
         ->mSetDesc("Set the DRY (Don't Repeat Yourself) repetition penalty multiplier (0 = disabled)"));

  vAdd((new field_num("dry_base", pParams.vSampling.vDryBase))
         ->mSetDesc("Set the DRY repetition penalty base value (must be >= 1.0, any values < 1.0 will be replaced with "
                    "the default value)")
         ->mSetHandler([&](field_eval_context & pCtx, const json & data) {
           float vV                        = data.at("dry_base").get<float>();
           pCtx.vParams.vSampling.vDryBase = (vV < 1.0f) ? pParamsBase.vSampling.vDryBase : vV;
         }));

  vAdd((new field_num("dry_allowed_length", pParams.vSampling.vDryAllowedLength))
         ->mSetHardLimits(0, INT32_MAX)
         ->mSetDesc("Tokens that extend repetition beyond this length receive exponentially increasing penalty: "
                    "multiplier * base ^ (sequence_length - allowed_length)"));

  vAdd((new field_num("dry_penalty_last_n", pParams.vSampling.vDryPenaltyLastN))
         ->mSetHardLimits(0, INT32_MAX)
         ->mSetDesc("How many tokens to scan for repetitions (0 = disabled)"));

  vAdd((new field_num("mirostat", pParams.vSampling.vMirostat))
         ->mSetLimits(0, 2)
         ->mSetDesc("Enable Mirostat sampling, controlling perplexity during text generation (0 = disabled, 1 = "
                    "Mirostat, 2 = Mirostat 2.0)"));

  vAdd((new field_num("mirostat_tau", pParams.vSampling.vMirostatTau))
         ->mSetDesc("Set the Mirostat target entropy, parameter tau"));

  vAdd((new field_num("mirostat_eta", pParams.vSampling.vMirostatEta))
         ->mSetDesc("Set the Mirostat learning rate, parameter eta"));

  vAdd((new field_num("adaptive_target", pParams.vSampling.vAdaptiveTarget))
         ->mSetLimits(-std::numeric_limits<float>::max(), 1.0f)
         ->mSetDesc("Adaptive sampling target entropy (valid range 0.0 to 1.0; negative = disabled)"));

  vAdd((new field_num("adaptive_decay", pParams.vSampling.vAdaptiveDecay))
         ->mSetHardLimits(0.0f, 0.99f)
         ->mSetDesc("EMA decay for adaptive sampling; history approximates 1/(1-decay) tokens"));

  // seed is uint32_t; field_num uses int32_t so use a handler
  vAdd((new field_num("seed", pParams.vSampling.vSeed))
         ->mSetDesc("Set the random number generator (RNG) seed (-1 = random)"));

  vAdd((new field_num("n_probs", pParams.vSampling.vNProbs))
         ->mAddAlias("logprobs")  // use "logprobs" if "n_probs" wasn't provided
         ->mSetDesc("If greater than 0, output the probabilities of top N tokens for each generated token"));

  vAdd((new field_num("min_keep", pParams.vSampling.vMinKeep))
         ->mSetHardLimits(0, INT32_MAX)
         ->mSetDesc("If greater than 0, force samplers to return at least N possible tokens"));

  vAdd((new field_bool("backend_sampling", pParams.vSampling.vBackendSampling))
         ->mSetDesc("Use backend sampling instead of dgs.cpp sampling"));

  vAdd((new field_bool("post_sampling_probs", pParams.vPostSamplingProbs))
         ->mSetDesc("Return probabilities of top n_probs tokens after applying the sampling chain"));

  //
  // Speculative decoding params
  //

  // TODO: to keep things simple, we disable speculative parameter adjustments for now
#if 0
    // TODO: for now, be able to adjust only the draft-model based speculative parameters
    add((new field_num("speculative.n_max", params.speculative.draft.n_max))
        ->set_hard_limits(0, INT32_MAX)
        ->set_desc("Maximum number of tokens to draft during speculative decoding"));

    add((new field_num("speculative.n_min", params.speculative.draft.n_min))
        ->set_hard_limits(0, INT32_MAX)
        ->set_desc("Minimum number of draft tokens to use for speculative decoding");

    add((new field_num("speculative.p_min", params.speculative.draft.p_min))
        ->set_hard_limits(0.0f, 1.0f)
        ->set_desc("Minimum speculative decoding probability for draft tokens (0 = greedy)"));


    add((new field_str("speculative.type"))
        ->set_desc("Speculative decoding method (for debugging and research purposes)")
        ->set_handler([&](field_eval_context & ctx, const json & data) {
            ctx.params.speculative.types = { fCommonSpeculativeTypeFromName(data.at("speculative.type").get<std::string>()) };
        }));

    add((new field_num("speculative.ngram_size_n", params.speculative.ngram_simple.size_n))
        ->set_desc("Ngram size for lookup in ngram-based speculative decoding"));

    add((new field_num("speculative.ngram_size_m", params.speculative.ngram_simple.size_m))
        ->set_desc("Mgram size for speculative tokens in ngram-based speculative decoding"));

    add((new field_num("speculative.ngram_min_hits", params.speculative.ngram_simple.min_hits))
        ->set_desc("Minimum hits at ngram lookup for mgram to be proposed"));
#endif

  vAdd((new field_json("lora"))
         ->mSetDesc("A list of LoRA adapters to apply to this request. Each entry must have `id` and `scale` fields. "
                    "Adapters not listed default to scale 0.0")
         ->mSetHandler([&](field_eval_context & pCtx, const json & data) {
           const auto & lora = data.at("lora");
           if (!lora.is_array()) {
             throw std::runtime_error("Error: 'lora' must be an array of objects with 'id' and 'scale' fields");
           }
           pCtx.vParams.lora = fParseLoraRequest(lora);
         }));

  // sequence breakers for DRY
  // Currently, this is not compatible with TextGen WebUI, Koboldcpp and SillyTavern format
  // Ref: https://github.com/oobabooga/text-generation-webui/blob/d1af7a41ade7bd3c3a463bfa640725edb818ebaf/extensions/openai/typing.py#L39
  vAdd((new field_json("dry_sequence_breakers"))
         ->mSetDesc("Specify an array of sequence breakers for DRY sampling. Only a JSON array of strings is accepted")
         ->mSetHandler([&](field_eval_context & pCtx, const json & data) {
           pCtx.vParams.vSampling.lDrySequenceBreakers =
             fJsonValue(data, "dry_sequence_breakers", std::vector<std::string>());
           if (pCtx.vParams.vSampling.lDrySequenceBreakers.empty()) {
             throw std::runtime_error("Error: dry_sequence_breakers must be a non-empty array of strings");
           }
         }));

  // handle both "json_schema" and "grammar"
  vAdd((new field_json("json_schema"))
         ->mAddAlias("grammar")
         ->mSetDesc("Set a JSON schema (json_schema) or GBNF grammar string (grammar) for constrained generation. "
                    "json_schema takes precedence if both are provided")
         ->mSetHandler([&](field_eval_context & pCtx, const json & data) {
           auto & vParams = pCtx.vParams;
           if (data.contains("json_schema") && !data.contains("grammar")) {
             try {
               auto vSchema = fJsonValue(data, "json_schema", json::mObject());
               if (vSchema.is_object() && vSchema.empty()) {
                 // an empty schema means any object
                 vSchema["type"] = "object";
               }
               SRV_DBG("JSON schema: %s\n", vSchema.mDump(2).c_str());
               std::string vGrammarStr = fJsonSchemaToGrammar(vSchema);
               SRV_DBG("Converted grammar: %s\n", vGrammarStr.c_str());
               vParams.vSampling.vGrammar = { cCommonGrammarTypeOutputFormat, std::move(vGrammarStr) };
             } catch (const std::exception & vE) {
               throw std::runtime_error(std::string("\"json_schema\": ") + vE.what());
             }
           } else {
             std::string vGrammarStr = fJsonValue(data, "grammar", std::string());
             if (!vGrammarStr.empty()) {
               // grammar_type key is set by the server when converting chat template grammars
               std::string vGrammarType = fJsonValue(data, "grammar_type", std::string());
               if (vGrammarType == "tool_calls") {
                 vParams.vSampling.vGrammar = { cCommonGrammarTypeToolCalls, std::move(vGrammarStr) };
               } else {
                 // explicit grammar from the user (API field "grammar")
                 vParams.vSampling.vGrammar = { cCommonGrammarTypeUser, std::move(vGrammarStr) };
               }
               SRV_DBG("Grammar (%s): %s\n", vGrammarType.c_str(),
                       fCommonGrammarValue(vParams.vSampling.vGrammar).c_str());
             }
           }
         }));

  vAdd((new field_bool("grammar_lazy", pParams.vSampling.vGrammarLazy))
         ->mSetDesc("Whether to apply grammar constraints lazily, only when triggered (instead of at every step)"));

  //
  // Chat parser params
  //

  vAdd((new field_str("reasoning_format"))
         ->mSetDesc("Reasoning format for chain-of-thought models")
         ->mSetHandler([&](field_eval_context & pCtx, const json & data) {
           pCtx.vParams.vReasoningFormat =
             fCommonReasoningFormatFromName(data.at("reasoning_format").get<std::string>());
         }));

  vAdd((new field_json("continue_final_message"))
         ->mSetDesc("Whether to continue the final message of the chat template")
         ->mSetHandler([&](field_eval_context &, const json & data) {
           fCommonChatContinuationParse(data.at("continue_final_message"));
         }));

  vAdd((new field_json("echo"))
         ->mSetDesc("Whether to include the continued assistant message in the output")
         ->mSetHandler([&](field_eval_context &, const json & data) { data.at("echo").get<bool>(); }));

  //
  // Token-level fields (require vocab)
  //

  vAdd((new field_json("preserved_tokens"))
         ->mSetDesc("List of token strings that must not be split during tokenization")
         ->mSetHandler([&](field_eval_context & pCtx, const json & data) {
           fCommonSamplingAddPreservedTokens(pCtx.vParams.vSampling, pCtx.vVocab,
                                             data.at("preserved_tokens").get<std::vector<std::string>>());
         }));

  vAdd((new field_json("grammar_triggers"))
         ->mSetDesc("List of strings or patterns that trigger grammar-constrained generation")
         ->mSetHandler([&](field_eval_context & pCtx, const json & data) {
           std::vector<common_grammar_trigger> lTriggers;
           for (const auto & vT : data.at("grammar_triggers")) {
             lTriggers.push_back(server_grammar_trigger(vT).value);
           }
           fCommonSamplingAddGrammarTriggers(pCtx.vParams.vSampling, pCtx.vVocab, std::move(lTriggers));
         }));

  vAdd((new field_bool("reasoning_control", pParams.vSampling.vReasoningControl))
         ->mSetDesc("Create the budget sampler on demand so reasoning can be ended at runtime"));

  vAdd((new field_num("reasoning_budget_tokens", pParams.vSampling.vReasoningBudgetTokens))
         ->mSetHardLimits(-1, INT32_MAX)
         ->mSetDesc("Number of tokens in the reasoning budget (-1 = disabled)"));

  vAdd((new field_str("reasoning_budget_start_tag"))
         ->mSetDesc("Token string marking the start of the reasoning budget section")
         ->mSetHandler([&](field_eval_context & pCtx, const json & data) {
           GGML_ASSERT(pCtx.vVocab != nullptr);
           pCtx.vParams.vSampling.lReasoningBudgetStart =
             fCommonTokenize(pCtx.vVocab, data.at("reasoning_budget_start_tag").get<std::string>(), false, true);
         }));

  vAdd((new field_json("reasoning_budget_end_tags"))
         ->mAddAlias("reasoning_budget_end_tag")
         ->mSetDesc(
           "Token strings marking the end of the reasoning budget section; the first is forced when the budget expires")
         ->mSetHandler([&](field_eval_context & pCtx, const json & data) {
           GGML_ASSERT(pCtx.vVocab != nullptr);
           pCtx.vParams.vSampling.lReasoningBudgetEnd.clear();
           if (data.contains("reasoning_budget_end_tags")) {
             for (const auto & vT : data.at("reasoning_budget_end_tags")) {
               std::string vTag = vT.get<std::string>();
               if (!vTag.empty()) {
                 pCtx.vParams.vSampling.lReasoningBudgetEnd.push_back(fCommonTokenize(pCtx.vVocab, vTag, false, true));
               }
             }
           } else if (data.contains("reasoning_budget_end_tag")) {
             std::string vTag = data.at("reasoning_budget_end_tag").get<std::string>();
             if (!vTag.empty()) {
               pCtx.vParams.vSampling.lReasoningBudgetEnd.push_back(fCommonTokenize(pCtx.vVocab, vTag, false, true));
             }
           }
         }));

  vAdd((new field_str("reasoning_budget_message"))
         ->mSetDesc("Message to prepend to the reasoning budget end tag when forcing it")
         ->mSetHandler([&](field_eval_context & pCtx, const json & data) {
           GGML_ASSERT(pCtx.vVocab != nullptr);
           if (!pCtx.vParams.vSampling.lReasoningBudgetEnd.empty()) {
             dgs_tokens  lEndTag  = pCtx.vParams.vSampling.lReasoningBudgetEnd.front();
             std::string vMessage = fJsonValue(data, "reasoning_budget_message", std::string());
             if (!vMessage.empty()) {
               dgs_tokens lMessageTokens = fCommonTokenize(pCtx.vVocab, vMessage, false, true);
               lEndTag.insert(lEndTag.begin(), lMessageTokens.begin(), lMessageTokens.end());
             }
             pCtx.vParams.vSampling.lReasoningBudgetForced = std::move(lEndTag);
           }
         }));

  vAdd((new field_json("logit_bias"))
         ->mSetDesc("Modify the likelihood of specific tokens. Accepts an array of [token, bias] pairs or an object "
                    "mapping token to bias. Use false as bias to ban a token")
         ->mSetHandler([&](field_eval_context & pCtx, const json & data) {
           GGML_ASSERT(pCtx.vVocab != nullptr);
           pCtx.vParams.vSampling.lLogitBias.clear();
           const auto & vLogitBias = data.at("logit_bias");
           const int    cNVocab    = fDgsVocabNTokens(pCtx.vVocab);
           auto         vParseBias = [](const json & pV, float & pBias) -> bool {
             if (pV.is_number()) {
               pBias = pV.get<float>();
               return true;
             }
             if (pV.is_boolean() && !pV.get<bool>()) {
               pBias = -INFINITY;
               return true;
             }
             return false;
           };
           if (vLogitBias.is_array()) {
             for (const auto & vEl : vLogitBias) {
               if (!vEl.is_array() || vEl.size() != 2) {
                 continue;
               }
               float vBias;
               if (!vParseBias(vEl[1], vBias)) {
                 continue;
               }
               if (vEl[0].mIsNumberInteger()) {
                 dgs_token vTok = vEl[0].get<dgs_token>();
                 if (vTok >= 0 && vTok < cNVocab) {
                   pCtx.vParams.vSampling.lLogitBias.push_back({ vTok, vBias });
                 }
               } else if (vEl[0].is_string()) {
                 for (auto vTok : fCommonTokenize(pCtx.vVocab, vEl[0].get<std::string>(), false)) {
                   pCtx.vParams.vSampling.lLogitBias.push_back({ vTok, vBias });
                 }
               }
             }
           } else if (vLogitBias.is_object()) {
             for (const auto & vEl : vLogitBias.mItems()) {
               float vBias;
               if (!vParseBias(vEl.value(), vBias)) {
                 continue;
               }
               char *    end;
               dgs_token vTok = strtol(vEl.mKey().c_str(), &end, 10);
               if (*end == 0) {
                 if (vTok >= 0 && vTok < cNVocab) {
                   pCtx.vParams.vSampling.lLogitBias.push_back({ vTok, vBias });
                 }
               } else {
                 for (auto vT : fCommonTokenize(pCtx.vVocab, vEl.mKey(), false)) {
                   pCtx.vParams.vSampling.lLogitBias.push_back({ vT, vBias });
                 }
               }
             }
           }
         }));

  vAdd((new field_bool("ignore_eos", pParams.vSampling.vIgnoreEos))
         ->mSetDesc("Ignore the end-of-sequence token and continue generating")
         ->mSetHandler([&](field_eval_context & pCtx, const json & data) {
           GGML_ASSERT(pCtx.lLogitBiasEog != nullptr);
           pCtx.vParams.vSampling.vIgnoreEos = data.at("ignore_eos").get<bool>();
           if (pCtx.vParams.vSampling.vIgnoreEos && pCtx.lLogitBiasEog) {
             pCtx.vParams.vSampling.lLogitBias.insert(pCtx.vParams.vSampling.lLogitBias.end(),
                                                      pCtx.lLogitBiasEog->begin(), pCtx.lLogitBiasEog->end());
           }
         }));

  vAdd(
    (new field_json("stop"))
      ->mSetDesc(
        "Specify stopping strings. Generation stops when one is produced, and the string is not included in the output")
      ->mSetHandler([&](field_eval_context & pCtx, const json & data) {
        pCtx.vParams.lAntiprompt.clear();
        const auto & vStop = data.at("stop");
        if (vStop.is_array()) {
          for (const auto & vWord : vStop) {
            if (!vWord.empty()) {
              pCtx.vParams.lAntiprompt.push_back(vWord);
            }
          }
        } else if (vStop.is_string()) {
          pCtx.vParams.lAntiprompt.push_back(vStop.get<std::string>());
        }
        // fall back to CLI defaults if the request provided no effective stop strings
        if (pCtx.vParams.lAntiprompt.empty()) {
          pCtx.vParams.lAntiprompt = pParamsBase.lAntiprompt;
        }
      }));

  vAdd(
    (new field_json("samplers"))
      ->mSetDesc(
        "The order in which samplers are applied. An array of sampler type names, or a single string of sampler chars")
      ->mSetHandler([&](field_eval_context & pCtx, const json & data) {
        const auto & vSamplers = data.at("samplers");
        if (vSamplers.is_array()) {
          pCtx.vParams.vSampling.lSamplers = fCommonSamplerTypesFromNames(vSamplers.get<std::vector<std::string>>());
        } else if (vSamplers.is_string()) {
          pCtx.vParams.vSampling.lSamplers = fCommonSamplerTypesFromChars(vSamplers.get<std::string>());
        }
      }));

  return lFields;
}

task_params fEvalDgsCmplSchema(const dgs_vocab *                   pVocab,
                               const common_params &               pParamsBase,
                               const std::vector<dgs_logit_bias> & pLogitBiasEog,
                               const json &                        data) {
  task_params vParams;

  // Sampling parameter defaults are loaded from the global server context (but individual requests can still override them)
  vParams.vSampling        = pParamsBase.vSampling;
  vParams.vSpeculative     = pParamsBase.vSpeculative;
  vParams.vNKeep           = pParamsBase.vNKeep;
  vParams.vNPredict        = pParamsBase.vNPredict;
  vParams.vNCacheReuse     = pParamsBase.vNCacheReuse;
  vParams.vCachePrompt     = pParamsBase.vCachePrompt;
  vParams.lAntiprompt      = pParamsBase.lAntiprompt;
  vParams.vSsePingInterval = pParamsBase.vSsePingInterval;

  // enabling this will output extra debug information in the HTTP responses from the server
  vParams.vVerbose = pParamsBase.vVerbosity > 9;

  vParams.vReasoningFormat = pParamsBase.vReasoningFormat;

  // create context and schema
  field_eval_context vCtx(vParams);
  vCtx.vVocab        = pVocab;
  vCtx.lLogitBiasEog = &pLogitBiasEog;

  auto lSchema = fMakeDgsCmplSchema(pParamsBase, vParams);

  // eval all fields in the schema
  for (const auto & vF : lSchema) {
    vF->mEval(vCtx, data);
  }

  // debugging
  {
    auto vBudget = vParams.vSampling.vReasoningBudgetTokens;
    SRV_DBG("reasoning budget: tokens=%d, generation_prompt='%s', start=%zu toks, end=%zu seqs, forced=%zu toks\n",
            vBudget, vParams.vSampling.vGenerationPrompt.c_str(), vParams.vSampling.lReasoningBudgetStart.size(),
            vParams.vSampling.lReasoningBudgetEnd.size(), vParams.vSampling.lReasoningBudgetForced.size());
  }

  return vParams;
}

//
// eval() implementations
//

static void handle_with_catch(const char * name, std::function<void()> pFunc) {
  try {
    pFunc();
  } catch (const std::exception & vE) {
    throw std::invalid_argument(fStringFormat("Field '%s': %s", name, vE.what()));
  }
}

// treat a null value as absent so clients can send null to request the server default
static bool has_value(const json & data, const char * n) {
  return data.contains(n) && !data.at(n).is_null();
}

template <typename T> void field_num<T>::mEval(field_eval_context & pCtx, const json & data) {
  for (const auto & n : name) {
    if (has_value(data, n)) {
      handle_with_catch(n, [&]() {
        if (vCustomHandler) {
          vCustomHandler(pCtx, data);
        } else if (!vIsHardLimit) {
          vVal = std::max(min, std::min(max, data.at(n).template get<T>()));
        } else {
          T vTmp = data.at(n).template get<T>();
          if (vTmp < min || vTmp > max) {
            throw std::invalid_argument(std::string("Value must be between ") + std::to_string(min) +
                                        " <= value <= " + std::to_string(max) + ", but got " + std::to_string(vTmp));
          }
          vVal = vTmp;
        }
      });
      return;
    }
  }
}

void field_str::mEval(field_eval_context & pCtx, const json & data) {
  GGML_ASSERT(vCustomHandler);
  for (const auto & n : name) {
    if (has_value(data, n)) {
      handle_with_catch(n, [&]() { vCustomHandler(pCtx, data); });
      return;
    }
  }
}

void field_bool::mEval(field_eval_context & pCtx, const json & data) {
  for (const auto & n : name) {
    if (has_value(data, n)) {
      handle_with_catch(n, [&]() {
        if (vCustomHandler) {
          vCustomHandler(pCtx, data);
        } else {
          vVal = data.at(n).get<bool>();
        }
      });
      return;
    }
  }
}

void field_json::mEval(field_eval_context & pCtx, const json & data) {
  GGML_ASSERT(vCustomHandler);
  for (const auto & n : name) {
    if (has_value(data, n)) {
      handle_with_catch(n, [&]() { vCustomHandler(pCtx, data); });
      return;
    }
  }
}

void field_nested::mEval(field_eval_context & pCtx, const json & data) {
  for (const auto & n : name) {
    if (data.contains(n) && data.at(n).is_object()) {
      for (auto & vF : lSubfields) {
        vF->mEval(pCtx, data.at(n));
      }
      return;
    }
  }
}

}  // namespace server_schema
