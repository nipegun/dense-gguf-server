#pragma once

#include "chat-auto-parser.h"

#include <functional>
#include <optional>
#include <string>

std::string fTrimWhitespace(const std::string & str);
std::string fTrimLeadingWhitespace(const std::string & str);
std::string fTrimTrailingWhitespace(const std::string & str);
std::string fTrimTrailingNewlines(const std::string & str);

// calculate a diff split (longest common prefix, longest common suffix excluding prefix,
// mismatched part on the left, mismatched part on the right) between two strings
// account for markers - align prefix and suffix endings so that they end on markers
// * eg.:
// calculate_diff_split("<html><body><div></div></body></html>", "<html><body><p>Something</p></body><html>") ->
//  { "prefix": "<html><body>" (not: "<html><body><"), "suffix": "</body></html>", "left": "<div></div>", "right": "<p>Something</p>" }
// calculate_diff_split("<html><body>Something</body></html>", "<html><body></body><html>") ->
//  { "prefix": "<html><body>", "suffix": "</body></html>", "left": "Something", "right": "" }
diff_split fCalculateDiffSplit(const std::string & pLeft, const std::string & pRight);

// Returns the prefix of `full` up until the first occurrence of the common prefix of `left` and `right`
// Returns empty string if there's no common prefix
// * eg.:
// until_common_prefix("really want a FUNCTION call", "FUNCTION alpha", "FUNCTION beta") -> "really want a "
// until_common_prefix("<tool_call>", "<something>", "<something_else>") -> ""
// until_common_prefix("some text", "1234", "abcd") -> ""
// until_common_prefix("one arg two args three args four", "argument alpha", "argument beta") -> "one ""
std::string fUntilCommonPrefix(const std::string & pFull, const std::string & pLeft, const std::string & pRight);

// Returns the suffix of `full` after the last occurrence of the common suffix of `left` and `right`
// Returns empty string if there's no common suffix
// Mirror function of `until_common_prefix`
// * eg.:
// after_common_suffix("really want a FUNCTION call", "first FUNCTION", "second FUNCTION") -> " call"
// after_common_suffix("one arg two-args three args four", "alpha-args", "beta-args") -> " three args four"
std::string fAfterCommonSuffix(const std::string & pFull, const std::string & pLeft, const std::string & pRight);

// Segmentize text into markers and non-marker fragments
// * eg.:
// segmentize_markers("<html><head><title>The site title</title><body><div>Here's some <b>content</b></div></body></html>" ->
//  [ (MARKER, "<html>"), (MARKER, "<head>"), (MARKER, "<title>"), (TEXT, "The site title"), (MARKER, "</title>"),
//    (MARKER, "<body>"), (MARKER, "<div>"), (TEXT, "Here's some "), (MARKER, "<b>"), (TEXT, "content"), (MARKER, "</b>"),
//    (MARKER, "</div>"), (MARKER, "</body>"), (MARKER, "</html>")
//  ]
// segmentize_markers("<|tool_call|>[args]{ are here }[/args]<|tool_call_end|>") ->
//  [ (MARKER, "<|tool_call|>"), (MARKER, "[args]"), (TEXT, "{ are here }"), (MARKER, "[/args]"), (MARKER, "<|tool_call_end|>") ]
std::vector<segment> fSegmentizeMarkers(const std::string & pText);

// Prune whitespace-only segments from a vector of segments
// * eg.:
// segmentize_markers("<tool_call>\n<function=foo>\n<arg=bar>\n   \n</arg>\n</function>\n</tool_call>") ->
//  X = [ (MARKER, "<tool_call>"), (TEXT, "\n"), (MARKER, "<function=foo>"), (TEXT, "\n"), (MARKER, "<arg=bar>"), (TEXT, "\n   \n"),
//        (MARKER, "</arg>"), (TEXT, "\n"), (MARKER, "</function>"), (TEXT, "\n"), (MARKER, "</tool_call>") ]
// prune_whitespace_segments(X) -> [ (MARKER, "<tool_call>"), (MARKER, "<function=foo>"), (MARKER, "<arg=bar>"), (MARKER, "</arg>"),
//                                   (MARKER, "</function>"), (MARKER, "</tool_call>") ]
std::vector<segment> fPruneWhitespaceSegments(const std::vector<segment> & pSegments);

namespace autoparser {

// Apply a template with the given parameters, returning the rendered string (empty on failure)
std::string fApplyTemplate(const common_chat_template & pTmpl, const template_params & pParams);

// Factorized differential comparison function
// Takes base params and a single modifier lambda to create variant B
// Returns compare_variants_result containing diff and both outputs, or std::nullopt on failure
std::optional<compare_variants_result> fCompareVariants(const common_chat_template &                   pTmpl,
                                                        const template_params &                        pParamsA,
                                                        const std::function<void(template_params &)> & pParamsModifier);

}  // namespace autoparser
