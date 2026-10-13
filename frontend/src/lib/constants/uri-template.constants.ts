/**
 * URI Template constants for RFC 6570 template processing.
 */

/** URI scheme separator */
export const cUriSchemeSeparator = '://';

/** Regex to match template expressions like {var}, {+var}, {#var}, {/var} */
export const cTemplateExpressionRegex = /\{([+#./;?&]?)([^}]+)\}/g;

/** RFC 6570 URI template operators and separators. A single object covers both: an operator prefix character doubles as the separator written into the expansion (e.g. `{/a}`/`{;a}` use `/` and `;` for both), so the characters live here once. */
export const cUriTemplateSymbols = {
  /** Comma separator for list expansion */
  COMMA: ',',
  /** Form-style query */
  FORM_CONTINUATION: '&',
  /** Form-style query prefix */
  FORM_QUERY: '?',
  /** Fragment expansion */
  FRAGMENT: '#',
  /** Label expansion */
  LABEL: '.',
  /** Path-style parameters */
  PATH_PARAM: ';',
  /** Path segment expansion */
  PATH_SEGMENT: '/',
  /** Reserved expansion */
  RESERVED: '+',
  /** Simple string expansion (default) */
  SIMPLE: ''
} as const;

/** Maximum number of leading slashes to strip during URI normalization */
export const cMaxLeadingSlashesToStrip = 3;

/** Regex to strip explode modifier (*) from variable names */
export const cVariableExplodeModifierRegex = /[*]$/;

/** Regex to strip prefix modifier (:N) from variable names */
export const cVariablePrefixModifierRegex = /:[\d]+$/;

/** Regex to strip one or more leading slashes */
export const cLeadingSlashesRegex = /^\/+/;

/** Regex to match base64-encoded image URIs (format: "data:image/[media type];base64,[data]")*/
export const cBase64ImageUriRegex = /^data:(image\/[a-z0-9.\-+]+);base64/;
