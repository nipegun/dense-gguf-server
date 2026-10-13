import {
  cLeadingSlashesRegex,
  cTemplateExpressionRegex,
  cUriSchemeSeparator,
  cUriTemplateSymbols,
  cVariableExplodeModifierRegex,
  cVariablePrefixModifierRegex
} from '../constants';

/**
 * Normalize a resource URI for comparison.
 *
 * URI template expansion (especially with path operators like {/var})
 * can produce URIs that differ from listed resource URIs in slash placement.
 * For example, the template `svelte://{/slug*}.md` with slug="svelte/$effect"
 * expands to `svelte:///svelte/$effect.md`, while the listed resource URI is
 * `svelte://svelte/$effect.md`.
 *
 * This function strips extra leading slashes after the scheme to normalize
 * both forms to the same string for comparison purposes.
 *
 * @param uri - The URI to normalize
 * @returns Normalized URI string
 */
export function fNormalizeResourceUri(pUri: string): string {
  const cSchemeEnd = pUri.indexOf(cUriSchemeSeparator);

  if (cSchemeEnd === -1) return pUri;

  const cScheme = pUri.substring(0, cSchemeEnd);
  const cRest = pUri
    .substring(cSchemeEnd + cUriSchemeSeparator.length)
    .replace(cLeadingSlashesRegex, '');

  return `${cScheme}${cUriSchemeSeparator}${cRest}`;
}

/**
 * A parsed variable from a URI template expression.
 */
export interface UriTemplateVariable {
  /** Variable name */
  name: string;
  /** Operator prefix (+, #, /, etc.) or empty string */
  operator: string;
}

/**
 * Extract all variable names from a URI template string.
 *
 * @param template - URI template string (RFC 6570)
 * @returns Array of unique variable descriptors
 *
 * @example
 * ```ts
 * extractTemplateVariables("file:///{path}")
 * // => [{ name: "path", operator: "" }]
 *
 * extractTemplateVariables("db://{schema}/{table}")
 * // => [{ name: "schema", operator: "" }, { name: "table", operator: "" }]
 * ```
 */
export function fExtractTemplateVariables(pTemplate: string): UriTemplateVariable[] {
  const aVariables: UriTemplateVariable[] = [];
  const sSeen = new Set<string>();

  let vMatch;

  cTemplateExpressionRegex.lastIndex = 0;

  while ((vMatch = cTemplateExpressionRegex.exec(pTemplate)) !== null) {
    const cOperator = vMatch[1] || '';
    const cVarList = vMatch[2];

    // RFC 6570 allows comma-separated variable lists: {x,y,z}
    for (const cVarSpec of cVarList.split(',')) {
      // Strip explode modifier (*) and prefix modifier (:N)
      const name = cVarSpec
        .replace(cVariableExplodeModifierRegex, '')
        .replace(cVariablePrefixModifierRegex, '')
        .trim();

      if (name && !sSeen.has(name)) {
        sSeen.add(name);
        aVariables.push({ name, operator: cOperator });
      }
    }
  }

  return aVariables;
}

/**
 * Expand a URI template with the given variable values.
 * Implements a simplified RFC 6570 Level 2 expansion.
 *
 * @param template - URI template string
 * @param values - Map of variable name to value
 * @returns Expanded URI string
 *
 * @example
 * ```ts
 * expandTemplate("file:///{path}", { path: "src/main.rs" })
 * // => "file:///src/main.rs"
 * ```
 */
export function fExpandTemplate(pTemplate: string, pValues: Record<string, string>): string {
  cTemplateExpressionRegex.lastIndex = 0;

  return pTemplate.replace(
    cTemplateExpressionRegex,
    (_match, pOperator: string, pVarList: string) => {
      const aVarNames = pVarList
        .split(',')
        .map((pV: string) =>
          pV
            .replace(cVariableExplodeModifierRegex, '')
            .replace(cVariablePrefixModifierRegex, '')
            .trim()
        );
      const aExpandedParts = aVarNames
        .map((pName: string) => pValues[pName] ?? '')
        .filter((pV: string) => pV !== '');

      if (aExpandedParts.length === 0) return '';

      switch (pOperator) {
        case cUriTemplateSymbols.RESERVED:
          // Reserved expansion: no encoding
          return aExpandedParts.join(cUriTemplateSymbols.COMMA);
        case cUriTemplateSymbols.FRAGMENT:
          // Fragment expansion
          return cUriTemplateSymbols.FRAGMENT + aExpandedParts.join(cUriTemplateSymbols.COMMA);
        case cUriTemplateSymbols.PATH_SEGMENT:
          // Path segments
          return (
            cUriTemplateSymbols.PATH_SEGMENT + aExpandedParts.join(cUriTemplateSymbols.PATH_SEGMENT)
          );
        case cUriTemplateSymbols.LABEL:
          // Label expansion
          return cUriTemplateSymbols.LABEL + aExpandedParts.join(cUriTemplateSymbols.LABEL);
        case cUriTemplateSymbols.PATH_PARAM:
          // Path-style parameters
          return aVarNames
            .filter((p: string, pI: number) => aExpandedParts[pI])
            .map(
              (pName: string, pI: number) =>
                `${cUriTemplateSymbols.PATH_PARAM}${pName}=${aExpandedParts[pI]}`
            )
            .join('');
        case cUriTemplateSymbols.FORM_QUERY:
          // Form-style query
          return (
            cUriTemplateSymbols.FORM_QUERY +
            aVarNames
              .filter((p: string, pI: number) => aExpandedParts[pI])
              .map(
                (pName: string, pI: number) =>
                  `${encodeURIComponent(pName)}=${encodeURIComponent(aExpandedParts[pI])}`
              )
              .join(cUriTemplateSymbols.FORM_CONTINUATION)
          );
        case cUriTemplateSymbols.FORM_CONTINUATION:
          // Form-style query continuation
          return (
            cUriTemplateSymbols.FORM_CONTINUATION +
            aVarNames
              .filter((p: string, pI: number) => aExpandedParts[pI])
              .map(
                (pName: string, pI: number) =>
                  `${encodeURIComponent(pName)}=${encodeURIComponent(aExpandedParts[pI])}`
              )
              .join(cUriTemplateSymbols.COMMA)
          );
        default:
          // Simple string expansion (default operator)
          return aExpandedParts
            .map((pV: string) => encodeURIComponent(pV))
            .join(cUriTemplateSymbols.COMMA);
      }
    }
  );
}

/**
 * Check whether all required variables in a template have been provided.
 *
 * @param template - URI template string
 * @param values - Map of variable name to value
 * @returns true if all variables have non-empty values
 */
export function fIsTemplateComplete(pTemplate: string, pValues: Record<string, string>): boolean {
  const aVariables = fExtractTemplateVariables(pTemplate);

  return aVariables.every((pV) => (pValues[pV.name] ?? '').trim() !== '');
}
