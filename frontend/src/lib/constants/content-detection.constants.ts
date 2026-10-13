/**
 * String patterns for detecting content kind from MIME types and URIs.
 * Used with startsWith/includes checks, not as discriminated values.
 */

export const cMimeTypePrefixes = {
  IMAGE: 'image/',
  TEXT: 'text'
} as const;

export const cMimeTypeSubstrings = {
  JAVASCRIPT: 'javascript',
  JSON: 'json',
  TYPESCRIPT: 'typescript'
} as const;

export const cUriPatterns = {
  DATABASE_KEYWORD: 'database',
  DATABASE_SCHEME: 'db://'
} as const;
