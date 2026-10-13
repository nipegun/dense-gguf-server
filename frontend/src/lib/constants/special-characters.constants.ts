// Control / whitespace / formatting characters that appear literally inside rendered text.

/** Line feed. */
export const cNewline = '\n';

/** Horizontal tab. */
export const cTab = '\t';

/** Non-breaking space. */
export const cNbsp = '\u00a0';

/** Non-breaking spaces used to render a tab stop that whitespace collapsing would otherwise squash. */
export const cTabAsSpaces = cNbsp.repeat(4);

/** Matches a CR-terminated or bare LF line break. */
export const cLineBreak = /\r?\n/;
