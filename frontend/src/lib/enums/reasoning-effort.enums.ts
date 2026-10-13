/**
 * Reasoning effort levels for thinking models.
 * These values are sent to the server and mapped to token budgets.
 */
export enum ReasoningEffort {
  cDefault = 'default',
  cHigh = 'high',
  cLow = 'low',
  MAX = 'max',
  cMedium = 'medium',
  cOff = 'off'
}
