import { fTranslate } from '$lib/i18n';
import { ReasoningEffort } from '$lib/enums';
import type { ReasoningEffortLevel } from '$lib/types';

/**
 * Reasoning effort UI labels.
 * Keys match the ReasoningEffort enum values for type-safe lookups.
 */
export const dReasoningEffortLabels: Record<string, string> = {
  [ReasoningEffort.cDefault]: 'Default',
  [ReasoningEffort.cHigh]: 'High',
  [ReasoningEffort.cLow]: 'Low',
  [ReasoningEffort.MAX]: 'Max',
  [ReasoningEffort.cMedium]: 'Medium',
  [ReasoningEffort.cOff]: 'Off'
};

export const aReasoningEffortLevels: ReasoningEffortLevel[] = [
  { label: fTranslate('text_21b111cbfe6e'), value: ReasoningEffort.cDefault },
  { label: fTranslate('text_ca7981b46ecf'), value: ReasoningEffort.cOff },
  { label: fTranslate('text_f793de205ead'), value: ReasoningEffort.cLow },
  { label: fTranslate('text_8e588cd18774'), value: ReasoningEffort.cMedium },
  { label: fTranslate('text_c4ebc6d4a583'), value: ReasoningEffort.cHigh },
  { hasInfo: true, label: fTranslate('text_a1a5936d3b0f'), value: ReasoningEffort.MAX }
];

/**
 * Reasoning effort to token budget mapping.
 * Maps the ReasoningEffort enum values to concrete token counts for the server.
 */
export const dReasoningEffortTokens: Record<string, number> = {
  [ReasoningEffort.cHigh]: 8192,
  [ReasoningEffort.cLow]: 512,
  [ReasoningEffort.MAX]: -1, // unlimited
  [ReasoningEffort.cMedium]: 2048
};
