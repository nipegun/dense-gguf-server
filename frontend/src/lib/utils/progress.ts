/**
 * Model load progress helpers for the /models/sse surfaces
 * (selector row and chat message).
 */

import { dModelLoadStageLabels, cModelLoadTailShare } from '$lib/constants';

/**
 * Human label for a model load stage.
 */
export function fModelLoadStageLabel(pStage: ApiModelLoadStage): string {
  return dModelLoadStageLabels[pStage];
}

/**
 * Overall load fraction (0.0 -> 1.0) across the declared stage plan.
 * text_model fills [0, 1 - tail], each later phase owns one tail slice.
 */
export function fModelLoadFraction(pProgress: ModelLoadProgress | null): number {
  if (!pProgress) return 0;

  // The server may emit a progress event before the stage plan is known, so
  // `stages` can be absent. Fall back to the raw value in that case.
  const { current: vCurrent, stages: vStages = [], value: vValue } = pProgress;
  const cTailCount = Math.max(vStages.length - 1, 0);
  const cTextCeiling = 1 - cTailCount * cModelLoadTailShare;
  const cIdx = vStages.indexOf(vCurrent);

  if (cIdx <= 0) {
    return vValue * cTextCeiling;
  }

  return cTextCeiling + (cIdx - 1 + vValue) * cModelLoadTailShare;
}

/**
 * Single line describing load progress: active stage label and overall percent.
 * Returns null when there is no progress to show.
 */
export function fModelLoadProgressText(pProgress: ModelLoadProgress | null): string | null {
  if (!pProgress) return null;

  const cLabel = fModelLoadStageLabel(pProgress.current);

  if (!cLabel) return null;

  return `${cLabel} ${Math.round(fModelLoadFraction(pProgress) * 100)}%`;
}
