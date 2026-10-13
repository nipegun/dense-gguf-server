import { MODEL_ID, type ModelSidecar, aSidecarTokens } from '$lib/constants';
import { ModelAuxSidecar, ModelDraftSidecar } from '$lib/enums';

const sSidecarTokenSet = new Set<string>(aSidecarTokens);
const sDraftSidecarSet = new Set<string>(Object.values(ModelDraftSidecar));
const sAuxSidecarSet = new Set<string>(Object.values(ModelAuxSidecar));

/** Map a lowercase filename token (e.g. `mtp`) to its sidecar enum value. */
export function fSidecarFromFileToken(pToken: string): ModelSidecar | null {
  return sSidecarTokenSet.has(pToken) ? (pToken as ModelSidecar) : null;
}

/**
 * Sidecar a download tag points at: the segment after the last dash,
 * e.g. `q4_0-mtp` -> `mtp`, `mmproj` -> `mmproj`. Returns null for quant-only
 * tags and tags whose tail is not a sidecar token.
 */
export function fSidecarFromTag(pTag: string): ModelSidecar | null {
  const cToken = pTag.toLowerCase().split(MODEL_ID.SEGMENT_SEPARATOR).pop() ?? '';

  return fSidecarFromFileToken(cToken);
}

export function fIsDraftSidecar(pSidecar: ModelSidecar): pSidecar is ModelDraftSidecar {
  return sDraftSidecarSet.has(pSidecar);
}

export function fIsAuxSidecar(pSidecar: ModelSidecar): pSidecar is ModelAuxSidecar {
  return sAuxSidecarSet.has(pSidecar);
}
