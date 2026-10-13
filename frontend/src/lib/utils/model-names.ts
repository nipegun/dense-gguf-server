import { cFilePathSeparatorRegex, MODEL_ID } from '$lib/constants';

/**
 * Normalizes a model name by extracting the filename from a path, but preserves Hugging Face repository format.
 *
 * Handles both forward slashes (/) and backslashes (\) as path separators.
 * - If the model name has exactly one slash (org/model format), preserves the full "org/model" name
 * - If the model name has no slash or multiple slashes, extracts just the filename
 * - If the model name is just a filename (no path), returns it as-is.
 *
 * @param modelName - The model name or path to normalize
 * @returns The normalized model name
 *
 * @example
 * normalizeModelName('models/dgs-3.1-8b') // Returns: 'dgs-3.1-8b' (multiple slashes -> filename)
 * normalizeModelName('C:\\Models\\gpt-4') // Returns: 'gpt-4' (multiple slashes -> filename)
 * normalizeModelName('Qwen/Qwen3.5-4B') // Returns: 'Qwen/Qwen3.5-4B' (Hugging Face format)
 * normalizeModelName('simple-model') // Returns: 'simple-model' (no slash)
 * normalizeModelName('  spaced  ') // Returns: 'spaced'
 * normalizeModelName('') // Returns: ''
 */
export function fNormalizeModelName(pModelName: string): string {
  const cTrimmed = pModelName.trim();

  if (!cTrimmed) {
    return '';
  }

  const aSegments = cTrimmed.split(cFilePathSeparatorRegex);

  // If we have exactly 2 segments (one slash), treat it as Hugging Face repo format
  // and preserve the full "org/model" format
  if (aSegments.length === 2) {
    const [vOrg, vModel] = aSegments;
    const cTrimmedOrg = vOrg?.trim();
    const cTrimmedModel = vModel?.trim();

    if (cTrimmedOrg && cTrimmedModel) {
      return `${cTrimmedOrg}/${cTrimmedModel}`;
    }
  }

  // For other cases (no slash, or multiple slashes), extract just the filename
  const cCandidate = aSegments.pop();
  const cNormalized = cCandidate?.trim();

  return cNormalized && cNormalized.length > 0 ? cNormalized : cTrimmed;
}

/**
 * Validates if a model name is valid (non-empty after normalization).
 *
 * @param modelName - The model name to validate
 * @returns true if valid, false otherwise
 */
export function fIsValidModelName(pModelName: string): boolean {
  return fNormalizeModelName(pModelName).length > 0;
}

/**
 * Org segment of a HuggingFace repo id (`unsloth/Qwen3.5-4B` -> `unsloth`).
 * Returns the input itself when it carries no org separator, and an empty string
 * for a missing id, so callers can use `||` against their own fallback org.
 */
export function fOrgOf(pRepoId: string | null | undefined): string {
  if (!pRepoId) return '';

  return pRepoId.split(MODEL_ID.ORG_SEPARATOR)[0] || pRepoId;
}

/**
 * Repo of a quantized model id (`unsloth/Qwen3.5-4B-GGUF:Q4_K_M` ->
 * `unsloth/Qwen3.5-4B-GGUF`), the id itself when it carries no quant tag.
 */
export function fRepoOf(pModelId: string): string {
  return pModelId.split(MODEL_ID.QUANTIZATION_SEPARATOR)[0] || pModelId;
}

/**
 * Name of a repo id past its org (`unsloth/Qwen3.5-4B-GGUF` -> `Qwen3-8B-GGUF`),
 * the input itself when it carries no org separator.
 */
export function fNameOf(pRepoId: string): string {
  return pRepoId.split(MODEL_ID.ORG_SEPARATOR).pop() || pRepoId;
}
