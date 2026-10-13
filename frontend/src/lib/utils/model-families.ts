import { fNameOf } from '$lib/utils/model-names';

/** One family of a model list, and the entries it covers. */
export interface ModelFamilyGroup<T> {
  entries: T[];
  key: string;
  label: string;
}

/** Leading letters of a repo name, e.g. the `Qwen` of `Qwen3.8-27B`. */
const cFamilyLetters = /^[A-Za-z]+/;
/** Separators between the segments of a repo name, e.g. the `Qwen3` of `Qwen3-8B`. */
const cFamilySeparators = /[-_.]/;

/**
 * Family a repo belongs to: the leading letters of its name, so `Qwen3.5` and
 * `Qwen3.8-27B` both read as `Qwen`.
 */
export function fModelFamilyKey(pRepo: string): string {
  const name = fNameOf(pRepo);
  const cLetters = name.match(cFamilyLetters);

  return cLetters ? cLetters[0] : name.split(cFamilySeparators)[0] || name;
}

/** Fold entries into families, so `Qwen` collects its sizes and variants. */
export function fGroupModelFamilies<T>(
  entries: T[],
  pModelOf: (pEntry: T) => string
): ModelFamilyGroup<T>[] {
  const dFamilies = new Map<string, ModelFamilyGroup<T>>();

  for (const cEntry of entries) {
    const cKey = fModelFamilyKey(pModelOf(cEntry));
    const cFamily = dFamilies.get(cKey);

    if (cFamily) {
      cFamily.entries.push(cEntry);

      continue;
    }

    dFamilies.set(cKey, { entries: [cEntry], key: cKey, label: cKey });
  }

  return Array.from(dFamilies.values());
}
