import { ModelModality } from '$lib/enums';
import type { ModelOption } from '$lib/types/models';

/** Model list helpers: search, grouping and windowing of the selector rows. */

export interface ModelItem {
  option: ModelOption;
}

export interface OrgGroup {
  orgName: string | null;
  items: ModelItem[];
}

export interface GroupedModelOptions {
  available: OrgGroup[];
}

function fMatchesModality(pOption: ModelOption, pTerm: string): boolean {
  const cModalities = pOption.modalities;

  if (!cModalities) return false;

  switch (pTerm) {
    case ModelModality.VISION.toLowerCase():
      return cModalities.vision;
    case ModelModality.AUDIO.toLowerCase():
      return cModalities.audio;
    case ModelModality.VIDEO.toLowerCase():
      return cModalities.video;
    default:
      return false;
  }
}

export function fFilterModelOptions(pOptions: ModelOption[], pSearchTerm: string): ModelOption[] {
  const cTerm = pSearchTerm.trim().toLowerCase();

  if (!cTerm) return pOptions;

  return pOptions.filter(
    (pOption) =>
      pOption.model.toLowerCase().includes(cTerm) ||
      pOption.name?.toLowerCase().includes(cTerm) ||
      pOption.aliases?.some((pAlias: string) => pAlias.toLowerCase().includes(cTerm)) ||
      pOption.tags?.some((pTag: string) => pTag.toLowerCase().includes(cTerm)) ||
      fMatchesModality(pOption, cTerm)
  );
}

/**
 * Favorite models across every backend, in list order: the favorites tab spans
 * all backends, so they come from the full option list.
 */
export function fGroupFavoriteOptions(
  pOptions: ModelOption[],
  pFavoriteIds: Set<string>
): ModelItem[] {
  return pOptions
    .filter((pOption) => pFavoriteIds.has(pOption.model))
    .map((pOption) => ({ option: pOption }));
}

/**
 * Cut the local groups down to a window of rows, so the caller can grow it as
 * the list scrolls.
 */
export function fWindowLocalGroups(
  pGroups: GroupedModelOptions,
  pLimit: number
): { available: OrgGroup[]; shown: number } {
  let vBudget = pLimit;

  const aAvailable: OrgGroup[] = [];

  let vShown = 0;

  for (const cGroup of pGroups.available) {
    if (vBudget <= 0) break;

    const aItems = cGroup.items.slice(0, vBudget);

    vBudget -= aItems.length;
    vShown += aItems.length;

    if (aItems.length > 0) aAvailable.push({ ...cGroup, items: aItems });
  }

  return { available: aAvailable, shown: vShown };
}

export function fGroupModelOptions(pFilteredOptions: ModelOption[]): GroupedModelOptions {
  const aAvailable: OrgGroup[] = [];
  const dOrgGroups = new Map<string, ModelItem[]>();

  for (const cOption of pFilteredOptions) {
    const cKey = cOption.parsedId?.orgName ?? '';

    if (!dOrgGroups.has(cKey)) dOrgGroups.set(cKey, []);

    dOrgGroups.get(cKey)!.push({ option: cOption });
  }

  for (const [vOrgName, aItems] of dOrgGroups) {
    aAvailable.push({ items: aItems, orgName: vOrgName || null });
  }

  return { available: aAvailable };
}
