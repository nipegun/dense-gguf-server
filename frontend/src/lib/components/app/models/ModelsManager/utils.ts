import { cLocalBackendId, type ModalityKey } from '$lib/constants';
import { ModelCapability, ModelGroupKind, ModelsTableGroupKind } from '$lib/enums';
import { HuggingFaceService, ModelsService } from '$lib/services';
import { cModelsStore } from '$lib/stores';
import type { ModelDownloadEntry, ModelDownloadProgress, ModelOption } from '$lib/types/models';
import { fDetectThinkingSupport, fDetectToolUseSupport, fRepoOf } from '$lib/utils';

/** One repo of the table, with the rows it ships as. */
export interface ModelQuantGroup {
  base: ModelOption;
  key: string;
  kind: ModelGroupKind;
  quants: ModelOption[];
}

/** True when any of the manager's filters narrows the table. */
export function fHasActiveFilters(
  pContextLimit: number,
  pModalities: ModalityKey[],
  pCapabilities: ModelCapability[]
): boolean {
  return pContextLimit > 0 || pModalities.length > 0 || pCapabilities.length > 0;
}

/** Byte counts of a tracked download: live while it runs, frozen while paused. */
export function fDownloadProgressFor(pRepoWithTag: string): ModelDownloadProgress | null {
  return (
    cModelsStore.status.mGetDownloadProgress(pRepoWithTag) ??
    cModelsStore.status.mGetPausedDownloadProgress(pRepoWithTag)
  );
}

/**
 * One table row per tracked download, so the table lists them the way it lists
 * any other model. A paused download is already in the router's listing, so its
 * own option is reused; a fresh one stands for its tag alone.
 */
export function fDownloadGroups(
  entries: ModelDownloadEntry[],
  pModels: ModelOption[]
): ModelQuantGroup[] {
  return entries.map((pEntry) => {
    const cOption = pModels.find((pCandidate) => pCandidate.model === pEntry.repoWithTag) ?? {
      backendId: cLocalBackendId,
      capabilities: [],
      id: pEntry.repoWithTag,
      model: pEntry.repoWithTag,
      name: pEntry.repoWithTag
    };

    return {
      base: cOption,
      key: `download::${pEntry.repoWithTag}`,
      kind: ModelGroupKind.cQuants,
      quants: [cOption]
    };
  });
}

/** One collapsible block of the manager's table. */
export interface ModelsTableGroup {
  /** One entry per repo, its quants hanging off it. */
  items: ModelQuantGroup[];
  key: string;
  kind: ModelsTableGroupKind;
  label: string;
}

/** Context the model runs with: what a loaded model reports. */
export function fConfiguredContext(pOption: ModelOption): number | null {
  return cModelsStore.mIsModelRunning(pOption.model)
    ? cModelsStore.props.mGetModelContextSize(pOption.model)
    : null;
}

/**
 * Capability a model reports: true or false once its listing or its chat template
 * answers, null while the Hub record that carries the template is not read yet.
 */
export function fModelCapability(
  pOption: ModelOption,
  pCapability: ModelCapability
): boolean | null {
  if (pOption.capabilities.includes(pCapability)) return true;

  const cDetails = HuggingFaceService.mCachedDetails(pOption.model);

  // only the chat template answers this, and it arrives with the Hub record
  if (cDetails === undefined) return null;

  const cTemplate = cDetails?.gguf?.chat_template ?? '';

  return pCapability === ModelCapability.cToolUse
    ? fDetectToolUseSupport(cTemplate)
    : fDetectThinkingSupport(cTemplate);
}

/**
 * Context a model reports: the provider listing first, then the cached Hub record.
 * Null while neither of them has answered.
 */
export function fModelContextLength(pOption: ModelOption): number | null {
  return (
    pOption.contextLength ??
    HuggingFaceService.mCachedDetails(pOption.model)?.gguf?.context_length ??
    null
  );
}

/** Fold the quants of one repo into a single entry, so the table shows one row per model. */
export function fGroupModelQuants(pModels: ModelOption[]): ModelQuantGroup[] {
  const dGroups = new Map<string, ModelQuantGroup>();

  for (const cOption of pModels) {
    const cKey = fRepoOf(cOption.model);
    const cGroup = dGroups.get(cKey);

    if (cGroup) {
      cGroup.quants.push(cOption);

      continue;
    }

    dGroups.set(cKey, {
      base: cOption,
      key: cKey,
      kind: ModelGroupKind.cQuants,
      quants: [cOption]
    });
  }

  return Array.from(dGroups.values()).flatMap((pGroup) => {
    const cKind = fGroupKind(pGroup.quants);
    const aModelIds = pGroup.quants.map((pOption) => pOption.model);

    // the very same id twice is not a quant set; keep those rows apart
    if (aModelIds.length > 1 && aModelIds.every((pModel) => pModel === aModelIds[0])) {
      return pGroup.quants.map((pOption) => ({
        ...pGroup,
        base: pOption,
        key: `${pGroup.key}::${pOption.id}`,
        kind: ModelGroupKind.cVariants,
        quants: [pOption]
      }));
    }

    return [{ ...pGroup, kind: cKind }];
  });
}

/** What a group folds: quants or variants of one repo. */
function fGroupKind(pQuants: ModelOption[]): ModelGroupKind {
  const cIsQuant = pQuants.every(
    (pOption) => (pOption.parsedId ?? ModelsService.mParseModelId(pOption.model)).quantization
  );

  return cIsQuant ? ModelGroupKind.cQuants : ModelGroupKind.cVariants;
}

/**
 * Split the repos of a section into the quants that are hidden and the ones that
 * are not, so a partly hidden repo lists in both blocks instead of dragging its
 * visible quants into the hidden one.
 */
export function fSplitHiddenQuants(
  entries: ModelQuantGroup[],
  pIsHidden: (pOption: ModelOption) => boolean
): { hidden: ModelQuantGroup[]; local: ModelQuantGroup[] } {
  const aHidden: ModelQuantGroup[] = [];
  const aLocal: ModelQuantGroup[] = [];

  for (const cEntry of entries) {
    const aHiddenQuants = cEntry.quants.filter((pQuant) => pIsHidden(pQuant));
    const aLocalQuants = cEntry.quants.filter((pQuant) => !pIsHidden(pQuant));

    if (aHiddenQuants.length > 0) {
      aHidden.push({
        ...cEntry,
        base: aHiddenQuants[0],
        key: `${cEntry.key}::hidden`,
        quants: aHiddenQuants
      });
    }

    if (aLocalQuants.length > 0) {
      aLocal.push({ ...cEntry, base: aLocalQuants[0], quants: aLocalQuants });
    }
  }

  return { hidden: aHidden, local: aLocal };
}
