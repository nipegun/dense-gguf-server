/** Labels of the models manager table sections. */

import { ModelsTableGroupKind } from '$lib/enums';

export const dModelsTableGroupLabels: Record<ModelsTableGroupKind, string> = {
  [ModelsTableGroupKind.cDownloading]: 'Downloading',
  [ModelsTableGroupKind.cFavorites]: 'Favorites',
  [ModelsTableGroupKind.cHidden]: 'Hidden models',
  [ModelsTableGroupKind.cLoaded]: 'Loaded models',
  [ModelsTableGroupKind.cLocal]: 'Local models'
};

/**
 * Sticky offset of a group heading under its section header: one section-header
 * row (`py-2` over `text-[13px]` line height) minus the header's bottom border,
 * so the two stacked sticky rows do not show a gap or overlap by a pixel.
 */
export const cModelsTableGroupStickyOffset = 'top: calc(2.25rem - 1px)';
