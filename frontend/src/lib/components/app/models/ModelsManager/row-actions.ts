import { fTranslate } from '$lib/i18n';
import { Eye, EyeOff, Heart, HeartOff, Trash2 } from '@lucide/svelte';
import { cModelDownloadIcons } from '$lib/constants';
import { ModelRowDownloadState } from '$lib/enums';
import { cModelsStore } from '$lib/stores';
import type { ModelOption } from '$lib/types/models';

/** Row actions follow the app's dropdown pattern: icon, label, separators, variants. */
export function fModelRowActions(
  pOption: ModelOption,
  pFavorite: boolean,
  pIsHidden: boolean,
  pOnDelete: (pOption: ModelOption) => void,
  /** Download state, when the row stands for a tracked download. */
  pDownload?: ModelRowDownloadState | null
) {
  // a tracked download is paused, resumed or dropped: it is not loaded, hidden or
  // drafted. Deleting it stops the download and removes what is on disk, which is
  // what the same wording offers on a cached model
  if (pDownload) {
    const cIsPaused = pDownload === ModelRowDownloadState.cPaused;

    return [
      {
        icon: cIsPaused ? cModelDownloadIcons.resume : cModelDownloadIcons.pause,
        label: cIsPaused ? 'Resume downloading' : 'Pause downloading',
        onclick: () =>
          void (cIsPaused
            ? cModelsStore.status.mDownloadModel(pOption.model)
            : cModelsStore.status.mPauseDownload(pOption.model))
      },
      {
        icon: Trash2,
        label: fTranslate('dgs_2a445ca5b1c7'),
        onclick: () => pOnDelete(pOption),
        separator: true,
        variant: 'destructive' as const
      }
    ];
  }

  return [
    {
      icon: pFavorite ? HeartOff : Heart,
      label: pFavorite ? 'Remove from favorites' : 'Add to favorites',
      onclick: () => cModelsStore.mToggleFavorite(pOption.model)
    },
    {
      icon: Trash2,
      label: fTranslate('dgs_2a445ca5b1c7'),
      onclick: () => pOnDelete(pOption),
      separator: true,
      variant: 'destructive' as const
    },
    {
      icon: pIsHidden ? Eye : EyeOff,
      label: pIsHidden ? 'Unhide model' : 'Hide model',
      onclick: () => cModelsStore.mToggleHidden(pOption.id),
      separator: true
    }
  ];
}
