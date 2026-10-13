import { cUiSettingsStore } from '$lib/stores/ui-settings.svelte';

/** String-valued UI preferences within the server's current settings document. */
export class SharedPreferencesService {
  static mGetItem(pKey: string): string | null {
    return cUiSettingsStore.mGet('preferences')?.[pKey] ?? null;
  }

  static mSetItem(pKey: string, pValue: string): void {
    cUiSettingsStore.mSet('preferences', {
      ...cUiSettingsStore.mGet('preferences'),
      [pKey]: pValue
    });
  }
}
