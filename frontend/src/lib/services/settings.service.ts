import { browser } from '$app/environment';
import { cUiSettingsStore } from '$lib/stores/ui-settings.svelte';

const cApiKeyStorage = 'DgsUi.apiKey';

/** Shared configuration is read from the server; only the authentication key stays here. */
export class SettingsService {
  static mLoadApiKey(): string {
    return browser ? (localStorage.getItem(cApiKeyStorage) ?? '') : '';
  }

  static mLoadConfig(): {
    config: Record<string, unknown>;
    userOverrides: string[];
    isFirstVisit: boolean;
  } {
    return {
      config: { ...cUiSettingsStore.mGet('config'), apiKey: this.mLoadApiKey() },
      userOverrides: cUiSettingsStore.mGet('userOverrides') ?? [],
      isFirstVisit: cUiSettingsStore.vIsFirstRun
    };
  }

  static mSaveConfig(pConfig: Record<string, unknown>, pUserOverrides: string[]): void {
    if (!browser) return;
    const { apiKey: vApiKey, ...dConfig } = pConfig;
    localStorage.setItem(cApiKeyStorage, typeof vApiKey === 'string' ? vApiKey : '');
    cUiSettingsStore.mSet(
      'config',
      dConfig as Record<string, string | number | boolean | undefined>
    );
    cUiSettingsStore.mSet('userOverrides', pUserOverrides);
    if (!cUiSettingsStore.vIsLoaded) void cUiSettingsStore.mLoad();
  }
}
