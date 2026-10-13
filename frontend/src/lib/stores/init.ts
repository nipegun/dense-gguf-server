import { cUiSettingsStore } from './ui-settings.svelte';
import { cModelsStore } from './models/index.svelte';
import { SettingsService } from '$lib/services/settings.service';
// direct imports, not via the barrel, to avoid circular deps
import { cConversationsStore } from './conversations/index.svelte';
import { cPermissionsStore } from './permissions.svelte';
import { cSettingsStore } from './settings/index.svelte';
import { cTabsStore } from './tabs.svelte';
import { cToolsStore } from './tools.svelte';
import { cVersionStore } from './version.svelte';
import { browser } from '$app/environment';

let vStartup: Promise<void> | null = null;

export function fInitStores(): Promise<void> {
  if (!browser) return Promise.resolve();

  vStartup ??= (async () => {
    cSettingsStore.vConfig.apiKey = SettingsService.mLoadApiKey();
    await cUiSettingsStore.mLoad();
    if (!cUiSettingsStore.vIsLoaded) {
      vStartup = null;
      return;
    }
    cSettingsStore.mInitialize();
    cModelsStore.mReloadSharedSettings();
    cConversationsStore.preferences.mReloadSharedSettings();
    cPermissionsStore.mInitialize();
    cToolsStore.mInitialize();
    void cVersionStore.mInitialize();

    // the full conversation list loads in the background; once it is back,
    // prune persisted tabs against the conversations that still exist
    void cConversationsStore.mInitialize().then(() => {
      cTabsStore.mInit(cConversationsStore.aConversations.map((pC) => pC.id));
    });
  })();

  return vStartup;
}
