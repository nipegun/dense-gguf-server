import { SharedPreferencesService } from '$lib/services/shared-preferences.service';
/**
 * permissionsStore - Allowed tool permissions
 *
 * Owns the set of tools the user has permanently allowed, persisted to
 * localStorage. The agentic loop's permission gates consult it to run a
 * tool without prompting.
 */

import { browser } from '$app/environment';
import { cAlwaysAllowedToolsLocalstorageKey } from '$lib/constants';
import { SvelteSet } from 'svelte/reactivity';

class PermissionsStore {
  private _tools = $state(new SvelteSet<string>());

  get tools(): ReadonlySet<string> {
    return this._tools;
  }

  mAllowTool(pKey: string): void {
    this._tools.add(pKey);
    this.mPersist();
  }

  mAllowTools(pKeys: string[]): void {
    for (const cKey of pKeys) this._tools.add(cKey);
    this.mPersist();
  }

  mHasTool(pKey: string): boolean {
    return this._tools.has(pKey);
  }

  /**
   * Load persisted permissions. Called by initStores() after migrations
   * have run.
   */
  mInitialize(): void {
    // browser-only init: skip on SSR to avoid localStorage side effects
    if (!browser) return;

    try {
      const cStored = SharedPreferencesService.mGetItem(cAlwaysAllowedToolsLocalstorageKey);

      if (cStored) {
        for (const name of JSON.parse(cStored) as string[]) {
          if (typeof name === 'string') this._tools.add(name);
        }
      }
    } catch (vErr) {
      console.error(
        `Failed to load permissions from localStorage ("${cAlwaysAllowedToolsLocalstorageKey}"):`,
        vErr
      );
    }
  }

  mRevokeTool(pKey: string): void {
    this._tools.delete(pKey);
    this.mPersist();
  }

  private mPersist(): void {
    try {
      SharedPreferencesService.mSetItem(
        cAlwaysAllowedToolsLocalstorageKey,
        JSON.stringify([...this._tools])
      );
    } catch (vErr) {
      console.error(
        `Failed to persist to localStorage ("${cAlwaysAllowedToolsLocalstorageKey}"):`,
        vErr
      );
    }
  }
}

export const cPermissionsStore = new PermissionsStore();
