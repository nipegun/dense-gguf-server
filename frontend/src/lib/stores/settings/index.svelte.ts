import { fTranslate } from '$lib/i18n';
/**
 * settingsStore - Application configuration and theme management
 *
 * Owns generation parameters, UI preferences and theme, persisted to
 * localStorage with Svelte 5 runes. Applies the admin's server ui_settings
 * as defaults on first visit; sampling parameters sync with the server via
 * ParameterSyncService.
 */

import { browser } from '$app/environment';
import { dSettingConfigDefault, cSettingsKeys } from '$lib/constants';
import { ColorMode } from '$lib/enums';
import { ParameterSyncService } from '$lib/services/parameter-sync.service';
import { SettingsService } from '$lib/services/settings.service';
import { cDeviceStore } from '$lib/stores/device.svelte';
// direct imports between stores, not via the barrel, to avoid circular deps
import { cServerStore } from '$lib/stores/server.svelte';
import type { SettingsExportType } from '$lib/types';
import {
  fConfigToParameterRecord,
  fGetConfigValue,
  fNormalizeFloatingPoint,
  fSetConfigValue
} from '$lib/utils';
import { setMode } from 'mode-watcher';

class SettingsStore {
  vConfig = $state<SettingsConfigType>({ ...dSettingConfigDefault });
  vIsInitialized = $state(false);
  sUserOverrides = $state<Set<string>>(new Set());

  // True until a config exists in localStorage; gates the one-time
  // application of server ui_settings defaults for new users.
  private vIsFirstVisit = false;

  mCanSyncParameter(pKey: string): boolean {
    return ParameterSyncService.mCanSyncParameter(pKey);
  }
  /**
   * Clear all user overrides (for debugging)
   */
  mClearAllUserOverrides(): void {
    this.sUserOverrides.clear();
    this.mSaveConfig();
    console.log('Cleared all user overrides');
  }

  /**
   * Export all settings as a versioned JSON-compatible object.
   * The export captures the full config (excluding sensitive values like API key)
   * and user overrides. Sensitive fields are filtered out for security by default.
   * @param includeSensitiveData - If true, include sensitive fields (apiKey, MCP server headers) in export
   */
  mExportSettings(pIncludeSensitiveData: boolean = false): SettingsExportType {
    // Build config excluding sensitive data unless user opts in
    const dConfigToExport: Record<string, string | number | boolean | undefined> =
      pIncludeSensitiveData
        ? { ...this.vConfig }
        : Object.fromEntries(Object.entries(this.vConfig).filter(([vKey]) => vKey !== 'apiKey'));

    // Handle MCP servers: exclude custom headers unless user opts in
    if ('mcpServers' in dConfigToExport && !pIncludeSensitiveData) {
      try {
        const aMcpServers = JSON.parse(dConfigToExport.mcpServers as string) as Array<
          Record<string, unknown>
        >;
        const aSafeServers = aMcpServers.map((pServer) => {
          delete pServer.headers;

          return pServer;
        });

        dConfigToExport.mcpServers = JSON.stringify(aSafeServers);
      } catch {
        // If parsing fails, just exclude the entire mcpServers field
        delete (dConfigToExport as Record<string, unknown>).mcpServers;
      }
    }

    return {
      config: dConfigToExport,
      timestamp: Date.now(),
      userOverrides: Array.from(this.sUserOverrides),
      version: 1
    };
  }

  /**
   * Reset all parameters to their default values (from props)
   * This is used by the "Reset to Default" functionality
   * Prioritizes Server defaults from /props, falls back to UI defaults
   */
  mForceSyncWithServerDefaults(): void {
    const dPropsDefaults = this.mGetServerDefaults();
    const dUiSettings = cServerStore.uiSettings;

    for (const cKey of ParameterSyncService.mGetSyncableParameterKeys()) {
      if (dUiSettings && cKey in dUiSettings) {
        // UI setting from admin config: write actual value
        fSetConfigValue(this.vConfig, cKey, dUiSettings[cKey]);
      } else if (dPropsDefaults[cKey] !== undefined) {
        // sampling param: clear it, let server decide
        fSetConfigValue(this.vConfig, cKey, '');
      } else if (cKey in dSettingConfigDefault) {
        fSetConfigValue(this.vConfig, cKey, fGetConfigValue(dSettingConfigDefault, cKey));
      }

      this.sUserOverrides.delete(cKey);
    }

    // Non-syncable keys: reset is a full return to the instance state, the
    // admin baseline value when defined, the factory default otherwise.
    for (const cKey of Object.keys(dSettingConfigDefault)) {
      if (ParameterSyncService.mCanSyncParameter(cKey)) {
        continue;
      }

      const cValue =
        dUiSettings && cKey in dUiSettings && dUiSettings[cKey] !== undefined
          ? dUiSettings[cKey]
          : fGetConfigValue(dSettingConfigDefault, cKey);

      fSetConfigValue(this.vConfig, cKey, cValue);

      if (cKey === cSettingsKeys.THEME) {
        setMode(cValue as ColorMode);
      }

      this.sUserOverrides.delete(cKey);
    }

    this.mSaveConfig();
  }

  /**
   * Get the entire configuration object
   * @returns The complete configuration object
   */
  mGetAllConfig(): SettingsConfigType {
    return { ...this.vConfig };
  }

  /**
   * Get a specific configuration value
   * @param key - The configuration key to get
   * @returns The configuration value
   */
  mGetConfig<K extends keyof SettingsConfigType>(pKey: K): SettingsConfigType[K] {
    return this.vConfig[pKey];
  }

  /**
   * Get diff between current settings and server defaults
   */
  mGetParameterDiff() {
    const dServerDefaults = this.mGetServerDefaults();

    if (Object.keys(dServerDefaults).length === 0) return {};

    const dConfigAsRecord = fConfigToParameterRecord(
      this.vConfig,
      ParameterSyncService.mGetSyncableParameterKeys()
    );

    return ParameterSyncService.mCreateParameterDiff(dConfigAsRecord, dServerDefaults);
  }

  /**
   * Get parameter information including source for a specific parameter
   */
  mGetParameterInfo(pKey: string) {
    const dPropsDefaults = this.mGetServerDefaults();
    const cCurrentValue = fGetConfigValue(this.vConfig, pKey);

    return ParameterSyncService.mGetParameterInfo(
      pKey,
      cCurrentValue ?? '',
      dPropsDefaults,
      this.sUserOverrides
    );
  }

  /**
   * Import settings from a previously exported object.
   * Restores config (including theme) and user overrides.
   * @param data - The exported settings object
   */
  mImportSettings(pData: SettingsExportType): void {
    if (!browser) return;

    if (!pData || !pData.config) {
      throw new Error(fTranslate('text_0a8d4a6abfc5'));
    }

    // Restore config (theme is included in config)
    this.vConfig = {
      ...dSettingConfigDefault,
      ...pData.config
    };

    // Restore user overrides (derived state — may be stale if server defaults differ)
    this.sUserOverrides = new Set(pData.userOverrides ?? []);

    // Persist to localStorage
    this.mSaveConfig();

    // Apply theme for immediate visual feedback
    setMode(this.vConfig[cSettingsKeys.THEME] as ColorMode);

    console.log('Settings imported successfully');
  }

  /**
   * Initialize the settings store by loading from localStorage.
   * Called during current-schema store initialization.
   */
  mInitialize() {
    if (!browser) return;

    try {
      this.mLoadConfig();
      // Apply the persisted theme from config on initial load
      setMode(this.vConfig[cSettingsKeys.THEME] as ColorMode);
      this.vIsInitialized = true;
    } catch (error) {
      console.error('Failed to initialize settings store:', error);
    }
  }

  /**
   * Reset all settings to defaults.
   */
  mResetAll() {
    this.mResetConfig();

    this.mResetTheme();
  }

  /**
   * Reset configuration to defaults
   */
  mResetConfig() {
    this.vConfig = { ...dSettingConfigDefault };

    this.mSaveConfig();
  }

  /**
   * Reset a parameter to Server default (or UI default if no Server default)
   */
  mResetParameterToServerDefault(pKey: string): void {
    const dServerDefaults = this.mGetServerDefaults();
    const dUiSettings = cServerStore.uiSettings;

    if (dUiSettings && pKey in dUiSettings) {
      // UI setting from admin config: write actual value
      fSetConfigValue(this.vConfig, pKey, dUiSettings[pKey]);
    } else if (dServerDefaults[pKey] !== undefined) {
      // sampling param known by server: clear it, let server decide
      fSetConfigValue(this.vConfig, pKey, '');
    } else if (pKey in dSettingConfigDefault) {
      fSetConfigValue(this.vConfig, pKey, fGetConfigValue(dSettingConfigDefault, pKey));
    }

    this.sUserOverrides.delete(pKey);
    this.mSaveConfig();
  }

  /**
   * Reset theme to default value.
   * Theme is now stored inside the config object.
   */
  mResetTheme() {
    this.mUpdateConfig(cSettingsKeys.THEME, dSettingConfigDefault[cSettingsKeys.THEME]);

    setMode(dSettingConfigDefault[cSettingsKeys.THEME] as ColorMode);
  }

  /**
   * Initialize settings with props defaults when server properties are first loaded
   * This sets up the default values from /props endpoint
   */
  mSyncWithServerDefaults(): void {
    const dPropsDefaults = this.mGetServerDefaults();
    const dUiSettings = cServerStore.uiSettings;

    // a router main instance reports no sampling defaults, but its
    // ui_settings still need the first visit pass below
    if (Object.keys(dPropsDefaults).length === 0 && !(dUiSettings && this.vIsFirstVisit)) return;

    const sUiSettingsKeys = new Set(dUiSettings ? Object.keys(dUiSettings) : []);

    for (const [vKey, vPropsValue] of Object.entries(dPropsDefaults)) {
      const cCurrentValue = fGetConfigValue(this.vConfig, vKey);
      const cNormalizedCurrent = fNormalizeFloatingPoint(cCurrentValue);
      const cNormalizedDefault = fNormalizeFloatingPoint(vPropsValue);

      // if user value matches server, it's not a real override
      if (cNormalizedCurrent === cNormalizedDefault) {
        this.sUserOverrides.delete(vKey);

        if (
          !sUiSettingsKeys.has(vKey) &&
          fGetConfigValue(dSettingConfigDefault, vKey) === undefined
        ) {
          fSetConfigValue(this.vConfig, vKey, undefined);
        }
      }
    }

    // UI settings are the admin's defaults for new users: applied once on
    // the first visit, never on later loads, so the user's config can
    // diverge. "Reset to Default" is the explicit way back to the baseline.
    // A first visit config carries factory values only, so a key that
    // already diverges here was set by the user before the baseline could
    // be reached, through the API key splash, and stays theirs.
    if (dUiSettings && this.vIsFirstVisit) {
      this.vIsFirstVisit = false;

      for (const [vKey, vValue] of Object.entries(dUiSettings)) {
        if (vValue === undefined || this.sUserOverrides.has(vKey)) continue;

        if (fGetConfigValue(this.vConfig, vKey) !== fGetConfigValue(dSettingConfigDefault, vKey)) {
          continue;
        }

        fSetConfigValue(this.vConfig, vKey, vValue);

        // theme lives in mode-watcher, not just in config -> propagate
        if (vKey === cSettingsKeys.THEME) {
          setMode(vValue as ColorMode);
        }
      }
    }

    this.mSaveConfig();
    console.log('User overrides after sync:', Array.from(this.sUserOverrides));
  }

  /**
   * Update a specific configuration setting
   * @param key - The configuration key to update
   * @param value - The new value for the configuration key
   */
  mUpdateConfig<K extends keyof SettingsConfigType>(pKey: K, pValue: SettingsConfigType[K]): void {
    this.vConfig[pKey] = pValue;

    if (ParameterSyncService.mCanSyncParameter(pKey as string)) {
      const dPropsDefaults = this.mGetServerDefaults();
      const cPropsDefault = dPropsDefaults[pKey as string];

      if (cPropsDefault !== undefined) {
        const cNormalizedValue = fNormalizeFloatingPoint(pValue);
        const cNormalizedDefault = fNormalizeFloatingPoint(cPropsDefault);

        if (cNormalizedValue === cNormalizedDefault) {
          this.sUserOverrides.delete(pKey as string);
        } else {
          this.sUserOverrides.add(pKey as string);
        }
      }
    }

    this.mSaveConfig();
  }

  /**
   *
   *
   * Import / Export
   *
   *
   */

  /**
   * Update multiple configuration settings at once
   * @param updates - Object containing the configuration updates
   */
  mUpdateMultipleConfig(pUpdates: Partial<SettingsConfigType>) {
    Object.assign(this.vConfig, pUpdates);

    const dPropsDefaults = this.mGetServerDefaults();

    for (const [vKey, vValue] of Object.entries(pUpdates)) {
      if (ParameterSyncService.mCanSyncParameter(vKey)) {
        const cPropsDefault = dPropsDefaults[vKey];

        if (cPropsDefault !== undefined) {
          const cNormalizedValue = fNormalizeFloatingPoint(vValue);
          const cNormalizedDefault = fNormalizeFloatingPoint(cPropsDefault);

          if (cNormalizedValue === cNormalizedDefault) {
            this.sUserOverrides.delete(vKey);
          } else {
            this.sUserOverrides.add(vKey);
          }
        }
      }
    }

    this.mSaveConfig();
  }

  /**
   * Update the theme setting.
   * @param newTheme - The new theme value
   */
  mUpdateTheme(pNewTheme: string) {
    this.mUpdateConfig(cSettingsKeys.THEME, pNewTheme);

    setMode(pNewTheme as ColorMode);
  }

  /**
   *
   *
   * Utilities (private helpers)
   *
   *
   */

  /**
   * Helper method to get server defaults with null safety
   * Centralizes the pattern of getting and extracting server defaults
   */
  private mGetServerDefaults(): Record<string, string | number | boolean> {
    return ParameterSyncService.mExtractServerDefaults(cServerStore.defaultParams);
  }

  /**
   * Load configuration from localStorage via the persistence service.
   * Returns default values for missing keys to prevent breaking changes.
   */
  private mLoadConfig() {
    if (!browser) return;

    const {
      config: dSavedVal,
      isFirstVisit: vIsFirstVisit,
      userOverrides: aSavedOverrides
    } = SettingsService.mLoadConfig();

    // First visit: no stored config yet. Server ui_settings apply once in
    // this state, then the user's config diverges freely.
    this.vIsFirstVisit = vIsFirstVisit;

    // Merge with defaults to prevent breaking changes
    this.vConfig = {
      ...dSettingConfigDefault,
      ...dSavedVal
    };

    // Default sendOnEnter to false on mobile when the user has no saved preference
    if (!(cSettingsKeys.SEND_ON_ENTER in dSavedVal)) {
      if (cDeviceStore.isMobile) {
        this.vConfig[cSettingsKeys.SEND_ON_ENTER] = false;
      }
    }

    // Load user overrides
    this.sUserOverrides = new Set(aSavedOverrides);
  }

  /**
   * Save the current configuration to localStorage via the persistence service.
   */
  private mSaveConfig() {
    if (!browser) return;

    SettingsService.mSaveConfig(this.vConfig, Array.from(this.sUserOverrides));
  }
}

export const cSettingsStore = new SettingsStore();
