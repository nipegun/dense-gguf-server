/**
 * ParameterSyncService - Syncs sampling parameters with the server
 *
 * Decides for each sampling parameter whether the user's setting is an
 * override of the server default, and normalizes floating-point values.
 * No reactive state; consumed by settingsStore.
 */

import { cSettingsKeys, aSettingsRegistry } from '$lib/constants';
import { ParameterSource, SyncableParameterType } from '$lib/enums';
import type { ParameterInfo, ParameterRecord, ParameterValue, SyncableParameter } from '$lib/types';
import { fNormalizeFloatingPoint } from '$lib/utils';

/** Mapping of UI setting keys to server parameter keys, derived from the registry. */
export const aSyncableParameters: SyncableParameter[] = aSettingsRegistry
  .flatMap((pSection) => pSection.settings)
  .filter((pS) => pS.sync !== undefined)
  .map((pS) => ({
    canSync: true,
    key: pS.key,
    serverKey: pS.sync!.serverKey,
    type: pS.sync!.paramType
  }));

export class ParameterSyncService {
  /**
   * Check if a parameter can be synced from server.
   *
   * @param key - The parameter key to check
   * @returns True if the parameter is in the syncable parameters list
   */
  static mCanSyncParameter(pKey: string): boolean {
    return aSyncableParameters.some((pParam) => pParam.key === pKey && pParam.canSync);
  }

  /**
   * Create a diff between current settings and server defaults.
   * Shows which parameters differ from server values, useful for debugging
   * and for the "Reset to defaults" functionality.
   *
   * @param currentSettings - Current parameter values in the settings store
   * @param serverDefaults - Default values extracted from server props
   * @returns Record of parameter diffs with current value, server value, and whether they differ
   */
  static mCreateParameterDiff(
    pCurrentSettings: ParameterRecord,
    pServerDefaults: ParameterRecord
  ): Record<string, { current: ParameterValue; server: ParameterValue; differs: boolean }> {
    const dDiff: Record<
      string,
      { current: ParameterValue; server: ParameterValue; differs: boolean }
    > = {};

    for (const cKey of this.mGetSyncableParameterKeys()) {
      const cCurrentValue = pCurrentSettings[cKey];
      const cServerValue = pServerDefaults[cKey];

      if (cServerValue !== undefined) {
        dDiff[cKey] = {
          current: cCurrentValue,
          differs: cCurrentValue !== cServerValue,
          server: cServerValue
        };
      }
    }

    return dDiff;
  }

  /**
   * Extract server default parameters that can be synced from `/props` response.
   * Handles both generation settings parameters and UI-specific settings.
   * Converts samplers array to semicolon-delimited string for UI display.
   *
   * @param serverParams - Raw generation settings from server `/props` endpoint
   * @returns Record of extracted parameter key-value pairs with normalized precision
   */
  static mExtractServerDefaults(
    pServerParams: ApiDgsServerProps['default_generation_settings']['params'] | null
  ): ParameterRecord {
    const cExtracted: ParameterRecord = {};

    if (pServerParams) {
      for (const cParam of aSyncableParameters) {
        if (cParam.canSync && cParam.serverKey in pServerParams) {
          const cValue = (pServerParams as unknown as Record<string, ParameterValue>)[
            cParam.serverKey
          ];

          if (cValue !== undefined) {
            // Apply precision rounding to avoid JavaScript floating-point issues
            cExtracted[cParam.key] = this.mRoundFloatingPoint(cValue);
          }
        }
      }

      // Handle samplers array conversion to string
      if (pServerParams.samplers && Array.isArray(pServerParams.samplers)) {
        cExtracted[cSettingsKeys.SAMPLERS] = pServerParams.samplers.join(';');
      }
    }

    return cExtracted;
  }

  /**
   * Get parameter information including source and values.
   * Used by SettingsChatParameterSourceIndicator to display the correct badge
   * (Custom vs Default) for each parameter in the settings UI.
   *
   * @param key - The parameter key to get info for
   * @param currentValue - The current value of the parameter
   * @param propsDefaults - Server default values from `/props`
   * @param userOverrides - Set of parameter keys explicitly overridden by the user
   * @returns Parameter info with source, server default, and user override values
   */
  static mGetParameterInfo(
    pKey: string,
    pCurrentValue: ParameterValue,
    pPropsDefaults: ParameterRecord,
    pUserOverrides: Set<string>
  ): ParameterInfo {
    const cHasPropsDefault = pPropsDefaults[pKey] !== undefined;
    const cIsUserOverride = pUserOverrides.has(pKey);
    // Simple logic: either using default (from props) or custom (user override)
    const cSource = cIsUserOverride ? ParameterSource.cCustom : ParameterSource.cDefault;

    return {
      serverDefault: cHasPropsDefault ? pPropsDefaults[pKey] : undefined, // Keep same field name for compatibility
      source: cSource,
      userOverride: cIsUserOverride ? pCurrentValue : undefined,
      value: pCurrentValue
    };
  }

  /**
   * Get all syncable parameter keys.
   *
   * @returns Array of parameter keys that can be synced from server
   */
  static mGetSyncableParameterKeys(): string[] {
    return aSyncableParameters.filter((pParam) => pParam.canSync).map((pParam) => pParam.key);
  }

  /**
   * Merge server defaults with current user settings.
   * User overrides always take priority — only parameters not in `userOverrides`
   * set will be updated from server defaults.
   *
   * @param currentSettings - Current parameter values in the settings store
   * @param serverDefaults - Default values extracted from server props
   * @param userOverrides - Set of parameter keys explicitly overridden by the user
   * @returns Merged parameter record with user overrides preserved
   */
  static mMergeWithServerDefaults(
    pCurrentSettings: ParameterRecord,
    pServerDefaults: ParameterRecord,
    pUserOverrides: Set<string> = new Set()
  ): ParameterRecord {
    const cMerged = { ...pCurrentSettings };

    for (const [vKey, vServerValue] of Object.entries(pServerDefaults)) {
      // Only update if user hasn't explicitly overridden this parameter
      if (!pUserOverrides.has(vKey)) {
        cMerged[vKey] = this.mRoundFloatingPoint(vServerValue);
      }
    }

    return cMerged;
  }

  /**
   * Validate a server parameter value against its expected type.
   *
   * @param key - The parameter key to validate
   * @param value - The value to validate
   * @returns True if value matches the expected type for this parameter
   */
  static mValidateServerParameter(pKey: string, pValue: ParameterValue): boolean {
    const cParam = aSyncableParameters.find((pP) => pP.key === pKey);

    if (!cParam) return false;

    switch (cParam.type) {
      case SyncableParameterType.cNumber:
        return typeof pValue === 'number' && !isNaN(pValue);
      case SyncableParameterType.cString:
        return typeof pValue === 'string';
      case SyncableParameterType.cBoolean:
        return typeof pValue === 'boolean';
      default:
        return false;
    }
  }

  /**
   * Round floating-point numbers to avoid JavaScript precision issues.
   * E.g., 0.1 + 0.2 = 0.30000000000000004 → 0.3
   *
   * @param value - Parameter value to normalize
   * @returns Precision-normalized value
   */
  private static mRoundFloatingPoint(pValue: ParameterValue): ParameterValue {
    return fNormalizeFloatingPoint(pValue) as ParameterValue;
  }
}
