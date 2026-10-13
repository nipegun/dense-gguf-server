/** Shared settings document; API keys and conversations stay in the browser. */
export interface UiSettingsDocument {
  config?: Record<string, string | number | boolean | undefined>;
  userOverrides?: string[];
  language?: string;
  preferences?: Record<string, string>;
}
