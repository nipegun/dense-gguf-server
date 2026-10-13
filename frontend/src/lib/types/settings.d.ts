import type { ChatMessagePromptProgress, ChatMessageTimings } from './chat';
import type { DatabaseMessageExtra } from './database';
import type { OpenAIToolDefinition } from './mcp';
import type { Icon } from '@lucide/svelte';
import type { dSettingConfigDefault, cSettingsSectionTitles } from '$lib/constants';
import type {
  ParameterSource,
  ReasoningEffort,
  SettingsFieldType,
  StreamConnectionState,
  SyncableParameterType
} from '$lib/enums';
import type { Component } from 'svelte';

export type SettingsConfigValue = string | number | boolean | undefined;

/** Section title type derived from registry section titles. */
export type SettingsSectionTitle =
  (typeof cSettingsSectionTitles)[keyof typeof cSettingsSectionTitles];

/** Per-setting metadata — one entry per setting. */
export interface SettingsEntry {
  key: string;
  label: string;
  help: string;
  defaultValue: SettingsConfigValue;
  type: SettingsFieldType;
  options?: Array<{ value: string; label: string; icon: Component }>;
  /** Options rendered for RADIO fields. Each entry maps a `value` (the radio's selected value) to the underlying config `key` whose boolean state mirrors it. */
  radioOptions?: Array<{ value: string; label: string; key: string; isExperimental?: boolean }>;
  isExperimental?: boolean;
  isPositiveInteger?: boolean;
  /** When true, the field is rendered as a password input (e.g. API key). */
  isPrivate?: boolean;
  /** When false, the setting is stored/synced but has no standalone field; it is rendered by a sibling control or a dedicated page. */
  standaloneField?: boolean;
  placeholder?: string;
  min?: number;
  max?: number;
  dependsOn?: string;
  sync?: {
    serverKey: string;
    paramType: SyncableParameterType;
  };
}

/** A settings section with its icon, slug, title, and ordered settings. */
export interface SettingsSectionEntry {
  title: SettingsSectionTitle;
  slug: string;
  icon: Component;
  settings: SettingsEntry[];
}

export interface SettingsFieldConfig {
  key: string;
  label: string;
  type: SettingsFieldType;
  isExperimental?: boolean;
  isPositiveInteger?: boolean;
  isPrivate?: boolean;
  placeholder?: string;
  min?: number;
  max?: number;
  dependsOn?: string;
  help?: string;
  options?: Array<{ value: string; label: string; icon?: typeof Icon }>;
  /** Options rendered for RADIO fields. Each entry maps a `value` (the radio's selected value) to the underlying config `key` whose boolean state mirrors it. */
  radioOptions?: Array<{ value: string; label: string; key: string; isExperimental?: boolean }>;
}

/** Current settings section schema. */
export interface SettingsSection {
  fields?: SettingsFieldConfig[];
  icon: Component;
  slug: string;
  title: SettingsSectionTitle;
}

export interface SettingsChatServiceOptions {
  stream?: boolean;
  // Model (required in ROUTER mode, optional in MODEL mode)
  model?: string;
  // System message to inject
  systemMessage?: string;
  // Disable reasoning parsing (use 'none' instead of 'auto')
  disableReasoningParsing?: boolean;
  // Strip reasoning content from context before sending
  excludeReasoningFromContext?: boolean;
  // Enable model thinking/reasoning via chat_template_kwargs
  enableThinking?: boolean;
  // Reasoning effort level (low/medium/high/max) for thinking models
  reasoningEffort?: ReasoningEffort;
  tools?: OpenAIToolDefinition[];
  // Generation parameters
  temperature?: number;
  max_tokens?: number;
  // Sampling parameters
  dynatemp_range?: number;
  dynatemp_exponent?: number;
  top_k?: number;
  top_p?: number;
  min_p?: number;
  xtc_probability?: number;
  xtc_threshold?: number;
  typ_p?: number;
  // Penalty parameters
  repeat_last_n?: number;
  repeat_penalty?: number;
  presence_penalty?: number;
  frequency_penalty?: number;
  dry_multiplier?: number;
  dry_base?: number;
  dry_allowed_length?: number;
  dry_penalty_last_n?: number;
  // Sampler configuration
  samplers?: string | string[];
  backend_sampling?: boolean;
  // Custom JSON parameters
  customJson?: string;
  timings_per_token?: boolean;
  // Continuation control (vLLM compat), opt in to the explicit continue final message flag
  continueFinalMessage?: boolean;
  // Callbacks
  onChunk?: (pChunk: string) => void;
  onReasoningChunk?: (pChunk: string) => void;
  onToolCallChunk?: (pChunk: string) => void;
  onAttachments?: (pExtras: DatabaseMessageExtra[]) => void;
  onModel?: (pModel: string) => void;
  onCompletionId?: (pId: string) => void;
  onTimings?: (pTimings?: ChatMessageTimings, pPromptProgress?: ChatMessagePromptProgress) => void;
  onComplete?: (
    pResponse: string,
    pReasoningContent?: string,
    pTimings?: ChatMessageTimings,
    pToolCalls?: string
  ) => void;
  onError?: (pError: Error) => void;
  onConnectionState?: (pState: StreamConnectionState) => void;
}

export type SettingsConfigType = typeof dSettingConfigDefault & {
  [pKey: string]: SettingsConfigValue;
};

/**
 * Parameter synchronization types for server defaults and user overrides
 * Note: ParameterSource and SyncableParameterType enums are imported from '$lib/enums'
 */
export type ParameterValue = string | number | boolean;
export type ParameterRecord = Record<string, ParameterValue>;

export interface ParameterInfo {
  value: string | number | boolean;
  source: ParameterSource;
  serverDefault?: string | number | boolean;
  userOverride?: string | number | boolean;
}

export interface SyncableParameter {
  key: string;
  serverKey: string;
  type: SyncableParameterType;
  canSync: boolean;
}

/**
 * Shape of the settings JSON export file.
 * Versioned to allow future schema evolution.
 */
export interface SettingsExportType {
  /** Export format version — bumped on breaking changes */
  version: number;
  /** Unix timestamp of export */
  timestamp: number;
  /** Full settings config (includes theme as a config key) */
  config: SettingsConfigType;
  /** Keys that differ from server defaults (derived, but persisted for fidelity) */
  userOverrides: string[];
}
