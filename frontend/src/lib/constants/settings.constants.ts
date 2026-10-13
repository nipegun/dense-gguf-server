import { fTranslate } from '$lib/i18n';
import { cCliFlags } from './cli-flags.constants';
import { cDefaultMcpConfig } from './mcp.constants';
import { cSettingsKeys } from './settings-keys.constants';
import { cTitleGeneration } from './title-generation.constants';
import { cFileGlobSearchPickers } from './working-directory.constants';
import {
  Code,
  Database,
  Funnel,
  ListRestart,
  Monitor,
  Moon,
  PencilRuler,
  SlidersVertical,
  Sun
} from '@lucide/svelte';
import { SyncableParameterType } from '$lib/enums';
import { SettingsFieldType } from '$lib/enums/settings.enums';
import { ColorMode } from '$lib/enums/ui.enums';
import type {
  SettingsConfigValue,
  SettingsEntry,
  SettingsFieldConfig,
  SettingsSection,
  SettingsSectionEntry
} from '$lib/types';

/** Settings sections — slug is the routing identity, title is the display label. */
export const cSettingsSections = {
  AGENTIC: { slug: 'agentic', title: fTranslate('text_7b1f65c25801') },
  DEVELOPER: { slug: 'developer', title: fTranslate('text_3fb7b39416f1') },
  DISPLAY: { slug: 'display', title: fTranslate('text_34e108c0896d') },
  GENERAL: { slug: 'general', title: fTranslate('text_c910d474dcd7') },
  IMPORT_EXPORT: { slug: 'import-export', title: fTranslate('text_42191ef46da3') },
  SAMPLING_PENALTIES: { slug: 'sampling-penalties', title: fTranslate('dgs_c2a6d6f845bf') },
  TOOLS: { slug: 'tools', title: fTranslate('text_ea93d6a262ec') }
} as const;

export const cSettingsSectionSlugs = {
  AGENTIC: cSettingsSections.AGENTIC.slug,
  DEVELOPER: cSettingsSections.DEVELOPER.slug,
  DISPLAY: cSettingsSections.DISPLAY.slug,
  GENERAL: cSettingsSections.GENERAL.slug,
  IMPORT_EXPORT: cSettingsSections.IMPORT_EXPORT.slug,
  SAMPLING_PENALTIES: cSettingsSections.SAMPLING_PENALTIES.slug,
  TOOLS: cSettingsSections.TOOLS.slug
} as const;

export const cSettingsSectionTitles = {
  AGENTIC: cSettingsSections.AGENTIC.title,
  DEVELOPER: cSettingsSections.DEVELOPER.title,
  DISPLAY: cSettingsSections.DISPLAY.title,
  GENERAL: cSettingsSections.GENERAL.title,
  IMPORT_EXPORT: cSettingsSections.IMPORT_EXPORT.title,
  SAMPLING_PENALTIES: cSettingsSections.SAMPLING_PENALTIES.title,
  TOOLS: cSettingsSections.TOOLS.title
} as const;

export const aSettingsRegistry: SettingsSectionEntry[] = [
  // General
  {
    icon: SlidersVertical,
    settings: [
      {
        key: cSettingsKeys.RESPOND_IN_INTERFACE_LANGUAGE,
        defaultValue: false,
        type: SettingsFieldType.cCheckbox,
        label: fTranslate('dgs_36a3e09d460a'),
        help: fTranslate('dgs_72904dc67da0')
      },

      {
        defaultValue: ColorMode.cSystem,
        help: fTranslate('text_d8c2701b0dce'),
        key: cSettingsKeys.THEME,
        label: fTranslate('text_efb52e7172b7'),
        options: [
          { icon: Monitor, label: fTranslate('text_6725e7bbcd28'), value: ColorMode.cSystem },
          { icon: Sun, label: fTranslate('text_dbcd5e7bb7a0'), value: ColorMode.cLight },
          { icon: Moon, label: fTranslate('text_60acc53f13a5'), value: ColorMode.DARK }
        ],
        type: SettingsFieldType.cSelect
      },
      {
        defaultValue: '',
        help: `Set the API Key if you are using <code> ${cCliFlags.API_KEY} </code> option for the server.`,
        isPrivate: true,
        key: cSettingsKeys.API_KEY,
        label: fTranslate('text_23189d55f697'),
        type: SettingsFieldType.INPUT
      },
      {
        defaultValue: '',
        help: fTranslate('text_fbd01c02e202'),
        key: cSettingsKeys.SYSTEM_MESSAGE,
        label: fTranslate('text_8e5a3143184d'),
        type: SettingsFieldType.cTextarea
      },
      {
        defaultValue: true,
        help: fTranslate('text_8cae084b4d35'),
        key: cSettingsKeys.SHOW_SYSTEM_MESSAGE,
        label: fTranslate('text_b710ee80506f'),
        standaloneField: false,
        type: SettingsFieldType.cCheckbox
      },
      {
        defaultValue: 2500,
        help: fTranslate('text_d50c4251977a'),
        key: cSettingsKeys.PASTE_LONG_TEXT_TO_FILE_LEN,
        label: fTranslate('text_401b499e8e8f'),
        type: SettingsFieldType.INPUT
      },
      {
        defaultValue: true,
        help: fTranslate('text_819c91d143be'),
        key: cSettingsKeys.SEND_ON_ENTER,
        label: fTranslate('text_7240e16f8c6b'),
        type: SettingsFieldType.cCheckbox
      },
      {
        defaultValue: true,
        help: fTranslate('text_569a6b44e53d'),
        key: cSettingsKeys.AUTO_MIC_ON_EMPTY,
        label: fTranslate('text_8666460b20dc'),
        type: SettingsFieldType.cCheckbox
      },
      {
        defaultValue: false,
        help: fTranslate('text_c25279367499'),
        isExperimental: true,
        key: cSettingsKeys.ENABLE_CONTINUE_GENERATION,
        label: fTranslate('text_086d634416fb'),
        type: SettingsFieldType.cCheckbox
      },
      {
        // Deliberately off for now: the natural place to turn it on is the first
        // run experience, once onboarding exists to ask the user about it.
        defaultValue: false,
        help: fTranslate('dgs_184e84dffa8b'),
        key: cSettingsKeys.USE_HUGGING_FACE_HUB,
        label: fTranslate('dgs_7d0ac338d59c'),
        type: SettingsFieldType.cCheckbox
      },
      {
        defaultValue: true,
        help: fTranslate('text_2798ae6854ed'),
        key: cSettingsKeys.TITLE_GENERATION_USE_FIRST_LINE,
        label: fTranslate('text_5b5cd3e853ae'),
        radioOptions: [
          {
            key: cSettingsKeys.TITLE_GENERATION_USE_FIRST_LINE,
            label: fTranslate('text_d260b3167eca'),
            value: 'firstLine'
          },
          {
            isExperimental: true,
            key: cSettingsKeys.TITLE_GENERATION_USE_LLM,
            label: fTranslate('text_979477c8d433'),
            value: 'llm'
          }
        ],
        type: SettingsFieldType.cRadio
      },
      {
        defaultValue: cTitleGeneration.DEFAULT_PROMPT,
        dependsOn: cSettingsKeys.TITLE_GENERATION_USE_LLM,
        help: fTranslate('text_d737cb824eb4'),
        key: cSettingsKeys.TITLE_GENERATION_PROMPT,
        label: fTranslate('text_20e4c1fbfbc0'),
        type: SettingsFieldType.cTextarea
      },
      {
        defaultValue: false,
        help: fTranslate('text_0e24ac7791c3'),
        key: cSettingsKeys.TITLE_GENERATION_USE_LLM,
        label: fTranslate('text_979477c8d433'),
        standaloneField: false,
        type: SettingsFieldType.cCheckbox
      },
      {
        defaultValue: false,
        help: fTranslate('text_bac76e8ab9fb'),
        key: cSettingsKeys.COPY_TEXT_ATTACHMENTS_AS_PLAIN_TEXT,
        label: fTranslate('text_ba488dcfbd3e'),
        type: SettingsFieldType.cCheckbox
      },
      {
        defaultValue: false,
        help: fTranslate('text_96fe65afdb37'),
        key: cSettingsKeys.PDF_AS_IMAGE,
        label: fTranslate('text_9ec72f537886'),
        type: SettingsFieldType.cCheckbox
      },
      {
        defaultValue: 0,
        help: fTranslate('text_a9e27241ec14'),
        key: cSettingsKeys.MAX_IMAGE_RESOLUTION,
        label: fTranslate('text_2954877fdbea'),
        type: SettingsFieldType.INPUT
      }
    ],
    slug: cSettingsSectionSlugs.GENERAL,
    title: cSettingsSectionTitles.GENERAL
  },
  // Display
  {
    icon: Monitor,
    settings: [
      {
        key: cSettingsKeys.SHOW_STATUS_BAR,
        defaultValue: true,
        type: SettingsFieldType.cCheckbox,
        label: fTranslate('dgs_5f34fb50b1b2'),
        help: fTranslate('dgs_365440b4d1ad')
      },
      {
        key: cSettingsKeys.EXPAND_LOCAL_SHELL_COMMANDS,
        defaultValue: false,
        type: SettingsFieldType.cCheckbox,
        label: fTranslate('dgs_a57e42071b2b'),
        help: fTranslate('dgs_eae0134b7637')
      },

      {
        defaultValue: true,
        help: fTranslate('text_1e886427b492'),
        key: cSettingsKeys.SHOW_MESSAGE_STATS,
        label: fTranslate('text_58767901d07b'),
        type: SettingsFieldType.cCheckbox
      },
      {
        defaultValue: false,
        dependsOn: cSettingsKeys.SHOW_MESSAGE_STATS,
        help: fTranslate('text_d531a39769ee'),
        key: cSettingsKeys.SHOW_AGENTIC_TURN_STATS,
        label: fTranslate('text_795b48b44734'),
        type: SettingsFieldType.cCheckbox
      },
      {
        defaultValue: true,
        help: fTranslate('text_a51f7f782ae8'),
        key: cSettingsKeys.SHOW_THOUGHT_IN_PROGRESS,
        label: fTranslate('text_c699c01d5c4c'),
        type: SettingsFieldType.cCheckbox
      },
      {
        defaultValue: false,
        help: fTranslate('text_0a711551837d'),
        key: cSettingsKeys.ALWAYS_SHOW_TOOL_CALL_CONTENT,
        label: fTranslate('text_9ae2d6e1f7ed'),
        type: SettingsFieldType.cCheckbox
      },
      {
        defaultValue: true,
        help: fTranslate('dgs_9f72dcfe1de8'),
        key: cSettingsKeys.RENDER_USER_CONTENT_AS_MARKDOWN,
        label: fTranslate('text_11910b83c707'),
        type: SettingsFieldType.cCheckbox
      },
      {
        defaultValue: true,
        help: fTranslate('text_3b2811cf8cbc'),
        key: cSettingsKeys.RENDER_THINKING_AS_MARKDOWN,
        label: fTranslate('text_047a9804886c'),
        type: SettingsFieldType.cCheckbox
      },
      {
        defaultValue: false,
        help: fTranslate('text_7f545155cd7c'),
        key: cSettingsKeys.FULL_HEIGHT_CODE_BLOCKS,
        label: fTranslate('text_ee7633d878e8'),
        type: SettingsFieldType.cCheckbox
      },
      {
        defaultValue: false,
        help: fTranslate('text_3253a89f0cd9'),
        key: cSettingsKeys.DISABLE_AUTO_SCROLL,
        label: fTranslate('text_1d17af5ec5f4'),
        type: SettingsFieldType.cCheckbox
      },
      {
        defaultValue: false,
        help: fTranslate('text_2d79bf0c56fd'),
        key: cSettingsKeys.ALWAYS_SHOW_SIDEBAR_ON_DESKTOP,
        label: fTranslate('text_87776621d9c2'),
        type: SettingsFieldType.cCheckbox
      },
      {
        defaultValue: true,
        help: fTranslate('dgs_af8f1c9f28ca'),
        key: cSettingsKeys.CONVERSATION_TABS,
        label: fTranslate('dgs_b97856610c7e'),
        type: SettingsFieldType.cCheckbox
      },
      {
        defaultValue: false,
        help: fTranslate('text_980c9fab0818'),
        key: cSettingsKeys.SHOW_RAW_MODEL_NAMES,
        label: fTranslate('text_8dfac9650c4a'),
        type: SettingsFieldType.cCheckbox
      },
      {
        defaultValue: true,
        help: fTranslate('text_49aedab8ac92'),
        key: cSettingsKeys.SHOW_MODEL_QUANTIZATION,
        label: fTranslate('text_b1c372445053'),
        type: SettingsFieldType.cCheckbox
      },
      {
        defaultValue: true,
        help: fTranslate('text_0ed5075d0dba'),
        key: cSettingsKeys.SHOW_MODEL_TAGS,
        label: fTranslate('text_68d073cd1d69'),
        type: SettingsFieldType.cCheckbox
      },
      {
        defaultValue: false,
        help: fTranslate('dgs_d1ff795aca20'),
        key: cSettingsKeys.SHOW_MODEL_CAPABILITIES_IN_SELECTOR,
        label: fTranslate('dgs_665edbbc7697'),
        type: SettingsFieldType.cCheckbox
      },
      {
        defaultValue: true,
        help: fTranslate('dgs_696df85e7885'),
        key: cSettingsKeys.SHOW_MODEL_ORG_NAME,
        label: fTranslate('dgs_b2a824c71ab7'),
        type: SettingsFieldType.cCheckbox
      },
      {
        defaultValue: false,
        help: fTranslate('text_216237a41c51'),
        key: cSettingsKeys.SHOW_BUILD_VERSION,
        label: fTranslate('text_4d8d312c1cf1'),
        type: SettingsFieldType.cCheckbox
      },
      {
        defaultValue: false,
        help: fTranslate('dgs_bd846f86005a'),
        key: cSettingsKeys.SHOW_FULL_PATH_IN_MENTIONS,
        label: fTranslate('dgs_c1ad5789407e'),
        type: SettingsFieldType.cCheckbox
      },
      {
        defaultValue: true,
        help: fTranslate('dgs_f142dcecb9bf'),
        key: cSettingsKeys.GROUP_MODELS_BY_FAMILY,
        label: fTranslate('dgs_06d0d4a15adc'),
        type: SettingsFieldType.cCheckbox
      }
    ],
    slug: cSettingsSectionSlugs.DISPLAY,
    title: cSettingsSectionTitles.DISPLAY
  },
  // MCP Servers (non-UI config object)
  {
    icon: PencilRuler,
    settings: [
      {
        defaultValue: '[]',
        help: fTranslate('text_7b544c202be9'),
        key: cSettingsKeys.MCP_SERVERS,
        label: fTranslate('text_22a7559f09bf'),
        standaloneField: false,
        type: SettingsFieldType.INPUT
      }
    ],
    slug: cSettingsSectionSlugs.TOOLS,
    title: cSettingsSectionTitles.TOOLS
  },
  // Tools
  {
    icon: ListRestart,
    settings: [
      {
        defaultValue: 10,
        help: fTranslate('text_f8db79fce786'),
        isPositiveInteger: true,
        key: cSettingsKeys.AGENTIC_MAX_TURNS,
        label: fTranslate('text_7e334c6eec91'),
        type: SettingsFieldType.INPUT
      },
      {
        defaultValue: cDefaultMcpConfig.requestTimeoutSeconds,
        help: fTranslate('text_eb6b3380faf2'),
        isPositiveInteger: true,
        key: cSettingsKeys.MCP_REQUEST_TIMEOUT_SECONDS,
        label: fTranslate('text_ce1c49ac8a96'),
        type: SettingsFieldType.INPUT
      },
      {
        defaultValue: cFileGlobSearchPickers.DEFAULT_SEARCH_DEPTH,
        help: fTranslate('dgs_e75e8c21b6d9'),
        isPositiveInteger: true,
        key: cSettingsKeys.MENTION_SEARCH_MAX_DEPTH,
        label: fTranslate('dgs_4183fb353c42'),
        max: cFileGlobSearchPickers.MAX_SEARCH_DEPTH,
        min: 1,
        placeholder: `${cFileGlobSearchPickers.DEFAULT_SEARCH_DEPTH}`,
        type: SettingsFieldType.INPUT
      }
    ],
    slug: cSettingsSectionSlugs.AGENTIC,
    title: cSettingsSectionTitles.AGENTIC
  },
  // Import/Export
  {
    icon: Database,
    settings: [],
    slug: cSettingsSectionSlugs.IMPORT_EXPORT,
    title: cSettingsSectionTitles.IMPORT_EXPORT
  },
  // Sampling
  {
    icon: Funnel,
    settings: [
      {
        defaultValue: undefined,
        help: fTranslate('text_4a6ef954759e'),
        key: cSettingsKeys.TEMPERATURE,
        label: fTranslate('text_b958ce8b871a'),
        sync: {
          paramType: SyncableParameterType.cNumber,
          serverKey: cSettingsKeys.TEMPERATURE
        },
        type: SettingsFieldType.INPUT
      },
      {
        defaultValue: undefined,
        help: fTranslate('text_ff2049bb8185'),
        key: cSettingsKeys.DYNATEMP_RANGE,
        label: fTranslate('text_262220def130'),
        sync: {
          paramType: SyncableParameterType.cNumber,
          serverKey: cSettingsKeys.DYNATEMP_RANGE
        },
        type: SettingsFieldType.INPUT
      },
      {
        defaultValue: undefined,
        help: fTranslate('text_ed4443b1e4aa'),
        key: cSettingsKeys.DYNATEMP_EXPONENT,
        label: fTranslate('text_e7499a75dbee'),
        sync: {
          paramType: SyncableParameterType.cNumber,
          serverKey: cSettingsKeys.DYNATEMP_EXPONENT
        },
        type: SettingsFieldType.INPUT
      },
      {
        defaultValue: undefined,
        help: fTranslate('text_8dbff0f8b0d9'),
        key: cSettingsKeys.TOP_K,
        label: fTranslate('text_74fc1150e902'),
        sync: { paramType: SyncableParameterType.cNumber, serverKey: cSettingsKeys.TOP_K },
        type: SettingsFieldType.INPUT
      },
      {
        defaultValue: undefined,
        help: fTranslate('text_c42b88da2f69'),
        key: cSettingsKeys.TOP_P,
        label: fTranslate('text_c7a8872a31da'),
        sync: { paramType: SyncableParameterType.cNumber, serverKey: cSettingsKeys.TOP_P },
        type: SettingsFieldType.INPUT
      },
      {
        defaultValue: undefined,
        help: fTranslate('text_bbd244420e7b'),
        key: cSettingsKeys.MIN_P,
        label: fTranslate('text_b0815a3244ed'),
        sync: { paramType: SyncableParameterType.cNumber, serverKey: cSettingsKeys.MIN_P },
        type: SettingsFieldType.INPUT
      },
      {
        defaultValue: undefined,
        help: fTranslate('text_c718b5a0c65c'),
        key: cSettingsKeys.XTC_PROBABILITY,
        label: fTranslate('text_f137399227a7'),
        sync: {
          paramType: SyncableParameterType.cNumber,
          serverKey: cSettingsKeys.XTC_PROBABILITY
        },
        type: SettingsFieldType.INPUT
      },
      {
        defaultValue: undefined,
        help: fTranslate('text_48793e125258'),
        key: cSettingsKeys.XTC_THRESHOLD,
        label: fTranslate('text_d4ba9412017a'),
        sync: {
          paramType: SyncableParameterType.cNumber,
          serverKey: cSettingsKeys.XTC_THRESHOLD
        },
        type: SettingsFieldType.INPUT
      },
      {
        defaultValue: undefined,
        help: fTranslate('text_2a4f56aa43d8'),
        key: cSettingsKeys.TYP_P,
        label: fTranslate('text_866fba8d4873'),
        sync: { paramType: SyncableParameterType.cNumber, serverKey: cSettingsKeys.TYP_P },
        type: SettingsFieldType.INPUT
      },
      {
        defaultValue: undefined,
        help: fTranslate('text_0afe6457ffaf'),
        key: cSettingsKeys.MAX_TOKENS,
        label: fTranslate('text_409e75c09fcc'),
        sync: {
          paramType: SyncableParameterType.cNumber,
          serverKey: cSettingsKeys.MAX_TOKENS
        },
        type: SettingsFieldType.INPUT
      },
      {
        defaultValue: '',
        help: fTranslate('text_0527a017301b'),
        key: cSettingsKeys.SAMPLERS,
        label: fTranslate('text_1c9adac18d8e'),
        sync: { paramType: SyncableParameterType.cString, serverKey: cSettingsKeys.SAMPLERS },
        type: SettingsFieldType.INPUT
      },
      {
        defaultValue: false,
        help: fTranslate('text_d18beca92501'),
        key: cSettingsKeys.BACKEND_SAMPLING,
        label: fTranslate('text_20d1867091ca'),
        type: SettingsFieldType.cCheckbox
      },
      {
        defaultValue: undefined,
        help: fTranslate('text_ea4a39630c59'),
        key: cSettingsKeys.REPEAT_LAST_N,
        label: fTranslate('text_c917bb45649a'),
        sync: {
          paramType: SyncableParameterType.cNumber,
          serverKey: cSettingsKeys.REPEAT_LAST_N
        },
        type: SettingsFieldType.INPUT
      },
      {
        defaultValue: undefined,
        help: fTranslate('text_df8fffc1e7e8'),
        key: cSettingsKeys.REPEAT_PENALTY,
        label: fTranslate('text_bbbc898044f9'),
        sync: {
          paramType: SyncableParameterType.cNumber,
          serverKey: cSettingsKeys.REPEAT_PENALTY
        },
        type: SettingsFieldType.INPUT
      },
      {
        defaultValue: undefined,
        help: fTranslate('text_1f5fec90f468'),
        key: cSettingsKeys.PRESENCE_PENALTY,
        label: fTranslate('text_5891319b9abe'),
        sync: {
          paramType: SyncableParameterType.cNumber,
          serverKey: cSettingsKeys.PRESENCE_PENALTY
        },
        type: SettingsFieldType.INPUT
      },
      {
        defaultValue: undefined,
        help: fTranslate('text_cc7e2f1ca212'),
        key: cSettingsKeys.FREQUENCY_PENALTY,
        label: fTranslate('text_7f9413709c95'),
        sync: {
          paramType: SyncableParameterType.cNumber,
          serverKey: cSettingsKeys.FREQUENCY_PENALTY
        },
        type: SettingsFieldType.INPUT
      },
      {
        defaultValue: undefined,
        help: fTranslate('text_e52fc11a5708'),
        key: cSettingsKeys.DRY_MULTIPLIER,
        label: fTranslate('text_a473fa39828e'),
        sync: {
          paramType: SyncableParameterType.cNumber,
          serverKey: cSettingsKeys.DRY_MULTIPLIER
        },
        type: SettingsFieldType.INPUT
      },
      {
        defaultValue: undefined,
        help: fTranslate('text_29ed70517b86'),
        key: cSettingsKeys.DRY_BASE,
        label: fTranslate('text_f982fb90c90b'),
        sync: { paramType: SyncableParameterType.cNumber, serverKey: cSettingsKeys.DRY_BASE },
        type: SettingsFieldType.INPUT
      },
      {
        defaultValue: undefined,
        help: fTranslate('text_06ba2e536c75'),
        key: cSettingsKeys.DRY_ALLOWED_LENGTH,
        label: fTranslate('text_88e074dbde59'),
        sync: {
          paramType: SyncableParameterType.cNumber,
          serverKey: cSettingsKeys.DRY_ALLOWED_LENGTH
        },
        type: SettingsFieldType.INPUT
      },
      {
        defaultValue: undefined,
        help: fTranslate('text_e804042a5e31'),
        key: cSettingsKeys.DRY_PENALTY_LAST_N,
        label: fTranslate('text_c8ab732e8d70'),
        sync: {
          paramType: SyncableParameterType.cNumber,
          serverKey: cSettingsKeys.DRY_PENALTY_LAST_N
        },
        type: SettingsFieldType.INPUT
      }
    ],
    slug: cSettingsSectionSlugs.SAMPLING_PENALTIES,
    title: cSettingsSectionTitles.SAMPLING_PENALTIES
  },
  // Developer
  {
    icon: Code,
    settings: [
      {
        defaultValue: false,
        help: fTranslate('text_b59a8e16b7c0'),
        key: cSettingsKeys.PRE_ENCODE_CONVERSATION,
        label: fTranslate('text_642329b42435'),
        type: SettingsFieldType.cCheckbox
      },
      {
        defaultValue: false,
        help: fTranslate('text_da5c0c85580d'),
        key: cSettingsKeys.DISABLE_REASONING_PARSING,
        label: fTranslate('text_70edf5da9d33'),
        type: SettingsFieldType.cCheckbox
      },
      {
        defaultValue: false,
        help: fTranslate('text_6c6741612135'),
        key: cSettingsKeys.EXCLUDE_REASONING_FROM_CONTEXT,
        label: fTranslate('text_1f3579b2968e'),
        type: SettingsFieldType.cCheckbox
      },
      {
        defaultValue: false,
        help: fTranslate('text_2da37fca08fe'),
        key: cSettingsKeys.SHOW_RAW_OUTPUT_SWITCH,
        label: fTranslate('text_922618c6a92e'),
        type: SettingsFieldType.cCheckbox
      },
      {
        defaultValue: false,
        help: fTranslate('text_328ff7861fd0'),
        key: cSettingsKeys.JS_SANDBOX_ENABLED,
        label: fTranslate('text_085046e2a92a'),
        type: SettingsFieldType.cCheckbox
      },
      {
        defaultValue: false,
        dependsOn: cSettingsKeys.JS_SANDBOX_ENABLED,
        help: fTranslate('text_03344003f473'),
        key: cSettingsKeys.SYMBOLIC_MATH_ENABLED,
        label: fTranslate('text_072ccd309854'),
        type: SettingsFieldType.cCheckbox
      },
      {
        defaultValue: '',
        help: fTranslate('text_c0eecd6c7ea5'),
        key: cSettingsKeys.CUSTOM_JSON,
        label: fTranslate('text_4b225723787e'),
        type: SettingsFieldType.cTextarea
      },
      {
        defaultValue: '',
        help: fTranslate('text_eddb4b393605'),
        key: cSettingsKeys.CUSTOM_CSS,
        label: fTranslate('text_9d3b9a0e78ea'),
        type: SettingsFieldType.cTextarea
      }
    ],
    slug: cSettingsSectionSlugs.DEVELOPER,
    title: cSettingsSectionTitles.DEVELOPER
  }
];

function fGetAllSettings(): SettingsEntry[] {
  const aResult: SettingsEntry[] = [];

  for (const cSection of aSettingsRegistry) {
    aResult.push(...cSection.settings);
  }

  return aResult;
}

/** Flat config object stored in localStorage. */
export const dSettingConfigDefault: Record<string, SettingsConfigValue> = Object.fromEntries(
  fGetAllSettings().map((pS) => [pS.key, pS.defaultValue])
) as Record<string, SettingsConfigValue>;

/** Help text for every setting (including non-UI). */
export const dSettingConfigInfo: Record<string, string> = Object.fromEntries(
  fGetAllSettings().map((pS) => [pS.key, pS.help])
) as Record<string, string>;

/** Sidebar sections + field configs (as consumed by UI). */
function fToSettingsSection(pSection: SettingsSectionEntry): SettingsSection {
  return {
    fields: pSection.settings
      .filter((pS) => pS.standaloneField !== false)
      .map((pS) => ({
        dependsOn: pS.dependsOn,
        help: pS.help,
        isExperimental: pS.isExperimental,
        isPositiveInteger: pS.isPositiveInteger,
        isPrivate: pS.isPrivate,
        key: pS.key,
        label: pS.label,
        max: pS.max,
        min: pS.min,
        options: pS.options as SettingsFieldConfig['options'],
        placeholder: pS.placeholder,
        radioOptions: pS.radioOptions,
        type: pS.type
      })),
    icon: pSection.icon,
    slug: pSection.slug,
    title: pSection.title
  };
}

/** Sidebar sections in custom display order (the registry array order). */
export const aSettingsChatSections: SettingsSection[] = aSettingsRegistry.map(fToSettingsSection);

/** INPUT-type settings whose value is a number. */
export const aNumericFields = fGetAllSettings()
  .filter((pS) => pS.type === SettingsFieldType.INPUT && typeof pS.defaultValue !== 'string')
  .map((pS) => pS.key) as readonly string[];

/** Numeric fields clamped to >= 1 and rounded. */
export const aPositiveIntegerFields = fGetAllSettings()
  .filter((pS) => pS.isPositiveInteger)
  .map((pS) => pS.key) as readonly string[];
