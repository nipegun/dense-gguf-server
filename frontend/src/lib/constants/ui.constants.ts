import { fTranslate } from '$lib/i18n';
import { ModelIcon } from './icons.constants';
import { Activity, Search, Settings, SquarePen } from '@lucide/svelte';
import McpLogo from '$lib/components/app/mcp/McpLogo.svelte';
import { SidebarAction, ToolSource } from '$lib/enums';
import type { DesktopIconStripItem } from '$lib/types';

export const cForkTreeDepthPadding = 8;
export const cSystemMessagePlaceholder = 'System message';

/** Data attributes for app-level DOM contracts. */
export const cUiDataAttrs = {
  ACTIVE: 'data-active',
  ACTIVE_TAB: 'data-active-tab',
  CONVERSATION_ROW: 'data-conversation-row',
  HIGHLIGHT_THEME_PREVIEW: 'data-highlight-theme-preview',
  PICKER_INDEX: 'data-picker-index',
  RESULT_INDEX: 'data-result-index',
  THUMBNAIL_INDEX: 'data-thumbnail-index'
} as const;

export const cToolGroupLabels = {
  [ToolSource.cBrowser]: 'Browser',
  [ToolSource.cCustom]: 'JSON Schema',
  [ToolSource.SERVER]: 'Server'
} as const;

export const cToolServerLabels = {
  [ToolSource.cBrowser]: 'Browser Tools',
  [ToolSource.cCustom]: 'Custom Tools',
  [ToolSource.SERVER]: 'Server Tools'
} as const;

export const cTooltipDelayDuration = 500;

export const cViewportGutter = 8;
export const cMenuOffset = 6;

export const cProcessingInfoTimeout = 2000;

/**
 * Statistics units labels
 */
export const cStatsUnits = {
  TOKENS_PER_SECOND: 't/s'
} as const;

export const cDefaultMobileBreakpoint = 768;

/** Models listed per remote provider before the "+ X more" line; search covers the rest. */
export const cRemoteProviderModelLimit = 12;

/** Orgs whose avatar is dark and needs inverting in dark mode. */
export const aDarkInvertAvatarOrgs = ['openai'];

/** Model rows mounted before the list scrolls; a local catalog can hold hundreds. */
export const cModelRowWindow = 50;

/** Models of one family shown before the rest fold behind a "show more" row. */
export const cFamilyRowWindow = 6;

/** Recently used models kept per browser, most recent first. */
export const cRecentModelLimit = 20;

export const cIconStripTransitionDuration = 150;
export const cIconStripTransitionDelayMultiplier = 50;

/** Max height for tool-result code blocks (json / source / diff / streaming code). */
export const cMaxHeightCodeBlock = '22rem';

export const aSidebarActionsItems: DesktopIconStripItem[] = [
  {
    action: SidebarAction.cNewChat,
    icon: SquarePen,
    keys: ['shift', 'cmd', 'o'],
    tooltip: fTranslate('text_db18382a249e')
  },
  { icon: Search, keys: ['cmd', 'k'], tooltip: fTranslate('text_49c266baaaa7') },
  {
    action: SidebarAction.cManageModels,
    icon: ModelIcon,
    tooltip: fTranslate('dgs_d17d2d78d76e')
  },
  {
    action: SidebarAction.cMcp,
    icon: McpLogo,
    tooltip: fTranslate('dgs_53f13ae99ed5')
  },
  {
    action: SidebarAction.cSettings,
    icon: Settings,
    tooltip: fTranslate('text_74a883a037bc')
  },
  {
    icon: Activity,
    tooltip: fTranslate('text_4c2e1df4570b'),
    route: '#/monitor',
    activeRouteId: '/monitor'
  }
];
