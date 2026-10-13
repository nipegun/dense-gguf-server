export enum ColorMode {
  DARK = 'dark',
  cLight = 'light',
  cSystem = 'system'
}

export enum TooltipSide {
  BOTTOM = 'bottom',
  LEFT = 'left',
  RIGHT = 'right',
  TOP = 'top'
}

/**
 * ScrollCarousel arrow placement.
 */
export enum ScrollCarouselVariant {
  CENTER = 'center',
  TOP = 'top'
}

/**
 * Sidebar icon strip actions handled directly by the sidebar.
 */
export enum SidebarAction {
  cManageModels = 'manage-models',
  cMcp = 'mcp',
  cNewChat = 'new-chat',
  cSettings = 'settings'
}

/**
 * MCP prompt display variant
 */
export enum McpPromptVariant {
  ATTACHMENT = 'attachment',
  MESSAGE = 'message'
}

/**
 * URL prefixes for protocol detection
 */
export enum UrlProtocol {
  DATA = 'data:',
  FILE = 'file:',
  cHttp = 'http:',
  cHttps = 'https:',
  cWebsocket = 'ws:',
  cWebsocketSecure = 'wss:'
}

export enum HtmlInputType {
  FILE = 'file'
}

/**
 * Alert level that drives the context gauge dial color.
 */
export enum ColorLevel {
  cCritical = 'critical',
  cNeutral = 'neutral',
  OK = 'ok',
  cWarning = 'warning'
}
