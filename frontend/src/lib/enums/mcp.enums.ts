/**
 * Connection lifecycle phases for MCP protocol
 */
export enum MCPConnectionPhase {
  cCapabilitiesExchanged = 'capabilities_exchanged',
  cConnected = 'connected',
  cDisconnected = 'disconnected',
  ERROR = 'error',
  cIdle = 'idle',
  cInitializing = 'initializing',
  cListingTools = 'listing_tools',
  cTransportCreating = 'transport_creating',
  cTransportReady = 'transport_ready'
}

/**
 * Log level for connection events
 */
export enum MCPLogLevel {
  ERROR = 'error',
  INFO = 'info',
  cWarn = 'warn'
}

/**
 * Transport types for MCP connections
 */
export enum MCPTransportType {
  cStreamableHttp = 'streamable_http',
  cWebsocket = 'websocket'
}

/**
 * Health check status for MCP servers
 */
export enum HealthCheckStatus {
  cConnecting = 'connecting',
  ERROR = 'error',
  cIdle = 'idle',
  SUCCESS = 'success'
}

/**
 * Content types for MCP tool results
 */
export enum MCPContentType {
  IMAGE = 'image',
  RESOURCE = 'resource',
  TEXT = 'text'
}

/**
 * JSON Schema types used in MCP tool definitions
 */
export enum JsonSchemaType {
  cNumber = 'number',
  cObject = 'object',
  cString = 'string'
}

/**
 * Reference types for MCP completions
 */
export enum MCPRefType {
  PROMPT = 'ref/prompt',
  RESOURCE = 'ref/resource'
}
