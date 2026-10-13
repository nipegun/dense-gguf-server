export enum ToolSource {
  cBrowser = 'browser',
  cCustom = 'custom',
  cMcp = 'mcp',
  SERVER = 'server'
}

export enum ToolPermissionDecision {
  cAlways = 'always',
  cAlwaysServer = 'always_server',
  cDeny = 'deny',
  cOnce = 'once'
}

export enum ToolResponseField {
  ERROR = 'error',
  cPlainText = 'plain_text_response'
}

/**
 * Entry types accepted by the `file_glob_search` tool's `type` parameter.
 * Mirrors the server-side validation in server-tools.cpp.
 */
export enum GlobSearchType {
  cAll = 'all',
  cDir = 'dir',
  FILE = 'file'
}

/**
 * Wire-format identifiers for server and browser tools. The string
 * value matches what the model emits in tool call names, so comparing
 * against `BuiltInTool.SERVER_READ_FILE` is equivalent to comparing
 * against the raw `'read_file'` literal - the enum just keeps the two in
 * lock-step and gives TypeScript a single source of truth for autocomplete
 * / rename support.
 *
 * The `SERVER_` / `BROWSER_` prefixes mirror the tool's primary source
 * (dgs-server vs dgs-ui). `get_info` is the exception: it is served by
 * the server, but dgs-ui falls back to a browser implementation when the
 * server does not provide it, so it can surface under both categories in
 * the UI while keeping a single wire name.
 */
export enum BuiltInTool {
  cBrowserGetDatetime = 'get_datetime',
  cBrowserReadMedia = 'read_media',
  cBrowserRunJavascript = 'run_javascript',
  cServerEditFile = 'edit_file',
  cServerExecShellCommand = 'exec_shell_command',
  cServerFileGlobSearch = 'file_glob_search',
  cServerGetInfo = 'get_info',
  cServerGrepSearch = 'grep_search',
  cServerReadFile = 'read_file',
  cServerWriteFile = 'write_file'
}
