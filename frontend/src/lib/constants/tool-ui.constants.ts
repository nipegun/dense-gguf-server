import { fTranslate } from '$lib/i18n';
// Registry of server and browser tools whose renderer
// shows a recognizable icon and friendly label inline in the chat UI.
//
// To add a new tool, add an entry to TOOL_UI. To give a
// tool a custom title or body renderer, add a dedicated component under
// ChatMessageToolCall/ and route it in ChatMessageToolCallBlock.svelte
// (see ChatMessageToolCallBlockGetDatetime and
// ChatMessageToolCallBlockSearchResults for prior art).

import {
  Braces,
  Clock,
  Eye,
  FilePen,
  FilePlus,
  FileSearch,
  FileText,
  Info,
  SearchCode,
  Terminal
} from '@lucide/svelte';
import { BuiltInTool, ToolSource } from '$lib/enums';
import type { ToolUiEntry } from '$lib/types';

export const TOOL_UI: Readonly<Record<BuiltInTool, ToolUiEntry>> = {
  [BuiltInTool.cBrowserGetDatetime]: {
    icon: Clock,
    label: fTranslate('text_06dafda7802b'),
    source: ToolSource.cBrowser
  },
  [BuiltInTool.cBrowserReadMedia]: {
    icon: Eye,
    label: fTranslate('dgs_aa4c7701ac82'),
    source: ToolSource.cBrowser
  },
  [BuiltInTool.cBrowserRunJavascript]: {
    icon: Braces,
    label: fTranslate('text_9c90cb6e8c27'),
    source: ToolSource.cBrowser
  },
  [BuiltInTool.cServerEditFile]: {
    icon: FilePen,
    label: fTranslate('text_9608dd142d9b'),
    source: ToolSource.SERVER
  },
  [BuiltInTool.cServerExecShellCommand]: {
    icon: Terminal,
    label: fTranslate('text_87e30f34875f'),
    source: ToolSource.SERVER
  },
  [BuiltInTool.cServerFileGlobSearch]: {
    icon: FileSearch,
    label: fTranslate('text_179fed85ec50'),
    source: ToolSource.SERVER
  },
  [BuiltInTool.cServerGetInfo]: {
    icon: Info,
    label: fTranslate('dgs_0bc0f77ade2d'),
    source: ToolSource.SERVER
  },
  [BuiltInTool.cServerGrepSearch]: {
    icon: SearchCode,
    label: fTranslate('text_5fcd436a8042'),
    source: ToolSource.SERVER
  },
  [BuiltInTool.cServerReadFile]: {
    icon: FileText,
    label: fTranslate('text_47659c3dccc8'),
    source: ToolSource.SERVER
  },
  [BuiltInTool.cServerWriteFile]: {
    icon: FilePlus,
    label: fTranslate('text_997357233881'),
    source: ToolSource.SERVER
  }
} as const;
