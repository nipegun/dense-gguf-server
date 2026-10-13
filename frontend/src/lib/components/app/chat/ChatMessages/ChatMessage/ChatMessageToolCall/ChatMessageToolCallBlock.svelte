<script lang="ts">
  import ChatMessageToolCallBlockLocalShell from './ChatMessageToolCallBlockLocalShell.svelte';
  import { fIsLocalShellTool } from './parsers/local-shell-run';
  import ChatMessageToolCallBlockDefault from './ChatMessageToolCallBlockDefault.svelte';
  import ChatMessageToolCallBlockEditFile from './ChatMessageToolCallBlockEditFile.svelte';
  import ChatMessageToolCallBlockExecShellCommand from './ChatMessageToolCallBlockExecShellCommand.svelte';
  import ChatMessageToolCallBlockFileGlobSearch from './ChatMessageToolCallBlockFileGlobSearch.svelte';
  import ChatMessageToolCallBlockGetDatetime from './ChatMessageToolCallBlockGetDatetime.svelte';
  import ChatMessageToolCallBlockGetInfo from './ChatMessageToolCallBlockGetInfo.svelte';
  import ChatMessageToolCallBlockGrepSearch from './ChatMessageToolCallBlockGrepSearch.svelte';
  import ChatMessageToolCallBlockReadFile from './ChatMessageToolCallBlockReadFile.svelte';
  import ChatMessageToolCallBlockReadMedia from './ChatMessageToolCallBlockReadMedia.svelte';
  import ChatMessageToolCallBlockRunJavascript from './ChatMessageToolCallBlockRunJavascript.svelte';
  import ChatMessageToolCallBlockSearchResults from './ChatMessageToolCallBlockSearchResults.svelte';
  import ChatMessageToolCallBlockWriteFile from './ChatMessageToolCallBlockWriteFile.svelte';
  import { BuiltInTool } from '$lib/enums';
  import type { AgenticSection, DatabaseMessageExtra } from '$lib/types';
  import {
    fExtractSearchQuery,
    fExtractSearchResults,
    fIsWebSearchToolName,
    fLooksLikeSearchResult
  } from '$lib/utils';

  interface Props {
    section: AgenticSection;
    attachments?: DatabaseMessageExtra[];
    open: boolean;
    isStreaming: boolean;
    isExecuting?: boolean;
    onToggle?: () => void;
  }

  let { attachments, isExecuting, isStreaming, onToggle, open, section }: Props = $props();

  // Runs for every tool block on mount, before the body renders: the cheap
  // content prefilter and the tool-name allow-list come first so blobs from
  // exec/file tools are never line-split or JSON-parsed here
  const cIsSearchCall = $derived.by(() => {
    if (fLooksLikeSearchResult(section.toolResult)) {
      return fExtractSearchResults(section.toolResult).length > 0;
    }

    return (
      fIsWebSearchToolName(section.toolName) && fExtractSearchQuery(section.toolArgs).length > 0
    );
  });
</script>

{#if fIsLocalShellTool(section.toolName)}
  <ChatMessageToolCallBlockLocalShell {isStreaming} {onToggle} {open} {section} />
{:else if cIsSearchCall}
  <ChatMessageToolCallBlockSearchResults {isStreaming} {onToggle} {open} {section} />
{:else if section.toolName === BuiltInTool.cBrowserGetDatetime}
  <ChatMessageToolCallBlockGetDatetime {isStreaming} {section} />
{:else if section.toolName === BuiltInTool.cServerGetInfo}
  <ChatMessageToolCallBlockGetInfo {isStreaming} {onToggle} {open} {section} />
{:else if section.toolName === BuiltInTool.cServerReadFile}
  <ChatMessageToolCallBlockReadFile {isStreaming} {onToggle} {open} {section} />
{:else if section.toolName === BuiltInTool.cBrowserReadMedia}
  <ChatMessageToolCallBlockReadMedia {isStreaming} {onToggle} {open} {section} />
{:else if section.toolName === BuiltInTool.cServerEditFile}
  <ChatMessageToolCallBlockEditFile {isStreaming} {onToggle} {open} {section} />
{:else if section.toolName === BuiltInTool.cServerWriteFile}
  <ChatMessageToolCallBlockWriteFile {isStreaming} {onToggle} {open} {section} />
{:else if section.toolName === BuiltInTool.cServerExecShellCommand}
  <ChatMessageToolCallBlockExecShellCommand
    {attachments}
    {isExecuting}
    {isStreaming}
    {onToggle}
    {open}
    {section}
  />
{:else if section.toolName === BuiltInTool.cServerFileGlobSearch}
  <ChatMessageToolCallBlockFileGlobSearch {isStreaming} {onToggle} {open} {section} />
{:else if section.toolName === BuiltInTool.cServerGrepSearch}
  <ChatMessageToolCallBlockGrepSearch {isStreaming} {onToggle} {open} {section} />
{:else if section.toolName === BuiltInTool.cBrowserRunJavascript}
  <ChatMessageToolCallBlockRunJavascript {isStreaming} {onToggle} {open} {section} />
{:else}
  <ChatMessageToolCallBlockDefault {attachments} {isStreaming} {onToggle} {open} {section} />
{/if}
