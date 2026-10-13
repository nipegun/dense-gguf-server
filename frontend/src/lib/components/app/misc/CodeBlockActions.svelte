<script lang="ts">
  import { Eye } from '@lucide/svelte';
  import { ActionIcon, ActionIconCopyToClipboard } from '$lib/components/app';
  import { FileTypeText } from '$lib/enums';

  interface Props {
    code: string;
    language: string;
    disabled?: boolean;
    onPreview?: (pCode: string, pLanguage: string) => void;
  }

  let { code, disabled = false, language, onPreview }: Props = $props();

  const cShowPreview = $derived(language?.toLowerCase() === FileTypeText.HTML);
</script>

<div class="code-block-actions">
  <ActionIconCopyToClipboard
    ariaLabel={disabled ? 'Code incomplete' : 'Copy code'}
    canCopy={!disabled}
    text={code}
  />

  {#if cShowPreview}
    <ActionIcon
      {disabled}
      icon={Eye}
      onclick={() => onPreview!(code, language)}
      tooltip={disabled ? 'Code incomplete' : 'Preview code'}
    />
  {/if}
</div>
