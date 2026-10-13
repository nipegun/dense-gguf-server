<script lang="ts">
  import ActionIcon from './ActionIcon.svelte';
  import { Copy } from '@lucide/svelte';
  import { cIconClassDefault } from '$lib/constants';
  import { fCopyToClipboard } from '$lib/utils';

  export let ariaLabel: string = 'Copy to clipboard';
  export let canCopy: boolean = true;
  export let text: string;
</script>

<ActionIcon
  disabled={!canCopy}
  icon={Copy}
  iconSize={cIconClassDefault}
  onclick={() => canCopy && fCopyToClipboard(text)}
  tooltip={ariaLabel}
/>
