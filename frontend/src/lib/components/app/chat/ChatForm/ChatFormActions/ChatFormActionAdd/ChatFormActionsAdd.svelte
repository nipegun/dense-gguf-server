<script lang="ts">
  import ChatFormActionAddButton from './ChatFormActionAddButton.svelte';
  import ChatFormActionAddDrawer from './ChatFormActionAddDrawer.svelte';
  import ChatFormActionAddDropdown from './ChatFormActionAddDropdown.svelte';
  import { cDeviceStore } from '$lib/stores';
</script>

{#if cDeviceStore.isMobile}
  <ChatFormActionAddDrawer>
    {#snippet trigger({ disabled, onclick: vOnclick })}
      <ChatFormActionAddButton {disabled} onclick={vOnclick} />
    {/snippet}
  </ChatFormActionAddDrawer>
{:else}
  <ChatFormActionAddDropdown />
{/if}
