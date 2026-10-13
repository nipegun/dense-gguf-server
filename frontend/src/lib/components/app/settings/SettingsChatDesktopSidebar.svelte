<script lang="ts">
  import { cIconClassDefault } from '$lib/constants';
  import type { SettingsSection, SettingsSectionTitle } from '$lib/types';

  interface Props {
    sections: SettingsSection[];
    isActive: (pSection: SettingsSection) => boolean;
    onSectionChange?: (pSection: SettingsSectionTitle) => void;
  }

  let { isActive, onSectionChange, sections }: Props = $props();
</script>

<div class="sticky top-12 hidden w-64 flex-col self-start bg-background md:flex gap-6">
  <nav class="space-y-1">
    {#each sections as vSection (vSection.title)}
      <button
        class="flex w-full cursor-pointer items-center gap-3 rounded-lg px-3 py-2 text-left text-sm transition-colors hover:bg-accent {isActive(
          vSection
        )
          ? 'bg-accent text-accent-foreground'
          : 'text-muted-foreground'}"
        onclick={() => onSectionChange?.(vSection.title)}
      >
        <vSection.icon class={cIconClassDefault} />

        <span class="ml-2">{vSection.title}</span>
      </button>
    {/each}
  </nav>
</div>
