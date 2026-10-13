<script lang="ts">
  import { fColorLevelTextClass } from './context-gauge';
  import type { ColorLevel } from '$lib/enums';

  interface Props {
    percent: number | null;
    level: ColorLevel;
    size?: 'sm' | 'md';
  }

  let { level, percent, size = 'sm' }: Props = $props();

  // the md dial fills its box: the ring would sit inside a padding ring otherwise
  const RADIUS = $derived(size === 'md' ? 13 : 11);
  const cCircumference = $derived(2 * Math.PI * RADIUS);

  const cStrokeLevelClass = $derived(fColorLevelTextClass(level));
  const cDimensions = $derived(size === 'md' ? 'h-6 w-6' : 'h-5 w-5');
  const cStrokeWidth = $derived(size === 'md' ? 4 : 3);
</script>

<svg class={cDimensions} fill="none" viewBox="0 0 32 32">
  <circle
    cx="16"
    cy="16"
    r={RADIUS}
    stroke="currentColor"
    stroke-opacity="0.1"
    stroke-width={cStrokeWidth}
  />

  <circle
    class="transition-colors duration-300 {cStrokeLevelClass}"
    cx="16"
    cy="16"
    r={RADIUS}
    stroke="currentColor"
    stroke-dasharray={cCircumference}
    stroke-dashoffset={percent !== null ? cCircumference * (1 - percent / 100) : cCircumference}
    stroke-linecap="round"
    stroke-width={cStrokeWidth}
    transform="rotate(-90 16 16)"
  />
</svg>
