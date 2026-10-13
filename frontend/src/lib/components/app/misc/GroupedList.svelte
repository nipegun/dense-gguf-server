<script lang="ts" module>
  /** One group of a grouped list, rendered under its own heading. */
  export interface GroupedListGroup<G, E> {
    entries: E[];
    group: G;
    key: string;
  }

  /** What a show-more row counts: rows of one group, or whole groups of the list. */
  export const cGroupedListUnit = {
    ENTRIES: 'entries',
    GROUPS: 'families'
  } as const;

  export type cGroupedListUnit = (typeof cGroupedListUnit)[keyof typeof cGroupedListUnit];
</script>

<script generics="G, E" lang="ts">
  import CollapsibleRegion from './CollapsibleRegion.svelte';
  import type { Snippet } from 'svelte';
  import { SvelteMap, SvelteSet } from 'svelte/reactivity';

  interface Props {
    /** Heading of one group. Omit it for a list whose rows need no heading. */
    group?: Snippet<
      [{ depth: number; expanded: boolean; group: G; key: string; toggle: () => void }]
    >;
    /** Groups to render. Omit to render `items` as a flat list. */
    groups?: GroupedListGroup<G, E>[] | null;
    /** Entries of one group shown before its show more row. 0 shows every entry. */
    groupWindow?: number;
    /** Group ids folded away, read once so a caller can restore what it persisted. */
    initialCollapsed?: string[];
    /** One row. Depth is 1 under a group, 0 in a flat list. */
    item: Snippet<[{ depth: number; entry: E }]>;
    /** Identity of a row, used for the keyed each. */
    keyOf: (pEntry: E) => string;
    /** The show more row. Without it the rest of a window stays hidden. */
    more?: Snippet<[{ count: number; onMore: () => void; unit: cGroupedListUnit }]>;
    /** Flat entries, rendered when the list has no groups. */
    items?: E[];
    /** Entries of the list shown before its show more row. 0 shows every entry. */
    sectionWindow?: number;
    /** Called on every group toggle, for a caller that persists the state. */
    onCollapsedChange?: ((pKey: string, pCollapsed: boolean) => void) | null;
    /** Sticky class of a group heading, so a surface can shade it differently. */
    stickyClass?: string;
    /** Sticky offset of a group heading, e.g. `top: 2.25rem`. Empty keeps it scrolling. */
    stickyStyle?: string;
    /** Weight of one entry against a window, e.g. the rows it renders. */
    weightOf?: (pEntry: E) => number;
  }

  let {
    group,
    groups = null,
    groupWindow = 0,
    initialCollapsed = [],
    item,
    items = [],
    keyOf,
    more,
    onCollapsedChange = null,
    sectionWindow = 0,
    stickyClass = 'bg-popover',
    stickyStyle = '',
    weightOf
  }: Props = $props();

  // groups start open: this tracks the ones the user folded away, seeded once from
  // whatever the caller persisted
  // svelte-ignore state_referenced_locally
  const cCollapsed = new SvelteSet<string>(initialCollapsed);
  // windows grow one step at a time, per group and per list
  const cGroupSteps = new SvelteMap<string, number>();
  let vSectionSteps = $state(0);

  function fToggleGroup(pKey: string): void {
    if (cCollapsed.has(pKey)) {
      cCollapsed.delete(pKey);
    } else {
      cCollapsed.add(pKey);
    }

    onCollapsedChange?.(pKey, cCollapsed.has(pKey));
  }

  function fGrowGroup(pKey: string): void {
    cGroupSteps.set(pKey, (cGroupSteps.get(pKey) ?? 0) + 1);
  }

  let fGroupCap = $derived((pKey: string) =>
    groupWindow > 0 ? groupWindow * (1 + (cGroupSteps.get(pKey) ?? 0)) : Infinity
  );
  let fWeight = $derived(weightOf ?? (() => 1));

  /** Groups with their entries cut to the window, and what each cut leaves behind. */
  let aWindowed = $derived.by((): Array<{ group: G; hidden: number; key: string; rows: E[] }> => {
    const aShown: Array<{ group: G; hidden: number; key: string; rows: E[] }> = [];
    const cCap = sectionWindow > 0 ? sectionWindow + vSectionSteps : Infinity;

    let vUsed = 0;

    for (const cEntry of groups ?? []) {
      if (vUsed >= cCap) break;

      const aRows = cEntry.entries.slice(0, fGroupCap(cEntry.key));

      aShown.push({
        group: cEntry.group,
        hidden: cEntry.entries.length - aRows.length,
        key: cEntry.key,
        rows: aRows
      });
      vUsed += aRows.reduce((pSum, pRow) => pSum + fWeight(pRow), 0);
    }

    return aShown;
  });

  let aFlatRows = $derived(
    sectionWindow > 0 ? items.slice(0, sectionWindow + vSectionSteps) : items
  );

  /** Rows one group weighs against the window. */
  const fGroupWeight = (pEntry: GroupedListGroup<G, E>) =>
    pEntry.entries.reduce((pSum, pRow) => pSum + fWeight(pRow), 0);

  /** The groups one page reveals, and the rows they weigh against the window. */
  let vGroupPage = $derived.by(() => {
    const aPage = (groups ?? []).slice(aWindowed.length, aWindowed.length + sectionWindow);

    return {
      count: aPage.length,
      weight: aPage.reduce((pSum, pEntry) => pSum + fGroupWeight(pEntry), 0)
    };
  });
</script>

{#if groups}
  {#each aWindowed as vEntry (vEntry.key)}
    {@const cExpanded = !cCollapsed.has(vEntry.key)}

    {#if group}
      <div class="sticky z-10 {stickyClass}" style={stickyStyle}>
        {@render group({
          depth: 0,
          expanded: cExpanded,
          group: vEntry.group,
          key: vEntry.key,
          toggle: () => fToggleGroup(vEntry.key)
        })}
      </div>
    {/if}

    <CollapsibleRegion open={cExpanded}>
      {#each vEntry.rows as vRow (keyOf(vRow))}
        {@render item({ depth: 1, entry: vRow })}
      {/each}

      {#if vEntry.hidden > 0 && more}
        {@render more({
          count: Math.min(groupWindow, vEntry.hidden),
          onMore: () => fGrowGroup(vEntry.key),
          unit: cGroupedListUnit.ENTRIES
        })}
      {/if}
    </CollapsibleRegion>
  {/each}

  {#if aWindowed.length < (groups?.length ?? 0) && more}
    {@render more({
      count: vGroupPage.count,
      onMore: () => (vSectionSteps += Math.max(1, vGroupPage.weight)),
      unit: cGroupedListUnit.GROUPS
    })}
  {/if}
{:else}
  {#each aFlatRows as vRow (keyOf(vRow))}
    {@render item({ depth: 0, entry: vRow })}
  {/each}

  {#if aFlatRows.length < items.length && more}
    {@render more({
      count: Math.min(sectionWindow, items.length - aFlatRows.length),
      onMore: () => (vSectionSteps += Math.min(sectionWindow, items.length - aFlatRows.length)),
      unit: cGroupedListUnit.ENTRIES
    })}
  {/if}
{/if}
