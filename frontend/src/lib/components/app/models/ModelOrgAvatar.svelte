<script lang="ts">
  import { aDarkInvertAvatarOrgs } from '$lib/constants';
  import { HuggingFaceService } from '$lib/services';
  import { SvelteMap } from 'svelte/reactivity';

  // Avatars a failed request is cooling down for. A failure hides the image
  // immediately, but the org gets another chance on a later mount: the cooldown
  // expires on its own, so a transient network blip is not permanent for the
  // session. A row remounting inside the cooldown renders the monogram instead
  // of re-requesting the same failing URL.
  const cAvatarRetryDelayMs = 60_000;
  const cFailedAvatarOrgs = new SvelteMap<string, number>();

  /** True while an org's failed avatar is still inside its retry cooldown. */
  function fIsAvatarCoolingDown(pOrg: string): boolean {
    const cFailedAt = cFailedAvatarOrgs.get(pOrg);

    // a stale entry reads as not failed and is dropped on the next onerror
    return cFailedAt !== undefined && Date.now() - cFailedAt < cAvatarRetryDelayMs;
  }

  interface Props {
    class?: string;
    org: string;
    quantOrg?: string;
    size?: string;
    baseImageClass?: string;
    quantImageClass?: string;
    quantPositionClass?: string;
    quantSize?: string;
  }

  let {
    baseImageClass = '',
    class: className = '',
    org,
    quantImageClass = 'h-full w-full',
    quantOrg,
    quantPositionClass = '-bottom-0.75 -right-0.75',
    quantSize = 'h-4.25 w-4.25',
    size = 'h-9 w-9'
  }: Props = $props();

  // With the Hub metadata setting off there is nothing to fetch and nothing to
  // show: the avatar is hidden entirely instead of falling back to a monogram.
  let vHubEnabled = $derived(HuggingFaceService.mIsEnabled());
  let vOrgAvatarFailed = $derived(fIsAvatarCoolingDown(org));
  let vQuantAvatarFailed = $derived(fIsAvatarCoolingDown(quantOrg ?? ''));

  let vInvertAvatar = $derived(aDarkInvertAvatarOrgs.includes(org));
  let vInvertQuant = $derived(aDarkInvertAvatarOrgs.includes(quantOrg ?? ''));

  // Monogram fallback: org initial on a hue derived from its name, so each org
  // gets a stable distinct color.
  let vHue = $derived.by(() => {
    let vH = 0;

    for (let vI = 0; vI < org.length; vI++) vH = (vH * 31 + org.charCodeAt(vI)) >>> 0;

    return vH % 360;
  });

  let vQuantHue = $derived.by(() => {
    const name = quantOrg ?? '';

    let vH = 0;

    for (let vI = 0; vI < name.length; vI++) vH = (vH * 31 + name.charCodeAt(vI)) >>> 0;

    return vH % 360;
  });
</script>

{#if vHubEnabled}
  <span class="relative inline-flex shrink-0 {className}">
    {#if vOrgAvatarFailed}
      <span
        aria-hidden="true"
        class="flex {size} items-center justify-center rounded-md text-sm font-semibold text-white"
        style="background-color: hsl({vHue} 60% 45%)"
      >
        {org.charAt(0).toUpperCase()}
      </span>
    {:else}
      <div class="rounded-md">
        <!-- the server serves the app under COEP require-corp, so a cross-origin
				     image has to be fetched in CORS mode to be allowed -->
        <img
          alt=""
          class="{size} rounded-md {vInvertAvatar ? 'dark:invert' : ''} {baseImageClass}"
          crossorigin="anonymous"
          loading="lazy"
          onerror={() => {
            cFailedAvatarOrgs.set(org, Date.now());
          }}
          src={HuggingFaceService.mGetAvatarUrl(org)}
        />
      </div>
    {/if}

    {#if quantOrg && quantOrg !== org}
      <!-- native title instead of a floating tooltip: long model lists mount one badge per row -->
      <span
        class="absolute {quantPositionClass} {quantSize} overflow-hidden rounded-full border border-background bg-muted"
        title={quantOrg}
      >
        {#if vQuantAvatarFailed}
          <span
            aria-hidden="true"
            class="flex h-full w-full items-center justify-center rounded-full text-[8px] font-semibold text-white"
            style="background-color: hsl({vQuantHue} 60% 45%)"
          >
            {quantOrg.charAt(0).toUpperCase()}
          </span>
        {:else}
          <img
            alt=""
            class="{quantImageClass} rounded-full {vInvertQuant ? 'dark:invert' : ''}"
            crossorigin="anonymous"
            loading="lazy"
            onerror={() => {
              cFailedAvatarOrgs.set(quantOrg ?? '', Date.now());
            }}
            src={HuggingFaceService.mGetAvatarUrl(quantOrg)}
          />
        {/if}
      </span>
    {/if}
  </span>
{/if}
