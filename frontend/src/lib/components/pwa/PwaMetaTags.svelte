<script lang="ts">
  import { cAppName, cAppleMetaTags, cMediaQueries, cThemeColors } from '$lib/constants';

  let { appName = cAppName } = $props();
</script>

<svelte:head>
  <!-- Theme color for light/dark modes -->
  <meta content={cThemeColors.LIGHT} media={cMediaQueries.PREFERS_LIGHT} name="theme-color" />

  <meta content={cThemeColors.DARK} media={cMediaQueries.PREFERS_DARK} name="theme-color" />

  <!-- Apple mobile web app meta tags -->
  <meta
    content={cAppleMetaTags.MOBILE_WEB_APP_CAPABLE.content}
    name={cAppleMetaTags.MOBILE_WEB_APP_CAPABLE.name}
  />

  <meta
    content={cAppleMetaTags.STATUS_BAR_STYLE.content}
    name={cAppleMetaTags.STATUS_BAR_STYLE.name}
  />

  <meta content={appName} name={cAppleMetaTags.MOBILE_WEB_APP_TITLE.name} />
</svelte:head>
