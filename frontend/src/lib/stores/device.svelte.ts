/**
 * deviceStore - Browser environment signals
 *
 * Device capabilities, OS theme and viewport in one class store:
 * deviceStore.isMobile, deviceStore.isIOSDevice / isIOSSafari / isWKWebView /
 * isStandalone, deviceStore.systemTheme.isDark.
 *
 * UA-derived flags are static for the session; isStandalone and systemTheme
 * track live media query changes.
 */

import { browser } from '$app/environment';
import { cDefaultMobileBreakpoint, cMediaQueries } from '$lib/constants';
import { MediaQuery } from 'svelte/reactivity';

/**
 * iOS UA token detection.
 *
 * iPadOS 13+ ships a desktop macOS UA, so 'iPad' is no longer present in it;
 * a Macintosh UA combined with touch support is treated as an iPad instead.
 * Third-party iOS browsers (Chrome, Firefox, Edge) and in-app WKWebViews all
 * run on WKWebView and emit their own tokens (CriOS/FxiOS/EdgiOS/GSA) instead
 * of the trailing 'Safari/' the Safari app keeps.
 */
const cUaPatterns = {
  IOS_PHONE: /iPhone|iPod/,
  MACINTOSH: /Macintosh/,
  SAFARI: /Safari/,
  WEBVIEW_IOS: /CriOS|FxiOS|EdgiOS|GSA/
} as const;

class DeviceStore {
  /** Any iOS/iPadOS device, regardless of which app or browser embeds the page. */
  readonly vIsIOSDevice: boolean = false;
  /** The Safari browser app on iOS, excluding other iOS browsers and WKWebViews. */
  readonly vIsIOSSafari: boolean = false;
  /** PWA standalone mode: the page was launched from the home screen icon. */
  vIsStandalone = $state(false);
  /** Any WKWebView context on iOS: in-app browsers, embedded web views, and the
   *  third-party iOS browsers (all of which share the WKWebView engine). */
  readonly vIsWKWebView: boolean = false;
  /** OS color scheme preference; the user override lives in settingsStore. */
  readonly vSystemTheme = $state({ isDark: false });

  private vMobile = new MediaQuery(`max-width: ${cDefaultMobileBreakpoint - 1}px`);

  get isMobile(): boolean {
    return this.vMobile.current;
  }

  constructor() {
    if (!browser) return;

    const cUa = navigator.userAgent;
    const cIsTouch = navigator.maxTouchPoints > 0;

    this.vIsIOSDevice =
      cUaPatterns.IOS_PHONE.test(cUa) || (cUaPatterns.MACINTOSH.test(cUa) && cIsTouch);
    // Safari keeps 'Safari/' in the UA; non-Safari iOS browsers emit their own
    // token instead. WKWebView typically omits 'Safari/' entirely.
    const cHasSafariToken = cUaPatterns.SAFARI.test(cUa) && !cUaPatterns.WEBVIEW_IOS.test(cUa);

    this.vIsIOSSafari = this.vIsIOSDevice && cHasSafariToken;
    this.vIsWKWebView = this.vIsIOSDevice && !cHasSafariToken;
    this.vIsStandalone = window.matchMedia(cMediaQueries.DISPLAY_MODE_STANDALONE).matches;
    this.vSystemTheme.isDark = window.matchMedia(cMediaQueries.PREFERS_DARK).matches;

    // isStandalone and systemTheme can change at runtime (e.g. user installs the
    // PWA while the tab is open); the UA-derived flags are static for the session
    const cStandaloneMql = window.matchMedia(cMediaQueries.DISPLAY_MODE_STANDALONE);

    cStandaloneMql.addEventListener('change', (pE) => {
      this.vIsStandalone = pE.matches;
    });

    const cDarkMql = window.matchMedia(cMediaQueries.PREFERS_DARK);

    cDarkMql.addEventListener('change', (pE) => {
      this.vSystemTheme.isDark = pE.matches;
    });
  }
}

export const cDeviceStore = new DeviceStore();
