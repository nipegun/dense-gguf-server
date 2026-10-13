/**
 * serverStore - Server connection state, configuration and role detection
 *
 * Owns the connection state and properties fetched from /props, plus MODEL
 * vs ROUTER role detection and server-wide generation defaults. Uses
 * PropsService for the /props fetch.
 */

import { ServerRole } from '$lib/enums';
import { PropsService } from '$lib/services/props.service';
import { ApiError } from '$lib/utils';

const cLoadingRetryIntervalMs = 1000;

class ServerStore {
  error = $state<string | null>(null);
  loading = $state(false);
  props = $state<ApiDgsServerProps | null>(null);
  vRole = $state<ServerRole | null>(null);
  status = $state<number | null>(null);
  private vFetchPromise: Promise<void> | null = null;
  private vRetryTimer: ReturnType<typeof setTimeout> | null = null;

  get contextSize(): number | null {
    const cNCtx = this.props?.default_generation_settings?.n_ctx;

    return typeof cNCtx === 'number' ? cNCtx : null;
  }

  get defaultParams(): ApiDgsServerProps['default_generation_settings']['params'] | null {
    return this.props?.default_generation_settings?.params || null;
  }

  get isModelMode(): boolean {
    return this.vRole === ServerRole.MODEL;
  }

  get isRouterMode(): boolean {
    return this.vRole === ServerRole.ROUTER;
  }

  get uiSettings(): Record<string, string | number | boolean> | undefined {
    return this.props?.ui_settings;
  }

  mClear(): void {
    this.mClearRetryTimer();
    this.props = null;
    this.error = null;
    this.status = null;
    this.loading = false;
    this.vRole = null;
    this.vFetchPromise = null;
  }

  /**
   * @param background - Set by the automatic "still loading" poll. Skips the
   * `loading` flag flip so the UI doesn't bounce between the full loading
   * splash and the chat screen every retry tick.
   */
  async mFetch({ background: vBackground = false }: { background?: boolean } = {}): Promise<void> {
    if (this.vFetchPromise) return this.vFetchPromise;

    this.mClearRetryTimer();

    if (!vBackground) {
      this.loading = true;
    }

    // Don't clear an existing "still loading" error before a retry -
    // doing so would unmount/remount the error banner every second.
    if (this.status !== 503) {
      this.error = null;
    }

    const cFetchPromise = (async () => {
      try {
        const cProps = await PropsService.mFetch();

        this.props = cProps;
        this.error = null;
        this.status = null;
        this.mDetectRole(cProps);
      } catch (error: unknown) {
        this.error = error instanceof Error ? error.message : String(error);
        this.status = error instanceof ApiError ? error.status : null;
        console.error('Error fetching server properties:', error);

        if (this.status === 503) {
          this.mScheduleRetry();
        }
      } finally {
        if (!vBackground) {
          this.loading = false;
        }

        this.vFetchPromise = null;
      }
    })();

    this.vFetchPromise = cFetchPromise;
    await cFetchPromise;
  }

  private mClearRetryTimer(): void {
    if (this.vRetryTimer) {
      clearTimeout(this.vRetryTimer);
      this.vRetryTimer = null;
    }
  }

  private mDetectRole(pProps: ApiDgsServerProps): void {
    const cNewRole = pProps?.role === ServerRole.ROUTER ? ServerRole.ROUTER : ServerRole.MODEL;

    if (this.vRole !== cNewRole) {
      this.vRole = cNewRole;
      console.info(`Server running in ${cNewRole === ServerRole.ROUTER ? 'ROUTER' : 'MODEL'} mode`);
    }
  }

  private mScheduleRetry(): void {
    if (this.vRetryTimer) return;

    this.vRetryTimer = setTimeout(() => {
      this.vRetryTimer = null;
      this.mFetch({ background: true });
    }, cLoadingRetryIntervalMs);
  }
}

export const cServerStore = new ServerStore();
