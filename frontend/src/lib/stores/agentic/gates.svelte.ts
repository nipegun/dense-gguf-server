/**
 * AgenticGates - User interaction gates for the agentic loop
 *
 * Owns the state the loop waits on between turns: tool permission requests,
 * turn-limit continue prompts and queued steering messages. The loop awaits
 * requestPermission/requestContinue; the UI resolves them through
 * resolvePermission/resolveContinue. Owned by agenticStore, no host coupling.
 */

import { ToolPermissionDecision } from '$lib/enums';
// direct imports between stores, not via the barrel, to avoid circular deps
import { cPermissionsStore } from '$lib/stores/permissions.svelte';
import { cToolsStore } from '$lib/stores/tools.svelte';
import type { DatabaseMessageExtra, SteeringMessage } from '$lib/types';
import { SvelteMap } from 'svelte/reactivity';

export class AgenticGates {
  /** Resolve functions for pending continue Promises; nothing derives from this map */
  private vContinueResolvers = new SvelteMap<string, (pShouldContinue: boolean) => void>();
  /** Dedicated reactive state for pending continue requests (turn limit reached) */
  private vPendingContinueRequests = new SvelteMap<string, boolean>();

  /** Dedicated reactive state for pending permission requests (ensures immediate UI updates) */
  private vPendingPermissions = new SvelteMap<
    string,
    { toolName: string; serverLabel: string } | null
  >();
  /** Resolve functions for pending permission Promises; nothing derives from this map */
  private vPermissionResolvers = new SvelteMap<
    string,
    (pDecision: ToolPermissionDecision) => void
  >();

  /** Reactive: queued steering messages to inject between turns */
  private vSteeringMessages = new SvelteMap<string, SteeringMessage>();

  /**
   * Drop all pending gate state for a conversation, e.g. when a flow exits.
   */
  mClear(pConversationId: string): void {
    this.vPendingPermissions.set(pConversationId, null);
    this.vPermissionResolvers.delete(pConversationId);
    this.vPendingContinueRequests.set(pConversationId, false);
    this.vContinueResolvers.delete(pConversationId);
    this.vSteeringMessages.delete(pConversationId);
  }

  /**
   * Clear the pending steering message without consuming it.
   */
  mClearSteeringMessage(pConversationId: string): void {
    this.vSteeringMessages.delete(pConversationId);
  }

  /**
   * Consume and return the pending steering message for re-sending.
   * Called by chatStore after the agentic flow exits.
   */
  mConsumePendingSteeringMessage(pConversationId: string): SteeringMessage | null {
    const cMsg = this.vSteeringMessages.get(pConversationId);

    if (!cMsg) return null;

    this.vSteeringMessages.delete(pConversationId);

    return cMsg;
  }

  mGetPendingContinueRequest(pConversationId: string): boolean {
    return this.vPendingContinueRequests.get(pConversationId) ?? false;
  }

  mGetPendingPermissionRequest(
    pConversationId: string
  ): { toolName: string; serverLabel: string } | null {
    return this.vPendingPermissions.get(pConversationId) ?? null;
  }

  mGetPendingSteeringMessageContent(pConversationId: string): string | null {
    return this.vSteeringMessages.get(pConversationId)?.content ?? null;
  }

  mGetPendingSteeringMessageExtras(pConversationId: string): DatabaseMessageExtra[] | undefined {
    return this.vSteeringMessages.get(pConversationId)?.extras;
  }

  mHasPendingSteeringMessage(pConversationId: string): boolean {
    return this.vSteeringMessages.has(pConversationId);
  }

  /**
   * Queue a steering message. When the current agentic turn completes,
   * the flow exits and the caller re-sends the message as a normal chat message.
   */
  mInjectSteeringMessage(
    pConversationId: string,
    pContent: string,
    pExtras?: DatabaseMessageExtra[]
  ): void {
    this.vSteeringMessages.set(pConversationId, { content: pContent, extras: pExtras });
  }

  async mRequestContinue(pConversationId: string, pSignal?: AbortSignal): Promise<boolean> {
    this.vPendingContinueRequests.set(pConversationId, true);

    return new Promise<boolean>((pResolve) => {
      if (pSignal?.aborted) {
        this.vPendingContinueRequests.set(pConversationId, false);
        pResolve(false);

        return;
      }

      this.vContinueResolvers.set(pConversationId, (pShouldContinue) => {
        this.vPendingContinueRequests.set(pConversationId, false);
        pResolve(pShouldContinue);
      });

      pSignal?.addEventListener(
        'abort',
        () => {
          const cResolver = this.vContinueResolvers.get(pConversationId);

          if (cResolver) {
            this.vContinueResolvers.delete(pConversationId);
            this.vPendingContinueRequests.set(pConversationId, false);
            pResolve(false);
          }
        },
        { once: true }
      );
    });
  }

  async mRequestPermission(
    pConversationId: string,
    pToolName: string,
    pServerLabel: string,
    pSignal?: AbortSignal
  ): Promise<ToolPermissionDecision> {
    const cPermissionKey = cToolsStore.mGetPermissionKey(pToolName);

    if (cPermissionKey && cPermissionsStore.mHasTool(cPermissionKey)) {
      return ToolPermissionDecision.cOnce;
    }

    this.vPendingPermissions.set(pConversationId, {
      serverLabel: pServerLabel,
      toolName: pToolName
    });

    return new Promise<ToolPermissionDecision>((pResolve) => {
      if (pSignal?.aborted) {
        this.vPendingPermissions.set(pConversationId, null);
        pResolve(ToolPermissionDecision.cDeny);

        return;
      }

      this.vPermissionResolvers.set(pConversationId, (pDecision) => {
        this.vPendingPermissions.set(pConversationId, null);

        if (pDecision === ToolPermissionDecision.cAlways && cPermissionKey) {
          cPermissionsStore.mAllowTool(cPermissionKey);
        } else if (pDecision === ToolPermissionDecision.cAlwaysServer) {
          const aServerToolKeys = cToolsStore.allTools
            .filter((pT) =>
              pT.serverName
                ? pT.serverName === pServerLabel
                : cToolsStore.mGetToolServerLabel(pT.definition.function.name) === pServerLabel
            )
            .map((pT) => cToolsStore.mGetPermissionKey(pT.definition.function.name)!)
            .filter((pK): pK is string => pK !== null);

          cPermissionsStore.mAllowTools(aServerToolKeys);
        }

        pResolve(pDecision);
      });

      pSignal?.addEventListener(
        'abort',
        () => {
          const cResolver = this.vPermissionResolvers.get(pConversationId);

          if (cResolver) {
            this.vPermissionResolvers.delete(pConversationId);
            this.vPendingPermissions.set(pConversationId, null);
            pResolve(ToolPermissionDecision.cDeny);
          }
        },
        { once: true }
      );
    });
  }

  mResolveContinue(pConversationId: string, pShouldContinue: boolean): void {
    const cResolver = this.vContinueResolvers.get(pConversationId);

    if (cResolver) {
      this.vContinueResolvers.delete(pConversationId);
      cResolver(pShouldContinue);
    }
  }

  mResolvePermission(pConversationId: string, pDecision: ToolPermissionDecision): void {
    const cResolver = this.vPermissionResolvers.get(pConversationId);

    if (cResolver) {
      this.vPermissionResolvers.delete(pConversationId);
      cResolver(pDecision);
    }
  }
}
