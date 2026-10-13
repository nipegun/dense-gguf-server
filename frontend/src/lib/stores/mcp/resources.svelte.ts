/**
 * mcpResourceStore - Reactive State Store for MCP Resources
 *
 * Manages MCP protocol resources:
 * - Resource discovery and listing per server
 * - Resource content caching
 * - Resource subscriptions
 * - Resource attachments for chat context
 *
 * @see MCP Protocol Specification: https://modelcontextprotocol.io/specification/2025-06-18/server/resources
 */

import {
  cBinaryContentLabel,
  cMcpResourceAttachmentIdPrefix,
  cMcpResourceCache,
  cNewline,
  cResourceUnknownType
} from '$lib/constants';
import { AttachmentType } from '$lib/enums';
import type {
  DatabaseMessageExtraMcpResource,
  MCPCachedResource,
  MCPResource,
  MCPResourceAttachment,
  MCPResourceContent,
  MCPResourceInfo,
  MCPResourceSubscription,
  MCPResourceTemplate,
  MCPResourceTemplateInfo,
  MCPServerResources
} from '$lib/types';
import { fNormalizeResourceUri } from '$lib/utils';
import { SvelteMap } from 'svelte/reactivity';

function fGenerateAttachmentId(): string {
  return `${cMcpResourceAttachmentIdPrefix}-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
}

class MCPResourceStore {
  private _attachments = $state<MCPResourceAttachment[]>([]);
  private vCachedResources = $state<SvelteMap<string, MCPCachedResource>>(new SvelteMap());
  private _isLoading = $state(false);
  private vServerResources = $state<SvelteMap<string, MCPServerResources>>(new SvelteMap());
  private vSubscriptions = $state<SvelteMap<string, MCPResourceSubscription>>(new SvelteMap());

  get attachmentCount(): number {
    return this._attachments.length;
  }

  get attachments(): MCPResourceAttachment[] {
    return this._attachments;
  }

  get cachedResources(): Map<string, MCPCachedResource> {
    return this.vCachedResources;
  }

  get hasAttachments(): boolean {
    return this._attachments.length > 0;
  }

  get isLoading(): boolean {
    return this._isLoading;
  }

  get serverResources(): Map<string, MCPServerResources> {
    return this.vServerResources;
  }

  get subscriptions(): Map<string, MCPResourceSubscription> {
    return this.vSubscriptions;
  }

  get totalResourceCount(): number {
    let vCount = 0;

    for (const cServerRes of this.vServerResources.values()) {
      vCount += cServerRes.resources.length;
    }

    return vCount;
  }

  get totalTemplateCount(): number {
    let vCount = 0;

    for (const cServerRes of this.vServerResources.values()) {
      vCount += cServerRes.templates.length;
    }

    return vCount;
  }

  /**
   * Add a resource attachment to the current chat context
   */
  mAddAttachment(pResource: MCPResourceInfo): MCPResourceAttachment {
    const attachment: MCPResourceAttachment = {
      id: fGenerateAttachmentId(),
      loading: true,
      resource: pResource
    };

    this._attachments = [...this._attachments, attachment];
    console.log(`[MCPResources] Added attachment: ${pResource.uri}`);

    return attachment;
  }

  /**
   * Register a subscription for a resource
   */
  mAddSubscription(pUri: string, pServerName: string): void {
    this.vSubscriptions.set(pUri, {
      serverName: pServerName,
      subscribedAt: new Date(),
      uri: pUri
    });

    const cCached = this.vCachedResources.get(pUri);

    if (cCached) {
      this.vCachedResources.set(pUri, { ...cCached, subscribed: true });
    }

    console.log(`[MCPResources] Added subscription: ${pUri}`);
  }

  /**
   * Cache resource content after reading
   */
  mCacheResourceContent(pResource: MCPResourceInfo, pContent: MCPResourceContent[]): void {
    // Enforce cache size limit
    if (this.vCachedResources.size >= cMcpResourceCache.MAX_ENTRIES) {
      const cOldestKey = this.vCachedResources.keys().next().value;

      if (cOldestKey) {
        this.vCachedResources.delete(cOldestKey);
      }
    }

    this.vCachedResources.set(pResource.uri, {
      content: pContent,
      fetchedAt: new Date(),
      resource: pResource,
      subscribed: this.vSubscriptions.has(pResource.uri)
    });
    console.log(`[MCPResources] Cached content for: ${pResource.uri}`);
  }

  /**
   * Clear all state (e.g., on full reset)
   */
  mClear(): void {
    this.vServerResources.clear();
    this.vCachedResources.clear();
    this.vSubscriptions.clear();
    this._attachments = [];
    this._isLoading = false;
    console.log(`[MCPResources] Cleared all state`);
  }

  /**
   * Clear all attachments
   */
  mClearAttachments(): void {
    this._attachments = [];
    console.log(`[MCPResources] Cleared all attachments`);
  }

  /**
   * Clear all cached content
   */
  mClearCache(): void {
    this.vCachedResources.clear();
    console.log(`[MCPResources] Cleared all cached content`);
  }

  /**
   * Clear resources for a server (e.g., when disconnected)
   */
  mClearServerResources(pServerName: string): void {
    this.vServerResources.delete(pServerName);

    for (const [vUri, vCached] of this.vCachedResources) {
      if (vCached.resource.serverName === pServerName) {
        this.vCachedResources.delete(vUri);
      }
    }

    for (const [vUri, vSub] of this.vSubscriptions) {
      if (vSub.serverName === pServerName) {
        this.vSubscriptions.delete(vUri);
      }
    }

    console.log(`[MCPResources][${pServerName}] Cleared all resources`);
  }

  /**
   * Find resource info by URI across all servers
   */
  mFindResourceByUri(pUri: string): MCPResourceInfo | undefined {
    const cNormalizedUri = fNormalizeResourceUri(pUri);

    for (const [vServerName, vServerRes] of this.vServerResources) {
      const resource =
        vServerRes.resources.find((pR) => pR.uri === pUri) ??
        vServerRes.resources.find((pR) => fNormalizeResourceUri(pR.uri) === cNormalizedUri);

      if (resource) {
        return {
          annotations: resource.annotations,
          description: resource.description,
          icons: resource.icons,
          mimeType: resource.mimeType,
          name: resource.name,
          serverName: vServerName,
          title: resource.title,
          uri: resource.uri
        };
      }
    }

    return undefined;
  }

  /**
   * Find server name for a resource URI
   */
  mFindServerForUri(pUri: string): string | undefined {
    for (const [vServerName, vServerRes] of this.vServerResources) {
      if (vServerRes.resources.some((pR) => pR.uri === pUri)) {
        return vServerName;
      }
    }

    return undefined;
  }

  /**
   * Get resource content as text for chat context
   * Formats content for inclusion in LLM prompts
   */
  mFormatAttachmentsForContext(): string {
    if (this._attachments.length === 0) return '';

    const aParts: string[] = [];

    for (const attachment of this._attachments) {
      if (attachment.error) continue;

      if (!attachment.content || attachment.content.length === 0) continue;

      const cResourceName = attachment.resource.title || attachment.resource.name;
      const cServerName = attachment.resource.serverName;

      for (const cContent of attachment.content) {
        if ('text' in cContent && cContent.text) {
          aParts.push(
            `\n\n--- Resource: ${cResourceName} (from ${cServerName}) ---\n${cContent.text}`
          );
        } else if ('blob' in cContent && cContent.blob) {
          // For binary content, just note it exists
          aParts.push(
            `\n\n--- Resource: ${cResourceName} (from ${cServerName}) ---\n[${cBinaryContentLabel}: ${cContent.mimeType || cResourceUnknownType}]`
          );
        }
      }
    }

    return aParts.join('');
  }

  /**
   * Get all resources as MCPResourceInfo array (flattened with server names)
   */
  mGetAllResourceInfos(): MCPResourceInfo[] {
    const aResult: MCPResourceInfo[] = [];

    for (const [vServerName, vServerRes] of this.vServerResources) {
      for (const resource of vServerRes.resources) {
        aResult.push({
          annotations: resource.annotations,
          description: resource.description,
          icons: resource.icons,
          mimeType: resource.mimeType,
          name: resource.name,
          serverName: vServerName,
          title: resource.title,
          uri: resource.uri
        });
      }
    }

    return aResult;
  }

  /**
   * Get all templates as MCPResourceTemplateInfo array (flattened with server names)
   */
  mGetAllTemplateInfos(): MCPResourceTemplateInfo[] {
    const aResult: MCPResourceTemplateInfo[] = [];

    for (const [vServerName, vServerRes] of this.vServerResources) {
      for (const cTemplate of vServerRes.templates) {
        aResult.push({
          annotations: cTemplate.annotations,
          description: cTemplate.description,
          icons: cTemplate.icons,
          mimeType: cTemplate.mimeType,
          name: cTemplate.name,
          serverName: vServerName,
          title: cTemplate.title,
          uriTemplate: cTemplate.uriTemplate
        });
      }
    }

    return aResult;
  }

  /**
   * Get attachment by ID
   */
  mGetAttachment(pAttachmentId: string): MCPResourceAttachment | undefined {
    return this._attachments.find((pAtt) => pAtt.id === pAttachmentId);
  }

  /**
   * Get cached content for a resource
   */
  mGetCachedContent(pUri: string): MCPCachedResource | undefined {
    const cCached = this.vCachedResources.get(pUri);

    if (!cCached) return undefined;

    const cAge = Date.now() - cCached.fetchedAt.getTime();

    if (cAge > cMcpResourceCache.TTL_MS && !cCached.subscribed) {
      // Cache expired and not subscribed, remove it
      this.vCachedResources.delete(pUri);

      return undefined;
    }

    return cCached;
  }

  /**
   * Get resources for a specific server
   */
  mGetServerResources(pServerName: string): MCPServerResources | undefined {
    return this.vServerResources.get(pServerName);
  }

  /**
   * Handle resources list changed notification
   */
  mHandleResourcesListChanged(pServerName: string): void {
    const cExisting = this.vServerResources.get(pServerName);

    if (cExisting) {
      this.vServerResources.set(pServerName, {
        ...cExisting,
        lastFetched: undefined
      });
    }

    console.log(`[MCPResources][${pServerName}] Resources list changed, needs refresh`);
  }

  /**
   *
   *
   * Attachments (for chat context)
   *
   *
   */

  /**
   * Handle resource update notification
   */
  mHandleResourceUpdate(pUri: string): void {
    // Invalidate cache so next read gets fresh content
    this.mInvalidateCache(pUri);

    const cSub = this.vSubscriptions.get(pUri);

    if (cSub) {
      this.vSubscriptions.set(pUri, { ...cSub, lastUpdate: new Date() });
    }

    console.log(`[MCPResources] Resource updated: ${pUri}`);
  }

  /**
   * Invalidate cached content for a resource (e.g., on update notification)
   */
  mInvalidateCache(pUri: string): void {
    this.vCachedResources.delete(pUri);
    console.log(`[MCPResources] Invalidated cache for: ${pUri}`);
  }

  /**
   * Check if a resource is already attached
   */
  mIsAttached(pUri: string): boolean {
    const cNormalizedUri = fNormalizeResourceUri(pUri);

    return this._attachments.some(
      (pAtt) =>
        pAtt.resource.uri === pUri || fNormalizeResourceUri(pAtt.resource.uri) === cNormalizedUri
    );
  }

  /**
   * Check if a resource is subscribed
   */
  mIsSubscribed(pUri: string): boolean {
    return this.vSubscriptions.has(pUri);
  }

  /**
   * Remove an attachment
   */
  mRemoveAttachment(pAttachmentId: string): void {
    this._attachments = this._attachments.filter((pAtt) => pAtt.id !== pAttachmentId);
    console.log(`[MCPResources] Removed attachment: ${pAttachmentId}`);
  }

  /**
   * Remove a subscription for a resource
   */
  mRemoveSubscription(pUri: string): void {
    this.vSubscriptions.delete(pUri);

    const cCached = this.vCachedResources.get(pUri);

    if (cCached) {
      this.vCachedResources.set(pUri, { ...cCached, subscribed: false });
    }

    console.log(`[MCPResources] Removed subscription: ${pUri}`);
  }

  /**
   * Set global loading state
   */
  mSetLoading(pLoading: boolean): void {
    this._isLoading = pLoading;
  }

  /**
   * Set error state for a server's resources
   */
  mSetServerError(pServerName: string, pError: string): void {
    const cExisting = this.vServerResources.get(pServerName);

    if (cExisting) {
      this.vServerResources.set(pServerName, { ...cExisting, error: pError, loading: false });
    } else {
      this.vServerResources.set(pServerName, {
        error: pError,
        loading: false,
        resources: [],
        serverName: pServerName,
        templates: []
      });
    }
  }

  /**
   * Set loading state for a server's resources
   */
  mSetServerLoading(pServerName: string, pLoading: boolean): void {
    const cExisting = this.vServerResources.get(pServerName);

    if (cExisting) {
      this.vServerResources.set(pServerName, { ...cExisting, loading: pLoading });
    } else {
      this.vServerResources.set(pServerName, {
        error: undefined,
        loading: pLoading,
        resources: [],
        serverName: pServerName,
        templates: []
      });
    }
  }

  /**
   * Set resources for a server (called after listResources)
   */
  mSetServerResources(
    pServerName: string,
    pResources: MCPResource[],
    pTemplates: MCPResourceTemplate[]
  ): void {
    this.vServerResources.set(pServerName, {
      error: undefined,
      lastFetched: new Date(),
      loading: false,
      resources: pResources,
      serverName: pServerName,
      templates: pTemplates
    });
    console.log(
      `[MCPResources][${pServerName}] Set ${pResources.length} resources, ${pTemplates.length} templates`
    );
  }

  /**
   * Convert current resource attachments to DatabaseMessageExtra[] for persisting with a message.
   * Each attachment becomes a DatabaseMessageExtraMcpResource stored on the user message.
   */
  mToMessageExtras(): DatabaseMessageExtraMcpResource[] {
    const aExtras: DatabaseMessageExtraMcpResource[] = [];

    for (const attachment of this._attachments) {
      if (attachment.error) continue;

      if (!attachment.content || attachment.content.length === 0) continue;

      const cResourceName = attachment.resource.title || attachment.resource.name;
      const aContentParts: string[] = [];

      for (const cContent of attachment.content) {
        if ('text' in cContent && cContent.text) {
          aContentParts.push(cContent.text);
        } else if ('blob' in cContent && cContent.blob) {
          aContentParts.push(
            `[${cBinaryContentLabel}: ${cContent.mimeType || cResourceUnknownType}]`
          );
        }
      }

      if (aContentParts.length > 0) {
        aExtras.push({
          content: aContentParts.join(cNewline),
          mimeType: attachment.resource.mimeType,
          name: cResourceName,
          serverName: attachment.resource.serverName,
          type: AttachmentType.MCP_RESOURCE,
          uri: attachment.resource.uri
        });
      }
    }

    return aExtras;
  }

  /**
   * Update attachment with fetched content
   */
  mUpdateAttachmentContent(pAttachmentId: string, pContent: MCPResourceContent[]): void {
    this._attachments = this._attachments.map((pAtt) =>
      pAtt.id === pAttachmentId
        ? { ...pAtt, content: pContent, error: undefined, loading: false }
        : pAtt
    );
  }

  /**
   * Update attachment with error
   */
  mUpdateAttachmentError(pAttachmentId: string, pError: string): void {
    this._attachments = this._attachments.map((pAtt) =>
      pAtt.id === pAttachmentId ? { ...pAtt, error: pError, loading: false } : pAtt
    );
  }
}

export const cMcpResourceStore = new MCPResourceStore();
