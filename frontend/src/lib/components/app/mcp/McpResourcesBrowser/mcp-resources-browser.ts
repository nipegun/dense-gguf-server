import type { MCPResource, MCPResourceInfo } from '$lib/types';
import { fParseResourcePath } from '$lib/utils';

export interface ResourceTreeNode {
  name: string;
  resource?: MCPResourceInfo;
  children: Map<string, ResourceTreeNode>;
  isFiltered?: boolean;
}

function fResourceMatchesSearch(pResource: MCPResource, pQuery: string): boolean {
  return (
    pResource.title?.toLowerCase().includes(pQuery) || pResource.uri.toLowerCase().includes(pQuery)
  );
}

export function fBuildResourceTree(
  pResourceList: MCPResource[],
  pServerName: string,
  pSearchQuery?: string
): ResourceTreeNode {
  const root: ResourceTreeNode = { children: new Map(), name: 'root' };

  if (!pSearchQuery || !pSearchQuery.trim()) {
    for (const resource of pResourceList) {
      const aPathParts = fParseResourcePath(resource.uri);

      let vCurrent = root;

      for (let vI = 0; vI < aPathParts.length - 1; vI++) {
        const cPart = aPathParts[vI];

        if (!vCurrent.children.has(cPart)) {
          vCurrent.children.set(cPart, { children: new Map(), name: cPart });
        }

        vCurrent = vCurrent.children.get(cPart)!;
      }

      const cFileName = aPathParts[aPathParts.length - 1] || resource.name;

      vCurrent.children.set(resource.uri, {
        children: new Map(),
        name: cFileName,
        resource: { ...resource, serverName: pServerName }
      });
    }

    return root;
  }

  const cQuery = pSearchQuery.toLowerCase();

  // Build tree with filtering
  for (const resource of pResourceList) {
    if (!fResourceMatchesSearch(resource, cQuery)) continue;

    const aPathParts = fParseResourcePath(resource.uri);

    let vCurrent = root;

    for (let vI = 0; vI < aPathParts.length - 1; vI++) {
      const cPart = aPathParts[vI];

      if (!vCurrent.children.has(cPart)) {
        vCurrent.children.set(cPart, { children: new Map(), isFiltered: true, name: cPart });
      }

      vCurrent = vCurrent.children.get(cPart)!;
    }

    const cFileName = aPathParts[aPathParts.length - 1] || resource.name;

    vCurrent.children.set(resource.uri, {
      children: new Map(),
      isFiltered: true,
      name: cFileName,
      resource: { ...resource, serverName: pServerName }
    });
  }

  function fCleanupEmptyFolders(node: ResourceTreeNode): boolean {
    if (node.resource) return true;

    const aToDelete: string[] = [];

    for (const [vName, vChild] of node.children.entries()) {
      if (!fCleanupEmptyFolders(vChild)) {
        aToDelete.push(vName);
      }
    }

    for (const name of aToDelete) {
      node.children.delete(name);
    }

    return node.children.size > 0;
  }

  fCleanupEmptyFolders(root);

  return root;
}

export function fCountTreeResources(node: ResourceTreeNode): number {
  if (node.resource) return 1;

  let vCount = 0;

  for (const cChild of node.children.values()) {
    vCount += fCountTreeResources(cChild);
  }

  return vCount;
}

export function fSortTreeChildren(pChildren: ResourceTreeNode[]): ResourceTreeNode[] {
  return pChildren.sort((pA, pB) => {
    const aIsFolder = !pA.resource && pA.children.size > 0;
    const cBIsFolder = !pB.resource && pB.children.size > 0;

    if (aIsFolder && !cBIsFolder) return -1;

    if (!aIsFolder && cBIsFolder) return 1;

    return pA.name.localeCompare(pB.name);
  });
}
