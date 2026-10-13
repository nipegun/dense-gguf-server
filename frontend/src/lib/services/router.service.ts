/**
 * RouterService - Builds app route paths
 *
 * Returns chat route strings from a single source of truth (ROUTES). No state.
 */

import { cRoutes } from '$lib/constants';

export class RouterService {
  static mChat(pId: string): string {
    return `${cRoutes.CHAT}/${pId}`;
  }
}
