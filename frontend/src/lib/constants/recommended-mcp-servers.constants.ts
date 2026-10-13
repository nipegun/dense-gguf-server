import { fTranslate } from '$lib/i18n';
import type { RecommendedMCPServer } from '$lib/types';

// Suggested MCP servers shown as opt-in cards in the "Add New Server" dialog.
// Rendering these cards never reaches the upstream domain - favicons come
// from local bundles in static/recommended-mcp/ and the URL is only used
// after the user clicks Add.
export const aRecommendedMcpServers: RecommendedMCPServer[] = [
  {
    description: fTranslate('text_2fc097e2378b'),
    iconUrl: '/recommended-mcp/exa.ico',
    id: 'exa',
    name: 'Exa',
    url: 'https://mcp.exa.ai/mcp'
  },
  {
    description: fTranslate('text_464508c5f0f4'),
    iconUrl: '/recommended-mcp/huggingface.ico',
    id: 'huggingface',
    name: 'Hugging Face',
    url: 'https://huggingface.co/mcp'
  },
  {
    description: fTranslate('text_5a80bd7f18ef'),
    iconUrlDark: '/recommended-mcp/github-dark.png',
    iconUrlLight: '/recommended-mcp/github-light.png',
    id: 'github',
    name: 'GitHub',
    needsAuthorization: true,
    url: 'https://api.githubcopilot.com/mcp'
  },
  {
    description: fTranslate('text_df8d5fe20367'),
    iconUrl: '/recommended-mcp/context7.png',
    id: 'context7',
    name: 'Context7',
    url: 'https://mcp.context7.com/mcp'
  }
];
