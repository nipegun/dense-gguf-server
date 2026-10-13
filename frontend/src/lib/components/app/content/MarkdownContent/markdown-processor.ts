// Shared remark/rehype pipeline factory for MarkdownContent.
//
// The frozen plugin chain is expensive to build ( ~15 plugin instances ),
// and MarkdownContent used to rebuild it on every processMarkdown call:
// once per block at mount, and again on every coalesced chunk while
// streaming. Pipelines without attachments are shared process-wide per
// math flag; attachment-bearing pipelines are cached by the attachments
// array identity, which changes whenever extras are updated.

import { fRehypeEnhanceCodeBlocks } from './plugins/rehype/enhance-code-blocks';
import { fRehypeEnhanceLinks } from './plugins/rehype/enhance-links';
import { fRehypeEnhanceMermaidBlocks } from './plugins/rehype/enhance-mermaid-blocks';
import { fRehypeEnhanceSvgBlocks } from './plugins/rehype/enhance-svg-blocks';
import { fRehypeEnhanceTables } from './plugins/rehype/enhance-tables';
import { fRehypeFileBadge } from './plugins/rehype/file-badge';
import { fRehypeMermaidPre } from './plugins/rehype/mermaid-pre';
import { fRehypeRtlSupport } from './plugins/rehype/rehype-rtl-support';
import { fRehypeResolveAttachmentImages } from './plugins/rehype/resolve-attachment-images';
import { fRehypeSvgPre } from './plugins/rehype/svg-pre';
import { fRehypeRestoreTableHtml } from './plugins/rehype/table-html-restorer';
import { fRemarkLiteralHtml } from './plugins/remark/literal-html';
import { FileTypeText } from '$lib/enums/files.enums';
import type { DatabaseMessageExtra } from '$lib/types/database';
import type { Root as HastRoot } from 'hast';
import { all as lowlightAll } from 'lowlight';
import type { Root as MdastRoot } from 'mdast';
import rehypeHighlight from 'rehype-highlight';
import rehypeKatex from 'rehype-katex';
import rehypeStringify from 'rehype-stringify';
import { remark } from 'remark';
import remarkBreaks from 'remark-breaks';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';

export interface MarkdownProcessor {
  mParse(pMarkdown: string): MdastRoot;
  mRun(pTree: MdastRoot): Promise<HastRoot>;
  mStringify(pTree: HastRoot): string;
}

export interface MarkdownProcessorOptions {
  attachments?: DatabaseMessageExtra[];
  disableMath?: boolean;
}

const dSharedPipelines = new Map<string, MarkdownProcessor>();
const cAttachmentPipelines = new WeakMap<object, MarkdownProcessor>();

function fBuildPipeline({
  attachments: vAttachments,
  disableMath: vDisableMath = false
}: MarkdownProcessorOptions): MarkdownProcessor {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let vProc: any = remark().use(remarkGfm); // GitHub Flavored Markdown

  if (!vDisableMath) {
    vProc = vProc.use(remarkMath); // Parse $inline$ and $$block$$ math
  }

  vProc = vProc
    .use(remarkBreaks) // Convert line breaks to <br>
    // Treat raw HTML as literal text with preserved indentation
    .use(fRemarkLiteralHtml)
    .use(remarkRehype); // Convert Markdown AST to rehype

  if (!vDisableMath) {
    vProc = vProc.use(rehypeKatex); // Render math using KaTeX
  }

  const cPipeline = vProc
    .use(rehypeHighlight, {
      aliases: { [FileTypeText.cXml]: [FileTypeText.cSvelte, FileTypeText.cVue] },
      languages: lowlightAll
    }) // Add syntax highlighting
    .use(fRehypeRestoreTableHtml) // Restore limited HTML (e.g. <br>, <ul>) inside Markdown tables
    .use(fRehypeEnhanceTables) // Wrap tables in a horizontal scroll container
    .use(fRehypeEnhanceLinks) // Add target="_blank" to links
    .use(fRehypeFileBadge) // Render file:// anchors as inline badge chips
    .use(fRehypeMermaidPre) // Convert mermaid blocks to <pre class="mermaid">
    .use(fRehypeSvgPre) // Convert svg blocks to <pre class="svg-block">
    .use(fRehypeEnhanceCodeBlocks) // Wrap code blocks with header and actions
    .use(fRehypeEnhanceMermaidBlocks) // Wrap mermaid blocks with header and actions
    .use(fRehypeEnhanceSvgBlocks) // Wrap svg blocks with header and actions
    .use(fRehypeResolveAttachmentImages, { attachments: vAttachments })
    .use(fRehypeRtlSupport) // Add bidirectional text support
    .use(rehypeStringify, { allowDangerousHtml: true }); // Convert to HTML string

  return cPipeline as MarkdownProcessor;
}

export function fGetMarkdownProcessor(pOptions: MarkdownProcessorOptions): MarkdownProcessor {
  if (pOptions.attachments && pOptions.attachments.length > 0) {
    let vCached = cAttachmentPipelines.get(pOptions.attachments);

    if (!vCached) {
      vCached = fBuildPipeline(pOptions);
      cAttachmentPipelines.set(pOptions.attachments, vCached);
    }

    return vCached;
  }

  const cKey = String(Boolean(pOptions.disableMath));

  let vCached = dSharedPipelines.get(cKey);

  if (!vCached) {
    vCached = fBuildPipeline(pOptions);
    dSharedPipelines.set(cKey, vCached);
  }

  return vCached;
}
