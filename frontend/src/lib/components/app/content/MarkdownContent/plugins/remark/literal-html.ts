import { cLineBreak, cNbsp, sPhraseParents, cTabAsSpaces } from '$lib/constants';
import type { Break, Content, Paragraph, PhrasingContent, Root, Text } from 'mdast';
import type { Plugin } from 'unified';
import { visit } from 'unist-util-visit';

/**
 * remark plugin that rewrites raw HTML nodes into plain-text equivalents.
 *
 * remark parses inline HTML into `html` nodes even when we do not want to render
 * them. We turn each of those nodes into regular text (plus `<br>` break markers)
 * so the downstream rehype pipeline escapes the characters instead of executing
 * them. Leading spaces and tab characters are converted to non‑breaking spaces to
 * keep indentation identical to the original author input.
 */

function fPreserveIndent(pLine: string): string {
  let vIndex = 0;
  let vOutput = '';

  while (vIndex < pLine.length) {
    const cChar = pLine[vIndex];

    if (cChar === ' ') {
      vOutput += cNbsp;
      vIndex += 1;

      continue;
    }

    if (cChar === '\t') {
      vOutput += cTabAsSpaces;
      vIndex += 1;

      continue;
    }

    break;
  }

  return vOutput + pLine.slice(vIndex);
}

function fCreateLiteralChildren(pValue: string): PhrasingContent[] {
  const aLines = pValue.split(cLineBreak);
  const aNodes: PhrasingContent[] = [];

  for (const [vLineIndex, vRawLine] of aLines.entries()) {
    if (vLineIndex > 0) {
      aNodes.push({ type: 'break' } as Break as unknown as PhrasingContent);
    }

    aNodes.push({
      type: 'text',
      value: fPreserveIndent(vRawLine)
    } as Text as unknown as PhrasingContent);
  }

  if (!aNodes.length) {
    aNodes.push({ type: 'text', value: '' } as Text as unknown as PhrasingContent);
  }

  return aNodes;
}

export const fRemarkLiteralHtml: Plugin<[], Root> = () => {
  return (pTree) => {
    visit(pTree, 'html', (node, pIndex, pParent) => {
      if (!pParent || typeof pIndex !== 'number') {
        return;
      }

      const aReplacement = fCreateLiteralChildren(node.value);

      if (!sPhraseParents.has(pParent.type as string)) {
        const cParagraph: Paragraph = {
          children: aReplacement as Paragraph['children'],
          data: { literalHtml: true },
          type: 'paragraph'
        };
        const aSiblings = pParent.children as unknown as Content[];

        aSiblings.splice(pIndex, 1, cParagraph as unknown as Content);

        if (pIndex > 0) {
          const cPrevious = aSiblings[pIndex - 1] as Paragraph | undefined;

          if (
            cPrevious?.type === 'paragraph' &&
            (cPrevious.data as { literalHtml?: boolean } | undefined)?.literalHtml
          ) {
            const aPrevChildren = cPrevious.children as unknown as PhrasingContent[];

            if (aPrevChildren.length) {
              const cLastChild = aPrevChildren[aPrevChildren.length - 1];

              if (cLastChild.type !== 'break') {
                aPrevChildren.push({
                  type: 'break'
                } as Break as unknown as PhrasingContent);
              }
            }

            aPrevChildren.push(...(cParagraph.children as unknown as PhrasingContent[]));

            aSiblings.splice(pIndex, 1);

            return pIndex;
          }
        }

        return pIndex + 1;
      }

      (pParent.children as unknown as PhrasingContent[]).splice(
        pIndex,
        1,
        ...(aReplacement as unknown as PhrasingContent[])
      );

      return pIndex + aReplacement.length;
    });
  };
};
