/**
 * Source-space undo/redo history for the ChatFormInputRich, whose
 * imperative DOM rebuilds destroy the browser's native undo stack.
 * Entries record the state BEFORE an edit; edits within `groupWindowMs`
 * extend the open group so a typing burst undoes as a unit, while
 * structural edits (paste, mention insert, clear) pass `newGroup`.
 */

export interface SourceHistoryEntry {
  value: string;
  caret: number;
}

export class SourceHistory {
  private vLastPush = 0;
  private aRedoStack: SourceHistoryEntry[] = [];
  private aUndoStack: SourceHistoryEntry[] = [];

  constructor(
    private pLimit = 100,
    private pGroupWindowMs = 800
  ) {}

  mPush(pEntry: SourceHistoryEntry, pNow: number, pNewGroup = false): void {
    if (pNewGroup || pNow - this.vLastPush >= this.pGroupWindowMs || this.aUndoStack.length === 0) {
      this.aUndoStack.push(pEntry);

      if (this.aUndoStack.length > this.pLimit) this.aUndoStack.shift();
    }

    this.vLastPush = pNow;
    this.aRedoStack = [];
  }

  mRedo(pCurrent: SourceHistoryEntry): SourceHistoryEntry | null {
    const cEntry = this.aRedoStack.pop();

    if (!cEntry) return null;

    this.aUndoStack.push(pCurrent);
    this.vLastPush = 0;

    return cEntry;
  }

  mUndo(pCurrent: SourceHistoryEntry): SourceHistoryEntry | null {
    const cEntry = this.aUndoStack.pop();

    if (!cEntry) return null;

    this.aRedoStack.push(pCurrent);
    this.vLastPush = 0; // the next edit after an undo starts a new group

    return cEntry;
  }
}
