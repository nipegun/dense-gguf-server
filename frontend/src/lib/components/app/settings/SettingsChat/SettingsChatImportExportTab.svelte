<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import SettingsChatImportExportSection from './SettingsChatImportExportSection.svelte';
  import { Download, Trash2, Upload } from '@lucide/svelte';
  import {
    DialogConfirmation,
    DialogConversationSelection,
    DialogExportSettings
  } from '$lib/components/app';
  import SettingsGroup from '$lib/components/app/settings/SettingsGroup.svelte';
  import { ConversationSelectionMode, FileExtensionText, HtmlInputType } from '$lib/enums';
  import { ConversationTransferService } from '$lib/services';
  import { cConversationsStore, cSettingsStore } from '$lib/stores';
  import { fCreateMessageCountMap } from '$lib/utils';
  import { fade } from 'svelte/transition';
  import { toast } from 'svelte-sonner';

  let aExportedConversations = $state<DatabaseConversation[]>([]);
  let aImportedConversations = $state<DatabaseConversation[]>([]);
  let vShowExportSummary = $state(false);
  let vShowImportSummary = $state(false);

  let vShowExportDialog = $state(false);
  let vShowImportDialog = $state(false);
  let aAvailableConversations = $state<DatabaseConversation[]>([]);
  let dMessageCountMap = $state<Map<string, number>>(new Map());
  let aFullImportData = $state<Array<{ conv: DatabaseConversation; messages: DatabaseMessage[] }>>(
    []
  );

  // Delete functionality state
  let vShowDeleteDialog = $state(false);

  // Settings import/export state
  let vShowSettingsExportSummary = $state(false);
  let vShowSettingsImportSummary = $state(false);
  let vShowSettingsExportDialog = $state(false);
  let vIncludeSensitiveData = $state(false);

  function fHandleSettingsExport() {
    vShowSettingsExportDialog = true;
    vIncludeSensitiveData = false;
  }

  function fHandleSettingsExportConfirm() {
    vShowSettingsExportDialog = false;

    try {
      const data = cSettingsStore.mExportSettings(vIncludeSensitiveData);
      const cBlob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(cBlob);
      const cA = document.createElement('a');

      cA.href = url;
      cA.download = `dgs_settings_${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(cA);
      cA.click();
      document.body.removeChild(cA);
      URL.revokeObjectURL(url);

      vShowSettingsExportSummary = true;
      vShowSettingsImportSummary = false;
      toast.success(fTranslate('text_5bfca92e998f'));
    } catch (vErr) {
      console.error('Failed to export settings:', vErr);
      toast.error(fTranslate('text_3cf664c4bd2b'));
    }
  }

  function fHandleSettingsExportCancel() {
    vShowSettingsExportDialog = false;
  }

  function fHandleSettingsImport() {
    try {
      const input = document.createElement('input');

      input.type = HtmlInputType.FILE;
      input.accept = FileExtensionText.cJson;

      input.onchange = async (pE) => {
        const file = (pE.target as HTMLInputElement)?.files?.[0];

        if (!file) return;

        try {
          const text = await file.text();
          const data = JSON.parse(text);

          if (!data || typeof data !== 'object' || !data.config) {
            toast.error(fTranslate('text_e65771eee330'));

            return;
          }

          cSettingsStore.mImportSettings(data);

          vShowSettingsImportSummary = true;
          vShowSettingsExportSummary = false;
          toast.success(fTranslate('text_f3addd823e40'));
        } catch (vErr) {
          console.error('Failed to import settings:', vErr);
          toast.error(fTranslate('text_6ffd4b51379f'));
        }
      };

      input.click();
    } catch (vErr) {
      console.error('Failed to open file picker:', vErr);
      toast.error(fTranslate('text_5843f72974fd'));
    }
  }

  async function fHandleExportClick() {
    try {
      const aAllConversations = cConversationsStore.aConversations;

      if (aAllConversations.length === 0) {
        toast.info(fTranslate('text_bf0779e06dd8'));

        return;
      }

      const aConversationsWithMessages = await Promise.all(
        aAllConversations.map(async (pConv: DatabaseConversation) => {
          const aMessages = await cConversationsStore.mGetConversationMessages(pConv.id);

          return { conv: pConv, messages: aMessages };
        })
      );

      dMessageCountMap = fCreateMessageCountMap(aConversationsWithMessages);
      aAvailableConversations = aAllConversations;
      vShowExportDialog = true;
    } catch (vErr) {
      console.error('Failed to load conversations:', vErr);
      alert('Failed to load conversations');
    }
  }

  async function fHandleExportConfirm(pSelectedConversations: DatabaseConversation[]) {
    try {
      const aAllData = await cConversationsStore.mGetConversationsForExport(
        pSelectedConversations.map((pConv) => pConv.id)
      );

      if (aAllData.length === 1) {
        ConversationTransferService.mDownloadConversationFile(aAllData[0]);
      } else {
        ConversationTransferService.mDownloadConversationsArchive(aAllData);
      }

      aExportedConversations = pSelectedConversations;
      vShowExportSummary = true;
      vShowImportSummary = false;
      vShowExportDialog = false;
    } catch (vErr) {
      console.error('Export failed:', vErr);
      alert('Failed to export conversations');
    }
  }

  async function fHandleImportClick() {
    try {
      const input = document.createElement('input');

      // No `accept` filter: iOS resolves each entry to a UTI and has none for
      // `.jsonl`, which greys out exported conversations in the file picker.
      // `parseImportFile` detects the format from the file contents instead.
      input.type = HtmlInputType.FILE;

      input.onchange = async (pE) => {
        const file = (pE.target as HTMLInputElement)?.files?.[0];

        if (!file) return;

        try {
          const aImportedData = await ConversationTransferService.mParseImportFile(file);

          if (aImportedData.length === 0) {
            throw new Error(fTranslate('text_5db93832d23b'));
          }

          aFullImportData = aImportedData;
          aAvailableConversations = aImportedData.map((pItem) => pItem.conv);
          dMessageCountMap = fCreateMessageCountMap(aImportedData);
          vShowImportDialog = true;
        } catch (vErr: unknown) {
          const message = vErr instanceof Error ? vErr.message : 'Unknown error';

          console.error('Failed to parse file:', vErr);
          alert(`Failed to parse file: ${message}`);
        }
      };

      input.click();
    } catch (vErr) {
      console.error('Import failed:', vErr);
      alert('Failed to import conversations');
    }
  }

  async function fHandleImportConfirm(pSelectedConversations: DatabaseConversation[]) {
    try {
      const sSelectedIds = new Set(pSelectedConversations.map((pC) => pC.id));
      const aSelectedData = $state
        .snapshot(aFullImportData)
        .filter((pItem) => sSelectedIds.has(pItem.conv.id));
      const { imported: aImported, skipped: aSkipped } =
        await cConversationsStore.mImportConversationsData(aSelectedData);

      // A conversation already in the database is left untouched, so the summary
      // lists what was written and the toast accounts for the rest.
      if (aSkipped.length > 0) {
        toast.info(
          `Skipped ${aSkipped.length} conversation${aSkipped.length === 1 ? '' : 's'} already in your library`
        );
      }

      aImportedConversations = aImported;
      vShowImportSummary = true;
      vShowExportSummary = false;
      vShowImportDialog = false;
    } catch (vErr) {
      console.error('Import failed:', vErr);
      alert('Failed to import conversations. Please check the file format.');
    }
  }

  async function fHandleDeleteAllClick() {
    try {
      const aAllConversations = cConversationsStore.aConversations;

      if (aAllConversations.length === 0) {
        toast.info(fTranslate('text_5c9a78a4a817'));

        return;
      }

      vShowDeleteDialog = true;
    } catch (vErr) {
      console.error('Failed to load conversations for deletion:', vErr);
      toast.error(fTranslate('text_ae55fc046c4c'));
    }
  }

  async function fHandleDeleteAllConfirm() {
    try {
      await cConversationsStore.mDeleteAll();

      vShowDeleteDialog = false;
    } catch (vErr) {
      console.error('Failed to delete conversations:', vErr);
    }
  }

  function fHandleDeleteAllCancel() {
    vShowDeleteDialog = false;
  }
</script>

<div in:fade={{ duration: 150 }} class="space-y-12">
  <SettingsGroup title={fTranslate('text_1d432f58690c')}>
    <SettingsChatImportExportSection
      IconComponent={Download}
      buttonText="Export conversations"
      description={fTranslate('text_e1c29b8d7188')}
      onclick={fHandleExportClick}
      summary={{ items: aExportedConversations, show: vShowExportSummary, verb: 'Exported' }}
      title={fTranslate('text_3664895579f0')}
    />

    <SettingsChatImportExportSection
      IconComponent={Upload}
      buttonText="Import conversations"
      description={fTranslate('text_a659626215c6')}
      onclick={fHandleImportClick}
      summary={{ items: aImportedConversations, show: vShowImportSummary, verb: 'Imported' }}
      title={fTranslate('text_2cff9baabf56')}
    />

    <SettingsChatImportExportSection
      IconComponent={Trash2}
      buttonClass="text-destructive-foreground justify-start justify-self-start bg-destructive hover:bg-destructive/80 md:w-auto"
      buttonText="Delete all conversations"
      buttonVariant="destructive"
      description={fTranslate('text_21814f9e74f4')}
      onclick={fHandleDeleteAllClick}
      title={fTranslate('text_ef85afa1b92c')}
      titleClass="text-destructive"
    />
  </SettingsGroup>

  <SettingsGroup title={fTranslate('text_74a883a037bc')}>
    <SettingsChatImportExportSection
      IconComponent={Download}
      buttonText="Export settings"
      description={fTranslate('text_72df65773b40')}
      onclick={fHandleSettingsExport}
      summary={{ items: [], show: vShowSettingsExportSummary, verb: 'Exported' }}
      title={fTranslate('text_3664895579f0')}
    />

    <SettingsChatImportExportSection
      IconComponent={Upload}
      buttonText="Import settings"
      description={fTranslate('text_6235faca5e63')}
      onclick={fHandleSettingsImport}
      summary={{ items: [], show: vShowSettingsImportSummary, verb: 'Imported' }}
      title={fTranslate('text_2cff9baabf56')}
    />
  </SettingsGroup>
</div>

<DialogExportSettings
  bind:includeSensitiveData={vIncludeSensitiveData}
  bind:open={vShowSettingsExportDialog}
  onCancel={fHandleSettingsExportCancel}
  onConfirm={fHandleSettingsExportConfirm}
/>

<DialogConversationSelection
  bind:open={vShowExportDialog}
  conversations={aAvailableConversations}
  messageCountMap={dMessageCountMap}
  mode={ConversationSelectionMode.cExport}
  onCancel={() => (vShowExportDialog = false)}
  onConfirm={fHandleExportConfirm}
/>

<DialogConversationSelection
  bind:open={vShowImportDialog}
  conversations={aAvailableConversations}
  messageCountMap={dMessageCountMap}
  mode={ConversationSelectionMode.cImport}
  onCancel={() => (vShowImportDialog = false)}
  onConfirm={fHandleImportConfirm}
/>

<DialogConfirmation
  bind:open={vShowDeleteDialog}
  cancelText="Cancel"
  confirmText="Delete All"
  description={fTranslate('text_45145fe2a7ce')}
  icon={Trash2}
  onCancel={fHandleDeleteAllCancel}
  onConfirm={fHandleDeleteAllConfirm}
  title={fTranslate('text_43a72ccf4424')}
  variant="destructive"
/>
