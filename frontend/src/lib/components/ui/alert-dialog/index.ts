import Action from './alert-dialog-action.svelte';
import Cancel from './alert-dialog-cancel.svelte';
import Content from './alert-dialog-content.svelte';
import Description from './alert-dialog-description.svelte';
import Footer from './alert-dialog-footer.svelte';
import Header from './alert-dialog-header.svelte';
import Overlay from './alert-dialog-overlay.svelte';
import Title from './alert-dialog-title.svelte';
import Trigger from './alert-dialog-trigger.svelte';
import { AlertDialog as AlertDialogPrimitive } from 'bits-ui';

const fRoot = AlertDialogPrimitive.Root;
const fPortal = AlertDialogPrimitive.Portal;

export {
  fRoot,
  Title,
  Action,
  Cancel,
  fPortal,
  Footer,
  Header,
  Trigger,
  Overlay,
  Content,
  Description,
  //
  fRoot as AlertDialog,
  Title as AlertDialogTitle,
  Action as AlertDialogAction,
  Cancel as AlertDialogCancel,
  fPortal as AlertDialogPortal,
  Footer as AlertDialogFooter,
  Header as AlertDialogHeader,
  Trigger as AlertDialogTrigger,
  Overlay as AlertDialogOverlay,
  Content as AlertDialogContent,
  Description as AlertDialogDescription
};
