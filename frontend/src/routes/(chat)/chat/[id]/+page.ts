import type { PageLoad } from './$types';
import { fInitStores } from '$lib/stores/init';
import { fValidateApiKey } from '$lib/utils';

export const load: PageLoad = async ({ fetch: vFetch }) => {
  // loads run before the root layout script, so the stored API key reaches
  // the probe only once the settings store has read localStorage
  await fInitStores();
  await fValidateApiKey(vFetch);
};
