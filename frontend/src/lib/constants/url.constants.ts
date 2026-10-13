import { UrlProtocol } from '$lib/enums';

const aStd = ['com', 'net', 'org', 'gov', 'edu'] as const;
const aStdMil = [...aStd, 'mil'] as const;
const dCcTLDPrefixes: Record<string, readonly string[]> = {
  ae: ['co', 'net', 'org', 'gov', 'ac', 'sch'],
  // --- Standard 5 only ---
  ar: aStd,
  au: [...aStdMil, 'id', 'asn', 'csiro'],
  bd: aStd,
  bg: aStd,
  br: [
    ...aStdMil,
    'art',
    'eco',
    'eng',
    'inf',
    'med',
    'psi',
    'tmp',
    'etc',
    'adm',
    'adv',
    'arq',
    'bio',
    'bmd',
    'cim',
    'cng',
    'cnt',
    'coop',
    'ecn',
    'esp',
    'far',
    'fm',
    'fnd',
    'fot',
    'fst',
    'g12',
    'ggf',
    'imb',
    'ind',
    'jor',
    'jus',
    'leg',
    'lel',
    'mat',
    'mp',
    'mus',
    'not',
    'ntr',
    'odo',
    'ppg',
    'pro',
    'psc',
    'qsl',
    'rec',
    'slg',
    'srv',
    'trd',
    'tur',
    'tv',
    'vet',
    'vlog',
    'wiki',
    'zlg'
  ],
  cn: aStdMil,
  eg: aStd,
  gr: aStd,
  hk: aStd,
  hr: aStd,
  hu: ['co', 'net', 'org', 'gov', 'edu'],
  id: [...aStdMil, 'co', 'go', 'or', 'web', 'sch'],
  il: ['co', 'net', 'org', 'gov', 'ac', 'muni'],
  in: [...aStdMil, 'co', 'gen', 'ind', 'firm', 'ernet', 'nic'],
  jp: ['ac', 'ad', 'co', 'ed', 'go', 'gr', 'lg', 'ne', 'or'],
  ke: ['co', 'or', 'ne', 'go', 'ac', 'sc'],
  kr: [...aStdMil, 'co', 'go', 'or', 'ac', 're'],
  lk: aStd,
  mx: aStdMil,
  my: aStdMil,
  ng: aStd,
  nz: [
    ...aStdMil,
    'co',
    'gen',
    'geek',
    'kiwi',
    'maori',
    'school',
    'govt',
    'health',
    'iwi',
    'parliament'
  ],

  ph: aStd,
  pk: aStd,
  pl: aStd,
  ro: aStd,
  rs: ['co', 'net', 'org', 'gov', 'edu'],
  ru: aStd,
  sa: aStd,
  sg: [...aStd, 'per'],

  si: aStd,
  th: ['co', 'go', 'or', 'in', 'ac', 'mi', 'net'],
  tr: aStd,
  tw: aStd,
  ua: aStd,
  uk: ['co', 'org', 'net', 'ac', 'gov', 'mil', 'nhs', 'police', 'mod', 'ltd', 'plc', 'me', 'sch'],
  ve: aStd,
  za: ['co', 'org', 'net', 'web', 'law', 'mil']
};
const dWildcardBases: Record<string, readonly string[]> = {
  br: ['nom', 'blog'],
  jp: [
    'kobe',
    'kyoto',
    'nagoya',
    'osaka',
    'sapporo',
    'sendai',
    'tokyo',
    'yokohama',
    'aichi',
    'akita',
    'aomori',
    'chiba',
    'ehime',
    'fukui',
    'fukuoka',
    'fukushima',
    'gifu',
    'gunma',
    'hiroshima',
    'hokkaido',
    'hyogo',
    'ibaraki',
    'ishikawa',
    'iwate',
    'kagawa',
    'kagoshima',
    'kanagawa',
    'kochi',
    'kumamoto',
    'mie',
    'miyagi',
    'miyazaki',
    'nagano',
    'nara',
    'niigata',
    'oita',
    'okayama',
    'okinawa',
    'saga',
    'saitama',
    'shiga',
    'shimane',
    'shizuoka',
    'tochigi',
    'tokushima',
    'tottori',
    'toyama',
    'wakayama',
    'yamagata',
    'yamaguchi',
    'yamanashi'
  ]
};

function fBuildSuffixSet(pSuffixes: Record<string, readonly string[]>): Set<string> {
  const sSet = new Set<string>();

  for (const [vTld, aParts] of Object.entries(pSuffixes)) {
    for (const cPart of aParts) {
      sSet.add(`${cPart}.${vTld}`);
    }
  }

  return sSet;
}

export const sTwoPartPublicSuffixes = fBuildSuffixSet(dCcTLDPrefixes);
export const sWildcardPublicSuffixes = fBuildSuffixSet(dWildcardBases);

// Matches one or more trailing "/" characters at the end of a URL/path.
export const cTrailingSlashesRegex = /\/+$/;

// Protocols that apiFetch treats as absolute and passes through untouched.
// Add a protocol here when a caller needs to fetch an absolute URL with it.
export const aApiAbsoluteUrlProtocols = [UrlProtocol.cHttp, UrlProtocol.cHttps] as const;
