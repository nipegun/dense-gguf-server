/**
 * Comprehensive dictionary of all supported file types in dgs-ui
 * Organized by category with TypeScript enums for better type safety
 */

// File type category enum
export enum FileTypeCategory {
  AUDIO = 'audio',
  IMAGE = 'image',
  PDF = 'pdf',
  TEXT = 'text',
  VIDEO = 'video'
}

/**
 * Special file types for internal use (not MIME types)
 */
export enum SpecialFileType {
  MCP_PROMPT = 'mcp-prompt'
}

// Specific file type enums for each category
export enum FileTypeImage {
  cGif = 'gif',
  cHeic = 'heic',
  cHeif = 'heif',
  cJpeg = 'jpeg',
  cPng = 'png',
  SVG = 'svg',
  cWebp = 'webp'
}

export enum FileTypeAudio {
  cMp3 = 'mp3',
  cWav = 'wav',
  cWebm = 'webm'
}

export enum FileTypeVideo {
  cMp4 = 'mp4',
  cOgg = 'ogg',
  cWebm = 'webm'
}

export enum FileTypePdf {
  PDF = 'pdf'
}

export enum FileTypeText {
  cAsciidoc = 'asciidoc',
  cBibtex = 'bibtex',
  cCpp = 'cpp',
  cCsharp = 'csharp',
  cCss = 'css',
  cCsv = 'csv',
  cCuda = 'cuda',
  cDart = 'dart',
  cGo = 'go',
  cHaskell = 'haskell',
  HTML = 'html',
  cJava = 'java',
  cJavascript = 'js',
  cJson = 'json',
  cJsx = 'jsx',
  cKotlin = 'kotlin',
  cLatex = 'latex',
  cLog = 'log',
  MARKDOWN = 'md',
  cPhp = 'php',
  cPlainText = 'plainText',
  cProperties = 'properties',
  cPython = 'python',
  R = 'r',
  cRuby = 'ruby',
  cRust = 'rust',
  cScala = 'scala',
  cShell = 'shell',
  cSql = 'sql',
  cSvelte = 'svelte',
  cSwift = 'swift',
  cTsx = 'tsx',
  cTypescript = 'ts',
  cVue = 'vue',
  cVulkan = 'vulkan',
  cXml = 'xml',
  cYaml = 'yaml'
}

// File extension enums
export enum FileExtensionImage {
  cGif = '.gif',
  cHeic = '.heic',
  cHeif = '.heif',
  cJpeg = '.jpeg',
  cJpg = '.jpg',
  cPng = '.png',
  SVG = '.svg',
  cWebp = '.webp'
}

export enum FileExtensionAudio {
  cMp3 = '.mp3',
  cWav = '.wav'
}

export enum FileExtensionVideo {
  cMp4 = '.mp4',
  cOgg = '.ogg',
  cWebm = '.webm'
}

export enum FileExtensionPdf {
  PDF = '.pdf'
}

export enum FileExtensionText {
  cAdoc = '.adoc',
  cBat = '.bat',
  cBib = '.bib',
  C = '.c',
  cComp = '.comp',
  cCpp = '.cpp',
  cCs = '.cs',
  cCss = '.css',
  cCsv = '.csv',
  cCu = '.cu',
  cCuh = '.cuh',
  cDart = '.dart',
  cGo = '.go',
  cH = '.h',
  cHpp = '.hpp',
  cHs = '.hs',
  cHtm = '.htm',
  HTML = '.html',
  cJava = '.java',
  cJs = '.js',
  cJson = '.json',
  JSONL = '.jsonl',
  cJsx = '.jsx',
  cKt = '.kt',
  cLog = '.log',
  cMd = '.md',
  cPhp = '.php',
  cProperties = '.properties',
  cPy = '.py',
  R = '.r',
  cRb = '.rb',
  cRs = '.rs',
  cScala = '.scala',
  cSh = '.sh',
  cSql = '.sql',
  cSvelte = '.svelte',
  cSwift = '.swift',
  cTex = '.tex',
  cTs = '.ts',
  cTsx = '.tsx',
  cTxt = '.txt',
  cVue = '.vue',
  cXml = '.xml',
  cYaml = '.yaml',
  cYml = '.yml',
  cZip = '.zip'
}

// MIME type prefixes and includes for content detection
export enum MimeTypePrefix {
  AUDIO = 'audio/',
  IMAGE = 'image/',
  TEXT = 'text'
}

export enum MimeTypeIncludes {
  cJavascript = 'javascript',
  cJson = 'json',
  cTypescript = 'typescript'
}

// URI patterns for content detection
export enum UriPattern {
  cDatabaseKeyword = 'database',
  cDatabaseScheme = 'db://'
}

// MIME type enums
export enum MimeTypeApplication {
  cJson = 'application/json',
  cOctetStream = 'application/octet-stream',
  PDF = 'application/pdf',
  cZip = 'application/zip'
}

export enum MimeTypeAudio {
  cMp3 = 'audio/mp3',
  cMp3Mpeg = 'audio/mpeg',
  cMp4 = 'audio/mp4',
  cVndWave = 'audio/vnd.wave',
  cWav = 'audio/wav',
  cWave = 'audio/wave',
  cWebm = 'audio/webm',
  cWebmOpus = 'audio/webm;codecs=opus',
  cXPnWav = 'audio/x-pn-wav',
  cXWav = 'audio/x-wav',
  cXWave = 'audio/x-wave'
}

export enum MimeTypeVideo {
  cMp4 = 'video/mp4',
  cOgg = 'video/ogg',
  cWebm = 'video/webm'
}

export enum MimeTypeImage {
  cGif = 'image/gif',
  cHeic = 'image/heic',
  cHeif = 'image/heif',
  cIco = 'image/x-icon',
  cIcoMicrosoft = 'image/vnd.microsoft.icon',
  cJpeg = 'image/jpeg',
  cJpg = 'image/jpg',
  cPng = 'image/png',
  SVG = 'image/svg+xml',
  cWebp = 'image/webp'
}

export enum MimeTypeText {
  cAsciidoc = 'text/asciidoc',
  cBat = 'application/x-bat',
  cBibtex = 'text/x-bibtex',
  cCHdr = 'text/x-chdr',
  cCSrc = 'text/x-csrc',
  cCppHdr = 'text/x-c++hdr',
  cCppSrc = 'text/x-c++src',
  cCsharp = 'text/x-csharp',
  cCss = 'text/css',
  cCsv = 'text/csv',
  cCuda = 'text/x-cuda',
  cDart = 'text/x-dart',
  cGo = 'text/x-go',
  cHaskell = 'text/x-haskell',
  HTML = 'text/html',
  cJava = 'text/x-java-source',
  cJavascript = 'text/javascript',
  cJavascriptApp = 'application/javascript',
  cJson = 'application/json',
  JSONL = 'application/jsonl',
  cJsx = 'text/jsx',
  cKotlin = 'text/x-kotlin',
  cLatex = 'application/x-latex',
  MARKDOWN = 'text/markdown',
  cPhp = 'text/x-php',
  cPlain = 'text/plain',
  cProperties = 'text/properties',
  cPython = 'text/x-python',
  R = 'text/x-r',
  cRuby = 'text/x-ruby',
  cRust = 'text/x-rust',
  cScala = 'text/x-scala',
  cShell = 'text/x-shellscript',
  cSql = 'text/x-sql',
  cSvelte = 'text/x-svelte',
  cSwift = 'text/x-swift',
  cTex = 'text/x-tex',
  cTexApp = 'application/x-tex',
  cTsx = 'text/tsx',
  cTypescript = 'text/typescript',
  cVue = 'text/x-vue',
  cXmlApp = 'application/xml',
  cXmlText = 'text/xml',
  cYamlApp = 'application/yaml',
  cYamlText = 'text/yaml'
}
