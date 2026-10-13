import {
  cAudioFileTypes,
  cImageFileTypes,
  cPdfFileTypes,
  cTextFileTypes,
  cVideoFileTypes
} from '$lib/constants';
import {
  FileExtensionAudio,
  FileExtensionImage,
  FileExtensionPdf,
  FileExtensionText,
  FileTypeCategory,
  MimeTypeApplication,
  MimeTypeAudio,
  MimeTypeImage,
  MimeTypeText,
  MimeTypeVideo
} from '$lib/enums';

function fNormalizeMimeType(pMimeType: string): string {
  return pMimeType.trim().toLowerCase();
}

export function fGetFileTypeCategory(pMimeType: string): FileTypeCategory | null {
  switch (fNormalizeMimeType(pMimeType)) {
    // Images
    case MimeTypeImage.cJpeg:
    case MimeTypeImage.cPng:
    case MimeTypeImage.cGif:
    case MimeTypeImage.cWebp:
    case MimeTypeImage.SVG:
    case MimeTypeImage.cHeic:
    case MimeTypeImage.cHeif:
      return FileTypeCategory.IMAGE;

    // Audio
    case MimeTypeAudio.cMp3Mpeg:
    case MimeTypeAudio.cMp3:
    case MimeTypeAudio.cMp4:
    case MimeTypeAudio.cWav:
    case MimeTypeAudio.cWave:
    case MimeTypeAudio.cXWav:
    case MimeTypeAudio.cXWave:
    case MimeTypeAudio.cVndWave:
    case MimeTypeAudio.cXPnWav:
    case MimeTypeAudio.cWebm:
    case MimeTypeAudio.cWebmOpus:
      return FileTypeCategory.AUDIO;

    // Video
    case MimeTypeVideo.cMp4:
    case MimeTypeVideo.cOgg:
    case MimeTypeVideo.cWebm:
      return FileTypeCategory.VIDEO;

    // PDF
    case MimeTypeApplication.PDF:
      return FileTypeCategory.PDF;

    // Text
    case MimeTypeText.cPlain:
    case MimeTypeText.MARKDOWN:
    case MimeTypeText.cAsciidoc:
    case MimeTypeText.cJavascript:
    case MimeTypeText.cJavascriptApp:
    case MimeTypeText.cTypescript:
    case MimeTypeText.cJsx:
    case MimeTypeText.cTsx:
    case MimeTypeText.cCss:
    case MimeTypeText.HTML:
    case MimeTypeText.cJson:
    case MimeTypeText.cXmlText:
    case MimeTypeText.cXmlApp:
    case MimeTypeText.cYamlText:
    case MimeTypeText.cYamlApp:
    case MimeTypeText.cCsv:
    case MimeTypeText.cPython:
    case MimeTypeText.cJava:
    case MimeTypeText.cCppSrc:
    case MimeTypeText.cCSrc:
    case MimeTypeText.cCHdr:
    case MimeTypeText.cPhp:
    case MimeTypeText.cRuby:
    case MimeTypeText.cGo:
    case MimeTypeText.cRust:
    case MimeTypeText.cShell:
    case MimeTypeText.cBat:
    case MimeTypeText.cSql:
    case MimeTypeText.R:
    case MimeTypeText.cScala:
    case MimeTypeText.cKotlin:
    case MimeTypeText.cSwift:
    case MimeTypeText.cDart:
    case MimeTypeText.cVue:
    case MimeTypeText.cSvelte:
    case MimeTypeText.cLatex:
    case MimeTypeText.cBibtex:
    case MimeTypeText.cCuda:
    case MimeTypeText.cCppHdr:
    case MimeTypeText.cCsharp:
    case MimeTypeText.cHaskell:
    case MimeTypeText.cProperties:
    case MimeTypeText.cTex:
    case MimeTypeText.cTexApp:
      return FileTypeCategory.TEXT;

    default:
      return null;
  }
}

export function fGetFileTypeCategoryByExtension(pFilename: string): FileTypeCategory | null {
  const cExtension = pFilename.toLowerCase().substring(pFilename.lastIndexOf('.'));

  switch (cExtension) {
    // Images
    case FileExtensionImage.cJpg:
    case FileExtensionImage.cJpeg:
    case FileExtensionImage.cPng:
    case FileExtensionImage.cGif:
    case FileExtensionImage.cWebp:
    case FileExtensionImage.SVG:
    case FileExtensionImage.cHeic:
    case FileExtensionImage.cHeif:
      return FileTypeCategory.IMAGE;

    // Audio
    case FileExtensionAudio.cMp3:
    case FileExtensionAudio.cWav:
      return FileTypeCategory.AUDIO;

    // PDF
    case FileExtensionPdf.PDF:
      return FileTypeCategory.PDF;

    // Text
    case FileExtensionText.cTxt:
    case FileExtensionText.cMd:
    case FileExtensionText.cAdoc:
    case FileExtensionText.cJs:
    case FileExtensionText.cTs:
    case FileExtensionText.cJsx:
    case FileExtensionText.cTsx:
    case FileExtensionText.cCss:
    case FileExtensionText.HTML:
    case FileExtensionText.cHtm:
    case FileExtensionText.cJson:
    case FileExtensionText.cXml:
    case FileExtensionText.cYaml:
    case FileExtensionText.cYml:
    case FileExtensionText.cCsv:
    case FileExtensionText.cLog:
    case FileExtensionText.cPy:
    case FileExtensionText.cJava:
    case FileExtensionText.cCpp:
    case FileExtensionText.C:
    case FileExtensionText.cH:
    case FileExtensionText.cPhp:
    case FileExtensionText.cRb:
    case FileExtensionText.cGo:
    case FileExtensionText.cRs:
    case FileExtensionText.cSh:
    case FileExtensionText.cBat:
    case FileExtensionText.cSql:
    case FileExtensionText.R:
    case FileExtensionText.cScala:
    case FileExtensionText.cKt:
    case FileExtensionText.cSwift:
    case FileExtensionText.cDart:
    case FileExtensionText.cVue:
    case FileExtensionText.cSvelte:
    case FileExtensionText.cTex:
    case FileExtensionText.cBib:
    case FileExtensionText.cComp:
    case FileExtensionText.cCu:
    case FileExtensionText.cCuh:
    case FileExtensionText.cHpp:
    case FileExtensionText.cHs:
    case FileExtensionText.cProperties:
      return FileTypeCategory.TEXT;

    default:
      return null;
  }
}

export function fGetFileTypeByExtension(pFilename: string): string | null {
  const cExtension = pFilename.toLowerCase().substring(pFilename.lastIndexOf('.'));

  for (const [vKey, vType] of Object.entries(cImageFileTypes)) {
    if ((vType.extensions as readonly string[]).includes(cExtension)) {
      return `${FileTypeCategory.IMAGE}:${vKey}`;
    }
  }

  for (const [vKey, vType] of Object.entries(cAudioFileTypes)) {
    if ((vType.extensions as readonly string[]).includes(cExtension)) {
      return `${FileTypeCategory.AUDIO}:${vKey}`;
    }
  }

  for (const [vKey, vType] of Object.entries(cVideoFileTypes)) {
    if ((vType.extensions as readonly string[]).includes(cExtension)) {
      return `${FileTypeCategory.VIDEO}:${vKey}`;
    }
  }

  for (const [vKey, vType] of Object.entries(cPdfFileTypes)) {
    if ((vType.extensions as readonly string[]).includes(cExtension)) {
      return `${FileTypeCategory.PDF}:${vKey}`;
    }
  }

  for (const [vKey, vType] of Object.entries(cTextFileTypes)) {
    if ((vType.extensions as readonly string[]).includes(cExtension)) {
      return `${FileTypeCategory.TEXT}:${vKey}`;
    }
  }

  return null;
}

export function fIsFileTypeSupported(pFilename: string, pMimeType?: string): boolean {
  // Images are detected and handled separately for vision models
  if (pMimeType) {
    const cCategory = fGetFileTypeCategory(pMimeType);

    if (
      cCategory === FileTypeCategory.IMAGE ||
      cCategory === FileTypeCategory.AUDIO ||
      cCategory === FileTypeCategory.PDF
    ) {
      return true;
    }
  }

  // Check extension for known types (especially images without MIME)
  const cExtCategory = fGetFileTypeCategoryByExtension(pFilename);

  if (
    cExtCategory === FileTypeCategory.IMAGE ||
    cExtCategory === FileTypeCategory.AUDIO ||
    cExtCategory === FileTypeCategory.PDF
  ) {
    return true;
  }

  // Fallback: treat everything else as text (inclusive by default)
  return true;
}
