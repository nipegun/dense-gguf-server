/**
 * Comprehensive dictionary of all supported file types in dgs-ui
 * Organized by category with TypeScript enums for better type safety
 */

import {
  FileExtensionAudio,
  FileExtensionImage,
  FileExtensionPdf,
  FileExtensionText,
  FileTypeAudio,
  FileTypeImage,
  FileTypePdf,
  FileTypeText,
  MimeTypeApplication,
  MimeTypeAudio,
  MimeTypeImage,
  MimeTypeText,
  MimeTypeVideo
} from '$lib/enums';
import { FileExtensionVideo, FileTypeVideo } from '$lib/enums/files.enums';

// File type configuration using enums
export const cAudioFileTypes = {
  [FileTypeAudio.cMp3]: {
    extensions: [FileExtensionAudio.cMp3],
    mimeTypes: [MimeTypeAudio.cMp3Mpeg, MimeTypeAudio.cMp3]
  },
  [FileTypeAudio.cWav]: {
    extensions: [FileExtensionAudio.cWav],
    mimeTypes: [MimeTypeAudio.cWav]
  }
} as const;

export const cVideoFileTypes = {
  [FileTypeVideo.cMp4]: {
    extensions: [FileExtensionVideo.cMp4],
    mimeTypes: [MimeTypeVideo.cMp4]
  },
  [FileTypeVideo.cOgg]: {
    extensions: [FileExtensionVideo.cOgg],
    mimeTypes: [MimeTypeVideo.cOgg]
  },
  [FileTypeVideo.cWebm]: {
    extensions: [FileExtensionVideo.cWebm],
    mimeTypes: [MimeTypeVideo.cWebm]
  }
} as const;

export const cImageFileTypes = {
  [FileTypeImage.cGif]: {
    extensions: [FileExtensionImage.cGif],
    mimeTypes: [MimeTypeImage.cGif]
  },
  [FileTypeImage.cHeic]: {
    extensions: [FileExtensionImage.cHeic, FileExtensionImage.cHeif],
    mimeTypes: [MimeTypeImage.cHeic, MimeTypeImage.cHeif]
  },
  [FileTypeImage.cJpeg]: {
    extensions: [FileExtensionImage.cJpg, FileExtensionImage.cJpeg],
    mimeTypes: [MimeTypeImage.cJpeg]
  },
  [FileTypeImage.cPng]: {
    extensions: [FileExtensionImage.cPng],
    mimeTypes: [MimeTypeImage.cPng]
  },
  [FileTypeImage.SVG]: {
    extensions: [FileExtensionImage.SVG],
    mimeTypes: [MimeTypeImage.SVG]
  },
  [FileTypeImage.cWebp]: {
    extensions: [FileExtensionImage.cWebp],
    mimeTypes: [MimeTypeImage.cWebp]
  }
} as const;

export const cPdfFileTypes = {
  [FileTypePdf.PDF]: {
    extensions: [FileExtensionPdf.PDF],
    mimeTypes: [MimeTypeApplication.PDF]
  }
} as const;

export const cTextFileTypes = {
  [FileTypeText.cAsciidoc]: {
    extensions: [FileExtensionText.cAdoc],
    mimeTypes: [MimeTypeText.cAsciidoc]
  },
  [FileTypeText.cBibtex]: {
    extensions: [FileExtensionText.cBib],
    mimeTypes: [MimeTypeText.cBibtex]
  },
  [FileTypeText.cCpp]: {
    extensions: [
      FileExtensionText.cCpp,
      FileExtensionText.C,
      FileExtensionText.cH,
      FileExtensionText.cHpp
    ],
    mimeTypes: [MimeTypeText.cCppSrc, MimeTypeText.cCppHdr, MimeTypeText.cCSrc, MimeTypeText.cCHdr]
  },
  [FileTypeText.cCsharp]: {
    extensions: [FileExtensionText.cCs],
    mimeTypes: [MimeTypeText.cCsharp]
  },
  [FileTypeText.cCss]: {
    extensions: [FileExtensionText.cCss],
    mimeTypes: [MimeTypeText.cCss]
  },
  [FileTypeText.cCsv]: {
    extensions: [FileExtensionText.cCsv],
    mimeTypes: [MimeTypeText.cCsv]
  },
  [FileTypeText.cCuda]: {
    extensions: [FileExtensionText.cCu, FileExtensionText.cCuh],
    mimeTypes: [MimeTypeText.cCuda]
  },
  [FileTypeText.cDart]: {
    extensions: [FileExtensionText.cDart],
    mimeTypes: [MimeTypeText.cDart]
  },
  [FileTypeText.cGo]: {
    extensions: [FileExtensionText.cGo],
    mimeTypes: [MimeTypeText.cGo]
  },
  [FileTypeText.cHaskell]: {
    extensions: [FileExtensionText.cHs],
    mimeTypes: [MimeTypeText.cHaskell]
  },
  [FileTypeText.HTML]: {
    extensions: [FileExtensionText.HTML, FileExtensionText.cHtm],
    mimeTypes: [MimeTypeText.HTML]
  },
  [FileTypeText.cJava]: {
    extensions: [FileExtensionText.cJava],
    mimeTypes: [MimeTypeText.cJava]
  },
  [FileTypeText.cJavascript]: {
    extensions: [FileExtensionText.cJs],
    mimeTypes: [MimeTypeText.cJavascript, MimeTypeText.cJavascriptApp]
  },
  [FileTypeText.cJson]: {
    extensions: [FileExtensionText.cJson],
    mimeTypes: [MimeTypeText.cJson]
  },
  [FileTypeText.cJsx]: {
    extensions: [FileExtensionText.cJsx],
    mimeTypes: [MimeTypeText.cJsx]
  },
  [FileTypeText.cKotlin]: {
    extensions: [FileExtensionText.cKt],
    mimeTypes: [MimeTypeText.cKotlin]
  },
  [FileTypeText.cLatex]: {
    extensions: [FileExtensionText.cTex],
    mimeTypes: [MimeTypeText.cLatex, MimeTypeText.cTex, MimeTypeText.cTexApp]
  },
  [FileTypeText.cLog]: {
    extensions: [FileExtensionText.cLog],
    mimeTypes: [MimeTypeText.cPlain]
  },
  [FileTypeText.MARKDOWN]: {
    extensions: [FileExtensionText.cMd],
    mimeTypes: [MimeTypeText.MARKDOWN]
  },
  [FileTypeText.cPhp]: {
    extensions: [FileExtensionText.cPhp],
    mimeTypes: [MimeTypeText.cPhp]
  },
  [FileTypeText.cPlainText]: {
    extensions: [FileExtensionText.cTxt],
    mimeTypes: [MimeTypeText.cPlain]
  },
  [FileTypeText.cProperties]: {
    extensions: [FileExtensionText.cProperties],
    mimeTypes: [MimeTypeText.cProperties]
  },
  [FileTypeText.cPython]: {
    extensions: [FileExtensionText.cPy],
    mimeTypes: [MimeTypeText.cPython]
  },
  [FileTypeText.R]: {
    extensions: [FileExtensionText.R],
    mimeTypes: [MimeTypeText.R]
  },
  [FileTypeText.cRuby]: {
    extensions: [FileExtensionText.cRb],
    mimeTypes: [MimeTypeText.cRuby]
  },
  [FileTypeText.cRust]: {
    extensions: [FileExtensionText.cRs],
    mimeTypes: [MimeTypeText.cRust]
  },
  [FileTypeText.cScala]: {
    extensions: [FileExtensionText.cScala],
    mimeTypes: [MimeTypeText.cScala]
  },
  [FileTypeText.cShell]: {
    extensions: [FileExtensionText.cSh, FileExtensionText.cBat],
    mimeTypes: [MimeTypeText.cShell, MimeTypeText.cBat]
  },
  [FileTypeText.cSql]: {
    extensions: [FileExtensionText.cSql],
    mimeTypes: [MimeTypeText.cSql]
  },
  [FileTypeText.cSvelte]: {
    extensions: [FileExtensionText.cSvelte],
    mimeTypes: [MimeTypeText.cSvelte]
  },
  [FileTypeText.cSwift]: {
    extensions: [FileExtensionText.cSwift],
    mimeTypes: [MimeTypeText.cSwift]
  },
  [FileTypeText.cTsx]: {
    extensions: [FileExtensionText.cTsx],
    mimeTypes: [MimeTypeText.cTsx]
  },
  [FileTypeText.cTypescript]: {
    extensions: [FileExtensionText.cTs],
    mimeTypes: [MimeTypeText.cTypescript]
  },
  [FileTypeText.cVue]: {
    extensions: [FileExtensionText.cVue],
    mimeTypes: [MimeTypeText.cVue]
  },
  [FileTypeText.cVulkan]: {
    extensions: [FileExtensionText.cComp],
    mimeTypes: [MimeTypeText.cPlain]
  },
  [FileTypeText.cXml]: {
    extensions: [FileExtensionText.cXml],
    mimeTypes: [MimeTypeText.cXmlText, MimeTypeText.cXmlApp]
  },
  [FileTypeText.cYaml]: {
    extensions: [FileExtensionText.cYaml, FileExtensionText.cYml],
    mimeTypes: [MimeTypeText.cYamlText, MimeTypeText.cYamlApp]
  }
} as const;
