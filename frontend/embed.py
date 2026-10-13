#!/usr/bin/env -S PYTHONDONTWRITEBYTECODE=1 python3

"""Embed the locally built web application without downloading upstream assets."""

import gzip
import hashlib
import json
import mimetypes
from pathlib import Path
import sys

def fMain():
  vSource, vOutput = (Path(vPath) for vPath in sys.argv[1:3])
  vEnabled = sys.argv[3].upper() == 'ON'
  aFiles = sorted(vPath for vPath in (vSource / 'dist').rglob('*') if vPath.is_file()) if vEnabled else []
  if vEnabled and not (vSource / 'dist/index.html').is_file():
    raise RuntimeError('Build frontend/dist/index.html before embedding the web application')
  aArrays = []
  aTable = []
  for vIndex, vPath in enumerate(aFiles):
    vName = vPath.relative_to(vSource / 'dist').as_posix()
    vData = gzip.compress(vPath.read_bytes(), mtime=0)
    vSymbol = f'aAsset{vIndex}'
    aArrays.append(f'static const unsigned char {vSymbol}[] = {{' + ','.join(str(vByte) for vByte in vData) + '};')
    vMime = {'.js': 'application/javascript', '.webmanifest': 'application/manifest+json'}.get(vPath.suffix, mimetypes.guess_type(vName)[0] or 'application/octet-stream')
    vEtag = '"' + hashlib.sha256(vData).hexdigest() + '"'
    aTable.append('  {' + f'{json.dumps(vName)}, {vSymbol}, sizeof({vSymbol}), {json.dumps(vEtag)}, {json.dumps(vMime)}' + '},')
  dValues = {'N_ASSETS': str(len(aFiles)), 'ASSET_ARRAYS': '\n'.join(aArrays), 'ASSET_TABLE': '\n'.join(aTable), 'USE_GZIP': 'true'}
  vOutput.mkdir(parents=True, exist_ok=True)
  for vName in ['ui.cpp', 'ui.h']:
    vText = (vSource / (vName + '.in')).read_text()
    for vKey, vValue in dValues.items():
      vText = vText.replace('@' + vKey + '@', vValue)
    vText = vText.replace('#cmakedefine DGS_UI_HAS_ASSETS 1', '#define DGS_UI_HAS_ASSETS 1' if aFiles else '// No embedded assets')
    vTarget = vOutput / vName
    if not vTarget.exists() or vTarget.read_text() != vText:
      vTarget.write_text(vText)

if __name__ == '__main__':
  try:
    fMain()
  except Exception as vError:
    print(f'Cannot embed DGS web assets: {vError}', file=sys.stderr)
    sys.exit(1)
