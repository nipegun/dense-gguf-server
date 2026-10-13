# Inputs that invalidate the locally generated web application.
set(aUiSourceGlobs
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*"
  "${CMAKE_CURRENT_SOURCE_DIR}/static/*"
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/*"
)

set(aUiSourceFiles
  package.json
  package-lock.json
  vite.config.ts
  svelte.config.js
  tsconfig.json
  pwa-assets.config.ts
  pwa-assets-dark.config.ts
)
