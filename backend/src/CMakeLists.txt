dgs_add_compile_flags()

#
# libraries
#

# dgs

set(aDgsModelSources
  models/delta-net-base.cpp
  models/qwen35.cpp
)

set(aDgsCoreSources
  dgs.cpp
  dgs-adapter.cpp
  dgs-arch.cpp
  dgs-batch.cpp
  dgs-context.cpp
  dgs-cparams.cpp
  dgs-grammar.cpp
  dgs-graph.cpp
  dgs-hparams.cpp
  dgs-impl.cpp
  dgs-io.cpp
  dgs-kv-cache.cpp
  dgs-kv-cache-iswa.cpp
  dgs-kv-cache-dsa.cpp
  dgs-kv-cache-dsa-iswa.cpp
  dgs-kv-cache-msa.cpp
  dgs-kv-cache-dsv4.cpp
  dgs-memory.cpp
  dgs-memory-hybrid.cpp
  dgs-memory-hybrid-iswa.cpp
  dgs-memory-hybrid-idx.cpp
  dgs-memory-recurrent.cpp
  dgs-mmap.cpp
  dgs-model-loader.cpp
  dgs-model-policy.cpp
  dgs-model.cpp
  dgs-sampler.cpp
  dgs-vocab.cpp
  unicode-data.cpp
  unicode.cpp
)

add_library(dgs
      ../include/dgs.h
      ${aDgsCoreSources}
      unicode.h
      ${aDgsModelSources}
      )

set_target_properties(dgs PROPERTIES
  VERSION ${DGS_VERSION_BASE}
  SOVERSION ${DGS_VERSION_MAJOR}
  MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
  UNITY_BUILD OFF
)

configure_file(dgs-version.h.in ${CMAKE_CURRENT_BINARY_DIR}/dgs-version.h @ONLY)

target_include_directories(dgs PRIVATE . ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(dgs PUBLIC ../include)
target_compile_features   (dgs PRIVATE cxx_std_17) # don't bump

target_link_libraries(dgs PUBLIC ggml)

if (BUILD_SHARED_LIBS)
  set_target_properties(dgs PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(dgs PRIVATE DGS_BUILD)
  target_compile_definitions(dgs PUBLIC  DGS_SHARED)
endif()
