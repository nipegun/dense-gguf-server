include(sources.cmake)
include("${PROJECT_SOURCE_DIR}/../deploy/scripts/ui-assets.cmake")

add_custom_target(dgs-ui-assets DEPENDS "${vUiCpp}" "${vUiHeader}")
add_library(dgs-ui STATIC "${vUiCpp}" "${vUiHeader}")
target_compile_features(dgs-ui PRIVATE cxx_std_17)
target_include_directories(dgs-ui PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
set_target_properties(dgs-ui PROPERTIES POSITION_INDEPENDENT_CODE ON)
