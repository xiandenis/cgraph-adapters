cmake_minimum_required(VERSION 3.24)
project(cgraph_adapters VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Stale Eigen3 redirects from a previous RTR configure would recurse on
# find_package(Eigen3). Remove them before the real Eigen find.
if(DEFINED CMAKE_FIND_PACKAGE_REDIRECTS_DIR)
  file(REMOVE
    "${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/Eigen3Config.cmake"
    "${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/Eigen3ConfigVersion.cmake"
    "${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/eigen3-config.cmake"
    "${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}/eigen3-config-version.cmake"
  )
endif()
unset(Eigen3_DIR CACHE)
unset(Eigen3_DIR)

find_package(cgraph CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# --- gate/fixture (kernel-acceptance fixtures; not production) ---
add_library(cgraph_gate_fixture
  gate/fixture/src/fx_types.cpp
  gate/fixture/src/const.cpp
  gate/fixture/src/const_scalar.cpp
  gate/fixture/src/echo.cpp
  gate/fixture/src/probe.cpp
  gate/fixture/src/dup.cpp
  gate/fixture/src/scale.cpp
  gate/fixture/src/stamp.cpp
  gate/fixture/src/abs.cpp
  gate/fixture/src/add.cpp
  gate/fixture/src/math_print.cpp
  gate/fixture/src/inspect.cpp
  gate/fixture/src/math_scalar.cpp
  gate/fixture/src/math_branch.cpp
  gate/fixture/src/math_linalg.cpp
  gate/fixture/src/math_svd.cpp
  gate/fixture/src/math_solve.cpp
  gate/fixture/src/math_gd.cpp
  gate/fixture/src/math_map.cpp
  gate/fixture/src/math_artifact.cpp
  gate/fixture/src/dump_json.cpp
  gate/fixture/src/mem_blob.cpp
  gate/fixture/src/stamp_proc.cpp
  gate/fixture/src/cat.cpp
  gate/fixture/src/sneak.cpp
  gate/fixture/src/foo.cpp
  gate/fixture/src/box.cpp
  gate/fixture/src/conv.cpp
  gate/fixture/src/coarse_reg.cpp
  gate/fixture/src/rand.cpp
  gate/fixture/src/constrained.cpp
  gate/fixture/src/bad_cost.cpp
  gate/fixture/src/gpu_hold.cpp
  gate/fixture/src/slow_proc.cpp
  gate/fixture/src/ram_gate.cpp
  gate/fixture/src/sleep_ms.cpp
  gate/fixture/src/subproc_echo.cpp
  gate/fixture/src/subproc_flaky.cpp
  gate/fixture/src/subproc_chunk.cpp
  gate/fixture/src/subproc_sneak.cpp
  gate/fixture/src/photogram.cpp
  gate/fixture/src/register.cpp
)
target_include_directories(cgraph_gate_fixture
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/gate/fixture/include>
)
target_link_libraries(cgraph_gate_fixture PUBLIC cgraph::cgraph Eigen3::Eigen)
if(MSVC)
  target_compile_options(cgraph_gate_fixture PUBLIC /utf-8)
endif()

add_library(cgraph_gate_fixture_plugin SHARED gate/fixture/src/plugin_register.cpp)
target_link_libraries(cgraph_gate_fixture_plugin PRIVATE cgraph_gate_fixture)
# Keep DLL name cgraph_gate_fixture; put import .lib under operators/ so it
# does not collide with the static cgraph_gate_fixture.lib in Release/.
set_target_properties(cgraph_gate_fixture_plugin PROPERTIES
  OUTPUT_NAME cgraph_gate_fixture
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/operators"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/operators"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/operators"
)
foreach(_cfg IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  string(TOUPPER "${_cfg}" _cfgu)
  set_target_properties(cgraph_gate_fixture_plugin PROPERTIES
    "RUNTIME_OUTPUT_DIRECTORY_${_cfgu}" "${CMAKE_BINARY_DIR}/operators"
    "LIBRARY_OUTPUT_DIRECTORY_${_cfgu}" "${CMAKE_BINARY_DIR}/operators"
    "ARCHIVE_OUTPUT_DIRECTORY_${_cfgu}" "${CMAKE_BINARY_DIR}/operators"
  )
endforeach()
if(MSVC)
  target_compile_options(cgraph_gate_fixture_plugin PRIVATE /utf-8)
endif()

add_custom_command(TARGET cgraph_gate_fixture_plugin POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
    "${CMAKE_SOURCE_DIR}/gate/fixture/catalog/fx-types.yaml"
    "$<TARGET_FILE_DIR:cgraph_gate_fixture_plugin>/fx-types.yaml"
)
install(FILES "${CMAKE_SOURCE_DIR}/gate/fixture/catalog/fx-types.yaml"
  DESTINATION operators
)

file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/native")

install(TARGETS cgraph_gate_fixture_plugin
  RUNTIME DESTINATION operators
  LIBRARY DESTINATION operators
)

enable_testing()
function(cgraph_fixture_add_test name)
  add_executable(${name} ${ARGN})
  target_include_directories(${name} PRIVATE
    ${CMAKE_SOURCE_DIR}/gate/fixture/src
    ${CMAKE_SOURCE_DIR}/gate/fixture/include
    ${CMAKE_SOURCE_DIR}/vendor/rtr/tests
  )
  target_link_libraries(${name} PRIVATE cgraph_gate_fixture)
  if(MSVC)
    target_compile_options(${name} PRIVATE /utf-8)
  endif()
  add_test(NAME ${name} COMMAND ${name})
endfunction()
cgraph_fixture_add_test(cgraph_fixture_test_fx_typed
  gate/fixture/tests/test_fx_typed.cpp)
cgraph_fixture_add_test(cgraph_fixture_test_math_typed_wire
  gate/fixture/tests/test_math_typed_wire.cpp)
cgraph_fixture_add_test(cgraph_fixture_test_math_case01
  gate/fixture/tests/test_math_case01.cpp)
cgraph_fixture_add_test(cgraph_fixture_test_math_cases
  gate/fixture/tests/test_math_cases_02_05.cpp)
cgraph_fixture_add_test(cgraph_fixture_test_math_cases_wave2
  gate/fixture/tests/test_math_cases_06_10.cpp)

option(CGRAPH_ADAPTERS_WITH_RTR "Wrap RTR leaf algorithms" ON)
if(CGRAPH_ADAPTERS_WITH_RTR)
  # RTR Config: find_dependency(Eigen3 3.3 NO_MODULE). Eigen 5.x ConfigVersion
  # uses SameMajorVersion and rejects that request. Drop a redirect into
  # CMake's pkgRedirects dir (do not replace CMAKE_FIND_PACKAGE_REDIRECTS_DIR).
  file(COPY
    "${CMAKE_SOURCE_DIR}/cmake/rtr-dep-shim/Eigen3Config.cmake"
    "${CMAKE_SOURCE_DIR}/cmake/rtr-dep-shim/Eigen3ConfigVersion.cmake"
    DESTINATION "${CMAKE_FIND_PACKAGE_REDIRECTS_DIR}"
  )
  find_package(real_time_registration CONFIG REQUIRED)
  add_library(cgraph_rtr_ops
    vendor/rtr/src/json.cpp
    vendor/rtr/src/align_result_unpack.cpp
    vendor/rtr/src/rough_global_reg.cpp
    vendor/rtr/src/fine_registration.cpp
    vendor/rtr/src/point_cloud_load.cpp
    vendor/rtr/src/registration_initializer.cpp
    vendor/rtr/src/registration_report.cpp
  )
  target_include_directories(cgraph_rtr_ops PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/vendor/rtr/include>
  )
  target_link_libraries(cgraph_rtr_ops PUBLIC
    cgraph::cgraph
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    real_time_registration::rough_global_reg
    real_time_registration::fine_registration
    real_time_registration::point_cloud_io
    real_time_registration::registration_initializer
    real_time_registration::registration_report
  )
  if(MSVC)
    target_compile_options(cgraph_rtr_ops PUBLIC /utf-8)
  endif()

  add_library(cgraph_rtr_plugin SHARED vendor/rtr/src/plugin_register.cpp)
  target_link_libraries(cgraph_rtr_plugin PRIVATE cgraph_rtr_ops)
  set_target_properties(cgraph_rtr_plugin PROPERTIES
    OUTPUT_NAME cgraph_rtr
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/operators"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/operators"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/operators"
  )
  foreach(_cfg IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    string(TOUPPER "${_cfg}" _cfgu)
    set_target_properties(cgraph_rtr_plugin PROPERTIES
      "RUNTIME_OUTPUT_DIRECTORY_${_cfgu}" "${CMAKE_BINARY_DIR}/operators"
      "LIBRARY_OUTPUT_DIRECTORY_${_cfgu}" "${CMAKE_BINARY_DIR}/operators"
      "ARCHIVE_OUTPUT_DIRECTORY_${_cfgu}" "${CMAKE_BINARY_DIR}/operators"
    )
  endforeach()
  if(MSVC)
    target_compile_options(cgraph_rtr_plugin PRIVATE /utf-8)
  endif()
  # Type Pack sidecar beside the DLL (Type Catalog §5). Kernel Process scan
  # skips cgraph-types.yaml / pack_id documents; do not treat this as a Process.
  add_custom_command(TARGET cgraph_rtr_plugin POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      "${CMAKE_SOURCE_DIR}/vendor/rtr/catalog/cgraph-types.yaml"
      "$<TARGET_FILE_DIR:cgraph_rtr_plugin>/cgraph-types.yaml"
  )
  install(FILES "${CMAKE_SOURCE_DIR}/vendor/rtr/catalog/cgraph-types.yaml"
    DESTINATION operators
  )
  install(TARGETS cgraph_rtr_plugin
    RUNTIME DESTINATION operators
    LIBRARY DESTINATION operators
  )

  enable_testing()
  function(cgraph_rtr_add_test name)
    add_executable(${name} ${ARGN})
    target_include_directories(${name} PRIVATE
      ${CMAKE_SOURCE_DIR}/vendor/rtr/tests
    )
    target_link_libraries(${name} PRIVATE cgraph_rtr_ops)
    add_test(NAME ${name} COMMAND ${name})
  endfunction()
  cgraph_rtr_add_test(cgraph_rtr_test_json vendor/rtr/tests/test_json.cpp)
  cgraph_rtr_add_test(cgraph_rtr_test_unpack vendor/rtr/tests/test_unpack.cpp)
  cgraph_rtr_add_test(cgraph_rtr_test_rough_fine vendor/rtr/tests/test_rough_fine.cpp)
  cgraph_rtr_add_test(cgraph_rtr_test_mainchain vendor/rtr/tests/test_mainchain.cpp)

  add_executable(cgraph_rtr_test_wire vendor/rtr/tests/test_wire.cpp)
  target_include_directories(cgraph_rtr_test_wire PRIVATE
    ${CMAKE_SOURCE_DIR}/vendor/rtr/tests
    ${CMAKE_SOURCE_DIR}/gate/fixture/include
  )
  target_link_libraries(cgraph_rtr_test_wire PRIVATE cgraph_rtr_ops cgraph_gate_fixture)
  add_test(NAME cgraph_rtr_test_wire COMMAND cgraph_rtr_test_wire)

  add_executable(cgraph_rtr_test_plugin_smoke vendor/rtr/tests/test_plugin_smoke.cpp)
  target_include_directories(cgraph_rtr_test_plugin_smoke PRIVATE
    ${CMAKE_SOURCE_DIR}/vendor/rtr/tests
  )
  target_link_libraries(cgraph_rtr_test_plugin_smoke PRIVATE cgraph::cgraph_load_operators)
  add_dependencies(cgraph_rtr_test_plugin_smoke cgraph_rtr_plugin)
  add_test(NAME cgraph_rtr_test_plugin_smoke
    COMMAND cgraph_rtr_test_plugin_smoke "$<TARGET_FILE:cgraph_rtr_plugin>"
  )
endif()
