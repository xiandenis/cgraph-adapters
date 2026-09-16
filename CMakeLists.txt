cmake_minimum_required(VERSION 3.24)
project(cgraph_adapters VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(cgraph CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# --- gate/fixture (kernel-acceptance fixtures; not production) ---
add_library(cgraph_gate_fixture
  gate/fixture/src/const.cpp
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

# --- gate/cloud (gate stubs; not production domain ops) ---
add_library(cgraph_gate_cloud
  gate/cloud/src/register.cpp
  gate/cloud/src/cloud_value.cpp
  gate/cloud/src/ply_pcd_ascii.cpp
  gate/cloud/src/load.cpp
  gate/cloud/src/dump.cpp
  gate/cloud/src/voxel.cpp
  gate/cloud/src/geom/rigid.cpp
  gate/cloud/src/geom/knn.cpp
  gate/cloud/src/ops/normal_est.cpp
  gate/cloud/src/ops/fpfh.cpp
  gate/cloud/src/ops/shot.cpp
  gate/cloud/src/ops/match.cpp
  gate/cloud/src/ops/ransac.cpp
  gate/cloud/src/ops/icp_step.cpp
  gate/cloud/src/ops/filter/passthrough.cpp
  gate/cloud/src/ops/filter/stat_outlier.cpp
  gate/cloud/src/ops/sample/uniform_sampling.cpp
  gate/cloud/src/ops/segment/euclid_cluster.cpp
)
target_include_directories(cgraph_gate_cloud
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/gate/cloud/include>
)
target_link_libraries(cgraph_gate_cloud PUBLIC cgraph::cgraph Eigen3::Eigen)
if(MSVC)
  target_compile_options(cgraph_gate_cloud PUBLIC /utf-8)
endif()

add_library(cgraph_gate_cloud_plugin SHARED gate/cloud/src/plugin_register.cpp)
target_link_libraries(cgraph_gate_cloud_plugin PRIVATE cgraph_gate_cloud)
set_target_properties(cgraph_gate_cloud_plugin PROPERTIES
  OUTPUT_NAME cgraph_gate_cloud
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/operators"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/operators"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/operators"
)
foreach(_cfg IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  string(TOUPPER "${_cfg}" _cfgu)
  set_target_properties(cgraph_gate_cloud_plugin PROPERTIES
    "RUNTIME_OUTPUT_DIRECTORY_${_cfgu}" "${CMAKE_BINARY_DIR}/operators"
    "LIBRARY_OUTPUT_DIRECTORY_${_cfgu}" "${CMAKE_BINARY_DIR}/operators"
    "ARCHIVE_OUTPUT_DIRECTORY_${_cfgu}" "${CMAKE_BINARY_DIR}/operators"
  )
endforeach()
if(MSVC)
  target_compile_options(cgraph_gate_cloud_plugin PRIVATE /utf-8)
endif()

# Placeholder dirs for future work.
file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/native")
file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/vendor/rtr")

install(TARGETS cgraph_gate_fixture_plugin cgraph_gate_cloud_plugin
  RUNTIME DESTINATION operators
  LIBRARY DESTINATION operators
)
install(DIRECTORY gate/cloud/descriptions DESTINATION operators/cloud OPTIONAL)
install(DIRECTORY gate/cloud/usage DESTINATION operators/cloud OPTIONAL)
