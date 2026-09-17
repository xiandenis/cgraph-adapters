# RTR Config does find_dependency(Eigen3 3.3). Eigen 5.x rejects that version.
# Adapters already imported Eigen3::Eigen; this redirect only satisfies the version check.
if(NOT TARGET Eigen3::Eigen)
  message(FATAL_ERROR "rtr-dep-shim Eigen3Config: Eigen3::Eigen is not defined")
endif()
set(Eigen3_FOUND TRUE)
