cmake_minimum_required(VERSION 3.16)
project(volume_detector CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# CGAL ships a config package (CGALConfig.cmake) that also wires up GMP/MPFR.
find_package(CGAL REQUIRED)

# The grid/visibility detector source lives in this directory.
add_executable(volume_detector ${CMAKE_CURRENT_LIST_DIR}/volume_detector.cpp)
target_link_libraries(volume_detector PRIVATE CGAL::CGAL)

# Best-fit-plane triangulation uses CGAL's Eigen diagonalize traits.
# Some Eigen3 installs set a *_FOUND variable without creating the imported
# target (module vs config package), so guard on the target itself.
find_package(Eigen3 QUIET)
if(TARGET Eigen3::Eigen)
    target_link_libraries(volume_detector PRIVATE Eigen3::Eigen)
elseif(DEFINED EIGEN3_INCLUDE_DIR)
    target_include_directories(volume_detector PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()
