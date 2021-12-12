cmake_minimum_required (VERSION 3.0.2)
project (darts VERSION 3.2 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
find_package(utf8cpp REQUIRED)


if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(IS_ROOT_PROJECT ON)
else()
    set(IS_ROOT_PROJECT OFF)
endif()

