cmake_minimum_required (VERSION 3.0.2)
project (darts VERSION 1.2 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(IS_ROOT_PROJECT ON)
else()
    set(IS_ROOT_PROJECT OFF)
endif()



set(XTENSOR_USE_XSIMD 0)
set(XTENSOR_USE_TBB 0)
# xsimd and TBB dependencies are automatically
# searched when the following is executed
find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
find_package(utf8cpp REQUIRED)
add_compile_options(-fPIC)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(src ${xtensor_INCLUDE_DIRS} ${utf8cpp_INCLUDE_DIR})


add_executable(darts src/main/main.cpp)
target_link_libraries(darts utf8cpp xtensor xtensor::optimize)
set_property(TARGET darts PROPERTY CXX_STANDARD 17)

# add_library(ocr SHARED src/main/darts4py.cpp)
# target_link_libraries(ocr obase  "${OpenCV_LIBS}" "${TORCH_LIBRARIES}" "${CUDA_LIBRARIES}")
# set_property(TARGET ocr PROPERTY CXX_STANDARD 11)


