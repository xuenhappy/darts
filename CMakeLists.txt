cmake_minimum_required (VERSION 3.0.2)
project (darts VERSION 1.2 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(IS_ROOT_PROJECT ON)
else()
    set(IS_ROOT_PROJECT OFF)
endif()


find_path(Jsoncpp_INCLUDE_DIR
  NAMES json/json.h
  PATH_SUFFIXES jsoncpp
  PATHS ${Jsoncpp_PKGCONF_INCLUDE_DIRS} # /usr/include/jsoncpp/json
)
# find_path(utf8cpp_INCLUDE_DIR
#   NAMES utf8.h
#   PATH_SUFFIXES utf8cpp
#   DOC "utf8cpp include directory"
# )
# mark_as_advanced(utf8cpp_INCLUDE_DIR)
# include(FindPackageHandleStandardArgs)
# find_package_handle_standard_args(utf8cpp REQUIRED_VARS utf8cpp_INCLUDE_DIR)
# if (utf8cpp_FOUND)
#   set(utf8cpp_INCLUDE_DIRS "${utf8cpp_INCLUDE_DIR}")
#   if (NOT TARGET utf8cpp::utf8cpp)
#     add_library(utf8cpp::utf8cpp INTERFACE IMPORTED)
#     set_target_properties(utf8cpp::utf8cpp PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${utf8cpp_INCLUDE_DIR}")
#   endif ()
# endif ()

# xsimd and TBB dependencies are automatically
# searched when the following is executed
find_package (Eigen3 REQUIRED NO_MODULE)
message("${EIGEN3_INCLUDE_DIR}")
message("${Jsoncpp_INCLUDE_DIR}")


add_compile_options(-fPIC)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(src ${EIGEN3_INCLUDE_DIR} ${Jsoncpp_INCLUDE_DIR})

add_executable(darts src/main/main.cpp)
target_link_libraries(darts Eigen3::Eigen ${Jsoncpp_LIBRARY})
set_property(TARGET darts PROPERTY CXX_STANDARD 17)

# add_library(ocr SHARED src/main/darts4py.cpp)
# target_link_libraries(ocr obase  "${OpenCV_LIBS}" "${TORCH_LIBRARIES}" "${CUDA_LIBRARIES}")
# set_property(TARGET ocr PROPERTY CXX_STANDARD 11)


