cmake_minimum_required (VERSION 3.0.2)
project (darts VERSION 1.2 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(IS_ROOT_PROJECT ON)
else()
    set(IS_ROOT_PROJECT OFF)
endif()

#some self define function
message("find required package path....")
function(DARTS_PROTOBUF_GENERATE_CPP PATH SRCS HDRS)
  if(NOT ARGN)
    message(SEND_ERROR "Error: PROTOBUF_GENERATE_CPP() called without any proto files")
    return()
  endif()

  if(PROTOBUF_GENERATE_CPP_APPEND_PATH)
    # Create an include path for each file specified
    foreach(FIL ${ARGN})
      get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
      get_filename_component(ABS_PATH ${ABS_FIL} PATH)
      list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
      if(${_contains_already} EQUAL -1)
          list(APPEND _protobuf_include_path -I ${ABS_PATH})
      endif()
    endforeach()
  else()
    set(_protobuf_include_path -I ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  if(DEFINED PROTOBUF_IMPORT_DIRS)
    foreach(DIR ${PROTOBUF_IMPORT_DIRS})
      get_filename_component(ABS_PATH ${DIR} ABSOLUTE)
      list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
      if(${_contains_already} EQUAL -1)
          list(APPEND _protobuf_include_path -I ${ABS_PATH})
      endif()
    endforeach()
  endif()

  set(${SRCS})
  set(${HDRS})
  foreach(FIL ${ARGN})
    get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
    get_filename_component(FIL_WE ${FIL} NAME_WE)

    list(APPEND ${SRCS} "${CMAKE_CURRENT_BINARY_DIR}/${PATH}/${FIL_WE}.pb.cc")
    list(APPEND ${HDRS} "${CMAKE_CURRENT_BINARY_DIR}/${PATH}/${FIL_WE}.pb.h")

    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/${PATH})

    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${PATH}/${FIL_WE}.pb.cc"
             "${CMAKE_CURRENT_BINARY_DIR}/${PATH}/${FIL_WE}.pb.h"
      COMMAND  ${PROTOBUF_PROTOC_EXECUTABLE}
      ARGS --cpp_out  ${CMAKE_CURRENT_BINARY_DIR}/${PATH} ${_protobuf_include_path} ${ABS_FIL}
      DEPENDS ${ABS_FIL}
      COMMENT "Running C++ protocol buffer compiler on ${FIL}"
      VERBATIM )
  endforeach()

  set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
endfunction()


# search the package use
find_path(Jsoncpp_INCLUDE_DIR
  NAMES json/json.h
  PATH_SUFFIXES jsoncpp
  PATHS ${Jsoncpp_PKGCONF_INCLUDE_DIRS} # /usr/include/jsoncpp/json
)

find_library(Jsoncpp_LIBRARY
  NAMES jsoncpp
  PATHS ${Jsoncpp_PKGCONF_LIBRARY_DIRS}
)

find_package (Eigen3 REQUIRED NO_MODULE)
find_package(Protobuf REQUIRED)
find_package(HDF5 REQUIRED)
find_package(ZLIB REQUIRED)
file(GLOB ProtoFiles "${CMAKE_CURRENT_LIST_DIR}/src/utils/*.proto")
set(proto_gen_path proto_gen)
DARTS_PROTOBUF_GENERATE_CPP(${proto_gen_path} PROTO_SRCS PROTO_HDRS ${ProtoFiles})

message("-----------------include----path----------------")
message("proto src file: ${ProtoFiles}")
message("gen proto head file: ${PROTO_HDRS}")
message("jsoncpp path: ${Jsoncpp_INCLUDE_DIR}")
message("eigen3 path: ${EIGEN3_INCLUDE_DIR}")
message("hdf5 path: ${HDF5_INCLUDE_DIR}")
message("-----------------================")

#set bin ouput dir
add_compile_options(-fPIC)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/dist)

include_directories(src ${ZLIB_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${Jsoncpp_INCLUDE_DIR} ${PROTOBUF_INCLUDE_DIRS} "${CMAKE_CURRENT_BINARY_DIR}/${proto_gen_path}" ${HDF5_INCLUDE_DIR})
set(darts_libs Eigen3::Eigen ${Jsoncpp_LIBRARY} ${PROTOBUF_LIBRARY} ${HDF5_LIBRARY} ${ZLIB_LIBRARIES})

#build darts c libray 
add_library(cdarts STATIC src/main/darts.cxx)
target_link_libraries(cdarts PRIVATE ${darts_libs})
set_target_properties(cdarts PROPERTIES PUBLIC_HEADER "src/main/darts.h")
set_property(TARGET cdarts PROPERTY CXX_STANDARD 17)

#build darts
add_executable(darts ${PROTO_SRCS} src/main/main.cpp)
target_link_libraries(darts PRIVATE ${darts_libs})
set_property(TARGET darts PROPERTY CXX_STANDARD 17)

INSTALL(TARGETS cdarts darts)
INSTALL(DIRECTORY data DESTINATION .)
#build test
add_executable(test_darts ${PROTO_SRCS} test/test_func.cc)
target_link_libraries(test_darts PRIVATE ${darts_libs})
set_property(TARGET test_darts PROPERTY CXX_STANDARD 17)



