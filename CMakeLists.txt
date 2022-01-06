list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
cmake_minimum_required (VERSION 3.0.2)
project (darts VERSION 1.2 LANGUAGES C CXX)
#set complier
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-fPIC -c -O2)

if (APPLE)
    add_compile_options(-mmacosx-version-min=11.0 )
	set (CMAKE_CXX_COMPILER clang++)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -stdlib=libc++") 
    set (CMAKE_C_COMPILER clang)
endif()
message("Use C compiler is [${CMAKE_C_COMPILER}] and CXX compiler is [${CMAKE_CXX_COMPILER}]")
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
find_package(Jsoncpp REQUIRED)
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
message("--------------- lib--------path-----------------------")
message("${Jsoncpp_LIBRARY}")
message("${Protobuf_LIBRARIES}")
message("${HDF5_LIBRARIES}")
message("${ZLIB_LIBRARIES}")

#set bin ouput dir

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/dist)

include_directories(src ${ZLIB_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${Jsoncpp_INCLUDE_DIR} ${PROTOBUF_INCLUDE_DIRS} "${CMAKE_CURRENT_BINARY_DIR}/${proto_gen_path}" ${HDF5_INCLUDE_DIR})
set(darts_libs ${Jsoncpp_LIBRARY} ${Protobuf_LIBRARIES} ${HDF5_LIBRARIES} ${ZLIB_LIBRARIES})


#build darts
add_executable(darts ${PROTO_SRCS} src/main/main.cpp)
target_link_libraries(darts PRIVATE ${darts_libs})
set_property(TARGET darts PROPERTY CXX_STANDARD 17)


#build darts c libray 
add_library(cdarts STATIC src/main/darts.cxx)
target_link_libraries(cdarts PRIVATE ${darts_libs})
set_target_properties(cdarts PROPERTIES PUBLIC_HEADER "src/main/darts.h")
set_property(TARGET cdarts PROPERTY CXX_STANDARD 17)

INSTALL(TARGETS cdarts darts)
INSTALL(DIRECTORY data DESTINATION .)
#build test
add_executable(test_darts ${PROTO_SRCS} test/test_func.cc)
target_link_libraries(test_darts PRIVATE ${darts_libs})
set_property(TARGET test_darts PROPERTY CXX_STANDARD 17)



