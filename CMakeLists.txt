cmake_minimum_required(VERSION 3.21)
project(tevr_asr_tool)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(tensorflow_src/tensorflow/lite)

add_executable(tevr_asr_tool main.cpp)
target_link_libraries(tevr_asr_tool tensorflow-lite)

install(TARGETS tevr_asr_tool)

SET(CPACK_GENERATOR "DEB")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Hajo Nils Krabbenhöft")
INCLUDE(CPack)
