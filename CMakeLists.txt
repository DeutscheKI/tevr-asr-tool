cmake_minimum_required(VERSION 3.21)
project(tevr_asr_tool)

set(CMAKE_CXX_STANDARD 14)

add_executable(tevr_asr_tool main.cpp)


add_subdirectory(tensorflow_src/tensorflow/lite)
target_link_libraries(tevr_asr_tool tensorflow-lite)


include(kenlm/cmake/KenLMFunctions.cmake)
list(APPEND CMAKE_MODULE_PATH kenlm/cmake/modules)

find_package(Boost 1.41.0 REQUIRED COMPONENTS
        program_options
        system
        thread
        unit_test_framework
        )

include_directories(${Boost_INCLUDE_DIRS})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_subdirectory(kenlm/util)
add_subdirectory(kenlm/lm)

target_link_libraries(tevr_asr_tool kenlm)


install(TARGETS tevr_asr_tool)

set(CPACK_GENERATOR "DEB")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Hajo Nils Krabbenhöft")
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
set(CPACK_PACKAGE_DESCRIPTION "German speech recognition with wav2vec 2.0 XLS-R 1B enhanced with TEVR tokens")
include(CPack)
