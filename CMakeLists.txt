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

SET(CPACK_GENERATOR "DEB")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Hajo Nils Krabbenhöft")
INCLUDE(CPack)
