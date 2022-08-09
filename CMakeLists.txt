cmake_minimum_required(VERSION 3.21)
project(tevr_asr_tool)

set(CMAKE_CXX_STANDARD 14)

add_executable(tevr_asr_tool tevr_asr_tool.cc op_erf.cc)

add_subdirectory(tensorflow_src/tensorflow/lite)

target_link_libraries(tevr_asr_tool tensorflow-lite absl::flags absl::flags_parse)

find_package(Threads)
macro(install)
endmacro()
macro(find_package)
endmacro()

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


add_library(wave
    ${PROJECT_SOURCE_DIR}/wave/src/wave/header/data_header.h
    ${PROJECT_SOURCE_DIR}/wave/src/wave/header/data_header.cc
    ${PROJECT_SOURCE_DIR}/wave/src/wave/header/riff_header.h
    ${PROJECT_SOURCE_DIR}/wave/src/wave/header/riff_header.cc
    ${PROJECT_SOURCE_DIR}/wave/src/wave/header/fmt_header.h
    ${PROJECT_SOURCE_DIR}/wave/src/wave/header/fmt_header.cc
    ${PROJECT_SOURCE_DIR}/wave/src/wave/header/wave_header.h
    ${PROJECT_SOURCE_DIR}/wave/src/wave/header/wave_header.cc

    ${PROJECT_SOURCE_DIR}/wave/src/wave/header.h
    ${PROJECT_SOURCE_DIR}/wave/src/wave/header.cc
    ${PROJECT_SOURCE_DIR}/wave/src/wave/header_list.h
    ${PROJECT_SOURCE_DIR}/wave/src/wave/header_list.cc

    ${PROJECT_SOURCE_DIR}/wave/src/wave/error.h
    ${PROJECT_SOURCE_DIR}/wave/src/wave/file.h
    ${PROJECT_SOURCE_DIR}/wave/src/wave/file.cc
)
target_include_directories(wave PUBLIC ${PROJECT_SOURCE_DIR}/wave/src)

target_link_libraries(tevr_asr_tool wave)


_install(TARGETS tevr_asr_tool)
_install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tevr-asr-data/" DESTINATION "/usr/share/tevr_asr_tool")
set(CPACK_GENERATOR "DEB")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Hajo Nils Krabbenhöft")
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS OFF)
set(CPACK_PACKAGE_DESCRIPTION "German speech recognition with wav2vec 2.0 XLS-R 1B enhanced with TEVR tokens" )
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6 (>= 2.27), libgcc1 (>= 1:4.7), libstdc++6 (>= 6)" )
include(CPack)
