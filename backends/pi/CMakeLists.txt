cmake_minimum_required(VERSION 3.20)
project(mstar_pi_native VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

option(MSTAR_PI_GGML "Build the resident ggml Vulkan provider" ON)

find_package(Threads REQUIRED)
find_package(Vulkan REQUIRED)
find_path(DRM_INCLUDE_DIR xf86drm.h PATH_SUFFIXES libdrm REQUIRED)
find_path(DRM_UAPI_INCLUDE_DIR v3d_drm.h PATH_SUFFIXES libdrm drm REQUIRED)
find_library(DRM_LIBRARY NAMES drm libdrm.so.2 REQUIRED)
find_program(GLSLANG_VALIDATOR glslangValidator REQUIRED)

add_library(mstar_pi_native STATIC native/src/runtime.cpp)
set_target_properties(mstar_pi_native PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
target_include_directories(mstar_pi_native
    PUBLIC native/include
    PRIVATE "${DRM_INCLUDE_DIR}" "${DRM_UAPI_INCLUDE_DIR}")
target_link_libraries(mstar_pi_native PUBLIC Vulkan::Vulkan "${DRM_LIBRARY}" Threads::Threads)
target_compile_options(mstar_pi_native PRIVATE -Wall -Wextra -Wpedantic)

set(TRANSFER_SPV "${CMAKE_BINARY_DIR}/shaders/transfer.spv")
add_custom_command(OUTPUT "${TRANSFER_SPV}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/shaders"
    COMMAND "${GLSLANG_VALIDATOR}" -V --target-env vulkan1.1
            -o "${TRANSFER_SPV}" "${CMAKE_CURRENT_SOURCE_DIR}/native/shaders/transfer.comp"
    DEPENDS native/shaders/transfer.comp
    VERBATIM)
add_custom_target(mstar_pi_shaders ALL DEPENDS "${TRANSFER_SPV}")

if(MSTAR_PI_GGML)
    # The Torch extension links this exact provider, never a previously installed
    # ggml shared object. CPU execution and ggml's dynamic backend discovery are
    # outside this provider's contract.
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
    set(GGML_CPU OFF CACHE BOOL "" FORCE)
    set(GGML_BLAS OFF CACHE BOOL "" FORCE)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
    set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
    if(GGML_VULKAN_CHECK_RESULTS OR GGML_VULKAN_RUN_TESTS)
        message(FATAL_ERROR "Managed provider builds exclude ggml's host reference paths")
    endif()
    add_subdirectory(third_party/ggml)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(mstar_pi_stream_prefixes native/tests/stream_prefixes_test.cpp)
    target_include_directories(mstar_pi_stream_prefixes PRIVATE native/include native/src)
    add_test(NAME native_stream_prefixes COMMAND mstar_pi_stream_prefixes)

    add_executable(mstar_pi_preflight native/tests/preflight_test.cpp)
    target_link_libraries(mstar_pi_preflight PRIVATE mstar_pi_native)
    add_test(NAME native_preflight COMMAND mstar_pi_preflight)

    if(MSTAR_PI_GGML)
        find_package(SPIRV-Headers CONFIG REQUIRED)
        add_executable(mstar_pi_memory_logger native/tests/ggml_memory_logger_test.cpp)
        target_include_directories(mstar_pi_memory_logger PRIVATE
            third_party/ggml/include third_party/ggml/src third_party/ggml/src/ggml-vulkan
            "${CMAKE_CURRENT_BINARY_DIR}/third_party/ggml/src/ggml-vulkan")
        target_link_libraries(mstar_pi_memory_logger PRIVATE
            ggml-vulkan Vulkan::Vulkan SPIRV-Headers::SPIRV-Headers)
        add_test(NAME native_shared_import_logging COMMAND mstar_pi_memory_logger)
    endif()
endif()

install(TARGETS mstar_pi_native ARCHIVE DESTINATION lib)
install(DIRECTORY native/include/ DESTINATION include)
install(FILES "${TRANSFER_SPV}" DESTINATION share/mstar_pi)
