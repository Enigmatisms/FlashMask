# CP Balance bridge: standalone sub-CMake.
#
# Invoked by setup.py via subprocess (cmake configure + build). Compiles the
# framework-neutral cp_balance load-balancing kernels + extern "C" surface into
# one shared lib (libflashmask_cpbalance.so), driven from Python by ctypes.
# No framework headers: tensors cross as raw device pointers.
#
# 3.18 is the floor: CUDA_ARCHITECTURES accepting arch strings like "90a"/"100a"
# landed in 3.18.
cmake_minimum_required(VERSION 3.18)
project(flashmask_cpbalance LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# target sm arch; setup.py passes the local GPU's (e.g. 90a / 100a). The kernels
# use __reduce_max_sync/__reduce_min_sync (sm_80+), so any sm >= 80 works.
set(CPB_CUDA_ARCH "90a" CACHE STRING "target sm arch, e.g. 90a or 100a")

# Derive compute_ capability: gencode 'compute_' must drop the trailing 'a'
# (compute_90a is illegal; the correct pair is arch=compute_90,code=sm_90a).
string(REGEX REPLACE "a$" "" CPB_CUDA_ARCH_NUM "${CPB_CUDA_ARCH}")

add_library(flashmask_cpbalance SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/cp_balance_bridge.cu
)

target_compile_options(flashmask_cpbalance PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcompiler=-fPIC
    -Xcompiler=-O3
    -std=c++17
    -gencode arch=compute_${CPB_CUDA_ARCH_NUM},code=sm_${CPB_CUDA_ARCH}
    -lineinfo
    -DNDEBUG
    --expt-relaxed-constexpr
>)

target_include_directories(flashmask_cpbalance PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}       # cp_balance_fast.hpp lives next to the bridge
)

target_link_libraries(flashmask_cpbalance PRIVATE CUDA::cudart)

# Export only the fm_cpb_* C ABI; keep implementation symbols local so the lib
# does not interpose with libpaddle.so / torch.
target_link_options(flashmask_cpbalance PRIVATE
    -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cp_balance.map
    -Wl,-Bsymbolic
    -Wl,--exclude-libs,ALL
)

set_target_properties(flashmask_cpbalance PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CUDA_STANDARD 17
    CUDA_ARCHITECTURES "${CPB_CUDA_ARCH}"
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH ON
)
