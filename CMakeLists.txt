cmake_minimum_required(VERSION 3.7)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -Wl,--no-as-needed")

file(GLOB SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

file(GLOB PYTHON_BINDING_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*pybind.cpp
)

list(REMOVE_ITEM SRCS ${PYTHON_BINDING_SRCS})

# Base for hma, no operators, no python bindings
add_library(hma_base SHARED ${SRCS})
target_include_directories(hma_base PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
# C++ interface, operators but no python
set(HMA_LIBS hma_base)

file(GLOB OPERATOR_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/operators/*.cpp
)

add_library(hma_cpu SHARED ${OPERATOR_SRCS})
target_link_libraries(hma_cpu hma_base)
list(APPEND HMA_LIBS hma_cpu)

find_package(CUDA)
if (CUDA_FOUND)
  file(GLOB CUDA_OPERATOR_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/operators/cuda/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/operators/cuda/*.cpp
  )
  cuda_add_library(hma_cuda SHARED ${CUDA_OPERATOR_SRCS})
  target_link_libraries(hma_cuda hma_base)
  list(APPEND HMA_LIBS hma_cuda)
endif()

add_subdirectory(pybind11)
pybind11_add_module(hma SHARED ${PYTHON_BINDING_SRCS})
target_link_libraries(hma PUBLIC pybind11 ${HMA_LIBS})
target_include_directories(hma PUBLIC
    ${PYBIND11_INCLUDE_DIR}
)

link_directories(${PYTORCH_DIR}/lib)
add_executable(cpp_test test.cpp)
target_link_libraries(cpp_test PUBLIC ${HMA_LIBS} c10 torch)
target_include_directories(cpp_test PUBLIC
    ${PYTORCH_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

