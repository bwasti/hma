cmake_minimum_required(VERSION 3.7)

file(GLOB SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

file(GLOB PYTHON_BINDING_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*pybind.cpp
)

list(REMOVE_ITEM SRCS ${PYTHON_BINDING_SRCS})

file(GLOB OPERATOR_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/operators/*.cpp
)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

add_library(libhma SHARED ${SRCS} ${OPERATOR_SRCS})
target_include_directories(libhma PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_subdirectory(pybind11)
pybind11_add_module(hma SHARED ${PYTHON_BINDING_SRCS})
target_link_libraries(hma PUBLIC pybind11 libhma)
target_include_directories(hma PUBLIC
    ${PYBIND11_INCLUDE_DIR}
)

link_directories(${PYTORCH_DIR}/lib)
add_executable(cpp_test test.cpp)
target_link_libraries(cpp_test PUBLIC libhma c10 torch)
target_include_directories(cpp_test PUBLIC
    ${PYTORCH_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

