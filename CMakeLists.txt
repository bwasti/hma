cmake_minimum_required(VERSION 3.7)

file(GLOB SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

file(GLOB OPERATOR_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/operators/*.cpp
)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(pybind11)
pybind11_add_module(hma SHARED ${SRCS} ${OPERATOR_SRCS})
target_link_libraries(hma PUBLIC pybind11)
target_include_directories(hma PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${PYBIND11_INCLUDE_DIR}
)
