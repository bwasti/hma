cmake_minimum_required(VERSION 3.7)

file(GLOB SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)
find_package(MKL)
if (${MKL_FOUND})
  set(SRCS ${SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/extensions/mkl.cpp)
endif()

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(pybind11)
pybind11_add_module(hma SHARED ${SRCS})

target_link_libraries(hma PUBLIC pybind11)
target_include_directories(hma PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${PYBIND11_INCLUDE_DIR}
)
message("-- ${MKL_LIBRARIES} ${MKL_INCLUDE_DIR}")
if (${MKL_FOUND})
  target_link_libraries(hma PUBLIC
		${MKL_OPENMP_LIBRARY}
		${MKL_LIBRARIES}
  )
  target_include_directories(hma PUBLIC
      ${MKL_INCLUDE_DIR}
  )
endif()
