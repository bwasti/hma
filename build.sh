PYTORCH_DIR=$(python -c 'import os, torch; print(os.path.dirname(os.path.realpath(torch.__file__)))')
mkdir -p build && cd build
CMAKE_BUILD_TYPE="Release"
while getopts ":d" opt; do
  case ${opt} in
    d ) CMAKE_BUILD_TYPE="Debug"; echo "Building in debug mode"
      ;;
    \? ) echo "Usage: cmd [-h] [-t]"
      ;;
  esac
done
cmake .. -DPYTORCH_DIR=${PYTORCH_DIR} -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE
make -j 24
