## Build + Test

Setup a clean Python env:

```
python3.7 -m venv env
source env/bin/activate
pip install numpy torch
```

Build from source:

```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j24
PYTHONPATH=. python ../test.py
```
