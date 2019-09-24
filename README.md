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

## Code

Register a method with this macro:

```
REGISTER_METHOD(mul, [](Context &ctx) {
  const auto &t1 = ctx.input(0);
  const auto &t2 = ctx.input(1);
  auto *out = ctx.output(0);
  out->resize(t1.shape(), t1.dtype());

  auto *d1 = static_cast<const float *>(t1.ptr());
  auto *d2 = static_cast<const float *>(t2.ptr());
  auto *out_d = static_cast<float *>(out->ptr());
  for (auto i = 0; i < t1.size(); ++i) {
    out_d[i] = d1[i] * d2[i];
  }
});
```

which exposes it to

```
outputs = hma.mul([inputs])
```
