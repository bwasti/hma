## Why?

`hma` attempts to achieve front-end flexbility while being as backend friendly as possible.
Key tenets are:

- Execution is lazy
- Every operation is pure
- All gradients are defined symbolically
- All shapes are defined symbolically

Although these restrictions depart somewhat from conventional eager frameworks,
they yield many interesting properties, such as:

- Ability to differentiate with respect to any variable (i.e. no `requires_grad`)
- Ability to trade memory for compute
- "Constness" deduction and compute caching
- Hot-path compilation and optimization
- Simple high-latency remote device integration (i.e. accelerators)


## Status

Toy-mode.  The `test_model.py` file shows the limit of what can be done today.
Basically linear regression (with autograd for that usecase).

## Build + Test

Setup a clean Python env:

```bash
python3.7 -m venv env
source env/bin/activate
pip install numpy torch
```

Build from source:

```bash
./build.sh
PYTHONPATH=build python test_model.py
```

## Usage

```python
import numpy as np
import pyhma as ph

# use numpy to seed values
a_ = np.random.randn(128,128).astype(np.float32)
b_ = np.random.randn(128,128).astype(np.float32)
ones = np.ones((128,128)).astype(np.float32)

# create a pyhma Tensor
a = ph.Tensor(a_)
b = ph.Tensor(b_)
c = func(a, b)
# create jacobian
g = c.grad(a)
# call the jacobian
a_grad = g(ph.Tensor(ones))
# no execution happens until this line
print(a_grad.np())
```

## Code

Register a method with this macro:

```cpp
REGISTER_METHOD(mul, [](Context &ctx) {
  const auto &t1 = ctx.input(0);
  const auto &t2 = ctx.input(1);
  auto *out = ctx.output(0);
  out->resize(t1.shape(), t1.dtype());

  // Note this assumes t1.dtype() == Tensor::Dtype::float_
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



## How autograd works

The differentiation of any tensor (`y`) with respect to another tensor (`x`) consists of
walking the graph of deps and finding all paths between the variables.

## Optimizations

Every Tensor is "lazy," which means they tote around a full history of their creation.
This allows a suite of "fusion" optimizations.

Every operation is "pure" so these histories can recomputed at any time.
This allows caching and automatic tradeoffs between compute and memory consumption.

## Python Wrapper

`pyhma.py` contains useful wrappers to make `hma` feel more like a Python library.

## TODO

Although size/shape stuff has been stubbed out, it hasn't been implemented in a usable way.
