## Why?

`hma` attempts to achieve front-end flexbility while being as backend friendly and lightweight as possible.
Note that it is a very early work in progress!

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

**Toy-mode**.  The `test_model.py` file shows the limit of what can be done today.
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

## Contribution

### Adding a differentiable method

There are [only a few methods registered so far](https://github.com/bwasti/hma/tree/master/src/operators).
You can register a method with these macros:

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

REGISTER_GRAD(mul,
              [](const std::vector<Variable *> &inputs,
                 const std::vector<Variable *> &ginputs)
                  -> std::vector<Variable *> {
                return {call("mul", {ginputs[0], inputs[1]})[0],
                        call("mul", {ginputs[0], inputs[0]})[0]};
              });

REGISTER_SHAPE(mul,
               [](const std::vector<Variable *> &inputs) -> std::vector<Shape> {
                 return { inputs[0]->shape };
               });
```

which exposes it to

```
outputs = hma.mul([inputs])
```

Note that gradients are registered symbolically, so repeated differentiation is possible.
You can also register non-differentiable methods if you'd like.

### Messing with the runtime

There are really only [two API entry points](https://github.com/bwasti/hma/blob/master/src/exec.h):
```cpp
// Main API, call is a lazy invocation. debug_info optional
std::vector<Variable *> call(const std::string &,
                             const std::vector<Variable *> &,
                             std::string debug_info = "");
// Resolve evaluates the recorded operations and produces
// a real Tensor.
Tensor *resolve(const Variable *v);
```
`call` doesn't do anything but record computation on `Variable`s, which can later be materialized into `Tensor`s.
All `Variable`s are symbolic representations of `Tensor`s that optionally hold a pointer to a materialized `Tensor`.
This means `call` might immediately call `resolve` (which is the case when laziness is set to 0).

It might make sense to allow static registration of overrides for both of these, but that hasn't been done yet.

## How autograd works

The differentiation of any tensor (`y`) with respect to another tensor (`x`) consists of
walking the graph of deps created by invocations to `call` and finding all paths between the variables.

## Python Wrapper

`pyhma.py` contains useful wrappers to make `hma` feel more like a Python library.

## TODO

- Pointer sharing on no-ops
- Automatic recompute
- Garbage collection
- Many more methods
- Real shape propagation
- Overrideable runtime
