#include "exec.h"
#include "method.h"

static size_t laziness = DEFAULT_LAZINESS;
void setLaziness(size_t laziness_) { laziness = laziness_; }

std::vector<Variable *> call(const std::string &name,
                             const std::vector<Variable *> &vs,
                             std::string debug_info) {
  HMA_ENFORCE(vs.size());
  auto *graph = vs[0]->graph;
  for (const auto &v : vs) {
    HMA_ENFORCE(v, std::string("Invalid input, perhaps an invalid index into the inputs of a grad function that calls ") + name);
    HMA_ENFORCE(graph == v->graph);
  }
  const auto &method = getMethod(name);
  auto op = graph->create_op(name, vs, method.num_outputs);

  size_t index = 0;
  if (!method.shape) {
    std::stringstream ss;
    ss << "method \"" << method.name << "\" has no shape function";
    HMA_ENFORCE(method.shape, ss.str());
  }
  const auto &shapes = method.shape(vs);
  for (auto &output : op->outputs) {
    output->shape = shapes[index];
    // Really, we only need to resolve the first and the rest are free
    if (output->depth >= laziness) {
      output->tensor = resolve(output);
    }
    index++;
  }
  for (auto &v : vs) {
    v->deps.emplace_back(op);
  }
  return op->outputs;
}



Tensor *resolve(const Variable *v) {
  if (v->tensor) {
    v->depth = 0;
    return v->tensor;
  } else {
    HMA_ENFORCE(v->op, std::string("root variable unresolvable"));
    auto &method = *v->op->method;
    std::vector<const Tensor *> inputs;
    for (const auto &i : v->op->inputs) {
      inputs.emplace_back(resolve(i));
    }

    // For debugging
    auto index = 0;
    const auto &tag = inputs[index]->tag();

    for (const auto &input : inputs) {
      if (input->tag() != tag && method.name != "tag_like") {
        std::stringstream ss;
        ss << "method \"" << method.name << "\" passed invalid tag "
           << getTagName(input->tag()) << " at index " << index << ", expected "
           << getTagName(tag);
        HMA_ENFORCE(input->tag() == tag, ss.str());
      }
      index++;
    }

    std::vector<Tensor *> outputs;
    for (const auto &o : v->op->outputs) {
      o->tensor = new Tensor(tag);
      outputs.emplace_back(o->tensor);
    }
    Context ctx{inputs, outputs};
    if (method.kernels.size() <= tag) {
      std::stringstream ss;
      ss << "no method \"" << method.name << "\" on device \""
         << getTagName(tag) << "\"";
      HMA_ENFORCE(method.kernels.size() > tag, ss.str());
    }
    const auto &f = method.kernels[tag];
    if (!f) {
      std::stringstream ss;
      ss << "no method \"" << method.name << "\" on device \""
         << getTagName(tag) << "\"";
      HMA_ENFORCE(f, ss.str());
    }
    f(ctx);
    v->depth = 0;
    return v->tensor;
  }
}

