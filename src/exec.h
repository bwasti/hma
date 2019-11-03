#pragma once
#include "variable.h"
#include "tensor.h"

// Main API, call is a lazy invocation. debug_info optional
std::vector<Variable *> call(const std::string &,
                             const std::vector<Variable *> &,
                             std::string debug_info = "");
// Resolve evaluates the recorded operations and produces
// a real Tensor.
Tensor *resolve(const Variable *v);

// By default, call only records a maximum of DEFAULT_LAZINESS
// operations before resolving itself into cache.
#define DEFAULT_LAZINESS 100
void setLaziness(size_t laziness);
