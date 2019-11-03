#include "grad.h"
#include "error.h"
#include "method.h"
#include "exec.h"

#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>

Variable *grad(Variable *y, Variable *x, Variable *j) {
  std::unordered_set<Variable *> need_grad;
  need_grad.insert(y);
  std::unordered_set<Variable *> no_grad;
  using Route = std::unordered_set<Variable *>;
  std::queue<std::pair<Variable *, Route>> q;
  // Iterate from X, as most nets work this way
  Route init_route;
  init_route.insert(x);
  q.push(std::make_pair(x, init_route));
  // q contains variables that haven't been
  // traversed.
  while (q.size()) {
    // Take a variable and try to find y,
    // "staying left" (first dep every time).
    //
    //   |
    //   v
    //  dep1  dep2
    //    \   /
    //     var
    //
    // Every time we "stay left," add the other deps to q
    // If we find y -- add the whole route to need_grad
    // If we can't find y -- add the whole route to no_grad
    Variable *var;
    std::unordered_set<Variable *> route;
    std::tie(var, route) = q.front();
    q.pop();
    route.insert(var);

    while (var) {
      if (var == y) {
        need_grad.insert(route.begin(), route.end());
        break;
      }
      // add to q
      std::vector<Variable *> next;
      for (auto dep : var->deps) {
        auto i = 0;
        for (auto inp : dep->inputs) {
          if (inp == var) {
            for (const auto &out : dep->outputs) {
              next.emplace_back(out);
            }
          }
          i++;
        }
      }
      if (!next.size()) {
        no_grad.insert(route.begin(), route.end());
        break;
      }
      auto iter = next.begin();
      var = *iter;
      route.insert(var);
      iter++;
      while (iter != next.end()) {
        q.push(std::make_pair(*iter, route));
        iter++;
      }
    }
  }
  // Now calculate the gradients
  std::unordered_map<Variable *, Variable *> grad_map;
  // This is the input
  grad_map[y] = j;
  std::vector<Operator *> frontier{y->op};
  std::vector<Operator *> next_frontier;
  // This could be way more efficient
  std::set<Operator *> seen_ops{y->op};
  while (frontier.size()) {
    next_frontier.clear();
    for (const auto &op : frontier) {
      std::vector<Variable *> grad_inputs;
      for (const auto &op_out : op->outputs) {
        HMA_ENFORCE(need_grad.find(op_out) != need_grad.end());
        auto grad_inp_iter = grad_map.find(op_out);
        HMA_ENFORCE(grad_inp_iter != grad_map.end());
        grad_inputs.emplace_back(grad_inp_iter->second);
      }
      bool run_grad = false;
      for (const auto &input : op->inputs) {
        if (need_grad.find(input) != need_grad.end()) {
          run_grad = true;
          break;
        }
      }
      if (run_grad) {
        const auto &g = op->method->grad;
        if (!g) {
          std::stringstream ss;
          ss << "no known grad for method \"" << op->method->name << "\"";
          HMA_ENFORCE(g, ss.str());
        }
        auto g_outs = g(op->inputs, grad_inputs);
        for (auto i = 0; i < g_outs.size(); ++i) {
          auto input = op->inputs[i];
          if (need_grad.find(input) != need_grad.end()) {
            if (grad_map.find(input) != grad_map.end()) {
              grad_map[input] = call("add", {grad_map[input], g_outs[i]})[0];
            } else {
              grad_map[input] = g_outs[i];
            }
            if (input->op && seen_ops.find(input->op) == seen_ops.end()) {
              next_frontier.emplace_back(input->op);
              seen_ops.insert(input->op);
            }
          }
        }
      }
    }
    frontier = next_frontier;
  }
  HMA_ENFORCE(grad_map.find(x) != grad_map.end());
  return grad_map[x];
}
