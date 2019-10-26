#include "grad.h"
#include "error.h"
#include "method.h"

#include <queue>
#include <set>
#include <unordered_map>

Variable* grad(Variable* y, Variable* x, Variable* j) {
  std::unordered_map<Variable*, std::set<int>> need_grad;
  need_grad[y] = {-1};
  std::unordered_map<Variable*, std::set<int>> no_grad;
  std::queue<std::pair<Variable*, int>> q;
  // Iterate from X, as most nets work this way
  q.push(std::make_pair(x, -1));
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
    Variable* var;
    int index;
    std::tie(var, index) = q.front();
    q.pop();
    std::unordered_map<Variable*, std::set<int>> route;
    route[var] = {index};

    while (var) {
      if (var == y) {
        need_grad.insert(route.begin(), route.end());
        break;
      }
      // add to q
      std::vector<std::pair<Variable*, int>> next;
      for (auto dep : var->deps) {
        auto i = 0;
        for (auto inp : dep->inputs) {
          if (inp == var) {
            for (const auto& out : dep->outputs) {
              next.emplace_back(std::make_pair(out, i));
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
      var = iter->first;
      route[var].insert(0);
      iter++;
      while (iter != next.end()) {
        q.push(*iter);
        iter++;
      }
    }
  }
  // Now calculate the gradients
  std::unordered_map<Variable*, Variable*> grad_map;
  // This is the input
  grad_map[y] = j;
  std::vector<Operator*> frontier{y->op};
  std::vector<Operator*> next_frontier;
  while (frontier.size()) {
    next_frontier.clear();
    for (const auto& op : frontier) {
      std::vector<Variable*> grad_inputs;
      for (const auto& op_out : op->outputs) {
        HMA_ENFORCE(need_grad.find(op_out) != need_grad.end());
        auto grad_inp_iter = grad_map.find(op_out);
        HMA_ENFORCE(grad_inp_iter != grad_map.end());
        grad_inputs.emplace_back(grad_inp_iter->second);
      }
      auto g_outs = op->method->grad(op->inputs, grad_inputs);
      for (auto i = 0; i < g_outs.size(); ++i) {
        auto input = op->inputs[i];
        if (need_grad.find(input) != need_grad.end()) {
          if (grad_map.find(input) != grad_map.end()) {
            grad_map[input] = call("add", {grad_map[input], g_outs[i]})[0];
          } else {
            grad_map[input] = g_outs[i];
          }
          if (input->op) {
            next_frontier.emplace_back(input->op);
          } else {
            HMA_ENFORCE(input == x);
          }
        }
      }
    }
    frontier = next_frontier;
  }
  return grad_map[x];
}
