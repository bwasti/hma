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
  // TODO make this a set, remove set<int>
  using Route = std::unordered_map<Variable*, std::set<int>>;
  std::queue<std::tuple<Variable*, int, Route>> q;
  // Iterate from X, as most nets work this way
  Route init_route;
  int use_count = 0;
  init_route[x] = {-1};
  q.push(std::make_tuple(x, -1, init_route));
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
    std::unordered_map<Variable*, std::set<int>> route;
    std::tie(var, index, route) = q.front();
    q.pop();
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
        q.push(std::make_tuple(
        iter->first,
        iter->second,
        route
        ));
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
  // This could be way more efficient
  std::set<Operator*> seen_ops{y->op};
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
          if (input->op && seen_ops.find(input->op) == seen_ops.end()) {
            next_frontier.emplace_back(input->op);
            seen_ops.insert(input->op);
          }
        }
      }
    }
    frontier = next_frontier;
  }
  HMA_ENFORCE(grad_map.find(x) != grad_map.end());
  return grad_map[x];
}
