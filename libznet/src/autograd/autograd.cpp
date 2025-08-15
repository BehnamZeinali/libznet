// #include <znet/autograd/autograd.hpp>
// #include <znet/autograd/tensor.hpp>
// #include <unordered_set>
// namespace znet {

// // Topological sort helper
// void build_topo(AutogradFunction* fn,
//                 std::vector<AutogradFunction*>& order,
//                 std::unordered_set<AutogradFunction*>& visited) {
//     if (!fn || visited.count(fn)) return;
//     visited.insert(fn);

//     for (const znet::Tensor* input : fn->inputs()) {
//         if (input->grad_fn()) {
//             build_topo(input->grad_fn().get(), order, visited);
//         }
//     }

//     order.push_back(fn);  // post-order
// }

// // Top-level backward engine
// void backward(const Tensor& loss) {
//     if (!loss.grad()) {
//         const_cast<Tensor&>(loss).set_grad(std::make_shared<Tensor>(
//             loss.shape(), std::vector<float>(loss.numel(), 1.0f)));
//     }

//     std::vector<AutogradFunction*> topo_order;
//     std::unordered_set<AutogradFunction*> visited;

//     if (loss.grad_fn()) {
//         build_topo(loss.grad_fn().get(), topo_order, visited);
//     }

//     for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
//         (*it)->backward();
//     }
// }

// }
