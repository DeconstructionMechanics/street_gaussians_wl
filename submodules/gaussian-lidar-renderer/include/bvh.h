#ifndef BVH_BVH_H
#define BVH_BVH_H
#include <torch/extension.h>
#include <cstdint>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
create_bvh(const torch::Tensor& means3D, const torch::Tensor& scales, const torch::Tensor& rotations, const torch::Tensor& nodes, const torch::Tensor& aabbs);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
trace_bvh(const torch::Tensor& nodes, const torch::Tensor& aabbs,
          const torch::Tensor& rays_o, const torch::Tensor& rays_d,
          const torch::Tensor& means3D, const torch::Tensor& covs3D,
          const torch::Tensor& opacities);

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor
>
trace_bvh_opacity(const torch::Tensor& nodes, const torch::Tensor& aabbs,
          const torch::Tensor& rays_o, const torch::Tensor& rays_d,
          const torch::Tensor& means3D, const torch::Tensor& covs3D,
          const torch::Tensor& opacities, const torch::Tensor& shs, int32_t sh_degree, bool needs_grad, int32_t n_gaussian_backward)

// std::tuple<torch::Tensor, torch::Tensor>
// backward_shs(const torch::Tensor& contribute_gid, const torch::Tensor& contribute_clamp,
//           const torch::Tensor& grad_intensityprime, const torch::Tensor& grad_raydropprime,
//           const torch::Tensor& rays_o, const torch::Tensor& rays_d,
//           const torch::Tensor& means3D,
//           const torch::Tensor& shs, int32_t sh_degree);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
backward_trace(const torch::Tensor& contribute_gid, const torch::Tensor& contribute_T, const torch::Tensor& contribute_clamp,
          const torch::Tensor& contribute_tprime, const torch::Tensor& contribute_intensityprime, const torch::Tensor& contribute_raydropprime,
          const torch::Tensor& weights, const torch::Tensor& tvalues, const torch::Tensor& intensity, const torch::Tensor& raydrop,
          const torch::Tensor& means3D, const torch::Tensor& covs3D, const torch::Tensor& opacity, const torch::Tensor& shs, int32_t sh_degree
          const torch::Tensor& rays_o, const torch::Tensor& rays_d,
          const torch::Tensor& grad_up_weights, const torch::Tensor& grad_up_tvalues, const torch::Tensor& grad_up_intensity, const torch::Tensor& grad_up_raydrop);

#endif //BVH_BVH_H
