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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
trace_bvh_opacity(const torch::Tensor& nodes, const torch::Tensor& aabbs,
          const torch::Tensor& rays_o, const torch::Tensor& rays_d,
          const torch::Tensor& means3D, const torch::Tensor& covs3D,
          const torch::Tensor& opacities, const torch::Tensor& shs, int32_t sh_degree);

std::tuple<torch::Tensor, torch::Tensor>
backward_shs(const torch::Tensor& contribute_gid, const torch::Tensor& contribute_clamp,
          const torch::Tensor& grad_intensityprime, const torch::Tensor& grad_raydropprime,
          const torch::Tensor& rays_o, const torch::Tensor& rays_d,
          const torch::Tensor& means3D,
          const torch::Tensor& shs, int32_t sh_degree);

#endif //BVH_BVH_H
