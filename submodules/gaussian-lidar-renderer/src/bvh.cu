#ifndef BVH_BVH_CU
#define BVH_BVH_CU
#include <tuple>
#include "bvh.h"
#include "construct.cuh"
#include "trace.cuh"
#include "backward.cuh"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
create_bvh(const torch::Tensor& means3D, const torch::Tensor& scales, const torch::Tensor& rotations, const torch::Tensor& nodes, const torch::Tensor& aabbs){
    const uint32_t P = means3D.size(0);

    auto int_opts = means3D.options().dtype(torch::kInt32);
    auto float_opts = means3D.options().dtype(torch::kFloat32);

    torch::Tensor mortons = torch::zeros({P}, means3D.options().dtype(torch::kLong));

    construct_bvh(
            P,
            means3D.contiguous().data_ptr<float>(),
            scales.contiguous().data_ptr<float>(),
            rotations.contiguous().data_ptr<float>(),
            (int32_t*)nodes.contiguous().data_ptr<int>(),
            aabbs.contiguous().data_ptr<float>(),
            (uint64_t*)mortons.contiguous().data_ptr<int64_t>()
    );
    return std::make_tuple(nodes, aabbs, mortons);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
trace_bvh(const torch::Tensor& nodes, const torch::Tensor& aabbs,
          const torch::Tensor& rays_o, const torch::Tensor& rays_d,
          const torch::Tensor& means3D, const torch::Tensor& covs3D,
          const torch::Tensor& opacities){
    int32_t num_rays = rays_o.size(0);

    auto int_opts = rays_o.options().dtype(torch::kInt32);
    auto float_opts = rays_o.options();
    torch::Tensor num_contributes = torch::zeros({num_rays, 1}, int_opts);

    auto result = trace_bvh_cuda(num_rays,
                   nodes.contiguous().data_ptr<int32_t>(),
                   aabbs.contiguous().data_ptr<float>(),
                   (float3*)rays_o.contiguous().data_ptr<float>(),
                   (float3*)rays_d.contiguous().data_ptr<float>(),
                   (float3*)means3D.contiguous().data_ptr<float>(),
                   covs3D.contiguous().data_ptr<float>(),
                   opacities.contiguous().data_ptr<float>(),
                   num_contributes.contiguous().data_ptr<int32_t>());

//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start);
//     float milliseconds = 0;

    int32_t num_rendered = std::get<0>(result);
    thrust::device_vector<int32_t>& point_list_vec = std::get<1>(result);
    thrust::device_vector<float3>& position_list_vec = std::get<2>(result);
    thrust::device_vector<int32_t>& ray_id_list_vec = std::get<3>(result);
    if (num_rendered == 0){
        torch::Tensor point_list_tensor = torch::zeros({0, 1}, int_opts);
        torch::Tensor position_list_tensor = torch::zeros({0, 3}, float_opts);
        torch::Tensor ray_id_list_tensor = torch::zeros({0, 3}, float_opts);
        return std::make_tuple(num_contributes, point_list_tensor, position_list_tensor, ray_id_list_tensor);
    }
    int32_t* point_list = thrust::raw_pointer_cast(point_list_vec.data());
    int32_t size = point_list_vec.size();
    torch::Tensor point_list_tensor = torch::from_blob(point_list, {size, 1}, int_opts);
    point_list_tensor = point_list_tensor.clone();

    float* position_list = (float*)thrust::raw_pointer_cast(position_list_vec.data());
    torch::Tensor position_list_tensor = torch::from_blob(position_list, {size, 3}, float_opts);
    position_list_tensor = position_list_tensor.clone();

    int32_t* ray_id_list = thrust::raw_pointer_cast(ray_id_list_vec.data());
    torch::Tensor ray_id_list_tensor = torch::from_blob(ray_id_list, {size, 1}, int_opts);
    ray_id_list_tensor = ray_id_list_tensor.clone();

//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     std::cout << "after time: " << milliseconds << " ms" << std::endl;
//     cudaEventRecord(start);
    return std::make_tuple(num_contributes, point_list_tensor, position_list_tensor, ray_id_list_tensor);
}


std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor
>
trace_bvh_opacity(const torch::Tensor& nodes, const torch::Tensor& aabbs,
          const torch::Tensor& rays_o, const torch::Tensor& rays_d,
          const torch::Tensor& means3D, const torch::Tensor& covs3D,
          const torch::Tensor& opacities, const torch::Tensor& shs, int32_t sh_degree, bool needs_grad, int32_t n_gaussian_backward){
    int32_t num_rays = rays_o.numel() / rays_o.size(-1);
    int32_t M = shs.size(1);
    auto rays_o_shape = rays_o.sizes().slice(0, rays_o.dim() - 1);
//     auto rays_o_shape = rays_o.sizes().vec();
//     rays_o_shape.pop_back();
//     rays_o_shape.push_back(1);

    auto int_opts = rays_o.options().dtype(torch::kInt32);
    auto float_opts = rays_o.options();
    auto bool_opts = rays_o.options().dtype(torch::kBool);

    torch::Tensor num_contributes = torch::zeros(rays_o_shape, int_opts);
    torch::Tensor rendered_opacity = torch::zeros(rays_o_shape, float_opts);
    torch::Tensor rendered_tvalue = torch::ones(rays_o_shape, float_opts);
    torch::Tensor rendered_intensity = torch::ones(rays_o_shape, float_opts);
    torch::Tensor rendered_raydrop = torch::ones(rays_o_shape, float_opts);

    torch::Tensor contribute_gid = torch::full({num_rays, n_gaussian_backward}, -1, int_opts);
    torch::Tensor contribute_depth = torch::full({num_rays, n_gaussian_backward}, -1, int_opts);
    torch::Tensor contribute_T = torch::zeros({num_rays, n_gaussian_backward}, float_opts);
    torch::Tensor contribute_clamp = torch::zeros({num_rays, n_gaussian_backward, 2}, bool_opts);
    torch::Tensor contribute_tprime = torch::zeros({num_rays, n_gaussian_backward}, float_opts);
    torch::Tensor contribute_intensityprime = torch::zeros({num_rays, n_gaussian_backward}, float_opts);
    torch::Tensor contribute_raydropprime = torch::zeros({num_rays, n_gaussian_backward}, float_opts);
    torch::Tensor contribute_Talpha = torch::zeros({num_rays, n_gaussian_backward}, float_opts);

    trace_bvh_opacity_cuda(num_rays, sh_degree, M, n_gaussian_backward,
                   nodes.contiguous().data_ptr<int32_t>(),
                   aabbs.contiguous().data_ptr<float>(),
                   (float3*)rays_o.contiguous().data_ptr<float>(),
                   (float3*)rays_d.contiguous().data_ptr<float>(),
                   (float3*)means3D.contiguous().data_ptr<float>(),
                   covs3D.contiguous().data_ptr<float>(),
                   opacities.contiguous().data_ptr<float>(),
                   shs.contiguous().data_ptr<float>(),
                   num_contributes.contiguous().data_ptr<int32_t>(),
                   rendered_opacity.contiguous().data_ptr<float>(),
                   rendered_tvalue.contiguous().data_ptr<float>(),
                   rendered_intensity.contiguous().data_ptr<float>(),
                   rendered_raydrop.contiguous().data_ptr<float>(),
                   needs_grad,
                   contribute_gid.contiguous().data_ptr<int32_t>(),
                   contribute_depth.contiguous().data_ptr<int32_t>(),
                   contribute_T.contiguous().data_ptr<float>(),
                   contribute_clamp.contiguous().data_ptr<bool>(),
                   contribute_tprime.contiguous().data_ptr<float>(),
                   contribute_intensityprime.contiguous().data_ptr<float>(),
                   contribute_raydropprime.contiguous().data_ptr<float>(),
                   contribute_Talpha.contiguous().data_ptr<float>()
                   );

    return std::make_tuple(
        num_contributes, rendered_opacity, rendered_tvalue, rendered_intensity, rendered_raydrop,
        contribute_gid, contribute_depth, contribute_T, contribute_clamp, contribute_tprime, contribute_intensityprime, contribute_raydropprime
        );
}


// std::tuple<torch::Tensor, torch::Tensor>
// backward_shs(const torch::Tensor& contribute_gid, const torch::Tensor& contribute_clamp,
//           const torch::Tensor& grad_intensityprime, const torch::Tensor& grad_raydropprime,
//           const torch::Tensor& rays_o, const torch::Tensor& rays_d,
//           const torch::Tensor& means3D,
//           const torch::Tensor& shs, int32_t sh_degree){
//     int32_t num_rays = rays_o.numel() / rays_o.size(-1);
//     int32_t M = shs.size(1);
//     int32_t G = contribute_gid.size(1);

//     auto int_opts = rays_o.options().dtype(torch::kInt32);
//     auto float_opts = rays_o.options();
    
//     torch::Tensor grad_shs_from_shs = torch::zeros(shs.sizes(), float_opts);
//     torch::Tensor grad_means_from_shs = torch::zeros(means3D.sizes(), float_opts);

//     backward_shs_cuda(num_rays, sh_degree, M, G,
//                    contribute_gid.contiguous().data_ptr<int32_t>(),
//                    contribute_clamp.contiguous().data_ptr<bool>(),
//                    grad_intensityprime.contiguous().data_ptr<float>(),
//                    grad_raydropprime.contiguous().data_ptr<float>(),
//                    (float3*)rays_o.contiguous().data_ptr<float>(),
//                    (float3*)rays_d.contiguous().data_ptr<float>(),
//                    (float3*)means3D.contiguous().data_ptr<float>(),
//                    shs.contiguous().data_ptr<float>(),
//                    grad_shs_from_shs.contiguous().data_ptr<float>(),
//                    grad_means_from_shs.contiguous().data_ptr<float>());
//     return std::make_tuple(grad_shs_from_shs, grad_means_from_shs);
// }

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
backward_trace(const torch::Tensor& contribute_gid, const torch::Tensor& contribute_T, const torch::Tensor& contribute_clamp,
          const torch::Tensor& contribute_tprime, const torch::Tensor& contribute_intensityprime, const torch::Tensor& contribute_raydropprime,
          const torch::Tensor& weights, const torch::Tensor& tvalues, const torch::Tensor& intensity, const torch::Tensor& raydrop,
          const torch::Tensor& means3D, const torch::Tensor& covs3D, const torch::Tensor& opacity, const torch::Tensor& shs, int32_t sh_degree
          const torch::Tensor& rays_o, const torch::Tensor& rays_d,
          const torch::Tensor& grad_up_weights, const torch::Tensor& grad_up_tvalues, const torch::Tensor& grad_up_intensity, const torch::Tensor& grad_up_raydrop){
    int32_t num_rays = rays_o.numel() / rays_o.size(-1);
    int32_t M = shs.size(1);
    int32_t G = contribute_gid.size(1);

    auto int_opts = rays_o.options().dtype(torch::kInt32);
    auto float_opts = rays_o.options();
    
    torch::Tensor grad_means3D = torch::zeros(means3D.sizes(), float_opts);
    torch::Tensor grad_covs3D = torch::zeros(covs3D.sizes(), float_opts);
    torch::Tensor grad_opacity = torch::zeros(opacity.sizes(), float_opts);
    torch::Tensor grad_shs = torch::zeros(shs.sizes(), float_opts);

    backward_trace_cuda(num_rays, sh_degree, M, G,
                   contribute_gid.contiguous().data_ptr<int32_t>(),
                   contribute_T.contiguous().data_ptr<float>(),
                   contribute_clamp.contiguous().data_ptr<bool>(),
                   contribute_tprime.contiguous().data_ptr<float>(),
                   contribute_intensityprime.contiguous().data_ptr<float>(),
                   contribute_raydropprime.contiguous().data_ptr<float>(),
                   weights.contiguous().data_ptr<float>(),
                   tvalues.contiguous().data_ptr<float>(),
                   intensity.contiguous().data_ptr<float>(),
                   raydrop.contiguous().data_ptr<float>(),
                   (float3*)means3D.contiguous().data_ptr<float>(),
                   covs3D.contiguous().data_ptr<float>(),
                   opacity.contiguous().data_ptr<float>(),
                   shs.contiguous().data_ptr<float>(),
                   (float3*)rays_o.contiguous().data_ptr<float>(),
                   (float3*)rays_d.contiguous().data_ptr<float>(),
                   grad_up_weights.contiguous().data_ptr<float>(),
                   grad_up_tvalues.contiguous().data_ptr<float>(),
                   grad_up_intensity.contiguous().data_ptr<float>(),
                   grad_up_raydrop.contiguous().data_ptr<float>(),
                   (float3*)grad_means3D.contiguous().data_ptr<float>(),
                   grad_covs3D.contiguous().data_ptr<float>(),
                   grad_opacity.contiguous().data_ptr<float>(),
                   (float2*)grad_shs.contiguous().data_ptr<float>());
    return std::make_tuple(grad_means3D, grad_covs3D, grad_opacity, grad_shs);
}


#endif //BVH_BVH_CU