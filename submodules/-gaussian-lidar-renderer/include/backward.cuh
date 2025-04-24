#ifndef BACKWARD_H
#define BACKWARD_H
#include <cstdint>
#include <thrust/swap.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include "utility.cuh"

// void backward_shs_cuda(int32_t num_rays, int32_t D, int32_t M, int32_t G,
//     int32_t* contribute_gid, bool* contribute_clamp,
//     float* grad_intensityprime, float* grad_raydropprime,
//     float3* rays_o, float3* rays_d,
//     float3* means3D,
//     float* shs,
//     float* grad_shs_from_shs,
//     float* grad_means_from_shs);

void backward_trace_cuda(int32_t num_rays, int32_t D, int32_t M, int32_t G,
    int32_t* contribute_gid, float* contribute_T, bool* contribute_clamp,
	float* contribute_tprime, float* contribute_intensityprime, float* contribute_raydropprime,
    float* weights, float* tvalues, float* intensity, float* raydrop,
    float3* means3D, float* covs3D, float* opacity, float* shs,
	float3* rays_o, float3* rays_d,
	float* grad_up_weights, float* grad_up_tvalues, float* grad_up_intensity, float* grad_up_raydrop,
	float3* grad_means3D, float* grad_covs3D, float* grad_opacity, float2* grad_shs);

#endif //BACKWARD_H