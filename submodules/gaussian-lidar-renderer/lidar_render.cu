#include "lidar_render.h"
#include <iostream>
#include <glm/glm.hpp>

__global__ void gaussian_intersect(
    int n_g, int idx_b, int n_b,
    const float* lidar_position,
    const float* beam,
    const float* covariance,
    const float* xyz,
    const float* opacity,
    float* const out_alpha,
    float* const out_tvalue)
{
    int idx_g = blockIdx.x * blockDim.x + threadIdx.x;;
    if (idx_g >= n_g || idx_b >= n_b) {
        return;
    }
    glm::vec3 u = glm::vec3(xyz[idx_g * 3], xyz[idx_g * 3 + 1], xyz[idx_g * 3 + 2]);
    glm::vec3 r0 = glm::vec3(lidar_position[0], lidar_position[1], lidar_position[2]);
    glm::vec3 rd = glm::vec3(beam[idx_b * 3], beam[idx_b * 3 + 1], beam[idx_b * 3 + 2]);
    glm::mat3 sigma = glm::mat3(covariance[idx_g * 6], covariance[idx_g * 6 + 1], covariance[idx_g * 6 + 2],
                                covariance[idx_g * 6 + 1], covariance[idx_g * 6 + 3], covariance[idx_g * 6 + 4],
                                covariance[idx_g * 6 + 2], covariance[idx_g * 6 + 4], covariance[idx_g * 6 + 5]);
    glm::vec3 sigma_rd = sigma * rd;
    float t = glm::dot(u - r0, sigma_rd) / glm::dot(rd, sigma_rd);

    glm::vec3 du = r0 - u + t * rd;
    float power = -0.5f * glm::dot(du, sigma * du);
    if (power > 0.0f){
        out_alpha[idx_g] = 0;
        out_tvalue[idx_g] = 0;
		return;
    }
    float alpha = min(0.99f, opacity[idx_g] * exp(power));
    if (alpha < 1.0f / 255.0f){
        out_alpha[idx_g] = 0;
        out_tvalue[idx_g] = 0;
        return;
    }
    out_alpha[idx_g] = alpha;
    out_tvalue[idx_g] = t;
}

__global__ void ray_intersect(

)

// def render_beam(self, lidar_position: torch.Tensor, beam: torch.Tensor, covariance: torch.Tensor, xyz: torch.Tensor, opacity: torch.Tensor)    
std::tuple<float, float> render_beam(torch::Tensor lidar_position, torch::Tensor beam, torch::Tensor covariance, torch::Tensor xyz, torch::Tensor opacity){
    
    int n = xyz.sizes()[0];

    float* device_alpha;
    float* host_alpha = new float[n];
    cudaMalloc(&device_alpha, n * sizeof(float));
    float* device_tvalue;
    float* host_tvalue = new float[n];
    cudaMalloc(&device_tvalue, n * sizeof(float));

    int threads_per_block = 1024;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    gaussian_intersect<<<blocks_per_grid, threads_per_block>>>(
        n, 0, 1,
        lidar_position.data_ptr<float>(),
        beam.contiguous().data_ptr<float>(),
        covariance.contiguous().data_ptr<float>(),
        xyz.contiguous().data_ptr<float>(),
        opacity.contiguous().data_ptr<float>(),
        device_alpha,
        device_tvalue
        );

    cudaMemcpy(host_alpha, device_alpha, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_tvalue, device_tvalue, n * sizeof(float), cudaMemcpyDeviceToHost);
    float w = host_alpha[0];  
    float t = host_tvalue[0];

    cudaFree(device_alpha);
    delete[] host_alpha;
    cudaFree(device_tvalue);
    delete[] host_tvalue;

    return std::make_tuple(t, w);
}


