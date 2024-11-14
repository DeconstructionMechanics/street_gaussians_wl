#include <torch/extension.h>
#include <tuple>

float* load_gaussians(torch::Tensor covariance, torch::Tensor xyz, torch::Tensor opacity);
void unload_gaussians(float* gaussians_ptr);
std::tuple<float, float> render_beam(torch::Tensor lidar_position, torch::Tensor beam, torch::Tensor covariance, torch::Tensor xyz, torch::Tensor opacity);
