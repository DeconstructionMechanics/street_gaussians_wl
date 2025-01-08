#include "trace.cuh"

struct id_intersection{
    int32_t id;
    float2 intersection;
};




// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};
__device__ float2 computeFeaturesFromSH(int idx, int deg, int max_coeffs, const float3* means, float3 campos, const float* shs)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	float3 pos = means[idx];
	float3 dir = {
        pos.x - campos.x,
        pos.y - campos.y,
        pos.z - campos.z,
        };
    float dir_len = sqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    dir.x /= dir_len;
    dir.y /= dir_len;
    dir.z /= dir_len;

	float2* sh = ((float2*)shs) + idx * max_coeffs;
	float2 result = {
        SH_C0 * sh[0].x,
        SH_C0 * sh[0].y
        };

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = {
            result.x - SH_C1 * y * sh[1].x + SH_C1 * z * sh[2].x - SH_C1 * x * sh[3].x,
            result.y - SH_C1 * y * sh[1].y + SH_C1 * z * sh[2].y - SH_C1 * x * sh[3].y
            };

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = {
                result.x +
				SH_C2[0] * xy * sh[4].x +
				SH_C2[1] * yz * sh[5].x +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6].x +
				SH_C2[3] * xz * sh[7].x +
				SH_C2[4] * (xx - yy) * sh[8].x,
                result.y +
				SH_C2[0] * xy * sh[4].y +
				SH_C2[1] * yz * sh[5].y +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6].y +
				SH_C2[3] * xz * sh[7].y +
				SH_C2[4] * (xx - yy) * sh[8].y
                };

			if (deg > 2)
			{
				result = {
                    result.x +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9].x +
					SH_C3[1] * xy * z * sh[10].x +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11].x +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12].x +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13].x +
					SH_C3[5] * z * (xx - yy) * sh[14].x +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15].x,
                    result.y +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9].y +
					SH_C3[1] * xy * z * sh[10].y +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11].y +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12].y +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13].y +
					SH_C3[5] * z * (xx - yy) * sh[14].y +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15].y
                    };
			}
		}
	}
	result = {max(result.x + 0.5f, 0.0f), max(result.y + 0.5f, 0.0f)};
	return result;
}



std::tuple<int32_t, thrust::device_vector<int32_t>, thrust::device_vector<float3>, thrust::device_vector<int32_t>>
trace_bvh_cuda(int32_t num_rays, int32_t* nodes, float* aabbs,
    float3* rays_o, float3* rays_d,
    float3* means3D, float* covs3D,
    float* opacities, int32_t* num_contributes){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    float milliseconds = 0;


    auto* aabbs_internal = reinterpret_cast<aabb_type*>(aabbs);

    thrust::for_each(thrust::device,
        thrust::make_counting_iterator<int32_t>(0),
        thrust::make_counting_iterator<int32_t>(num_rays),
        [nodes, aabbs_internal, rays_o, rays_d, num_contributes] __device__(int32_t idx){
        IndexStack<int32_t> stack_device;
        stack_device.push(0);
        int32_t count = 0;
        float3 ray_o = rays_o[idx], ray_d = rays_d[idx];
        while(!stack_device.empty()){
            int32_t node_id = stack_device.pop();
            int32_t* node = nodes + node_id * 5;
            if(node[4] <= 4){
                count += node[4];
            }
            else{
                int32_t lid = node[1], rid = node[2];
                float2 interection_l = ray_intersects(aabbs_internal[lid], ray_o, ray_d);
                float2 interection_r = ray_intersects(aabbs_internal[rid], ray_o, ray_d);
                if(interection_l.y > interection_r.y){
                    if(interection_l.y > 0){
                        stack_device.push(lid);
                    }
                    if(interection_r.y > 0){
                        stack_device.push(rid);
                    }
                }                
                else{
                    if(interection_r.y > 0){
                        stack_device.push(rid);
                    }
                    if(interection_l.y > 0){
                        stack_device.push(lid);
                    }
                }
            }
        }
        num_contributes[idx] = count;
    });

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "num_contributes time: " << milliseconds << " ms" << std::endl;
    cudaEventRecord(start);

    thrust::device_vector<int32_t> ray_offsets_vec(num_rays);
    thrust::device_ptr<int32_t> num_contributes_ptr = thrust::device_pointer_cast(num_contributes);

    thrust::inclusive_scan(num_contributes_ptr, num_contributes_ptr + num_rays, ray_offsets_vec.begin());
    int32_t* ray_offsets = thrust::raw_pointer_cast(ray_offsets_vec.data());
    int32_t num_rendered;
    cudaMemcpy(&num_rendered, ray_offsets + num_rays - 1, sizeof(int32_t), cudaMemcpyDeviceToHost);
    // printf("num_rendered: %d\n",num_rendered);

    thrust::device_vector<uint64_t> ray_list_key_vec(num_rendered);
    uint64_t* ray_list_key = thrust::raw_pointer_cast(ray_list_key_vec.data());

    thrust::device_vector<int32_t> point_list_vec(num_rendered);
    int32_t* point_list = thrust::raw_pointer_cast(point_list_vec.data());
    thrust::device_vector<float3> position_list_vec(num_rendered);
    float3* position_list = thrust::raw_pointer_cast(position_list_vec.data());
    thrust::device_vector<int32_t> ray_id_list_vec(num_rendered);
    int32_t* ray_id_list = thrust::raw_pointer_cast(ray_id_list_vec.data());

    //     cudaEventRecord(stop);
    //     cudaEventSynchronize(stop);
    //     cudaEventElapsedTime(&milliseconds, start, stop);
    //     std::cout << "inclusive_scan time: " << milliseconds << " ms" << std::endl;
    cudaEventRecord(start);

    thrust::for_each(thrust::device,
        thrust::make_counting_iterator<uint32_t>(0),
        thrust::make_counting_iterator<uint32_t>(num_rays),
        [nodes, aabbs_internal, rays_o, rays_d, num_contributes,
        ray_offsets, ray_list_key, point_list, ray_id_list,
        means3D, covs3D, position_list
        ] __device__(uint32_t idx){
        if(num_contributes[idx] == 0) return;
        uint32_t offset = (idx == 0) ? 0 : ray_offsets[idx - 1];
        uint64_t* ray_key_ptr = ray_list_key + offset;
        int32_t* point_ptr = point_list + offset;
        float3* position_ptr = position_list + offset;
        int32_t* ray_id_ptr = ray_id_list + offset;

        IndexStack<id_intersection> stack_device;
        stack_device.push({ 0, {-1000, 1000} });
        int32_t count = 0;
        float3 ray_o = rays_o[idx], ray_d = rays_d[idx];
        while(!stack_device.empty()){
            id_intersection pop_result = stack_device.pop();
            int32_t node_id = pop_result.id;
            float2 intersection = pop_result.intersection;
            int32_t* node = nodes + node_id * 5;
            if(node[4] <= 4){
                IndexStack<int32_t> stack2;
                stack2.push(node_id);
                int32_t count2 = 0;
                int32_t count2_total = node[4];
                while(!stack2.empty()){
                    int32_t node_id = stack2.pop();
                    int32_t* node = nodes + node_id * 5;
                    if(node[3] >= 0){

                        int32_t object_id = node[3];
                        float t = ray_intersects(means3D[object_id], ray_o, ray_d);
                        // float t = ray_intersects(means3D[object_id], covs3D + object_id*6, ray_o, ray_d);
                        if(t < 0.01 || t < intersection.x || t > intersection.y){
                            t = 1000000.f;
                            object_id = -1;
                        }
                        uint64_t key = idx;
                        key <<= 32;
                        key |= *(uint32_t*)&t;
                        ray_key_ptr[count] = key;
                        point_ptr[count] = object_id;
                        ray_id_ptr[count] = idx;
                        position_ptr[count] = {
                                ray_o.x + t * ray_d.x,
                                ray_o.y + t * ray_d.y,
                                ray_o.z + t * ray_d.z,
                        };
                        ++count2;
                        ++count;
                    }
                    else{
                        stack2.push(node[1]);
                        stack2.push(node[2]);
                    }
                }
                assert(count2 == count2_total);



            }
            else{
                int32_t lid = node[1], rid = node[2];
                float2 interection_l = ray_intersects(aabbs_internal[lid], ray_o, ray_d);
                float2 interection_r = ray_intersects(aabbs_internal[rid], ray_o, ray_d);
                if(interection_l.y > interection_r.y){
                    if(interection_l.y > 0){
                        stack_device.push({ lid, interection_l });
                    }
                    if(interection_r.y > 0){
                        stack_device.push({ rid, interection_r });
                    }
                }
                else{
                    if(interection_r.y > 0){
                        stack_device.push({ rid, interection_r });
                    }
                    if(interection_l.y > 0){
                        stack_device.push({ lid, interection_l });
                    }
                }
            }
        }
    });
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "trace time: " << milliseconds << " ms" << std::endl;
    cudaEventRecord(start);
    thrust::stable_sort_by_key(ray_list_key_vec.begin(),
        ray_list_key_vec.end(),
        thrust::make_zip_iterator(
            thrust::make_tuple(point_list_vec.begin(),
                position_list_vec.begin())));
    //     cudaEventRecord(stop);
    //     cudaEventSynchronize(stop);
    //     cudaEventElapsedTime(&milliseconds, start, stop);
    //     std::cout << "sort time: " << milliseconds << " ms" << std::endl;
    //     cudaEventRecord(start);
    return std::make_tuple(num_rendered, point_list_vec, position_list_vec, ray_id_list_vec);
}


void trace_bvh_opacity_cuda(int32_t num_rays, int32_t D, int32_t M, int32_t* nodes, float* aabbs,
    float3* rays_o, float3* rays_d,
    float3* means3D, float* covs3D,
    float* opacities,
    float* shs,
    int32_t* num_contributes,
    float* rendered_opacity,
    float* rendered_tvalue,
    float* rendered_intensity,
    float* rendered_raydrop){
    //         cudaEvent_t start, stop;
    //         cudaEventCreate(&start);
    //         cudaEventCreate(&stop);
    //         cudaEventRecord(start);
    //         float milliseconds = 0;
    auto* aabbs_internal = reinterpret_cast<aabb_type*>(aabbs);
    thrust::for_each(thrust::device,
        thrust::make_counting_iterator<int32_t>(0),
        thrust::make_counting_iterator<int32_t>(num_rays),
        [D, M, nodes, aabbs_internal, rays_o, rays_d, num_contributes,
        means3D, covs3D, opacities, shs, rendered_opacity, rendered_tvalue, rendered_intensity, rendered_raydrop] __device__(int32_t idx){
        IndexStack<int32_t> stack_device;
        stack_device.push(0);
        int32_t count = 0;
        float3 ray_o = rays_o[idx], ray_d = rays_d[idx];
        float ray_opacity = 1.0f;
        float t_value = 0.0f; // added
        float intensity = 0.0f; // added
        float raydrop = 0.0f; // added
        while(!stack_device.empty()){
            int32_t node_id = stack_device.pop();
            int32_t* node = nodes + node_id * 5;
            if(node[4] <= 1){  // 8
                IndexStack<int32_t> stack_device2;
                stack_device2.push(node_id);
                while(!stack_device2.empty()){
                    int32_t node_id2 = stack_device2.pop();
                    int32_t* node2 = nodes + node_id2 * 5;
                    if(node2[4] > 1){
                        stack_device2.push(node2[1]);
                        stack_device2.push(node2[2]);
                    }
                    else{
                        int32_t object_id2 = node2[3];
                        if(opacities[object_id2] < 1.f / 255.f) continue;
                        
                        float t = ray_intersects(means3D[object_id2], covs3D + object_id2 * 6, ray_o, ray_d);
                        if(t < 0.01){
                            continue;
                        }

                        float2 features = computeFeaturesFromSH(object_id2, D, M, (float3*)means3D, ray_o, shs);

                        float3 pos = {
                                        ray_o.x + t * ray_d.x,
                                        ray_o.y + t * ray_d.y,
                                        ray_o.z + t * ray_d.z,
                        };
                        float power = gaussian_fn(means3D[object_id2], pos, covs3D + object_id2 * 6);
                        if(power > 0) continue;
                        count += 1;
                        float alpha = opacities[object_id2] * __expf(power);
                        t_value += ray_opacity * alpha * t;
                        intensity += ray_opacity * alpha * features.x;
                        raydrop += ray_opacity * alpha * features.y;
                        ray_opacity *= 1 - alpha;
                        if(ray_opacity < 0.0001f){
                            break;
                        }
                    }
                }
            }
            else{
                int32_t lid = node[1], rid = node[2];
                float2 interection_l = ray_intersects(aabbs_internal[lid], ray_o, ray_d);
                float2 interection_r = ray_intersects(aabbs_internal[rid], ray_o, ray_d);
                if(interection_l.y > interection_r.y){
                    if(interection_l.y > 0){
                        stack_device.push(lid);
                    }
                    if(interection_r.y > 0){
                        stack_device.push(rid);
                    }
                }                
                else{
                    if(interection_r.y > 0){
                        stack_device.push(rid);
                    }
                    if(interection_l.y > 0){
                        stack_device.push(lid);
                    }
                }
            }
        }
        num_contributes[idx] = count;
        rendered_tvalue[idx] = t_value / (1 - ray_opacity);
        rendered_opacity[idx] = 1 - ray_opacity;
        rendered_intensity[idx] = intensity;
        rendered_raydrop[idx] = raydrop;
    });

    //     cudaEventRecord(stop);
    //     cudaEventSynchronize(stop);
    //     cudaEventElapsedTime(&milliseconds, start, stop);
    //     std::cout << "tracing time: " << milliseconds << " ms" << std::endl;
    //     cudaEventRecord(start);
}