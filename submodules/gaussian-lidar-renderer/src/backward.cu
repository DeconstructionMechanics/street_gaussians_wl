#include "backward.cuh"

// Backward pass for conversion of spherical harmonics to feature for
// each Gaussian.
__device__ void computeFeaturesFromSHBackward(int idx, int deg, int max_coeffs, const float3* means, float3 campos, const float* shs, const bool* clamped, float2 dL_dfeature, float3* dL_dmeans, float2* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	// accumulate(plus) gradients
	float3 pos = means[idx];
    float3 dir_orig = {
        pos.x - campos.x,
        pos.y - campos.y,
        pos.z - campos.z,
        };
    float dir_len = sqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    float3 dir = {
        dir_orig.x / dir_len,
        dir_orig.y / dir_len,
        dir_orig.z / dir_len,
        };
    

	float2* sh = ((float2*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	// float2 dL_dfeature = dL_dfeatures[idx];
	dL_dfeature.x *= clamped[0] ? 0 : 1;
	dL_dfeature.y *= clamped[1] ? 0 : 1;
	
	float2 dfeaturedx(0, 0);
	float2 dfeaturedy(0, 0);
	float2 dfeaturedz(0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	float2* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dfeaturedsh0 = SH_C0;
	dL_dsh[0] += {
        dfeaturedsh0 * dL_dfeature.x,
        dfeaturedsh0 * dL_dfeature.y
        };
	if (deg > 0)
	{
		float dfeaturedsh1 = -SH_C1 * y;
		float dfeaturedsh2 = SH_C1 * z;
		float dfeaturedsh3 = -SH_C1 * x;
		dL_dsh[1] += {
            dfeaturedsh1 * dL_dfeature.x,
            dfeaturedsh1 * dL_dfeature.y
            };
		dL_dsh[2] += {
            dfeaturedsh2 * dL_dfeature.x,
            dfeaturedsh2 * dL_dfeature.y
            };
		dL_dsh[3] += {
            dfeaturedsh3 * dL_dfeature.x,
            dfeaturedsh3 * dL_dfeature.y
            };

		dfeaturedx = {
            -SH_C1 * sh[3].x,
            -SH_C1 * sh[3].y
            };
		dfeaturedy = {
            -SH_C1 * sh[1].x,
            -SH_C1 * sh[1].y            
            };
		dfeaturedz = {
            SH_C1 * sh[2].x,
            SH_C1 * sh[2].y
            };

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dfeaturedsh4 = SH_C2[0] * xy;
			float dfeaturedsh5 = SH_C2[1] * yz;
			float dfeaturedsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dfeaturedsh7 = SH_C2[3] * xz;
			float dfeaturedsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] += {
                dfeaturedsh4 * dL_dfeature.x,
                dfeaturedsh4 * dL_dfeature.y
                };
			dL_dsh[5] += {
                dfeaturedsh5 * dL_dfeature.x,
                dfeaturedsh5 * dL_dfeature.y
                };
			dL_dsh[6] += {
                dfeaturedsh6 * dL_dfeature.x,
                dfeaturedsh6 * dL_dfeature.y
                };
			dL_dsh[7] += {
                dfeaturedsh7 * dL_dfeature.x,
                dfeaturedsh7 * dL_dfeature.y
                };
			dL_dsh[8] += {
                dfeaturedsh8 * dL_dfeature.x,
                dfeaturedsh8 * dL_dfeature.y
                };

			dfeaturedx += {
                SH_C2[0] * y * sh[4].x + SH_C2[2] * 2.f * -x * sh[6].x + SH_C2[3] * z * sh[7].x + SH_C2[4] * 2.f * x * sh[8].x,
                SH_C2[0] * y * sh[4].y + SH_C2[2] * 2.f * -x * sh[6].y + SH_C2[3] * z * sh[7].y + SH_C2[4] * 2.f * x * sh[8].y
                };
			dfeaturedy += {
                SH_C2[0] * x * sh[4].x + SH_C2[1] * z * sh[5].x + SH_C2[2] * 2.f * -y * sh[6].x + SH_C2[4] * 2.f * -y * sh[8].x,
                SH_C2[0] * x * sh[4].y + SH_C2[1] * z * sh[5].y + SH_C2[2] * 2.f * -y * sh[6].y + SH_C2[4] * 2.f * -y * sh[8].y
                };
			dfeaturedz += {
                SH_C2[1] * y * sh[5].x + SH_C2[2] * 2.f * 2.f * z * sh[6].x + SH_C2[3] * x * sh[7].x,
                SH_C2[1] * y * sh[5].y + SH_C2[2] * 2.f * 2.f * z * sh[6].y + SH_C2[3] * x * sh[7].y
                };

			if (deg > 2)
			{
				float dfeaturedsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dfeaturedsh10 = SH_C3[1] * xy * z;
				float dfeaturedsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dfeaturedsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dfeaturedsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dfeaturedsh14 = SH_C3[5] * z * (xx - yy);
				float dfeaturedsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] += {
                    dfeaturedsh9 * dL_dfeature.x,
                    dfeaturedsh9 * dL_dfeature.y
                    };
				dL_dsh[10] += {
                    dfeaturedsh10 * dL_dfeature.x,
                    dfeaturedsh10 * dL_dfeature.y
                    };
				dL_dsh[11] += {
                    dfeaturedsh11 * dL_dfeature.x,
                    dfeaturedsh11 * dL_dfeature.y
                    };
				dL_dsh[12] += {
                    dfeaturedsh12 * dL_dfeature.x,
                    dfeaturedsh12 * dL_dfeature.y
                    };
				dL_dsh[13] += {
                    dfeaturedsh13 * dL_dfeature.x,
                    dfeaturedsh13 * dL_dfeature.y
                    };
				dL_dsh[14] += {
                    dfeaturedsh14 * dL_dfeature.x,
                    dfeaturedsh14 * dL_dfeature.y
                    };
				dL_dsh[15] += {
                    dfeaturedsh15 * dL_dfeature.x,
                    dfeaturedsh15 * dL_dfeature.y
                    };

				dfeaturedx += {(
					SH_C3[0] * sh[9].x * 3.f * 2.f * xy +
					SH_C3[1] * sh[10].x * yz +
					SH_C3[2] * sh[11].x * -2.f * xy +
					SH_C3[3] * sh[12].x * -3.f * 2.f * xz +
					SH_C3[4] * sh[13].x * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14].x * 2.f * xz +
					SH_C3[6] * sh[15].x * 3.f * (xx - yy)),
                    (
					SH_C3[0] * sh[9].y * 3.f * 2.f * xy +
					SH_C3[1] * sh[10].y * yz +
					SH_C3[2] * sh[11].y * -2.f * xy +
					SH_C3[3] * sh[12].y * -3.f * 2.f * xz +
					SH_C3[4] * sh[13].y * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14].y * 2.f * xz +
					SH_C3[6] * sh[15].y * 3.f * (xx - yy))};

				dfeaturedy += {(
					SH_C3[0] * sh[9].x * 3.f * (xx - yy) +
					SH_C3[1] * sh[10].x * xz +
					SH_C3[2] * sh[11].x * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12].x * -3.f * 2.f * yz +
					SH_C3[4] * sh[13].x * -2.f * xy +
					SH_C3[5] * sh[14].x * -2.f * yz +
					SH_C3[6] * sh[15].x * -3.f * 2.f * xy),
                    (
					SH_C3[0] * sh[9].y * 3.f * (xx - yy) +
					SH_C3[1] * sh[10].y * xz +
					SH_C3[2] * sh[11].y * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12].y * -3.f * 2.f * yz +
					SH_C3[4] * sh[13].y * -2.f * xy +
					SH_C3[5] * sh[14].y * -2.f * yz +
					SH_C3[6] * sh[15].y * -3.f * 2.f * xy)};

				dfeaturedz += {(
					SH_C3[1] * sh[10].x * xy +
					SH_C3[2] * sh[11].x * 4.f * 2.f * yz +
					SH_C3[3] * sh[12].x * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13].x * 4.f * 2.f * xz +
					SH_C3[5] * sh[14].x * (xx - yy)),
                    (
					SH_C3[1] * sh[10].y * xy +
					SH_C3[2] * sh[11].y * 4.f * 2.f * yz +
					SH_C3[3] * sh[12].y * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13].y * 4.f * 2.f * xz +
					SH_C3[5] * sh[14].y * (xx - yy))};
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	float3 dL_ddir((dfeaturedx.x * dL_dfeature.x + dfeaturedx.y * dL_dfeature.y), (dfeaturedy.x * dL_dfeature.x + dfeaturedy.y * dL_dfeature.y), (dfeaturedz.x * dL_dfeature.x + dfeaturedz.y * dL_dfeature.y));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(dir_orig, dL_ddir);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += dL_dmean;
}



// void backward_shs_cuda(int32_t num_rays, int32_t D, int32_t M, int32_t G,
//     int32_t* contribute_gid, bool* contribute_clamp,
//     float* grad_intensityprime, float* grad_raydropprime,
//     float3* rays_o, float3* rays_d,
//     float3* means3D,
//     float* shs,
//     float* grad_shs_from_shs,
//     float* grad_means_from_shs){
//     thrust::for_each(thrust::device,
//         thrust::make_counting_iterator<int32_t>(0),
//         thrust::make_counting_iterator<int32_t>(num_rays),
//         [D, M, contribute_gid, contribute_clamp, grad_intensityprime, grad_raydropprime, rays_o, rays_d,
//         means3D, shs, grad_shs_from_shs, grad_means_from_shs] __device__(int32_t idx){
        
//         float3 ray_o = rays_o[idx], ray_d = rays_d[idx];
//         for (int32_t iG = 0; iG < G; iG++){
//             int32_t gaussian_idx = *(contribute_gid + idx * G + iG);
//             if (gaussian_idx < 0){
//                 continue;
//             }
//             bool clamped = *(contribute_clamp + idx * G + iG);
//             float2 dL_dfeature = {*(grad_intensityprime + idx * G + iG), *(grad_raydropprime + idx * G + iG)};
//             computeFeaturesFromSHBackward(gaussian_idx, D, M, (float3*)means3D, ray_o, shs, clamped, dL_dfeature, (float3*)grad_means_from_shs, (float2*)grad_shs_from_shs);
//         }
//     }); 
// }

#define DELTA 0.0001f

void backward_trace_cuda(int32_t num_rays, int32_t D, int32_t M, int32_t G,
    int32_t* contribute_gid, float* contribute_T, bool* contribute_clamp,
	float* contribute_tprime, float* contribute_intensityprime, float* contribute_raydropprime,
    float* weights, float* tvalues, float* intensity, float* raydrop,
    float3* means3D, float* covs3D, float* opacity, float* shs,
	float3* rays_o, float3* rays_d,
	float* grad_up_weights, float* grad_up_tvalues, float* grad_up_intensity, float* grad_up_raydrop,
	float3* grad_means3D, float* grad_covs3D, float* grad_opacity, float2* grad_shs){
		// contribute_gid (num_rays, G), G sorted by depth
		// covs3D stripped to 6
    thrust::for_each(thrust::device,
        thrust::make_counting_iterator<int32_t>(0),
        thrust::make_counting_iterator<int32_t>(num_rays),
        [D, M, G,
		contribute_gid, contribute_T, contribute_clamp,
		contribute_tprime, contribute_intensityprime, contribute_raydropprime,
		weights, tvalues, intensity, raydrop,
        means3D, covs3D, opacity, shs,
		rays_o, rays_d,
		grad_up_weights, grad_up_tvalues, grad_up_intensity, grad_up_raydrop,
		grad_means3D, grad_covs3D, grad_opacity, grad_shs] __device__(int32_t ray_idx){
        
		float ray_weight = weights[ray_idx];
		float ray_tvalue = tvalues[ray_idx];
		float ray_intensity = intensity[ray_idx];
		float ray_raydrop = raydrop[ray_idx];
        float3 ray_o = rays_o[ray_idx];
		float3 ray_d = rays_d[ray_idx];
		float ray_grad_up_tvalue = grad_up_tvalues[ray_idx];
		float ray_grad_up_intensity = grad_up_intensity[ray_idx];
		float ray_grad_up_raydrop = grad_up_raydrop[ray_idx];
		
        for (int32_t iG = 0; iG < G; iG++){
            int32_t gaussian_idx = *(contribute_gid + idx * G + iG);
            if (gaussian_idx < 0){
                continue;
            }
			float T = *(contribute_T + idx * G + iG);
            bool* clamp = contribute_clamp +  2 * (idx * G + iG);
			float tprime = *(contribute_tprime + idx * G + iG);
			float intensityprime = *(contribute_intensityprime + idx * G + iG);
			float raydropprime = *(contribute_raydropprime + idx * G + iG);
			float* covs3D_ptr = covs3D + gaussian_idx * 6;
			float* grad_covs3D_ptr = grad_covs3D + gaussian_idx * 6;

			float3 pos = {
							ray_o.x + tprime * ray_d.x,
							ray_o.y + tprime * ray_d.y,
							ray_o.z + tprime * ray_d.z,
			};
			float exp_power = __expf(gaussian_fn(means3D[gaussian_idx], pos, covs3D_ptr))
			float alpha = opacity[gaussian_idx] * exp_power;

			// feature -> alpha
			float dtfeature_dalpha = T * tprime;
			float dintensity_dalpha = T * intensityprime;
			float draydrop_dalpha = T * raydropprime;
			float dposttfeature_dalpha = 0;
			float dpostintensity_dalpha = 0;
			float dpostraydrop_dalpha = 0;
			for (int32_t iG2 = 0; iG2 < G; iG2++){
				int32_t gaussian_idx2 = *(contribute_gid + idx * G + iG2);
				if (gaussian_idx2 < 0){
					continue;
				}
				float T2 = *(contribute_T + idx * G + iG2);
				float tprime2 = *(contribute_tprime + idx * G + iG2);
				float intensityprime2 = *(contribute_intensityprime + idx * G + iG2);
				float raydropprime2 = *(contribute_raydropprime + idx * G + iG2);
				float* covs3D_ptr2 = covs3D + gaussian_idx2 * 6;
				float3 pos2 = {
					ray_o.x + tprime2 * ray_d.x,
					ray_o.y + tprime2 * ray_d.y,
					ray_o.z + tprime2 * ray_d.z,
				};
				float alpha2 = opacity[gaussian_idx2] * __expf(gaussian_fn(means3D[gaussian_idx2], pos2, covs3D_ptr2));
				dposttfeature_dalpha -= T2 * alpha2 * tprime2;
				dpostintensity_dalpha -= T2 * alpha2 * intensityprime2;
				dpostraydrop_dalpha -= T2 * alpha2 * raydropprime2;
			}
			dtfeature_dalpha += dposttfeature_dalpha / fmaxf((1 - alpha), DELTA);
			dintensity_dalpha += dpostintensity_dalpha / fmaxf((1 - alpha), DELTA);
			draydrop_dalpha += dpostraydrop_dalpha / fmaxf((1 - alpha), DELTA);

			float dtvalue_dalpha = (dtfeature_dalpha - ray_tvalue * (1 - ray_weight) / fmaxf((1 - alpha), DELTA)) / fmaxf(ray_weight, DELTA);
			float grad_alpha = dtvalue_dalpha * ray_grad_up_tvalue + dintensity_dalpha * ray_grad_up_intensity + draydrop_dalpha * ray_grad_up_raydrop;

			// alpha -> opacity, means3D, covs3D
			grad_opacity[gaussian_idx] +=  exp_power * grad_alpha;
			float grad_power = grad_alpha;
			float3 d = {means3D[gaussian_idx].x - pos.x, means3D[gaussian_idx].y - pos.y, means3D[gaussian_idx].z - pos.z};
			grad_covs3D_ptr[0] += -0.5 * d.x * d.x * grad_power;
			grad_covs3D_ptr[1] += -1 * d.x * d.y * grad_power;
			grad_covs3D_ptr[2] += -1 * d.x * d.z * grad_power;
			grad_covs3D_ptr[3] += -0.5 * d.y * d.y * grad_power;
			grad_covs3D_ptr[4] += -1 * d.y * d.z * grad_power;
			grad_covs3D_ptr[5] += -0.5 * d.z * d.z * grad_power;
			float3 grad_d = {
				-1 * (covs3D_ptr[0] * d.x + covs3D_ptr[1] * d.y + covs3D_ptr[2] * d.z) * grad_power,
				-1 * (covs3D_ptr[1] * d.x + covs3D_ptr[3] * d.y + covs3D_ptr[4] * d.z) * grad_power,
				-1 * (covs3D_ptr[2] * d.x + covs3D_ptr[4] * d.y + covs3D_ptr[5] * d.z) * grad_power,
			};
			grad_means3D[gaussian_idx] += grad_d;
			float grad_tprime = -1 * (ray_d.x * grad_d.x + ray_d.y * grad_d.y + ray_d.z * grad_d.z);

			// feature -> featureprime
			float dfeature_dfeatureprime = T * alpha;
			grad_tprime += dfeature_dfeatureprime * ray_grad_up_tvalue / fmaxf(ray_weight, DELTA);
			float grad_intensityprime = dfeature_dfeatureprime * ray_grad_up_intensity;
			float grad_raydropprime = dfeature_dfeatureprime * ray_grad_up_raydrop;


			// tprime -> means3D
			float3 sigma_rd = {
				covs3D_ptr[0] * ray_d.x + covs3D_ptr[1] * ray_d.y + covs3D_ptr[2] * ray_d.z,
				covs3D_ptr[1] * ray_d.x + covs3D_ptr[3] * ray_d.y + covs3D_ptr[4] * ray_d.z,
				covs3D_ptr[2] * ray_d.x + covs3D_ptr[4] * ray_d.y + covs3D_ptr[5] * ray_d.z,
			};
			float rd_sigma_rd = ray_d.x * sigma_rd.x + ray_d.y * sigma_rd.y + ray_d.z * sigma_rd.z;
			grad_means3D[gaussian_idx] += {
				sigma_rd.x * grad_tprime / rd_sigma_rd,
				sigma_rd.y * grad_tprime / rd_sigma_rd,
				sigma_rd.z * grad_tprime / rd_sigma_rd,
			};

			// tprime -> covs3D
			float3 miu = {means3D[gaussian_idx].x - ray_o.x, means3D[gaussian_idx].y - ray_o.y, means3D[gaussian_idx].z - ray_o.z};
			float miu_sigma_rd = miu.x * sigma_rd.x + miu.y * sigma_rd.y + miu.z * sigma_rd.z;
			grad_covs3D_ptr[0] = (miu.x * ray_d.x * rd_sigma_rd - miu_sigma_rd * ray_d.x * ray_d.x) / (rd_sigma_rd * rd_sigma_rd) * grad_tprime;
			grad_covs3D_ptr[1] = ((miu.x * ray_d.y + miu.y * ray_d.x) * rd_sigma_rd - 2 * miu_sigma_rd * ray_d.x * ray_d.y) / (rd_sigma_rd * rd_sigma_rd) * grad_tprime;
			grad_covs3D_ptr[2] = ((miu.x * ray_d.z + miu.z * ray_d.x) * rd_sigma_rd - 2 * miu_sigma_rd * ray_d.x * ray_d.z) / (rd_sigma_rd * rd_sigma_rd) * grad_tprime;
			grad_covs3D_ptr[3] = (miu.y * ray_d.y * rd_sigma_rd - miu_sigma_rd * ray_d.y * ray_d.y) / (rd_sigma_rd * rd_sigma_rd) * grad_tprime;
			grad_covs3D_ptr[4] = ((miu.y * ray_d.z + miu.z * ray_d.y) * rd_sigma_rd - 2 * miu_sigma_rd * ray_d.y * ray_d.z) / (rd_sigma_rd * rd_sigma_rd) * grad_tprime;
			grad_covs3D_ptr[5] = (miu.z * ray_d.z * rd_sigma_rd - miu_sigma_rd * ray_d.z * ray_d.z) / (rd_sigma_rd * rd_sigma_rd) * grad_tprime;

			// featureprime -> sh
            float2 grad_feature = {grad_intensityprime, grad_raydropprime};
            computeFeaturesFromSHBackward(gaussian_idx, D, M, means3D, ray_o, shs, clamp, grad_feature, grad_means3D, grad_shs);
        }
    }); 
}
