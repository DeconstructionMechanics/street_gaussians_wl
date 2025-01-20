#ifndef BVH_UTILITY_CUH
#define BVH_UTILITY_CUH
#include <vector_types.h>
#include <math_constants.h>

struct aabb_type{
    float3 lower;
    float3 upper;
};

__device__ __host__
inline float3 centroid(const aabb_type& box) noexcept
{
    float3 c;
    c.x = (box.upper.x + box.lower.x) * 0.5;
    c.y = (box.upper.y + box.lower.y) * 0.5;
    c.z = (box.upper.z + box.lower.z) * 0.5;
    return c;
}

__device__ __host__
inline aabb_type merge(const aabb_type& lhs, const aabb_type& rhs) noexcept
{
    aabb_type merged{};
    merged.upper.x = fmaxf(lhs.upper.x, rhs.upper.x);
    merged.upper.y = fmaxf(lhs.upper.y, rhs.upper.y);
    merged.upper.z = fmaxf(lhs.upper.z, rhs.upper.z);
    merged.lower.x = fminf(lhs.lower.x, rhs.lower.x);
    merged.lower.y = fminf(lhs.lower.y, rhs.lower.y);
    merged.lower.z = fminf(lhs.lower.z, rhs.lower.z);
    return merged;
}

__device__ __host__
inline float2 ray_intersects(const aabb_type& lhs, const float3& ray_o, const float3& ray_d) noexcept
{
    float tmin = (lhs.lower.x - ray_o.x) / ray_d.x;
    float tmax = (lhs.upper.x - ray_o.x) / ray_d.x;

    if (tmin > tmax) {
        thrust::swap(tmin, tmax);
    }

    float tymin = (lhs.lower.y - ray_o.y) / ray_d.y;
    float tymax = (lhs.upper.y - ray_o.y) / ray_d.y;

    if (tymin > tymax) {
        thrust::swap(tymin, tymax);
    }

    if (tmin > tymax || tymin > tmax) {
        return { -1.0f, -1.0f };
    }

    if (tymin > tmin) {
        tmin = tymin;
    }

    if (tymax < tmax) {
        tmax = tymax;
    }

    float tzmin = (lhs.lower.z - ray_o.z) / ray_d.z;
    float tzmax = (lhs.upper.z - ray_o.z) / ray_d.z;

    if (tzmin > tzmax) {
        thrust::swap(tzmin, tzmax);
    }

    if (tmin > tzmax || tzmin > tmax) {
        return { -1.0f, -1.0f };
    }

    if (tzmin > tmin) {
        tmin = tzmin;
    }

    if (tzmax < tmax) {
        tmax = tzmax;
    }
    return { tmin, tmax };
}

__device__ __host__
inline float ray_intersects(const float3 mean, const float3& ray_o, const float3& ray_d) noexcept
{
    return (mean.x - ray_o.x)*ray_d.x + (mean.y - ray_o.y)*ray_d.y + (mean.z - ray_o.z)*ray_d.z;
}

__device__ __host__
inline float ray_intersects(const float3 mean, const float* cov3D_inverse, const float3& ray_o, const float3& ray_d) noexcept
{
     float3 miu = {mean.x - ray_o.x, mean.y - ray_o.y, mean.z - ray_o.z};
     float t1=cov3D_inverse[0]*miu.x*ray_d.x+cov3D_inverse[1]*miu.x*ray_d.y+cov3D_inverse[2]*miu.x*ray_d.z+
             cov3D_inverse[1]*miu.y*ray_d.x+cov3D_inverse[3]*miu.y*ray_d.y+cov3D_inverse[4]*miu.y*ray_d.z+
             cov3D_inverse[2]*miu.z*ray_d.x+cov3D_inverse[4]*miu.z*ray_d.y+cov3D_inverse[5]*miu.z*ray_d.z;
     float t2=cov3D_inverse[0]*ray_d.x*ray_d.x+cov3D_inverse[1]*ray_d.x*ray_d.y+cov3D_inverse[2]*ray_d.x*ray_d.z+
             cov3D_inverse[1]*ray_d.y*ray_d.x+cov3D_inverse[3]*ray_d.y*ray_d.y+cov3D_inverse[4]*ray_d.y*ray_d.z+
             cov3D_inverse[2]*ray_d.z*ray_d.x+cov3D_inverse[4]*ray_d.z*ray_d.y+cov3D_inverse[5]*ray_d.z*ray_d.z;
    return t1 / t2;
}

__device__ __host__
inline float gaussian_fn(const float3 mean, const float3& pos, const float* cov3D_inverse) noexcept
{
    float3 d = {mean.x-pos.x,mean.y-pos.y,mean.z-pos.z};
    float power = -0.5 * (d.x*d.x*cov3D_inverse[0] + d.y*d.y*cov3D_inverse[3] + d.z*d.z*cov3D_inverse[5] +
                          2*d.x*d.y*cov3D_inverse[1] + 2*d.x*d.z*cov3D_inverse[2] + 2*d.y*d.z*cov3D_inverse[4]);
    return power;
}

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

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__host__ __device__ inline float2& operator+=(float2& lhs, const float2& rhs) {
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    return lhs;
}

#endif //BVH_UTILITY_CUH