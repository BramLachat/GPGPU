#pragma once

#include <cuda_runtime.h>

__host__ __device__ int NoDivTriTriIsect(float V0[3], float V1[3], float V2[3],
	float U0[3], float U1[3], float U2[3]);

__host__ __device__ int coplanar_tri_tri(float N[3], float V0[3], float V1[3], float V2[3],
	float U0[3], float U1[3], float U2[3]);

__global__ void triangle_triangle_GPU_BPCD_ThreadPerInnerTriangle(int3* cudaInsideTriangles, float3* cudaInsideVertices, int3* cudaOutsideTriangles, float3* cudaOutsideVertices, bool* inside, int numberOfInsideTriangles, int numberOfOutsideTriangles, float2* cudaOutsideTriangleIntervals);

__global__ void triangle_triangle_GPU_ThreadPerInnerTriangle(int3* cudaInsideTriangles, float3* cudaInsideVertices, int3* cudaOutsideTriangles, float3* cudaOutsideVertices, bool* inside, int numberOfInsideTriangles, int numberOfOutsideTriangles);

__global__ void triangle_triangle_GPU_BPCD_BlockPerInnerTriangle(int3* cudaInsideTriangles, float3* cudaInsideVertices, int3* cudaOutsideTriangles, float3* cudaOutsideVertices, bool* inside, int numberOfInsideTriangles, int numberOfOutsideTriangles, float2* cudaOutsideTriangleIntervals);

__global__ void triangle_triangle_GPU_BlockPerInnerTriangle(int3* cudaInsideTriangles, float3* cudaInsideVertices, int3* cudaOutsideTriangles, float3* cudaOutsideVertices, bool* inside, int numberOfInsideTriangles, int numberOfOutsideTriangles);

__global__ void triangle_triangle_GPU_BPCD_ThreadPerOuterTriangle(int3* cudaInsideTriangles, float3* cudaInsideVertices, int3* cudaOutsideTriangles, float3* cudaOutsideVertices, bool* inside, int numberOfInsideTriangles, int numberOfOutsideTriangles, float2* cudaInsideTriangleIntervals);

__global__ void triangle_triangle_GPU_ThreadPerOuterTriangle(int3* cudaInsideTriangles, float3* cudaInsideVertices, int3* cudaOutsideTriangles, float3* cudaOutsideVertices, bool* inside, int numberOfInsideTriangles, int numberOfOutsideTriangles);