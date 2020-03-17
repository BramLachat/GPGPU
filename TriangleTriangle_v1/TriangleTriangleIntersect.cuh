#pragma once

#include <cuda_runtime.h>

__host__ __device__ int NoDivTriTriIsect(float V0[3], float V1[3], float V2[3],
	float U0[3], float U1[3], float U2[3]);

__host__ __device__ int coplanar_tri_tri(float N[3], float V0[3], float V1[3], float V2[3],
	float U0[3], float U1[3], float U2[3]);

__host__ __device__ int BPCD(float V0[3], float V1[3], float V2[3],
	float U0[3], float U1[3], float U2[3]);

/*__host__ __device__ int BPCD_Bram(float V0[3], float V1[3], float V2[3],
	float U0[3], float U1[3], float U2[3]);*/ //Nice try, maar helaas!!! ;)

__global__ void triangle_triangle_GPU_BPCD_3_2(
	int3* cudaInsideTriangles, 
	float3* cudaInsideVertices, 
	int3* cudaOutsideTriangles, 
	float3* cudaOutsideVertices, 
	bool* inside, 
	int numberOfInsideTriangles, 
	int* intersectingTriangles, 
	int* triangleIndices);

__global__ void triangle_triangle_GPU_BPCD_3_1(
	int3* cudaInsideTriangles, 
	float3* cudaInsideVertices, 
	int3* cudaOutsideTriangles, 
	float3* cudaOutsideVertices, 
	int numberOfInsideTriangles, 
	int numberOfOutsideTriangles, 
	int* intersectingTriangles, 
	int* triangleIndices, 
	int* size);

__global__ void triangle_triangle_GPU(int3* cudaInsideTriangles, float3* cudaInsideVertices, int3* cudaOutsideTriangles, float3* cudaOutsideVertices, bool* inside, int numberOfInsideTriangles, int numberOfOutsideTriangles);

__global__ void triangle_triangle_GPU_BPCD_2_1(
	int3* cudaInsideTriangles, 
	float3* cudaInsideVertices, 
	int3* cudaOutsideTriangles, 
	float3* cudaOutsideVertices, 
	int numberOfInsideTriangles, 
	int numberOfOutsideTriangles, 
	int* intersectingTriangles);

__global__ void triangle_triangle_GPU_BPCD_2_2(
	int3* cudaInsideTriangles, 
	float3* cudaInsideVertices, 
	int3* cudaOutsideTriangles, 
	float3* cudaOutsideVertices, 
	bool* inside, 
	int numberOfInsideTriangles, 
	int* intersectingTriangles);

__global__ void triangle_triangle_GPU_BPCD_1_1(
	int3* cudaInsideTriangles, 
	float3* cudaInsideVertices, 
	int3* cudaOutsideTriangles, 
	float3* cudaOutsideVertices, 
	int numberOfInsideTriangles, 
	int numberOfOutsideTriangles, 
	int2* cudaIntervals);

__global__ void triangle_triangle_GPU_BPCD_1_2(
	int3* cudaInsideTriangles,
	float3* cudaInsideVertices,
	int3* cudaOutsideTriangles, 
	float3* cudaOutsideVertices, 
	bool* inside, 
	int numberOfInsideTriangles,
	int2* cudaIntervals);