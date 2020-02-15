#ifndef RAYTRIANGLEINTERSECT_H
#define RAYTRIANGLEINTERSECT_H

#include <cuda_runtime.h>

	__device__ int intersect_triangle3(float orig[3], float dir[3],
		float vert0[3], float vert1[3], float vert2[3],
		float* t, float* u, float* v);

	int intersect_triangleCPU(float orig[3], float dir[3],
		float vert0[3], float vert1[3], float vert2[3],
		float* t, float* u, float* v);

	__global__ void intersect_triangleGPU_BlockPerOrigin(float3* origins, float dir[3],
		int3* triangles, float3* vertices, int numberOfTriangles, bool* inside);

	__global__ void intersect_triangleGPU_ThreadPerOrigin(float3* origins, float dir[3],
		int3* triangles, float3* vertices, int numberOfOrigins, int numberOfTriangles, bool* inside);

	__global__ void intersect_triangleGPU_ThreadPerTriangle(float3* origins, float dir[3],
		int3* triangles, float3* vertices, int numberOfOrigins, int numberOfTriangles, int* intersectionsPerOrigin);
#endif
