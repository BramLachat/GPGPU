#ifndef RAYTRIANGLEINTERSECT_H
#define RAYTRIANGLEINTERSECT_H

#include <cuda_runtime.h>

namespace Intersection {

	int intersect_triangle(float orig[3], float dir[3],
		float vert0[3], float vert1[3], float vert2[3],
		float* t, float* u, float* v);
	
	int intersect_triangle1(double orig[3], double dir[3],
		double vert0[3], double vert1[3], double vert2[3],
		double* t, double* u, double* v);
	
	int intersect_triangle2(double orig[3], double dir[3],
		double vert0[3], double vert1[3], double vert2[3],
		double* t, double* u, double* v);
	
	__device__ int intersect_triangle3(float orig[3], float dir[3],
		float vert0[3], float vert1[3], float vert2[3],
		float* t, float* u, float* v);

	int intersect_triangleCPU(float orig[3], float dir[3],
		float vert0[3], float vert1[3], float vert2[3],
		float* t, float* u, float* v);

	__global__ void intersect_triangle4(float orig[3], float dir[3],
		int* triangles, float* vertices, int* result, int* numberOfCalculations);

	__global__ void intersect_triangleGPU(float3* origins, float dir[3],
		int3* triangles, float3* vertices, int numberOfOrigins, int numberOfTriangles, int* intersectionsPerOrigin, float3* d_outsideVertices);
}

#endif
