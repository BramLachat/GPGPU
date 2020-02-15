#pragma once

#include <cuda_runtime.h>

/*__host__ __device__ int NoDivTriTriIsect(float V0[3], float V1[3], float V2[3],
	float U0[3], float U1[3], float U2[3]);

__host__ __device__ int coplanar_tri_tri(float N[3], float V0[3], float V1[3], float V2[3],
	float U0[3], float U1[3], float U2[3]);*/

	 // Three-dimensional Triangle-Triangle Overlap Test
__host__ __device__ int tri_tri_overlap_test_3d(float p1[3], float q1[3], float r1[3],
	float p2[3], float q2[3], float r2[3]);


// Three-dimensional Triangle-Triangle Overlap Test
// additionaly computes the segment of intersection of the two triangles if it exists. 
// coplanar returns whether the triangles are coplanar, 
// source and target are the endpoints of the line segment of intersection 
__host__ __device__ int tri_tri_intersection_test_3d(float p1[3], float q1[3], float r1[3],
	float p2[3], float q2[3], float r2[3],
	int* coplanar,
	float source[3], float target[3]);


__host__ __device__ int coplanar_tri_tri3d(float  p1[3], float  q1[3], float  r1[3],
	float  p2[3], float  q2[3], float  r2[3],
	float  N1[3], float  N2[3]);


// Two dimensional Triangle-Triangle Overlap Test
__host__ __device__ int tri_tri_overlap_test_2d(float p1[2], float q1[2], float r1[2],
	float p2[2], float q2[2], float r2[2]);

__global__ void triangle_triangle_GPU(int3* cudaInsideTriangles, float3* cudaInsideVertices, int3* cudaOutsideTriangles, float3* cudaOutsideVertices, bool* inside, int numberOfInsideTriangles, int numberOfOutsideTriangles, float2* cudaOutsideTriangleIntervals);