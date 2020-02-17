/* Ray-Triangle Intersection Test Routines          */
/* Different optimizations of my and Ben Trumbore's */
/* code from journals of graphics tools (JGT)       */
/* http://www.acm.org/jgt/                          */
/* by Tomas Moller, May 2000                        */

#include <math.h>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "RayTriangleIntersect.cuh"

#define EPSILON 0.000001
#define CROSS(dest,v1,v2) \
          dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; \
          dest[1]=v1[2]*v2[0]-v1[0]*v2[2]; \
          dest[2]=v1[0]*v2[1]-v1[1]*v2[0];
#define DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])
#define SUB(dest,v1,v2) \
          dest[0]=v1[0]-v2[0]; \
          dest[1]=v1[1]-v2[1]; \
          dest[2]=v1[2]-v2[2]; 

	/* code rewritten to do tests on the sign of the determinant */
	/* the division is before the test of the sign of the det    */
	/* and one CROSS has been moved out from the if-else if-else */
	__device__ int intersect_triangle3(float orig[3], float dir[3],
		float vert0[3], float vert1[3], float vert2[3],
		float* t, float* u, float* v)
	{
		//std::cout << "vert0 = " << vert0[0] << ", " << vert0[1] << ", " << vert0[2] << std::endl;
		//std::cout << "vert1 = " << vert1[0] << ", " << vert1[1] << ", " << vert1[2] << std::endl;
		//std::cout << "vert2 = " << vert2[0] << ", " << vert2[1] << ", " << vert2[2] << std::endl;
		//std::cout << "orig = " << orig[0] << ", " << orig[1] << ", " << orig[2] << std::endl;

		float edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
		float det, inv_det;

		/* find vectors for two edges sharing vert0 */
		SUB(edge1, vert1, vert0);
		SUB(edge2, vert2, vert0);

		/* begin calculating determinant - also used to calculate U parameter */
		CROSS(pvec, dir, edge2);

		/* if determinant is near zero, ray lies in plane of triangle */
		det = DOT(edge1, pvec);

		/* calculate distance from vert0 to ray origin */
		SUB(tvec, orig, vert0);
		inv_det = 1.0 / det;

		CROSS(qvec, tvec, edge1);

		if (det > EPSILON)
		{
			*u = DOT(tvec, pvec);
			if (*u < 0.0 || *u > det)
				return 0;

			/* calculate V parameter and test bounds */
			*v = DOT(dir, qvec);
			if (*v < 0.0 || *u + *v > det)
				return 0;

		}
		else if (det < -EPSILON)
		{
			/* calculate U parameter and test bounds */
			*u = DOT(tvec, pvec);
			if (*u > 0.0 || *u < det)
				return 0;

			/* calculate V parameter and test bounds */
			*v = DOT(dir, qvec);
			if (*v > 0.0 || *u + *v < det)
				return 0;
		}
		else return 0;  /* ray is parallell to the plane of the triangle */

		*t = DOT(edge2, qvec) * inv_det;
		(*u) *= inv_det;
		(*v) *= inv_det;

		if (*t > 0)
		{
			return 1;
		}
		else
		{
			return 0;
		}
	}

	/* code rewritten to do tests on the sign of the determinant */
	/* the division is before the test of the sign of the det    */
	/* and one CROSS has been moved out from the if-else if-else */
	int intersect_triangleCPU(float orig[3], float dir[3],
		float vert0[3], float vert1[3], float vert2[3],
		float* t, float* u, float* v)
	{
		//std::cout << "vert0 = " << vert0[0] << ", " << vert0[1] << ", " << vert0[2] << std::endl;
		//std::cout << "vert1 = " << vert1[0] << ", " << vert1[1] << ", " << vert1[2] << std::endl;
		//std::cout << "vert2 = " << vert2[0] << ", " << vert2[1] << ", " << vert2[2] << std::endl;
		//std::cout << "orig = " << orig[0] << ", " << orig[1] << ", " << orig[2] << std::endl;

		float edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
		float det, inv_det;

		/* find vectors for two edges sharing vert0 */
		SUB(edge1, vert1, vert0);
		SUB(edge2, vert2, vert0);

		/* begin calculating determinant - also used to calculate U parameter */
		CROSS(pvec, dir, edge2);

		/* if determinant is near zero, ray lies in plane of triangle */
		det = DOT(edge1, pvec);

		/* calculate distance from vert0 to ray origin */
		SUB(tvec, orig, vert0);
		inv_det = 1.0 / det;

		CROSS(qvec, tvec, edge1);

		if (det > EPSILON)
		{
			*u = DOT(tvec, pvec);
			if (*u < 0.0 || *u > det)
				return 0;

			/* calculate V parameter and test bounds */
			*v = DOT(dir, qvec);
			if (*v < 0.0 || *u + *v > det)
				return 0;

		}
		else if (det < -EPSILON)
		{
			/* calculate U parameter and test bounds */
			*u = DOT(tvec, pvec);
			if (*u > 0.0 || *u < det)
				return 0;

			/* calculate V parameter and test bounds */
			*v = DOT(dir, qvec);
			if (*v > 0.0 || *u + *v < det)
				return 0;
		}
		else return 0;  /* ray is parallell to the plane of the triangle */

		*t = DOT(edge2, qvec) * inv_det;
		(*u) *= inv_det;
		(*v) *= inv_det;

		if (*t > 0)
		{
			return 1;
		}
		else
		{
			return 0;
		}
	}

	//block per origin
	__global__ void intersect_triangleGPU_BlockPerOrigin(float3* origins, float dir[3],
		int3* triangles, float3* vertices, int numberOfTriangles, bool* inside) // , int* intersectionsPerOrigin, float3* outsideVertices
	{
		int threadidx = threadIdx.x;
		float orig[3] = { origins[blockIdx.x].x, origins[blockIdx.x].y, origins[blockIdx.x].z };

		__shared__ int intersectionsPerBlock[128];	//!!!Threads per block moet een macht van 2 zijn!!!
													//zoniet krijg je problemen met lijn 494 (i /= 2)
		int numberOfIntersections = 0;
		int punt1;
		int punt2;
		int punt3;
		while (threadidx < numberOfTriangles) {
			//if (*inside) {
				punt1 = triangles[threadidx].x;
				punt2 = triangles[threadidx].y;
				punt3 = triangles[threadidx].z;
				float vert0[3] = { vertices[punt1].x, vertices[punt1].y, vertices[punt1].z };
				float vert1[3] = { vertices[punt2].x, vertices[punt2].y, vertices[punt2].z };
				float vert2[3] = { vertices[punt3].x, vertices[punt3].y, vertices[punt3].z };
				float t, u, v;
				if (intersect_triangle3(orig, dir, vert0, vert1, vert2, &t, &u, &v) == 1)
				{
					numberOfIntersections += 1;
				}
				threadidx += 128;
			/*}
			else {
				return;
			}*/
		}
		threadidx = threadIdx.x;
		intersectionsPerBlock[threadidx] = numberOfIntersections;
		__syncthreads();
		int i = blockDim.x / 2;
		while (i != 0) {
			if (threadidx < i) {
				intersectionsPerBlock[threadidx] += intersectionsPerBlock[threadidx + i];
			}
			__syncthreads();
			i /= 2;
		}
		if (threadidx == 0) {
			//intersectionsPerOrigin[blockIdx.x] = intersectionsPerBlock[0];
			if (intersectionsPerBlock[0] % 2 == 0)
			{
				*inside = false;
				//return;
				/*outsideVertices[blockIdx.x].x = orig[0];
				outsideVertices[blockIdx.x].y = orig[1];
				outsideVertices[blockIdx.x].z = orig[2];*/
			}
		}
	}

	//thread per origin
	__global__ void intersect_triangleGPU_ThreadPerOrigin(float3* origins, float dir[3],
		int3* triangles, float3* vertices, int numberOfOrigins, int numberOfTriangles, bool* inside)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid < numberOfOrigins)
		{
			float orig[3] = { origins[tid].x, origins[tid].y, origins[tid].z };
			int numberOfIntersections = 0;
			for (int i = 0; i < numberOfTriangles; i++)
			{
				//if (*inside) {
					float vert0[3] = { vertices[triangles[i].x].x, vertices[triangles[i].x].y, vertices[triangles[i].x].z };
					float vert1[3] = { vertices[triangles[i].y].x, vertices[triangles[i].y].y, vertices[triangles[i].y].z };
					float vert2[3] = { vertices[triangles[i].z].x, vertices[triangles[i].z].y, vertices[triangles[i].z].z };
					float t, u, v;
					if (intersect_triangle3(orig, dir, vert0, vert1, vert2, &t, &u, &v) == 1)
					{
						numberOfIntersections++;
					}
				/*}
				else {
					return;
				}*/
			}
			//intersectionsPerOrigin[tid] = numberOfIntersections;
			if (numberOfIntersections % 2 == 0)
			{
				*inside = false;
				//return;
				/*outsideVertices[tid].x = orig[0];
				outsideVertices[tid].y = orig[1];
				outsideVertices[tid].z = orig[2];*/
			}
		}
	}

	//thread per triangle
	__global__ void intersect_triangleGPU_ThreadPerTriangle(float3* origins, float dir[3],
		int3* triangles, float3* vertices, int numberOfOrigins, int numberOfTriangles, int* intersectionsPerOrigin)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		__shared__ int intersectionsPerBlock[128];
		intersectionsPerBlock[threadIdx.x] = 0;

		if (tid < numberOfTriangles)
		{
			float vert0[3] = { vertices[triangles[tid].x].x, vertices[triangles[tid].x].y, vertices[triangles[tid].x].z };
			float vert1[3] = { vertices[triangles[tid].y].x, vertices[triangles[tid].y].y, vertices[triangles[tid].y].z };
			float vert2[3] = { vertices[triangles[tid].z].x, vertices[triangles[tid].z].y, vertices[triangles[tid].z].z };
			int i = 0;
			while (i < numberOfOrigins)
			{
				float orig[3] = { origins[i].x, origins[i].y, origins[i].z };
				float t, u, v;
				if (intersect_triangle3(orig, dir, vert0, vert1, vert2, &t, &u, &v) == 1)
				{
					intersectionsPerBlock[threadIdx.x] = 1;
				}
				__syncthreads();
				int j = blockDim.x / 2;
				while (j != 0) {
					if (threadIdx.x < j) {
						intersectionsPerBlock[threadIdx.x] += intersectionsPerBlock[threadIdx.x + j]; // intersectionsPerBlock[]: Index = 0 houdt de som van alle threads binnen deze block bij
					}
					__syncthreads();
					j /= 2;
				}
				if (threadIdx.x == 0) {
					atomicAdd(&intersectionsPerOrigin[i], intersectionsPerBlock[0]);
				}
				// Als niet alle blocks tegelijk kunnen worden uitgevoerd dan zal het resultaat dat in 'intersectionsPerOrigin[i]' zit nog niet volledig zijn als deze wordt opgevraagd.
				// Dit kan zorgen voor verkeerde resultaten als het tussenresultaat toevallig even zou zijn.
				/*if (threadIdx.x == 0) {
					if (intersectionsPerOrigin[i] % 2 == 0) {
						inside = false;
					}
				}*/
				intersectionsPerBlock[threadIdx.x] = 0;
				i++;
			}
		}
	}