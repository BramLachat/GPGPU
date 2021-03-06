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
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <cooperative_groups.h>

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

namespace Intersection {

	/* the original jgt code */
	int intersect_triangle(float orig[3], float dir[3],
		float vert0[3], float vert1[3], float vert2[3],
		float* t, float* u, float* v)
	{
		double edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
		double det, inv_det;

		/* find vectors for two edges sharing vert0 */
		SUB(edge1, vert1, vert0);
		SUB(edge2, vert2, vert0);

		/* begin calculating determinant - also used to calculate U parameter */
		CROSS(pvec, dir, edge2);

		/* if determinant is near zero, ray lies in plane of triangle */
		det = DOT(edge1, pvec);

		if (det > -EPSILON && det < EPSILON)
			return 0;
		inv_det = 1.0 / det;

		/* calculate distance from vert0 to ray origin */
		SUB(tvec, orig, vert0);

		/* calculate U parameter and test bounds */
		*u = DOT(tvec, pvec) * inv_det;
		if (*u < 0.0 || *u > 1.0)
			return 0;

		/* prepare to test V parameter */
		CROSS(qvec, tvec, edge1);

		/* calculate V parameter and test bounds */
		*v = DOT(dir, qvec) * inv_det;
		if (*v < 0.0 || *u + *v > 1.0)
			return 0;

		/* calculate t, ray intersects triangle */
		*t = DOT(edge2, qvec) * inv_det;

		return 1;
	}


	/* code rewritten to do tests on the sign of the determinant */
	/* the division is at the end in the code                    */
	int intersect_triangle1(double orig[3], double dir[3],
		double vert0[3], double vert1[3], double vert2[3],
		double* t, double* u, double* v)
	{
		double edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
		double det, inv_det;

		/* find vectors for two edges sharing vert0 */
		SUB(edge1, vert1, vert0);
		SUB(edge2, vert2, vert0);

		/* begin calculating determinant - also used to calculate U parameter */
		CROSS(pvec, dir, edge2);

		/* if determinant is near zero, ray lies in plane of triangle */
		det = DOT(edge1, pvec);

		if (det > EPSILON)
		{
			/* calculate distance from vert0 to ray origin */
			SUB(tvec, orig, vert0);

			/* calculate U parameter and test bounds */
			*u = DOT(tvec, pvec);
			if (*u < 0.0 || *u > det)
				return 0;

			/* prepare to test V parameter */
			CROSS(qvec, tvec, edge1);

			/* calculate V parameter and test bounds */
			*v = DOT(dir, qvec);
			if (*v < 0.0 || *u + *v > det)
				return 0;

		}
		else if (det < -EPSILON)
		{
			/* calculate distance from vert0 to ray origin */
			SUB(tvec, orig, vert0);

			/* calculate U parameter and test bounds */
			*u = DOT(tvec, pvec);
			/*      printf("*u=%f\n",(float)*u); */
			/*      printf("det=%f\n",det); */
			if (*u > 0.0 || *u < det)
				return 0;

			/* prepare to test V parameter */
			CROSS(qvec, tvec, edge1);

			/* calculate V parameter and test bounds */
			*v = DOT(dir, qvec);
			if (*v > 0.0 || *u + *v < det)
				return 0;
		}
		else return 0;  /* ray is parallell to the plane of the triangle */


		inv_det = 1.0 / det;

		/* calculate t, ray intersects triangle */
		*t = DOT(edge2, qvec) * inv_det;
		(*u) *= inv_det;
		(*v) *= inv_det;

		return 1;
	}

	/* code rewritten to do tests on the sign of the determinant */
	/* the division is before the test of the sign of the det    */
	int intersect_triangle2(double orig[3], double dir[3],
		double vert0[3], double vert1[3], double vert2[3],
		double* t, double* u, double* v)
	{
		double edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
		double det, inv_det;

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

		if (det > EPSILON)
		{
			/* calculate U parameter and test bounds */
			*u = DOT(tvec, pvec);
			if (*u < 0.0 || *u > det)
				return 0;

			/* prepare to test V parameter */
			CROSS(qvec, tvec, edge1);

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

			/* prepare to test V parameter */
			CROSS(qvec, tvec, edge1);

			/* calculate V parameter and test bounds */
			*v = DOT(dir, qvec);
			if (*v > 0.0 || *u + *v < det)
				return 0;
		}
		else return 0;  /* ray is parallell to the plane of the triangle */

		/* calculate t, ray intersects triangle */
		*t = DOT(edge2, qvec) * inv_det;
		(*u) *= inv_det;
		(*v) *= inv_det;

		return 1;
	}

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

	__global__ void intersect_triangle4(float orig[3], float dir[3],
		int* triangles, float* vertices, int* result, int* numberOfCalculations) //hier bepaalde waarden nog eens uitprinten voor eenvoudig voorbeeld om te kijken of wel degelijk gebeurt wat je verwacht
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		/*for (int i = 0; i < 24; i++)
		{
			printf("vertices = %f\n", vertices[i]);
		}
		for (int i = 0; i < 36; i++)
		{
			printf("triangles = %d\n", triangles[i]);
		}*/
		/*if (tid == 0) 
		{
			printf("numberOfCalculations: %d", *numberOfCalculations);
		}*/
		if (tid < *numberOfCalculations)
		{
			float vert0[3] = { vertices[triangles[tid * 3] * 3], vertices[triangles[tid * 3] * 3 + 1], vertices[triangles[tid * 3] * 3 + 2] };
			float vert1[3] = { vertices[triangles[(tid * 3) + 1] * 3], vertices[triangles[(tid * 3) + 1] * 3 + 1], vertices[triangles[(tid * 3) + 1] * 3 + 2] };
			float vert2[3] = { vertices[triangles[(tid * 3) + 2] * 3], vertices[triangles[(tid * 3) + 2] * 3 + 1], vertices[triangles[(tid * 3) + 2] * 3 + 2] };
			//printf("vert0 = %f, %f, %f\n", vert0[0], vert0[1], vert0[2]);
			//printf("vert1 = %f, %f, %f\n", vert1[0], vert1[1], vert1[2]);
			//printf("vert2 = %f, %f, %f\n", vert2[0], vert2[1], vert2[2]);

			//float vert0[3] = { 1.0, 0.0, 0.0 };
			//float vert1[3] = { 0.0, 1.0, 0.0 };
			//float vert2[3] = { 0.0, 0.0, 1.0 };

			/*float newDir[3];
			newDir[0] = dir[0] - orig[0];
			newDir[1] = dir[1] - orig[1];
			newDir[2] = dir[2] - orig[2];*/

			float edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
			float det, inv_det;
			float t, u, v;

			/* find vectors for two edges sharing vert0 */
			SUB(edge1, vert1, vert0);
			SUB(edge2, vert2, vert0);

			/* begin calculating determinant - also used to calculate U parameter */
			//CROSS(pvec, newDir, edge2);
			CROSS(pvec, dir, edge2);

			/* if determinant is near zero, ray lies in plane of triangle */
			det = DOT(edge1, pvec);

			/* calculate distance from vert0 to ray origin */
			SUB(tvec, orig, vert0);
			inv_det = 1.0 / det;

			CROSS(qvec, tvec, edge1);

			if (det > EPSILON)
			{
				u = DOT(tvec, pvec);
				if (u < 0.0 || u > det) {
					result[tid] = 0;
					return;
				}

				/* calculate V parameter and test bounds */
				//v = DOT(newDir, qvec);
				v = DOT(dir, qvec);
				if (v < 0.0 || u + v > det) {
					result[tid] = 0;
					return;
				}

			}
			else if (det < -EPSILON)
			{
				/* calculate U parameter and test bounds */
				u = DOT(tvec, pvec);
				if (u > 0.0 || u < det) {
					result[tid] = 0;
					return;
				}

				/* calculate V parameter and test bounds */
				//v = DOT(newDir, qvec);
				v = DOT(dir, qvec);
				if (v > 0.0 || u + v < det) {
					result[tid] = 0;
					return;
				}
			}
			else
			{
				result[tid] = 0;  /* ray is parallell to the plane of the triangle */
				return;
			}

			t = DOT(edge2, qvec) * inv_det;
			(u) *= inv_det;
			(v) *= inv_det;

			if (t > 0)
			{
				result[tid] = 1;
				return;
			}
			else
			{
				result[tid] = 0;
				return;
			}
		}
		
	}

	// Kernel laten stoppen vanaf er 1 punt buiten ligt.
	// Dit lukt enkel als het aantal block beperkt blijft, anders ben je niet meer zeker dat deze parallel worden uitgevoerd.
	//__device__ bool inside = true;

	//thread per triangle
	__global__ void intersect_triangleGPU(float3* origins, float dir[3],
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

	//thread per triangle versie 2
	__global__ void intersect_triangleGPU_v2(float3* origins, float dir[3],
		int3* triangles, float3* vertices, int numberOfOrigins, int numberOfTriangles, int* intersectionsPerOrigin)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		//__shared__ int intersectionsPerBlock[128];
		//intersectionsPerBlock[threadIdx.x] = 0;

		if (tid < numberOfTriangles)
		{
			float vert0[3] = { vertices[triangles[tid].x].x, vertices[triangles[tid].x].y, vertices[triangles[tid].x].z };
			float vert1[3] = { vertices[triangles[tid].y].x, vertices[triangles[tid].y].y, vertices[triangles[tid].y].z };
			float vert2[3] = { vertices[triangles[tid].z].x, vertices[triangles[tid].z].y, vertices[triangles[tid].z].z };
			int i = tid%numberOfOrigins;
			int teller = 0;

			while (teller < numberOfOrigins)
			{
				float orig[3] = { origins[i].x, origins[i].y, origins[i].z };
				float t, u, v;
				if (intersect_triangle3(orig, dir, vert0, vert1, vert2, &t, &u, &v) == 1)
				{
					atomicAdd(&intersectionsPerOrigin[i], 1);
				}
				/*__syncthreads();
				int j = blockDim.x / 2;
				while (j != 0) {
					if (threadIdx.x < j) {
						intersectionsPerBlock[threadIdx.x] += intersectionsPerBlock[threadIdx.x + j]; // intersectionsPerBlock[]: Index = 0 houdt de som van alle threads binnen deze block bij
					}
					__syncthreads();
					j /= 2;
				}*/
				/*if (threadIdx.x == 0) {
					atomicAdd(&intersectionsPerOrigin[i], intersectionsPerBlock[0]);
				}*/
				// Als niet alle blocks tegelijk kunnen worden uitgevoerd dan zal het resultaat dat in 'intersectionsPerOrigin[i]' zit nog niet volledig zijn als deze wordt opgevraagd.
				// Dit kan zorgen voor verkeerde resultaten als het tussenresultaat toevallig even zou zijn.
				/*if (threadIdx.x == 0) {
					if (intersectionsPerOrigin[i] % 2 == 0) {
						inside = false;
					}
				}*/
				//intersectionsPerBlock[threadIdx.x] = 0;
				i++;
				teller++;
				i = i % numberOfOrigins;
			}
		}
	}
}