/* Ray-Triangle Intersection Test Routines          */
/* Different optimizations of my and Ben Trumbore's */
/* code from journals of graphics tools (JGT)       */
/* http://www.acm.org/jgt/                          */
/* by Tomas Moller, May 2000                        */

#include <math.h>
#include <iostream>
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
	int intersect_triangle3(float orig[3], float dir[3],
		float vert0[3], float vert1[3], float vert2[3],
		float* t, float* u, float* v)
	{
		//std::cout << "vert0 = " << vert0[0] << ", " << vert0[1] << ", " << vert0[2] << std::endl;
		//std::cout << "vert1 = " << vert1[0] << ", " << vert1[1] << ", " << vert1[2] << std::endl;
		//std::cout << "vert2 = " << vert2[0] << ", " << vert2[1] << ", " << vert2[2] << std::endl;
		//std::cout << "orig = " << orig[0] << ", " << orig[1] << ", " << orig[2] << std::endl;

		//Transfer parameters to device memory (boek: programming massively parallel processors fig 3.10 p50)
			//cudaMalloc(pointerToPointer, size);
			//cudaMemcpy(pointerToPointer, pointer, size, cudaMemcpyHostToDevice);

		//Kernel invocation code

		//Transfer variables from device back to host
			//cudaMemcpy(..., ..., size, cudaMemcpyDeviceToHost);
			//cudaFree(pointer);

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
		int* triangles, float* vertices, int* result)
	{
		printf("orig = %f, %f, %f", orig[0], orig[1], orig[2]);
		printf("dir = %f, %f, %f", dir[0], dir[1], dir[2]);
		float vert0[3] = { vertices[triangles[threadIdx.x]], vertices[triangles[threadIdx.x]+1], vertices[triangles[threadIdx.x]+2] };
		float vert1[3] = { vertices[triangles[threadIdx.x+1]], vertices[triangles[threadIdx.x+1] + 1], vertices[triangles[threadIdx.x+1] + 2] };
		float vert2[3] = { vertices[triangles[threadIdx.x+2]], vertices[triangles[threadIdx.x+2] + 1], vertices[triangles[threadIdx.x+2] + 2] };

		float edge1[3], edge2[3], tvec[3], pvec[3], qvec[3];
		float det, inv_det;
		float t, u, v;

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
			u = DOT(tvec, pvec);
			if (u < 0.0 || u > det) {
				result[threadIdx.x] = 0;
				return;
			}

			/* calculate V parameter and test bounds */
			v = DOT(dir, qvec);
			if (v < 0.0 || u + v > det) {
				result[threadIdx.x] = 0;
				return;
			}

		}
		else if (det < -EPSILON)
		{
			/* calculate U parameter and test bounds */
			u = DOT(tvec, pvec);
			if (u > 0.0 || u < det) {
				result[threadIdx.x] = 0;
				return;
			}

			/* calculate V parameter and test bounds */
			v = DOT(dir, qvec);
			if (v > 0.0 || u + v < det) {
				result[threadIdx.x] = 0;
				return;
			}
		}
		else 
		{
			result[threadIdx.x] = 0;  /* ray is parallell to the plane of the triangle */
			return;
		}

		t = DOT(edge2, qvec) * inv_det;
		(u) *= inv_det;
		(v) *= inv_det;

		if (t > 0)
		{
			result[threadIdx.x] = 1;
			return;
		}
		else
		{
			result[threadIdx.x] = 0;
			return;
		}
	}

}