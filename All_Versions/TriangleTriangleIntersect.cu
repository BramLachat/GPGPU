/* Triangle/triangle intersection test routine,
 * by Tomas Moller, 1997.
 * See article "A Fast Triangle-Triangle Intersection Test",
 * Journal of Graphics Tools, 2(2), 1997
 *
 * Updated June 1999: removed the divisions -- a little faster now!
 * Updated October 1999: added {} to CROSS and SUB macros
 *
 * int NoDivTriTriIsect(float V0[3],float V1[3],float V2[3],
 *                      float U0[3],float U1[3],float U2[3])
 *
 * parameters: vertices of triangle 1: V0,V1,V2
 *             vertices of triangle 2: U0,U1,U2
 * result    : returns 1 if the triangles intersect, otherwise 0
 *
 */

#include <math.h>
#include <device_launch_parameters.h>

#include "TriangleTriangleIntersect.cuh"

#define FABS(x) (float(fabs(x)))        /* implement as is fastest on your machine */

 /* if USE_EPSILON_TEST is true then we do a check:
		  if |dv|<EPSILON then dv=0.0;
	else no check is done (which is less robust)
 */
#define USE_EPSILON_TEST TRUE
#define EPSILON 0.000001


 /* some macros */
#define CROSS(dest,v1,v2){                     \
              dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; \
              dest[1]=v1[2]*v2[0]-v1[0]*v2[2]; \
              dest[2]=v1[0]*v2[1]-v1[1]*v2[0];}

#define DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])

#define SUB(dest,v1,v2){         \
            dest[0]=v1[0]-v2[0]; \
            dest[1]=v1[1]-v2[1]; \
            dest[2]=v1[2]-v2[2];}

/* sort so that a<=b */
#define SORT(a,b)       \
             if(a>b)    \
             {          \
               float c; \
               c=a;     \
               a=b;     \
               b=c;     \
             }


/* this edge to edge test is based on Franlin Antonio's gem:
   "Faster Line Segment Intersection", in Graphics Gems III,
   pp. 199-202 */
#define EDGE_EDGE_TEST(V0,U0,U1)                      \
  Bx=U0[i0]-U1[i0];                                   \
  By=U0[i1]-U1[i1];                                   \
  Cx=V0[i0]-U0[i0];                                   \
  Cy=V0[i1]-U0[i1];                                   \
  f=Ay*Bx-Ax*By;                                      \
  d=By*Cx-Bx*Cy;                                      \
  if((f>0 && d>=0 && d<=f) || (f<0 && d<=0 && d>=f))  \
  {                                                   \
    e=Ax*Cy-Ay*Cx;                                    \
    if(f>0)                                           \
    {                                                 \
      if(e>=0 && e<=f) return 1;                      \
    }                                                 \
    else                                              \
    {                                                 \
      if(e<=0 && e>=f) return 1;                      \
    }                                                 \
  }

#define EDGE_AGAINST_TRI_EDGES(V0,V1,U0,U1,U2) \
{                                              \
  float Ax,Ay,Bx,By,Cx,Cy,e,d,f;               \
  Ax=V1[i0]-V0[i0];                            \
  Ay=V1[i1]-V0[i1];                            \
  /* test edge U0,U1 against V0,V1 */          \
  EDGE_EDGE_TEST(V0,U0,U1);                    \
  /* test edge U1,U2 against V0,V1 */          \
  EDGE_EDGE_TEST(V0,U1,U2);                    \
  /* test edge U2,U1 against V0,V1 */          \
  EDGE_EDGE_TEST(V0,U2,U0);                    \
}

#define POINT_IN_TRI(V0,U0,U1,U2)           \
{                                           \
  float a,b,c,d0,d1,d2;                     \
  /* is T1 completly inside T2? */          \
  /* check if V0 is inside tri(U0,U1,U2) */ \
  a=U1[i1]-U0[i1];                          \
  b=-(U1[i0]-U0[i0]);                       \
  c=-a*U0[i0]-b*U0[i1];                     \
  d0=a*V0[i0]+b*V0[i1]+c;                   \
                                            \
  a=U2[i1]-U1[i1];                          \
  b=-(U2[i0]-U1[i0]);                       \
  c=-a*U1[i0]-b*U1[i1];                     \
  d1=a*V0[i0]+b*V0[i1]+c;                   \
                                            \
  a=U0[i1]-U2[i1];                          \
  b=-(U0[i0]-U2[i0]);                       \
  c=-a*U2[i0]-b*U2[i1];                     \
  d2=a*V0[i0]+b*V0[i1]+c;                   \
  if(d0*d1>0.0)                             \
  {                                         \
    if(d0*d2>0.0) return 1;                 \
  }                                         \
}

#define NEWCOMPUTE_INTERVALS(VV0,VV1,VV2,D0,D1,D2,D0D1,D0D2,A,B,C,X0,X1) \
{ \
        if(D0D1>0.0f) \
        { \
                /* here we know that D0D2<=0.0 */ \
            /* that is D0, D1 are on the same side, D2 on the other or on the plane */ \
                A=VV2; B=(VV0-VV2)*D2; C=(VV1-VV2)*D2; X0=D2-D0; X1=D2-D1; \
        } \
        else if(D0D2>0.0f)\
        { \
                /* here we know that d0d1<=0.0 */ \
            A=VV1; B=(VV0-VV1)*D1; C=(VV2-VV1)*D1; X0=D1-D0; X1=D1-D2; \
        } \
        else if(D1*D2>0.0f || D0!=0.0f) \
        { \
                /* here we know that d0d1<=0.0 or that D0!=0.0 */ \
                A=VV0; B=(VV1-VV0)*D0; C=(VV2-VV0)*D0; X0=D0-D1; X1=D0-D2; \
        } \
        else if(D1!=0.0f) \
        { \
                A=VV1; B=(VV0-VV1)*D1; C=(VV2-VV1)*D1; X0=D1-D0; X1=D1-D2; \
        } \
        else if(D2!=0.0f) \
        { \
                A=VV2; B=(VV0-VV2)*D2; C=(VV1-VV2)*D2; X0=D2-D0; X1=D2-D1; \
        } \
        else \
        { \
                /* triangles are coplanar */ \
                return coplanar_tri_tri(N1,V0,V1,V2,U0,U1,U2); \
        } \
}


	__host__ __device__ int NoDivTriTriIsect(float V0[3], float V1[3], float V2[3],
		float U0[3], float U1[3], float U2[3])
	{
		float E1[3], E2[3];
		float N1[3], N2[3], d1, d2;
		float du0, du1, du2, dv0, dv1, dv2;
		float D[3];
		float isect1[2], isect2[2];
		float du0du1, du0du2, dv0dv1, dv0dv2;
		short index;
		float vp0, vp1, vp2;
		float up0, up1, up2;
		float bb, cc, max;

		/* compute plane equation of triangle(V0,V1,V2) */
		SUB(E1, V1, V0);
		SUB(E2, V2, V0);
		CROSS(N1, E1, E2);
		d1 = -DOT(N1, V0);
		/* plane equation 1: N1.X+d1=0 */

		/* put U0,U1,U2 into plane equation 1 to compute signed distances to the plane*/
		du0 = DOT(N1, U0) + d1;
		du1 = DOT(N1, U1) + d1;
		du2 = DOT(N1, U2) + d1;

		/* coplanarity robustness check */
#if USE_EPSILON_TEST==TRUE
		if (FABS(du0) < EPSILON) du0 = 0.0;
		if (FABS(du1) < EPSILON) du1 = 0.0;
		if (FABS(du2) < EPSILON) du2 = 0.0;
#endif
		du0du1 = du0 * du1;
		du0du2 = du0 * du2;

		if (du0du1 > 0.0f && du0du2 > 0.0f) /* same sign on all of them + not equal 0 ? */
			return 0;                    /* no intersection occurs */

		  /* compute plane of triangle (U0,U1,U2) */
		SUB(E1, U1, U0);
		SUB(E2, U2, U0);
		CROSS(N2, E1, E2);
		d2 = -DOT(N2, U0);
		/* plane equation 2: N2.X+d2=0 */

		/* put V0,V1,V2 into plane equation 2 */
		dv0 = DOT(N2, V0) + d2;
		dv1 = DOT(N2, V1) + d2;
		dv2 = DOT(N2, V2) + d2;

#if USE_EPSILON_TEST==TRUE
		if (FABS(dv0) < EPSILON) dv0 = 0.0;
		if (FABS(dv1) < EPSILON) dv1 = 0.0;
		if (FABS(dv2) < EPSILON) dv2 = 0.0;
#endif

		dv0dv1 = dv0 * dv1;
		dv0dv2 = dv0 * dv2;

		if (dv0dv1 > 0.0f && dv0dv2 > 0.0f) /* same sign on all of them + not equal 0 ? */
			return 0;                    /* no intersection occurs */

		  /* compute direction of intersection line */
		CROSS(D, N1, N2);

		/* compute and index to the largest component of D */
		max = (float)FABS(D[0]);
		index = 0;
		bb = (float)FABS(D[1]);
		cc = (float)FABS(D[2]);
		if (bb > max) max = bb, index = 1;
		if (cc > max) max = cc, index = 2;

		/* this is the simplified projection onto L*/
		vp0 = V0[index];
		vp1 = V1[index];
		vp2 = V2[index];

		up0 = U0[index];
		up1 = U1[index];
		up2 = U2[index];

		/* compute interval for triangle 1 */
		float a, b, c, x0, x1;
		NEWCOMPUTE_INTERVALS(vp0, vp1, vp2, dv0, dv1, dv2, dv0dv1, dv0dv2, a, b, c, x0, x1);

		/* compute interval for triangle 2 */
		float d, e, f, y0, y1;
		NEWCOMPUTE_INTERVALS(up0, up1, up2, du0, du1, du2, du0du1, du0du2, d, e, f, y0, y1);

		float xx, yy, xxyy, tmp;
		xx = x0 * x1;
		yy = y0 * y1;
		xxyy = xx * yy;

		tmp = a * xxyy;
		isect1[0] = tmp + b * x1 * yy;
		isect1[1] = tmp + c * x0 * yy;

		tmp = d * xxyy;
		isect2[0] = tmp + e * xx * y1;
		isect2[1] = tmp + f * xx * y0;

		SORT(isect1[0], isect1[1]);
		SORT(isect2[0], isect2[1]);

		if (isect1[1] < isect2[0] || isect2[1] < isect1[0]) return 0;
		return 1;
	}

	__host__ __device__ int coplanar_tri_tri(float N[3], float V0[3], float V1[3], float V2[3],
		float U0[3], float U1[3], float U2[3])
	{
		float A[3];
		short i0, i1;
		/* first project onto an axis-aligned plane, that maximizes the area */
		/* of the triangles, compute indices: i0,i1. */
		A[0] = FABS(N[0]);
		A[1] = FABS(N[1]);
		A[2] = FABS(N[2]);
		if (A[0] > A[1])
		{
			if (A[0] > A[2])
			{
				i0 = 1;      /* A[0] is greatest */
				i1 = 2;
			}
			else
			{
				i0 = 0;      /* A[2] is greatest */
				i1 = 1;
			}
		}
		else   /* A[0]<=A[1] */
		{
			if (A[2] > A[1])
			{
				i0 = 0;      /* A[2] is greatest */
				i1 = 1;
			}
			else
			{
				i0 = 0;      /* A[1] is greatest */
				i1 = 2;
			}
		}

		/* test all edges of triangle 1 against the edges of triangle 2 */
		EDGE_AGAINST_TRI_EDGES(V0, V1, U0, U1, U2);
		EDGE_AGAINST_TRI_EDGES(V1, V2, U0, U1, U2);
		EDGE_AGAINST_TRI_EDGES(V2, V0, U0, U1, U2);

		/* finally, test if tri1 is totally contained in tri2 or vice versa */
		POINT_IN_TRI(V0, U0, U1, U2);
		POINT_IN_TRI(U0, V0, V1, V2);

		return 0;
	}

	__host__ __device__ int BPCD(float V0[3], float V1[3], float V2[3],
		float U0[3], float U1[3], float U2[3])
	{
		float E1[3], E2[3];
		float N1[3], N2[3], d1, d2;
		float du0, du1, du2, dv0, dv1, dv2;
		//float D[3];
		//float isect1[2], isect2[2];
		float du0du1, du0du2, dv0dv1, dv0dv2;
		//short index;
		//float vp0, vp1, vp2;
		//float up0, up1, up2;
		//float bb, cc, max;

		/* compute plane equation of triangle(V0,V1,V2) */
		SUB(E1, V1, V0);
		SUB(E2, V2, V0);
		CROSS(N1, E1, E2);
		d1 = -DOT(N1, V0);
		/* plane equation 1: N1.X+d1=0 */

		/* put U0,U1,U2 into plane equation 1 to compute signed distances to the plane*/
		du0 = DOT(N1, U0) + d1;
		du1 = DOT(N1, U1) + d1;
		du2 = DOT(N1, U2) + d1;

		/* coplanarity robustness check */
#if USE_EPSILON_TEST==TRUE
		if (FABS(du0) < EPSILON) du0 = 0.0;
		if (FABS(du1) < EPSILON) du1 = 0.0;
		if (FABS(du2) < EPSILON) du2 = 0.0;
#endif
		du0du1 = du0 * du1;
		du0du2 = du0 * du2;

		if (du0du1 > 0.0f && du0du2 > 0.0f) /* same sign on all of them + not equal 0 ? */
			return 0;                    /* no intersection occurs */

		  /* compute plane of triangle (U0,U1,U2) */
		SUB(E1, U1, U0);
		SUB(E2, U2, U0);
		CROSS(N2, E1, E2);
		d2 = -DOT(N2, U0);
		/* plane equation 2: N2.X+d2=0 */

		/* put V0,V1,V2 into plane equation 2 */
		dv0 = DOT(N2, V0) + d2;
		dv1 = DOT(N2, V1) + d2;
		dv2 = DOT(N2, V2) + d2;

#if USE_EPSILON_TEST==TRUE
		if (FABS(dv0) < EPSILON) dv0 = 0.0;
		if (FABS(dv1) < EPSILON) dv1 = 0.0;
		if (FABS(dv2) < EPSILON) dv2 = 0.0;
#endif

		dv0dv1 = dv0 * dv1;
		dv0dv2 = dv0 * dv2;

		if (dv0dv1 > 0.0f && dv0dv2 > 0.0f) /* same sign on all of them + not equal 0 ? */
			return 0;                    /* no intersection occurs */

		return 1;
	}

	//thread per (inner) triangle
	__global__ void triangle_triangle_GPU_ThreadPerInnerTriangle(
		int3* cudaInsideTriangles, 
		float3* cudaInsideVertices, 
		int3* cudaOutsideTriangles, 
		float3* cudaOutsideVertices, 
		bool* inside, 
		int numberOfInsideTriangles, 
		int numberOfOutsideTriangles) { // , int* cudaIntersectionsPerInsideTriangle
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid < numberOfInsideTriangles)
		{
			float vert1_1[3] = { cudaInsideVertices[cudaInsideTriangles[tid].x].x, cudaInsideVertices[cudaInsideTriangles[tid].x].y, cudaInsideVertices[cudaInsideTriangles[tid].x].z };
			float vert1_2[3] = { cudaInsideVertices[cudaInsideTriangles[tid].y].x, cudaInsideVertices[cudaInsideTriangles[tid].y].y, cudaInsideVertices[cudaInsideTriangles[tid].y].z };
			float vert1_3[3] = { cudaInsideVertices[cudaInsideTriangles[tid].z].x, cudaInsideVertices[cudaInsideTriangles[tid].z].y, cudaInsideVertices[cudaInsideTriangles[tid].z].z };

			//int numberOfIntersections = 0;
			for (int i = 0; i < numberOfOutsideTriangles; i++)
			{
				//if (*inside) {
					float vert2_1[3] = { cudaOutsideVertices[cudaOutsideTriangles[i].x].x, cudaOutsideVertices[cudaOutsideTriangles[i].x].y, cudaOutsideVertices[cudaOutsideTriangles[i].x].z };
					float vert2_2[3] = { cudaOutsideVertices[cudaOutsideTriangles[i].y].x, cudaOutsideVertices[cudaOutsideTriangles[i].y].y, cudaOutsideVertices[cudaOutsideTriangles[i].y].z };
					float vert2_3[3] = { cudaOutsideVertices[cudaOutsideTriangles[i].z].x, cudaOutsideVertices[cudaOutsideTriangles[i].z].y, cudaOutsideVertices[cudaOutsideTriangles[i].z].z };
					if (NoDivTriTriIsect(vert1_1, vert1_2, vert1_3, vert2_1, vert2_2, vert2_3) == 1)
					{
						//numberOfIntersections++;
						*inside = false;
						//return;
						//cudaIntersectionsPerInsideTriangle[tid] = 1; // Sneller als je dit weg laat in het geval de meshes elkaar niet sijden ==> dit zorgt er voor dat het trager wordt als de meshes in elkaar liggen
					}
					//if(intersect){ cudaIntersectionsPerInsideTriangle[tid] = 1; } // Sneller als je dit weg laat in het geval de meshes elkaar niet sijden
				/*}
				else {
					return;
				}*/
			}
			//printf("numberOfIntersections = %d\n", numberOfIntersections);
			//cudaIntersectionsPerInsideTriangle[tid] = numberOfIntersections;
		}
	}

	//block per (inner) triangle
	__global__ void triangle_triangle_GPU_BlockPerInnerTriangle(
		int3* cudaInsideTriangles,
		float3* cudaInsideVertices,
		int3* cudaOutsideTriangles,
		float3* cudaOutsideVertices,
		bool* inside,
		int numberOfInsideTriangles,
		int numberOfOutsideTriangles) { // , int* cudaIntersectionsPerInsideTriangle

		int threadidx = threadIdx.x;
		float vert1_1[3] = { cudaInsideVertices[cudaInsideTriangles[blockIdx.x].x].x, cudaInsideVertices[cudaInsideTriangles[blockIdx.x].x].y, cudaInsideVertices[cudaInsideTriangles[blockIdx.x].x].z };
		float vert1_2[3] = { cudaInsideVertices[cudaInsideTriangles[blockIdx.x].y].x, cudaInsideVertices[cudaInsideTriangles[blockIdx.x].y].y, cudaInsideVertices[cudaInsideTriangles[blockIdx.x].y].z };
		float vert1_3[3] = { cudaInsideVertices[cudaInsideTriangles[blockIdx.x].z].x, cudaInsideVertices[cudaInsideTriangles[blockIdx.x].z].y, cudaInsideVertices[cudaInsideTriangles[blockIdx.x].z].z };

		//int numberOfIntersections = 0;
		while (threadidx < numberOfOutsideTriangles) { // && *inside
			float vert2_1[3] = { cudaOutsideVertices[cudaOutsideTriangles[threadidx].x].x, cudaOutsideVertices[cudaOutsideTriangles[threadidx].x].y, cudaOutsideVertices[cudaOutsideTriangles[threadidx].x].z };
			float vert2_2[3] = { cudaOutsideVertices[cudaOutsideTriangles[threadidx].y].x, cudaOutsideVertices[cudaOutsideTriangles[threadidx].y].y, cudaOutsideVertices[cudaOutsideTriangles[threadidx].y].z };
			float vert2_3[3] = { cudaOutsideVertices[cudaOutsideTriangles[threadidx].z].x, cudaOutsideVertices[cudaOutsideTriangles[threadidx].z].y, cudaOutsideVertices[cudaOutsideTriangles[threadidx].z].z };
			if (NoDivTriTriIsect(vert1_1, vert1_2, vert1_3, vert2_1, vert2_2, vert2_3) == 1)
			{
				//numberOfIntersections++;
				*inside = false;
				//return;
				//cudaIntersectionsPerInsideTriangle[tid] = 1; // Sneller als je dit weg laat in het geval de meshes elkaar niet sijden ==> dit zorgt er voor dat het trager wordt als de meshes in elkaar liggen
			}
			threadidx += blockDim.x;
			//if(intersect){ cudaIntersectionsPerInsideTriangle[tid] = 1; } // Sneller als je dit weg laat in het geval de meshes elkaar niet sijden
		}
		//printf("numberOfIntersections = %d\n", numberOfIntersections);
		//cudaIntersectionsPerInsideTriangle[tid] = numberOfIntersections;
	}

	//thread per (outer) triangle
	__global__ void triangle_triangle_GPU_ThreadPerOuterTriangle(
		int3* cudaInsideTriangles,
		float3* cudaInsideVertices,
		int3* cudaOutsideTriangles,
		float3* cudaOutsideVertices,
		bool* inside,
		int numberOfInsideTriangles,
		int numberOfOutsideTriangles) { // , int* cudaIntersectionsPerInsideTriangle

		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid < numberOfOutsideTriangles)
		{
			float vert1_1[3] = { cudaOutsideVertices[cudaOutsideTriangles[tid].x].x, cudaOutsideVertices[cudaOutsideTriangles[tid].x].y, cudaOutsideVertices[cudaOutsideTriangles[tid].x].z };
			float vert1_2[3] = { cudaOutsideVertices[cudaOutsideTriangles[tid].y].x, cudaOutsideVertices[cudaOutsideTriangles[tid].y].y, cudaOutsideVertices[cudaOutsideTriangles[tid].y].z };
			float vert1_3[3] = { cudaOutsideVertices[cudaOutsideTriangles[tid].z].x, cudaOutsideVertices[cudaOutsideTriangles[tid].z].y, cudaOutsideVertices[cudaOutsideTriangles[tid].z].z };

			//int numberOfIntersections = 0;
			int i = 0;
			while (i < numberOfInsideTriangles) { // && *inside
				float vert2_1[3] = { cudaInsideVertices[cudaInsideTriangles[i].x].x, cudaInsideVertices[cudaInsideTriangles[i].x].y, cudaInsideVertices[cudaInsideTriangles[i].x].z };
				float vert2_2[3] = { cudaInsideVertices[cudaInsideTriangles[i].y].x, cudaInsideVertices[cudaInsideTriangles[i].y].y, cudaInsideVertices[cudaInsideTriangles[i].y].z };
				float vert2_3[3] = { cudaInsideVertices[cudaInsideTriangles[i].z].x, cudaInsideVertices[cudaInsideTriangles[i].z].y, cudaInsideVertices[cudaInsideTriangles[i].z].z };
				if (NoDivTriTriIsect(vert1_1, vert1_2, vert1_3, vert2_1, vert2_2, vert2_3) == 1)
				{
					//numberOfIntersections++;
					*inside = false;
					//return;
					//cudaIntersectionsPerInsideTriangle[tid] = 1; // Sneller als je dit weg laat in het geval de meshes elkaar niet sijden ==> dit zorgt er voor dat het trager wordt als de meshes in elkaar liggen
				}
				i++;
				//if(intersect){ cudaIntersectionsPerInsideTriangle[tid] = 1; } // Sneller als je dit weg laat in het geval de meshes elkaar niet sijden
			}
			//printf("numberOfIntersections = %d\n", numberOfIntersections);
			//cudaIntersectionsPerInsideTriangle[tid] = numberOfIntersections;
		}
	}

	//block per (outer) triangle
	__global__ void triangle_triangle_GPU_BlockPerOuterTriangle(
		int3* cudaInsideTriangles,
		float3* cudaInsideVertices,
		int3* cudaOutsideTriangles,
		float3* cudaOutsideVertices,
		bool* inside,
		int numberOfInsideTriangles,
		int numberOfOutsideTriangles) {

		int threadidx = threadIdx.x;
		float vert1_1[3] = { cudaOutsideVertices[cudaOutsideTriangles[blockIdx.x].x].x, cudaOutsideVertices[cudaOutsideTriangles[blockIdx.x].x].y, cudaOutsideVertices[cudaOutsideTriangles[blockIdx.x].x].z };
		float vert1_2[3] = { cudaOutsideVertices[cudaOutsideTriangles[blockIdx.x].y].x, cudaOutsideVertices[cudaOutsideTriangles[blockIdx.x].y].y, cudaOutsideVertices[cudaOutsideTriangles[blockIdx.x].y].z };
		float vert1_3[3] = { cudaOutsideVertices[cudaOutsideTriangles[blockIdx.x].z].x, cudaOutsideVertices[cudaOutsideTriangles[blockIdx.x].z].y, cudaOutsideVertices[cudaOutsideTriangles[blockIdx.x].z].z };

		while (threadidx < numberOfInsideTriangles) { // && *inside
			float vert2_1[3] = { cudaInsideVertices[cudaInsideTriangles[threadidx].x].x, cudaInsideVertices[cudaInsideTriangles[threadidx].x].y, cudaInsideVertices[cudaInsideTriangles[threadidx].x].z };
			float vert2_2[3] = { cudaInsideVertices[cudaInsideTriangles[threadidx].y].x, cudaInsideVertices[cudaInsideTriangles[threadidx].y].y, cudaInsideVertices[cudaInsideTriangles[threadidx].y].z };
			float vert2_3[3] = { cudaInsideVertices[cudaInsideTriangles[threadidx].z].x, cudaInsideVertices[cudaInsideTriangles[threadidx].z].y, cudaInsideVertices[cudaInsideTriangles[threadidx].z].z };
			if (NoDivTriTriIsect(vert1_1, vert1_2, vert1_3, vert2_1, vert2_2, vert2_3) == 1)
			{
				*inside = false;
			}
			threadidx += blockDim.x;
		}
	}

	__global__ void triangle_triangle_GPU_BPCD_1_ThreadPerInnerTriangle(
		int3* cudaInsideTriangles, 
		float3* cudaInsideVertices, 
		int3* cudaOutsideTriangles, 
		float3* cudaOutsideVertices, 
		int numberOfInsideTriangles, 
		int numberOfOutsideTriangles, 
		int* intersectingTriangles) {

		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		int start = 0;
		int end = 0;
		int counter = 0;
		if (tid < numberOfInsideTriangles)
		{
			float vert1_1[3] = { cudaInsideVertices[cudaInsideTriangles[tid].x].x, cudaInsideVertices[cudaInsideTriangles[tid].x].y, cudaInsideVertices[cudaInsideTriangles[tid].x].z };
			float vert1_2[3] = { cudaInsideVertices[cudaInsideTriangles[tid].y].x, cudaInsideVertices[cudaInsideTriangles[tid].y].y, cudaInsideVertices[cudaInsideTriangles[tid].y].z };
			float vert1_3[3] = { cudaInsideVertices[cudaInsideTriangles[tid].z].x, cudaInsideVertices[cudaInsideTriangles[tid].z].y, cudaInsideVertices[cudaInsideTriangles[tid].z].z };

			for (int i = 0; i < numberOfOutsideTriangles; i++)
			{

				float vert2_1[3] = { cudaOutsideVertices[cudaOutsideTriangles[i].x].x, cudaOutsideVertices[cudaOutsideTriangles[i].x].y, cudaOutsideVertices[cudaOutsideTriangles[i].x].z };
				float vert2_2[3] = { cudaOutsideVertices[cudaOutsideTriangles[i].y].x, cudaOutsideVertices[cudaOutsideTriangles[i].y].y, cudaOutsideVertices[cudaOutsideTriangles[i].y].z };
				float vert2_3[3] = { cudaOutsideVertices[cudaOutsideTriangles[i].z].x, cudaOutsideVertices[cudaOutsideTriangles[i].z].y, cudaOutsideVertices[cudaOutsideTriangles[i].z].z };
				if (BPCD(vert1_1, vert1_2, vert1_3, vert2_1, vert2_2, vert2_3) == 1)
				{
					if (counter == 0) {
						start = i;
					}
					end = i + 1;
					if (counter < 10) {
						intersectingTriangles[tid * 10 + counter] = i;
					}
					counter++;
				}
			}
			if (counter == 0 || counter > 10) {
				intersectingTriangles[tid * 10] = -1;
				intersectingTriangles[tid * 10 + 1] = start;
				intersectingTriangles[tid * 10 + 2] = end;
			}
		}
	}

	__global__ void triangle_triangle_GPU_BPCD_1_ThreadPerOuterTriangle(
		int3* cudaInsideTriangles,
		float3* cudaInsideVertices,
		int3* cudaOutsideTriangles,
		float3* cudaOutsideVertices,
		int numberOfInsideTriangles,
		int numberOfOutsideTriangles,
		int* intersectingTriangles) {

		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		int start = 0;
		int end = 0;
		int counter = 0;
		if (tid < numberOfOutsideTriangles)
		{
			float vert1_1[3] = { cudaOutsideVertices[cudaOutsideTriangles[tid].x].x, cudaOutsideVertices[cudaOutsideTriangles[tid].x].y, cudaOutsideVertices[cudaOutsideTriangles[tid].x].z };
			float vert1_2[3] = { cudaOutsideVertices[cudaOutsideTriangles[tid].y].x, cudaOutsideVertices[cudaOutsideTriangles[tid].y].y, cudaOutsideVertices[cudaOutsideTriangles[tid].y].z };
			float vert1_3[3] = { cudaOutsideVertices[cudaOutsideTriangles[tid].z].x, cudaOutsideVertices[cudaOutsideTriangles[tid].z].y, cudaOutsideVertices[cudaOutsideTriangles[tid].z].z };

			for (int i = 0; i < numberOfInsideTriangles; i++)
			{

				float vert2_1[3] = { cudaInsideVertices[cudaInsideTriangles[i].x].x, cudaInsideVertices[cudaInsideTriangles[i].x].y, cudaInsideVertices[cudaInsideTriangles[i].x].z };
				float vert2_2[3] = { cudaInsideVertices[cudaInsideTriangles[i].y].x, cudaInsideVertices[cudaInsideTriangles[i].y].y, cudaInsideVertices[cudaInsideTriangles[i].y].z };
				float vert2_3[3] = { cudaInsideVertices[cudaInsideTriangles[i].z].x, cudaInsideVertices[cudaInsideTriangles[i].z].y, cudaInsideVertices[cudaInsideTriangles[i].z].z };
				if (BPCD(vert1_1, vert1_2, vert1_3, vert2_1, vert2_2, vert2_3) == 1)
				{
					if (counter == 0) {
						start = i;
					}
					end = i + 1;
					if (counter < 10) {
						intersectingTriangles[tid * 10 + counter] = i;
					}
					counter++;
				}
			}
			if (counter == 0 || counter > 10) {
				intersectingTriangles[tid * 10] = -1;
				intersectingTriangles[tid * 10 + 1] = start;
				intersectingTriangles[tid * 10 + 2] = end;
			}
		}
	}

	__global__ void triangle_triangle_GPU_BPCD_2_ThreadPerInnerTriangle(
		int3* cudaInsideTriangles, 
		float3* cudaInsideVertices, 
		int3* cudaOutsideTriangles, 
		float3* cudaOutsideVertices, 
		bool* inside, 
		int numberOfInsideTriangles, 
		int* intersectingTriangles) {
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid < numberOfInsideTriangles)
		{
			float vert1_1[3] = { cudaInsideVertices[cudaInsideTriangles[tid].x].x, cudaInsideVertices[cudaInsideTriangles[tid].x].y, cudaInsideVertices[cudaInsideTriangles[tid].x].z };
			float vert1_2[3] = { cudaInsideVertices[cudaInsideTriangles[tid].y].x, cudaInsideVertices[cudaInsideTriangles[tid].y].y, cudaInsideVertices[cudaInsideTriangles[tid].y].z };
			float vert1_3[3] = { cudaInsideVertices[cudaInsideTriangles[tid].z].x, cudaInsideVertices[cudaInsideTriangles[tid].z].y, cudaInsideVertices[cudaInsideTriangles[tid].z].z };

			if (intersectingTriangles[tid * 10] == -1) {
				int start = intersectingTriangles[tid * 10 + 1];
				int end = intersectingTriangles[tid * 10 + 2];

				int i = start;
				while(i < end && *inside)//for (int i = start; i < end; i++)
				{
					float vert2_1[3] = { cudaOutsideVertices[cudaOutsideTriangles[i].x].x, cudaOutsideVertices[cudaOutsideTriangles[i].x].y, cudaOutsideVertices[cudaOutsideTriangles[i].x].z };
					float vert2_2[3] = { cudaOutsideVertices[cudaOutsideTriangles[i].y].x, cudaOutsideVertices[cudaOutsideTriangles[i].y].y, cudaOutsideVertices[cudaOutsideTriangles[i].y].z };
					float vert2_3[3] = { cudaOutsideVertices[cudaOutsideTriangles[i].z].x, cudaOutsideVertices[cudaOutsideTriangles[i].z].y, cudaOutsideVertices[cudaOutsideTriangles[i].z].z };
					if (NoDivTriTriIsect(vert1_1, vert1_2, vert1_3, vert2_1, vert2_2, vert2_3) == 1)
					{
						*inside = false;
						return;
					}
					i++;
				}
			}
			else {
				int counter = 0;
				int i = intersectingTriangles[tid * 10 + counter];
				while (i != 0 && counter < 10 && *inside)
				{
					float vert2_1[3] = { cudaOutsideVertices[cudaOutsideTriangles[i].x].x, cudaOutsideVertices[cudaOutsideTriangles[i].x].y, cudaOutsideVertices[cudaOutsideTriangles[i].x].z };
					float vert2_2[3] = { cudaOutsideVertices[cudaOutsideTriangles[i].y].x, cudaOutsideVertices[cudaOutsideTriangles[i].y].y, cudaOutsideVertices[cudaOutsideTriangles[i].y].z };
					float vert2_3[3] = { cudaOutsideVertices[cudaOutsideTriangles[i].z].x, cudaOutsideVertices[cudaOutsideTriangles[i].z].y, cudaOutsideVertices[cudaOutsideTriangles[i].z].z };
					if (NoDivTriTriIsect(vert1_1, vert1_2, vert1_3, vert2_1, vert2_2, vert2_3) == 1)
					{
						*inside = false;
						return;
					}
					counter++;
					i = intersectingTriangles[tid * 10 + counter];
				}
			}


		}
	}

	__global__ void triangle_triangle_GPU_BPCD_2_ThreadPerOuterTriangle(
		int3* cudaInsideTriangles,
		float3* cudaInsideVertices,
		int3* cudaOutsideTriangles,
		float3* cudaOutsideVertices,
		bool* inside,
		int numberOfOutsideTriangles,
		int* intersectingTriangles) {
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid < numberOfOutsideTriangles)
		{
			float vert1_1[3] = { cudaOutsideVertices[cudaOutsideTriangles[tid].x].x, cudaOutsideVertices[cudaOutsideTriangles[tid].x].y, cudaOutsideVertices[cudaOutsideTriangles[tid].x].z };
			float vert1_2[3] = { cudaOutsideVertices[cudaOutsideTriangles[tid].y].x, cudaOutsideVertices[cudaOutsideTriangles[tid].y].y, cudaOutsideVertices[cudaOutsideTriangles[tid].y].z };
			float vert1_3[3] = { cudaOutsideVertices[cudaOutsideTriangles[tid].z].x, cudaOutsideVertices[cudaOutsideTriangles[tid].z].y, cudaOutsideVertices[cudaOutsideTriangles[tid].z].z };

			if (intersectingTriangles[tid * 10] == -1) {
				int start = intersectingTriangles[tid * 10 + 1];
				int end = intersectingTriangles[tid * 10 + 2];

				int i = start;
				while(i < end && *inside)//for (int i = start; i < end; i++)
				{
					float vert2_1[3] = { cudaInsideVertices[cudaInsideTriangles[i].x].x, cudaInsideVertices[cudaInsideTriangles[i].x].y, cudaInsideVertices[cudaInsideTriangles[i].x].z };
					float vert2_2[3] = { cudaInsideVertices[cudaInsideTriangles[i].y].x, cudaInsideVertices[cudaInsideTriangles[i].y].y, cudaInsideVertices[cudaInsideTriangles[i].y].z };
					float vert2_3[3] = { cudaInsideVertices[cudaInsideTriangles[i].z].x, cudaInsideVertices[cudaInsideTriangles[i].z].y, cudaInsideVertices[cudaInsideTriangles[i].z].z };
					if (NoDivTriTriIsect(vert1_1, vert1_2, vert1_3, vert2_1, vert2_2, vert2_3) == 1)
					{
						*inside = false;
						return;
					}
					i++;
				}
			}
			else {
				int counter = 0;
				int i = intersectingTriangles[tid * 10 + counter];
				while (i != 0 && counter < 10 && *inside)
				{
					float vert2_1[3] = { cudaInsideVertices[cudaInsideTriangles[i].x].x, cudaInsideVertices[cudaInsideTriangles[i].x].y, cudaInsideVertices[cudaInsideTriangles[i].x].z };
					float vert2_2[3] = { cudaInsideVertices[cudaInsideTriangles[i].y].x, cudaInsideVertices[cudaInsideTriangles[i].y].y, cudaInsideVertices[cudaInsideTriangles[i].y].z };
					float vert2_3[3] = { cudaInsideVertices[cudaInsideTriangles[i].z].x, cudaInsideVertices[cudaInsideTriangles[i].z].y, cudaInsideVertices[cudaInsideTriangles[i].z].z };
					if (NoDivTriTriIsect(vert1_1, vert1_2, vert1_3, vert2_1, vert2_2, vert2_3) == 1)
					{
						*inside = false;
						return;
					}
					counter++;
					i = intersectingTriangles[tid * 10 + counter];
				}
			}


		}
	}

	__global__ void triangle_triangle_GPU_BPCD_2_BlockPerInnerTriangle(
		int3* cudaInsideTriangles,
		float3* cudaInsideVertices,
		int3* cudaOutsideTriangles,
		float3* cudaOutsideVertices,
		bool* inside,
		int* intersectingTriangles) {

		float vert1_1[3] = { cudaInsideVertices[cudaInsideTriangles[blockIdx.x].x].x, cudaInsideVertices[cudaInsideTriangles[blockIdx.x].x].y, cudaInsideVertices[cudaInsideTriangles[blockIdx.x].x].z };
		float vert1_2[3] = { cudaInsideVertices[cudaInsideTriangles[blockIdx.x].y].x, cudaInsideVertices[cudaInsideTriangles[blockIdx.x].y].y, cudaInsideVertices[cudaInsideTriangles[blockIdx.x].y].z };
		float vert1_3[3] = { cudaInsideVertices[cudaInsideTriangles[blockIdx.x].z].x, cudaInsideVertices[cudaInsideTriangles[blockIdx.x].z].y, cudaInsideVertices[cudaInsideTriangles[blockIdx.x].z].z };

		if (intersectingTriangles[blockIdx.x * 10] == -1) {
			int start = intersectingTriangles[blockIdx.x * 10 + 1];
			int end = intersectingTriangles[blockIdx.x * 10 + 2];
			int i = threadIdx.x + start;
			while (i < end && *inside)
			{
				float vert2_1[3] = { cudaOutsideVertices[cudaOutsideTriangles[i].x].x, cudaOutsideVertices[cudaOutsideTriangles[i].x].y, cudaOutsideVertices[cudaOutsideTriangles[i].x].z };
				float vert2_2[3] = { cudaOutsideVertices[cudaOutsideTriangles[i].y].x, cudaOutsideVertices[cudaOutsideTriangles[i].y].y, cudaOutsideVertices[cudaOutsideTriangles[i].y].z };
				float vert2_3[3] = { cudaOutsideVertices[cudaOutsideTriangles[i].z].x, cudaOutsideVertices[cudaOutsideTriangles[i].z].y, cudaOutsideVertices[cudaOutsideTriangles[i].z].z };
				if (NoDivTriTriIsect(vert1_1, vert1_2, vert1_3, vert2_1, vert2_2, vert2_3) == 1)
				{
					*inside = false;
					return;
				}
				i += blockDim.x;
			}
		}
		else {
			int i = intersectingTriangles[blockIdx.x * 10 + threadIdx.x];
			if (i != 0 && threadIdx.x < 10 && *inside)
			{
				float vert2_1[3] = { cudaOutsideVertices[cudaOutsideTriangles[i].x].x, cudaOutsideVertices[cudaOutsideTriangles[i].x].y, cudaOutsideVertices[cudaOutsideTriangles[i].x].z };
				float vert2_2[3] = { cudaOutsideVertices[cudaOutsideTriangles[i].y].x, cudaOutsideVertices[cudaOutsideTriangles[i].y].y, cudaOutsideVertices[cudaOutsideTriangles[i].y].z };
				float vert2_3[3] = { cudaOutsideVertices[cudaOutsideTriangles[i].z].x, cudaOutsideVertices[cudaOutsideTriangles[i].z].y, cudaOutsideVertices[cudaOutsideTriangles[i].z].z };
				if (NoDivTriTriIsect(vert1_1, vert1_2, vert1_3, vert2_1, vert2_2, vert2_3) == 1)
				{
					*inside = false;
					return;
				}
			}
		}
	}

	__global__ void triangle_triangle_GPU_BPCD_2_BlockPerOuterTriangle(
		int3* cudaInsideTriangles,
		float3* cudaInsideVertices,
		int3* cudaOutsideTriangles,
		float3* cudaOutsideVertices,
		bool* inside,
		int* intersectingTriangles) {

		float vert1_1[3] = { cudaOutsideVertices[cudaOutsideTriangles[blockIdx.x].x].x, cudaOutsideVertices[cudaOutsideTriangles[blockIdx.x].x].y, cudaOutsideVertices[cudaOutsideTriangles[blockIdx.x].x].z };
		float vert1_2[3] = { cudaOutsideVertices[cudaOutsideTriangles[blockIdx.x].y].x, cudaOutsideVertices[cudaOutsideTriangles[blockIdx.x].y].y, cudaOutsideVertices[cudaOutsideTriangles[blockIdx.x].y].z };
		float vert1_3[3] = { cudaOutsideVertices[cudaOutsideTriangles[blockIdx.x].z].x, cudaOutsideVertices[cudaOutsideTriangles[blockIdx.x].z].y, cudaOutsideVertices[cudaOutsideTriangles[blockIdx.x].z].z };

		if (intersectingTriangles[blockIdx.x * 10] == -1) {
			int start = intersectingTriangles[blockIdx.x * 10 + 1];
			int end = intersectingTriangles[blockIdx.x * 10 + 2];
			int i = threadIdx.x + start;
			while (i < end && *inside)
			{
				float vert2_1[3] = { cudaInsideVertices[cudaInsideTriangles[i].x].x, cudaInsideVertices[cudaInsideTriangles[i].x].y, cudaInsideVertices[cudaInsideTriangles[i].x].z };
				float vert2_2[3] = { cudaInsideVertices[cudaInsideTriangles[i].y].x, cudaInsideVertices[cudaInsideTriangles[i].y].y, cudaInsideVertices[cudaInsideTriangles[i].y].z };
				float vert2_3[3] = { cudaInsideVertices[cudaInsideTriangles[i].z].x, cudaInsideVertices[cudaInsideTriangles[i].z].y, cudaInsideVertices[cudaInsideTriangles[i].z].z };
				if (NoDivTriTriIsect(vert1_1, vert1_2, vert1_3, vert2_1, vert2_2, vert2_3) == 1)
				{
					*inside = false;
					return;
				}
				i += blockDim.x;
			}
		}
		else {
			int i = intersectingTriangles[blockIdx.x * 10 + threadIdx.x];
			if (i != 0 && threadIdx.x < 10 && *inside)
			{
				float vert2_1[3] = { cudaInsideVertices[cudaInsideTriangles[i].x].x, cudaInsideVertices[cudaInsideTriangles[i].x].y, cudaInsideVertices[cudaInsideTriangles[i].x].z };
				float vert2_2[3] = { cudaInsideVertices[cudaInsideTriangles[i].y].x, cudaInsideVertices[cudaInsideTriangles[i].y].y, cudaInsideVertices[cudaInsideTriangles[i].y].z };
				float vert2_3[3] = { cudaInsideVertices[cudaInsideTriangles[i].z].x, cudaInsideVertices[cudaInsideTriangles[i].z].y, cudaInsideVertices[cudaInsideTriangles[i].z].z };
				if (NoDivTriTriIsect(vert1_1, vert1_2, vert1_3, vert2_1, vert2_2, vert2_3) == 1)
				{
					*inside = false;
					return;
				}
			}
		}
	}