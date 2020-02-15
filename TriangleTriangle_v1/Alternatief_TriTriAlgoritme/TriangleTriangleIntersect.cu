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

 /* function prototype */

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






/* some 3D macros */

#define CROSS(dest,v1,v2)                       \
               dest[0]=v1[1]*v2[2]-v1[2]*v2[1]; \
               dest[1]=v1[2]*v2[0]-v1[0]*v2[2]; \
               dest[2]=v1[0]*v2[1]-v1[1]*v2[0];

#define DOT(v1,v2) (v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])



#define SUB(dest,v1,v2) dest[0]=v1[0]-v2[0]; \
                        dest[1]=v1[1]-v2[1]; \
                        dest[2]=v1[2]-v2[2]; 


#define SCALAR(dest,alpha,v) dest[0] = alpha * v[0]; \
                             dest[1] = alpha * v[1]; \
                             dest[2] = alpha * v[2];



#define CHECK_MIN_MAX(p1,q1,r1,p2,q2,r2) {\
  SUB(v1,p2,q1)\
  SUB(v2,p1,q1)\
  CROSS(N1,v1,v2)\
  SUB(v1,q2,q1)\
  if (DOT(v1,N1) > 0.0f) return 0;\
  SUB(v1,p2,p1)\
  SUB(v2,r1,p1)\
  CROSS(N1,v1,v2)\
  SUB(v1,r2,p1) \
  if (DOT(v1,N1) > 0.0f) return 0;\
  else return 1; }



/* Permutation in a canonical form of T2's vertices */

#define TRI_TRI_3D(p1,q1,r1,p2,q2,r2,dp2,dq2,dr2) { \
  if (dp2 > 0.0f) { \
     if (dq2 > 0.0f) CHECK_MIN_MAX(p1,r1,q1,r2,p2,q2) \
     else if (dr2 > 0.0f) CHECK_MIN_MAX(p1,r1,q1,q2,r2,p2)\
     else CHECK_MIN_MAX(p1,q1,r1,p2,q2,r2) }\
  else if (dp2 < 0.0f) { \
    if (dq2 < 0.0f) CHECK_MIN_MAX(p1,q1,r1,r2,p2,q2)\
    else if (dr2 < 0.0f) CHECK_MIN_MAX(p1,q1,r1,q2,r2,p2)\
    else CHECK_MIN_MAX(p1,r1,q1,p2,q2,r2)\
  } else { \
    if (dq2 < 0.0f) { \
      if (dr2 >= 0.0f)  CHECK_MIN_MAX(p1,r1,q1,q2,r2,p2)\
      else CHECK_MIN_MAX(p1,q1,r1,p2,q2,r2)\
    } \
    else if (dq2 > 0.0f) { \
      if (dr2 > 0.0f) CHECK_MIN_MAX(p1,r1,q1,p2,q2,r2)\
      else  CHECK_MIN_MAX(p1,q1,r1,q2,r2,p2)\
    } \
    else  { \
      if (dr2 > 0.0f) CHECK_MIN_MAX(p1,q1,r1,r2,p2,q2)\
      else if (dr2 < 0.0f) CHECK_MIN_MAX(p1,r1,q1,r2,p2,q2)\
      else return coplanar_tri_tri3d(p1,q1,r1,p2,q2,r2,N1,N2);\
     }}}



/*
*
*  Three-dimensional Triangle-Triangle Overlap Test
*
*/


__host__ __device__ int tri_tri_overlap_test_3d(float p1[3], float q1[3], float r1[3],

	float p2[3], float q2[3], float r2[3])
{
	float dp1, dq1, dr1, dp2, dq2, dr2;
	float v1[3], v2[3];
	float N1[3], N2[3];

	/* Compute distance signs  of p1, q1 and r1 to the plane of
	   triangle(p2,q2,r2) */


	SUB(v1, p2, r2)
		SUB(v2, q2, r2)
		CROSS(N2, v1, v2)

		SUB(v1, p1, r2)
		dp1 = DOT(v1, N2);
	SUB(v1, q1, r2)
		dq1 = DOT(v1, N2);
	SUB(v1, r1, r2)
		dr1 = DOT(v1, N2);

	if (((dp1 * dq1) > 0.0f) && ((dp1 * dr1) > 0.0f))  return 0;

	/* Compute distance signs  of p2, q2 and r2 to the plane of
	   triangle(p1,q1,r1) */


	SUB(v1, q1, p1)
		SUB(v2, r1, p1)
		CROSS(N1, v1, v2)

		SUB(v1, p2, r1)
		dp2 = DOT(v1, N1);
	SUB(v1, q2, r1)
		dq2 = DOT(v1, N1);
	SUB(v1, r2, r1)
		dr2 = DOT(v1, N1);

	if (((dp2 * dq2) > 0.0f) && ((dp2 * dr2) > 0.0f)) return 0;

	/* Permutation in a canonical form of T1's vertices */


	if (dp1 > 0.0f) {
		if (dq1 > 0.0f) TRI_TRI_3D(r1, p1, q1, p2, r2, q2, dp2, dr2, dq2)
		else if (dr1 > 0.0f) TRI_TRI_3D(q1, r1, p1, p2, r2, q2, dp2, dr2, dq2)
		else TRI_TRI_3D(p1, q1, r1, p2, q2, r2, dp2, dq2, dr2)
	}
	else if (dp1 < 0.0f) {
		if (dq1 < 0.0f) TRI_TRI_3D(r1, p1, q1, p2, q2, r2, dp2, dq2, dr2)
		else if (dr1 < 0.0f) TRI_TRI_3D(q1, r1, p1, p2, q2, r2, dp2, dq2, dr2)
		else TRI_TRI_3D(p1, q1, r1, p2, r2, q2, dp2, dr2, dq2)
	}
	else {
		if (dq1 < 0.0f) {
			if (dr1 >= 0.0f) TRI_TRI_3D(q1, r1, p1, p2, r2, q2, dp2, dr2, dq2)
			else TRI_TRI_3D(p1, q1, r1, p2, q2, r2, dp2, dq2, dr2)
		}
		else if (dq1 > 0.0f) {
			if (dr1 > 0.0f) TRI_TRI_3D(p1, q1, r1, p2, r2, q2, dp2, dr2, dq2)
			else TRI_TRI_3D(q1, r1, p1, p2, q2, r2, dp2, dq2, dr2)
		}
		else {
			if (dr1 > 0.0f) TRI_TRI_3D(r1, p1, q1, p2, q2, r2, dp2, dq2, dr2)
			else if (dr1 < 0.0f) TRI_TRI_3D(r1, p1, q1, p2, r2, q2, dp2, dr2, dq2)
			else return coplanar_tri_tri3d(p1, q1, r1, p2, q2, r2, N1, N2);
		}
	}
};



__host__ __device__ int coplanar_tri_tri3d(float p1[3], float q1[3], float r1[3],
	float p2[3], float q2[3], float r2[3],
	float normal_1[3], float normal_2[3]) {

	float P1[2], Q1[2], R1[2];
	float P2[2], Q2[2], R2[2];

	float n_x, n_y, n_z;

	n_x = ((normal_1[0] < 0) ? -normal_1[0] : normal_1[0]);
	n_y = ((normal_1[1] < 0) ? -normal_1[1] : normal_1[1]);
	n_z = ((normal_1[2] < 0) ? -normal_1[2] : normal_1[2]);


	/* Projection of the triangles in 3D onto 2D such that the area of
	   the projection is maximized. */


	if ((n_x > n_z) && (n_x >= n_y)) {
		// Project onto plane YZ

		P1[0] = q1[2]; P1[1] = q1[1];
		Q1[0] = p1[2]; Q1[1] = p1[1];
		R1[0] = r1[2]; R1[1] = r1[1];

		P2[0] = q2[2]; P2[1] = q2[1];
		Q2[0] = p2[2]; Q2[1] = p2[1];
		R2[0] = r2[2]; R2[1] = r2[1];

	}
	else if ((n_y > n_z) && (n_y >= n_x)) {
		// Project onto plane XZ

		P1[0] = q1[0]; P1[1] = q1[2];
		Q1[0] = p1[0]; Q1[1] = p1[2];
		R1[0] = r1[0]; R1[1] = r1[2];

		P2[0] = q2[0]; P2[1] = q2[2];
		Q2[0] = p2[0]; Q2[1] = p2[2];
		R2[0] = r2[0]; R2[1] = r2[2];

	}
	else {
		// Project onto plane XY

		P1[0] = p1[0]; P1[1] = p1[1];
		Q1[0] = q1[0]; Q1[1] = q1[1];
		R1[0] = r1[0]; R1[1] = r1[1];

		P2[0] = p2[0]; P2[1] = p2[1];
		Q2[0] = q2[0]; Q2[1] = q2[1];
		R2[0] = r2[0]; R2[1] = r2[1];
	}

	return tri_tri_overlap_test_2d(P1, Q1, R1, P2, Q2, R2);

};



/*
*
*  Three-dimensional Triangle-Triangle Intersection
*
*/

/*
   This macro is called when the triangles surely intersect
   It constructs the segment of intersection of the two triangles
   if they are not coplanar.
*/

#define CONSTRUCT_INTERSECTION(p1,q1,r1,p2,q2,r2) { \
  SUB(v1,q1,p1) \
  SUB(v2,r2,p1) \
  CROSS(N,v1,v2) \
  SUB(v,p2,p1) \
  if (DOT(v,N) > 0.0f) {\
    SUB(v1,r1,p1) \
    CROSS(N,v1,v2) \
    if (DOT(v,N) <= 0.0f) { \
      SUB(v2,q2,p1) \
      CROSS(N,v1,v2) \
      if (DOT(v,N) > 0.0f) { \
  SUB(v1,p1,p2) \
  SUB(v2,p1,r1) \
  alpha = DOT(v1,N2) / DOT(v2,N2); \
  SCALAR(v1,alpha,v2) \
  SUB(source,p1,v1) \
  SUB(v1,p2,p1) \
  SUB(v2,p2,r2) \
  alpha = DOT(v1,N1) / DOT(v2,N1); \
  SCALAR(v1,alpha,v2) \
  SUB(target,p2,v1) \
  return 1; \
      } else { \
  SUB(v1,p2,p1) \
  SUB(v2,p2,q2) \
  alpha = DOT(v1,N1) / DOT(v2,N1); \
  SCALAR(v1,alpha,v2) \
  SUB(source,p2,v1) \
  SUB(v1,p2,p1) \
  SUB(v2,p2,r2) \
  alpha = DOT(v1,N1) / DOT(v2,N1); \
  SCALAR(v1,alpha,v2) \
  SUB(target,p2,v1) \
  return 1; \
      } \
    } else { \
      return 0; \
    } \
  } else { \
    SUB(v2,q2,p1) \
    CROSS(N,v1,v2) \
    if (DOT(v,N) < 0.0f) { \
      return 0; \
    } else { \
      SUB(v1,r1,p1) \
      CROSS(N,v1,v2) \
      if (DOT(v,N) >= 0.0f) { \
  SUB(v1,p1,p2) \
  SUB(v2,p1,r1) \
  alpha = DOT(v1,N2) / DOT(v2,N2); \
  SCALAR(v1,alpha,v2) \
  SUB(source,p1,v1) \
  SUB(v1,p1,p2) \
  SUB(v2,p1,q1) \
  alpha = DOT(v1,N2) / DOT(v2,N2); \
  SCALAR(v1,alpha,v2) \
  SUB(target,p1,v1) \
  return 1; \
      } else { \
  SUB(v1,p2,p1) \
  SUB(v2,p2,q2) \
  alpha = DOT(v1,N1) / DOT(v2,N1); \
  SCALAR(v1,alpha,v2) \
  SUB(source,p2,v1) \
  SUB(v1,p1,p2) \
  SUB(v2,p1,q1) \
  alpha = DOT(v1,N2) / DOT(v2,N2); \
  SCALAR(v1,alpha,v2) \
  SUB(target,p1,v1) \
  return 1; \
      }}}} 



#define TRI_TRI_INTER_3D(p1,q1,r1,p2,q2,r2,dp2,dq2,dr2) { \
  if (dp2 > 0.0f) { \
     if (dq2 > 0.0f) CONSTRUCT_INTERSECTION(p1,r1,q1,r2,p2,q2) \
     else if (dr2 > 0.0f) CONSTRUCT_INTERSECTION(p1,r1,q1,q2,r2,p2)\
     else CONSTRUCT_INTERSECTION(p1,q1,r1,p2,q2,r2) }\
  else if (dp2 < 0.0f) { \
    if (dq2 < 0.0f) CONSTRUCT_INTERSECTION(p1,q1,r1,r2,p2,q2)\
    else if (dr2 < 0.0f) CONSTRUCT_INTERSECTION(p1,q1,r1,q2,r2,p2)\
    else CONSTRUCT_INTERSECTION(p1,r1,q1,p2,q2,r2)\
  } else { \
    if (dq2 < 0.0f) { \
      if (dr2 >= 0.0f)  CONSTRUCT_INTERSECTION(p1,r1,q1,q2,r2,p2)\
      else CONSTRUCT_INTERSECTION(p1,q1,r1,p2,q2,r2)\
    } \
    else if (dq2 > 0.0f) { \
      if (dr2 > 0.0f) CONSTRUCT_INTERSECTION(p1,r1,q1,p2,q2,r2)\
      else  CONSTRUCT_INTERSECTION(p1,q1,r1,q2,r2,p2)\
    } \
    else  { \
      if (dr2 > 0.0f) CONSTRUCT_INTERSECTION(p1,q1,r1,r2,p2,q2)\
      else if (dr2 < 0.0f) CONSTRUCT_INTERSECTION(p1,r1,q1,r2,p2,q2)\
      else { \
        *coplanar = 1; \
  return coplanar_tri_tri3d(p1,q1,r1,p2,q2,r2,N1,N2);\
     } \
  }} }


/*
   The following version computes the segment of intersection of the
   two triangles if it exists.
   coplanar returns whether the triangles are coplanar
   source and target are the endpoints of the line segment of intersection
*/

__host__ __device__ int tri_tri_intersection_test_3d(float p1[3], float q1[3], float r1[3],
	float p2[3], float q2[3], float r2[3],
	int* coplanar,
	float source[3], float target[3])

{
	float dp1, dq1, dr1, dp2, dq2, dr2;
	float v1[3], v2[3], v[3];
	float N1[3], N2[3], N[3];
	float alpha;

	// Compute distance signs  of p1, q1 and r1 
	// to the plane of triangle(p2,q2,r2)


	SUB(v1, p2, r2)
		SUB(v2, q2, r2)
		CROSS(N2, v1, v2)

		SUB(v1, p1, r2)
		dp1 = DOT(v1, N2);
	SUB(v1, q1, r2)
		dq1 = DOT(v1, N2);
	SUB(v1, r1, r2)
		dr1 = DOT(v1, N2);

	if (((dp1 * dq1) > 0.0f) && ((dp1 * dr1) > 0.0f))  return 0;

	// Compute distance signs  of p2, q2 and r2 
	// to the plane of triangle(p1,q1,r1)


	SUB(v1, q1, p1)
		SUB(v2, r1, p1)
		CROSS(N1, v1, v2)

		SUB(v1, p2, r1)
		dp2 = DOT(v1, N1);
	SUB(v1, q2, r1)
		dq2 = DOT(v1, N1);
	SUB(v1, r2, r1)
		dr2 = DOT(v1, N1);

	if (((dp2 * dq2) > 0.0f) && ((dp2 * dr2) > 0.0f)) return 0;

	// Permutation in a canonical form of T1's vertices


	if (dp1 > 0.0f) {
		if (dq1 > 0.0f) TRI_TRI_INTER_3D(r1, p1, q1, p2, r2, q2, dp2, dr2, dq2)
		else if (dr1 > 0.0f) TRI_TRI_INTER_3D(q1, r1, p1, p2, r2, q2, dp2, dr2, dq2)

		else TRI_TRI_INTER_3D(p1, q1, r1, p2, q2, r2, dp2, dq2, dr2)
	}
	else if (dp1 < 0.0f) {
		if (dq1 < 0.0f) TRI_TRI_INTER_3D(r1, p1, q1, p2, q2, r2, dp2, dq2, dr2)
		else if (dr1 < 0.0f) TRI_TRI_INTER_3D(q1, r1, p1, p2, q2, r2, dp2, dq2, dr2)
		else TRI_TRI_INTER_3D(p1, q1, r1, p2, r2, q2, dp2, dr2, dq2)
	}
	else {
		if (dq1 < 0.0f) {
			if (dr1 >= 0.0f) TRI_TRI_INTER_3D(q1, r1, p1, p2, r2, q2, dp2, dr2, dq2)
			else TRI_TRI_INTER_3D(p1, q1, r1, p2, q2, r2, dp2, dq2, dr2)
		}
		else if (dq1 > 0.0f) {
			if (dr1 > 0.0f) TRI_TRI_INTER_3D(p1, q1, r1, p2, r2, q2, dp2, dr2, dq2)
			else TRI_TRI_INTER_3D(q1, r1, p1, p2, q2, r2, dp2, dq2, dr2)
		}
		else {
			if (dr1 > 0.0f) TRI_TRI_INTER_3D(r1, p1, q1, p2, q2, r2, dp2, dq2, dr2)
			else if (dr1 < 0.0f) TRI_TRI_INTER_3D(r1, p1, q1, p2, r2, q2, dp2, dr2, dq2)
			else {
				// triangles are co-planar

				*coplanar = 1;
				return coplanar_tri_tri3d(p1, q1, r1, p2, q2, r2, N1, N2);
			}
		}
	}
};





/*
*
*  Two dimensional Triangle-Triangle Overlap Test
*
*/


/* some 2D macros */

#define ORIENT_2D(a, b, c)  ((a[0]-c[0])*(b[1]-c[1])-(a[1]-c[1])*(b[0]-c[0]))


#define INTERSECTION_TEST_VERTEX(P1, Q1, R1, P2, Q2, R2) {\
  if (ORIENT_2D(R2,P2,Q1) >= 0.0f)\
    if (ORIENT_2D(R2,Q2,Q1) <= 0.0f)\
      if (ORIENT_2D(P1,P2,Q1) > 0.0f) {\
  if (ORIENT_2D(P1,Q2,Q1) <= 0.0f) return 1; \
  else return 0;} else {\
  if (ORIENT_2D(P1,P2,R1) >= 0.0f)\
    if (ORIENT_2D(Q1,R1,P2) >= 0.0f) return 1; \
    else return 0;\
  else return 0;}\
    else \
      if (ORIENT_2D(P1,Q2,Q1) <= 0.0f)\
  if (ORIENT_2D(R2,Q2,R1) <= 0.0f)\
    if (ORIENT_2D(Q1,R1,Q2) >= 0.0f) return 1; \
    else return 0;\
  else return 0;\
      else return 0;\
  else\
    if (ORIENT_2D(R2,P2,R1) >= 0.0f) \
      if (ORIENT_2D(Q1,R1,R2) >= 0.0f)\
  if (ORIENT_2D(P1,P2,R1) >= 0.0f) return 1;\
  else return 0;\
      else \
  if (ORIENT_2D(Q1,R1,Q2) >= 0.0f) {\
    if (ORIENT_2D(R2,R1,Q2) >= 0.0f) return 1; \
    else return 0; }\
  else return 0; \
    else  return 0; \
 };



#define INTERSECTION_TEST_EDGE(P1, Q1, R1, P2, Q2, R2) { \
  if (ORIENT_2D(R2,P2,Q1) >= 0.0f) {\
    if (ORIENT_2D(P1,P2,Q1) >= 0.0f) { \
        if (ORIENT_2D(P1,Q1,R2) >= 0.0f) return 1; \
        else return 0;} else { \
      if (ORIENT_2D(Q1,R1,P2) >= 0.0f){ \
  if (ORIENT_2D(R1,P1,P2) >= 0.0f) return 1; else return 0;} \
      else return 0; } \
  } else {\
    if (ORIENT_2D(R2,P2,R1) >= 0.0f) {\
      if (ORIENT_2D(P1,P2,R1) >= 0.0f) {\
  if (ORIENT_2D(P1,R1,R2) >= 0.0f) return 1;  \
  else {\
    if (ORIENT_2D(Q1,R1,R2) >= 0.0f) return 1; else return 0;}}\
      else  return 0; }\
    else return 0; }}



__host__ __device__ int ccw_tri_tri_intersection_2d(float p1[2], float q1[2], float r1[2],
	float p2[2], float q2[2], float r2[2]) {
	if (ORIENT_2D(p2, q2, p1) >= 0.0f) {
		if (ORIENT_2D(q2, r2, p1) >= 0.0f) {
			if (ORIENT_2D(r2, p2, p1) >= 0.0f) return 1;
			else INTERSECTION_TEST_EDGE(p1, q1, r1, p2, q2, r2)
		}
		else {
			if (ORIENT_2D(r2, p2, p1) >= 0.0f)
				INTERSECTION_TEST_EDGE(p1, q1, r1, r2, p2, q2)
			else INTERSECTION_TEST_VERTEX(p1, q1, r1, p2, q2, r2)
		}
	}
	else {
		if (ORIENT_2D(q2, r2, p1) >= 0.0f) {
			if (ORIENT_2D(r2, p2, p1) >= 0.0f)
				INTERSECTION_TEST_EDGE(p1, q1, r1, q2, r2, p2)
			else  INTERSECTION_TEST_VERTEX(p1, q1, r1, q2, r2, p2)
		}
		else INTERSECTION_TEST_VERTEX(p1, q1, r1, r2, p2, q2)
	}
};


__host__ __device__ int tri_tri_overlap_test_2d(float p1[2], float q1[2], float r1[2],
	float p2[2], float q2[2], float r2[2]) {
	if (ORIENT_2D(p1, q1, r1) < 0.0f)
		if (ORIENT_2D(p2, q2, r2) < 0.0f)
			return ccw_tri_tri_intersection_2d(p1, r1, q1, p2, r2, q2);
		else
			return ccw_tri_tri_intersection_2d(p1, r1, q1, p2, q2, r2);
	else
		if (ORIENT_2D(p2, q2, r2) < 0.0f)
			return ccw_tri_tri_intersection_2d(p1, q1, r1, p2, r2, q2);
		else
			return ccw_tri_tri_intersection_2d(p1, q1, r1, p2, q2, r2);

};


//thread per triangle
__global__ void triangle_triangle_GPU(int3* cudaInsideTriangles, float3* cudaInsideVertices, int3* cudaOutsideTriangles, float3* cudaOutsideVertices, bool* inside, int numberOfInsideTriangles, int numberOfOutsideTriangles, float2* cudaOutsideTriangleIntervals) { // , int* cudaIntersectionsPerInsideTriangle
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < numberOfInsideTriangles)
	{
		float vert1_1[3] = { cudaInsideVertices[cudaInsideTriangles[tid].x].x, cudaInsideVertices[cudaInsideTriangles[tid].x].y, cudaInsideVertices[cudaInsideTriangles[tid].x].z };
		float vert1_2[3] = { cudaInsideVertices[cudaInsideTriangles[tid].y].x, cudaInsideVertices[cudaInsideTriangles[tid].y].y, cudaInsideVertices[cudaInsideTriangles[tid].y].z };
		float vert1_3[3] = { cudaInsideVertices[cudaInsideTriangles[tid].z].x, cudaInsideVertices[cudaInsideTriangles[tid].z].y, cudaInsideVertices[cudaInsideTriangles[tid].z].z };

		/*float max_temp = (vert1_1[0] < vert1_2[0]) ? vert1_2[0] : vert1_1[0];
		float max =  ((max_temp < vert1_3[0]) ? vert1_3[0] : max_temp);

		float min_temp = (vert1_1[0] > vert1_2[0]) ? vert1_2[0] : vert1_1[0];
		float min = ((min_temp > vert1_3[0]) ? vert1_3[0] : min_temp);*/

		//int numberOfIntersections = 0;
		for (int i = 0; i < numberOfOutsideTriangles; i++)
		{
			if (*inside) {
				float vert2_1[3] = { cudaOutsideVertices[cudaOutsideTriangles[i].x].x, cudaOutsideVertices[cudaOutsideTriangles[i].x].y, cudaOutsideVertices[cudaOutsideTriangles[i].x].z };
				float vert2_2[3] = { cudaOutsideVertices[cudaOutsideTriangles[i].y].x, cudaOutsideVertices[cudaOutsideTriangles[i].y].y, cudaOutsideVertices[cudaOutsideTriangles[i].y].z };
				float vert2_3[3] = { cudaOutsideVertices[cudaOutsideTriangles[i].z].x, cudaOutsideVertices[cudaOutsideTriangles[i].z].y, cudaOutsideVertices[cudaOutsideTriangles[i].z].z };
				//if(cudaOutsideTriangleIntervals[i].x <= max && cudaOutsideTriangleIntervals[i].y >= min) // Broad Phase Collision Detection (x = min, y = max) 
				{
					if (tri_tri_overlap_test_3d(vert1_1, vert1_2, vert1_3, vert2_1, vert2_2, vert2_3) == 1)
					{
						//numberOfIntersections++;
						*inside = false;
						return;
						//cudaIntersectionsPerInsideTriangle[tid] = 1; // Sneller als je dit weg laat in het geval de meshes elkaar niet sijden ==> dit zorgt er voor dat het trager wordt als de meshes in elkaar liggen
					}
					//if(intersect){ cudaIntersectionsPerInsideTriangle[tid] = 1; } // Sneller als je dit weg laat in het geval de meshes elkaar niet sijden
				}
			}
			else {
				return;
			}
		}
		//printf("numberOfIntersections = %d\n", numberOfIntersections);
		//cudaIntersectionsPerInsideTriangle[tid] = numberOfIntersections;
	}
}