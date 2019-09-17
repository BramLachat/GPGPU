#ifndef RAYTRIANGLEINTERSECT_H
#define RAYTRIANGLEINTERSECT_H

namespace Intersection {

	int intersect_triangle(double orig[3], double dir[3],
		double vert0[3], double vert1[3], double vert2[3],
		double* t, double* u, double* v);
	
	int intersect_triangle1(double orig[3], double dir[3],
		double vert0[3], double vert1[3], double vert2[3],
		double* t, double* u, double* v);
	
	int intersect_triangle2(double orig[3], double dir[3],
		double vert0[3], double vert1[3], double vert2[3],
		double* t, double* u, double* v);
	
	int intersect_triangle3(float orig[3], float dir[3],
		float vert0[3], float vert1[3], float vert2[3],
		float* t, float* u, float* v);

}

#endif
