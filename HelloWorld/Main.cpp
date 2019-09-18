#include <cassert>
#include <iostream>
//#include <memory> //needed for smart pointers

#include "Mesh.h"
#include "parse_stl.h"
#include "RayTriangleIntersect.h"

int main(int argc, char* argv[]) {
	std::string stl_file_name = "BoxBin.stl";

	if (argc == 2) {
		stl_file_name = argv[1];
	}
	else if (argc > 2) {
		std::cout << "ERROR: Too many command line arguments" << std::endl;
	}

	//Only reads STL-file in binary format!!!
	auto info = stl::parse_stl(stl_file_name);

	std::vector<stl::triangle> triangles = info.triangles;
	std::cout << "STL HEADER = " << info.name << std::endl;
	std::cout << "# triangles = " << triangles.size() << std::endl;
	


	//std::unique_ptr<float[]> orig = std::make_unique<float[]>(3); //smart pointer
	//std::vector<float> orig;
	float* orig = new float[3];
	orig[0] = 0.0;
	orig[1] = 0.0;
	orig[2] = 0.0;
	float* dir = new float[3];
	dir[0] = -1.0;
	dir[1] = -1.0;
	dir[2] = -0.2;
	float* vert1 = new float[3];
	float* vert2 = new float[3];
	float* vert3 = new float[3];
	float* t = new float;
	float* u = new float;
	float* v = new float;

	for (stl::triangle tri : info.triangles) {
		std::cout << tri << std::endl;
		vert1[0] = tri.v1.x; vert1[1] = tri.v1.y; vert1[2] = tri.v1.z;
		vert2[0] = tri.v2.x; vert2[1] = tri.v2.y; vert2[2] = tri.v2.z;
		vert3[0] = tri.v3.x; vert3[1] = tri.v3.y; vert3[2] = tri.v3.z;
		int snijden = Intersection::intersect_triangle(orig, dir, vert1, vert2, vert3, t, u, v);
		std::cout << "snijden = " << snijden << std::endl;
	}
	delete orig;
	delete dir;
	delete vert1;
	delete vert2;
	delete vert3;
	delete t;
	delete u;
	delete v;

}