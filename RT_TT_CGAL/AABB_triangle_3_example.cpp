// Author(s) : Camille Wormser, Pierre Alliez

#include <iostream>
#include <list>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <streambuf>
#include <memory>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>

typedef CGAL::Simple_cartesian<double> K;

typedef K::FT FT;
typedef K::Ray_3 Ray;
typedef K::Line_3 Line;
typedef K::Point_3 Point;
typedef K::Triangle_3 Triangle;

typedef std::list<Triangle>::iterator Iterator;
typedef CGAL::AABB_triangle_primitive<K, Iterator> Primitive;
typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;

std::list<Triangle> parse_stl(const std::string& stl_path);
std::list<Point> parse_stl2(const std::string& stl_path);
Point parse_point(std::ifstream& s);
float parse_float(std::ifstream& s);
int findDuplicate(const Point& v, std::map<std::string, int>& VertexIndices);

int main()
{
	std::string stl_file_inside;
	std::string stl_file_outside;
	std::cout << "Enter filename of inside mesh:" << std::endl;
	std::cin >> stl_file_inside;
	std::cout << "Enter filename of outside mesh:" << std::endl;
	std::cin >> stl_file_outside;

	bool RT_TT = true; //RayTriangle_TriangleTriangle
	std::cout << "Ray-Triangle (1) or Triangle-Triangle (0)" << std::endl;
	std::cin >> RT_TT;

	if (RT_TT) {
		std::cout << "Reading files:" << std::endl;
		std::list<Point> triangleMesh_Inside = parse_stl2(stl_file_inside);
		std::list<Triangle> triangleMesh_Outside = parse_stl(stl_file_outside);

		/*Point a(1.0, 0.0, 0.0);
		Point b(0.0, 1.0, 0.0);
		Point c(0.0, 0.0, 1.0);
		Point d(0.0, 0.0, 0.0);

		std::list<Triangle> triangles;
		triangles.push_back(Triangle(a,b,c));
		triangles.push_back(Triangle(a,b,d));
		triangles.push_back(Triangle(a,d,c));*/

		auto it0 = std::next(triangleMesh_Outside.begin(), 0);

		Triangle t = *it0;

		Point V1 = t[0];
		Point V2 = t[1];
		Point V3 = t[2];

		float xCenter = (V1[0] + V2[0] + V3[0]) / 3;
		float yCenter = (V1[1] + V2[1] + V3[1]) / 3;
		float zCenter = (V1[2] + V2[2] + V3[2]) / 3;

		Point direction(xCenter, yCenter, zCenter);

		std::cout << "direction = " << direction[0] << ", " << direction[1] << ", " << direction[2] << std::endl;
		std::cout << "# triangles: " << triangleMesh_Outside.size() << std::endl;
		std::cout << "# vertices: " << triangleMesh_Inside.size() << std::endl;

		auto t1 = std::chrono::high_resolution_clock::now(); //start time measurement

		// constructs AABB tree
		Tree tree(triangleMesh_Outside.begin(), triangleMesh_Outside.end());

		/*for (std::list<Triangle>::iterator it = triangleMesh_Outside.begin(); it != triangleMesh_Outside.end(); ++it) {
			std::cout << *it << std::endl;
		}*/

		// counts #intersections
		int counter = 0;
		bool inside = true;
		int numberOfIntersections;
		for (std::list<Point>::iterator it = triangleMesh_Inside.begin(); it != triangleMesh_Inside.end(); ++it) {
			//std::cout << (*it)[0] << ", " << (*it)[1] << ", " << (*it)[2] << ", " << std::endl;
			Ray ray_query(*it, direction);
			//std::cout << tree.number_of_intersected_primitives(ray_query) << " intersections(s) with ray query" << std::endl;
			numberOfIntersections = tree.number_of_intersected_primitives(ray_query);
			//std::cout << "numberOfIntersections: " << numberOfIntersections << std::endl;
			if (numberOfIntersections % 2 == 0) {
				inside = false;
				break;
			}
			//counter += numberOfIntersections;
		}

		auto t2 = std::chrono::high_resolution_clock::now(); //stop time measurement
		auto time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
		std::cout << "Time = " << time << " milliseconds" << std::endl;

		std::cout << "Number of intersections: " << counter << std::endl;
		std::cout << "Inside: " << inside << std::endl;

		std::cout << "Press Enter to quit program!" << std::endl;
		std::cin.get();
		std::cin.get();
	}
	else {
		std::cout << "Reading files:" << std::endl;
		std::list<Triangle> triangleMesh_Inside = parse_stl(stl_file_inside);
		std::list<Triangle> triangleMesh_Outside = parse_stl(stl_file_outside);

		/*auto it0 = std::next(triangleMesh_Outside.begin(), 0);

		Triangle t = *it0;

		Point V1 = t[0];
		Point V2 = t[1];
		Point V3 = t[2];

		float xCenter = (V1[0] + V2[0] + V3[0]) / 3;
		float yCenter = (V1[1] + V2[1] + V3[1]) / 3;
		float zCenter = (V1[2] + V2[2] + V3[2]) / 3;

		Point direction(xCenter, yCenter, zCenter);*/

		//std::cout << "direction = " << direction[0] << ", " << direction[1] << ", " << direction[2] << std::endl;
		std::cout << "# triangles outer mesh: " << triangleMesh_Outside.size() << std::endl;
		std::cout << "# triangles inner mesh: " << triangleMesh_Inside.size() << std::endl;

		auto t1 = std::chrono::high_resolution_clock::now(); //start time measurement

		// constructs AABB tree
		Tree tree(triangleMesh_Outside.begin(), triangleMesh_Outside.end());

		// counts #intersections
		int counter = 0;
		bool inside = true;
		bool intersection;
		std::list<Triangle>::iterator it = triangleMesh_Inside.begin();
		while (it != triangleMesh_Inside.end() && inside) {
			//std::cout << (*it)[0] << ", " << (*it)[1] << ", " << (*it)[2] << ", " << std::endl;
			//std::cout << tree.number_of_intersected_primitives(ray_query) << " intersections(s) with ray query" << std::endl;
			intersection = tree.do_intersect(*it);
			//std::cout << "numberOfIntersections: " << numberOfIntersections << std::endl;
			inside = !intersection;
			//counter += numberOfIntersections;
			++it;
		}

		auto t2 = std::chrono::high_resolution_clock::now(); //stop time measurement
		auto time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
		std::cout << "Time = " << time << " milliseconds" << std::endl;

		std::cout << "Number of intersections: " << counter << std::endl;
		std::cout << "Inside: " << inside << std::endl;

		std::cout << "Press Enter to quit program!" << std::endl;
		std::cin.get();
		std::cin.get();
	}
	
    return EXIT_SUCCESS;
}

std::list<Triangle> parse_stl(const std::string& stl_path) {

	std::map<std::string, int> VertexIndices;
	std::list<Point> vertices;
	std::list<Triangle> triangles;

	std::ifstream stl_file(stl_path.c_str(), std::ios::in | std::ios::binary);
	if (!stl_file) {
		std::cout << "ERROR: COULD NOT READ FILE" << std::endl;
		assert(false);
	}

	char header_info[80] = "";
	char n_triangles[4];
	stl_file.read(header_info, 80);
	stl_file.read(n_triangles, 4);
	std::string h(header_info);
	unsigned int* num_triangles = (unsigned int*)n_triangles;
	for (unsigned int i = 0; i < *num_triangles; i++) {
		Point p = parse_point(stl_file); //normalvector --> wordt niet gebruikt!
		int triangleVertices[3];
		int duplicateVertexIndex;
		for (int i = 0; i < 3; i++)
		{
			Point p = parse_point(stl_file);
			duplicateVertexIndex = findDuplicate(p, VertexIndices);//to string methode niet 2 keer oproepen
			std::string s = std::to_string(p[0]) + std::to_string(p[1]) + std::to_string(p[2]);
			VertexIndices.insert(std::pair<std::string, int>(s, vertices.size()));
			if (duplicateVertexIndex == -1)
			{
				vertices.push_back(p);
				triangleVertices[i] = vertices.size() - 1;
			}
			else
			{
				triangleVertices[i] = duplicateVertexIndex;
			}
		}
		auto it0 = std::next(vertices.begin(), triangleVertices[0]);
		auto it1 = std::next(vertices.begin(), triangleVertices[1]);
		auto it2 = std::next(vertices.begin(), triangleVertices[2]);
		triangles.push_back(Triangle(*it0, *it1, *it2));
		char dummy[2];
		stl_file.read(dummy, 2);
	}
	return triangles;
}

std::list<Point> parse_stl2(const std::string& stl_path) {

	std::map<std::string, int> VertexIndices;
	std::list<Point> vertices;
	std::list<Triangle> triangles;

	std::ifstream stl_file(stl_path.c_str(), std::ios::in | std::ios::binary);
	if (!stl_file) {
		std::cout << "ERROR: COULD NOT READ FILE" << std::endl;
		assert(false);
	}

	char header_info[80] = "";
	char n_triangles[4];
	stl_file.read(header_info, 80);
	stl_file.read(n_triangles, 4);
	std::string h(header_info);
	unsigned int* num_triangles = (unsigned int*)n_triangles;
	for (unsigned int i = 0; i < *num_triangles; i++) {
		Point p = parse_point(stl_file); //normalvector --> wordt niet gebruikt!
		int triangleVertices[3];
		int duplicateVertexIndex;
		for (int i = 0; i < 3; i++)
		{
			Point p = parse_point(stl_file);
			duplicateVertexIndex = findDuplicate(p, VertexIndices);
			std::string s = std::to_string(p[0]) + std::to_string(p[1]) + std::to_string(p[2]);
			VertexIndices.insert(std::pair<std::string, int>(s, vertices.size()));
			if (duplicateVertexIndex == -1)
			{
				vertices.push_back(p);
				triangleVertices[i] = vertices.size() - 1;
			}
			else
			{
				triangleVertices[i] = duplicateVertexIndex;
			}
		}
		auto it0 = std::next(vertices.begin(), triangleVertices[0]);
		auto it1 = std::next(vertices.begin(), triangleVertices[1]);
		auto it2 = std::next(vertices.begin(), triangleVertices[2]);
		triangles.push_back(Triangle(*it0, *it1, *it2));
		char dummy[2];
		stl_file.read(dummy, 2);
	}
	return vertices;
}

Point parse_point(std::ifstream& s) {
	float x = parse_float(s);
	float y = parse_float(s);
	float z = parse_float(s);
	Point v(x, y, z);
	return v;
}

float parse_float(std::ifstream& s) {
	char f_buf[sizeof(float)];
	s.read(f_buf, 4);
	float* fptr = (float*)f_buf;
	return *fptr;
}

int findDuplicate(const Point& v, std::map<std::string, int>& VertexIndices)
{
	std::string s = std::to_string(v[0]) + std::to_string(v[1]) + std::to_string(v[2]);
	std::map<std::string, int>::iterator itr = VertexIndices.find(s);
	if (itr != VertexIndices.end())
	{
		return itr->second;
	}
	else
	{
		return -1;
	}
}