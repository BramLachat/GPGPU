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
void writeResultsToFile(std::vector<std::string>& result);

/* Console output wegschrijven naar file*/
std::vector<std::string> output;

int main()
{
	std::string stl_file_inside;
	std::string stl_file_outside;
	std::cout << "Enter filename of inside mesh:" << std::endl;
	std::cin >> stl_file_inside;
	std::cout << "Enter filename of outside mesh:" << std::endl;
	std::cin >> stl_file_outside;


	std::string delimiter = "\\";

	//Only reads STL-file in binary format!!!
	std::cout << "Reading file: triangleMesh_Inside_Point" << std::endl;
	std::list<Point> triangleMesh_Inside_Point = parse_stl2(stl_file_inside);
	std::cout << "Reading file: triangleMesh_Inside_Triangle" << std::endl;
	std::list<Triangle> triangleMesh_Inside_Triangle = parse_stl(stl_file_inside);
	std::cout << "Reading file: triangleMesh_Outside" << std::endl;
	std::list<Triangle> triangleMesh_Outside = parse_stl(stl_file_outside);

	size_t pos = 0;
	std::string token;
	while ((pos = stl_file_inside.find(delimiter)) != std::string::npos) {
		token = stl_file_inside.substr(0, pos);
		stl_file_inside.erase(0, pos + delimiter.length());
	}
	stl_file_inside = stl_file_inside.substr(0, stl_file_inside.find(".stl"));

	pos = 0;
	while ((pos = stl_file_outside.find(delimiter)) != std::string::npos) {
		token = stl_file_outside.substr(0, pos);
		stl_file_outside.erase(0, pos + delimiter.length());
	}
	stl_file_outside = stl_file_outside.substr(0, stl_file_outside.find(".stl"));


	std::cout << "Calculating file: " << stl_file_inside << "-" << stl_file_outside << std::endl;

	output.push_back(stl_file_inside + "-" + stl_file_outside + ";");

	//output.push_back(";RT_CGAL(ms);;TT_CGAL(ms);\n");

	/**********************************************************************************************
								CGAL - Ray Triangle algorithm
	***********************************************************************************************/

	auto it0 = std::next(triangleMesh_Outside.begin(), 0);

	Triangle t = *it0;

	Point V1 = t[0];
	Point V2 = t[1];
	Point V3 = t[2];

	float xCenter = (V1[0] + V2[0] + V3[0]) / 3;
	float yCenter = (V1[1] + V2[1] + V3[1]) / 3;
	float zCenter = (V1[2] + V2[2] + V3[2]) / 3;

	Point direction(xCenter, yCenter, zCenter);

	// constructs AABB tree
	Tree tree(triangleMesh_Outside.begin(), triangleMesh_Outside.end());

	/*for (std::list<Triangle>::iterator it = triangleMesh_Outside.begin(); it != triangleMesh_Outside.end(); ++it) {
		std::cout << *it << std::endl;
	}*/

	// counts #intersections
	int counter = 0;
	bool inside = true;
	int numberOfIntersections;

	auto t1 = std::chrono::high_resolution_clock::now(); //start time measurement

	for (std::list<Point>::iterator it = triangleMesh_Inside_Point.begin(); it != triangleMesh_Inside_Point.end(); ++it) {
		//std::cout << (*it)[0] << ", " << (*it)[1] << ", " << (*it)[2] << ", " << std::endl;
		Ray ray_query(*it, direction);
		//std::cout << tree.number_of_intersected_primitives(ray_query) << " intersections(s) with ray query" << std::endl;
		numberOfIntersections = tree.number_of_intersected_primitives(ray_query);
		//std::cout << "numberOfIntersections: " << numberOfIntersections << std::endl;
		if (numberOfIntersections % 2 == 0) {
			inside = false;
			//break;
		}
		//counter += numberOfIntersections;
	}

	auto t2 = std::chrono::high_resolution_clock::now(); //stop time measurement
	auto time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

	std::string result;
	if (inside) result = "INSIDE";
	else result = "OUTSIDE";
	output.push_back(std::to_string(time) + ";" + result + ";");

	/*std::cout << "Time = " << time << " milliseconds" << std::endl;

	std::cout << "Number of intersections: " << counter << std::endl;
	std::cout << "Inside: " << inside << std::endl;*/

	/**********************************************************************************************
								CGAL - Triangle Triangle algorithm
	***********************************************************************************************/

	// constructs AABB tree
	//Tree tree(triangleMesh_Outside.begin(), triangleMesh_Outside.end());

	// counts #intersections
	counter = 0;
	inside = true;
	bool intersection;

	t1 = std::chrono::high_resolution_clock::now(); //start time measurement

	std::list<Triangle>::iterator it = triangleMesh_Inside_Triangle.begin();
	while (it != triangleMesh_Inside_Triangle.end()) { // && inside
		//std::cout << (*it)[0] << ", " << (*it)[1] << ", " << (*it)[2] << ", " << std::endl;
		//std::cout << tree.number_of_intersected_primitives(ray_query) << " intersections(s) with ray query" << std::endl;
		intersection = tree.do_intersect(*it);
		//std::cout << "numberOfIntersections: " << numberOfIntersections << std::endl;
		inside = !intersection;
		//counter += numberOfIntersections;
		++it;
	}

	t2 = std::chrono::high_resolution_clock::now(); //stop time measurement
	time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

	if (inside) result = "INSIDE";
	else result = "OUTSIDE";
	output.push_back(std::to_string(time) + ";" + result + "\n");

	/*std::cout << "Time = " << time << " milliseconds" << std::endl;

	std::cout << "Number of intersections: " << counter << std::endl;
	std::cout << "Inside: " << inside << std::endl;*/

	writeResultsToFile(output);

	std::cout << "Press Enter to quit program!" << std::endl;
	std::cin.get();
	std::cin.get();
	
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

void writeResultsToFile(std::vector<std::string>& result)
{
	std::vector<std::string>::iterator itr;
	std::string path = "output.csv";
	std::ofstream ofs;
	ofs.open(path, std::ofstream::out | std::ofstream::app);
	for (itr = result.begin(); itr != result.end(); ++itr)
	{
		ofs << (*itr);
	}
}