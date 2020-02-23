// Author(s) : Camille Wormser, Pierre Alliez

#include <iostream>
#include <list>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <streambuf>
#include <memory>
#include "parse_stl.h"
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
	std::cout << "Reading files:" << std::endl;
	std::unique_ptr<Mesh> triangleMesh_Inside = stl::parse_stl(stl_file_inside);
	std::unique_ptr<Mesh> triangleMesh_Outside = stl::parse_stl(stl_file_outside);

	//Only reads STL-file in binary format!!!
	std::cout << "Reading file: triangleMesh_Inside_Point" << std::endl;
	std::list<Point> triangleMesh_Inside_Point = triangleMesh_Inside->getPointList();
	std::cout << "Reading file: triangleMesh_Inside_Triangle" << std::endl;
	std::list<Triangle> triangleMesh_Inside_Triangle = triangleMesh_Inside->getTriangleList();
	std::cout << "Reading file: triangleMesh_Outside" << std::endl;
	std::list<Triangle> triangleMesh_Outside_Triangle = triangleMesh_Outside->getTriangleList();

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

	auto it0 = std::next(triangleMesh_Outside_Triangle.begin(), 0);

	Triangle t = *it0;

	Point V1 = t[0];
	Point V2 = t[1];
	Point V3 = t[2];

	float xCenter = (V1[0] + V2[0] + V3[0]) / 3;
	float yCenter = (V1[1] + V2[1] + V3[1]) / 3;
	float zCenter = (V1[2] + V2[2] + V3[2]) / 3;

	Point direction(xCenter, yCenter, zCenter);

	// constructs AABB tree
	Tree tree(triangleMesh_Outside_Triangle.begin(), triangleMesh_Outside_Triangle.end());

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
	auto time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

	std::string result;
	if (inside) result = "INSIDE";
	else result = "OUTSIDE";
	output.push_back(std::to_string((float)time/1000) + ";" + result + ";");

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
		if (inside) {
			inside = !intersection;
		}
		//counter += numberOfIntersections;
		++it;
	}

	t2 = std::chrono::high_resolution_clock::now(); //stop time measurement
	time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

	if (inside) result = "INSIDE";
	else result = "OUTSIDE";
	output.push_back(std::to_string((float)time/1000) + ";" + result + "\n");

	/*std::cout << "Time = " << time << " milliseconds" << std::endl;

	std::cout << "Number of intersections: " << counter << std::endl;
	std::cout << "Inside: " << inside << std::endl;*/

	writeResultsToFile(output);

	std::cout << "Press Enter to quit program!" << std::endl;
	std::cin.get();
	std::cin.get();
	
    return EXIT_SUCCESS;
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