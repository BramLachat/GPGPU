#include <cassert>
#include <iostream>
//#include <memory> //needed for smart pointers

#include "Mesh.h"
#include "parse_stl.h"

int main(int argc, char* argv[]) {
	std::string stl_file_inside = "BB_BallBin.stl";
	std::string stl_file_outside = "SphericonBin.stl";

	if (argc == 2) {
		stl_file_inside = argv[1];
	}
	else if (argc > 2) {
		std::cout << "ERROR: Too many command line arguments" << std::endl;
	}

	//Only reads STL-file in binary format!!!
	std::cout << "lezen" << std::endl;
	std::unique_ptr<Mesh> triangleMesh_Cube = stl::parse_stl(stl_file_inside);
	std::unique_ptr<Mesh> triangleMesh_Ball = stl::parse_stl(stl_file_outside);

	std::cout << "STL HEADER = " << triangleMesh_Cube->getName() << std::endl;
	std::cout << "# triangles = " << triangleMesh_Cube->getNumberOfTriangles() << std::endl;

	//triangleMesh_Cube.schrijf();

	std::cout << "STL HEADER = " << triangleMesh_Ball->getName() << std::endl;
	std::cout << "# triangles = " << triangleMesh_Ball->getNumberOfTriangles() << std::endl;

	//triangleMesh_Ball.schrijf();

	float direction[3] = {0.0, 0.0, 100.0};

	//2 opties om unique ptr mee te geven als argument aan een functie:
	//https://stackoverflow.com/questions/30905487/how-can-i-pass-stdunique-ptr-into-a-function
	triangleMesh_Ball->findIntersections(direction, triangleMesh_Cube);
}