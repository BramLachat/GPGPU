#include <cassert>
#include <iostream>
#include <chrono>
//#include <memory> //needed for smart pointers

#include "Mesh.h"
#include "parse_stl.h"

int main(int argc, char* argv[]) {
	std::string stl_file_inside = "Hull_GravBin.stl";
	std::string stl_file_outside = "Hull_Grav_BigBin.stl";
	/*std::cout << "Enter filename of inside mesh:" << std::endl;
	std::cin >> stl_file_inside;
	std::cout << "Enter filename of outside mesh:" << std::endl;
	std::cin >> stl_file_outside;*/

	if (argc == 2) {
		stl_file_inside = argv[1];
	}
	else if (argc > 2) {
		std::cout << "ERROR: Too many command line arguments" << std::endl;
	}

	//Only reads STL-file in binary format!!!
	std::cout << "lezen" << std::endl;
	std::unique_ptr<Mesh> triangleMesh_Inside = stl::parse_stl(stl_file_inside);
	std::unique_ptr<Mesh> triangleMesh_Outside = stl::parse_stl(stl_file_outside);

	std::cout << "STL HEADER = " << triangleMesh_Inside->getName() << std::endl;
	std::cout << "# triangles = " << triangleMesh_Inside->getNumberOfTriangles() << std::endl;

	//triangleMesh_Inside.schrijf();

	std::cout << "STL HEADER = " << triangleMesh_Outside->getName() << std::endl;
	std::cout << "# triangles = " << triangleMesh_Outside->getNumberOfTriangles() << std::endl;

	//triangleMesh_Outside.schrijf();

	float direction[3] = {0.5, 1.0, -0.5};

	auto start = std::chrono::high_resolution_clock::now(); //start time measurement

	//2 opties om unique ptr mee te geven als argument aan een functie:
	//https://stackoverflow.com/questions/30905487/how-can-i-pass-stdunique-ptr-into-a-function
	triangleMesh_Outside->findIntersections(direction, triangleMesh_Inside);

	auto end = std::chrono::high_resolution_clock::now(); //stop time measurement
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "Time = " << duration << "ms" << std::endl;

	std::cout << "Press Enter to quit program!" << std::endl;
	std::cin.get();
	std::cin.get();
	return 0;
}