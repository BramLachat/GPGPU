#include <cassert>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
//#include <memory> //needed for smart pointers

#include "Mesh.h"
#include "parse_stl.h"
#include "RayTriangleIntersect.cuh"

void writeResultsToFile(std::vector<std::string>& result);

/* Console output wegschrijven naar file*/
std::vector<std::string> output;


int main(int argc, char* argv[]) {
	
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

	int number_of_threads;
	std::cout << "Number of threads?" << std::endl;
	std::cin >> number_of_threads;

	//output.push_back(";RT_CPU_OpenMP(ms);;RT_CPU(ms);TT_CPU(ms)\n");

	Vertex* V1 = triangleMesh_Outside->getVertexAtIndex(0);
	Vertex* V2 = triangleMesh_Outside->getVertexAtIndex(1);
	Vertex* V3 = triangleMesh_Outside->getVertexAtIndex(2);

	float xCenter = (V1->getCoordinates()[0] + V2->getCoordinates()[0] + V3->getCoordinates()[0]) / 3;
	float yCenter = (V1->getCoordinates()[1] + V2->getCoordinates()[1] + V3->getCoordinates()[1]) / 3;
	float zCenter = (V1->getCoordinates()[2] + V2->getCoordinates()[2] + V3->getCoordinates()[2]) / 3;

	float direction[3] = { xCenter, yCenter, zCenter };


	output.push_back(std::to_string((float)(triangleMesh_Outside->rayTriangleIntersectOpenMP(direction, triangleMesh_Inside, number_of_threads))/1000) + ";Number of threads: " + std::to_string(number_of_threads) + ";"); // CPU version

	//2 opties om unique ptr mee te geven als argument aan een functie:
	//https://stackoverflow.com/questions/30905487/how-can-i-pass-stdunique-ptr-into-a-function
	output.push_back(std::to_string((float)(triangleMesh_Outside->rayTriangleIntersect(direction, triangleMesh_Inside))/1000) + ";"); // CPU version

	output.push_back(std::to_string((float)(triangleMesh_Outside->triangleTriangleIntersectOpenMP(triangleMesh_Inside, number_of_threads)) / 1000) + ";Number of threads: " + std::to_string(number_of_threads) + ";"); // CPU version

	output.push_back(std::to_string((float)(triangleMesh_Outside->triangleTriangleIntersect(triangleMesh_Inside))/1000) + "\n");

	writeResultsToFile(output);

	std::cout << "Press Enter to quit program!" << std::endl;
	std::cin.get();
	std::cin.get();
	return 0;
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