#include <cassert>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
//#include <memory> //needed for smart pointers

#include "Mesh.h"
#include "parse_stl.h"
#include "RayTriangleIntersect.cuh"

void rayTriangleIntersect(float dir[3], std::unique_ptr<Mesh>& innerMesh, std::unique_ptr<Mesh>& outerMesh);
void handleCudaError(cudaError_t cudaERR);

int main(int argc, char* argv[]) {
	std::string stl_file_inside;
	std::string stl_file_outside;
	int RayTriangle;
	std::cout << "Enter filename of inside mesh:" << std::endl;
	std::cin >> stl_file_inside;
	std::cout << "Enter filename of outside mesh:" << std::endl;
	std::cin >> stl_file_outside;
	std::cout << "0 = RayTriangleIntersection, 1 = TriangleTriangleIntersection" << std::endl;
	std::cin >> RayTriangle;

	if (argc == 2) {
		stl_file_inside = argv[1];
	}
	else if (argc > 2) {
		std::cout << "ERROR: Too many command line arguments" << std::endl;
	}

	auto t1 = std::chrono::high_resolution_clock::now(); //start time measurement

	//Only reads STL-file in binary format!!!
	std::cout << "lezen" << std::endl;
	std::unique_ptr<Mesh> triangleMesh_Inside = stl::parse_stl(stl_file_inside);
	std::unique_ptr<Mesh> triangleMesh_Outside = stl::parse_stl(stl_file_outside);

	auto t2 = std::chrono::high_resolution_clock::now(); //stop time measurement
	auto time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	std::cout << "Time = " << time << "ms" << std::endl;

	std::cout << "STL HEADER = " << triangleMesh_Inside->getName() << std::endl;
	std::cout << "# triangles = " << triangleMesh_Inside->getNumberOfTriangles() << std::endl;
	std::cout << "# vertices = " << triangleMesh_Inside->getNumberOfVertices() << std::endl;

	//triangleMesh_Inside.schrijf();

	std::cout << "STL HEADER = " << triangleMesh_Outside->getName() << std::endl;
	std::cout << "# triangles = " << triangleMesh_Outside->getNumberOfTriangles() << std::endl;
	std::cout << "# vertices = " << triangleMesh_Outside->getNumberOfVertices() << std::endl;

	//triangleMesh_Outside.schrijf();

	Vertex* V1 = triangleMesh_Outside->getVertexAtIndex(0);
	Vertex* V2 = triangleMesh_Outside->getVertexAtIndex(1);
	Vertex* V3 = triangleMesh_Outside->getVertexAtIndex(2);

	float xCenter = (V1->getCoordinates()[0] + V2->getCoordinates()[0] + V3->getCoordinates()[0])/3;
	float yCenter = (V1->getCoordinates()[1] + V2->getCoordinates()[1] + V3->getCoordinates()[1])/3;
	float zCenter = (V1->getCoordinates()[2] + V2->getCoordinates()[2] + V3->getCoordinates()[2])/3;

	float direction[3] = { xCenter, yCenter, zCenter };
	//float direction[3] = { 1.0, 1.0, 1.0 };

	std::cout << "direction = " << direction[0] << ", " << direction[1] << ", " << direction[2] << std::endl;

	//auto start = std::chrono::high_resolution_clock::now(); //start time measurement

	if (RayTriangle == 0)
	{
		auto start = std::chrono::high_resolution_clock::now(); //start time measurement

		//2 opties om unique ptr mee te geven als argument aan een functie:
		//https://stackoverflow.com/questions/30905487/how-can-i-pass-stdunique-ptr-into-a-function
		triangleMesh_Outside->rayTriangleIntersect(direction, triangleMesh_Inside);

		auto end = std::chrono::high_resolution_clock::now(); //stop time measurement
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "Time = " << duration << "ms" << std::endl;

		start = std::chrono::high_resolution_clock::now(); //start time measurement

		rayTriangleIntersect(direction, triangleMesh_Inside, triangleMesh_Outside);

		end = std::chrono::high_resolution_clock::now(); //stop time measurement
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "Time = " << duration << "ms" << std::endl;
	}
	else
	{
		triangleMesh_Outside->triangleTriangleIntersect(triangleMesh_Inside);
	}	
	
	//auto end = std::chrono::high_resolution_clock::now(); //stop time measurement
	//auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	//std::cout << "Time = " << duration << "ms" << std::endl;

	std::cout << "Press Enter to quit program!" << std::endl;
	std::cin.get();
	std::cin.get();
	return 0;
}

void rayTriangleIntersect(float dir[3], std::unique_ptr<Mesh>& innerMesh, std::unique_ptr<Mesh>& outerMesh)
{
	bool inside = true;
	int numberOfTriangles = outerMesh->getNumberOfTriangles();

	//nodig om in kernel te controleren dat aantal keer dat test wordt uitgevoerd <= is dan het aantal driehoeken
	int* numberOfCalculations;
	handleCudaError(cudaMalloc((void**)& numberOfCalculations, sizeof(int)));
	handleCudaError(cudaMemcpy(numberOfCalculations, &numberOfTriangles, sizeof(int), cudaMemcpyHostToDevice));

	float* orig;
	float* cudaOrig;
	
	float* cudaDir;
	handleCudaError(cudaMalloc((void**)& cudaDir, 3*sizeof(float)));
	handleCudaError(cudaMemcpy(cudaDir, dir, 3*sizeof(float), cudaMemcpyHostToDevice));

	int* result = new int[numberOfTriangles];
	int* cudaResult;
	handleCudaError(cudaMalloc((void**)& cudaResult, numberOfTriangles * sizeof(int))); //dit hoeft geen lijst van int's te zijn boolean is voldoende

	int* triangles = outerMesh->getIntArrayTriangles();
	int* cudaTriangles;
	int sizeTriangles = 3 * numberOfTriangles * sizeof(int);
	handleCudaError(cudaMalloc((void**)& cudaTriangles, sizeTriangles));
	handleCudaError(cudaMemcpy(cudaTriangles, triangles, sizeTriangles, cudaMemcpyHostToDevice));

	float* vertices = outerMesh->getFloatArrayVertices();
	float* cudaVertices;
	int sizeVertices = 3 * outerMesh->getNumberOfVertices() * sizeof(float);
	handleCudaError(cudaMalloc((void**)& cudaVertices, sizeVertices));
	handleCudaError(cudaMemcpy(cudaVertices, vertices, sizeVertices, cudaMemcpyHostToDevice));

	int totalIntersections = 0;

	for (int j = 0; j < innerMesh->getNumberOfVertices(); j++)
	{
		orig = (innerMesh->getVertexAtIndex(j))->getCoordinates();
		handleCudaError(cudaMalloc((void**)& cudaOrig, 3 * sizeof(float)));
		handleCudaError(cudaMemcpy(cudaOrig, orig, 3 * sizeof(float), cudaMemcpyHostToDevice));

		int numberOfIntersections = 0;

		int numberOfBlocks = ((int)((numberOfTriangles+255)/256));

		Intersection::intersect_triangle4<<<numberOfBlocks,256>>>(cudaOrig, cudaDir, cudaTriangles, cudaVertices, cudaResult, numberOfCalculations);
		cudaError_t err = cudaGetLastError();
		handleCudaError(err);

		handleCudaError(cudaMemcpy(result, cudaResult, numberOfTriangles * sizeof(int), cudaMemcpyDeviceToHost));

		//std::cout << "result = ";
		for (int i = 0; i < numberOfTriangles; i++)
		{
			if (result[i] == 1) { numberOfIntersections++; }
			//std::cout << result[i] << ", ";
		}
		totalIntersections += numberOfIntersections;
		//std::cout << "numberOfIntersections: " << numberOfIntersections;
		//std::cout << std::endl;
		if (numberOfIntersections % 2 == 0)
		{
			inside = false;
		}
		cudaFree(cudaOrig);
	}
	cudaFree(cudaDir);
	cudaFree(cudaResult);
	cudaFree(cudaTriangles);
	cudaFree(cudaVertices);
	cudaFree(numberOfCalculations);
	delete result;
	std::cout << "totaal intersecties: " << totalIntersections << std::endl;
	if (inside) { std::cout << "INSIDE" << std::endl; }
	else { std::cout << "OUTSIDE" << std::endl; }
}

void handleCudaError(cudaError_t cudaERR) {
	if (cudaERR != cudaSuccess) {
		printf("CUDA ERROR : %s\n", cudaGetErrorString(cudaERR));
	}
}