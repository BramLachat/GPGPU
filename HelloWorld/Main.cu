#include <cassert>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
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
	int numberOfOutsideTriangles = outerMesh->getNumberOfTriangles();
	int numberOfInsideVertices = innerMesh->getNumberOfVertices();

	//nodig om in kernel te controleren dat aantal keer dat test wordt uitgevoerd <= is dan het aantal driehoeken
	int numberOfCudaCalculations = numberOfInsideVertices;
	//handleCudaError(cudaMalloc((void**)& numberOfCudaCalculations, sizeof(int)));
	//handleCudaError(cudaMemcpy(numberOfCudaCalculations, &numberOfInsideVertices, sizeof(int), cudaMemcpyHostToDevice));

	//float* orig;
	//float* cudaOrig;
	float* insideOrigins = innerMesh->getFloatArrayVertices();
	float* cudaInsideOrigins;
	int sizeInsideVertices = 3 * numberOfInsideVertices * sizeof(float);
	handleCudaError(cudaMalloc((void**)& cudaInsideOrigins, sizeInsideVertices));
	handleCudaError(cudaMemcpy(cudaInsideOrigins, insideOrigins, sizeInsideVertices, cudaMemcpyHostToDevice));
	
	float* cudaDir;
	handleCudaError(cudaMalloc((void**)& cudaDir, 3*sizeof(float)));
	handleCudaError(cudaMemcpy(cudaDir, dir, 3*sizeof(float), cudaMemcpyHostToDevice));

	bool* result = new bool[numberOfInsideVertices];
	bool* cudaResult;
	handleCudaError(cudaMalloc((void**)& cudaResult, numberOfInsideVertices * sizeof(bool)));

	int* outsideTriangles = outerMesh->getIntArrayTriangles();
	int* cudaOutsideTriangles;
	int sizeOutsideTriangles = 3 * numberOfOutsideTriangles * sizeof(int);
	handleCudaError(cudaMalloc((void**)& cudaOutsideTriangles, sizeOutsideTriangles));
	handleCudaError(cudaMemcpy(cudaOutsideTriangles, outsideTriangles, sizeOutsideTriangles, cudaMemcpyHostToDevice));

	float* outsideVertices = outerMesh->getFloatArrayVertices();
	float* cudaOutsideVertices;
	int sizeOutsideVertices = 3 * outerMesh->getNumberOfVertices() * sizeof(float);
	handleCudaError(cudaMalloc((void**)& cudaOutsideVertices, sizeOutsideVertices));
	handleCudaError(cudaMemcpy(cudaOutsideVertices, outsideVertices, sizeOutsideVertices, cudaMemcpyHostToDevice));

	thrust::device_vector<int> intersectionsPerThread(numberOfInsideVertices);
	int* d_intersectionsPerThread = thrust::raw_pointer_cast(&intersectionsPerThread[0]);

	thrust::device_vector<float3> resultVertices(numberOfInsideVertices);
	float3* d_resultVertices = thrust::raw_pointer_cast(&resultVertices[0]);
	//float3* d_outsideVertices = thrust::raw_pointer_cast(&outsideVertices[0]); LUKT NIET!!!

	int totalIntersections = 0;
	/*
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
		//cudaFree(cudaOrig);
	}*/
	int numberOfBlocks = ((int)((numberOfInsideVertices + 255) / 256));
	Intersection::intersect_triangleGPU<<<numberOfBlocks,256>>>(cudaInsideOrigins, cudaDir, cudaOutsideTriangles, cudaOutsideVertices, cudaResult, numberOfCudaCalculations, numberOfOutsideTriangles, d_intersectionsPerThread, d_resultVertices);
	cudaError_t err = cudaGetLastError();
	handleCudaError(err);

	handleCudaError(cudaMemcpy(result, cudaResult, numberOfInsideVertices * sizeof(bool), cudaMemcpyDeviceToHost));

	std::vector<int> h_intersectionsPerThread(intersectionsPerThread.size());
	thrust::copy(intersectionsPerThread.begin(), intersectionsPerThread.end(), h_intersectionsPerThread.begin());

	std::vector<float3> h_resultVertices(resultVertices.size());
	thrust::copy(resultVertices.begin(), resultVertices.end(), h_resultVertices.begin());
	std::unique_ptr<std::vector<Vertex>> verticesToWrite = std::make_unique<std::vector<Vertex>>();
	verticesToWrite->reserve(h_resultVertices.size());
	float x, y, z;
	for (int i = 0; i < h_resultVertices.size(); i++) 
	{
		x = h_resultVertices[i].x;
		y = h_resultVertices[i].y;
		z = h_resultVertices[i].z;
		if (x + y + z != 0) { verticesToWrite->emplace_back(x, y, z); }
	}
	innerMesh->writeVerticesToFile(verticesToWrite, "OutsideVerticesCUDA.stl");

	for (int i = 0; i < numberOfInsideVertices; i++)
	{
		if (!result[i])
		{
			inside = false;
		}
		totalIntersections += h_intersectionsPerThread[i];
	}

	cudaFree(cudaInsideOrigins);
	cudaFree(cudaDir);
	cudaFree(cudaResult);
	cudaFree(cudaOutsideTriangles);
	cudaFree(cudaOutsideVertices);
	//cudaFree(numberOfCudaCalculations);
	delete result;
	delete insideOrigins;
	delete outsideTriangles;
	delete outsideVertices;

	std::cout << "totaal intersecties: " << totalIntersections << std::endl;
	if (inside) { std::cout << "INSIDE" << std::endl; }
	else { std::cout << "OUTSIDE" << std::endl; }
}

void handleCudaError(cudaError_t cudaERR) {
	if (cudaERR != cudaSuccess) {
		printf("CUDA ERROR : %s\n", cudaGetErrorString(cudaERR));
	}
}