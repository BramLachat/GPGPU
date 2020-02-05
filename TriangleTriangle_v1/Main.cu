#include <cassert>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
//#include <memory> //needed for smart pointers

#include "Mesh.h"
#include "parse_stl.h"
#include "RayTriangleIntersect.cuh"
#include "TriangleTriangleIntersect.cuh"

void rayTriangleIntersect(float dir[3], std::unique_ptr<Mesh>& innerMesh, std::unique_ptr<Mesh>& outerMesh);
void TriangleTriangleIntersect(std::unique_ptr<Mesh>& innerMesh, std::unique_ptr<Mesh>& outerMesh);
void handleCudaError(cudaError_t cudaERR);
__global__ void startGPU();

int main(int argc, char* argv[]) {
	std::string stl_file_inside;
	std::string stl_file_outside;
	int RayTriangle;
	int CPU;
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
	std::cout << "Reading files:" << std::endl;
	std::unique_ptr<Mesh> triangleMesh_Inside = stl::parse_stl(stl_file_inside);
	std::unique_ptr<Mesh> triangleMesh_Outside = stl::parse_stl(stl_file_outside);

	auto t2 = std::chrono::high_resolution_clock::now(); //stop time measurement
	auto time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	std::cout << "Time = " << time << " milliseconds" << std::endl;

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
		std::cout << "CPU? (yes = 1, no = 0)" << std::endl;
		std::cin >> CPU;
		if (CPU == 1)
		{
			//2 opties om unique ptr mee te geven als argument aan een functie:
			//https://stackoverflow.com/questions/30905487/how-can-i-pass-stdunique-ptr-into-a-function
			triangleMesh_Outside->rayTriangleIntersect(direction, triangleMesh_Inside); // CPU version
		}
		rayTriangleIntersect(direction, triangleMesh_Inside, triangleMesh_Outside); // GPU version
	}
	else
	{
		std::cout << "CPU? (yes = 1, no = 0)" << std::endl;
		std::cin >> CPU;
		if (CPU == 1)
		{
			triangleMesh_Outside->triangleTriangleIntersect(triangleMesh_Inside); // CPU version
		}
		TriangleTriangleIntersect(triangleMesh_Inside, triangleMesh_Outside); // GPU version
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
	auto start = std::chrono::high_resolution_clock::now(); //start time measurement
	startGPU<<<1,1>>>();
	cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now(); //stop time measurement
	auto transferDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "\t\t\tStartup time GPU = " << transferDuration << " milliseconds" << std::endl;

	std::cout << "\t\t\tCalculating intersections! (GPU)" << std::endl;
	std::cout << "--- Data Transfer ---" << std::endl;
	start = std::chrono::high_resolution_clock::now(); //start time measurement

	bool* inside = new bool;
	*inside = true;
	bool* cudaInside;
	handleCudaError(cudaMalloc((void**)& cudaInside, sizeof(bool)));
	handleCudaError(cudaMemcpy(cudaInside, inside, sizeof(bool), cudaMemcpyHostToDevice));

	int numberOfOutsideTriangles = outerMesh->getNumberOfTriangles();
	int numberOfInsideVertices = innerMesh->getNumberOfVertices();

	//nodig om in kernel te controleren dat aantal keer dat test wordt uitgevoerd <= is dan het aantal driehoeken
	//int numberOfCudaCalculations = numberOfInsideVertices * numberOfOutsideTriangles;

	//bool* threadResult;
	//handleCudaError(cudaMalloc((void**)& threadResult, numberOfCudaCalculations*sizeof(bool)));

	/*thrust::host_vector<float3> insideOrigins = innerMesh->getVerticesVector();
	thrust::device_vector<float3> cudaInsideOrigins(insideOrigins.size());
	cudaInsideOrigins = insideOrigins;
	thrust::device_ptr<float3> d_cudaInsideOrigins = cudaInsideOrigins.data();
	float3* deviceInsideOrigins = thrust::raw_pointer_cast(d_cudaInsideOrigins);*/

	float3* insideOrigins = innerMesh->getFloat3ArrayVertices();
	float3* cudaInsideOrigins;
	int sizeInsideVertices = numberOfInsideVertices * sizeof(float3);
	//handleCudaError(cudaMallocManaged((void**)& cudaInsideOrigins, sizeInsideVertices));
	//memcpy(cudaInsideOrigins, insideOrigins, sizeInsideVertices);  waarschijnlijk moet ik zoiets ook nog doen bij cudaHostAlloc???
	handleCudaError(cudaMalloc((void**)& cudaInsideOrigins, sizeInsideVertices));
	handleCudaError(cudaMemcpy(cudaInsideOrigins, insideOrigins, sizeInsideVertices, cudaMemcpyHostToDevice));
	
	float* cudaDir;
	//handleCudaError(cudaMallocManaged((void**)& cudaDir, 3*sizeof(float)));
	//memcpy(cudaDir, dir, 3 * sizeof(float));
	handleCudaError(cudaMalloc((void**)& cudaDir, 3 * sizeof(float)));
	handleCudaError(cudaMemcpy(cudaDir, dir, 3*sizeof(float), cudaMemcpyHostToDevice));

	/*thrust::host_vector<int3> outsideTriangles = outerMesh->getTrianglesVector();;
	thrust::device_vector<int3> cudaOutsideTriangles(outsideTriangles.size());
	cudaOutsideTriangles = outsideTriangles;
	thrust::device_ptr<int3> d_cudaOutsideTriangles = cudaOutsideTriangles.data();
	int3* deviceOutsideTriangles = thrust::raw_pointer_cast(d_cudaOutsideTriangles);*/

	int3* outsideTriangles = outerMesh->getInt3ArrayTriangles();
	int3* cudaOutsideTriangles;
	int sizeOutsideTriangles = numberOfOutsideTriangles * sizeof(int3);
	//handleCudaError(cudaMallocManaged((void**)& cudaOutsideTriangles, sizeOutsideTriangles));
	//memcpy(cudaOutsideTriangles, outsideTriangles, sizeOutsideTriangles);
	handleCudaError(cudaMalloc((void**)& cudaOutsideTriangles, sizeOutsideTriangles));
	handleCudaError(cudaMemcpy(cudaOutsideTriangles, outsideTriangles, sizeOutsideTriangles, cudaMemcpyHostToDevice));

	/*thrust::host_vector<float3> outsideVertices = outerMesh->getVerticesVector();
	thrust::device_vector<float3> cudaOutsideVertices(outsideVertices.size());
	cudaOutsideVertices = outsideVertices;
	thrust::device_ptr<float3> d_cudaOutsideVertices = cudaOutsideVertices.data();
	float3* deviceOutsideVertices = thrust::raw_pointer_cast(d_cudaOutsideVertices);*/

	float3* outsideVertices = outerMesh->getFloat3ArrayVertices();
	float3* cudaOutsideVertices;
	int sizeOutsideVertices = outerMesh->getNumberOfVertices() * sizeof(float3);
	//handleCudaError(cudaMallocManaged((void**)& cudaOutsideVertices, sizeOutsideVertices));
	//memcpy(cudaOutsideVertices, outsideVertices, sizeOutsideVertices);
	handleCudaError(cudaMalloc((void**)& cudaOutsideVertices, sizeOutsideVertices));
	handleCudaError(cudaMemcpy(cudaOutsideVertices, outsideVertices, sizeOutsideVertices, cudaMemcpyHostToDevice));
	
	//thrust::device_vector<int> intersectionsPerOrigin(numberOfInsideVertices);
	//int* d_intersectionsPerOrigin = thrust::raw_pointer_cast(&intersectionsPerOrigin[0]);
	/*int* intersectionsPerOrigin = new int[numberOfInsideVertices];
	int* cudaIntersectionsPerOrigin;
	handleCudaError(cudaMalloc((void**)& cudaIntersectionsPerOrigin, numberOfInsideVertices*sizeof(int)));*/

	//thrust::device_vector<float3> resultVertices(numberOfInsideVertices);
	//float3* d_resultVertices = thrust::raw_pointer_cast(&resultVertices[0]);
	/*float3* resultVertices = new float3[numberOfInsideVertices];
	float3* cudaResultVertices;
	handleCudaError(cudaMalloc((void**)& cudaResultVertices, numberOfInsideVertices * sizeof(float3)));*/

	int totalIntersections = 0;

	std::cout << "--- End Data Transfer ---" << std::endl;
	end = std::chrono::high_resolution_clock::now(); //stop time measurement
	transferDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "\t\t\tTime Data Transfer = " << transferDuration << " milliseconds" << std::endl;

	std::cout << "--- Calculating ---" << std::endl;
	start = std::chrono::high_resolution_clock::now(); //start time measurement

	int numberOfBlocks = ((int)((numberOfInsideVertices + 511) / 512));
	Intersection::intersect_triangleGPU<<<numberOfBlocks,512>>>(cudaInsideOrigins, cudaDir, cudaOutsideTriangles, cudaOutsideVertices, numberOfInsideVertices, numberOfOutsideTriangles, cudaInside);
	cudaError_t err = cudaGetLastError();
	handleCudaError(err);

	handleCudaError(cudaMemcpy(inside, cudaInside, sizeof(bool), cudaMemcpyDeviceToHost));

	//std::vector<int> h_intersectionsPerOrigin(intersectionsPerOrigin.size());
	//hrust::copy(intersectionsPerOrigin.begin(), intersectionsPerOrigin.end(), h_intersectionsPerOrigin.begin());
	//handleCudaError(cudaMemcpy(intersectionsPerOrigin, cudaIntersectionsPerOrigin, numberOfInsideVertices * sizeof(int), cudaMemcpyDeviceToHost));

	//std::vector<float3> h_resultVertices(resultVertices.size());
	//thrust::copy(resultVertices.begin(), resultVertices.end(), h_resultVertices.begin());
	//handleCudaError(cudaMemcpy(resultVertices, cudaResultVertices, numberOfInsideVertices * sizeof(float3), cudaMemcpyDeviceToHost));

	std::cout << "--- End Calculating ---" << std::endl;
	end = std::chrono::high_resolution_clock::now(); //stop time measurement
	auto calculatingDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "\t\t\tTime Calculating = " << calculatingDuration << " microseconds" << std::endl;
	//std::cout << "\t\t\tTotal Time GPU = " << calculatingDuration + transferDuration << "ms" << std::endl;

	/*std::cout << "Writing to file!" << std::endl;

	std::unique_ptr<std::vector<Vertex>> verticesToWrite = std::make_unique<std::vector<Vertex>>();
	verticesToWrite->reserve(numberOfInsideVertices);
	float x, y, z;
	for (int i = 0; i < numberOfInsideVertices; i++)
	{
		totalIntersections += intersectionsPerOrigin[i];
		x = resultVertices[i].x;
		y = resultVertices[i].y;
		z = resultVertices[i].z;
		if (x + y + z != 0)
		{
			verticesToWrite->emplace_back(x, y, z);
			inside = false;
		}
	}

	innerMesh->writeVerticesToFile(verticesToWrite, "OutsideVerticesCUDA.stl");*/

	cudaFree(cudaInsideOrigins);
	cudaFree(cudaDir);
	cudaFree(cudaOutsideTriangles);
	cudaFree(cudaOutsideVertices);
	cudaFree(cudaInside);
	/*cudaFree(cudaIntersectionsPerOrigin);
	cudaFree(cudaResultVertices);*/
	cudaFreeHost(insideOrigins);
	cudaFreeHost(outsideTriangles);
	cudaFreeHost(outsideVertices);
	//delete insideOrigins;
	//delete outsideTriangles;
	//delete outsideVertices;
	/*delete intersectionsPerOrigin;
	delete resultVertices;*/

	std::cout << "totaal intersecties: " << totalIntersections << std::endl;
	if (*inside) { std::cout << "INSIDE" << std::endl; }
	else { std::cout << "OUTSIDE" << std::endl; }
	
	delete inside;
}

void TriangleTriangleIntersect(std::unique_ptr<Mesh>& innerMesh, std::unique_ptr<Mesh>& outerMesh)
{
	auto start = std::chrono::high_resolution_clock::now(); //start time measurement
	startGPU<<<1,1>>>();
	cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now(); //stop time measurement
	auto transferDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "\t\t\tStartup time GPU = " << transferDuration << " milliseconds" << std::endl;

	std::cout << "\t\t\tCalculating intersections! (GPU)" << std::endl;
	std::cout << "--- Data Transfer ---" << std::endl;
	start = std::chrono::high_resolution_clock::now(); //start time measurement

	bool* inside = new bool;
	*inside = true;
	bool* cudaInside;
	handleCudaError(cudaMalloc((void**)&cudaInside, sizeof(bool)));
	handleCudaError(cudaMemcpy(cudaInside, inside, sizeof(bool), cudaMemcpyHostToDevice));

	int numberOfOutsideTriangles = outerMesh->getNumberOfTriangles();
	int numberOfOutsideVertices = outerMesh->getNumberOfVertices();
	int numberOfInsideTriangles = innerMesh->getNumberOfTriangles();
	int numberOfInsideVertices = innerMesh->getNumberOfVertices();

	/* Alloceren en kopiëren hoekpunten binnenste mesh naar GPU*/
	float3* insideVertices = innerMesh->getFloat3ArrayVertices();
	float3* cudaInsideVertices;
	int sizeInsideVertices = numberOfInsideVertices * sizeof(float3);
	handleCudaError(cudaMalloc((void**)&cudaInsideVertices, sizeInsideVertices));
	handleCudaError(cudaMemcpyAsync(cudaInsideVertices, insideVertices, sizeInsideVertices, cudaMemcpyHostToDevice));

	/* Alloceren en kopiëren driehoeken binnenste mesh naar GPU*/
	int3* insideTriangles = innerMesh->getInt3ArrayTriangles();
	int3* cudaInsideTriangles;
	int sizeInsideTriangles = numberOfInsideTriangles * sizeof(int3);
	handleCudaError(cudaMalloc((void**)&cudaInsideTriangles, sizeInsideTriangles));
	handleCudaError(cudaMemcpyAsync(cudaInsideTriangles, insideTriangles, sizeInsideTriangles, cudaMemcpyHostToDevice));

	/* Alloceren en kopiëren hoekpunten buitenste mesh naar GPU*/
	float3* outsideVertices = outerMesh->getFloat3ArrayVertices();
	float3* cudaOutsideVertices;
	int sizeOutsideVertices = numberOfOutsideVertices * sizeof(float3);
	handleCudaError(cudaMalloc((void**)&cudaOutsideVertices, sizeOutsideVertices));
	handleCudaError(cudaMemcpyAsync(cudaOutsideVertices, outsideVertices, sizeOutsideVertices, cudaMemcpyHostToDevice));
	
	/* Alloceren en kopiëren driehoeken buitenste mesh naar GPU*/
	int3* outsideTriangles = outerMesh->getInt3ArrayTriangles();
	int3* cudaOutsideTriangles;
	int sizeOutsideTriangles = numberOfOutsideTriangles * sizeof(int3);
	handleCudaError(cudaMalloc((void**)&cudaOutsideTriangles, sizeOutsideTriangles));
	handleCudaError(cudaMemcpyAsync(cudaOutsideTriangles, outsideTriangles, sizeOutsideTriangles, cudaMemcpyHostToDevice));

	/* Alloceren van geheugen op GPU om bij te houden met hoeveel driehoeken van de buitenste mesh deze ene driehoek van de binnenste mesh snijdt*/
	//int* intersectionsPerInsideTriangle = new int[numberOfInsideTriangles];
	//int* cudaIntersectionsPerInsideTriangle;
	//handleCudaError(cudaMalloc((void**)&cudaIntersectionsPerInsideTriangle, numberOfInsideTriangles * sizeof(int)));

	/* Als deze waarde > 0 ==> De binnenste mesh ligt niet volledig in de buitenste mesh*/
	int totalIntersections = 0;

	std::cout << "--- End Data Transfer ---" << std::endl;
	end = std::chrono::high_resolution_clock::now(); //stop time measurement
	transferDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "\t\t\tTime Data Transfer = " << transferDuration << " milliseconds" << std::endl;

	std::cout << "--- Calculating ---" << std::endl;
	start = std::chrono::high_resolution_clock::now(); //start time measurement

	/* Uitvoeren CUDA kernel*/
	int numberOfBlocks = ((int)((numberOfInsideTriangles + 511) / 512));
	triangle_triangle_GPU<<<numberOfBlocks,512>>>(cudaInsideTriangles, cudaInsideVertices, cudaOutsideTriangles, cudaOutsideVertices, cudaInside, numberOfInsideTriangles, numberOfOutsideTriangles);
	cudaError_t err = cudaGetLastError();
	handleCudaError(err);

	/* Kopiëren van de resultaten van GPU naar CPU*/
	//handleCudaError(cudaMemcpy(intersectionsPerInsideTriangle, cudaIntersectionsPerInsideTriangle, numberOfInsideTriangles * sizeof(int), cudaMemcpyDeviceToHost));
	handleCudaError(cudaMemcpy(inside, cudaInside, sizeof(bool), cudaMemcpyDeviceToHost));

	std::cout << "--- End Calculating ---" << std::endl;
	end = std::chrono::high_resolution_clock::now(); //stop time measurement
	auto calculatingDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "\t\t\tTime Calculating = " << calculatingDuration << " microseconds" << std::endl;

	/*for (int i = 0; i < numberOfInsideTriangles; i++)
	{
		totalIntersections += intersectionsPerInsideTriangle[i];
	}
	if (totalIntersections > 0) { inside = false; }*/

	cudaFree(cudaInsideTriangles);
	cudaFree(cudaInsideVertices);
	cudaFree(cudaOutsideTriangles);
	cudaFree(cudaOutsideVertices);
	cudaFree(cudaInside);
	//cudaFree(cudaIntersectionsPerInsideTriangle);
	cudaFreeHost(outsideTriangles);
	cudaFreeHost(outsideVertices);
	cudaFreeHost(insideTriangles);
	cudaFreeHost(insideVertices);
	//delete intersectionsPerInsideTriangle;

	std::cout << "totaal intersecties: " << totalIntersections << std::endl;
	if (*inside) { std::cout << "SNIJDEN NIET" << std::endl; }
	else { std::cout << "SNIJDEN WEL" << std::endl; }

	delete inside;
}

void handleCudaError(cudaError_t cudaERR) {
	if (cudaERR != cudaSuccess) {
		printf("CUDA ERROR : %s\n", cudaGetErrorString(cudaERR));
	}
}

__global__ void startGPU() {
	printf("GPU ready!\n");
}