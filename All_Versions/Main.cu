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
#include "TriangleTriangleIntersect.cuh"

void rayTriangle_BlockPerOrigin(float dir[3], std::unique_ptr<Mesh>& innerMesh, std::unique_ptr<Mesh>& outerMesh);
void rayTriangle_ThreadPerOrigin(float dir[3], std::unique_ptr<Mesh>& innerMesh, std::unique_ptr<Mesh>& outerMesh);
void rayTriangle_ThreadPerTriangle(float dir[3], std::unique_ptr<Mesh>& innerMesh, std::unique_ptr<Mesh>& outerMesh);

void TriangleTriangle_ThreadPerInnerTriangle(std::unique_ptr<Mesh>& innerMesh, std::unique_ptr<Mesh>& outerMesh);
void TriangleTriangle_BlockPerInnerTriangle(std::unique_ptr<Mesh>& innerMesh, std::unique_ptr<Mesh>& outerMesh);
void TriangleTriangle_ThreadPerOuterTriangle(std::unique_ptr<Mesh>& innerMesh, std::unique_ptr<Mesh>& outerMesh);

void handleCudaError(cudaError_t cudaERR);
__global__ void startGPU();
void writeResultsToFile(std::vector<std::string>& result);

/* Console output wegschrijven naar file*/
std::vector<std::string> output;

int main(int argc, char* argv[]) {
	//output.push_back(";RT_v1(BPO);;RT_v2(TPO);;RT_v3(TPT);;TT_v1(TPIT);;TT_v2(BPIT);;TT_v3(TPOT);\n");

	std::string stl_file_inside;
	std::string stl_file_outside;
	std::cout << "Enter filename of inside mesh:" << std::endl;
	std::cin >> stl_file_inside;

	std::string delimiter = ".stl";
	std::string token = stl_file_inside.substr(28, stl_file_inside.find(delimiter));
	token = token.substr(0, token.find(delimiter));
	output.push_back(token + "-");

	std::cout << "Enter filename of outside mesh:" << std::endl;
	std::cin >> stl_file_outside;

	token = stl_file_outside.substr(28, stl_file_outside.find(delimiter));
	token = token.substr(0, token.find(delimiter));
	output.push_back(token + ";");

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

	std::cout << "STL HEADER = " << triangleMesh_Outside->getName() << std::endl;
	std::cout << "# triangles = " << triangleMesh_Outside->getNumberOfTriangles() << std::endl;
	std::cout << "# vertices = " << triangleMesh_Outside->getNumberOfVertices() << std::endl;

	Vertex* V1 = triangleMesh_Outside->getVertexAtIndex(0);
	Vertex* V2 = triangleMesh_Outside->getVertexAtIndex(1);
	Vertex* V3 = triangleMesh_Outside->getVertexAtIndex(2);

	float xCenter = (V1->getCoordinates()[0] + V2->getCoordinates()[0] + V3->getCoordinates()[0])/3;
	float yCenter = (V1->getCoordinates()[1] + V2->getCoordinates()[1] + V3->getCoordinates()[1])/3;
	float zCenter = (V1->getCoordinates()[2] + V2->getCoordinates()[2] + V3->getCoordinates()[2])/3;

	float direction[3] = { xCenter, yCenter, zCenter };

	std::cout << "direction = " << direction[0] << ", " << direction[1] << ", " << direction[2] << std::endl;

	//triangleMesh_Outside->rayTriangleIntersectOpenMP(direction, triangleMesh_Inside); // CPU version
	//triangleMesh_Outside->rayTriangleIntersect(direction, triangleMesh_Inside); // CPU version

	rayTriangle_BlockPerOrigin(direction, triangleMesh_Inside, triangleMesh_Outside);
	rayTriangle_ThreadPerOrigin(direction, triangleMesh_Inside, triangleMesh_Outside);
	rayTriangle_ThreadPerTriangle(direction, triangleMesh_Inside, triangleMesh_Outside);

	//triangleMesh_Outside->triangleTriangleIntersect(triangleMesh_Inside);

	TriangleTriangle_ThreadPerInnerTriangle(triangleMesh_Inside, triangleMesh_Outside); // GPU version
	TriangleTriangle_BlockPerInnerTriangle(triangleMesh_Inside, triangleMesh_Outside); // GPU version
	TriangleTriangle_ThreadPerOuterTriangle(triangleMesh_Inside, triangleMesh_Outside); // GPU version

	writeResultsToFile(output);

	std::cout << "Press Enter to quit program!" << std::endl;
	std::cin.get();
	std::cin.get();
	return 0;
}

void rayTriangle_BlockPerOrigin(float dir[3], std::unique_ptr<Mesh>& innerMesh, std::unique_ptr<Mesh>& outerMesh)
{
	auto start = std::chrono::high_resolution_clock::now(); //start time measurement
	startGPU<<<1,1>>>();
	cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now(); //stop time measurement
	auto transferDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "\t\t\tStartup time GPU = " << transferDuration << "ms" << std::endl;

	std::cout << "\t\t\tCalculating intersections! (GPU)" << std::endl;
	std::cout << "--- Data Transfer ---" << std::endl;
	start = std::chrono::high_resolution_clock::now(); //start time measurement

	bool* inside = new bool;
	*inside = true;
	bool* cudaInside;
	handleCudaError(cudaMalloc((void**)&cudaInside, sizeof(bool)));
	handleCudaError(cudaMemcpy(cudaInside, inside, sizeof(bool), cudaMemcpyHostToDevice));

	int numberOfOutsideTriangles = outerMesh->getNumberOfTriangles();
	int numberOfInsideVertices = innerMesh->getNumberOfVertices();

	float3* insideOrigins = innerMesh->getFloat3ArrayVertices();
	float3* cudaInsideOrigins;
	int sizeInsideVertices = numberOfInsideVertices * sizeof(float3);
	handleCudaError(cudaMalloc((void**)&cudaInsideOrigins, sizeInsideVertices));
	handleCudaError(cudaMemcpyAsync(cudaInsideOrigins, insideOrigins, sizeInsideVertices, cudaMemcpyHostToDevice));
	
	float* cudaDir;
	handleCudaError(cudaMalloc((void**)&cudaDir, 3 * sizeof(float)));
	handleCudaError(cudaMemcpy(cudaDir, dir, 3 * sizeof(float), cudaMemcpyHostToDevice));

	int3* outsideTriangles = outerMesh->getInt3ArrayTriangles();
	int3* cudaOutsideTriangles;
	int sizeOutsideTriangles = numberOfOutsideTriangles * sizeof(int3);
	handleCudaError(cudaMalloc((void**)&cudaOutsideTriangles, sizeOutsideTriangles));
	handleCudaError(cudaMemcpyAsync(cudaOutsideTriangles, outsideTriangles, sizeOutsideTriangles, cudaMemcpyHostToDevice));

	float3* outsideVertices = outerMesh->getFloat3ArrayVertices();
	float3* cudaOutsideVertices;
	int sizeOutsideVertices = outerMesh->getNumberOfVertices() * sizeof(float3);
	handleCudaError(cudaMalloc((void**)&cudaOutsideVertices, sizeOutsideVertices));
	handleCudaError(cudaMemcpyAsync(cudaOutsideVertices, outsideVertices, sizeOutsideVertices, cudaMemcpyHostToDevice));

	int totalIntersections = 0;

	std::cout << "--- End Data Transfer ---" << std::endl;
	end = std::chrono::high_resolution_clock::now(); //stop time measurement
	transferDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "\t\t\tTime Data Transfer = " << transferDuration << "ms" << std::endl;

	std::cout << "--- Calculating ---" << std::endl;
	start = std::chrono::high_resolution_clock::now(); //start time measurement

	int numberOfBlocks = numberOfInsideVertices;
	intersect_triangleGPU_BlockPerOrigin<<<numberOfBlocks,128>>>(cudaInsideOrigins, cudaDir, cudaOutsideTriangles, cudaOutsideVertices, numberOfOutsideTriangles, cudaInside);
	cudaError_t err = cudaGetLastError();
	handleCudaError(err);

	handleCudaError(cudaMemcpy(inside, cudaInside, sizeof(bool), cudaMemcpyDeviceToHost));

	std::cout << "--- End Calculating ---" << std::endl;
	end = std::chrono::high_resolution_clock::now(); //stop time measurement
	auto calculatingDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "\t\t\tTime Calculating = " << calculatingDuration << " microseconds" << std::endl;
	
	cudaFree(cudaInsideOrigins);
	cudaFree(cudaDir);
	cudaFree(cudaInside);
	cudaFree(cudaOutsideTriangles);
	cudaFree(cudaOutsideVertices);
	
	cudaFreeHost(insideOrigins);
	cudaFreeHost(outsideTriangles);
	cudaFreeHost(outsideVertices);

	std::string result;
	std::cout << "totaal intersecties: " << totalIntersections << std::endl;
	if (*inside) { result = "INSIDE"; }
	else { result = "OUTSIDE"; }
	std::cout << result << std::endl;
	output.push_back(std::to_string(calculatingDuration) + ";" + result + ";");

	delete inside;
}

void rayTriangle_ThreadPerOrigin(float dir[3], std::unique_ptr<Mesh>& innerMesh, std::unique_ptr<Mesh>& outerMesh)
{
	auto start = std::chrono::high_resolution_clock::now(); //start time measurement
	startGPU<<<1,1>>>();
	cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now(); //stop time measurement
	auto transferDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "\t\t\tStartup time GPU = " << transferDuration << "ms" << std::endl;

	std::cout << "\t\t\tCalculating intersections! (GPU)" << std::endl;
	std::cout << "--- Data Transfer ---" << std::endl;
	start = std::chrono::high_resolution_clock::now(); //start time measurement

	bool* inside = new bool;
	*inside = true;
	bool* cudaInside;
	handleCudaError(cudaMalloc((void**)&cudaInside, sizeof(bool)));
	handleCudaError(cudaMemcpy(cudaInside, inside, sizeof(bool), cudaMemcpyHostToDevice));

	int numberOfOutsideTriangles = outerMesh->getNumberOfTriangles();
	int numberOfInsideVertices = innerMesh->getNumberOfVertices();

	float3* insideOrigins = innerMesh->getFloat3ArrayVertices();
	float3* cudaInsideOrigins;
	int sizeInsideVertices = numberOfInsideVertices * sizeof(float3);
	handleCudaError(cudaMalloc((void**)&cudaInsideOrigins, sizeInsideVertices));
	handleCudaError(cudaMemcpyAsync(cudaInsideOrigins, insideOrigins, sizeInsideVertices, cudaMemcpyHostToDevice));

	float* cudaDir;
	handleCudaError(cudaMalloc((void**)&cudaDir, 3 * sizeof(float)));
	handleCudaError(cudaMemcpy(cudaDir, dir, 3 * sizeof(float), cudaMemcpyHostToDevice));

	int3* outsideTriangles = outerMesh->getInt3ArrayTriangles();
	int3* cudaOutsideTriangles;
	int sizeOutsideTriangles = numberOfOutsideTriangles * sizeof(int3);
	handleCudaError(cudaMalloc((void**)&cudaOutsideTriangles, sizeOutsideTriangles));
	handleCudaError(cudaMemcpyAsync(cudaOutsideTriangles, outsideTriangles, sizeOutsideTriangles, cudaMemcpyHostToDevice));

	float3* outsideVertices = outerMesh->getFloat3ArrayVertices();
	float3* cudaOutsideVertices;
	int sizeOutsideVertices = outerMesh->getNumberOfVertices() * sizeof(float3);
	handleCudaError(cudaMalloc((void**)&cudaOutsideVertices, sizeOutsideVertices));
	handleCudaError(cudaMemcpyAsync(cudaOutsideVertices, outsideVertices, sizeOutsideVertices, cudaMemcpyHostToDevice));

	int totalIntersections = 0;

	std::cout << "--- End Data Transfer ---" << std::endl;
	end = std::chrono::high_resolution_clock::now(); //stop time measurement
	transferDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "\t\t\tTime Data Transfer = " << transferDuration << "ms" << std::endl;

	std::cout << "--- Calculating ---" << std::endl;
	start = std::chrono::high_resolution_clock::now(); //start time measurement

	int numberOfBlocks = ((int)((numberOfInsideVertices + 511) / 512));
	intersect_triangleGPU_ThreadPerOrigin<<<numberOfBlocks,512>>>(cudaInsideOrigins, cudaDir, cudaOutsideTriangles, cudaOutsideVertices, numberOfInsideVertices, numberOfOutsideTriangles, cudaInside);
	cudaError_t err = cudaGetLastError();
	handleCudaError(err);

	handleCudaError(cudaMemcpy(inside, cudaInside, sizeof(bool), cudaMemcpyDeviceToHost));

	std::cout << "--- End Calculating ---" << std::endl;
	end = std::chrono::high_resolution_clock::now(); //stop time measurement
	auto calculatingDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "\t\t\tTime Calculating = " << calculatingDuration << " microseconds" << std::endl;
	
	cudaFree(cudaInsideOrigins);
	cudaFree(cudaDir);
	cudaFree(cudaInside);
	cudaFree(cudaOutsideTriangles);
	cudaFree(cudaOutsideVertices);
	
	cudaFreeHost(insideOrigins);
	cudaFreeHost(outsideTriangles);
	cudaFreeHost(outsideVertices);
	
	std::string result;
	std::cout << "totaal intersecties: " << totalIntersections << std::endl;
	if (*inside) { result = "INSIDE"; }
	else { result = "OUTSIDE"; }
	std::cout << result << std::endl;
	output.push_back(std::to_string(calculatingDuration) + ";" + result + ";");

	delete inside;
}

void rayTriangle_ThreadPerTriangle(float dir[3], std::unique_ptr<Mesh>& innerMesh, std::unique_ptr<Mesh>& outerMesh)
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

	bool inside = true;
	int numberOfOutsideTriangles = outerMesh->getNumberOfTriangles();
	int numberOfInsideVertices = innerMesh->getNumberOfVertices();

	float3* insideOrigins = innerMesh->getFloat3ArrayVertices();
	float3* cudaInsideOrigins;
	int sizeInsideVertices = numberOfInsideVertices * sizeof(float3);
	handleCudaError(cudaMalloc((void**)&cudaInsideOrigins, sizeInsideVertices));
	handleCudaError(cudaMemcpyAsync(cudaInsideOrigins, insideOrigins, sizeInsideVertices, cudaMemcpyHostToDevice));

	float* cudaDir;
	handleCudaError(cudaMalloc((void**)&cudaDir, 3 * sizeof(float)));
	handleCudaError(cudaMemcpy(cudaDir, dir, 3 * sizeof(float), cudaMemcpyHostToDevice));

	int3* outsideTriangles = outerMesh->getInt3ArrayTriangles();
	int3* cudaOutsideTriangles;
	int sizeOutsideTriangles = numberOfOutsideTriangles * sizeof(int3);
	handleCudaError(cudaMalloc((void**)&cudaOutsideTriangles, sizeOutsideTriangles));
	handleCudaError(cudaMemcpyAsync(cudaOutsideTriangles, outsideTriangles, sizeOutsideTriangles, cudaMemcpyHostToDevice));

	float3* outsideVertices = outerMesh->getFloat3ArrayVertices();
	float3* cudaOutsideVertices;
	int sizeOutsideVertices = outerMesh->getNumberOfVertices() * sizeof(float3);
	handleCudaError(cudaMalloc((void**)&cudaOutsideVertices, sizeOutsideVertices));
	handleCudaError(cudaMemcpyAsync(cudaOutsideVertices, outsideVertices, sizeOutsideVertices, cudaMemcpyHostToDevice));
	int* intersectionsPerOrigin = new int[numberOfInsideVertices];
	int* cudaIntersectionsPerOrigin;
	handleCudaError(cudaMalloc((void**)&cudaIntersectionsPerOrigin, numberOfInsideVertices * sizeof(int)));

	int totalIntersections = 0;

	std::cout << "--- End Data Transfer ---" << std::endl;
	end = std::chrono::high_resolution_clock::now(); //stop time measurement
	transferDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "\t\t\tTime Data Transfer = " << transferDuration << " milliseconds" << std::endl;

	std::cout << "--- Calculating ---" << std::endl;
	start = std::chrono::high_resolution_clock::now(); //start time measurement

	int numberOfBlocks = ((int)((numberOfOutsideTriangles + 127) / 128));
	intersect_triangleGPU_ThreadPerTriangle<<<numberOfBlocks,128>>>(cudaInsideOrigins, cudaDir, cudaOutsideTriangles, cudaOutsideVertices, numberOfInsideVertices, numberOfOutsideTriangles, cudaIntersectionsPerOrigin);
	cudaError_t err = cudaGetLastError();
	handleCudaError(err);

	handleCudaError(cudaMemcpy(intersectionsPerOrigin, cudaIntersectionsPerOrigin, numberOfInsideVertices * sizeof(int), cudaMemcpyDeviceToHost));

	int i = 0;
	while (i < numberOfInsideVertices && inside)
	{
		if (intersectionsPerOrigin[i] % 2 == 0) {
			inside = false;
		}
		i++;
	}

	std::cout << "--- End Calculating ---" << std::endl;
	end = std::chrono::high_resolution_clock::now(); //stop time measurement
	auto calculatingDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "\t\t\tTime Calculating = " << calculatingDuration << " microseconds" << std::endl;

	for (int i = 0; i < numberOfInsideVertices; i++)
	{
		totalIntersections += intersectionsPerOrigin[i];
	}

	cudaFree(cudaInsideOrigins);
	cudaFree(cudaDir);
	cudaFree(cudaOutsideTriangles);
	cudaFree(cudaOutsideVertices);
	cudaFree(cudaIntersectionsPerOrigin);
	
	cudaFreeHost(insideOrigins);
	cudaFreeHost(outsideTriangles);
	cudaFreeHost(outsideVertices);

	delete intersectionsPerOrigin;

	std::string result;
	std::cout << "totaal intersecties: " << totalIntersections << std::endl;
	if (inside) { result = "INSIDE"; }
	else { result = "OUTSIDE"; }
	std::cout << result << std::endl;
	output.push_back(std::to_string(calculatingDuration) + ";" + result + ";");
}

void TriangleTriangle_ThreadPerInnerTriangle(std::unique_ptr<Mesh>& innerMesh, std::unique_ptr<Mesh>& outerMesh)
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

	//TODO: extra lijst met outside mesh driehoekintervallen meegeven (ongesorteerd, omdat ik op dit moment geen idee heb waarom die gesorteerd zou moeten zijn)
	float2* outsideTriangleIntervals = outerMesh->getTriangleInterval();
	float2* cudaOutsideTriangleIntervals;
	int sizeOutsideIntervals = numberOfOutsideTriangles * sizeof(float2);
	handleCudaError(cudaMalloc((void**)&cudaOutsideTriangleIntervals, sizeOutsideIntervals));
	handleCudaError(cudaMemcpyAsync(cudaOutsideTriangleIntervals, outsideTriangleIntervals, sizeOutsideIntervals, cudaMemcpyHostToDevice));

	/* Als deze waarde > 0 ==> De binnenste mesh ligt niet volledig in de buitenste mesh*/
	int totalIntersections = 0;

	std::cout << "--- End Data Transfer ---" << std::endl;
	end = std::chrono::high_resolution_clock::now(); //stop time measurement
	transferDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "\t\t\tTime Data Transfer = " << transferDuration << " milliseconds" << std::endl;


	/****************************************************************************
	Uitvoeren CUDA kernel - Triangle Triangle met Broad Phase Collision Detection
	*****************************************************************************/
	std::cout << "--- Calculating ---" << std::endl;
	start = std::chrono::high_resolution_clock::now(); //start time measurement

	/* Uitvoeren CUDA kernel*/
	int numberOfBlocks = ((int)((numberOfInsideTriangles + 511) / 512));

	triangle_triangle_GPU_BPCD_ThreadPerInnerTriangle<<<numberOfBlocks,512>>>(cudaInsideTriangles, cudaInsideVertices, cudaOutsideTriangles, cudaOutsideVertices, cudaInside, numberOfInsideTriangles, numberOfOutsideTriangles, cudaOutsideTriangleIntervals);
	cudaError_t err = cudaGetLastError();
	handleCudaError(err);

	/* Kopiëren van de resultaten van GPU naar CPU*/
	//handleCudaError(cudaMemcpy(intersectionsPerInsideTriangle, cudaIntersectionsPerInsideTriangle, numberOfInsideTriangles * sizeof(int), cudaMemcpyDeviceToHost));
	handleCudaError(cudaMemcpy(inside, cudaInside, sizeof(bool), cudaMemcpyDeviceToHost));

	std::cout << "--- End Calculating ---" << std::endl;
	end = std::chrono::high_resolution_clock::now(); //stop time measurement
	auto calculatingDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "\t\t\tTime Calculating (BPCD) = " << calculatingDuration << " microseconds" << std::endl;

	std::cout << "totaal intersecties: " << totalIntersections << std::endl;
	if (*inside) {
		std::cout << "SNIJDEN NIET" << std::endl;
	}
	else {
		std::cout << "SNIJDEN WEL" << std::endl;
	}

	/*******************************************************************************
	Uitvoeren CUDA kernel - Triangle Triangle zonder Broad Phase Collision Detection
	********************************************************************************/
	std::cout << "--- Calculating ---" << std::endl;
	start = std::chrono::high_resolution_clock::now(); //start time measurement

	/* Uitvoeren CUDA kernel*/
	numberOfBlocks = ((int)((numberOfInsideTriangles + 511) / 512));

	triangle_triangle_GPU_ThreadPerInnerTriangle<<<numberOfBlocks,512>>>(cudaInsideTriangles, cudaInsideVertices, cudaOutsideTriangles, cudaOutsideVertices, cudaInside, numberOfInsideTriangles, numberOfOutsideTriangles);
	err = cudaGetLastError();
	handleCudaError(err);

	/* Kopiëren van de resultaten van GPU naar CPU*/
	//handleCudaError(cudaMemcpy(intersectionsPerInsideTriangle, cudaIntersectionsPerInsideTriangle, numberOfInsideTriangles * sizeof(int), cudaMemcpyDeviceToHost));
	handleCudaError(cudaMemcpy(inside, cudaInside, sizeof(bool), cudaMemcpyDeviceToHost));

	std::cout << "--- End Calculating ---" << std::endl;
	end = std::chrono::high_resolution_clock::now(); //stop time measurement
	calculatingDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "\t\t\tTime Calculating = " << calculatingDuration << " microseconds" << std::endl;

	std::string result;
	std::cout << "totaal intersecties: " << totalIntersections << std::endl;
	if (*inside) {
		result = "SNIJDEN NIET";
	}
	else {
		result = "SNIJDEN WEL";
	}
	std::cout << result << std::endl;
	output.push_back(std::to_string(calculatingDuration) + ";" + result + ";");

	cudaFree(cudaInsideTriangles);
	cudaFree(cudaInsideVertices);
	cudaFree(cudaOutsideTriangles);
	cudaFree(cudaOutsideVertices);
	cudaFree(cudaInside);
	cudaFree(cudaOutsideTriangleIntervals);
	//cudaFree(cudaIntersectionsPerInsideTriangle);
	cudaFreeHost(outsideTriangles);
	cudaFreeHost(outsideVertices);
	cudaFreeHost(insideTriangles);
	cudaFreeHost(insideVertices);
	cudaFreeHost(outsideTriangleIntervals);
	//delete intersectionsPerInsideTriangle;

	delete inside;
}

void TriangleTriangle_BlockPerInnerTriangle(std::unique_ptr<Mesh>& innerMesh, std::unique_ptr<Mesh>& outerMesh)
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

	//TODO: extra lijst met outside mesh driehoekintervallen meegeven (ongesorteerd, omdat ik op dit moment geen idee heb waarom die gesorteerd zou moeten zijn)
	float2* outsideTriangleIntervals = outerMesh->getTriangleInterval();
	float2* cudaOutsideTriangleIntervals;
	int sizeOutsideIntervals = numberOfOutsideTriangles * sizeof(float2);
	handleCudaError(cudaMalloc((void**)&cudaOutsideTriangleIntervals, sizeOutsideIntervals));
	handleCudaError(cudaMemcpyAsync(cudaOutsideTriangleIntervals, outsideTriangleIntervals, sizeOutsideIntervals, cudaMemcpyHostToDevice));

	/* Als deze waarde > 0 ==> De binnenste mesh ligt niet volledig in de buitenste mesh*/
	int totalIntersections = 0;

	std::cout << "--- End Data Transfer ---" << std::endl;
	end = std::chrono::high_resolution_clock::now(); //stop time measurement
	transferDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "\t\t\tTime Data Transfer = " << transferDuration << " milliseconds" << std::endl;

	/****************************************************************************
	Uitvoeren CUDA kernel - Triangle Triangle met Broad Phase Collision Detection
	*****************************************************************************/
	std::cout << "--- Calculating ---" << std::endl;
	start = std::chrono::high_resolution_clock::now(); //start time measurement

	/* Uitvoeren CUDA kernel*/
	int numberOfBlocks = numberOfInsideTriangles;

	triangle_triangle_GPU_BPCD_BlockPerInnerTriangle<<<numberOfBlocks,128>>>(cudaInsideTriangles, cudaInsideVertices, cudaOutsideTriangles, cudaOutsideVertices, cudaInside, numberOfInsideTriangles, numberOfOutsideTriangles, cudaOutsideTriangleIntervals);
	cudaError_t err = cudaGetLastError();
	handleCudaError(err);

	/* Kopiëren van de resultaten van GPU naar CPU*/
	//handleCudaError(cudaMemcpy(intersectionsPerInsideTriangle, cudaIntersectionsPerInsideTriangle, numberOfInsideTriangles * sizeof(int), cudaMemcpyDeviceToHost));
	handleCudaError(cudaMemcpy(inside, cudaInside, sizeof(bool), cudaMemcpyDeviceToHost));

	std::cout << "--- End Calculating ---" << std::endl;
	end = std::chrono::high_resolution_clock::now(); //stop time measurement
	auto calculatingDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "\t\t\tTime Calculating (BPCD) = " << calculatingDuration << " microseconds" << std::endl;

	std::cout << "totaal intersecties: " << totalIntersections << std::endl;
	if (*inside) {
		std::cout << "SNIJDEN NIET" << std::endl;
	}
	else {
		std::cout << "SNIJDEN WEL" << std::endl;
	}

	/*******************************************************************************
	Uitvoeren CUDA kernel - Triangle Triangle zonder Broad Phase Collision Detection
	********************************************************************************/
	std::cout << "--- Calculating ---" << std::endl;
	start = std::chrono::high_resolution_clock::now(); //start time measurement

	/* Uitvoeren CUDA kernel*/
	numberOfBlocks = numberOfInsideTriangles;

	triangle_triangle_GPU_BlockPerInnerTriangle<<<numberOfBlocks,128>>>(cudaInsideTriangles, cudaInsideVertices, cudaOutsideTriangles, cudaOutsideVertices, cudaInside, numberOfInsideTriangles, numberOfOutsideTriangles);
	err = cudaGetLastError();
	handleCudaError(err);

	/* Kopiëren van de resultaten van GPU naar CPU*/
	//handleCudaError(cudaMemcpy(intersectionsPerInsideTriangle, cudaIntersectionsPerInsideTriangle, numberOfInsideTriangles * sizeof(int), cudaMemcpyDeviceToHost));
	handleCudaError(cudaMemcpy(inside, cudaInside, sizeof(bool), cudaMemcpyDeviceToHost));

	std::cout << "--- End Calculating ---" << std::endl;
	end = std::chrono::high_resolution_clock::now(); //stop time measurement
	calculatingDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "\t\t\tTime Calculating = " << calculatingDuration << " microseconds" << std::endl;

	std::string result;
	std::cout << "totaal intersecties: " << totalIntersections << std::endl;
	if (*inside) {
		result = "SNIJDEN NIET";
	}
	else {
		result = "SNIJDEN WEL";
	}
	std::cout << result << std::endl;
	output.push_back(std::to_string(calculatingDuration) + ";" + result + ";");

	cudaFree(cudaInsideTriangles);
	cudaFree(cudaInsideVertices);
	cudaFree(cudaOutsideTriangles);
	cudaFree(cudaOutsideVertices);
	cudaFree(cudaInside);
	cudaFree(cudaOutsideTriangleIntervals);
	//cudaFree(cudaIntersectionsPerInsideTriangle);
	cudaFreeHost(outsideTriangles);
	cudaFreeHost(outsideVertices);
	cudaFreeHost(insideTriangles);
	cudaFreeHost(insideVertices);
	cudaFreeHost(outsideTriangleIntervals);
	//delete intersectionsPerInsideTriangle;

	delete inside;
}

void TriangleTriangle_ThreadPerOuterTriangle(std::unique_ptr<Mesh>& innerMesh, std::unique_ptr<Mesh>& outerMesh)
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

	//TODO: extra lijst met outside mesh driehoekintervallen meegeven (ongesorteerd, omdat ik op dit moment geen idee heb waarom die gesorteerd zou moeten zijn)
	float2* insideTriangleIntervals = innerMesh->getTriangleInterval();
	float2* cudaInsideTriangleIntervals;
	int sizeInsideIntervals = numberOfInsideTriangles * sizeof(float2);
	handleCudaError(cudaMalloc((void**)&cudaInsideTriangleIntervals, sizeInsideIntervals));
	handleCudaError(cudaMemcpyAsync(cudaInsideTriangleIntervals, insideTriangleIntervals, sizeInsideIntervals, cudaMemcpyHostToDevice));

	/* Als deze waarde > 0 ==> De binnenste mesh ligt niet volledig in de buitenste mesh*/
	int totalIntersections = 0;

	std::cout << "--- End Data Transfer ---" << std::endl;
	end = std::chrono::high_resolution_clock::now(); //stop time measurement
	transferDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "\t\t\tTime Data Transfer = " << transferDuration << " milliseconds" << std::endl;

	/****************************************************************************
	Uitvoeren CUDA kernel - Triangle Triangle met Broad Phase Collision Detection
	*****************************************************************************/
	std::cout << "--- Calculating ---" << std::endl;
	start = std::chrono::high_resolution_clock::now(); //start time measurement

	/* Uitvoeren CUDA kernel*/
	int numberOfBlocks = ((int)((numberOfOutsideTriangles + 511) / 512));
	triangle_triangle_GPU_BPCD_ThreadPerOuterTriangle<<<numberOfBlocks,512>>>(cudaInsideTriangles, cudaInsideVertices, cudaOutsideTriangles, cudaOutsideVertices, cudaInside, numberOfInsideTriangles, numberOfOutsideTriangles, cudaInsideTriangleIntervals);
	cudaError_t err = cudaGetLastError();
	handleCudaError(err);

	/* Kopiëren van de resultaten van GPU naar CPU*/
	//handleCudaError(cudaMemcpy(intersectionsPerInsideTriangle, cudaIntersectionsPerInsideTriangle, numberOfInsideTriangles * sizeof(int), cudaMemcpyDeviceToHost));
	handleCudaError(cudaMemcpy(inside, cudaInside, sizeof(bool), cudaMemcpyDeviceToHost));

	std::cout << "--- End Calculating ---" << std::endl;
	end = std::chrono::high_resolution_clock::now(); //stop time measurement
	auto calculatingDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "\t\t\tTime Calculating (BPCD) = " << calculatingDuration << " microseconds" << std::endl;

	std::cout << "totaal intersecties: " << totalIntersections << std::endl;
	if (*inside) {
		std::cout << "SNIJDEN NIET" << std::endl;
	}
	else {
		std::cout << "SNIJDEN WEL" << std::endl;
	}

	/*******************************************************************************
	Uitvoeren CUDA kernel - Triangle Triangle zonder Broad Phase Collision Detection
	********************************************************************************/
	std::cout << "--- Calculating ---" << std::endl;
	start = std::chrono::high_resolution_clock::now(); //start time measurement

	/* Uitvoeren CUDA kernel*/
	numberOfBlocks = ((int)((numberOfOutsideTriangles + 511) / 512));
	triangle_triangle_GPU_ThreadPerOuterTriangle<<<numberOfBlocks,512>>>(cudaInsideTriangles, cudaInsideVertices, cudaOutsideTriangles, cudaOutsideVertices, cudaInside, numberOfInsideTriangles, numberOfOutsideTriangles);
	err = cudaGetLastError();
	handleCudaError(err);

	/* Kopiëren van de resultaten van GPU naar CPU*/
	//handleCudaError(cudaMemcpy(intersectionsPerInsideTriangle, cudaIntersectionsPerInsideTriangle, numberOfInsideTriangles * sizeof(int), cudaMemcpyDeviceToHost));
	handleCudaError(cudaMemcpy(inside, cudaInside, sizeof(bool), cudaMemcpyDeviceToHost));

	std::cout << "--- End Calculating ---" << std::endl;
	end = std::chrono::high_resolution_clock::now(); //stop time measurement
	calculatingDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "\t\t\tTime Calculating = " << calculatingDuration << " microseconds" << std::endl;

	std::string result;
	std::cout << "totaal intersecties: " << totalIntersections << std::endl;
	if (*inside) {
		result = "SNIJDEN NIET";
	}
	else {
		result = "SNIJDEN WEL";
	}
	std::cout << result << std::endl;
	output.push_back(std::to_string(calculatingDuration) + ";" + result + "\n");

	cudaFree(cudaInsideTriangles);
	cudaFree(cudaInsideVertices);
	cudaFree(cudaOutsideTriangles);
	cudaFree(cudaOutsideVertices);
	cudaFree(cudaInside);
	cudaFree(cudaInsideTriangleIntervals);
	//cudaFree(cudaIntersectionsPerInsideTriangle);
	cudaFreeHost(outsideTriangles);
	cudaFreeHost(outsideVertices);
	cudaFreeHost(insideTriangles);
	cudaFreeHost(insideVertices);
	cudaFreeHost(insideTriangleIntervals);
	//delete intersectionsPerInsideTriangle;

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