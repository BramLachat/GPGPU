#include <cassert>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <cooperative_groups.h>
//#include <memory> //needed for smart pointers

#include "Mesh.h"
#include "parse_stl.h"
#include "RayTriangleIntersect.cuh"
#include "TriangleTriangleIntersect.cuh"

void rayTriangleIntersect(float dir[3], std::unique_ptr<Mesh>& innerMesh, std::unique_ptr<Mesh>& outerMesh);
void rayTriangleIntersect_v2(float dir[3], std::unique_ptr<Mesh>& innerMesh, std::unique_ptr<Mesh>& outerMesh);
void rayTriangleIntersect_v3(float dir[3], std::unique_ptr<Mesh>& innerMesh, std::unique_ptr<Mesh>& outerMesh);
void rayTriangle_BlockPerTriangle(float dir[3], std::unique_ptr<Mesh>& innerMesh, std::unique_ptr<Mesh>& outerMesh);
//void rayTriangle_BlockPerTriangle_v2(float dir[3], std::unique_ptr<Mesh>& innerMesh, std::unique_ptr<Mesh>& outerMesh);
void handleCudaError(cudaError_t cudaERR);
__global__ void startGPU();

std::ofstream outfile;
cudaEvent_t start_event, stop_event;

int main(int argc, char* argv[]) {

	outfile.open("file.csv", std::ios::app);

	std::string stl_file_inside;
	std::string stl_file_outside;
	std::cout << "Enter filename of inside mesh:" << std::endl;
	std::cin >> stl_file_inside;
	std::cout << "Enter filename of outside mesh:" << std::endl;
	std::cin >> stl_file_outside;

	//Only reads STL-file in binary format!!!
	std::cout << "Reading files:" << std::endl;
	std::unique_ptr<Mesh> triangleMesh_Inside = stl::parse_stl(stl_file_inside);
	std::unique_ptr<Mesh> triangleMesh_Outside = stl::parse_stl(stl_file_outside);

	Vertex* V1 = triangleMesh_Outside->getVertexAtIndex(0);
	Vertex* V2 = triangleMesh_Outside->getVertexAtIndex(1);
	Vertex* V3 = triangleMesh_Outside->getVertexAtIndex(2);

	float xCenter = (V1->getCoordinates()[0] + V2->getCoordinates()[0] + V3->getCoordinates()[0])/3;
	float yCenter = (V1->getCoordinates()[1] + V2->getCoordinates()[1] + V3->getCoordinates()[1])/3;
	float zCenter = (V1->getCoordinates()[2] + V2->getCoordinates()[2] + V3->getCoordinates()[2])/3;

	float direction[3] = { xCenter, yCenter, zCenter };
	
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);

	std::cout << "direction = " << direction[0] << ", " << direction[1] << ", " << direction[2] << std::endl;

	rayTriangleIntersect(direction, triangleMesh_Inside, triangleMesh_Outside);	
	rayTriangleIntersect_v2(direction, triangleMesh_Inside, triangleMesh_Outside);
	rayTriangleIntersect_v3(direction, triangleMesh_Inside, triangleMesh_Outside);
	rayTriangle_BlockPerTriangle(direction, triangleMesh_Inside, triangleMesh_Outside);
	//rayTriangle_BlockPerTriangle_v2(direction, triangleMesh_Inside, triangleMesh_Outside);
	
	std::cout << "Press Enter to quit program!" << std::endl;
	std::cin.get();
	std::cin.get();

	outfile.close();
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

	std::cout << "Transfering data from cpu to gpu!" << std::endl;
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

	cudaDeviceSynchronize();
	end = std::chrono::high_resolution_clock::now(); //stop time measurement
	transferDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

	int totalIntersections = 0;
	std::cout << "Kernel execution: rayTriangleIntersect" << std::endl;

	int numberOfBlocks = ((int)((numberOfOutsideTriangles + 127) / 128));

	cudaEventRecord(start_event);
	Intersection::intersect_triangleGPU<<<numberOfBlocks,128>>>(cudaInsideOrigins, cudaDir, cudaOutsideTriangles, cudaOutsideVertices, numberOfInsideVertices, numberOfOutsideTriangles, cudaIntersectionsPerOrigin);
	cudaEventRecord(stop_event);

	cudaError_t err = cudaGetLastError();
	handleCudaError(err);

	handleCudaError(cudaMemcpy(intersectionsPerOrigin, cudaIntersectionsPerOrigin, numberOfInsideVertices * sizeof(int), cudaMemcpyDeviceToHost));
	cudaEventSynchronize(stop_event);

	int i = 0;
	while (i < numberOfInsideVertices && inside)
	{
		if (intersectionsPerOrigin[i] % 2 == 0) {
			inside = false;
		}
		i++;
	}

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start_event, stop_event);

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
	if (inside) { result = "INSIDE"; }
	else { result = "OUTSIDE"; }
	std::cout << result << std::endl;
	outfile << std::to_string(milliseconds) + ";" + result + ";" + std::to_string(totalIntersections) + ";";
}

void rayTriangleIntersect_v2(float dir[3], std::unique_ptr<Mesh>& innerMesh, std::unique_ptr<Mesh>& outerMesh)
{
	std::cout << "Transfering data from cpu to gpu!" << std::endl;
	auto start = std::chrono::high_resolution_clock::now(); //start time measurement

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

	cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now(); //stop time measurement
	auto transferDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

	int totalIntersections = 0;
	std::cout << "Kernel execution: rayTriangleIntersect_v2" << std::endl;

	int numberOfBlocks = ((int)((numberOfOutsideTriangles + 127) / 128));

	cudaEventRecord(start_event);
	Intersection::intersect_triangleGPU_v2<< <numberOfBlocks, 128 >> > (cudaInsideOrigins, cudaDir, cudaOutsideTriangles, cudaOutsideVertices, numberOfInsideVertices, numberOfOutsideTriangles, cudaIntersectionsPerOrigin);
	cudaEventRecord(stop_event);

	cudaError_t err = cudaGetLastError();
	handleCudaError(err);

	handleCudaError(cudaMemcpy(intersectionsPerOrigin, cudaIntersectionsPerOrigin, numberOfInsideVertices * sizeof(int), cudaMemcpyDeviceToHost));
	cudaEventSynchronize(stop_event);

	int i = 0;
	while (i < numberOfInsideVertices && inside)
	{
		if (intersectionsPerOrigin[i] % 2 == 0) {
			inside = false;
		}
		i++;
	}

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start_event, stop_event);

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
	if (inside) { result = "INSIDE"; }
	else { result = "OUTSIDE"; }
	std::cout << result << std::endl;
	outfile << std::to_string(milliseconds) + ";" + result + ";" + std::to_string(totalIntersections) + ";";
}

void rayTriangleIntersect_v3(float dir[3], std::unique_ptr<Mesh>& innerMesh, std::unique_ptr<Mesh>& outerMesh)
{
	std::cout << "Transfering data from cpu to gpu!" << std::endl;
	auto start = std::chrono::high_resolution_clock::now(); //start time measurement

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

	cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now(); //stop time measurement
	auto transferDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

	int totalIntersections = 0;
	std::cout << "Kernel execution: rayTriangleIntersect_v3" << std::endl;

	int numberOfBlocks = ((int)((numberOfOutsideTriangles + 127) / 128));

	cudaEventRecord(start_event);
	Intersection::intersect_triangleGPU_v3 << <numberOfBlocks, 128 >> > (cudaInsideOrigins, cudaDir, cudaOutsideTriangles, cudaOutsideVertices, numberOfInsideVertices, numberOfOutsideTriangles, cudaIntersectionsPerOrigin);
	cudaEventRecord(stop_event);

	cudaError_t err = cudaGetLastError();
	handleCudaError(err);

	handleCudaError(cudaMemcpy(intersectionsPerOrigin, cudaIntersectionsPerOrigin, numberOfInsideVertices * sizeof(int), cudaMemcpyDeviceToHost));
	cudaEventSynchronize(stop_event);

	int i = 0;
	while (i < numberOfInsideVertices && inside)
	{
		if (intersectionsPerOrigin[i] % 2 == 0) {
			inside = false;
		}
		i++;
	}

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start_event, stop_event);

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
	if (inside) { result = "INSIDE"; }
	else { result = "OUTSIDE"; }
	std::cout << result << std::endl;
	outfile << std::to_string(milliseconds) + ";" + result + ";" + std::to_string(totalIntersections) + ";";
}

void rayTriangle_BlockPerTriangle(float dir[3], std::unique_ptr<Mesh>& innerMesh, std::unique_ptr<Mesh>& outerMesh)
{
	std::cout << "Transfering data from cpu to gpu!" << std::endl;
	auto start = std::chrono::high_resolution_clock::now(); //start time measurement

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

	cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now(); //stop time measurement
	auto transferDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

	int totalIntersections = 0;
	std::cout << "Kernel execution: rayTriangle_BlockPerTriangle" << std::endl;

	cudaEventRecord(start_event);
	Intersection::intersect_triangleGPU_BlockPerTriangle << <numberOfOutsideTriangles, 128 >> > (cudaInsideOrigins, cudaDir, cudaOutsideTriangles, cudaOutsideVertices, numberOfInsideVertices, cudaIntersectionsPerOrigin);
	cudaEventRecord(stop_event);

	cudaError_t err = cudaGetLastError();
	handleCudaError(err);

	handleCudaError(cudaMemcpy(intersectionsPerOrigin, cudaIntersectionsPerOrigin, numberOfInsideVertices * sizeof(int), cudaMemcpyDeviceToHost));
	cudaEventSynchronize(stop_event);

	int i = 0;
	while (i < numberOfInsideVertices && inside)
	{
		if (intersectionsPerOrigin[i] % 2 == 0) {
			inside = false;
		}
		i++;
	}

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start_event, stop_event);

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
	if (inside) { result = "INSIDE"; }
	else { result = "OUTSIDE"; }
	std::cout << result << std::endl;
	outfile << std::to_string(milliseconds) + ";" + result + ";" + std::to_string(totalIntersections) + ";";
}

/*void rayTriangle_BlockPerTriangle_v2(float dir[3], std::unique_ptr<Mesh>& innerMesh, std::unique_ptr<Mesh>& outerMesh)
{
	std::cout << "Transfering data from cpu to gpu!" << std::endl;
	auto start = std::chrono::high_resolution_clock::now(); //start time measurement

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

	cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now(); //stop time measurement
	auto transferDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

	int totalIntersections = 0;
	std::cout << "Kernel execution: rayTriangle_BlockPerTriangle_v2" << std::endl;

	cudaEventRecord(start_event);
	Intersection::intersect_triangleGPU_BlockPerTriangle_v2 << <numberOfOutsideTriangles, 128 >> > (cudaInsideOrigins, cudaDir, cudaOutsideTriangles, cudaOutsideVertices, numberOfInsideVertices, cudaIntersectionsPerOrigin);
	cudaEventRecord(stop_event);

	cudaError_t err = cudaGetLastError();
	handleCudaError(err);

	handleCudaError(cudaMemcpy(intersectionsPerOrigin, cudaIntersectionsPerOrigin, numberOfInsideVertices * sizeof(int), cudaMemcpyDeviceToHost));
	cudaEventSynchronize(stop_event);

	int i = 0;
	while (i < numberOfInsideVertices && inside)
	{
		if (intersectionsPerOrigin[i] % 2 == 0) {
			inside = false;
		}
		i++;
	}

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start_event, stop_event);

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
	if (inside) { result = "INSIDE"; }
	else { result = "OUTSIDE"; }
	std::cout << result << std::endl;
	outfile << std::to_string(milliseconds) + ";" + result + ";" + std::to_string(totalIntersections) + "\n";
}*/

void handleCudaError(cudaError_t cudaERR) {
	if (cudaERR != cudaSuccess) {
		printf("CUDA ERROR : %s\n", cudaGetErrorString(cudaERR));
	}
}

__global__ void startGPU() {
	printf("GPU ready!\n");
}