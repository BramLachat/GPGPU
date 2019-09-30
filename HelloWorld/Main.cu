#include <cassert>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
//#include <memory> //needed for smart pointers

#include "Mesh.h"
#include "parse_stl.h"
#include "RayTriangleIntersect.cuh"

void rayTriangleIntersect(float dir[3], std::unique_ptr<Mesh>& innerMesh, std::unique_ptr<Mesh>& outerMesh);

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

	std::cout << "direction = " << direction[0] << ", " << direction[1] << ", " << direction[2] << std::endl;

	auto start = std::chrono::high_resolution_clock::now(); //start time measurement

	if (RayTriangle == 0)
	{
		//2 opties om unique ptr mee te geven als argument aan een functie:
		//https://stackoverflow.com/questions/30905487/how-can-i-pass-stdunique-ptr-into-a-function
		//triangleMesh_Outside->rayTriangleIntersect(direction, triangleMesh_Inside);
		rayTriangleIntersect(direction, triangleMesh_Inside, triangleMesh_Outside);
	}
	else
	{
		triangleMesh_Outside->triangleTriangleIntersect(triangleMesh_Inside);
	}	
	
	auto end = std::chrono::high_resolution_clock::now(); //stop time measurement
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "Time = " << duration << "ms" << std::endl;

	std::cout << "Press Enter to quit program!" << std::endl;
	std::cin.get();
	std::cin.get();
	return 0;
}

void rayTriangleIntersect(float dir[3], std::unique_ptr<Mesh>& innerMesh, std::unique_ptr<Mesh>& outerMesh)
{
	/*float* t = new float;
	float* u = new float;
	float* v = new float;
	Vertex* V1;
	Vertex* V2;
	Vertex* V3;
	float* vert1;
	float* vert2;
	float* vert3;*/
	float* orig;
	float* cudaOrig;
	float* cudaDir;
	int verticesSize;
	int trianglesSize;
	int* cudaVerticesSize;
	int* cudaTrianglesSize;


	Vertex* innerVertex;
	Vertex* outerVerticesCuda;
	Triangle* outerTrianglesCuda;

	std::vector<Vertex> innerVertices = innerMesh->getVertices();
	std::vector<Triangle> outertriangles = outerMesh->getTriangles();
	std::vector<Vertex> outerVertices = outerMesh->getVertices();

	verticesSize = outerVertices.size();
	trianglesSize = outertriangles.size();
	
	outerVerticesCuda = &outerVertices[0];
	cudaMalloc((void**)& outerVerticesCuda, outerVertices.size()*sizeof(Vertex));
	cudaMemcpy(outerVerticesCuda, &outerVertices, outerVertices.size() * sizeof(Vertex), cudaMemcpyHostToDevice);

	outerTrianglesCuda = &outertriangles[0];
	cudaMalloc((void**)& outerTrianglesCuda, outertriangles.size()*sizeof(Triangle));
	cudaMemcpy(outerTrianglesCuda, &outertriangles, outertriangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

	cudaMalloc((void**)& cudaDir, 3*sizeof(float));
	cudaMemcpy(cudaDir, dir, 3*sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)& cudaTrianglesSize, sizeof(int));
	cudaMemcpy(cudaTrianglesSize, &trianglesSize, sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)& cudaVerticesSize, sizeof(int));
	cudaMemcpy(cudaVerticesSize, &verticesSize, sizeof(int), cudaMemcpyHostToDevice);

	bool inside = true;

	for (int j = 0; j < innerMesh->getNumberOfVertices(); j++)
	{
		innerVertex = &(innerVertices.at(j));
		orig = innerVertex->getCoordinates();

		int* result = new int[trianglesSize];
		int* cudaResult;

		cudaMalloc((void**)& cudaOrig, 3*sizeof(float));
		cudaMemcpy(cudaOrig, orig, 3*sizeof(float), cudaMemcpyHostToDevice);

		cudaMalloc((void**)& cudaResult, trianglesSize*sizeof(int));


		int numberOfIntersections = 0;

		Intersection::intersect_triangle4<<<1,trianglesSize>>>(cudaOrig, cudaDir, outerTrianglesCuda, outerVerticesCuda, cudaVerticesSize, cudaTrianglesSize, cudaResult);

		cudaMemcpy(result, cudaResult, trianglesSize * sizeof(int), cudaMemcpyDeviceToHost);

		std::cout << "result = ";
		for (int i = 0; i < trianglesSize; i++)
		{
			std::cout << result[i] << ", ";
		}
		std::cout << std::endl;

		/*for (int i = 0; i < triangles.size(); i++)
		{
			V1 = &(vertices.at(triangles.at(i).getIndexOfVertexInMesh(0)));
			V2 = &(vertices.at(triangles.at(i).getIndexOfVertexInMesh(1)));
			V3 = &(vertices.at(triangles.at(i).getIndexOfVertexInMesh(2)));
			vert1 = V1->getCoordinates();
			vert2 = V2->getCoordinates();
			vert3 = V3->getCoordinates();
			if (Intersection::intersect_triangle3(orig, dir, vert1, vert2, vert3, t, u, v) == 1)
			{
				numberOfIntersections++;
			}
		}*/
		//std::cout << "aantal intersecties = " << numberOfIntersections << std::endl;
		if (numberOfIntersections % 2 == 0)
		{
			inside = false;
		}
		cudaFree(cudaOrig);
		cudaFree(cudaResult);
		delete result;
	}
	cudaFree(outerVerticesCuda);
	cudaFree(outerTrianglesCuda);
	cudaFree(cudaDir);
	cudaFree(cudaTrianglesSize);
	cudaFree(cudaVerticesSize);
	//delete t; delete u; delete v;
	if (inside) { std::cout << "INSIDE" << std::endl; }
	else { std::cout << "OUTSIDE" << std::endl; }
}