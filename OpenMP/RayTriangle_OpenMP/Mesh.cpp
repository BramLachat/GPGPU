#include <vector>
#include <string>
#include <iostream>
#include <iterator>
#include <map>
#include <fstream>
#include <chrono>

#include "omp.h"
#include "Mesh.h"
#include "RayTriangleIntersect.cuh"
#include "TriangleTriangleIntersect.h"


Mesh::Mesh(std::string n, unsigned int size)
{
	name = n;
	triangles.reserve(size);
	vertices.reserve(size);
}
std::string Mesh::getName()
{
	return name;
}
int Mesh::getNumberOfTriangles()
{
	return triangles.size();
}
int Mesh::getNumberOfVertices()
{
	return vertices.size();
}
void Mesh::addTriangle(const Triangle& t)
{
	triangles.push_back(t);
}
void Mesh::addVertex(const Vertex& v)
{
	vertices.push_back(v);
}
int Mesh::findDuplicate(const Vertex& v)
{
	std::map<std::string, int>::iterator itr = VertexIndices.find(v.toString());
	if (itr != VertexIndices.end())
	{
		return itr->second;
	}
	else
	{
		return -1;
	}

	/*std::vector<Vertex>::iterator it = std::find(vertices.begin(), vertices.end(), v);
	if (it != vertices.end())//returned index of last value if nothing found!
	{
		return std::distance(vertices.begin(), it);
	}
	else
	{
		return -1;
	}*/
}
void Mesh::rayTriangleIntersect(float dir[3], std::unique_ptr<Mesh>& innerMesh)
{
	std::vector<Vertex> outermesh_vertices = vertices; //Nodig voor OpenMP
	std::vector<Triangle> outermesh_triangles = triangles; //Nodig voor OpenMP

	int aantal_threads = 1;
	std::cout << "Choose the number of threads!" << std::endl;
	std::cin >> aantal_threads;
	omp_set_num_threads(aantal_threads);
	std::cout << "aantal threads: " << omp_get_num_threads() << std::endl;

	auto start = std::chrono::high_resolution_clock::now(); //start time measurement

	std::unique_ptr<std::vector<Vertex>> outsideVertices = std::make_unique<std::vector<Vertex>>(innerMesh->getNumberOfVertices());
	bool inside = true;
	std::unique_ptr<std::vector<int>> totalIntersections = std::make_unique<std::vector<int>>(aantal_threads);
	int totaalAantalIntersecties = 0;

#pragma omp parallel shared(innerMesh,outermesh_vertices,outermesh_triangles,totalIntersections,outsideVertices)
	{

	//std::unique_ptr<float[]> orig = std::make_unique<float[]>(3); //smart pointer
	//std::vector<float> orig;
	float* t = new float;
	float* u = new float;
	float* v = new float;
	Vertex* V1;
	Vertex* V2;
	Vertex* V3;
	float* vert1;
	float* vert2;
	float* vert3;
	float* orig;
	//float dirPerPoint[3];
	int tid = omp_get_thread_num();
	totalIntersections->at(tid) = 0;

	Vertex* innerVertex;

	std::cout << "\t\t\tCalculating intersections! (CPU)" << std::endl;

#pragma omp for schedule(static)
	for(int j = 0 ; j < innerMesh->getNumberOfVertices() ; j++)
	{
		innerVertex = &(innerMesh->vertices.at(j));
		orig = innerVertex->getCoordinates();
		//dirPerPoint[0] = dir[0] - orig[0];
		//dirPerPoint[1] = dir[1] - orig[1];
		//dirPerPoint[2] = dir[2] - orig[2];
		//std::cout << "orig = " << orig[0] << ", " << orig[1] << ", " << orig[2] << std::endl;

		int numberOfIntersections = 0;

		for (int i = 0; i < triangles.size(); i++)
		{
			V1 = &(vertices.at(triangles.at(i).getIndexOfVertexInMesh(0)));
			V2 = &(vertices.at(triangles.at(i).getIndexOfVertexInMesh(1)));
			V3 = &(vertices.at(triangles.at(i).getIndexOfVertexInMesh(2)));
			vert1 = V1->getCoordinates();
			vert2 = V2->getCoordinates();
			vert3 = V3->getCoordinates();
			//if (Intersection::intersect_triangle3(orig, dirPerPoint, vert1, vert2, vert3, t, u, v) == 1)
			if (Intersection::intersect_triangleCPU(orig, dir, vert1, vert2, vert3, t, u, v) == 1)
			{
				numberOfIntersections++;
				//std::cout << "1, ";
			}
			else {
				//std::cout << "0, ";
			}
		}
		totalIntersections->at(tid) += numberOfIntersections;
		//std::cout << "aantal intersecties = " << numberOfIntersections << std::endl;
		if (numberOfIntersections % 2 == 0)
		{
			inside = false;
			outsideVertices->at(j) = *innerVertex;
		}
	}
	if (tid == 0) {
		for (int i = 1; i < aantal_threads; i++) {
			totalIntersections->at(0) += totalIntersections->at(i);
		}
		totaalAantalIntersecties = totalIntersections->at(0);
	}
	delete t; delete u; delete v;
	}

	auto end = std::chrono::high_resolution_clock::now(); //stop time measurement
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "\t\t\tTime CPU = " << duration << "ms" << std::endl;

	std::cout << "totaal intersecties = " << totaalAantalIntersecties << std::endl;
	if (inside) { std::cout << "INSIDE" << std::endl; }
	else { std::cout << "OUTSIDE" << std::endl; }

	std::cout << "Writing to file!" << std::endl;
	writeVerticesToFile(outsideVertices, "OutsideVertices.stl");
}
void Mesh::triangleTriangleIntersect(std::unique_ptr<Mesh>& innerMesh)
{
	auto start = std::chrono::high_resolution_clock::now(); //start time measurement

	float* vert1_1;
	float* vert1_2;
	float* vert1_3;
	float* vert2_1;
	float* vert2_2;
	float* vert2_3;
	Triangle* t1;
	Triangle* t2;
	std::vector<Vertex>* innerVertices = &(innerMesh->vertices);

	std::unique_ptr<std::vector<Triangle>> intersectingTriangles1 = std::make_unique<std::vector<Triangle>>();
	std::unique_ptr<std::vector<Triangle>> intersectingTriangles2 = std::make_unique<std::vector<Triangle>>();

	bool inside = true;

	std::cout << "\t\t\tCalculating intersecting triangles! (CPU)" << std::endl;

	for (int j = 0; j < innerMesh->getNumberOfTriangles(); j++)
	{
		t1 = &(innerMesh->triangles.at(j));
		vert1_1 = innerVertices->at(t1->getIndexOfVertexInMesh(0)).getCoordinates();
		vert1_2 = innerVertices->at(t1->getIndexOfVertexInMesh(1)).getCoordinates();
		vert1_3 = innerVertices->at(t1->getIndexOfVertexInMesh(2)).getCoordinates();

		int numberOfIntersections = 0;

		for (int i = 0; i < triangles.size(); i++)
		{
			t2 = &(triangles.at(i));
			vert2_1 = vertices.at(t2->getIndexOfVertexInMesh(0)).getCoordinates();
			vert2_2 = vertices.at(t2->getIndexOfVertexInMesh(1)).getCoordinates();
			vert2_3 = vertices.at(t2->getIndexOfVertexInMesh(2)).getCoordinates();
			if (Intersection::NoDivTriTriIsect(vert1_1, vert1_2, vert1_3, vert2_1, vert2_2, vert2_3) == 1)
			{
				//list printed with intersecting triangles
				intersectingTriangles1->push_back(*t1);
				intersectingTriangles2->push_back(*t2);

				numberOfIntersections++;
			}
		}
		//std::cout << "aantal intersecties = " << numberOfIntersections << std::endl;
		if (numberOfIntersections != 0)
		{
			inside = false;
		}
	}
	if (inside) { std::cout << "SNIJDEN NIET" << std::endl; }
	else { std::cout << "SNIJDEN WEL" << std::endl; }

	auto end = std::chrono::high_resolution_clock::now(); //stop time measurement
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "\t\t\tTime CPU = " << duration << "ms" << std::endl;

	std::cout << "Writing to file!" << std::endl;
	writeTrianglesToFile(intersectingTriangles1, innerVertices, "IntersectingTriangles1.stl");
	writeTrianglesToFile(intersectingTriangles2, &vertices, "IntersectingTriangles2.stl");
}
int Mesh::getLastVertex()
{
	return (vertices.size() - 1);
}
void Mesh::schrijf()
{
	for (int i = 0; i < triangles.size(); i++)
	{
		std::cout << "driehoek " << i << std::endl;
		for (int j = 0; j < 3; j++)
		{
			std::cout << "vertex " << j << ": index = " << triangles.at(i).getIndexOfVertexInMesh(j) << std::endl;
			Vertex v = vertices.at(triangles.at(i).getIndexOfVertexInMesh(j));
			v.schrijf();
		}
		std::cout << std::endl;
	}
	std::map<std::string, int>::iterator itr;
	for (itr = VertexIndices.begin(); itr != VertexIndices.end(); itr++)
	{
		std::cout << "key: " << itr->first  // string (key)
			<< ':'
			<< "value: " << itr->second   // string's value 
			<< std::endl;
	}
}
Vertex* Mesh::getVertexAtIndex(int index)
{
	return &(vertices.at(index));
}
void Mesh::resize()
{
	vertices.shrink_to_fit();
}
void Mesh::addVertexIndex(const std::string& s, int index)
{
	VertexIndices.insert(std::pair<std::string, int>(s, index));
}
void Mesh::writeTrianglesToFile(std::unique_ptr<std::vector<Triangle>>& triangles, std::vector<Vertex>* vertices, std::string fileName)
{
	std::vector<Triangle>::iterator itr;
	//std::string path = "C:\\Users\\hla\\Documents\\Masterproef\\GPGPU\\Output\\" + fileName;
	std::string path = "D:\\Masterproef\\GPGPU\\Output\\" + fileName;
	std::ofstream ofs(path);
	ofs << "solid IntersectingTriangles" << std::endl;
	for (itr = triangles->begin(); itr != triangles->end(); ++itr)
	{
		ofs << "  facet normal  0.0  0.0  0.0" << std::endl;
		ofs << "    outer loop" << std::endl;
		float* vert;
		for (int j = 0; j < 3; j++)
		{
			vert = vertices->at(itr->getIndexOfVertexInMesh(j)).getCoordinates();
			ofs << "      vertex  " << vert[0] << "  "
				<< vert[1] << "  "
				<< vert[2] << std::endl;
		}
		ofs << "    endloop" << std::endl;
		ofs << "  endfacet" << std::endl;
	}
	ofs << "endsolid vcg" << std::endl;
}
int3* Mesh::getInt3ArrayTriangles()
{
	int3* triangleArray;
	cudaError_t status = cudaHostAlloc((void**)& triangleArray, triangles.size() * sizeof(int3), cudaHostAllocDefault);
	if (status != cudaSuccess) printf("Error allocating pinned host memory\n");
	//int3* triangleArray = new int3[triangles.size()];
	for (int i = 0 ; i < triangles.size() ; i++)
	{
		triangleArray[i] = triangles[i].getIndexOfVerticesInMesh();
	}
	return triangleArray;
}
thrust::host_vector<int3> Mesh::getTrianglesVector()
{
	thrust::host_vector<int3> result(triangles.size());
	for (int i = 0; i < triangles.size(); i++) 
	{

		result[i] = triangles[i].getIndexOfVerticesInMesh();
	}
	return result;
}
float3* Mesh::getFloat3ArrayVertices()
{
	float3* vertexArray;
	cudaError_t status = cudaHostAlloc((void**)& vertexArray, vertices.size() * sizeof(float3), cudaHostAllocDefault);
	if (status != cudaSuccess) printf("Error allocating pinned host memory\n");
	//float3* vertexArray = new float3[vertices.size()];
	for (int i = 0; i < vertices.size(); i++)
	{
		vertexArray[i] = vertices[i].getCoordinatesFloat3();
	}
	return vertexArray;
}
thrust::host_vector<float3> Mesh::getVerticesVector()
{
	thrust::host_vector<float3> result(vertices.size());
	for (int i = 0; i < vertices.size(); i++)
	{

		result[i] = vertices[i].getCoordinatesFloat3();
	}
	return result;
}
void Mesh::writeVerticesToFile(std::unique_ptr<std::vector<Vertex>>& vertices, std::string fileName)
{
	std::vector<Vertex>::iterator itr;
	//std::string path = "C:\\Users\\hla\\Documents\\Masterproef\\GPGPU\\Output\\" + fileName;
	std::string path = "D:\\Masterproef\\GPGPU\\Output\\" + fileName;
	std::ofstream ofs(path);
	ofs << "solid OutsideVertices" << std::endl;
	float* vert;
	for (itr = vertices->begin(); itr != vertices->end(); ++itr)
	{
		vert = itr->getCoordinates();
		if(vert[0] + vert[1] + vert[2] != 0){
			ofs << "  facet normal  0.0  0.0  0.0" << std::endl;
			ofs << "    outer loop" << std::endl;
			ofs << "      vertex  " << vert[0] << "  "
				<< vert[1] << "  "
				<< vert[2] << std::endl;
			ofs << "      vertex  " << (vert[0]-0.1) << "  "
				<< (vert[1]-0.1) << "  "
				<< (vert[2]-0.1) << std::endl;
			ofs << "      vertex  " << (vert[0] + 0.1) << "  "
				<< (vert[1] + 0.1) << "  "
				<< (vert[2] - 0.1) << std::endl;
			ofs << "    endloop" << std::endl;
			ofs << "  endfacet" << std::endl;
		}
	}
	ofs << "endsolid vcg" << std::endl;
}