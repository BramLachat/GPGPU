#include <vector>
#include <string>
#include <iostream>
#include <iterator>
#include <map>
#include <fstream>
#include <chrono>
#include <algorithm> 

#include "omp.h"
#include "Mesh.h"


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
void Mesh::addVertex(const Point& v)
{
	vertices.push_back(v);
}
int Mesh::findDuplicate(const Point& v)
{
	std::map<std::string, int>::iterator itr = VertexIndices.find(std::to_string(v[0]) + std::to_string(v[1]) + std::to_string(v[2]));
	if (itr != VertexIndices.end())
	{
		return itr->second;
	}
	else
	{
		return -1;
	}
}
int Mesh::getLastVertex()
{
	return (vertices.size() - 1);
}
Point* Mesh::getVertexAtIndex(int index)
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
std::list<Point> Mesh::getPointList()
{
	std::list<Point> result(vertices.begin(), vertices.end());
	return result;
}
std::list<Triangle> Mesh::getTriangleList()
{
	std::list<Triangle> result(triangles.begin(), triangles.end());
	return result;
}
/*int3* Mesh::getInt3ArrayTriangles()
{
	int3* triangleArray;
	cudaError_t status = cudaHostAlloc((void**)&triangleArray, triangles.size() * sizeof(int3), cudaHostAllocDefault);
	if (status != cudaSuccess) printf("Error allocating pinned host memory\n");
	//int3* triangleArray = new int3[triangles.size()];
	for (int i = 0; i < triangles.size(); i++)
	{
		triangleArray[i] = triangles[i].getIndexOfVerticesInMesh();
	}
	return triangleArray;
}
float3* Mesh::getFloat3ArrayVertices()
{
	float3* vertexArray;
	cudaError_t status = cudaHostAlloc((void**)&vertexArray, vertices.size() * sizeof(float3), cudaHostAllocDefault);
	if (status != cudaSuccess) printf("Error allocating pinned host memory\n");
	//float3* vertexArray = new float3[vertices.size()];
	for (int i = 0; i < vertices.size(); i++)
	{
		vertexArray[i] = vertices[i].getCoordinatesFloat3();
	}
	return vertexArray;
}*/