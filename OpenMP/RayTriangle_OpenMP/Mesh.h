#ifndef MESH_H
#define MESH_H

#include <string>
#include <map>
#include <thrust/device_vector.h>

#include "Vertex.h"
#include "Triangle.h"

class Mesh
{
private:
	std::string name;
	std::vector<Triangle> triangles;
	std::vector<Vertex> vertices;
	std::map<std::string, int> VertexIndices;
public:
	Mesh(std::string name, unsigned int size);
	std::string getName();
	int getNumberOfTriangles();
	int getNumberOfVertices();
	void addTriangle(const Triangle& t);
	void addVertex(const Vertex& v);
	int findDuplicate(const Vertex& v);
	int rayTriangleIntersectOpenMP(float dir[3], std::unique_ptr<Mesh>& innerMesh, int number_of_threads);
	int rayTriangleIntersect(float dir[3], std::unique_ptr<Mesh>& innerMesh);
	int triangleTriangleIntersect(std::unique_ptr<Mesh>& innerMesh);
	int getLastVertex();
	void schrijf();
	Vertex* getVertexAtIndex(int index);
	void resize();
	void addVertexIndex(const std::string& s, int index);
	void writeTrianglesToFile(std::unique_ptr<std::vector<Triangle>>& triangles, std::vector<Vertex>* vertices, std::string fileName);
	int3* getInt3ArrayTriangles();
	thrust::host_vector<int3> getTrianglesVector();
	float3* getFloat3ArrayVertices();
	thrust::host_vector<float3> getVerticesVector();
	void writeVerticesToFile(std::unique_ptr<std::vector<Vertex>>& vertices, std::string fileName);
};
#endif // !MESH_H