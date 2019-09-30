#ifndef MESH_H
#define MESH_H

#include <string>
#include <map>

#include "Vertex.h"
#include "Triangle.h"

class Mesh
{
private:
	std::string name;
	std::vector<Triangle> triangles;
	std::vector<Vertex> vertices;
	std::map<std::string, int> VertexIndices;
	std::vector<Triangle> intersectingTriangles;
public:
	Mesh(std::string name, unsigned int size);
	std::string getName();
	int getNumberOfTriangles();
	int getNumberOfVertices();
	void addTriangle(const Triangle& t);
	void addVertex(const Vertex& v);
	int findDuplicate(const Vertex& v);
	void rayTriangleIntersect(float dir[3], std::unique_ptr<Mesh>& innerMesh);
	void triangleTriangleIntersect(std::unique_ptr<Mesh>& innerMesh);
	int getLastVertex();
	void schrijf();
	Vertex* getVertexAtIndex(int index);
	void resize();
	void addVertexIndex(const std::string& s, int index);
	void writeTrianglesToFile(std::unique_ptr<Mesh>& innerMesh);
	int* getIntArrayTriangles();
	float* Mesh::getFloatArrayVertices();
};
#endif // !MESH_H