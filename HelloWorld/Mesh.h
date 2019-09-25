#ifndef MESH_H
#define MESH_H

#include <string>

#include "Vertex.h"
#include "Triangle.h"

class Mesh
{
private:
	std::string name;
	std::vector<Triangle> triangles;
	std::vector<Vertex> vertices;
public:
	Mesh(std::string name, unsigned int size);
	std::string getName();
	int getNumberOfTriangles();
	int getNumberOfVertices();
	void addTriangle(const Triangle& t);
	void addVertex(const Vertex& v);
	int findDuplicate(const Vertex& v);
	void findIntersections(float dir[3], std::unique_ptr<Mesh>& innerMesh);
	int getLastVertex();
	void schrijf();
	Vertex* getVertexAtIndex(int index);
	void resize();
};
#endif // !MESH_H