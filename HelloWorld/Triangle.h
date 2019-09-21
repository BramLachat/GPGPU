#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <vector>

#include "Vertex.h"

class Triangle
{
private:
	std::vector<int> vertexIndexes;
public:
	Triangle();
	Triangle(const Triangle& t);
	void addVertexIndex(const int& index);
	int getIndexOfVertexInMesh(const int& index);
	void clear();
};
#endif