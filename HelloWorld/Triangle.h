#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <vector>

#include "Vertex.h"

class Triangle
{
private:
	int vertexIndexes[3];
public:
	Triangle();
	Triangle(const Triangle& t);
	void addVertexIndex(int index, int vertexNumber);
	int getIndexOfVertexInMesh(int index);
	void clear();
};
#endif