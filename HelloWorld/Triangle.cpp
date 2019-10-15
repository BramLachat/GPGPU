#include <vector>

#include "Triangle.h"

Triangle::Triangle()
{
	vertexIndexes[0] = -1;vertexIndexes[1] = -1;vertexIndexes[2] = -1;
}
Triangle::Triangle(const Triangle& t)
{
	vertexIndexes[0] = t.vertexIndexes[0];
	vertexIndexes[1] = t.vertexIndexes[1];
	vertexIndexes[2] = t.vertexIndexes[2];
}
void Triangle::addVertexIndex(int index, int vertexNumber)
{
	vertexIndexes[vertexNumber] = index;
}
int Triangle::getIndexOfVertexInMesh(int index)
{
	return vertexIndexes[index];
}
int3 Triangle::getIndexOfVerticesInMesh()
{
	return make_int3(vertexIndexes[0], vertexIndexes[1], vertexIndexes[2]);
}
void Triangle::clear()
{
	vertexIndexes[0] = -1;vertexIndexes[1] = -1;vertexIndexes[2] = -1;
}