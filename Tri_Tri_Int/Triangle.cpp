#include <vector>

#include "Triangle.h"

Triangle::Triangle()
{
	vertexIndexes.reserve(3);
}
Triangle::Triangle(const Triangle& t)
{
	vertexIndexes.reserve(3);
	for (int i : t.vertexIndexes) 
	{
		vertexIndexes.push_back(i);
	}
}
void Triangle::addVertexIndex(const int& index)
{
	vertexIndexes.push_back(index);
}
int Triangle::getIndexOfVertexInMesh(const int& index)
{
	return vertexIndexes.at(index);
}
void Triangle::clear()
{
	vertexIndexes.clear();
}