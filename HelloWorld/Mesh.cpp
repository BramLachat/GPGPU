#include <vector>
#include <string>
#include <iostream>

#include "Mesh.h"
#include "RayTriangleIntersect.h"


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
	for(std::size_t i = 0 ; i < vertices.size() ; i++)
	{
		if (v.isDuplicate(vertices.at(i)))
		{
			return (int)i;
		}
	}
	return -1;
}
void Mesh::findIntersections(float dir[3], std::unique_ptr<Mesh>& innerMesh)
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

	Vertex* innerVertex;

	bool inside = true;

	std::cout << "start berekening" << std::endl;

	for(int j = 0 ; j < innerMesh->getNumberOfVertices() ; j++)
	{
		innerVertex = &(innerMesh->vertices.at(j));
		orig = innerVertex->getCoordinates();
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
			if (Intersection::intersect_triangle3(orig, dir, vert1, vert2, vert3, t, u, v) == 1)
			{
				numberOfIntersections++;
			}
		}
		std::cout << "aantal intersecties = " << numberOfIntersections << std::endl;
		if (numberOfIntersections % 2 == 0)
		{
			inside = false;
		}
	}
	delete t; delete u; delete v;
	if (inside) { std::cout << "INSIDE" << std::endl; }
	else { std::cout << "OUTSIDE" << std::endl; }
}
int Mesh::getLastVertex()
{
	return (vertices.size() - 1);
}
void Mesh::schrijf()
{
	for (int i = 0; i < triangles.size(); i++)
	{
		for (int j = 0; j < 3; j++)
		{
			Vertex v = vertices.at(triangles.at(i).getIndexOfVertexInMesh(j));
			v.schrijf();
		}
		std::cout << std::endl;
	}
}
/*Vertex Mesh::getVertexByIndex(int index)
{
	return vertices.at(index);
}*/
void Mesh::resize()
{
	vertices.shrink_to_fit();
}