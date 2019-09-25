#include <vector>
#include <string>
#include <iostream>

#include "Mesh.h"
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
}
void Mesh::findIntersections(float dir[3], std::unique_ptr<Mesh>& innerMesh)
{
	//std::unique_ptr<float[]> orig = std::make_unique<float[]>(3); //smart pointer
	//std::vector<float> orig;
	Vertex* V1_1;
	Vertex* V1_2;
	Vertex* V1_3;
	float* vert1_1;
	float* vert1_2;
	float* vert1_3;
	Vertex* V2_1;
	Vertex* V2_2;
	Vertex* V2_3;
	float* vert2_1;
	float* vert2_2;
	float* vert2_3;

	bool inside = true;

	std::cout << "start berekening" << std::endl;

	for(int j = 0 ; j < innerMesh->getNumberOfTriangles() ; j++)
	{
		V1_1 = &(innerMesh->vertices.at(innerMesh->triangles.at(j).getIndexOfVertexInMesh(0)));
		V1_2 = &(innerMesh->vertices.at(innerMesh->triangles.at(j).getIndexOfVertexInMesh(1)));
		V1_3 = &(innerMesh->vertices.at(innerMesh->triangles.at(j).getIndexOfVertexInMesh(2)));
		vert1_1 = V1_1->getCoordinates();
		vert1_2 = V1_2->getCoordinates();
		vert1_3 = V1_3->getCoordinates();

		int numberOfIntersections = 0;

		for (int i = 0; i < triangles.size(); i++)
		{
			V2_1 = &(vertices.at(triangles.at(i).getIndexOfVertexInMesh(0)));
			V2_2 = &(vertices.at(triangles.at(i).getIndexOfVertexInMesh(1)));
			V2_3 = &(vertices.at(triangles.at(i).getIndexOfVertexInMesh(2)));
			vert2_1 = V2_1->getCoordinates();
			vert2_2 = V2_2->getCoordinates();
			vert2_3 = V2_3->getCoordinates();
			if (Intersection::NoDivTriTriIsect(vert1_1, vert1_2, vert1_3, vert2_1, vert2_2, vert2_3) == 1)
			{
				numberOfIntersections++;
			}
		}
		std::cout << "aantal intersecties = " << numberOfIntersections << std::endl;
		if (numberOfIntersections != 0)
		{
			inside = false;
		}
	}
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