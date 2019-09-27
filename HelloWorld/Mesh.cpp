#include <vector>
#include <string>
#include <iostream>
#include <iterator>
#include <map>
#include <fstream>

#include "Mesh.h"
#include "RayTriangleIntersect.h"
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

	/*std::vector<Vertex>::iterator it = std::find(vertices.begin(), vertices.end(), v);
	if (it != vertices.end())//returned index of last value if nothing found!
	{
		return std::distance(vertices.begin(), it);
	}
	else
	{
		return -1;
	}*/
}
void Mesh::rayTriangleIntersect(float dir[3], std::unique_ptr<Mesh>& innerMesh)
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
		//std::cout << "aantal intersecties = " << numberOfIntersections << std::endl;
		if (numberOfIntersections % 2 == 0)
		{
			inside = false;
		}
	}
	delete t; delete u; delete v;
	if (inside) { std::cout << "INSIDE" << std::endl; }
	else { std::cout << "OUTSIDE" << std::endl; }
}
void Mesh::triangleTriangleIntersect(std::unique_ptr<Mesh>& innerMesh)
{
	float* vert1_1;
	float* vert1_2;
	float* vert1_3;
	float* vert2_1;
	float* vert2_2;
	float* vert2_3;
	Triangle* t1;
	Triangle* t2;
	std::vector<Vertex>* innerVertices = &(innerMesh->vertices);

	bool inside = true;

	std::cout << "start berekening" << std::endl;

	for (int j = 0; j < innerMesh->getNumberOfTriangles(); j++)
	{
		t1 = &(innerMesh->triangles.at(j));
		vert1_1 = innerVertices->at(t1->getIndexOfVertexInMesh(0)).getCoordinates();
		vert1_2 = innerVertices->at(t1->getIndexOfVertexInMesh(1)).getCoordinates();
		vert1_3 = innerVertices->at(t1->getIndexOfVertexInMesh(2)).getCoordinates();

		int numberOfIntersections = 0;

		for (int i = 0; i < triangles.size(); i++)
		{
			t2 = &(triangles.at(i));
			vert2_1 = vertices.at(t2->getIndexOfVertexInMesh(0)).getCoordinates();
			vert2_2 = vertices.at(t2->getIndexOfVertexInMesh(1)).getCoordinates();
			vert2_3 = vertices.at(t2->getIndexOfVertexInMesh(2)).getCoordinates();
			if (Intersection::NoDivTriTriIsect(vert1_1, vert1_2, vert1_3, vert2_1, vert2_2, vert2_3) == 1)
			{
				//list printed with intersecting triangles
				innerMesh->intersectingTriangles.push_back(innerMesh->triangles.at(j));
				intersectingTriangles.push_back(triangles.at(i));

				numberOfIntersections++;
			}
		}
		std::cout << "aantal intersecties = " << numberOfIntersections << std::endl;
		if (numberOfIntersections != 0)
		{
			inside = false;
		}
	}
	if (inside) { std::cout << "SNIJDEN NIET" << std::endl; }
	else { std::cout << "SNIJDEN WEL" << std::endl; }
	writeTrianglesToFile(innerMesh);
}
int Mesh::getLastVertex()
{
	return (vertices.size() - 1);
}
void Mesh::schrijf()
{
	for (int i = 0; i < triangles.size(); i++)
	{
		std::cout << "driehoek " << i << std::endl;
		for (int j = 0; j < 3; j++)
		{
			std::cout << "vertex " << j << ": index = " << triangles.at(i).getIndexOfVertexInMesh(j) << std::endl;
			Vertex v = vertices.at(triangles.at(i).getIndexOfVertexInMesh(j));
			v.schrijf();
		}
		std::cout << std::endl;
	}
	std::map<std::string, int>::iterator itr;
	for (itr = VertexIndices.begin(); itr != VertexIndices.end(); itr++)
	{
		std::cout << "key: " << itr->first  // string (key)
			<< ':'
			<< "value: " << itr->second   // string's value 
			<< std::endl;
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
void Mesh::writeTrianglesToFile(std::unique_ptr<Mesh>& innerMesh)
{
	std::vector<Triangle>::iterator itr;
	std::ofstream ofs("IntersectingTriangles.stl");
	ofs << "solid IntersectingTriangles" << std::endl;
	for (itr = intersectingTriangles.begin(); itr != intersectingTriangles.end(); ++itr)
	{
		ofs << "  facet normal  0.0  0.0  0.0" << std::endl;
		ofs << "    outer loop" << std::endl;
		float* vert;
		for (int j = 0; j < 3; j++)
		{
			vert = vertices.at(itr->getIndexOfVertexInMesh(j)).getCoordinates();
			ofs << "      vertex  " << vert[0] << "  "
				<< vert[1] << "  "
				<< vert[2] << std::endl;
		}
		ofs << "    endloop" << std::endl;
		ofs << "  endfacet" << std::endl;
	}
	for (itr = innerMesh->intersectingTriangles.begin(); itr != innerMesh->intersectingTriangles.end(); ++itr)
	{
		ofs << "  facet normal  0.0  0.0  0.0" << std::endl;
		ofs << "    outer loop" << std::endl;
		float* vert;
		for (int j = 0; j < 3; j++)
		{
			vert = innerMesh->vertices.at(itr->getIndexOfVertexInMesh(j)).getCoordinates();
			ofs << "      vertex  " << vert[0] << "  "
				<< vert[1] << "  "
				<< vert[2] << std::endl;
		}
		ofs << "    endloop" << std::endl;
		ofs << "  endfacet" << std::endl;
	}
	ofs << "endsolid vcg" << std::endl;
}