#pragma once
#ifndef MESH_H
#define MESH_H

#include <string>
#include <map>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>

typedef CGAL::Simple_cartesian<double> K;

typedef K::FT FT;
typedef K::Ray_3 Ray;
typedef K::Line_3 Line;
typedef K::Point_3 Point;
typedef K::Triangle_3 Triangle;

class Mesh
{
private:
	std::string name;
	std::vector<Triangle> triangles;
	std::vector<Point> vertices;
	std::map<std::string, int> VertexIndices;
public:
	Mesh(std::string name, unsigned int size);
	std::string getName();
	int getNumberOfTriangles();
	int getNumberOfVertices();
	void addTriangle(const Triangle& t);
	void addVertex(const Point& v);
	int findDuplicate(const Point& v);
	void rayTriangleIntersectOpenMP(float dir[3], std::unique_ptr<Mesh>& innerMesh);
	void rayTriangleIntersect(float dir[3], std::unique_ptr<Mesh>& innerMesh);
	void triangleTriangleIntersect(std::unique_ptr<Mesh>& innerMesh);
	int getLastVertex();
	void schrijf();
	Point* getVertexAtIndex(int index);
	void resize();
	void addVertexIndex(const std::string& s, int index);
	void writeTrianglesToFile(std::unique_ptr<std::vector<Triangle>>& triangles, std::vector<Point>* vertices, std::string fileName);
	void writeVerticesToFile(std::unique_ptr<std::vector<Point>>& vertices, std::string fileName);
	std::list<Point> Mesh::getPointList();
	std::list<Triangle> Mesh::getTriangleList();
};
#endif // !MESH_H