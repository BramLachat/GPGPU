/*#include "Mesh.h"
#include <vector>
#include <string>

class Mesh
{
private:
	std::string name;
	std::vector<Triangle> triangles;
	std::vector<Vertex> vertices;
public:
	Mesh(std::string name) : name(name) {}
	void addTriangle(Triangle t) 
	{
		triangles.push_back(t);
		delete triangles;
	}
	void addVertex(Vertex v)
	{
		vertices.push_back(v);
	}
};
*/