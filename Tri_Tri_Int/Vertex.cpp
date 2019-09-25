#include <memory>
#include <iostream>
#include <string>

#include "Vertex.h"

int Vertex::copies = 0;
Vertex::Vertex()
{
	point[0] = 0; point[1] = 0; point[2] = 0;
}
Vertex::Vertex(float x, float y, float z)
{
	//point = new float[3];
	point[0] = x;
	point[1] = y;
	point[2] = z;
}
Vertex::Vertex(const Vertex& v) //copyconstructor
{
	//point = new float[3];
//	memcpy(point, v.point, 3);
	point[0] = v.point[0];
	point[1] = v.point[1];
	point[2] = v.point[2];
	//std::cout << "Copied" << std::endl;
	Vertex::copies++;
}
Vertex& Vertex::operator=(const Vertex& v) //copy-assignment
{
	if (this != &v) {
		//if (point != nullptr) delete[] point;
		//point = new float[3];
		point[0] = v.point[0];
		point[1] = v.point[1];
		point[2] = v.point[2];
		return *this;
	}
	else {
		return *this;
	}
}
bool Vertex::isDuplicate(const Vertex& v) const
{
	if (v.point[0] == point[0] && v.point[1] == point[1] && v.point[2] == point[2])
	{
		return true;
	}
	else
	{
		return false;
	}
}
float* Vertex::getCoordinates()
{
	return point;
}
/*Vertex::~Vertex()
{
	delete[] point;
}*/
void Vertex::schrijf()
{
	std::cout << point[0] << ", " << point[1] << ", " << point[2] << std::endl;
}
std::string Vertex::toString() const
{
	return std::to_string(point[0]) + std::to_string(point[1]) + std::to_string(point[2]);
}