#ifndef VERTEX_H
#define VERTEX_H

#include <string>
#include <cuda_runtime.h>

class Vertex
{
private:
	float point[3];
public:
	static int copies;
	Vertex();
	Vertex(float x, float y, float z);
	Vertex(const Vertex& v);
	Vertex& operator=(const Vertex& v);
	bool operator==(const Vertex& v);
	bool isDuplicate(const Vertex& v) const;
	float* getCoordinates();
	float getCoordinate(int i);
	void schrijf();
	std::string toString() const;
	float3 getCoordinatesFloat3();
	//~Vertex();
};
#endif

