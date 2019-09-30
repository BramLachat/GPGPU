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
	__device__ __host__ float* getCoordinates();
	void schrijf();
	std::string toString() const;
	//~Vertex();
};
#endif

