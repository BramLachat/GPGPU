#ifndef VERTEX_H
#define VERTEX_H

class Vertex
{
private:
	float* point;
public:
	static int copies;
	Vertex();
	Vertex(float x, float y, float z);
	Vertex(const Vertex& v);
	Vertex& operator=(const Vertex& v);
	bool operator==(const Vertex& v);
	bool isDuplicate(const Vertex& v) const;
	float* getCoordinates();
	void schrijf();
	~Vertex();
};
#endif

