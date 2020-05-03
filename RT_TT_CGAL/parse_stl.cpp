#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <streambuf>
#include <memory>

#include "parse_stl.h"
#include "Mesh.h"

namespace stl {

	std::ostream& operator<<(std::ostream& out, const point p) {
		out << "(" << p.x << ", " << p.y << ", " << p.z << ")" << std::endl;
		return out;
	}

	std::ostream& operator<<(std::ostream& out, const triangle& t) {
		out << "---- TRIANGLE ----" << std::endl;
		out << t.normal << std::endl;
		out << t.v1 << std::endl;
		out << t.v2 << std::endl;
		out << t.v3 << std::endl;
		return out;
	}

	float parse_float(std::ifstream& s) {
		char f_buf[sizeof(float)];
		s.read(f_buf, 4);
		float* fptr = (float*)f_buf;
		return *fptr;
	}

	Point parse_point(std::ifstream& s) {
		float x = parse_float(s);
		float y = parse_float(s);
		float z = parse_float(s);
		Point v(x, y, z);
		return v;
	}

	std::unique_ptr<Mesh> parse_stl(const std::string& stl_path) {
		std::ifstream stl_file(stl_path.c_str(), std::ios::in | std::ios::binary);
		if (!stl_file) {
			std::cout << "ERROR: COULD NOT READ FILE" << std::endl;
			assert(false);
		}

		char header_info[80] = "";
		char n_triangles[4];
		stl_file.read(header_info, 80);
		stl_file.read(n_triangles, 4);
		std::string h(header_info);
		unsigned int* num_triangles = (unsigned int*)n_triangles;
		std::unique_ptr<Mesh> mesh = std::make_unique<Mesh>(h, *num_triangles);
		Point v;
		for (unsigned int i = 0; i < *num_triangles; i++) {
			v = parse_point(stl_file); //normalvector --> wordt niet gebruikt!
			int duplicateVertexIndex;
			int triangleVertices[3];
			for (int i = 0; i < 3; i++)
			{
				v = parse_point(stl_file);
				duplicateVertexIndex = mesh->findDuplicate(v);//to string methode niet 2 keer oproepen
				mesh->addVertexIndex(std::to_string(v[0]) + std::to_string(v[1]) + std::to_string(v[2]), mesh->getNumberOfVertices());//index niet 2 keer oproepen
				if (duplicateVertexIndex == -1)
				{
					mesh->addVertex(v);
					triangleVertices[i] = mesh->getLastVertex();
				}
				else
				{
					triangleVertices[i] = duplicateVertexIndex;
				}
			}
			mesh->addTriangle(Triangle(*(mesh->getVertexAtIndex(triangleVertices[0])), *(mesh->getVertexAtIndex(triangleVertices[1])), *(mesh->getVertexAtIndex(triangleVertices[2]))));
			char dummy[2];
			stl_file.read(dummy, 2);
		}
		mesh->resize();
		return mesh;
	}

}