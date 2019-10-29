/*
 *Deze werkwijze is nog een stukje sneller. Er zijn enkel nog paar probleempjes:
 *1) totaal aantal intersecties bleek niet te kloppen met for-loop versie (thread per origin)
 *2) je moet nog eens apart aantal intersecties van dezelfde origins die apart berekend zijn gaan optellen
 *3) de vertices worden nog niet naar een outsideVertices geschreven
 */
__global__ void intersect_triangleGPU(float3* origins, float dir[3],
		int3* triangles, float3* vertices, int numberOfOrigins, int numberOfTriangles, int* intersectionsPerOrigin, float3* outsideVertices)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		int tid2 = tid % numberOfOrigins;
		if (tid < numberOfOrigins*2)
		{
			float orig[3] = { origins[tid2].x, origins[tid2].y, origins[tid2].z };
			int i = 0;
			int end = 0;
			if (blockIdx.x % 2 == 0) {
				i = 0;
				end = numberOfTriangles / 2;
				//printf("%d --> %d = %d, %d\n", tid, tid2, i, end);
			}
			else {
				i = numberOfTriangles / 2;
				end = numberOfTriangles;
				//printf("%d --> %d = %d, %d\n", tid, tid2, i, end);
			}
			int numberOfIntersections = 0;
			for (i; i < end; i++)
			{
				float vert0[3] = { vertices[triangles[i].x].x, vertices[triangles[i].x].y, vertices[triangles[i].x].z };
				float vert1[3] = { vertices[triangles[i].y].x, vertices[triangles[i].y].y, vertices[triangles[i].y].z };
				float vert2[3] = { vertices[triangles[i].z].x, vertices[triangles[i].z].y, vertices[triangles[i].z].z };
				float t, u, v;
				if (intersect_triangle3(orig, dir, vert0, vert1, vert2, &t, &u, &v) == 1)
				{
					//printf("1 \n");
					numberOfIntersections++;
				}
			}
			//printf("numberOfIntersections = %d\n", numberOfIntersections);
			intersectionsPerOrigin[tid] = numberOfIntersections;
			/*if (numberOfIntersections % 2 == 0)
			{
				outsideVertices[tid].x = orig[0];
				outsideVertices[tid].y = orig[1];
				outsideVertices[tid].z = orig[2];
			}*/
		}
	}