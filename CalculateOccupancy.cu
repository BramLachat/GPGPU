int numBlocks;        // Occupancy in terms of active blocks
	int blockSize = 128;//meegeven volgens welke configuratie ge uw kernel zou launchen

	// These variables are used to convert occupancy to warps
	int device;
	cudaDeviceProp prop;
	int activeWarps;
	int maxWarps;

	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);

	cudaOccupancyMaxActiveBlocksPerMultiprocessor(
		&numBlocks,
		Intersection::intersect_triangleGPU,
		blockSize,
		0);

	activeWarps = numBlocks * blockSize / prop.warpSize;
	maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

	std::cout << "Occupancy: " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;