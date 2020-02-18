//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>
#include <chrono>
#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>

#include "optixTriangle.h"
#include "Mesh.h"
#include "parse_stl.h"

#include <array>
#include <iomanip>
#include <iostream>
#include <string>
#include <fstream>

#include <sutil/Camera.h>
#include <sutil/Trackball.h>

// SBT record with an appropriately aligned and sized data block
template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;
typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;
typedef SbtRecord<AnyHitData>   AnyHitSbtRecord;


void configureCamera( sutil::Camera& cam, const uint32_t width, const uint32_t height )
{
    cam.setEye( {0.0f, 0.0f, 2.0f} );
    cam.setLookat( {0.0f, 0.0f, 0.0f} );
    cam.setUp( {0.0f, 1.0f, 3.0f} );
    cam.setFovY( 45.0f );
    cam.setAspectRatio( (float)width / (float)height );
}


void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      Specify file for image output\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 512x384\n";
    exit( 1 );
}


static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
    << message << "\n";
}

void writeResultsToFile(std::vector<std::string>& result)
{
	std::vector<std::string>::iterator itr;
	std::string path = "output.csv";
	std::ofstream ofs;
	ofs.open(path, std::ofstream::out | std::ofstream::app);
	for (itr = result.begin(); itr != result.end(); ++itr)
	{
		ofs << (*itr);
	}
}

/* Console output wegschrijven naar file*/
std::vector<std::string> output;

int main( int argc, char* argv[] )
{
	std::string delimiter = "\\";
	std::string stl_file_inside;
	std::string stl_file_outside;

	std::cout << "Enter filename of inside mesh:" << std::endl;
	std::cin >> stl_file_inside;

	std::cout << "Enter filename of outside mesh:" << std::endl;
	std::cin >> stl_file_outside;

	//auto t1 = std::chrono::high_resolution_clock::now(); //start time measurement

	//Only reads STL-file in binary format!!!
	std::cout << "Reading files:" << std::endl;
	std::unique_ptr<Mesh> triangleMesh_Inside = stl::parse_stl(stl_file_inside);
	std::unique_ptr<Mesh> triangleMesh_Outside = stl::parse_stl_with_duplicate_vertices(stl_file_outside);

	size_t pos = 0;
	std::string token;
	while ((pos = stl_file_inside.find(delimiter)) != std::string::npos) {
		token = stl_file_inside.substr(0, pos);
		stl_file_inside.erase(0, pos + delimiter.length());
	}
	stl_file_inside = stl_file_inside.substr(0, stl_file_inside.find(".stl"));
	output.push_back(stl_file_inside + "-");

	pos = 0;
	while ((pos = stl_file_outside.find(delimiter)) != std::string::npos) {
		token = stl_file_outside.substr(0, pos);
		stl_file_outside.erase(0, pos + delimiter.length());
	}
	stl_file_outside = stl_file_outside.substr(0, stl_file_outside.find(".stl"));
	output.push_back(stl_file_outside + ";");

	//auto t2 = std::chrono::high_resolution_clock::now(); //stop time measurement
	//auto time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	//std::cout << "Time = " << time << " milliseconds" << std::endl;

	/*std::cout << "STL HEADER = " << triangleMesh_Inside->getName() << std::endl;
	std::cout << "# triangles = " << triangleMesh_Inside->getNumberOfTriangles() << std::endl;
	std::cout << "# vertices = " << triangleMesh_Inside->getNumberOfVertices() << std::endl;*/

	//triangleMesh_Inside.schrijf();

	/*std::cout << "STL HEADER = " << triangleMesh_Outside->getName() << std::endl;
	std::cout << "# triangles = " << triangleMesh_Outside->getNumberOfTriangles() << std::endl;
	std::cout << "# vertices = " << triangleMesh_Outside->getNumberOfVertices() << std::endl;*/

	//triangleMesh_Outside.schrijf();

	Vertex* V1 = triangleMesh_Outside->getVertexAtIndex(0);
	Vertex* V2 = triangleMesh_Outside->getVertexAtIndex(1);
	Vertex* V3 = triangleMesh_Outside->getVertexAtIndex(2);

	float xCenter = (V1->getCoordinates()[0] + V2->getCoordinates()[0] + V3->getCoordinates()[0]) / 3;
	float yCenter = (V1->getCoordinates()[1] + V2->getCoordinates()[1] + V3->getCoordinates()[1]) / 3;
	float zCenter = (V1->getCoordinates()[2] + V2->getCoordinates()[2] + V3->getCoordinates()[2]) / 3;

	float direction[3] = { xCenter, yCenter, zCenter };
	//float direction[3] = { 1.0, 1.0, 1.0 };

	//std::cout << "direction = " << direction[0] << ", " << direction[1] << ", " << direction[2] << std::endl;
	//std::cout << "AnyHit (0) or ClosestHit(1)?" << std::endl;
	int AH_CH = 0;
	//std::cin >> AH_CH;

	//**********************************************************************************************************
	//----------------------------------------------------------------------------------------------------------
	//**********************************************************************************************************

    std::string outfile;
    //int         width  = 1024;
    //int         height =  768;

    /*for( int i = 1; i < argc; ++i )
    {
        const std::string arg( argv[i] );
        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "--file" || arg == "-f" )
        {
            if( i < argc - 1 )
            {
                outfile = argv[++i];
            }
            else
            {
                printUsageAndExit( argv[0] );
            }
        }
        else if( arg.substr( 0, 6 ) == "--dim=" )
        {
            const std::string dims_arg = arg.substr( 6 );
            sutil::parseDimensions( dims_arg.c_str(), width, height );
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }*/

    try
    {
        char log[2048]; // For error reporting from OptiX creation functions


        //
        // Initialize CUDA and create OptiX context
        //
        OptixDeviceContext context = nullptr;
        {
            // Initialize CUDA
            CUDA_CHECK( cudaFree( 0 ) );

            CUcontext cuCtx = 0;  // zero means take the current context
            OPTIX_CHECK( optixInit() );
            OptixDeviceContextOptions options = {};
            options.logCallbackFunction       = &context_log_cb;
            options.logCallbackLevel          = 4;
            OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );
        }


        //
        // accel handling
        //
        OptixTraversableHandle gas_handle;
        CUdeviceptr            d_gas_output_buffer;
        {
			// Specify options for the build. We use default options for simplicity.
            OptixAccelBuildOptions accel_options = {};
            accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
            accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

			// Triangle build input: simple list of three vertices
            /*const std::vector<float3> vertices =
            { {
                  { -1.0f, -1.0f, 0.0f },
                  {  0.1f, -0.1f, 0.0f },
                  {  0.0f,  0.1f, 0.0f }
            } };*/
			const std::vector<float3> vertices = triangleMesh_Outside->getVerticesVector();

			/*// Declaring iterator to a vector 
			std::vector<float3>::iterator ptr;

			// Displaying vector elements using begin() and end() 
			std::cout << "The vector elements are : ";
			int teller = 0;
			for (ptr = vertices.begin(); ptr < vertices.end(); ptr++) { // als je hier fout krijgt, 'const' bij const std::vector<float3> vertices weg doen
				std::cout << ptr->x << "\t";
				teller++;
				if (teller%3 == 0) {
					std::cout << std::endl;
				}
			}*/

			// Allocate and copy device memory for our input triangle vertices
            const size_t vertices_size = sizeof( float3 )*vertices.size();
            CUdeviceptr d_vertices=0;
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_vertices ), vertices_size ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( d_vertices ),
                        vertices.data(),
                        vertices_size,
                        cudaMemcpyHostToDevice
                        ) );

			// Populate the build input struct with our triangle data as well as
			// information about the sizes and types of our data
            const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
            OptixBuildInput triangle_input = {};
            triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangle_input.triangleArray.numVertices   = static_cast<uint32_t>( vertices.size() );
            triangle_input.triangleArray.vertexBuffers = &d_vertices;
            triangle_input.triangleArray.flags         = triangle_input_flags;
            triangle_input.triangleArray.numSbtRecords = 1;

			// Query OptiX for the memory requirements for our GAS
            OptixAccelBufferSizes gas_buffer_sizes;
            OPTIX_CHECK( optixAccelComputeMemoryUsage( 
				context,			// The device context we are using
				&accel_options, 
				&triangle_input,	// Describes our geometry
				1,					// Number of build input
				&gas_buffer_sizes ) );

			// Allocate device memory for the scratch space buffer as well
			// as the GAS itself
            CUdeviceptr d_temp_buffer_gas;
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer_gas ), gas_buffer_sizes.tempSizeInBytes ) );

            // non-compacted output
            CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
            size_t compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(
                            &d_buffer_temp_output_gas_and_compacted_size ),
                        compactedSizeOffset + 8
                        ) );

            OptixAccelEmitDesc emitProperty = {};
            emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

			// Now build the GAS
            OPTIX_CHECK( optixAccelBuild(
                        context,
                        0,              // CUDA stream
                        &accel_options,
                        &triangle_input,
                        1,              // num build inputs
                        d_temp_buffer_gas,
                        gas_buffer_sizes.tempSizeInBytes,
                        d_buffer_temp_output_gas_and_compacted_size,
                        gas_buffer_sizes.outputSizeInBytes,
                        &gas_handle,	// Output handle to the struct
                        &emitProperty,  // emitted property list
                        1               // num emitted properties
                        ) );

			// We can now free scratch space used during the build
            CUDA_CHECK( cudaFree( (void*)d_temp_buffer_gas ) );
            CUDA_CHECK( cudaFree( (void*)d_vertices ) );

            size_t compacted_gas_size;
            CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost ) );

            if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
            {
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_gas_output_buffer ), compacted_gas_size ) );

                // use handle as input and output
                OPTIX_CHECK( optixAccelCompact( context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle ) );

                CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
            }
            else
            {
                d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
            }
        }

        //
        // Create module
        //
        OptixModule module = nullptr; // The output module

		// Pipeline options must be consistent for all modules used in a
		// single pipeline
        OptixPipelineCompileOptions pipeline_compile_options = {};
        {
            OptixModuleCompileOptions module_compile_options = {};
            module_compile_options.maxRegisterCount     = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
            module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

            pipeline_compile_options.usesMotionBlur        = false;

			// This option is important to ensure we compile code which is optimal
			// for our scene hierarchy. We use a single GAS – no instancing or
			// multi-level hierarchies
            pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;

			// Our device code uses 3 payload registers (r,g,b output value)
            pipeline_compile_options.numPayloadValues      = 3;
            pipeline_compile_options.numAttributeValues    = 3;
            pipeline_compile_options.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;

			// This is the name of the param struct variable in our device code
            pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

            const std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, "optixTriangle.cu" );
            size_t sizeof_log = sizeof( log );

            OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
                        context,
                        &module_compile_options,
                        &pipeline_compile_options,
                        ptx.c_str(),
                        ptx.size(),
                        log,
                        &sizeof_log,
                        &module
                        ) );
        }

        //
        // Create program groups
        //
        OptixProgramGroup raygen_prog_group   = nullptr;
        OptixProgramGroup miss_prog_group     = nullptr;
		OptixProgramGroup anyhit_prog_group = nullptr;
		OptixProgramGroup hitgroup_prog_group = nullptr;
        {
            OptixProgramGroupOptions program_group_options   = {}; // Initialize to zeros

            OptixProgramGroupDesc raygen_prog_group_desc    = {}; //
            raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_prog_group_desc.raygen.module            = module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
            size_t sizeof_log = sizeof( log );
            OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        context,
                        &raygen_prog_group_desc,
                        1,   // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &raygen_prog_group
                        ) );

            OptixProgramGroupDesc miss_prog_group_desc  = {};
            miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
            miss_prog_group_desc.miss.module            = module;
            miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
            sizeof_log = sizeof( log );
            OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        context,
                        &miss_prog_group_desc,
                        1,   // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &miss_prog_group
                        ) );
			if (AH_CH == 0) { //ANYHIT
				OptixProgramGroupDesc anyhit_prog_group_desc = {};
				anyhit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
				anyhit_prog_group_desc.hitgroup.moduleAH = module;
				anyhit_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
				sizeof_log = sizeof(log);
				OPTIX_CHECK_LOG(optixProgramGroupCreate(
					context,
					&anyhit_prog_group_desc,
					1,   // num program groups
					&program_group_options,
					log,
					&sizeof_log,
					&anyhit_prog_group
				));
			}
			else { //CLOSESTHIT
				OptixProgramGroupDesc hitgroup_prog_group_desc = {};
			hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
			hitgroup_prog_group_desc.hitgroup.moduleCH            = module;
			hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
			sizeof_log = sizeof( log );
			OPTIX_CHECK_LOG( optixProgramGroupCreate(
						context,
						&hitgroup_prog_group_desc,
						1,   // num program groups
						&program_group_options,
						log,
						&sizeof_log,
						&hitgroup_prog_group
						) );
			}
        }

        //
        // Link pipeline
        //
        OptixPipeline pipeline = nullptr;
        {
			OptixProgramGroup program_groups[3];
			if (AH_CH == 0) { //ANYHIT
				program_groups[0] = raygen_prog_group;
				program_groups[1] = miss_prog_group;
				program_groups[2] = anyhit_prog_group;
			}
			else { //CLOSESTHIT
				program_groups[0] = raygen_prog_group;
				program_groups[1] = miss_prog_group;
				program_groups[2] = hitgroup_prog_group;
			}
            OptixPipelineLinkOptions pipeline_link_options = {};
            pipeline_link_options.maxTraceDepth          = 5;
            pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
            pipeline_link_options.overrideUsesMotionBlur = false;
            size_t sizeof_log = sizeof( log );
            OPTIX_CHECK_LOG( optixPipelineCreate(
                        context,
                        &pipeline_compile_options,
                        &pipeline_link_options,
                        program_groups,
                        sizeof( program_groups ) / sizeof( program_groups[0] ),
                        log,
                        &sizeof_log,
                        &pipeline
                        ) );
        }

        //
        // Set up shader binding table
        //

		// The shader binding table struct we will populate
        OptixShaderBindingTable sbt = {};
        {
            CUdeviceptr  raygen_record;
            const size_t raygen_record_size = sizeof( RayGenSbtRecord );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &raygen_record ), raygen_record_size ) );
            sutil::Camera cam;
            //configureCamera( cam, width, height );
            RayGenSbtRecord rg_sbt;
            rg_sbt.data ={};
            //rg_sbt.data.cam_eye = cam.eye();
			rg_sbt.data.origins = triangleMesh_Inside->getFloat3ArrayVertices();
			rg_sbt.data.direction = make_float3(direction[0], direction[1], direction[2]);
            //cam.UVWFrame( rg_sbt.data.camera_u, rg_sbt.data.camera_v, rg_sbt.data.camera_w );
            OPTIX_CHECK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( raygen_record ),
                        &rg_sbt,
                        raygen_record_size,
                        cudaMemcpyHostToDevice
                        ) );

			// Allocate the miss record on the device
            CUdeviceptr miss_record;
            size_t      miss_record_size = sizeof( MissSbtRecord );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &miss_record ), miss_record_size ) );

			// Populate host-side copy of the record with header and data
            MissSbtRecord ms_sbt;
            ms_sbt.data = { 0.3f, 0.1f, 0.2f };
            OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt ) );

			// Now copy our host record to the device
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( miss_record ),
                        &ms_sbt,
                        miss_record_size,
                        cudaMemcpyHostToDevice
                        ) );

            CUdeviceptr anyhit_record;
			CUdeviceptr hitgroup_record;

			if (AH_CH == 0) { //ANYHIT
				size_t      anyhit_record_size = sizeof(AnyHitSbtRecord);
				CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&anyhit_record), anyhit_record_size));
				AnyHitSbtRecord ah_sbt;
				ah_sbt.data = { 1.5f };
				OPTIX_CHECK(optixSbtRecordPackHeader(anyhit_prog_group, &ah_sbt));
				CUDA_CHECK(cudaMemcpy(
					reinterpret_cast<void*>(anyhit_record),
					&ah_sbt,
					anyhit_record_size,
					cudaMemcpyHostToDevice
				));
			}
			else { //CLOSESTHIT
				size_t      hitgroup_record_size = sizeof(HitGroupSbtRecord);
				CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size));
				HitGroupSbtRecord hg_sbt;
				hg_sbt.data = { 1.5f };
				OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
				CUDA_CHECK(cudaMemcpy(
					reinterpret_cast<void*>(hitgroup_record),
					&hg_sbt,
					hitgroup_record_size,
					cudaMemcpyHostToDevice
				));
			}

			// Finally we specify how many records and how they are packed in memory
            sbt.raygenRecord                = raygen_record;
            sbt.missRecordBase              = miss_record;
            sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
            sbt.missRecordCount             = 1;
            
			if (AH_CH == 0) { //ANYHIT
				sbt.hitgroupRecordBase = anyhit_record;
			}
			else { //CLOSESTHIT
				sbt.hitgroupRecordBase = hitgroup_record;
			}

            sbt.hitgroupRecordStrideInBytes = sizeof( AnyHitSbtRecord );
            sbt.hitgroupRecordCount         = 1;
        }

        sutil::CUDAOutputBuffer<uchar1> output_buffer( sutil::CUDAOutputBufferType::CUDA_DEVICE, triangleMesh_Inside->getNumberOfVertices(), 1 );

        //
        // launch
        //
        {
            CUstream stream;
            CUDA_CHECK( cudaStreamCreate( &stream ) );

            Params params;
            params.image        = output_buffer.map();
            params.image_width  = triangleMesh_Inside->getNumberOfVertices();
            params.image_height = 1;
            //params.origin_x     = width / 2;
            //params.origin_y     = height / 2;
            params.handle       = gas_handle;

            CUdeviceptr d_param;
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( Params ) ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( d_param ),
                        &params, sizeof( params ),
                        cudaMemcpyHostToDevice
                        ) );

			//std::cout << "--- Calculating ---" << std::endl;
			auto start = std::chrono::high_resolution_clock::now(); //start time measurement
			
			//printf("width: %d, height: %d", width, height);
            OPTIX_CHECK( optixLaunch( pipeline, stream, d_param, sizeof( Params ), &sbt, triangleMesh_Inside->getNumberOfVertices(), 1, /*depth=*/1 ) );

            CUDA_SYNC_CHECK();
			
            output_buffer.unmap();

			auto end = std::chrono::high_resolution_clock::now(); //stop time measurement

			bool inside = true;
			uchar1* hopeloos = output_buffer.getHostPointer();
			int teller = 0;
			for (int j = 0; j < triangleMesh_Inside->getNumberOfVertices(); j++) {
				//printf("output: %d\n", hopeloos[j].x);
				teller += hopeloos[j].x;
				if (hopeloos[j].x % 2 == 0) {
					inside = false;
					break;
				}
			}

			
			//std::cout << "--- End Calculating ---" << std::endl;
			auto calculatingDuration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
			std::string result;
			if (inside) { result = "INSIDE"; }
			else { result = "OUTSIDE"; }
			output.push_back(std::to_string((float)calculatingDuration / 1000) + ";" + result+ "\n");

			//std::cout << "\t\t\tTime Calculating = " << calculatingDuration << " microseconds" << std::endl;

			//std::cout << "teller: " << teller << std::endl;
			//std::cout << "inside: " << inside << std::endl;
        }
			
        //
        // Display results
        //
        /*{
            sutil::ImageBuffer buffer;
            buffer.data         = output_buffer.getHostPointer();
            buffer.width        = width;
            buffer.height       = height;
            buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
			if (outfile.empty())
                sutil::displayBufferWindow( argv[0], buffer );
            else
                sutil::displayBufferFile( outfile.c_str(), buffer, false );
        }*/

        //
        // Cleanup
        //
        {
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.raygenRecord       ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.missRecordBase     ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.hitgroupRecordBase ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_gas_output_buffer    ) ) );

            OPTIX_CHECK( optixPipelineDestroy( pipeline ) );
			if (AH_CH == 0) { //ANYHIT
				OPTIX_CHECK(optixProgramGroupDestroy(anyhit_prog_group));
			}
			else { //CLOSESTHIT
				OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_group));
			}		
            OPTIX_CHECK( optixProgramGroupDestroy( miss_prog_group ) );
            OPTIX_CHECK( optixProgramGroupDestroy( raygen_prog_group ) );
            OPTIX_CHECK( optixModuleDestroy( module ) );

            OPTIX_CHECK( optixDeviceContextDestroy( context ) );
        }

		writeResultsToFile(output);
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
	std::cout << "Press Enter to quit program!" << std::endl;
	std::cin.get();
	std::cin.get();

    return 0;
}
