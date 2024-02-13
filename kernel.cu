#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <map>
#include <functional>
#include <string>
#include <stdio.h>
#include <malloc.h>
#include <cstdlib>
#include <time.h>

#include "cpucounters.h"

extern "C" {
#include "vcmalloc.h"
}

#include "mimalloc.h"

using namespace pcm;

void log_csv(
	const char* operation,
	const char* x,
	const char* y,
	const char* z,
	const char* xtype,
	const char* ytype,
	const char* ztype,
	const char* xunit,
	const char* yunit,
	const char* zunit
) {
	// Print the results
	printf(
		"%s\n"
		"%s (%s): %s\n"
		"%s (%s): %s\n"
		"%s (%s): %s\n",
		operation,
		ztype, zunit, z,
		xtype, xunit, x,
		ytype, yunit, y
	);

	// Save results to a CSV file
	FILE* csv_file = fopen("results.csv", "a");

	// Check if the file is empty; if it is, add headings
	fseek(csv_file, 0, SEEK_END);
	if (ftell(csv_file) == 0) {
		fprintf(csv_file, "operation, x, y, z, xtype, ytype, ztype, xunit, yunit, zunit\n");
	}

	if (csv_file) {
		fprintf(csv_file, "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n",
			operation, x, y, z, xtype, ytype, ztype, xunit, yunit, zunit);
		fclose(csv_file);
	}
}

// Define functors for different benchmark types
struct vcm {
	int operator()(int argc, char* argv[]) {

		const char* allocatorname = "vcm";

		PCM* m;
		m = PCM::getInstance();
		m->cleanup();
		PCM::ErrorCode returnResult = m->program();
		if (returnResult != PCM::Success) {
			std::cerr << "PCM couldn't start" << std::endl;
			std::cerr << "Error code: " << returnResult << std::endl;
			exit(1);
		}

		cudaError_t cudaStatus;
		clock_t start, end;
		SystemCounterState before_sstate, after_sstate;

		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);

		size_t total_size = prop.totalGlobalMem;
		size_t N = std::strtoull(argv[2], nullptr, 10);
		size_t M = total_size / (N * sizeof(double));

		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		}

		double* dev_buffer;

		cudaStatus = cudaMalloc((void**)&dev_buffer, total_size);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		hca_init(total_size + N * sizeof(double*), N + 1, 1);

		double** buffer = (double**)vca_malloc(N * sizeof(double*));
		for (int i = 0; i < N; ++i)
			buffer[i] = (double*)vca_malloc(M * sizeof(double));

		for (size_t i = 0; i < N; i++)
			for (size_t j = 0; j < M; j++)
				buffer[i][j] = i * M + j;

		start = clock();
		before_sstate = getSystemCounterState();

		for (size_t i = 0; i < N; i++)
		{
			cudaStatus = cudaMemcpy(dev_buffer + (i * M), buffer[i], M * sizeof(double), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
			}
		}

		after_sstate = getSystemCounterState();
		end = clock();

		double time = (double)(end - start) / CLOCKS_PER_SEC;
		double cpu_energy = getConsumedJoules(before_sstate, after_sstate);
		uint64 L3CacheMisses = getL3CacheMisses(before_sstate, after_sstate);
		uint64 L2CacheMisses = getL2CacheMisses(before_sstate, after_sstate);

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(time).c_str(), allocatorname,
			"N", "Time", "Memory Allocator", "", "Seconds", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(cpu_energy).c_str(), allocatorname,
			"N", "Energy", "Memory Allocator", "", "Joules", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(L3CacheMisses).c_str(), allocatorname,
			"N", "L3 Cache Misses", "Memory Allocator", "", "", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(L2CacheMisses).c_str(), allocatorname,
			"N", "L2 Cache Misses", "Memory Allocator", "", "", "");


		start = clock();
		before_sstate = getSystemCounterState();

		for (size_t i = 0; i < N; i++)
		{
			cudaStatus = cudaMemcpy(buffer[i], dev_buffer + (i * M), M * sizeof(double), cudaMemcpyDeviceToHost);

			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
			}
		}

		after_sstate = getSystemCounterState();
		end = clock();

		time = (double)(end - start) / CLOCKS_PER_SEC;
		cpu_energy = getConsumedJoules(before_sstate, after_sstate);
		L3CacheMisses = getL3CacheMisses(before_sstate, after_sstate);
		L2CacheMisses = getL2CacheMisses(before_sstate, after_sstate);

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(time).c_str(), allocatorname,
			"N", "Time", "Memory Allocator", "", "Seconds", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(cpu_energy).c_str(), allocatorname,
			"N", "Energy", "Memory Allocator", "", "Joules", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(L3CacheMisses).c_str(), allocatorname,
			"N", "L3 Cache Misses", "Memory Allocator", "", "", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(L2CacheMisses).c_str(), allocatorname,
			"N", "L2 Cache Misses", "Memory Allocator", "", "", "");

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}

		return 0;
	}
};

struct vcma {
	int operator()(int argc, char* argv[]) {

		const char* allocatorname = "vcma";

		PCM* m;
		m = PCM::getInstance();
		m->cleanup();
		PCM::ErrorCode returnResult = m->program();
		if (returnResult != PCM::Success) {
			std::cerr << "PCM couldn't start" << std::endl;
			std::cerr << "Error code: " << returnResult << std::endl;
			exit(1);
		}

		cudaError_t cudaStatus;
		clock_t start, end;
		SystemCounterState before_sstate, after_sstate;

		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);

		size_t total_size = prop.totalGlobalMem;
		size_t N = std::strtoull(argv[2], nullptr, 10);
		size_t M = total_size / (N * sizeof(double));

		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		}

		double* dev_buffer;

		cudaStatus = cudaMalloc((void**)&dev_buffer, total_size);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}


		hca_init(total_size + N * sizeof(double*), N + 1, 1);

		double** buffer = (double**)vca_malloc(N * sizeof(double*));
		for (int i = 0; i < N; ++i)
			buffer[i] = (double*)vca_malloc(M * sizeof(double));

		double* buffer_start = buffer[0];

		for (size_t i = 0; i < N; i++)
			for (size_t j = 0; j < M; j++)
				buffer[i][j] = i * M + j;

		start = clock();
		before_sstate = getSystemCounterState();

		for (size_t i = 0; i < N; i++)
		{
			cudaStatus = cudaMemcpy(dev_buffer + (i * M), buffer_start + (i * M), M * sizeof(double), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
			}
		}

		after_sstate = getSystemCounterState();
		end = clock();

		double time = (double)(end - start) / CLOCKS_PER_SEC;
		double cpu_energy = getConsumedJoules(before_sstate, after_sstate);
		uint64 L3CacheMisses = getL3CacheMisses(before_sstate, after_sstate);
		uint64 L2CacheMisses = getL2CacheMisses(before_sstate, after_sstate);

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(time).c_str(), allocatorname,
			"N", "Time", "Memory Allocator", "", "Seconds", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(cpu_energy).c_str(), allocatorname,
			"N", "Energy", "Memory Allocator", "", "Joules", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(L3CacheMisses).c_str(), allocatorname,
			"N", "L3 Cache Misses", "Memory Allocator", "", "", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(L2CacheMisses).c_str(), allocatorname,
			"N", "L2 Cache Misses", "Memory Allocator", "", "", "");


		start = clock();
		before_sstate = getSystemCounterState();

		for (size_t i = 0; i < N; i++)
		{
			cudaStatus = cudaMemcpy(buffer_start + (i * M), dev_buffer + (i * M), M * sizeof(double), cudaMemcpyDeviceToHost);

			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
			}
		}

		after_sstate = getSystemCounterState();
		end = clock();

		time = (double)(end - start) / CLOCKS_PER_SEC;
		cpu_energy = getConsumedJoules(before_sstate, after_sstate);
		L3CacheMisses = getL3CacheMisses(before_sstate, after_sstate);
		L2CacheMisses = getL2CacheMisses(before_sstate, after_sstate);

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(time).c_str(), allocatorname,
			"N", "Time", "Memory Allocator", "", "Seconds", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(cpu_energy).c_str(), allocatorname,
			"N", "Energy", "Memory Allocator", "", "Joules", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(L3CacheMisses).c_str(), allocatorname,
			"N", "L3 Cache Misses", "Memory Allocator", "", "", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(L2CacheMisses).c_str(), allocatorname,
			"N", "L2 Cache Misses", "Memory Allocator", "", "", "");

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}

		return 0;
	}
};

struct vcms {
	int operator()(int argc, char* argv[]) {

		const char* allocatorname = "vcms";

		PCM* m;
		m = PCM::getInstance();
		m->cleanup();
		PCM::ErrorCode returnResult = m->program();
		if (returnResult != PCM::Success) {
			std::cerr << "PCM couldn't start" << std::endl;
			std::cerr << "Error code: " << returnResult << std::endl;
			exit(1);
		}

		cudaError_t cudaStatus;
		clock_t start, end;
		SystemCounterState before_sstate, after_sstate;

		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);

		size_t total_size = prop.totalGlobalMem;
		size_t N = std::strtoull(argv[2], nullptr, 10);
		size_t M = total_size / (N * sizeof(double));

		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		}

		double* dev_buffer;

		cudaStatus = cudaMalloc((void**)&dev_buffer, total_size);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		hca_init(total_size + N * sizeof(double*), N + 1, 1);

		double** buffer = (double**)vca_malloc(N * sizeof(double*));
		for (int i = 0; i < N; ++i)
			buffer[i] = (double*)vca_malloc(M * sizeof(double));

		for (size_t i = 0; i < N; i++)
			for (size_t j = 0; j < M; j++)
				buffer[i][j] = i * M + j;

		start = clock();
		before_sstate = getSystemCounterState();

		cudaStatus = cudaMemcpy(dev_buffer, buffer[0], N * M * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) fprintf(stderr, "cudaMemcpy failed!");

		after_sstate = getSystemCounterState();
		end = clock();

		double time = (double)(end - start) / CLOCKS_PER_SEC;
		double cpu_energy = getConsumedJoules(before_sstate, after_sstate);
		uint64 L3CacheMisses = getL3CacheMisses(before_sstate, after_sstate);
		uint64 L2CacheMisses = getL2CacheMisses(before_sstate, after_sstate);

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(time).c_str(), allocatorname,
			"N", "Time", "Memory Allocator", "", "Seconds", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(cpu_energy).c_str(), allocatorname,
			"N", "Energy", "Memory Allocator", "", "Joules", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(L3CacheMisses).c_str(), allocatorname,
			"N", "L3 Cache Misses", "Memory Allocator", "", "", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(L2CacheMisses).c_str(), allocatorname,
			"N", "L2 Cache Misses", "Memory Allocator", "", "", "");


		start = clock();
		before_sstate = getSystemCounterState();

		cudaStatus = cudaMemcpy(buffer[0], dev_buffer, N * M * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) fprintf(stderr, "cudaMemcpy failed!");

		after_sstate = getSystemCounterState();
		end = clock();

		time = (double)(end - start) / CLOCKS_PER_SEC;
		cpu_energy = getConsumedJoules(before_sstate, after_sstate);
		L3CacheMisses = getL3CacheMisses(before_sstate, after_sstate);
		L2CacheMisses = getL2CacheMisses(before_sstate, after_sstate);

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(time).c_str(), allocatorname,
			"N", "Time", "Memory Allocator", "", "Seconds", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(cpu_energy).c_str(), allocatorname,
			"N", "Energy", "Memory Allocator", "", "Joules", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(L3CacheMisses).c_str(), allocatorname,
			"N", "L3 Cache Misses", "Memory Allocator", "", "", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(L2CacheMisses).c_str(), allocatorname,
			"N", "L2 Cache Misses", "Memory Allocator", "", "", "");

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}

		return 0;
	}
};

struct m {
	int operator()(int argc, char* argv[]) {

		const char* allocatorname = "m";

		PCM* m;
		m = PCM::getInstance();
		m->cleanup();
		PCM::ErrorCode returnResult = m->program();
		if (returnResult != PCM::Success) {
			std::cerr << "PCM couldn't start" << std::endl;
			std::cerr << "Error code: " << returnResult << std::endl;
			exit(1);
		}

		cudaError_t cudaStatus;
		clock_t start, end;
		SystemCounterState before_sstate, after_sstate;

		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);

		size_t total_size = prop.totalGlobalMem;
		size_t N = std::strtoull(argv[2], nullptr, 10);
		size_t M = total_size / (N * sizeof(double));

		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		}

		double* dev_buffer;

		cudaStatus = cudaMalloc((void**)&dev_buffer, total_size);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		double** buffer = (double**)malloc(N * sizeof(double*));
		for (int i = 0; i < N; ++i)
			buffer[i] = (double*)malloc(M * sizeof(double));



		for (size_t i = 0; i < N; i++)
			for (size_t j = 0; j < M; j++)
				buffer[i][j] = i * M + j;

		start = clock();
		before_sstate = getSystemCounterState();

		for (size_t i = 0; i < N; i++)
		{
			cudaStatus = cudaMemcpy(dev_buffer + (i * M), buffer[i], M * sizeof(double), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
			}
		}

		after_sstate = getSystemCounterState();
		end = clock();

		double time = (double)(end - start) / CLOCKS_PER_SEC;
		double cpu_energy = getConsumedJoules(before_sstate, after_sstate);
		uint64 L3CacheMisses = getL3CacheMisses(before_sstate, after_sstate);
		uint64 L2CacheMisses = getL2CacheMisses(before_sstate, after_sstate);

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(time).c_str(), allocatorname,
			"N", "Time", "Memory Allocator", "", "Seconds", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(cpu_energy).c_str(), allocatorname,
			"N", "Energy", "Memory Allocator", "", "Joules", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(L3CacheMisses).c_str(), allocatorname,
			"N", "L3 Cache Misses", "Memory Allocator", "", "", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(L2CacheMisses).c_str(), allocatorname,
			"N", "L2 Cache Misses", "Memory Allocator", "", "", "");


		start = clock();
		before_sstate = getSystemCounterState();

		for (size_t i = 0; i < N; i++)
		{
			cudaStatus = cudaMemcpy(buffer[i], dev_buffer + (i * M), M * sizeof(double), cudaMemcpyDeviceToHost);

			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
			}
		}

		after_sstate = getSystemCounterState();
		end = clock();

		time = (double)(end - start) / CLOCKS_PER_SEC;
		cpu_energy = getConsumedJoules(before_sstate, after_sstate);
		L3CacheMisses = getL3CacheMisses(before_sstate, after_sstate);
		L2CacheMisses = getL2CacheMisses(before_sstate, after_sstate);

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(time).c_str(), allocatorname,
			"N", "Time", "Memory Allocator", "", "Seconds", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(cpu_energy).c_str(), allocatorname,
			"N", "Energy", "Memory Allocator", "", "Joules", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(L3CacheMisses).c_str(), allocatorname,
			"N", "L3 Cache Misses", "Memory Allocator", "", "", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(L2CacheMisses).c_str(), allocatorname,
			"N", "L2 Cache Misses", "Memory Allocator", "", "", "");

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}

		return 0;
	}
};

struct mim {
	int operator()(int argc, char* argv[]) {

		const char* allocatorname = "mim";

		PCM* m;
		m = PCM::getInstance();
		m->cleanup();
		PCM::ErrorCode returnResult = m->program();
		if (returnResult != PCM::Success) {
			std::cerr << "PCM couldn't start" << std::endl;
			std::cerr << "Error code: " << returnResult << std::endl;
			exit(1);
		}

		cudaError_t cudaStatus;
		clock_t start, end;
		SystemCounterState before_sstate, after_sstate;

		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);

		size_t total_size = prop.totalGlobalMem;
		size_t N = std::strtoull(argv[2], nullptr, 10);
		size_t M = total_size / (N * sizeof(double));

		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		}

		double* dev_buffer;

		cudaStatus = cudaMalloc((void**)&dev_buffer, total_size);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		double** buffer = (double**)mi_malloc(N * sizeof(double*));
		for (int i = 0; i < N; ++i)
			buffer[i] = (double*)mi_malloc(M * sizeof(double));

		for (size_t i = 0; i < N; i++)
			for (size_t j = 0; j < M; j++)
				buffer[i][j] = i * M + j;

		start = clock();
		before_sstate = getSystemCounterState();

		for (size_t i = 0; i < N; i++)
		{
			cudaStatus = cudaMemcpy(dev_buffer + (i * M), buffer[i], M * sizeof(double), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
			}
		}

		after_sstate = getSystemCounterState();
		end = clock();

		double time = (double)(end - start) / CLOCKS_PER_SEC;
		double cpu_energy = getConsumedJoules(before_sstate, after_sstate);
		uint64 L3CacheMisses = getL3CacheMisses(before_sstate, after_sstate);
		uint64 L2CacheMisses = getL2CacheMisses(before_sstate, after_sstate);

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(time).c_str(), allocatorname,
			"N", "Time", "Memory Allocator", "", "Seconds", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(cpu_energy).c_str(), allocatorname,
			"N", "Energy", "Memory Allocator", "", "Joules", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(L3CacheMisses).c_str(), allocatorname,
			"N", "L3 Cache Misses", "Memory Allocator", "", "", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(L2CacheMisses).c_str(), allocatorname,
			"N", "L2 Cache Misses", "Memory Allocator", "", "", "");


		start = clock();
		before_sstate = getSystemCounterState();

		for (size_t i = 0; i < N; i++)
		{
			cudaStatus = cudaMemcpy(buffer[i], dev_buffer + (i * M), M * sizeof(double), cudaMemcpyDeviceToHost);

			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
			}
		}

		after_sstate = getSystemCounterState();
		end = clock();

		time = (double)(end - start) / CLOCKS_PER_SEC;
		cpu_energy = getConsumedJoules(before_sstate, after_sstate);
		L3CacheMisses = getL3CacheMisses(before_sstate, after_sstate);
		L2CacheMisses = getL2CacheMisses(before_sstate, after_sstate);

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(time).c_str(), allocatorname,
			"N", "Time", "Memory Allocator", "", "Seconds", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(cpu_energy).c_str(), allocatorname,
			"N", "Energy", "Memory Allocator", "", "Joules", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(L3CacheMisses).c_str(), allocatorname,
			"N", "L3 Cache Misses", "Memory Allocator", "", "", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(L2CacheMisses).c_str(), allocatorname,
			"N", "L2 Cache Misses", "Memory Allocator", "", "", "");

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}

		return 0;
	}
};

struct ch {
	int operator()(int argc, char* argv[]) {

		const char* allocatorname = "ch";

		PCM* m;
		m = PCM::getInstance();
		m->cleanup();
		PCM::ErrorCode returnResult = m->program();
		if (returnResult != PCM::Success) {
			std::cerr << "PCM couldn't start" << std::endl;
			std::cerr << "Error code: " << returnResult << std::endl;
			exit(1);
		}

		cudaError_t cudaStatus;
		clock_t start, end;
		SystemCounterState before_sstate, after_sstate;

		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);

		size_t total_size = prop.totalGlobalMem;
		size_t N = std::strtoull(argv[2], nullptr, 10);
		size_t M = total_size / (N * sizeof(double));

		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		}

		double* dev_buffer;

		cudaStatus = cudaMalloc((void**)&dev_buffer, total_size);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		double** buffer;
		cudaStatus = cudaHostAlloc(&buffer, N * sizeof(double*), cudaHostAllocDefault);
		if (cudaStatus != cudaSuccess) fprintf(stderr, "cudaHostAlloc failed!");
		for (int i = 0; i < N; ++i) {
			cudaStatus = cudaHostAlloc(&buffer[i], M * sizeof(double), cudaHostAllocDefault);
			if (cudaStatus != cudaSuccess) fprintf(stderr, "cudaHostAlloc failed!");
		}

		for (size_t i = 0; i < N; i++)
			for (size_t j = 0; j < M; j++)
				buffer[i][j] = i * M + j;

		start = clock();
		before_sstate = getSystemCounterState();

		for (size_t i = 0; i < N; i++)
		{
			cudaStatus = cudaMemcpy(dev_buffer + (i * M), buffer[i], M * sizeof(double), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
			}
		}

		after_sstate = getSystemCounterState();
		end = clock();

		double time = (double)(end - start) / CLOCKS_PER_SEC;
		double cpu_energy = getConsumedJoules(before_sstate, after_sstate);
		uint64 L3CacheMisses = getL3CacheMisses(before_sstate, after_sstate);
		uint64 L2CacheMisses = getL2CacheMisses(before_sstate, after_sstate);

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(time).c_str(), allocatorname,
			"N", "Time", "Memory Allocator", "", "Seconds", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(cpu_energy).c_str(), allocatorname,
			"N", "Energy", "Memory Allocator", "", "Joules", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(L3CacheMisses).c_str(), allocatorname,
			"N", "L3 Cache Misses", "Memory Allocator", "", "", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(L2CacheMisses).c_str(), allocatorname,
			"N", "L2 Cache Misses", "Memory Allocator", "", "", "");


		start = clock();
		before_sstate = getSystemCounterState();

		for (size_t i = 0; i < N; i++)
		{
			cudaStatus = cudaMemcpy(buffer[i], dev_buffer + (i * M), M * sizeof(double), cudaMemcpyDeviceToHost);

			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
			}
		}

		after_sstate = getSystemCounterState();
		end = clock();

		time = (double)(end - start) / CLOCKS_PER_SEC;
		cpu_energy = getConsumedJoules(before_sstate, after_sstate);
		L3CacheMisses = getL3CacheMisses(before_sstate, after_sstate);
		L2CacheMisses = getL2CacheMisses(before_sstate, after_sstate);

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(time).c_str(), allocatorname,
			"N", "Time", "Memory Allocator", "", "Seconds", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(cpu_energy).c_str(), allocatorname,
			"N", "Energy", "Memory Allocator", "", "Joules", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(L3CacheMisses).c_str(), allocatorname,
			"N", "L3 Cache Misses", "Memory Allocator", "", "", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(L2CacheMisses).c_str(), allocatorname,
			"N", "L2 Cache Misses", "Memory Allocator", "", "", "");

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}

		return 0;
	}
};

struct vcml {
	int operator()(int argc, char* argv[]) {

		const char* allocatorname = "vcml";

		PCM* m;
		m = PCM::getInstance();
		m->cleanup();
		PCM::ErrorCode returnResult = m->program();
		if (returnResult != PCM::Success) {
			std::cerr << "PCM couldn't start" << std::endl;
			std::cerr << "Error code: " << returnResult << std::endl;
			exit(1);
		}

		cudaError_t cudaStatus;
		clock_t start, end;
		SystemCounterState before_sstate, after_sstate;

		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);

		size_t total_size = prop.totalGlobalMem;
		size_t N = std::strtoull(argv[2], nullptr, 10);
		size_t M = total_size / (N * sizeof(double));

		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		}

		double* dev_buffer;

		cudaStatus = cudaMalloc((void**)&dev_buffer, total_size);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}


		hca_init(total_size + N * sizeof(double*), N + 1, 1);

		double** buffer = (double**)vca_malloc(N * sizeof(double*));
		for (int i = 0; i < N; ++i)
			buffer[i] = (double*)vca_malloc(M * sizeof(double));

		//locking memory

		start = clock();
		before_sstate = getSystemCounterState();

		for (int i = 0; i < N; ++i) {
			cudaStatus = cudaHostRegister(buffer[i], M * sizeof(double), cudaHostRegisterDefault);
			if (cudaStatus != cudaSuccess) fprintf(stderr, "cudaHostRegister failed!");
		}

		after_sstate = getSystemCounterState();
		end = clock();

		double time = (double)(end - start) / CLOCKS_PER_SEC;
		double cpu_energy = getConsumedJoules(before_sstate, after_sstate);
		uint64 L3CacheMisses = getL3CacheMisses(before_sstate, after_sstate);
		uint64 L2CacheMisses = getL2CacheMisses(before_sstate, after_sstate);

		log_csv("CUDA Host Register",
			std::to_string(N).c_str(), std::to_string(time).c_str(), allocatorname,
			"N", "Time", "Memory Allocator", "", "Seconds", "");

		log_csv("CUDA Host Register",
			std::to_string(N).c_str(), std::to_string(cpu_energy).c_str(), allocatorname,
			"N", "Energy", "Memory Allocator", "", "Joules", "");

		log_csv("CUDA Host Register",
			std::to_string(N).c_str(), std::to_string(L3CacheMisses).c_str(), allocatorname,
			"N", "L3 Cache Misses", "Memory Allocator", "", "", "");

		log_csv("CUDA Host Register",
			std::to_string(N).c_str(), std::to_string(L2CacheMisses).c_str(), allocatorname,
			"N", "L2 Cache Misses", "Memory Allocator", "", "", "");

		for (size_t i = 0; i < N; i++)
			for (size_t j = 0; j < M; j++)
				buffer[i][j] = i * M + j;

		start = clock();
		before_sstate = getSystemCounterState();

		for (size_t i = 0; i < N; i++)
		{
			cudaStatus = cudaMemcpy(dev_buffer + (i * M), buffer[i], M * sizeof(double), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
			}
		}

		after_sstate = getSystemCounterState();
		end = clock();

		time = (double)(end - start) / CLOCKS_PER_SEC;
		cpu_energy = getConsumedJoules(before_sstate, after_sstate);
		L3CacheMisses = getL3CacheMisses(before_sstate, after_sstate);
		L2CacheMisses = getL2CacheMisses(before_sstate, after_sstate);

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(time).c_str(), allocatorname,
			"N", "Time", "Memory Allocator", "", "Seconds", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(cpu_energy).c_str(), allocatorname,
			"N", "Energy", "Memory Allocator", "", "Joules", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(L3CacheMisses).c_str(), allocatorname,
			"N", "L3 Cache Misses", "Memory Allocator", "", "", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(L2CacheMisses).c_str(), allocatorname,
			"N", "L2 Cache Misses", "Memory Allocator", "", "", "");


		start = clock();
		before_sstate = getSystemCounterState();

		for (size_t i = 0; i < N; i++)
		{
			cudaStatus = cudaMemcpy(buffer[i], dev_buffer + (i * M), M * sizeof(double), cudaMemcpyDeviceToHost);

			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
			}
		}

		after_sstate = getSystemCounterState();
		end = clock();

		time = (double)(end - start) / CLOCKS_PER_SEC;
		cpu_energy = getConsumedJoules(before_sstate, after_sstate);
		L3CacheMisses = getL3CacheMisses(before_sstate, after_sstate);
		L2CacheMisses = getL2CacheMisses(before_sstate, after_sstate);

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(time).c_str(), allocatorname,
			"N", "Time", "Memory Allocator", "", "Seconds", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(cpu_energy).c_str(), allocatorname,
			"N", "Energy", "Memory Allocator", "", "Joules", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(L3CacheMisses).c_str(), allocatorname,
			"N", "L3 Cache Misses", "Memory Allocator", "", "", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(L2CacheMisses).c_str(), allocatorname,
			"N", "L2 Cache Misses", "Memory Allocator", "", "", "");

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}

		return 0;
	}
};

struct vcmal {
	int operator()(int argc, char* argv[]) {

		const char* allocatorname = "vcmal";

		PCM* m;
		m = PCM::getInstance();
		m->cleanup();
		PCM::ErrorCode returnResult = m->program();
		if (returnResult != PCM::Success) {
			std::cerr << "PCM couldn't start" << std::endl;
			std::cerr << "Error code: " << returnResult << std::endl;
			exit(1);
		}

		cudaError_t cudaStatus;
		clock_t start, end;
		SystemCounterState before_sstate, after_sstate;

		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);

		size_t total_size = prop.totalGlobalMem;
		size_t N = std::strtoull(argv[2], nullptr, 10);
		size_t M = total_size / (N * sizeof(double));

		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		}

		double* dev_buffer;

		cudaStatus = cudaMalloc((void**)&dev_buffer, total_size);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}


		hca_init(total_size + N * sizeof(double*), N + 1, 1);

		double** buffer = (double**)vca_malloc(N * sizeof(double*));
		for (int i = 0; i < N; ++i)
			buffer[i] = (double*)vca_malloc(M * sizeof(double));

		double* buffer_start = buffer[0];

		//locking memory

		start = clock();
		before_sstate = getSystemCounterState();


		for (int i = 0; i < N; ++i) {
			cudaStatus = cudaHostRegister(buffer_start + (i * M), M * sizeof(double), cudaHostRegisterDefault);
			if (cudaStatus != cudaSuccess) fprintf(stderr, "cudaHostRegister failed!");
		}

		after_sstate = getSystemCounterState();
		end = clock();

		double time = (double)(end - start) / CLOCKS_PER_SEC;
		double cpu_energy = getConsumedJoules(before_sstate, after_sstate);
		uint64 L3CacheMisses = getL3CacheMisses(before_sstate, after_sstate);
		uint64 L2CacheMisses = getL2CacheMisses(before_sstate, after_sstate);

		log_csv("CUDA Host Register",
			std::to_string(N).c_str(), std::to_string(time).c_str(), allocatorname,
			"N", "Time", "Memory Allocator", "", "Seconds", "");

		log_csv("CUDA Host Register",
			std::to_string(N).c_str(), std::to_string(cpu_energy).c_str(), allocatorname,
			"N", "Energy", "Memory Allocator", "", "Joules", "");

		log_csv("CUDA Host Register",
			std::to_string(N).c_str(), std::to_string(L3CacheMisses).c_str(), allocatorname,
			"N", "L3 Cache Misses", "Memory Allocator", "", "", "");

		log_csv("CUDA Host Register",
			std::to_string(N).c_str(), std::to_string(L2CacheMisses).c_str(), allocatorname,
			"N", "L2 Cache Misses", "Memory Allocator", "", "", "");

		for (size_t i = 0; i < N; i++)
			for (size_t j = 0; j < M; j++)
				buffer[i][j] = i * M + j;

		start = clock();
		before_sstate = getSystemCounterState();

		for (size_t i = 0; i < N; i++)
		{
			cudaStatus = cudaMemcpy(dev_buffer + (i * M), buffer_start + (M * i), M * sizeof(double), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
			}
		}

		after_sstate = getSystemCounterState();
		end = clock();

		time = (double)(end - start) / CLOCKS_PER_SEC;
		cpu_energy = getConsumedJoules(before_sstate, after_sstate);
		L3CacheMisses = getL3CacheMisses(before_sstate, after_sstate);
		L2CacheMisses = getL2CacheMisses(before_sstate, after_sstate);

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(time).c_str(), allocatorname,
			"N", "Time", "Memory Allocator", "", "Seconds", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(cpu_energy).c_str(), allocatorname,
			"N", "Energy", "Memory Allocator", "", "Joules", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(L3CacheMisses).c_str(), allocatorname,
			"N", "L3 Cache Misses", "Memory Allocator", "", "", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(L2CacheMisses).c_str(), allocatorname,
			"N", "L2 Cache Misses", "Memory Allocator", "", "", "");


		start = clock();
		before_sstate = getSystemCounterState();

		for (size_t i = 0; i < N; i++)
		{
			cudaStatus = cudaMemcpy(buffer_start + (M * i), dev_buffer + (i * M), M * sizeof(double), cudaMemcpyDeviceToHost);

			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
			}
		}

		after_sstate = getSystemCounterState();
		end = clock();

		time = (double)(end - start) / CLOCKS_PER_SEC;
		cpu_energy = getConsumedJoules(before_sstate, after_sstate);
		L3CacheMisses = getL3CacheMisses(before_sstate, after_sstate);
		L2CacheMisses = getL2CacheMisses(before_sstate, after_sstate);

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(time).c_str(), allocatorname,
			"N", "Time", "Memory Allocator", "", "Seconds", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(cpu_energy).c_str(), allocatorname,
			"N", "Energy", "Memory Allocator", "", "Joules", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(L3CacheMisses).c_str(), allocatorname,
			"N", "L3 Cache Misses", "Memory Allocator", "", "", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(L2CacheMisses).c_str(), allocatorname,
			"N", "L2 Cache Misses", "Memory Allocator", "", "", "");

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}

		return 0;
	}
};

struct vcmsl {
	int operator()(int argc, char* argv[]) {

		const char* allocatorname = "vcmsl";

		PCM* m;
		m = PCM::getInstance();
		m->cleanup();
		PCM::ErrorCode returnResult = m->program();
		if (returnResult != PCM::Success) {
			std::cerr << "PCM couldn't start" << std::endl;
			std::cerr << "Error code: " << returnResult << std::endl;
			exit(1);
		}

		cudaError_t cudaStatus;
		clock_t start, end;
		SystemCounterState before_sstate, after_sstate;

		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);

		size_t total_size = prop.totalGlobalMem;
		size_t N = std::strtoull(argv[2], nullptr, 10);
		size_t M = total_size / (N * sizeof(double));

		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		}

		double* dev_buffer;

		cudaStatus = cudaMalloc((void**)&dev_buffer, total_size);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}


		hca_init(total_size + N * sizeof(double*), N + 1, 1);

		double** buffer = (double**)vca_malloc(N * sizeof(double*));
		for (int i = 0; i < N; ++i)
			buffer[i] = (double*)vca_malloc(M * sizeof(double));


		double* buffer_start = buffer[0];

		//locking memory

		start = clock();
		before_sstate = getSystemCounterState();

		cudaStatus = cudaHostRegister(buffer_start, M * N * sizeof(double), cudaHostRegisterDefault);
		if (cudaStatus != cudaSuccess) fprintf(stderr, "cudaHostRegister failed!");

		after_sstate = getSystemCounterState();
		end = clock();

		double time = (double)(end - start) / CLOCKS_PER_SEC;
		double cpu_energy = getConsumedJoules(before_sstate, after_sstate);
		uint64 L3CacheMisses = getL3CacheMisses(before_sstate, after_sstate);
		uint64 L2CacheMisses = getL2CacheMisses(before_sstate, after_sstate);

		log_csv("CUDA Host Register",
			std::to_string(N).c_str(), std::to_string(time).c_str(), allocatorname,
			"N", "Time", "Memory Allocator", "", "Seconds", "");

		log_csv("CUDA Host Register",
			std::to_string(N).c_str(), std::to_string(cpu_energy).c_str(), allocatorname,
			"N", "Energy", "Memory Allocator", "", "Joules", "");

		log_csv("CUDA Host Register",
			std::to_string(N).c_str(), std::to_string(L3CacheMisses).c_str(), allocatorname,
			"N", "L3 Cache Misses", "Memory Allocator", "", "", "");

		log_csv("CUDA Host Register",
			std::to_string(N).c_str(), std::to_string(L2CacheMisses).c_str(), allocatorname,
			"N", "L2 Cache Misses", "Memory Allocator", "", "", "");

		for (size_t i = 0; i < N; i++)
			for (size_t j = 0; j < M; j++)
				buffer[i][j] = i * M + j;

		start = clock();
		before_sstate = getSystemCounterState();

		cudaStatus = cudaMemcpy(dev_buffer, buffer[0], N * M * sizeof(double), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) fprintf(stderr, "cudaMemcpy failed!");

		after_sstate = getSystemCounterState();
		end = clock();

		time = (double)(end - start) / CLOCKS_PER_SEC;
		cpu_energy = getConsumedJoules(before_sstate, after_sstate);
		L3CacheMisses = getL3CacheMisses(before_sstate, after_sstate);
		L2CacheMisses = getL2CacheMisses(before_sstate, after_sstate);

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(time).c_str(), allocatorname,
			"N", "Time", "Memory Allocator", "", "Seconds", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(cpu_energy).c_str(), allocatorname,
			"N", "Energy", "Memory Allocator", "", "Joules", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(L3CacheMisses).c_str(), allocatorname,
			"N", "L3 Cache Misses", "Memory Allocator", "", "", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(L2CacheMisses).c_str(), allocatorname,
			"N", "L2 Cache Misses", "Memory Allocator", "", "", "");


		start = clock();
		before_sstate = getSystemCounterState();

		cudaStatus = cudaMemcpy(buffer[0], dev_buffer, N * M * sizeof(double), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) fprintf(stderr, "cudaMemcpy failed!");

		after_sstate = getSystemCounterState();
		end = clock();

		time = (double)(end - start) / CLOCKS_PER_SEC;
		cpu_energy = getConsumedJoules(before_sstate, after_sstate);
		L3CacheMisses = getL3CacheMisses(before_sstate, after_sstate);
		L2CacheMisses = getL2CacheMisses(before_sstate, after_sstate);

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(time).c_str(), allocatorname,
			"N", "Time", "Memory Allocator", "", "Seconds", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(cpu_energy).c_str(), allocatorname,
			"N", "Energy", "Memory Allocator", "", "Joules", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(L3CacheMisses).c_str(), allocatorname,
			"N", "L3 Cache Misses", "Memory Allocator", "", "", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(L2CacheMisses).c_str(), allocatorname,
			"N", "L2 Cache Misses", "Memory Allocator", "", "", "");

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}

		return 0;
	}
};

struct ml {
	int operator()(int argc, char* argv[]) {

		const char* allocatorname = "ml";

		PCM* m;
		m = PCM::getInstance();
		m->cleanup();
		PCM::ErrorCode returnResult = m->program();
		if (returnResult != PCM::Success) {
			std::cerr << "PCM couldn't start" << std::endl;
			std::cerr << "Error code: " << returnResult << std::endl;
			exit(1);
		}

		cudaError_t cudaStatus;
		clock_t start, end;
		SystemCounterState before_sstate, after_sstate;

		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);

		size_t total_size = prop.totalGlobalMem;
		size_t N = std::strtoull(argv[2], nullptr, 10);
		size_t M = total_size / (N * sizeof(double));

		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		}

		double* dev_buffer;

		cudaStatus = cudaMalloc((void**)&dev_buffer, total_size);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}


		double** buffer = (double**)malloc(N * sizeof(double*));
		for (int i = 0; i < N; ++i)
			buffer[i] = (double*)malloc(M * sizeof(double));

		//locking memory

		start = clock();
		before_sstate = getSystemCounterState();

		for (int i = 0; i < N; ++i) {
			cudaStatus = cudaHostRegister(buffer[i], M * sizeof(double), cudaHostRegisterDefault);
			if (cudaStatus != cudaSuccess) fprintf(stderr, "cudaHostRegister failed!");
		}

		after_sstate = getSystemCounterState();
		end = clock();

		double time = (double)(end - start) / CLOCKS_PER_SEC;
		double cpu_energy = getConsumedJoules(before_sstate, after_sstate);
		uint64 L3CacheMisses = getL3CacheMisses(before_sstate, after_sstate);
		uint64 L2CacheMisses = getL2CacheMisses(before_sstate, after_sstate);

		log_csv("CUDA Host Register",
			std::to_string(N).c_str(), std::to_string(time).c_str(), allocatorname,
			"N", "Time", "Memory Allocator", "", "Seconds", "");

		log_csv("CUDA Host Register",
			std::to_string(N).c_str(), std::to_string(cpu_energy).c_str(), allocatorname,
			"N", "Energy", "Memory Allocator", "", "Joules", "");

		log_csv("CUDA Host Register",
			std::to_string(N).c_str(), std::to_string(L3CacheMisses).c_str(), allocatorname,
			"N", "L3 Cache Misses", "Memory Allocator", "", "", "");

		log_csv("CUDA Host Register",
			std::to_string(N).c_str(), std::to_string(L2CacheMisses).c_str(), allocatorname,
			"N", "L2 Cache Misses", "Memory Allocator", "", "", "");

		for (size_t i = 0; i < N; i++)
			for (size_t j = 0; j < M; j++)
				buffer[i][j] = i * M + j;

		start = clock();
		before_sstate = getSystemCounterState();

		for (size_t i = 0; i < N; i++)
		{
			cudaStatus = cudaMemcpy(dev_buffer + (i * M), buffer[i], M * sizeof(double), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
			}
		}

		after_sstate = getSystemCounterState();
		end = clock();

		time = (double)(end - start) / CLOCKS_PER_SEC;
		cpu_energy = getConsumedJoules(before_sstate, after_sstate);
		L3CacheMisses = getL3CacheMisses(before_sstate, after_sstate);
		L2CacheMisses = getL2CacheMisses(before_sstate, after_sstate);

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(time).c_str(), allocatorname,
			"N", "Time", "Memory Allocator", "", "Seconds", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(cpu_energy).c_str(), allocatorname,
			"N", "Energy", "Memory Allocator", "", "Joules", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(L3CacheMisses).c_str(), allocatorname,
			"N", "L3 Cache Misses", "Memory Allocator", "", "", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(L2CacheMisses).c_str(), allocatorname,
			"N", "L2 Cache Misses", "Memory Allocator", "", "", "");


		start = clock();
		before_sstate = getSystemCounterState();

		for (size_t i = 0; i < N; i++)
		{
			cudaStatus = cudaMemcpy(buffer[i], dev_buffer + (i * M), M * sizeof(double), cudaMemcpyDeviceToHost);

			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
			}
		}

		after_sstate = getSystemCounterState();
		end = clock();

		time = (double)(end - start) / CLOCKS_PER_SEC;
		cpu_energy = getConsumedJoules(before_sstate, after_sstate);
		L3CacheMisses = getL3CacheMisses(before_sstate, after_sstate);
		L2CacheMisses = getL2CacheMisses(before_sstate, after_sstate);

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(time).c_str(), allocatorname,
			"N", "Time", "Memory Allocator", "", "Seconds", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(cpu_energy).c_str(), allocatorname,
			"N", "Energy", "Memory Allocator", "", "Joules", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(L3CacheMisses).c_str(), allocatorname,
			"N", "L3 Cache Misses", "Memory Allocator", "", "", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(L2CacheMisses).c_str(), allocatorname,
			"N", "L2 Cache Misses", "Memory Allocator", "", "", "");

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}

		return 0;
	}
};

struct miml {
	int operator()(int argc, char* argv[]) {

		const char* allocatorname = "miml";

		PCM* m;
		m = PCM::getInstance();
		m->cleanup();
		PCM::ErrorCode returnResult = m->program();
		if (returnResult != PCM::Success) {
			std::cerr << "PCM couldn't start" << std::endl;
			std::cerr << "Error code: " << returnResult << std::endl;
			exit(1);
		}

		cudaError_t cudaStatus;
		clock_t start, end;
		SystemCounterState before_sstate, after_sstate;

		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);

		size_t total_size = prop.totalGlobalMem;
		size_t N = std::strtoull(argv[2], nullptr, 10);
		size_t M = total_size / (N * sizeof(double));

		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		}

		double* dev_buffer;

		cudaStatus = cudaMalloc((void**)&dev_buffer, total_size);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
		}

		double** buffer = (double**)mi_malloc(N * sizeof(double*));
		for (int i = 0; i < N; ++i)
			buffer[i] = (double*)mi_malloc(M * sizeof(double));

		//locking memory

		start = clock();
		before_sstate = getSystemCounterState();

		for (int i = 0; i < N; ++i) {
			cudaStatus = cudaHostRegister(buffer[i], M * sizeof(double), cudaHostRegisterDefault);
			if (cudaStatus != cudaSuccess) fprintf(stderr, "cudaHostRegister failed!");
		}

		after_sstate = getSystemCounterState();
		end = clock();

		double time = (double)(end - start) / CLOCKS_PER_SEC;
		double cpu_energy = getConsumedJoules(before_sstate, after_sstate);
		uint64 L3CacheMisses = getL3CacheMisses(before_sstate, after_sstate);
		uint64 L2CacheMisses = getL2CacheMisses(before_sstate, after_sstate);

		log_csv("CUDA Host Register",
			std::to_string(N).c_str(), std::to_string(time).c_str(), allocatorname,
			"N", "Time", "Memory Allocator", "", "Seconds", "");

		log_csv("CUDA Host Register",
			std::to_string(N).c_str(), std::to_string(cpu_energy).c_str(), allocatorname,
			"N", "Energy", "Memory Allocator", "", "Joules", "");

		log_csv("CUDA Host Register",
			std::to_string(N).c_str(), std::to_string(L3CacheMisses).c_str(), allocatorname,
			"N", "L3 Cache Misses", "Memory Allocator", "", "", "");

		log_csv("CUDA Host Register",
			std::to_string(N).c_str(), std::to_string(L2CacheMisses).c_str(), allocatorname,
			"N", "L2 Cache Misses", "Memory Allocator", "", "", "");

		for (size_t i = 0; i < N; i++)
			for (size_t j = 0; j < M; j++)
				buffer[i][j] = i * M + j;

		start = clock();
		before_sstate = getSystemCounterState();

		for (size_t i = 0; i < N; i++)
		{
			cudaStatus = cudaMemcpy(dev_buffer + (i * M), buffer[i], M * sizeof(double), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
			}
		}

		after_sstate = getSystemCounterState();
		end = clock();

		time = (double)(end - start) / CLOCKS_PER_SEC;
		cpu_energy = getConsumedJoules(before_sstate, after_sstate);
		L3CacheMisses = getL3CacheMisses(before_sstate, after_sstate);
		L2CacheMisses = getL2CacheMisses(before_sstate, after_sstate);

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(time).c_str(), allocatorname,
			"N", "Time", "Memory Allocator", "", "Seconds", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(cpu_energy).c_str(), allocatorname,
			"N", "Energy", "Memory Allocator", "", "Joules", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(L3CacheMisses).c_str(), allocatorname,
			"N", "L3 Cache Misses", "Memory Allocator", "", "", "");

		log_csv("CUDA Host to Device",
			std::to_string(N).c_str(), std::to_string(L2CacheMisses).c_str(), allocatorname,
			"N", "L2 Cache Misses", "Memory Allocator", "", "", "");


		start = clock();
		before_sstate = getSystemCounterState();

		for (size_t i = 0; i < N; i++)
		{
			cudaStatus = cudaMemcpy(buffer[i], dev_buffer + (i * M), M * sizeof(double), cudaMemcpyDeviceToHost);

			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
			}
		}

		after_sstate = getSystemCounterState();
		end = clock();

		time = (double)(end - start) / CLOCKS_PER_SEC;
		cpu_energy = getConsumedJoules(before_sstate, after_sstate);
		L3CacheMisses = getL3CacheMisses(before_sstate, after_sstate);
		L2CacheMisses = getL2CacheMisses(before_sstate, after_sstate);

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(time).c_str(), allocatorname,
			"N", "Time", "Memory Allocator", "", "Seconds", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(cpu_energy).c_str(), allocatorname,
			"N", "Energy", "Memory Allocator", "", "Joules", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(L3CacheMisses).c_str(), allocatorname,
			"N", "L3 Cache Misses", "Memory Allocator", "", "", "");

		log_csv("CUDA Device to Host",
			std::to_string(N).c_str(), std::to_string(L2CacheMisses).c_str(), allocatorname,
			"N", "L2 Cache Misses", "Memory Allocator", "", "", "");

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}

		return 0;
	}
};


std::map<std::string, std::function<int(int argc, char* argv[])>> benchmarkMap = {
	{"vcm", vcm()},
	{"vcma", vcma()},
	{"vcms", vcms()},
	{"vcml", vcml()},
	{"vcmal", vcmal()},
	{"vcmsl", vcmsl()},

	{"m", m()},
	{"ml", ml()},

	{"mim", mim()},
	{"miml", miml()},

	{"ch", ch()},
};

int main(int argc, char* argv[]) {

	if (argc != 3) {
		std::cerr << "Usage: " << argv[0] << " <type> <N>" << std::endl;
		return 1;
	}

	std::string benchmarkType = argv[1];

	auto it = benchmarkMap.find(benchmarkType);
	if (it != benchmarkMap.end()) {
		return it->second(argc, argv);
	}
	else {
		std::cerr << "Invalid benchmark type: " << benchmarkType << std::endl;
		return 1;
	}

}
