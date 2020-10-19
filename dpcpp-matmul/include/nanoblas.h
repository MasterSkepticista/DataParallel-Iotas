#pragma once
#include <CL/sycl.hpp>
#include "common.h"
#include "dpc_common.hpp"
#include <vector>

using namespace sycl;

#define PROFILING 1
#if PROFILING
#define PROFILE_SCOPE(name) InstrumentationTimer timer##__LINE__(name) // TOO SMART. How to give same timernames in one function. You append LINE_NUMBER!!!
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCSIG__) // __FUNCTION__ means function name, but overloading is useless here
#endif

queue create_device_queue() {
	PROFILE_FUNCTION();
	try {
		default_selector d_selector;
		queue q(d_selector, dpc_common::exception_handler);
		std::cout << "Enumerated Device: " << q.get_device().get_info<info::device::name>() << "\n";
		return q;
	}
	catch (sycl::exception const& e) {
		std::cout << "Exception while creating queue.\n";
		std::terminate();
	}
}

template<typename T>
void MatrixMulParallelNaive(queue& q, 
	size_t M, size_t N, size_t P,
	const std::vector<T>& a_host,
	const std::vector<T>& b_host,
	std::vector<T>& c_gpu) {
	/*
		1. Create buffers for array
		2. Create a command group containing kernel as a lambda
		3. Give access permissions inside the submission
	*/
	PROFILE_FUNCTION();
	try {
		
		buffer<T, 1> a(a_host.data(), range<1>{a_host.size()});
		buffer<T, 1> b(b_host.data(), range<1>{b_host.size()});
		buffer<T, 2> c(c_gpu.data(), range<2>{M, P});
		PROFILE_SCOPE("Starting Multiply on GPU");
		std::cout << "GPU::Multiplying A and B into C.\n";
		auto e = q.submit([&](handler& h) {

			auto A = a.template get_access<access::mode::read>(h);
			auto B = b.template get_access<access::mode::read>(h);
			auto C = c.template get_access<access::mode::write>(h);
			
			h.parallel_for(range<2>{M, P}, [=](id<2> index) {
				size_t row = index[0];
				size_t col = index[1];
				auto sum = 0;
				// Compute result of ONE element of C
				for (size_t i = 0; i < N; i++)
					sum += A[row * M + i] * B[i * N + col];
				C[index] = sum;
				});
			});
		e.wait();
	}
	catch (sycl::exception const& e) {
		std::cout << "An exception is caught while multiplying matrices.\n";
		terminate();
	}
}

template <typename T>
void MatrixMulTiled(queue& q,
					size_t M, size_t N, size_t P,
					const std::vector<T>& a_host,
					const std::vector<T>& b_host,
					std::vector<T>& c_gpu) {
	PROFILE_FUNCTION();
	try {
		/* window size TS x TS */
		size_t TS = 2;

		/* Create buffers */
		buffer<T, 1> a(a_host.data(), range<1>{a_host.size()});
		buffer<T, 1> b(b_host.data(), range<1>{b_host.size()});
		buffer<T, 2> c(c_gpu.data(), range<2>{M, P});

		auto e = q.submit([&](handler& h) {
			/* Create accessors */
			auto A = a.template get_access<access::mode::read>(h);
			auto B = b.template get_access<access::mode::read>(h);
			auto C = c.template get_access<access::mode::write>(h);

			/* Local accessor hyperfast cache */
			accessor<T, 1, access::mode::read_write, access::target::local> C_cache(range<1>{c_gpu.size()}, h);

			/* Create kernel */
			h.parallel_for(nd_range<2>(range<2>(M, P), /* global range */
									range<2>(TS, TS)), /* local range */
									[=](nd_item<2> item) {
				
				/* ID in global x-y of each tile: Which tile is it, in x-y? */
				const auto id_x = item.get_global_id(0);
				const auto id_y = item.get_global_id(1);
				
				/* Map the 2D x-y coords to 1D */
				const auto width = item.get_group_range(0) * item.get_local_range(0);

				/* Map each memory block to 1D now. This is to access each element */
				const auto index = id_x * width + id_y;

				/* ... computation ... */

				/* Barrier to sync the read-write */
				item.barrier(access::fence_space::local_space);

				/* Write from cache to host memory */
				C[index] = C_cache[index];
				
				});

			});
		e.wait();
	}
	catch (sycl::exception const& e) {
		std::cout << "An exception is caught while multiplying matrices.\n";
		terminate();
	}
}

// Function set to verify results of different kernels you implement
template <typename T>
void MatrixMulCPU(size_t M, size_t N, size_t P,
				const std::vector<T> &a_host,
				const std::vector<T> &b_host,
				std::vector<T> &c_host) {

	PROFILE_FUNCTION();
	
	for (size_t i = 0; i < M; i++)
		for (size_t k = 0; k < N; k++)
			for (size_t j = 0; j < P; j++)
				c_host[i * M + j] += a_host[i * M + k] * b_host[k * N + j];
}

template <typename T>
bool AreSame(T a, T b) {
	return fabs(a - b) < std::numeric_limits<float>::epsilon();
}

template <typename T>
bool VerifyResult(std::vector<T>& c_gpu, std::vector<T>& c_host) {
	PROFILE_FUNCTION();
	std::cout << "Comparing results of CPU and GPU.\n";

	int errors = 0;
	bool mismatched = false;
	for (size_t i = 0; i < c_host.size(); i++) {
		if (!AreSame<T>(c_gpu[i], c_host[i])) {
			std::cout << "Unexpected Result for [" << i << "]"
				<< " Expected -> " << c_host[i]
				<< " Computed -> " << c_gpu[i] << "\n";
			errors++;
			mismatched = true;
		} if (errors == 5) break;
		if (errors == 5) break;
	}

	// Need not worry about gc, they are destroyed at end of scope.
	if (!mismatched) {
		size_t indices[]{ 0, 1, 2, (c_host.size() - 1) };
		constexpr size_t indices_size = sizeof(indices) / sizeof(size_t);

		// Print out few indices for sanity check
		for (size_t i = 0; i < indices_size; i++) {
			size_t j = indices[i];
			if (i == indices_size - 1) std::cout << "...\n";
			std::cout << "[" << j << "]: " << c_host[j] << " = " << c_gpu[j] << "\n";
		}
		std::cout << ":) Results Match.\n";
		return 0;
	}
	else {
		std::cout << ":( Failed.";
		return -1;
	}
}
