#pragma once
#include <CL/sycl.hpp>
#include "common.h"
#include "dpc_common.hpp"
#include <vector>
#include "settings.h"

using namespace sycl;

#define PROFILING 1
#if PROFILING
#define PROFILE_SCOPE(name) InstrumentationTimer timer##__LINE__(name) // Same timernames in one fn: append LINE_NUMBER!!!
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCSIG__) // __FUNCTION__ only fn name, __FUNCSIG__ shows overloads
#endif

queue create_device_queue() {
	PROFILE_FUNCTION();
	try {
		default_selector d_selector;
		queue q(d_selector, dpc_common::exception_handler);
		std::cout << "Enumerated Device: " << q.get_device().get_info<info::device::name>() << "\n";
		auto wgroup_size = q.get_device().get_info<info::device::max_work_group_size>();
		auto local_mem_size = q.get_device().get_info<info::device::local_mem_size>();
		auto global_mem_size = q.get_device().get_info<info::device::global_mem_size>();

		std::cout << "Maximum workgroup size\t:" << wgroup_size << "\n" 
			<< "Global Memory Size\t:" << global_mem_size / 1024 / 1024 << " MB\n"
			<< "Local Memory Size\t:" << local_mem_size / 1024 << " KB\n";

		if (wgroup_size % 2 != 0)
			std::cout << "[WARNING] Workgroup size has to be even.\n";

		
		return q;
	}
	catch (sycl::exception const& e) {
		std::cout << "Exception while creating queue.\n";
		std::terminate();
	}
}


//-----------------------------------------------------------------------------
// Kernel-1: Naive approach (roofline model) ~1.0x
//-----------------------------------------------------------------------------
template<typename T>
void MatrixMulParallelNaive(queue& q, 
	size_t M, size_t N, size_t P,
	const std::vector<T>& a_host,
	const std::vector<T>& b_host,
	std::vector<T>& c_gpu) {
	
	PROFILE_FUNCTION();
	try {
		
		buffer<T, 1> a(a_host.data(), range<1>{a_host.size()});
		buffer<T, 1> b(b_host.data(), range<1>{b_host.size()});
		buffer<T, 1> c(c_gpu.data(), range<1>{c_gpu.size()});
		
		auto e = q.submit([&](handler& h) {

			auto A = a.template get_access<access::mode::read>(h);
			auto B = b.template get_access<access::mode::read>(h);
			auto C = c.template get_access<access::mode::write>(h);
			
			h.parallel_for(range<1>{c_gpu.size()}, [=](id<1> index) {
				size_t row = index / M;
				size_t col = index % M;
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

//-----------------------------------------------------------------------------
// Kernel-2: Tiled approach -> use on-chip cache registers ~2.5x
//-----------------------------------------------------------------------------
template <typename T>
void MatrixMulTiled(queue& q,
					size_t M, size_t N, size_t P,
					const std::vector<T>& a_host,
					const std::vector<T>& b_host,
					std::vector<T>& c_gpu) {
	PROFILE_FUNCTION();
	try {
		/* Create buffers */
		buffer<T, 1> a(a_host.data(), range<1>{a_host.size()});
		buffer<T, 1> b(b_host.data(), range<1>{b_host.size()});
		buffer<T, 1> c(c_gpu.data(), range<1>{c_gpu.size()});

		auto e = q.submit([&](handler& h) {
			/* Create accessors */
			auto A = a.template get_access<access::mode::read>(h);
			auto B = b.template get_access<access::mode::read>(h);
			auto C = c.template get_access<access::mode::write>(h);

			/* Local accessor TILES: hyperfast cache */
			accessor<T, 2, access::mode::read_write, access::target::local> Asub(range<2>{TS, TS}, h);
			accessor<T, 2, access::mode::read_write, access::target::local> Bsub(range<2>{TS, TS}, h);

			/* Create kernel */
			h.parallel_for(nd_range<2>(range<2>(M, P), range<2>(TS, TS)), [=](nd_item<2> item) {
				/* row, col thread identifier for each tile */
				size_t row = item.get_local_id(0);
				size_t col = item.get_local_id(1);

				/* row, col thread identifer of C */
				size_t globalRow = TS * item.get_group().get_id(0) + row;
				size_t globalCol = TS * item.get_group().get_id(1) + col;

				T acc = 0;
				/* loop over all tiles */
				const size_t num_tiles = P / TS;
				for (size_t t = 0; t < num_tiles; t++) {
					/* Load one tile of A and B into cache */
					const size_t tiledRow = TS * t + row;
					const size_t tiledCol = TS * t + col;
					Asub[row][col] = A[globalRow * M + tiledCol];
					Bsub[row][col] = B[tiledRow * N + globalCol];
					
					/* Barrier to sync the read-write */
					item.barrier(access::fence_space::local_space);
					
					/* Do the matmul between Asub and Bsub */
					for (size_t k = 0; k < TS; k++) {
						acc += Asub[row][k] * Bsub[k][col];
					}

					/* Barrier to sync the read-write */
					item.barrier(access::fence_space::local_space);
				}
				/* Write from cache to host memory */
				C[globalRow * M + globalCol] = acc;
			});

		});
		e.wait();
	}
	catch (sycl::exception const& e) {
		std::cout << "An exception is caught while multiplying matrices.\n";
		terminate();
	}
}

//-----------------------------------------------------------------------------
// Kernel-3: Increase WPT (Work per thread) ~?x
//-----------------------------------------------------------------------------
template <typename T>
void MatrixMulWPT(queue &q, 
				size_t M, size_t N, size_t P,
				const std::vector<T>& a_host,
				const std::vector<T>& b_host,
				std::vector<T>& c_gpu) {
	PROFILE_FUNCTION();
	try {
		/* Create buffers */
		buffer<T, 1> a_buf(a_host.data(), range<1>{a_host.size()});
		buffer<T, 1> b_buf(b_host.data(), range<1>{b_host.size()});
		buffer<T, 1> c_buf(c_gpu.data(), range<1>{c_gpu.size()});

		/* Submit to queue with buffer accessors */
		auto e = q.submit([&](handler& h) {

			auto A = a_buf.template get_access<access::mode::read>(h);
			auto B = b_buf.template get_access<access::mode::read>(h);
			auto C = c_buf.template get_access<access::mode::write>(h);

			/* Create cache reservations for workgroup */
			accessor<T, 2, access::mode::read_write, access::target::local> Asub(range<2>{TS, TS}, h);
			accessor<T, 2, access::mode::read_write, access::target::local> Bsub(range<2>{TS, TS}, h);

			h.parallel_for(nd_range<2>(range<2>{M, P/WPT}, range<2>{TS, RTS}), [=](nd_item<2> item) {
				/* Thread identifiers of work-item */
				const int row = item.get_local_id(0); // 0-3 (1-TS)
				const int col = item.get_local_id(1); // 0-1 (1-TS/WPT)

				/* Thread identifiers across C */
				const int globalRow = TS * item.get_group().get_id(0) + row;
				const int globalCol = TS * item.get_group().get_id(1) + col;

				/* WPT allocators */
				T acc[WPT];
				for (int w = 0; w < WPT; w++) { acc[w] = 0; }

				/* For all tiles */
				const int num_tiles = P / TS;
				for (int t = 0; t < num_tiles; t++) {
					/* Load a tile into cache for both A and B */
					for (int w = 0; w < WPT; w++) {
						const int tiledRow = TS * t + row;
						const int tiledCol = TS * t + col;
						Asub[row][col + w*RTS] = A[globalRow * M + tiledCol + w*RTS];
						Bsub[row][col + w*RTS] = B[tiledRow * N + globalCol + w*RTS];
					}
					/* cache-sync */
					item.barrier(access::fence_space::local_space);

					/* Compute result for one element */
					for (int k = 0; k < TS; k++) {
						for (int w = 0; w < WPT; w++)
							acc[w] += Asub[row][k] * Bsub[k][col + w * RTS];
					}
					/* cache-sync */
					item.barrier(access::fence_space::local_space);
				}
				/* store values to C */
				for (int w = 0; w < WPT; w++)
					C[globalRow * M + (globalCol + w*RTS)] = acc[w]; //?
			});
		});
		e.wait();
	}
	catch (sycl::exception const &e) {
		std::cout << "Exception occured while executing the kernel.\n";
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
	std::cout << "Computing CPU results...\n";
	for (size_t i = 0; i < M; i++)
		for (size_t k = 0; k < N; k++)
			for (size_t j = 0; j < P; j++)
				c_host[i * M + j] += a_host[i * M + k] * b_host[k * N + j];
}

template <typename T>
void print_matrix(size_t rows, size_t cols, std::vector<T>& mat) {
	for (int i = 0; i < rows; i++) {
		std::cout << "[ ";
		for (int j = 0; j < cols; j++) {
			std::cout << mat[i * cols + j];
			if (j != cols - 1)
				std::cout << " | ";
		}
		std::cout << " ]\n";
	}
}

template <typename T>
class Verify {
private:
	static bool AreSame(T a, T b) {
		return fabs(a - b) < std::numeric_limits<float>::epsilon();
	}

public:
	static bool VerifyResult(std::vector<T>& c_gpu, std::vector<T>& c_host) {
		PROFILE_FUNCTION();
		std::cout << "Comparing results of CPU and GPU.\n";

		int errors = 0;
		bool mismatched = false;
		for (size_t i = 0; i < c_host.size(); i++) {
			if (!AreSame(c_gpu[i], c_host[i])) {
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
			std::cout << ":( Failed.\n";
			return -1;
		}
	}
};