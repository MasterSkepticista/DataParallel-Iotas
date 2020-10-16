#pragma once
#include <CL/sycl.hpp>
#include <Eigen/Dense>
#include "common.h"

using namespace Eigen;
using namespace sycl;

#define PROFILING 1
#if PROFILING
#define PROFILE_SCOPE(name) InstrumentationTimer timer##__LINE__(name) // TOO SMART. How to give same timernames in one function. You append LINE_NUMBER!!!
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCSIG__) // __FUNCTION__ means function name, but overloading is useless here
#endif

void MatrixMulParallelNaive(queue& q, Matrix<double, -1, -1, RowMajor>& a_host,
	Matrix<double, -1, -1, RowMajor>& b_host,
	Matrix<double, -1, -1, RowMajor>& c_gpu) {
	/*
		1. Create buffers for array
		2. Create a command group containing kernel as a lambda
		3. Give access permissions inside the submission
	*/
	PROFILE_FUNCTION();
	try {
		size_t M = a_host.rows();
		size_t N = a_host.cols();
		size_t P = b_host.cols();
		buffer<double, 2> a(a_host.data(), range<2>{M, N});
		buffer<double, 2> b(b_host.data(), range<2>{N, P});
		buffer<double, 2> c(c_gpu.data(), range<2>{M, P});
		PROFILE_SCOPE("Starting Multiply on GPU");
		std::cout << "GPU::Multiplying A and B into C.\n";
		auto e = q.submit([&](handler& h) {

			auto A = a.get_access<access::mode::read>(h);
			auto B = b.get_access<access::mode::read>(h);
			auto C = c.get_access<access::mode::write>(h);
			// buffer.get_range() returns the dimensions (shape) of the buffer/array
			int num_cols_a = a.get_range()[1];

			h.parallel_for(range<2>{M, P}, [=](id<2> index) {
				// index[0] allows accessing ROW index, index[1] is column index
				// Also called thread identifiers
				int row = index[0];
				int col = index[1];
				auto sum = 0.0;
				// Compute result of ONE element of C
				for (int i = 0; i < num_cols_a; i++)
					sum += A[row][i] * B[i][col];
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

// Function set to verify results of different kernels you implement
bool AreSame(double a, double b) {
	return fabs(a - b) < std::numeric_limits<float>::epsilon();
}


bool VerifyResult(Matrix<double, -1, -1, RowMajor>& c_gpu, Matrix<double, -1, -1, RowMajor>& c_host) {
	PROFILE_FUNCTION();
	std::cout << "Comparing results of CPU and GPU.\n";

	int errors = 0;
	bool mismatched = false;
	for (int i = 0; i < c_gpu.rows(); i++) {
		for (int j = 0; j < c_gpu.cols(); j++) {
			if (!AreSame(c_gpu(i, j), c_host(i, j))) {
				std::cout << "Unexpected Result for [" << i << "][" << j << "]:"
					<< " Expected -> " << c_host(i, j)
					<< " Computed -> " << c_gpu(i, j) << "\n";
				errors++;
				mismatched = true;
			} if (errors == 5) break;
		}
		if (errors == 5) break;
	}

	// Need not worry about gc, they are destroyed at end of scope.
	if (!mismatched) {
		std::cout << ":) Results Match.\n";
		return 0;
	}
	else {
		std::cout << ":( Failed.";
		return -1;
	}
}
