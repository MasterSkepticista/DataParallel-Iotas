/*
DataParallel Matrix Mulitplication using device buffers.
C[M, P] = A[M, N] * B[N, P]
TODO: 
	1. [x] Create init functions: Not possible, since it requires P in float(*arr)[P] to be a constant across all arrays passed. Use std::vector.
	2. [x] Change array sizes to be arbitrary and not just multiples of 8.
	3. [x] Understand what is float(*c_gpu)[P]. DONE: float() is a typecast. [P] is number of pointers for each 1D arr. We pass by reference here.
	4. [x] WHY reinterpret_cast? Can you use normal <float, 2> to see if it is same?: Not same, reinterpret_cast converts a pointer of one datatype to another. 
			vector.data() gives you the pointer to that vector start.
			Hence, reinterpret_cast<float *>(vector1.data()) is what you need, to get <float> type buffer.
	5. [x] Use Eigen Matrix instead of Vector
	6. [x] Can you use multiple h.parallel_for(..) to initialize both arrays in one group? No, try not to. Create different cgh's
	7. [ ] Calculate FLOPS 
	8. [ ] Calculate Memory Accesses
*/

#include <CL/sycl.hpp>
#include <Eigen/Dense>
#include <iostream>
#include "dpc_common.hpp"
#include <limits>

// For timing functions
#include "common.h"

// Select namespaces
using namespace std::this_thread;
using namespace std::chrono;
using namespace Eigen;
using namespace sycl;

#define PROFILING 1
#if PROFILING
#define PROFILE_SCOPE(name) InstrumentationTimer timer##__LINE__(name) // TOO SMART. How to give same timernames in one function. You append LINE_NUMBER!!!
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCSIG__) // __FUNCTION__ means function name, but overloading is useless here
#endif

bool VerifyResult(Matrix<double, -1, -1, RowMajor>& c_gpu, Matrix<double, -1, -1, RowMajor>& c_host);
bool AreSame(double a, double b) {
	return fabs(a - b) < std::numeric_limits<float>::epsilon();
}

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

void MatrixMulParallel(queue& q, Matrix<double, -1, -1, RowMajor>& a_host,
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

int main() {

	Instrumentor::Get().BeginSession("GPU MatMul");
	PROFILE_FUNCTION();
	// Matrix size definitions
	constexpr size_t multiplier = 800;
	std::cout << "Specify a workload multiplier:\n";
	//std::cin >> multiplier;
	constexpr size_t M = 1 * multiplier;
	constexpr size_t N = 2 * multiplier;
	constexpr size_t P = 4 * multiplier;

	// MatrixXd: X means X-dim, d means double
	Matrix<double, -1, -1, RowMajor> a_host = Matrix<double, -1, -1>::Ones(M, N);
	Matrix<double, -1, -1, RowMajor> b_host = Matrix<double, -1, -1>::Random(N, P) * 10;
	Matrix<double, -1, -1, RowMajor> c_host = Matrix<double, -1, -1>::Zero(M, P);

	// Device write-back placeholder
	Matrix<double, -1, -1, RowMajor> c_gpu = Matrix<double, -1, -1>::Zero(M, P);
	
	/*
	Now the standard procedure of 
		1. Create a device queue with d_selector and exception handler
		2. Call the kernel enclosed in function
		3. Enclose all within a try-catch block
	*/
	try {
		queue q = create_device_queue();
		// Multiply parallel two matrices
		MatrixMulParallel(q, a_host, b_host, c_gpu);	
	}
	catch (std::exception const& e) {
		std::cout << "Exception while multiplying on GPU.\n";
	}

	// Now that you are out of try {..}, buffer destructors will copy back the data.
	std::cout << "Computation on GPU finished.\nSwitching to CPU...\n";
	std::cout << "CPU::Multiplying A and B into C.\n";
	
	InstrumentationTimer *t = new InstrumentationTimer("CPU run");
	c_host = a_host * b_host;
	delete t;
	
	bool result = VerifyResult(c_gpu, c_host);
	Instrumentor::Get().EndSession();
	return result;
}

bool VerifyResult(Matrix<double, -1, -1, RowMajor>& c_gpu, Matrix<double, -1, -1, RowMajor>& c_host) {
	PROFILE_FUNCTION();
	std::cout << "CPU::Comparing results of CPU and GPU.\n";

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
		std::cout << "Passed! Pat yourself dude.";
		return 0;
	}
	else {
		std::cout << ":( Failed.";
		return -1;
	}
}

