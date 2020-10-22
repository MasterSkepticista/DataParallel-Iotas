/*
SGEMM optimization using kernel-tricks, on SYCL.
C[M, P] = A[M, N] * B[N, P]
Progress (M = N = P = 4096):
	1. [x] Kernel-1: ~7  GFLOPS | Naive parallelism with gargantuan memory accesses (roofline model)
	2. [x] Kernel-2: ~20 GFLOPS | Tiling blocks of A, B in register/on-die cache. TS=8 is ideal for iGPU
	3. [ ] Kernel-3: ~37 GFLOPS | More Work Per Thread. Reducing the total load/stores
	4. [ ] Kernel-4: 

*/

#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#include "common.h"
#include "nanoblas.h"

// Select namespaces
using namespace sycl;

#define PROFILING 1
#define DEBUG 0
#define VERIFY 1
#define PRECISION float

int main() {
	Instrumentor::Get().BeginSession("GPU MatMul");
	PROFILE_FUNCTION();

	/* set seed */
	srand(39872);

	// Matrix size definitions
	constexpr size_t multiplier = 4096;
	constexpr size_t M = 1 * multiplier;
	constexpr size_t N = 1 * multiplier;
	constexpr size_t P = 1 * multiplier;

	std::vector<PRECISION> b_host(N * P);
	std::vector<PRECISION> a_host(M * N);
	std::vector<PRECISION> c_host(M * P);

	std::vector<PRECISION> c_gemm1(M * P);
	std::vector<PRECISION> c_gemm2(M * P);
	std::vector<PRECISION> c_gemm3(M * P);

	for (auto& item : a_host) { item = rand() % 5; }
	for (auto& item : b_host) { item = rand() % 5; }
	/*
	Now the standard procedure of 
		1. Create a device queue with d_selector and exception handler
		2. Call the kernel enclosed in function
		3. Enclose all within a try-catch block
	*/
	try {
		queue q = create_device_queue();
		/* Kernel-1 Basic Parallel method with too many memory accesses */
		MatrixMulParallelNaive<PRECISION>(q, M, N, P, a_host, b_host, c_gemm1);	
		/* Kernel-2 8x8 tiled method */
		MatrixMulTiled<PRECISION>(q, M, N, P, a_host, b_host, c_gemm2);
		/* Kernel-3 Tiling + WPT */
		MatrixMulWPT<PRECISION>(q, M, N, P, a_host, b_host, c_gemm3);

	}
	catch (std::exception const& e) {
		std::cout << "Exception while multiplying on GPU.\n";
	}

	#if DEBUG 
	std::cout << "A_host\n";
	print_matrix<float>(M, N, a_host);
	std::cout << "B_host\n";
	print_matrix<float>(N, P, b_host);

	std::cout << "C_gemm1\n";
	print_matrix<float>(M, P, c_gemm1);
	std::cout << "C_gemm2\n";
	print_matrix<float>(M, P, c_gemm2);
	std::cout << "C_gemm3\n";
	print_matrix<float>(M, P, c_gemm3);
	#endif

	#if VERIFY
	MatrixMulCPU<PRECISION>(M, N, P, a_host, b_host, c_host);
	Verify<PRECISION>::VerifyResult(c_gemm1, c_host);
	Verify<PRECISION>::VerifyResult(c_gemm2, c_host);
	Verify<PRECISION>::VerifyResult(c_gemm3, c_host);
	#endif
	Instrumentor::Get().EndSession();
	return 0;
}

