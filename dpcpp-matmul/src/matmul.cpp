/*
SGEMM optimization using kernel-tricks, on SYCL.
C[M, P] = A[M, N] * B[N, P]
Progress (M = N = P = 4096):
	1. [x] Kernel-1: ~7  GFLOPS | Naive parallelism with gargantuan memory accesses (roofline model)
	2. [x] Kernel-2: ~15 GFLOPS | Tiling blocks of A, B in register/on-die cache. TS=8 is ideal for iGPU
	3. [x] Kernel-3: ~25 GFLOPS | More Work Per Thread. Reducing the total load/stores
	4. [x] Kernel-4: ~27 GFLOPS | Wider Load/Store. No WPT 

*/

#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#include "common.h"
#include "nanoblas.h"

#if SIZE <= 16
#define DEBUG 1
#endif
#define VERIFY 1

int main() {
	pfr::Instrumentor::Get().BeginSession("GPU MatMul");
	PROFILE_FUNCTION("time");

	/* set seed */
	srand(39872);

	// Matrix size definitions
	constexpr size_t M = 1 * SIZE;
	constexpr size_t N = 1 * SIZE;
	constexpr size_t P = 1 * SIZE;

	// float ptr matrix. &a_host points to the data.
	float* a_host = (float*)malloc(M * N * sizeof(float*));
	float* b_host = (float*)malloc(N * P * sizeof(float*));

	float* c_gemm = (float*)malloc(M * P * sizeof(float*));
	float* c_gemm2 = (float*)malloc(M * P * sizeof(float*));
	float* c_gemm3 = (float*)malloc(M * P * sizeof(float*));
	float* c_gemm4 = (float*)malloc(M * P * sizeof(float*));
	
	for (size_t i = 0; i < M * N; i++) { a_host[i] = rand() % 5; }
	for (size_t i = 0; i < N * P; i++) { b_host[i] = rand() % 5; }

	/*
	Now the standard procedure of 
		1. Create a device queue with d_selector and exception handler
		2. Call the kernel enclosed in function
		3. Enclose all within a try-catch block
	*/
	#if true
	try {
		sycl::queue q = create_device_queue();
		/* Kernel-1 Basic Parallel method with too many memory accesses */
		MatrixMulParallelNaive(q, M, N, P, a_host, b_host, c_gemm);	
		/* Kernel-2 8x8 tiled method */
		MatrixMulTiled(q, M, N, P, a_host, b_host, c_gemm2);
		MatrixMulTiled(q, M, N, P, a_host, b_host, c_gemm2);
		MatrixMulTiled(q, M, N, P, a_host, b_host, c_gemm2);
		/* Kernel-3 Tiling + WPT */
		MatrixMulWPT(q, M, N, P, a_host, b_host, c_gemm3);
		MatrixMulWPT(q, M, N, P, a_host, b_host, c_gemm3);
		MatrixMulWPT(q, M, N, P, a_host, b_host, c_gemm3);
		/* Kernel-4 Tiling + Wide WPT */
		MatrixMulWideWPT(q, M, N, P, a_host, b_host, c_gemm4);
		MatrixMulWideWPT(q, M, N, P, a_host, b_host, c_gemm4);
		MatrixMulWideWPT(q, M, N, P, a_host, b_host, c_gemm4);

	}
	catch (std::exception const& e) {
		std::cout << "Exception while multiplying on GPU.\n";
	}
	#endif

	#if DEBUG
	std::cout << "A_host\n";
	print_matrix(M, N, a_host);
	std::cout << "B_host\n";
	print_matrix(N, P, b_host);

	/*floatX* a_hostX = reinterpret_cast<floatX*>(a_host);
	floatX* b_hostX = reinterpret_cast<floatX*>(b_host);
	std::cout << "A_hostX\n";
	print_matrix(M, N, a_hostX);
	std::cout << "B_hostX\n";
	print_matrix(N, P, b_hostX);*/

	std::cout << "C_gemm1\n";
	print_matrix(M, P, c_gemm);
	std::cout << "C_gemm4\n";
	print_matrix(M, P, c_gemm4);
	#endif

	#if VERIFY
	//float* c_host = (float*)malloc(M * P * sizeof(float*));
	//MatrixMulCPU(M, N, P, a_host, b_host, c_host);
	float* c_host = c_gemm;
	Verify<float>::VerifyResult(M, P, c_gemm, c_host);
	Verify<float>::VerifyResult(M, P, c_gemm2, c_host);
	Verify<float>::VerifyResult(M, P, c_gemm3, c_host);
	Verify<float>::VerifyResult(M, P, c_gemm4, c_host);
	#endif
	pfr::Instrumentor::Get().EndSession();
	return 0;
}

