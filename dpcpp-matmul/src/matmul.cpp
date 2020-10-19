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
#include <iostream>
#include "dpc_common.hpp"
#include <limits>
#include <vector>
#include <stdlib.h>

#include "common.h"
#include "nanoblas.h"

// Select namespaces
using namespace sycl;

#define PROFILING 1
#if PROFILING
#define PROFILE_SCOPE(name) InstrumentationTimer timer##__LINE__(name) // TOO SMART. How to give same timernames in one function. You append LINE_NUMBER!!!
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCSIG__) // __FUNCTION__ means function name, but overloading is useless here
#endif


int main() {
	Instrumentor::Get().BeginSession("GPU MatMul");
	PROFILE_FUNCTION();
	
	/* set seed */
	srand(39872);

	// Matrix size definitions
	constexpr size_t multiplier = 4;
	constexpr size_t M = 1 * multiplier;
	constexpr size_t N = 1 * multiplier;
	constexpr size_t P = 1 * multiplier;

	std::vector<int> b_host(N*P);
	std::vector<int> a_host(M*N);
	std::vector<int> c_gpu(M*P);
	std::vector<int> c_host(M*P);

	for (int& item : b_host) { item = rand() % 10; }
	for (int& item : a_host) { item = rand() % 10; }
	/*
	Now the standard procedure of 
		1. Create a device queue with d_selector and exception handler
		2. Call the kernel enclosed in function
		3. Enclose all within a try-catch block
	*/
	try {
		queue q = create_device_queue();
		//MatrixMulParallelNaive<int>(q, M, N, P, a_host, b_host, c_gpu);	
		MatrixMulTiled<int>(q, M, N, P, a_host, b_host, c_gpu);
	}
	catch (std::exception const& e) {
		std::cout << "Exception while multiplying on GPU.\n";
	}

	MatrixMulCPU<int>(M, N, P, a_host, b_host, c_host);
	
	bool result = VerifyResult(c_gpu, c_host);
	Instrumentor::Get().EndSession();
	return result;
}

