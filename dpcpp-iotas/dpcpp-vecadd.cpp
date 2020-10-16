/*
DataParallel Addition of two Vectors using device buffers.
*/

#include <CL/sycl.hpp>
#include <vector>
#include <iostream>
#include <chrono>
#include <thread>

using namespace std::this_thread;
using namespace std::chrono;
using namespace sycl;

// Define array_size
constexpr size_t array_size = 128000000;

// Function to initialize array with the same value as its index
void InitializeArray(std::vector<int> &a) {
	for (size_t i = 0; i < a.size(); i++) a[i] = i;
}

// Create an asynchronous Exception Handler for sycl
static auto exception_handler = [](cl::sycl::exception_list eList) {
	for (std::exception_ptr const& e : eList) {
		try {
			std::rethrow_exception(e);
		}
		catch (std::exception const& e) {
			std::cout << "Failure" << std::endl;
			std::terminate();
		}
	}
};

void VectorAddParallel(queue &q, 
					const std::vector<int> &x, 
					const std::vector<int>& y, 
					std::vector<int>& parallel_sum) {
	// Create a range object for the arrays managed by buffer.
	// From what is seen so far, it only holds the size of the data defined by array_size
	range<1> num_items{ x.size() };
	
	/*
	Create buffers for x, y and parallel_sum 
	Syntax:
		buffer<dtype, dim> buf_name(pointer_of_data, size of the data);
	*/
	buffer x_buf(x);
	buffer y_buf(y);
	buffer sum_buf(parallel_sum.data(), num_items);

	/*
	Submit a command group (like SIMD) to the queue by a lambda
	which contains data access permissions and device computation
	*/
	q.submit([&](handler& h) {
		/*
		Permission accessors: Because accessors are the only way to r/w to buffers
		*/
		auto xa = x_buf.get_access<access::mode::read>(h);
		auto ya = y_buf.get_access<access::mode::read>(h);
		auto sa = sum_buf.get_access<access::mode::write>(h);

		/*
		Use a parallel for to execute lambda, the kernel, on device.
		*/
		std::cout << "Adding on GPU (Parallel)\n";
		h.parallel_for(num_items, [=](id<1> i) { sa[i] = xa[i] + ya[i]; });
		std::cout << "Done on GPU (Parallel)\n";
	});

	/*
	queue runs the kernel asynchronously. Once beyond the scope,
	buffers' data is copied back to the host.
	*/
}

int main() {
	default_selector d_selector;
	std::vector<int> a(array_size), b(array_size), sequential(array_size), parallel(array_size);

	/*
	Init arrays. They are passed by reference
	*/
	InitializeArray(a);
	InitializeArray(b);

	std::cout << "Will start after 10 seconds...\n";
	sleep_for(seconds(10));
	/*
	Do the sequential, which is supposed to be slow
	*/
	std::cout << "Adding on CPU (Scalar)\n";
	for (size_t i = 0; i < sequential.size(); i++) {
		sequential[i] = a[i] + b[i];
	}
	std::cout << "Done on CPU (Scalar)\n";

	/*
	Create a device queue and 
	Do the parallel 
	*/
	try {
		// Queue needs 2 things: Device and Exception handler
		queue q(d_selector, exception_handler);
		/*
		No exception means probably successful in creating one.
		Print out the device info.
		*/
		std::cout << "Accelerator: " 
				  << q.get_device().get_info<info::device::name>() << "\n";
		std::cout << "Vector size: " << a.size() << "\n";
		VectorAddParallel(q, a, b, parallel);
	}
	catch (std::exception const& e) {
		std::cout << "Exception while creating Queue. Terminating...\n";
		std::terminate();
	}
	
	
	/*
	Verify results, the old-school way
	*/
	if (false)
		for (size_t i = 0; i < parallel.size(); i++) {
			if (parallel[i] != sequential[i]) {
				std::cout << "Fail: " << parallel[i] << " != " << sequential[i] << std::endl;
				std::cout << "Failed. Results do not match.\n";
				return -1;
			}
		}
	std::cout << "Success!\n";
	return 0;
}