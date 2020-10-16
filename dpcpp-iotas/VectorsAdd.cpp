/*
DataParallel Addition of two Vectors using device buffers.
*/

#include <CL/sycl.hpp>
#include <array>
#include <iostream>

using namespace cl::sycl;
using namespace std;

/*
Define short definitions for quickly modifying buffer R/W permissions
*/
constexpr access::mode dp_read = access::mode::read;
constexpr access::mode dp_write = access::mode::write;

/*
Create an IntArray of some size
*/
constexpr size_t array_size = 100000;
typedef array<int, array_size> IntArray;

/*
Function to initialize array with the same value as its index
*/
void initialize_array(IntArray& a) {
	for (size_t i = 0; i < a.size(); i++) a[i] = i;
}

/*
Create a device queue using either default or specific selector
*/
queue create_device_queue() {
	// Whichever found fastest will be chosen as default
	default_selector dselector;

	// Create an asynchronous exception handler for graceful termination
	auto ehandler = [](cl::sycl::exception_list exceptionList) {
		// Iterate through each exception
		for (std::exception_ptr const& e : exceptionList) {
			try {
				std::rethrow_exception(e);
			}
			catch (cl::sycl::exception const& e) {
				std::cout << "Asynchronous DPC++ exception!\n";
				std::terminate();
			}
		}
	};

	try {
		// Queue needs 2 things: Device and Exception handler
		queue q(dselector, ehandler);
		// Assuming successful
		return q;
	}
	catch(cl::sycl::exception const &e) {
		std::cout << "Exception while creating Queue. Terminating...\n";
		std::terminate();
	}
}

void VectorAddParallel(const IntArray& x, const IntArray& y, const IntArray& parallel_sum) {
	// Create a device queue.
	queue q = create_device_queue();

	/*
	No exception means probably successful in creating one.
	Print out the device info.
	*/
	std::cout << "Accelerator: " << q.get_device().get_info<info::device::name>() << "\n";
	
	// Create a range object for the arrays managed by buffer.
	// From what is seen so far, it only holds the size of the data defined by array_size
	cl::sycl::range<1> num_items{ array_size };
	
	/*
	Create buffers for x, y and parallel_sum 
	Syntax:
		buffer<dtype, dim> buf_name(pointer_of_data, size of the data);
	*/
	buffer<int, 1> x_buf(x.data(), num_items);
	buffer<int, 1> y_buf(y.data(), num_items);
	buffer<int, 1> sum_buf(parallel_sum.data(), num_items);

	/*
	Submit a command group (like SIMD) to the queue by a lambda
	which contains data access permissions and device computation
	*/
	q.submit([&](handler& h) {
		/*
		Permission accessors: Because accessors are the only way to r/w to buffers
		*/
		auto x_accessor = x_buf.get_access<dp_read>(h);
		auto y_accessor = y_buf.get_access<dp_read>(h);
		auto sum_accessor = sum_buf.get_access<dp_write>(h);

		/*
		Use a parallel for to execute lambda, the kernel, on device.
		*/
		h.parallel_for(num_items, [=](id<1> i) {
			sum_accessor[i] = x_accessor[i] + y_accessor[i];
		});
	});

	/*
	queue runs the kernel asynchronously. Once beyond the scope,
	buffers' data is copied back to the host.
	*/
}

int main() {
	IntArray x, y, sequential, parallel;

	/*
	Init arrays. They are passed by reference
	*/
	initialize_array(x);
	initialize_array(y);

	/*
	Do the sequential, which is supposed to be slow
	*/
	cout << "Adding on CPU (Scalar)\n";
	for (size_t i = 0; i < x.size(); i++) {
		sequential[i] = x[i] + y[i];
	}
	cout << "Done on CPU (Scalar)\n";

	/*
	Do the parallel 
	*/
	cout << "Adding on GPU (Parallel)\n";
	VectorAddParallel(x, y, parallel);
	cout << "Done on GPU (Parallel)\n";

	/*
	Verify results, the old-school way
	*/
	for (size_t i = 0; i < x.size(); i++) {
		if (parallel[i] != sequential[i]) {
			std::cout << "Failed. Results do not match.\n";
			return -1;
		}
	}
	std::cout << "Success!\n";
	return 0;
}