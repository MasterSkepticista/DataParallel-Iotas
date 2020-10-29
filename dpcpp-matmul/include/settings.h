#pragma once

#define SIZE 4096

// For kernels 2, 3, 4
#define TS 16				// Tile Size

// For kernel 3 
#define WPT 2				// Work-per-Thread
#define RTS TS/WPT			// Reduced tile size

// For kernel 4 (wider load/stores)
#define WIDTH 2
#if WIDTH == 1
typedef float floatX;
#elif WIDTH == 2
typedef sycl::float2 floatX;
#elif WIDTH == 4
typedef sycl::float4 floatX;
#elif WIDTH == 8
typedef sycl::float8 floatX;
#endif
