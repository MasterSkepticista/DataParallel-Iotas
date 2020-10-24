#pragma once

// For kernels 2-4
#define TS 16				// Tile Size
#define WPT 2				// Work-per-Thread
#define RTS TS/WPT			// Reduced tile size
