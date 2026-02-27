#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector_types.h>
#include <vector_functions.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "cuda_texture_types.h"

#define IMAGE_DIM 2048
#define SAMPLE_SIZE 6
#define NUMBER_OF_SAMPLES (((SAMPLE_SIZE*2)+1)*((SAMPLE_SIZE*2)+1))

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF     2e10f

void output_image_file(uchar4* image);
void input_image_file(char* filename, uchar4* image);
void checkCUDAError(const char *msg);


__global__ void image_blur(uchar4 *image, uchar4 *image_output) {
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int output_offset = x + y * blockDim.x * gridDim.x;
	uchar4 pixel;
	float4 average = make_float4(0,0,0,0);

	for (int i = -SAMPLE_SIZE; i <= SAMPLE_SIZE; i++){
		for (int j = -SAMPLE_SIZE; j <= SAMPLE_SIZE; j++){
			int x_offset = x + i;
			int y_offset = y + j;
			//wrap bounds
			if (x_offset < 0)
				x_offset += IMAGE_DIM;
			if (x_offset >= IMAGE_DIM)
				x_offset -= IMAGE_DIM;
			if (y_offset < 0)
				y_offset += IMAGE_DIM;
			if (y_offset >= IMAGE_DIM)
				y_offset -= IMAGE_DIM;
			int offset = x_offset + y_offset * blockDim.x * gridDim.x;
			pixel = image[offset];

			//sum values
			average.x += pixel.x;
			average.y += pixel.y;
			average.z += pixel.z;
		}
	}
	//calculate average
	average.x /= (float)NUMBER_OF_SAMPLES;
	average.y /= (float)NUMBER_OF_SAMPLES;
	average.z /= (float)NUMBER_OF_SAMPLES;

	image_output[output_offset].x = (unsigned char)average.x;
	image_output[output_offset].y = (unsigned char)average.y;
	image_output[output_offset].z = (unsigned char)average.z;
	image_output[output_offset].w = 255;
}


/* Host code */

int main(void) {
	unsigned int image_size;
	uchar4 *d_image, *d_image_output;
	uchar4 *h_image;
	cudaEvent_t start, stop;
	float3 ms; //[0]=normal,[1]=tex1d,[2]=tex2d

	image_size = IMAGE_DIM*IMAGE_DIM*sizeof(uchar4);

	// create timers
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// allocate memory on the GPU for the output image
	cudaMalloc((void**)&d_image, image_size);
	cudaMalloc((void**)&d_image_output, image_size);
	checkCUDAError("CUDA malloc");

	// allocate and load host image
	h_image = (uchar4*)malloc(image_size);
	input_image_file("input.png", h_image);

	// copy image to device memory
	cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy to device");

	//cuda layout and execution
	dim3    blocksPerGrid(IMAGE_DIM / 16, IMAGE_DIM / 16);
	dim3    threadsPerBlock(16, 16);

	// normal version
	cudaEventRecord(start, 0);
	image_blur << <blocksPerGrid, threadsPerBlock >> >(d_image, d_image_output);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&ms.x, start, stop);
	checkCUDAError("kernel normal");


	// copy the image back from the GPU for output to file
	cudaMemcpy(h_image, d_image_output, image_size, cudaMemcpyDeviceToHost);
	checkCUDAError("CUDA memcpy from device");

	//output timings
	printf("Execution times:\n");
	printf("\tNormal version: %f\n", ms.x);
	printf("\ttex1D version: %f\n", ms.y);
	printf("\ttex2D version: %f\n", ms.z);

	// output image
	output_image_file(h_image);

	//cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_image);
	cudaFree(d_image_output);
	free(h_image);

	return 0;
}

void output_image_file(uchar4* image)
{
	if (!stbi_write_png("output.png", IMAGE_DIM, IMAGE_DIM, 4, image, IMAGE_DIM * 4)) {
		fprintf(stderr, "Error writing to file 'output.png'\n");
	}
}

void input_image_file(char* filename, uchar4* image)
{
	int width = 0;
	int height = 0;
	int channels = 0;
	void *data = (uchar4 *)stbi_load(filename, &width, &height, &channels, 0);
	if (!image) {
		fprintf(stderr, "Unable to load image '%s', please try a different file.\n", filename);
		exit(EXIT_FAILURE);
	}
	if (channels != 4) {
		fprintf(stderr, "Image is %d channels, must be 4, please try a different file.\n", channels);
		exit(EXIT_FAILURE);
	}
	if (width != IMAGE_DIM || height != IMAGE_DIM) {
		fprintf(stderr, "Image dimensions is %d x %d, should be %d x %d, please try a different file.\n", width, height, IMAGE_DIM, IMAGE_DIM);
		exit(EXIT_FAILURE);
	}
	memcpy(image, data, sizeof(char) * 4 * IMAGE_DIM * IMAGE_DIM);
	stbi_image_free(data);
}
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
