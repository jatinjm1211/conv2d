#include <iostream>
#include <cuda.h>
#include <random>
#include <fstream>
#include <iomanip>
#include <cmath>

// The cuda kernel
__global__ void conv2d_kernel(float * img_data, float * ker_data, float * out_data, int kernel_dim, int img_rows, int img_cols, dim3 threadsPerBlock, int image_rows, int image_cols) {

  int tid_kernel;
  int block_base_index = blockIdx.y * blockDim.y * img_cols + blockIdx.x * blockDim.x;
  int tid_image;
  int out_index = block_base_index + threadIdx.y * img_cols + threadIdx.x;  
  int x_index = blockIdx.x * blockDim.x + threadIdx.x;
  int y_index = blockIdx.y * blockDim.y + threadIdx.y;

  float image_data;
  float kernel_data;
  float sum = 0.0;
  //printf("thread : (%d,%d)\t block : (%d,%d)\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
  for(int i = 0; i < kernel_dim; i++){
    for(int j = 0; j < kernel_dim; j++){    
      //if(out_index == 0){
        //printf("(%d,%d) out index -> y_index : (%d,%d), x_index : (%d,%d), monitoring (%d, %d) index\n", x_index, y_index, y_index - int(kernel_dim/2), y_index + int(kernel_dim/2), x_index - int(kernel_dim/2), x_index + int(kernel_dim/2), j, i);  
        tid_image = (y_index - int(kernel_dim/2) + i) * img_cols + (x_index - int(kernel_dim/2) + j);   
        tid_kernel = i * kernel_dim + j;      
        //printf("loop -> tid_image : %d, tid_kernel : %d, img_rows : %d, img_cols : %d\n", tid_image, tid_kernel, img_rows, img_cols);
        if(x_index - int(kernel_dim/2) + j >= 0 && x_index - int(kernel_dim/2) + j < img_cols && y_index - int(kernel_dim/2) + i >= 0 && y_index - int(kernel_dim/2) + i < img_rows){
          image_data = img_data[tid_image];
          kernel_data = ker_data[tid_kernel];
          //printf("sumb : %.16lf\n",sum);
          sum += image_data * kernel_data;
          //printf("sum : %.16lf\n",sum);       
        }        
      //}      
    }
  }

  out_data[out_index] = sum;

}


int main(int argc, char *argv[]) {

  char *image, *kernel;

  // Read the inputs from command line
  image = argv[1];
  kernel = argv[2];

  std::cout << std::fixed << std::setprecision(3);

  FILE* kernel_file = fopen(kernel, "r");
  if (!kernel_file) {
      printf("Error opening file\n");
      return 1;
  }


  int kernel_dims;
  fscanf(kernel_file, "%d", &kernel_dims);
  //printf("kernel dims : %d x %d\n", kernel_dims, kernel_dims);

  std::vector<std::vector<float>> kernel_data(kernel_dims, std::vector<float>(kernel_dims));

  for (int i = 0; i < kernel_dims; i++) {
      for (int j = 0; j < kernel_dims; j++) {
          fscanf(kernel_file, "%f", &kernel_data[i][j]);
          //printf("%f\t", kernel_data[i][j]);
      }
      //printf("\n");
  }

  fclose(kernel_file);
  //std::cout << "printing 1d kernel " << std::endl;
  float * kernel_data_1d = new float[kernel_dims * kernel_dims];
  for (int i = 0; i < kernel_dims; i++) {
      for (int j = 0; j < kernel_dims; j++) {
          kernel_data_1d[i * kernel_dims + j] = kernel_data[i][j];
          //printf("%f\t", kernel_data[i][j]);
      }
      //printf("\n");
  }

  //std::cout << "reading image" << std::endl;

  FILE* image_file = fopen(image, "r");
  if (!image_file) {
      printf("Error opening file\n");
      return 1;
  }

  int image_rows, image_cols, img_rows, img_cols;

  fscanf(image_file, "%d%d", &image_rows, &image_cols);
  img_rows = image_rows;
  img_cols = image_cols;

  std::vector<std::vector<float>> image_data(img_rows, std::vector<float>(img_cols));

  for (int i = 0; i < image_rows; i++) {
      for (int j = 0; j < image_cols; j++) {
        fscanf(image_file, "%f", &image_data[i][j]);
      }
  }

  fclose(image_file);
  //std::cout << "reading image complete" << std::endl;

  //std::cout << "printing 1d image " << std::endl;
  float * image_data_1d = new float[img_rows * img_cols];
  for (int i = 0; i < img_rows; i++) {
      for (int j = 0; j < img_cols; j++) {
          image_data_1d[i * img_cols + j] = image_data[i][j];
      }
  }

  float * image_data_gpu;
  size_t size_img = img_rows * img_cols;
  cudaMalloc((void**)&image_data_gpu, size_img * sizeof(float));
  cudaError_t err = cudaMemcpy(image_data_gpu, image_data_1d, size_img * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "CUDA error image: " << cudaGetErrorString(err) << std::endl;
}

  float * kernel_data_gpu;
  size_t size_ker = kernel_dims * kernel_dims;
  cudaMalloc((void**)&kernel_data_gpu, size_ker * sizeof(float));
  err = cudaMemcpy(kernel_data_gpu, kernel_data_1d, size_ker * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "CUDA error filter: " << cudaGetErrorString(err) << std::endl;
}

  float * output_data_gpu;
  size_t size_out = image_rows * image_cols;
  cudaMalloc((void**)&output_data_gpu, size_out * sizeof(float));

  // Launch the kernel
  size_t sharedMemSize = 2 * kernel_dims * kernel_dims * sizeof(float);
  //std::cout << "block size : " << 16  << " x " << 16 << std::endl;
  dim3 threadsPerBlock(16,16);
  //std::cout << "number of blocks : " << (image_cols + 15)/16  << " x " << (image_rows + 15)/16 << std::endl;
  dim3 numBlocks((image_cols + 15)/16, (image_rows + 15)/16);
  conv2d_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(image_data_gpu, kernel_data_gpu, output_data_gpu, kernel_dims, img_rows, img_cols, threadsPerBlock, image_rows, image_cols);

  float * output_data  = new float[image_rows * image_cols];
  err = cudaMemcpy(output_data, output_data_gpu, size_out* sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::cerr << "CUDA error output: " << cudaGetErrorString(err) << std::endl;
}

  // Print the output
  for(int i = 0; i < image_rows; ++i){
    for(int j = 0; j < image_cols; ++j){
      printf("%.3f\n",output_data[i * image_cols + j]);
    }
  }

  // Clean up the memory
  cudaFree(image_data_gpu);
  cudaFree(kernel_data_gpu);
  cudaFree(output_data_gpu);

  return 0;
}