#include <iostream>
#include <cuda.h>
#include <random>
#include <fstream>
#include <iomanip>


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
        //printf("(%d,%d) out index -> y_index : (%d,%d), x_index : (%d,%d), monitoring (%d, %d) index\n", x_index, y_index, y_index - ((kernel_dim-1)/2), y_index + ((kernel_dim-1)/2), x_index - ((kernel_dim-1)/2), x_index + ((kernel_dim-1)/2), j, i);  
        tid_image = (y_index - ((kernel_dim-1)/2) + i) * img_cols + (x_index - ((kernel_dim-1)/2) + j);   
        tid_kernel = i * kernel_dim + j;   
        //printf("out_index : %d -> tid_image : %d, tid_kernel : %d, condition x : %d, condition y: %d\n", out_index, tid_image, tid_kernel, x_index - ((kernel_dim-1)/2) + j, y_index - ((kernel_dim-1)/2) + i);       
        if(x_index - ((kernel_dim-1)/2) + j >= 0 && x_index - ((kernel_dim-1)/2) + j < img_cols && y_index - ((kernel_dim-1)/2) + i >= 0 && y_index - ((kernel_dim-1)/2) + i < img_rows){
          image_data = img_data[tid_image];
          kernel_data = ker_data[tid_kernel];
          //printf("out_index : %d -> tid_image : %d, tid_kernel : %d, image : %f, filter : %f, sum : %f\n", out_index, tid_image, tid_kernel, image_data, kernel_data, sum);
          //printf("sumb : %.16lf\n",sum);
          sum += image_data * kernel_data;
          //printf("out_index1 : %d -> tid_image : %d, tid_kernel : %d, image : %f, filter : %f, sum : %f\n", out_index, tid_image, tid_kernel, image_data, kernel_data, sum);
          //printf("sum : %.16lf\n",sum);       
        }        
      //}      
    } 
  }

  out_data[out_index] = sum;
}


int main(int argc, char *argv[]) {

  // Read the inputs from command line

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

  float * kernel_data;
  cudaMallocManaged((void**)&kernel_data, kernel_dims * kernel_dims * sizeof(float));
  for (int i = 0; i < kernel_dims * kernel_dims; i++) {
    if (fscanf(kernel_file, "%f", &kernel_data[i]) != 1) {
        printf("Error reading float value\n");
        cudaFree(kernel_data);
        fclose(kernel_file);
        return 1;
    }
  }

  fclose(kernel_file);

  // std::cout << "printing 1d kernel " << std::endl;
  // for (int i = 0; i < kernel_dims; i++) {
  //     for (int j = 0; j < kernel_dims; j++) {
  //         printf("%f\t", kernel_data[i*kernel_dims+j]);
  //     }
  //     printf("\n");
  // }

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

  float * image_data;
  cudaMallocManaged((void**)&image_data, img_rows * image_cols * sizeof(float));
  for (int i = 0; i < image_rows * image_cols; i++) {
    if (fscanf(image_file, "%f", &image_data[i]) != 1) {
        printf("Error reading float value\n");
        cudaFree(image_data);
        fclose(image_file);
        return 1;
    }
  }

  fclose(image_file);
  
  //std::cout << "reading image complete" << std::endl;

  // std::cout << "printing 1d image " << std::endl;
  // for (int i = 0; i < img_rows; i++) {
  //     for (int j = 0; j < img_cols; j++) {
  //       printf("%f\t", image_data[i*img_cols+j]);
  //     }
  //     printf("\n");
  // }

  float * output_data;
  cudaMallocManaged((void**)&output_data, img_rows * image_cols * sizeof(float));

  // Launch the kernel

  //std::cout << "block size : " << 16  << " x " << 16 << std::endl;
  dim3 threadsPerBlock(16,16);
  //std::cout << "number of blocks : " << (image_cols + 15)/16  << " x " << (image_rows + 15)/16 << std::endl;
  dim3 numBlocks((image_cols + 15)/16, (image_rows + 15)/16);
  conv2d_kernel<<<numBlocks, threadsPerBlock>>>(image_data, kernel_data, output_data, kernel_dims, img_rows, img_cols, threadsPerBlock, image_rows, image_cols);
  cudaDeviceSynchronize();


  // Print the output
  for(int i = 0; i < image_rows; ++i){
    for(int j = 0; j < image_cols; ++j){
      printf("%.3f\n",output_data[i * image_cols + j]);
    }
  }

  // Clean up the memory
  cudaFree(image_data);
  cudaFree(kernel_data);
  cudaFree(output_data);

  return 0;
}