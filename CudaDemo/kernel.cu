#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>
#include <ctime>

using namespace std;

#define THREAD_NUM 256

#define VEC_TOTAL 19545
#define VEC_SIZE 2960
#define NUM_CENTROID 10

__global__ void mul(int n, float *vec, float *centroid, float *distances)
{
	//int i_index = blockIdx.x;
	//int i_stide = gridDim.x;
	//int j_index = threadIdx.x;
	//int j_stride = blockDim.x;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride) {
		int v_index = i / NUM_CENTROID;
		int c_index = i % NUM_CENTROID;
		distances[i] = 0;
		float temp = 0;
		for (int j = 0; j < VEC_SIZE; j++) {
			temp = vec[v_index * VEC_SIZE + j] - centroid[c_index * VEC_SIZE + j];
			distances[i] += temp * temp;
		}
	}
}

__global__ void cluster(int n, int *indices, float *distances)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride) {
		float min_distance = VEC_SIZE;
		int min_index = -1;
		for (int j = 0; j < NUM_CENTROID; j++) {
			float temp = distances[i * NUM_CENTROID + j];
			if (temp < min_distance) {
				min_distance = temp;
				min_index = j;
			}
		}
		indices[i] = min_index;
	}
}

__global__ void check_equal(int n, int *indices1, int *indices2, bool *equal)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride) {
		if (indices1[i] != indices2[i]) {
			*equal = false;
		}
	}
}

__global__ void clear_centroid(int n, float *centroid)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride) {
		centroid[i] = 0;
	}
		
}

__global__ void add(int n, float *centroid, float *vec)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride) {
		centroid[i] = centroid[i] + vec[i];
	}
}

__global__ void div(int n, float *centroid, int *centroid_count)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride) {
		centroid[i] /= centroid_count[i / VEC_SIZE];
	}
}

int main(void)
{
	time_t start, end;
	int count = 0;
	bool active = true;

	float *vec, *centroid;
	int *indices, *indices1, *indices2, *centroid_count;
	bool *equal;
	cudaMallocManaged(&vec, VEC_TOTAL * VEC_SIZE * sizeof(float));
	cudaMallocManaged(&centroid, NUM_CENTROID * VEC_SIZE * sizeof(float));
	cudaMallocManaged(&indices1, VEC_TOTAL * sizeof(int));
	cudaMallocManaged(&indices2, VEC_TOTAL * sizeof(int));
	cudaMallocManaged(&centroid_count, NUM_CENTROID * sizeof(int));
	cudaMallocManaged(&equal, sizeof(bool));
	
	ifstream file;
	file.open("vector.txt");
	for (int i = 0; i < NUM_CENTROID; i++) {
		for (int j = 0; j < VEC_SIZE; j++) {
			file >> vec[i * VEC_SIZE + j];
			centroid[i * VEC_SIZE + j] = vec[i * VEC_SIZE + j];
		}
		indices1[i] = 0;
		indices2[i] = 0;
	}
	for (int i = NUM_CENTROID; i < VEC_TOTAL; i++) {
		cout << i << endl;
		for (int j = 0; j < VEC_SIZE; j++) {
			file >> vec[i * VEC_SIZE + j];
		}
		indices1[i] = 0;
		indices2[i] = 0;
	}
	file.close();

	time(&start);
	while (true) {
		cout << ++count << endl;
		indices = active ? indices1 : indices2;
		float *dist, *distances;
		cudaMallocManaged(&distances, VEC_TOTAL * NUM_CENTROID * sizeof(float));
		mul << <(VEC_TOTAL * NUM_CENTROID + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM >> >(VEC_TOTAL * NUM_CENTROID, vec, centroid, distances);
		cudaDeviceSynchronize();
		cluster << <(VEC_TOTAL + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM >> >(VEC_TOTAL, indices, distances);
		cudaDeviceSynchronize();
		//cout << distances[(VEC_TOTAL - 1) * NUM_CENTROID + NUM_CENTROID - 2] << endl;
		//cout << indices1[9] << endl;
		cudaFree(distances);
		*equal = true;
		check_equal << <(VEC_TOTAL + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM >> > (VEC_TOTAL, indices1, indices2, equal);
		cudaDeviceSynchronize();
		if (*equal) break;

		clear_centroid << <(NUM_CENTROID * VEC_SIZE + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM >> > (NUM_CENTROID * VEC_SIZE, centroid);
		cudaDeviceSynchronize();
		for (int i = 0; i < NUM_CENTROID; i++) {
			centroid_count[i] = 0;
		}
		/*for (int i = 0; i < VEC_TOTAL; i++) {
			int index = indices[i];
			centroid_count[index]++;
			add << <(VEC_SIZE + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM >> >(VEC_SIZE, &centroid[index * VEC_SIZE], &vec[i * VEC_SIZE]);
			cudaDeviceSynchronize();
		}*/
		for (int i = 0; i < VEC_TOTAL; i++) {
			int index = indices[i];
			centroid_count[index]++;
			for (int j = 0; j < VEC_SIZE; j++) {
				centroid[index * VEC_SIZE + j] += vec[i * VEC_SIZE + j];
			}
		}
		div << <(NUM_CENTROID * VEC_SIZE + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM >> >(NUM_CENTROID * VEC_SIZE, centroid, centroid_count);
		cudaDeviceSynchronize();
		active = !active;
		time(&end);
		cout << "time: " << end - start << endl;
	}
	cout << centroid[0] << endl;

	cudaFree(vec);
	cudaFree(centroid);
	cudaFree(indices1);
	cudaFree(indices2);
	cudaFree(centroid_count);
	cudaFree(equal);

	return 0;
}
