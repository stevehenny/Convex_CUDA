#include "convex_hull_cuda.cuh"
#include <GL/glut.h>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <hash_set>
#include <unordered_set>
#include <vector>

#define BLOCK_DIM 256

using namespace std;

__host__ __device__ double Distance(Point p1, Point p2, Point p)
{
  return std::abs((p.y - p1.y) * (p2.x - p1.x) - (p2.y - p1.y) * (p.x - p1.x));
}

// Find which side point p is on in comparison to line segment p1->p2
__host__ __device__ int Side(Point p1, Point p2, Point p)
{
  auto side = (p.y - p1.y) * (p2.x - p1.x) - (p2.y - p1.y) * (p.x - p1.x);
  if (side > 0)
    return 1;
  if (side < 0)
    return -1;
  return 0;
}

// Finds and updates local max distances and indexes
__device__ void FindMaxIndexGPU(Point a, Point b, Point candidate, int side, int candidateIndex,
                                int *maxIndex, double *maxDistance)
{
  int s = Side(a, b, candidate);
  if (s != side)
    return;

  double candidateDistance = Distance(a, b, candidate);

  if (candidateDistance > *maxDistance)
  {
    *maxIndex = candidateIndex;
    *maxDistance = candidateDistance;
  }
}

// Finds the max index but is a host function, only to be called as a final reduction
__host__ void FindMaxIndexCPU(Point a, Point b, Point candidate, int side, int candidateIndex,
                              int *maxIndex, double *maxDistance)
{
  int s = Side(a, b, candidate);
  if (s != side)
    return;

  double candidateDistance = Distance(a, b, candidate);
  if (candidateDistance > *maxDistance)
  {
    *maxIndex = candidateIndex;
    *maxDistance = candidateDistance;
  }
}
// Kernel that finds the max index and does most of the heavy lifting
__global__ void FindMaxIndexKernel(Point *input, int length, int side, Point a, Point b,
                                   DataBlock *output)
{
  // Define a shared memory buffer to hold the results of the thread-level computation
  __shared__ double s_distance[BLOCK_DIM];
  __shared__ int s_index[BLOCK_DIM];

  // Compute the start index for this block
  int start_index = blockIdx.x * blockDim.x;

  // Initialize the thread-level results with default values
  double distance = -1.0f;
  int index = -1;

  // Compute the maximum distance and index for the points assigned to this block
  for (int i = threadIdx.x; i < blockDim.x && start_index + i < length; i += blockDim.x)
  {
    FindMaxIndexGPU(a, b, input[start_index + i], side, start_index + i, &index, &distance);
  }
  __syncthreads();

  s_distance[threadIdx.x] = distance;
  s_index[threadIdx.x] = index;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (threadIdx.x < s)
    {
      if (s_distance[threadIdx.x + s] > distance)
      {
        distance = s_distance[threadIdx.x + s];
        index = s_index[threadIdx.x + s];
        s_index[threadIdx.x] = index;
        s_distance[threadIdx.x] = distance;
      }
    }
    __syncthreads();
  }

  // Only the first thread of each block needs to write the results to global memory
  if (threadIdx.x == 0)
  {
    output[blockIdx.x] = DataBlock(index, distance);
  }
}

// Recursively called divide and conquer function
void FindHull(std::vector<Point> &points, Point p1, Point p2, int side,
              std::unordered_set<Point, Point::Hash> &result, Point *d_points)
{
  // Calculate grid and block dimensions
  dim3 blockDim(BLOCK_DIM);                                    // 16 threads per block
  dim3 gridDim((points.size() + blockDim.x - 1) / blockDim.x); // Adjust grid size

  // Allocate memory for DataBlock on device
  DataBlock *d_output;
  cudaMalloc((void **)&d_output, gridDim.x * sizeof(DataBlock));

  // Kernel call to find maximum index
  FindMaxIndexKernel<<<gridDim, blockDim>>>(d_points, points.size(), side, p1, p2, d_output);
  cudaDeviceSynchronize();

  // Copy results back to host
  std::vector<DataBlock> output(gridDim.x);
  cudaMemcpy(output.data(), d_output, gridDim.x * sizeof(DataBlock), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_output);

  // Determine the maximum index and distance
  int maxIndex = -1;
  double maxDistance = 0.0f;

  for (const auto &candidate : output)
  {
    if (candidate.index < 0) // Skip invalid candidates
      continue;

    // Refine the max index and distance using the CPU
    FindMaxIndexCPU(p1, p2, points[candidate.index], side, candidate.index, &maxIndex,
                    &maxDistance);
  }

  // If no valid point is found, add endpoints to the result
  if (maxIndex == -1)
  {
    result.insert(p1);
    result.insert(p2);
    return;
  }

  // Recur for two partitions
  int newSide = Side(points[maxIndex], p1, p2);
  FindHull(points, points[maxIndex], p1, -newSide, result, d_points);
  FindHull(points, points[maxIndex], p2, newSide, result, d_points);
}

// QuickHull algorithm. Utilizes hashing for quicker access in a unordered_set
// Also calls FindHull on left hull and right hull, which are then recursively called
std::unordered_set<Point, Point::Hash> QuickHull(std::vector<Point> &points)
{

  std::unordered_set<Point, Point::Hash> result;

  Point left = points[0], right = points[0];
  for (int i = 1; i < points.size(); i++)
  {
    if (points[i].x < left.x)
      left = points[i];
    if (points[i].x > right.x)
      right = points[i];
  }

  Point *d_points;
  cudaMalloc((void **)&d_points, points.size() * sizeof(Point));
  cudaMemcpy(d_points, points.data(), points.size() * sizeof(Point), cudaMemcpyHostToDevice);

  FindHull(points, left, right, 1, result, d_points);
  FindHull(points, left, right, -1, result, d_points);

  cudaFree(d_points);
  return result;
}

// Function to compute the cross product of vectors (p0p1 and p0p2)
// Positive result indicates counter-clockwise turn, negative indicates clockwise turn
double crossProduct(const Point &p0, const Point &p1, const Point &p2)
{
  return (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
}

// Function to find the point with the lowest y-coordinate (or leftmost if there are ties)
Point findLeftmostPoint(const std::vector<Point> &points)
{
  Point leftmost = points[0];
  for (const auto &p : points)
  {
    if (p.y < leftmost.y || (p.y == leftmost.y && p.x < leftmost.x))
    {
      leftmost = p;
    }
  }
  return leftmost;
}

// Comparator function to sort points in clockwise order around the reference point
bool compareClockwise(const Point &p1, const Point &p2, const Point &reference)
{
  double cross = crossProduct(reference, p1, p2);
  if (cross == 0)
  {
    // If points are collinear, the one closer to the reference comes first
    return (p1.x - reference.x) * (p1.x - reference.x) +
               (p1.y - reference.y) * (p1.y - reference.y) <
           (p2.x - reference.x) * (p2.x - reference.x) +
               (p2.y - reference.y) * (p2.y - reference.y);
  }
  return cross > 0; // We want a clockwise order
}
