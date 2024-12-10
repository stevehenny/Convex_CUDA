#ifndef QUICKHULL_CUH
#define QUICKHULL_CUH

#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <unordered_set>
#include <vector>

using namespace std;
// Structure representing a 2D point
struct Point
{
  double x;
  double y;

  // Default constructor
  __host__ __device__ Point() : x(0), y(0)
  {
  }

  // Parameterized constructor
  __host__ __device__ Point(double x, double y) : x(x), y(y)
  {
  }

  struct Hash
  {
    __host__ __device__ std::size_t operator()(const Point &p) const
    {
      std::size_t xHash = static_cast<std::size_t>(p.x * 1000000);
      std::size_t yHash = static_cast<std::size_t>(p.y * 1000000);
      return xHash ^ (yHash << 1);
    }
  };

  __host__ __device__ bool operator<(const Point &other) const
  {
    if (x == other.x)
      return y < other.y;
    return x < other.x;
  }

  __host__ __device__ bool operator>(const Point &other) const
  {
    return !(*this < other) && !(*this == other);
  }

  __host__ __device__ bool operator==(const Point &other) const
  {
    return x == other.x && y == other.y;
  }
};
// Structure representing a data block for storing index and distance
struct DataBlock
{
  int index;
  double distance;

  __host__ __device__ DataBlock()
  {
  }
  __host__ __device__ DataBlock(int _index, double _distance) : index(_index), distance(_distance)
  {
  }
};

// Host and device functions for geometry calculations
__host__ __device__ double Distance(Point p1, Point p2, Point p);
__host__ __device__ int Side(Point p1, Point p2, Point p);

// Device functions for atomic assignments
__device__ void atomicAssign(int *var, int val);
__device__ void atomicAssigndouble(double *var, double val);

// Device and host functions for finding maximum index
__device__ void FindMaxIndexGPU(Point a, Point b, Point candidate, int side, int candidateIndex,
                                int *maxIndex, double *maxDistance);
__host__ void FindMaxIndexCPU(Point a, Point b, Point candidate, int side, int candidateIndex,
                              int *maxIndex, double *maxDistance);

// CUDA kernel for finding the maximum index
__global__ void FindMaxIndexKernel(Point *input, int length, int side, Point a, Point b,
                                   DataBlock *output);

// Host function for QuickHull's recursive hull finding
void FindHull(std::vector<Point> &points, Point p1, Point p2, int side,
              std::unordered_set<Point, Point::Hash> &result, Point *d_points);

// Host function for QuickHull algorithm
std::unordered_set<Point, Point::Hash> QuickHull(std::vector<Point> &points);

// Function to compute the cross product of vectors (p0p1 and p0p2)
// Positive result indicates counter-clockwise turn, negative indicates clockwise turn
double crossProduct(const Point &p0, const Point &p1, const Point &p2);

// Function to find the point with the lowest y-coordinate (or leftmost if there are ties)
Point findLeftmostPoint(const std::vector<Point> &points);

// Comparator function to sort points in clockwise order around the reference point
bool compareClockwise(const Point &p1, const Point &p2, const Point &reference);

#endif // QUICKHULL_CUH
