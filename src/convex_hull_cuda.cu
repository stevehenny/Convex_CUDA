#include "convex_hull_general.h"
#include "convex_hull_serial.h"
#include <GL/glut.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fenv.h>
#include <iostream>
#include <iterator>
#include <random>
#include <thread>

#define htkCheck(stmt)                                                                             \
  do                                                                                               \
  {                                                                                                \
    cudaError_t err = stmt;                                                                        \
    if (err != cudaSuccess)                                                                        \
    {                                                                                              \
      std::cerr << "Failed to run stmt: " << #stmt << std::endl;                                   \
      std::cerr << "Got CUDA error (" << err << "): " << cudaGetErrorString(err) << std::endl;     \
      std::cerr << "File: " << __FILE__ << ", Line: " << __LINE__ << std::endl;                    \
      exit(1);                                                                                     \
    }                                                                                              \
  } while (0)

using namespace std;

vector<Point> global_points;
vector<Point> global_hull;

void display()
{
  glClear(GL_COLOR_BUFFER_BIT);
  glColor3f(1.0, 1.0, 1.0);

  // Draw all points
  glBegin(GL_POINTS);
  for (const auto &point : global_points)
  {
    glVertex2f(point.x, point.y);
  }
  glEnd();

  // Draw the hull
  glColor3f(1.0, 0.0, 0.0);
  glBegin(GL_LINE_LOOP);
  for (const auto &point : global_hull)
  {
    glVertex2f(point.x, point.y);
  }
  glEnd();

  glutSwapBuffers();
}

// Timer function for real-time updates
void timer(int)
{
  // Re-render the display
  glutPostRedisplay();

  // Add a delay for visualization purposes
  this_thread::sleep_for(chrono::milliseconds(100));

  // Register the timer callback again
  glutTimerFunc(100, timer, 0);
}

// Initialize OpenGL
void initOpenGL()
{
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0.0, 1000.0, 0.0, 1000.0);
}

// Helper function for atomicMax with double using casting and shared memory
__device__ double atomicMaxDouble(double *address, double val)
{
  unsigned long long *address_as_ull = (unsigned long long *)address;
  unsigned long long old = *address_as_ull, assumed;

  do
  {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(max(val, __longlong_as_double(assumed))));
  } while (assumed != old);

  return __longlong_as_double(old);
}

// Kernel to find the farthest point
__global__ void findFarthestPoint(Point *points, int num_points, Point lineStart, Point lineEnd,
                                  int *farthestIdx, double *maxDist)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_points)
    return;

  double crossProduct = fabs((lineEnd.x - lineStart.x) * (lineStart.y - points[idx].y) -
                             (lineStart.x - points[idx].x) * (lineEnd.y - lineStart.y));
  atomicMaxDouble(maxDist, crossProduct);

  if (crossProduct == *maxDist)
  {
    *farthestIdx = idx;
  }
}

// Kernel to classify points as left or right of the line
__global__ void classifyPoints(Point *points, int num_points, Point lineStart, Point lineEnd,
                               Point farthest, Point *leftSubset, Point *rightSubset,
                               int *leftCount, int *rightCount)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_points)
    return;

  double crossProduct = (lineEnd.x - lineStart.x) * (lineStart.y - points[idx].y) -
                        (lineStart.x - points[idx].x) * (lineEnd.y - lineStart.y);

  if (crossProduct > 0)
  {
    int pos = atomicAdd(leftCount, 1);
    leftSubset[pos] = points[idx];
  }
  else if (crossProduct < 0)
  {
    int pos = atomicAdd(rightCount, 1);
    rightSubset[pos] = points[idx];
  }
}

// Host function to manage recursion and kernel launches
void divide_kernel_caller(Point *points, int num_points, Point lineStart, Point lineEnd,
                          Point *hull, int *hullCount)
{
  if (num_points == 0)
  {
    return;
  }

  // Allocate device memory
  Point *d_points, *leftSubset, *rightSubset;
  int *d_leftCount, *d_rightCount, *d_farthestIdx;
  double *d_maxDist;

  htkCheck(cudaMalloc(&d_points, num_points * sizeof(Point)));
  cout << "malloced d_points" << endl;

  htkCheck(cudaMemcpy(d_points, points, num_points * sizeof(Point), cudaMemcpyHostToDevice));

  htkCheck(cudaMalloc(&leftSubset, num_points * sizeof(Point)));
  htkCheck(cudaMalloc(&rightSubset, num_points * sizeof(Point)));
  htkCheck(cudaMalloc(&d_leftCount, sizeof(int)));
  htkCheck(cudaMalloc(&d_rightCount, sizeof(int)));
  htkCheck(cudaMalloc(&d_farthestIdx, sizeof(int)));
  htkCheck(cudaMalloc(&d_maxDist, sizeof(double)));

  // Initialize counts
  htkCheck(cudaMemset(d_leftCount, 0, sizeof(int)));
  htkCheck(cudaMemset(d_rightCount, 0, sizeof(int)));
  htkCheck(cudaMemset(d_maxDist, 0, sizeof(double)));

  cout << "Malloced everything" << endl;
  // Kernel configuration
  dim3 grid((num_points + 255) / 256);
  dim3 block(256);

  // Find the farthest point
  findFarthestPoint<<<grid, block>>>(d_points, num_points, lineStart, lineEnd, d_farthestIdx,
                                     d_maxDist);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
  htkCheck(cudaDeviceSynchronize());
  // cout << "successfully synced" << endl;

  // Retrieve results
  int farthestIdx;
  double maxDist;
  htkCheck(cudaMemcpy(&farthestIdx, d_farthestIdx, sizeof(int), cudaMemcpyDeviceToHost));
  htkCheck(cudaMemcpy(&maxDist, d_maxDist, sizeof(double), cudaMemcpyDeviceToHost));

  if (maxDist == 0)
  {
    htkCheck(cudaFree(d_points));
    htkCheck(cudaFree(leftSubset));
    htkCheck(cudaFree(rightSubset));
    htkCheck(cudaFree(d_leftCount));
    htkCheck(cudaFree(d_rightCount));
    htkCheck(cudaFree(d_farthestIdx));
    htkCheck(cudaFree(d_maxDist));
    return;
  }

  Point farthest = points[farthestIdx];
  hull[(*hullCount)++] = farthest;

  // Classify points
  classifyPoints<<<grid, block>>>(d_points, num_points, lineStart, lineEnd, farthest, leftSubset,
                                  rightSubset, d_leftCount, d_rightCount);
  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
  htkCheck(cudaDeviceSynchronize());
  cout << "Successfully synced" << endl;

  // Retrieve counts
  int leftCount, rightCount;
  htkCheck(cudaMemcpy(&leftCount, d_leftCount, sizeof(int), cudaMemcpyDeviceToHost));
  htkCheck(cudaMemcpy(&rightCount, d_rightCount, sizeof(int), cudaMemcpyDeviceToHost));

  // Recursively process left and right subsets
  if (leftCount > 0)
  {
    divide_kernel_caller(rightSubset, rightCount, farthest, lineEnd, hull, hullCount);
  }

  // Free device memory
  htkCheck(cudaFree(d_points));
  htkCheck(cudaFree(leftSubset));
  htkCheck(cudaFree(rightSubset));
  htkCheck(cudaFree(d_leftCount));
  htkCheck(cudaFree(d_rightCount));
  htkCheck(cudaFree(d_farthestIdx));
  htkCheck(cudaFree(d_maxDist));
}

int main(int argc, char *argv[])
{
  Config config;
  int exit_status = parse_args(argc, argv, &config);
  if (exit_status != 0)
  {
    return 1;
  }

  cout << config.num_points << endl;
  cout << config.command << endl;
  int case_id;
  if (strcmp(config.command, "both") == 0)
    case_id = 0;
  else if (strcmp(config.command, "serial") == 0)
    case_id = 1;
  else if (strcmp(config.command, "parallel") == 0)
    case_id = 2;
  else
  {
    cerr << "Invalid command" << endl;
    return 1;
  }

  cout << "case_id: " << case_id << endl;

  global_points = generate_random_points(config.num_points);

  cout << "Generated points" << endl;

  // Sort points by x value, y value is tie breaker
  sort(global_points.begin(), global_points.end(),
       [](const Point &a, const Point &b) { return (a.x < b.x) || (a.x == b.x && a.y < b.y); });

  // Initialize OpenGL
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
  glutInitWindowSize(800, 800);
  glutCreateWindow("Convex Hull Visualization");

  initOpenGL();
  cout << "Window initialized" << endl;

  vector<Point> serial_hull, parallel_hull;
  std::chrono::time_point<std::chrono::high_resolution_clock> serial_start, serial_end,
      parallel_start, parallel_end;
  double serial_time, parallel_time;
  int hullCount;

  cout << "Variables needed for case statements declared" << endl;

  // NOTE: WHERE I LEFT OFF: Implement the switch statment logic for the command
  switch (case_id)
  {
  case 0: // both case
    serial_start = std::chrono::high_resolution_clock::now();
    serial_hull = divide(global_points);
    serial_end = std::chrono::high_resolution_clock::now();
    serial_time = std::chrono::duration<double>(serial_end - serial_start).count();
    break;
  case 1: // serial case
    serial_start = std::chrono::high_resolution_clock::now();
    serial_hull = divide(global_points);
    serial_end = std::chrono::high_resolution_clock::now();
    serial_time = std::chrono::duration<double>(serial_end - serial_start).count();
    global_hull = serial_hull;
    break;
  case 2: // parallel case
    hullCount = 0;
    parallel_start = std::chrono::high_resolution_clock::now();
    parallel_hull.resize(global_points.size());
    cout << "hull size resized properlly" << endl;
    divide_kernel_caller(global_points.data(), global_points.size(), global_points[0],
                         global_points.back(), parallel_hull.data(), &hullCount);
    cout << "divided kernel caller returned" << endl;
    parallel_end = std::chrono::high_resolution_clock::now();
    parallel_time = std::chrono::duration<double>(parallel_end - parallel_start).count();
    global_hull = parallel_hull;
    break;
  default:
    break;
  }
  glutDisplayFunc(display);
  glutTimerFunc(0, timer, 0);
  cout << "Drawing" << endl;
  // Start the main loop
  glutMainLoop();
  return 0;
}
