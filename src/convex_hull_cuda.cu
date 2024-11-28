#include "convex_hull_general.h"
#include <GL/glut.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fenv.h>
#include <iostream>
#include <random>
#include <thread>

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

__global__ void findFarthestPoint(const Point *points, int num_points, Point lineStart,
                                  Point lineEnd, int *farthestIdx, double *maxDist)
{
  __shared__ double localMaxDist;
  __shared__ int localFarthestIdx;

  int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

  // bound checking thread ids
  if (thread_id >= num_points)
  {
    return;
  }

  double dist = fabs((lineEnd.y - lineStart.y) * (points[thread_id].x - lineStart.x) -
                     (lineEnd.x - lineStart.x) * (points[thread_id].y - lineStart.y));

  // Initialize localMaxDist and localFarthestIdx for each block
  if (threadIdx.x == 0)
  {
    localMaxDist = 0.0f;
    localFarthestIdx = -1;
  }
  __syncthreads();

  atomicMax(&localMaxDist, dist);
  __syncthreads();

  if (dist == localMaxDist)
  {
    atomicExch(&localFarthestIdx, thread_id);
  }

  if (threadIdx.x == 0 && localMaxDist > (*maxDist))
  {
    *maxDist = localMaxDist;
    *farthestIdx = localFarthestIdx;
  }
}

__global__ void classifyPoints(const Point *points, int numPoints, Point lineStart, Point lineEnd,
                               Point farthest, Point *leftSubset, Point *rightSubset,
                               int *leftCount, int *rightCount)
{
  int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (thread_id >= numPoints)
    return;

  float cross = (lineEnd.x - lineStart.x) * (points[thread_id].y - lineStart.y) -
                (lineEnd.y - lineStart.y) * (points[thread_id].x - lineStart.x);

  if (cross > 0)
  {
    int idx = atomicAdd(leftCount, 1);
    leftSubset[idx] = points[thread_id];
  }
  else if (cross < 0)
  {
    int idx = atomicAdd(rightCount, 1);
    rightSubset[idx] = points[thread_id];
  }
}

__global__ void divide_kernel(Point *points, int num_points, Point lineStart, Point lineEnd,
                              Point *hull, int *hullCount)
{
  if (num_points == 0)
  {
    return;
  }

  int farthestIdx;
  double maxDist = 0.0f;

  dim3 grid(ceil(num_points / 256.0), 1, 1);
  dim3 block(256, 1, 1);

  findFarthestPoint<<<grid, block>>>(points, num_points, lineStart, lineEnd, &farthestIdx,
                                     &maxDist);

  cudaDeviceSynchronize();

  Point farthest = points[farthestIdx];

  // Add the farthest point to the hull
  int idx = atomicAdd(hullCount, 1);
  hull[idx] = farthest;

  // Partition points into left and right subsets
  Point *leftSubset, *rightSubset;
  int leftCount = 0, rightCount = 0;
  classifyPoints<<<grid, block>>>(points, num_points, lineStart, lineEnd, farthest, leftSubset,
                                  rightSubset, &leftCount, &rightCount);

  // Recursive calls
  if (leftCount > 0)
  {
    divide_kernel<<<1, 1>>>(leftSubset, leftCount, lineStart, farthest, hull, hullCount);
  }
  if (rightCount > 0)
  {
    divide_kernel<<<1, 1>>>(rightSubset, rightCount, farthest, lineEnd, hull, hullCount);
  }
}

static void kernel_caller(const vector<Point> &points, vector<Point> &hull)
{
  int numPoints = points.size();
  Point *d_points, *d_hull;
  int *d_hull_count;

  // copying over points to d_points
  cudaMalloc(&d_points, numPoints * sizeof(Point));
  cudaMemcpy(d_points, points.data(), numPoints * sizeof(Point), cudaMemcpyHostToDevice);

  // Allocing space for d_hull and d_hull_count
  cudaMalloc(&d_hull, numPoints * sizeof(Point));
  cudaMalloc(&d_hull_count, sizeof(int));
  cudaMemset(d_hull_count, 0, sizeof(int));

  Point lineStart = points[0];
  Point lineEnd = points.back();

  divide_kernel<<<1, 1>>>(d_points, numPoints, lineStart, lineEnd, d_hull, d_hull_count);

  // Make sure all resulting kernel calls on device are synchronized at this points
  cudaDeviceSynchronize();

  int hullCount;
  cudaMemcpy(&hullCount, d_hull_count, sizeof(int), cudaMemcpyDeviceToHost);

  // Resize hull vector properlly and copy d_hull back over to hull.data()
  hull.resize(hullCount);
  cudaMemcpy(hull.data(), d_hull, sizeof(Point) * hullCount, cudaMemcpyDeviceToHost);

  cudaFree(d_points);
  cudaFree(d_hull_count);
  cudaFree(d_hull);
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

  vector<Point> hull;
  vector<Point> points = generate_random_points(config.num_points);

  // sort points by x value, y value is tie breaker
  sort(points.begin(), points.end(),
       [](const Point &a, const Point &b) { return (a.x < b.x) || (a.x == b.x && a.y < b.y); });

  // Initialize OpenGL
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
  glutInitWindowSize(800, 800);
  glutCreateWindow("Convex Hull Visualization");

  initOpenGL();

  // NOTE: WHERE I LEFT OFF: Implement the switch statment logic for the command

  return 0;
}
