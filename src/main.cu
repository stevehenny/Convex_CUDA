#include "convex_hull_cuda.cuh"
#include "convex_hull_general.h"
#include "convex_hull_serial.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <ctime>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#ifdef USE_OPENGL
#include <GL/glut.h>
#include <thread>
#endif

#define BLOCK_SIZE 1024
#define THRESHOLD (BLOCK_SIZE * 32)
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

#ifdef USE_OPENGL
// OpenGL Visualization Functions
void display()
{
  glClear(GL_COLOR_BUFFER_BIT);
  glColor3f(1.0, 1.0, 1.0);
  glBegin(GL_POINTS);
  for (const auto &point : global_points)
  {
    glVertex2f(point.x, point.y);
  }
  glEnd();

  glColor3f(1.0, 0.0, 0.0);
  glBegin(GL_LINE_LOOP);
  for (const auto &point : global_hull)
  {
    glVertex2f(point.x, point.y);
  }
  glEnd();
  glutSwapBuffers();
}

void timer(int)
{
  glutPostRedisplay();
  this_thread::sleep_for(chrono::milliseconds(100));
  glutTimerFunc(100, timer, 0);
}

void initOpenGL()
{
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0.0, 1000.0, 0.0, 1000.0);
}
#endif

void sort_points_clockwise(std::vector<Point> &points)
{
  // Compute the centroid
  float centroid_x = 0, centroid_y = 0;
  for (const auto &point : points)
  {
    centroid_x += point.x;
    centroid_y += point.y;
  }
  centroid_x /= points.size();
  centroid_y /= points.size();

  // Sort the points clockwise around the centroid
  std::sort(points.begin(), points.end(), [centroid_x, centroid_y](const Point &a, const Point &b) {
    float angle_a = atan2(a.y - centroid_y, a.x - centroid_x);
    float angle_b = atan2(b.y - centroid_y, b.x - centroid_x);
    return angle_a > angle_b; // Descending order for clockwise sorting
  });
}

static void run_parallel_config(vector<Point> &host_points)
{

  int N = host_points.size();

  auto parallel_sort_start = std::chrono::high_resolution_clock::now();
  // Sort using cuda wrapper library thrust vectors
  thrust::device_vector<Point> d_points(host_points);
  thrust::sort(d_points.begin(), d_points.end(),
               [] __host__ __device__(const Point &a, const Point &b) {
                 if (a.x < b.x)
                 {
                   return true;
                 }
                 else if (a.x > b.x)
                 {
                   return false;
                 }
                 else
                 {
                   return a.y < b.y;
                 }
               });

  thrust::copy(d_points.begin(), d_points.end(), host_points.begin());
  auto parallel_sort_end = std::chrono::high_resolution_clock::now();
  auto parallel_sort_time =
      std::chrono::duration_cast<chrono::milliseconds>(parallel_sort_end - parallel_sort_start)
          .count();

  // Exectuion time
  auto parallel_start = std::chrono::high_resolution_clock::now();
  // compute
  std::unordered_set hull = QuickHull(host_points);
  auto parallel_end = std::chrono::high_resolution_clock::now();
  auto parallel_time =
      std::chrono::duration_cast<chrono::milliseconds>(parallel_end - parallel_start).count();

  for (auto &point : hull)
  {
    global_hull.push_back(point);
  }
  sort_points_clockwise(global_hull);
  cout << "PARALLEL INFO" << endl;
  cout << "Parallel sort time: " << parallel_sort_time << "ms" << endl;
  cout << "Parallel algorithm execution time: " << parallel_time << "ms" << endl;
  cout << "Parallel total execution time: " << parallel_time + parallel_sort_time << "ms" << endl;
  cout << "Parallel hull size: " << global_hull.size() << endl;
}

static void run_serial_config(vector<Point> &host_points)
{

  auto serial_sort_start = std::chrono::high_resolution_clock::now();
  sort(host_points.begin(), host_points.end(),
       [](const Point &a, const Point &b) { return (a.x < b.x) || (a.x == b.x && a.y < b.y); });
  auto serial_sort_end = std::chrono::high_resolution_clock::now();
  auto serial_sort_time =
      std::chrono::duration_cast<chrono::milliseconds>(serial_sort_end - serial_sort_start).count();
  auto serial_start = std::chrono::high_resolution_clock::now();
  global_hull = divide(host_points);
  auto serial_end = std::chrono::high_resolution_clock::now();
  auto serial_time =
      std::chrono::duration_cast<chrono::milliseconds>(serial_end - serial_start).count();
  cout << "SERIAL INFO" << endl;
  cout << "Serial sort time: " << serial_sort_time << "ms" << endl;
  cout << "Serial algorithm execution time: " << serial_time << "ms" << endl;
  cout << "Serial total execution time: " << serial_time + serial_sort_time << "ms" << endl;
  cout << "Serial hull size: " << global_hull.size() << endl;
}

int main(int argc, char *argv[])
{
  Config config;
  int exit_status = parse_args(argc, argv, &config);
  if (exit_status != 0)
  {
    return 1;
  }

  vector<Point> host_points = generate_random_points(config.num_points);

  if (strcmp(config.command, "parallel") == 0)
  {
    run_parallel_config(host_points);
  }
  else if (strcmp(config.command, "serial") == 0)
  {
    run_serial_config(host_points);
  }
#ifdef NO_OPENGL
  else if (strcmp(config.command, "both") == 0)
  {
    run_parallel_config(host_points);
    cout << endl;
    run_serial_config(host_points);
  }
#endif
  else
  {
    cerr << "Invalid command" << endl;
    return 1;
  }

#ifdef USE_OPENGL
  global_points = host_points;
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
  glutInitWindowSize(800, 800);
  glutCreateWindow("Convex Hull Visualization");
  initOpenGL();
  glutDisplayFunc(display);
  glutTimerFunc(0, timer, 0);
  glutMainLoop();
#endif

  return 0;
}
