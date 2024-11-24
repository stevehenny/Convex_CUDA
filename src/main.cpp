#include "convex_hull_serial.h"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Interval_nt.h>
#include <CGAL/convex_hull_2.h>
#include <GL/glut.h>
#include <cstdlib>
#include <ctime>
#include <fenv.h>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

using namespace std;

vector<Point> global_points;
vector<Point> global_hull;

// CGAL Kernel and Point Type
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_2 Point_2;

// OpenGL display function
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

// Function to convert your custom Point to CGAL Point_2
Point_2 to_cgal_point(const Point &p)
{
  return Point_2(p.x, p.y);
}

// Function to convert CGAL Point_2 to your custom Point
Point to_custom_point(const Point_2 &p)
{
  return Point(p.x(), p.y());
}

int main(int argc, char **argv)
{
  if (!fesetround(FE_TONEAREST))
  {
    std::cerr << "Rounding mode set to nearest successfully.\n";
  }
  else
  {
    std::cerr << "Failed to set rounding mode.\n";
  }
  std::srand(std::time(0));

  random_device rd;
  mt19937 gen(rd());
  normal_distribution<> dist_x(500.0, 100.0);
  normal_distribution<> dist_y(500.0, 100.0);
  // Generate random points
  for (int i = 0; i < 1000000; ++i)
  {
    const double x = (dist_x(gen)); // Random x in [0.0, 100.0)
    const double y = (dist_y(gen)); // Random y in [0.0, 100.0)
    global_points.push_back(Point(x, y));
    // std::cout << std::fixed << std::setprecision(6) << "(" << x << ", " << y << ")\n";
  }

  std::sort(global_points.begin(), global_points.end(), [](const Point &a, const Point &b) {
    return (a.x < b.x) || (a.x == b.x && a.y < b.y);
  });
  // // Convert custom points to CGAL Point_2
  std::vector<Point_2> cgal_points;
  for (const auto &p : global_points)
  {
    cgal_points.push_back(to_cgal_point(p));
  }
  // Compute convex hull using CGAL
  std::vector<Point_2> cgal_hull;
  CGAL::convex_hull_2(cgal_points.begin(), cgal_points.end(), std::back_inserter(cgal_hull));
  vector<Point> converted_cgal;
  for (const auto &p : cgal_hull)
  {
    converted_cgal.push_back(to_custom_point(p));
  }
  // Initialize OpenGL
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
  glutInitWindowSize(800, 800);
  glutCreateWindow("Convex Hull Visualization");

  initOpenGL();
  // Compute convex hull using your implementation
  cout << "D AND CONQUER" << endl;
  global_hull = divide(global_points);
  // global_hull = converted_cgal;

  // Set OpenGL callback functions
  glutDisplayFunc(display);
  glutTimerFunc(0, timer, 0);

  // Start the main loop
  glutMainLoop();

  // // Convert CGAL hull back to custom Point type for comparison
  // std::vector<Point> converted_cgal_hull;
  // for (const auto &p : cgal_hull)
  // {
  //   converted_cgal_hull.push_back(to_custom_point(p));
  // }
  //
  // // Sort the hulls for comparison
  // // Sort the hulls for comparison
  // std::sort(converted_cgal_hull.begin(), converted_cgal_hull.end(),
  //           [](const Point &a, const Point &b) {
  //             return (a.x < b.x) || (a.x == b.x && a.y < b.y);
  //           }); // Added the missing parenthesis here
  //
  // std::sort(my_hull.begin(), my_hull.end(), [](const Point &a, const Point &b) {
  //   return (a.x < b.x) || (a.x == b.x && a.y < b.y);
  // });
  //
  // // Compare the two hulls
  // if (converted_cgal_hull == my_hull)
  // {
  //   std::cout << "Your implementation matches CGAL's result!\n";
  // }
  // else
  // {
  //   std::cout << "Mismatch between your implementation and CGAL's result.\n";
  // }

  return 0;
}
