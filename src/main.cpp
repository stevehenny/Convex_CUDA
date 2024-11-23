#include "convex_hull_serial.h"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Interval_nt.h>
#include <CGAL/convex_hull_2.h>
#include <cstdlib>
#include <ctime>
#include <fenv.h>
#include <iomanip> // For precise output formatting
#include <iostream>
#include <vector>

// CGAL Kernel and Point Type
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_2 Point_2;

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

int main()
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

  // Generate random points
  std::vector<Point> points;
  for (int i = 0; i < 100000; ++i)
  {
    double x = static_cast<double>(std::rand()) / RAND_MAX * 100.0; // Random x in [0.0, 100.0)
    double y = static_cast<double>(std::rand()) / RAND_MAX * 100.0; // Random y in [0.0, 100.0)
    points.push_back(Point(x, y));
    // std::cout << std::fixed << std::setprecision(6) << "(" << x << ", " << y << ")\n";
  }
  // Convert custom points to CGAL Point_2
  std::vector<Point_2> cgal_points;
  for (const auto &p : points)
  {
    cgal_points.push_back(to_cgal_point(p));
  }

  std::sort(points.begin(), points.end(), [](const Point &a, const Point &b) {
    return (a.x < b.x) || (a.x == b.x && a.y < b.y);
  });
  // Compute convex hull using CGAL
  std::vector<Point_2> cgal_hull;
  CGAL::convex_hull_2(cgal_points.begin(), cgal_points.end(), std::back_inserter(cgal_hull));

  // Compute convex hull using your implementation
  std::vector<Point> my_hull = divide(points);

  // Convert CGAL hull back to custom Point type for comparison
  std::vector<Point> converted_cgal_hull;
  for (const auto &p : cgal_hull)
  {
    converted_cgal_hull.push_back(to_custom_point(p));
  }

  // Sort the hulls for comparison
  // Sort the hulls for comparison
  std::sort(converted_cgal_hull.begin(), converted_cgal_hull.end(),
            [](const Point &a, const Point &b) {
              return (a.x < b.x) || (a.x == b.x && a.y < b.y);
            }); // Added the missing parenthesis here

  std::sort(my_hull.begin(), my_hull.end(), [](const Point &a, const Point &b) {
    return (a.x < b.x) || (a.x == b.x && a.y < b.y);
  });

  // Compare the two hulls
  if (converted_cgal_hull == my_hull)
  {
    std::cout << "Your implementation matches CGAL's result!\n";
  }
  else
  {
    std::cout << "Mismatch between your implementation and CGAL's result.\n";
    // std::cout << "CGAL Hull:\n";
    //   for (const auto &p : converted_cgal_hull)
    //   {
    //     std::cout << std::fixed << std::setprecision(6) << "(" << p.x << ", " << p.y << ")\n";
    //   }
    //   std::cout << "Your Hull:\n";
    //   for (const auto &p : my_hull)
    //   {
    //     std::cout << std::fixed << std::setprecision(6) << "(" << p.x << ", " << p.y << ")\n";
    //   }
  }

  return 0;
}
