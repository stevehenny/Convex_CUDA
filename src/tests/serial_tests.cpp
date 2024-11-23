#include "../convex_hull_serial.h"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/convex_hull_2.h>
#include <algorithm>
#include <chrono>
#include <gtest/gtest.h>
#include <random>
#include <vector>

// CGAL Kernel
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_2 CGAL_Point;

// Generate random points in a given range
std::vector<Point> generate_random_points(int num_points, double range_min = 0.0,
                                          double range_max = 100.0)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(range_min, range_max);

  std::vector<Point> points;
  for (int i = 0; i < num_points; ++i)
  {
    points.emplace_back(dist(gen), dist(gen));
  }
  return points;
}

// Convert points from your format to CGAL format
std::vector<CGAL_Point> convert_to_cgal_points(const std::vector<Point> &points)
{
  std::vector<CGAL_Point> cgal_points;
  for (const auto &point : points)
  {
    cgal_points.emplace_back(point.x, point.y);
  }
  return cgal_points;
}

// Test convex hull implementation against CGAL
TEST(ConvexHullComparisonTest, CompareAgainstCGAL)
{
  std::vector<int> test_sizes = {10, 50, 100, 1000, 10000, 100000};

  for (int size : test_sizes)
  {
    // Generate random points
    std::vector<Point> points = generate_random_points(size);

    // Sort points (needed for divide-and-conquer approach)
    std::sort(points.begin(), points.end(), [](const Point &a, const Point &b) {
      return (a.x < b.x) || (a.x == b.x && a.y < b.y);
    });

    // Compute convex hull using your implementation
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Point> my_hull = divide(points);
    auto end = std::chrono::high_resolution_clock::now();
    double my_time = std::chrono::duration<double>(end - start).count();

    // Convert points to CGAL format
    std::vector<CGAL_Point> cgal_points = convert_to_cgal_points(points);

    // Compute convex hull using CGAL
    std::vector<CGAL_Point> cgal_hull;
    start = std::chrono::high_resolution_clock::now();
    CGAL::convex_hull_2(cgal_points.begin(), cgal_points.end(), std::back_inserter(cgal_hull));
    end = std::chrono::high_resolution_clock::now();
    double cgal_time = std::chrono::duration<double>(end - start).count();

    // Compare results
    ASSERT_EQ(my_hull.size(), cgal_hull.size()) << "Hull size mismatch for " << size << " points.";

    for (size_t i = 0; i < my_hull.size(); ++i)
    {
      EXPECT_NEAR(my_hull[i].x, cgal_hull[i].x(), 1e-9) << "Mismatch at point " << i;
      EXPECT_NEAR(my_hull[i].y, cgal_hull[i].y(), 1e-9) << "Mismatch at point " << i;
    }

    // Print timing information (optional)
    std::cout << "Test size: " << size << ", Your implementation: " << my_time << "s"
              << ", CGAL: " << cgal_time << "s" << std::endl;
  }
}
