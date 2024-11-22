#include "../convex_hull_serial.h"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

// Helper function to create a few points for testing
std::vector<Point> create_test_points()
{
  return {Point(1.4, 2.3), Point(3.2, 1.4), Point(2.1, 2),   Point(8.2, 2),
          Point(2.1, 0.2), Point(2.6, 3.7), Point(1.9, 1.4), Point(3, 2.3)};
}

// Comparator to sort points by x, and by y if x is the same
bool point_comparator(const Point &a, const Point &b)
{
  return (a.x() < b.x()) || (a.x() == b.x() && a.y() < b.y());
}

TEST(ConvexHullTest, CheckPolyCross)
{
  Point a(0.1, 0.2), b(1.4, 2.8), c(3.2, 2.1);
  double result = check_cross(a, b, c);
  EXPECT_NEAR(result, -5.58999999999, 1e-9); // Adjust based on expected result
}

TEST(ConvexHullTest, ComputeUpperTangent)
{
  std::vector<Point> left = {Point(0.1, 5.1), Point(1.16, 1.12)};
  std::vector<Point> right = {Point(2.21, 23.2), Point(3.33, 33.3)};

  std::sort(left.begin(), left.end(), point_comparator);
  std::sort(right.begin(), right.end(), point_comparator);

  std::pair<int, int> result = compute_upper_tangent(left, right);
  EXPECT_EQ(result.first, 0);
  EXPECT_EQ(result.second, 1);
}

TEST(ConvexHullTest, ComputeLowerTangent)
{
  std::vector<Point> left = {Point(0, 0), Point(1, 2)};
  std::vector<Point> right = {Point(2, 3), Point(3, 1)};

  std::sort(left.begin(), left.end(), point_comparator);
  std::sort(right.begin(), right.end(), point_comparator);

  std::pair<int, int> result = compute_lower_tangent(left, right);
  EXPECT_EQ(result.first, 0);
  EXPECT_EQ(result.second, 1);
}

TEST(ConvexHullTest, MergeHulls)
{
  std::vector<Point> left = {Point(0, 0), Point(1, 1)};
  std::vector<Point> right = {Point(2, 2), Point(3, 3)};

  std::sort(left.begin(), left.end(), point_comparator);
  std::sort(right.begin(), right.end(), point_comparator);

  std::vector<Point> result = merge_hulls(left, right);
  ASSERT_EQ(static_cast<int>(result.size()), 4);
}

TEST(ConvexHullTest, FindConvexHull)
{
  std::vector<Point> points = create_test_points();

  // Sort all points before calling find_convex_hull
  std::sort(points.begin(), points.end(), point_comparator);

  std::vector<Point> result = find_convex_hull(points);

  // Adjust the expected points as per the actual expected convex hull points
  ASSERT_EQ(static_cast<int>(result.size()), 4); // Adjust based on expected hull size

  // Verify each point in the convex hull with approximate floating-point comparison
  EXPECT_NEAR(result[0].x(), 2.1, 1e-9);
  EXPECT_NEAR(result[0].y(), 0.2, 1e-9);
  EXPECT_NEAR(result[1].x(), 1.4, 1e-9);
  EXPECT_NEAR(result[1].y(), 2.3, 1e-9);
  EXPECT_NEAR(result[2].x(), 2.6, 1e-9);
  EXPECT_NEAR(result[2].y(), 3.7, 1e-9);
  EXPECT_NEAR(result[3].x(), 8.2, 1e-9);
  EXPECT_NEAR(result[3].y(), 2.0, 1e-9);
}

TEST(ConvexHullTest, FindConvexHull_10_Points)
{
  std::vector<Point> points;
  points.push_back(Point(0, 0));
  points.push_back(Point(1, 4));   // Changed from Point(1, -4)
  points.push_back(Point(1, 5));   // Changed from Point(-1, -5)
  points.push_back(Point(5, 3));   // Changed from Point(-5, -3)
  points.push_back(Point(3, 1));   // Changed from Point(-3, -1)
  points.push_back(Point(1, 3));   // Changed from Point(-1, -3)
  points.push_back(Point(2, 2));   // Changed from Point(-2, -2)
  points.push_back(Point(1, 1));   // Changed from Point(-1, -1)
  points.push_back(Point(2, 1));   // Changed from Point(-2, -1)
  points.push_back(Point(1.2, 2)); // Changed from Point(-1, 1)

  sort(points.begin(), points.end(), point_comparator);

  std::vector<Point> result = find_convex_hull(points);

  EXPECT_NEAR(result[0].x(), 0, 1e-9);
  EXPECT_NEAR(result[0].y(), 0, 1e-9);
  EXPECT_NEAR(result[1].x(), 1, 1e-9);
  EXPECT_NEAR(result[1].y(), 5, 1e-9);
  EXPECT_NEAR(result[2].x(), 5, 1e-9);
  EXPECT_NEAR(result[2].y(), 3, 1e-9);
  EXPECT_NEAR(result[3].x(), 3, 1e-9);
  EXPECT_NEAR(result[3].y(), 1, 1e-9);
  EXPECT_NEAR(result[4].x(), 1, 1e-9);
  EXPECT_NEAR(result[4].y(), 1, 1e-9);
}
