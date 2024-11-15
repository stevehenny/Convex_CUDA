#include "../convex_hull_serial.h"
#include <algorithm>
#include <vector>
#include <gtest/gtest.h>

// Helper function to create a few points for testing
std::vector<Point> create_test_points() {
  return {Point(0, 0), Point(1, 1), Point(2, 2), Point(0, 2),
          Point(2, 0), Point(3, 3), Point(1, 0), Point(3, 0)};
}

// Comparator to sort points by x, and by y if x is the same
bool point_comparator(const Point &a, const Point &b) {
  return (a.x() < b.x()) || (a.x() == b.x() && a.y() < b.y());
}

TEST(ConvexHullTest, CheckPolyCross) {
  Point a(0, 0), b(1, 1), c(2, 2);
  double result = CROSS(a, b, c);
  EXPECT_NEAR(result, 0.0, 1e-9); // Adjust based on expected result
}

TEST(ConvexHullTest, ComputeUpperTangent) {
  std::vector<Point> left = {Point(0, 0), Point(1, 1)};
  std::vector<Point> right = {Point(2, 2), Point(3, 3)};

  std::sort(left.begin(), left.end(), point_comparator);
  std::sort(right.begin(), right.end(), point_comparator);
  for (int i = 0; i < left.size(); i++){
    std::cout << left[i].x() << std::endl;
    std::cout << left[i].y() << std::endl;
    std::cout << right[i].x() << std::endl;
    std::cout << right[i].y() << std::endl;
  }
    
  std::pair<int, int> result = compute_upper_tangent(left, right);
  EXPECT_EQ(result.first, 0);
  EXPECT_EQ(result.second, 1);
}

TEST(ConvexHullTest, ComputeLowerTangent) {
  std::vector<Point> left = {Point(0, 0), Point(1, 1)};
  std::vector<Point> right = {Point(2, 2), Point(3, 3)};

  std::sort(left.begin(), left.end(), point_comparator);
  std::sort(right.begin(), right.end(), point_comparator);

  std::pair<int, int> result = compute_lower_tangent(left, right);
  EXPECT_EQ(result.first, 0);
  EXPECT_EQ(result.second, 1);
}

TEST(ConvexHullTest, MergeHulls) {
  std::vector<Point> left = {Point(0, 0), Point(1, 1)};
  std::vector<Point> right = {Point(2, 2), Point(3, 3)};

  std::sort(left.begin(), left.end(), point_comparator);
  std::sort(right.begin(), right.end(), point_comparator);

  std::vector<Point> result = merge_hulls(left, right);
  ASSERT_EQ(result.size(), 4);
}

TEST(ConvexHullTest, FindConvexHull) {
  std::vector<Point> points = create_test_points();

  // Sort all points before calling find_convex_hull
  std::sort(points.begin(), points.end(), point_comparator);

  std::vector<Point> result = find_convex_hull(points);

  ASSERT_EQ(result.size(), 4); // Assuming 4 points in the convex hull
  EXPECT_TRUE(result[0].x() == 0 && result[0].y() == 0);
  EXPECT_TRUE(result[1].x() == 2 && result[1].y() == 0);
  EXPECT_TRUE(result[2].x() == 2 && result[2].y() == 2);
  EXPECT_TRUE(result[3].x() == 0 && result[3].y() == 2);
}
