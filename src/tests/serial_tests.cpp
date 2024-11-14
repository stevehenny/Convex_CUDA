#include "../convex_hull_serial.h"
#include <gtest/gtest.h>
#include <vector>

// Helper function to create a few points for testing
std::vector<Point> create_test_points() {
  return {Point(0, 0), Point(1, 1), Point(2, 2), Point(0, 2),
          Point(2, 0), Point(3, 3), Point(1, 0), Point(3, 0)};
}

TEST(ConvexHullTest, CheckPolyCross) {
  Point a(0, 0), b(1, 1), c(2, 2);
  double result = CROSS(a, b, c);
  // Add an expected value based on your algorithm
  EXPECT_NEAR(result, 0.0, 1e-9); // Adjust based on expected result
}

TEST(ConvexHullTest, ComputeUpperTangent) {
  std::vector<Point> left = {Point(0, 0), Point(1, 1)};
  std::vector<Point> right = {Point(2, 2), Point(3, 3)};

  std::pair<int, int> result = compute_upper_tangent(left, right);
  // Adjust expected values based on the actual algorithm
  EXPECT_EQ(result.first, 0);
  EXPECT_EQ(result.second, 1);
}

TEST(ConvexHullTest, ComputeLowerTangent) {
  std::vector<Point> left = {Point(0, 0), Point(1, 1)};
  std::vector<Point> right = {Point(2, 2), Point(3, 3)};

  std::pair<int, int> result = compute_lower_tangent(left, right);
  // Adjust expected values based on the actual algorithm
  EXPECT_EQ(result.first, 0);
  EXPECT_EQ(result.second, 1);
}

TEST(ConvexHullTest, MergeHulls) {
  std::vector<Point> left = {Point(0, 0), Point(1, 1)};
  std::vector<Point> right = {Point(2, 2), Point(3, 3)};

  std::vector<Point> result = merge_hulls(left, right);
  // Adjust expected result based on the merge_hulls logic
  ASSERT_EQ(result.size(), 4);
}

TEST(ConvexHullTest, FindConvexHull) {
  std::vector<Point> points = create_test_points();
  std::vector<Point> result = find_convex_hull(points);

  // Test that the result is a convex hull
  // Adjust based on your expected result for this set of points
  ASSERT_EQ(result.size(), 4); // Assuming 4 points in the convex hull
  EXPECT_TRUE(result[0].x() == 0 && result[0].y() == 0);
  EXPECT_TRUE(result[1].x() == 2 && result[1].y() == 0);
  EXPECT_TRUE(result[2].x() == 2 && result[2].y() == 2);
  EXPECT_TRUE(result[3].x() == 0 && result[3].y() == 2);
}
