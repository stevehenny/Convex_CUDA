#include "convex_hull_serial.h"
#include <cmath>
#include <iostream>
#include <vector>

// Compute the upper tangent of two convex hulls ch_left and ch_right
std::pair<int, int> compute_upper_tangent(const std::vector<Point> &ch_left,
                                          const std::vector<Point> &ch_right) {
  int i = ch_left.size() - 1; // Start from the rightmost point of the left hull
  int j = 0;                  // Start from the leftmost point of the right hull

  // Try to find the upper tangent
  while (true) {
    bool found = false;

    // Check if the tangent can be moved from the left hull to the right
    while (CROSS(ch_left[i], ch_right[j], ch_right[(j + 1) % ch_right.size()]) <
           0) {
      j = (j + 1) % ch_right.size(); // Move to the next point on the right hull
      found = true;
    }

    // Check if the tangent can be moved from the right hull to the left
    while (CROSS(ch_right[j], ch_left[i], ch_left[(i + 1) % ch_left.size()]) >
           0) {
      i = (i + 1) % ch_left.size(); // Move to the next point on the left hull
      found = true;
    }

    // If no more movement can be made, we have found the tangent
    if (!found) {
      break;
    }
  }

  return std::make_pair(i, j); // Return the indices of the tangent points
}

// Compute the lower tangent of two convex hulls ch_left and ch_right
std::pair<int, int> compute_lower_tangent(const std::vector<Point> &ch_left,
                                          const std::vector<Point> &ch_right) {
  int i = 0; // Start from the leftmost point of the left hull
  int j =
      ch_right.size() - 1; // Start from the rightmost point of the right hull

  // Try to find the lower tangent
  while (true) {
    bool found = false;

    // Check if the tangent can be moved from the left hull to the right
    while (CROSS(ch_left[i], ch_right[j],
                 ch_right[(j - 1 + ch_right.size()) % ch_right.size()]) > 0) {
      j = (j - 1 + ch_right.size()) %
          ch_right.size(); // Move to the previous point on the right hull
      found = true;
    }

    // Check if the tangent can be moved from the right hull to the left
    while (CROSS(ch_right[j], ch_left[i],
                 ch_left[(i - 1 + ch_left.size()) % ch_left.size()]) < 0) {
      i = (i - 1 + ch_left.size()) %
          ch_left.size(); // Move to the previous point on the left hull
      found = true;
    }

    // If no more movement can be made, we have found the tangent
    if (!found) {
      break;
    }
  }

  return std::make_pair(i, j); // Return the indices of the tangent points
}

// Merges two convex hulls using their upper and lower tangents
std::vector<Point> merge_hulls(const std::vector<Point> &ch_left,
                               const std::vector<Point> &ch_right) {
  std::vector<Point> result;
  auto u_tan = compute_upper_tangent(ch_left, ch_right);
  auto l_tan = compute_lower_tangent(ch_left, ch_right);

  int l_length = ch_left.size(), r_length = ch_right.size();

  // Add points from left hull between lower and upper tangents
  int ind = l_tan.first;
  result.push_back(ch_left[ind]);
  while (ind != u_tan.first) {
    ind = (ind + 1) % l_length;
    result.push_back(ch_left[ind]);
  }

  // Add points from right hull between upper and lower tangents
  ind = u_tan.second;
  result.push_back(ch_right[ind]);
  while (ind != l_tan.second) {
    ind = (ind + 1) % r_length;
    result.push_back(ch_right[ind]);
  }
  return result;
}

// Recursive function to find the convex hull of a sorted set of points
std::vector<Point> find_convex_hull(const std::vector<Point> &points) {
  int n = points.size();

  // Base case: when there are 3 or fewer points
  if (n <= 3) {
    std::vector<Point> new_hull(points.begin(), points.end());
    return new_hull;
  }

  // Recursively find the convex hull of the left and right halves
  std::vector<Point> ch_left =
      find_convex_hull({points.begin(), points.begin() + n / 2});
  std::vector<Point> ch_right =
      find_convex_hull({points.begin() + n / 2, points.end()});

  // Merge the two convex hulls
  return merge_hulls(ch_left, ch_right);
}
