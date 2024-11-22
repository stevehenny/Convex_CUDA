#include "convex_hull_serial.h"
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

// Helper function to compute the cross product of vectors AB and AC
double check_cross(const Point &a, const Point &b, const Point &c)
{
  // std::cout << CROSS(a, b, c) << std::endl;
  return CROSS(a, b, c);
}

// Compute the upper tangent of two convex hulls ch_left and ch_right
std::pair<int, int> compute_upper_tangent(const std::vector<Point> &ch_left,
                                          const std::vector<Point> &ch_right)
{
  int l_length = ch_left.size();
  int r_length = ch_right.size();

  // Find the rightmost point of the left hull
  int left_index = 0;
  for (int i = 0; i < l_length; i++)
  {
    if (ch_left[i].x() > ch_left[left_index].x())
    {
      left_index = i;
    }
  }

  // Find the leftmost point of the right hull
  int right_index = 0;
  for (int i = 0; i < r_length; i++)
  {
    if (ch_right[i].x() < ch_right[right_index].x())
    {
      right_index = i;
    }
  }

  bool done = false;
  while (!done)
  {
    done = true;

    // Adjust left index if necessary
    while (check_cross(ch_left[left_index], ch_right[right_index],
                       ch_left[(l_length - 1) % l_length]) <= 0)
    {
      left_index = (left_index + l_length - 1) % l_length;
      done = false;
    }

    // Adjust right index if necessary
    while (check_cross(ch_right[right_index], ch_left[left_index],
                       ch_right[(right_index + 1) % r_length]) >= 0)
    {
      right_index = (right_index + 1) % r_length;
      done = false;
    }
  }
  return std::make_pair(left_index, right_index);
}

// Compute the lower tangent of two convex hulls ch_left and ch_right
std::pair<int, int> compute_lower_tangent(const std::vector<Point> &ch_left,
                                          const std::vector<Point> &ch_right)
{
  int l_length = ch_left.size();
  int r_length = ch_right.size();

  // Find the rightmost point of the left hull
  int left_index = 0;
  for (int i = 0; i < l_length; i++)
  {
    if (ch_left[i].x() > ch_left[left_index].x())
    {
      left_index = i;
    }
  }

  // Find the leftmost point of the right hull
  int right_index = 0;
  for (int i = 0; i < r_length; i++)
  {
    if (ch_right[i].x() < ch_right[right_index].x())
    {
      right_index = i;
    }
  }

  bool done = false;
  while (!done)
  {
    done = true;

    // Adjust left index if necessary
    while (check_cross(ch_left[left_index], ch_right[right_index],
                       ch_left[(left_index + 1) % l_length]) >= 0)
    {
      left_index = (left_index + 1) % l_length;
      done = false;
    }

    // Adjust right index if necessary
    while (check_cross(ch_right[right_index], ch_left[left_index],
                       ch_right[(right_index + r_length - 1) % r_length]) <= 0)
    {
      right_index = (right_index - 1) % r_length;
      done = false;
    }
  }
  return std::make_pair(left_index, right_index);
}
// Merges two convex hulls using their upper and lower tangents
std::vector<Point> merge_hulls(const std::vector<Point> &ch_left,
                               const std::vector<Point> &ch_right)
{
  std::vector<Point> result;
  auto u_tan = compute_upper_tangent(ch_left, ch_right);
  auto l_tan = compute_lower_tangent(ch_left, ch_right);

  int l_length = ch_left.size(), r_length = ch_right.size();

  // Add points from left hull between lower and upper tangents
  int ind = l_tan.first;
  result.push_back(ch_left[ind]);
  while (ind != u_tan.first)
  {
    ind = (ind + 1) % l_length;
    result.push_back(ch_left[ind]);
  }

  // Add points from right hull between upper and lower tangents
  ind = u_tan.second;
  result.push_back(ch_right[ind]);
  while (ind != l_tan.second)
  {
    ind = (ind + 1) % r_length;
    result.push_back(ch_right[ind]);
  }
  return result;
}

// Recursive function to find the convex hull of a sorted set of points
std::vector<Point> find_convex_hull(const std::vector<Point> &points)
{
  int n = points.size();

  // Base case: when there are 3 or fewer points
  if (n <= 3)
  {
    std::vector<Point> new_hull;
    new_hull.push_back(points[0]);

    if (n == 2)
    {
      new_hull.push_back(points[1]);
    }
    else if (n == 3)
    {
      if (check_cross(points[1], points[0], points[2]) > 0)
      {
        new_hull.push_back(points[2]);
        new_hull.push_back(points[1]);
      }
      else
      {
        new_hull.push_back(points[1]);
        new_hull.push_back(points[2]);
      }
    }
    return new_hull;
  }

  // Recursively find the convex hull of the left and right halves
  std::vector<Point> ch_left = find_convex_hull({points.begin(), points.begin() + n / 2});
  std::vector<Point> ch_right = find_convex_hull({points.begin() + n / 2, points.end()});

  std::cout << "LEFT HULL" << std::endl;
  for (int i = 0; i < ch_left.size(); i++)
  {
    std::cout << "Point " << i << ":" << std::endl
              << "X: " << ch_left[i].x() << std::endl
              << "Y: " << ch_left[i].y() << std::endl;
  }

  std::cout << std::endl;
  std::cout << "RIGHT HULL" << std::endl;

  for (int i = 0; i < ch_right.size(); i++)
  {
    std::cout << "Point " << i << ":" << std::endl
              << "X: " << ch_right[i].x() << std::endl
              << "Y: " << ch_right[i].y() << std::endl;
  }
  std::cout << std::endl;
  // Merge the two convex hulls
  return merge_hulls(ch_left, ch_right);
}
