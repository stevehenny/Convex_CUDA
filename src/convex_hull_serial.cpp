#include "convex_hull_serial.h"
#include <cmath>
#include <utility>
#include <vector>

using namespace std;

// Check the polygon cross value (slope) between ac and bc
double check_cross(const Point &a, const Point &b, const Point &c)
{
  double cross_product = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
  const double epsilon = 1e-9; // Threshold to handle precision issues
  if (fabs(cross_product) < epsilon)
    return 0;
  return cross_product;
}

// Finds the upper tangent and returns the indexes of the two Points
pair<int, int> compute_upper_tangent(const vector<Point> &left, const vector<Point> &right)
{
  int l_length = left.size();
  int r_length = right.size();

  int left_ind = 0;
  int right_ind = 0;

  // The next two for loops find the rightmost point of the leftmost
  // point of the right hull, and the rightmost point of the left hull
  for (int i = 1; i < l_length; i++)
  {
    if (left[i].x > left[left_ind].x)
    {
      left_ind = i;
    }
  }

  for (int i = 1; i < r_length; i++)
  {
    if (right[i].x < right[right_ind].x)
    {
      right_ind = i;
    }
  }

  bool done = false;
  // while any updates happen, rotate counterclockwise on the left hull for a more
  // negative slope, and clockwise on the right hull for a more positive slope
  while (!done)
  {
    done = true;
    while (check_cross(left[left_ind], right[right_ind],
                       left[((left_ind - 1 + l_length) % l_length)]) < 0)
    {
      left_ind = (left_ind - 1 + l_length) % l_length; // Ensure proper wrap-around
      done = false;
    }

    while (check_cross(right[right_ind], left[left_ind], right[((right_ind + 1) % r_length)]) > 0)
    {
      right_ind = (right_ind + 1) % r_length; // Ensure proper wrap-around
      done = false;
    }
  }
  return make_pair(left_ind, right_ind);
}

// Finds the lower tangent and returns the indexes of the two Points
pair<int, int> compute_lower_tangent(const vector<Point> &left, const vector<Point> &right)
{
  int l_length = left.size();
  int r_length = right.size();

  int left_ind = 0;
  int right_ind = 0;

  // The next two for loops find the rightmost point of the leftmost
  // point of the right hull, and the rightmost point of the left hull
  for (int i = 1; i < l_length; i++)
  {
    if (left[i].x > left[left_ind].x)
    {
      left_ind = i;
    }
  }

  for (int i = 1; i < r_length; i++)
  {
    if (right[i].x < right[right_ind].x)
    {
      right_ind = i;
    }
  }

  bool done = false;

  // while any updates happen, rotate counterclockwise on the righ hull for a more
  // negative slope, and clockwise on the left hull for a more positive slope
  while (!done)
  {
    done = true;
    while (check_cross(left[left_ind], right[right_ind], left[((left_ind + 1) % l_length)]) > 0)
    {
      left_ind = (left_ind + 1) % l_length;
      done = false;
    }

    while (check_cross(right[right_ind], left[left_ind],
                       right[((right_ind - 1 + r_length) % r_length)]) < 0)
    {
      right_ind = (right_ind - 1 + r_length) % r_length; // Ensure proper wrap-around
      done = false;
    }
  }
  return make_pair(left_ind, right_ind);
}

// merge two hulls together
vector<Point> merger(const std::vector<Point> &left, const std::vector<Point> &right)
{
  int l_length = left.size();
  int r_length = right.size();

  pair<int, int> u_tangent = compute_upper_tangent(left, right);
  pair<int, int> l_tangent = compute_lower_tangent(left, right);

  vector<Point> hull;
  int ind = l_tangent.first;
  hull.push_back(left[ind]);

  // rotate from lower tangent left hull to upper tangent right hull
  while (ind != u_tangent.first)
  {
    ind = (ind + 1) % l_length;
    hull.push_back(left[ind]);
  }

  ind = u_tangent.second;
  hull.push_back(right[ind]);

  // rotate from lower tangent to upper tangent of right hull to lower tangent of right hull
  while (ind != l_tangent.second)
  {
    ind = (ind + 1) % r_length;
    hull.push_back(right[ind]);
  }
  return hull;
}

// top level function of serial implementation. recursively calls divide and calls merger once hulls
// are of size 2 and 3 points. Organizes points in a clockwise fashion
vector<Point> divide(vector<Point> points)
{
  if (points.size() <= 3)
  {
    vector<Point> hull;
    if (points.size() == 2)
    {
      return points;
    }
    else
    {

      hull.push_back(points[0]);
      if (check_cross(points[1], points[0], points[2]) < 0)
      {
        hull.push_back(points[1]);
        hull.push_back(points[2]);
      }
      else
      {
        hull.push_back(points[2]);
        hull.push_back(points[1]);
      }
    }
    return hull;
  }
  // Split points into two halves
  vector<Point> left(points.begin(), points.begin() + points.size() / 2);
  vector<Point> right(points.begin() + points.size() / 2, points.end());

  // Recursive call on both halves
  vector<Point> ch_left = divide(left);
  vector<Point> ch_right = divide(right);

  // Merge the results
  return merger(ch_left, ch_right);
}
