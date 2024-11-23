#include "convex_hull_serial.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <vector>

using namespace std;

Point mid;

// Determines the quadrant of a point (used in compare())
int quad(const Point &p)
{
  if (p.x >= 0 && p.y >= 0)
    return 1;
  if (p.x <= 0 && p.y >= 0)
    return 2;
  if (p.x <= 0 && p.y <= 0)
    return 3;
  return 4;
}

// Checks the orientation of three points:
// 0 -> collinear, 1 -> clockwise, -1 -> counterclockwise
int orientation(const Point &a, const Point &b, const Point &c)
{
  double val = (b.y - a.y) * (c.x - b.x) - (c.y - b.y) * (b.x - a.x);
  if (fabs(val) < 1e-9) // Account for floating-point precision
    return 0;
  return (val > 0) ? 1 : -1;
}

// Compare function for sorting points based on polar angle with respect to the middle point
bool compare(const Point &p1, const Point &p2)
{
  Point p = {p1.x - mid.x, p1.y - mid.y};
  Point q = {p2.x - mid.x, p2.y - mid.y};

  int one = quad(p);
  int two = quad(q);

  if (one != two)
    return (one < two);
  return (p.y * q.x < q.y * p.x);
}

// Merge function to combine the convex hulls of two polygons
vector<Point> mergeHulls(const vector<Point> &left_hull, const vector<Point> &right_hull)
{
  int left_length = left_hull.size(), right_length = right_hull.size();

  int left_ind = 0, right_ind = 0;
  for (int i = 1; i < left_length; i++)
    if (left_hull[i].x > left_hull[left_ind].x)
      left_ind = i;

  for (int i = 1; i < right_length; i++)
    if (right_hull[i].x < right_hull[right_ind].x)
      right_ind = i;

  int upper_left = left_ind, upper_right = right_ind, lower_left = left_ind,
      lower_right = right_ind;

  bool done = false;
  while (!done)
  {
    done = true;
    while (orientation(right_hull[upper_right], left_hull[upper_left],
                       left_hull[(upper_left + 1) % left_length]) > 0)
      upper_left = (upper_left + 1) % left_length;

    while (orientation(left_hull[upper_left], right_hull[upper_right],
                       right_hull[(right_length + upper_right - 1) % right_length]) < 0)
    {
      upper_right = (right_length + upper_right - 1) % right_length;
      done = false;
    }
  }

  done = false;
  while (!done)
  {
    done = true;
    while (orientation(left_hull[lower_left], right_hull[lower_right],
                       right_hull[(lower_right + 1) % right_length]) > 0)
      lower_right = (lower_right + 1) % right_length;

    while (orientation(right_hull[lower_right], left_hull[lower_left],
                       left_hull[(left_length + lower_left - 1) % left_length]) < 0)
    {
      lower_left = (left_length + lower_left - 1) % left_length;
      done = false;
    }
  }

  vector<Point> result;

  int ind = upper_left;
  result.push_back(left_hull[upper_left]);
  while (ind != lower_left)
  {
    ind = (ind + 1) % left_length;
    result.push_back(left_hull[ind]);
  }

  ind = lower_right;
  result.push_back(right_hull[lower_right]);
  while (ind != upper_right)
  {
    ind = (ind + 1) % right_length;
    result.push_back(right_hull[ind]);
  }

  return result;
}

// Brute force algorithm to find the convex hull for a small set of points
vector<Point> bruteHull(vector<Point> points)
{
  set<Point> hull;

  for (int i = 0; i < points.size(); i++)
  {
    for (int j = i + 1; j < points.size(); j++)
    {
      double x1 = points[i].x, x2 = points[j].x;
      double y1 = points[i].y, y2 = points[j].y;

      double a1 = y1 - y2;
      double b1 = x2 - x1;
      double c1 = x1 * y2 - y1 * x2;

      int pos = 0, neg = 0;
      for (int k = 0; k < points.size(); k++)
      {
        double val = a1 * points[k].x + b1 * points[k].y + c1;
        if (val > 1e-9)
          pos++;
        else if (val < -1e-9)
          neg++;
      }
      if (pos == 0 || neg == 0)
      {
        hull.insert(points[i]);
        hull.insert(points[j]);
      }
    }
  }

  vector<Point> result(hull.begin(), hull.end());

  mid = {0, 0};
  for (const auto &p : result)
  {
    mid.x += p.x;
    mid.y += p.y;
  }
  mid.x /= result.size();
  mid.y /= result.size();

  sort(result.begin(), result.end(), compare);
  return result;
}

// Divide-and-conquer method for convex hull computation
vector<Point> divide(vector<Point> points)
{
  if (points.size() <= 5)
    return bruteHull(points);

  vector<Point> left(points.begin(), points.begin() + points.size() / 2);
  vector<Point> right(points.begin() + points.size() / 2, points.end());

  vector<Point> leftHull = divide(left);
  vector<Point> rightHull = divide(right);

  return mergeHulls(leftHull, rightHull);
}
