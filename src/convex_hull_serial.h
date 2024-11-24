#ifndef CONVEX_HULL_SERIAL
#define CONVEX_HULL_SERIAL

#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

using namespace std;
// Custom Point class
struct Point
{
  double x, y;
  Point(const double x = 0, const double y = 0) : x(x), y(y)
  {
  }

  // Overload the equality operator
  bool operator==(const Point &other) const
  {
    return x == other.x && y == other.y;
  }

  bool operator<(const Point &other) const
  {
    return (x < other.x) || (x == other.x && y < other.y);
  }
};
// check slope values
double check_cross(const Point &a, const Point &b, const Point &c);
// finds upper tangent
pair<int, int> compute_upper_tangent(const vector<Point> &left, const vector<Point> &right);
// finds lower tangent
pair<int, int> compute_lower_tangent(const vector<Point> &left, const vector<Point> &right);
// Finds upper tangent of two polygons 'a' and 'b' represented as two vectors.
vector<Point> merger(const vector<Point> &left, const vector<Point> &right);
// Returns the convex hull for the given set of points
vector<Point> divide(vector<Point> hull);

#endif // CONVEX_HULL_SERIAL
