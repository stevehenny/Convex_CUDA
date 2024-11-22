#ifndef CONVEX_HULL_SERIAL
#define CONVEX_HULL_SERIAL

#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

// Custom Point class
struct Point
{
  float x, y;
  Point(float x = 0, float y = 0) : x(x), y(y)
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
}; // Global variable for the center of the polygon

// Determines the quadrant of a point (used in compare())
int quad(const Point &p);
// Checks the orientation of three points
int orientation(const Point &a, const Point &b, const Point &c);
// Compare function for sorting
bool compare(const Point &p1, const Point &q1);
// Finds upper tangent of two polygons 'a' and 'b' represented as two vectors.
std::vector<Point> merger(const std::vector<Point> &a, const std::vector<Point> &b);
// Returns the convex hull for the given set of points
std::vector<Point> divide(std::vector<Point> a);

#endif // CONVEX_HULL_SERIAL
