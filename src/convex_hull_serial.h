#ifndef CONVEX_HULL_SERIAL
#define CONVEX_HULL_SERIAL

#include <utility>
#include <vector>

#define CROSS(A, B, C) ((B.y() - C.y()) / (B.x() - C.x()) - (B.y() - A.y()) / (B.x() - A.x()))
class Point
{

public:
  Point(double x, double y) : _x(x), _y(y)
  {
  }
  double x() const
  {
    return _x;
  }
  double y() const
  {
    return _y;
  }

private:
  double _x, _y;
};
double check_cross(const Point &a, const Point &b, const Point &c);
std::pair<int, int> compute_upper_tangent(const std::vector<Point> &ch_left,
                                          const std::vector<Point> &ch_right);
std::pair<int, int> compute_lower_tangent(const std::vector<Point> &ch_left,
                                          const std::vector<Point> &ch_right);
std::vector<Point> merge_hulls(const std::vector<Point> &ch_left,
                               const std::vector<Point> &ch_right);
std::vector<Point> find_convex_hull(const std::vector<Point> &points);
#endif // CONVEX_HULL_SERIAL
