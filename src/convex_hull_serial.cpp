#include "convex_hull_serial.h"
#include <bits/stdc++.h>
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

// Checks the orientation of three points
int orientation(const Point &a, const Point &b, const Point &c)
{
  float res = (b.y - a.y) * (c.x - b.x) - (c.y - b.y) * (b.x - a.x);

  if (res == 0)
    return 0;
  if (res > 0)
    return 1;
  return -1;
}

// Compare function for sorting
bool compare(const Point &p1, const Point &q1)
{
  Point p = {p1.x - mid.x, p1.y - mid.y};
  Point q = {q1.x - mid.x, q1.y - mid.y};

  int one = quad(p);
  int two = quad(q);

  if (one != two)
    return (one < two);
  return (p.y * q.x < q.y * p.x);
}

// Finds upper tangent of two polygons 'a' and 'b' represented as two vectors.
vector<Point> merger(const vector<Point> &a, const vector<Point> &b)
{
  int n1 = a.size(), n2 = b.size();

  int ia = 0, ib = 0;
  for (int i = 1; i < n1; i++)
    if (a[i].x > a[ia].x)
      ia = i;

  for (int i = 1; i < n2; i++)
    if (b[i].x < b[ib].x)
      ib = i;

  int inda = ia, indb = ib;
  bool done = false;
  while (!done)
  {
    done = true;
    while (orientation(b[indb], a[inda], a[(inda + 1) % n1]) >= 0)
      inda = (inda + 1) % n1;

    while (orientation(a[inda], b[indb], b[(n2 + indb - 1) % n2]) <= 0)
    {
      indb = (n2 + indb - 1) % n2;
      done = false;
    }
  }

  int uppera = inda, upperb = indb;
  inda = ia, indb = ib;
  done = false;
  while (!done)
  {
    done = true;
    while (orientation(a[inda], b[indb], b[(indb + 1) % n2]) >= 0)
      indb = (indb + 1) % n2;

    while (orientation(b[indb], a[inda], a[(n1 + inda - 1) % n1]) <= 0)
    {
      inda = (n1 + inda - 1) % n1;
      done = false;
    }
  }

  int lowera = inda, lowerb = indb;
  vector<Point> ret;

  int ind = uppera;
  ret.push_back(a[uppera]);
  while (ind != lowera)
  {
    ind = (ind + 1) % n1;
    ret.push_back(a[ind]);
  }

  ind = lowerb;
  ret.push_back(b[lowerb]);
  while (ind != upperb)
  {
    ind = (ind + 1) % n2;
    ret.push_back(b[ind]);
  }
  return ret;
}

// Brute force algorithm to find convex hull for a small set of points
vector<Point> bruteHull(vector<Point> a)
{
  set<Point> s;

  for (int i = 0; i < a.size(); i++)
  {
    for (int j = i + 1; j < a.size(); j++)
    {
      float x1 = a[i].x, x2 = a[j].x;
      float y1 = a[i].y, y2 = a[j].y;

      float a1 = y1 - y2;
      float b1 = x2 - x1;
      float c1 = x1 * y2 - y1 * x2;

      int pos = 0, neg = 0;
      for (int k = 0; k < a.size(); k++)
      {
        if (a1 * a[k].x + b1 * a[k].y + c1 <= 0)
          neg++;
        if (a1 * a[k].x + b1 * a[k].y + c1 >= 0)
          pos++;
      }
      if (pos == a.size() || neg == a.size())
      {
        s.insert(a[i]);
        s.insert(a[j]);
      }
    }
  }

  vector<Point> ret(s.begin(), s.end());
  mid = {0, 0};
  int n = ret.size();
  for (int i = 0; i < n; i++)
  {
    mid.x += ret[i].x;
    mid.y += ret[i].y;
    ret[i].x *= n;
    ret[i].y *= n;
  }
  sort(ret.begin(), ret.end(), compare);
  for (int i = 0; i < n; i++)
  {
    ret[i].x /= n;
    ret[i].y /= n;
  }
  return ret;
}

// Returns the convex hull for the given set of points
vector<Point> divide(vector<Point> a)
{
  cout << a.size() << endl;
  if (a.size() <= 5)
    return bruteHull(a);

  vector<Point> left(a.begin(), a.begin() + a.size() / 2);
  vector<Point> right(a.begin() + a.size() / 2, a.end());

  vector<Point> left_hull = divide(left);
  vector<Point> right_hull = divide(right);

  return merger(left_hull, right_hull);
}
