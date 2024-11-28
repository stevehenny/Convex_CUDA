#pragma once
#ifndef CONVEX_HULL_SERIAL
#define CONVEX_HULL_SERIAL

#include "convex_hull_general.h"
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

using namespace std;
// Custom Point class
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
