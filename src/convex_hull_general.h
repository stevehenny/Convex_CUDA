#ifndef CONVEX_HULL_GENERAL_
#define CONVEX_HULL_GENERAL_
#define DEFAULT_NUMBER_POINTS 100000
#define DEFAULT_COMMAND "both"
#define USAGE_STRING                                                                               \
  "Usage: tcp_client [--help] [-v] [-h HOST] [-p PORT] FILE\n"                                     \
  "\n"                                                                                             \
  "\n"                                                                                             \
  "Options:\n"                                                                                     \
  "  --help\n"                                                                                     \
  "  --num_points POINTS, -n POINTS (Default is 100000)\n"                                         \
  "  --command COMMAND, -c COMMAND (Default is both)\n"                                            \
  "                   Commands: both, serial, parallel"

#include <getopt.h>
#include <vector>
using namespace std;

struct Config
{
  int num_points;
  char *command;
};

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

struct DataBlock
{
  int index;
  double distance;
};

int parse_args(int argc, char *argv[], Config *config);
vector<Point> generate_random_points(int n);

#endif // CONVEX_HULL_GENERAL
