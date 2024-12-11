#ifndef CONVEX_HULL_GENERAL_
#define CONVEX_HULL_GENERAL_
#define DEFAULT_NUMBER_POINTS 100000
#define DEFAULT_COMMAND ((char *)"serial")
#define USAGE_STRING                                                                               \
  "Usage: tcp_client [--help] [-v] [-h HOST] [-p PORT] FILE\n"                                     \
  "\n"                                                                                             \
  "\n"                                                                                             \
  "Options:\n"                                                                                     \
  "  --help\n"                                                                                     \
  "  --num_points POINTS, -n POINTS (Default is 100000)\n"                                         \
  "  --command COMMAND, -c COMMAND (Default is serial)\n"                                          \
  "                   Commands: both, serial, parallel"

#include "convex_hull_cuda.cuh"
#include <getopt.h>
#include <vector>
using namespace std;

// Config passed into parse_args
struct Config
{
  int num_points;
  char *command;
};

int parse_args(int argc, char *argv[], Config *config);
vector<Point> generate_random_points(int n);
void scramble_points(vector<Point> &points);
#endif // CONVEX_HULL_GENERAL
