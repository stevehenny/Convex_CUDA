#include "convex_hull_general.h"
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdlib.h>
#include <string.h>
using namespace std;

int parse_args(int argc, char **argv, Config *config)
{

  config->num_points = DEFAULT_NUMBER_POINTS;
  config->command = DEFAULT_COMMAND;

  for (int i = 1; i < argc; i++)
  {
    if (strcmp(argv[i], "--help") == 0)
    {
      cerr << USAGE_STRING;
      return 1;
    }
  }
  int c;

  while (1)
  {
    int option_index = 0;
    static struct option long_options[] = {{"num_points", required_argument, 0, 'n'},
                                           {"command", required_argument, 0, 'c'},
                                           {0, 0, 0}};
    c = getopt_long(argc, argv, "n:c:", long_options, &option_index);

    if (c == -1)
      break;

    switch (c)
    {

    case 'n':
      config->num_points = atoi(optarg);
      break;

    case 'c':
      config->command = optarg;
      break;

    case '?':
      cerr << USAGE_STRING;
      return 1;
      break;
    }
  }
  return 0;
}

vector<Point> generate_random_points(int n)
{
  vector<Point> points;

  random_device rd;
  mt19937 gen(rd());
  normal_distribution<> dist_x(500.0, 50.0);
  normal_distribution<> dist_y(500.0, 50.0);
  // Generate random points
  for (int i = 0; i < n; ++i)
  {
    const double x = (dist_x(gen));
    const double y = (dist_y(gen));
    points.push_back(Point{x, y});
  }

  return points;
}
