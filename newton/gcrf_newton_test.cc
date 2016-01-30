
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include "gcrf_newton.h"

using namespace std;
using namespace Eigen;

void TestSmall() {
  double small[] = {
    0.9471,   -0.0997,   -0.0146,   -0.0522,   -0.1149,
    -0.0997,   0.9341,   -0.0266,    0.0343,    0.0380,
    -0.0146,  -0.0266,    0.3205,    0.0583,   -0.0335,
    -0.0522,   0.0343,    0.0583,    0.3031,    0.0081,
    -0.1149,   0.0380,   -0.0335,    0.0081,    0.3516
  };

  int p = 2;
  int n = 3;
  MatrixXd S = Map<MatrixXd>((double*)&small, n+p, n+p);
  MatrixXd Lambda = MatrixXd::Identity(p, p);
  MatrixXd Theta = MatrixXd::Zero(n, p);

  GCRFParams params;
  GCRFStats stats;
  OptimizeGCRF(S, 0.05, params, Lambda, Theta, &stats);
  assert(fabs(stats.objval.back() - 1.8474) < 1e-4);
  cout << "PASSED" << endl;
}

int main() {
  TestSmall();
  // TODO(mwytock): Add larger tests
}
