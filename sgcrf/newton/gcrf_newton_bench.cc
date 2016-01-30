
// Linux: 
// g++ -O2 -o gcrf_newton_bench -I/usr/include/eigen3
//   -I/usr/local/MATLAB/R2012b/extern/include  gcrf_newton_bench.cc
//   gcrf_newton.cc  -Wl,-rpath-link,/usr/local/MATLAB/R2012b/bin/glnxa64
//   -L/usr/local/MATLAB/R2012b/bin/glnxa64 -lmat -lmx  

#include <mat.h>
#include <iostream>
#include <Eigen/Dense>
#include "gcrf_newton.h"

using namespace Eigen;
using namespace std;

void LoadInput(const char* filename, MatrixXd* S, int* p, double* lambda) {
  MATFile* file = matOpen(filename, "r");
  assert(file);
  mxArray* array = matGetVariable(file, "S");
  assert(array);
  *S = Map<MatrixXd>(mxGetPr(array), mxGetM(array), mxGetN(array));
  *p = (int)mxGetScalar(matGetVariable(file, "p"));
  *lambda = mxGetScalar(matGetVariable(file, "lambda"));
  assert(p);
  assert(lambda);
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: ./gcrf_newton_bench <bench1.mat> [bench2.mat] ...\n");
    return 1;
  }

  for (int i = 1; i < argc; i++) {
    MatrixXd S;
    int p;
    double lambda;
    cout << "Loading input " << argv[i] << " ..." << endl;
    LoadInput(argv[i], &S, &p, &lambda);
    int n = S.rows() - p;

    cout << "Done loading input" << endl 
         << "n=" << n << " p=" << p << " lambda=" << lambda << endl;
    
    GCRFParams params;
    GCRFStats stats;
    MatrixXd Lambda = MatrixXd::Identity(p, p);
    MatrixXd Theta = MatrixXd::Zero(n, p);

    params.max_iters = 10;
    
    cout << "Optimizing GCRF ..." << endl;
    double start = clock();
    OptimizeGCRF(S, lambda, params, Lambda, Theta, &stats);
    cout << "Total time: " << (clock()-start)/CLOCKS_PER_SEC;
  }
  return 0;
}
 
