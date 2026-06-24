#include "mve.h"

namespace mve {


Vec weight(Mat ret, double gamma) {
    if (ret.cols() ==  1) return Vec::Ones(1);
    const int N = ret.cols();
    Mat cov = ret.transpose() * ret / double(ret.rows()) + gamma * Mat::Identity(N, N);
    Vec w = cov.completeOrthogonalDecomposition().solve(ret.colwise().mean().transpose());
    w = w.array() - w.mean();
    w /= w.cwiseAbs().sum();
    return w;
}

Vec factor(Mat ret, double gamma) {
    Vec w = weight(ret, gamma);
    return ret * w;
}

double sharpe(Mat ret, double gamma) {
    Vec f = factor(ret, gamma);
    double mean = f.mean();
    double stddev = std::sqrt((f.array() - mean).square().mean());
    return mean / stddev;
}


} // namespace mve