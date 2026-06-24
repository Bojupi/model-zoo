#pragma once

#include <Eigen/Dense>

namespace mve {


using Mat    = Eigen::MatrixXd;
using Vec    = Eigen::VectorXd;
using Indice = Eigen::VectorXi;

template<typename T> using Ref = Eigen::Ref<T>;


/** 均值方差切线组合权重
 *
 * @param ret:   资产收益矩阵 (T x N) 
 * @param gamma: 正则化系数
 * @return (N x 1) 权重向量
 */
Vec weight(Mat ret, double gamma=1e-4);


/** 均值方差切线组合收益
 * 
 * @param ret:   资产收益矩阵 (T x N) 
 * @param gamma: 正则化系数
 * @return (N x 1) 收益向量
 */
Vec factor(Mat ret, double gamma=1e-4);


/** 均值方差切线组合夏普比率
 * 
 * @param ret:   资产收益矩阵 (T x N) 
 * @param gamma: 正则化系数
 * @return (N x 1) 夏普比率
 */
double sharpe(Mat ret, double gamma=1e-4);


} // namespace mve