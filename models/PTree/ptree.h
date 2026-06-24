#pragma once

#include <queue>
#include <chrono>

#include <Eigen/Dense>

namespace ptree {

using Clock  = std::chrono::steady_clock;

using Mat    = Eigen::MatrixXd;
using Vec    = Eigen::VectorXd;
using Indice = Eigen::VectorXi;

template<typename T> using Ref = Eigen::Ref<T>;
template<typename T> using Map = Eigen::Map<T>;


struct Node {

    /** 基于非平衡面板数据初始化节点
     * 
     * @param value:  非平衡面板数据 ((T x N_t) x (D + 2))，最后两列为收益和预定义权重
     * @param offset: 偏移项 ((T + 1) x 1)
     * @param id:     树中节点的索引（根节点为0）
     * @param depth:  节点深度（根节点为1）
     * @param parent: 父节点id（根节点为-1）
     */
    Node(const Ref<const Mat> value, const Ref<const Indice> offset, int id, int depth, int parent);

    /** 节点分裂后，释放value/offset以节省内存，并更新子节点信息
     * 
     * @param left:      左节点id
     * @param right:     右节点id
     * @param fid:       分裂所有的特征编号
     * @param threshold: 分裂阈值
    */
    void clear(int left, int right, int fid, double threshold);

    bool isLeaf() const;

    bool operator==(const Node& other) const {
        return id == other.id;
    }

    bool operator!=(const Node& other) const {
        return id != other.id;
    }

    int id        = -1;
    int left      = -1;
    int right     = -1;
    int parent    = -1;
    int fid       = -1;
    int depth     = 0;

    double weight    = 0.0;
    double threshold = 0.0;

    Mat    value;
    Indice offset;
    Vec    ret;
};


struct Record {
    int    id        = -1;
    int    fid       = -1;
    double threshold = -1.0;
    double sharpe    = -100.0;

    bool operator<(const Record& other) const {
        return sharpe < other.sharpe;
    }
};


class PTree {
public:

    /** 构建面板树
     * 
     * @param value:           非平衡面板数据 ((T x N_t) x (D + 2))，最后两列为收益和预定义权重
     * @param offset:          偏移项 ((T + 1) x 1)
     * @param max_depth:       树的最大深度
     * @param q:               分裂阈值分位数
     * @param min_sample_leaf: 叶子节点最小样本数
     * @param gamma            权重估计的正则化系数
     */
    PTree(const Ref<const Mat> value, const Ref<const Indice> offset, int max_depth, int q, int min_sample_leaf, double gamma=1e-4);

    /** 构建提升树（面板树+基准因子）
     * 
     * @param factor: 基准因子收益序列 (T)
     */
    PTree(const Ref<const Mat> value, const Ref<const Indice> offset, int max_depth, int q, int min_sample_leaf, const Ref<const Vec> factor, double gamma=1e-4);

    /** 生长
     * 
     * @return: SDF因子收益序列 (T x 1)
     */
    Vec grow();

    /** 获取新样本所属叶子节点的编号和权重
     * 
     * @param value: 资产特征矩阵 ((T x N_t) x D)
     * @return: (N x 2)矩阵，第一列为权重，第二列为编号
     */
    Mat predict(const Ref<const Mat> value);

    double tree_weight  = 1.0; // 提升树的权重（简单面板树的权重为1）

private:

    /** 对选定叶子节点遍历评估所有候选切分
     * 
     * @param node: 选定叶子节点
     */
    void evaluate(const Node& node);

    /** 执行最优分裂
     * 
     * @param record: 最优分裂方案；
     */
    void split(const Record& record);

    /** 输出分裂日志
     * 
     * @param record: 最优分裂方案；
     * @param time:   上一次分裂时间
     */
    void trace(const Record& record, Clock::time_point& time);

    /** 获取除选定节点外的所有叶组合收益面板
     * 
     * @param id: 排除指定节点（如evaluate中排除自身）
     * @return: (T x L) 所有叶组合收益面板
     */
    Mat leaves(int id=-1);

    int max_depth       = 0;
    int min_sample_leaf = 0;
    int T               = 0;
    int D               = 0;
    double max_sharpe   = -100.0;
    double gamma        = 1e-4;

    std::vector<Node> nodes;
    std::vector<double> thresholds;
    std::priority_queue<Record> candidates;

    Vec factor;
    bool boost          = false;
};


} // namespace ptree