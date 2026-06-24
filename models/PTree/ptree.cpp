#include <cmath>
#include <iostream>
#include <iomanip>

#include <omp.h>

#include "ptree.h"
#include "mve.h"

namespace ptree {


Node::Node(const Ref<const Mat> value, const Ref<const Indice> offset, int id, int depth, int parent) 
    : value(value), offset(offset), id(id), depth(depth), parent(parent) {
    
    ret = Vec(offset.size() - 1);

    const auto& wcol = value.col(value.cols() - 1);
    const auto& rcol = value.col(value.cols() - 2);

    for (int t = 0; t < offset.size() - 1; ++t) {
        int start  = offset[t];
        int length = offset[t+1] - offset[t];
        auto w     = wcol.segment(start, length);
        auto r     = rcol.segment(start, length);
        ret[t]     = (w.array() * r.array()).sum() / w.sum();
    }
}

void Node::clear(int left, int right, int fid, double threshold) {
    value           = Mat();
    offset          = Indice();
    this->left      = left;
    this->right     = right;
    this->fid       = fid;
    this->threshold = threshold;
}

bool Node::isLeaf() const { return left == -1 && right == -1; }

PTree::PTree(const Ref<const Mat> value, const Ref<const Indice> offset, int max_depth, int q, int min_sample_leaf, double gamma) 
    : max_depth(max_depth), min_sample_leaf(min_sample_leaf), gamma(gamma), T(offset.size() - 1), D(value.cols() - 2) {
    Node root(value, offset, 0, 1, -1);
    nodes.push_back(root);
    nodes.reserve(std::pow(2, max_depth) - 1);
    max_sharpe = mve::sharpe(root.ret, gamma);
    const double step = 2.0 / q;
    for (int i = 0; i < q - 1; ++i) thresholds.push_back(-1 + step * (i + 1));

}

PTree::PTree(const Ref<const Mat> value, const Ref<const Indice> offset, int max_depth, int q, int min_sample_leaf, const Ref<const Vec> factor, double gamma)
    : PTree(value, offset, max_depth, q, min_sample_leaf, gamma){
    this->factor = factor;
    boost        = true;
}

Mat PTree::leaves(int id) {
    int L = 0;
    for (auto& n : nodes) if (n.isLeaf()) L++;
    Mat ret(T, L + boost + (id != -1 ? 1 : 0));
    int count = 0;
    if (boost) ret.col(count++) = factor;
    for (auto& n: nodes) if (n.isLeaf() && n.id != id) ret.col(count++) = n.ret;
    return ret;
}

constexpr const char* C_RESET  = "\033[0m";
constexpr const char* C_BLUE   = "\033[34m";

void PTree::trace(const Record& record, Clock::time_point& time) {
    double elapsed = std::chrono::duration<double>(Clock::now() - time).count();
    std::cout << std::setfill('0')
              << C_BLUE           << "[split] "           << C_RESET
              << "depth="         << std::setw(2)         << nodes[record.id].depth
              << " node="         << std::setw(2)         << record.id
              << " fid="          << std::setw(2)         << record.fid
              << std::fixed       << std::setprecision(4) << std::showpos
              << " threshold="    << record.threshold
              << " sharpe="       << record.sharpe
              << std::noshowpos   << std::setprecision(2)
              << " elapsed="      << elapsed   <<   's'   << std::endl;
    time = Clock::now();
}

Vec PTree::grow() {
    Clock::time_point time = Clock::now();
    while (nodes.size() < nodes.capacity()) {
        for (auto& n: nodes)
            if (n.isLeaf()) evaluate(n);
        if (candidates.size() == 0) break;
        const Record top = candidates.top();
        split(top);
        trace(top, time);
        max_sharpe = top.sharpe;
        candidates = {};
    }
    Mat ret = leaves();
    if (boost) tree_weight =  mve::weight(ret, gamma)[0];
    return mve::factor(ret, gamma);
}

void PTree::evaluate(const Node& node) {
    if (node.depth + 1 > max_depth) return;
    Mat ret = leaves(node.id);

    std::vector<Record> buf(D * thresholds.size());

    std::vector<Mat> xsec(T);
    for (int t = 0; t < T; ++t) {
        int start  = node.offset[t];
        int length = node.offset[t+1] - node.offset[t];
        xsec[t] = node.value.middleRows(start, length);
    }

    #pragma omp parallel 
    {
        Mat retcp = ret;
        Vec left(T); Vec right(T);
        #pragma omp for collapse(2) schedule(dynamic)
        for (int fid = 0; fid < D; ++fid) {
            for (int i = 0; i < thresholds.size(); ++i) {
                bool flag = false;
                for (int t = 0; t < T; ++t) {
                    auto m   = (xsec[t].col(fid).array() <= thresholds[i]);
                    if (m.count() < min_sample_leaf || (!m).count() < min_sample_leaf) { flag = true; break; }
                    auto r   = xsec[t].col(D);
                    auto w   = xsec[t].col(D+1);
                    left[t]  = m.select(w.array() * r.array(), 0.0).sum() / m.select(w, 0.0).sum();
                    right[t] = (!m).select(w.array() * r.array(), 0.0).sum() / (!m).select(w, 0.0).sum();
                }
                if (flag) continue;
                retcp.col(retcp.cols() - 2) = left;
                retcp.col(retcp.cols() - 1) = right;
                double sharpe = mve::sharpe(retcp, gamma);
                buf[fid * thresholds.size() + i] = Record({node.id, fid, thresholds[i], sharpe});
            } 
        }
    }
    for (auto& rec : buf) if (rec.id != -1) candidates.push(rec);
}

void PTree::split(const Record& record) {
    Node& parent       = nodes[record.id];
    const auto& value  = parent.value;
    const auto& offset = parent.offset;

    std::vector<int> l_rows,    r_rows;
    std::vector<int> l_offs{0}, r_offs{0};
    l_offs.reserve(T+1); r_offs.reserve(T+1);

    for (int t = 0; t < T; ++t) {
        int start  = offset[t];
        int length = offset[t+1] - offset[t];
        auto xsec  = value.middleRows(start, length);
        auto mask  = xsec.col(record.fid).array() <= record.threshold;
        int count  = mask.count();
        for (int i = 0; i < xsec.rows(); ++i) (mask[i] ? l_rows : r_rows).push_back(start + i);
        l_offs.push_back(l_offs.back() + count);
        r_offs.push_back(r_offs.back() + (xsec.rows() - count));
    }

    Node left (value(l_rows, Eigen::all), Map<Indice>(l_offs.data(), l_offs.size()), nodes.size()  , parent.depth+1, parent.id);
    Node right(value(r_rows, Eigen::all), Map<Indice>(r_offs.data(), r_offs.size()), nodes.size()+1, parent.depth+1, parent.id);
    parent.clear(left.id, right.id, record.fid, record.threshold);
    nodes.push_back(std::move(left));
    nodes.push_back(std::move(right));
}

Mat PTree::predict(const Ref<const Mat> value) {
    Mat out(value.rows(), 2);
    Vec w = mve::weight(leaves(), gamma);
    int count = boost;
    for (auto& n : nodes) {
        if (n.isLeaf()) n.weight = w[count++];
        else n.weight = 0.0;
    }
    for (int i = 0; i < value.rows(); i++) {
        int id = 0;
        const auto& sample = value.row(i);
        while (!nodes[id].isLeaf()) {
            Node& n = nodes[id];
            if (sample[n.fid] <= n.threshold) id = n.left;
            else id = n.right;
        }
        out.col(0)[i] = nodes[id].weight;
        out.col(1)[i] = id;
    }
    return out;
}


} // namespace ptree