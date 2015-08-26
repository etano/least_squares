#include "least_squares.h"

using umap = std::unordered_map<unsigned,double>;
using umap_umap = std::unordered_map<unsigned,umap>;

class test{
public:
    static umap f_test(const umap& xs, int a){
        return std::move(xs);
    }
    static umap_umap grad_f_test(const umap& xs, const umap& ys, int a){
        umap_umap grad_ys;
        for(const auto& x : xs)
            for(const auto& y : ys){
                if(y.first == x.first)
                    grad_ys[y.first][x.first] = 1;
                else
                    grad_ys[y.first][x.first] = 0;
            }
        return std::move(grad_ys);
    }
};

int main(int argc, char** argv)
{
    unsigned n_var = 5;
    double tol = 1.e-16;
    nlopt::opt opt(nlopt::LD_MMA, n_var);
    auto y_orig = umap{{0,-1},{1,10},{2,3},{3,5},{4,-20}};
    std::tuple<int> args{1};
    test t;
    least_squares::LeastSquares<umap,umap,umap_umap,int> lsq(y_orig,t.f_test,t.grad_f_test,args);
    opt.set_min_objective(least_squares::residual<decltype(lsq)>, &lsq);
    opt.set_lower_bounds(std::vector<double>(n_var,-100));
    opt.set_upper_bounds(std::vector<double>(n_var,100));
    opt.set_xtol_abs(tol);
    std::vector<double> x(n_var,100);
    double diff;
    opt.optimize(x,diff);
    for(const auto& y : y_orig)
        std::cout << y.first << " " << x[y.first] << std::endl;
    std::cout << diff << std::endl;
    return 0;
}
