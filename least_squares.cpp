#include "least_squares.h"

using umap = std::unordered_map<unsigned,double>;

umap f_test(const umap& x, int y){
    return x;
}

int main(int argc, char** argv)
{
    unsigned n_var = 5;
    double tol = 1.e-16;
    nlopt::opt opt(nlopt::LN_NELDERMEAD, n_var);
    auto y_orig = umap{{0,-1},{1,10},{2,3},{3,5},{4,-20}};
    std::tuple<int> args{1};
    least_squares::LeastSquares<umap,umap,int> lsq(y_orig,f_test,args);
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
