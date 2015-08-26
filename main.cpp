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
                    grad_ys[x.first][y.first] = 1;
                else
                    grad_ys[x.first][y.first] = 0;
            }
        return std::move(grad_ys);
    }
};

int main(int argc, char** argv)
{
    unsigned n_par = 5;

    // Parameters structs
    double p[n_par];
    mp_par pars[n_par];
    for(unsigned i=0; i<n_par; i++){
        p[i] = 1;
        pars[i].limited[0] = true;
        pars[i].limited[1] = true;
        pars[i].limits[0] = -100;
        pars[i].limits[1] = 100;
        pars[i].side = 0;
        pars[i].deriv_debug = true;
    }

    // Config struct
    mp_config config;
    config.ftol = 1.e-10;
    config.nprint = 1;

    // Result struct
    mp_result result;
    memset(&result,0,sizeof(result));       /* Zero results structure */
    double perror[n_par];
    result.xerror = perror;

    // Functor
    auto y_orig = umap{{0,-1},{1,10},{2,3},{3,5},{4,-20}};
    std::tuple<int> args{1};
    test t;
    least_squares::CMPFitFunctor<umap,umap,umap_umap,int> functor(y_orig,t.f_test,t.grad_f_test,args);

    // Run least squares
    int status = mpfit(least_squares::residual<decltype(functor),umap,umap,umap_umap>, y_orig.size(), n_par, p, pars, &config, (void *) &functor, &result);

    printf("*** testlinfit status = %d\n", status);
    for(unsigned i=0; i<n_par; i++)
        std::cout << y_orig[i] << " " << p[i] << " " << perror[i] << std::endl;

    return 0;
}
