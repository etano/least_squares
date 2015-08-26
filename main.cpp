#include "least_squares.h"

using Vec = std::vector<double>;

class test{
public:
    static Vec f(const Vec& xs, int a){
        return std::move(xs);
    }
    static std::vector<Vec> grad_f(const Vec& xs, const Vec& ys, int a){
        std::vector<Vec> grad_ys(xs.size());
        for(unsigned i=0; i<xs.size(); i++){
            Vec t_grad_ys(ys.size());
            for(unsigned j=0; j<ys.size(); j++){
                if(i==j)
                    t_grad_ys[j] = 1;
                else
                    t_grad_ys[j] = 0;
            }
            grad_ys[i] = t_grad_ys;
        }
        return std::move(grad_ys);
    }
};

int main(int argc, char** argv)
{
    Vec y{-1,10,3,5,-20};
    Vec y_err{1.e-8,1.e-8,1.e-8,1.e-8,1.e-8};
    Vec p0{1,1,1,1,1};
    Vec lb{-100,-100,-100,-100,-100};
    Vec ub{100,100,100,100,100};
    double tol = 1.e-10;
    test t;
    int args0 = 10;
    std::tuple<int> args{args0};
    Vec p, p_err;
    std::cout << "Finished with result: " << least_squares::fit(y,y_err,p0,p,p_err,lb,ub,tol,t.f,t.grad_f,args) << std::endl;
    for(unsigned i=0; i<p.size(); i++){
        std::cout << y[i] << " " << p[i] << " " << p_err[i] << std::endl;
    }

    return 0;
}
