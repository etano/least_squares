#include "least_squares.h"

using Vec = std::vector<double>;

class test{
public:
    /// Calculate Pls from inv model with parameters ps
    static Vec calc_Pls_from_Pl_inv_ps(const Vec& ps, int N){
        Vec Pls(N);
        for(unsigned l=0; l<Pls.size(); l++){
            Pls[l] = 1;
            for(unsigned i=0; i<ps.size(); i++)
                Pls[l] += ps[i]/pow(l+1,i+1);
        }
        return std::move(Pls);
    }
    /// Calculate Pls from exponential model with parameters ps
    static Vec calc_Pls_from_Pl_exp_ps(const Vec& ps, int N){
        Vec Pls(N);
        for(unsigned l=0; l<Pls.size(); l++){
            double tot(0);
            for(unsigned i=1; i<ps.size(); i+=2)
                tot += ps[i]*pow(l+1,ps[i+1]);
            Pls[l] = 1 + ps[0]*exp(-tot);
        }
        return std::move(Pls);
    }
    static Vec f(const Vec& xs, int a){
        return std::move(xs);
    }
    /// Calculate Grad Pls from exponential model with parameters ps
    static std::vector<Vec> calc_grad_Pls_from_Pl_exp_ps(const Vec& ps, const Vec& Pls, int N){
        std::vector<Vec> grad_Pls_ps(Pls.size());
        for(unsigned l=0; l<Pls.size(); l++){
            Vec grad_Pls_ps_l(ps.size());
            grad_Pls_ps_l[0] = (Pls[l]-1)/ps[0];
            for(unsigned i=1; i<ps.size(); i+=2){
                grad_Pls_ps_l[i] = -ps[0]*pow(double(l+1),ps[i+1])*(Pls[l]-1)/ps[0];
                grad_Pls_ps_l[i+1] = -ps[0]*ps[i]*log(double(l+1))*pow(double(l+1),ps[i+1])*(Pls[l]-1)/ps[0];
            }
            grad_Pls_ps[l] = grad_Pls_ps_l;
        }
        return std::move(grad_Pls_ps);
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
    Vec y{15.50976275363625,5.485410905620537,3.011517956446927,2.028025211656552,1.561240183601053,1.318552146349282,1.185164432377111,1.109217028127513,1.06499953739507,1.03889526616426,1.023351744819556,1.014047903852249,1.008461153453396,1.005099943217989,1.003075338900046,1.001854968149406,1.001119050921592,1.000675158031187,1.000407367607005,1.000245800585313,1.000148316205731,1.000089495231626,1.000054002586685,1.000032586016012,1.000019662973773,1.000011865003624,1.000007159570965,1.000004320225254,1.000002606909456,1.000001573061137,1.00000094921658,1.000000572776334,1.000000345624751};
    Vec y_err{1.e-8,1.e-8,1.e-8,1.e-8,1.e-8,1.e-8,1.e-8,1.e-8,1.e-8,1.e-8,1.e-8,1.e-8,1.e-8,1.e-8,1.e-8,1.e-8,1.e-8,1.e-8,1.e-8,1.e-8,1.e-8,1.e-8,1.e-8,1.e-8,1.e-8,1.e-8,1.e-8,1.e-8,1.e-8,1.e-8,1.e-8,1.e-8,1.e-8};
    Vec p0{1,1,1};
    Vec lb{1e-6,1e-6,1e-6};
    Vec ub{1e3,1e1,1};
    double tol = 1.e-10;
    test t;
    int args0 = y.size();
    std::tuple<int> args{args0};
    Vec p(p0), p_err(p0);
    least_squares::fit_cmpfit(y,y_err,p0,p,p_err,lb,ub,tol,t.calc_Pls_from_Pl_exp_ps,t.calc_grad_Pls_from_Pl_exp_ps,args,true);
    for(unsigned i=0; i<p.size(); i++){
        std::cout << p0[i] << " " << p[i] << " " << p_err[i] << std::endl;
    }
    auto Pls = t.calc_Pls_from_Pl_exp_ps(p,y.size());
    double tot = 0;
    for(unsigned l=0; l<Pls.size(); l++){
        std::cout << l << " " << Pls[l] << " " << y[l] << std::endl;
        tot += pow(Pls[l]-y[l],2);
    }
    std::cout << tot << std::endl;

    least_squares::fit_nlopt(y,y_err,p0,p,p_err,lb,ub,tol,t.calc_Pls_from_Pl_exp_ps,t.calc_grad_Pls_from_Pl_exp_ps,args,true);
    for(unsigned i=0; i<p.size(); i++){
        std::cout << p[i] << " " << p_err[i] << std::endl;
    }
    Pls = t.calc_Pls_from_Pl_exp_ps(p,y.size());
    tot = 0;
    for(unsigned l=0; l<Pls.size(); l++){
        std::cout << l << " " << Pls[l] << " " << y[l] << std::endl;
        tot += pow(Pls[l]-y[l],2);
    }
    std::cout << tot << std::endl;

    least_squares::fit_ceres(y,y_err,p0,p,p_err,lb,ub,tol,t.calc_Pls_from_Pl_exp_ps,t.calc_grad_Pls_from_Pl_exp_ps,args,true);
    for(unsigned i=0; i<p.size(); i++){
        std::cout << p[i] << " " << p_err[i] << std::endl;
    }
    Pls = t.calc_Pls_from_Pl_exp_ps(p,y.size());
    tot = 0;
    for(unsigned l=0; l<Pls.size(); l++){
        std::cout << l << " " << Pls[l] << " " << y[l] << std::endl;
        tot += pow(Pls[l]-y[l],2);
    }
    std::cout << tot << std::endl;


    return 0;
}
