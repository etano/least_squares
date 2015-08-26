#ifndef LEAST_SQUARES_HPP
#define LEAST_SQUARES_HPP

#include "cmpfit/mpfit.h"

namespace least_squares{

template<typename Func, typename Tup, std::size_t... index>
decltype(auto) invoke_helper(Func&& func, Tup&& tup, std::index_sequence<index...>)
{
    return func(std::get<index>(std::forward<Tup>(tup))...);
}

template<typename Func, typename Tup>
decltype(auto) invoke(Func&& func, Tup&& tup)
{
    constexpr auto Size = std::tuple_size<typename std::decay<Tup>::type>::value;
    return invoke_helper(std::forward<Func>(func),std::forward<Tup>(tup),std::make_index_sequence<Size>{});
}

template<typename T, typename U>
U convert_map(const std::vector<T>& x){
    U m(x.size());
    for(size_t i=0; i<x.size(); i++)
        m[i] = x[i];
    return std::move(m);
}

template<typename T, typename U>
U convert_container(unsigned n, T* x){
    U c(n);
    for(size_t i=0; i<n; i++)
        c[i] = x[i];
    return std::move(c);
}

template<typename T, typename U, typename V>
U convert_container(const unsigned n, const unsigned m, T** x){
    U c(n);
    for(size_t i=0; i<n; i++){
        V t_c(m);
        for(size_t j=0; j<m; j++)
            t_c[j] = x[i][j];
        c[i] = t_c;
    }
    return std::move(c);
}

template<class T, class U>
std::vector<T> convert_vec(const U& m){
    std::vector<T> w(m.size());
    for(const auto& e : m)
        w[e.first] = e.second;
    return std::move(w);
}

template<typename T, typename U, typename V, typename... Args>
class LeastSquares{
public:
    /// Constructor with gradient
    LeastSquares(const T& y_orig, T (*f)(const U&,Args...)&, V (*grad_f)(const U&,const T&,Args...)&, const std::tuple<const Args&...>& args)
        : f_(f), grad_f_(grad_f), args_(args), y_orig_(y_orig)
    {}

    /// Compute the residual
    double get_residual(const std::vector<double>& x, std::vector<double>& grad){
        auto xs = convert_map<double,U>(x);
        auto ys = invoke(f_,std::tuple_cat(std::tuple<U>(xs),args_));
        double tot(0);
        for(const auto& y : ys)
            tot += pow(y.second-y_orig_.at(y.first),2);
        double r = tot;
        if(!grad.empty()){
            auto grad_ys = invoke(grad_f_,std::tuple_cat(std::tuple<U,T>(xs,ys),args_));
            for(unsigned i=0; i<grad.size(); i++)
                grad[i] = 0;
            for(const auto& y : ys){
                auto diff = y.second-y_orig_.at(y.first);
                for(unsigned i=0; i<grad.size(); i++)
                    grad[i] += 2*diff*grad_ys.at(y.first).at(i);
            }
        }
        return r;
    }
private:
    T (*f_)(const U&,Args...); ///> The function which to to be fitted
    V (*grad_f_)(const U&,const T&,Args...); ///> The gradient of the function which to to be fitted
    const std::tuple<const Args&...>& args_; /// Additional arguments
    const T& y_orig_; ///> Original y values
};

template<typename T>
double residual(const std::vector<double>& x, std::vector<double>& grad, void* f_data){
    T* lsq = static_cast<T*>(f_data);
    return lsq->get_residual(x,grad);
}

template<typename T, typename U, typename F, typename GradF, typename... Args>
std::unordered_map<T,U> perform_least_squares(const std::unordered_map<T,U>& y, const std::unordered_map<T,U>& x0, const std::vector<U>& lb, const std::vector<U>& ub, const U& tol, F f, GradF grad_f, const std::tuple<Args...>& args, bool use_grad=false){
    unsigned n_var = x0.size();
    LeastSquares<std::unordered_map<T,U>,std::unordered_map<T,U>,std::unordered_map<T,std::unordered_map<T,U>>,Args...> lsq(y,f,grad_f,args);
    if(use_grad){
        nlopt::opt opt(nlopt::LN_BOBYQA, n_var);
        opt.set_min_objective(residual<decltype(lsq)>, &lsq);
        opt.set_lower_bounds(lb);
        opt.set_upper_bounds(ub);
        opt.set_ftol_rel(tol);
        std::vector<U> x(convert_vec<U,std::unordered_map<T,U>>(x0));
        U diff;
        opt.optimize(x,diff);
        return std::move(convert_map<U,std::unordered_map<T,U>>(x));
    }else{
        nlopt::opt opt(nlopt::LN_NELDERMEAD, n_var);
        opt.set_min_objective(residual<decltype(lsq)>, &lsq);
        opt.set_lower_bounds(lb);
        opt.set_upper_bounds(ub);
        opt.set_ftol_rel(tol);
        std::vector<U> x(convert_vec<U,std::unordered_map<T,U>>(x0));
        U diff;
        opt.optimize(x,diff);
        return std::move(convert_map<U,std::unordered_map<T,U>>(x));
    }
}

/// Compute the residual
///
/// m - number of data points
/// n - number of parameters
/// p - array of fit parameters
/// res - array of residuals to be returned
/// dvec - array of user defined derivatives (default 0)
/// vars - private data (struct vars_struct *)
///
/// RETURNS: error code (0 = success)
template<typename FunctorType, typename T, typename U, typename V>
int residual(int m, int n, double *p, double *res, double **dvec, void *vars){
    FunctorType& lsq(*static_cast<FunctorType*>(vars));
    U ps = convert_container<double,U>(n,p);
    T ress = convert_container<double,T>(m,res);
    V grad_ps;
    if(dvec)
        grad_ps = convert_container<double,V,U>(m,n,dvec);
    int result = lsq(ps,ress,grad_ps);
    for(int i=0; i<n; i++){
        res[i] = ress[i];
        if(!grad_ps.empty()){
            for(int j=0; j<m; j++)
                dvec[j][i] = grad_ps[i][j];
        }
    }
    return result;
}

template<typename T, typename U, typename V, typename... Args>
struct CMPFitFunctor{
    /// Constructor with gradient
    CMPFitFunctor(const T& y_orig, const T& y_err_orig, T (*f)(const U&,Args...)&, V (*grad_f)(const U&,const T&,Args...)&, const std::tuple<Args...>& args)
        : f_(f), grad_f_(grad_f), args_(args), y_orig_(y_orig), y_err_orig_(y_err_orig)
    {}

    /// Sets residual and gradients of residuals with respect to the parameters
    int operator()(U& ps, T& ress, V& grad_ps){
        T ys(invoke(f_,std::tuple_cat(std::tuple<U>(ps),args_)));
        for(unsigned i=0; i<ys.size(); i++){
            ress[i] = ys.at(i)-y_orig_.at(i);
            if(y_err_orig_.at(i)!=0)
                ress[i] /= y_err_orig_.at(i);
        }
        if(!grad_ps.empty()){
            grad_ps = invoke(grad_f_,std::tuple_cat(std::tuple<U,T>(ps,ress),args_));
            for(unsigned i=0; i<ys.size(); i++){
                if(y_err_orig_.at(i)!=0)
                for(unsigned j=0; j<ps.size(); j++)
                    grad_ps[i][j] /= y_err_orig_.at(i);
            }
        }
        return 0;
    }

    T (*f_)(const U&,Args...); ///> The fitted function
    V (*grad_f_)(const U&,const T&,Args...); ///> The gradient of the fitted function
    const std::tuple<Args...>& args_; /// Additional arguments for the fitted function
    const T& y_orig_; ///> Original y values
    const T& y_err_orig_; ///> Original y error values
};

template<typename VecT, typename T, typename F, typename GradF, typename... Args>
int fit(const VecT& y, const VecT& y_err, const VecT& p0, VecT& ps, VecT& p_errs, const VecT& lb, const VecT& ub, const T& tol, F f, GradF grad_f, const std::tuple<Args...>& args){
    // Problem size
    unsigned n_y = y.size();
    unsigned n_par = p0.size();

    // Parameters structs
    double p[n_par];
    mp_par pars[n_par];
    for(unsigned i=0; i<n_par; i++){
        p[i] = p0[i];
        pars[i].limited[0] = true;
        pars[i].limited[1] = true;
        pars[i].limits[0] = lb[i];
        pars[i].limits[1] = ub[i];
        //pars[i].deriv_debug = true;
    }

    // Config struct
    mp_config config;
    config.ftol = tol;
    config.nprint = 1;

    // Result struct
    mp_result result;
    memset(&result,0,sizeof(result));
    double perror[n_par];
    result.xerror = perror;

    // Functor
    CMPFitFunctor<VecT,VecT,std::vector<VecT>,Args...> functor(y,y_err,f,grad_f,args);

    // Run least squares
    int status = mpfit(least_squares::residual<decltype(functor),VecT,VecT,std::vector<VecT>>,n_y,n_par,p,pars,&config,(void *) &functor,&result);

    // Set parameters
    ps.resize(p0.size());
    p_errs.resize(p0.size());
    for(unsigned i=0; i<n_par; i++){
        ps[i] = p[i];
        p_errs[i] = perror[i];
    }

    return status;
}

}

#endif
