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
        pars[i].fixed = false;
        pars[i].limited[0] = true;
        pars[i].limited[1] = true;
        pars[i].limits[0] = lb[i];
        pars[i].limits[1] = ub[i];
        pars[i].parname = "";
        pars[i].step = 0;
        pars[i].relstep = 0;
        pars[i].side = 0;
        pars[i].deriv_debug = false;
    }

    // Config struct
    mp_config config;
    config.ftol = tol;
    config.nprint = 1;
    config.xtol = 0;
    config.gtol = 0;
    config.stepfactor = 0;
    config.epsfcn = 0;
    config.maxiter = 0;
    config.douserscale = 0;
    config.covtol = 0;
    config.nofinitecheck = 0;
    config.maxfev = 0;

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
