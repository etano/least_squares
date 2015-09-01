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

template<typename T, typename U, typename... Args>
class CeresNumericDiffFunctor{
public:
    /// Constructor with gradient
    CeresNumericDiffFunctor(const T& y_orig, const T& y_err_orig, const unsigned n_par, T (*f)(const U&,Args...), const std::tuple<Args...>& args)
        : f_(f), args_(args), y_orig_(y_orig), y_err_orig_(y_err_orig), n_par_(n_par)
    {}

    /// Sets residual and gradients of residuals with respect to the parameters
    template <typename CT>
    bool operator()(CT const* const* parameters, CT* residuals) const {
        std::vector<double> ps;
        for(unsigned i=0; i<n_par_; i++)
            ps.push_back(parameters[0][i]);
        T ys(invoke(f_,std::tuple_cat(std::tuple<U>(ps),args_)));
        for(unsigned i=0; i<y_orig_.size(); i++)
            residuals[i] = CT(y_orig_[i]) - CT(ys[i]);
        return true;
    }

private:
    T (*f_)(const U&,Args...); ///> The fitted function
    const std::tuple<Args...>& args_; /// Additional arguments for the fitted function
    const T& y_orig_; ///> Original y values
    const T& y_err_orig_; ///> Original y error values
    const unsigned n_par_; ///> Number of parameters
};

template<typename T, typename U, typename V, typename... Args>
class CeresAnalyticDiffFunctor : public ceres::CostFunction{
public:
    /// Constructor with gradient
    CeresAnalyticDiffFunctor(const T& y_orig, const T& y_err_orig, const unsigned n_par, T (*f)(const U&,Args...), V (*grad_f)(const U&,const T&,Args...), const std::tuple<Args...>& args)
        : f_(f), grad_f_(grad_f), args_(args), y_orig_(y_orig), y_err_orig_(y_err_orig), n_par_(n_par)
    {
        mutable_parameter_block_sizes()->push_back(n_par_);
        set_num_residuals(y_orig_.size());
    }

    /// Sets residual and gradients of residuals with respect to the parameters
    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
        std::vector<double> ps;
        for(unsigned i=0; i<n_par_; i++)
            ps.push_back(parameters[0][i]);
        T ys(invoke(f_,std::tuple_cat(std::tuple<U>(ps),args_)));
        for(unsigned i=0; i<y_orig_.size(); i++)
            residuals[i] = y_orig_[i] - ys[i];
        if(jacobians != NULL){
            if(jacobians[0] != NULL){
                auto grad_ys = invoke(grad_f_,std::tuple_cat(std::tuple<U,T>(ps,ys),args_));
                for(unsigned i=0; i<n_par_; i++){
                    for(unsigned j=0; j<ys.size(); j++){
                        jacobians[0][j*n_par_ + i] = -grad_ys[j][i];
                    }
                }
            }
        }
        return true;
    }

private:
    T (*f_)(const U&,Args...); ///> The fitted function
    V (*grad_f_)(const U&,const T&,Args...); ///> Gradient of the fitted function
    const std::tuple<Args...>& args_; /// Additional arguments for the fitted function
    const T& y_orig_; ///> Original y values
    const T& y_err_orig_; ///> Original y error values
    const unsigned n_par_; ///> Number of parameters
};

template<typename VecT, typename T, typename F, typename GradF, typename... Args>
T fit_ceres(const VecT& y, const VecT& y_err, const VecT& p0, VecT& ps, VecT& p_errs, const VecT& lb, const VecT& ub, const T& tol, F f, GradF grad_f, const std::tuple<Args...>& args, bool use_grad=false){
    int n_par = p0.size();
    ps = p0;
    double *p = ps.data();

    ceres::Problem problem;
    problem.AddParameterBlock(p,n_par);
    for(int i=0; i<n_par; i++){
        problem.SetParameterLowerBound(p,i,lb[i]);
        problem.SetParameterUpperBound(p,i,ub[i]);
    }

    if(use_grad){
        CeresAnalyticDiffFunctor<VecT,VecT,std::vector<VecT>,Args...> *cost_function = new CeresAnalyticDiffFunctor<VecT,VecT,std::vector<VecT>,Args...>(y,y_err,n_par,f,grad_f,args);
        problem.AddResidualBlock(cost_function,NULL,p);
    }else{
        ceres::DynamicNumericDiffCostFunction<CeresNumericDiffFunctor<VecT,VecT,Args...>,ceres::CENTRAL> *cost_function =
            new ceres::DynamicNumericDiffCostFunction<CeresNumericDiffFunctor<VecT,VecT,Args...>,ceres::CENTRAL>(
                new CeresNumericDiffFunctor<VecT,VecT,Args...>(y,y_err,n_par,f,args));
        cost_function->AddParameterBlock(n_par);
        cost_function->SetNumResiduals(y.size());
        problem.AddResidualBlock(cost_function,NULL,p);
    }
    ceres::Solver::Options options;
    options.max_num_iterations = 10000;
    options.parameter_tolerance = tol;
    options.check_gradients = false;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";

    //ceres::Covariance::Options cov_options;
    //ceres::Covariance covariance(cov_options);

    //std::vector<std::pair<const double*, const double*> > covariance_blocks;
    //covariance_blocks.push_back(std::make_pair(p, p));

    //CHECK(covariance.Compute(covariance_blocks, &problem));

    //double covariance_pp[n_par * n_par];
    //covariance.GetCovarianceBlock(p, p, covariance_pp);

    p_errs.resize(n_par);
    for(int i=0; i<n_par; i++)
        p_errs[i] = 0.;//sqrt(covariance_pp[i*n_par + i]);
    return 0;
}

}

#endif
