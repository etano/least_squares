#ifndef LEAST_SQUARES_HPP
#define LEAST_SQUARES_HPP

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
U convert_vec(const std::vector<T>& x){
    U m(x.size());
    for(size_t i=0; i<x.size(); i++)
        m[i] = x[i];
    return std::move(m);
}

template<typename T, typename U, typename... Args>
class LeastSquares{
public:
    /// Constructor
    LeastSquares(const T& y_orig, T (*f)(const U&,Args...)&, const std::tuple<Args...>& args)
        : f_(f), args_(args), y_orig_(y_orig)
    {}

    /// Compute the residual
    double get_residual(const std::vector<double>& x){
        T y_new(invoke(f_,std::tuple_cat(std::tuple<U>(convert_vec<double,U>(x)),args_)));
        double tot(0);
        for(size_t i=0; i<y_orig_.size(); i++)
            tot += pow(y_new.at(i)-y_orig_.at(i),2);
        return tot;
    }
private:
    T (*f_)(const U&,Args...); ///> The function which to to be fitted
    const std::tuple<Args...>& args_; /// Additional arguments
    const T& y_orig_; ///> Original y values
};

template<typename T>
double residual(const std::vector<double>& x, std::vector<double>& grad, void* f_data){
    T* lsq = static_cast<T*>(f_data);
    return lsq->get_residual(x);
}

}

#endif
