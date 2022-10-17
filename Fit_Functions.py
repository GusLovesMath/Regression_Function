import matplotlib.pyplot as pl
import numpy as np
from scipy.optimize import curve_fit


'''Fits Function based on a inputted function'''
def fit_function(
        xdata, ydata, func, n=3, yerr=None, xerr=None, p0=None,
        pc='k', dc='C3o', title='Function', ylab='f(x)', xlab='x',
        bounds=(-np.inf, np.inf), dps=5, elc='k', elw=1.5, cs=5,
        maxfew=800, param_return=False, cov_show=False,
        chi_show=False):

    # fitting data and computing error
    params, covariance_matrix = curve_fit(f=func, xdata=xdata, ydata=ydata,
                                          p0=p0, sigma=yerr, bounds=bounds,
                                          maxfev=maxfew)
    params_error = np.sqrt(np.diag(covariance_matrix))

    # returns parameters
    if param_return:
        return params, params_error

    # plots fitted function, with data points and error bars
    else:
        # Formatting fitted terms
        params_formated = ['%.1e' % param for param in params]
        params_error_formated = ["%.1e" % err for err in params_error]

        # plotting our fitted data
        x = np.arange(min(xdata), max(xdata) + 0.01, 0.01)
        pl.errorbar(xdata, ydata, yerr=yerr, xerr=xerr, label='Data Points',
                    ms=dps, fmt=dc, ecolor=elc, elinewidth=elw, capsize=cs)

        # 2 parameters being fit
        if n == 2:
            def fit_graph_terms(x, func, params):
                return func(x, params[0], params[1])

            pl.plot(x, fit_graph_terms(x, func, params), pc,
                label=f'Fitted Graph:\
                       \nA = {params_formated[0]} ± {params_error_formated[0]}\
                       \nB = {params_formated[1]} ± {params_error_formated[1]}')

        # 3 parameters being fit
        elif n == 3:
            def fit_graph_terms(x, func, params):
                return func(x, params[0], params[1], params[2])

            pl.plot(x, fit_graph_terms(x, func, params), pc,
                label=f'Fitted:\
                       \nA = {params_formated[0]} ± {params_error_formated[0]}\
                       \nB = {params_formated[1]} ± {params_error_formated[1]}\
                       \nC = {params_formated[2]} ± {params_error_formated[2]}')

        # 4 parameters being fit
        elif n == 4:
            def fit_graph_terms(x, func, params):
                return func(x, params[0], params[1], params[2], params[3])

            pl.plot(x, fit_graph_terms(x, func, params), pc,
                label=f'Fitted:\
                       \nA = {params_formated[0]} ± {params_error_formated[0]}\
                       \nB = {params_formated[1]} ± {params_error_formated[1]}\
                       \nC = {params_formated[2]} ± {params_error_formated[2]}\
                       \nD = {params_formated[3]} ± {params_error_formated[3]}')

        else:
            print('Input not supported; Parameters allowed 2 to 4.')

        # Plotting
        pl.title(title)
        pl.ylabel(ylab)
        pl.xlabel(xlab)
        pl.legend(bbox_to_anchor=(1.04,1), loc='upper left')
        pl.grid(alpha=0.3)
        pl.show()

        # Displaying fitted function and Parameters plus Uncertainties
        from IPython.display import display, Latex
        display(Latex(title))
        n_list = ['A', 'B', 'C', 'D']
        for i in range(n):
            display(
                Latex(f'Parameter {n_list[i]} = {params[i]} $\pm$ {params_error[i]}'))

        # Chi Calculation
        if chi_show:

            # Chi Squared chi^2
            yfit = fit_graph_terms(xdata, func, params)

            if yerr is not None:
                chi2 = sum(((ydata - yfit) / yerr) ** 2)
            else:
                chi2 = sum((ydata - yfit) ** 2)

            # Reduced Chi Squared
            dof = len(xdata) - len(params)     # degrees of Freedom
            red_chi2 = chi2 / dof              # generally

            display(Latex(f'$\chi^2$:  Chi-Squared: {chi2}'))
            display(Latex(f'$\chi_R^2$: Reduced Chi-Squared: {red_chi2}'))

        # Display Covarience Matrix
        if cov_show:
            from sympy import Matrix
            print('\nCovariance Matrix:')
            display(Matrix(covariance_matrix))



'''Fits Function based on a general polynomial degree'''
def poly_fit(
        xdata, ydata, n=1, pc='k--', dc='bo', title=None,
        ylab='f(x)', xlab='x', yerr=None, xerr=None, dps=5,
        elc='k', elw=1.5, cs=5, param_return=False,
        p_func=False, p_cov=False):

    params, covariance_matrix = np.polyfit(xdata, ydata, n, cov=True)   # x-data, y-data, polynomial degree
    params_error = np.sqrt(np.diag(covariance_matrix))                  # error is given by the diagnols of √cov 

    # returns parameters
    if param_return:
        return params, params_error

    # plots polynomial function, with data points and error bars
    else:
        # Linear function
        print_func = np.poly1d(params)

        x = np.arange(min(xdata), max(xdata) + 0.01, 0.01)
        y_func = np.polyval(params, x)                                  # creats usable function from expression

        if p_func:
            print(f'Poly Fit: {print_func}')

        # Plotting
        #pl.plot(xdata, ydata, dc, label='Data Points')                # Plots Data points
        pl.errorbar(xdata, ydata, yerr, xerr, ms=dps, fmt=dc, ecolor=elc,
                    elinewidth=elw, capsize=cs, label='Data Points')
        pl.plot(x, y_func, pc, label='Fitted Function')                # Plots fitted function

        if title is None:
            pl.title(rf'f(x) = {print_func}')

        else:
            pl.title(title)
        pl.ylabel(ylab)
        pl.xlabel(xlab)
        pl.grid(alpha=0.3)
        pl.legend()
        pl.show()

        for i, p in enumerate(params):
            print(f'Parameter {i} = {p} +/- {params_error[i]}')

        if p_cov:
            if n >= 4:
                from IPython.display import display, Latex
                from sympy import Matrix
                print('\nCovariance Matrix:')
                display(Matrix(covariance_matrix))

            else:
                print('\nCovariance Matrix: \n', covariance_matrix)
