# File: Simple_Linear_Regression_using_OLS.py

import time
import numpy as np
import matplotlib.pyplot as plt

# scikit learn
from sklearn import datasets
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore")


def mse(y_tst, y_predicted):
    """ Mean squared error regression loss """
    # MSE = 1/N ∑i=1:n (yi−(xi.w + b))^2
    _MSE = np.mean(np.power((y_tst - y_predicted), 2))

    return _MSE


def rmse(y_tst, y_predicted):
    """ Root Mean Squared Error """
    _mse = mse(y_tst, y_predicted)
    return np.sqrt(_mse)


def r2score(y_tst, y_predicted):
    """ (coefficient of determination) regression score function """
    # Sum of square of residuals
    RSS = np.sum((y_predicted - y_tst) ** 2)
    #  Total sum of squares
    TSS = np.sum((y_tst - np.mean(y_tst)) ** 2)
    # R2 score
    r2 = 1 - (RSS / TSS)

    return r2


class SimpleLinearRegression:
    """ Linear regression using Ordinary least squares """
    """
    it concerns two-dimensional sample points with one independent 
    variable and one dependent variable (conventionally, the x 
    and y coordinates in a Cartesian coordinate system) and finds 
    a linear function (a non-vertical straight line) that, as 
    accurately as possible, predicts the dependent variable values
    as a function of the independent variable. 
    The adjective simple refers to the fact that the outcome 
    variable is related to a single predictor.
    
    In statistics, ordinary least squares is a type of linear 
    least squares method for estimating the unknown parameters 
    in a linear regression model.
    linear model : y = α + βX  --> yi = α + β * xi.T
    """

    def __repr__(self):
        return "The best-fit line is {0:8.6f} + {1:8.6f} * x"\
            .format(self.alpha_hat, self.beta_hat)

    def fit(self, x_trn, y_trn):
        """ Fitting data """

        # Initialize the parameters
        self.alpha_hat = 0
        self.beta_hat = 0

        # Check input array sizes:
        if len(x_trn.shape) < 2:
            print("Reshaping features array.")
            x_trn = x_trn.reshape(x_trn.shape[0], 1)

        if len(y_trn.shape) < 2:
            print("Reshaping observations array.")
            y_trn = y_trn.reshape(y_trn.shape[0], 1)

        # Calculate the least squares estimates - OLS:

        # Calculate Means:
        y_bar = np.mean(y_trn)
        x_bar = np.mean(x_trn)

        """
            numerator =  ∑i=1:n (xi - x_bar)(yi - y_bar)
            denominator = ∑i=1:n (xi - x_bar)^2
            β̂ = numerator / denominator
        """
        # Build the numerator term:
        numerator = np.sum((x_trn - x_bar) * (y_trn - y_bar))
        # Build the denominator term:
        denominator = np.sum((x_trn - x_bar) ** 2)
        # Calculate the slope β
        self.alpha_hat = numerator / denominator

        # Calculate the intercept α̂ = y̅ - β̂x̅
        self.beta_hat = y_bar - self.alpha_hat * x_bar

        # return np.array([self.beta_hat, self.alpha_hat])

    def predict(self, x_tst):
        """ Prediction - Best Fit Line"""
        # Calculate y_hat =  yi = α + β * xi.T
        y_predicted = self.beta_hat + self.alpha_hat * x_tst.T

        return y_predicted


def draw_plot(xs, pl, x_trn, x_tst,
              y_trn, y_tst):
    """
    """
    cmap = plt.get_cmap('viridis')
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(x_trn,
                y_trn,
                color=cmap(0.8),
                s=20)
    plt.scatter(x_tst,
                y_tst,
                color=cmap(0.5),
                s=20)
    plt.plot(xs, pl.T,
             color='red',
             linewidth=1,
             ls='--',
             label="Prediction")

    plt.legend()
    plt.show()


def programmed_slr(Xs, X_trn, X_tst, y_trn, y_tst):
    """"""
    start = time.time()

    # Create a Linear Regression object
    reg = SimpleLinearRegression()
    # Fit the training model using gradient descent
    reg.fit(X_trn, y_trn)
    # Print the object information - to verify the minimum
    print(reg)
    # Predict labels using the test data and the training model parameters
    y_predictions = reg.predict(X_tst)

    # Measure the accuracy between the actual data and the predicted ones
    _mse = mse(y_tst, y_predictions)
    print("\n ----------\n MSE: {:.2f}".format(_mse))
    _rmse = rmse(y_tst, y_predictions)
    print("\n ----------\n RMSE: {:.2f}".format(_rmse))
    _r2_score = r2score(y_tst, y_predictions)
    print("\n ----------\n R^2 score: %.2f%%" % (_r2_score * 100))

    end = time.time()  # ----------------------------------------------
    print('\n ----------\n Execution Time: {%f}' \
          % ((end - start) / 1000) + ' seconds.')

    # Draw the regression
    predicted_line = reg.predict(Xs)
    draw_plot(Xs, predicted_line, X_trn, X_tst, y_trn, y_tst)


if __name__ == '__main__':
    """"""
    try:
        # Create regression data
        X, y = datasets.make_regression(n_samples=1000,
                                        n_features=1,
                                        noise=20,
                                        random_state=5)
        # Split the data to training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=5)

        programmed_slr(X, X_train, X_test, y_train, y_test)

    except:
        pass
