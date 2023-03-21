import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import locale

class PreformLinearRegression:

    museum_path = "./cleaned_data/wikipedia_most_visited_cleaned.csv"
    city_population_path = "./cleaned_data/UN_data.csv"

    def __init__(self):
        museum_df = pd.read_csv(self.museum_path)
        city_population = pd.read_csv(self.city_population_path)

        data_to_use = museum_df.merge(city_population, left_on=['city', 'country'],
                                      right_on=['city', 'country_or_area'])[
            ['Name', 'Location', 'visitor_cleaned', 'population']]

        x,y,self.regr = self.linear_regression(data_to_use)
        self.make_linear_regression_visual(x,y)
        self.get_linear_regression_results(x,y)

    def linear_regression(self, data_to_use):
        """
        :param data_to_use: the data set
        :return regr: linear regression model fit to x & y
        """
        length = data_to_use.shape[0]
        x_temp = data_to_use['visitor_cleaned'].values
        y_temp = data_to_use['population'].values
        x = x_temp.reshape(length, 1)
        y = y_temp.reshape(length, 1)

        regr = linear_model.LinearRegression()
        return x,y,regr.fit(x, y)

    def make_linear_regression_visual(self,x,y):
        """
        make scatter plot of museum and city population data and save it as a jpg
        :param x: visitor cleaned data
        :param y: population data
        :return: nothing
        """
        plt.scatter(x, y, color='black')
        plt.plot(x, self.regr.predict(x), color='blue', linewidth=3)
        plt.xticks(())
        plt.yticks(())

        plt.xlabel('museum visitors')
        plt.ylabel('city population')
        plt.title('Linear Regression of Museum Visitors vs City Population')

        plt.savefig("./artifacts/scatter_plot.jpg")

    def get_linear_regression_results(self,x,y):
        """
        make regression results csv and save it as a csv
        :param x: visitor cleaned data
        :param y: population data
        :return:
        """
        metric = ["mean absolute error", "coefficient", "intercept"]
        result = [str(mean_absolute_error(y, self.regr.predict(x))),
                  str(self.regr.coef_[0][0]),
                  str(self.regr.intercept_[0])
                  ]
        result_dict = {"metric": metric, "result": result}
        df = pd.DataFrame(result_dict)
        print(df)
        # saving the dataframe
        df.to_csv('./artifacts/regression_results.csv')

