import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import locale
from datetime import date

class ProcessUNData:
    un_data = pd.read_csv(r'./UNdata_Export_20230308_220221493.csv', low_memory=False)

    clean_dict = {'Country':
                      {'Russian Federation': 'Russia',
                       'United States of America': 'United States'},
                  'City':
                      {'Firenze': 'florence',
                       'St. Petersburg': 'Saint Petersburg'}
                  }

    def __init__(self):
        self.cleaned_un_df = self.clean_un_data()
        self.cleaned_un_df.to_csv("./cleaned_data/UN_data.csv")

    def clean_un_data(self):
        self.un_data['numeric_year'] = self.un_data['Year'].apply(lambda x: int(x) if str(x).isnumeric() else 0)
        self.un_data['City'] = self.un_data['City'].str.split(' \(', 1, expand=True)[0]
        self.un_data['City'] = self.un_data['City'].apply(
            lambda x: self.clean_dict['City'][x] if x in self.clean_dict['City'].keys() else x)
        self.un_data['Country or Area'] = self.un_data['Country or Area'].apply(
            lambda x: self.clean_dict['Country'][x] if x in self.clean_dict['Country'].keys() else x)
        # only keep final data, city type of city proper and data on both sexes and get data for the most recent
        # year per city and country or area
        one_record = self.un_data[
            (self.un_data['Reliability'] == 'Final figure, complete') &
            (self.un_data['City type'] == 'City proper') & (
                    self.un_data['Sex'].astype('str') == 'Both Sexes')].sort_values(
            by=['City', 'Country or Area', 'numeric_year'], ascending=False).groupby(
            ['City', 'Country or Area']).first().reset_index()
        one_record['city'] = one_record['City'].str.lower()
        one_record['country_or_area'] = one_record['Country or Area'].str.lower()
        one_record['population'] = one_record['Value'].astype('int64')

        return one_record[['city', 'country_or_area', 'numeric_year', 'population']]

class ProcessMuseumData:

    url = "https://en.wikipedia.org/wiki/List_of_most-visited_museums"

    def __init__(self):
        today = date.today()
        d1 = today.strftime("%d_%m_%Y")

        self.museum_df = self.clean_museum_data(pd.read_html(self.url)[0])
        self.museum_df.to_csv("./cleaned_data/wikipedia_most_visited_"+d1+"_cleaned.csv")

    def get_city_country(self, pair):
        list_pair = pair.split(", ")
        return [list_pair[0], list_pair[-1]]

    def clean_museum_data(self, museum_df):
        locale.setlocale(locale.LC_ALL, '')

        museum_df[['city', 'country']] = museum_df['Location'].apply(lambda x: self.get_city_country(x)).to_list()
        museum_df['visitor_cleaned'] = museum_df['Number of visitors'].str.split('[', 1, expand=True)[0].apply(
            lambda x: locale.atof(x.replace(',','')))

        museum_df['city'] = museum_df['city'].str.lower()
        museum_df['country'] = museum_df['country'].str.lower()

        return museum_df

class PreformLinearRegression:

    today = date.today()
    d1 = today.strftime("%d_%m_%Y")
    museum_path = "./cleaned_data/wikipedia_most_visited_"+d1+"_cleaned.csv"
    city_population_path = "./cleaned_data/UN_data.csv"

    def __init__(self):

        museum_df = pd.read_csv(self.museum_path)
        city_population = pd.read_csv(self.city_population_path)

        data_to_use = museum_df.merge(city_population, left_on=['city', 'country'],
                                      right_on=['city', 'country_or_area'])[
            ['Name', 'Location', 'visitor_cleaned', 'population']]

        self.linear_regression(data_to_use)
        self.make_linear_regression_visual()
        self.get_linear_regression_results()

    def linear_regression(self, data_to_use):
        length = data_to_use.shape[0]
        x = data_to_use['visitor_cleaned'].values
        y = data_to_use['population'].values
        self.x = x.reshape(length, 1)
        self.y = y.reshape(length, 1)

        self.regr = linear_model.LinearRegression()
        self.regr.fit(self.x, self.y)

    def make_linear_regression_visual(self):
        plt.scatter(self.x, self.y, color='black')
        plt.plot(self.x, self.regr.predict(self.x), color='blue', linewidth=3)
        plt.xticks(())
        plt.yticks(())

        plt.xlabel('museum visitors')
        plt.ylabel('city population')
        plt.title('Linear Regression of Museum Visitors vs City Population')

        plt.savefig("./artifacts/scatter_plot.jpg")

    def get_linear_regression_results(self):

        metric = ["mean absolute error","coefficient","intercept"]
        result = [str(mean_absolute_error(self.y, self.regr.predict(self.x))),
                  str(self.regr.coef_[0][0]),
                  str(self.regr.intercept_[0])
                  ]
        result_dict = {"metric":metric,"result":result}
        df = pd.DataFrame(result_dict)
        print(df)
        # saving the dataframe
        df.to_csv('./artifacts/regression_results.csv')


if __name__ == "__main__":
    ProcessUNData()
    ProcessMuseumData()
    PreformLinearRegression()





