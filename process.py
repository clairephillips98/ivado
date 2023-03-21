import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import locale

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
        """
        clean the un data set
        :return: dataframe of important metrics
        """
        self.un_data['numeric_year'] = self.un_data['Year'].apply(lambda x: int(x) if str(x).isnumeric() else 0)
        self.un_data['City'] = self.un_data['City'].str.split(' \(', 1, expand=True)[0]
        self.un_data['City'] = self.un_data['City'].apply(
            lambda x: self.clean_dict['City'][x] if x in self.clean_dict['City'].keys() else x)
        self.un_data['Country or Area'] = self.un_data['Country or Area'].apply(
            lambda x: self.clean_dict['Country'][x] if x in self.clean_dict['Country'].keys() else x)
        # only keep final data, city type of city proper and data on both sexes and get data for the most recent
        # year per city and country or area
        un_data_cleaned = self.un_data[
            (self.un_data['Reliability'] == 'Final figure, complete') &
            (self.un_data['City type'] == 'City proper') & (
                    self.un_data['Sex'].astype('str') == 'Both Sexes')].sort_values(
            by=['City', 'Country or Area', 'numeric_year'], ascending=False).groupby(
            ['City', 'Country or Area']).first().reset_index()
        un_data_cleaned['city'] = un_data_cleaned['City'].str.lower()
        un_data_cleaned['country_or_area'] = un_data_cleaned['Country or Area'].str.lower()
        un_data_cleaned['population'] = un_data_cleaned['Value'].astype('int64')

        return un_data_cleaned[['city', 'country_or_area', 'numeric_year', 'population']]


class ProcessMuseumData:
    url = "https://en.wikipedia.org/wiki/List_of_most-visited_museums"

    def __init__(self):
        self.museum_df = self.clean_museum_data(pd.read_html(self.url)[0])
        self.museum_df.to_csv("./cleaned_data/wikipedia_most_visited_cleaned.csv")

    def get_city_country(self, pair):
        """
        :param pair: city country string seperated by common and sometimes additional data
        :return: list of city than country
        """
        list_pair = pair.split(", ")
        return [list_pair[0], list_pair[-1]]

    def clean_museum_data(self, museum_df):
        """
        :param museum_df: museum dataframe from wikipedia
        :return: cleaned museum dataframe
        """
        locale.setlocale(locale.LC_ALL, '')

        museum_df[['city', 'country']] = museum_df['Location'].apply(lambda x: self.get_city_country(x)).to_list()
        museum_df['visitor_cleaned'] = museum_df['Number of visitors'].str.split('[', 1, expand=True)[0].apply(
            lambda x: locale.atof(x.replace(',', '')))

        museum_df['city'] = museum_df['city'].str.lower()
        museum_df['country'] = museum_df['country'].str.lower()

        return museum_df

