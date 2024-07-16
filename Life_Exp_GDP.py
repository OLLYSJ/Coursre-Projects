import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

life_exp_gdp_df = pd.read_csv("/Users/olly/Downloads/Life-Expectancy-and-GDP-Starter2/Life Expectancy and GDP Data.csv")
life_exp_gdp_df.columns = ['Country', 'Year', 'Life_exp', 'GDP']
life_exp_gdp_df.GDP = life_exp_gdp_df.GDP.astype(int)
print(life_exp_gdp_df)

print(life_exp_gdp_df.groupby('Country').GDP.describe())

# figure 1 contains 2 plots representing data pertaining to the rise of life expectancy and gdp separately
plt.figure(figsize=(10, 9))
plt.subplot(1,2,1)
sns.scatterplot(life_exp_gdp_df, x='Year', y='GDP', hue='Country', palette='bright')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.title('GDP against Year')
plt.subplot(1,2,2)
sns.scatterplot(life_exp_gdp_df, x='Year', y='Life_exp', hue='Country', palette='bright')
plt.xlabel('Year')
plt.ylabel('Life Expectancy at birth (years)')
plt.title('Life Expectancy against Year')
# plt.savefig('/Users/olly/Documents/Figure_One.png')
# plt.show()
plt.clf()

# countries gdp's and life expectancies min and max to show growth

Germany_df = life_exp_gdp_df[life_exp_gdp_df.Country == 'Germany']
Germany_df = Germany_df.drop(columns=['Country'])
Germany_df.columns = ['Year', 'Life_exp', 'GDP']
Germany_gdp_growth = (np.max(Germany_df.GDP) - np.min(Germany_df.GDP))/15
Germany_life_exp_growth = (np.max(Germany_df.Life_exp) - np.min(Germany_df.Life_exp))/15
# print(Germany_df)
# print(Germany_gdp_growth)
# print(Germany_life_exp_growth)

Chile_df = life_exp_gdp_df[life_exp_gdp_df.Country == 'Chile']
Chile_df = Chile_df.drop(columns=['Country'])
Chile_df.columns = ['Year', 'Life_exp', 'GDP']
Chile_gdp_growth = (np.max(Chile_df.GDP) - np.min(Chile_df.GDP))/15
Chile_life_exp_growth = (np.max(Chile_df.Life_exp) - np.min(Chile_df.Life_exp))/15
# print(Chile_df)
# print(Chile_gdp_growth)
# print(Chile_life_exp_growth)

China_df = life_exp_gdp_df[life_exp_gdp_df.Country == 'China']
China_df = China_df.drop(columns=['Country'])
China_df.columns = ['Year', 'Life_exp', 'GDP']
China_gdp_growth = (np.max(China_df.GDP) - np.min(China_df.GDP))/15
China_life_exp_growth = (np.max(China_df.Life_exp) - np.min(China_df.Life_exp))/15
# print(China_df)
# print(China_gdp_growth)
# print(China_life_exp_growth)

Mexico_df = life_exp_gdp_df[life_exp_gdp_df.Country == 'Mexico']
Mexico_df = Mexico_df.drop(columns=['Country'])
Mexico_df.columns = ['Year', 'Life_exp', 'GDP']
Mexico_gdp_growth = (np.max(Mexico_df.GDP) - np.min(Mexico_df.GDP))/15
Mexico_life_exp_growth = (np.max(Mexico_df.Life_exp) - np.min(Mexico_df.Life_exp))/15
# print(Mexico_df)
# print(Mexico_gdp_growth)
# print(Mexico_life_exp_growth)

US_df = life_exp_gdp_df[life_exp_gdp_df.Country == 'United States']
US_df = US_df.drop(columns=['Country'])
US_df.columns = ['Year', 'Life_exp', 'GDP']
US_gdp_growth = (np.max(US_df.GDP) - np.min(US_df.GDP))/15
US_life_exp_growth = (np.max(US_df.Life_exp) - np.min(US_df.Life_exp))/15
# print(US_df)
# print(US_gdp_growth)
# print(US_life_exp_growth)

Zimbabwe_df = life_exp_gdp_df[life_exp_gdp_df.Country == 'Zimbabwe']
Zimbabwe_df = Zimbabwe_df.drop(columns=['Country'])
Zimbabwe_df.columns = ['Year', 'Life_exp', 'GDP']
Zimbabwe_gdp_growth = (np.max(Zimbabwe_df.GDP) - np.min(Zimbabwe_df.GDP))/15
Zimbabwe_life_exp_growth = (np.max(Zimbabwe_df.Life_exp) - np.min(Zimbabwe_df.Life_exp))/15
print(Zimbabwe_df)
# print(Zimbabwe_gdp_growth)
# print(Zimbabwe_life_exp_growth)

gdp_avg_growth = [Germany_gdp_growth, Chile_gdp_growth, China_gdp_growth, Mexico_gdp_growth, US_gdp_growth, Zimbabwe_gdp_growth]
life_exp_avg_growth = [Germany_life_exp_growth, Chile_life_exp_growth, China_life_exp_growth, Mexico_life_exp_growth, US_life_exp_growth, Zimbabwe_life_exp_growth]
countries = ['Germany', 'Chile', 'China', 'Mexico', 'United States', 'Zimbabwe']

plt.figure(figsize=(10, 9))
sns.scatterplot(x=life_exp_avg_growth, y=gdp_avg_growth, hue=countries, palette='bright')
plt.xlabel('Life Expectancy avg growth per year')
plt.ylabel('GDP avg growth per year')
plt.legend(['Germany', 'Chile', 'China', 'Mexico', 'United States', 'Zimbabwe'], loc=9)
# plt.savefig('/Users/olly/Documents/Figure_Two.png')
# plt.show()
plt.clf()

plt.figure(figsize=(10, 9))
plt.subplot(1,2,1)
plt.plot(Mexico_df.Year, Mexico_df.GDP)
plt.xlabel('Year')
plt.ylabel('GDP')
plt.title('GDP against Year')
plt.subplot(1, 2, 2)
plt.plot(Mexico_df.Year, Mexico_df.Life_exp)
plt.xlabel('Year')
plt.ylabel('Life expectancy at birth(years)')
plt.title('Life expectancy against Year')
plt.savefig('/Users/olly/Documents/Mexico_figure.png')
# plt.show()
plt.clf()

plt.figure(figsize=(10, 9))
plt.subplot(1,2,1)
plt.plot(Zimbabwe_df.Year, Zimbabwe_df.GDP)
plt.xlabel('Year')
plt.ylabel('GDP')
plt.title('GDP against Year')
plt.subplot(1, 2, 2)
plt.plot(Zimbabwe_df.Year, Zimbabwe_df.Life_exp)
plt.xlabel('Year')
plt.ylabel('Life expectancy at birth(years)')
plt.title('Life expectancy against Year')
plt.savefig('/Users/olly/Documents/Zimbabwe_figure.png')
# plt.show()
plt.clf()

# probability a rise in gdp leads to a rise in life expectancy

print("General Data")
covariance_matrix = np.cov(life_exp_gdp_df.GDP, life_exp_gdp_df.Life_exp)
print(covariance_matrix)
corr_assignment, p = pearsonr(life_exp_gdp_df.GDP, life_exp_gdp_df.Life_exp)
print(p)

print("Mexico")
corr_assignment, p = pearsonr(Mexico_df.GDP, Mexico_df.Life_exp)
print(p)

print("Zimbabwe")
corr_assignment, p = pearsonr(Zimbabwe_df.GDP, Zimbabwe_df.Life_exp)
print(p)
