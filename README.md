# nyc_realestate_proj

<!--
*** Thanks for checking out this README Template. If you have a suggestion that would
*** make this better, please fork the repo and create a pull request or simply open
*** an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
-->





<!-- PROJECT SHIELDS -->





  <h3 align="center">NYC Real Estate Property Sale Project</h3>

  <p align="center">
    Predicting Property Prices using Sklearn and Tensorflow
    <br />
    <a href="https://github.com/k-falk/ds_salary_proj"
    ><strong>Explore the docs »</strong></a>
    <br />
    <br />
    ·
    <a href="https://github.com/k-falk/ds_salary_proj/issues">Report Bug</a>
    ·
    <a href="https://github.com/k-falk/ds_salary_proj/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Data Cleaning and Database Creation](#data-cleaning)
* [EDA](#eda)
* [Model Building](#model-building)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)



<!-- ABOUT THE PROJECT -->
## About The Project

This is a data science project designed to aid in the prediction of the sale prices of properties in NYC from the dates 10/15/19 - 10/14/20. We created a MySQL database using SQLAlchemy and created various visualizations in Tableau to help visualize various factors in our data. Then we used Python to further explore our data and build a model using Tensorflow and Sklearn. 

### Built With
This project uses SQL to store our data. Tableau to visualize the data and create a dashboard to present to stakeholders. And Python to build a model to predict the data. We then use Flask to productionize our model and enable predictions using user inputted data. 
* [SQL Alchemy](https://www.sqlalchemy.org/)
* [Tableau](https://www.tableau.com/)
* [Pandas](https://pandas.pydata.org/)
* [Sklearn](https://scikit-learn.org/stable/)
* [Tensorflow](https://scikit-learn.org/stable/)
* [Flask](https://flask.palletsprojects.com/en/1.1.x/)



<!-- Input-->
## Input
The input for our data was acquired using NYC's open data project dataset. It can be found here. It takes the rolling data from the past year of property sales. It came with the following notable variables: 

 - Sale Price
 - Address
 - Borough
 - Building Category
 - Neighborhood
 - Tax Class
 - Residential Units, Commercial Units, Total Units
 - Square Feet
 - Year Built

<!-- Data Cleaning-->
## Data Cleaning
There were a few transformations we had to do with our data. We removed some unusable variables such as Apartment number or Easement. In the future, some of these variables may be able to be further cleaned to give us more features for our model. We also transformed some of our logarithmic variables such as sale price to make them more normally distributed which will be better for our model

<!-- Feature Creation-->
## Feature Creation
We were able to glean a couple of more features from our data. Notably, we used Python's cartopy library to get the Latitude and Longitude of our addresses. This allowed us to map our data to the distance of a landmark in NYC as well as visualize property sales in Tableau. We found the distance from each address to Central Park. I chose that as it is in the center of Manhattan and had strong correlation with sale price. We also found the date between the sale and the introduction of Coronavirus into NYC but this had very little affect on property price. 

<!-- EDA-->
## EDA

We did some exploratory analysis on our data using Tableau and Python. Tableau allowed us to create an easy to read dashboard that we could present to stakeholders while Python allowed us to look at key variables that would affect our model. 

Here are some highlights:

![Salary By Position Boxplot](https://github.com/k-falk/ds_salary_proj/blob/master/salary_title_boxplot.PNG)

![Corr Map](https://github.com/k-falk/ds_salary_proj/blob/master/corr_map.PNG)

![Statemap](https://github.com/k-falk/ds_salary_proj/blob/master/statemap.PNG)

![Salary by Title Pivot](https://github.com/k-falk/ds_salary_proj/blob/master/title_salary_pivot.PNG)

![Title Barchart](https://github.com/k-falk/ds_salary_proj/blob/master/title_barchart.PNG)
<!-- Model building -->
## Model building

For our model building, we used sklearn and its cross_val_score function to evaluate each model. We tried the following models and picked the best one: 
* Linear Regression
* Lasso Regression
* Random Forest
* Elastic Net
* K Neighbors
* Decision Tree
* Gradient Boosting

The best model here was Gradient Boosting with an MAE of .249. We then used RandomizedGridSearch to find the best model. This gave us a final MAE of .245. 

We also tested our data on a deep learning model. We used Tensorflow keras to build our model. This gave us better results and we got an MAE of 0.232. Transforming the MAE into real numbers gives us an MAE of 140k. The deeplearning model's results can be visualized here:
<!-- New Data-->
## Testing on new data
Our model's results were uninspiring to say the least. Although our residuals showed our errors were normally distributed and our MAE had a small standard deviation, we still hoped for better results. We hypothesized that this was because of our data input. So we went ahead and tested our model building on new data. 

There is a dataset on Kaggle that has the same NYC Property data but from previous years. We found that using the same techniques we used for the current year, we had much better results. Our findings can be summarized here

WIP


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Kevin Falk - [LinkedIn](in-k-falk) - [Github](github.com/k-falk) 
kevin.falk.631@gmail.com

Project Link: [https://github.com/k-falk/nyc_realestate_proj](https://github.com/k-falk/nyc_realestate_proj)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [* [Stack Overflow](https://stackoverflow.com/) (A lot of stackoverflow was used on this project :) )
* 




