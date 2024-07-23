# Project 2:  Machine Learning and Sports Analytics
## Presented by:  Krishan Fotedar and Cristina Frisby
### In Partial Fulfillment of The Requirements of AI Bootcamp

## About the Project
This project utilizes existing basketball and baseball statistics and historical
data to build predictive models.  The data exploration intends to provide insights
into what the critical components are in predicting the outcomes.

## Why Study Sports Data?
Areas of interest include the following: <br>
    1. Professional <br>
    2. Fan Interests <br>
    3. Global Impacts <br>

Professional sports is a multi-billion dollar global industry.  Coaching staff
that understand the use of analytics can make better decisions real time during
the course of a game, or adjust strategies that lean towards statistically
significant outcomes.  For fans using data analytics, the modelling may give an
advantage on fantasy leagues and gambling. 

## About the Code
The code for this project is broken down into two folders:

**frisby_mlb_analytics** <br><br>
This file includes the original data in the "baseball_resources" folder, and an
output file for the cleaned data in "baseball_out".
The main model code is in File "01 Baseball_Game_Scores_Model".  Run this model 
first, as it generates the clean data for later in that file and also for the
secondary model file.
"01 Baseball_Game_Scores_Model" is a set of Random Forest Models for all innnings 
of the data set.  This file can be run from "Run All" without any user input and 
will generate the final accuracy scores for each inning one through 8 in bar 
graph format.

A secondary set of models is in File "02_Baseball_Additional_Model_Types".  This
includes a selection of alternative modesl to Random Forest for comparative
purposes only.

Results:  Using the Random Forest Models, balanced accuracy scores for predicting
the winner were as high as 80% after 5 innings, and 95% after 8 innings.

**fotedar_nba_analytics**<br><br>
The files in the 'fotedar_nba_analytics' consists of the following: <br>
    1/ nba_project_season - this file studies playoff nba data <br>
    2/ nba_project_playoffs - this file studies in season nba data <br>
    3/ nba_project_kmeans and pca - this file performs kmeans and pca analysis on the data <br>
    4/ csv - contains the input data <br>
    5/ output - output files <br>

I used data I found on <a href="https://www.kaggle.com/datasets/wyattowalsh/basketball">Kaggle</a>.
The data covers every game from 1946-2023 - it includes teams, winners losers, points, shooting percentage, rebound margin etc.

For the purposes of this projects I only used games from 2004 as this is the year the NBA took its current form.
The programs convert some of the  statistics into advance metrics (true shooting percentage, assist to turnover ratio) to see how 
they determine winning margin and who wins. 

I removed pre-season and All-Star games and then split the data into two separate dataframes - one for playoffs and another for in-season games.

I used Linear Regression to determine what factors drive winning margin, Logistic Regression to see who wins (regardless of score)
as well as Random Forest Classifier. The project also does kmeans and pca analysis, but this was not needed as the solution was able to handle the full data set.
In any case, if we did use kmeans, the optimal kmeans number was 25.

The analysis shows that true-shooting percentage is the primary driver for wins and loses.
In the regular season, the shooting percentage of the home team is the primary driver. 
In the playoffs it is the opposite, the shooting percentage of the visiting team is the primary driver.
Assist to turnover and rebound margin play a role but a much lesser role than shooting percentage.





