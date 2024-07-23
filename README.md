# Project 2:  Machine Learning and Sports Analytics
## Presented by:  Krishan Fotedar and Cristina Frisby
### In Partial Fulfillment of The Requirements of AI Bootcamp

## About the Project
This project utilizes existing basketball and baseball statistics and historical
data to build predictive models.  The data exploration intends to provide insights
into what the critical components are in predicting the outcomes.

## Why Study Sports Data?
Areas of interest include the following:
    1. Professional
    2. Fan Interests
    3. Global Impacts

Professional sports is a multi-billion dollar global industry.  Coaching staff
that understand the use of analytics can make better decisions real time during
the course of a game, or adjust strategies that lean towards statistically
significant outcomes.  For fans using data analytics, the modelling may give an
advantage on fantasy leagues and gambling. 

## About the Code
The code for this project is broken down into two folders:

**frisby_mlb_analytics**
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


**fotedar_nba_analytics**
Enter Descriptoin here



