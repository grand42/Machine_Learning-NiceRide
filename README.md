# Machine Learning Nice Ride Bike Sharing

## Background: 
Our group used machine learning to predict the number of bikes that are rented each day based on a number of inputs. Bike-Sharing has become increasingly popular in major cities. Here in the twin cities, we have the bike-sharing nonprofit Nice Ride MN. We want to analyze these data to draw some insights and help the Nice Ride to make better business decisions

* A linear regression model that predict the daily bike transaction based on daily weather inputs. 
* A logistic regression model to classify whether daily bike use is above or below the average for the year. 
* Analyzed the station demand to determine if adequate docks are provided at each station, does it has a seasonal demand change, and etc. 


## Data Source
Climate data comes from National Oceanic and Atmospheric Administration,  OpenWeatherMap, Kaggle, and also open data shared by the company Nice Ride MN. 

## Tools and Languages

- Python
    * Scikit-learn
    * Pandas
    * Matplotlib
    * Sqlalchemy
    * Statsmodels

- HTML
- CSS
- Tableau

## Work Distribution
* Database: Zean Zhang

* HTML Set-Up: Gretchen Anderson

* Classification model: Gretchen Anderson

* Linear Regression model: Zean Zhang

* Station Demand Analysis: Saif Ahmed

* Visualizations: Richard Lee

## Analysis

#### Setup Dependencies and Connect to Database

      # Python SQL toolkit and Object Relational Mapper
      import numpy as np
      import pandas as pd
      import sqlalchemy
      from sqlalchemy.ext.automap import automap_base
      from sqlalchemy.orm import Session
      from sqlalchemy import create_engine, func,inspect
      from config import pgkey
      from sklearn.neighbors import KNeighborsClassifier
      import os
      
      # Connect to PostgreSQL database
      con_string = f"sqlite:///../Data/Final_Project_DB.db"
      engine = create_engine(con_string)
      connection = engine.connect()
      
      # reflect an existing database into a new model
      Base = automap_base()
      # reflect the tables
      Base.prepare(engine, reflect = True
      
      # Save references to each table
      station_df = Base.classes.station_location
      weather_df = Base.classes.weather_data
      bike_df = Base.classes.bike_transaction
      humidity_df = Base.classes.humidity
      wind_speed_df = Base.classes.wind_speed
      pressure_df = Base.classes.pressure
      
      # Create our session (link) from Python to the DB
      session = Session(engine)
      # Create the inspector and connect it to the engine
      inspector = inspect(engine)
      
#### Query data and transfrom into pandas dataframes

      # Query all bike transaction data 
      bike_info = session.query(
          bike_df.id,
          bike_df.start_date,
          bike_df.start_time,
          bike_df.start_station_name,
          bike_df.start_station_code,
          bike_df.end_date,
          bike_df.end_time,
          bike_df.end_station_name,
          bike_df.end_station_code,
          bike_df.user_type,
          bike_df.total_duration
      ).all()

      session.close()

      # Convert query results to DataFrame
      bikes_df = pd.DataFrame(bike_info)
      bikes_df['start_date'] = pd.to_datetime(bikes_df['start_date'])
      bikes_df['end_date'] = pd.to_datetime(bikes_df['end_date'])
      
#### Clean and Transform the data

For each of the models, calculations of the total daily rides, avg daily rides, and station demand was calculated and added to the data frames.  The date columns were converted into Datetime format using pd.to_datetime

#### Logistic Regression Model

Using SKLearn, a logistic model was created to predict the demand for bike use based on the days weather.  The data was categorized as either above or below average demand.  Then using train_test_split, the data was split into train and test data to create the model.

      from sklearn.model_selection import train_test_split

      X_train, X_test, y_train, y_test = train_test_split(Bikes_test_data, target, random_state=0)
      
      # Fit the model and calculate scores
      classifier.fit(X_train, y_train)
      
      print(f"Training Data Score: {classifier.score(X_train, y_train)}")
      print(f"Testing Data Score: {classifier.score(X_test, y_test)}")
      
      
