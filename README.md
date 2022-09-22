<img src="http://imgur.com/1ZcRyrc.png" style="float: left; margin: 20px; height: 55px">

# DSI - Project 2 : Linear Regression Model on Ames Housing Dataset to Predict Sale Prices

### Contents:
- [Problem Statement](#Problem-Statement)
- [Repository Structure](#Repository-Structure)
- [Scenario Analysis](#Scenario-Analysis)
- [Conclusions and Recommendations](#Conclusions-and-Recommendations)


### Problem Statement 

Regardless of whether one chooses to buy or sell a home, knowing how much that home is worth is essential before making such a significant financial choice. That is the concern that Ames, Iowa homebuyers and sellers have. If we have familiar with the features of a home in Ames, how can we estimate its price? To answer that question, the price of a house at the sale will be predicted using a linear regression model built using the Ames Housing Dataset.


### Repository Structure 

```
dsi_project_ames
|
|__ code
|   |__ 01_data_cleaning.ipynb
|   |     - EDA to understand the dataset
|   |     - Cleaning of missing values using different  
|   |       methods
|   |__ 02_data_preprocessing.ipynb 
|   |     - Grouping of columns
|   |     - Removing redundant columns
|   |     - Cleaning up categorical values  
|   |__ 02_data_preprocessing_discrete_to_category.ipynb
|   |     - Converting discrete columns to categorical      
|   |       instead of numerical
|   |     - Everything else is the same as above
|   |__ 03_model_benchmarking.ipynb 
|   |     - Creating a benchmarking model 
|   |__ 04_model_comparison.ipynb
|   |     - Iterating over many combinations of models   
|   |       to find the optimal model
|   |__ 05_kaggle_competition.ipynb
|   |     - File to generate kaggle competition format   
|   |       of prediction csv to submit on kaggle
|   |__ 06_production_model_evaluation.ipynb
|           - This display performance of selection model   
|           for production as well as gives an         
|           overview of why this model was selected.        
|                         
|__ datasets
|   |__ train.csv
|   |__ test.csv
|   |__ train_fill.csv
|   |__ test_fill.csv
|   |__ train_cleaned.csv
|   |__ test_cleaned.csv
|   |__ test_cleaned_discrete.csv
|   |__ submission.csv
|
|__ figure
|   |__ 2nd_floor.png
|   |__ basement_sf.png
|   |__ bath.png
|   |__ ...
|
|__ model
|   |__ all_num_6_quality_cat.sav
|   |__ all_num_6qual_neighbour.sav
|   |__ all_num_10_cat.sav
|   |__ all_num_many_cat.sav
|
|__ README.md
|
|__ data_dictionary.md
```

### Scenario Analysis

The first **01_data_cleaning.ipynb** notebook begins by importing and cleaning a training data set 2197 homes with 82 different features. Once cleaned, we examine many more columns that have missing values. 
 
Explore the datasets to understand what each columns mean as many terms are american housing related terms. This process involves researching and understanding what feature of american house contributes to the price and other technical terms e.g. sale type and sale condition.

Once I have a clear understanding of the dataset, I begin to explore the dataset with EDA in order to understand the dataset further and ultimately deal with missing values. Missing values from this dataset was dealt with in 4 ways (dropping, impute mode, impute 0, and impute 'None')

Once there are no missing values, the relationship between columns was then investigated in order to combine and drop redundant columns. Leaving columns that are highly related (i.e. dependent columns) will confuse the model which usually results in suboptimal coefficients and model performance.

Cleaning of categorical columns involves exploring the mean of each groupby of column results, and combining the values of columns with uneven categorical value. This method allows us to simplify our categorical column and model to create flexibility and ability to take in new information.

Feature engineering was then carried out as our feature correlates better with natural log of sale price. Comparison of models during model iteration will support this decision.

With the data ready, I proceeded with creating a benchmarking model with top 5 numerical features ranks by correlation with target variable y (sale price). This benchmarking model serves as a baseline to assess our iterations performance.

Iterating through the model options systematically, from numerical to category, gives us the finalize production model consisting of 25 numerical features and 6 categorical feautres. This model is selected because it's simple and clean, allowing the model to become more robust and is less likely to overfit in the future.



### Conclusions and Recommendations
The model selected for production consists of:

```
# Numerical Columns 25

num_cols = ['overall_quality', 'total_sf','garage_cars','total_bath','year_built', 'has_fireplace', 'total_rooms_above_ground','has_open_porch', 'masonry_area', 'log_lot_area', 'lot_frontage',
'has_wood_deck', 'central_air', 'has_basement_sf','bedroom_above_ground', 'functional', 'street','has_2nd_floor_sf','month_sold', 'year_sold', 'lot_contour', 'lot_slope',
'overall_condition', 'kitchen_above_ground', 'lot_shape']

# Categorical Columns 6

cat_cols = ['external_quality','basement_quality','heating_quality','kitchen_quality','fireplace_quality','garage_quality']

```   
<br>

![production_model](./figure/production_model.png)

Kaggle submission model consists of 25 numerical columns and 14 categorical columns which performs well with 20,788 RMSE on kaggle, and the R^2 score of 0.93 and 0.91 for training and testing datasets.

However, if I was to select one model for production it would be our iteration model that consists of 25 numerical columns and 6 categorical columns (generic quality related columns) - performing very well with 21,084 RMSE on test dataset and an R^2 score of 0.92 and 0.92. Selecting a more generic feature that every houses must have allows the model to be more robust in the future as well as more resistance to overfitting. Its predictive power is good with low sale price. Additionally, it predicts differences in high price ranges (more than 350000 dollar).
