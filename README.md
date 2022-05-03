# Health_Insurance_Cost_Predictor
Repo for the data science health insurance cost prediction project

## Project Objective
Every insurance company needs a smooth way that helps it identify the charges(premium) of their customers based on different aspects and situations, for this sake, this project was built in order to solve this issue and to draw insights
based on the company's data to extract some conclusions that would boost it's work and to understad more how it should interact with customers. 

## Project Overview
* Created a tool that estimates the charge that a customer should pay based on his/her situation.
* Drawing insights and some business strategies or plans that a health insurance company should implement.
* Used cross validation to draw out which will give the best model among: Linear, Lasso, and Random Forest Regressors.

### Resources Used
* Python Version: 3.9.7
* You can reach out the dataset used [here](https://www.kaggle.com/datasets/mirichoi0218/insurance).

This dataset is made up of:

* age
* sex
* bmi
* children
* smoker
* region
* charges

### Methods Used
* Data Cleaning
* Data Visualization & Exploratory Data Analysis(EDA)
* Machine Learning

### Technologies & Libraries
* Python
* jupyter
* Numpy
* Pandas
* Matplotlib
* Seaborn
* Sklearn

## Data Cleaning
After I imported the kaggle dataset, thankfully it was almost clean, but I edited some places such as:

* Changing the smoker column's values like "yes" => 1 & "no" => 0.

    ```python
    all_data['smoker'] = all_data['smoker'].apply(lambda x: 1 if(x == 'yes') else 0)
    ```
* Rounding the charges values in order to have more straightforward values.

    ```python
    all_data['charges'] = np.around(all_data['charges']) 
    ```
* Exporting the new dataset.

     ```python
    all_data.to_csv("cleaned_data.csv", index=False) 
    ```
    
## Exploratory Data Analysis (EDA)
I looked at the distributions of the data and the value counts for the various numerical & categorical variables. Below are a few highlights of the visualizations.

![alt text](https://github.com/amjadalbaba/Health_Insurance_Cost_Predictor/blob/master/Images/heatmap.png)

![alt text](https://github.com/amjadalbaba/Health_Insurance_Cost_Predictor/blob/master/Images/female_male_smoker_no_smoker.png)

![alt text](https://github.com/amjadalbaba/Health_Insurance_Cost_Predictor/blob/master/Images/histograms.png)

![alt text](https://github.com/amjadalbaba/Health_Insurance_Cost_Predictor/blob/master/Images/scatter.png)

## Model Building 

First, I transformed the categorical variables into dummy variables as a prerequisite:

```python
all_data_dum = pd.get_dummies(all_data)
```

Then, I made the data split into train and tests sets with a test size of 20%.   

I tried three different models and evaluated them using ```cross_val_score```. I chose cross validation because it can show how each algorithm will act on my model and which one will give the highest score by applying different train/test splits.   

I tried three different models:

* Linear Regression
* Lasso Regression
* Random Forest  

## Model performance
The Random Forest model far outperformed the other approaches regarding it's score. 

*	**Linear Regression** : ```[0.75229225, 0.78832069, 0.7388152 , 0.69943941, 0.7462162]```
*	**Lasso Regression**  : ```[0.75224359, 0.78835653, 0.73883871, 0.69949229, 0.74622792]```
*	**Random Forest**     : ```[0.84178716, 0.86191325, 0.8137814 , 0.8308621 , 0.82653594]```

## Model Tuning
To sum it all up, always we need to find what help us in using less computational complexity and since there is no significant score difference when utilizing different number of estimators as shown below, I will use the n_estimators = 100.

```python
for i in range(100, 200, 10):
          print((i, np.mean(cross_val_score(RandomForestRegressor(n_estimators = i), X, np.ravel(y)))))
```

## Contact
* You can email me on: amjad.baba91@gmail.com.  
* Get in touch with my blog posts on [medium](https://medium.com/@amjad.baba913), and don't forget to drop your comments!
