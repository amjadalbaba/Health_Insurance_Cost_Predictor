# Health_Insurance_Cost_Predictor
Repo for the data science health insurance cost prediction project

## Project Objective
Every insurance company needs a smooth way that helps it identify the charges(premium) of their customers based on different aspects and situations, for this sake, this project was built in order to solve this issue and to draw insights
based on the company's data to extract some conclusions that would boost it's work and to understad more how it should interact with customers. 

## Project Overview
* Created a tool that estimates the charge that a customer should pay based on his/her situation.
* Drawing insights and some business startegies or plans that a health insurance company should implement.
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

* Changing the smoker columns values like "yes" => 1 & "no" => 0.

    ```python
    all_data['smoker'] = all_data['smoker'].apply(lambda x: 1 if(x == 'yes') else 0)
    ```
* Rounding the charges values in order to have more straightforward values.

    ```python
    all_data['charges'] = np.around(all_data['charges']) 
    ```
* Exporing the new dataset.

     ```python
    all_data.to_csv("cleaned_data.csv", index=False) 
    ```

## Exploratory Data Analysis (EDA)
I looked at the distributions of the data and the value counts for the various numerical & categorical variables. Below are a few highlights of the visualizations.

![alt text](https://github.com/amjadalbaba/Health_Insurance_Cost_Predictor/blob/master/Images/heatmap.png)

![alt text](https://github.com/amjadalbaba/Health_Insurance_Cost_Predictor/blob/master/Images/female_male_smoker_no_smoker.png)

![alt text](https://github.com/amjadalbaba/Health_Insurance_Cost_Predictor/blob/master/Images/histograms.png)

![alt text](https://github.com/amjadalbaba/Health_Insurance_Cost_Predictor/blob/master/Images/scatter.png)


## Contact
* You can email me [here](amjad.baba91@gmail.com).  
* Get in touch with my blog posts on [medium](https://medium.com/@amjad.baba913), and don't forget to drop your comments!
