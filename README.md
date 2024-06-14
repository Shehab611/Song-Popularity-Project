<h1 align="center" id="title">Song Popularity Project</h1>

<p id="description">The objective of the project is to prepare me to apply various machine-learning algorithms to real-world tasks.
  This helped me enhance my understanding of the workflow involved in machine learning tasks. 
  I learned how to clean data apply pre-processing techniques conduct feature engineering and implement regression and classification methods.
  Each milestone showcases my proficiency in these techniques.
  The First Milestone is to apply different regression techniques to find the model that fits your data with minimum error. 
  The Second Milestone is to apply different classification techniques to find the model that fits in the data with minimum error.</p>


<h2>Preprocessing</h2>
1. Handle Null Data

  * Train Time:
    1. if the column type is an object then we choose the mode data of this column to use in filling the nulls
    2. if the column type is numerical then we get the median of this column to use in filling the nulls
    3. save this filling data in a text file to use in Test time
  * Test Time:
    1. get the filling data from the text file to use them
    2. use these data to fill the null in test data
2. Encode Data

    Encode all the label data to numerical ones so that the model can use it
3. Normalize the Data

    Normalize all the features using the min-max scaler 
4. Drop unique and low-correlated data

    As They won't affect the model accuracy and make further unwanted processing
  
<h2>Machine Learning Techniques</h2>
1. Train 10 different regression models

  * Linear Regression
  * Polynomial Regression
  * Ridge Regression
  * Lasso Regression
  * ElasticNet Regression
  * Support Vector Regression
  * Decision Tree Regression
  * Random Forest Regression
  * Gradient Boosting Regression
  * Polynomial with Lasso Regression

  Save Each Model After Training in a Pickle with its name

2. Train 12 different classification models

  * Logistic Regression
  * Decision Tree Classifier
  * Random Forest Classifier
  * Support Vector Classifier
  * Support Vector Linear Classifier
  * Support Vector RPF Classifier
  * K-Nearest Neighbors Classifier
  * Gaussian Naive Bayes Classifier
  * XGBoost Classifier
  * AdaBoost Classifier
  * Gradient Boost Classifier
  * Light Gradient Boost Classifier

  Save Each Model After Training in a Pickle with its name
<h2>💻 Built with</h2>

Technologies used in the project:

*   Python
*   train\_test\_split
*   LabelEncoder
*   MinMaxScaler
*   numpy
*   pickle
*   matplot
