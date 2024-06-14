<h1 align="center" id="title">Song Popularity Project</h1>

<p id="description">The objective of the project is to prepare me to apply various machine learning algorithms to real-world tasks.
  This helped me enhance my understanding of the workflow involved in machine learning tasks. 
  I learned how to clean data apply pre-processing techniques conduct feature engineering and implement regression and classification methods.
  Each milestone showcasing my proficiency in these techniques.
  The First Milestone is to apply different regression techniques to find the model that fit your data with minimum error. 
  The Second Milestone is to apply different classification techniques to find the model that fit in the data with minimum error.</p>


<h2>Preprocessing</h2>
1. Handle Null Data
  * Train Time:
    1. if the column type is object then we choose the mode data of this column to use it in filling the nulls
    2. if the column type is numerical then we get the median of this column to use it in filling the nulls
    3. save this filling data in text file to use them in Test time
  * Test Time:
    1. get the filling data from the text file to use them
    2. use these data to fill the null in test data
2. Encode Data
    Encode all the label data to numerical one so that the model can use it
3. Normalize the Data
    Normalize all the features using min max scaler 
4. Drop unique and low correlated data
    As They won't affect the model accuracy and make further un wanted processing
  
<h2>Machine Learning Techniques</h2>
1. Train 10 different regression models
    |  xx |  tt | b  | f  |q   |
    |---|---|---|---|---|
    | 1  |  2 |   |   |   |
    |  1 |  2 |   |   |   |
    |   1|  2 |   |   |   |

  5. Save the Model after training
    Save the Model in Pickle after training so i can use it when test without training again and consume more time
*   Get the saved model in test data
*   apply 10 different regression models
*   apply 12 different classification models
*   Plot the data
<h2>💻 Built with</h2>

Technologies used in the project:

*   Python
*   train\_test\_split
*   LabelEncoder
*   MinMaxScaler
*   numpy
*   pickle
*   matplot
