Feature Engineering:

1. There were lots of missing values in the columns GeoLocation but we were having Locationdesc which signifies the same thing and hence GeoLocation is a redundant variable. Hence, dropped it.
2. There are columns which provides desciption of the questions(Greater_Risk_Question,Description) and we have another variable called QuestionCode which more or less means the same. So,dropping the Greater_Risk_Question,Description variables

Visualizations:
1. Greater_Risk_Probability was higher for someone whose Grade is 2 compared to the others
2. Tried to visualize Greater_Risk_Probability against Sample_Size which had no relation and dropped the Sample Size while training model

Model Building:
Tried Linear Regression, Stochastic Gradient Descent Regressor and SVM Regressor and performed hyperparameters tuning using GridSearchCV and finally chose SVM Regressor with a certain set of parameters obtained from Grid Search CV.
