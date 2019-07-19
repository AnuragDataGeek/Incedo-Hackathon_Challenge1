import os,sys,time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.svm import NuSVR
from sklearn.svm import SVR
from sklearn import grid_search
from sklearn.preprocessing import StandardScaler

train_df=pd.read_csv('Dataset2147b1d/train_file.csv')
test_df=pd.read_csv('Dataset2147b1d/test_file.csv')

def unique_values(data,col):
    print(data[col].value_counts())

def missing_values(train_df):
    print(train_df.isnull().sum())
    '''
    c=0
    dic={}
    
    for i in range(train_df.shape[0]):
        if train_df.at[i,'GeoLocation'] is np.NaN:
            #print(train_df.at[i,'LocationDesc'])
            print(train_df[train_df['LocationDesc'] == train_df.at[i,'LocationDesc']]['GeoLocation'])
            
            c+=1
    print(c)
    '''
    #Dropping GeoLocation as we have LocationDesc variable, presence of GeoLocation variable is not required as it is a redundant variable
    train_df.drop(['GeoLocation'],axis=1,inplace=True)
    return train_df

def new_columns(train_df):
    train_df['Question_Description_combined']=train_df[['Greater_Risk_Question','Description']].apply(lambda x: ' '.join(x), axis=1)
    #print(train_df['Question_Description_combined'])
    #There is a column called QuestionCode which uniquely identifies a question
    train_df.drop(['Greater_Risk_Question','Description','Question_Description_combined'],axis=1,inplace=True)
    return train_df

def visualizations():
    y=train_df.groupby(['YEAR','Subtopic'])['Patient_ID'].count()
    print(y)
    y.plot.bar()
    plt.show()

def grade_vs_prob():
    y=train_df.groupby(['Grade'])['Greater_Risk_Probability'].mean()
    print(y)
    y.plot.bar()
    plt.show()

def samplesize_vs_prob():
    log_sample=[math.log(i) for i in train_df['Sample_Size']]
    #plt.plot(train_df['Greater_Risk_Probability'],log_sample)
    plt.scatter(train_df['Greater_Risk_Probability'],log_sample)
    plt.xlabel('Greater Risk Probability')
    plt.ylabel('Sample Size')
    plt.show()

def encoding(data,col):
    '''
    le=LabelEncoder()
    data.values[:,col]=le.fit_transform(data.values[:,col])
    oe=OneHotEncoder(categorical_features = [col])
    data=oe.fit_transform(data).toarray()
    print(data['YEAR'].head())
    print(type(data['YEAR'][7]))
    '''
    data=pd.get_dummies(data=data,columns=col)
    #print(data.head())
    #print(data.columns)
    print(data.shape)
    return data

def model1(X_train, X_test, y_train, y_test):
    lr = LinearRegression()
    lr.fit(X_train,y_train)
    y_pred=lr.predict(X_test)
    rms = sqrt(mean_squared_error(y_test, y_pred))
    print("Linear Regression:",rms)

def model2(X_train, X_test, y_train, y_test):
    st_time=time.time()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    parameters = {'kernel': ('linear', 'rbf','poly'), 'C':[0.001,0.01,0.1,1, 10],'gamma': [1e-7, 1e-4],'epsilon':[0.1,0.2,0.3]}
    #svm = NuSVR()
    svr = SVR()
    svm_grid = grid_search.GridSearchCV(svr, parameters)
    svm_grid.fit(X_train_scaled, y_train)
    print("Best parametesrs: ", svm_grid.best_params_)
    print("Best score:", svm_grid.best_score_)
    #y_pred = svm.predict(X_test_scaled)
    #rms = sqrt(mean_squared_error(y_test, y_pred))
    #print(" SVM Regression:",rms)
    print("Time taken :", time.time()-st_time)

def model3(X_train, X_test, y_train, y_test):
    st_time=time.time()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, loss='squared_loss')
    sgd_reg.fit(X_train_scaled, y_train)
    y_pred = sgd_reg.predict(X_test_scaled)
    rms = sqrt(mean_squared_error(y_test, y_pred))
    print(" Stochastic Gradient Regression:",rms)

def final_model(X_train, X_test, y_train, y_test,test_data):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    parameters = {'kernel': 'linear', 'C': 0.01,'gamma': 1e-7,'epsilon': 0.1}
    svr = SVR(kernel='linear', C=0.01, gamma=1e-7, epsilon=0.1)
    svr.fit(X_train_scaled, y_train)
    y_pred = svr.predict(X_test_scaled)
    rms = sqrt(mean_squared_error(y_test, y_pred))
    print(" SVM Regression:",rms)
    print(test_data.columns)
    print(test_data.shape)
    test_data_scaled=scaler.fit_transform(test_data.drop(['Patient_ID','Sample_Size'],axis=1))
    submit_pred = svr.predict(test_data_scaled)
    test_data['Greater_Risk_Probability']=submit_pred
    submission_data=test_data[['Patient_ID','Greater_Risk_Probability']]
    submission_data['Greater_Risk_Probability']=submission_data['Greater_Risk_Probability'].apply(lambda x: 0 if x<=0 else x)
    submission_data.to_csv('submission_file.csv', index=False)



if __name__ == "__main__":
    train_df=missing_values(train_df)
    unique_values(train_df,'LocationDesc')
    #print(train_df.groupby(['LocationDesc'])['GeoLocation'][0])
    print(train_df['LocationDesc'].nunique())
    print(train_df['GeoLocation'].nunique())
    visualizations()
    grade_vs_prob()
    samplesize_vs_prob()
    train_df=new_columns(train_df)
    
    #print(train_df.columns)
    encoders_list=['YEAR','LocationDesc','Sex','Race','Grade','QuestionCode','StratID1','StratID2','StratID3','StratificationType']
    data=encoding(train_df,encoders_list)
    #print(data.head())
    X_train, X_test, y_train, y_test = train_test_split(data.drop(['Patient_ID','Greater_Risk_Probability','Sample_Size'],axis=1),data['Greater_Risk_Probability'], test_size=0.2)
    model1(X_train, X_test, y_train, y_test)
    model2(X_train, X_test, y_train, y_test)
    model3(X_train,X_test, y_train, y_test)
    print(test_df.isnull().sum())
    test_df=missing_values(test_df)
    test_df=new_columns(test_df)
    test_data=encoding(test_df,encoders_list)
    
    final_model(X_train,X_test, y_train, y_test,test_data)
