import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# Maybe start with KNN

def main():
    testfile = open("UNSW_NB15_training-set.csv","r")
#    testset = pd.read_csv("UNSW_NB15_testing-set.csv")
    data = pd.read_csv(testfile, index_col=0)

    testfile.close()

    # Feature reduction stuff
#    X = trainset.iloc[:,0]
#    Y = trainset.iloc[:,-1]

    data = data.drop(columns="proto")
    data = data.drop(columns="state")
    data = data.drop(columns="attack_cat")
    data = data.drop(columns="service")


    X = data.iloc[:,:]  #independent columns
    y = data.iloc[:,-1]    #target column i.e price range
    print(X) 
    print(y)
    #apply SelectKBest class to extract top 10 best features


    bestfeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    print(featureScores.nlargest(10,'Score'))  #print 10 best features

if __name__ == '__main__':
    main()