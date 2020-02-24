import pandas as pd
from sklearn import svm
from sklearn import naive_bayes
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import model_selection
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# The above imports have many packages from when I was trying different models

def main():

    #Take in and store the files
    trainfile = open("UNSW_NB15_training-set.csv","r")
    traindata = pd.read_csv(trainfile, index_col=0)

    trainfile.close()

    testfile = open("UNSW_NB15_testing-set.csv","r")
    testdata = pd.read_csv(testfile, index_col=0)

    testfile.close()

    # Store observations from train set, ignoring proto, service, state, attack_cat, and label fields
    # With label included, forest hits 100% (big shocker.)
    X = traindata.drop(['proto','service','state','attack_cat','label'],axis = 1)
    X = X.loc[:,:].values

    Y = traindata.loc[:,'label'].values # Pull out malicious or not value

    # Store observations from test set, ignoring proto, service, state, attack_cat, and label fields
    X_test = testdata.drop(['proto','service','state','attack_cat','label'],axis = 1)
    X_test = X_test.loc[:,:].values
    
    Y_test = testdata.loc[:,'label'].values # Pull out malicious or not value

    model = RandomForestClassifier()

    model.fit(X,Y)

    y_pred = model.predict(X_test)

    # See where y_pred doesn't match Y_test, divide by number of observations, 1 - that to get accuracy instead of error rate
    print("Accuracy = {}" .format(1-((Y_test != y_pred).sum()/len(y_pred))))

    # generate confusion matrix
    confusionMatrix = plot_confusion_matrix(model, X_test, Y_test)

    print(confusionMatrix.confusion_matrix)
    plt.show()


    # Doing confusion matrix manually as well to prove I don't need a python package for everything :) 
    # Just most things :(
    TP=0
    FP=0
    TN=0
    FN=0

    for obs in range(len(Y_test)):
        if Y_test[obs] == 0:
            if y_pred[obs] == Y_test[obs]:
                TN+=1
            else:
                FP+=1
        else:
            if y_pred[obs] == Y_test[obs]:
                TP+=1
            else:
                FN+=1
    print("1 Predicted, 0 Predicted")
    print("{0} {1}".format(TP, FN))
    print("{0} {1}".format(FP, TN))

if __name__ == "__main__":
    main()
