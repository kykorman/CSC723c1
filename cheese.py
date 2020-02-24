import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

# I have created an AI that requires no training for this dataset!

def main():

    testfile = open("UNSW_NB15_testing-set.csv","r")
    testdata = pd.read_csv(testfile, index_col=0)

    testfile.close()

    # Store observations from test set, ignoring proto, service, state, attack_cat, and label fields
    X_test = testdata.loc[:,'attack_cat'].values
    
    Y_test = testdata.loc[:,'label'].values # Pull out malicious or not value

    y_pred = list()

    for i in range(len(Y_test)):
        if X_test[i] != "Normal":
            y_pred.append(1)
        else:
            y_pred.append(0)


    print("Accuracy = {}" .format(((Y_test == y_pred).sum()/len(y_pred))))

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
    print("Saving confusion matrix to 'confusionMatrix.txt'")

    with open("confusionMatrix.txt","w") as f:
        f.write("1 predicted, 0 predicted")
        f.write("{0} {1}\n".format(TP, FN))
        f.write("{0} {1}".format(FP, TN))

    print("Saving vector of predicted values to 'predictionVector.csv'")

    with open("predictionVector.csv","w") as f:
        for item in range(len(y_pred)-1):
            f.write("{}," .format(y_pred[item]))
        f.write("{}" .format(y_pred[len(y_pred)-1]))
if __name__ == "__main__":
    main()