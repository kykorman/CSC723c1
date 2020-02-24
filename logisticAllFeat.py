import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
def main():
    trainfile = open("UNSW_NB15_training-set.csv","r")
    traindata = pd.read_csv(trainfile, index_col=0)

    trainfile.close()

    testfile = open("UNSW_NB15_testing-set.csv","r")
    testdata = pd.read_csv(testfile, index_col=0)

    testfile.close()

# Scores higher without stcpb and dtcpb, 70.953 vs 70.598
    traindata = traindata.drop(['proto','service','state','attack_cat','stcpb','dtcpb'],axis=1)
    testdata = testdata.drop(['proto','service','state','attack_cat','stcpb','dtcpb'],axis=1)


    #features = ['stcpb','dtcpb','sload','dload','rate','dbytes','sinpkt','sbytes','response_body_len','djit']

    X = traindata.loc[:, :].values # Pull out features that matter
    Y = traindata.loc[:,'label'].values # Pull out malicious or not value

    X_test = testdata.loc[:, :].values # Pull out features that matter
    Y_test = testdata.loc[:,'label'].values # Pull out malicious or not value

    model = LogisticRegression()
    model.fit(X,Y)

    result = model.score(X_test,Y_test)

    print(result)



if __name__ == "__main__":
    main()