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

#70.6%
#    features = ['stcpb','dtcpb','sload','dload','rate','dbytes','sinpkt','sbytes','response_body_len','djit']
#74.32%
    #features = ['sload','dload','rate','dbytes', 'sinpkt']

# 74.6%
#    features = ['sload','dload','rate', 'sinpkt']

#no smean, no stcpb, dtcpb, sbytes
#74.7255
#    features = ['sload','dload','rate','sinpkt','response_body_len','dbytes']

#no trans_depth, ftp stuff, is_sm_ips_ports, ct_state_ttl, ct_src_ltm
#the entirety of previous state boosts the accuracy, with dload being the most important

# stcpb and dtcpb hurt the performance a LOT
    features = ['sload','dload','rate','sinpkt','response_body_len','dbytes','djit']

    X = traindata.loc[:, features].values # Pull out features that matter
    Y = traindata.loc[:,'label'].values # Pull out malicious or not value

    X_test = testdata.loc[:, features].values # Pull out features that matter
    Y_test = testdata.loc[:,'label'].values # Pull out malicious or not value

    model = LogisticRegression()
    model.fit(X,Y)

    y_pred = model.predict(X_test)

    print("Accuracy = {}" .format(1-((Y_test != y_pred).sum()/len(y_pred))))
#    result = model.score(X_test,Y_test)

#    print(result)



if __name__ == "__main__":
    main()
