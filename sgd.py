import pandas as pd
from sklearn import model_selection
from sklearn import linear_model
from sklearn import svm
from sklearn import naive_bayes
from sklearn.neighbors import NearestNeighbors

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



# 75.417% with restricted features and the other corresponding commented model. Model seems to have random variation
#    features = ['sload','dload','rate','sinpkt','response_body_len','dbytes','djit']


# seems more consistent than above, hovers around 74.5%+
#    features = ['dtcpb','sload','dload','rate','sinpkt','response_body_len','dbytes','djit']

#Might be better than above, hard to tell with instability
#    features = ['dtcpb','sload','dload','rate','sinpkt','response_body_len','sbytes','dbytes','djit']
 
    features = ['dtcpb','sload','dload','rate','sinpkt','response_body_len','sbytes','dbytes']


# get rid of text fields, use this if not using features = up above
 #   traindata = traindata.drop(['proto','service','state','attack_cat'],axis=1)
 #   testdata = testdata.drop(['proto','service','state','attack_cat'],axis=1)

# keep text fields except attack category as it's cheap
#    traindata = traindata.drop(['attack_cat'],axis=1)
#    testdata = testdata.drop(['attack_cat'],axis=1)

    X = traindata.loc[:, features].values # Pull out features that matter
#    X = traindata.loc[:, :].values # Pull out features that matter
    Y = traindata.loc[:,'label'].values # Pull out malicious or not value

    X_test = testdata.loc[:, features].values # Pull out features that matter
#    X_test = testdata.loc[:, :].values # Pull out features that matter
    Y_test = testdata.loc[:,'label'].values # Pull out malicious or not value

#75.417%, seems that there's randomness. Probably don't have a large enough dataset
    model = linear_model.SGDClassifier(tol=1e-6)

#SVC isn't really working
#Naive Bayes isn't really working
#idk how to do KNN, so haven't tested that

    model.fit(X,Y)

    y_pred = model.predict(X_test)

    # See where y_pred doesn't match Y_test, divide by number of observations, 1 - that to get accuracy instead of error rate
    print("Accuracy = {}" .format(1-((Y_test != y_pred).sum()/len(y_pred))))


if __name__ == "__main__":
    main()