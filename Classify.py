import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold,cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest,VarianceThreshold, f_classif
from scipy.stats import itemfreq
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier,VotingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

#file paths for train and test data
fnameTrain = 'data/train.dat.txt'
fnameTrainLbl = 'data/train.labels.txt'
fnameTest = 'data/test.dat.txt'

#load the dataset to memory
print "Reading data...."
dataTrain = np.loadtxt(fnameTrain)
trainLables = np.loadtxt(fnameTrainLbl,dtype = int)
dataTest = np.loadtxt(fnameTest)
print "Reading complete..."

#split the data to train and test -for training mode
#X_train, X_test, Y_train, Y_test = train_test_split(dataTrain, trainLables, test_size=0.40, random_state=15)


#for test mode
X_train = dataTrain
Y_train = trainLables
X_test = dataTest

#verify the item frequencies
print itemfreq(Y_train)

######################## SOME FAILED ATTEMPTS ################
#C = {4:500,5:400,6:200,7:250,10:100,11:180}
#C1={1:5000,2:3500,3:1000}
#rus = RandomUnderSampler(random_state=42, ratio=C)
#X_resampled, y_resampled = rus.fit_sample(X_train, Y_train)
#ros = RandomUnderSampler(random_state=1, ratio=C1)
#X_resampled, y_resampled = ros.fit_sample(X_train, Y_train)
#itemfreq(y_resampled)
#sm = SMOTE(random_state=42,k_neighbors=2)
#X_resampled, y_resampled = sm.fit_sample(X_train, Y_train)

# sc_x = StandardScaler()
# x_trn = sc_x.fit_transform(X_train)
# x_tst = sc_x.transform(X_test)

# scalar = preprocessing.MinMaxScaler(feature_range=(0, 1))
# x_trn = MinMaxScaler(x_trn, feature_range=(1e-10, 10))
# x_trn = scalar.fit_transform(X_train)
# x_tst = scalar.fit_transform(X_test)

# clf = PCA(n_components = 50, random_state=42)
# X_new = clf.fit_transform(x_trn,Y_train)
# X_test_new = clf.transform(x_tst)

# clf = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
# X_new = clf.fit_transform(x_trn,Y_train)
# X_test_new = clf.transform(x_tst)
################ ################ ################ ##############

#remove constant features from the train data
vmod = VarianceThreshold()
x_trn = vmod.fit_transform(X_train)
print x_trn.shape
#remove constant features from the test data
x_tst = vmod.transform(X_test)
print x_tst.shape

#perform feature selection using F-value on train data to select best scoring 60 features
featureSelector = SelectKBest(f_classif, k=60)
X_new = featureSelector.fit_transform(x_trn,Y_train)
print X_new.shape
#reduce dimensionality of train data using F-value
X_test_new = featureSelector.transform(x_tst)
print X_test_new.shape

#Perform classification using
print('Training...')
clf = ExtraTreesClassifier(random_state=1, class_weight ='balanced_subsample', n_estimators=200, min_samples_split = 5,max_features=None)
#Try with voting classifier
#alg1 = ExtraTreesClassifier(random_state=1, class_weight ='balanced_subsample', n_estimators=200, min_samples_split = 5, max_features=None)
#alg2 = KNeighborsClassifier(n_neighbors=3)
#CLF = VotingClassifier(estimators=[('ET',alg1), ('KN',alg2)],voting='soft',weights=[2,0.5])
clf.fit(X_new, Y_train)
print('Predicting...')
predict = clf.predict(X_test_new)

# perform cross validation and classification report - for train mode
#cv = np.mean(cross_val_score(clf, X_new, Y_train, cv=10))
#print cv
#print '\n clasification report:\n', classification_report(Y_test,predict)

#Save the output - for test mode
predict.tofile('data/output.txt', sep="\n", format="%d")
print('Done..')