import os
from six.moves import urllib
import tarfile
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV


import os
os.system('cls')

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH+ "/housing.tgz"

def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH):
    ''' download a file using http and unzip file '''
    # make a directory for data
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path , "housing.tgz")
    # download file from given url to the path
    urllib.request.urlretrieve(housing_url, tgz_path)
    # unzip the file and save it in a given path
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()

def load_housing_data(housing_path = HOUSING_PATH ):
    ''' load data which is in the format of csv
    OUTPUT : data frame'''
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

### take a quich look
data = load_housing_data()
print data.head()
data.info()

print data['ocean_proximity'].value_counts()

print data.describe()

data.hist(bins = 50)
data.plot( kind = 'box', sharex = False, subplots =True, layout = (3,3))

np.random.seed(42)
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(data.shape[0])
    m_test = np.int(data.shape[0]*test_ratio)
    test_data = data.iloc[shuffled_indices[:m_test]]
    train_data = data.iloc[shuffled_indices[m_test:]]
    return train_data, test_data
# def split_train_test(data, ratio):
#     data.sample(frac = 1)
#     m_test = np.int(len(data)* ratio)
#     test_data = data[:m_test]
#     train_data = data[m_test:]
#     return train_data, test_data
#train_data, test_data = split_train_test(data, .2)

# the same function as above in scikit learn, randomly split train and test
train_set , test_set = train_test_split(data, test_size = .2 , random_state = 42)
print " number of texst and train_data", len(train_set), len(test_set)

### consider we know median_income is important feature
# to avoid bias sampling in test set for that feature : use stratified sampling
plt.figure()
data['median_income'].hist(bins = 50)

# convert numeical data to categorical
data['income_cat'] = np.ceil(data['median_income']/1.5)
data['income_cat'].where(data['income_cat'] <  5, 5., inplace=True )


split = StratifiedShuffleSplit(n_splits = 1, test_size= .2, random_state = 42)
for train_index, test_index in split.split(data, data['income_cat']):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

# print data['income_cat'].value_counts()/len(data)
# print strat_test_set['income_cat'].value_counts()/len(strat_test_set)
# train_set , test_set = train_test_split(data, test_size = .2 , random_state = 42)
# print test_set['income_cat'].value_counts()/len(test_set)
#

for set in (strat_train_set, strat_test_set):
    set.drop("income_cat", axis=1, inplace=True)

### visualize the data to gain insights
explore_set = strat_train_set.copy()

explore_set.plot(kind = 'scatter', x = 'longitude', y = 'latitude' , alpha = .1)

explore_set.plot( kind = 'scatter', x='longitude', y = 'latitude' , alpha= .4,\
s = explore_set['population']/100 , label = "population",\
c = "median_house_value" , cmap = plt.get_cmap("jet"), colorbar = True)

explore_set.plot( kind = 'scatter', x='longitude', y = 'latitude' , alpha= .4,\
c = explore_set['median_income'] , cmap = plt.get_cmap("jet"), colorbar = True)

### corrolation matrix : shows just linear corrolation
corr_matrix = explore_set.corr()
print corr_matrix["median_house_value"].sort_values(ascending = False)

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(explore_set[attributes], figsize= (12,8))

### experimenting the attribute combination
explore_set['rooms_per_household'] = explore_set["total_rooms"]/explore_set["households"]
explore_set['bedrooms_per_rooms'] = explore_set["total_bedrooms"]/explore_set["total_rooms"]
explore_set['population_per_household'] = explore_set['population']/explore_set['households']

corr_matrix = explore_set.corr()
print corr_matrix["median_house_value"].sort_values(ascending = False)

### Data cleaning
x_train = strat_train_set.drop('median_house_value', axis = 1)
y_train = strat_train_set['median_house_value'].copy()
## handling missing Data
# # opion 1
# x_train.dropna(subset = ["total_bedrooms"]).info()
# # option 2
# x_train.drop(["total_bedrooms"], axis =1)
# #option3
# median = x_train['total_bedrooms'].median()
# x_train["total_bedrooms"].fillna(median)
# x_train.info()

imputer = Imputer(strategy = "median")
x_train_num = x_train.drop('ocean_proximity', axis =1)
imputer.fit(x_train_num)
print imputer.statistics_
# resul is numpy array
X_np = imputer.transform(x_train_num)
X_pd = pd.DataFrame(X_np, columns = x_train_num.columns)

## handling text data
# first step : text category to integer category to number
le = LabelEncoder()
le.fit(x_train["ocean_proximity"])
print le.classes_
x_train_cat_encoded = le.transform(x_train["ocean_proximity"])
print x_train_cat_encoded

# step2 from integer category to one hot vectors
ohe = OneHotEncoder()
# result is scipy sparse matrix
x_train_cat_1hot = ohe.fit_transform(x_train_cat_encoded.reshape(-1,1))
print type(x_train_cat_1hot)
#x_train_cat_1hot.toarray()

## or ( step1+step2)

# encoder = LabelBinarizer()
#x_train_cat_1hot = encoder.fit_transform(x_train["ocean_proximity"])
# return np

### custom transformation
class CombinedAttributesAdder(TransformerMixin, BaseEstimator):
    ''' the first base class is used for implementing fit_transform easily
    the second case class is used for getting get_params and set_params methods
    these two methods are usefull for automatic hyperparameter tuning'''
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y = None):
        rooms_per_household = X[:,rooms_ix]/X[:,households_ix]
        population_per_household = X[:,population_ix]/X[:,households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_rooms = X[:, bedrooms_ix]/X[:,rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
            bedrooms_per_rooms]
        else:
            return np.c_[X ,rooms_per_household, population_per_household]
            # resul is numpy array

rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room = False)
x_train_extra_attr = attr_adder.transform(x_train.values) # just works with .values
#x_train_extra_attr = attr_adder.transform(X_pd.values)
#fi = np.c_[x_train_extra_attr, x_train_cat_1hot.toarray() ]

### feature scaling
# min max scaling
scaler = MinMaxScaler()
scaler.fit(X_pd)
scaled_data = scaler.transform(X_pd)
print scaled_data[0,:]
# or X_np as input data
s_scaler = StandardScaler()
s_scaler.fit(X_pd)
scaled2_data = s_scaler.transform(X_pd)
print scaled2_data[0,:]

### pipeline
# first define a class called DataFrameSelector
class DataFrameSelector(TransformerMixin, BaseEstimator):
    def __init__(self, attributes_list):
        self.attributes_list = attributes_list
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attributes_list].values
        # .values cincert it to numpy array
        # we are going to use sklearn which is build upon numpy array

# pipeline for numerical data
# sequence of transformation
num_attributes_list = list(x_train_num)
cat_attributes_list = ["ocean_proximity"]

num_pipeline = Pipeline([\
('num_data_selector', DataFrameSelector(num_attributes_list)),
('imputer' , Imputer(strategy = "median")), \
('attr_adder' , CombinedAttributesAdder()),\
('s_scaler' , StandardScaler())\
 ])

# num_pipeline.fit(x_train_num)
# x_train_num_tr = num_pipeline.transform(x_train_num)

# pipeline for categorical data
# sequence of transformation
cat_pipeline = Pipeline([\
('cat_data_selector', DataFrameSelector(cat_attributes_list)),
('encoder',  LabelBinarizer())\
])

# join categorical and numerical pipeline
# transformer_list : list of transformation\pipelines(tuple), the first one is\
# the name of it
full_pipeline = FeatureUnion(transformer_list = [
("pipeline_1", num_pipeline),
("pipeline2", cat_pipeline),
])

x_train_prepared = full_pipeline.fit_transform(x_train)
print x_train_prepared.shape

### train a model

y_train_np  = y_train.as_matrix()
x_test = strat_test_set.drop('median_house_value', axis = 1, inplace=False)
y_test = strat_test_set['median_house_value'].copy()
x_test_prepared = full_pipeline.transform(x_test)
y_test_np = y_test.as_matrix()

print "**Linear Regression**"
lin_reg = LinearRegression()
lin_reg.fit(x_train_prepared, y_train_np)

coeficients = lin_reg.coef_
prediction = lin_reg.predict(x_train_prepared)
RSS_train = mean_squared_error(y_train_np, prediction)
RMSE_train = np.sqrt(RSS_train)
print "RMSE_train  :\t" , RMSE_train
prediction_test = lin_reg.predict(x_test_prepared)
RSS_test = mean_squared_error(y_test.as_matrix(), prediction_test)
RMSE_test = np.sqrt(RSS_test)
print "RMSE_test :\t", RMSE_test

### plot learning curve
# m_train = x_train_prepared.shape[0]
# RMSE_test_list,RMSE_train_list,m_list = [], [],[]
# for i in range(15):
#     m = np.int(((i+1)/15.)*m_train)
#     x_partial = x_train_prepared[:m,:]
#     y_partial = y_train_np[:m]
#     lin_reg_partial = LinearRegression()
#     lin_reg_partial.fit(x_partial, y_partial)
#     prediction_partial = lin_reg_partial.predict(x_partial)
#     RMSE_train = np.sqrt(mean_squared_error(y_partial, prediction_partial))
#     RMSE_train_list.append(RMSE_train)
#     prediction_test = lin_reg_partial.predict(x_test_prepared)
#     RMSE_test = np.sqrt(mean_squared_error(y_test.as_matrix(), prediction_test))
#     RMSE_test_list.append(RMSE_test)
#     m_list.append(m)
# plt.figure()
# plt.plot(m_list, RMSE_test_list, 'ro', label = 'test data')
# plt.plot(m_list, RMSE_train_list, 'bo', label = 'train data')
# plt.xlabel(' number of train data points')
# plt.ylabel('error')
# plt.title('Learning curve')
# plt.legend()

## using a more complex model : Decision Tree
tree_reg = DecisionTreeRegressor(random_state = 42)
tree_reg.fit(x_train_prepared, y_train_np)

prediction = tree_reg.predict(x_train_prepared)
RMSE_train_tree = np.sqrt(mean_squared_error(y_train_np, prediction))
print "**Decision Tree**"
print "RMSE_train  :\t" , RMSE_train_tree
prediction_test = tree_reg.predict(x_test_prepared)
RMSE_test_tree = np.sqrt(mean_squared_error(y_test.as_matrix(), prediction_test))
print "RMSE_test :\t", RMSE_test_tree


### validation
##1) use the sklearn.model_selection.train_test_split
## 2) cross validation
def my_scorer(estimator, X , y ):
    prediction = estimator.predict(X)
    rmse = np.sqrt(mean_squared_error(y, prediction))
    return rmse
#print my_scorer(estimator = lin_reg, X = x_test_prepared, y = y_test.as_matrix())
scores_cv_tree = cross_val_score(estimator = tree_reg, X =x_train_prepared, y=y_train_np\
, scoring = my_scorer, cv=10)

# or this one
scores_cv2= cross_val_score(estimator = tree_reg, X =x_train_prepared, y=y_train_np\
, scoring = "neg_mean_squared_error", cv=10)
scores_cv2_tree = np.sqrt(-scores_cv2)

def display_scores(CV_scores):
    #print ("scores :\n"), CV_scores
    print ("Validation sets Mean:\t\t"), np.mean(CV_scores)
    print ("Validation sets standard deviation"), np.std(CV_scores)

print "**Decision Tree**"
display_scores(scores_cv_tree)

scores_cv_lin = cross_val_score(estimator = lin_reg, X =x_train_prepared, y=y_train_np\
, scoring = my_scorer, cv=10)
print "**Linear Regression**"
display_scores(scores_cv_lin)

## another complex model : random forest regressor
forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(x_train_prepared, y_train_np)
print "\n**random forest resgressor**"
print "RMSE_train", my_scorer(forest_reg, x_train_prepared , y_train_np)
print "RMSE_test",my_scorer(forest_reg, x_test_prepared , y_test.as_matrix() )

scores_cv_forest = cross_val_score(estimator = forest_reg, X =x_train_prepared,\
 y=y_train_np, scoring = "neg_mean_squared_error", cv=10)
display_scores(np.sqrt(-scores_cv_forest))

## save models for later use
joblib.dump(lin_reg,'model_lin_reg.pkl')
# later
#lin_reg_model= joblib.load(''model_lin_reg.pkl'')

### fine tune model
forest_reg = RandomForestRegressor(random_state=42)
param_grid = [\
  {'n_estimators' : [3, 10,30], 'max_features':[2,4,6,8]},
  {'bootstrap': [False], 'n_estimators' : [3,10] ,'max_features':[2,3,4]},
 ]
grid_search = GridSearchCV(forest_reg, param_grid, cv = 5,
 scoring = "neg_mean_squared_error")
grid_search.fit(x_train_prepared, y_train_np)

a = grid_search.cv_results_['params']
b = np.sqrt(-grid_search.cv_results_['mean_test_score'])
print len(a)
for i in range(len(a)):
    print a[i], b[i]
print "best params found for random forest in  grid search" ,grid_search.best_params_
print grid_search.best_estimator_

final_model = grid_search.best_estimator_
final_model.fit(x_train_prepared, y_train_np)
print "\n**tuned random forest resgressor**"
print "RMSE_train", my_scorer(final_model, x_train_prepared , y_train_np)
print "RMSE_test",my_scorer(final_model, x_test_prepared , y_test_np )

feature_importance = final_model.feature_importances_
extra_attributes = ['rooms_per_household', 'population_per_household',
                    'bedrooms_per_rooms']
cat_one_hot_attribs = list(le.classes_)
attributes = list(x_train_num) + extra_attributes + cat_one_hot_attribs

tup = zip(feature_importance, attributes)
tup_sorted = sorted(tup, reverse = True)
for i in range(len(attributes)):
    print tup_sorted[i]

##excersise 1: using svr as a model and GreadSerachCV
# print "\n**SVR**"
# svm_reg = SVR()
# svm_reg.fit(x_train_prepared, y_train_np)
# print "SRV RMSE train:", my_scorer(svm_reg, x_train_prepared , y_train_np )
# print "SVR RMSE test:", my_scorer(svm_reg, x_test_prepared , y_test_np )
# scores_cv_svr = cross_val_score(estimator = svm_reg, X= x_train_prepared,\
#  y =y_train_np, scoring=my_scorer, cv=10)
# display_scores(scores_cv_svr)

###* excercise 3
class select_important_attr(TransformerMixin, BaseEstimator):
    def __init__(self, feature_importance, k):
        self.feature_importance = feature_importance
        self.k = k
    def fit(self, X,y=None):
        self.important_attr_ = np.argsort(self.feature_importance )[::-1][:self.k]
        return self
    def transform(self, X):
        return X[:, self.important_attr_]
# to check the new transformer
# select = select_important_attr(feature_importance, k = 5)
# x_new = select.fit_transform(x_train_prepared)
# print x_new.shape
# considered_attr  = [attributes[i] for i in select.important_attr_]
# print considered_attr

prepare_and_select_pipeline  = Pipeline([\
('FullPipeline', full_pipeline),
('select',  select_important_attr(feature_importance, k = 5))\
])

selection_x_train_prepared = prepare_and_select_pipeline.fit_transform(x_train)
print selection_x_train_prepared.shape

#** exercise 4
class SupervisionFriendlyLabelBinarizer(LabelBinarizer):
    def fit_transform(self, X, y=None):
        return super(SupervisionFriendlyLabelBinarizer, self).fit_transform(X)

class SupervisionFriendlyLabelBinarizer2(LabelBinarizer):
    def fit_transform(self, X, y=None):
        return LabelBinarizer.fit_transform(self, X)

cat_pipeline.steps[1] = ('new_encoder',  SupervisionFriendlyLabelBinarizer2())

prepared_select_model_pipeline = Pipeline([\
 ('FullPipeline', full_pipeline),
 ('select',  select_important_attr(feature_importance, k = 5)),
 ('tree_reg ', DecisionTreeRegressor(random_state = 42)),
 ])

prepared_select_model_pipeline.fit(x_train, y_train_np)
scores_cv_here = cross_val_score(estimator = prepared_select_model_pipeline, \
 X = x_train, y=y_train_np, scoring=my_scorer, cv=10)
display_scores(scores_cv_here)

# some_data = x_train[0:4]
# prediction_some_data = prepared_select_model_pipeline.predict(some_data)
# print prediction_some_data
# label_some_data = y_train_np[0:4]
# print label_some_data # still overfitted although remove some features

#* excersice 5
param_grid = [{'select__k' : [5,6,7],
              'FullPipeline__pipeline_1__imputer__strategy': ['mean', 'median', 'most_frequent']}\
              ]
grid_search_prepared = GridSearchCV(prepared_select_model_pipeline, param_grid,\
 scoring=my_scorer,cv=5, verbose=2)

grid_search_prepared.fit(x_train,y_train_np)

a = grid_search_prepared.cv_results_['mean_test_score']
b = grid_search_prepared.cv_results_['params']
for i in range(len(a)):
    print a[i], b[i]
print grid_search_prepared.best_params_
print grid_search_prepared.best_estimator_
