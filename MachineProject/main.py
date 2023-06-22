from ast import literal_eval
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error , r2_score
import plotly.express as ex
import plotly.graph_objs as go
import plotly.figure_factory as ff
import bs4 as bs
from sklearn.svm import SVR
import re
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MaxAbsScaler
from sklearn import preprocessing, metrics
from sklearn.metrics import accuracy_score
import json
import string
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.linear_model import Ridge

Movies_Data = pd.read_csv("movies-regression-dataset.csv")

# split data
X = Movies_Data.iloc[:, 0:-1]  # Features
Y = Movies_Data['vote_average']  # Label
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.20, random_state=0, shuffle=False)
# print(Movies_Data.isna().sum())




def pre(X_train):
    # preprocessing (homepage)
    x1 = X_train['homepage']
    x2 = X_train['original_title']
    for i in range(0, len(x1)):
        s = 'http://www.' + x2.iloc[i] + 'movie.com/'
        p = ""
        for j in range(0, len(s)):
            if s[j] != " ":
                p += s[j]
        x1[i] = p
    X_train['homepage'] = x1
    lehome = LabelEncoder()
    X_train.homepage = lehome.fit_transform(X_train.homepage)

    # preprocessing (genres)
    for i in range(0, len(X_train['genres'])):
        X_train['genres'].iloc[i] = json.loads(X_train['genres'].iloc[i])

    for i in range(0, len(X_train['genres'])):
        l = []
        for j in X_train['genres'].iloc[i]:
            l.append(j['name'])
        X_train['genres'].iloc[i] = l
    for i in range(0, len(X_train['genres'])):
        X_train['genres'].iloc[i].sort()
        s = ""
        for j in X_train['genres'].iloc[i]:
            s += j
        s = s.replace(" ", "")
        X_train['genres'].iloc[i] = s
    legenres = LabelEncoder()
    X_train.genres = legenres.fit_transform(X_train.genres)

    # preprocessing (budget)
    X_train['budget'] = X_train['budget'].replace(0, X_train['budget'].mean())
    X_train['budget'].fillna(value=X_train['budget'].mean(), inplace=True)

    # preprocessing (id)
    X_train.drop_duplicates(subset=['id'], keep='first', inplace=True)
    X_train.index = range(len(X_train))

    # preprocessing(original_language)
    leoriginal_l = LabelEncoder()
    X_train.original_language = leoriginal_l.fit_transform(X_train.original_language)

    # preprocessing(keywords)
    for i in range(0, len(X_train['keywords'])):
        X_train['keywords'][i] = json.loads(X_train['keywords'][i])

    for i in range(0, len(X_train['keywords'])):
        l = []
        for j in X_train['keywords'][i]:
            l.append(j['name'])
        X_train['keywords'][i] = l
    for i in range(0, len(X_train['keywords'])):
        X_train['keywords'][i].sort()
        s = ""
        for j in X_train['keywords'][i]:
            s += j
        s = s.replace(" ", "")
        X_train['keywords'][i] = s
    lekey = LabelEncoder()
    X_train.keywords = lekey.fit_transform(X_train.keywords)

    # preprocessing(original_title)
    for i in range(0, len(X_train['original_title'])):
        X_train['original_title'][i] = X_train['original_title'][i].replace(" ", "")
        X_train['original_title'][i] = X_train['original_title'][i].replace(string.punctuation, "")
        if not X_train['original_title'][i].isalnum():
            X_train['original_title'][i] = None
    leoriginal_t = LabelEncoder()
    X_train.original_title = leoriginal_t.fit_transform(X_train.original_title)

    # preprocessing(overview)
    leover = LabelEncoder()
    X_train.overview = leover.fit_transform(X_train.overview)

    # preprocessing(production_companies)
    for i in range(0, len(X_train['production_companies'])):
        X_train['production_companies'][i] = json.loads(X_train['production_companies'][i])

    for i in range(0, len(X_train['production_companies'])):
        l = []
        for j in X_train['production_companies'][i]:
            l.append(j['name'])
        X_train['production_companies'][i] = l
    for i in range(0, len(X_train['production_companies'])):
        X_train['production_companies'][i].sort()
        s = ""
        for j in X_train['production_companies'][i]:
            s += j
        s = s.replace(" ", "")
        X_train['production_companies'][i] = s
    lecompany = LabelEncoder()
    X_train.production_companies = lecompany.fit_transform(X_train.production_companies)

    # preprocessing(production_countries)
    for i in range(0, len(X_train['production_countries'])):
        X_train['production_countries'][i] = json.loads(X_train['production_countries'][i])
    for i in range(0, len(X_train['production_countries'])):
        l = []
        for j in X_train['production_countries'][i]:
            l.append(j['name'])
        X_train['production_countries'][i] = l
    for i in range(0, len(X_train['production_countries'])):
        X_train['production_countries'][i].sort()
        s = ""
        for j in X_train['production_countries'][i]:
            s += j
        s = s.replace(" ", "")
        X_train['production_countries'][i] = s
    lecountry = LabelEncoder()
    X_train.production_countries = lecountry.fit_transform(X_train.production_countries)

    # preprocessing(release_date)
    X_train.release_date = pd.to_datetime(X_train.release_date)
    X_train['Day_Of_Week'] = X_train.release_date.apply(lambda x: x.weekday())
    X_train['Month'] = X_train.release_date.apply(lambda x: x.month)
    X_train['Year'] = X_train.release_date.apply(lambda x: x.year)
    X_train.drop(columns=['release_date'], inplace=True)

    # preprocessing(revenue)
    X_train['revenue'] = X_train['revenue'].replace(0, X_train['revenue'].mean())
    X_train['revenue'].fillna(value=X_train['revenue'].mean(), inplace=True)

    # preprocessing(runtime)
    X_train['runtime'] = X_train['runtime'].replace(0, X_train['runtime'].mean())
    X_train['runtime'].fillna(value=X_train['runtime'].mean(), inplace=True)

    # preprocessing(spoken_languages)
    for i in range(0, len(X_train['spoken_languages'])):
        X_train['spoken_languages'][i] = json.loads(X_train['spoken_languages'][i])

    for i in range(0, len(X_train['spoken_languages'])):
        l = []
        for j in X_train['spoken_languages'][i]:
            l.append(j['iso_639_1'])
        X_train['spoken_languages'][i] = l
    for i in range(0, len(X_train['spoken_languages'])):
        X_train['spoken_languages'][i].sort()
        s = ""
        for j in X_train['spoken_languages'][i]:
            s += j
        s = s.replace(" ", "")
        X_train['spoken_languages'][i] = s
    lerun = LabelEncoder()
    X_train.spoken_languages = lerun.fit_transform(X_train.spoken_languages)

    # preprocessing(status)
    lestatus = LabelEncoder()
    X_train.status = lestatus.fit_transform(X_train.status)
    X_train.drop(columns=['status'], inplace=True)

    # preprocessing(tagline)
    letag = LabelEncoder()
    X_train.tagline = letag.fit_transform(X_train.tagline)
    X_train.drop(columns=['tagline'], inplace=True)
    # preprocessing(title)
    for i in range(0, len(X_train['title'])):
        X_train['title'][i] = X_train['title'][i].replace(" ", "")
        X_train['title'][i] = X_train['title'][i].replace(string.punctuation, "")
        if not X_train['title'][i].isalnum():
            X_train['title'][i] = None
    letitle = LabelEncoder()
    X_train.title = letitle.fit_transform(X_train.title)
    X_train.drop('title', axis='columns', inplace=True)
    return X_train

X_train = pre(X_train)

##################################
# feature_selection
data = X_train
data['vote_average'] = y_train
corr = data.corr()
top_feature = corr.index[abs(corr['vote_average']) > 0.07]
plt.subplots(figsize=(12, 8))
top_corr = data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
top_feature = top_feature.delete(-1)
X_train = data[top_feature]

# scaling
scaler = StandardScaler()
scaler1 = StandardScaler()
scaler1.fit(X_train)
X_train = pd.DataFrame(scaler1.transform(X_train),columns=X_train.columns)
y = np.array(y_train).reshape(-1, 1)
y_train = pd.DataFrame(scaler.fit_transform(y))
y_train.rename(columns={0: 'vote_average'}, inplace=True)

################################start##################################################

model =LinearRegression()
model.fit(X_train, y_train)
#################polynomial##################################
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

#####################lasso###################################
lasso = Lasso()
params = {
    'alpha': [0.01, 0.1, 1, 10]
}
g_lasso = GridSearchCV(lasso, param_grid=params, cv=5)
g_lasso.fit(X_train, y_train)

#########################ridge_model#######################
ridge_model = Ridge()
param_grid = {
    'alpha': [0.01, 0.1, 1, 10]
}
ridge_model.fit(X_train,y_train)

#######################pre_test##############################
X_test = pre(X_test)
X_test = X_test[top_feature]
X_test = pd.DataFrame(scaler1.transform(X_test),columns=X_test.columns)
y = np.array(y_test).reshape(-1, 1)
y_test = pd.DataFrame(scaler.transform(y))
y_test.rename(columns={0: 'vote_average'}, inplace=True)

###############predict#################################
X_test_poly = poly_features.transform(X_test)
y_predict4=poly_model.predict(X_test_poly)
print("error (polynomial)",metrics.mean_squared_error(y_test, y_predict4))
print("Score(polynomial)",r2_score(y_test,y_predict4))
print("------------")

width = 10
height = 5
plt.figure(figsize=(width, height))
sns.regplot(x=y_test, y=y_predict4)
plt.ylim(0,)
plt.show()

y_predict = model.predict(X_test)
print("error(linear)",metrics.mean_squared_error(y_test, y_predict))
print("score(linear)",r2_score(y_test,y_predict))
print("------------")

width = 10
height = 5
plt.figure(figsize=(width, height))
sns.regplot(x=y_test, y=y_predict)
plt.ylim(0,)
plt.show()

y_predict2 = g_lasso.best_estimator_
y_pred_lasso = y_predict2.predict(X_test)
print("error(lasso)",metrics.mean_squared_error(y_test, y_pred_lasso))
print("score(lasso)",r2_score(y_test,y_pred_lasso))
print("------------")

width = 10
height = 5
plt.figure(figsize=(width, height))
sns.regplot(x=y_test, y=y_pred_lasso)
plt.ylim(0,)
plt.show()

y_predict3 = ridge_model.predict(X_test)
print("error(ridge_model)",metrics.mean_squared_error(y_test, y_predict3))
print("score(ridge_model)",r2_score(y_test,y_predict3))
print("------------")

width = 10
height = 5
plt.figure(figsize=(width, height))
sns.regplot(x=y_test, y=y_predict3)
plt.ylim(0,)
plt.show()

########################################################

