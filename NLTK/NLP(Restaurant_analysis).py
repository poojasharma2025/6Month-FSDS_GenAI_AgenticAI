# resume project-sentiment analysis(change name using gpt)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\WELCOME\Documents\Data Science\September 2025\23rd sept-NLTK completed\4.CUSTOMERS REVIEW DATASET\Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

# cleaning the text before going advance to the ML steps
import re    # regular expression
import nltk     # library
from nltk.corpus import stopwords    # we have multiple stop words in english library
from nltk.stem.porter import PorterStemmer   # keeps just for root form of an word
 
corpus = []

for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english')) ]
    review = ' '.join(review)
    corpus.append(review)
# to convert text to vector(numbers) ,we use countVectorization algorithm and bag of word model(BOW)
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(corpus).toarray()  # x is independent variable
y = dataset.iloc[:,1].values

# Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20, random_state=0)

from sklearn.tree import DecisionTreeClassifier  # build all model and check the accuracy of all model and compare
classifier = DecisionTreeClassifier()
classifier.fit(x_train,y_train)

# Predicting the test set results
y_pred = classifier.predict(x_test)

# making the confusion metrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred) 
print(cm)

# in NLP Its a hard to get accuracy above 80
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print("accuracy is:",ac)

bias = classifier.score(x_train,y_train)
print("bias is:",bias)

variance = classifier.score(x_test,y_test)

print("variance is:",variance)

# High bias 0.99 and low variance 0.66 goes to underfitting model, if model underfit, we need to add more attribute
# after applying all model and algo , even though we didnt get better accuracy above 80, then we can duplicate the dataset 2-3 times

#===============================================
'''
CASE STUDY --> model is underfitted  & we got less accuracy 

1> Implementation of tfidf vectorization , lets check bias, variance, ac, auc, roc 
2> Impletemation of all classification algorihtm (logistic, knn, randomforest, decission tree, svm, xgboost,lgbm,nb) with bow & tfidf 
4> You can also reduce or increase test sample 
5> xgboost & lgbm as well
6> you can also try the model with stopword 


6> then please add more recores to train the data more records 
7> ac ,bias, varian - need to equal scale ( no overfit & not underfitt)
