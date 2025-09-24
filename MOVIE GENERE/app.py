# Importing essential libraries
from flask import Flask, render_template, request
import pickle

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename = r'C:\Users\A3MAX SOFTWARE TECH\A VS CODE\11. CAPSTONE PROJECT_DEPLOYMENT\MOVIE GENERE\movie-genre-mnb-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open(r'C:\Users\A3MAX SOFTWARE TECH\A VS CODE\11. CAPSTONE PROJECT_DEPLOYMENT\MOVIE GENERE\cv-transform.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
     message = request.form['message']
     data = [message]
     vect = cv.transform(data).toarray()
     my_prediction = classifier.predict(vect)
    return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)