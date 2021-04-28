# -*- coding: utf-8 -*-
from flask import Flask,render_template,url_for,request, Markup
import pickle
import pandas as pd
import preprocessing
import os
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage

app = Flask(__name__, template_folder='templates')
app.config ['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config ['UPLOAD_FOLDER'] = "G:\\NEWW\\SentimentAnalysis"
db = SQLAlchemy(app)


class database(db.Model):
    id=db.Column('user_id',db.Integer, primary_key=True)
    positive=db.Column(db.Integer)
    negative=db.Column(db.Integer)
    neutral=db.Column(db.Integer)

    def __init__(self, positive,negative, neutral):
        self.positive=positive
        self.negative=negative
        self.neutral=neutral

# load the model from disk
clf = pickle.load(open('nb_clf.pkl', 'rb'))
cv=pickle.load(open('tfidf_model.pkl','rb'))

total_negative=0
total_positive=0
total_neutral=0
values=[total_positive,total_negative,total_neutral]
labels=["POSITIVE","NEGATIVE","NEUTRAL"]
colors=["#F7464A", "#46BFBD", "#FDB45C"]

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['upload']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename)))
        review1=request.form['reviews']
        df=pd.read_csv(f.filename, usecols= [review1])
        count_positive=0
        count_negative=0
        count_neutral=0
        for i in range(len(df)) :
            one_review=df.loc[i, review1]
            #print(one_review)
            final_review=[one_review]
            data = preprocessing.text_Preprocessing(final_review)
            vect = cv.transform(data)
            my_prediction = clf.predict(vect)
            if (my_prediction[0]==1):
                count_positive=count_positive+1
            elif(my_prediction[0]==0):
                count_negative=count_negative+1
            else:
                count_neutral=count_neutral+1
        total_positive=(count_positive/len(df))*100
        total_negative=(count_negative/len(df))*100
        total_neutral=(count_neutral/len(df))*100
        data=database(total_positive,total_negative,total_neutral)
        db.session.add(data)
        db.session.commit()


    return render_template('home.html')

@app.route('/admin')
def admin():
    return render_template('testhome.html',database=database.query.first())
# @app.route('/analysis')
# def analysis():
#     pie_labels = labels
#     pie_values = values
#     return render_template('analysis.html', title='PIE CHART ANALYSIS', max=17000, set=zip(values, labels, colors))

if __name__ == '__main__':
    db.create_all()
    db.session.commit()
    app.run(debug=True)
