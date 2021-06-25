# -*- coding: utf-8 -*-
from flask import Flask,render_template,url_for,request, Markup, send_file, make_response
import pickle
import seaborn as sns
import pandas as pd
import preprocessing
import os
import io
import nltk
import base64
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
from io import BytesIO
import matplotlib.pyplot as plt
#from plot import do_plot
sid = SentimentIntensityAnalyzer() 
nltk.download('vader_lexicon')
app = Flask(__name__)
app.config ['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_BINDS'] = {'two' : 'sqlite:///site2.db'}
app.config ['UPLOAD_FOLDER'] = "C:\\Users\\Shashank Jain\\Desktop\\mlproject\\project"
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


class two(db.Model):
    __bind_key__ = 'two'
    id = db.Column(db.Integer, primary_key=True)
    positivity=db.Column(db.Integer)
    negativity=db.Column(db.Integer)
    review=db.Column(db.String(1000))

    def __init__(self, positivity,negativity,review):
        self.positivity=positivity
        self.negativity=negativity
        self.review=review


# load the model from disk
clf = pickle.load(open('nb_clf.pkl', 'rb'))
cv=pickle.load(open('tfidf_model.pkl','rb'))

total_negative=0
total_positive=0
total_neutral=0
@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['upload']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename)))
        review1=request.form['reviews']
        ratings=request.form['rating']
        
        df=pd.read_csv(f.filename, usecols= [review1,ratings])
        df["sentiments"] = df[review1].apply(lambda x: sid.polarity_scores(x))
        df = pd.concat([df.drop(['sentiments'], axis=1), df['sentiments'].apply(pd.Series)], axis=1)
        for i in range(len(df)) :
            one_review=df.loc[i, review1]
            final_review=[one_review]
            
            data = preprocessing.text_Preprocessing(final_review)
            string_data=data[0]
            positive=(sid.polarity_scores(string_data)).get('pos')
           
            negative=(sid.polarity_scores(string_data)).get('neg')
            print(positive)
            data1=two(positive,negative,one_review)
            db.session.add(data1)
            db.session.commit()
        #FOR DRAWING SENTIMENT DISTRIBUTION GRAPH
        df[ratings] = df[ratings].apply(lambda x: 1 if x < 3 else 0)
        
        for x in [0, 1]:
            subset = df[df[ratings] == x]
        # Draw the density plot
            if x == 0:
                label = "Good reviews"
            else:
                label = "Bad reviews"
            sns.distplot(subset['compound'], hist = False, label = label)

        #saving plot into bytes object
        img = BytesIO()
        plt.legend()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        #Counting total negative total positive total neutral and marking polarity scores with every review
        count_positive=0
        count_negative=0
        count_neutral=0
        for i in range(len(df)) :
            one_review=df.loc[i, review1]
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


    return render_template('testhome.html',two=two.query.order_by(two.positivity.desc()).limit(10).all(),database=database.query.first(),three=two.query.order_by(two.negativity.desc()).limit(10).all(),
    plot_url=plot_url)

# @app.route('/plots', methods=['GET'])
# def correlation_matrix():
#     bytes_obj = do_plot()
    
#     return send_file(bytes_obj,
#                      attachment_filename='plot.png',
#                      mimetype='image/png')
@app.route('/admin')
def admin():
    # f = request.files['upload']
    # csv_name=f.filename
    # print(csv_name)
    # df=pd.read_csv(csv_name, usecols= [review1,ratings])
    # for i in range(len(df)) :
    #     one_review=df.loc[i, review1]
    #     final_review=[one_review]
    #     data = preprocessing.text_Preprocessing(final_review)
    #     string_data=data[0]
    #     positive=(sid.polarity_scores(string_data)).get('pos')
    #     print(positive)
    #     negative=(sid.polarity_scores(string_data)).get('neg')
    #     data=database2(positive,negative,one_review)
    #     db.session.add(data)
    #     db.session.commit()
    
    return render_template('testhome.html',two=two.query.order_by(two.positivity.desc()).limit(10).all(),database=database.query.first(),three=two.query.order_by(two.negativity.desc()).limit(10).all())
# @app.route('/analysis')
# def analysis():
#     pie_labels = labels
#     pie_values = values
#     return render_template('analysis.html', title='PIE CHART ANALYSIS', max=17000, set=zip(values, labels, colors))

if __name__ == '__main__':
    db.create_all()
    db.create_all(bind=['two'])
    app.run(debug=True)
