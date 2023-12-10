from flask import request, render_template
from app import app
from app import models

# Route Sections

@app.route('/intro')
def root():
    return 'Week 4'

@app.route('/', methods = ['GET', 'POST'])
def home():
    if request.method == 'POST':
        sepal_length = request.form['s_length']
        sepal_width = request.form['s_width']
        petal_length = request.form['p_length']
        petal_width = request.form['p_width']

        y_pred = [[sepal_length, sepal_width, petal_length, petal_width]]
        model_tree = models.dec_tree()
        
        prediction = model_tree.predict(y_pred)

        setosa = 'The flower is classified as Setosa'
        versicolor = 'The flower is classified as Versicolor'
        virginica = 'The flower is classified as Virginica'

        if prediction == 0:
            return render_template('index.html', setosa = setosa)
        elif prediction == 1:
            return render_template('index.html', versicolor = versicolor)
        else:
            return render_template('index.html', virginica = virginica)

    return render_template('index.html')