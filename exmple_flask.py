from flask import Flask, request, render_template

app = Flask(__name__)

'''@app.route('/')
def my_form():
    return render_template('my-form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    processed_text = text.upper()
    return text'''
@app.route('/')
def hello():
    ...
    return '', 204

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5017)