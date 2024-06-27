import os
import PIL
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from model import predict_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        predicted_class, confidence = predict_image(filepath)
        return render_template('result.html', filename=filename, prediction=predicted_class, confidence="{:.2f}".format(confidence*100))
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
