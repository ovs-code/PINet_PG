from flask import Flask, url_for, render_template, request, session, redirect, flash
from markupsafe import escape
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# SEt the secrete key to some random bytes. Keep this really secret!
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# Set the upload folder for the user images
UPLOAD_FOLDER = 'static/imgs'

# Set the allowed extensions (i.e. only images)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# check for allowed extensions only
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files or not request.files['image']:
            flash('No image selected')
            return redirect(request.url)
        image = request.files['image']
        if not allowed_file(image.filename):
            flash(f'Image Type has to be one of {ALLOWED_EXTENSIONS}')
            return redirect(request.url)
        filename = secure_filename(image.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(file_path)
        return render_template('index.html', image=file_path) 
    elif request.method == 'GET':
        return render_template('index.html')
    else:
        return f'error {request.method} method not implemented'

@app.route('/impressum')
def impressum():
    return 'This is the impressum'

with app.test_request_context():
    print(url_for('index'))
    print(url_for('impressum'))