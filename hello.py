from flask import Flask, url_for, render_template, request, session, redirect, flash
from markupsafe import escape
from werkzeug.utils import secure_filename
import os
import sys
import torch

from PIL import Image

from pipeline import InferencePipeline
from options.infer_options import InferOptions
from tool import cords_to_map, get_coords, reorder_pose, load_pose_from_file

# necessary variable for using tf and torch simulatuosly
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# read inference options and create new pipeline
opt = InferOptions().parse(['--name', 'fashion_PInet_PG', '--pose_estimator', 'assets/pretrains/pose_estimator.h5'])
pipeline = InferencePipeline.from_opts(opt=opt)

# set the directory for the templates
template_dir = os.path.abspath('./webapp/templates')
static_dir = os.path.abspath('./webapp/static')
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Set the secrete key to some random bytes. Keep this really secret!
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# Set the upload folder for the user images
UPLOAD_FOLDER = 'webapp/static/imgs'

# for testing purpose only set the path to the target poses
TARGET_POSE_PATH = 'test_data/testK/randomphoto_small.jpg.npy'
# directly load the target pose
target_pose = load_pose_from_file(TARGET_POSE_PATH)

# Set the allowed extensions (i.e. only images)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# check for allowed extensions only
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print('POST request received')
        if 'image' not in request.files or not request.files['image']:
            flash('No image selected')
            return redirect(request.url)
        image = request.files['image']
        if not allowed_file(image.filename):
            flash(f'Image Type has to be one of {ALLOWED_EXTENSIONS}')
            return redirect(request.url)
        
        # get the input pose id from the form data
        # inputPose = request.form['inputPose']
        # do remapping of the inputPose

        source_image = Image.open(image).convert(mode='RGB')
        print(source_image.size, source_image.mode)
        
        # generate the output image
        output_image = pipeline(image=source_image, target_pose_map=target_pose)

        # filename = secure_filename(output_image.filename)
        # file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # output_image.save(file_path)
        return render_template('index.html', output_image=output_image) 
    elif request.method == 'GET':
        print('GET request received')
        return render_template('index.html')
    else:
        return f'error {request.method} method not implemented'

@app.route('/impressum')
def impressum():
    return 'This is the impressum'

with app.test_request_context():
    print(url_for('index'))
    print(url_for('impressum'))