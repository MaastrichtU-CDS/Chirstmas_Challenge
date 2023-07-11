import os
import urllib.request
from app import app
from flask import Flask, request, redirect, jsonify, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from flask import send_file, send_from_directory, safe_join, abort
from pathlib import Path
from compute_avg import compute_avg
import time

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'h5'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_file(file1, file2, iteration):
    path = app.config['DOWNLOAD_FOLDER']
    filename = str(iteration) + 'iteration.h5'
    if os.path.exists(os.path.join(path,filename)):
        return filename
    else:
        if os.path.exists(file1) and os.path.exists(file2):
            print("Both File Available")
            filename = compute_avg(file1, file2, iteration)
            if filename == False:
                return False
            else:
                return filename
        else:
            return 'File Not available....waiting'

@app.route('/file-upload', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    '''if 'file' not in request.files:
        resp = jsonify({'message': 'No file part in the request'})
        resp.status_code = 400
        return resp'''
    iteration = request.args.get('iteration')
    id = request.args.get('node_id')
    print("iteration")
    file = request.files['files']
    if file.filename == '':
        resp = jsonify({'message': 'No file selected for uploading'})
        resp.status_code = 400
        return resp
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print(filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        resp = jsonify({'message': 'Upload Complete'})
        return resp
    else:
        resp = jsonify({'message': 'Allowed file types are txt, pdf, png, jpg, jpeg, gif, h5'})
        resp.status_code = 400
        return resp

@app.route('/file_download', methods=['GET'])
def download_file():
    iteration = request.args.get('iteration')
    file1 = Path(os.path.join(app.config['UPLOAD_FOLDER'], '1' + '_' + str(iteration) + 'iteration.h5'))
    print(file1)
    file2 = Path(os.path.join(app.config['UPLOAD_FOLDER'], '2' + '_' + str(iteration) + 'iteration.h5'))
    filename = process_file(file1, file2, iteration)
    if not filename:
        resp = jsonify({'message': 'Fuck the shit'})
        resp.status_code = 400
        return resp
    else:
        return redirect(url_for('uploaded_file', filename=filename))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug = False, threaded = False) # change this arg could solve the version conflict of falsk and keras new version.