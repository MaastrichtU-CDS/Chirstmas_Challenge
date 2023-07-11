from flask import Flask
import os
from pathlib import Path

if not os.path.exists('upload'):
    os.mkdir('upload')
    
if not os.path.exists('download'):
    os.mkdir('download')

UPLOAD_FOLDER = Path(os.path.join(os.getcwd(),'upload'))
DOWNLOAD_FOLDER = Path(os.path.join(os.getcwd(),'download'))

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['DOWNLOAD_FOLDER'] = str(DOWNLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024