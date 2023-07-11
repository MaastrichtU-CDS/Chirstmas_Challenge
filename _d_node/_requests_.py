import requests
import time
from pathlib import Path
import os
import shutil
import h5py
import numpy as np
from io import BytesIO
class _requests_():
	def _download_file(iteration,url_download,save_directory):
		filename = os.path.join(save_directory,str(iteration) + "iteration.h5")
		print("Downloading File! Waiting......")
		while True:
			print("Why its stucks")
			if not os.path.exists(filename):
				file = requests.get(url=url_download, params={'iteration': iteration})
				if file.status_code == 200:
					raw_content=file.content
					hf=h5py.File(filename,'w')
					npdata=np.array(raw_content)
					dset=hf.create_dataset(filename,data=npdata)
					break
				else:
					print("File not available.... waiting")
					time.sleep(30)
					continue
			else:
				break
			return filename


	def upload_file(files,iteration,url_upload,node_id):
		params = {'iteration':iteration,'node_id':node_id}
		'''headers = {
			'Transfer-encoding': 'chunked',
			'Cache-Control': 'no-cache',
			'Connection': 'Keep-Alive',
			# 'User-Agent': 'ExpressionEncoder'
		}'''
		with open(files, 'rb') as f:
			r = requests.post(url_upload, files={'files': f}, params=params)
			while r.status_code != 200:
				time.sleep(2)

