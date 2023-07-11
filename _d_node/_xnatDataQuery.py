import re
import os
import xnat
import zipfile
from pathlib import Path
import shutil

class _xnatDataQuery():
    def __init__(self):
        self.size = "Hello"

    def extract_data(self,temp_path, path, ext_path, res_id):
        file = zipfile.ZipFile(path)
        for filename in file.namelist():
            file.extract(filename, temp_path)
            file.close()
        png = os.path.join(Path(temp_path), Path(filename))
        shutil.move(png, Path(ext_path))

        for paths, dirs, files in os.walk(ext_path):
            for dir in dirs:
                shutil.rmtree(os.path.join(ext_path, dir))
            for filenames in files:
                fname, fext = os.path.splitext(filenames)
                if fext in ['.zip']:
                    os.remove(Path(os.path.join(ext_path, filenames)))

    def downloadXNATdata(self,train_dir,test_dir):
        train_dir=str(train_dir)
        test_dir = str(test_dir)
        database_uri = "http://ec2-18-221-87-238.us-east-2.compute.amazonaws.com:8081"
        #database_uri = os.getenv('DATABASE_URI', 'no_database_uri')  #read database_uri
        session = xnat.connect(database_uri, user="admin",password="admin")
        myProjects = session.projects

        for project in myProjects:
            test = re.search("^dnode_.*_png", project)
            if test:
                project_id = project
                # val = re.search(r'[0-9]', project_id).group(0)
                # with open(input.txt,'w') as fp:
                #     fp.write(str(val))
                #     fp.close()
            else:
                project_id = "dnode_png"

        currentDirectory = os.getcwd()
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
            os.chdir(test_dir)
            os.mkdir('0')
            os.mkdir('1')
            os.chdir(currentDirectory)
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
            os.chdir(train_dir)
            os.mkdir('0')
            os.mkdir('1')
            os.chdir(currentDirectory)

        myProject = session.projects[project_id]
        mySubjectList = myProject.subjects.values()

        for entries in mySubjectList:
            mySubjectID = entries.label
            mySubject = myProject.subjects[mySubjectID]
            myExperiments = mySubject.experiments.values()

            for experiments in myExperiments:
                myExperimentID = experiments.label
                myExperiment = mySubject.experiments[myExperimentID]
                myResources = myExperiment.resources.values()

                train_0 = re.search("^0.*TRAIN$", myExperimentID)
                train_1 = re.search("^1.*TRAIN$", myExperimentID)
                test_0 = re.search("^0.*TEST$", myExperimentID)
                test_1 = re.search("^1.*TEST$", myExperimentID)

                if (train_0):
                    #  print('Downloaded'+myExperimentID+'In /TRAIN/0 folder')
                    for resources in myResources:
                        myResourceID = resources.label
                        myResource = myExperiment.resources[myResourceID]
                        myResource.download(currentDirectory + '/TRAIN/0/' + myResourceID + '.zip')
                        temp_dir = str(currentDirectory + '/TRAIN/0/' + myResourceID)
                        zip_dir = currentDirectory + '/TRAIN/0/' + myResourceID + '.zip'
                        ext_dir = currentDirectory + '/TRAIN/0/'
                        self.extract_data(temp_dir, zip_dir, ext_dir, myResourceID)
                elif (train_1):
                    #  print('Downloaded'+myExperimentID+'In /TRAIN/1 folder')
                    for resources in myResources:
                        myResourceID = resources.label
                        myResource = myExperiment.resources[myResourceID]
                        myResource.download(currentDirectory + '/TRAIN/1/' + myResourceID + '.zip')
                        temp_dir = str(currentDirectory + '/TRAIN/1/' + myResourceID)
                        zip_dir = currentDirectory + '/TRAIN/1/' + myResourceID + '.zip'
                        ext_dir = currentDirectory + '/TRAIN/1/'
                        self.extract_data(temp_dir, zip_dir, ext_dir, myResourceID)
                elif (test_0):
                    #  print('Downloaded'+myExperimentID+'In /TEST/0 folder')
                    for resources in myResources:
                        myResourceID = resources.label
                        myResource = myExperiment.resources[myResourceID]
                        myResource.download(currentDirectory + '/TEST/0/' + myResourceID + '.zip')
                        temp_dir = str(currentDirectory + '/TEST/0/' + myResourceID)
                        zip_dir = currentDirectory + '/TEST/0/' + myResourceID + '.zip'
                        ext_dir = currentDirectory + '/TEST/0/'
                        self.extract_data(temp_dir, zip_dir, ext_dir, myResourceID)
                elif (test_1):
                    #  print('Downloaded'+myExperimentID+'In /TRAIN/1 folder')
                    for resources in myResources:
                        myResourceID = resources.label
                        myResource = myExperiment.resources[myResourceID]
                        myResource.download(currentDirectory + '/TEST/1/' + myResourceID + '.zip')
                        temp_dir = str(currentDirectory + '/TEST/1/' + myResourceID)
                        zip_dir = currentDirectory + '/TEST/1/' + myResourceID + '.zip'
                        ext_dir = currentDirectory + '/TEST/1/'
                        self.extract_data(temp_dir, zip_dir, ext_dir, myResourceID)
                else:
                    print("Something Went Wrong")