#main
#env: ARCGIS_ENABLE_TF_BACKEND=1
import os
from pathlib import Path
#from arcgis.mapping import WebMap
from arcgis.gis import GIS 
from arcgis.learn import prepare_data, FeatureClassifier


gis = GIS()

training_data = gis.content.get('81932a51f77b4d2d964218a7c5a4af17')
training_data
filepath = training_data.download(file_name=training_data.name)

import zipfile
with zipfile.ZipFile(filepath, 'r') as zip_ref:
    zip_ref.extractall(Path(filepath).parent)

data_path = Path(os.path.join(os.path.splitext(filepath)[0]))

from glob import glob
from PIL import Image
for image_filepath in glob(os.path.join(data_path, 'images', '**','*.jpg')):
    if Image.open(image_filepath).mode != 'RGB':
        os.remove(image_filepath)


data = prepare_data(
    path=data_path,
    dataset_type='Imagenet',
    batch_size=64,
    chip_size=300
)
print("data prepared")
