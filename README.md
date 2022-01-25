# Pipeline to extract text from DFS Files
This repository is the result of "Uso de Aprendizaje de Máquina para el análisis de los ficheros de la Dirección Federal de Seguridad (DFS)".

# Examples

[Example of all pipeline execute in files](notebooks/detecting_dfs_files.ipynb)

# Installation
The install.sh file will download yolov5 especific version used to train file_dfs_detector.  

Execute install.sh: chmod +x install.sh & ./install.sh  
Install requeriments: pip install -r requirements.txt  

# Structure 

| Module  | Description |
| ------------- | ------------- |
| ocr_dfs.files_dfs_pipeline.py  | This module contains functions and classes developed to implement all pipeline suggested for the authors to detect and extract dfs files information.   |
| ocr_dfs.files_dfs_detector.py  | This module contains functions and classes developed to detect orientation, dfs files into images.   |
| ocr_dfs.text_dfs_analyzer.py  | This module contains functions and classes developed to implement ocr text quality.   |
| ocr_dfs.clean_image_pipeline.py  | This module contains functions and classes developed to pre-process dfs files images.  |