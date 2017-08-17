# Comicolorization

## Training

### Colorization Task

### Super Resolution Task
#### 1. modify config file
Modify the information with these keys in `bin/config_super_resolution.json`.
* `dataset/images_glob`: set the image file names
* `model/other/path_result_directory`: set the colorization model directory
* `project/name`: set the name of this task (this will be the directory name)
* `project/result_path`: set the path for the result directory

#### 2. run
```bash
python bin/train_super_resolution.py bin/config_super_resolution.json
```
