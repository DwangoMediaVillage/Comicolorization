# Comicolorization

## Run Sample Code

### Colorization
```bash
python sample/sample_painting.py

# Help
# python sample/sample_painting.py --help
```

* optional arguments
* `--input_image`: path of input page image
* `--panel_rectangle`: path of json file written panel rectangle
* `--reference_images`: paths of reference images
* `--comicolorizatoin_model_directory`: the trained model directory for the comicolorization task.
* `--comicolorizatoin_model_iteration`: the trained model iteration for the comicolorization task.
* `--super_resolution_model_directory`: the trained model directory for the super resolution task.
* `--super_resolution_model_iteration`: the trained model iteration for the super resolution task.
* `--gpu`: gpu number (-1 means the cpu mode).
* `--output`: the path of colorized image.

### Auto Panel Rectangle Detection
#### 1. get `manga-frame-extension` and build
```bash
git submodule init
git submodule update
cd manga-frame-extraction/MangaFrameExtraction
cmake ./
make
```

Please read [manga-frame-extraction's README.md](http://github.com/DwangoMediaVillage/manga_frame_extraction/blob/master/README.md) for details.

### 2. run
```bash
cd ../../
python sample/sample_detecting_panels.py

# Help
# python sample/sample_detecting_panels.py --help
```

* optional arguments
* `--input`: the path of input page image.
* `--mfe`: the path of manga-frame-extraction's binary file.
* `--output`: the path of output panel rectangle information.

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

## Copyright
* We cite monochrome mangas from [Manga109 dataset](http://www.manga109.org/).
* <img src="./sample/Belmondo-1.png" width="96px"> ©Ishioka Shoei
* <img src="./sample/HinagikuKenzan_026.jpg" width="96px"> ©Sakurano Minene
* <img src="./sample/HinagikuKenzan-1.png" width="96px"> ©Sakurano Minene
* <img src="./sample/TasogareTsushin-1.png" width="96px"> ©Tanaka Masato
