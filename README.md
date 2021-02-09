# Comicolorization

This is an implementation of the [Comicolorization : Semi-automatic Manga Colorization](https://arxiv.org/abs/1706.06759).

With this repository, you can
* run sample codes
* train colorization task
* use as colorization library

## Sample Code
We prepare two sample codes, `sample_painting.py` and `sample_detecting_panels.py`.

1. `sample_painting.py` is example of colorization the manga page.
1. `sample_detecting_panels.py` is example of detection auto panel rectangle.

First of all, install the requirements.
```bash
pip install -r requirements.txt
```
In addition, you must install [OpenCV](http://docs.opencv.org/master/)-Python.

### Colorization
```bash
python sample/sample_painting.py

# Help
# python sample/sample_painting.py --help
```

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

#### 2. run
```bash
cd ../../
python sample/sample_detecting_panels.py

# Help
# python sample/sample_detecting_panels.py --help
```

### Copyright
Following images are from [Manga109 dataset](http://www.manga109.org/).

<table style="border-collapse: collapse">
<tr>
    <td><img src="./sample/Belmondo-1.png" width="96px"></td>
    <td><img src="./sample/HinagikuKenzan_026.jpg" width="96px"></td>
    <td><img src="./sample/HinagikuKenzan-1.png" width="96px"></td>
    <td><img src="./sample/TasogareTsushin-1.png" width="96px"></td>
</tr>
<tr>
    <td>©Ishioka Shoei</td>
    <td>©Sakurano Minene</td>
    <td>©Sakurano Minene</td>
    <td>©Tanaka Masato</td>
</tr>
</table>

## Training
There are two training task.

1. the colorization task for generate the low resolution colorized image.
2. the super resolution task for generate the higher resolution colorized image.

First of all, install the requirements.
```bash
pip install -r requirements.txt
```
In addition, you must install [OpenCV](http://docs.opencv.org/master/)-Python.

### Colorization Task

#### 1. prepare dataset
For training, three type data are required.

* color images that [Pillow](https://pillow.readthedocs.io/) can load
* json file written label ID list that has 'id' key
    * ex. ```{
        "id": [
            "a",
            "b",
            "c"
        ]
    }```
* json file written label ID list for each image
    * the key is image file name without extension
    * ex. ```{
        "imageX": ["a", "b"],
        "imageY": ["a"],
        "imageZ": ["c"],
        ...
    }```

#### 2. run
```bash
# run same as paper
python bin/train.py \
    /path/to/images/directory \
    /path/to/save \
    --path_tag_list /path/to/label_ID_list.json \
    --path_tag_list_each_image /path/to/label_ID_list_for_each_image.json \
    --network_model LTBC \
    --num_dataset_test 1000 \
    --batchsize 30 \
    --size_image 224 \
    --augmentation True \
    --size_image_augmentation 256 \
    --save_result_iteration 1000 \
    --random_seed_test 0 \
    --loss_type Lab \
    --alpha_ltbc_classification 0.00333333333 \
    --ltbc_classification_loss_function multi_label \
    --line_drawing_mode otsu_threshold \
    --max_pixel_drawing 15 \
    --use_adversarial_network \
    --blend_adversarial_generator 1.0 \
    --discriminator_first_pooling_size 2 \
    --optimizer_adam_alpha 0.0001 \
    --blend_mse_color 1.0 \
    --mse_loss_mode color_space \
    --weight_decay 0.0001 \
    --log_interval 200 \
    --gpu -1 \
    --ltbc_classification_num_output_list 512 428 \  # last number should be same as number of labels
    {color_feature}  # other params depending on task, please see below

# Help
# python bin/train.py --help
```

When palette mode, change `color_feature` to
```bash
    --threshold_histogram_palette 0.0 \
    --use_histogram_network \
    --num_bins_histogram 6 \
    --use_multidimensional_histogram
```

When histogram mode, change `color_feature` to
```bash
    --use_histogram_network \
    --num_bins_histogram 6 \
    --use_multidimensional_histogram
```

When without any color feature, `color_feature` is empty.

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

## Use as colorization library
### Install
```bash
pip install git+http://github.com/DwangoMediaVillage/Comicolorization
```

### Download trained model
Please download [model files](https://s3-us-west-1.amazonaws.com/nico-opendata-distribution/550000.model) at `model` directory in this repository.

### How to use
1. import `comicolorization` and `comicolorization_sr`
1. initialize `Drawer`
1. load model
1. initialize `PagePipeline`
1. call `process` method

Please see the [sample code](./sample/sample_painting.py) for more details.

## License
MIT License, see [LICENSE](./LICENSE).
