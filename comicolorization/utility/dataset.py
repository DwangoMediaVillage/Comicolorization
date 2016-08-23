import typing

import comicolorization


def choose_dataset(
        paths: typing.Iterable,
        num_dataset_test: int,
        loss_type: str,
        augmentation: bool,
        size_image_augmentation: typing.List[int],
        size_image: typing.List[int],
        use_ltbc_classification: bool,
        path_tag_list: str,
        path_tag_list_each_image: str,
        line_drawing_mode: str,
        max_pixel_drawing: int,
        max_size_pixel_drawing: int,
        use_binarization_dataset: bool,
):
    if augmentation:
        resize = size_image_augmentation
        crop_size = size_image
        random_flip = True
    else:
        resize = size_image
        crop_size = None
        random_flip = False

    train_paths = paths[:-num_dataset_test]
    test_paths = paths[-num_dataset_test:]
    train_for_evaluate_paths = paths[:num_dataset_test]

    if use_binarization_dataset:
        base_dataset_class = comicolorization.dataset.BinarizationImageDataset
    elif line_drawing_mode is not None:
        if line_drawing_mode == 'otsu_threshold':
            base_dataset_class = comicolorization.dataset.LabOtsuThresholdImageDataset
        elif line_drawing_mode == 'adaptive_threshold':
            base_dataset_class = comicolorization.dataset.LabAdaptiveThresholdImageDataset
        elif line_drawing_mode == 'canny':
            base_dataset_class = comicolorization.dataset.LabCannyImageDataset
        elif line_drawing_mode == 'three_value_threshold':
            base_dataset_class = comicolorization.dataset.LabThreeValueThresholdImageDataset
        elif line_drawing_mode == 'dilate-diff':
            base_dataset_class = comicolorization.dataset.LabDilateDiffImageDataset
        else:
            raise NotImplementedError
        assert loss_type == "Lab"
    elif loss_type == 'RGB':
        base_dataset_class = comicolorization.dataset.ColorMonoImageDataset
    elif loss_type == 'Lab':
        base_dataset_class = comicolorization.dataset.LabImageDataset
    elif loss_type == 'ab':
        base_dataset_class = comicolorization.dataset.LabOnlyChromaticityDataset
    else:
        raise NotImplementedError

    train_dataset = base_dataset_class(
        train_paths,
        resize=resize,
        random_crop_size=crop_size,
        random_flip=random_flip,
        test=False
    )
    test_dataset = base_dataset_class(
        test_paths,
        resize=resize,
        random_crop_size=crop_size,
        test=True
    )
    train_for_evaluate_dataset = base_dataset_class(
        train_for_evaluate_paths,
        resize=resize,
        random_crop_size=crop_size,
        test=True
    )

    if max_pixel_drawing is not None:
        def _make_dataset(base_lineimage_dataset, fix_position):
            return comicolorization.dataset.LabSeveralPixelDrawingImageDataset(
                base_lineimage_dataset=base_lineimage_dataset,
                max_point=max_pixel_drawing,
                max_size=max_size_pixel_drawing,
                fix_position=fix_position,
            )

        train_dataset = _make_dataset(train_dataset, fix_position=False)
        test_dataset = _make_dataset(test_dataset, fix_position=True)
        train_for_evaluate_dataset = _make_dataset(train_for_evaluate_dataset, fix_position=True)

    if not use_ltbc_classification:
        pass
    elif path_tag_list is None:
        def _make_dataset(paths, base_dataset):
            return comicolorization.dataset.LabeledByDirectoryDataset(
                paths,
                base_dataset=base_dataset,
            )

        train_dataset = _make_dataset(train_paths, train_dataset)
        test_dataset = _make_dataset(test_paths, test_dataset)
        train_for_evaluate_dataset = _make_dataset(train_for_evaluate_paths, train_for_evaluate_dataset)
    else:
        def _make_dataset(paths, base_dataset):
            return comicolorization.dataset.MultiTagLabeledDataset(
                paths,
                base_dataset=base_dataset,
                path_tag_list=path_tag_list,
                path_tag_list_each_image=path_tag_list_each_image,
            )

        train_dataset = _make_dataset(train_paths, train_dataset)
        test_dataset = _make_dataset(test_paths, test_dataset)
        train_for_evaluate_dataset = _make_dataset(train_for_evaluate_paths, train_for_evaluate_dataset)

    return {
        'train': train_dataset,
        'test': test_dataset,
        'train_for_evaluate': train_for_evaluate_dataset,
    }
