import typing

import comicolorization


def choose_dataset(
        paths,
        num_dataset_test,
        loss_type,
        augmentation,
        size_image_augmentation,
        size_image,
        use_ltbc_classification,
        path_tag_list,
        path_tag_list_each_image,
        line_drawing_mode,
        max_pixel_drawing,
        max_size_pixel_drawing,
        use_binarization_dataset,
):
    # type: (typing.Iterable, int, str, bool, typing.List[int], typing.List[int], bool, str, str, str, int, int, bool) -> any
    if augmentation:
        resize = size_image_augmentation
        crop_size = size_image
        random_flip = True
    else:
        resize = size_image
        crop_size = None
        random_flip = False

    def _make_dataset(paths, test):
        # type: (any, bool) -> any
        dataset = comicolorization.dataset.PILImageDataset(
            paths,
            resize=resize,
            random_crop_size=crop_size,
            random_flip=random_flip & test,
            test=test,
        )
        dataset = comicolorization.dataset.ColorMonoImageDataset(dataset)

        # color space of input image
        if loss_type == 'RGB':
            pass
        elif loss_type == 'Lab' or loss_type == 'ab' or use_binarization_dataset:
            dataset = comicolorization.dataset.LabImageDataset(dataset)

            if use_binarization_dataset:
                dataset = comicolorization.dataset.BinarizationImageDataset(dataset)
            elif loss_type == 'ab':
                dataset = comicolorization.dataset.LabOnlyChromaticityDataset(dataset)
        else:
            raise ValueError(loss_type)

        # make line-drawing
        if line_drawing_mode is not None:
            if line_drawing_mode == 'otsu_threshold':
                dataset = comicolorization.dataset.LabOtsuThresholdImageDataset(dataset)
            elif line_drawing_mode == 'adaptive_threshold':
                dataset = comicolorization.dataset.LabAdaptiveThresholdImageDataset(dataset)
            elif line_drawing_mode == 'canny':
                dataset = comicolorization.dataset.LabCannyImageDataset(dataset)
            elif line_drawing_mode == 'three_value_threshold':
                dataset = comicolorization.dataset.LabThreeValueThresholdImageDataset(dataset)
            elif line_drawing_mode == 'dilate-diff':
                dataset = comicolorization.dataset.LabDilateDiffImageDataset(dataset)
            else:
                raise ValueError(line_drawing_mode)

        # colorize part
        if max_pixel_drawing is not None:
            dataset = comicolorization.dataset.LabSeveralPixelDrawingImageDataset(
                base=dataset,
                max_point=max_pixel_drawing,
                max_size=max_size_pixel_drawing,
                fix_position=test,
            )

        # classification
        if not use_ltbc_classification:
            pass
        elif path_tag_list is None:
            dataset = comicolorization.dataset.LabeledByDirectoryDataset(
                paths,
                base_dataset=dataset,
            )
        else:
            dataset = comicolorization.dataset.MultiTagLabeledDataset(
                paths,
                base_dataset=dataset,
                path_tag_list=path_tag_list,
                path_tag_list_each_image=path_tag_list_each_image,
            )

        return dataset

    train_paths = paths[:-num_dataset_test]
    test_paths = paths[-num_dataset_test:]
    train_for_evaluate_paths = paths[:num_dataset_test]

    return {
        'train': _make_dataset(train_paths, test=False),
        'test': _make_dataset(test_paths, test=True),
        'train_for_evaluate': _make_dataset(train_for_evaluate_paths, test=True),
    }
