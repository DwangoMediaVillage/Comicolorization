from chainer.dataset import dataset_mixin
import json
import os
import six
import numpy


class LabeledByDirectoryDataset(dataset_mixin.DatasetMixin):
    @staticmethod
    def get_directory_name(path):
        return os.path.basename(os.path.dirname(path))

    def __init__(self, paths, base_dataset):
        """
        ex. paths = root/a/1.png root/b/2.png root/c/3.png -> labeled a,b,c
        """
        if isinstance(paths, six.string_types):
            with open(paths) as paths_file:
                paths = [path.strip() for path in paths_file]
        self._paths = paths

        self.class_name_list = list(set(self.get_directory_name(path) for path in paths))

        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def get_example(self, i):
        """
        :return: (base_dataset_class's data, label)
        """
        class_name = self.get_directory_name(self._paths[i])
        label = numpy.array(self.class_name_list.index(class_name), numpy.int32)
        return self.base_dataset[i] + (label,)

    def get_input_range(self):
        return self.base_dataset.get_input_range()

    def get_input_luminance_range(self):
        return self.base_dataset.get_input_luminance_range()

    def get_output_range(self):
        return self.base_dataset.get_output_range()


class MultiTagLabeledDataset(dataset_mixin.DatasetMixin):
    """
    path_tag_list's json is like : {
        "id": [
            12492,
            7576,
            129275,
            ...
        ],
        "name": [
            "aaaaa",
            "bbbbb",
            "ccccc",
            ...
        ]
    }

    path_tag_list_each_image's json is like : {
        "2874747": [],
        "4971903": [12492],
        "2267731": [110, 313, 12492],
        ...
    }
    """

    @staticmethod
    def get_image_id(path):
        """
        >>> MultiTagLabeledDataset.get_image_id('/path/to/image/12345.png')
        '12345'
        >>> MultiTagLabeledDataset.get_image_id('/path/to/image/12345-67.png')
        '12345'
        """
        filename = os.path.splitext(os.path.basename(path))[0]
        image_id = filename.split('-')[0]
        return image_id

    def __init__(self, paths, base_dataset, path_tag_list, path_tag_list_each_image):
        if isinstance(paths, six.string_types):
            with open(paths) as paths_file:
                paths = [path.strip() for path in paths_file]
        self._paths = paths

        self.tag_id_list = json.load(open(path_tag_list, encoding='utf-8'))['id']
        self.num_tag = len(self.tag_id_list)
        self.tag_id_list_each_image = json.load(open(path_tag_list_each_image, encoding='utf-8'))

        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def get_example(self, i):
        """
        :return: (base_dataset_class's data, binary label of tag)
        """
        image_id = self.get_image_id(self._paths[i])
        tag_id_list_this_image = self.tag_id_list_each_image[image_id]
        label_index_list = [self.tag_id_list.index(tag_id) for tag_id in tag_id_list_this_image]

        label = numpy.zeros(self.num_tag, numpy.int32)
        label[label_index_list] = 1

        return self.base_dataset[i] + (label,)

    def get_input_range(self):
        return self.base_dataset.get_input_range()

    def get_input_luminance_range(self):
        return self.base_dataset.get_input_luminance_range()

    def get_output_range(self):
        return self.base_dataset.get_output_range()
