import chainer
import collections
import glob

from comicolorization_sr import config
from comicolorization_sr.data_process import *


class BaseImageDataset(chainer.dataset.DatasetMixin):
    def get_example(self, i) -> Image.Image:
        raise NotImplementedError


class BaseImageArrayDataset(chainer.dataset.DatasetMixin):
    def get_example(self, i) -> numpy.ndarray:
        raise NotImplementedError


class PILImageDataset(BaseImageDataset):
    def __init__(self, paths):
        self._paths = paths

    def __len__(self):
        return len(self._paths)

    def get_example(self, i):
        path = self._paths[i]
        return Image.open(path).convert('RGB')


class RandomCropImageDataset(BaseImageDataset):
    def __init__(self, base_image_dataset, test, crop_width, crop_height):
        self._base_dataset = base_image_dataset
        self._test = test
        self._crop_width = crop_width
        self._crop_height = crop_height

    def __len__(self):
        return len(self._base_dataset)

    def get_example(self, i):
        image = self._base_dataset[i]
        width, height = image.size
        assert width >= self._crop_width and height >= self._crop_height

        if not self._test:
            top = numpy.random.randint(height - self._crop_height + 1)
            left = numpy.random.randint(width - self._crop_width + 1)
        else:
            top = (height - self._crop_height) // 2
            left = (width - self._crop_width) // 2

        bottom = top + self._crop_height
        right = left + self._crop_width

        image = image.crop((left, top, right, bottom))
        return image


class RandomFlipImageDataset(BaseImageDataset):
    def __init__(self, base_image_dataset, test, p_flip_horizontal=0.5, p_flip_vertical=0.0):
        self._base_dataset = base_image_dataset
        self._test = test
        self.p_flip_horizontal = p_flip_horizontal
        self.p_flip_vertical = p_flip_vertical

    def __len__(self):
        return len(self._base_dataset)

    def get_example(self, i):
        image = self._base_dataset[i]

        if not self._test:
            if numpy.random.rand(1) < self.p_flip_horizontal:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

            if numpy.random.rand(1) < self.p_flip_vertical:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)

        return image


class PairImageDataset(chainer.dataset.DatasetMixin):
    def __init__(
            self,
            base_dataset: BaseImageDataset,
            test,
            input_process: BaseDataProcess,
            concat_process: BaseDataProcess,
            target_process: BaseDataProcess,
    ):
        self._test = test
        self._base_dataset = base_dataset
        self._input_process = input_process
        self._concat_process = concat_process
        self._target_process = target_process

    def __len__(self):
        return len(self._base_dataset)

    def get_example(self, i):
        image = self._base_dataset[i]
        input_image = self._input_process(image, self._test)  # 小さい入力画像
        concat_image = self._concat_process(image, self._test)  # 大きいconcat用の入力画像
        target_image = self._target_process(image, self._test)  # 目標画像
        return {
            'input': input_image,
            'concat': concat_image,
            'target': target_image,
        }


def create(
        config: config.DatasetConfig,
        input_process: BaseDataProcess,
        concat_process: BaseDataProcess,
):
    """
    :param input_process: targetのPILImageを、入力arrayに変換する処理
    :param concat_process: targetのPILImageを、超解像入力arrayに結合させるarrayに変換する処理
    """
    paths = glob.glob(config.images_glob)

    random_state = numpy.random.RandomState(seed=config.seed_evaluation)
    paths = random_state.permutation(paths)

    s = config.scale_input
    w = config.target_width

    num_test = config.num_test
    train_paths = paths[num_test:]
    test_paths = paths[:num_test]
    train_varidation_paths = train_paths[:num_test]

    keys = ['train', 'test', 'train_varidation']
    datasets = collections.namedtuple('Datasets', keys)
    for key, paths, test in zip(
            keys,
            [train_paths, test_paths, train_varidation_paths],
            [False, True, True],
    ):
        d = PILImageDataset(paths=paths)
        d = RandomCropImageDataset(d, crop_width=w, crop_height=w, test=test)
        d = RandomFlipImageDataset(d, test=test)

        # 小さい入力画像、大きいconcat用の入力画像、目標画像それぞれのデータ処理
        d = PairImageDataset(
            base_dataset=d,
            test=test,
            input_process=ChainProcess([
                RandomScaleImageProcess(min_scale=s, max_scale=s),
                input_process,
            ]),
            concat_process=concat_process,
            target_process=LabImageArrayProcess(),
        )

        setattr(datasets, key, d)

    return datasets
