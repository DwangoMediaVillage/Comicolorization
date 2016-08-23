import copy
import typing
import chainer

import comicolorization


def _make_ltbc_base(setting: typing.Dict):
    model = comicolorization.models.Ltbc(
        use_global=not setting['disable_ltbc_global'],
        use_classification=setting['alpha_ltbc_classification'] is not None,
        classification_num_output_list=setting['ltbc_classification_num_output_list'],
        use_histogram=setting['use_histogram_network'],
        use_multidimensional_histogram=setting['use_multidimensional_histogram'],
        num_bins_histogram=setting['num_bins_histogram'],
        threshold_histogram_palette=setting['threshold_histogram_palette'],
        reinput_mode=setting['reinput_mode'],
        loss_type=setting['loss_type'],
    )
    return model


def _make_ltbc_main(setting: typing.Dict):
    setting_copy = copy.deepcopy(setting)
    if not setting['separate_model_reinput']:
        setting_copy['reinput_mode'] = setting['reinput_mode']
    else:
        setting_copy['reinput_mode'] = None

    model = _make_ltbc_base(setting_copy)

    if setting['path_pretrained_model'] is not None:
        chainer.serializers.load_npz(setting['path_pretrained_model'], model)

    return model


def _make_ltbc_reinput(main_model, setting: typing.Dict):
    if not setting['separate_model_reinput']:
        model_list = [main_model for _ in setting['loss_blend_ratio_reinput']]
    else:
        model_list = [_make_ltbc_base(setting) for _ in setting['loss_blend_ratio_reinput']]
    return model_list


def make_ltbc(setting: typing.Dict):
    model_main = _make_ltbc_main(setting)
    mdoel_reinput_list = _make_ltbc_reinput(model_main, setting)
    return model_main, mdoel_reinput_list
