import chainer
import typing

import comicolorization
import numpy


class LossMaker(object):
    def __init__(
            self,
            args,
            model,
            model_reinput_list,
            range_input_luminance,
            range_output_luminance,
            discriminator
    ):
        # type: (any, comicolorization.models.BaseModel, typing.List[comicolorization.models.BaseModel], any, any, comicolorization.models.Discriminator) -> None
        self.args = args
        self.model = model
        self.model_reinput_list = model_reinput_list
        self.range_input_luminance = range_input_luminance
        self.range_output_luminance = range_output_luminance
        self.discriminator = discriminator
        self.use_classification = args.alpha_ltbc_classification is not None

        if args.ltbc_classification_loss_function == 'softmax':
            self.loss_func_classify = chainer.functions.softmax_cross_entropy
        elif args.ltbc_classification_loss_function == 'multi_label':
            self.loss_func_classify = chainer.functions.sigmoid_cross_entropy

    def _forward_model(self, model, image_input, image_rgb, image_real, test):
        if self.args.use_histogram_network:
            color, other = model(image_input, x_rgb=image_rgb, test=test)
        else:
            color, other = model(image_input, test=test)

        if self.discriminator is not None:
            disc_real, disc_gen = self._forward_discriminator(image_real, image_generated=color, test=test)
        else:
            disc_real, disc_gen = None, None

        return color, other, disc_real, disc_gen

    def _forward_reinput(self, color, image_rgb, image_real, test):
        outputs = []

        for i_reinput in range(len(self.args.loss_blend_ratio_reinput)):
            model_reinput = self.model_reinput_list[i_reinput]
            image_input = model_reinput.xp.copy(color.data)
            image_input_residual = model_reinput.xp.copy(image_input)

            # convert gray color range '''output to input'''
            image_input[:, 0, :, :] = comicolorization.utility.color.normalize(
                image_input[:, 0, :, :],
                in_min=self.range_output_luminance[0], in_max=self.range_output_luminance[1],
                out_min=self.range_input_luminance[0], out_max=self.range_input_luminance[1],
            )

            color, other, disc_real, disc_gen = \
                self._forward_model(model_reinput, image_input, image_rgb, image_real, test=test)

            if self.args.use_residual_reinput:
                color += chainer.Variable(image_input_residual)

            outputs.append([color, other, disc_real, disc_gen])

        return outputs

    def _forward_discriminator(self, image_real, image_generated, test):
        disc_real = self.discriminator(image_real, test=test)
        disc_gen = self.discriminator(image_generated, test=test)
        return disc_real, disc_gen

    def _forward(self, image_input, image_rgb, image_real, test):
        color, other, disc_real, disc_gen = \
            self._forward_model(self.model, image_input, image_rgb, image_real, test=test)
        outputs_reinput = self._forward_reinput(color, image_rgb, image_real, test=test)
        return {
            'main': (color, other, disc_real, disc_gen),
            'reinput': outputs_reinput,
        }

    def _calc_loss_adversarial(self, disc_real, disc_gen):
        f_sce = chainer.functions.sigmoid_cross_entropy
        label_true = self.discriminator.xp.ones((disc_real.shape[0], 1), dtype=numpy.int32)
        label_false = self.discriminator.xp.zeros((disc_real.shape[0], 1), dtype=numpy.int32)
        loss_generator = f_sce(disc_gen, label_true)
        loss_disc_real = f_sce(disc_real, label_true)
        loss_disc_fake = f_sce(disc_gen, label_false)

        tp = (disc_real.data > 0.5).sum()
        fp = (disc_gen.data > 0.5).sum()
        fn = (disc_real.data <= 0.5).sum()
        tn = (disc_gen.data <= 0.5).sum()
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        if not self.discriminator.xp.isfinite(precision):
            precision = self.discriminator.xp.zeros(1, dtype=numpy.float32)

        loss_disc = loss_disc_real + loss_disc_fake
        return loss_disc, loss_generator, {
            'loss_discriminator_real': loss_disc_real,
            'loss_discriminator_fake': loss_disc_fake,
            'loss_generator': loss_generator,
            'accuracy_discriminator': accuracy,
            'precision_discriminator': precision,
            'recall_discriminator': recall,
        }

    def _calc_loss_main(self, color, other, disc_real, disc_gen, image_target, label):
        """
        :return: (sum of loss, loss detail)
        """
        if self.args.mse_loss_mode == 'color_space':
            loss_mse = chainer.functions.mean_squared_error(color, image_target)
        elif self.args.mse_loss_mode == 'before_sigmoid':
            before_sigmoid = other['before_sigmoid']
            image_zero_one = comicolorization.utility.color.normalize_zero_one(
                array=image_target,
                in_type=self.args.loss_type,
                split=lambda x: chainer.functions.split_axis(x, x.shape[1], axis=1, force_tuple=True),
                concat=lambda x: chainer.functions.concat(x, axis=1),
            )
            loss_mse = chainer.functions.mean_squared_error(before_sigmoid, image_zero_one)
        else:
            raise NotImplementedError

        loss = loss_mse * self.args.blend_mse_color

        if other['classification'] is not None:
            loss += self.loss_func_classify(other['classification'], label) * self.args.alpha_ltbc_classification

        if self.discriminator is not None:
            loss_discriminator, loss_generator, loss_adversarial = \
                self._calc_loss_adversarial(disc_real, disc_gen)
            loss += loss_generator * self.args.blend_adversarial_generator
        else:
            loss_discriminator = None
            loss_generator = None
            loss_adversarial = None

        return loss, loss_discriminator, {
            'loss': loss,
            'loss_mse': loss_mse,
            'loss_discriminator': loss_discriminator,
            'loss_generator': loss_generator,
            'loss_adversarial': loss_adversarial,
        }

    def _calc_loss(self, outputs, image_target, label):
        """
        :return: (sum of loss, loss detail)
        """
        color, other, disc_real, disc_gen = outputs['main']
        outputs_reinput = outputs['reinput']

        sum_loss, sum_loss_discriminator, loss_detail_main = \
            self._calc_loss_main(color, other, disc_real, disc_gen, image_target, label)

        loss_detail_reinput = []
        for i_reinput, blend in enumerate(self.args.loss_blend_ratio_reinput):
            color, other, disc_real, disc_gen = outputs_reinput[i_reinput]
            loss, loss_discriminator, loss_detail = \
                self._calc_loss_main(color, other, disc_real, disc_gen, image_target, label)
            sum_loss += loss
            if loss_discriminator is not None:
                sum_loss_discriminator += loss_discriminator
            loss_detail_reinput.append(loss_detail)

        return sum_loss, {
            'sum_loss': sum_loss,
            'sum_loss_discriminator': sum_loss_discriminator,
            'main': loss_detail_main,
            'reinput': loss_detail_reinput,
        }

    def calc_loss(self, image_target, image_gray, image_rgb, label=None, test=None):
        assert test is not None, "Please input True of Fase for the value of test."

        # forward
        outputs = self._forward(image_input=image_gray, image_rgb=image_rgb, image_real=image_target, test=test)

        # loss
        sum_loss, loss_detail = self._calc_loss(outputs=outputs, image_target=image_target, label=label)

        # report
        loss_flatten = comicolorization.utility.object.flatten(loss_detail['main'], sep='/')
        chainer.reporter.report(loss_flatten, self.model)

        for i_reinput, loss_reinput in enumerate(loss_detail['reinput']):
            model_reinput = self.model_reinput_list[i_reinput]
            loss_reinput = {'reinput{}/{}'.format(i_reinput, key): value for key, value in loss_reinput.items()}
            chainer.reporter.report(loss_reinput, model_reinput)

        chainer.reporter.report({'sum_loss': sum_loss}, self.model)
        return loss_detail

    def get_loss_names(self):
        return [
            'sum_loss',
            'loss',
            'loss_mse',
            'loss_generator',
            'loss_discriminator',
            'loss_adversarial/loss_discriminator_fake',
            'loss_adversarial/loss_discriminator_real',
            'loss_adversarial/recall_discriminator',
            'loss_adversarial/precision_discriminator',
            'loss_adversarial/accuracy_discriminator',
        ]

    def loss_test(self, image_target, image_gray, image_rgb):
        loss_detail = self.calc_loss(image_target, image_gray, image_rgb, test=True)
        return loss_detail['sum_loss']
