import chainer
import typing
import comicolorization


class MultiUpdater(chainer.training.StandardUpdater):
    def __init__(
            self,
            args,
            loss_maker: comicolorization.loss.LossMaker,
            main_optimizer,
            main_lossfun: typing.Callable[[typing.Dict], any],
            reinput_optimizer: typing.List[chainer.Optimizer] = None,
            reinput_lossfun: typing.Callable[[int, typing.Dict], any] = None,
            discriminator_optimizer=None,
            discriminator_lossfun: typing.Callable[[typing.Dict], any] = None,
            *_args, **kwargs
    ):
        optimizers = {'main': main_optimizer}
        if reinput_optimizer is not None:
            for i_reinput, optimizer in enumerate(reinput_optimizer):
                optimizers['reinput{}'.format(i_reinput)] = optimizer

        if discriminator_optimizer is not None:
            optimizers['discriminator'] = discriminator_optimizer

        super().__init__(optimizer=optimizers, *_args, **kwargs)

        # 同じモデルに複数のoptimizerを適用するとreporterが壊れるため特別処置
        if args.separate_backward_reinput and reinput_optimizer is None:
            reinput_optimizer = [main_optimizer for _ in range(len(args.loss_blend_ratio_reinput))]

        self.args = args
        self.loss_maker = loss_maker
        self.main_optimizer = main_optimizer
        self.main_lossfun = main_lossfun
        self.reinput_optimizer = reinput_optimizer
        self.reinput_lossfun = reinput_lossfun
        self.discriminator_optimizer = discriminator_optimizer
        self.discriminator_lossfun = discriminator_lossfun

    def update_core(self):
        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)

        loss_detail = self.loss_maker.calc_loss(*tuple(chainer.Variable(x) for x in in_arrays), test=False)

        # main network
        main_optimizer = self.main_optimizer
        main_optimizer.update(self.main_lossfun, loss_detail)

        # reinput network
        reinput_optimizer_list = self.reinput_optimizer
        if reinput_optimizer_list is not None:
            for i_reinput, reinput_optimizer in enumerate(reinput_optimizer_list):
                reinput_optimizer.update(self.reinput_lossfun, i_reinput, loss_detail)

        if self.discriminator_optimizer is not None:
            self.discriminator_optimizer.update(self.discriminator_lossfun, loss_detail)
