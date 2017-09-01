import chainer

from comicolorization_sr.loss import LossMaker


class Updater(chainer.training.StandardUpdater):
    def __init__(self, loss_maker, *args, **kwargs):
        # type: (LossMaker, *any, **any) -> None
        super(Updater, self).__init__(*args, **kwargs)
        self.loss_maker = loss_maker

    def update_core(self):
        optimizers = self.get_all_optimizers()

        batch = self.converter(self.get_iterator('main').next(), self.device)
        input, concat, target = batch['input'], batch['concat'], batch['target']

        loss = self.loss_maker.make_loss(input=input, concat=concat, target=target, test=False)
        optimizers['main'].update(self.loss_maker.sum_loss, loss['main'])
