import sys
import os
import json
import glob
import typing
import chainer
from chainer.training import extensions
import numpy
ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(ROOT_PATH)
import comicolorization
from comicolorization.extensions import SaveRawImageExtension, SaveGeneratedImageExtension

parser = comicolorization.utility.config.get_train_parser()

args = parser.parse_args()
print(args)

use_classification = args.alpha_ltbc_classification is not None

model = None  # type: chainer.Chain
discriminator = None
if args.network_model == 'SimpleConvolution':
    model = comicolorization.models.SimpleConvolution(loss_type=args.loss_type)
    model_reinput_list = []
elif args.network_model == 'LTBC':
    model, model_reinput_list = comicolorization.utility.model.make_ltbc(vars(args))
else:
    raise ValueError(args.network_model)

if args.use_adversarial_network:
    discriminator = comicolorization.models.Discriminator(
        size=args.size_image,
        first_pooling_size=args.discriminator_first_pooling_size,
    )

if args.gpu >= 0:
    chainer.cuda.get_device(args.gpu).use()  # Make the GPU current
    model.to_gpu()
    if discriminator is not None:
        discriminator.to_gpu()

    if args.separate_model_reinput:
        for model_reinput in model_reinput_list:
            model_reinput.to_gpu()

# make dataset
paths = glob.glob("{}/*".format(args.dataset_path))
random_state = numpy.random.RandomState(args.random_seed_test)
paths = random_state.permutation(paths)

datasets = comicolorization.utility.dataset.choose_dataset(
    paths=paths,
    num_dataset_test=args.num_dataset_test,
    loss_type=args.loss_type,
    augmentation=args.augmentation,
    size_image_augmentation=[args.size_image_augmentation, args.size_image_augmentation],
    size_image=[args.size_image, args.size_image],
    use_ltbc_classification=args.alpha_ltbc_classification is not None,
    path_tag_list=args.path_tag_list,
    path_tag_list_each_image=args.path_tag_list_each_image,
    line_drawing_mode=args.line_drawing_mode,
    max_pixel_drawing=args.max_pixel_drawing,
    max_size_pixel_drawing=args.max_size_pixel_drawing,
    use_binarization_dataset=False,
)
train_dataset = datasets['train']
test_dataset = datasets['test']
train_for_evaluate_dataset = datasets['train_for_evaluate']

train_iterator = chainer.iterators.MultiprocessIterator(
    train_dataset,
    batch_size=args.batchsize,
    repeat=True,
    shuffle=False,
)
test_iterator = chainer.iterators.MultiprocessIterator(
    test_dataset,
    batch_size=args.batchsize,
    repeat=False,
    shuffle=False,
)
train_for_evaluate_iterator = chainer.iterators.MultiprocessIterator(
    train_for_evaluate_dataset,
    batch_size=args.batchsize,
    repeat=False,
    shuffle=False,
)

range_input = train_dataset.get_input_range()
range_input_luminance = train_dataset.get_input_luminance_range()
range_output_luminance = train_dataset.get_output_range()[0]

# make loss
loss_maker = comicolorization.loss.LossMaker(
    args=args,
    model=model,
    model_reinput_list=model_reinput_list,
    range_input_luminance=range_input_luminance,
    range_output_luminance=range_output_luminance,
    discriminator=discriminator
)


# make trainer
def make_optimizer(_model):
    _optimizer = chainer.optimizers.Adam()
    _optimizer.setup(_model)

    if args.weight_decay is not None:
        _optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

    return _optimizer


optimizer = make_optimizer(model)
if discriminator is not None:
    discriminator_optimizer = make_optimizer(discriminator)
else:
    discriminator_optimizer = None

if (not args.separate_backward_reinput) & (not args.separate_model_reinput):
    main_lossfun = lambda loss_detail: loss_detail['sum_loss']
    reinput_lossfun = None
    reinput_optimizer = None
else:
    main_lossfun = lambda loss_detail: loss_detail['main']['loss']
    reinput_lossfun = lambda i_reinput, loss_detail: loss_detail['reinput'][i_reinput]['loss']
    if not args.separate_model_reinput:
        reinput_optimizer = None
    else:
        reinput_optimizer = []
        for i in range(len(args.loss_blend_ratio_reinput)):
            optimizer_reinput = make_optimizer(model_reinput_list[i])
            reinput_optimizer.append(optimizer_reinput)

discriminator_lossfun = lambda loss_detail: loss_detail['sum_loss_discriminator']

updater = comicolorization.updater.MultiUpdater(
    args=args,
    loss_maker=loss_maker,
    main_optimizer=optimizer,
    main_lossfun=main_lossfun,
    reinput_optimizer=reinput_optimizer,
    reinput_lossfun=reinput_lossfun,
    iterator=train_iterator,
    device=args.gpu,
    discriminator_optimizer=discriminator_optimizer,
    discriminator_lossfun=discriminator_lossfun
)
trainer = chainer.training.Trainer(updater, (args.max_epoch, 'epoch'), out=args.save_result_path)


def save_json(filename, obj):
    json.dump(obj, open(filename, 'w'), sort_keys=True, indent=4)


input_image_mode = 'gray' if len(range_input) != 3 else 'Lab'

train_images = [train_for_evaluate_dataset[i] for i in range(10)]
train_color_images = numpy.concatenate([numpy.expand_dims(images[0], axis=0) for images in train_images])
train_gray_images = numpy.concatenate([numpy.expand_dims(images[1], axis=0) for images in train_images])
train_rgb_images = numpy.concatenate([numpy.expand_dims(images[2], axis=0) for images in train_images])
train_extend_generated_image = SaveGeneratedImageExtension(train_gray_images, train_rgb_images, model, prefix_directory='train_{.updater.iteration}', image_mode=args.loss_type)
train_extend_gray_image = SaveRawImageExtension(train_gray_images, prefix_directory='train_gray_image', prefix_filename='gray_', image_mode=input_image_mode, linedrawing=args.line_drawing_mode)
train_extend_raw_image = SaveRawImageExtension(train_color_images, prefix_directory='train_raw_image', prefix_filename='color_', image_mode=args.loss_type)

test_images = [test_dataset[i] for i in range(10)]
test_color_images = numpy.concatenate([numpy.expand_dims(images[0], axis=0) for images in test_images])
test_gray_images = numpy.concatenate([numpy.expand_dims(images[1], axis=0) for images in test_images])
test_rgb_images = numpy.concatenate([numpy.expand_dims(images[2], axis=0) for images in test_images])
test_extend_generated_image = SaveGeneratedImageExtension(test_gray_images, test_rgb_images, model, prefix_directory='test_{.updater.iteration}', image_mode=args.loss_type)
test_extend_gray_image = SaveRawImageExtension(test_gray_images, prefix_directory='test_gray_image', prefix_filename='gray_', image_mode=input_image_mode, linedrawing=args.line_drawing_mode)
test_extend_raw_image = SaveRawImageExtension(test_color_images, prefix_directory='test_raw_image', prefix_filename='color_', image_mode=args.loss_type)

trainer.extend(extensions.dump_graph('main/sum_loss', out_name='main_graph.dot'))

trainer.extend(extensions.snapshot_object(args.__dict__, 'argument.json', savefun=save_json), invoke_before_training=True)
trainer.extend(train_extend_gray_image, invoke_before_training=True, trigger=lambda _: False)
trainer.extend(train_extend_raw_image, invoke_before_training=True, trigger=lambda _: False)
trainer.extend(test_extend_gray_image, invoke_before_training=True, trigger=lambda _: False)
trainer.extend(test_extend_raw_image, invoke_before_training=True, trigger=lambda _: False)

save_interval = (args.save_result_iteration, 'iteration')
trainer.extend(train_extend_generated_image, trigger=save_interval)
trainer.extend(test_extend_generated_image, trigger=save_interval)
trainer.extend(extensions.snapshot_object(model, '{.updater.iteration}.model'), trigger=save_interval)

if args.separate_model_reinput:
    for i_reinput, model_reinput in enumerate(model_reinput_list):
        name = 'reinput{}'.format(i_reinput)
        trainer.extend(extensions.dump_graph(name + '/' + name + '/loss', out_name=name + '_graph.dot'))

        name = 'reinput{}'.format(i_reinput) + '_{.updater.iteration}.model'
        trainer.extend(extensions.snapshot_object(model_reinput, name), trigger=save_interval)

num_reinput = len(args.loss_blend_ratio_reinput)

report_target = ['epoch', 'iteration']
for evaluater_name in ['', 'validation/', 'validation/train/']:
    for model_name in ['main/'] + ['discriminator/'] + ['reinput{}/'.format(i) for i in range(num_reinput)]:
        for reinput_name in [''] + ['reinput{}/'.format(i) for i in range(num_reinput)]:
            for loss_name in loss_maker.get_loss_names():
                report_target.append(evaluater_name + model_name + reinput_name + loss_name)

log_interval = (args.log_interval, 'iteration')
targets = {'main': model}
if args.separate_model_reinput:
    for i_reinput, model_reinput in enumerate(model_reinput_list):
        targets['reinput{}'.format(i_reinput)] = model_reinput
trainer.extend(extensions.Evaluator(test_iterator, target=targets, eval_func=loss_maker.loss_test, device=args.gpu), trigger=log_interval)
trainer.extend(extensions.Evaluator(train_for_evaluate_iterator, target=targets, eval_func=loss_maker.loss_test, device=args.gpu), name='validation/train', trigger=log_interval)
trainer.extend(extensions.LogReport(trigger=log_interval, log_name="log.txt"))
trainer.extend(extensions.PrintReport(report_target))
trainer.extend(extensions.ProgressBar(update_interval=10))

trainer.run()
