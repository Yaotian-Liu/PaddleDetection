import paddle
import paddle.fluid as fluid
import paddleslim as slim

paddle.enable_static()

exe, train_program, val_program, inputs, outputs = slim.models.image_classification(
    "MobileNet", [1, 300, 300], 10, use_gpu=True
)
place = fluid.CUDAPlace(0)

import paddle.dataset.voc2012 as reader

train_reader = paddle.fluid.io.batch(reader.train(), batch_size=64, drop_last=True)
test_reader = paddle.fluid.io.batch(reader.test(), batch_size=64, drop_last=True)
data_feeder = fluid.DataFeeder(inputs, place)

import numpy as np


def test(program):
    acc_top1_ns = []
    acc_top5_ns = []
    for data in test_reader():
        acc_top1_n, acc_top5_n, _, _ = exe.run(
            program, feed=data_feeder.feed(data), fetch_list=outputs
        )
        acc_top1_ns.append(np.mean(acc_top1_n))
        acc_top5_ns.append(np.mean(acc_top5_n))
    print(
        "Final eva - acc_top1: {}; acc_top5: {}".format(
            np.mean(np.array(acc_top1_ns)), np.mean(np.array(acc_top5_ns))
        )
    )
    return np.mean(np.array(acc_top1_ns))


for data in train_reader():
    acc1, acc5, loss, _ = exe.run(
        train_program, feed=data_feeder.feed(data), fetch_list=outputs
    )
print(np.mean(acc1), np.mean(acc5), np.mean(loss))

test(val_program)

params = []
for param in train_program.global_block().all_parameters():
    if "_sep_weights" in param.name:
        params.append(param.name)
print(params)
params = params[:5]

sens_0 = slim.prune.sensitivity(
    val_program,
    place,
    params,
    test,
    sensitivities_file="sensitivities_0.data",
    pruned_ratios=[0.1, 0.2, 0.3, 0.4],
)
print(sens_0)

loss = 0.01
ratios = slim.prune.get_ratios_by_loss(sens_0, loss)
print(ratios)

pruner = slim.prune.Pruner()
print("FLOPs before pruning: {}".format(slim.analysis.flops(val_program)))
pruned_val_program, _, _ = pruner.prune(
    val_program,
    fluid.global_scope(),
    params=ratios.keys(),
    ratios=ratios.values(),
    place=place,
    only_graph=True,
)
print("FLOPs after pruning: {}".format(slim.analysis.flops(pruned_val_program)))
