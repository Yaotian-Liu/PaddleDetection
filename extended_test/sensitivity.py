import os
import re
import yaml
import pandas as pd

cfg = {
    "pretrain_weights": "https://paddlemodels.bj.bcebos.com/object_detection/dygraph/ssd_mobilenet_v1_300_120e_voc.pdparams",
    "slim": "PruneQAT",
    "pruner": "Pruner",
    "Pruner": {
        "criterion": "fpgm",
        "print_params": False,
    },
    "qat": "QAT",
    "QAT": {
        "quant_config": {
            "weight_quantize_type": "channel_wise_abs_max",
            "activation_quantize_type": "moving_average_abs_max",
            "weight_bits": 8,
            "activation_bits": 8,
            "dtype": "int8",
            "window_size": 10000,
            "moving_rate": 0.9,
            "quantizable_layer_type": ["Conv2D", "Linear"],
        },
        "print_model": False,
    },
}


def generate_yaml(pruned_params, pruned_ratios):
    cfg["Pruner"]["pruned_params"] = [pruned_params]
    cfg["Pruner"]["pruned_ratios"] = [pruned_ratios]
    with open("./extended/prune_qat_test.yml", "w") as f:
        yaml.dump(cfg, f)


def train(config, slim_config):
    os.system("rm -rf " + "output/" + slim_config[9:-4])
    train_cmd = (
        "python tools/train.py"
        + " -c "
        + config
        + " --eval "
        + " --slim_config "
        + slim_config
    )
    print("training")
    print(train_cmd)
    os.system(train_cmd)


def read_acc(cmd):
    f = os.popen(cmd, "r")
    res = f.readlines()
    f.close()
    acc = re.findall(r"[\d.]+%", res[-2])[0][:-1]
    return float(acc)


def evaluate(config, slim_config):
    eval_cmd = (
        "python tools/eval.py"
        + " -c "
        + config
        + " --slim_config "
        + slim_config
        + " -o "
        + "weights=output/"
        + slim_config[9:-4]
        + "/best_model"
    )
    print("evaluating")
    print(eval_cmd)
    acc = read_acc(eval_cmd)
    return acc


if __name__ == "__main__":

    config = "configs/ssd/ssd_mobilenet_v1_300_40e_roadsign_voc.yml"
    slim_config = "extended/prune_qat_test.yml"

    # baseline
    generate_yaml(None, 0)
    train(config, slim_config)
    baseline_acc = evaluate(config, slim_config)

    # test sensitivities
    param_list = []
    ratio_list = []
    acc_list = []
    sensitivity_list = []

    pruned_params = [
        "conv2d_8.w_0",
        "conv2d_10.w_0",
        "conv2d_12.w_0",
        "conv2d_14.w_0",
        "conv2d_16.w_0",
        "conv2d_18.w_0",
        "conv2d_20.w_0",
        "conv2d_22.w_0",
    ]
    pruned_ratios = [0.1, 0.3, 0.5, 0.7]
    for param in pruned_params:
        for ratio in pruned_ratios:
            generate_yaml(param, ratio)
            train(config, slim_config)
            acc = evaluate(config, slim_config)
            sensitivity = baseline_acc - acc

            param_list.append(param)
            ratio_list.append(ratio)
            acc_list.append(acc)
            sensitivity_list.append(sensitivity)

    result_df = pd.DataFrame(
        {
            "param": param_list,
            "ratio": ratio_list,
            "acc": acc_list,
            "sensitivity": sensitivity_list,
        }
    )

    print(result_df)
    result_df.to_csv("extended/sensitivity.csv")
