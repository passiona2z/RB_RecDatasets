# @Time   : 2020/9/18
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn


import argparse
import importlib



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_class_name', type=str, default="AmazonMusicalInstrumentsDataset_review")
    parser.add_argument('--input_path', type=str, default="")
    parser.add_argument('--output_path', type=str, default="output_data/MI")

    parser.add_argument('--convert_inter', action='store_true')
    parser.add_argument('--convert_item', action='store_true')
    parser.add_argument('--convert_user', action='store_true')

    args = parser.parse_args()

    assert args.input_path is not None, 'input_path can not be None, please specify the input_path'
    assert args.output_path is not None, 'output_path can not be None, please specify the output_path'

    input_args = [args.input_path, args.output_path]
    dataset_class = getattr(importlib.import_module('src.extended_dataset'), args.dataset_class_name)

    datasets = dataset_class(*input_args)

    if args.convert_inter:
        datasets.convert_inter()
    if args.convert_item:
        datasets.convert_item()
    if args.convert_user:
        datasets.convert_user()
