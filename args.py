import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    '--gpu',
    default = "7",
    type = str,
    help='choose gpu device')

parser.add_argument(
    '--dataset',
    default = 'ICEWS05-15',
    type = str,
    help='choose dataset')

parser.add_argument(
    '--seed_num',
    default = 1000,
    type = int,
    help='choose the number of alignment seeds')

parser.add_argument(
    '--dim',
    default = 512,
    type = int,
    help='choose embedding dimension')

parser.add_argument(
    '--depth',
    default = 2,
    type = int,
    help='choose number of label propogation layers')

parser.add_argument(
    '--top_k',
    default = 500,
    type = int,
    help='choose the top_k neighbors')

parser.add_argument(
    '--alpha',
    default = 0.6,
    type = float,
    help='choose balance factor for relational-aspect LP and temporal-aspect LP')
    
parser.add_argument(
    '--beta',
    default = 0.4,
    type = float,
    help='choose balance factor for label similarity and time similarity')

args = parser.parse_args()