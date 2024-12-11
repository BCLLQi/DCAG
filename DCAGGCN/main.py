import os
import argparse
import warnings
from trainer import GNN_RUL_trainer
import sklearn.exceptions

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', default='experiments_logs', type=str, help='Directory containing all experiments')

parser.add_argument('--run_description', default='test', type=str, help='name of your runs')
parser.add_argument('--GNN_method', default='DCAG_GCN', type=str)
parser.add_argument('--data_path', default=r'./Data_Process/Processed_dataset', type=str, help='Path containing dataset')
parser.add_argument('--dataset', default='PHM2012', type=str, help='Dataset choice: (PHM2012 - XJTU_SY)')
parser.add_argument('--dataset_id', default='Condition_1', type=str)
parser.add_argument('--num_runs', default=5, type=int, help='Number of consecutive run with different seeds')
parser.add_argument('--device', default='cuda:0', type=str, help='cpu or cuda')
args = parser.parse_args()

if __name__ == "__main__":

    method_bearing_names = ['DCAG_GCN']

    trainer = GNN_RUL_trainer(args)
    trainer.train()


