from config.configurator import configs
from trainer.trainer import init_seed
from models.bulid_model import build_model
from trainer.logger import Logger
from data_utils.build_data_handler import build_data_handler
from trainer.build_trainer import build_trainer
from trainer.tuner import Tuner
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    # First Step: Create data_handler
    init_seed()
    data_handler = build_data_handler()
    data_handler.load_data()

    # Second Step: Create model
    model = build_model(data_handler).to(configs['device'])

    # Third Step: Create logger
    logger = Logger()

    # Fourth Step: Create trainer
    trainer = build_trainer(data_handler, logger)

    # Fifth Step: training
    best_model = trainer.train(model)

    # Sixth Step: test
    trainer.test(best_model)

def tune():
    # First Step: Create data_handler
    init_seed()
    data_handler = build_data_handler()
    data_handler.load_data()

    # Second Step: Create logger
    logger = Logger()

    # Third Step: Create tuner
    tuner = Tuner(logger)

    # Fourth Step: Create trainer
    trainer = build_trainer(data_handler, logger)

    # Fifth Step: Start grid search
    tuner.grid_search(data_handler, trainer)


def test():
    # First Step: Create data_handler
    init_seed()
    data_handler = build_data_handler()
    data_handler.load_data()

    # Second Step: Create model
    model = build_model(data_handler).to(configs['device'])
    # best_model = torch.load("/home/asaliao/SSLRec/checkpoint/bert4rec/bert4rec-reddit-1710574833.pth")

    # Third Step: Create logger
    logger = Logger()

    # Fourth Step: Create trainer
    trainer = build_trainer(data_handler, logger)

    # Fifth Step: load model from pretrain_path
    best_model = trainer.load_model(model)

    # Sixth Step: test
    trainer.test(best_model)

# if not configs['tune']['enable']:
#     main()
# else:
#     tune()
def visualize():
    # First Step: Create data_handler
    init_seed()
    data_handler = build_data_handler()
    data_handler.load_data()
    model = build_model(data_handler).to(configs['device'])
    logger = Logger()
    trainer = build_trainer(data_handler, logger)
    best_model = trainer.load_model(model)
    pretrain_path = "/home/asaliao/SSLRec/checkpoint/bert4rec/bert4rec-sports-1710580531.pth"
    model.load_state_dict(torch.load(pretrain_path))
    print(torch.load(pretrain_path))
    layer_index = 1
    head_index = [0, 1]  # 选择头部 1 和 2

    # 计算所选层和头部的平均注意力权重
    average_attention_weights = np.mean(best_model[layer_index][:, head_index], axis=1)

    # 可视化平均注意力权重
    plt.figure(figsize=(10, 6))
    sns.heatmap(average_attention_weights, cmap='viridis')
    plt.title(f'Average Attention Weights for Layer {layer_index} and Heads {head_index}')
    plt.xlabel('Output Sequence Position')
    plt.ylabel('Input Sequence Position')
    plt.show()

if __name__ == '__main__':
    # test()
    visualize()


