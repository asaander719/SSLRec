from config.configurator import configs
from trainer.trainer import init_seed
from models.bulid_model import build_model
from trainer.logger import Logger
from data_utils.build_data_handler import build_data_handler
from trainer.build_trainer import build_trainer
from trainer.tuner import Tuner

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

    # Third Step: Create logger
    logger = Logger()

    # Fourth Step: Create trainer
    trainer = build_trainer(data_handler, logger)

    # Fifth Step: load model from pretrain_path
    best_model = trainer.load(model)

    # Sixth Step: test
    trainer.test(best_model)

if not configs['tune']['enable']:
#       aug_with_maxself: [1, 0]
#   replace_ratio: [0.1, 0.2, 0.3, 0.5, 0.8, 1]
#   aug_ratio: [0.1, 0.2, 0.3, 0.5, 0.8, 1]
#   cl_weight: [0.1, 0.2, 0.5, 1, 2]
#   aug_k: [1,2,3,4,5]
    for i in [1, 0]:
        configs['model']['aug_with_maxself'] = i
        for j in [0.1, 0.2, 0.3, 0.5, 0.8, 1]:
            configs['model']['replace_ratio'] = j
            for k in [0.1, 0.2, 0.3, 0.5, 0.8, 1]:
                configs['model']['aug_ratio'] = k
                for m in [0.1, 0.2, 0.5, 1, 2]:
                    configs['model']['cl_weight'] = m
                    for n in [1,2,3,4,5]:
                        configs['model']['aug_k'] = n
                        print('aug_with_maxself:', configs['model']['aug_with_maxself'], 'replace_ratio:', configs['model']['replace_ratio'],
                              'aug_ratio:',configs['model']['aug_ratio'], 'cl_weight:',  configs['model']['cl_weight'], 'aug_k:', configs['model']['aug_k'] )
                        main()
else:
    tune()


