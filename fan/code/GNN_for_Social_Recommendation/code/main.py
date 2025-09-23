import torch
import torch.nn as nn
import torch.optim as optim
import logging
from model.graphrec import GraphRec
from trainer.trainer import Trainer
from trainer.evaluator import Evaluator
from trainer.data_loader import get_dataloader
from config import config
from utils.log_utils import init_logger

def main():
    # 1. 初始化日志
    logger = init_logger(config["log_path"])
    logger.info("Starting GraphRec training...")
    logger.info(f"Using device: {config['device']}")
    
    # 2. 加载数据
    logger.info("Loading data...")
    try:
        dataloaders = get_dataloader(
            train_path=config["train_data_path"],
            val_path=config["val_data_path"],
            test_path=config["test_data_path"],
            batch_size=config["batch_size"]
        )
        logger.info("Data loaded successfully")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return
    
    # 3. 初始化模型、损失函数、优化器
    logger.info("Initializing model...")
    try:
        model = GraphRec(
            n_users=config["n_users"],
            n_items=config["n_items"],
            n_ratings=config["n_ratings"],
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            num_hidden_layers=config["num_hidden_layers"]
        ).to(config["device"])
        
        criterion = nn.MSELoss()
        optimizer = optim.RMSprop(
            model.parameters(), 
            lr=config["lr"], 
            alpha=0.99, 
            eps=1e-08
        )
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        return
    
    # 4. 初始化评估器和训练器
    evaluator = Evaluator()
    trainer = Trainer(model, criterion, optimizer, config, dataloaders)
    
    # 5. 启动训练
    logger.info("Starting training...")
    try:
        trainer.train(evaluator)
        logger.info("Training completed")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return
    
    # 6. 测试最优模型
    logger.info("Evaluating best model on test set...")
    try:
        model.load_state_dict(torch.load(config["best_model_path"]))
        test_mae, test_rmse = evaluator.evaluate(model, dataloaders[2], config)
        logger.info(f"Test MAE: {test_mae:.4f} | Test RMSE: {test_rmse:.4f}")
        print(f"\nTest MAE: {test_mae:.4f} | Test RMSE: {test_rmse:.4f}")
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        return

if __name__ == "__main__":
    main()
