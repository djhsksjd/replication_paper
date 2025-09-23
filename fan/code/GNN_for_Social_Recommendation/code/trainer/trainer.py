import torch
import torch.nn as nn

class Trainer:
    def __init__(self, model, criterion, optimizer, config, dataloaders):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config  # 从config.py传入的全局配置
        self.train_loader, self.val_loader, self.test_loader = dataloaders
        self.device = config["device"]
        self.best_val_rmse = float("inf")
        self.patience_counter = 0

    def train_epoch(self):
        # 单轮训练逻辑
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            user_idx, item_idx, rating = [x.to(self.device) for x in batch]
            self.optimizer.zero_grad()
            
            # 前向传播
            r_hat = self.model(user_idx, item_idx, self.config["C"], self.config["N"], self.config["B"], self.config["R"])
            
            # 计算损失
            loss = self.criterion(r_hat, rating)
            
            # 反向传播与优化
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * user_idx.size(0)
        
        return total_loss / len(self.train_loader.dataset)

    def validate(self, evaluator):
        # 验证逻辑
        self.model.eval()
        all_preds, all_trues = [], []
        
        with torch.no_grad():
            for batch in self.val_loader:
                user_idx, item_idx, rating = [x.to(self.device) for x in batch]
                r_hat = self.model(user_idx, item_idx, self.config["C"], self.config["N"], self.config["B"], self.config["R"])
                all_preds.extend(r_hat.cpu().numpy())
                all_trues.extend(rating.cpu().numpy())
        
        return evaluator.compute_metrics(all_trues, all_preds)

    def train(self, evaluator):
        # 完整训练流程（含早停）
        for epoch in range(self.config["epochs"]):
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_mae, val_rmse = self.validate(evaluator)
            
            # 打印日志
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f}")
            
            # 早停与模型保存
            if val_rmse < self.best_val_rmse:
                self.best_val_rmse = val_rmse
                torch.save(self.model.state_dict(), self.config["best_model_path"])
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config["patience"]:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
