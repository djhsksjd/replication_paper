from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch

class Evaluator:
    def compute_metrics(self, true_labels, predictions):
        """
        计算MAE和RMSE指标
        """
        mae = mean_absolute_error(true_labels, predictions)
        rmse = mean_squared_error(true_labels, predictions, squared=False)
        return mae, rmse
    
    def evaluate(self, model, data_loader, config):
        """
        在指定数据集上评估模型
        """
        model.eval()
        all_preds = []
        all_trues = []
        
        with torch.no_grad():
            for batch in data_loader:
                user_idx, item_idx, rating = batch
                user_idx = user_idx.to(config["device"])
                item_idx = item_idx.to(config["device"])
                rating = rating.to(config["device"])
                
                r_hat = model(user_idx, item_idx, config["C"], config["N"], config["B"], config["R"])
                all_preds.extend(r_hat.cpu().numpy())
                all_trues.extend(rating.cpu().numpy())
        
        return self.compute_metrics(all_trues, all_preds)
