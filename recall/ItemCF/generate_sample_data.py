import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

class DataGenerator:
    """
    生成ItemCF推荐系统的示例数据
    """
    def __init__(self, data_dir: str, num_users: int = 50, num_items: int = 30, 
                 interactions_per_user: tuple = (3, 10), num_categories: int = 5):
        """
        初始化数据生成器
        
        参数:
            data_dir: 数据保存目录
            num_users: 用户数量
            num_items: 物品数量
            interactions_per_user: 每个用户的交互记录数量范围
            num_categories: 物品类别数量
        """
        self.data_dir = data_dir
        self.num_users = num_users
        self.num_items = num_items
        self.interactions_per_user = interactions_per_user
        self.num_categories = num_categories
        
        # 确保数据目录存在
        os.makedirs(data_dir, exist_ok=True)
        
        # 定义数据参数
        self.user_ids = [1000 + i for i in range(num_users)]
        self.item_ids = [2000 + i for i in range(num_items)]
        
        # 物品类别和名称模板
        self.categories = ["电子产品", "家居用品", "服装配饰", "食品饮料", "图书文具"]
        self.item_type_templates = {
            "电子产品": ["智能手表", "无线耳机", "平板电脑", "蓝牙音箱", "移动电源"],
            "家居用品": ["保温杯", "抱枕", "收纳盒", "台灯", "香薰蜡烛"],
            "服装配饰": ["围巾", "帽子", "墨镜", "手链", "钱包"],
            "食品饮料": ["巧克力", "茶叶", "咖啡", "坚果礼盒", "果干"],
            "图书文具": ["笔记本", "钢笔", "小说", "工具书", "创意文具"]
        }
        
        # 交互类型和权重
        self.interaction_types = ["点击", "收藏", "加入购物车", "购买"]
        self.interaction_weights = [1, 2, 3, 5]  # 不同交互类型的重要性权重
        
        # 生成基础日期范围
        self.start_date = datetime(2025, 9, 1)
        self.end_date = datetime(2025, 9, 30)
    
    def generate_user_data(self) -> pd.DataFrame:
        """生成用户数据"""
        users = pd.DataFrame({
            'user_id': self.user_ids,
            'age': np.random.randint(18, 65, size=self.num_users),
            'gender': np.random.choice(['男', '女'], size=self.num_users, p=[0.55, 0.45])
        })
        return users
    
    def generate_item_data(self) -> pd.DataFrame:
        """生成物品数据"""
        item_categories = np.random.choice(self.categories[:self.num_categories], size=self.num_items)
        
        item_names = []
        for category in item_categories:
            # 为每个类别随机选择一个物品类型模板
            item_type = random.choice(self.item_type_templates[category])
            # 添加一些随机修饰词增加多样性
            adjectives = ["高级", "智能", "时尚", "经典", "迷你", "便携", "多功能", "高品质"]
            adj = random.choice(adjectives)
            item_names.append(f"{adj}{item_type}")
        
        items = pd.DataFrame({
            'item_id': self.item_ids,
            'item_name': item_names,
            'category': item_categories,
            'price': np.round(np.random.uniform(10, 1000, size=self.num_items), 2)
        })
        return items
    
    def generate_interaction_data(self, users: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
        """生成用户-物品交互数据"""
        interactions = []
        
        # 为每个用户生成交互记录
        for user_id in self.user_ids:
            # 随机决定该用户的交互记录数量
            num_interactions = random.randint(self.interactions_per_user[0], self.interactions_per_user[1])
            
            # 用户偏好的类别（基于性别）
            user_gender = users.loc[users['user_id'] == user_id, 'gender'].iloc[0]
            category_weights = np.ones(len(self.categories))
            
            # 根据性别稍微调整类别偏好
            if user_gender == '男':
                category_weights[self.categories.index("电子产品")] *= 1.5
                category_weights[self.categories.index("图书文具")] *= 1.2
            else:
                category_weights[self.categories.index("服装配饰")] *= 1.5
                category_weights[self.categories.index("家居用品")] *= 1.2
            
            category_weights = category_weights / category_weights.sum()
            
            # 为用户选择物品
            interacted_items = set()
            while len(interacted_items) < num_interactions:
                # 基于类别偏好选择物品
                preferred_category = np.random.choice(self.categories[:self.num_categories], p=category_weights)
                category_items = items[items['category'] == preferred_category]['item_id'].tolist()
                
                if category_items:
                    item_id = random.choice(category_items)
                    if item_id not in interacted_items:
                        interacted_items.add(item_id)
                        
                        # 生成交互类型（购买的概率较低）
                        interaction_type = random.choices(
                            self.interaction_types,
                            weights=[0.4, 0.2, 0.2, 0.2],  # 点击概率最高，购买次之
                            k=1
                        )[0]
                        
                        # 生成交互时间
                        delta_days = random.randint(0, (self.end_date - self.start_date).days)
                        interaction_time = (self.start_date + timedelta(days=delta_days)).strftime('%Y-%m-%d')
                        
                        interactions.append({
                            'user_id': user_id,
                            'item_id': item_id,
                            'interaction_type': interaction_type,
                            'interaction_time': interaction_time
                        })
        
        return pd.DataFrame(interactions)
    
    def save_data(self, users: pd.DataFrame, items: pd.DataFrame, interactions: pd.DataFrame) -> None:
        """保存数据到CSV文件"""
        # 保存用户数据
        user_path = os.path.join(self.data_dir, 'user_table.csv')
        users.to_csv(user_path, index=False)
        print(f"用户数据已保存到：{user_path}，共{len(users)}条记录")
        
        # 保存物品数据
        item_path = os.path.join(self.data_dir, 'item_table.csv')
        items.to_csv(item_path, index=False)
        print(f"物品数据已保存到：{item_path}，共{len(items)}条记录")
        
        # 保存交互数据
        interaction_path = os.path.join(self.data_dir, 'interaction_table.csv')
        interactions.to_csv(interaction_path, index=False)
        print(f"交互数据已保存到：{interaction_path}，共{len(interactions)}条记录")
    
    def generate_and_save_all(self) -> None:
        """生成并保存所有数据"""
        print(f"开始生成示例数据：{self.num_users}个用户，{self.num_items}个物品")
        
        # 生成用户数据
        users = self.generate_user_data()
        
        # 生成物品数据
        items = self.generate_item_data()
        
        # 生成交互数据
        interactions = self.generate_interaction_data(users, items)
        
        # 保存数据
        self.save_data(users, items, interactions)
        
        print("数据生成完成！")


# 主函数
if __name__ == "__main__":
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    
    # 创建数据生成器实例
    # 可以根据需要调整参数：用户数量、物品数量、每个用户的交互记录数量范围
    generator = DataGenerator(
        data_dir=data_dir,
        num_users=50,      # 50个用户
        num_items=30,      # 30个物品
        interactions_per_user=(3, 10),  # 每个用户3-10条交互记录
        num_categories=5   # 5个物品类别
    )
    
    # 生成并保存所有数据
    generator.generate_and_save_all()