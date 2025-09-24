# 读取epinions数据集
import scipy.io
import numpy as np
import pandas as pd
from pathlib import Path  # 更优雅地处理路径


def extract_mat_variables(mat_file_path):
    """
    读取.mat文件并返回所有变量的字典
    
    参数:
        mat_file_path: .mat文件的路径
        
    返回:
        variables: 包含所有变量的字典（键为变量名，值为数据）
    """
    # 加载.mat文件（返回字典，键为变量名，值为数据）
    mat_data = scipy.io.loadmat(mat_file_path)
    
    # 过滤MATLAB自动添加的元数据（以'__'开头和结尾的键）
    variables = {
        key: value for key, value in mat_data.items()
        if not (key.startswith('__') and key.endswith('__'))
    }
    
    return variables
    
# 2. 加载.mat文件并提取rating数组
def get_data(file_path):
    mat_variables = extract_mat_variables(file_path)
    # 加载文件（自动过滤__header__等元数据，仅关注用户变量）

    print("从",file_path,".mat文件中提取的变量：", list(mat_variables.keys()))
    if 'rating' in mat_variables:
        mat_data = scipy.io.loadmat(file_path)
    
        # 提取rating数组（关键步骤：通过键名'rating'获取二维数组）
        rating_array = mat_data['rating']  # 此时rating_array是numpy.ndarray类型
        
        # 3. 查看数据基本信息（验证是否正确提取）
        print("=== rating数组基本信息 ===")
        print(f"数组形状：{rating_array.shape}")  # 输出 (922267, 4)，确认行数和列数
        print(f"数据类型：{rating_array.dtype}")  # 输出 int32，确认数据类型
        print(f"前5行数据：\n{rating_array[:5]}")  # 查看前5行，了解列的含义
        
        # 4. 按GraphRec需求处理数据（关键：提取user_id、item_id、rating三列）
        # 假设数组列顺序：第0列=user_id，第1列=item_id，第2列=rating，第3列=额外信息（可忽略）
        rating_core = rating_array[:, [0, 1, 2, 3]]  # 只保留前3列核心数据
        
        # 5. 转换为DataFrame（便于后续预处理，如构建用户/物品索引映射）
        rating_df = pd.DataFrame(
            rating_core,  # 核心数据（user_id, item_id, rating）
            columns=["user_id", "item_id", "category","rating"]  # 列名，对应GraphRec原始数据格式
        )
        
        # 6. 保存为文本文件（存入data/raw/，符合之前的项目结构）
        save_path = Path("./data/raw/epinions_ratings.txt")
        save_path.parent.mkdir(parents=True, exist_ok=True)  # 确保文件夹存在
        rating_df.to_csv(
            save_path,
            sep="\t",  # 用制表符分隔，和论文原始数据格式一致
            index=False,  # 不保存行索引
            header=False  # 不保存列名（符合之前定义的原始数据格式）
        )
        
        print(f"\n=== 处理完成 ===")
        print(f"核心数据形状：{rating_core.shape}")  # 输出 (922267, 4)
        print(f"DataFrame前100行：\n{rating_df.head()}")
        print(f"文本文件已保存到：{save_path}")

    
    if 'trustnetwork' in mat_variables:
        social = mat_variables['trustnetwork']    # 社交数据（numpy数组）
        
        mat_data = scipy.io.loadmat(file_path)
    
        # 提取rating数组（关键步骤：通过键名'rating'获取二维数组）
        trustnetwork_array = mat_data['trustnetwork']  # 此时rating_array是numpy.ndarray类型
        
        # 3. 查看数据基本信息（验证是否正确提取）
        print("=== trustnetwork数组基本信息 ===")
        print(f"数组形状：{trustnetwork_array.shape}") 
        print(f"数据类型：{trustnetwork_array.dtype}")  # 输出 int32，确认数据类型
        print(f"前5行数据：\n{trustnetwork_array[:5]}")  # 查看前5行，了解列的含义
        
        # 4. 按GraphRec需求处理数据（关键：提取user_id、item_id、rating三列）
        # 假设数组列顺序：第0列=user_id，第1列=item_id，第2列=rating，第3列=额外信息（可忽略）
        trustnetwork_core = trustnetwork_array[:, [0, 1]]  # 只保留前3列核心数据
        
        # # 5. 转换为DataFrame（便于后续预处理，如构建用户/物品索引映射）
        trustnetwork_df = pd.DataFrame(
            trustnetwork_core,  # 核心数据（user_id, item_id, rating）
            columns=["user1_id", "user2_id"]  # 列名，对应GraphRec原始数据格式
        )
        # 6. 保存为文本文件（存入data/raw/，符合之前的项目结构）
        save_path = Path("./data/raw/epinions_trustnetwork.txt")
        save_path.parent.mkdir(parents=True, exist_ok=True)  # 确保文件夹存在
        trustnetwork_df.to_csv(
            save_path,
            sep="\t",  # 用制表符分隔，和论文原始数据格式一致
            index=False,  # 不保存行索引
            header=False  # 不保存列名（符合之前定义的原始数据格式）
        )
        
        print(f"\n=== 处理完成 ===")
        print("user 1 trust user 2")
        print(f"核心数据形状：{trustnetwork_core.shape}")  # 输出 (922267, 4)
        print(f"DataFrame前100行：\n{trustnetwork_df.head()}")
        print(f"文本文件已保存到：{save_path}")

# 1. 定义文件路径（替换为你的实际路径）
trustnetwork_file_path = Path("./epinions/trustnetwork.mat")  # 相对路径，可根据实际位置调整
rating_file_path = Path("./epinions/rating.mat")  # 相对路径，可根据实际位置调整


get_data(rating_file_path)
get_data(trustnetwork_file_path)
    
# *********
    # 
    
    
