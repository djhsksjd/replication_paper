# 读取 .mat 文件并打印其内容
import scipy.io
import os
# 使用相对路径访问文件
file_path = './epinions/rating.mat'

# 检查文件是否存在
if os.path.exists(file_path):
    mat_data = scipy.io.loadmat(file_path)
    print(mat_data)
    # 查看MAT文件中的变量
    print("MAT 文件中的变量：", mat_data.keys())
else:
    print(f"文件不存在: {file_path}")
