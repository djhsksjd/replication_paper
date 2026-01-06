# UserCF Swing 推荐系统 Web界面

这是一个用于可视化UserCF Swing推荐算法完整流程的Web应用。

## 功能特点

- 📊 **数据可视化**：展示用户、物品和交互数据的统计信息
- 🔍 **算法流程展示**：可视化展示从数据加载到推荐生成的完整流程
- 👥 **相似用户分析**：展示目标用户的相似用户及其相似度
- 🎯 **个性化推荐**：生成推荐结果并显示推荐原因
- 📈 **系统评估**：评估推荐系统的效果指标

## 安装和运行

### 1. 安装依赖

```bash
cd recall/UserCF_Swing/web_app
pip install -r requirements.txt
```

### 2. 运行应用

```bash
python app.py
```

### 3. 访问界面

打开浏览器访问：http://127.0.0.1:5000

## 使用流程

1. **初始化系统**：点击"初始化系统"按钮加载数据
2. **计算相似度**：点击"计算相似度"按钮使用Swing算法计算用户相似度
3. **选择用户**：从下拉菜单中选择目标用户
4. **加载用户数据**：查看用户的交互历史和相似用户
5. **生成推荐**：点击"生成推荐"按钮获取个性化推荐结果
6. **系统评估**：点击"开始评估"查看推荐系统的效果指标

## 项目结构

```
web_app/
├── app.py                 # Flask后端应用
├── requirements.txt       # Python依赖
├── README.md             # 说明文档
├── templates/
│   └── index.html        # 前端HTML页面
└── static/
    ├── css/
    │   └── style.css     # 样式文件
    └── js/
        └── main.js       # JavaScript交互逻辑
```

## API接口

- `POST /api/init` - 初始化推荐系统
- `GET /api/status` - 获取算法状态
- `GET /api/data/stats` - 获取数据统计
- `GET /api/users` - 获取用户列表
- `POST /api/calculate_similarity` - 计算用户相似度
- `GET /api/user/<user_id>/interactions` - 获取用户交互历史
- `GET /api/user/<user_id>/similar_users` - 获取相似用户
- `GET /api/user/<user_id>/recommendations` - 获取推荐结果
- `POST /api/evaluate` - 评估推荐系统

## 技术栈

- **后端**：Flask (Python)
- **前端**：HTML5, CSS3, JavaScript
- **算法**：UserCF Swing推荐算法
- **数据**：Pandas, NumPy

## 注意事项

- 确保`code`目录下有完整的数据文件（user_table.csv, item_table.csv, interaction_table.csv）
- 首次运行可能需要重新计算相似度矩阵，耗时较长
- 建议使用Chrome或Firefox浏览器以获得最佳体验

