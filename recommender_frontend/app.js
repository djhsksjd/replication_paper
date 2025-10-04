// 应用主逻辑模块
import { mockUsers, mockItems, mockInteractions } from './mock_data.js';
import { createRecommender, getAvailableAlgorithms } from './algorithms.js';

// 全局变量
let currentRecommender = null;
let performanceChart = null;
let currentAlgorithm = 'itemcf';

// DOM元素初始化
function initDOM() {
    // 初始化用户ID输入框
    document.getElementById('user-id').value = 1; // 默认选择用户1
    
    // 初始化推荐数量输入框
    document.getElementById('top-n').value = 5;
}

// 事件监听初始化
function initEventListeners() {
    // 算法选择按钮点击事件
    document.querySelectorAll('.algorithm-btn').forEach(button => {
        button.addEventListener('click', () => {
            // 移除所有按钮的active类
            document.querySelectorAll('.algorithm-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            // 为当前点击的按钮添加active类
            button.classList.add('active');
            // 更新当前算法
            currentAlgorithm = button.dataset.algorithm;
            // 生成推荐
            generateRecommendations();
        });
    });
    
    // 加载用户按钮点击事件
    document.getElementById('load-user-btn').addEventListener('click', () => {
        updateUserInfo();
    });
    
    // 用户ID输入框回车事件
    document.getElementById('user-id').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            updateUserInfo();
        }
    });
    
    // 生成推荐按钮点击事件
    document.getElementById('generate-rec-btn').addEventListener('click', () => {
        generateRecommendations();
    });
    
    // 推荐数量输入框回车事件
    document.getElementById('top-n').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            generateRecommendations();
        }
    });
    
    // 交互模拟按钮点击事件
    document.getElementById('simulate-click-btn').addEventListener('click', () => {
        simulateUserInteraction('点击');
    });
    
    document.getElementById('simulate-collect-btn').addEventListener('click', () => {
        simulateUserInteraction('收藏');
    });
    
    document.getElementById('simulate-cart-btn').addEventListener('click', () => {
        simulateUserInteraction('加入购物车');
    });
    
    document.getElementById('simulate-purchase-btn').addEventListener('click', () => {
        simulateUserInteraction('购买');
    });
}

// 更新用户信息
function updateUserInfo() {
    const userId = parseInt(document.getElementById('user-id').value);
    const user = mockUsers.find(u => u.user_id === userId);
    
    if (user) {
        document.getElementById('user-info').innerHTML = `
            <p>用户ID: ${user.user_id}</p>
            <p>年龄: ${user.age}</p>
            <p>性别: ${user.gender}</p>
            <p>偏好类别: ${user.preferred_categories.join(', ')}</p>
        `;
        
        // 更新用户交互历史
        updateUserInteractions(userId);
        
        // 自动生成推荐
        generateRecommendations();
    } else {
        document.getElementById('user-info').innerHTML = '<p class="error">用户不存在</p>';
        document.getElementById('interaction-history').innerHTML = '';
        document.getElementById('recommendations').innerHTML = '';
    }
}

// 更新用户交互历史
function updateUserInteractions(userId) {
    const userInteractions = mockInteractions.filter(i => i.user_id === userId);
    const interactionList = document.getElementById('interaction-history');
    
    interactionList.innerHTML = '';
    
    if (userInteractions.length === 0) {
        interactionList.innerHTML = '<p class="no-data">暂无交互历史</p>';
        return;
    }
    
    // 按时间倒序排序
    userInteractions.sort((a, b) => new Date(b.interaction_time) - new Date(a.interaction_time));
    
    userInteractions.forEach(interaction => {
        const item = mockItems.find(i => i.item_id === interaction.item_id);
        if (item) {
            const div = document.createElement('div');
            div.className = 'interaction-item';
            div.innerHTML = `
                <span class="item-name">${item.item_name}</span>
                <span class="interaction-type">${interaction.interaction_type}</span>
                <span class="interaction-time">${formatDate(interaction.interaction_time)}</span>
                <span class="item-category">${item.category}</span>
            `;
            interactionList.appendChild(div);
        }
    });
}

// 生成推荐结果
function generateRecommendations() {
    const userId = parseInt(document.getElementById('user-id').value);
    const user = mockUsers.find(u => u.user_id === userId);
    const topN = parseInt(document.getElementById('top-n').value);
    
    if (!user) {
        document.getElementById('recommendations').innerHTML = '<p class="error">请先选择有效用户</p>';
        return;
    }
    
    try {
        // 创建推荐器实例
        currentRecommender = createRecommender(currentAlgorithm);
        
        // 生成推荐结果
        const recommendations = currentRecommender.recommendItems(userId, topN);
        
        // 显示推荐结果
        displayRecommendations(recommendations);
        
        // 计算并显示性能指标
        const metrics = currentRecommender.evaluate([userId], topN);
        displayMetrics(metrics);
        
        // 更新性能对比图表
        updatePerformanceChart(metrics);
        
    } catch (error) {
        console.error('生成推荐失败:', error);
        document.getElementById('recommendations').innerHTML = `<p class="error">生成推荐失败: ${error.message}</p>`;
    }
}

// 显示推荐结果
function displayRecommendations(recommendations) {
    const recommendationsList = document.getElementById('recommendations');
    recommendationsList.innerHTML = '';
    
    if (recommendations.length === 0) {
        recommendationsList.innerHTML = '<p class="no-data">暂无推荐物品</p>';
        return;
    }
    
    recommendations.forEach((rec, index) => {
        const div = document.createElement('div');
        div.className = 'recommendation-item';
        div.innerHTML = `
            <div class="rank">${index + 1}</div>
            <div class="item-info">
                <h4>${rec.item.item_name || `物品${rec.item_id}`}</h4>
                <p class="category">类别: ${rec.item.category || '未知'}</p>
                <p class="price">价格: ¥${rec.item.price || '未知'}</p>
            </div>
            <div class="score">推荐分数: ${rec.score.toFixed(4)}</div>
        `;
        recommendationsList.appendChild(div);
    });
}

// 显示性能指标
function displayMetrics(metrics) {
    // 计算F1值
    const precision = metrics.avg_precision || 0;
    const recall = metrics.avg_recall || 0;
    const f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0;
    
    document.getElementById('recall-value').textContent = `${(recall * 100).toFixed(2)}%`;
    document.getElementById('precision-value').textContent = `${(precision * 100).toFixed(2)}%`;
    document.getElementById('f1-value').textContent = `${(f1 * 100).toFixed(2)}%`;
}

// 初始化性能对比图表
function initPerformanceChart() {
    const ctx = document.getElementById('performance-chart').getContext('2d');
    
    performanceChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['召回率', '精确率', 'F1值'],
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    min: 0,
                    max: 1,
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'bottom'
                },
                title: {
                    display: true,
                    text: '算法性能对比'
                }
            }
        }
    });
}

// 更新性能对比图表
function updatePerformanceChart(metrics) {
    if (!performanceChart) {
        return;
    }
    
    const precision = metrics.avg_precision || 0;
    const recall = metrics.avg_recall || 0;
    const f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0;
    
    // 获取算法名称
    const algorithmNames = {
        'itemcf': 'ItemCF',
        'usercf_swing': 'UserCF+Swing',
        'content_based': '基于内容'
    };
    
    const algorithmName = algorithmNames[currentAlgorithm] || currentAlgorithm;
    
    // 检查是否已存在该算法的数据
    let datasetIndex = -1;
    for (let i = 0; i < performanceChart.data.datasets.length; i++) {
        if (performanceChart.data.datasets[i].label === algorithmName) {
            datasetIndex = i;
            break;
        }
    }
    
    // 定义颜色
    const colors = {
        'itemcf': 'rgba(255, 99, 132, 0.2)',
        'usercf_swing': 'rgba(54, 162, 235, 0.2)',
        'content_based': 'rgba(75, 192, 192, 0.2)'
    };
    
    const borderColors = {
        'itemcf': 'rgba(255, 99, 132, 1)',
        'usercf_swing': 'rgba(54, 162, 235, 1)',
        'content_based': 'rgba(75, 192, 192, 1)'
    };
    
    const data = [recall, precision, f1];
    
    if (datasetIndex !== -1) {
        // 更新现有数据
        performanceChart.data.datasets[datasetIndex].data = data;
    } else {
        // 添加新数据
        performanceChart.data.datasets.push({
            label: algorithmName,
            data: data,
            backgroundColor: colors[currentAlgorithm] || 'rgba(200, 200, 200, 0.2)',
            borderColor: borderColors[currentAlgorithm] || 'rgba(200, 200, 200, 1)',
            borderWidth: 2,
            pointBackgroundColor: borderColors[currentAlgorithm] || 'rgba(200, 200, 200, 1)'
        });
    }
    
    performanceChart.update();
}

// 模拟用户交互
function simulateUserInteraction(interactionType) {
    const userId = parseInt(document.getElementById('user-id').value);
    const user = mockUsers.find(u => u.user_id === userId);
    
    if (!user) {
        showSimulationFeedback('请先选择有效用户', 'error');
        return;
    }
    
    // 获取推荐结果中的物品
    const recommendationItems = document.querySelectorAll('.recommendation-item');
    
    if (recommendationItems.length === 0) {
        showSimulationFeedback('请先生成推荐结果', 'error');
        return;
    }
    
    // 随机选择一个推荐物品
    const randomIndex = Math.floor(Math.random() * recommendationItems.length);
    const selectedItem = recommendationItems[randomIndex];
    const itemName = selectedItem.querySelector('h4').textContent;
    
    // 查找物品ID
    const item = mockItems.find(i => i.item_name === itemName);
    
    if (!item) {
        showSimulationFeedback('无法找到物品信息', 'error');
        return;
    }
    
    // 检查用户是否已经与该物品交互过
    const hasInteracted = mockInteractions.some(
        i => i.user_id === userId && i.item_id === item.item_id && i.interaction_type === interactionType
    );
    
    if (hasInteracted) {
        showSimulationFeedback(`用户已经${interactionType}过${itemName}`, 'info');
        return;
    }
    
    // 创建新的交互记录
    const newInteraction = {
        user_id: userId,
        item_id: item.item_id,
        interaction_type: interactionType,
        interaction_time: new Date().toISOString()
    };
    
    // 添加到模拟数据中
    mockInteractions.push(newInteraction);
    
    // 更新显示
    updateUserInteractions(userId);
    
    // 显示反馈
    showSimulationFeedback(`已模拟用户${userId}${interactionType}了物品：${itemName}`, 'success');
    
    // 自动重新生成推荐
    generateRecommendations();
}

// 显示模拟反馈
function showSimulationFeedback(message, type = 'info') {
    const feedbackElement = document.getElementById('simulation-feedback');
    
    feedbackElement.textContent = message;
    feedbackElement.className = 'feedback';
    feedbackElement.classList.add(type);
    
    // 3秒后清除反馈
    setTimeout(() => {
        feedbackElement.textContent = '';
        feedbackElement.className = 'feedback';
    }, 3000);
}

// 格式化日期
function formatDate(dateString) {
    const date = new Date(dateString);
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    
    return `${year}-${month}-${day} ${hours}:${minutes}`;
}

// 应用初始化
function initApp() {
    // 初始化DOM元素
    initDOM();
    
    // 初始化事件监听
    initEventListeners();
    
    // 初始化性能图表
    if (typeof Chart !== 'undefined') {
        initPerformanceChart();
    }
    
    // 初始化显示
    updateUserInfo();
}

// 页面加载完成后初始化应用
window.addEventListener('DOMContentLoaded', initApp);