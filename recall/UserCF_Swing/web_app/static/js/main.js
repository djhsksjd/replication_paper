// 全局状态
let currentUserId = null;
let systemInitialized = false;
let similarityCalculated = false;

// API基础URL
const API_BASE = '';

// 初始化系统
async function initSystem() {
    try {
        updateStepStatus(1, 'processing', '正在初始化系统...');
        
        const response = await fetch(`${API_BASE}/api/init`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
            updateStepStatus(1, 'completed', '系统初始化成功');
            systemInitialized = true;
            
            // 加载数据统计
            await loadDataStats();
            
            // 加载用户列表
            await loadUserList();
            
            // 启用下一步按钮
            document.getElementById('calcSimBtn').disabled = false;
            
            showNotification('系统初始化成功！', 'success');
        } else {
            updateStepStatus(1, 'error', data.message || '初始化失败');
            showNotification('初始化失败: ' + data.message, 'error');
        }
    } catch (error) {
        updateStepStatus(1, 'error', '网络错误: ' + error.message);
        showNotification('初始化失败: ' + error.message, 'error');
    }
}

// 计算相似度
async function calculateSimilarity() {
    if (!systemInitialized) {
        showNotification('请先初始化系统', 'warning');
        return;
    }
    
    try {
        updateStepStatus(2, 'processing', '正在计算用户相似度...');
        
        const response = await fetch(`${API_BASE}/api/calculate_similarity`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                use_weights: true,
                return_steps: true
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            updateStepStatus(2, 'completed', `相似度计算完成，共计算了 ${data.num_users} 个用户的相似度`);
            similarityCalculated = true;
            
            // 显示计算步骤
            if (data.calculation_steps) {
                displaySimilaritySteps(data.calculation_steps);
            }
            
            // 启用推荐按钮
            document.getElementById('recommendBtn').disabled = false;
            document.getElementById('evaluateBtn').disabled = false;
            
            showNotification('相似度计算完成！', 'success');
        } else {
            updateStepStatus(2, 'error', data.error || '计算失败');
            showNotification('相似度计算失败: ' + data.error, 'error');
        }
    } catch (error) {
        updateStepStatus(2, 'error', '网络错误: ' + error.message);
        showNotification('计算失败: ' + error.message, 'error');
    }
}

// 加载数据统计
async function loadDataStats() {
    try {
        const response = await fetch(`${API_BASE}/api/data/stats`);
        const data = await response.json();
        
        document.getElementById('statUsers').textContent = data.num_users.toLocaleString();
        document.getElementById('statItems').textContent = data.num_items.toLocaleString();
        document.getElementById('statInteractions').textContent = data.num_interactions.toLocaleString();
        document.getElementById('statAvgInteractions').textContent = data.avg_interactions_per_user;
        
        document.getElementById('statsPanel').style.display = 'block';
    } catch (error) {
        console.error('加载数据统计失败:', error);
    }
}

// 加载用户列表
async function loadUserList() {
    try {
        const response = await fetch(`${API_BASE}/api/users`);
        const data = await response.json();
        
        const select = document.getElementById('userIdSelect');
        select.innerHTML = '<option value="">请选择用户...</option>';
        
        data.users.forEach(user => {
            const option = document.createElement('option');
            option.value = user.user_id;
            option.textContent = `用户 ${user.user_id} (${user.gender}, ${user.age}岁)`;
            select.appendChild(option);
        });
        
        updateStepStatus(3, 'waiting', '请选择用户');
    } catch (error) {
        console.error('加载用户列表失败:', error);
    }
}

// 加载用户数据
async function loadUserData() {
    const userId = document.getElementById('userIdSelect').value;
    
    if (!userId) {
        showNotification('请先选择用户', 'warning');
        return;
    }
    
    if (!similarityCalculated) {
        showNotification('请先计算相似度', 'warning');
        return;
    }
    
    currentUserId = parseInt(userId);
    
    try {
        updateStepStatus(3, 'processing', '正在加载用户数据...');
        
        // 加载用户交互历史
        await loadUserInteractions(userId);
        
        // 加载相似用户
        await loadSimilarUsers(userId);
        
        updateStepStatus(3, 'completed', `用户 ${userId} 数据加载完成`);
        document.getElementById('recommendBtn').disabled = false;
        
        showNotification('用户数据加载完成！', 'success');
    } catch (error) {
        updateStepStatus(3, 'error', '加载失败: ' + error.message);
        showNotification('加载失败: ' + error.message, 'error');
    }
}

// 加载用户交互历史
async function loadUserInteractions(userId) {
    try {
        const response = await fetch(`${API_BASE}/api/user/${userId}/interactions`);
        
        if (!response.ok) {
            throw new Error(`HTTP错误: ${response.status}`);
        }
        
        const data = await response.json();
        
        // 检查是否有错误
        if (data.error) {
            throw new Error(data.error);
        }
        
        const userInfo = document.getElementById('userInfo');
        userInfo.innerHTML = `
            <div style="display: flex; align-items: center; gap: 15px;">
                <div style="font-size: 2em;">👤</div>
                <div>
                    <div style="font-size: 1.5em; font-weight: bold;">用户 ${userId}</div>
                    <div style="color: #6b7280;">查看交互历史和相似用户</div>
                </div>
            </div>
        `;
        
        const interactionsList = document.getElementById('interactionsList');
        
        // 安全地检查interactions是否存在且为数组
        if (!data.interactions || !Array.isArray(data.interactions)) {
            interactionsList.innerHTML = '<p style="color: #6b7280;">该用户暂无交互历史数据</p>';
        } else if (data.interactions.length === 0) {
            interactionsList.innerHTML = '<p style="color: #6b7280;">该用户暂无交互历史</p>';
        } else {
            interactionsList.innerHTML = data.interactions.map(interaction => `
                <div class="interaction-item">
                    <div>
                        <div style="font-weight: 600;">${interaction.item_name || '物品 ' + interaction.item_id}</div>
                        <div style="font-size: 0.9em; color: #6b7280;">
                            ${interaction.item_category || ''} | ¥${interaction.item_price || '0'}
                        </div>
                    </div>
                    <div>
                        <span style="background: #667eea; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.85em;">
                            ${interaction.interaction_type || '未知'}
                        </span>
                    </div>
                </div>
            `).join('');
        }
        
        document.getElementById('userPanel').style.display = 'block';
    } catch (error) {
        console.error('加载交互历史失败:', error);
        const interactionsList = document.getElementById('interactionsList');
        if (interactionsList) {
            interactionsList.innerHTML = `<p style="color: #ef4444;">加载失败: ${error.message}</p>`;
        }
    }
}

// 加载相似用户
async function loadSimilarUsers(userId) {
    try {
        const response = await fetch(`${API_BASE}/api/user/${userId}/similar_users?top_k=5`);
        
        if (!response.ok) {
            throw new Error(`HTTP错误: ${response.status}`);
        }
        
        const data = await response.json();
        
        // 检查是否有错误
        if (data.error) {
            throw new Error(data.error);
        }
        
        const similarUsersList = document.getElementById('similarUsersList');
        
        // 安全地检查similar_users是否存在且为数组
        if (!data.similar_users || !Array.isArray(data.similar_users)) {
            similarUsersList.innerHTML = '<p style="color: #6b7280;">未找到相似用户数据</p>';
            return;
        }
        
        if (data.similar_users.length === 0) {
            similarUsersList.innerHTML = '<p style="color: #6b7280;">未找到相似用户</p>';
        } else {
            similarUsersList.innerHTML = data.similar_users.map((user, index) => `
                <div class="similar-user-item">
                    <div>
                        <div style="font-weight: 600;">${index + 1}. 用户 ${user.user_id}</div>
                        <div style="font-size: 0.9em; color: #6b7280;">
                            ${user.gender || '未知'} | ${user.age || '未知'}岁
                        </div>
                    </div>
                    <div>
                        <span style="background: #10b981; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 600;">
                            相似度: ${user.similarity || '0.0000'}
                        </span>
                    </div>
                </div>
            `).join('');
        }
    } catch (error) {
        console.error('加载相似用户失败:', error);
        const similarUsersList = document.getElementById('similarUsersList');
        if (similarUsersList) {
            similarUsersList.innerHTML = `<p style="color: #ef4444;">加载失败: ${error.message}</p>`;
        }
    }
}

// 生成推荐
async function generateRecommendations() {
    if (!currentUserId) {
        showNotification('请先选择用户', 'warning');
        return;
    }
    
    if (!similarityCalculated) {
        showNotification('请先计算相似度', 'warning');
        return;
    }
    
    try {
        updateStepStatus(4, 'processing', '正在生成推荐...');
        
        const response = await fetch(`${API_BASE}/api/user/${currentUserId}/recommendations?top_n=10&return_steps=true`);
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            const errorMsg = errorData.error || `HTTP错误: ${response.status}`;
            updateStepStatus(4, 'error', errorMsg);
            showNotification('生成推荐失败: ' + errorMsg, 'error');
            return;
        }
        
        const data = await response.json();
        
        // 检查是否有错误
        if (data.error) {
            updateStepStatus(4, 'error', data.error);
            showNotification('生成推荐失败: ' + data.error, 'error');
            
            // 即使有错误，也尝试显示空结果
            if (data.recommendations) {
                displayRecommendations(data.recommendations);
            }
            return;
        }
        
        // 检查推荐结果
        if (!data.recommendations || data.recommendations.length === 0) {
            const message = data.message || '未找到推荐结果';
            updateStepStatus(4, 'completed', message);
            showNotification(message, 'warning');
            displayRecommendations([]);
            return;
        }
        
        displayRecommendations(data.recommendations);
        
        // 显示计算步骤
        if (data.calculation_steps) {
            displayRecommendationSteps(data.calculation_steps);
        }
        
        updateStepStatus(4, 'completed', `成功生成 ${data.recommendations.length} 条推荐`);
        
        showNotification('推荐生成成功！', 'success');
    } catch (error) {
        console.error('生成推荐错误:', error);
        updateStepStatus(4, 'error', '网络错误: ' + error.message);
        showNotification('生成推荐失败: ' + error.message, 'error');
    }
}

// 显示推荐结果
function displayRecommendations(recommendations) {
    const recommendationsGrid = document.getElementById('recommendationsGrid');
    
    // 安全地检查recommendations是否存在且为数组
    if (!recommendations || !Array.isArray(recommendations)) {
        recommendationsGrid.innerHTML = '<p style="color: #6b7280;">暂无推荐结果数据</p>';
        return;
    }
    
    if (recommendations.length === 0) {
        recommendationsGrid.innerHTML = '<p style="color: #6b7280;">暂无推荐结果</p>';
    } else {
        recommendationsGrid.innerHTML = recommendations.map((rec, index) => `
            <div class="recommendation-card">
                <div class="recommendation-header">
                    <div class="recommendation-title">${index + 1}. ${rec.item_name}</div>
                    <div class="recommendation-score">${rec.score}</div>
                </div>
                <div class="recommendation-info">
                    <span><i class="fas fa-tag"></i> ${rec.category}</span>
                    <span><i class="fas fa-yen-sign"></i> ¥${rec.price}</span>
                </div>
                <div class="recommendation-reason">
                    <div class="reason-title">
                        <i class="fas fa-lightbulb"></i> 推荐原因
                    </div>
                    ${(rec.reason && rec.reason.similar_users && Array.isArray(rec.reason.similar_users)) 
                        ? rec.reason.similar_users.map(simUser => `
                        <div class="similar-user-reason">
                            <div style="font-weight: 600; margin-bottom: 5px;">
                                用户 ${simUser.user_id || '未知'} (${simUser.gender || '未知'}, ${simUser.age || '未知'}岁)
                            </div>
                            <div style="font-size: 0.85em; color: #6b7280;">
                                相似度: ${simUser.similarity || '0.0000'} | 
                                对该物品: ${simUser.interaction_type || '未知'} | 
                                贡献度: ${simUser.contribution || '0.0000'}
                            </div>
                        </div>
                    `).join('') : '<p style="color: #6b7280; font-size: 0.9em;">暂无相似用户信息</p>'}
                    ${(rec.reason && rec.reason.common_items && Array.isArray(rec.reason.common_items) && rec.reason.common_items.length > 0) ? `
                        <div class="common-items">
                            <i class="fas fa-link"></i> 共同喜欢: 
                            ${rec.reason.common_items.map(item => item.item_name || '未知物品').join(', ')}
                        </div>
                    ` : ''}
                </div>
            </div>
        `).join('');
    }
    
    document.getElementById('recommendationsPanel').style.display = 'block';
}

// 评估系统
async function evaluateSystem() {
    if (!similarityCalculated) {
        showNotification('请先计算相似度', 'warning');
        return;
    }
    
    try {
        updateStepStatus(5, 'processing', '正在评估系统...');
        
        const response = await fetch(`${API_BASE}/api/evaluate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                test_ratio: 0.2,
                top_n: 10
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            updateStepStatus(5, 'error', data.error);
            showNotification('评估失败: ' + data.error, 'error');
            return;
        }
        
        displayEvaluationResults(data.results);
        updateStepStatus(5, 'completed', '评估完成');
        
        showNotification('评估完成！', 'success');
    } catch (error) {
        updateStepStatus(5, 'error', '网络错误: ' + error.message);
        showNotification('评估失败: ' + error.message, 'error');
    }
}

// 显示评估结果
function displayEvaluationResults(results) {
    const evaluationResults = document.getElementById('evaluationResults');
    
    evaluationResults.innerHTML = Object.entries(results).map(([key, value]) => {
        let displayValue = value;
        if (typeof value === 'number' && value < 1 && value > 0) {
            displayValue = (value * 100).toFixed(2) + '%';
        } else if (typeof value === 'number') {
            displayValue = value.toFixed(2);
        }
        
        return `
            <div class="evaluation-card">
                <div class="evaluation-label">${key}</div>
                <div class="evaluation-value">${displayValue}</div>
            </div>
        `;
    }).join('');
    
    document.getElementById('evaluationPanel').style.display = 'block';
}

// 更新步骤状态
function updateStepStatus(stepNum, status, message) {
    const step = document.getElementById(`step${stepNum}`);
    const statusEl = document.getElementById(`status${stepNum}`);
    const detailsEl = document.getElementById(`details${stepNum}`);
    
    // 移除所有状态类
    step.classList.remove('active', 'completed', 'waiting');
    statusEl.classList.remove('waiting', 'processing', 'completed', 'error');
    
    // 添加新状态类
    if (status === 'processing') {
        step.classList.add('active');
        statusEl.classList.add('processing');
    } else if (status === 'completed') {
        step.classList.add('completed');
        statusEl.classList.add('completed');
    } else if (status === 'error') {
        statusEl.classList.add('error');
    } else {
        step.classList.add('waiting');
        statusEl.classList.add('waiting');
    }
    
    // 更新状态文本
    const statusText = {
        'waiting': '等待开始',
        'processing': '处理中...',
        'completed': '完成',
        'error': '错误'
    };
    
    statusEl.textContent = statusText[status] || status;
    
    // 更新详细信息
    if (detailsEl && message) {
        detailsEl.textContent = message;
        detailsEl.style.display = message ? 'block' : 'none';
    }
}

// 重置系统
function resetSystem() {
    if (confirm('确定要重置系统吗？这将清除所有数据。')) {
        location.reload();
    }
}

// 显示通知
function showNotification(message, type = 'info') {
    // 创建通知元素
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        z-index: 10000;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        animation: slideIn 0.3s ease;
    `;
    
    const colors = {
        'success': '#10b981',
        'error': '#ef4444',
        'warning': '#f59e0b',
        'info': '#3b82f6'
    };
    
    notification.style.background = colors[type] || colors.info;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // 3秒后自动移除
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// 添加CSS动画
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// 显示相似度计算步骤
function displaySimilaritySteps(steps) {
    const panel = document.getElementById('calculationStepsPanel');
    const container = document.getElementById('calculationSteps');
    
    if (!steps || !steps.example_steps) {
        return;
    }
    
    let html = `
        <div class="step-summary">
            <h3>📊 计算摘要</h3>
            <div class="summary-grid">
                <div class="summary-item">
                    <span class="summary-label">处理物品数:</span>
                    <span class="summary-value">${steps.total_items}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">用户对数:</span>
                    <span class="summary-value">${steps.total_pairs}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">用户数:</span>
                    <span class="summary-value">${steps.num_users}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">耗时:</span>
                    <span class="summary-value">${steps.time_cost}秒</span>
                </div>
            </div>
        </div>
        
        <div class="steps-detail">
            <h3>🔍 计算步骤示例（前3个）</h3>
    `;
    
    steps.example_steps.forEach((step, index) => {
        html += `
            <div class="calculation-step-card">
                <div class="step-number">步骤 ${index + 1}</div>
                <div class="step-content-detail">
                    <div class="step-row">
                        <span class="step-label">物品ID:</span>
                        <span class="step-value">${step.item_id}</span>
                    </div>
                    <div class="step-row">
                        <span class="step-label">用户对:</span>
                        <span class="step-value">用户 ${step.user_u} ↔ 用户 ${step.user_v}</span>
                    </div>
                    <div class="step-row">
                        <span class="step-label">交互该物品的用户数:</span>
                        <span class="step-value">${step.user_count}</span>
                    </div>
                    <div class="step-row">
                        <span class="step-label">热门惩罚系数:</span>
                        <span class="step-value">1 / log(1 + ${step.user_count}) = ${step.penalty}</span>
                    </div>
        `;
        
        if (step.weights && Object.keys(step.weights).length > 0) {
            html += `
                    <div class="step-row">
                        <span class="step-label">交互权重:</span>
                        <div class="step-nested">
                            <div>用户 ${step.user_u}: ${step.weights.user_u.type} (权重: ${step.weights.user_u.weight})</div>
                            <div>用户 ${step.user_v}: ${step.weights.user_v.type} (权重: ${step.weights.user_v.weight})</div>
                            <div>组合权重: ${step.weights.user_u.weight} × ${step.weights.user_v.weight} = ${step.weights.combined}</div>
                        </div>
                    </div>
            `;
        }
        
        html += `
                    <div class="step-row">
                        <span class="step-label">贡献值:</span>
                        <span class="step-value">${step.base_contribution}${step.weights.combined ? ' × ' + step.weights.combined : ''} = ${step.final_contribution}</span>
                    </div>
                    <div class="step-row">
                        <span class="step-label">相似度更新:</span>
                        <span class="step-value">${step.similarity_before} + ${step.final_contribution} = ${step.similarity_after}</span>
                    </div>
                </div>
            </div>
        `;
    });
    
    html += `</div>`;
    
    container.innerHTML = html;
    panel.style.display = 'block';
}

// 显示推荐计算步骤
function displayRecommendationSteps(steps) {
    const panel = document.getElementById('calculationStepsPanel');
    const container = document.getElementById('calculationSteps');
    
    if (!steps || !steps.steps) {
        return;
    }
    
    let html = `<h3>🎯 为用户 ${steps.user_id} 生成推荐的详细步骤</h3>`;
    
    steps.steps.forEach((step, index) => {
        if (step.error) {
            html += `
                <div class="calculation-step-card error">
                    <div class="step-number">步骤 ${step.step}</div>
                    <div class="step-content-detail">
                        <div class="step-description">${step.description}</div>
                    </div>
                </div>
            `;
            return;
        }
        
        html += `
            <div class="calculation-step-card">
                <div class="step-number">步骤 ${step.step}</div>
                <div class="step-content-detail">
                    <div class="step-description">${step.description}</div>
        `;
        
        // 根据步骤类型显示不同内容
        if (step.step === 1 && step.user_items) {
            html += `
                    <div class="step-row">
                        <span class="step-label">用户交互的物品:</span>
                        <span class="step-value">${step.user_items.join(', ')}${step.total_items > 10 ? ' ... (共' + step.total_items + '个)' : ''}</span>
                    </div>
            `;
        } else if (step.step === 2 && step.similar_users) {
            html += `<div class="step-row"><span class="step-label">Top相似用户:</span></div>`;
            step.similar_users.forEach(sim => {
                html += `
                    <div class="step-nested">
                        <div>用户 ${sim.user_id}: 相似度 = ${sim.similarity}</div>
                    </div>
                `;
            });
        } else if (step.step === 3 && step.common_items_example) {
            html += `
                    <div class="step-row">
                        <span class="step-label">示例（用户 ${step.common_items_example.similar_user}）:</span>
                        <span class="step-value">${step.common_items_example.common_items.join(', ') || '无共同物品'}</span>
                    </div>
            `;
        } else if (step.step === 4 && step.calculation_example) {
            html += `<div class="step-row"><span class="step-label">计算示例（前5个）:</span></div>`;
            step.calculation_example.forEach(calc => {
                html += `
                    <div class="step-nested">
                        <div><strong>物品 ${calc.item_id}</strong></div>
                        <div>相似用户 ${calc.similar_user} (相似度: ${calc.similarity}) × 交互权重 ${calc.weight} (${calc.interaction_type}) = 贡献 ${calc.contribution}</div>
                        <div>物品分数: ${calc.item_score_before} + ${calc.contribution} = ${calc.item_score_after}</div>
                    </div>
                `;
            });
        } else if (step.step === 5 && step.top_items) {
            html += `<div class="step-row"><span class="step-label">Top推荐物品:</span></div>`;
            step.top_items.forEach((item, idx) => {
                html += `
                    <div class="step-nested">
                        <div>${idx + 1}. 物品 ${item.item_id}: 分数 = ${item.score}</div>
                    </div>
                `;
            });
        }
        
        html += `
                </div>
            </div>
        `;
    });
    
    container.innerHTML = html;
    panel.style.display = 'block';
}

// 页面加载完成后的初始化
document.addEventListener('DOMContentLoaded', function() {
    // 初始化所有步骤状态
    for (let i = 1; i <= 5; i++) {
        updateStepStatus(i, 'waiting', '');
    }
    
    console.log('UserCF Swing 推荐系统界面已加载');
});

