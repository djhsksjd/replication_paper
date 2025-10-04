// 推荐算法实现模块
import { mockUsers, mockItems, mockInteractions } from './mock_data.js';

// 基础数据处理函数
function getUserItemMatrix(interactions) {
    const users = [...new Set(interactions.map(i => i.user_id))];
    const items = [...new Set(interactions.map(i => i.item_id))];
    
    const matrix = {};
    users.forEach(user => {
        matrix[user] = {};
        items.forEach(item => {
            matrix[user][item] = 0;
        });
    });
    
    interactions.forEach(interaction => {
        const weight = getInteractionWeight(interaction.interaction_type);
        matrix[interaction.user_id][interaction.item_id] = weight;
    });
    
    return matrix;
}

function getInteractionWeight(interactionType) {
    const weightMap = {
        "点击": 1,
        "收藏": 2,
        "加入购物车": 3,
        "购买": 5
    };
    return weightMap[interactionType] || 0;
}

// ItemCF推荐算法实现
class ItemCFRecommender {
    constructor() {
        this.userItemMatrix = null;
        this.itemSimilarity = null;
        this.items = mockItems;
        this.interactions = mockInteractions;
        this.buildModel();
    }
    
    buildModel() {
        // 构建用户-物品矩阵
        this.userItemMatrix = getUserItemMatrix(this.interactions);
        // 计算物品相似度
        this.calculateItemSimilarity();
    }
    
    calculateItemSimilarity() {
        const items = Object.keys(Object.values(this.userItemMatrix)[0] || {});
        const itemVectors = {};
        
        // 构建物品向量
        items.forEach(item => {
            itemVectors[item] = Object.values(this.userItemMatrix).map(userVector => userVector[item]);
        });
        
        // 计算余弦相似度
        this.itemSimilarity = {};
        items.forEach(itemA => {
            this.itemSimilarity[itemA] = {};
            items.forEach(itemB => {
                if (itemA === itemB) {
                    this.itemSimilarity[itemA][itemB] = 1;
                    return;
                }
                
                const vecA = itemVectors[itemA];
                const vecB = itemVectors[itemB];
                
                let dotProduct = 0;
                let normA = 0;
                let normB = 0;
                
                for (let i = 0; i < vecA.length; i++) {
                    dotProduct += vecA[i] * vecB[i];
                    normA += vecA[i] * vecA[i];
                    normB += vecB[i] * vecB[i];
                }
                
                if (normA === 0 || normB === 0) {
                    this.itemSimilarity[itemA][itemB] = 0;
                } else {
                    this.itemSimilarity[itemA][itemB] = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
                }
            });
        });
    }
    
    recommendItems(user_id, top_n = 5, filter_interacted = true) {
        // 获取用户交互过的物品
        const userInteractedItems = {};
        this.interactions
            .filter(interaction => interaction.user_id === user_id)
            .forEach(interaction => {
                userInteractedItems[interaction.item_id] = getInteractionWeight(interaction.interaction_type);
            });
        
        // 如果用户没有交互历史，返回空数组
        if (Object.keys(userInteractedItems).length === 0) {
            return [];
        }
        
        // 计算候选物品分数
        const candidateScores = {};
        Object.entries(userInteractedItems).forEach(([itemId, weight]) => {
            Object.entries(this.itemSimilarity[itemId] || {}).forEach(([similarItemId, similarity]) => {
                if (filter_interacted && userInteractedItems[similarItemId]) {
                    return;
                }
                candidateScores[similarItemId] = (candidateScores[similarItemId] || 0) + similarity * weight;
            });
        });
        
        // 按分数降序排序并返回Top-N
        return Object.entries(candidateScores)
            .map(([itemId, score]) => ({
                item_id: parseInt(itemId),
                score: score,
                item: this.items.find(item => item.item_id === parseInt(itemId)) || {}
            }))
            .sort((a, b) => b.score - a.score)
            .slice(0, top_n);
    }
    
    evaluate(testUserIds = null, top_n = 5) {
        const activeUsers = [...new Set(this.interactions.map(i => i.user_id))];
        if (activeUsers.length === 0) {
            return { test_users: 0, avg_recall: 0, avg_precision: 0 };
        }
        
        // 选择测试用户
        const testUsers = testUserIds || activeUsers.slice(0, Math.max(1, Math.floor(activeUsers.length * 0.2)));
        
        let totalRecall = 0;
        let totalPrecision = 0;
        let validUsers = 0;
        
        testUsers.forEach(user_id => {
            // 真实交互物品
            const realItems = new Set(
                this.interactions
                    .filter(interaction => interaction.user_id === user_id)
                    .map(interaction => interaction.item_id)
            );
            
            if (realItems.size < 2) {
                return;
            }
            
            // 推荐物品
            const recommendedItems = new Set(
                this.recommendItems(user_id, top_n)
                    .map(recommendation => recommendation.item_id)
            );
            
            if (recommendedItems.size === 0) {
                return;
            }
            
            // 计算命中物品
            let hits = 0;
            recommendedItems.forEach(itemId => {
                if (realItems.has(itemId)) {
                    hits++;
                }
            });
            
            validUsers++;
            totalRecall += hits / realItems.size;
            totalPrecision += hits / recommendedItems.size;
        });
        
        return {
            test_users: validUsers,
            avg_recall: validUsers > 0 ? Math.round(totalRecall / validUsers * 10000) / 10000 : 0,
            avg_precision: validUsers > 0 ? Math.round(totalPrecision / validUsers * 10000) / 10000 : 0
        };
    }
}

// UserCF+Swing推荐算法实现
class UserCFSwingRecommender {
    constructor() {
        this.userSimilarity = null;
        this.itemToUsers = null;
        this.users = mockUsers;
        this.items = mockItems;
        this.interactions = mockInteractions;
        this.buildModel();
    }
    
    buildModel() {
        // 构建物品到用户的倒排表
        this.buildItemToUsers();
        // 计算用户相似度
        this.calculateUserSimilarity();
    }
    
    buildItemToUsers() {
        this.itemToUsers = {};
        
        this.interactions.forEach(interaction => {
            const { item_id, user_id } = interaction;
            if (!this.itemToUsers[item_id]) {
                this.itemToUsers[item_id] = [];
            }
            if (!this.itemToUsers[item_id].includes(user_id)) {
                this.itemToUsers[item_id].push(user_id);
            }
        });
    }
    
    calculateUserSimilarity() {
        this.userSimilarity = {};
        
        // 遍历每个物品的用户列表
        Object.entries(this.itemToUsers).forEach(([itemId, users]) => {
            const userCount = users.length;
            // 计算热门惩罚项
            const penalty = 1.0 / Math.log(1 + userCount);
            
            // 遍历用户对，计算相似度贡献
            for (let i = 0; i < users.length; i++) {
                for (let j = i + 1; j < users.length; j++) {
                    const u = users[i];
                    const v = users[j];
                    
                    // 初始化相似度对象
                    if (!this.userSimilarity[u]) {
                        this.userSimilarity[u] = {};
                    }
                    if (!this.userSimilarity[v]) {
                        this.userSimilarity[v] = {};
                    }
                    
                    // 累加相似度贡献
                    this.userSimilarity[u][v] = (this.userSimilarity[u][v] || 0) + penalty;
                    this.userSimilarity[v][u] = (this.userSimilarity[v][u] || 0) + penalty;
                }
            }
        });
    }
    
    recommendItems(user_id, top_n = 5, k_similar_users = 50) {
        // 获取用户已交互的物品集合
        const userItems = new Set(
            this.interactions
                .filter(interaction => interaction.user_id === user_id)
                .map(interaction => interaction.item_id)
        );
        
        if (userItems.size === 0) {
            return [];
        }
        
        // 获取相似用户
        const similarUsers = Object.entries(this.userSimilarity[user_id] || {})
            .sort((a, b) => b[1] - a[1])
            .slice(0, k_similar_users);
        
        if (similarUsers.length === 0) {
            return [];
        }
        
        // 计算候选物品分数
        const itemScores = {};
        similarUsers.forEach(([similarUserId, similarity]) => {
            // 获取相似用户交互过的物品
            const similarUserInteractions = this.interactions
                .filter(interaction => interaction.user_id === parseInt(similarUserId));
            
            similarUserInteractions.forEach(interaction => {
                const { item_id, interaction_type } = interaction;
                // 过滤掉用户已交互的物品
                if (!userItems.has(item_id)) {
                    const weight = getInteractionWeight(interaction_type);
                    itemScores[item_id] = (itemScores[item_id] || 0) + similarity * weight;
                }
            });
        });
        
        // 按分数排序并返回Top-N
        return Object.entries(itemScores)
            .map(([itemId, score]) => ({
                item_id: parseInt(itemId),
                score: score,
                item: this.items.find(item => item.item_id === parseInt(itemId)) || {}
            }))
            .sort((a, b) => b.score - a.score)
            .slice(0, top_n);
    }
    
    evaluate(testUserIds = null, top_n = 5) {
        const activeUsers = [...new Set(this.interactions.map(i => i.user_id))];
        if (activeUsers.length === 0) {
            return { test_users: 0, avg_recall: 0, avg_precision: 0 };
        }
        
        // 选择测试用户
        const testUsers = testUserIds || activeUsers.slice(0, Math.max(1, Math.floor(activeUsers.length * 0.2)));
        
        let totalRecall = 0;
        let totalPrecision = 0;
        let validUsers = 0;
        
        testUsers.forEach(user_id => {
            // 真实交互物品
            const realItems = new Set(
                this.interactions
                    .filter(interaction => interaction.user_id === user_id)
                    .map(interaction => interaction.item_id)
            );
            
            if (realItems.size < 2) {
                return;
            }
            
            // 推荐物品
            const recommendedItems = new Set(
                this.recommendItems(user_id, top_n)
                    .map(recommendation => recommendation.item_id)
            );
            
            if (recommendedItems.size === 0) {
                return;
            }
            
            // 计算命中物品
            let hits = 0;
            recommendedItems.forEach(itemId => {
                if (realItems.has(itemId)) {
                    hits++;
                }
            });
            
            validUsers++;
            totalRecall += hits / realItems.size;
            totalPrecision += hits / recommendedItems.size;
        });
        
        return {
            test_users: validUsers,
            avg_recall: validUsers > 0 ? Math.round(totalRecall / validUsers * 10000) / 10000 : 0,
            avg_precision: validUsers > 0 ? Math.round(totalPrecision / validUsers * 10000) / 10000 : 0
        };
    }
}

// 基于内容的推荐算法实现
class ContentBasedRecommender {
    constructor() {
        this.users = mockUsers;
        this.items = mockItems;
        this.interactions = mockInteractions;
    }
    
    recommendItems(user_id, top_n = 5, filter_interacted = true) {
        // 获取用户交互历史
        const userInteractions = this.interactions.filter(interaction => interaction.user_id === user_id);
        
        if (userInteractions.length === 0) {
            return [];
        }
        
        // 计算用户对各个类别的偏好
        const categoryPreferences = {};
        userInteractions.forEach(interaction => {
            const item = this.items.find(i => i.item_id === interaction.item_id);
            if (item) {
                const weight = getInteractionWeight(interaction.interaction_type);
                categoryPreferences[item.category] = (categoryPreferences[item.category] || 0) + weight;
            }
        });
        
        // 过滤已交互物品
        const interactedItemIds = new Set(userInteractions.map(i => i.item_id));
        
        // 计算物品得分
        const itemScores = this.items
            .filter(item => !filter_interacted || !interactedItemIds.has(item.item_id))
            .map(item => ({
                item_id: item.item_id,
                score: categoryPreferences[item.category] || 0,
                item: item
            }))
            .sort((a, b) => b.score - a.score)
            .slice(0, top_n);
        
        return itemScores;
    }
    
    evaluate(testUserIds = null, top_n = 5) {
        const activeUsers = [...new Set(this.interactions.map(i => i.user_id))];
        if (activeUsers.length === 0) {
            return { test_users: 0, avg_recall: 0, avg_precision: 0 };
        }
        
        // 选择测试用户
        const testUsers = testUserIds || activeUsers.slice(0, Math.max(1, Math.floor(activeUsers.length * 0.2)));
        
        let totalRecall = 0;
        let totalPrecision = 0;
        let validUsers = 0;
        
        testUsers.forEach(user_id => {
            // 真实交互物品
            const realItems = new Set(
                this.interactions
                    .filter(interaction => interaction.user_id === user_id)
                    .map(interaction => interaction.item_id)
            );
            
            if (realItems.size < 2) {
                return;
            }
            
            // 推荐物品
            const recommendedItems = new Set(
                this.recommendItems(user_id, top_n)
                    .map(recommendation => recommendation.item_id)
            );
            
            if (recommendedItems.size === 0) {
                return;
            }
            
            // 计算命中物品
            let hits = 0;
            recommendedItems.forEach(itemId => {
                if (realItems.has(itemId)) {
                    hits++;
                }
            });
            
            validUsers++;
            totalRecall += hits / realItems.size;
            totalPrecision += hits / recommendedItems.size;
        });
        
        return {
            test_users: validUsers,
            avg_recall: validUsers > 0 ? Math.round(totalRecall / validUsers * 10000) / 10000 : 0,
            avg_precision: validUsers > 0 ? Math.round(totalPrecision / validUsers * 10000) / 10000 : 0
        };
    }
}

// 推荐算法工厂，用于创建不同类型的推荐器
export function createRecommender(type) {
    switch(type) {
        case 'itemcf':
            return new ItemCFRecommender();
        case 'usercfswing':
            return new UserCFSwingRecommender();
        case 'content':
            return new ContentBasedRecommender();
        default:
            throw new Error(`不支持的推荐算法类型: ${type}`);
    }
}

// 获取算法列表
export function getAvailableAlgorithms() {
    return [
        { value: 'itemcf', label: 'ItemCF (基于物品的协同过滤)' },
        { value: 'usercfswing', label: 'UserCF+Swing (基于用户的协同过滤)' },
        { value: 'content', label: '基于内容的推荐' }
    ];
}