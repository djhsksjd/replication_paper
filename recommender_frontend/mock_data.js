// 模拟数据模块

// 用户数据
export const mockUsers = [
  {
    user_id: 1,
    name: "张三",
    age: 28,
    gender: "男",
    preferred_categories: ["电子产品", "运动装备"],
    avatar: "https://picsum.photos/id/1/100/100"
  },
  {
    user_id: 2,
    name: "李四",
    age: 32,
    gender: "女",
    preferred_categories: ["时尚服饰", "美妆护肤"],
    avatar: "https://picsum.photos/id/2/100/100"
  },
  {
    user_id: 3,
    name: "王五",
    age: 25,
    gender: "男",
    preferred_categories: ["图书", "音乐"],
    avatar: "https://picsum.photos/id/3/100/100"
  },
  {
    user_id: 4,
    name: "赵六",
    age: 40,
    gender: "女",
    preferred_categories: ["家居用品", "厨具"],
    avatar: "https://picsum.photos/id/4/100/100"
  },
  {
    user_id: 5,
    name: "钱七",
    age: 35,
    gender: "男",
    preferred_categories: ["汽车用品", "电子产品"],
    avatar: "https://picsum.photos/id/5/100/100"
  }
];

// 物品数据
export const mockItems = [
  {
    item_id: 101,
    item_name: "无线降噪耳机",
    category: "电子产品",
    price: 999,
    description: "高品质无线降噪耳机，续航可达24小时",
    image: "https://picsum.photos/id/10/300/300"
  },
  {
    item_id: 102,
    item_name: "智能手表",
    category: "电子产品",
    price: 1299,
    description: "多功能智能手表，支持心率监测和运动追踪",
    image: "https://picsum.photos/id/11/300/300"
  },
  {
    item_id: 103,
    item_name: "篮球鞋",
    category: "运动装备",
    price: 799,
    description: "专业篮球鞋，提供良好的支撑和抓地力",
    image: "https://picsum.photos/id/12/300/300"
  },
  {
    item_id: 104,
    item_name: "瑜伽垫",
    category: "运动装备",
    price: 199,
    description: "环保材质瑜伽垫，防滑设计",
    image: "https://picsum.photos/id/13/300/300"
  },
  {
    item_id: 105,
    item_name: "时尚连衣裙",
    category: "时尚服饰",
    price: 599,
    description: "优雅时尚连衣裙，适合各种场合",
    image: "https://picsum.photos/id/14/300/300"
  },
  {
    item_id: 106,
    item_name: "护肤套装",
    category: "美妆护肤",
    price: 899,
    description: "全套护肤产品，温和不刺激",
    image: "https://picsum.photos/id/15/300/300"
  },
  {
    item_id: 107,
    item_name: "科幻小说集",
    category: "图书",
    price: 129,
    description: "经典科幻小说合集，包含多部获奖作品",
    image: "https://picsum.photos/id/16/300/300"
  },
  {
    item_id: 108,
    item_name: "蓝牙耳机",
    category: "电子产品",
    price: 399,
    description: "小巧便携蓝牙耳机，音质清晰",
    image: "https://picsum.photos/id/17/300/300"
  },
  {
    item_id: 109,
    item_name: "机械键盘",
    category: "电子产品",
    price: 499,
    description: "专业机械键盘，打字手感舒适",
    image: "https://picsum.photos/id/18/300/300"
  },
  {
    item_id: 110,
    item_name: "咖啡机",
    category: "厨具",
    price: 1599,
    description: "全自动咖啡机，支持多种咖啡制作",
    image: "https://picsum.photos/id/19/300/300"
  }
];

// 交互历史数据
export const mockInteractions = [
  { user_id: 1, item_id: 101, interaction_type: "点击", interaction_time: "2023-10-01T10:00:00" },
  { user_id: 1, item_id: 103, interaction_type: "购买", interaction_time: "2023-10-02T14:30:00" },
  { user_id: 1, item_id: 108, interaction_type: "点击", interaction_time: "2023-10-03T09:15:00" },
  { user_id: 2, item_id: 105, interaction_type: "购买", interaction_time: "2023-10-01T16:45:00" },
  { user_id: 2, item_id: 106, interaction_type: "点击", interaction_time: "2023-10-02T11:20:00" },
  { user_id: 3, item_id: 107, interaction_type: "购买", interaction_time: "2023-10-01T08:30:00" },
  { user_id: 3, item_id: 101, interaction_type: "点击", interaction_time: "2023-10-03T15:10:00" },
  { user_id: 4, item_id: 110, interaction_type: "购买", interaction_time: "2023-10-02T10:50:00" },
  { user_id: 5, item_id: 102, interaction_type: "点击", interaction_time: "2023-10-01T13:25:00" },
  { user_id: 5, item_id: 109, interaction_type: "购买", interaction_time: "2023-10-03T14:40:00" }
];

// 推荐结果数据（模拟不同算法的推荐结果）
export const recommendationResults = {
  itemCF: {
    1: [102, 109, 108, 104], // 用户1的ItemCF推荐
    2: [106, 105, 101, 108], // 用户2的ItemCF推荐
    3: [101, 108, 102, 109], // 用户3的ItemCF推荐
    4: [110, 104, 105, 106], // 用户4的ItemCF推荐
    5: [109, 102, 101, 108]  // 用户5的ItemCF推荐
  },
  userCF: {
    1: [109, 102, 108, 104], // 用户1的UserCF推荐
    2: [105, 106, 101, 108], // 用户2的UserCF推荐
    3: [108, 101, 102, 109], // 用户3的UserCF推荐
    4: [110, 104, 105, 106], // 用户4的UserCF推荐
    5: [102, 109, 101, 108]  // 用户5的UserCF推荐
  },
  contentBased: {
    1: [102, 108, 109, 104], // 用户1的基于内容推荐
    2: [105, 106, 101, 103], // 用户2的基于内容推荐
    3: [107, 101, 108, 102], // 用户3的基于内容推荐
    4: [110, 104, 105, 106], // 用户4的基于内容推荐
    5: [102, 109, 101, 108]  // 用户5的基于内容推荐
  }
};

// 性能指标数据
export const performanceMetrics = {
  itemCF: {
    recall: 0.75,
    precision: 0.65,
    f1: 0.70
  },
  userCF: {
    recall: 0.72,
    precision: 0.68,
    f1: 0.70
  },
  contentBased: {
    recall: 0.68,
    precision: 0.72,
    f1: 0.70
  }
};

// 获取用户的交互历史
export function getUserHistory(userId) {
  return mockInteractions
    .filter(interaction => interaction.user_id === userId)
    .map(interaction => {
      const item = mockItems.find(i => i.item_id === interaction.item_id);
      return {
        ...interaction,
        itemName: item?.item_name,
        itemCategory: item?.category,
        itemImage: item?.image
      };
    });
}

// 获取推荐物品详情
export function getRecommendationItems(algorithm, userId, topN = 4) {
  const itemIds = recommendationResults[algorithm]?.[userId] || [];
  return itemIds.slice(0, topN).map(itemId => {
    return mockItems.find(item => item.item_id === itemId);
  }).filter(Boolean);
}

// 获取算法性能指标
export function getAlgorithmMetrics(algorithm) {
  return performanceMetrics[algorithm] || {
    recall: 0,
    precision: 0,
    f1: 0
  };
}