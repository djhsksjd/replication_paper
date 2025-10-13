let currentCategory = 'all';

// 加载单词本
function loadWordList(category = 'all') {
  chrome.storage.sync.get(["wordList"], (result) => {
    const wordList = result.wordList || [];
    const wordListElement = document.getElementById('wordList');
    
    // 根据分类筛选单词
    let filteredWords = wordList;
    if (category !== 'all') {
      filteredWords = wordList.filter(item => item.category === category);
    }
    
    if (filteredWords.length === 0) {
      wordListElement.innerHTML = '<div class="empty">暂无保存的单词</div>';
    } else {
      // 按时间倒序排序
      filteredWords.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
      
      wordListElement.innerHTML = '';
      filteredWords.forEach(item => {
        const wordItem = document.createElement('div');
        wordItem.className = 'word-item';
        wordItem.innerHTML = `
          <div class="word-info">
            <div class="word">${item.word}</div>
            <div class="translation">${item.translation}</div>
            <div class="category">分类：${item.category}</div>
          </div>
          <button class="delete-btn" data-word="${item.word}">删除</button>
        `;
        wordListElement.appendChild(wordItem);
        
        // 添加删除按钮事件
        const deleteBtn = wordItem.querySelector('.delete-btn');
        deleteBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          const word = e.currentTarget.dataset.word;
          deleteWord(word);
        });
      });
    }
  });
}

// 加载分类列表
function loadCategories() {
  chrome.runtime.sendMessage({ action: "getCategories" }, (response) => {
    const categories = response.categories || ['默认分类'];
    const categoryFilter = document.getElementById('categoryFilter');
    
    // 清空现有选项（保留第一个"所有分类"选项）
    while (categoryFilter.options.length > 1) {
      categoryFilter.remove(1);
    }
    
    // 添加分类选项
    categories.forEach(category => {
      const option = document.createElement('option');
      option.value = category;
      option.textContent = category;
      if (category === currentCategory) {
        option.selected = true;
      }
      categoryFilter.appendChild(option);
    });
  });
}

// 删除单个单词
function deleteWord(word) {
  if (confirm(`确定要删除单词 "${word}" 吗？`)) {
    chrome.runtime.sendMessage({ 
      action: "deleteWord",
      word: word
    }, (response) => {
      if (response && response.success) {
        loadWordList(currentCategory);
      }
    });
  }
}

// 清空单词本
function clearWordList() {
  if (confirm('确定要清空所有单词吗？')) {
    chrome.storage.sync.set({ wordList: [] }, () => {
      loadWordList(currentCategory);
    });
  }
}

// 导出单词本
function exportWordList() {
  chrome.storage.sync.get(["wordList"], (result) => {
    const wordList = result.wordList || [];
    const dataStr = JSON.stringify(wordList, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    
    // 创建下载链接
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `单词本_${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(link);
    link.click();
    
    // 清理
    setTimeout(() => {
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    }, 0);
  });
}

// 添加新分类
function addCategory() {
  const categoryInput = document.getElementById('newCategory');
  const category = categoryInput.value.trim();
  
  if (category) {
    chrome.runtime.sendMessage({ 
      action: "addCategory",
      category: category
    }, (response) => {
      if (response && response.success) {
        categoryInput.value = '';
        loadCategories();
      } else if (response && response.reason) {
        alert(response.reason);
      }
    });
  } else {
    alert('请输入分类名称');
  }
}

// 初始化
function init() {
  // 加载单词和分类
  loadWordList(currentCategory);
  loadCategories();
  
  // 清空按钮事件
  document.getElementById('clearBtn').addEventListener('click', clearWordList);
  
  // 导出按钮事件
  document.getElementById('exportBtn').addEventListener('click', exportWordList);
  
  // 分类筛选事件
  document.getElementById('categoryFilter').addEventListener('change', (e) => {
    currentCategory = e.target.value;
    loadWordList(currentCategory);
  });
  
  // 添加分类按钮事件
  document.getElementById('addCategoryBtn').addEventListener('click', addCategory);
  
  // 按Enter键添加分类
  document.getElementById('newCategory').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      addCategory();
    }
  });
}

// 当DOM加载完成后初始化
window.addEventListener('DOMContentLoaded', init);