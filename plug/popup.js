// 加载单词本
function loadWordList() {
  chrome.storage.sync.get(["wordList"], (result) => {
    const wordList = result.wordList || [];
    const wordListElement = document.getElementById('wordList');
    
    if (wordList.length === 0) {
      wordListElement.innerHTML = '<div class="empty">暂无保存的单词</div>';
    } else {
      // 按时间倒序排序
      wordList.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
      
      wordListElement.innerHTML = '';
      wordList.forEach(item => {
        const wordItem = document.createElement('div');
        wordItem.className = 'word-item';
        wordItem.innerHTML = `
          <div class="word">${item.word}</div>
          <div class="translation">${item.translation}</div>
        `;
        wordListElement.appendChild(wordItem);
      });
    }
  });
}

// 清空单词本
function clearWordList() {
  if (confirm('确定要清空所有单词吗？')) {
    chrome.storage.sync.set({ wordList: [] }, () => {
      loadWordList();
    });
  }
}

// 初始化
document.addEventListener('DOMContentLoaded', () => {
  loadWordList();
  
  // 清空按钮事件
  document.getElementById('clearBtn').addEventListener('click', clearWordList);
});