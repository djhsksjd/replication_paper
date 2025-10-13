// 创建翻译弹窗
function createTooltip() {
  let tooltip = document.getElementById('word-translator-tooltip');
  if (!tooltip) {
    tooltip = document.createElement('div');
    tooltip.id = 'word-translator-tooltip';
    tooltip.className = 'word-translator-tooltip';
    document.body.appendChild(tooltip);
  }
  return tooltip;
}

// 翻译函数 (使用DeepSeek API)
async function translateText(text) {
  try {
    // DeepSeek API 配置
    const apiKey = '';
    const apiUrl = 'https://api.deepseek.com/v1/chat/completions';
    
    // 构建翻译请求
    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        model: 'deepseek-chat',
        messages: [
          {
            role: 'system',
            content: '你是一个专业的翻译助手，请将用户提供的文本翻译成中文。如果用户提供的已经是中文，请翻译成英文。直接返回翻译结果，不要添加任何额外的解释或说明。'
          },
          {
            role: 'user',
            content: text
          }
        ],
        max_tokens: 200
      })
    });
    
    const data = await response.json();
    
    // 处理API响应
    if (data && data.choices && data.choices.length > 0) {
      return data.choices[0].message.content.trim();
    } else {
      console.error('DeepSeek API 响应异常:', data);
      return '翻译失败';
    }
  } catch (error) {
    console.error('翻译出错:', error);
    return '翻译服务暂时不可用';
  }
}

// 显示翻译弹窗
function showTranslation(x, y, text, translation) {
  const tooltip = createTooltip();
  tooltip.innerHTML = `
    <div class="tooltip-header">
      <span class="original-text">${text}</span>
      <button class="save-btn" data-word="${text}" data-translation="${translation}">+ 单词本</button>
    </div>
    <div class="translation-text">${translation}</div>
  `;
  
  // 设置位置
  tooltip.style.left = `${x}px`;
  tooltip.style.top = `${y}px`;
  tooltip.style.display = 'block';
  
  // 添加保存按钮事件
  const saveBtn = tooltip.querySelector('.save-btn');
  saveBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    const word = e.currentTarget.dataset.word;
    const translation = e.currentTarget.dataset.translation;
    
    // 获取用户选择的分类（这里简化处理，使用默认分类）
    const category = "默认分类";
    
    // 发送保存单词的消息给后台脚本
    chrome.runtime.sendMessage({
      action: "saveWord",
      word: word,
      translation: translation,
      category: category
    }, (response) => {
      if (response && response.success) {
        saveBtn.textContent = '已保存';
        saveBtn.disabled = true;
      } else if (response && response.reason) {
        alert(response.reason);
      }
    });
  });
}

// 隐藏翻译弹窗
function hideTranslation() {
  const tooltip = document.getElementById('word-translator-tooltip');
  if (tooltip) {
    tooltip.style.display = 'none';
  }
}

// 监听鼠标选中文本事件
document.addEventListener('mouseup', async (e) => {
  const selection = window.getSelection();
  const text = selection.toString().trim();
  
  if (text.length > 0 && text.length < 500) { // 限制长度，避免过长文本
    const rect = selection.getRangeAt(0).getBoundingClientRect();
    const x = rect.right + 10;
    const y = rect.top - 10;
    
    // 翻译选中的文本
    const translation = await translateText(text);
    showTranslation(x, y, text, translation);
  }
});

// 点击其他地方隐藏翻译弹窗
document.addEventListener('mousedown', (e) => {
  const tooltip = document.getElementById('word-translator-tooltip');
  if (tooltip && !tooltip.contains(e.target)) {
    hideTranslation();
  }
});

// 监听来自后台的翻译请求
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "translate" && message.text) {
    translateText(message.text).then(translation => {
      // 显示翻译结果
      const selection = window.getSelection();
      if (selection.rangeCount > 0) {
        const rect = selection.getRangeAt(0).getBoundingClientRect();
        showTranslation(rect.right + 10, rect.top - 10, message.text, translation);
      }
    });
  }
});