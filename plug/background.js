// 监听安装事件
chrome.runtime.onInstalled.addListener(() => {
  // 创建右键菜单
  chrome.contextMenus.create({
    id: "translate",
    title: "翻译 '%s'",
    contexts: ["selection"]
  });
  
  // 初始化存储
  chrome.storage.sync.set({ wordList: [] }, () => {
    console.log('单词本已初始化');
  });
});

// 监听右键菜单点击事件
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "translate" && info.selectionText) {
    // 发送消息给内容脚本进行翻译
    chrome.tabs.sendMessage(tab.id, {
      action: "translate",
      text: info.selectionText
    });
  }
});

// 监听来自内容脚本的消息
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "saveWord") {
    // 保存单词到单词本
    chrome.storage.sync.get(["wordList"], (result) => {
      const wordList = result.wordList || [];
      // 检查是否已存在该单词
      const exists = wordList.some(item => item.word === message.word);
      if (!exists) {
        wordList.push({
          word: message.word,
          translation: message.translation || "点击翻译查看释义",
          timestamp: new Date().toISOString()
        });
        chrome.storage.sync.set({ wordList }, () => {
          console.log('单词已保存到单词本');
          sendResponse({ success: true });
        });
      } else {
        sendResponse({ success: false, reason: "单词已存在" });
      }
    });
    return true; // 保持消息通道开放直到sendResponse被调用
  } else if (message.action === "updateWordTranslation") {
    // 更新单词的翻译
    chrome.storage.sync.get(["wordList"], (result) => {
      const wordList = result.wordList || [];
      const wordIndex = wordList.findIndex(item => item.word === message.word);
      
      if (wordIndex !== -1) {
        wordList[wordIndex].translation = message.translation;
        chrome.storage.sync.set({ wordList }, () => {
          console.log('单词翻译已更新');
          sendResponse({ success: true });
        });
      } else {
        sendResponse({ success: false, reason: "单词不存在" });
      }
    });
    return true; // 保持消息通道开放直到sendResponse被调用
  }
});