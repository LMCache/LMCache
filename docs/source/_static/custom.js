document.addEventListener("DOMContentLoaded", function () {
  var script = document.createElement("script");
  script.type = "module";
  script.id = "runllm-widget-script"

  script.src = "https://widget.runllm.com";

  script.setAttribute("version", "stable");
  script.setAttribute("crossorigin", "true");
  script.setAttribute("runllm-keyboard-shortcut", "Mod+j");
  script.setAttribute("runllm-name", "LMCache Assistant");
  script.setAttribute("runllm-position", "BOTTOM_RIGHT");
  script.setAttribute("runllm-assistant-id", "1185");

  script.async = true;
  document.head.appendChild(script);

  // Initialize language switcher
  initLanguageSwitcher();
});

function initLanguageSwitcher() {
  // Wait for the SVG to be available
  setTimeout(function() {
    var switcher = document.getElementById('language-switcher');
    if (switcher) {
      // Find the parent link and override its click behavior
      var link = switcher.closest('a');
      if (link) {
        link.addEventListener('click', function(e) {
          e.preventDefault();
          switchLanguage();
        });
        
        // Update icon based on current language
        updateLanguageIcon();
      }
    } else {
      // Retry if not found yet
      if (document.readyState === 'complete') {
        console.warn('Language switcher icon not found');
      } else {
        initLanguageSwitcher();
      }
    }
  }, 100);
}

function getCurrentLanguage() {
  var path = window.location.pathname;
  if (path.includes('/zh_CN/') || path.includes('/zh/') || path.includes('/cn/')) {
    return 'zh_CN';
  }
  return 'en';
}

function updateLanguageIcon() {
  var switcher = document.getElementById('language-switcher');
  if (!switcher) return;
  
  var currentLang = getCurrentLanguage();
  var title = currentLang === 'zh_CN' ? 'Switch to English' : '切换到中文';
  
  var link = switcher.closest('a');
  if (link) {
    link.setAttribute('title', title);
    link.setAttribute('aria-label', title);
  }
}

function switchLanguage() {
  var currentPath = window.location.pathname;
  var currentLang = getCurrentLanguage();
  var newPath;
  
  if (currentLang === 'en') {
    // Switch to Chinese
    newPath = '/zh_CN' + currentPath;
  } else {
    // Switch to English
    if (currentPath.includes('/zh_CN/')) {
      newPath = currentPath.replace('/zh_CN', '');
    } else if (currentPath.includes('/zh/')) {
      newPath = currentPath.replace('/zh', '');
    } else if (currentPath.includes('/cn/')) {
      newPath = currentPath.replace('/cn', '');
    } else {
      newPath = currentPath;
    }
  }
  
  window.location.href = newPath;
}
