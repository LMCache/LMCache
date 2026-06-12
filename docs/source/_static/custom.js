/**
 * Add the RunLLM assistant widget script once per page.
 *
 * @returns {void}
 */
function addRunLlmWidget() {
  if (document.getElementById("runllm-widget-script")) {
    return;
  }

  var script = document.createElement("script");
  script.type = "module";
  script.id = "runllm-widget-script";

  script.src = "https://widget.runllm.com";

  script.setAttribute("version", "stable");
  script.setAttribute("crossorigin", "true");
  script.setAttribute("runllm-keyboard-shortcut", "Mod+j");
  script.setAttribute("runllm-name", "LMCache Assistant");
  script.setAttribute("runllm-position", "BOTTOM_RIGHT");
  script.setAttribute("runllm-assistant-id", "1185");

  script.async = true;
  document.head.appendChild(script);
}

/**
 * Return true when the current page is rendered under the Chinese docs prefix.
 *
 * @returns {boolean} Whether the current path is a Chinese documentation page.
 */
function isChineseDocsPage() {
  return window.location.pathname.split("/").includes("zh_CN");
}

/**
 * Build the target language URL for the current page.
 *
 * @param {"en" | "zh_CN"} language The target language.
 * @returns {string} The target URL.
 */
function buildLanguageUrl(language) {
  var pathParts = window.location.pathname.split("/");
  var zhIndex = pathParts.indexOf("zh_CN");

  if (language === "zh_CN" && zhIndex === -1) {
    pathParts.splice(1, 0, "zh_CN");
  } else if (language === "en" && zhIndex !== -1) {
    pathParts.splice(zhIndex, 1);
  }

  var nextPath = pathParts.join("/") || "/";
  return nextPath + window.location.search + window.location.hash;
}

/**
 * Build the home URL for a target language.
 *
 * @param {"en" | "zh_CN"} language The target language.
 * @returns {string} The target language home URL.
 */
function buildLanguageHomeUrl(language) {
  return language === "zh_CN" ? "/zh_CN/" : "/";
}

/**
 * Fall back to the target language home page if the equivalent page is absent.
 *
 * @param {HTMLAnchorElement} link The language link to validate.
 * @param {"en" | "zh_CN"} language The target language.
 * @returns {void}
 */
function fallbackMissingLanguagePage(link, language) {
  window
    .fetch(link.href, { method: "HEAD" })
    .then(function (response) {
      if (!response.ok) {
        link.href = buildLanguageHomeUrl(language);
      }
    })
    .catch(function () {
      link.href = buildLanguageHomeUrl(language);
    });
}

/**
 * Add a compact language switcher to the docs header.
 *
 * @returns {void}
 */
function addLanguageSwitcher() {
  if (document.querySelector(".lmcache-language-switcher")) {
    return;
  }

  var switcher = document.createElement("div");
  var chineseLink = document.createElement("a");
  var divider = document.createElement("span");
  var englishLink = document.createElement("a");
  var isChinesePage = isChineseDocsPage();

  switcher.className = "lmcache-language-switcher";
  switcher.setAttribute("aria-label", "Documentation language");

  chineseLink.href = buildLanguageUrl("zh_CN");
  chineseLink.textContent = "中文";
  chineseLink.setAttribute("aria-label", "Switch to Chinese");

  divider.className = "lmcache-language-switcher__divider";
  divider.textContent = "|";

  englishLink.href = buildLanguageUrl("en");
  englishLink.textContent = "Eng";
  englishLink.setAttribute("aria-label", "Switch to English");

  if (isChinesePage) {
    chineseLink.setAttribute("aria-current", "page");
  } else {
    englishLink.setAttribute("aria-current", "page");
  }

  switcher.appendChild(chineseLink);
  switcher.appendChild(divider);
  switcher.appendChild(englishLink);

  // Place the switcher in the top nav bar with the other icons.
  // If the nav bar isn't there, show it as a floating button instead.
  var navbar = findDocsNavbar();
  if (navbar) {
    navbar.appendChild(switcher);
  } else {
    switcher.classList.add("lmcache-language-switcher--fallback");
    document.body.appendChild(switcher);
  }

  fallbackMissingLanguagePage(chineseLink, "zh_CN");
  fallbackMissingLanguagePage(englishLink, "en");
}

/**
 * Locate the top nav bar that holds the GitHub / profile / theme-toggle
 * icons. Prefers the structural `header nav` selector; falls back to
 * the GitHub link's parent if the theme markup differs.
 *
 * @returns {HTMLElement | null} The nav bar element, or null if not found.
 */
function findDocsNavbar() {
  var navbar = document.querySelector("header nav");
  if (navbar) {
    return navbar;
  }
  var githubLink = document.querySelector('a[title="Visit GitHub"]');
  return githubLink ? githubLink.parentElement : null;
}

/**
 * Initialize docs widgets after the DOM is ready.
 *
 * @returns {void}
 */
function initializeDocsWidgets() {
  addLanguageSwitcher();
  addRunLlmWidget();
}

document.addEventListener("DOMContentLoaded", initializeDocsWidgets);
