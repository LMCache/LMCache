/**
 * Goblin image pair and placement metadata.
 *
 * @typedef {Object} GoblinVariant
 * @property {string} name The variant name used for CSS classes and session state.
 * @property {string} image The default image filename in the Sphinx static directory.
 * @property {string} hitImage The clicked-state image filename in the Sphinx static directory.
 * @property {"edge" | "free" | "link"} placement The placement strategy for this variant.
 * @property {[number, number]} size The minimum and maximum rendered size in pixels.
 * @property {number} weight The relative selection weight for this variant.
 */

/**
 * Get the base URL for docs static assets by locating the loaded custom.js file.
 *
 * @returns {string} The absolute or relative base URL for static assets.
 */
function getDocsStaticBaseUrl() {
  var currentScript = document.querySelector('script[src*="custom.js"]');

  if (!currentScript) {
    return "_static/";
  }

  return new URL(".", currentScript.src).href;
}

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
 * Remove all goblin easter egg elements from the current page.
 *
 * @returns {void}
 */
function removeGoblinEasterEgg() {
  document.querySelectorAll(".lmcache-goblin").forEach(function (goblin) {
    goblin.remove();
  });
}

/**
 * Load an image and run a callback after the browser has loaded the source.
 *
 * @param {string} src The image URL to load.
 * @param {() => void} onLoad Callback invoked when the image has loaded.
 * @returns {void}
 */
function loadGoblinImage(src, onLoad) {
  var image = new Image();

  image.onload = onLoad;
  image.src = src;
}

/**
 * Return a random number in the half-open range [minimum, maximum).
 *
 * @param {number} minimum The lower bound.
 * @param {number} maximum The upper bound.
 * @returns {number} A random number between the provided bounds.
 */
function randomNumber(minimum, maximum) {
  return Math.random() * (maximum - minimum) + minimum;
}

/**
 * Pick a random item from a non-empty array.
 *
 * @template T
 * @param {T[]} items The candidate items.
 * @returns {T} One randomly selected item.
 */
function pickRandomItem(items) {
  return items[Math.floor(Math.random() * items.length)];
}

/**
 * Pick a goblin variant according to each variant's relative weight.
 *
 * @param {GoblinVariant[]} variants The variants to choose from.
 * @returns {GoblinVariant} The selected variant.
 */
function pickWeightedGoblinVariant(variants) {
  var totalWeight = variants.reduce(function (total, variant) {
    return total + variant.weight;
  }, 0);
  var threshold = Math.random() * totalWeight;

  for (var index = 0; index < variants.length; index += 1) {
    threshold -= variants[index].weight;
    if (threshold <= 0) {
      return variants[index];
    }
  }

  return variants[variants.length - 1];
}

/**
 * Pick the next goblin variant while avoiding the previous session variant.
 *
 * @returns {GoblinVariant} The selected goblin variant.
 */
function pickGoblinVariant() {
  var variants = [
    {
      name: "sneaky",
      image: "goblin_sneaky.png",
      hitImage: "goblin_sneaky_hit.png",
      placement: "edge",
      size: [118, 146],
      weight: 1,
    },
    {
      name: "standing",
      image: "goblin_standing.png",
      hitImage: "goblin_standing_hit.png",
      placement: "free",
      size: [124, 156],
      weight: 1,
    },
    {
      name: "walking",
      image: "goblin_walking.png",
      hitImage: "goblin_walking_hit.png",
      placement: "free",
      size: [118, 150],
      weight: 1,
    },
    {
      name: "dancing",
      image: "goblin_dancing.png",
      hitImage: "goblin_dancing_hit.png",
      placement: "free",
      size: [130, 166],
      weight: 1,
    },
    {
      name: "grimace-face",
      image: "goblin_grimace_face.png",
      hitImage: "goblin_grimace_face_hit.png",
      placement: "free",
      size: [118, 150],
      weight: 1,
    },
    {
      name: "focus",
      image: "goblin_focus.png",
      hitImage: "goblin_focus_hit.png",
      placement: "link",
      size: [118, 148],
      weight: 1,
    },
  ];
  var previousVariant = window.sessionStorage.getItem(
    "lmcache-goblin-variant",
  );
  var availableVariants = variants.filter(function (variant) {
    return variant.name !== previousVariant;
  });
  var nextVariant = pickWeightedGoblinVariant(availableVariants);

  window.sessionStorage.setItem("lmcache-goblin-variant", nextVariant.name);
  return nextVariant;
}

/**
 * Pick an edge placement while avoiding the previous session placement.
 *
 * @returns {string} A CSS class for the selected edge placement.
 */
function pickEdgePlacement() {
  var positions = [
    "lmcache-goblin--bottom-left",
    "lmcache-goblin--middle-left",
    "lmcache-goblin--top-left",
    "lmcache-goblin--top-right",
    "lmcache-goblin--middle-right",
  ];
  var previousPosition = window.sessionStorage.getItem(
    "lmcache-goblin-position",
  );
  var availablePositions = positions.filter(function (position) {
    return position !== previousPosition;
  });
  var nextPosition = pickRandomItem(availablePositions);

  window.sessionStorage.setItem("lmcache-goblin-position", nextPosition);
  return nextPosition;
}

/**
 * Place a goblin at a random free-floating position in the viewport.
 *
 * @param {HTMLButtonElement} goblinButton The goblin button element.
 * @returns {void}
 */
function positionGoblinFreely(goblinButton) {
  goblinButton.classList.add("lmcache-goblin--free");
  goblinButton.style.setProperty(
    "--goblin-left",
    randomNumber(28, 72).toFixed(0) + "vw",
  );
  goblinButton.style.setProperty(
    "--goblin-top",
    randomNumber(26, 70).toFixed(0) + "vh",
  );
}

/**
 * Place a goblin near a currently visible documentation link.
 *
 * @param {HTMLButtonElement} goblinButton The goblin button element.
 * @returns {void}
 */
function positionGoblinNearLink(goblinButton) {
  var links = Array.from(
    document.querySelectorAll("main a[href], aside a[href]"),
  );
  var visibleLinks = links.filter(function (link) {
    var rect = link.getBoundingClientRect();

    return (
      rect.width > 24 &&
      rect.height > 12 &&
      rect.bottom > 80 &&
      rect.top < window.innerHeight - 80 &&
      rect.right > 0 &&
      rect.left < window.innerWidth
    );
  });

  if (visibleLinks.length === 0) {
    positionGoblinFreely(goblinButton);
    return;
  }

  var rect = pickRandomItem(visibleLinks).getBoundingClientRect();
  var left = Math.min(window.innerWidth - 150, Math.max(16, rect.left - 74));
  var top = Math.min(window.innerHeight - 150, Math.max(82, rect.top - 108));

  goblinButton.classList.add("lmcache-goblin--free");
  goblinButton.style.setProperty("--goblin-left", left.toFixed(0) + "px");
  goblinButton.style.setProperty("--goblin-top", top.toFixed(0) + "px");
}

/**
 * Apply the placement strategy for a selected goblin variant.
 *
 * @param {HTMLButtonElement} goblinButton The goblin button element.
 * @param {GoblinVariant} variant The selected goblin variant.
 * @returns {void}
 */
function positionGoblin(goblinButton, variant) {
  if (variant.placement === "edge") {
    goblinButton.classList.add(pickEdgePlacement());
  } else if (variant.placement === "link") {
    positionGoblinNearLink(goblinButton);
  } else {
    positionGoblinFreely(goblinButton);
  }
}

/**
 * Add a randomized clickable goblin easter egg to the current docs page.
 *
 * @returns {void}
 */
function addGoblinEasterEgg() {
  var highVisibilityPages = [
    "/getting_started/quickstart.html",
    "/mp/index.html",
    "/developer_guide/contributing.html",
  ];
  var currentPath = window.location.pathname;
  var isHighVisibilityPage = highVisibilityPages.some(function (pagePath) {
    return currentPath.endsWith(pagePath);
  });
  var appearanceChance = isHighVisibilityPage ? 0.95 : 0.65;

  if (Math.random() > appearanceChance) {
    return;
  }

  removeGoblinEasterEgg();

  var goblinVariant = pickGoblinVariant();
  var staticBaseUrl = getDocsStaticBaseUrl();
  var goblinButton = document.createElement("button");
  var goblinImage = document.createElement("img");
  var normalGoblinSrc = staticBaseUrl + goblinVariant.image;
  var hitGoblinSrc = staticBaseUrl + goblinVariant.hitImage;
  var pagePath = window.location.pathname;

  goblinButton.type = "button";
  goblinButton.className =
    "lmcache-goblin lmcache-goblin--" + goblinVariant.name;
  goblinButton.setAttribute("aria-label", "Dismiss the hidden LMCache goblin");
  goblinButton.style.setProperty(
    "--goblin-size",
    randomNumber(goblinVariant.size[0], goblinVariant.size[1]).toFixed(0) +
      "px",
  );
  goblinButton.style.setProperty(
    "--goblin-random-offset",
    randomNumber(-18, 18).toFixed(0) + "px",
  );
  goblinButton.style.setProperty(
    "--goblin-random-rotation",
    randomNumber(-5, 5).toFixed(1) + "deg",
  );
  goblinButton.style.setProperty(
    "--goblin-random-scale",
    randomNumber(0.88, 1.12).toFixed(2),
  );
  positionGoblin(goblinButton, goblinVariant);

  goblinImage.alt = "";
  goblinImage.decoding = "async";
  goblinImage.src = normalGoblinSrc;

  goblinButton.appendChild(goblinImage);

  goblinButton.addEventListener("pointerdown", function () {
    goblinButton.classList.add("lmcache-goblin--mouse-active");
  });

  goblinButton.addEventListener(
    "click",
    function () {
      if (goblinButton.classList.contains("lmcache-goblin--hit")) {
        return;
      }

      goblinButton.blur();
      goblinButton.tabIndex = -1;
      goblinButton.setAttribute("aria-hidden", "true");
      goblinButton.classList.add("lmcache-goblin--hit");
      goblinImage.src = hitGoblinSrc;

      window.setTimeout(function () {
        goblinButton.classList.add("lmcache-goblin--leaving");
      }, 1600);

      window.setTimeout(function () {
        goblinButton.remove();
      }, 2000);
    },
    { once: true },
  );

  loadGoblinImage(normalGoblinSrc, function () {
    if (window.location.pathname !== pagePath) {
      return;
    }

    document.body.appendChild(goblinButton);
  });

  loadGoblinImage(hitGoblinSrc, function () {});
}

/**
 * Initialize docs widgets after the DOM is ready.
 *
 * @returns {void}
 */
function initializeDocsWidgets() {
  addRunLlmWidget();
  addGoblinEasterEgg();
}

/**
 * Restore the goblin after returning from the browser back-forward cache.
 *
 * @param {PageTransitionEvent} event The page show event.
 * @returns {void}
 */
function restoreGoblinFromPageCache(event) {
  if (event.persisted) {
    addGoblinEasterEgg();
  }
}

document.addEventListener("DOMContentLoaded", initializeDocsWidgets);

window.addEventListener("pagehide", removeGoblinEasterEgg);
window.addEventListener("pageshow", restoreGoblinFromPageCache);
