// Saketh Saridena, portfolio behavior: theme, menu, header border and the current-section marker.

(function () {
  "use strict";

  var root = document.documentElement;
  var darkQuery = window.matchMedia("(prefers-color-scheme: dark)");

  function storedTheme() {
    try {
      return localStorage.getItem("theme");
    } catch (e) {
      return null;
    }
  }

  function currentTheme() {
    return root.getAttribute("data-theme") || (darkQuery.matches ? "dark" : "light");
  }

  // Keep the browser's UI color in step with the page, including a manual choice.
  function syncThemeColor() {
    var bg = getComputedStyle(document.body).backgroundColor;
    document.querySelectorAll('meta[name="theme-color"]').forEach(function (m) {
      m.setAttribute("content", bg);
    });
  }

  // ----- Theme toggle -----
  var toggle = document.querySelector(".theme-toggle");
  function labelToggle() {
    if (!toggle) return;
    var next = currentTheme() === "dark" ? "light" : "dark";
    toggle.setAttribute("aria-label", "Switch to " + next + " theme");
  }
  if (toggle) {
    labelToggle();
    toggle.addEventListener("click", function () {
      var next = currentTheme() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try {
        localStorage.setItem("theme", next);
      } catch (e) {
        /* storage blocked: the choice lasts for this page view */
      }
      labelToggle();
      syncThemeColor();
    });
    darkQuery.addEventListener("change", function () {
      if (!storedTheme()) {
        labelToggle();
        syncThemeColor();
      }
    });
  }
  if (storedTheme()) syncThemeColor();

  // ----- Phone menu: the toggle comes before the links, so Tab moves into them -----
  var menuButton = document.querySelector(".menu-toggle");
  var navLinks = document.getElementById("nav-links");
  function setMenu(open) {
    menuButton.setAttribute("aria-expanded", String(open));
    menuButton.setAttribute("aria-label", open ? "Close menu" : "Open menu");
    navLinks.classList.toggle("is-open", open);
  }
  if (menuButton && navLinks) {
    menuButton.addEventListener("click", function () {
      setMenu(menuButton.getAttribute("aria-expanded") !== "true");
    });
    navLinks.addEventListener("click", function (e) {
      if (e.target.closest("a")) setMenu(false);
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && navLinks.classList.contains("is-open")) {
        setMenu(false);
        menuButton.focus();
      }
    });
  }

  // ----- Header border once the page scrolls, and the nav link for the section in view -----
  var header = document.querySelector(".site-header");
  var sections = [];
  if (navLinks) {
    navLinks.querySelectorAll("a").forEach(function (a) {
      if (a.pathname !== location.pathname || !a.hash) return;
      var section = document.getElementById(a.hash.slice(1));
      if (section) sections.push({ link: a, section: section });
    });
  }

  var ticking = false;
  function update() {
    ticking = false;
    if (header) header.classList.toggle("is-scrolled", window.scrollY > 8);
    if (!sections.length) return;
    // The current section is the last one whose top has passed 40% of the viewport height.
    var line = window.innerHeight * 0.4;
    var current = null;
    sections.forEach(function (s) {
      if (s.section.getBoundingClientRect().top <= line) current = s;
    });
    sections.forEach(function (s) {
      if (s === current) s.link.setAttribute("aria-current", "location");
      else s.link.removeAttribute("aria-current");
    });
  }
  window.addEventListener(
    "scroll",
    function () {
      if (!ticking) {
        ticking = true;
        window.requestAnimationFrame(update);
      }
    },
    { passive: true }
  );
  window.addEventListener("resize", update);
  update();

  var year = document.getElementById("year");
  if (year) year.textContent = String(new Date().getFullYear());
})();
