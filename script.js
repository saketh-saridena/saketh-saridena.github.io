// Saketh Saridena, portfolio behavior: theme, menu, active section, case studies.

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
    });
    darkQuery.addEventListener("change", function () {
      if (!storedTheme()) labelToggle();
    });
  }

  // ----- Mobile menu -----
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

  // ----- Header border once the page scrolls -----
  var header = document.querySelector(".site-header");
  if (header) {
    var onScroll = function () {
      header.classList.toggle("is-scrolled", window.scrollY > 8);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  }

  // ----- Mark the nav link for the section in view -----
  if ("IntersectionObserver" in window && navLinks) {
    var links = {};
    navLinks.querySelectorAll('a[href^="#"]').forEach(function (a) {
      links[a.getAttribute("href").slice(1)] = a;
    });
    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          var link = links[entry.target.id];
          if (!link || !entry.isIntersecting) return;
          Object.keys(links).forEach(function (id) {
            links[id].removeAttribute("aria-current");
          });
          link.setAttribute("aria-current", "true");
        });
      },
      { rootMargin: "-45% 0px -50% 0px" }
    );
    Object.keys(links).forEach(function (id) {
      var section = document.getElementById(id);
      if (section) observer.observe(section);
    });
  }

  // ----- Case study dialogs -----
  document.querySelectorAll("[data-case]").forEach(function (button) {
    var dialog = document.getElementById(button.getAttribute("data-case"));
    if (!dialog || typeof dialog.showModal !== "function") {
      button.hidden = true;
      return;
    }
    button.addEventListener("click", function () {
      dialog.showModal();
      dialog.scrollTop = 0;
    });
  });

  document.querySelectorAll("dialog.case").forEach(function (dialog) {
    dialog.addEventListener("click", function (e) {
      // A click on the backdrop lands on the dialog element itself
      if (e.target === dialog) dialog.close();
    });
    var close = dialog.querySelector(".case-close");
    if (close) {
      close.addEventListener("click", function () {
        dialog.close();
      });
    }
  });

  var year = document.getElementById("year");
  if (year) year.textContent = String(new Date().getFullYear());
})();
