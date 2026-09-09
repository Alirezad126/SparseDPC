(function () {
  'use strict';

  /* Mobile navigation */
  var burger = document.getElementById('burger');
  var menu = document.getElementById('menu');

  burger.addEventListener('click', function () {
    var open = menu.classList.toggle('open');
    burger.setAttribute('aria-expanded', String(open));
  });

  menu.addEventListener('click', function (e) {
    if (e.target.tagName === 'A') {
      menu.classList.remove('open');
      burger.setAttribute('aria-expanded', 'false');
    }
  });

  /* FR / EN toggle — every translatable node carries data-fr and data-en */
  var LANGS = ['fr', 'en'];
  var stored = null;
  try { stored = localStorage.getItem('lrsm-lang'); } catch (e) {}

  function apply(lang) {
    document.documentElement.lang = lang;
    var nodes = document.querySelectorAll('[data-' + lang + ']');
    for (var i = 0; i < nodes.length; i++) {
      nodes[i].textContent = nodes[i].getAttribute('data-' + lang);
    }
    try { localStorage.setItem('lrsm-lang', lang); } catch (e) {}
  }

  document.getElementById('lang').addEventListener('click', function () {
    apply(document.documentElement.lang === 'fr' ? 'en' : 'fr');
  });

  if (LANGS.indexOf(stored) !== -1 && stored !== 'fr') apply(stored);

  /* Footer year */
  document.getElementById('year').textContent = new Date().getFullYear();
})();
