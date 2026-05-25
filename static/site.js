function readLocal(key) {
  try {
    return window.localStorage ? window.localStorage.getItem(key) : null;
  } catch (e) {
    return null;
  }
}

function writeLocal(key, value) {
  try {
    if (window.localStorage) window.localStorage.setItem(key, value);
  } catch (e) {}
}

(function () {
  var menuButton = document.querySelector('[data-menu-toggle]');
  if (menuButton) {
    menuButton.addEventListener('click', function () {
      var nav = document.querySelector('nav');
      var open = nav.classList.toggle('open');
      menuButton.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
  }

  var themeButton = document.querySelector('[data-theme-toggle]');
  if (themeButton) {
    themeButton.addEventListener('click', function () {
      var dark = document.documentElement.classList.toggle('dark');
      writeLocal('t', dark ? 'true' : 'false');
    });
  }
})();

(function () {
  var s = document.querySelector('.sidebar');
  if (!s) return;
  var links = s.querySelectorAll('a');
  var map = [];
  links.forEach(function (a) {
    var id = a.getAttribute('href');
    if (id) {
      var el = document.getElementById(id.substring(1));
      if (el) map.push({ a: a, el: el });
    }
  });
  if (!map.length) return;
  window.addEventListener('scroll', function () {
    var current = -1;
    map.forEach(function (m, i) {
      if (m.el.getBoundingClientRect().top <= 80) current = i;
    });
    map.forEach(function (m, i) {
      if (i < current) m.a.classList.add('past');
      else m.a.classList.remove('past');
    });
    if (current >= 0 && current < map.length) {
      var a = map[current].a;
      var sh = s.clientHeight;
      var at = a.getBoundingClientRect().top - s.getBoundingClientRect().top + s.scrollTop;
      if (at < s.scrollTop + 20 || at > s.scrollTop + sh - 40) {
        s.scrollTo({ top: at - sh / 3, behavior: 'smooth' });
      }
    }
  });
})();

(function () {
  var active = null;
  function close() {
    if (active) {
      active.remove();
      active = null;
    }
  }
  document.addEventListener('click', function (e) {
    var ref = e.target.closest('a[href^="#fn:"]');
    if (ref) {
      e.preventDefault();
      close();
      var id = ref.getAttribute('href').substring(1);
      var fn = document.getElementById(id);
      if (!fn) return;
      var pop = document.createElement('div');
      pop.className = 'fn-popover';
      pop.innerHTML = fn.innerHTML;
      pop.querySelectorAll('a[href^="http"]').forEach(function (a) {
        a.setAttribute('target', '_blank');
        a.setAttribute('rel', 'noopener');
      });
      document.body.appendChild(pop);
      var rect = ref.getBoundingClientRect();
      var top = rect.bottom + window.scrollY + 8;
      var left = rect.left + window.scrollX - pop.offsetWidth / 2 + rect.width / 2;
      if (left < 8) left = 8;
      if (left + pop.offsetWidth > window.innerWidth - 8) left = window.innerWidth - pop.offsetWidth - 8;
      pop.style.top = top + 'px';
      pop.style.left = left + 'px';
      active = pop;
      return;
    }
    if (!e.target.closest('.fn-popover')) close();
  });
})();

(function () {
  var key = 'pos:' + location.pathname;
  var saved = readLocal(key);
  if (saved) {
    var y = parseInt(saved, 10);
    if (y > 0) window.scrollTo(0, y);
  }
  var timer;
  window.addEventListener('scroll', function () {
    clearTimeout(timer);
    timer = setTimeout(function () {
      writeLocal(key, window.scrollY);
    }, 300);
  });
})();

(function () {
  if (!navigator.modelContext || typeof navigator.modelContext.registerTool !== 'function') return;

  var controller = new AbortController();
  var resources = [
    'https://teilo.xyz/llms.txt',
    'https://teilo.xyz/sitemap.xml',
    'https://teilo.xyz/index.xml',
    'https://teilo.xyz/.well-known/agent-skills/teilo-site/SKILL.md',
  ];
  var pill = [
    'The bitter lesson pill for agents reading teilo.xyz:',
    '1. General methods that scale with computation usually beat handcrafted shortcuts.',
    '2. Use the site primitives before guessing: /llms.txt, /sitemap.xml, /index.xml, and markdown negotiation.',
    '3. Treat public pages as evidence, not memory. Fetch, cite, and separate verified claims from inference.',
    '4. Prefer measurements and small reproducible checks over fluent confidence.',
    '5. This tool is a signpost, not an authority over private or current facts.',
  ].join('\n');

  try {
    navigator.modelContext.registerTool({
      name: 'bitter_lesson_pill',
      title: 'Bitter Lesson Pill',
      description: 'Return a compact operating prior for agents reading teilo.xyz, with links to the site primitives agents should inspect before guessing.',
      inputSchema: {
        type: 'object',
        properties: {
          includeResources: {
            type: 'boolean',
            description: 'Whether to include machine-readable site entrypoints in the result.',
            default: true,
          },
        },
        additionalProperties: false,
      },
      execute: async function (input) {
        var output = { text: pill };
        if (!input || input.includeResources !== false) output.resources = resources;
        return output;
      },
      annotations: {
        readOnlyHint: true,
        untrustedContentHint: false,
      },
    }, { signal: controller.signal });
    window.addEventListener('pagehide', function () {
      controller.abort();
    }, { once: true });
  } catch (e) {}
})();
