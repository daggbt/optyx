<script>
(function () {
  function findNavLinks() {
    // Gather the left sidebar links for the current section only (siblings)
    const sidebar = document.querySelector('#quarto-sidebar');
    if (!sidebar) return [];

    // Try to find the active link and use its parent section
    let active = sidebar.querySelector('a.sidebar-link.active');
    if (!active) {
      // if there's no active class, try to match by URL
      const path = window.location.pathname.replace(/.*\//, '');
      const byHref = Array.from(sidebar.querySelectorAll('a.sidebar-link')).find(a => a.href && a.href.endsWith(path));
      if (byHref) active = byHref;
    }

    let sectionRoot = null;
    if (active) {
      // find closest list root (ul)
      sectionRoot = active.closest('ul');
    }

    const scope = sectionRoot || sidebar;
    const links = Array.from(scope.querySelectorAll('a.sidebar-link'))
      .map(a => ({ href: a.getAttribute('href'), text: a.textContent.trim() }))
      .filter(x => x.href && x.href !== '#');
    return links;
  }

  function createPageNav(prev, next) {
    const navFooterCenter = document.querySelector('.nav-footer-center');
    if (!navFooterCenter) return;
    const container = document.createElement('div');
    container.className = 'page-nav-buttons';

    const prevBtn = document.createElement('a');
    prevBtn.className = 'btn btn-outline-primary me-2';
    if (prev) {
      prevBtn.href = prev.href;
      prevBtn.innerHTML = '← ' + prev.text;
    } else {
      prevBtn.href = '#';
      prevBtn.className += ' disabled';
      prevBtn.innerHTML = '← None';
    }

    const nextBtn = document.createElement('a');
    nextBtn.className = 'btn btn-outline-primary ms-2';
    if (next) {
      nextBtn.href = next.href;
      nextBtn.innerHTML = next.text + ' →';
    } else {
      nextBtn.href = '#';
      nextBtn.className += ' disabled';
      nextBtn.innerHTML = 'None →';
    }

    container.appendChild(prevBtn);
    container.appendChild(nextBtn);
    navFooterCenter.appendChild(container);
  }

  function init() {
    const links = findNavLinks();
    if (!links.length) return;
    const currentPath = window.location.pathname.replace(/.*\//, ''); // page.html
    // try to find matching href end
    let currentIndex = links.findIndex(l => l.href && l.href.endsWith(currentPath));
    if (currentIndex === -1) {
      // Sometimes links are relative without html extension
      const pathNoExt = currentPath.replace(/\.html$/, '');
      currentIndex = links.findIndex(l => l.href.includes(pathNoExt));
    }

    const prev = currentIndex > 0 ? links[currentIndex - 1] : null;
    const next = currentIndex >= 0 && currentIndex < links.length - 1 ? links[currentIndex + 1] : null;

    createPageNav(prev, next);
  }

  // Run when DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
</script>
