'use strict';

// Progressive enhancement: the complete page remains readable without JavaScript.
const filters = document.querySelector('.case-filters');
if (filters) {
  filters.hidden = false;
  filters.addEventListener('click', (event) => {
    const button = event.target.closest('button[data-filter]');
    if (!button) return;
    filters.querySelectorAll('button').forEach((item) => {
      item.setAttribute('aria-pressed', String(item === button));
    });
    document.querySelectorAll('.case').forEach((item) => {
      item.hidden = button.dataset.filter !== 'all' && item.dataset.case !== button.dataset.filter;
    });
  });
}

const sort = document.querySelector('#sort-results');
const table = document.querySelector('#results-table');
if (sort && table) {
  sort.parentElement.hidden = false;
  sort.addEventListener('change', () => {
    const column = Number(sort.value);
    const rows = Array.from(table.tBodies[0].rows);
    rows.sort((a, b) => Number(b.cells[column].textContent) - Number(a.cells[column].textContent));
    rows.forEach((row) => table.tBodies[0].appendChild(row));
    table.querySelectorAll('thead th').forEach((cell, index) => {
      if (index === column) cell.setAttribute('aria-sort', 'descending');
      else cell.removeAttribute('aria-sort');
    });
  });
  table.querySelectorAll('thead th')[1].setAttribute('aria-sort', 'descending');
}

const dialog = document.querySelector('#image-dialog');
if (dialog && typeof dialog.showModal === 'function') {
  document.querySelectorAll('.case .comparison img').forEach((img) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'image-expand';
    button.setAttribute('aria-label', `Enlarge comparison: ${img.alt}`);
    img.replaceWith(button);
    button.appendChild(img);
    button.addEventListener('click', () => {
      const article = button.closest('.case');
      dialog.querySelector('#dialog-title').textContent = article.querySelector('h3').textContent;
      const target = dialog.querySelector('.dialog-images');
      target.replaceChildren();
      article.querySelectorAll('.comparison figure').forEach((figure) => {
        const copy = document.createElement('figure');
        const image = figure.querySelector('img').cloneNode();
        image.loading = 'eager';
        copy.append(image, figure.querySelector('figcaption').cloneNode(true));
        target.appendChild(copy);
      });
      dialog.showModal();
    });
  });
  dialog.querySelector('#dialog-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', (event) => {
    const bounds = dialog.getBoundingClientRect();
    if (event.target === dialog && (event.clientX < bounds.left || event.clientX > bounds.right ||
        event.clientY < bounds.top || event.clientY > bounds.bottom)) dialog.close();
  });
}
