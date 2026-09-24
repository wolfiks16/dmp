// ВЫПАДАЮЩИЕ МЕНЮ ИНТЕРФЕЙСА — одно на 2D и 3D (docs/ui_rules.md §1): стандартный вид, величина.
// Разметка: кнопка + <div class="vmenu" role="menu" hidden> с пунктами <button data-v="…">. Пункт с вложенным
// списком — <div class="vmenu-sub"><button class="vmenu-parent">…</button><div class="vmenu nested">…</div></div>:
// вложенный список открывается справа при наведении или фокусе. Открыто одно меню; щелчок мимо и Esc закрывают.
(function () {
  let open = null;                                           // {btn, menu}
  const items = menu => [...menu.querySelectorAll('button')].filter(b => b.offsetParent !== null);

  function close() {
    if (!open) return;
    open.menu.hidden = true;
    open.btn.setAttribute('aria-expanded', 'false');
    open = null;
  }
  // Меню ставится по координатам кнопки (position: fixed): панели вида обрезают своё содержимое (overflow),
  // а меню должно быть видно целиком. .up — над кнопкой (строка вида внизу), иначе — под ней; по горизонтали —
  // в пределах окна.
  function place(btn, menu) {
    const b = btn.getBoundingClientRect(), w = menu.offsetWidth, h = menu.offsetHeight, pad = 8;
    const up = menu.classList.contains('up');
    menu.style.left = Math.max(pad, Math.min(b.left, innerWidth - w - pad)) + 'px';
    menu.style.top = Math.max(pad, Math.min(up ? b.top - 6 - h : b.bottom + 6, innerHeight - h - pad)) + 'px';
  }
  function show(btn, menu) {
    if (open && open.menu === menu) { close(); return; }
    close();
    menu.hidden = false;
    place(btn, menu);
    btn.setAttribute('aria-expanded', 'true');
    open = { btn, menu };
    const cur = menu.querySelector('[aria-checked="true"]');
    const first = (cur && cur.closest('.nested') ? cur.closest('.vmenu-sub').querySelector('.vmenu-parent') : cur)
      || items(menu)[0];
    if (first) first.focus({ preventScroll: true });
  }
  document.addEventListener('click', e => {
    if (open && !open.menu.contains(e.target) && !open.btn.contains(e.target)) close();
  });
  document.addEventListener('keydown', e => {
    if (!open) return;
    if (e.key === 'Escape') { const b = open.btn; close(); b.focus(); e.stopPropagation(); return; }
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp' && e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
    const a = document.activeElement;
    if (e.key === 'ArrowRight' && a && a.classList.contains('vmenu-parent')) {          // во вложенный список
      const n = a.parentElement.querySelector('.nested button'); if (n) n.focus(); e.preventDefault(); return;
    }
    if (e.key === 'ArrowLeft' && a && a.closest('.nested')) {                           // назад к родителю
      a.closest('.vmenu-sub').querySelector('.vmenu-parent').focus(); e.preventDefault(); return;
    }
    const scope = a && a.closest('.nested') ? a.closest('.nested') : open.menu;
    const list = items(scope).filter(b => b.closest('.vmenu') === scope);
    if (!list.length) return;
    const i = list.indexOf(a), d = e.key === 'ArrowDown' ? 1 : e.key === 'ArrowUp' ? -1 : 0;
    if (d) { list[(i + d + list.length) % list.length].focus(); e.preventDefault(); }
  });

  window.Menu = {
    // кнопка открывает меню; выбор пункта (data-v) закрывает его и зовёт onPick(значение, пункт)
    bind(btn, menu, onPick) {
      btn.setAttribute('aria-haspopup', 'menu');
      btn.setAttribute('aria-expanded', 'false');
      btn.addEventListener('click', e => { e.stopPropagation(); show(btn, menu); });
      menu.addEventListener('click', e => {
        const it = e.target.closest('button[data-v]');
        if (!it || !menu.contains(it)) return;
        close(); onPick(it.dataset.v, it);
      });
    },
    // отметить выбранный пункт (aria-checked); у родителя вложенного списка — отметка, что выбор внутри
    check(menu, v) {
      menu.querySelectorAll('button[data-v]').forEach(b => b.setAttribute('aria-checked', String(b.dataset.v === String(v))));
      menu.querySelectorAll('.vmenu-sub').forEach(s => s.querySelector('.vmenu-parent')
        .classList.toggle('has-checked', !!s.querySelector('.nested [aria-checked="true"]')));
    },
    close,
  };
})();
