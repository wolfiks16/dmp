# three.js (сторонний компонент)

Объёмный вид 3D-режима (этап 3D-5). Решение Sergey 2026-09-14 — после сравнения с canvas 2D,
своим WebGL, VTK.js, Plotly и отрисовкой в Python (см. `docs/plan_3d_2026-09-11.md`).

- Пакет: npm `three`, версия **0.186.0** (файлы без изменений; скачаны 2026-09-14 через зеркало
  npm jsDelivr, `https://cdn.jsdelivr.net/npm/three@0.186.0/…`).
- Лицензия: MIT (`LICENSE` рядом).
- Интернет для работы не нужен: страница берёт файлы отсюда.

| файл | путь в пакете | байт | SHA-256 (base64, совпал с опубликованным) |
|---|---|---|---|
| `three.module.js` | `build/three.module.js` | 662 772 | `kFIELWdssP3B3f7+GTBT80t6wFE6YW/axFNdSZh4Euo=` |
| `three.core.js` | `build/three.core.js` | 1 458 113 | `nt3gArBmqaBWdqYSf2dzW2K685m96lKfL34xZX2naeY=` |
| `OrbitControls.js` | `examples/jsm/controls/OrbitControls.js` | 40 755 | `PXnQfstoa05dkyMu7aslUzHBvu9xHhMWTqofaGVaXys=` |
| `LICENSE` | `LICENSE` | 1 081 | `izeOvmDi/lABWMsKxxy16LfZKVPCq8xjoOuQSZZTtbw=` |
| `lines/LineSegments2.js` | `examples/jsm/lines/LineSegments2.js` | 11 477 | `/Lwg9Xbog0POoSEx9JtwA9erRO3kIvy2lOVnRWAi96A=` |
| `lines/LineSegmentsGeometry.js` | `examples/jsm/lines/LineSegmentsGeometry.js` | 6 893 | `Rx8KlUoMnFnT0VE5L0JBQqkWSnIoAvY7NxlxN4qEobE=` |
| `lines/LineMaterial.js` | `examples/jsm/lines/LineMaterial.js` | 14 020 | `1FFwAfnXs+iF6luqkw3gB3NR6IXbUu9FZc3kx90X6yM=` |

Папка `lines/` — толстые линии (этап 3D-7, силовые линии): обычные линии WebGL всегда толщиной в один
пиксель, а `LineSegments2` рисует их треугольниками и задаёт толщину. Скачаны 2026-09-18 с разрешения
Sergey оттуда же (jsDelivr, three@0.186.0), SHA-256 сверены с опубликованными. Три файла подключают только
`'three'` и друг друга.

`three.module.js` подключает `./three.core.js`; `OrbitControls.js` подключает `'three'` — в
`index.html` для этого карта импортов (`importmap`). Обновлять версию — только осознанно: интерфейс
библиотеки меняется от версии к версии.
