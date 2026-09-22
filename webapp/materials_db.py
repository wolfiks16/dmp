# -*- coding: utf-8 -*-
"""
Хранилище СВОИХ материалов — SQLite вместо плоского JSON.

Зачем: материалы теперь можно не только добавлять, но и РЕДАКТИРОВАТЬ и УДАЛЯТЬ прямо из
интерфейса, а для этого нужен устойчивый идентификатор, не зависящий от имени (в JSON-версии
id считался хешем имени, и переименование материала порождало дубль вместо правки).

Схема одна таблица:
    materials(id TEXT PK, kind TEXT, name TEXT, family TEXT, spec TEXT JSON,
              created_at TEXT, updated_at TEXT)

`spec` — та же структура, что раньше лежала в materials.json (СИ: Br [Тл], Hcb/Hk/Hcj [А/м];
для стали — списки H и B), поэтому остальной код читает материалы без изменений.

Старый materials.json импортируется автоматически при первом обращении и переименовывается
в materials.json.imported (не удаляем — это данные пользователя).
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "materials.db"
LEGACY_JSON = Path(__file__).parent / "materials.json"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS materials (
    id         TEXT PRIMARY KEY,
    kind       TEXT NOT NULL CHECK (kind IN ('magnet','steel')),
    name       TEXT NOT NULL,
    family     TEXT NOT NULL DEFAULT '',
    spec       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS materials_kind ON materials(kind);
-- Удалённые справочные марки. Сам справочник — код (magnet_catalog), строку из него стереть
-- нельзя, поэтому удаление из БИБЛИОТЕКИ фиксируется здесь: такая марка больше не выдаётся.
CREATE TABLE IF NOT EXISTS deleted (
    id TEXT PRIMARY KEY,
    at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(_SCHEMA)
    return con


def _import_legacy(con: sqlite3.Connection) -> int:
    """Перенести materials.json в базу (один раз). Возвращает число перенесённых записей."""
    if not LEGACY_JSON.exists():
        return 0
    try:
        store = json.loads(LEGACY_JSON.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — битый файл не должен рушить запуск
        return 0
    n = 0
    for mid, spec in (store or {}).items():
        kind = str(spec.get("kind", "magnet"))
        if kind not in ("magnet", "steel"):
            continue
        con.execute(
            "INSERT OR IGNORE INTO materials(id,kind,name,family,spec,created_at,updated_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (mid, kind, str(spec.get("name", mid)), str(spec.get("family", "")),
             json.dumps(spec, ensure_ascii=False), _now(), _now()))
        n += 1
    con.commit()
    LEGACY_JSON.rename(LEGACY_JSON.with_suffix(".json.imported"))
    return n


def init() -> None:
    """Создать базу и, если есть, втянуть старый JSON. Безопасно вызывать многократно."""
    with _connect() as con:
        _import_legacy(con)


def load_all() -> dict:
    """Все свои материалы как {id: spec} — формат, совместимый с прежним materials.json."""
    init()
    with _connect() as con:
        rows = con.execute("SELECT id, spec FROM materials ORDER BY name").fetchall()
    out = {}
    for r in rows:
        try:
            out[r["id"]] = json.loads(r["spec"])
        except Exception:  # noqa: BLE001 — одна битая запись не должна ломать список
            continue
    return out


def get(mid: str) -> dict | None:
    init()
    with _connect() as con:
        r = con.execute("SELECT spec FROM materials WHERE id=?", (mid,)).fetchone()
    return json.loads(r["spec"]) if r else None


def upsert(spec: dict, mid: str | None = None) -> str:
    """
    Создать материал (mid=None) или ПЕРЕЗАПИСАТЬ существующий по id.
    Идентификатор не зависит от имени — переименование правит запись, а не плодит дубли.
    """
    kind = str(spec.get("kind", "magnet"))
    if kind not in ("magnet", "steel"):
        raise ValueError("kind должен быть 'magnet' или 'steel'.")
    name = str(spec.get("name", "")).strip()
    if not name:
        raise ValueError("нужно имя материала.")
    payload = json.dumps(spec, ensure_ascii=False)
    init()
    with _connect() as con:
        if mid:
            cur = con.execute(
                "UPDATE materials SET kind=?, name=?, family=?, spec=?, updated_at=? WHERE id=?",
                (kind, name, str(spec.get("family", "")), payload, _now(), mid))
            if cur.rowcount:
                con.commit()
                return mid
        mid = mid or ("cust_" + uuid.uuid4().hex[:10])
        con.execute(
            "INSERT INTO materials(id,kind,name,family,spec,created_at,updated_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (mid, kind, name, str(spec.get("family", "")), payload, _now(), _now()))
        con.commit()
    return mid


def delete(mid: str) -> bool:
    init()
    with _connect() as con:
        cur = con.execute("DELETE FROM materials WHERE id=?", (mid,))
        con.commit()
    return bool(cur.rowcount)


# --- справочные марки: правка и удаление ---------------------------------------------------
# Правка каталожной марки сохраняется как ЗАПИСЬ С ТЕМ ЖЕ id и перекрывает справочную.
# Удаление БЕЗВОЗВРАТНОЕ: сам справочник — это код (magnet_catalog), строку оттуда стереть
# нельзя, поэтому факт удаления фиксируется в `deleted`, и библиотека такую марку не выдаёт.

def delete_catalog(mid: str) -> bool:
    """Убрать справочную марку из библиотеки навсегда (вместе с правками, если были)."""
    init()
    with _connect() as con:
        con.execute("INSERT OR REPLACE INTO deleted(id, at) VALUES(?,?)", (mid, _now()))
        con.execute("DELETE FROM materials WHERE id=?", (mid,))
        con.commit()
    return True


def deleted_ids() -> set[str]:
    init()
    with _connect() as con:
        return {r["id"] for r in con.execute("SELECT id FROM deleted").fetchall()}
