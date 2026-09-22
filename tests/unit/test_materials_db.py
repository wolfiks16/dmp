from __future__ import annotations

import json

import pytest

from webapp import materials_db as db


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    """Каждый тест — своя база и свой «старый» JSON, реальные файлы проекта не трогаем."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "materials.db")
    monkeypatch.setattr(db, "LEGACY_JSON", tmp_path / "materials.json")
    return tmp_path


def _magnet(name="Мой SmCo", Br=1.05):
    return {"kind": "magnet", "name": name, "family": "SmCo", "Br": Br,
            "Hcb": 780e3, "Hk": 1100e3, "Hcj": 1600e3, "alpha_Br": 0.03, "gamma_Hc": 0.20}


def test_create_read_delete_roundtrip() -> None:
    mid = db.upsert(_magnet())
    assert db.get(mid)["name"] == "Мой SmCo"
    assert list(db.load_all()) == [mid]
    assert db.delete(mid) is True
    assert db.get(mid) is None and db.load_all() == {}
    assert db.delete(mid) is False           # повторное удаление — не ошибка, но и не успех


def test_rename_edits_the_record_and_does_not_create_a_duplicate() -> None:
    """Главная причина перехода с JSON: там id считался из имени, и правка плодила дубли."""
    mid = db.upsert(_magnet(name="Партия 1", Br=1.05))
    same = db.upsert(_magnet(name="Партия 1 (уточнено)", Br=1.09), mid)
    assert same == mid
    assert len(db.load_all()) == 1
    spec = db.get(mid)
    assert (spec["name"], spec["Br"]) == ("Партия 1 (уточнено)", 1.09)


def test_two_materials_with_equal_names_stay_separate() -> None:
    a, b = db.upsert(_magnet(name="Образец")), db.upsert(_magnet(name="Образец"))
    assert a != b and len(db.load_all()) == 2


def test_legacy_json_is_imported_once_and_file_is_kept_as_backup(tmp_db) -> None:
    legacy = {"cust_old": {"kind": "magnet", "name": "Старый", "Br": 1.0, "Hcb": 700e3,
                           "Hk": 900e3, "Hcj": 1200e3}}
    db.LEGACY_JSON.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
    assert db.load_all()["cust_old"]["name"] == "Старый"
    assert not db.LEGACY_JSON.exists()                       # переименован, не удалён
    assert (tmp_db / "materials.json.imported").exists()
    db.load_all()                                            # повторный вызов не дублирует
    assert len(db.load_all()) == 1


def test_broken_legacy_json_does_not_break_startup(tmp_db) -> None:
    db.LEGACY_JSON.write_text("{это не json", encoding="utf-8")
    assert db.load_all() == {}


def test_invalid_kind_and_empty_name_are_rejected() -> None:
    with pytest.raises(ValueError):
        db.upsert({"kind": "воздух", "name": "х"})
    with pytest.raises(ValueError):
        db.upsert({"kind": "magnet", "name": "   "})


def test_steel_spec_survives_roundtrip() -> None:
    mid = db.upsert({"kind": "steel", "name": "Моя сталь", "family": "сталь",
                     "H": [0.0, 100.0], "B": [0.0, 1.2]})
    assert db.get(mid)["B"] == [0.0, 1.2]
