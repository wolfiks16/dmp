import json

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from webapp import server  # noqa: E402

# Тема оформления по умолчанию хранится на сервере (файл настроек интерфейса), одна для всех браузеров и
# переживает перезапуск; неверное значение не принимается, испорченный файл не ломает приложение.

client = TestClient(server.app)


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    monkeypatch.setattr(server, "_UI_SETTINGS_PATH", path)
    return path


def test_without_settings_file_the_default_theme_is_color(settings_file):
    assert not settings_file.exists()
    d = client.get("/api/settings").json()
    assert d["default_theme"] == "color"
    assert d["themes"] == ["white", "grey", "color"]


def test_default_theme_is_saved_to_file_and_survives_restart(settings_file):
    d = client.post("/api/settings", json={"default_theme": "white"}).json()
    assert "error" not in d and d["default_theme"] == "white"
    assert json.loads(settings_file.read_text(encoding="utf-8")) == {"default_theme": "white"}
    # «перезапуск»: значение читается из файла заново, а не из памяти процесса
    assert server._ui_settings() == {"default_theme": "white"}
    assert client.get("/api/settings").json()["default_theme"] == "white"


def test_unknown_theme_is_refused_and_nothing_changes(settings_file):
    client.post("/api/settings", json={"default_theme": "grey"})
    for bad in ("eskd", "", None, 3):
        d = client.post("/api/settings", json={"default_theme": bad}).json()
        assert "error" in d and d["default_theme"] == "grey"
    assert json.loads(settings_file.read_text(encoding="utf-8"))["default_theme"] == "grey"


def test_broken_settings_file_falls_back_to_color(settings_file):
    for text in ("{не json", "[1, 2]", json.dumps({"default_theme": "purple"})):
        settings_file.write_text(text, encoding="utf-8")
        assert client.get("/api/settings").json()["default_theme"] == "color"


def test_parallel_limit_works_as_before_and_does_not_touch_the_theme(settings_file):
    before = client.get("/api/settings").json()["max_parallel"]
    try:
        d = client.post("/api/settings", json={"max_parallel": 2}).json()
        assert d["max_parallel"] == 2 and d["default_theme"] == "color"
        assert not settings_file.exists()                     # тему не меняли — файл не создаётся
    finally:
        client.post("/api/settings", json={"max_parallel": before})
