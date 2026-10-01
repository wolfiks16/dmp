from html.parser import HTMLParser
from pathlib import Path

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from magcore.domain import magnet_catalog as mc  # noqa: E402
from webapp import materials_db, server  # noqa: E402

# Окно «Новый материал» → «Магнит». Поле колена H_k необязательно: если оно пустое, сервер берёт колено долей H_cJ
# ПО СЕМЕЙСТВУ (magnet_catalog.HK_RATIO: NdFeB 0,95, SmCo 0,85), то есть семейство задаёт запас до необратимого
# размагничивания. Прежде пример в окне был SmCo («Мой SmCo (измер.)», α_Br 0,03, γ_Hc 0,20), а «Семейство» по
# умолчанию стояло на NdFeB: пример, сохранённый как есть, становился SmCo с коленом NdFeB (1520 кА/м вместо 1360).
#
# То же окно правит марку из таблицы (✎). У справочной марки колено — то же допущение HK_RATIO·H_cJ, а не паспорт.
# Прежде сервер отдавал его форме как обычное число: окно ставило его в поле H_k, сохранение записывало допущение
# как «введено из паспорта», и с новым H_cJ колено оставалось прежним.

INDEX_HTML = Path(__file__).resolve().parents[2] / "webapp" / "static" / "index.html"
client = TestClient(server.app)


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    """Своя база на каждый тест — библиотеку материалов пользователя (webapp/materials.db) не трогаем."""
    monkeypatch.setattr(materials_db, "DB_PATH", tmp_path / "materials.db")
    monkeypatch.setattr(materials_db, "LEGACY_JSON", tmp_path / "materials.json")


class _DialogDefaults(HTMLParser):
    """Поля окна «Новый материал» по умолчанию, как их видит браузер: у <input> — value, у <select> — option
    с атрибутом selected, а без него — первый."""

    def __init__(self):
        super().__init__()
        self.values: dict[str, str] = {}
        self._select: str | None = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "input" and str(a.get("id", "")).startswith("mat-"):
            self.values[a["id"]] = a.get("value") or ""
        elif tag == "select" and a.get("id") == "mat-fam-sel":
            self._select = a["id"]
        elif tag == "option" and self._select and ("selected" in a or self._select not in self.values):
            self.values[self._select] = a["value"]

    def handle_endtag(self, tag):
        if tag == "select":
            self._select = None


def _dialog_defaults() -> dict[str, str]:
    p = _DialogDefaults()
    p.feed(INDEX_HTML.read_text(encoding="utf-8"))
    return p.values


def _save(body: dict) -> dict:
    r = client.post("/api/materials", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _body(family, hk_kA="", hcj_kA="1600"):
    """Запрос, как его шлёт окно (saveMaterial): числа строками, пустое поле — пустая строка."""
    return {"id": None, "family": family, "name": "Образец", "Br": "1.05", "Hcb_kA": "780", "Hk_kA": hk_kA,
            "BHmax": "", "T_max": "", "Hcj_kA": hcj_kA, "alpha_Br": "0.03", "gamma_Hc": "0.20"}


def _model_knee_A_m(mid: str) -> float:
    """Колено, с которым считает модель магнита при T0 = 20 °C, А/м (сервер округляет до 0,1)."""
    d = client.post("/api/magnet_curve", json={"material": mid, "T": 20.0}).json()
    assert "error" not in d, d
    return d["Hk"]


def test_example_saved_as_is_is_smco_with_the_smco_knee():
    v = _dialog_defaults()
    assert "SmCo" in v["mat-name"] and v["mat-fam-sel"] == "SmCo"    # пример SmCo — и семейство SmCo
    assert v["mat-Hk"] == ""                                          # колено в примере не задано
    body = {"id": None, "family": v["mat-fam-sel"], "name": v["mat-name"], "Br": v["mat-Br"],
            "Hcb_kA": v["mat-Hcb"], "Hk_kA": v["mat-Hk"], "BHmax": v["mat-BH"], "T_max": v["mat-Tmax"],
            "Hcj_kA": v["mat-Hcj"], "alpha_Br": v["mat-aBr"], "gamma_Hc": v["mat-gHc"]}
    d = _save(body)
    assert d.get("ok"), d
    spec = materials_db.get(d["id"])
    hcj = float(v["mat-Hcj"]) * 1e3
    assert spec["family"] == "SmCo" and spec["hk_given"] is False
    assert spec["Hk"] == pytest.approx(mc.HK_RATIO["SmCo"] * hcj, rel=1e-12)
    assert _model_knee_A_m(d["id"]) == pytest.approx(mc.HK_RATIO["SmCo"] * hcj, abs=0.05)


@pytest.mark.parametrize("family", mc.FAMILIES)
@pytest.mark.parametrize("hk_kA", ["", "   ", None], ids=["пусто", "пробелы", "поля нет"])
def test_empty_knee_is_taken_by_the_family_of_the_material(family, hk_kA):
    body = _body(family, hk_kA)
    if hk_kA is None:
        del body["Hk_kA"]
    d = _save(body)
    assert d.get("ok"), d
    spec = materials_db.get(d["id"])
    knee = mc.HK_RATIO[family] * 1600e3
    assert spec["family"] == family and spec["hk_given"] is False
    assert spec["Hk"] == pytest.approx(knee, rel=1e-12)
    assert _model_knee_A_m(d["id"]) == pytest.approx(knee, abs=0.05)              # с ним и считает модель
    row = next(m for m in client.get("/api/materials").json()["magnets"] if m["id"] == d["id"])
    assert row["hk_given"] is False and row["Hk_kA"] == pytest.approx(knee / 1e3, abs=0.05)   # «!» в таблице


def test_same_datasheet_different_family_gives_a_different_knee():
    """Семейство не подпись: при тех же данных колено NdFeB дальше от рабочей точки, чем у SmCo."""
    nd = materials_db.get(_save(_body("NdFeB"))["id"])
    sm = materials_db.get(_save(_body("SmCo"))["id"])
    assert nd["Hcj"] == sm["Hcj"] == 1600e3
    assert nd["Hk"] / sm["Hk"] == pytest.approx(mc.HK_RATIO["NdFeB"] / mc.HK_RATIO["SmCo"], rel=1e-12)
    assert nd["Hk"] - sm["Hk"] == pytest.approx(160e3, rel=1e-12)    # 1520 против 1360 кА/м


@pytest.mark.parametrize("family", mc.FAMILIES)
def test_given_knee_is_kept_whatever_the_family(family):
    d = _save(_body(family, hk_kA="1100"))
    spec = materials_db.get(d["id"])
    assert spec["hk_given"] is True and spec["Hk"] == pytest.approx(1100e3, rel=1e-12)
    assert _model_knee_A_m(d["id"]) == pytest.approx(1100e3, abs=0.05)


@pytest.mark.parametrize("family", ["", "Ferrite", "smco", None], ids=["пусто", "чужое", "регистр", "поля нет"])
def test_without_a_known_family_the_magnet_is_not_saved(family):
    """Колено без H_k выбирается по семейству — без семейства материал не заводится, а не получает чужое колено."""
    body = _body(family)
    if family is None:
        del body["family"]
    d = _save(body)
    assert "error" in d and "класс материала" in d["error"]
    assert materials_db.load_all() == {}


# --- правка справочной марки -----------------------------------------------------------------------------------

def _edit_form(spec: dict) -> dict:
    """Поля окна правки так, как их заполняет nmEdit в index.html: колено, не заданное паспортом, — пустое поле."""
    opt = (lambda v: "" if v is None else str(v))
    return {"family": "SmCo" if spec["family"] == "SmCo" else "NdFeB", "name": spec["name"], "Br": str(spec["Br"]),
            "Hcb_kA": f"{spec['Hcb'] / 1e3:.1f}",
            "Hk_kA": "" if spec.get("hk_given") is False else f"{spec['Hk'] / 1e3:.1f}",
            "BHmax": opt(spec.get("BHmax")), "T_max": opt(spec.get("T_max")), "Hcj_kA": f"{spec['Hcj'] / 1e3:.1f}",
            "alpha_Br": str(spec["alpha_Br"]), "gamma_Hc": str(spec["gamma_Hc"])}


def _catalog_grade(family: str) -> mc.MagnetGrade:
    """Справочная марка семейства, id которой не занят встроенным пресетом: по такому id (n35) расчёт берёт пресет,
    а не марку и не её правку (Л-116, открыто)."""
    return next(g for g in mc.CATALOG if g.family == family and g.id not in server._BUILTIN_MAGNETS)


def test_edit_form_leaves_the_knee_field_empty_when_the_knee_is_not_given():
    """Правило окна, на которое опирается сервер (и `_edit_form` выше): hk_given === false — поле H_k пустое."""
    assert "$('mat-Hk').value=sp.hk_given===false?'':" in INDEX_HTML.read_text(encoding="utf-8")


@pytest.mark.parametrize("family", mc.FAMILIES)
def test_catalog_grade_goes_to_the_edit_form_with_the_knee_not_given(family):
    g = _catalog_grade(family)
    sp = client.get(f"/api/materials/{g.id}").json()["spec"]
    assert sp["hk_given"] is False
    assert sp["Hk"] == pytest.approx(mc.HK_RATIO[family] * g.Hcj, rel=1e-12)    # само значение — прежнее
    assert (sp["BHmax"], sp["T_max"]) == (g.BHmax, g.T_max)                    # в форме — весь паспорт марки


@pytest.mark.parametrize("family", mc.FAMILIES)
def test_edited_catalog_grade_takes_the_knee_from_its_new_hcj(family):
    g = _catalog_grade(family)
    form = _edit_form(client.get(f"/api/materials/{g.id}").json()["spec"])
    assert form["Hk_kA"] == ""
    form["Hcj_kA"] = f"{g.Hcj / 1e3 * 1.10:.1f}"                      # свой замер H_cJ — на 10 % выше справочного
    d = _save({"id": g.id, **form})
    assert d.get("ok") and d["id"] == g.id, d                        # правка той же марки, а не копия
    knee = mc.HK_RATIO[family] * float(form["Hcj_kA"]) * 1e3
    spec = materials_db.get(g.id)
    assert spec["hk_given"] is False and spec["Hk"] == pytest.approx(knee, rel=1e-12)
    assert _model_knee_A_m(g.id) == pytest.approx(knee, abs=0.05)    # с ним и считает модель
    row = next(m for m in client.get("/api/materials").json()["magnets"] if m["id"] == g.id)
    assert row["edited"] is True and row["hk_given"] is False         # в таблице — «!» допущения, а не «из паспорта»
    assert row["Hk_kA"] == pytest.approx(knee / 1e3, abs=0.05)
