from collections import Counter

import numpy as np
import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from magcore.domain import magnet_catalog as mc  # noqa: E402
from webapp import materials_db, server  # noqa: E402

# Библиотека материалов: запись — это id, и расчёт находит материал только по нему. Найдено 2026-10-01.
# Л-116. Встроенные пресеты перекрывали библиотеку: у Arnold N35 и пресета «NdFeB N35 (представит.)» был один id
#   'n35' (в таблице две строки, считался пресет); правка встроенного магнита ложилась записью с его id, но расчёт
#   брал пресет; ✎ у встроенных сталей отвечал «материал не найден». Теперь id не совпадают, а правка встроенного —
#   копия с новым id.
# Л-117. В правке магнита можно было сохранить со вкладки «Сталь»: свой магнит молча становился сталью, а справочная
#   марка-сталь роняла /api/materials. Теперь при правке вид записи не меняется — и на сервере, и в базе.

client = TestClient(server.app)


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    """Своя база на каждый тест — библиотеку материалов пользователя (webapp/materials.db) не трогаем."""
    monkeypatch.setattr(materials_db, "DB_PATH", tmp_path / "materials.db")
    monkeypatch.setattr(materials_db, "LEGACY_JSON", tmp_path / "materials.json")


def _post(url: str, body: dict) -> dict:
    r = client.post(url, json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _library() -> dict:
    r = client.get("/api/materials")
    assert r.status_code == 200, r.text                  # прежде справочная марка-сталь давала здесь 500
    return r.json()


def _br20(mid: str) -> float:
    d = _post("/api/magnet_curve", {"material": mid, "T": 20.0})
    assert "error" not in d, d
    return d["Br"]


def _magnet_body(mid=None, name="Мой SmCo", Br="1.05"):
    return {"id": mid, "family": "SmCo", "name": name, "Br": Br, "Hcb_kA": "780", "Hk_kA": "", "Hcj_kA": "1600",
            "alpha_Br": "0.03", "gamma_Hc": "0.20"}


def _steel_body(mid=None, name="Моя сталь", points=((0, 0), (100, 1.0), (1000, 1.6), (100000, 2.1))):
    return {"id": mid, "kind": "steel", "name": name, "points": [list(p) for p in points]}


# --- Л-116: id в библиотеке единственен --------------------------------------------------------------------------

def test_builtin_and_catalog_ids_do_not_collide():
    assert set(server._BUILTIN_IDS) & set(mc._BY_ID) == set()


def test_library_list_has_unique_ids():
    lib = _library()
    ids = [m["id"] for m in lib["magnets"] + lib["steels"]]
    assert [i for i, n in Counter(ids).items() if n > 1] == []


def test_arnold_n35_and_the_builtin_n35_are_two_materials_computed_with_their_own_data():
    rows = {m["id"]: m for m in _library()["magnets"]}
    assert rows["n35"]["name"] == server._BUILTIN_MAGNETS["n35"]["name"]
    assert rows["n35_arnold"]["name"] == "N35" and rows["n35_arnold"]["source"] == "arnold"
    assert _br20("n35") == pytest.approx(server._magnet_by_id("n35").Br0, abs=1e-4)          # пресет: 1,19 Тл
    assert _br20("n35_arnold") == pytest.approx(mc.by_id("n35_arnold").Br, abs=1e-4)        # Arnold: 1,21 Тл
    assert _br20("n35") != _br20("n35_arnold")


@pytest.mark.parametrize("mid", sorted(server._BUILTIN_MAGNETS))
def test_builtin_magnet_goes_to_the_form_as_a_copy_with_its_own_data(mid):
    d = client.get(f"/api/materials/{mid}").json()
    assert d["edit"] == "copy"
    m, sp = server._magnet_by_id(mid), d["spec"]
    assert (sp["kind"], sp["name"], sp["family"]) == ("magnet", server._BUILTIN_MAGNETS[mid]["name"],
                                                      server._BUILTIN_MAGNETS[mid]["family"])
    assert (sp["Br"], sp["Hcb"], sp["Hk"], sp["Hcj"]) == (m.Br0, m.Hcb0, m.Hk0, m.Hcj0)
    assert next(r for r in _library()["magnets"] if r["id"] == mid)["edit"] == "copy"


@pytest.mark.parametrize("mid", sorted(server._BUILTIN_STEELS))
def test_builtin_steel_goes_to_the_form_as_a_copy_with_its_bh_table(mid):
    d = client.get(f"/api/materials/{mid}").json()            # прежде — «материал не найден»
    assert d["edit"] == "copy" and d["spec"]["kind"] == "steel"
    c = server._steel_by_id(mid)
    assert np.array_equal(d["spec"]["H"], c.H_values) and np.array_equal(d["spec"]["B"], c.B_values)
    # копия, сохранённая формой как есть (новый материал), — та же кривая под новым id; встроенная не тронута
    s = _post("/api/materials", _steel_body(None, d["spec"]["name"] + " — копия", zip(d["spec"]["H"], d["spec"]["B"])))
    assert s.get("ok") and s["id"].startswith("cust_"), s
    copy = server._steel_by_id(s["id"])
    assert np.array_equal(copy.H_values, c.H_values) and np.array_equal(copy.B_values, c.B_values)
    assert np.array_equal(server._steel_by_id(mid).B_values, c.B_values)


def test_copy_of_a_builtin_magnet_is_a_new_material_and_the_builtin_stays():
    br0 = _br20("ndfeb")
    sp = client.get("/api/materials/ndfeb").json()["spec"]
    body = {"id": None, "family": sp["family"], "name": sp["name"] + " — копия", "Br": "1.30",
            "Hcb_kA": f"{sp['Hcb'] / 1e3:.1f}", "Hk_kA": f"{sp['Hk'] / 1e3:.1f}", "Hcj_kA": f"{sp['Hcj'] / 1e3:.1f}",
            "alpha_Br": str(sp["alpha_Br"]), "gamma_Hc": str(sp["gamma_Hc"])}
    d = _post("/api/materials", body)
    assert d.get("ok") and d["id"].startswith("cust_"), d
    assert _br20(d["id"]) == pytest.approx(1.30, abs=1e-4)                 # правка в копии считается
    assert _br20("ndfeb") == br0                                           # встроенный — как был
    assert Counter(m["id"] for m in _library()["magnets"])["ndfeb"] == 1   # прежде — вторая строка 'ndfeb'


@pytest.mark.parametrize("mid, body", [("ndfeb", _magnet_body), ("smco", _magnet_body), ("steel", _steel_body),
                                       ("steel10", _steel_body), ("m270", _steel_body)],
                         ids=["ndfeb", "smco", "steel", "steel10", "m270"])
def test_builtin_is_never_overwritten(mid, body):
    d = _post("/api/materials", body(mid))
    assert "error" in d and "встроенный" in d["error"]
    assert materials_db.load_all() == {}


# --- Л-117: при правке вид записи не меняется -------------------------------------------------------------------

def test_own_magnet_is_not_overwritten_by_a_steel():
    mid = _post("/api/materials", _magnet_body())["id"]
    d = _post("/api/materials", _steel_body(mid))
    assert "error" in d and "вид материала" in d["error"]
    assert materials_db.get(mid)["kind"] == "magnet" and _br20(mid) == pytest.approx(1.05, abs=1e-4)


def test_own_steel_is_not_overwritten_by_a_magnet():
    mid = _post("/api/materials", _steel_body())["id"]
    d = _post("/api/materials", _magnet_body(mid))
    assert "error" in d and "вид материала" in d["error"]
    assert materials_db.get(mid)["kind"] == "steel" and server._is_steel(mid)


def test_catalog_grade_is_not_overwritten_by_a_steel_and_the_library_still_opens():
    g = mc.CATALOG[0]
    d = _post("/api/materials", _steel_body(g.id))
    assert "error" in d and "сталь вместо магнита" in d["error"]
    assert materials_db.load_all() == {}
    row = next(m for m in _library()["magnets"] if m["id"] == g.id)
    assert row["Br"] == round(g.Br, 4) and not server._is_steel(g.id)


@pytest.mark.parametrize("body", [_magnet_body, _steel_body], ids=["магнит", "сталь"])
def test_edit_of_a_material_that_is_not_in_the_library_is_refused(body):
    d = _post("/api/materials", body("cust_0000000000"))
    assert "error" in d and "не найден" in d["error"]
    g = mc.CATALOG[1]
    assert _post("/api/materials/delete", {"id": g.id})["ok"]                # справочная марка удалена из библиотеки
    d = _post("/api/materials", _magnet_body(g.id))
    assert "error" in d and "не найден" in d["error"]
    assert materials_db.load_all() == {}


def test_own_material_and_catalog_grade_are_still_edited_in_place():
    mid = _post("/api/materials", _magnet_body(Br="1.05"))["id"]
    assert _post("/api/materials", _magnet_body(mid, Br="1.07"))["id"] == mid
    assert _br20(mid) == pytest.approx(1.07, abs=1e-4)
    g = next(g for g in mc.CATALOG if g.family == "SmCo")
    body = {"id": g.id, "family": g.family, "name": g.grade, "Br": str(round(g.Br + 0.02, 4)),
            "Hcb_kA": str(g.Hcb / 1e3), "Hk_kA": "", "Hcj_kA": str(g.Hcj / 1e3), "alpha_Br": str(g.alpha_Br),
            "gamma_Hc": str(g.gamma_Hc)}
    assert _post("/api/materials", body)["id"] == g.id
    assert _br20(g.id) == pytest.approx(g.Br + 0.02, abs=1e-4)
    assert client.get(f"/api/materials/{g.id}").json()["edit"] == "in_place"
