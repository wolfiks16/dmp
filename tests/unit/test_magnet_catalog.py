from __future__ import annotations

import pytest

from magcore.constants import MU0
from magcore.domain import magnet_catalog as mc


def test_catalog_covers_both_families_and_all_sources() -> None:
    s = mc.families_summary()
    assert set(s) == {"NdFeB", "SmCo"}
    assert s["NdFeB"] > 50 and s["SmCo"] >= 15
    assert sum(s.values()) == len(mc.CATALOG)
    assert {g.source for g in mc.CATALOG} == set(mc.SOURCES)


def test_ids_are_unique() -> None:
    ids = [g.id for g in mc.CATALOG]
    assert len(ids) == len(set(ids))


def test_every_grade_is_physically_orderly_and_model_buildable() -> None:
    """Инварианты, без которых кривая B(H) не строится: 0 < H_cB <= H_cJ, H_k < H_cJ, B_r > 0."""
    for g in mc.CATALOG:
        assert g.Br > 0.0, g.grade
        assert 0.0 < g.Hcb <= g.Hcj, g.grade
        assert g.Hk < g.Hcj, g.grade
        assert 0.0 < g.alpha_Br < 1.0 and 0.0 < g.gamma_Hc < 2.0, g.grade
        mc.to_magnet(g.id)          # падает, если параметры несовместимы с моделью


@pytest.mark.parametrize(
    "gid,grade,Br,Hcb_kA,Hcj_kA,src",
    [                                        # сверка переноса из источника, по одной марке на источник
        ("n42sh", "N42SH", 1.310, 955.0, 1592.0, "arnold"),
        ("ks25dts_240", "КС25ДЦ-240", 1.10, 780.0, 900.0, "gost21559"),
        ("nmb_250_130", "НмБ 250/130", 1.13, 840.0, 1300.0, "gost52956"),
    ],
)
def test_transcription_matches_source_tables(gid, grade, Br, Hcb_kA, Hcj_kA, src) -> None:
    g = mc.by_id(gid)
    assert (g.grade, g.source) == (grade, src)
    assert g.Br == pytest.approx(Br, rel=1e-6)
    assert g.Hcb == pytest.approx(Hcb_kA * 1e3, rel=1e-9)
    assert g.Hcj == pytest.approx(Hcj_kA * 1e3, rel=1e-9)


def test_search_finds_cyrillic_and_latin_case_insensitively() -> None:
    ks = [g.grade for g in mc.find("кс25дц")]
    assert ks == ["КС25ДЦ-150", "КС25ДЦ-175", "КС25ДЦ-190",
                  "КС25ДЦ-210", "КС25ДЦ-225", "КС25ДЦ-240"]
    assert [g.grade for g in mc.find("42sh")] == ["N42SH", "N42SHX"]
    assert mc.find("такой марки нет") == []


def test_family_filter_is_exclusive_and_complete() -> None:
    nd = mc.find(family="NdFeB")
    sm = mc.find(family="SmCo")
    assert all(g.family == "NdFeB" for g in nd)
    assert all(g.family == "SmCo" for g in sm)
    assert len(nd) + len(sm) == len(mc.CATALOG)
    assert all(g.source == "gost21559" for g in mc.find(family="SmCo", source="gost21559"))


def test_to_magnet_reproduces_catalog_parameters() -> None:
    g = mc.by_id("n42sh")
    m = mc.to_magnet(g.id)
    assert m.Br(20.0) == pytest.approx(g.Br, rel=1e-9)
    assert m.Hk(20.0) == pytest.approx(g.Hk, rel=1e-9)
    assert m.mu_rec == pytest.approx(g.Br / (MU0 * g.Hcb), rel=1e-9)
    # колено у N42SH лежит ЗА H_cB ⇒ нуль кривой совпадает с даташитной H_cB
    assert m.curve_at(20.0).Hcb_actual() == pytest.approx(g.Hcb, rel=1e-3)


def test_measured_knee_overrides_the_assumption() -> None:
    """H_k — ДОПУЩЕНИЕ hk_ratio*H_cJ; измеренное значение должно его вытеснять."""
    g = mc.by_id("ks25dts_240")
    assert g.Hk == pytest.approx(g.hk_ratio * g.Hcj, rel=1e-12)
    m = mc.to_magnet(g.id, hk=0.70 * g.Hcj)
    assert m.Hk(20.0) == pytest.approx(0.70 * g.Hcj, rel=1e-9)
    assert m.Hk(20.0) < mc.to_magnet(g.id).Hk(20.0)


def test_knee_ratio_depends_on_family() -> None:
    """У NdFeB петля прямоугольнее ⇒ колено ближе к H_cJ, чем у Sm-Co."""
    assert mc.HK_RATIO["NdFeB"] > mc.HK_RATIO["SmCo"]
    for g in mc.CATALOG:
        assert g.hk_ratio == pytest.approx(mc.HK_RATIO[g.family])
        assert g.Hk == pytest.approx(mc.HK_RATIO[g.family] * g.Hcj, rel=1e-12)
    nd, sm = mc.by_id("n42sh"), mc.by_id("ks25dts_240")
    assert nd.Hk / nd.Hcj > sm.Hk / sm.Hcj


def test_gost_ndfeb_uses_conservative_edge_of_the_range() -> None:
    """У ГОСТ Р 52956 параметры — диапазоны; берём НИЖНИЕ границы и ХУДШИЕ коэффициенты."""
    g = mc.by_id("nmb_200_80")
    assert g.Br == pytest.approx(1.00, rel=1e-9)      # диапазон 1,00-1,08 -> нижняя
    assert g.Hcb == pytest.approx(680e3, rel=1e-9)    # диапазон 680-700 -> нижняя
    assert g.alpha_Br == pytest.approx(0.12) and g.gamma_Hc == pytest.approx(0.59)
    assert "ДИАПАЗОН" in g.note.upper()


def test_unknown_id_and_bad_family_are_explicit_errors() -> None:
    with pytest.raises(KeyError):
        mc.by_id("нет такой марки")
    with pytest.raises(KeyError):
        mc.to_magnet("нет такой марки")
    with pytest.raises(ValueError):
        mc.find(family="AlNiCo")
