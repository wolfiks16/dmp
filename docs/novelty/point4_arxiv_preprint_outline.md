# arXiv-препринт по пункту (4) — структура и позиционирование

> Цель: зафиксировать **приоритет** на интегрированном вкладе. Депонировать как только будет минимальная демонстрация (срабатывание knee-переключения). Позиционирование (ниже) фиксируется уже сейчас.
>
> Основание: `docs/novelty/point4_certified_surrogate_risk.md`. Дата: 2026-06-02.

---

## Стратегия позиционирования (критично)

**НЕ** строить вокруг «нейросеть/оператор + conformal prediction» — это прайор-арт (UQNO/TMLR, Conformalized-DeepONet/Physica D, …). Заявляемая новизна — **тройка, которой нет ни у кого**:
1. CP впервые на **гибридном FEM/BEM-суррогате электрической машины** (магнитостатика open-domain);
2. **сертифицированная иерархия** FOM(FEM/BEM) → ROM(апостериорная RB-оценка) → NN(CP-интервал) как единый аппарат;
3. **переключение уровней trust-region MO-оптимизатора по физическому критерию** — близости рабочей точки магнита к knee-point по (B,T), а не только по ошибке суррогата.

Явная фраза в Introduction: *"Conformal prediction for neural/operator surrogates is established; our contribution is its integration into a certified FEM/BEM→ROM→NN hierarchy for open-domain PMSM design, with trust-region level-switching gated by a physical demagnetization-knee criterion."*

---

## Варианты заголовка (англ., для arXiv/IEEE)

1. *Demagnetization-Knee-Aware Certified Multi-Fidelity Optimization of Open-Domain Permanent-Magnet Machines* (FEM/BEM–ROM–Neural Surrogate hierarchy).
2. *A Certified FEM/BEM→ROM→Conformal-Surrogate Hierarchy with Physics-Gated Trust-Region Switching for PMSM Design.*
3. *Trust-Region Multi-Objective PMSM Design with Conformal-Certified Surrogates and Demagnetization-Knee Level Switching.*

Категории arXiv: `math.NA` + `cs.CE` (+ `cs.LG`). Целевой журнал: IEEE Trans. Magnetics / COMPEL.

---

## Структура

**Abstract.** Проблема: проектирование PMSM БПЛА у границы необратимой демагнетизации требует дорогого точного FEM/BEM; нужно ускорение *с гарантией*. Вклад: сертифицированная иерархия + физически-управляемое переключение. Цифры: ускорение ×N при гарантированном покрытии и без нарушения demag-границы.

**1. Introduction.** (a) Инженерный контекст — open-domain outrunner SPM, knee-риск в форсаже; (b) почему «ускорение с сертификатом» — каждый уровень доверия даёт *проверяемую* границу; (c) **позиционирование** (фраза выше): CP-на-суррогате — заимствовано, ново — EM-интеграция + knee-переключение; (d) вклад списком (3 пункта).

**2. Related work (с явным разграничением).** Три кластера + по строке разграничения (готовы в `point4_certified_surrogate_risk.md`):
- CP на нейрооператорах: UQNO (Ma et al., TMLR 2024, 2402.01960), Conformalized-DeepONet (Moya et al., Physica D 2025, 2402.15406), Millard/Lindemann/Baheri (2509.04623).
- CP на PINN: C-PINN (**Podina, Torabi-Rad, Kohandel**, 2405.08111), Yu/Ho/Wang (2509.13717), Gopakumar (2502.04406, 2408.09881).
- Суррогаты PMSM: Partovizadeh/Schöps/Loukrezis (2412.06485, Eng. w/ Computers 2025) — MC-UQ без CP/сертификата; Parekh PIBO-MESA; Lei/Bramerdorfer (IEEE TEC 2021).
- Multi-fidelity TR: March/Willcox, Klein & Ohlberger, Xu & Darve — **переключение по ошибке, не по физике** (⚠ до-проверить RQ4).

**3. Problem setting.** Open-domain SPM PMSM; гибридный FEM/BEM FOM (ссылка на `coupling.md`); целевые (момент, ripple, КПД, demag-margin); определение knee-point по (B,T) и индикатора близости `δ_knee`.

**4. Certified multi-level hierarchy.**
- Уровень 0 (FOM): гибрид FEM/BEM, эталон.
- Уровень 1 (ROM): RB/POD + **апостериорная оценка** `Δ_ROM(μ)` (сертификат уровня 1).
- Уровень 2 (NN): нейрооператор-суррогат + **split conformal** интервал `Δ_CP(μ)` при покрытии `1−α` (сертификат уровня 2). Явно: CP — стандартный, применён к EM-суррогату.

**5. Trust-region MO с физически-управляемым переключением (ядро новизны).**
- Правило выбора уровня: `level(μ) = f( max(Δ_ROM, Δ_CP), δ_knee(B,T;μ), Δ_TR )` — выписать формулой: у knee (`δ_knee` мал) форсировать FOM/ROM независимо от дешевизны NN; вдали — NN.
- TR-управление радиусом, согласование сертификата с радиусом доверия; MO/Pareto (NSGA-II + adjoint local refine, ссылка на фазу E).
- Теорема/утверждение: при таком переключении итерации не нарушают demag-границу с гарантией покрытия.

**6. Demonstration (gating-деливерабл).** Open-domain SPM PMSM: (a) срабатывание переключения у knee (график `Δ_CP` vs `δ_knee`); (b) ускорение vs чистый FOM; (c) эмпирическое покрытие CP `≈1−α`; (d) Pareto-фронт не хуже NSGA-II-FOM. **Это минимальный контент, который нужно успеть для депонирования.**

**7. Conclusion + приоритетная формулировка.**

---

## Что нужно для депонирования (минимум)
- Раздел 5 (правило переключения, выписанное) — можно сформулировать **уже сейчас** (не требует кода).
- Раздел 6 — **минимальная** демонстрация: даже грубый суррогат + CP на одном PMSM-кейсе со срабатыванием knee-переключения. Это gating-элемент; форсировать в начале фазы F.
- Разделы 1–4 — на основе `formulation_bounded.md`, `coupling.md`, `point4_certified_surrogate_risk.md`.

**Тактика приоритета:** допустимо «methods + preliminary results» препринт (раздел 6 — preliminary), чтобы зафиксировать дату; полные результаты — в журнальную версию. Мониторить Schöps/Loukrezis, Baheri.
