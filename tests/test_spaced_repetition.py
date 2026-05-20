"""tests fuer die sm-2-logik (ohne db)"""

from __future__ import annotations

from datetime import datetime

from generation.spaced_repetition import SM2State, sm2_step, DEFAULT_EF, MIN_EF


def test_perfekte_antwort_erhoeht_ef_und_interval():
    s0 = SM2State()
    s1 = sm2_step(s0, 5, now=datetime(2026, 4, 21))
    assert s1.ef > DEFAULT_EF
    assert s1.repetitions == 1
    assert s1.interval_days == 1
    assert s1.next_review_at is not None

    s2 = sm2_step(s1, 5, now=datetime(2026, 4, 22))
    assert s2.interval_days == 6
    assert s2.repetitions == 2


def test_falsche_antwort_setzt_wiederholungen_zurueck():
    s = SM2State(ef=2.7, interval_days=15, repetitions=4)
    s2 = sm2_step(s, 1)
    assert s2.repetitions == 0
    assert s2.interval_days == 1
    # ef muss unter den vorigen wert fallen (rating 1 ist schlecht)
    assert s2.ef < 2.7


def test_ef_bleibt_oberhalb_des_minimums():
    s = SM2State(ef=MIN_EF)
    # rating 0 versucht ef weiter zu senken darf aber nicht unter MIN_EF fallen
    s2 = sm2_step(s, 0)
    assert s2.ef >= MIN_EF


def test_ungueltiges_rating_wirft():
    import pytest
    with pytest.raises(ValueError):
        sm2_step(SM2State(), 99)
