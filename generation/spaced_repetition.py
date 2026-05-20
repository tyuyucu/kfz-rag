"""SM-2-Algorithmus (SuperMemo 2, Wozniak 1987) für den Quiz-Modus.

Verwaltet pro Nutzer und Frage den Easiness Factor (EF), das nächste
Wiederholungsintervall in Tagen und die Anzahl erfolgreicher
Wiederholungen in Folge. Auf dieser Basis kann der Quiz-Modus gezielt
Fragen auswählen, die jetzt oder demnächst zur Wiederholung anstehen.

Bewertung (Rating) pro Antwort, analog SuperMemo:
    0 = komplett vergessen
    1 = falsche Antwort, aber korrekt beim Nachdenken erkannt
    2 = falsche Antwort, die richtige fiel einem leicht ein
    3 = korrekte Antwort, aber mühsam erinnert
    4 = korrekte Antwort nach kurzem Zögern
    5 = perfekt und unmittelbar

Rückgabe: aktualisierter Nutzerstatus mit EF, Intervall, nächstem
Wiederholungszeitpunkt.

DB-Schema (wird vom Quiz-Modus genutzt):
    quiz_attempts(user_id, question_hash, ef, interval_days,
                  repetitions, next_review_at, last_rating, updated_at)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from db.database import get_connection

DEFAULT_EF = 2.5
MIN_EF = 1.3


# ─────────────────────────────────────────────────────────
# DB-Schema + Persistenz
# ─────────────────────────────────────────────────────────

SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS quiz_attempts (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    question_hash TEXT NOT NULL,
    ef REAL NOT NULL DEFAULT {DEFAULT_EF},
    interval_days INTEGER NOT NULL DEFAULT 0,
    repetitions INTEGER NOT NULL DEFAULT 0,
    next_review_at TIMESTAMP,
    last_rating INTEGER,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, question_hash)
);

CREATE INDEX IF NOT EXISTS idx_quiz_attempts_next_review
    ON quiz_attempts (user_id, next_review_at);
"""


@dataclass
class SM2State:
    ef: float = DEFAULT_EF
    interval_days: int = 0
    repetitions: int = 0
    last_rating: Optional[int] = None
    next_review_at: Optional[datetime] = None


def _clamp_ef(value: float) -> float:
    return max(MIN_EF, value)


def sm2_step(state: SM2State, rating: int, *, now: Optional[datetime] = None) -> SM2State:
    """Berechnet den neuen SM-2-Zustand nach einer Bewertung.

    Rating ≥ 3 gilt als korrekt, Rating < 3 setzt die Wiederholungen
    zurück.
    """
    if not 0 <= rating <= 5:
        raise ValueError(f"rating muss zwischen 0 und 5 liegen, erhielt {rating}")

    now = now or datetime.now()

    new_ef = _clamp_ef(
        state.ef + (0.1 - (5 - rating) * (0.08 + (5 - rating) * 0.02))
    )

    if rating < 3:
        new_reps = 0
        new_interval = 1
    else:
        if state.repetitions == 0:
            new_interval = 1
        elif state.repetitions == 1:
            new_interval = 6
        else:
            new_interval = max(1, round(state.interval_days * new_ef))
        new_reps = state.repetitions + 1

    return SM2State(
        ef=new_ef,
        interval_days=new_interval,
        repetitions=new_reps,
        last_rating=rating,
        next_review_at=now + timedelta(days=new_interval),
    )


def init_sm2_tables() -> None:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(SCHEMA_SQL)
        conn.commit()
        cur.close()
    finally:
        conn.close()


def load_state(user_id: str, question_hash: str) -> SM2State:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """SELECT ef, interval_days, repetitions, next_review_at, last_rating
                 FROM quiz_attempts
                WHERE user_id = %s AND question_hash = %s""",
            (user_id, question_hash),
        )
        row = cur.fetchone()
        cur.close()
        if not row:
            return SM2State()
        return SM2State(
            ef=float(row[0]),
            interval_days=int(row[1]),
            repetitions=int(row[2]),
            next_review_at=row[3],
            last_rating=row[4],
        )
    finally:
        conn.close()


def save_state(user_id: str, question_hash: str, state: SM2State) -> None:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO quiz_attempts
                   (user_id, question_hash, ef, interval_days, repetitions,
                    next_review_at, last_rating, updated_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
               ON CONFLICT (user_id, question_hash)
               DO UPDATE SET ef = EXCLUDED.ef,
                             interval_days = EXCLUDED.interval_days,
                             repetitions = EXCLUDED.repetitions,
                             next_review_at = EXCLUDED.next_review_at,
                             last_rating = EXCLUDED.last_rating,
                             updated_at = CURRENT_TIMESTAMP""",
            (
                user_id,
                question_hash,
                state.ef,
                state.interval_days,
                state.repetitions,
                state.next_review_at,
                state.last_rating,
            ),
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def due_questions(user_id: str, limit: int = 10, *, now: Optional[datetime] = None) -> list[str]:
    """Liefert die `question_hash`es, deren Review-Zeitpunkt erreicht ist,
    sortiert nach Fälligkeit (älteste zuerst).
    """
    now = now or datetime.now()
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """SELECT question_hash FROM quiz_attempts
                WHERE user_id = %s
                  AND (next_review_at IS NULL OR next_review_at <= %s)
                ORDER BY next_review_at NULLS FIRST
                LIMIT %s""",
            (user_id, now, limit),
        )
        rows = cur.fetchall()
        cur.close()
        return [r[0] for r in rows]
    finally:
        conn.close()
