"""API-Kosten-Schätzung für OpenAI-Aufrufe.

Preise in USD pro 1M Token. Stand: April 2026 — bei Preisänderungen
müssen die Werte hier aktualisiert werden. Die Pipeline verlässt sich
nicht darauf; die Werte dienen ausschließlich der Kennzahlenerhebung
in den Evaluations-Läufen.

Quelle: openai.com/api/pricing (Abruf April 2026).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    input_per_1m: float   # USD pro 1M Input-Tokens
    output_per_1m: float  # USD pro 1M Output-Tokens


# Stand April 2026
PRICING: dict[str, ModelPricing] = {
    "gpt-4o-mini": ModelPricing(input_per_1m=0.15, output_per_1m=0.60),
    "gpt-4o": ModelPricing(input_per_1m=2.50, output_per_1m=10.00),
    "text-embedding-3-small": ModelPricing(input_per_1m=0.02, output_per_1m=0.0),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    """Errechnet die USD-Kosten für eine Anzahl Tokens des gegebenen Modells.

    Unbekannte Modelle → 0.0 (mit Hinweis in der Tracing-DB dokumentiert).
    """
    p = PRICING.get(model)
    if p is None:
        return 0.0
    return (input_tokens / 1_000_000) * p.input_per_1m + (
        output_tokens / 1_000_000
    ) * p.output_per_1m


def estimate_run_cost(
    num_questions: int,
    avg_input_tokens_per_question: int = 4000,
    avg_output_tokens_per_question: int = 400,
    generator_model: str = "gpt-4o-mini",
    judge_model: str = "gpt-4o",
    embedding_tokens_per_question: int = 80,
) -> dict[str, float]:
    """Grobe Vorabschätzung der API-Kosten eines RAGAS-Laufs.

    Wird beim Start der Evaluations-Skripte angezeigt, damit der Nutzer
    die Größenordnung sieht, bevor Kosten anfallen.
    """
    gen_cost = num_questions * cost_usd(
        generator_model,
        avg_input_tokens_per_question,
        avg_output_tokens_per_question,
    )
    # RAGAS-Judge bewertet pro Frage ~2-3 Kontexte mit je ~1500 Token Prompt
    judge_cost = num_questions * cost_usd(judge_model, 5000, 300)
    embed_cost = num_questions * cost_usd(
        "text-embedding-3-small", embedding_tokens_per_question, 0
    )
    total = gen_cost + judge_cost + embed_cost
    return {
        "generator_usd": gen_cost,
        "judge_usd": judge_cost,
        "embedding_usd": embed_cost,
        "total_usd": total,
    }
