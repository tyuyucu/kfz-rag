"""api-kosten-schaetzung fuer openai-calls
preise in usd pro 1M token (stand april 2026)
bei preisaenderungen werte hier aktualisieren
nur fuer kennzahlen in den eval-laeufen die pipeline braucht das nicht
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    input_per_1m: float   # usd pro 1M input-tokens
    output_per_1m: float  # usd pro 1M output-tokens


# stand april 2026
PRICING: dict[str, ModelPricing] = {
    "gpt-4o-mini": ModelPricing(input_per_1m=0.15, output_per_1m=0.60),
    "gpt-4o": ModelPricing(input_per_1m=2.50, output_per_1m=10.00),
    "text-embedding-3-small": ModelPricing(input_per_1m=0.02, output_per_1m=0.0),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    """rechnet die usd-kosten fuer tokens des modells
    unbekannte modelle -> 0.0
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
    """grobe vorab-schaetzung der api-kosten fuer einen ragas-lauf
    wird beim start angezeigt damit man die groessenordnung sieht
    """
    gen_cost = num_questions * cost_usd(
        generator_model,
        avg_input_tokens_per_question,
        avg_output_tokens_per_question,
    )
    # ragas-judge bewertet pro frage ~2-3 kontexte mit je ~1500 token prompt
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
