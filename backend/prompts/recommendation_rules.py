"""Recommendation rules for the verdict generator (Issue #9)."""
from models import RecommendationRule

RECOMMENDATION_RULES: list[RecommendationRule] = [
    RecommendationRule(
        rule_id="R01",
        root_cause="ambiguous_retrieval_weak_generation",
        pipeline_component="top-k and reranker",
        action="reduce top-k or add a reranker to force selectivity before generation",
        render_hint="emphasize that the retriever couldn't decide what was relevant",
    ),
    RecommendationRule(
        rule_id="R02",
        root_cause="decisive_retrieval_generation_overstep",
        pipeline_component="chunk size",
        action="increase chunk size or use overlapping windows — the model is filling gaps with parametric knowledge",
        render_hint="emphasize that retrieval worked but chunks were too short",
    ),
    RecommendationRule(
        rule_id="R03",
        root_cause="decisive_retrieval_wrong_content",
        pipeline_component="chunk boundaries and embedding model",
        action="review chunk boundaries — relevant information may be split across chunks, or the embedding model may not suit this domain",
        render_hint="emphasize that the retriever was confident but retrieved the wrong thing",
    ),
    RecommendationRule(
        rule_id="R04",
        root_cause="overconfident_generation_on_weak_evidence",
        pipeline_component="prompt template",
        action="update the generation prompt to instruct the model to hedge when retrieved context is ambiguous",
        render_hint="emphasize that this is a prompt engineering problem, not a retrieval problem",
    ),
    RecommendationRule(
        rule_id="R05",
        root_cause="noisy_context_reaching_generator",
        pipeline_component="similarity threshold and top-k",
        action="raise the similarity threshold or reduce top-k to cut low-relevance chunks before generation",
        render_hint="emphasize that too much borderline content is diluting the signal",
    ),
    RecommendationRule(
        rule_id="R06",
        root_cause="underconfident_generation_on_strong_evidence",
        pipeline_component="prompt template",
        action="update the generation prompt to allow confident assertion when retrieved context directly supports a claim",
        render_hint="emphasize that this erodes user trust unnecessarily",
    ),
    RecommendationRule(
        rule_id="R07",
        root_cause="pipeline_healthy",
        pipeline_component="none",
        action="no changes indicated",
        render_hint="emphasize what's working — decisive retrieval, faithful generation, calibrated confidence",
    ),
    RecommendationRule(
        rule_id="R08",
        root_cause="query_phrasing_mismatch",
        pipeline_component="user query",
        action="rephrase the question — the corpus contains relevant information but the query didn't match the embedding space",
        render_hint="emphasize that the system can answer something close to what they asked",
    ),
    RecommendationRule(
        rule_id="R09",
        root_cause="corpus_coverage_gap",
        pipeline_component="knowledge base",
        action="the corpus does not appear to cover this topic — review the suggested questions to understand what the system can answer",
        render_hint="emphasize that this is a data coverage problem, not a query problem",
    ),
]


def get_rule(rule_id: str) -> RecommendationRule:
    return next(r for r in RECOMMENDATION_RULES if r.rule_id == rule_id)
