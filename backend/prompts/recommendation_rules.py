"""Recommendation rules for the verdict generator (Issue #9).

This file is a stub containing R08 and R09, added in Issue #8.
match_rule() and remaining rules will be implemented in Issue #9.
"""
from models import RecommendationRule

RECOMMENDATION_RULES: list[RecommendationRule] = [
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
