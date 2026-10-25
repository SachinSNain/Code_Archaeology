from src.graph.nodes.query_classifier import classify_query_deterministic

LABELED_EXAMPLES: list[tuple[str, str]] = [
    ("What does this function do?", "what"),
    ("Explain what this class implements.", "what"),
    ("Describe the behavior of this method.", "what"),
    ("What is the purpose of the parse_body function?", "what"),
    ("What happens when this exception is raised?", "what"),
    ("When was this authentication module added?", "when"),
    ("When did this function last change?", "when"),
    ("What date was this file created?", "when"),
    ("What year was this class introduced?", "when"),
    ("When was the caching layer added?", "when"),
    ("Who last modified the retry logic?", "who"),
    ("Who wrote this function originally?", "who"),
    ("Which contributor added this test?", "who"),
    ("Who is the maintainer of this file?", "who"),
    ("Who approved this pull request?", "who"),
    ("Why is this implemented this way?", "why"),
    ("Why was this design chosen over alternatives?", "why"),
    ("What motivated this refactor?", "why"),
    ("Why does this function use a lock here?", "why"),
    ("What was the reasoning behind this architectural decision?", "why"),
    ("How has the auth system evolved over time?", "evolution"),
    ("Show me the history of changes to this module.", "evolution"),
    ("What is the evolution of the caching layer?", "evolution"),
    ("How did this API change across versions?", "evolution"),
    ("Give me a timeline of how this module was rewritten.", "evolution"),
    ("Show me similar implementations in the codebase.", "similar"),
    ("Find functions similar to this one.", "similar"),
    ("Are there other places with the same pattern?", "similar"),
    ("Find analogous classes in the codebase.", "similar"),
    ("Are there duplicate implementations of this logic elsewhere?", "similar"),
]


def test_deterministic_rules_meet_accuracy_bar():
    correct = 0
    fallthrough = 0
    for query, expected in LABELED_EXAMPLES:
        result = classify_query_deterministic(query)
        if result is None:
            fallthrough += 1
        elif result == expected:
            correct += 1

    accuracy = correct / len(LABELED_EXAMPLES)
    fallthrough_rate = fallthrough / len(LABELED_EXAMPLES)

    # Critical rule: fallthrough rate should stay under 10% on representative queries.
    assert fallthrough_rate <= 0.10, f"fallthrough rate too high: {fallthrough_rate:.1%}"
    assert accuracy >= 0.85, f"rule accuracy too low: {accuracy:.1%}"


def test_why_takes_priority_over_what_when_both_present():
    # "why did the evolution ..." should route as evolution (checked before why/what).
    assert classify_query_deterministic("Why did the evolution of the auth system happen?") == "evolution"


def test_ambiguous_query_falls_through_to_none():
    assert classify_query_deterministic("retry logic") is None
