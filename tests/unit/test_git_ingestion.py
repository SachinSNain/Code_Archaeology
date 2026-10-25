from src.ingestion.git_ingestion import parse_linked_issues, parse_linked_pr


def test_parse_linked_issues_single():
    assert parse_linked_issues("Fixes #123") == [123]


def test_parse_linked_issues_multiple_and_case_insensitive():
    msg = "Closes #12 and Related to #34, also RESOLVES #56"
    assert parse_linked_issues(msg) == [12, 34, 56]


def test_parse_linked_issues_none():
    assert parse_linked_issues("Update release notes") == []


def test_parse_linked_pr_squash_merge_style():
    assert parse_linked_pr("Add retry logic to HTTP client (#4821)") == 4821


def test_parse_linked_pr_none():
    assert parse_linked_pr("Update release notes") is None


def test_parse_linked_pr_only_matches_trailing_reference():
    # A "#" earlier in the subject (e.g. referencing an issue) should not be
    # mistaken for the trailing "(#N)" PR-merge marker.
    assert parse_linked_pr("Fix bug from #100 discussion") is None
