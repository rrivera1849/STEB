"""Unit tests for manual cluster ep_config resolution.

Pins the auto-vs-fixed handling in ``_expected_ep_config`` so the
benchmark_clustering script's manual-cluster path keeps producing tables
for clusters whose tasks default to ``n_episodes_per_class="auto"``
(``time``, ``style``, ``dialect``, etc. — see issue noted on PR).
"""
from scripts.benchmark_clustering.manual_cluster import _expected_ep_config


# ---------------------------------------------------------------------------
# Fixed (non-auto) defaults pass through unchanged regardless of resolver.
# ---------------------------------------------------------------------------

def test_fixed_default_returned_verbatim():
    """`"1_2"` is a fixed default; resolver must not be consulted."""
    assert _expected_ep_config("1_2", _resolver_should_not_be_called) == "1_2"


def test_negative_episode_size_default_returned_verbatim():
    """The retrieval preset default `"-1_1"` must round-trip."""
    assert _expected_ep_config("-1_1", _resolver_should_not_be_called) == "-1_1"


# ---------------------------------------------------------------------------
# `_auto` defaults consult the resolver to produce the on-disk form.
# ---------------------------------------------------------------------------

def test_auto_default_resolves_to_concrete_value():
    """`"1_auto"` + resolver returning 25 -> `"1_25"`."""
    assert _expected_ep_config("1_auto", lambda ep_size: 25) == "1_25"


def test_auto_default_passes_episode_size_to_resolver():
    """Resolver receives the episode_size parsed from the default string."""
    captured: list = []

    def resolver(ep_size: int) -> int:
        captured.append(ep_size)
        return 50

    _expected_ep_config("3_auto", resolver)
    assert captured == [3]


def test_auto_default_returns_none_when_resolver_returns_none():
    """Resolver failure (e.g. raw data missing) propagates as None."""
    assert _expected_ep_config("1_auto", lambda _ep_size: None) is None


def test_non_integer_episode_size_returns_none():
    """A malformed default like `"foo_auto"` cannot be resolved."""
    assert _expected_ep_config("foo_auto", lambda _ep_size: 25) is None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolver_should_not_be_called(_ep_size: int) -> int:
    raise AssertionError("Resolver should not be called for fixed defaults")
