"""The optillm plugin, loaded exactly the way optillm loads it.

`spec_from_file_location` on the path under `plugins/optillm/plugins/`, because
that discovery convention *is* the interface: a file optillm cannot find, or a
`SLUG` it does not see, fails silently as "unknown approach" at sweep time.

optillm's own approach stack is not imported. The plugin's `dispatch` seam stands
in for it, so what these tests cover is the plugin's whole contribution: predict,
delegate, and report the prediction in the field the harness reads. Routing a
real request through a real server is the smoke test in supek/README.md - the
same precedent the job scripts and the container build follow.
"""

import importlib.util
from pathlib import Path

import pytest

from router_lab.labels import FINETUNED_ROUTER

PLUGIN_PATH = (
    Path(__file__).resolve().parents[1]
    / "plugins"
    / "optillm"
    / "plugins"
    / "router_ft_plugin.py"
)


@pytest.fixture
def plugin():
    """The plugin module, loaded from its path like `load_plugins()` does."""
    spec = importlib.util.spec_from_file_location("router_ft_plugin", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def routed(plugin, monkeypatch):
    """A plugin whose router always picks `bon` and whose dispatch is recorded."""
    calls = []

    class Router:
        def predict(self, system_prompt, query):
            calls.append(("predict", system_prompt, query))
            return "bon"

    def dispatch(approach, system_prompt, initial_query, client, model, request_config):
        calls.append(("dispatch", approach, request_config))
        return "the answer is 4", 128

    monkeypatch.setattr(plugin, "load_router", lambda: Router())
    monkeypatch.setattr(plugin, "dispatch", dispatch)
    return plugin, calls


def test_the_plugin_registers_under_its_own_approach_name(plugin):
    assert plugin.SLUG == FINETUNED_ROUTER == "router_ft"
    assert plugin.SLUG != "router", "the stock router keeps its own name"


def test_the_plugin_loads_without_the_repository_on_the_import_path(plugin):
    """optillm loads it by path from whatever directory the server was started in."""
    assert callable(plugin.run)


def test_it_routes_the_query_to_what_the_router_predicted(routed):
    plugin, calls = routed

    plugin.run("Be careful.", "2 + 2?", client=None, model="qwen3-8b")

    assert ("predict", "Be careful.", "2 + 2?") in calls
    assert [c for c in calls if c[0] == "dispatch"] == [("dispatch", "bon", {})]


def test_optillms_own_control_keys_are_not_forwarded_to_the_client(routed):
    """A `none` prediction forwards these to the OpenAI client, which rejects them."""
    plugin, calls = routed

    plugin.run(
        "p",
        "q",
        client=None,
        model="qwen3-8b",
        request_config={
            "optillm_approach": "router_ft",
            "temperature": 0.6,
            "max_tokens": 1536,
        },
    )

    forwarded = [c[2] for c in calls if c[0] == "dispatch"][0]
    assert forwarded == {"temperature": 0.6, "max_tokens": 1536}


def test_the_callers_parameters_survive_the_stripping(plugin):
    assert plugin.forwardable({"n": 2, "optillm_approach": "router_ft"}) == {"n": 2}
    assert plugin.forwardable(None) == {}


def test_the_prediction_is_reported_in_the_field_the_harness_reads(routed):
    plugin, _ = routed

    response, tokens = plugin.run("p", "q", client=None, model="qwen3-8b")

    assert response[plugin.ROUTER_APPROACH_FIELD] == "bon"
    assert response["choices"][0]["message"]["content"] == "the answer is 4"
    assert response["usage"]["completion_tokens"] == tokens == 128


def test_the_response_carries_the_pair_the_server_returns_verbatim_on(routed):
    """Without both `choices` and `usage`, the server rebuilds it and the
    prediction is lost."""
    plugin, _ = routed

    response, _ = plugin.run("p", "q", client=None, model="qwen3-8b")

    assert {"choices", "usage"} <= set(response)


def test_a_passthrough_response_keeps_its_own_usage(plugin):
    """The `none` approach hands back vLLM's whole response, prompt tokens and all."""
    upstream = {
        "model": "qwen3-8b",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "4"}}],
        "usage": {"completion_tokens": 12, "total_tokens": 250},
    }

    response = plugin.as_response(upstream, 0, "none", model="qwen3-8b")

    assert response["usage"] == {"completion_tokens": 12, "total_tokens": 250}
    assert response[plugin.ROUTER_APPROACH_FIELD] == "none"
    assert response["choices"] == upstream["choices"]


def test_several_completions_become_several_choices(plugin):
    response = plugin.as_response(["4", "four"], 20, "bon", model="qwen3-8b")

    assert [c["message"]["content"] for c in response["choices"]] == ["4", "four"]
    assert [c["index"] for c in response["choices"]] == [0, 1]


def test_a_failure_is_raised_rather_than_routed_around(plugin, monkeypatch):
    """A silent fallback would record rows that look like routing decisions."""

    def unconfigured():
        raise RuntimeError("no finetuned router checkpoint configured")

    monkeypatch.setattr(plugin, "load_router", unconfigured)

    with pytest.raises(RuntimeError, match="no finetuned router checkpoint"):
        plugin.run("p", "q", client=None, model="qwen3-8b")
