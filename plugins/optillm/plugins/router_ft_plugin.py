"""optillm plugin: route a query with the router this repo finetuned.

Registered under its own slug, so the finetuned router is swept exactly like any
other approach - `--approaches none bon router router_ft` puts it head to head
with the stock router over identical queries, in one run, and every existing
leaderboard, cost multiplier and per-query drill-down works on it with no new
code.

**The upstream checkout is not modified.** optillm discovers plugins in
`<--plugins-dir>/optillm/plugins`, which is why this file sits under a directory
whose name it does not choose. See the repository README for how the server is
started.

What this file owns is optillm's calling convention and nothing else:

- Prediction is `router_lab.finetuned_router`, where it can be tested without a
  server.
- Executing the chosen approach is optillm's own `execute_single_approach`, so
  all thirteen approaches are reachable without restating the stock router's
  dispatch chain - and without this plugin being able to drift from it.
- The prediction is reported as `optillm_router_approach`, the field the harness
  reads, by returning a full response dict - the convention optillm's own proxy
  plugin uses, and which the server hands straight back to the client. That is
  how a plugin adds a field to a response without the server needing to know the
  plugin exists.

Failures are *not* swallowed into a fallback the way the stock router's are. A
missing or unreadable checkpoint would otherwise produce rows that look like
routing decisions and are not, and the harness already records a failed request
as data.
"""

import sys
from pathlib import Path

# optillm loads this file by path, not as part of a package, so the repository is
# not necessarily on the import path - it depends on the server's working
# directory. Anchor it on this file instead: plugins/optillm/plugins/<this>.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from router_lab.finetuned_router import APPROACH, load_router  # noqa: E402

SLUG = APPROACH
"""`router_ft` - distinct from optillm's own `router`, so both can be swept."""

ROUTER_APPROACH_FIELD = "optillm_router_approach"
"""What the harness reads to learn which approach a router picked."""


CONTROL_PREFIX = "optillm_"
"""Request keys that are optillm's own controls rather than API parameters."""


def forwardable(request_config):
    """The caller's request parameters, without optillm's own control keys.

    optillm's `none` path forwards this dict to the OpenAI client as keyword
    arguments, and the dict still contains whatever routed the request here -
    `optillm_approach` at least - which the client rejects outright. So a
    prediction of `none` would fail every time, and only that prediction.

    Stripping the `optillm_` keys rather than passing nothing keeps the caller's
    temperature and max_tokens, which the stock router drops on this path by
    rebuilding the request from scratch.
    """
    return {
        key: value
        for key, value in (request_config or {}).items()
        if not key.startswith(CONTROL_PREFIX)
    }


def dispatch(approach, system_prompt, initial_query, client, model, request_config):
    """Run the chosen approach through optillm's own executor.

    Imported here rather than at module scope: this file is loaded by the server
    while it is starting up, and the import is only needed once a request
    arrives. It is also the seam the plugin's tests replace, which is what lets
    them run the plugin without optillm's whole approach stack.
    """
    from optillm.server import execute_single_approach

    return execute_single_approach(
        approach, system_prompt, initial_query, client, model, request_config
    )


def as_response(response, tokens, predicted, *, model):
    """Shape a plugin result into a response dict carrying the prediction.

    Three shapes arrive here, all of them optillm's: a full response dict (the
    `none` passthrough hands back vLLM's own, usage included), a string, or a
    list of strings from an approach that produced several completions. The
    prediction is added to whichever it is, and the token count is only invented
    when there was no usage block to keep.

    `choices` and `usage` are both always present, because that pair is what the
    server checks before returning a plugin's dict verbatim; a response missing
    either would be re-assembled by the server and lose the prediction.
    """
    if isinstance(response, dict) and "choices" in response:
        return {**response, ROUTER_APPROACH_FIELD: predicted}

    contents = response if isinstance(response, list) else [response]
    return {
        "model": model,
        "choices": [
            {
                "index": index,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
            for index, content in enumerate(contents)
        ],
        "usage": {"completion_tokens": tokens},
        ROUTER_APPROACH_FIELD: predicted,
    }


def run(
    system_prompt, initial_query, client, model, request_config: dict = None, **kwargs
):
    """Predict an approach for this query, run it, and report what was picked.

    Returns `(response, tokens)` like every optillm approach; the response is a
    full dict rather than a string, which is what carries the prediction.
    """
    router = load_router()
    predicted = router.predict(system_prompt, initial_query)
    print(f"Finetuned router predicted approach: {predicted}")

    response, tokens = dispatch(
        predicted,
        system_prompt,
        initial_query,
        client,
        model,
        forwardable(request_config),
    )
    return as_response(response, tokens, predicted, model=model), tokens
