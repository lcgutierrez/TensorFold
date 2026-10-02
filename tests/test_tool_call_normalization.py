"""The OpenAI wire format vs the Qwen3.6 chat template.

Clients send assistant tool calls with `function.arguments` as a JSON STRING
(that is the OpenAI spec). The template does

    {%- for args_name, args_value in tool_call.arguments|items %}

which needs a mapping, so a string raises TypeError inside Jinja and the
request fails. It triggers on every agentic turn whose history contains a
prior tool call.

Values that are not mappings after a JSON read (a truncated arguments string,
a JSON array, null, a bare list) are coerced to one too -- ``{"_raw": ...}``
keeps the original text visible to the model -- because a client that replays
its history verbatim would otherwise fail every later request of the session,
not just the turn that produced the bad call.
"""
from __future__ import annotations

from tensorfold.server.messages import _normalize_tool_call_arguments


def _msg(args):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "search_files", "arguments": args}}
        ],
    }


def _args(messages):
    return messages[0]["tool_calls"][0]["function"]["arguments"]


def test_json_string_arguments_become_a_dict():
    out = _normalize_tool_call_arguments([_msg('{"query": "retries"}')])
    assert _args(out) == {"query": "retries"}


def test_already_a_dict_is_left_alone():
    msgs = [_msg({"query": "retries"})]
    assert _normalize_tool_call_arguments(msgs) is msgs


def test_unparseable_arguments_become_raw_mapping():
    """A string that is not JSON would crash the template's ``arguments|items``;
    keep the text where the model can still see it instead of failing the request."""
    out = _normalize_tool_call_arguments([_msg("not json at all")])
    assert _args(out) == {"_raw": "not json at all"}


def test_truncated_arguments_become_raw_mapping():
    # A tool call cut off mid-stream (max tokens, an interrupted stream) and replayed
    # from a client's history must not 400 every later turn of the session.
    truncated = '{"command": "curl -sL https://example.com/docs > /tmp/'
    out = _normalize_tool_call_arguments([_msg(truncated)])
    assert _args(out) == {"_raw": truncated}


def test_json_that_is_not_an_object_becomes_raw_mapping():
    # "[1,2]" parses, but the template needs a mapping, not a list.
    out = _normalize_tool_call_arguments([_msg("[1, 2]")])
    assert _args(out) == {"_raw": "[1, 2]"}


def test_null_arguments_become_empty_mapping():
    # ``arguments: null`` is defined and != '', so the template would iterate None.
    out = _normalize_tool_call_arguments([_msg(None)])
    assert _args(out) == {}


def test_non_string_non_mapping_arguments_become_raw_mapping():
    out = _normalize_tool_call_arguments([_msg(["a", "b"])])
    assert _args(out) == {"_raw": ["a", "b"]}


def test_coercion_does_not_mutate_the_caller():
    original = _msg("not json at all")
    msgs = [original]
    _normalize_tool_call_arguments(msgs)
    assert original["tool_calls"][0]["function"]["arguments"] == "not json at all"


def test_multiple_tool_calls_all_normalized():
    m = {
        "role": "assistant",
        "tool_calls": [
            {"function": {"name": "a", "arguments": '{"x": 1}'}},
            {"function": {"name": "b", "arguments": '{"y": 2}'}},
        ],
    }
    out = _normalize_tool_call_arguments([m])
    got = [c["function"]["arguments"] for c in out[0]["tool_calls"]]
    assert got == [{"x": 1}, {"y": 2}]


def test_messages_without_tool_calls_pass_through_untouched():
    msgs = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}]
    assert _normalize_tool_call_arguments(msgs) is msgs


def test_empty_and_none_are_safe():
    assert _normalize_tool_call_arguments([]) == []


def test_original_messages_are_not_mutated():
    original = _msg('{"query": "x"}')
    msgs = [original]
    _normalize_tool_call_arguments(msgs)
    assert original["tool_calls"][0]["function"]["arguments"] == '{"query": "x"}'


def test_a_real_pi_shaped_history_renders():
    """The exact shape that hung: user -> assistant tool_call -> tool -> user."""
    msgs = [
        {"role": "user", "content": "take a look at this project."},
        _msg('{"query": "retry"}'),
        {"role": "tool", "tool_call_id": "c1", "content": "src/a.py"},
        {"role": "user", "content": "and now?"},
    ]
    out = _normalize_tool_call_arguments(msgs)
    assert out[1]["tool_calls"][0]["function"]["arguments"] == {"query": "retry"}
    assert out[0] == msgs[0] and out[2] == msgs[2] and out[3] == msgs[3]
