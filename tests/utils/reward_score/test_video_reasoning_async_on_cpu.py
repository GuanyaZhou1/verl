import pytest

from verl.utils.reward_score.video_reasoning_async import count_turns, extract_all_segments, extract_segments, format_reward


def _make_longvt_response() -> str:
    return (
        '<think>overview</think>'
        '<tool_call>{"name":"crop_video","arguments":{"start_time":1,"end_time":2}}</tool_call>'
        '<tool_call>{"name":"crop_video","arguments":{"start_time":2.5,"end_time":4}}</tool_call>'
        '<tool_response>frames</tool_response>'
        '<think>inspect</think>'
        '<tool_call>{"name":"crop_video","arguments":{"start_time":9,"end_time":11}}</tool_call>'
        '<tool_response>more frames</tool_response>'
        '<think>final</think><answer>done</answer>'
    )


def test_longvt_format_reward_is_supported():
    response = _make_longvt_response()
    assert format_reward(response) == 1.0
    assert format_reward(response, strict_segment=True) == 1.0


def test_longvt_segment_extraction_uses_tool_calls():
    response = _make_longvt_response()
    assert extract_all_segments(response) == [[(1.0, 2.0), (2.5, 4.0)], [(9.0, 11.0)]]
    assert extract_segments(response) == [(9.0, 11.0)]


def test_longvt_turn_count_maps_tool_tags():
    counts = count_turns(_make_longvt_response())
    assert counts == {"think": 3, "segment": 3, "observation": 2, "answer": 1}


@pytest.mark.parametrize(
    "payload",
    [
        '<think>x</think><tool_call>{"name":"crop_video","arguments":{"start_time":5,"end_time":1}}</tool_call><think>y</think><answer>z</answer>',
        '<think>x</think><tool_call>{"name":"other_tool","arguments":{"start_time":1,"end_time":2}}</tool_call><think>y</think><answer>z</answer>',
    ],
)
def test_longvt_strict_format_rejects_invalid_tool_calls(payload: str):
    assert format_reward(payload, strict_segment=True) == 0.0
