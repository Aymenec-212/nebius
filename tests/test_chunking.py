import pytest

from app.vad import Span, fixed_windows, make_chunks

MAX, GAP, PAD = 20.0, 0.5, 0.2


def _durations(spans):
    return [s.end - s.start for s in spans]


def test_empty_in_empty_out():
    assert make_chunks([], 100.0, MAX, GAP, PAD) == []


def test_close_spans_merge():
    spans = [Span(1.0, 3.0), Span(3.3, 5.0), Span(5.4, 6.0)]
    out = make_chunks(spans, 10.0, MAX, GAP, 0.0)
    assert out == [Span(1.0, 6.0)]


def test_distant_spans_stay_apart():
    spans = [Span(1.0, 3.0), Span(4.0, 5.0)]
    out = make_chunks(spans, 10.0, MAX, GAP, 0.0)
    assert out == [Span(1.0, 3.0), Span(4.0, 5.0)]


def test_merge_never_exceeds_max_chunk():
    spans = [Span(0.0, 12.0), Span(12.2, 24.0), Span(24.1, 30.0)]
    out = make_chunks(spans, 40.0, MAX, GAP, 0.0)
    assert all(d <= MAX for d in _durations(out))
    # first two cannot merge (24 s), second and third can (17.8 s)
    assert out == [Span(0.0, 12.0), Span(12.2, 30.0)]


def test_long_span_is_split_into_equal_parts():
    out = make_chunks([Span(0.0, 50.0)], 60.0, MAX, GAP, 0.0)
    assert len(out) == 3
    assert all(d <= MAX for d in _durations(out))
    assert all(abs(d - 50.0 / 3) < 1e-6 for d in _durations(out))
    assert out[0].start == 0.0 and out[-1].end == 50.0
    for a, b in zip(out, out[1:]):
        assert abs(a.end - b.start) < 1e-9


def test_padding_is_applied_and_clamped():
    out = make_chunks([Span(0.1, 2.0), Span(8.0, 9.95)], 10.0, MAX, GAP, PAD)
    assert out[0] == Span(0.0, 2.2)
    assert out[1] == Span(7.8, 10.0)


def test_result_sorted_by_start_even_for_unsorted_input():
    out = make_chunks([Span(5.0, 6.0), Span(1.0, 2.0)], 10.0, MAX, GAP, 0.0)
    assert [s.start for s in out] == sorted(s.start for s in out)


def test_padded_chunk_never_longer_than_max_plus_two_pads():
    spans = [Span(i * 7.0, i * 7.0 + 6.8) for i in range(20)]
    out = make_chunks(spans, 200.0, MAX, GAP, PAD)
    assert all(d <= MAX + 2 * PAD + 1e-9 for d in _durations(out))


def test_fixed_windows_cover_whole_duration():
    out = fixed_windows(45.0, MAX)
    assert out == [Span(0.0, 20.0), Span(20.0, 40.0), Span(40.0, 45.0)]
    assert out[0].start == 0.0 and out[-1].end == 45.0
    for a, b in zip(out, out[1:]):
        assert a.end == b.start


def test_fixed_windows_exact_multiple_and_empty():
    assert fixed_windows(40.0, MAX) == [Span(0.0, 20.0), Span(20.0, 40.0)]
    assert fixed_windows(0.0, MAX) == []


def test_invalid_max_chunk():
    with pytest.raises(ValueError):
        make_chunks([Span(0, 1)], 1.0, 0.0, GAP, PAD)
