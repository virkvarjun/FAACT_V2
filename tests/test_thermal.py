"""Tests for the thermal governor.

The governor guards a multi-hour laptop job, so the behaviour that matters is: does it
pass through when cool, block when hot, and give up rather than hang forever.
Thermal state is monkeypatched so these run anywhere, including the Linux GPU box.
"""

from __future__ import annotations

import faact.thermal as thermal
from faact.thermal import CRITICAL, FAIR, NOMINAL, SERIOUS, ThermalGovernor, safe_worker_count


def fake_states(monkeypatch, states):
    """Feed a scripted sequence of thermal readings, repeating the last one forever."""
    seq = list(states)

    def _state():
        return seq.pop(0) if len(seq) > 1 else seq[0]

    monkeypatch.setattr(thermal, "thermal_state", _state)
    return _state


def test_cool_machine_is_never_paused(monkeypatch):
    fake_states(monkeypatch, [NOMINAL])
    gov = ThermalGovernor(poll_seconds=0)
    assert gov.checkpoint() == 0.0
    assert gov.n_pauses == 0


def test_fair_is_not_paused(monkeypatch):
    """Sustained compute sits at 'fair'; pausing there would stall the job permanently."""
    fake_states(monkeypatch, [FAIR])
    gov = ThermalGovernor(poll_seconds=0)
    assert gov.checkpoint() == 0.0
    assert gov.n_pauses == 0


def test_serious_pauses_until_it_cools(monkeypatch):
    fake_states(monkeypatch, [SERIOUS, SERIOUS, SERIOUS, NOMINAL])
    gov = ThermalGovernor(poll_seconds=0)
    gov.checkpoint()
    assert gov.n_pauses == 1
    assert gov.max_state_seen == SERIOUS


def test_critical_also_pauses(monkeypatch):
    fake_states(monkeypatch, [CRITICAL, FAIR])
    gov = ThermalGovernor(poll_seconds=0)
    gov.checkpoint()
    assert gov.n_pauses == 1
    assert gov.max_state_seen == CRITICAL


def test_a_permanently_hot_machine_does_not_hang_the_job(monkeypatch):
    """Better to finish hot than to block forever — but say so."""
    fake_states(monkeypatch, [SERIOUS])
    gov = ThermalGovernor(poll_seconds=0, max_pause_seconds=0)
    gov.checkpoint()
    assert gov.n_pauses == 1


def test_unsupported_platform_degrades_to_a_noop(monkeypatch):
    monkeypatch.setattr(thermal, "thermal_state", lambda: None)
    gov = ThermalGovernor(poll_seconds=0)
    assert gov.checkpoint() == 0.0
    assert gov.report()["n_pauses"] == 0


def test_report_is_json_safe(monkeypatch):
    import json

    fake_states(monkeypatch, [SERIOUS, NOMINAL])
    gov = ThermalGovernor(poll_seconds=0)
    gov.checkpoint()
    json.dumps(gov.report())


def test_worker_count_leaves_the_machine_usable():
    """Never saturate every performance core, and never return a useless zero."""
    assert safe_worker_count(reserve=0) >= safe_worker_count() >= 1
    assert safe_worker_count(3) == 3
