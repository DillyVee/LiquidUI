"""GUI wiring tests for the certified-strategy preset.

Instantiates the real MainWindow offscreen and verifies that the
one-click preset and its two checkboxes actually change what the
optimizer would be constructed with - the settings must flow, not just
render.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtWidgets import QApplication
except ImportError as exc:  # missing system GL libs on bare containers
    pytest.skip(f"PyQt6 unavailable: {exc}", allow_module_level=True)

from gui.main_window import MainWindow  # noqa: E402

NO_CYCLE = ((250, 250), (0, 0), (0, 0))


@pytest.fixture(scope="module")
def window():
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    yield w
    w.close()
    app.processEvents()


def test_defaults_unchanged_by_new_controls(window):
    """Existing users see the exact pre-existing behavior by default"""
    assert not window.no_cycle_cb.isChecked()
    assert not window.vol_target_cb.isChecked()
    kwargs = window._signal_search_kwargs()
    assert kwargs["vol_targeting"] is False
    params = window._get_param_ranges()
    ranges = window._effective_cycle_ranges(params)
    assert ranges == (
        params["on"], params["off"], (0, params["on"][1] + params["off"][1])
    )


def test_certified_preset_applies_full_configuration(window):
    window.apply_certified_preset()

    assert window.no_cycle_cb.isChecked()
    assert window.vol_target_cb.isChecked()
    assert not window.cycle_only_cb.isChecked()
    assert window.combine_ind_cb.isChecked()
    assert all(cb.isChecked() for cb in window.indicator_checkboxes.values())
    assert window._selected_objective() == "sharpe"

    # The settings must reach the optimizer construction paths
    kwargs = window._signal_search_kwargs()
    assert kwargs["vol_targeting"] is True
    assert kwargs["cycle_only"] is False
    ranges = window._effective_cycle_ranges(window._get_param_ranges())
    assert ranges == NO_CYCLE

    # Cycle range spinboxes are visibly inert while the cycle is disabled
    assert not window.on_min.isEnabled()
    assert not window.off_max.isEnabled()


def test_cycle_only_and_no_cycle_are_mutually_exclusive(window):
    window.no_cycle_cb.setChecked(True)
    window.cycle_only_cb.setChecked(True)
    assert not window.no_cycle_cb.isChecked(), "cycle-only must clear no-cycle"

    window.no_cycle_cb.setChecked(True)
    assert not window.cycle_only_cb.isChecked(), "no-cycle must clear cycle-only"

    # Unchecking restores the normal search space and the spinboxes
    window.no_cycle_cb.setChecked(False)
    params = window._get_param_ranges()
    assert window._effective_cycle_ranges(params) != NO_CYCLE
    assert window.on_min.isEnabled()
