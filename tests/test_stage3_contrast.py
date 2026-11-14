import pytest

from stage3_contrast import first_and_reversal, visibility


def test_midpoint_applies_to_each_frame_at_each_contrast():
    assert visibility(.65, .6499, 1., .3) == (.65, True, False)
    assert visibility(.4749, .475, .65, .3) == (.475, False, True)


def test_first_visible_sigma_keeps_nonmonotonic_losses_of_visibility():
    assert first_and_reversal((False, False, True, False, True)) == (.1, True)
    assert first_and_reversal((False, True, True, True, True)) == (.05, False)
    assert first_and_reversal((False,)*5) == (None, False)
    with pytest.raises(ValueError):
        first_and_reversal((True,))
