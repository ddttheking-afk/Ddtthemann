import numpy as np

from teachai.games import TemplateReward
from teachai.games.matchers import similarity


def _banner(seed=0):
    """A distinctive 'victory' image: bright gold top stripe, dark below."""
    rng = np.random.default_rng(seed)
    img = np.zeros((60, 80, 3), dtype=np.uint8)
    img[:20] = [240, 200, 40]          # gold banner
    img[20:] = [20, 20, 30]            # dark background
    img = np.clip(img + rng.normal(0, 5, img.shape), 0, 255).astype(np.uint8)
    return img


def _gameplay(seed=1):
    """An unrelated busy scene."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)


def test_similarity_high_for_same_scene():
    a, b = _banner(0), _banner(99)
    assert similarity(a, b) > 0.9


def test_similarity_low_for_different_scene():
    assert similarity(_banner(0), _gameplay()) < 0.5


def test_template_reward_detects_and_rewards():
    reward = TemplateReward(_banner(0), threshold=0.7, reward=1.0)
    assert reward.detected(_banner(42)) is True
    assert reward(_banner(42)) == 1.0
    assert reward.detected(_gameplay()) is False
    assert reward(_gameplay()) == 0.0


def test_template_reward_region_crop():
    # Put the banner in the top-left corner of a larger frame.
    frame = _gameplay(3).copy()
    frame[:20, :30] = _banner(0)[:20, :30]
    reward = TemplateReward(_banner(0)[:20, :30], region=(0, 0, 30, 20), threshold=0.7)
    assert reward.detected(frame) is True
