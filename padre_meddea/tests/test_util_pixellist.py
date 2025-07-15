import pytest
import numpy as np

from padre_meddea.util.util import PixelList

def test_all_large():
    for i in range(4):
        px_list = PixelList.all_large(asics=[i])
        assert len(px_list) == 8
        assert i in px_list['asic']
        assert np.allclose(px_list['pixel'], np.arange(0, 8, dtype=np.uint8))

    px_list = PixelList.all_large(asics=[0, 1])
    assert len(px_list) == 8 * 2
    assert 0 in px_list['asic']
    px_list = PixelList.all_large(asics=[0, 1, 2])
    assert len(px_list) == 8 * 3
    px_list = PixelList.all_large(asics=[0, 1, 2, 3])
    assert len(px_list) == 8 * 4


def test_all_small():
    for i in range(4):
        px_list = PixelList.all_small(asics=[i])
        assert len(px_list) == 4
        assert i in px_list['asic']
        assert np.allclose(px_list['pixel'], np.arange(8, 12, dtype=np.uint8))

    asics = [[0, 1], [0, 1, 2], [0, 1, 2, 3]]
    for these_asics in asics:
        px_list = PixelList.all_small(asics=these_asics)
        assert len(px_list) == 4 * len(these_asics)
        for this_asic in these_asics:
            assert this_asic in px_list['asic']


def test_all():
    for i in range(4):
        px_list = PixelList.all(asics=[i])
        assert len(px_list) == 12
        assert i in px_list['asic']
        assert np.allclose(px_list['pixel'], np.arange(0, 12, dtype=np.uint8))

    asics = [[0, 1], [0, 1, 2], [0, 1, 2, 3]]
    for these_asics in asics:
        px_list = PixelList.all(asics=these_asics)
        assert len(px_list) == 12 * len(these_asics)
        for this_asic in these_asics:
            assert this_asic in px_list['asic']


def test_init():
    px_list = PixelList(asics=[0], pixels=[0])
    assert np.allclose(px_list['pixel'], np.array([0], dtype=np.uint8))
    assert np.allclose(px_list['asic'], np.array([0], dtype=np.uint8))

    px_list = PixelList(asics=[0, 1], pixels=[2, 3])
    assert np.allclose(px_list['pixel'], np.array([2, 3], dtype=np.uint8))
    assert np.allclose(px_list['asic'], np.array([0, 1], dtype=np.uint8))


def test_raises():
    with pytest.raises(ValueError):
        PixelList(asics=[5], pixels=[0])
        PixelList(asics=[0], pixels=[13])


def test_iterate():
    asics = [0, 1]
    pixels = [2, 3]
    px_list = PixelList(asics=[0, 1], pixels=[2, 3])
    for this_pixel in px_list:
        assert this_pixel['asic'] in asics
        assert this_pixel['pixel'] in pixels


def test_select():
    px_list = PixelList().all()
    asics = [0, 1]
    large_pixels = px_list.select_large(asics=asics)
    for this_pixel in large_pixels:
        assert this_pixel['asic'] in asics
    assert len(large_pixels) == 8 * len(asics)
    small_pixels = px_list.select_small(asics=asics)
    for this_pixel in small_pixels:
        assert this_pixel['asic'] in asics
    assert len(small_pixels) == 4 * len(asics)
    select_pixels = PixelList(asics=[0, 2], pixels=[1, 11])
    assert len(select_pixels.select_large()) == 1
    assert len(select_pixels.select_small()) == 1
    select_pixels = PixelList(asics=[1, 3], pixels=[1, 2])
    assert len(select_pixels.select_large()) == 2
    assert len(select_pixels.select_small()) == 0