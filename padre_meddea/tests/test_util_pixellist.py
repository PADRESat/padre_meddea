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


def test_init():
    px_list = PixelList(asics=[0], pixels=[0])
    assert np.allclose(px_list['pixel'], np.array([0], dtype=np.uint8))
    assert np.allclose(px_list['asic'], np.array([0], dtype=np.uint8))

    px_list = PixelList(asics=[0, 1], pixels=[2, 3])
    assert np.allclose(px_list['pixel'], np.array([2, 3], dtype=np.uint8))
    assert np.allclose(px_list['asic'], np.array([0, 1], dtype=np.uint8))