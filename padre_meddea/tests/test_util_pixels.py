import pytest
import numpy as np

from padre_meddea.util.pixels import *


def test_all_large():
    for i in range(4):
        px_list = PixelList.all_large(asics=[i])
        assert len(px_list) == 8
        assert i in px_list["asic"]
        assert np.allclose(px_list["pixel"], np.arange(0, 8, dtype=np.uint8))

    px_list = PixelList.all_large(asics=[0, 1])
    assert len(px_list) == 8 * 2
    assert 0 in px_list["asic"]
    px_list = PixelList.all_large(asics=[0, 1, 2])
    assert len(px_list) == 8 * 3
    px_list = PixelList.all_large(asics=[0, 1, 2, 3])
    assert len(px_list) == 8 * 4


def test_all_small():
    for i in range(4):
        px_list = PixelList.all_small(asics=[i])
        assert len(px_list) == 4
        assert i in px_list["asic"]
        assert np.allclose(px_list["pixel"], np.arange(8, 12, dtype=np.uint8))

    asics = [[0, 1], [0, 1, 2], [0, 1, 2, 3]]
    for these_asics in asics:
        px_list = PixelList.all_small(asics=these_asics)
        assert len(px_list) == 4 * len(these_asics)
        for this_asic in these_asics:
            assert this_asic in px_list["asic"]


def test_all():
    for i in range(4):
        px_list = PixelList.all(asics=[i])
        assert len(px_list) == 12
        assert i in px_list["asic"]
        assert np.allclose(px_list["pixel"], np.arange(0, 12, dtype=np.uint8))

    asics = [[0, 1], [0, 1, 2], [0, 1, 2, 3]]
    for these_asics in asics:
        px_list = PixelList.all(asics=these_asics)
        assert len(px_list) == 12 * len(these_asics)
        for this_asic in these_asics:
            assert this_asic in px_list["asic"]


def test_init():
    px_list = PixelList(asics=[0], pixels=[0])
    assert np.allclose(px_list["pixel"], np.array([0], dtype=np.uint8))
    assert np.allclose(px_list["asic"], np.array([0], dtype=np.uint8))

    px_list = PixelList(asics=[0, 1], pixels=[2, 3])
    assert np.allclose(px_list["pixel"], np.array([2, 3], dtype=np.uint8))
    assert np.allclose(px_list["asic"], np.array([0, 1], dtype=np.uint8))

    # test with lab asic number 7
    px_list = PixelList(asics=[7, 7], pixels=[2, 3])
    assert np.allclose(px_list["pixel"], np.array([2, 3], dtype=np.uint8))
    assert np.allclose(px_list["asic"], np.array([7, 7], dtype=np.uint8))


def test_raises():
    with pytest.raises(ValueError):
        PixelList(asics=[5], pixels=[0])
        PixelList(asics=[0], pixels=[13])


def test_iterate():
    asics = [0, 1]
    pixels = [2, 3]
    px_list = PixelList(asics=[0, 1], pixels=[2, 3])
    for this_pixel in px_list:
        assert this_pixel["asic"] in asics
        assert this_pixel["pixel"] in pixels


def test_select():
    px_list = PixelList().all()
    asics = [0, 1]
    large_pixels = px_list.select_large(asics=asics)
    for this_pixel in large_pixels:
        assert this_pixel["asic"] in asics
    assert len(large_pixels) == 8 * len(asics)
    small_pixels = px_list.select_small(asics=asics)
    for this_pixel in small_pixels:
        assert this_pixel["asic"] in asics
    assert len(small_pixels) == 4 * len(asics)
    select_pixels = PixelList(asics=[0, 2], pixels=[1, 11])
    assert len(select_pixels.select_large()) == 1
    assert len(select_pixels.select_small()) == 1
    select_pixels = PixelList(asics=[1, 3], pixels=[1, 2])
    assert len(select_pixels.select_large()) == 2
    assert len(select_pixels.select_small()) == 0


def test_duplicates():
    with pytest.raises(ValueError):
        PixelList(asics=[0, 0, 1], pixels=[0, 0, 1])
        PixelList(asics=[0, 0, 7, 1, 1], pixels=[11, 8, 0, 2, 2])


@pytest.mark.parametrize(
    "channel,pixel_num",
    [
        (26, 0),
        (15, 1),
        (8, 2),
        (1, 3),
        (29, 4),
        (18, 5),
        (5, 6),
        (0, 7),
        (30, 8),
        (21, 9),
        (11, 10),
        (3, 11),
        (31, 12),
        (14, 14 + 12),  # unconnected channel gets 12 added to it
    ],
)
def test_channel_to_pix(channel, pixel_num):
    assert channel_to_pixel(channel) == pixel_num
    if pixel_num < 12:
        assert pixel_to_channel(pixel_num) == channel


@pytest.mark.parametrize(
    "pixel_id,asic_num,pixel_num",
    [
        (51738, 0, 0),
        (51720, 0, 2),
        (51730, 0, 5),
        (51712, 0, 7),
        (51733, 0, 9),
        (51715, 0, 11),
        (51770, 1, 0),
        (51752, 1, 2),
        (51762, 1, 5),
        (51744, 1, 7),
        (51765, 1, 9),
        (51747, 1, 11),
        (51802, 2, 0),
        (51784, 2, 2),
        (51794, 2, 5),
        (51776, 2, 7),
        (51797, 2, 9),
        (51779, 2, 11),
        (51834, 3, 0),
        (51816, 3, 2),
        (51826, 3, 5),
        (51808, 3, 7),
        (51829, 3, 9),
        (51811, 3, 11),
    ],
)
def test_get_pixelid(pixel_id, asic_num, pixel_num):
    assert get_pixelid(asic_num, pixel_num) == pixel_id
    assert parse_pixelids(pixel_id) == (asic_num, pixel_to_channel(pixel_num))


def test_pixel_to_string():
    assert pixel_to_str(0) == "Pixel0L"
    assert pixel_to_str(8) == "Pixel8S"
    assert pixel_to_str(11) == "Pixel11S"


def test_pixel_to_string_error():
    with pytest.raises(ValueError):
        pixel_to_str(13)
    with pytest.raises(ValueError):
        pixel_to_str(16)
