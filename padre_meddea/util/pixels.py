"""Provides tools to deal with pixels"""

import warnings

import numpy as np
from astropy.table import Table

__all__ = [
    "PixelList",
    "channel_to_pixel",
    "pixel_to_channel",
    "parse_pixelids",
    "get_pixelid",
    "get_pixel_str",
    "pixelid_to_str",
    "pixel_to_str",
]


class PixelList(Table):
    """A list of pixels

    Parameters
    ----------
    asics : list
        The asic numbers
    pixels : list
        The pixel numbers
    pixel_ids : list
        The pixel ids

    Raises
    ------
    ValueError

    Examples
    --------
    >>> from padre_meddea.util import PixelList
    >>> px_list = PixelList(asics=[0, 0], pixels=[0, 1])  # two pixels both from asic 0
    >>> px_list = PixelList(asics=[0, 1], pixels=[8, 9])  # two small pixels, one from asic 0 and another from asic 1
    >>> px_list = PixelList(pixelids=[51738, 51720])  # using pixel ids
    >>> large_pixs = PixelList().all_large()  # every large pixel
    >>> small_pixs = PixelList().all_small()  # every large pixel
    >>> large_pixs_det1 = PixelList().all_large(asics=[1])  # all large pixels from asic 1
    """

    def __init__(self, *args, **kwargs):
        if ("asics" in kwargs) and ("pixels" in kwargs):
            asics = kwargs.pop("asics")
            pixels = kwargs.pop("pixels")
            super().__init__(*args, **kwargs)
            self["asic"] = np.array(asics, dtype=np.uint8)
            self["pixel"] = np.array(pixels, dtype=np.uint8)
        elif "pixelids" in kwargs:
            pixelids = kwargs.pop("pixelids")
            asics, channels = parse_pixelids(pixelids)
            pixels = channel_to_pixel(channels)
            super().__init__(*args, **kwargs)
            self["asic"] = np.array(asics, dtype=np.uint8)
            self["pixel"] = np.array(pixels, dtype=np.uint8)
        else:
            super().__init__(*args, **kwargs)
        if len(self) > 0:
            self._add_helper_columns()
            self._verify()

    @classmethod
    def all_large(cls, asics: list = [0, 1, 2, 3]):
        asic_nums = np.repeat(np.array(asics, dtype=np.uint8), 8)
        pixel_nums = np.resize(np.arange(8, dtype=np.uint8), 8 * len(asics))
        out = Table()
        out["asic"] = asic_nums
        out["pixel"] = pixel_nums
        return cls(out)

    @classmethod
    def all_small(cls, asics: list = [0, 1, 2, 3]):
        asic_nums = np.repeat(np.array(asics, dtype=np.uint8), 4)
        pixel_nums = np.resize(np.arange(8, 12, dtype=np.uint8), 4 * len(asics))
        out = Table()
        out["asic"] = asic_nums
        out["pixel"] = pixel_nums
        return cls(out)

    @classmethod
    def all(cls, asics: list = [0, 1, 2, 3]):
        asic_nums = np.repeat(np.array(asics, dtype=np.uint8), 12)
        pixel_nums = np.resize(np.arange(0, 12, dtype=np.uint8), 12 * len(asics))
        out = Table()
        out["asic"] = asic_nums
        out["pixel"] = pixel_nums
        return cls(out)

    def select_large(self, asics: list = [0, 1, 2, 3]):
        """Return only large pixels from an existing pixel list"""
        good_pixels = np.arange(0, 8)
        asic_list = []
        pixel_list = []
        for this_asic in asics:
            this_pixel_list = self[self["asic"] == this_asic]
            for this_pixel in this_pixel_list:
                if this_pixel["pixel"] in good_pixels:
                    asic_list.append(this_pixel["asic"])
                    pixel_list.append(this_pixel["pixel"])
        return PixelList(asics=asic_list, pixels=pixel_list)

    def select_small(self, asics: list = [0, 1, 2, 3]):
        """Return only small pixels from an existing pixel list"""
        good_pixels = np.arange(8, 12)
        asic_list = []
        pixel_list = []
        for this_asic in asics:
            this_pixel_list = self[self["asic"] == this_asic]
            for this_pixel in this_pixel_list:
                if this_pixel["pixel"] in good_pixels:
                    asic_list.append(this_pixel["asic"])
                    pixel_list.append(this_pixel["pixel"])
        return PixelList(asics=asic_list, pixels=pixel_list)

    def _verify(self):
        """Verify consistency of the data."""
        if "pixel" in self.columns and len(self) > 0 and np.any(self["pixel"] > 12):
            raise ValueError(
                f"Found a pixel number that is too large, {self['pixel'].max()}"
            )
        if "asic" in self.columns and len(self) > 0:
            good_asic_inds = (self["asic"] < 4) | (self["asic"] == 7)
            bad_asic_inds = ~good_asic_inds
            if np.any(bad_asic_inds):
                raise ValueError(
                    f"Found an unexpected asic number(s), {np.array(self['asic'][bad_asic_inds])}"
                )
        if "id" in self.columns and len(self) > 0:
            if len(np.unique(self["id"])) < len(self["id"]):
                raise ValueError("Found duplicate pixels.")

    def _add_helper_columns(self):
        """Add additional helper columns"""
        self["channel"] = np.array(
            [pixel_to_channel(this_pixel) for this_pixel in self["pixel"]],
            dtype=np.uint8,
        )
        self["id"] = np.array(
            [
                get_pixelid(this_asic, this_pixel)
                for this_asic, this_pixel in zip(self["asic"], self["pixel"])
            ],
            dtype=np.uint16,
        )
        self["label"] = [
            get_pixel_str(this_asic, this_pixel)
            for this_asic, this_pixel in zip(self["asic"], self["pixel"])
        ]


def _channel_to_pixel(channel: int) -> int:
    """
    Given a channel pixel number, return the pixel number.
    """
    CHANNEL_TO_PIX = {
        26: 0,
        15: 1,
        8: 2,
        1: 3,
        29: 4,
        18: 5,
        5: 6,
        0: 7,
        30: 8,
        21: 9,
        11: 10,
        3: 11,
        31: 12,
    }

    if channel in CHANNEL_TO_PIX.keys():
        return CHANNEL_TO_PIX[channel]
    else:
        warnings.warn(
            f"Found unconnected channel, {channel}. Returning channel + 12 ={channel + 12}."
        )
        return channel + 12


channel_to_pixel = np.vectorize(_channel_to_pixel)


def _pixel_to_channel(pixel_num: int) -> int:
    """
    Given a pixel number, return the channel number.
    """
    PIXEL_TO_CHANNEL = {
        0: 26,
        1: 15,
        2: 8,
        3: 1,
        4: 29,
        5: 18,
        6: 5,
        7: 0,
        8: 30,
        9: 21,
        10: 11,
        11: 3,
        12: 31,  # not a pixel, guard ring
    }

    if pixel_num in PIXEL_TO_CHANNEL.keys():
        return PIXEL_TO_CHANNEL[pixel_num]
    else:
        raise ValueError(f"Pixel number, {pixel_num}, not found.")


pixel_to_channel = np.vectorize(_pixel_to_channel)


def parse_pixelids(ids):
    """
    Given pixel id infomration, return the asic numbers and channel numbers
    """
    asic_nums = (np.array(ids) & 0b11100000) >> 5
    channel_nums = np.array(ids) & 0b00011111
    return asic_nums, channel_nums


def get_pixelid(asic_num: int, pixel_num: int) -> int:
    """Given an asic number and a pixel number return the pixelid"""
    return (asic_num << 5) + pixel_to_channel(pixel_num) + 0xCA00


def get_pixel_str(asic_num: int, pixel_num: int):
    return f"Det{str(asic_num)}{pixel_to_str(pixel_num)}"


def pixelid_to_str(ids):
    """
    Given unparsed pixel ids, return strings for each
    """
    asic_nums, channel_nums = parse_pixelids(ids)
    pixel_nums = [channel_to_pixel(this_chan) for this_chan in channel_nums]
    result = [
        get_pixel_str(this_asic, this_pixel)
        for this_asic, this_pixel in zip(asic_nums, pixel_nums)
    ]
    return result


def pixel_to_str(pixel_num: int) -> str:
    """
    Given a pixel number, return a standardized string.
    """
    if not (0 <= pixel_num <= 11):
        raise ValueError("Pixel integer number must be 0 to 11.")
    if 0 <= pixel_num <= 7:
        pixel_size = "L"
    elif 8 <= pixel_num <= 12:
        pixel_size = "S"
    return f"Pixel{pixel_num}{pixel_size}"
