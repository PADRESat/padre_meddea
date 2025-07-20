.. _pixels:

****************
Selecting Pixels
****************

The MeDDEA instrument is comprised of 4 individuals detectors each with 12 pixels for a total of 48 pixels.
Each detector has 8 large pixels and 4 small pixels.
It is frequently the case that it is necessary to consider sets of pixels.
For example, all large pixels on one detector or all small pixels across all detectors.
To faciliate this task, the `~padre_meddea.util.pixels.PixelList` is provided.

Each pixel is identified by its asic number, 0 to 3, and pixel number, 0 to 11.

.. code-block:: python

   >>> from padre_meddea.util import PixelList

We can create a `~padre_meddea.util.pixels.PixelList` in a variety of ways. First by just manually specifying each asic and pixel number pair:

.. code-block:: python

   >>> pixel_list = PixelList(asics=[0, 1], pixels=[0, 1])
   >>> pixel_list
   <PixelList length=2>
    asic pixel channel   id      label
   uint8 uint8  uint8  uint16    str11
   ----- ----- ------- ------ -----------
       0     0      26  51738 Det0Pixel0L
       1     1      15  51759 Det1Pixel1L

The channel column provides an internal specifier for how the detector identifies a pixel.
The id column provides a unique identifier for each pixel and the label provides a human-readable string.

Note that this object is a subclass of a `~astropy.table.Table` which means you can, for example, easily iterate over each element:

.. code-block:: python

    >>> asics = [this_pixel['asic'] for this_pixel in pixel_list]
    >>> asics
    [np.uint8(0), np.uint8(1)]
    >>> pixels = pixel_list['pixel']
    >>> pixels
    <Column name='pixel' dtype='uint8' length=2>
    0
    1

This also works for just one pixel:

.. code-block:: python

   >>> pixel_list = PixelList(asics=[3], pixels=[11])
   >>> pixel_list
   <PixelList length=1>
    asic pixel channel   id      label
   uint8 uint8  uint8  uint16    str12
    ----- ----- ------- ------ ------------
        3    11       3  51811 Det3Pixel11S

A few class methods make it easier to generate lists of pixels. For example, to get a list of all pixels:

.. code-block:: python

   >>> pixel_list = PixelList().all()
   >>> pixel_list
   <PixelList length=48>
    asic pixel channel   id      label
    uint8 uint8  uint8  uint16    str12
    ----- ----- ------- ------ ------------
        0     0      26  51738  Det0Pixel0L
        0     1      15  51727  Det0Pixel1L
        0     2       8  51720  Det0Pixel2L
        0     3       1  51713  Det0Pixel3L
        0     4      29  51741  Det0Pixel4L
      ...   ...     ...    ...          ...
        3     7       0  51808  Det3Pixel7L
        3     8      30  51838  Det3Pixel8S
        3     9      21  51829  Det3Pixel9S
        3    10      11  51819 Det3Pixel10S
        3    11       3  51811 Det3Pixel11S

You can also get a list of all large pixels or all small pixels:

.. code-block:: python

    >>> all_large = PixelList().all_large()
    >>> print(len(all_large))
    32
    >>> all_small = PixelList().all_small()
    >>> print(len(all_small))
    16

It is also possible to get all large pixels or all small pixels from a set of asics:

.. code-block:: python

    >>> all_large_det0 = PixelList().all_large(asics=[0])
    >>> print(len(all_large_det0))
    8
    >>> all_small_det1 = PixelList().all_small(asics=[1])
    >>> all_small_det1
    <PixelList length=4>
     asic pixel channel   id      label
    uint8 uint8  uint8  uint16    str12
    ----- ----- ------- ------ ------------
        1     8      30  51774  Det1Pixel8S
        1     9      21  51765  Det1Pixel9S
        1    10      11  51755 Det1Pixel10S
        1    11       3  51747 Det1Pixel11S

Finally, if given an existing pixel list, it is possible to select out a subset of pixels.
This is most useful to select a subset of the pixels that are monitored through the summary spectrum data.

.. code-block:: python

    >>> from padre_meddea.spectrum.spectrum import DEFAULT_SPEC_PIXEL_LIST
    >>> DEFAULT_SPEC_PIXEL_LIST.select_small()
    <PixelList length=8>
    asic pixel channel   id      label
    uint8 uint8  uint8  uint16    str12
    ----- ----- ------- ------ ------------
        0     9      21  51733  Det0Pixel9S
        0    11       3  51715 Det0Pixel11S
        1     9      21  51765  Det1Pixel9S
        1    11       3  51747 Det1Pixel11S
        2     9      21  51797  Det2Pixel9S
        2    11       3  51779 Det2Pixel11S
        3     9      21  51829  Det3Pixel9S
        3    11       3  51811 Det3Pixel11S

Both the `~padre_meddea.spectrum.spectrum.PhotonList` and `~padre_meddea.spectrum.spectrum.SpectrumList` provide pixel_list attributes which give a list of all pixels in the data set.
