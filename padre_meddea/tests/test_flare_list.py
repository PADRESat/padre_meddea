import urllib.error

import numpy as np
import pytest
from astropy.time import Time
from astropy.timeseries import TimeSeries
import astropy.units as u

from padre_meddea.io import flare_list


@pytest.fixture
def flare_csv(tmp_path):
    path = tmp_path / "flares.csv"

    def write(flux_header=",GOES Flux", first=",5.90E-05", second=","):
        path.write_text(
            "Date,Start Time (UTC),End Time (UTC),GOES Class,Quality Rating,"
            "Grafana Dashboard (Internal Link),Public,Link to L0 Data,Note"
            f"{flux_header}\n"
            f"2025-11-28,21:52:32,22:41:11,M5.9,1,internal,public,http://a/1.fits,note{first}\n"
            f"2025-11-28,22:41:11,00:11:32,M1.6,0,,,No link,{second}\n"
            f"#N/A,#N/A,#N/A,,,,,,,#VALUE!\n"
        )
        return path

    return write


def test_read_flare_list_parses_goes_flux(flare_csv):
    result = flare_list.read_flare_list(flare_csv())
    assert len(result) == 2
    assert result["goes_flux"].dtype.kind == "f"
    assert result["goes_flux"][0] == pytest.approx(5.90e-5)
    assert np.ma.is_masked(result["goes_flux"][1])
    assert result["goes_class"].tolist() == ["M5.9", "M1.6"]
    assert result["end_time"][1] == Time("2025-11-29T00:11:32", scale="utc")
    assert result["l0_urls"].tolist() == [["http://a/1.fits"], []]


def test_read_flare_list_supports_legacy_files_without_flux(flare_csv):
    result = flare_list.read_flare_list(flare_csv("", "", ""))
    assert len(result) == 2
    assert "goes_flux" not in result.colnames


@pytest.mark.parametrize("value", [",not-a-number", ""])
def test_read_flare_list_rejects_invalid_flux(flare_csv, value):
    with pytest.raises(ValueError, match="GOES Flux"):
        flare_list.read_flare_list(flare_csv(first=value))


def test_find_flare_photon_files(monkeypatch):
    """Files overlapping the flare are selected."""
    times = ["2025-11-28T20:00:00", "2025-11-28T21:00:00", "2025-11-28T22:00:00"]
    times += ["2025-11-28T23:30:00"]
    rows = [
        {"Time": Time(t), "Version": "v1.0.0", "url": np.str_(f"file_{i}")}
        for i, t in enumerate(times)
    ]

    class FakeResult(dict):
        pass

    monkeypatch.setattr(
        "sunpy.net.Fido.search", lambda *args, **kwargs: FakeResult(padre=rows)
    )

    flares = TimeSeries(time=Time(["2025-11-28T21:30:00"]))
    flares["end_time"] = Time(["2025-11-28T22:45:00"])
    listed = np.empty(1, dtype=object)
    listed[0] = ["file_1", "file_2"]
    flares["l0_urls"] = listed
    result = flare_list.find_flare_photon_files(flares)
    assert isinstance(result, TimeSeries)
    assert result["found"][0] == "file_1|file_2"
    assert result["matches_l0"].tolist() == [True]
    assert result["num_found"].tolist() == [2]
    assert "found" not in flares.colnames


def test_find_flare_photon_files_one_hour_overlap_and_boundaries(monkeypatch):
    start = Time("2025-11-29T00:30:00")
    end = Time("2025-11-29T01:00:00")
    rows = [
        {"Time": Time(t), "Version": version, "url": url}
        for t, version, url in [
            ("2025-11-28T22:00:00", "v1.0.0", "too_early"),
            ("2025-11-28T23:30:00", "v1.0.0", "ends_at_start"),
            ("2025-11-28T23:30:01", "v1.0.0", "overlap_old"),
            ("2025-11-28T23:30:01", "v1.0.1", "overlap_new"),
            ("2025-11-28T23:45:00", "v1.0.0", "overlap_second"),
            ("2025-11-29T00:30:00", "v1.0.0", "at_start"),
            ("2025-11-29T00:59:59", "v1.0.0", "before_end"),
            ("2025-11-29T01:00:00", "v1.0.0", "at_end"),
        ]
    ]
    queries = []

    def search(query, *args):
        queries.append(query)
        return {"padre": rows}

    monkeypatch.setattr("sunpy.net.Fido.search", search)
    flares = TimeSeries(time=Time([start]))
    flares["end_time"] = Time([end])
    listed = np.empty(1, dtype=object)
    listed[0] = ["overlap_new", "overlap_second", "at_start", "before_end"]
    flares["l0_urls"] = listed

    result = flare_list.find_flare_photon_files(flares)
    assert queries[0].start == start - 1 * u.hour
    assert queries[0].end == end
    assert result["found"][0] == "overlap_new|overlap_second|at_start|before_end"
    assert result["num_found"].tolist() == [4]
    assert result["matches_l0"].tolist() == [True]


def test_find_flare_photon_files_no_overlapping_files(monkeypatch):
    rows = [
        {"Time": Time("2025-11-28T20:00:00"), "Version": "v1.0.0", "url": "old"},
        {"Time": Time("2025-11-28T23:00:00"), "Version": "v1.0.0", "url": "later"},
    ]
    monkeypatch.setattr("sunpy.net.Fido.search", lambda *args, **kwargs: {"padre": rows})
    flares = TimeSeries(time=Time(["2025-11-28T21:30:00"]))
    flares["end_time"] = Time(["2025-11-28T22:45:00"])
    listed = np.empty(1, dtype=object)
    listed[0] = ["listed_but_not_found"]
    flares["l0_urls"] = listed

    result = flare_list.find_flare_photon_files(flares)
    assert result["found"][0] == ""
    assert result["num_found"].tolist() == [0]
    assert result["matches_l0"].tolist() == [False]


def test_find_flare_photon_files_compares_file_names(monkeypatch):
    rows = [
        {
            "Time": Time("2025-11-28T21:00:00"),
            "Version": "v1.0.0",
            "url": "https://server/a.fits",
        },
        {
            "Time": Time("2025-11-28T22:00:00"),
            "Version": "v1.0.0",
            "url": "https://server/b.fits",
        },
    ]
    searches = iter([{"padre": rows}, {"padre": rows[:1]}, {"padre": []}])
    monkeypatch.setattr("sunpy.net.Fido.search", lambda *args, **kwargs: next(searches))
    flares = TimeSeries(time=Time(["2025-11-28T21:30:00"] * 3))
    flares["end_time"] = Time(["2025-11-28T22:45:00"] * 3)
    listed = np.empty(3, dtype=object)
    listed[:] = [
        ["https://elsewhere/b.fits", "https://elsewhere/a.fits"],
        ["https://server/b.fits"],
        [],
    ]
    flares["l0_urls"] = listed

    result = flare_list.find_flare_photon_files(flares)
    assert result["matches_l0"].tolist() == [True, False, True]
    assert result["num_found"].tolist() == [2, 1, 0]
    assert result["found"].tolist() == [
        "https://server/a.fits|https://server/b.fits",
        "https://server/a.fits",
        "",
    ]
    assert result["matches_l0"].dtype.kind == "b"
    assert result["num_found"].dtype.kind == "i"


def test_find_flare_photon_files_requires_listed_urls():
    flares = TimeSeries(time=Time(["2025-11-28T21:30:00"]))
    flares["end_time"] = Time(["2025-11-28T22:45:00"])
    with pytest.raises(ValueError, match="l0_urls"):
        flare_list.find_flare_photon_files(flares)


def test_find_flare_photon_files_empty_input(monkeypatch):
    def unexpected_search(*args, **kwargs):
        raise AssertionError("Search called for an empty flare list")

    monkeypatch.setattr("sunpy.net.Fido.search", unexpected_search)
    flares = TimeSeries(time=Time([], format="isot", scale="utc"))
    flares["end_time"] = Time([], format="isot", scale="utc")
    flares["l0_urls"] = np.empty(0, dtype=object)
    result = flare_list.find_flare_photon_files(flares)
    assert len(result) == 0
    assert result["found"].dtype.kind == "U"
    assert result["matches_l0"].dtype.kind == "b"
    assert result["num_found"].dtype.kind == "i"


def test_check_flare_list_urls(monkeypatch):
    """Existing urls are True, failing ones are False and duplicates are checked once."""
    calls = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        if "missing" in request.full_url:
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)
        return Response()

    monkeypatch.setattr(flare_list.urllib.request, "urlopen", fake_urlopen)
    ts = TimeSeries(time=Time(["2025-11-28T21:52:32", "2025-11-29T03:47:25"]))
    ts["l0_urls"] = [
        ["http://a/ok.fits", "http://a/missing.fits"],
        ["http://a/ok.fits"],
    ]
    result = flare_list.check_flare_list_urls(ts)
    assert result == {"http://a/ok.fits": True, "http://a/missing.fits": False}
    assert len(calls) == 2


def test_check_flare_list_urls_ignores_empty_and_missing(monkeypatch):
    """Empty strings, None and empty lists are not requested or reported."""
    calls = []

    def fake_url_exists(url, timeout):
        calls.append(url)
        return True

    monkeypatch.setattr(flare_list, "_url_exists", fake_url_exists)
    ts = TimeSeries(
        time=Time(["2025-11-28T21:52:32", "2025-11-29T03:47:25", "2025-11-29T04:00:00"])
    )
    ts["l0_urls"] = [["", "  ", None, "http://a/ok.fits"], [], None]
    assert flare_list.check_flare_list_urls(ts) == {"http://a/ok.fits": True}
    assert calls == ["http://a/ok.fits"]
