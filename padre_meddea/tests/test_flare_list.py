import urllib.error

from astropy.time import Time
from astropy.timeseries import TimeSeries

from padre_meddea.io import flare_list


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
