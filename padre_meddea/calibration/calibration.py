"""
A module for all things calibration.
"""

from pathlib import Path

from astropy.io import fits
from astropy.table import Table

import padre_meddea
import padre_meddea.io.aws_db as aws_db
import padre_meddea.util.pixels as pixels
from padre_meddea import log
from padre_meddea.io import file_tools
from padre_meddea.io.fits_tools import get_comment, get_obs_header, get_primary_header
from padre_meddea.util import validation
from padre_meddea.util.util import (
    calc_time,
    create_meddea_filename,
)

__all__ = [
    "process_file",
]


def process_file(filename: Path, overwrite=False) -> list:
    """
    This is the entry point for the pipeline processing.
    It runs all of the various processing steps required.

    Parameters
    ----------
    data_filename: str
        Fully specificied filename of an input file

    Returns
    -------
    output_filenames: list
        Fully specificied filenames for the output files.
    """
    log.info(f"Processing file {filename}.")

    output_files = []
    file_path = Path(filename)

    if file_path.suffix.lower() in [".bin", ".dat"]:  # raw file
        # Before we process, validate the file with CCSDS
        custom_validators = [
            validation.validate_packet_checksums,
            lambda f: validation.validate_file_size(f, size_limit=10500000),  # 10 MB
        ]
        validation_findings = validation.validate(
            file_path,
            valid_apids=list(padre_meddea.APID.values()),
            custom_validators=custom_validators,
        )
        for finding in validation_findings:
            log.warning(f"Validation Finding for file : {filename} : {finding}")

        parsed_data = file_tools.read_raw_file(file_path)

        # Prepare Metadata for Output Files Naming and Headers
        test_flag = False
        input_level_str = "raw"
        level_str = "l0"

        # Look for each kind of data in the parsed packets
        if parsed_data["photons"] is not None:  # we have event list data
            # Set Data Type for L0 Data
            data_type = "photon"

            pkt_list, event_list = parsed_data["photons"]
            log.info(
                f"Found photon data, {len(event_list)} photons and {len(pkt_list)} packets."
            )

            event_list = Table(event_list)
            event_list.remove_column("time")

            # Get FITS Primary Header Template
            primary_hdr = get_primary_header(
                file_path, data_level=level_str, data_type=data_type
            )

            for this_keyword in ["DATE-BEG", "DATE-END", "DATE-AVG"]:
                primary_hdr[this_keyword] = (
                    event_list.meta.get(this_keyword, ""),
                    get_comment(this_keyword),
                )
            primary_hdr["DATEREF"] = (primary_hdr["DATE-BEG"], get_comment("DATEREF"))

            path = create_meddea_filename(
                time=primary_hdr["DATE-BEG"],
                level=level_str,
                descriptor=data_type,
                test=test_flag,
                overwrite=overwrite,
            )
            primary_hdr["FILENAME"] = (path.name, get_comment("FILENAME"))

            empty_primary_hdu = fits.PrimaryHDU(header=primary_hdr)

            # PKT HDU
            # record originating filename
            aws_db.record_filename(
                file_path.name,
                pkt_list.time[0],
                pkt_list.time[-1],
                level_str=input_level_str,
            )
            # record output filename
            aws_db.record_filename(
                path.name, pkt_list.time[0], pkt_list.time[-1], level_str=level_str
            )
            pkt_list = Table(pkt_list)
            pkt_list.remove_column("time")

            # PKT Header
            pkt_header = get_obs_header(data_level=level_str, data_type=data_type)
            pkt_header["DATE-BEG"] = (
                event_list.meta.get("DATE-BEG", ""),
                get_comment("DATE-BEG"),
            )
            pkt_header["DATEREF"] = (
                event_list.meta.get("DATE-BEG", ""),
                get_comment("DATEREF"),
            )
            pkt_header["FILENAME"] = (path.name, get_comment("FILENAME"))

            pkt_hdu = fits.BinTableHDU(pkt_list, header=pkt_header, name="PKT")
            pkt_hdu.add_checksum()

            # SCI HDU
            hit_header = get_obs_header(data_level=level_str, data_type=data_type)
            hit_header["DATE-BEG"] = (
                event_list.meta.get("DATE-BEG", ""),
                get_comment("DATE-BEG"),
            )
            hit_header["DATEREF"] = (
                event_list.meta.get("DATE-BEG", ""),
                get_comment("DATEREF"),
            )
            hit_header["FILENAME"] = (path.name, get_comment("FILENAME"))

            hit_hdu = fits.BinTableHDU(event_list, header=hit_header, name="SCI")
            hit_hdu.add_checksum()
            hdul = fits.HDUList([empty_primary_hdu, hit_hdu, pkt_hdu])

            # Write the file, with the overwrite option controlled by the environment variable
            hdul.writeto(path, overwrite=overwrite, checksum=True)
            hdul.close()
            # Store the output file path in a list
            output_files.append(path)
        if parsed_data["housekeeping"] is not None:
            # Set Data Type for L0 Data
            data_type = "housekeeping"

            hk_data = parsed_data["housekeeping"]
            # send data to AWS Timestream for Grafana dashboard
            aws_db.record_housekeeping(hk_data)
            hk_table = Table(hk_data)

            # Get FITS Primary Header Template
            primary_hdr = get_primary_header(
                file_path, data_level=level_str, data_type=data_type
            )

            date_beg = calc_time(hk_data["pkttimes"][0])
            date_end = calc_time(hk_data["pkttimes"][-1])
            primary_hdr["DATE-BEG"] = (date_beg.fits, get_comment("DATE-BEG"))
            primary_hdr["DATEREF"] = (date_beg.fits, get_comment("DATEREF"))

            hk_table["seqcount"] = hk_table["CCSDS_SEQUENCE_COUNT"]
            colnames_to_remove = [
                "CCSDS_VERSION_NUMBER",
                "CCSDS_PACKET_TYPE",
                "CCSDS_SECONDARY_FLAG",
                "CCSDS_SEQUENCE_FLAG",
                "CCSDS_APID",
                "CCSDS_SEQUENCE_COUNT",
                "CCSDS_PACKET_LENGTH",
                "CHECKSUM",
                "time",
            ]
            for this_col in colnames_to_remove:
                if this_col in hk_table.colnames:
                    hk_table.remove_column(this_col)

            path = create_meddea_filename(
                time=date_beg,
                level=level_str,
                descriptor=data_type,
                test=test_flag,
                overwrite=overwrite,
            )
            primary_hdr["FILENAME"] = (path.name, get_comment("FILENAME"))
            # record originating filename
            aws_db.record_filename(
                file_path.name, date_beg, date_end, level_str=input_level_str
            )
            # record output filename
            aws_db.record_filename(path.name, date_beg, date_end, level_str=level_str)
            empty_primary_hdu = fits.PrimaryHDU(header=primary_hdr)

            # Create HK HDU
            hk_header = get_obs_header(data_level=level_str, data_type=data_type)
            hk_header["DATE-BEG"] = (date_beg.fits, get_comment("DATE-BEG"))
            hk_header["DATEREF"] = (date_beg.fits, get_comment("DATEREF"))
            hk_header["FILENAME"] = (path.name, get_comment("FILENAME"))

            hk_hdu = fits.BinTableHDU(data=hk_table, header=hk_header, name="HK")
            hk_hdu.add_checksum()

            # add command response data if it exists  in the same fits file
            cmd_header = get_obs_header(data_level=level_str, data_type=data_type)
            cmd_header["FILENAME"] = (path.name, get_comment("FILENAME"))
            if parsed_data["cmd_resp"] is not None:
                data_ts = parsed_data["cmd_resp"]
                cmd_header["DATE-BEG"] = (
                    data_ts.time[0].fits,
                    get_comment("DATE-BEG"),
                )
                cmd_header["DATEREF"] = (
                    data_ts.time[0].fits,
                    get_comment("DATEREF"),
                )
                aws_db.record_cmd(data_ts)
                data_table = Table(data_ts)
                colnames_to_remove = [
                    "CCSDS_VERSION_NUMBER",
                    "CCSDS_PACKET_TYPE",
                    "CCSDS_SECONDARY_FLAG",
                    "CCSDS_SEQUENCE_FLAG",
                    "CCSDS_APID",
                    "CCSDS_SEQUENCE_COUNT",
                    "CCSDS_PACKET_LENGTH",
                    "CHECKSUM",
                    "time",
                ]
                for this_col in colnames_to_remove:
                    if this_col in hk_table.colnames:
                        data_table.remove_column(this_col)
                cmd_hdu = fits.BinTableHDU(
                    data=data_table, header=cmd_header, name="READ"
                )
                cmd_hdu.add_checksum()
            else:  # if None still end an empty Binary Table
                cmd_hdu = fits.BinTableHDU(data=None, header=cmd_header, name="READ")
            hdul = fits.HDUList([empty_primary_hdu, hk_hdu, cmd_hdu])

            hdul.writeto(path, overwrite=overwrite, checksum=True)
            hdul.close()
            output_files.append(path)
        if parsed_data["spectra"] is not None:
            from padre_meddea.spectrum.spectrum import SpectrumList

            # Set Data Type for L0 Data
            data_type = "spectrum"

            # TODO check that asic_nums and channel_nums do not change
            # the function below will remove any change in pixel ids
            pkt_ts, specs, pixel_ids = parsed_data["spectra"]
            ts, spectra, ids = file_tools.clean_spectra_data(pkt_ts, specs, pixel_ids)

            asic_nums, channel_nums = pixels.parse_pixelids(ids)

            # Get FITS Primary Header Template
            primary_hdr = get_primary_header(
                file_path, data_level=level_str, data_type=data_type
            )

            dates = {
                "DATE-BEG": ts.time[0].fits,
                "DATE-END": ts.time[-1].fits,
                "DATE-AVG": ts.time[len(ts.time) // 2].fits,
            }
            primary_hdr["DATEREF"] = (dates["DATE-BEG"], get_comment("DATEREF"))
            for this_keyword, value in dates.items():
                primary_hdr[this_keyword] = (
                    value,
                    get_comment(this_keyword),
                )

            path = create_meddea_filename(
                time=dates["DATE-BEG"],
                level=level_str,
                descriptor=data_type,
                test=test_flag,
                overwrite=overwrite,
            )
            # record originating filename
            aws_db.record_filename(
                file_path.name, ts.time[0], ts.time[-1], level_str=input_level_str
            )
            # record output filename
            aws_db.record_filename(
                path.name, ts.time[0], ts.time[-1], level_str=level_str
            )
            primary_hdr["FILENAME"] = (path.name, get_comment("FILENAME"))

            # Spectrum HDU
            spec_header = get_obs_header(data_level=level_str, data_type=data_type)
            spec_header["DATE-BEG"] = (primary_hdr["DATE-BEG"], get_comment("DATE-BEG"))
            spec_header["DATEREF"] = (primary_hdr["DATE-BEG"], get_comment("DATEREF"))
            spec_header["FILENAME"] = (path.name, get_comment("FILENAME"))

            spec_hdu = fits.CompImageHDU(
                data=spectra.data,
                header=spec_header,
                name="SPEC",
                compression_type="GZIP_1",
            )
            # NOTE: CompImageHDU does not support add_checksum, so we add checksum to the HDUList later

            data_table = Table()
            data_table["pkttimes"] = ts["pkttimes"]
            data_table["pktclock"] = ts["pktclock"]
            data_table["asic"] = asic_nums
            data_table["channel"] = channel_nums
            data_table["seqcount"] = ts["seqcount"]

            pkt_header = get_obs_header(data_level=level_str, data_type=data_type)
            pkt_header["DATE-BEG"] = (primary_hdr["DATE-BEG"], get_comment("DATE-BEG"))
            pkt_header["DATEREF"] = (primary_hdr["DATE-BEG"], get_comment("DATEREF"))
            pkt_header["FILENAME"] = (path.name, get_comment("FILENAME"))
            pkt_hdu = fits.BinTableHDU(data=data_table, header=pkt_header, name="PKT")
            pkt_hdu.add_checksum()

            empty_primary_hdu = fits.PrimaryHDU(header=primary_hdr)
            hdul = fits.HDUList([empty_primary_hdu, spec_hdu, pkt_hdu])

            hdul.writeto(path, overwrite=overwrite, checksum=True)
            hdul.close()
            # calibrate to ql data and send to AWS
            # spec_list = file_tools.read_file(path)
            spec_list = SpectrumList(ts, spectra, ids)
            aws_db.record_spectra(spec_list)
            output_files.append(path)

    # add other tasks below
    return output_files
