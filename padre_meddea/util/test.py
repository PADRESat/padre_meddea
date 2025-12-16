from io import BytesIO
from pathlib import Path

from padre_meddea.util.validation import (
    validate_ccsds_apids,
    validate_ccsds_seqnum,
    validate_packet_checksums,
)

data_dir = Path("C:/Users/sdchris1/OneDrive - NASA/Desktop/data/meddea")
files = list(data_dir.glob("*.DAT"))


def replace_byte_in_file(file, position_to_change, new_byte):
    with open(files[0], "rb") as fh:
        buffer = BytesIO(fh.read())
    # Move the cursor to the desired position
    buffer.seek(position_to_change)
    # Write the new byte
    buffer.write(new_byte)

    # Move the cursor back to the beginning to read the entire content
    buffer.seek(0)
    return buffer


# break checksum
buffer = replace_byte_in_file(files[0], 6, b"X")

validation_errors, bad_packets = validate_packet_checksums(
    buffer, output_bad_packets=True
)
print(validation_errors)
print(bad_packets[0].hex(":", 2))

# break APID
buffer = replace_byte_in_file(files[0], 1, b"X")

validation_errors, bad_packets = validate_ccsds_apids(buffer, output_bad_packets=True)
print(validation_errors)
print(bad_packets[0].hex(":", 2))

# no missing packets sequence count
buffer = replace_byte_in_file(files[0], 0, b"0")

validation_errors = validate_ccsds_seqnum(buffer)
print(validation_errors)

# no missing packets sequence count
buffer = replace_byte_in_file(files[0], 5, b"22")

validation_errors = validate_ccsds_seqnum(buffer)
print(validation_errors)
