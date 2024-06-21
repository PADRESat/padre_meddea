#  this scripts takes a raw packet file which includes all APIDs and creates
#  smaller invididual files with a smaller number of packets
import os

from ccsdspy.utils import (
    split_packet_bytes,
    count_packets,
    split_by_apid,
)

packets_per_file = 4

filename = "20240327_134604_calistesoproto_v02_ba133.bin"

packets_split = split_by_apid(filename)

for key, val in packets_split.items():
    output_filename = f"apid{key}_{packets_per_file}packets.bin"
    with open(output_filename, "wb") as binary_file:
        packet_bytes = split_packet_bytes(packets_split[key])
        for i in range(packets_per_file):
            binary_file.write(packet_bytes[i])

    # now confirm that this file is CCSDS and has the right number of packets
    print(f"Created {output_filename} with {count_packets(output_filename)} packets.")
    print(f"Filesize is {os.path.getsize(output_filename)/1000.0} kilobytes.")
