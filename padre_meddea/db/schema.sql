CREATE TABLE files (
    id INTEGER PRIMARY KEY,
    filename TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    data_type TEXT NOT NULL CHECK (data_type IN ('photon', 'housekeeping')),
    level TEXT,
    version TEXT,
    primary_header TEXT
);

CREATE TABLE source_extensions (
    file_id INTEGER NOT NULL REFERENCES files(id),
    hdu_index INTEGER NOT NULL CHECK (hdu_index > 0),
    name TEXT NOT NULL,
    header TEXT NOT NULL,
    PRIMARY KEY (file_id, hdu_index)
);

-- Timestamps use UTC ISO 8601 with nine fractional digits, e.g.
-- 2026-07-04T20:40:13.123456750Z. Original instrument time fields are retained.
CREATE TABLE photon_events (
    file_id INTEGER NOT NULL REFERENCES files(id),
    row_index INTEGER NOT NULL CHECK (row_index >= 0),
    packet_row_index INTEGER NOT NULL CHECK (packet_row_index >= 0),
    timestamp TEXT CHECK (
        timestamp IS NULL OR (
            length(timestamp) = 30
            AND timestamp GLOB (
                '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T'
                || '[0-9][0-9]:[0-9][0-9]:[0-9][0-9].'
                || '[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]Z'
            )
        )
    ),
    seqcount INTEGER,
    clocks INTEGER,
    asic INTEGER,
    channel INTEGER,
    atod INTEGER,
    baseline INTEGER,
    pkttimes INTEGER,
    pktclock INTEGER,
    pixel INTEGER,
    PRIMARY KEY (file_id, row_index),
    FOREIGN KEY (file_id, packet_row_index)
        REFERENCES photon_packets(file_id, row_index)
);

CREATE TABLE photon_packets (
    file_id INTEGER NOT NULL REFERENCES files(id),
    row_index INTEGER NOT NULL CHECK (row_index >= 0),
    -- Elapsed SI time in 50 ns ticks from 2000-01-01 00:00:00 UTC.
    time_ticks INTEGER,
    seqcount INTEGER,
    pkttimes INTEGER,
    pktclock INTEGER,
    pktlength INTEGER,
    livetime INTEGER,
    inttime INTEGER,
    flags INTEGER,
    decim_lvl INTEGER,
    drop_cnt INTEGER,
    int_time_flag INTEGER,
    num_hits INTEGER,
    rate REAL,
    corr_rate REAL,
    PRIMARY KEY (file_id, row_index)
);

CREATE TABLE housekeeping (
    file_id INTEGER NOT NULL REFERENCES files(id),
    row_index INTEGER NOT NULL CHECK (row_index >= 0),
    timestamp TEXT CHECK (
        timestamp IS NULL OR (
            length(timestamp) = 30
            AND timestamp GLOB (
                '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T'
                || '[0-9][0-9]:[0-9][0-9]:[0-9][0-9].'
                || '[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]Z'
            )
        )
    ),
    pkttimes INTEGER,
    raw_timestamp INTEGER,
    fp_temp INTEGER,
    dib_temp INTEGER,
    hvps_temp INTEGER,
    hvps_vsense INTEGER,
    hvps_csense INTEGER,
    csense_15v INTEGER,
    csense_33vd INTEGER,
    csense_33va INTEGER,
    hit_rate INTEGER,
    good_cmd_cnt INTEGER,
    error_cnt INTEGER,
    heater_pwm_duty_cycle INTEGER,
    decimation_rate INTEGER,
    error_summary INTEGER,
    seqcount INTEGER,
    checksum INTEGER,
    ccsds_apid INTEGER,
    ccsds_version_number INTEGER,
    ccsds_packet_type INTEGER,
    ccsds_secondary_flag INTEGER,
    ccsds_sequence_flag INTEGER,
    ccsds_sequence_count INTEGER,
    ccsds_packet_length INTEGER,
    PRIMARY KEY (file_id, row_index)
);

CREATE TABLE command_responses (
    file_id INTEGER NOT NULL REFERENCES files(id),
    row_index INTEGER NOT NULL CHECK (row_index >= 0),
    time_ticks INTEGER,
    time_part1 REAL,
    time_part2 REAL,
    pkttimes INTEGER,
    pktclock INTEGER,
    address INTEGER,
    value INTEGER,
    seqcount INTEGER,
    name TEXT,
    checksum INTEGER,
    PRIMARY KEY (file_id, row_index)
);

CREATE INDEX photon_events_time ON photon_events(timestamp);
CREATE INDEX photon_events_packet ON photon_events(file_id, packet_row_index);
CREATE INDEX photon_packets_time ON photon_packets(time_ticks);
CREATE INDEX photon_packets_identity
    ON photon_packets(file_id, seqcount, pkttimes, pktclock);
CREATE INDEX housekeeping_time ON housekeeping(timestamp);
CREATE INDEX command_responses_time ON command_responses(time_ticks);
