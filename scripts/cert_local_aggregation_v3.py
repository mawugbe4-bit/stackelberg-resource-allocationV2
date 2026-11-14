"""
CERT r4.2 Local Streaming Aggregation -- v3
--------------------------------------------------------------------------
Runs entirely on your own machine. Extracts the small log files normally and
streams http.csv straight out of the .tar.bz2 archive in chunks (never fully
written to disk).

v3 changes from v1 (all raised in supervisor review):
  1. AFTER-HOURS WINDOW FIXED. v1 used `hour > 20`, which treated 20:00-20:59
     as working hours, so the working window was really 06:00-20:59, not the
     06:00-20:00 stated in Methods 2.6. Working hours are now [06:00, 20:00),
     i.e. after-hours = hour < 6 or hour >= 20. THIS CHANGES EVASION FOR EVERY
     USER-WEEK, so every downstream result must be regenerated.
  2. NO INTERACTIVE PROMPT. v1 stopped at input() so it could not run end to
     end. v3 verifies the required archive members automatically and stops with
     a clear message if any is missing.
  3. ROW COUNTS AT EVERY STEP, written to aggregation_log.txt, including a
     check that the outer join drops no user-week (the "zero attrition" claim).
  4. RAW (UN-NORMALISED) FEATURES are saved too, so the raw distributions and a
     rank-normalisation check can be reported (see cert_feature_diagnostics.py).

Usage:  python cert_local_aggregation_v3.py [--list]
        (--list only prints the archive contents and exits)
"""

import os
import sys
import tarfile
import numpy as np
import pandas as pd

try:
    import config as _cfg
    ARCHIVE_PATH = str(_cfg.CERT_ARCHIVE)
    OUTPUT_DIR = str(_cfg.RESULTS_DIR)      # feature CSVs + log (committed)
    EXTRACT_DIR = str(_cfg.WORK_DIR)        # extracted raw logs (large; git-ignored)
except Exception:                                      # standalone use: edit these lines
    ARCHIVE_PATH = r"C:\Users\YOURNAME\Downloads\r4.2.tar.bz2"
    OUTPUT_DIR = r"C:\Users\YOURNAME\Documents\cert_features"
    EXTRACT_DIR = OUTPUT_DIR

WORK_START_HOUR = 6       # working hours are [06:00, 20:00)
WORK_END_HOUR = 20
SMALL_MEMBERS = ["r4.2/logon.csv", "r4.2/device.csv", "r4.2/email.csv", "r4.2/file.csv", "r4.2/psychometric.csv"]
HTTP_MEMBER = "r4.2/http.csv"

_LOG = []


def log(msg=""):
    print(msg)
    _LOG.append(str(msg))


def write_log():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "aggregation_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(_LOG) + "\n")


def _after_hours_mask(dt_series, lo=WORK_START_HOUR, hi=WORK_END_HOUR):
    """True for events outside [lo:00, hi:00). 20:00-20:59 IS after-hours (v1 wrongly excluded it)."""
    h = dt_series.dt.hour
    return (h < lo) | (h >= hi)


def list_contents():
    with tarfile.open(ARCHIVE_PATH, "r:bz2") as tf:
        for member in tf.getmembers():
            print(f"{member.size / 1024 / 1024:8.1f} MB   {member.name}")


def verify_archive_members():
    """Non-interactive replacement for v1's input() prompt."""
    with tarfile.open(ARCHIVE_PATH, "r:bz2") as tf:
        names = set(tf.getnames())
    missing = [m for m in SMALL_MEMBERS + [HTTP_MEMBER] if m not in names]
    if missing:
        raise SystemExit(f"ERROR: archive members not found: {missing}. Run with --list and edit "
                         f"SMALL_MEMBERS / HTTP_MEMBER (the leading folder name can differ).")
    log(f"Archive check OK: all {len(SMALL_MEMBERS) + 1} required members present.")


def extract_small_files():
    with tarfile.open(ARCHIVE_PATH, "r:bz2") as tf:
        for member_name in SMALL_MEMBERS:
            tf.extract(tf.getmember(member_name), path=EXTRACT_DIR)
            log(f"Extracted: {member_name}")


def stream_aggregate_http(chunksize=200_000):
    running, n_rows, n_chunks = None, 0, 0
    with tarfile.open(ARCHIVE_PATH, "r:bz2") as tf:
        fileobj = tf.extractfile(HTTP_MEMBER)
        if fileobj is None:
            raise FileNotFoundError(f"'{HTTP_MEMBER}' not found in archive")
        for chunk in pd.read_csv(fileobj, chunksize=chunksize):
            n_rows += len(chunk)
            chunk["date"] = pd.to_datetime(chunk["date"])
            chunk["week"] = chunk["date"].dt.to_period("W").astype(str)
            partial = (chunk.groupby(["user", "week"])
                       .agg(http_events=("date", "count"),
                            http_after_hours=("date", lambda s: _after_hours_mask(s).sum()))
                       .reset_index())
            running = partial if running is None else (
                pd.concat([running, partial]).groupby(["user", "week"], as_index=False).sum())
            n_chunks += 1
            if n_chunks % 20 == 0:
                print(f"  ...processed {n_rows:,} http.csv rows so far")
    log(f"http.csv: {n_rows:,} raw rows -> {len(running):,} distinct user-weeks "
        f"({running['user'].nunique():,} users); events kept = {int(running['http_events'].sum()):,} "
        f"(must equal raw rows: {'OK' if int(running['http_events'].sum()) == n_rows else 'MISMATCH'})")
    return running


def _parse_recipients(row):
    parts = set()
    for col in ("to", "cc", "bcc"):
        val = row.get(col)
        if isinstance(val, str) and val.strip():
            parts.update(p.strip() for p in val.split(";") if p.strip())
    return parts


def _read(name):
    df = pd.read_csv(os.path.join(EXTRACT_DIR, "r4.2", name))
    df["date"] = pd.to_datetime(df["date"])
    df["week"] = df["date"].dt.to_period("W").astype(str)
    return df


def _report(name, raw_rows, weekly, event_col):
    log(f"{name}: {raw_rows:,} raw rows -> {len(weekly):,} distinct user-weeks "
        f"({weekly['user'].nunique():,} users); events kept = {int(weekly[event_col].sum()):,} "
        f"({'OK' if int(weekly[event_col].sum()) == raw_rows else 'MISMATCH'})")


def aggregate_small_files():
    logon = _read("logon.csv")
    logon_weekly = (logon.groupby(["user", "week"]).agg(
        logon_events=("date", "count"),
        logon_after_hours=("date", lambda s: _after_hours_mask(s).sum()),
        distinct_pcs=("pc", "nunique")).reset_index())
    _report("logon.csv", len(logon), logon_weekly, "logon_events")

    device = _read("device.csv")
    device_weekly = (device.groupby(["user", "week"]).agg(
        device_events=("date", "count"),
        device_after_hours=("date", lambda s: _after_hours_mask(s).sum())).reset_index())
    _report("device.csv", len(device), device_weekly, "device_events")

    email = _read("email.csv")
    email["recipients"] = email.apply(_parse_recipients, axis=1)
    email_weekly = (email.groupby(["user", "week"]).agg(
        email_events=("date", "count"),
        email_after_hours=("date", lambda s: _after_hours_mask(s).sum()),
        large_attachments=("size", lambda s: (s > 1_000_000).sum())).reset_index())
    recipients_weekly = (email.groupby(["user", "week"])["recipients"]
                         .apply(lambda sets: len(set.union(*sets)) if len(sets) else 0)
                         .reset_index(name="distinct_recipients"))
    email_weekly = email_weekly.merge(recipients_weekly, on=["user", "week"], how="left")
    _report("email.csv", len(email), email_weekly, "email_events")

    filecsv = _read("file.csv")
    file_weekly = (filecsv.groupby(["user", "week"]).agg(
        file_events=("date", "count"),
        file_after_hours=("date", lambda s: _after_hours_mask(s).sum())).reset_index())
    _report("file.csv", len(filecsv), file_weekly, "file_events")
    return logon_weekly, device_weekly, email_weekly, file_weekly


# Impact      = file_events + large_attachments        (volume of data touched, per user-week)
# Propagation = distinct_pcs + distinct_recipients     (reach: machines + people, per user-week)
# Evasion     = after-hours events / all events, over logon/device/email/file/http,
#               after-hours = outside [06:00, 20:00)   (a weekly PROPORTION)
# Confidence  = NOT computed here (classifier output; see cert_confidence_classifier_v2.py)
def build_feature_table():
    logon_w, device_w, email_w, file_w = aggregate_small_files()
    http_w = stream_aggregate_http()

    streams = {"logon": logon_w, "device": device_w, "email": email_w, "file": file_w, "http": http_w}
    key_union = pd.concat([s[["user", "week"]] for s in streams.values()]).drop_duplicates()
    log(f"Union of (user, week) keys across the five streams: {len(key_union):,} "
        f"(distinct users {key_union['user'].nunique():,})")

    merged = (logon_w.merge(device_w, on=["user", "week"], how="outer")
              .merge(email_w, on=["user", "week"], how="outer")
              .merge(file_w, on=["user", "week"], how="outer")
              .merge(http_w, on=["user", "week"], how="outer").fillna(0))
    log(f"After outer joins + zero-fill: {len(merged):,} user-weeks")
    assert len(merged) == len(key_union), "outer join lost or duplicated user-weeks"
    log(f"ATTRITION CHECK: rows in = union of keys = rows out = {len(merged):,}  ->  zero user-weeks dropped")

    merged["impact_raw"] = merged["file_events"] + merged["large_attachments"]
    merged["propagation_raw"] = merged["distinct_pcs"] + merged["distinct_recipients"]
    total_events = (merged["logon_events"] + merged["device_events"] + merged["email_events"]
                    + merged["file_events"] + merged["http_events"])
    total_ah = (merged["logon_after_hours"] + merged["device_after_hours"] + merged["email_after_hours"]
                + merged["file_after_hours"] + merged["http_after_hours"])
    merged["evasion_raw"] = np.where(total_events > 0, total_ah / total_events, 0.0)
    merged["total_events"] = total_events
    merged["threat_id"] = merged["user"] + "_" + merged["week"]

    raw_path = os.path.join(OUTPUT_DIR, "cert_threat_resource_features_raw.csv")
    merged[["threat_id", "impact_raw", "propagation_raw", "evasion_raw", "total_events"]].to_csv(raw_path, index=False)
    log(f"Saved raw (un-normalised) features: {raw_path}")

    norm = merged[["threat_id"]].copy()
    for src, dst in (("impact_raw", "impact"), ("propagation_raw", "propagation"), ("evasion_raw", "evasion")):
        rng = merged[src].max() - merged[src].min()
        norm[dst] = (merged[src] - merged[src].min()) / rng if rng > 0 else 0.0
    norm["confidence"] = np.nan          # placeholder; produced by the classifier stage
    norm["resource_id"] = "TBD"
    norm["cost"] = 1.0
    out_path = os.path.join(OUTPUT_DIR, "cert_threat_resource_features.csv")
    norm[["threat_id", "resource_id", "impact", "propagation", "confidence", "evasion", "cost"]].to_csv(out_path, index=False)
    log(f"Saved normalised features: {out_path}  shape={norm.shape}")
    return merged


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    if "--list" in argv:
        list_contents()
        return
    log(f"Working hours: [{WORK_START_HOUR:02d}:00, {WORK_END_HOUR:02d}:00); after-hours = hour < {WORK_START_HOUR} or hour >= {WORK_END_HOUR}")
    verify_archive_members()
    extract_small_files()
    build_feature_table()
    write_log()
    print(f"\nLog written to {os.path.join(OUTPUT_DIR, 'aggregation_log.txt')}")


if __name__ == "__main__":
    main()
