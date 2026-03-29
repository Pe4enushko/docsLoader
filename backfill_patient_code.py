"""Backfill patient_code column in MedKard from a 1C JSON export.

Usage:
    python backfill_patient_code.py [path/to/file.json]

Defaults to one_c_response2.json when no path is given.
The script:
  1. Adds the patient_code column if it does not exist yet.
  2. Iterates every appointment in the JSON.
  3. Looks up the matching MedKard row by visit_guid_1c.
  4. Updates patient_code with the value from Пациент.CODE.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from engine.logging_utils import setup_logging
from engine.postgres import connect_postgres

DEFAULT_JSON_PATH = "one_c_response2.json"
LOG_FILE = "logs/backfill_patient_code.log"

ADD_COLUMN_SQL = """
ALTER TABLE public."MedKard"
ADD COLUMN IF NOT EXISTS patient_code varchar(10) NULL;
"""

UPDATE_SQL = """
UPDATE public."MedKard"
SET patient_code = %s
WHERE visit_guid_1c = %s;
"""


def main() -> None:
    setup_logging(LOG_FILE)
    log = logging.getLogger(__name__)

    json_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JSON_PATH
    log.info("Loading appointments from %s", json_path)
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    appointments: list[dict] = payload.get("appointments", [])

    conn = connect_postgres()
    try:
        # Ensure the column exists
        with conn.cursor() as cur:
            cur.execute(ADD_COLUMN_SQL)
        log.info("Ensured patient_code column exists")

        updated = 0
        skipped = 0
        missing = 0

        for appt in appointments:
            visit_info = appt.get("Прием", {})
            patient_info = appt.get("Пациент", {})

            guid: str = visit_info.get("GUID", "").strip()
            patient_code: str = patient_info.get("CODE", "").strip()

            if not guid:
                log.warning("Appointment missing GUID, skipping: %s", appt)
                skipped += 1
                continue

            if not patient_code:
                log.warning("Appointment guid=%s has no patient CODE, skipping", guid)
                skipped += 1
                continue

            with conn.cursor() as cur:
                cur.execute(UPDATE_SQL, (patient_code, guid))
                rows_affected = cur.rowcount

            if rows_affected == 0:
                log.debug("No MedKard row found for guid=%s (patient_code=%s)", guid, patient_code)
                missing += 1
            else:
                log.info("Updated guid=%s -> patient_code=%s", guid, patient_code)
                updated += 1

        conn.commit()
        log.info(
            "Done. updated=%d  missing_in_db=%d  skipped=%d  total=%d",
            updated, missing, skipped, len(appointments),
        )
        print(
            f"Done. updated={updated}  missing_in_db={missing}  "
            f"skipped={skipped}  total={len(appointments)}"
        )
    except Exception:
        conn.rollback()
        log.exception("Backfill failed, rolled back")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
