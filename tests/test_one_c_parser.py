from __future__ import annotations

from app.integrations.one_c.parser import extract_visit_guid, parse_appointments_payload


def test_parse_one_c_payload_list() -> None:
    payload = {
        "appointments": [
            {"Прием": {"GUID": "abc-1", "DATE": "01.01.2026"}},
            {"visit": {"guid": "abc-2", "date": "01.01.2026"}},
        ]
    }

    appointments = parse_appointments_payload(payload)
    assert len(appointments) == 2
    assert extract_visit_guid(appointments[0]) == "abc-1"
    assert extract_visit_guid(appointments[1]) == "abc-2"
