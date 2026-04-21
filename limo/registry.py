"""Mapping from LIMO ID (the number on the robot) to its IP address.

Add new LIMOs here as you discover their IPs (`ping limoXXX` on the ME416 wifi).
"""

LIMOS: dict[str, str] = {
    "793": "172.16.2.92",
    "809": "172.16.1.43",
    "814": "172.16.1.136",
}

DEFAULT_LIMO = "809"
UDP_PORT = 12346  # our server; coexists with the class's TCP 12345


def ip_for(limo_id: str) -> str:
    if limo_id not in LIMOS:
        known = ", ".join(sorted(LIMOS)) or "<none>"
        raise KeyError(f"LIMO {limo_id!r} not in registry. Known: {known}")
    return LIMOS[limo_id]
