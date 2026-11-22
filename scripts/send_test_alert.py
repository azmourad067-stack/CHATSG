"""Envoie un SMS de test pour vérifier les alertes de pari (Free Mobile).

Usage : python scripts/send_test_alert.py
Nécessite les variables FREE_MOBILE_USER et FREE_MOBILE_KEY (secrets GitHub).
"""

import sys

import _bootstrap  # noqa: F401
from horse_intelligence.alerts import AlertError, sender_from_env


def main() -> int:
    sender = sender_from_env()
    if sender is None:
        print("::error::secrets FREE_MOBILE_USER / FREE_MOBILE_KEY absents")
        return 1
    try:
        sender("Horse Intelligence : test réussi, les alertes de pari arriveront ici.")
    except AlertError as exc:
        print(f"::error::{exc}")
        return 1
    print("SMS de test envoyé")
    return 0


if __name__ == "__main__":
    sys.exit(main())
