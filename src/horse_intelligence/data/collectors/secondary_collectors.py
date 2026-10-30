"""Connecteurs secondaires identifiés lors de l'audit mais non opérationnels."""

from __future__ import annotations

from horse_intelligence.data.collectors.base import NotOperationalCollector


class OpenPmuCollector(NotOperationalCollector):
    key = "open_pmu_api"
    reason = ("Non audité : hôte inaccessible lors de l'audit, conditions d'utilisation "
              "et fiabilité inconnues. À activer uniquement pour du recoupement.")


class TurfBzhCollector(NotOperationalCollector):
    key = "turf_bzh"
    reason = "Exclu : le projet n'utilise que des API gratuites."
