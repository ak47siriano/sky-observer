"""Optional Home Assistant MQTT Auto-Discovery Publisher (Disabled by Default, 100% Offline-Safe)."""

from __future__ import annotations

import json
import logging
from typing import Any

from app.config import AppConfig

logger = logging.getLogger(__name__)


def build_ha_discovery_payloads(
    state_topic: str = "sky_observer/state",
    discovery_prefix: str = "homeassistant",
    device_id: str = "sky_observer_local",
    device_name: str = "Sky Observer",
) -> list[tuple[str, dict[str, Any]]]:
    """Generate Home Assistant MQTT Auto-Discovery sensor config topics & payloads."""
    device_block = {
        "identifiers": [device_id],
        "name": device_name,
        "manufacturer": "Sky Observer",
        "model": "Local AI Sky Observatory",
    }
    sensors = [
        ("cloud_cover", "Cloud Cover", "%", "{{ value_json.cloud_coverage_pct }}"),
        ("sky_state", "Sky State", None, "{{ value_json.state }}"),
        ("dominant_cloud", "Dominant Cloud Type", None, "{{ value_json.dominant_cloud_type }}"),
        ("sky_regime", "Sky Regime", None, "{{ value_json.sky_regime }}"),
        ("dark_cloud", "Dark Cloud Base", "%", "{{ value_json.dark_cloud_pct }}"),
        ("estimated_pv_kw", "Estimated PV Power", "kW", "{{ value_json.estimated_pv_kw }}"),
    ]
    out: list[tuple[str, dict[str, Any]]] = []
    for key, label, unit, tpl in sensors:
        topic = f"{discovery_prefix.rstrip('/')}/sensor/{device_id}/{key}/config"
        payload: dict[str, Any] = {
            "name": f"{device_name} {label}",
            "unique_id": f"{device_id}_{key}",
            "state_topic": state_topic,
            "value_template": tpl,
            "device": device_block,
        }
        if unit:
            payload["unit_of_measurement"] = unit
        out.append((topic, payload))
    return out


class HomeAssistantMQTTPublisher:
    """Lazy, non-blocking MQTT publisher using optional `paho-mqtt` only when explicitly enabled."""

    def __init__(self) -> None:
        self._discovery_sent = False

    def publish_observation(self, state_payload: dict[str, Any], config: AppConfig) -> bool:
        mqtt_cfg = getattr(config, "mqtt", None)
        if mqtt_cfg is None or not mqtt_cfg.enabled or not mqtt_cfg.broker_host.strip():
            return False
        if config.server.offline_mode and not mqtt_cfg.allow_in_offline_mode:
            return False
        try:
            import paho.mqtt.publish as mqtt_publish  # type: ignore[import-not-found]
        except ImportError:
            logger.debug("paho-mqtt not installed; skipping Home Assistant MQTT publish")
            return False

        auth = (
            {"username": mqtt_cfg.username, "password": mqtt_cfg.password}
            if mqtt_cfg.username
            else None
        )
        msgs: list[dict[str, Any]] = []
        if mqtt_cfg.ha_discovery_enabled and not self._discovery_sent:
            for topic, cfg_payload in build_ha_discovery_payloads(
                state_topic=mqtt_cfg.state_topic,
                discovery_prefix=mqtt_cfg.ha_discovery_prefix,
            ):
                msgs.append({"topic": topic, "payload": json.dumps(cfg_payload), "retain": True})
            self._discovery_sent = True

        msgs.append({"topic": mqtt_cfg.state_topic, "payload": json.dumps(state_payload), "retain": True})
        try:
            mqtt_publish.multiple(
                msgs,
                hostname=mqtt_cfg.broker_host.strip(),
                port=int(mqtt_cfg.broker_port),
                auth=auth,
                client_id="sky-observer-publisher",
                keepalive=15,
            )
            return True
        except Exception as exc:
            logger.warning("MQTT publish failed: %s", exc)
            return False


_mqtt_publisher = HomeAssistantMQTTPublisher()


def get_mqtt_publisher() -> HomeAssistantMQTTPublisher:
    return _mqtt_publisher
