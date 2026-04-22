"""Slack webhook notifier for crisis alerts.

Usage
-----
    from src.alerts.slack_notifier import SlackNotifier
    notifier = SlackNotifier()                        # reads SLACK_WEBHOOK_URL from .env
    notifier.send(alert)                              # no-op if URL not set
    notifier = SlackNotifier(min_level="CRITICAL")    # only fire on CRITICAL
"""

import os
import json
import logging

import requests
from dotenv import load_dotenv

from src.alerts.alert_schema import CrisisAlert, ALERT_LEVELS, LEVEL_COLORS

load_dotenv()

logger = logging.getLogger(__name__)

_LEVEL_EMOJI = {
    "LOW":      ":information_source:",
    "MEDIUM":   ":warning:",
    "HIGH":     ":rotating_light:",
    "CRITICAL": ":sos:",
}


class SlackNotifier:
    def __init__(self, webhook_url: str = None, min_level: str = "HIGH"):
        """
        Parameters
        ----------
        webhook_url : str, optional
            Slack Incoming Webhook URL. Defaults to SLACK_WEBHOOK_URL env var.
        min_level : str
            Minimum alert level to notify ("LOW", "MEDIUM", "HIGH", "CRITICAL").
        """
        assert min_level in ALERT_LEVELS, f"min_level must be one of {ALERT_LEVELS}"
        self._url = webhook_url or os.getenv("SLACK_WEBHOOK_URL", "")
        self._min_idx = ALERT_LEVELS.index(min_level)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def send(self, alert: CrisisAlert) -> bool:
        """Send a Slack notification for the given alert.

        Returns True on success, False if skipped or failed.
        """
        if not self._url:
            logger.warning("SLACK_WEBHOOK_URL not set — notification skipped.")
            return False

        if ALERT_LEVELS.index(alert.level) < self._min_idx:
            return False

        payload = self._build_payload(alert)
        try:
            resp = requests.post(
                self._url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            resp.raise_for_status()
            return True
        except requests.RequestException as exc:
            logger.error("Slack notification failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_payload(self, alert: CrisisAlert) -> dict:
        emoji = _LEVEL_EMOJI[alert.level]
        color = LEVEL_COLORS[alert.level]
        prob_pct = f"{alert.crisis_probability * 100:.1f}%"

        header_text = f"{emoji} *{alert.level} Crisis Alert* — {prob_pct} probability"

        scores_text = (
            f"*BERT:* {alert.bert_score:.3f}   "
            f"*LSTM:* {alert.lstm_score:.3f}   "
            f"*LDA:* {alert.lda_score:.3f}"
        )

        trigger_text = f"_{alert.trigger_text[:200]}_"

        blocks = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": header_text},
            },
            {"type": "divider"},
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Alert ID:*\n`{alert.alert_id}`"},
                    {"type": "mrkdwn", "text": f"*Timestamp:*\n{alert.timestamp}"},
                ],
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Model Scores*\n{scores_text}"},
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Trigger Tweet*\n{trigger_text}"},
            },
        ]

        if alert.top_tweets:
            top_lines = "\n".join(
                f"• _{t[:120]}_" for t in alert.top_tweets[:3])
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Top Tweets*\n{top_lines}"},
            })

        if alert.escalation_timing:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Escalation Timing*\n{alert.escalation_timing}"},
            })

        if alert.recommended_actions:
            action_lines = "\n".join(
                f"{i+1}. {a}" for i, a in enumerate(alert.recommended_actions))
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Recommended Actions*\n{action_lines}"},
            })

        blocks.append({"type": "divider"})

        return {
            "attachments": [
                {
                    "color": color,
                    "blocks": blocks,
                }
            ]
        }
