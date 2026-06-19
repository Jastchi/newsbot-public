"""Tests for utilities/email.py send_via_resend."""

from unittest.mock import MagicMock, patch


class TestSendViaResend:
    def test_sets_api_key_and_sends(self):
        from utilities.email import send_via_resend

        with patch("utilities.email.resend") as mock_resend:
            send_via_resend("key123", "from@ex.com", ["to@ex.com"], "Subj")
            assert mock_resend.api_key == "key123"
            mock_resend.Emails.send.assert_called_once()
            params = mock_resend.Emails.send.call_args[0][0]
            assert params["from"] == "from@ex.com"
            assert params["to"] == ["to@ex.com"]
            assert params["subject"] == "Subj"

    def test_omits_optional_fields_when_empty(self):
        from utilities.email import send_via_resend

        with patch("utilities.email.resend") as mock_resend:
            send_via_resend("k", "f@x.com", ["t@x.com"], "S")
            params = mock_resend.Emails.send.call_args[0][0]
            assert "text" not in params
            assert "html" not in params
            assert "headers" not in params

    def test_includes_text_when_provided(self):
        from utilities.email import send_via_resend

        with patch("utilities.email.resend") as mock_resend:
            send_via_resend("k", "f@x.com", ["t@x.com"], "S", text="hello")
            params = mock_resend.Emails.send.call_args[0][0]
            assert params["text"] == "hello"
            assert "html" not in params

    def test_includes_html_when_provided(self):
        from utilities.email import send_via_resend

        with patch("utilities.email.resend") as mock_resend:
            send_via_resend("k", "f@x.com", ["t@x.com"], "S", html="<b>hi</b>")
            params = mock_resend.Emails.send.call_args[0][0]
            assert params["html"] == "<b>hi</b>"
            assert "text" not in params

    def test_includes_headers_when_provided(self):
        from utilities.email import send_via_resend

        with patch("utilities.email.resend") as mock_resend:
            hdrs = {"List-Unsubscribe": "<mailto:u@x.com>"}
            send_via_resend("k", "f@x.com", ["t@x.com"], "S", headers=hdrs)
            params = mock_resend.Emails.send.call_args[0][0]
            assert params["headers"] == hdrs
