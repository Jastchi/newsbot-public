"""Shared email sending utilities."""

import resend


def send_via_resend(
    api_key: str,
    from_email: str,
    to: list[str],
    subject: str,
    *,
    text: str = "",
    html: str = "",
    headers: dict[str, str] | None = None,
) -> None:
    """Send an email via the Resend API."""
    resend.api_key = api_key
    params: resend.Emails.SendParams = {
        "from": from_email,
        "to": to,
        "subject": subject,
    }
    if text:
        params["text"] = text
    if html:
        params["html"] = html
    if headers:
        params["headers"] = headers
    resend.Emails.send(params)
