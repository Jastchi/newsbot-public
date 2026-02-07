"""Custom allauth adapters for linking social logins to users."""

from allauth.core.exceptions import ImmediateHttpResponse
from allauth.socialaccount.adapter import DefaultSocialAccountAdapter
from allauth.socialaccount.models import SocialLogin
from django.contrib.auth import get_user_model
from django.http import HttpRequest, HttpResponseRedirect
from django.urls import reverse

SESSION_KEY_SUBSCRIPTION_REQUEST_FROM_SOCIAL = (
    "newsserver_subscription_request_from_social"
)


class SocialAccountAdapter(DefaultSocialAccountAdapter):
    """
    Link social logins to an existing Subscriber when email matches.

    When the email matches, the user is logged in and the social account
    is connected. When there is no Subscriber yet, we do not create one:
    we redirect to the subscription-request flow and create a
    SubscriberRequest instead.
    """

    def pre_social_login(
        self,
        request: HttpRequest,
        sociallogin: SocialLogin,
    ) -> None:
        """
        Attach social login to Subscriber or redirect to request flow.

        Either attach to an existing Subscriber, or redirect to create a
        SubscriberRequest (no Subscriber created).
        """
        user = sociallogin.user
        if user is None:
            return
        if getattr(user, "pk", None):
            return
        email = (getattr(user, "email", None) or "").strip().lower()
        if not email:
            return
        user_model = get_user_model()
        try:
            existing = user_model.objects.get(email__iexact=email)
        except user_model.DoesNotExist:
            # New user: create SubscriberRequest instead of Subscriber
            request.session[SESSION_KEY_SUBSCRIPTION_REQUEST_FROM_SOCIAL] = {
                "email": email,
                "first_name": getattr(user, "first_name", "") or "",
                "last_name": getattr(user, "last_name", "") or "",
            }
            target = reverse("newsserver:subscription_request_from_social")
            raise ImmediateHttpResponse(
                HttpResponseRedirect(target),
            ) from None
        sociallogin.user = existing
