"""
URL configuration for web project.

The `urlpatterns` list routes URLs to views. For more information please
see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/

Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include,
        path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))

"""

from typing import Any

from django.contrib import admin
from django.contrib.admin import AdminSite
from django.http import HttpRequest, HttpResponse, HttpResponseRedirect
from django.urls import include, path, reverse

from web.newsserver.views import signup_google_only


def _admin_login_redirect(
    self: AdminSite,
    request: HttpRequest,
    extra_context: dict[str, Any] | None = None,
) -> HttpResponse:
    """Redirect admin login to app login (Google-only)."""
    next_url = request.GET.get("next", "/admin/")
    login_url = reverse("account_login")
    if next_url:
        login_url = f"{login_url}?next={next_url}"
    return HttpResponseRedirect(login_url)


# Use app login (Google) for admin; no separate admin password
admin.site.login = _admin_login_redirect

urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts/signup/", signup_google_only, name="account_signup"),
    path("accounts/", include("allauth.urls")),
    path("", include("web.newsserver.urls")),
]
