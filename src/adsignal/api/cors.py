"""CORS configuration, derived from settings.

Split from `app.py` so the policy is testable without building an app, and so
`app.py` stays a composition root rather than a place where policy lives.
"""

from adsignal.config import Settings


def cors_kwargs(settings: Settings) -> dict[str, object]:
    """The `CORSMiddleware` arguments this deployment should use.

    Credentials are never allowed: the API has no cookies and no auth, so
    permitting them would only widen the consequences of a wrong origin.
    """
    return {
        "allow_origins": list(settings.cors_origins),
        "allow_credentials": False,
        "allow_methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["content-type"],
    }
