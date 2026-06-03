from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import httpx
import jwt
from jwt import PyJWKClient


class MicrosoftIdentityError(Exception):
    pass


@dataclass(frozen=True)
class MicrosoftIdentitySettings:
    client_id: str
    tenant_id: str
    issuer: str
    openid_config_url: str


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise MicrosoftIdentityError(f"Missing required environment variable: {name}")
    return value


@lru_cache(maxsize=1)
def get_microsoft_settings() -> MicrosoftIdentitySettings:
    tenant_id = _required_env("MS_TENANT_ID")
    client_id = _required_env("MS_CLIENT_ID")
    issuer = f"https://login.microsoftonline.com/{tenant_id}/v2.0"
    openid_config_url = (
        f"https://login.microsoftonline.com/{tenant_id}/v2.0/.well-known/openid-configuration"
    )
    return MicrosoftIdentitySettings(
        client_id=client_id,
        tenant_id=tenant_id,
        issuer=issuer,
        openid_config_url=openid_config_url,
    )


@lru_cache(maxsize=1)
def _load_openid_config() -> dict[str, Any]:
    settings = get_microsoft_settings()
    response = httpx.get(settings.openid_config_url, timeout=10.0)
    response.raise_for_status()
    data = response.json()
    if "jwks_uri" not in data:
        raise MicrosoftIdentityError("OpenID config missing jwks_uri")
    return data


@lru_cache(maxsize=1)
def _jwks_client() -> PyJWKClient:
    config = _load_openid_config()
    return PyJWKClient(config["jwks_uri"])


def parse_admin_oids() -> set[str]:
    raw = os.getenv("MS_ADMIN_OIDS", "")
    return {oid.strip() for oid in raw.split(",") if oid.strip()}


def validate_microsoft_id_token(id_token: str) -> dict[str, Any]:
    settings = get_microsoft_settings()
    try:
        signing_key = _jwks_client().get_signing_key_from_jwt(id_token)
        claims = jwt.decode(
            id_token,
            key=signing_key.key,
            algorithms=["RS256"],
            audience=settings.client_id,
            issuer=settings.issuer,
            options={
                "require": ["exp", "iat", "iss", "aud", "tid", "oid"],
            },
        )
    except Exception as exc:
        raise MicrosoftIdentityError("Invalid Microsoft ID token") from exc

    token_tenant = str(claims.get("tid", "")).strip()
    if token_tenant != settings.tenant_id:
        raise MicrosoftIdentityError("Token tenant mismatch")

    oid = str(claims.get("oid", "")).strip()
    if not oid:
        raise MicrosoftIdentityError("Token missing oid claim")

    return claims
