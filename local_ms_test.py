from enum import verify
import os
import msal
import requests
import jwt
from app.security.microsoft_identity import validate_microsoft_id_token

API_BASE = "http://127.0.0.1:8000"
tenant_id = os.environ["MS_TENANT_ID"]
client_id = os.environ["MS_CLIENT_ID"]

app = msal.PublicClientApplication(
    client_id,
    authority=f"https://login.microsoftonline.com/{tenant_id}",
)

result = app.acquire_token_interactive(
    scopes=["User.read"]
)

if "id_token" not in result:
    raise SystemExit(f"Token error: {result}")

claims = jwt.decode(result["id_token"], options={"verify_signature": False})
print("aud:", claims.get("aud"))
print("tid:", claims.get("tid"))
print("iss:", claims.get("iss"))
print("oid:", claims.get("oid"))

try:
    out = validate_microsoft_id_token(result["id_token"])
    print("VALID:", out["oid"])
except Exception as e:
    print("ERROR:", type(e).__name__, str(e))
    c = getattr(e, "__cause__", None)
    if c:
        print("CAUSE:", type(c).__name__, str(c))
s = requests.Session()

r = s.post(f"{API_BASE}/auth/microsoft/login",
           json={"id_token": result["id_token"]},
           )
print("login:", r.status_code, r.text)

r = s.get(f"{API_BASE}/me")
print("me:", r.status_code, r.text)

# Admin-only: create a normal user (replace OID)
# r = s.post(f"{API_BASE}/admin/users", json={"microsoft_oid": "NORMAL_USER_OID", "role": "user", "permissions": []})
# print("create user:", r.status_code, r.text)
