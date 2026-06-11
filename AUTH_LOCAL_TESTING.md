# Microsoft Auth Local Testing

## 1) Configure Entra App Registration

1. Create or open your app registration in Microsoft Entra ID.
1. Add SPA redirect URI: `http://localhost:5173`.
1. Ensure ID tokens are enabled for your sign-in flow.
1. Note:
   - Application (client) ID
   - Directory (tenant) ID
   - Admin user object ID(s) for bootstrap

## 2) Backend Environment

Set these in your `.env`:

```env
MS_CLIENT_ID=your-client-id
MS_TENANT_ID=your-tenant-id
MS_ADMIN_OIDS=admin-oid-1,admin-oid-2
```

## 3) Login Flow (Local)

1. Frontend signs in with MSAL and receives `idToken`.
1. Frontend calls:

```http
POST /auth/microsoft/login
Content-Type: application/json

{
  "id_token": "<microsoft-id-token>"
}
```

1. Backend validates token and sets `session` cookie.
1. Frontend calls `GET /me` to confirm authenticated user.

## 4) Verify RBAC

1. First allowlisted admin logs in and is auto-created as `admin`.
1. Admin creates a normal user:
   - `POST /admin/users` with `microsoft_oid`.
1. Created user logs in with Microsoft.
1. Confirm:
   - unknown OID login fails
   - wrong tenant token fails
   - wrong audience token fails

## 5) Useful API Checks

- `GET /me` -> current user profile, role, permissions.
- `GET /admin/users` -> admin-only.
- `PATCH /admin/users/{id}` -> role/active/permissions updates.
- `POST /auth/logout` -> clears session.
