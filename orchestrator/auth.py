from fastapi import Request, HTTPException
from keycloak import KeycloakOpenID
import os

KEYCLOAK_URL = os.environ.get('KEYCLOAK_URL', 'http://keycloak:8080')
keycloak_openid = KeycloakOpenID(server_url=KEYCLOAK_URL, client_id='biomed-client', realm_name='master')

async def require_role(request: Request, role: str = 'researcher'):
    auth = request.headers.get('authorization')
    if not auth:
        raise HTTPException(status_code=401, detail='Missing token')
    token = auth.split(' ')[1]
    try:
        userinfo = keycloak_openid.userinfo(token)
    except Exception:
        raise HTTPException(status_code=401, detail='Invalid token')
    roles = userinfo.get('roles', [])
    if role not in roles:
        raise HTTPException(status_code=403, detail='Insufficient role')
    return userinfo
