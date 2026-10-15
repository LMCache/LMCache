# API-key authentication

LMCache services share transport-neutral API-key loading and verification from
`lmcache.security.api_key`. Protocol adapters are responsible for extracting a
credential from their wire format and passing only the token to
`APIKeyVerifier.verify_token()`.

The HTTP adapter in `lmcache.security.http` parses Bearer authorization headers
and rejects protected requests before route handling. A service supplies its
own environment-variable names and public paths; this keeps deployment policy
out of the common verifier.

Multiple accepted keys support rotation:

1. Deploy the new key alongside the current key.
2. Update clients to send the new key.
3. Remove the old key after clients have switched.

API keys authenticate callers but do not encrypt traffic. Deploy TLS at a
trusted reverse proxy when credentials cross an untrusted network.

The MP HTTP server uses the common loader and verifier through
`lmcache.v1.multiprocess.http_auth`. Its existing CLI options, environment
variables, public `/` and `/healthcheck` routes, and disabled-auth behavior
remain unchanged. Key-file validation uses common error messages, and token
comparisons use UTF-8 bytes so non-ASCII input does not raise a comparison error.
