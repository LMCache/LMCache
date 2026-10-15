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

Service entry points use repeated `--api-key-file`, `LMCACHE_API_KEY_FILE`,
and `LMCACHE_API_KEY` as one cumulative set of accepted credentials. The MP
HTTP server and request-transport adapters use the same loader. Clients send
one key, explicitly or through the shared environment variables.

The MP HTTP server uses the common verifier through
`lmcache.v1.multiprocess.http_auth`. `--http-api-key-file` remains a CLI alias;
legacy HTTP environment variables are fallbacks when their corresponding
unified variables are unset or empty. Its public `/` and `/healthcheck`
routes and disabled-auth behavior remain unchanged. Key-file validation uses
common error messages, and comparisons use UTF-8 bytes so non-ASCII input
does not raise a comparison error.

`lmcache.v1.multiprocess.transport.auth` supplies request-transport key loading
for ZMQ and gRPC adapters. Accepted server keys are cumulative, while a client
sends one explicit key or one environment-supplied key. This helper does not
enable authentication on any transport by itself.

gRPC carries a credential as Bearer metadata. A server interceptor rejects
invalid metadata before protobuf deserialization and handler dispatch; a
client interceptor injects the key into unary and server-streaming calls.
The default insecure channel does not encrypt tokens. Use TLS on untrusted
networks; mTLS can add certificate-based peer identity independently of the
API-key verifier.
