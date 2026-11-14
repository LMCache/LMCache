# S3 endpoint addressing and optional payload integrity

The S3 L2 adapter keeps the existing virtual-hosted `s3_endpoint` interface.
For services with a port or base path, callers can instead set
`s3_endpoint_url="http://storage.example:9000/api/s3"` and `s3_bucket="cache"`.
The pair is required together and conflicts with `s3_endpoint`. The URL is raw,
not percent-escaped; userinfo, query, fragment and dot segments are rejected
before CRT resources are allocated. Its scheme selects TLS.

The configuration resolves the request Host and root once. Object operations
append the existing shared ObjectKey codec output, URL-escaped once; list
operations append a separately encoded ListObjectsV2 query. Thus signing sees
the final port, base path, bucket and query. No codec or identity fields change.

```text
legacy endpoint             -> Host: bucket.example  path: /encoded-key
endpoint URL + bucket       -> Host: storage.example:9000
                               path: /api/s3/cache/encoded-key
list                        -> /api/s3/cache/?list-type=2&...
final request               -> CRT signing -> service
```

`s3_integrity_mode="none"` remains the default. Opt-in
`"singlepart_etag_md5"` requires a service that checks Content-MD5 and returns
quoted single-part MD5 ETags. It is unsuitable for services whose ETags have
other meanings. The adapter rejects empty or over-64-MiB PUTs before sending,
hashes the source buffer for Content-MD5, and uses CRT DEFAULT/PutObject to
prevent automatic multipart uploads. GET retains CRT GET_OBJECT; after its
future completes, exact length and complete destination MD5 must match the
single-part ETag before the load bitmap can publish a hit. Missing, malformed
or multipart ETags and corrupt/short/long bodies fail admission.

```text
submit task -> adapter asyncio thread -> CRT request/future
            -> request completes -> length/digest admission (load)
            -> result map / listener (store success) -> completion notifier
```

The opt-in path uses the finished future as the completion authority rather
than a later on_done callback. Failure preserves the existing store result or
load miss contracts and diagnostics. Caller buffers must remain valid through
asynchronous completion; this change adds no cancellation, deadline or owner
protocol. Existing client-local locks, presence caching, retry and shutdown
behavior remain outside this design's changes.

Hashing adds a full buffer read on each integrity PUT and successful GET; no
performance benefit is assumed. Fake-transport tests cover addressing,
pagination, signing and codec identity. Real CRT loopback tests cover single
PUT boundaries, digest rejection and withheld completion publication. These do
not establish production S3 service semantics, accelerator serving, safe
cancellation or cross-node reuse. See [user configuration](../../../../source/mp/l2_storage/s3.rst)
for deployment settings.
