# SPDX-License-Identifier: Apache-2.0
# shellcheck shell=bash
#
# Sourced before building lmcache and lmcache-server from one checkout.
# lmcache-server pins lmcache to its own version, so both must get the same pretend version.

# Mirror lmcache's _FOR_ variables onto lmcache-server, prefix by prefix, so both lookups
# agree whichever prefixes setuptools-scm reads. An empty server value counts as unset.
for split_prefix in SETUPTOOLS_SCM VCS_VERSIONING; do
    split_client_var="${split_prefix}_PRETEND_VERSION_FOR_LMCACHE"
    split_server_var="${split_prefix}_PRETEND_VERSION_FOR_LMCACHE_SERVER"
    if [[ -n "${!split_server_var:-}" ]]; then
        continue
    elif [[ -v $split_client_var ]]; then
        export "$split_server_var=${!split_client_var}"
    else
        unset "$split_server_var"
    fi
done

# setuptools-scm >= 10 takes the first of these that is set, even when empty (empty = git).
split_pretend_version() {
    local name
    for name in "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_$1" "VCS_VERSIONING_PRETEND_VERSION_FOR_$1" \
        SETUPTOOLS_SCM_PRETEND_VERSION VCS_VERSIONING_PRETEND_VERSION; do
        if [[ -v $name ]]; then
            printf '%s' "${!name}"
            return
        fi
    done
}
split_client="$(split_pretend_version LMCACHE)"
split_server="$(split_pretend_version LMCACHE_SERVER)"
if [[ "$split_server" != "$split_client" ]]; then
    echo "error: lmcache-server would get pretend version '${split_server:-none, taken from git}'" \
        "but lmcache '${split_client:-none, taken from git}'; give the lmcache-server _FOR_" \
        "variable the lmcache value or unset it" >&2
    exit 1
fi
# setuptools-scm < 10 reads no VCS_VERSIONING_ variables: give lmcache the agreeing server value.
if [[ -n "${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE_SERVER:-}" && ! -v SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE ]]; then
    export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE="$SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE_SERVER"
fi
unset -f split_pretend_version
unset split_prefix split_client_var split_server_var split_client split_server
