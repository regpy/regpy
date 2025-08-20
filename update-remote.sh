#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------------------
# Script to update a Git remote URL from cruegge/itreg.git to regpy/regpy.git
# while preserving the protocol (HTTPS vs SSH).
#
# This script was created by ChatGPT (OpenAI).
# -------------------------------------------------------------------

# Current remote URL
remote_url=$(git remote get-url origin)

# Target replacement
old_path="cruegge/itreg.git"
new_path="regpy/regpy.git"

if [[ "$remote_url" =~ ^git@ ]]; then
    # SSH style: git@host:user/repo.git
    if [[ "$remote_url" == *"$old_path" ]]; then
        new_url=$(echo "$remote_url" | sed -E "s#$old_path#$new_path#")
        echo "Updating SSH remote:"
        echo "  $remote_url → $new_url"
        git remote set-url origin "$new_url"
    else
        echo "Remote is SSH but does not match $old_path — no change."
    fi

elif [[ "$remote_url" =~ ^https:// ]]; then
    # HTTPS style: https://host/user/repo.git
    if [[ "$remote_url" == *"$old_path" ]]; then
        new_url=$(echo "$remote_url" | sed -E "s#$old_path#$new_path#")
        echo "Updating HTTPS remote:"
        echo "  $remote_url → $new_url"
        git remote set-url origin "$new_url"
    else
        echo "Remote is HTTPS but does not match $old_path — no change."
    fi

else
    echo "Unknown remote format: $remote_url"
    exit 1
fi

