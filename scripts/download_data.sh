#!/usr/bin/env bash
#
# Pulls the Echo dataset from Google Drive into ./data, keeping the layout the
# manifests expect: data/train, data/dev, data/test, and the RL stage's data/rl.
#
#   ./scripts/download_data.sh                 # all four
#   ./scripts/download_data.sh dev test rl     # only those
#   ./scripts/download_data.sh --codecs-only   # skip the audio, which training never reads
#   ./scripts/download_data.sh --list          # what is there, and how big
#   ./scripts/download_data.sh --dry-run dev   # what would be fetched
#
# Overridable: REMOTE, FOLDER_ID, DEST, TRANSFERS.

set -euo pipefail

REMOTE="${REMOTE:-gdrive}"
FOLDER_ID="${FOLDER_ID:-1Fz9t1dniEjYu5FUsxPD5gZF6hWscEk2k}"
DEST="${DEST:-./data}"
TRANSFERS="${TRANSFERS:-8}"

DRIVE=(--drive-root-folder-id "$FOLDER_ID")
COPY=(-P --transfers "$TRANSFERS" --checkers 16 --exclude "*.DS_Store" --tpslimit 10)

list_only=false
dry_run=false
codecs_only=false
parts=()

for argument in "$@"; do
    case "$argument" in
        --list)        list_only=true ;;
        --dry-run)     dry_run=true ;;
        --codecs-only) codecs_only=true ;;
        -h|--help)     sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*)            echo "unknown option: $argument" >&2; exit 2 ;;
        *)             parts+=("$argument") ;;
    esac
done

if [ ${#parts[@]} -eq 0 ]; then
    parts=(train dev test rl)
fi

if ! command -v rclone >/dev/null; then
    echo "rclone is not installed - see knowledge/downloading_the_data.md" >&2
    exit 1
fi

if ! rclone listremotes | grep -qx "${REMOTE}:"; then
    echo "no rclone remote called '${REMOTE}' - run 'rclone config'," >&2
    echo "or point REMOTE at one of: $(rclone listremotes | tr '\n' ' ')" >&2
    exit 1
fi

if [ "$list_only" = true ]; then
    rclone lsd "${DRIVE[@]}" "${REMOTE}:"
    rclone size "${DRIVE[@]}" "${REMOTE}:"
    exit 0
fi

# Copy never deletes at the destination and skips what is already there, so an
# interrupted download is resumed by running the same command again.
if [ "$dry_run" = true ]; then
    COPY+=(--dry-run)
fi

# Training reads the codecs alone - the waveforms are only ever needed to
# re-encode, or to listen. They are also the bulk of the download.
if [ "$codecs_only" = true ]; then
    COPY+=(--exclude "**/audio.zip" --exclude "**/audio/**")
fi

if [ ! -e "${DEST}/unpack.sh" ]; then
    echo "==> unpack.sh"
    # Not COPY: copyto refuses filters on a single file.
    rclone copyto "${DRIVE[@]}" "${REMOTE}:unpack.sh" "${DEST}/unpack.sh" -P $([ "$dry_run" = true ] && echo --dry-run)
fi

for part in "${parts[@]}"; do
    echo "==> ${part}"
    rclone copy "${DRIVE[@]}" "${REMOTE}:${part}" "${DEST}/${part}" "${COPY[@]}"
done

if [ "$dry_run" = false ]; then
    echo
    du -sh "${DEST}"/* 2>/dev/null || true
fi
