#!/usr/bin/env bash
# UserPromptSubmit hook: inject any new/changed keeper submits as context.
# Only triggers when running from the comfy2 project.
set -e

DATA_DIR="/home/chremmler/claude/comfy2/keeper_web/data"
MARKER="$DATA_DIR/.last_seen"

[ -d "$DATA_DIR" ] || exit 0

# Read cwd from hook JSON input on stdin; bail if not in comfy2
cwd=$(jq -r '.cwd // ""' 2>/dev/null || echo "")
case "$cwd" in
  */claude/comfy2*) ;;
  *) exit 0 ;;
esac

last=0
[ -f "$MARKER" ] && last=$(cat "$MARKER")

new_files=()
newest=$last
for f in "$DATA_DIR"/*_submit.md; do
  [ -f "$f" ] || continue
  mtime=$(stat -c %Y "$f")
  if [ "$mtime" -gt "$last" ]; then
    new_files+=("$f")
    [ "$mtime" -gt "$newest" ] && newest=$mtime
  fi
done

[ ${#new_files[@]} -eq 0 ] && exit 0

echo "<keeper-web-submission>"
echo "The user submitted keeper markings via the keeper-web UI. Sync the character memory file(s) (add keepers, remove rejects, capture refinement notes). Tell the user what you updated."
echo
for f in "${new_files[@]}"; do
  echo "=== $(basename "$f") ==="
  cat "$f"
  echo
done
echo "</keeper-web-submission>"

echo "$newest" > "$MARKER"
exit 0
