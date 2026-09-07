#!/usr/bin/env bash
# Record one of the demos the learn page has a slot for.
#
#   bash scripts/record_demo.sh --list
#   bash scripts/record_demo.sh climb-ascent
#   bash scripts/record_demo.sh climb-ascent --seconds 25 --region 1600x900+0+0
#
# Why this exists: docs/learn.html reserves a slot per demo and fills it in on
# load if the file is there. The only thing standing between an empty slot and a
# working player is a file with the right name, in the right place, small enough
# not to make the page slow. This handles the name, the place and the encoding
# so the only thing left to get right is the robot.
#
# It records the screen. There is deliberately no Gazebo automation here: what
# makes these clips worth watching is a person driving the robot and narrating
# with a terminal open beside it, and a scripted fly-through would lose that.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/docs/media"

# name|what to capture|what must be visible for the clip to be worth anything
DEMOS=(
"climb-ascent|Gazebo side view, foot of the flight to the landing|a terminal running: ros2 topic echo /climb/state"
"carrier-hold|The same flat-ground run twice, hold on then off|rqt_plot or echo of /terrain/state — the phantom risers"
"tof-terrain|Over the 22 mm sill, then refusing the landing edge|RViz with the /terrain/hazards cloud displayed"
"precise-dock|Nav2 arrival, then dock_precisely closing the last 20 cm|echo of /docking/status showing the residual fall"
"agent-errand|r2d2_mcp.agent running an errand end to end|the tool trace — ideally including a failure it recovers from"
)

SECONDS_LIMIT=30
REGION=""
NAME=""

usage() {
  echo "usage: bash scripts/record_demo.sh <name> [--seconds N] [--region WxH+X+Y]"
  echo "       bash scripts/record_demo.sh --list"
  exit "${1:-1}"
}

list_demos() {
  echo "Demos the learn page has a slot for:"
  echo
  for entry in "${DEMOS[@]}"; do
    IFS='|' read -r name what visible <<< "$entry"
    local status="needed"
    [ -f "$OUT/$name.mp4" ] && status="recorded"
    printf '  %-16s %-8s %s\n' "$name" "[$status]" "$what"
    printf '  %-16s %-8s also show: %s\n\n' "" "" "$visible"
  done
  echo "Files land in docs/media/. The page picks them up with no HTML edit."
}

[ $# -eq 0 ] && usage
while [ $# -gt 0 ]; do
  case "$1" in
    --list) list_demos; exit 0 ;;
    -h|--help) usage 0 ;;
    --seconds) SECONDS_LIMIT="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    -*) echo "unknown option: $1" >&2; usage ;;
    *) NAME="$1"; shift ;;
  esac
done

[ -z "$NAME" ] && usage

known=false
for entry in "${DEMOS[@]}"; do
  IFS='|' read -r name what visible <<< "$entry"
  if [ "$name" = "$NAME" ]; then known=true; DESC="$what"; SHOW="$visible"; fi
done
if [ "$known" = false ]; then
  echo "No slot named '$NAME'. Known slots:" >&2
  bash "${BASH_SOURCE[0]}" --list >&2
  exit 1
fi

command -v ffmpeg >/dev/null || {
  echo "ffmpeg is required. On Ubuntu: sudo apt install ffmpeg" >&2; exit 1; }

mkdir -p "$OUT"
MP4="$OUT/$NAME.mp4"
JPG="$OUT/$NAME.jpg"

if [ -z "$REGION" ]; then
  if command -v xdpyinfo >/dev/null; then
    REGION="$(xdpyinfo | awk '/dimensions:/{print $2}')+0+0"
  else
    REGION="1920x1080+0+0"
  fi
fi
SIZE="${REGION%%+*}"
OFFSET="${REGION#*+}"
X="${OFFSET%%+*}"
Y="${OFFSET##*+}"

cat <<INFO

  demo     : $NAME
  capturing: $DESC
  also show: $SHOW
  region   : $SIZE at +$X+$Y, up to ${SECONDS_LIMIT}s
  output   : docs/media/$NAME.mp4

  Arrange your windows now. Recording starts in 5 seconds; press q to stop early.

INFO
sleep 5

# CRF 28 and a 1280-wide cap keep a 30 s clip near 3 MB, which is what keeps the
# page fast. faststart moves the index to the front so it plays before it has
# fully downloaded.
ffmpeg -hide_banner -loglevel warning -stats \
  -f x11grab -framerate 25 -video_size "$SIZE" -i "${DISPLAY:-:0}+$X,$Y" \
  -t "$SECONDS_LIMIT" \
  -vf "scale='min(1280,iw)':-2" \
  -c:v libx264 -preset veryfast -crf 28 -pix_fmt yuv420p \
  -movflags +faststart \
  -y "$MP4"

# A poster frame from one second in, so the slot shows something before play.
ffmpeg -hide_banner -loglevel error -ss 1 -i "$MP4" -frames:v 1 -q:v 4 -y "$JPG"

BYTES=$(stat -c%s "$MP4" 2>/dev/null || stat -f%z "$MP4")
printf '\n  wrote %s (%s KB) and a poster frame\n' \
  "docs/media/$NAME.mp4" "$((BYTES / 1024))"
if [ "$BYTES" -gt 5242880 ]; then
  printf '  \033[33mover 5 MB — re-record shorter, or raise -crf\033[0m\n'
fi
printf '  Open docs/learn.html; the slot fills itself in.\n\n'
