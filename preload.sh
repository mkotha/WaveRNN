#!/bin/sh
n=0
find "$1" -type f | while read -r f; do
  md5sum "$f">/dev/null &
  n=$(($n+1))
  if [ $n = 1000 ]; then
    wait
    n=0
  fi
done
