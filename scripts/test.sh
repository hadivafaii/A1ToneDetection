#!/bin/bash

available_mem=$(awk '/MemFree/ { printf "%d \n", $2/1024/1024 }' /proc/meminfo)
min_mem=60

if (($available_mem < $min_mem)); then
  echo hi
  sleep 4s
  echo hihi
else
  echo bye
  sleep 10s
  echo byebye
fi

echo $available_mem
echo $min_mem
