#/usr/sh

ifconfig eth0 | grep "inet addr" | awk -F ":" '{print $2}' | awk -F " " '{print $1}'
