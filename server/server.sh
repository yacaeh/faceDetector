#!/usr/bin/env bash

export PYTHONPATH='/workspace:/workspace/server' 
export LANG=ko_KR.UTF-8
while ! python server.py
do
   sleep 1
   echo "Restarting Server"
done
