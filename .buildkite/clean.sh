#!/bin/bash

sleep 600
ps -e | grep pt_main_thread | awk '{print $1}' | xargs kill -9