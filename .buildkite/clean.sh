#!/bin/bash

ps -e | grep pt_main_thread | awk '{print $1}' | xargs kill -9 2>/dev/null || true
