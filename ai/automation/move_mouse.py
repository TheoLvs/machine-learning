#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""--------------------------------------------------------------------
AUTOMATION
Moves the mouse to avoid the computer to close
Started on the 28/12/2016

theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""


import ctypes
import time


def move(seconds):
    mouse_event = ctypes.windll.user32.mouse_event
    MOUSEEVENTF_MOVE = 0x0001
    while True:
        print('NEW LOOP')
        mouse_event(MOUSEEVENTF_MOVE, 0, 0, 0, 0)
        time.sleep(seconds)#sleep for 60 seconds