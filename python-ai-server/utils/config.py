# -- Modules -- #

# Web App
from flask import Flask, render_template, request
import signal

import requests
from selectolax.parser import HTMLParser

# Main
import os, sys, shutil, subprocess, json, pathlib, threading, pyperclip
from time import sleep
from datetime import datetime

# Capture
import webbrowser, pygetwindow

# Download
import m3u8
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Subtitles
import imdb, urllib.request, zipfile, srt, re

# Styling
import colorama
from termcolor import colored
# =========================================== #

# Initilizations
ROOT = os.path.dirname(os.path.dirname(__file__))
os.chdir(ROOT)
colorama.init()
# ------------------------------------------- #

# Settings
with open(os.path.join(ROOT, "setup", "settings.json")) as settings_file:
    settings = json.load(settings_file)
    
    QUALITY = settings["QUALITY"]
    MOVIES_DIR = settings["MOVIES_DIR"]
    TVSHOWS_DIR = settings["TVSHOWS_DIR"]
    SUB_LANGUAGE = settings["SUB_LANGUAGE"]
    FIDDLER_STARTUP_DELAY = settings["FIDDLER_STARTUP_DELAY"]

# CONSTANTS
BASE_URL = "https://solarmovie.pe"
APP = "app"

FIDDLER_PATH = os.path.join(ROOT, APP, "fiddler")
FIDDLER_TMP_REQUESTS = os.path.join(FIDDLER_PATH, "requests")
JSON_STAGED = os.path.join(ROOT, APP, "staged.json")

ALL_EXT = ["jpg", "html", "js", "css", "png", "webp", "txt", "ico"]
START = ["start", "cmd", "/c", "python"]

TIMEOUT = 3
MAX_TRIES = 3
MAX_EXT_TRIES = 2
SERVER = 1

MOVIES_API = f"{BASE_URL}/ajax/movie/episodes/"
SEASONS_API = f"{BASE_URL}/ajax/v2/tv/seasons/"
EPISODES_API = f"{BASE_URL}/ajax/v2/season/episodes/"
EPISODES_WATCH_API = f"{BASE_URL}/ajax/v2/episode/servers/"

LPAD = 100

SUB_WEIGHT = 500
SUBTITLE_RANKS = {
    "trusted": SUB_WEIGHT,
    "silver member": SUB_WEIGHT,
    "gold member": 2*SUB_WEIGHT,
    "platinum member": 3*SUB_WEIGHT,
    "vip lifetime member": 5*SUB_WEIGHT
}
CURRENT_YEAR = datetime.now().year + 1

SHIT_CHARS = ["?", "\\", "/", "|", "\"", ":", "*", "<", ">"]
# ------------------------------------------- #