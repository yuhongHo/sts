from datetime import datetime
import json
import os

from pytz import timezone


def read_json(file_path):
    with open(file_path) as f:
        return json.load(f)


def kst(sec, what):
    kst = datetime.now(timezone('Asia/Seoul'))
    return kst.timetuple()


def make_dirs(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
