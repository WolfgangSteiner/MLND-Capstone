import os
import sys
import errno
import uuid
import time
import urllib
import shutil

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def rmdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)


def uuid_file_name(extension):
    return str(uuid.uuid4()) + "." + extension


# Taken from https://blog.shichao.io/2012/10/04/progress_speed_indicator_for_urlretrieve_in_python.html
def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r%d, %d MB, %d KB/s" % (percent, progress_size / (1024 * 1024), speed))
    sys.stdout.flush()


def download(url, filename):
    urllib.urlretrieve(url, filename, reporthook)


def progress_bar(i, n, message=None, length=40):
    percent = float(i) / n
    dots = int(percent * length)
    head = "" if message is None else message + ' ... '
    bar = "[" + '#'*dots + '-'*(length - dots) + ']'
    bar += " %d%%" % (percent*100)
    sys.stdout.write('\r' + head + bar)
    sys.stdout.flush()
