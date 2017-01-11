import os
import sys
import errno
import uuid


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

        
def uuid_file_name(extension):
    return str(uuid.uuid4()) + "." + extension
