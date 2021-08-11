import subprocess
import string
import random
import os

def runJob( cmd, cwd='./'):
    """ Run command in a supbrocess
        return True if process finished correctly.
    """
    p = subprocess.Popen(cmd, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    p.wait()
    return 0 == p.returncode

def createHash(length=32):
    """ Creates a a temporary hash name of specificed length"""
    fnRandom = ''.join([random.choice(string.ascii_letters + string.digits) for i in range(length)])
    fnHash = "tmp"+fnRandom
    return fnHash

def create_directory(path):
    """ Create directory if it does not exist
    """
    if not os.path.isdir(path):
        os.mkdir(path)
