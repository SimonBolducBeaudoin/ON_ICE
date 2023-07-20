import os , platform

s = 'C:\\cygwin64\\usr\\x86_64-w64-mingw32\\sys-root\\mingw\\bin'

if ( platform.system() == 'Windows' ) and ( s not in os.environ['PATH'] ) :
    os.environ['PATH'] = s +os.path.pathsep+os.environ['PATH']

from .libfft import *
