import ctypes as C
import os, math
os.add_dll_directory(r"C:\Program Files\Nuke16.0v9")
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin")
DLL=r"C:\Users\Admin\Documents\Dev\TBlur2.0\tblur_plugins\tblur_plugin\bin\16.0\windows\x86_64\TBlur.dll"
lib=C.CDLL(DLL)
print('loaded ok')
