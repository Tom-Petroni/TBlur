import ctypes as C
import random
import math

DLL=r"C:\Users\Admin\Documents\Dev\TBlur2.0\tblur_plugins\tblur_plugin\bin\16.0\windows\x86_64\TBlur.dll"
lib=C.CDLL(DLL)

class TBlurInputDesc(C.Structure):
    _fields_=[
        ('packed_data', C.POINTER(C.c_float)),
        ('planar_data', C.POINTER(C.c_float)),
        ('plane_size_floats', C.c_int),
        ('row_stride', C.c_int),
        ('col_stride', C.c_int),
        ('chan_a', C.c_int),
        ('chan_b', C.c_int),
        ('chan_c', C.c_int),
        ('chan_d', C.c_int),
    ]

class TBlurDispatch(C.Structure):
    _fields_=[
        ('width', C.c_int),('height', C.c_int),
        ('source', TBlurInputDesc),
        ('guide', TBlurInputDesc),('guide_present', C.c_int),('guide_components', C.c_int),('guide_mode_rgb', C.c_int),('guide_mix', C.c_float),
        ('mask', TBlurInputDesc),('mask_present', C.c_int),('mask_components', C.c_int),('invert_mask', C.c_int),
        ('iterations', C.c_int),('radius_x', C.c_int),('radius_y', C.c_int),('sample_step_x', C.c_int),('sample_step_y', C.c_int),
        ('inv2_sig_t_x', C.c_float),('inv2_sig_t_y', C.c_float),('k', C.c_float),('edge_weight', C.c_float),('edge_despeckle_mix', C.c_float),
        ('hard_stop_mix', C.c_float),('edge_gate', C.c_float),('edge_norm_scale', C.c_float),
        ('show_guide_edge', C.c_int),('keep_alpha', C.c_int),('mix', C.c_float),('organic_cleanup_strength', C.c_float)
    ]

lib.cuda_is_available.restype=C.c_int
lib.cuda_prepare.argtypes=[C.c_int,C.c_int]
lib.cuda_prepare.restype=C.c_int
lib.cuda_process.argtypes=[C.POINTER(TBlurDispatch), C.POINTER(C.c_float)]
lib.cuda_process.restype=C.c_int
lib.cuda_get_last_error.argtypes=[C.c_char_p, C.c_int]
lib.cuda_get_last_error.restype=C.c_int

W,H=256,256
N=W*H
src=(C.c_float*(N*4))()
out=(C.c_float*(N*4))()

for y in range(H):
    for x in range(W):
        i=(y*W+x)*4
        # structured noisy pattern
        v=((x*13+y*7)%97)/97.0
        src[i+0]=v
        src[i+1]=(math.sin(x*0.13)+1.0)*0.5
        src[i+2]=(math.cos(y*0.11)+1.0)*0.5
        src[i+3]=1.0


def clear_desc(d):
    d.packed_data=None
    d.planar_data=None
    d.plane_size_floats=0
    d.row_stride=0
    d.col_stride=0
    d.chan_a=-1
    d.chan_b=-1
    d.chan_c=-1
    d.chan_d=-1


d=TBlurDispatch()
d.width=W; d.height=H
clear_desc(d.source); clear_desc(d.guide); clear_desc(d.mask)
d.source.packed_data=C.cast(src,C.POINTER(C.c_float))
d.guide_present=0; d.guide_components=0; d.guide_mode_rgb=0; d.guide_mix=C.c_float(0.0)
d.mask_present=0; d.mask_components=0; d.invert_mask=0
d.iterations=8
d.radius_x=32; d.radius_y=32
d.sample_step_x=1; d.sample_step_y=1
d.inv2_sig_t_x=C.c_float(1.0/(2.0*8.0*8.0))
d.inv2_sig_t_y=C.c_float(1.0/(2.0*8.0*8.0))
d.k=C.c_float(10.0)
d.edge_weight=C.c_float(0.03)
d.edge_despeckle_mix=C.c_float(0.0)
d.hard_stop_mix=C.c_float(0.0)
d.edge_gate=C.c_float(0.5)
d.edge_norm_scale=C.c_float(0.5)
d.show_guide_edge=0
d.keep_alpha=0
d.mix=C.c_float(1.0)
d.organic_cleanup_strength=C.c_float(0.0)

print('cuda_is_available', lib.cuda_is_available())
print('cuda_prepare', lib.cuda_prepare(W,H))
ok=lib.cuda_process(C.byref(d), out)
print('cuda_process', ok)

buf=C.create_string_buffer(2048)
lib.cuda_get_last_error(buf, len(buf))
print('last_error', buf.value.decode('utf-8', errors='ignore'))

if ok==1:
    acc=0.0
    for i in range(N*3):
        acc += abs(float(out[i])-float(src[i]))
    mad=acc/(N*3)
    print('MAD', mad)
