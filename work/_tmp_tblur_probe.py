import nuke, math
print('NUKE', nuke.NUKE_VERSION_STRING)
print('HAS_TBLUR', hasattr(nuke.nodes, 'TBlur'))
if not hasattr(nuke.nodes, 'TBlur'):
    raise RuntimeError('TBlur node class not found')

src = nuke.nodes.Noise(name='Src')
src['size'].setValue(28.0)
src['zoffset'].setValue(0.37)

b = nuke.nodes.TBlur(name='B')
b.setInput(0, src)

# force strong blur
b['mix'].setValue(1.0)
b['blur_size'].setValue(500.0)
b['edge_threshold'].setValue(0.0)
b['edge_smooth'].setValue(0.0)
b['iterations'].setValue(10)
b['guide_influence'].setValue(0.0)
b['show_guide_edge'].setValue(False)

# sample difference on a grid
W=256
H=256
step=16
acc=0.0
n=0
for y in range(8, H, step):
    for x in range(8, W, step):
        for c in ('rgba.red','rgba.green','rgba.blue'):
            a = nuke.sample(src, c, x, y)
            o = nuke.sample(b, c, x, y)
            acc += abs(o-a)
            n += 1
mad = acc / max(1,n)
print('MAD', mad)
print('MIX', b['mix'].value())
print('BLUR_SIZE', b['blur_size'].value())
print('ITER', b['iterations'].value())
