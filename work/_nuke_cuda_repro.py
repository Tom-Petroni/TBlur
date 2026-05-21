import nuke
import traceback
import sys

print('NUKE START')
try:
    n = nuke.nodes.Noise(name='TDbgNoise')
    n['format'].setValue('HD_1080')

    t = nuke.nodes.TEdgeAware(name='TDbgEdgeAware')
    t.setInput(0, n)
    t['blur_amount'].setValue(150.0)
    t['edge_threshold'].setValue(0.6)
    t['iterations'].setValue(1)

    t['use_gpu'].setValue(True)
    print('EXEC GPU ON')
    nuke.execute(t.name(), 1, 1)
    print('GPU EXEC OK')

    t['use_gpu'].setValue(False)
    print('EXEC GPU OFF')
    nuke.execute(t.name(), 1, 1)
    print('CPU EXEC OK')

    print('DONE OK')
except Exception as exc:
    print('PYTHON EXCEPTION:', exc)
    traceback.print_exc()
    sys.exit(2)
