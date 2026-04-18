import sys
import numpy as np
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/workspaces/SpLiCE/coco/clip_features/val/139.npy')
vec = np.load(path)
print('File:', path)
print('Shape:', vec.shape)
print('Min/Max/Mean:', vec.min(), vec.max(), vec.mean())
print('First 10 values:', vec)
print('Unknown indices 81-90:', vec[81:91])
