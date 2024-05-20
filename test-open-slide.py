#test-open-slide.py

OPENSLIDE_PATH = r'D:\openslide\openslide-bin-4.0.0.2-windows-x64\openslide-bin-4.0.0.2-windows-x64\bin'

import os
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

print(openslide.__version__)