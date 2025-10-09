REM From ChatGPT
pip uninstall -y numpy

REM Sometimes multiple wheels were layered; repeat until it says "not installed"

pip uninstall -y numpy

REM (Optional but helpful) remove stray folders
REM Adjust path from the printed np.__file__ above (go up to ...\site-packages\numpy)
REM Close Python apps first.
REM You can also nuke *.pyd/*.dll leftovers if needed.

pip install "numpy==1.26.4"
python -c "import numpy as np; print('NumPy OK:', np.__version__)"
