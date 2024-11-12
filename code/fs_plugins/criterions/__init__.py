import os
import importlib

# automatically import any Python files in the criterions/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        # if 'dag' not in file_name:
        try:
            importlib.import_module("fs_plugins.criterions." + file_name)
        except Exception as e:
            print('Error started to happen ...')
            print(e)
            print("Import M-DAT modules failed.")

try:
    from .multilingual_loss import *
except Exception as e:
    print('Error started to happen ...')
    print(e)
    print("Import M-DAT modules failed.")