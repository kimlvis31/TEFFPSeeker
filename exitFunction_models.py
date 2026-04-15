import os
import importlib
import traceback

#Import TEF Function Files
path_PROJECT      = os.path.dirname(os.path.realpath(__file__))
path_tefFunctions = os.path.join(path_PROJECT, 'teffunctions')
files_teffunctions = os.listdir(path_tefFunctions)

#Search TEF Function Files and Import
TEFFUNCTIONS_MODEL                = dict()
TEFFUNCTIONS_INPUTDATAKEY         = dict()
TEFFUNCTIONS_BATCHPROCESSFUNCTION = dict()
for name_file in files_teffunctions:
    if not (name_file.startswith('teff_') and name_file.endswith('.py')): continue
    name_module   = name_file[:-3]
    name_function = name_file[5:-3]
    try:
        module = importlib.import_module(f"teffunctions.{name_module}")
        TEFFUNCTIONS_MODEL[name_function]                = getattr(module, 'MODEL')
        TEFFUNCTIONS_INPUTDATAKEY[name_function]         = getattr(module, 'INPUTDATAKEYS')
        TEFFUNCTIONS_BATCHPROCESSFUNCTION[name_function] = getattr(module, 'PROCESSBATCH')
    except Exception as e:
        traceback.print_exc()