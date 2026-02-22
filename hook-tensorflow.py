from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# This forces PyInstaller to grab all hidden TensorFlow parts
hiddenimports = collect_submodules('tensorflow')
datas = collect_data_files('tensorflow')