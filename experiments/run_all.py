import sys, os
import glob

cwd = os.getcwd()

pythonFileList = [f for f in glob.glob('../experiments/**/*.py', recursive=True) if not f.endswith('run_all.py')]

for file in pythonFileList:
  print("Running: " + file)
  os.system('python ' + file)