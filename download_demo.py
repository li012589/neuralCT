import os
import subprocess

#url = "https://drive.google.com/file/d/1vkKrxT599XIQV_tbTXLIxKT9RC5EQohV/view?usp=sharing"
#url = "https://drive.google.com/uc?export=download&id=1vkKrxT599XIQV_tbTXLIxKT9RC5EQohV"
url = "https://www.googleapis.com/drive/v3/files/1HeLY9uY0897uSVq9-ve0E7Weo8KgXcj0?alt=media&key=AIzaSyA7V08DpkCyfdWPiJ40Kkcvru5Ay-UWHis"
cmds =  [["wget",url,"-O","demos.zip"],["unzip","demos.zip"],["mv","files/demo","demo"],["mv","files/database","database"],["rm","-r","files"]]

for cmd in cmds:
    subprocess.check_output(cmd)
