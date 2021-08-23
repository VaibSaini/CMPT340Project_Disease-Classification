import zipfile
import shutil 
with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
    zip_ref.extractall('sound_files')

source = 'sound_files/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files'
  
# Destination path 
destination = 'audio_and_txt_files'
  
# Move the content of 
# source to destination 
dest = shutil.move(source, destination) 
