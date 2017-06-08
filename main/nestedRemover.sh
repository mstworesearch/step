#This bash file will take all the nested files out of a given directory and move them only(without their folders) to somewhere else
#Also, Sulav was not here to figure this out

find /home/mw223/Documents/ignFiles/ -type f -print0 | xargs -0 mv -t /home/mw223/Documents/ignData
