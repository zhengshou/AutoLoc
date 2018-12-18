import os
import sys


if __name__ == '__main__':
    filein = sys.argv[1]

    idxout = os.path.splitext(filein)[0]+".lineidx"

    with open(filein,'r') as tsvin, open(idxout,'w') as tsvout:
        fsize = os.fstat(tsvin.fileno()).st_size
        fpos = 0
        while fpos!=fsize:
            tsvout.write(str(fpos)+"\n")
            tsvin.readline()
            fpos = tsvin.tell()

    print "{} generated.".format(idxout)
