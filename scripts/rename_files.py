import os
from os import path
import shutil

Source_Path = '/home/veer/Documents/mojo/stock-trend/preds/1000_companies_preds'
Destination = '/home/veer/Documents/mojo/stock-trend/preds/1000_companies_preds_main'
os.mkdir(Destination)
def main():
    for count, filename in enumerate(os.listdir(Source_Path)):
        v_index = filename.find('v')
        dst = filename[:v_index + 1]

        # rename all the files
        os.rename(os.path.join(Source_Path, filename),  os.path.join(Destination, dst))


# Driver Code
if __name__ == '__main__':
    main()