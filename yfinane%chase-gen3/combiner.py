import pandas as pd
import glob
import os
import tqdm as tqdm


"""
final_combined_data = pd.DataFrame()
for file_path in glob.glob(os.path.join(inputfolder2, '*')):
    if os.path.getsize(file_path) > 2048:
        df = pd.read_csv(file_path)
        final_combined_data = pd.concat([final_combined_data, df])
"""

from concurrent.futures import ProcessPoolExecutor

def process_file(file_path):
    if os.path.getsize(file_path) > 2048:
        df = pd.read_csv(file_path)
        return df.drop("Event",axis=1)
    return pd.DataFrame()  


if __name__ == '__main__':
    inputfolder1 = r"c:\Users\lndnc\Downloads\YF\modern"
    savepath1 = r"c:\Users\lndnc\Downloads\YF\fullmodern.csv"

    inputfolder2 = r"c:\Users\lndnc\Downloads\stockdataarchive\cleanedStocks"
    savepath2 = r"c:\Users\lndnc\Downloads\stockdataarchive\allstocks.csv"

    inputfolder3 = r"c:\Users\lndnc\Downloads\YF\timesplit\new"
    savepath3 = r"c:\Users\lndnc\Downloads\YF\timesplit\allnew.csv"

    inputfolder4 = r"c:\Users\lndnc\Downloads\YF\times\new"
    savepath4 = r"c:\Users\lndnc\Downloads\YF\times\allnew.csv"

    file_paths = glob.glob(os.path.join(inputfolder4, '*'))

    final_combined_data = pd.DataFrame()

    with ProcessPoolExecutor() as executor:
        results = list(tqdm.tqdm(executor.map(process_file, file_paths), total=len(file_paths)))

    final_combined_data = pd.concat(results, ignore_index=True)

    final_combined_data.to_csv(savepath4, index=False)
