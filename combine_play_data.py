# あるフォルダ配下にあるplay_dataのpklファイルを結合する
import os
import glob
import pickle

load_data_folder = 'data/play-data/LunarLander-v2/inada'
out_pkl_folder = 'data/pkl/decision/LunarLander-v2/normal'
os.makedirs(out_pkl_folder, exist_ok=True)
out_pkl_name = 'play_inada_LunarLander-v2.pkl'

out_pkl_path = f'{out_pkl_folder}/{out_pkl_name}'

if os.path.isfile(out_pkl_path):
    assert '既にファイル名が存在する'


load_data_files = glob.glob(f'{load_data_folder}/*.pkl')

data = []
for load_file in load_data_files:
    with open(load_file, mode="rb") as f:
        load_data = pickle.load(f)

        data.append(load_data)


os.makedirs(out_pkl_folder, exist_ok=True)
with open(out_pkl_path, 'wb') as tf:
    pickle.dump(data, tf)