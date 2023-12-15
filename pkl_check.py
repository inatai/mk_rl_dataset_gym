import pickle
import numpy as np
import matplotlib.pyplot as plt

load_data_path = 'C:/Users/user/Desktop/inada/rl-mkdataset/decision-transformer/gym/data/pkl/decision/case2-3.pkl'


def main():

    with open(load_data_path, mode="rb") as f:
        train_data = pickle.load(f)

        out_scatter(train_data)

        # for epi in train_data:
        #     for key in epi.keys(): 

        #         if key == "modes":
        #             print(epi[key][0])


        # for key in train_data.keys(): 
        #     print(f'key:"{key}" = {train_data[key].shape}')
        #     print(train_data[key][-1])

        # for epi in train_data:
        #     for key in epi.keys(): 
        #         # print(f'key:"{key}" = {epi[key].shape}')
        #         if key == "actions":
        #             for s in epi[key]:
        #                 print(s)
        #     break



def out_scatter(train_data):
    l_1 = [0] * 2000
    l_2 = [0] * 2000
    l = [0] * 750
    for epi in train_data:
        score = int(epi["rewards"][-1])
        # l[score] += 1
        if epi["modes"][-1] == 1:
            l_1[score] += 1
        if epi["modes"][-1] == 2:
            l_2[score] += 1

    # for s, num in enumerate(l):
    #     print(f'{s} : {num}')
    # print('########### m = 1 ##############')
    # for s, num in enumerate(l_1):
    #     print(f'{s} : {num}')
    # print('########### m = 2 ##############')
    # for s, num in enumerate(l_2):
    #     print(f'{s} : {num}')
    
    plt.plot(l_1)
    plt.plot(l_2)
    plt.show()

if __name__ == "__main__":
    main()