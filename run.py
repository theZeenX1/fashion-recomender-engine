import pandas as pd
import numpy as np
import catDetect as cd

men = pd.read_csv('listMen.csv')
women = pd.read_csv('listWomen.csv')

men['Rating_Rank'] = men['Rate'].rank(ascending = 0)
men = men.set_index('Rating_Rank')

men = men.sort_index()

women['Rating_Rank'] = women['Rate'].rank(ascending = 0)
women = women.set_index('Rating_Rank')

women = women.sort_index()


def Sort(sub_li):
    sub_li.sort(key = lambda x: x[1], reverse=True)
    return sub_li

def init(x):
    if(x == 0):
        for i in range(5):
            print(f"{i} {men.iloc[i]['URL']}")
        selection = input('Enter the clothes you want to wear:')
        return selection, 0
    else:
        for i in range(5):
            print(f"{i}) {women.iloc[i]['URL']}")
        selection = input('Enter the clothes you want to wear:')
        return selection, 1

    
def simCosine(i, gen):
    sim = []
    sub = []
    if gen == 0:
        
        for x, rows in men.iterrows():
            other = []
            if x != i:
                sub.append(np.array(men.loc[i][4:]).transpose().dot(np.array(men.loc[x][4:])))
                sub.append(x)
            sim.append(sub)
        sim = Sort(sim)
        print('Other items to consider: ')
        for i in range (5):
            print(f'{i+1} {men.loc[i][1]}')
                
    if gen == 1:
        root = list(women.loc[i][4:0])
        root = np.array(root).transpose()
        for x, rows in women.iterrows():
            other = []
            if x != i:
                other = list(women.loc[x][4:])
                other = np.array(other)
                sub.append(root.dot(other))
                sub.append(x)
            sim.append(sub)
        sim = Sort(sim)
        print('Other items to consider: ')
        for i in range (5):
            print(f'{i+1} {women.loc[i][1]}')
                
def start():
    cd.openCam()
    x = cd.prop
    print(x[0])
    if(x[0] == 'Male'):
        print('Shopping in Men')
        sel, i = init(0)
        simCosine(sel, i)
    else:
        print('Shopping in Women')
        sel, i = init(1)
        simCosine(sel, i)
        
        
x = input('Do you want to begin?(Y/N)')
if(x == 'Y' or x == 'y'):
    start()
else:
    print('Quitting...')
