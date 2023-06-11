import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import random

def initial_pop(grid_size=150,density=0.65,Ei=60,Ii=2,spread='random'):
    population = (grid_size ** 2) * density
    grid = np.zeros((grid_size,grid_size),dtype=int)
    used_locations = np.zeros((grid_size,grid_size),dtype=int)
    Ui = int(grid_size ** 2 - population)
    Si = int(population - Ei - Ii)
    Qi,Ri = 0,0
    count = [Si,Ei,Ii,Qi,Ri,Ui]
    if spread == 'random':
        for i in range(len(count)):
            for j in range(count[i]):
                used = True
                while used:
                    location = (random.randint(0,grid_size - 1),random.randint(0,grid_size - 1))
                    if used_locations[location] == 0:
                        used = False
                grid[location] = i
                used_locations[location] = 1
    elif spread == 'cluster':
        cluster_size = int(grid_size * 0.25)
        cluster_i_start,cluster_j_start = random.randint(0,grid_size - cluster_size - 1),random.randint(0,grid_size - cluster_size - 1)
        cluster_i_end,cluster_j_end = cluster_i_start + cluster_size, cluster_j_start + cluster_size
        for i in range(count[1]):
            used = True
            while used:
                location = (random.randint(cluster_i_start,cluster_i_end),random.randint(cluster_j_start,cluster_j_end))
                if used_locations[location] == 0:
                    used = False
            grid[location] = 1
            count[1] -= 1
            used_locations[location] = 1
        for i in range(count[2]):
            used = True
            while used:
                location = (random.randint(cluster_i_start,cluster_i_end),random.randint(cluster_j_start,cluster_j_end))
                if used_locations[location] == 0:
                    used = False
            grid[location] = 2
            count[2] -= 1
            used_locations[location] = 1
        for i in range(len(count)):
            for j in range(count[i]):
                used = True
                while used:
                    location = (random.randint(0,grid_size - 1),random.randint(0,grid_size - 1))
                    if used_locations[location] == 0:
                        used = False
                grid[location] = i
                used_locations[location] = 1
    return grid
    
def neighbourhood(population,d,i,j):
    height,width = len(population),len(population)
    count = [0,0,0,0,0,0]
    i_start,j_start = i - d, j - d
    i_end, j_end = i + d, j + d
    if i_start < 0:
        i_start = 0
    if j_start < 0:
        j_start = 0
    if i_end > height:
        i_end = height - 1
    if j_end > width:
        j_end = width - 1
    neighbourhood = population[i_start:i_end + 1,j_start:j_end + 1]
    state = population[i,j]
    for i in range(6):
        count[i] = np.count_nonzero(neighbourhood == i)
    count[state] -= 1
    return count

def change(state,count,Pe,Pi,Pq,Pr):
    if state == 0:
        n = count[1] + count[2]
        p = Pe
        change = np.random.binomial(n,p)
        if change:
            state = 1
    elif state == 1:
        change = np.random.binomial(1,Pi)
        if change:
            state = 2
    elif state == 2:
        change = np.random.binomial(1,Pq)
        if change:
            state = 3
    return state

def plot_grid(pop,day=0,title=False,fig_size=(4.5,4.5)):
    colours = ['white','orange','red','blue','green','black']
    newcmp = LinearSegmentedColormap.from_list('x',colours)
    plt.figure(figsize=fig_size)
    fig = plt.imshow(pop,cmap = newcmp,vmin=0,vmax=5)

    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    if not title:
        title = 'Day:' + str(day)
    else:
        title = str(title)
    plt.title(title)
    plt.show();
    
def plot_data(S,E,I,Q,R,title):
    fig = plt.figure(figsize=(5.5,5.5))
    ax = fig.add_subplot(111)

    plt.plot(S,'black',label = 'S')
    plt.plot(E,'orange' ,label = 'E')
    plt.plot(I, 'red',label = 'I')
    plt.plot(Q, 'blue' ,label = 'Q')
    plt.plot(R,'green', label = 'R')
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.title('Frequency of cells in each state')#
    if title != None:
        plt.suptitle(title)
    plt.legend(loc='best')
    
    I_max = int(max(I))
    day_max = int(I.index(I_max))
    text = 'infection peak: ' + str(I_max) + ', ' + 'peak day: ' + str(day_max)
    plt.annotate(text, xy=(day_max, I_max), xytext=(day_max, I_max+650),
            arrowprops=dict(facecolor='red', shrink=0.05),
            )

    plt.show()

def main(size=150,density=0.625,Ei=60,Ii=2,spread='random',time=60,d=1,Pe=0.5,Pi=0.5,Pq=0.12,Pr=0.1,Ti=2,Tq=2,Tr=10,showGrid=False,showPlot=True,figtitle=None):
    pop = initial_pop(size,density,Ei,Ii,spread)
    days = np.zeros((size,size),dtype=int)
    changed_pop = np.zeros((size,size),dtype=int)
    S,E,I,Q,R = [],[],[],[],[]
    data = [S,E,I,Q,R]
    for t in range(time):
        if showGrid:
            grid_interval = [0,2,5,10,20,40]
            if t in grid_interval:
                plot_grid(pop,t)
        for l in range(len(data)):
            data[l].append(np.count_nonzero(pop == l))
        for i in range(len(pop)):
            for j in range(len(pop[0])):
                state = pop[i,j]
                count = neighbourhood(pop,d,i,j)
                new_state = state
                if state == 0:
                    new_state = change(state,count,Pe,Pi,Pq,Pr)
                    if new_state == state:
                        days[i,j] += 1
                    else:
                        days[i,j] = 0
                elif state == 1:
                    if days[i,j] >= Ti:
                        new_state = change(state,count,Pe,Pi,Pq,Pr)
                        if new_state == state:
                            new_state = 0
                            days[i,j] = 0
                        else:
                            days[i,j] = 0
                    else:
                        days[i,j] += 1
                elif state == 2:
                    if days[i,j] == Tq:
                        new_state = change(state,count,Pe,Pi,Pq,Pr)
                    if days[i,j] >= Tr:
                        new_state = 4
                    if new_state == state:
                        days[i,j] += 1
                    else:
                        days[i,j] = 0
                elif state == 3:
                    if days[i,j] >= Tr:
                        new_state = 4
                    if new_state == state:
                        days[i,j] += 1
                    else:
                        days[i,j] = 0
                changed_pop[i,j] = new_state
        changed_pop = pop
    if showPlot:
        plot_data(S,E,I,Q,R,figtitle)
    return [S,E,I,Q,R]