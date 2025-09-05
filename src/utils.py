import os
import numpy as np
import matplotlib.pyplot as plt

def generate_base_path(location):
    base_path = os.path.join("results","Graphs")
    folder_to_check = os.path.join(base_path, f"{location}")
    
    if os.path.exists(folder_to_check) and os.path.isdir(folder_to_check):
        base_path = os.path.join("results","Graphs", f"{location}")
    else:
        os.makedirs(folder_to_check)
        base_path = os.path.join("results","Graphs", f"{location}")
    return base_path


def path_for_export(lv1:str = None,lv2:str = None,lv3:str = None, name:str = None, lv4:str = None, lv5:str=None):
    if lv1:
        path = os.path.join(lv1)
    if lv2:
        path = os.path.join(path,lv2)
    if lv3:
        path = os.path.join(path,lv3)
    if lv4:
        path = os.path.join(path,lv4)
    if lv5:
        path = os.path.join(path,lv5)
    if not os.path.exists(path):
        os.makedirs(path)
        
    if name:
        path = os.path.join(path,name)
    return path


def save_plot(base_path, location, fig_name):
    """Save figure to specified location."""
    file_path = path_for_export(lv1=base_path, name=f"{location}_{fig_name}.png")
    plt.savefig(file_path, dpi=150)
    plt.close()


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def area_under_curve(x,y):
    '''Calculate the area under the curve using the trapezoidal rule.
    x: x values
    y: y values
    
    '''
    return np.trapz(y,x)