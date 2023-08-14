# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 13:30:16 2023

@author: FRANT801
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

data = {'Farid': [ 1990,75,174 ],'Nigar': [ 1990,50,160 ],'Emil': [ 2019,17,90 ]}
df = pd.DataFrame ( data=data )
df.index = [ 'Dogum','Ceki','Boy' ]
labels = df.columns
record = df.loc[ df.index[ 0 ] ]
position = range ( len ( record ) )


class IndexTracker ():
    indx = 0

    def previous(self,event):
        print ( 'HAU' )

    def next(self,event):
        print ( "Fiiine" )


current_index = IndexTracker ()
fix,ax = plt.subplots ( figsize=(5,7) )
bars = ax.bar ( position,record )
plt.xticks ( position,labels )
# plt.xlim([1988,2023])
ax.set_title ( record.name )
ax.set_ylim ( [ 1980,2023 ] )

plt.subplots_adjust ( bottom=0.2 )
# create buttons area
ax_prev = plt.axes ( [ 0.58,0.05,0.15,0.07 ] )
ax_next = plt.axes ( [ 0.75,0.05,0.15,0.07 ] )

button_prev = Button ( ax_prev,'Previous',color='green',hovercolor='orange' )
button_next = Button ( ax_next,'Next',color='blue',hovercolor='red' )
button_prev.on_clicked ( current_index.previous )
button_next.on_clicked ( current_index.next )
plt.show()