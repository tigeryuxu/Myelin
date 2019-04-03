# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 14:53:43 2019

@author: Neuroimmunology Unit
"""

import random
from tkinter import Tk, Label, Button, Entry, StringVar, DISABLED, NORMAL, END, W, E

class GUI:
    def __init__(self, master):
        self.master = master
        master.title("Myelin Quantification - Parameters")
    
        self.scale = None
        self.minLength = None
        self.sensitivity = None
        self.rolling_ball = None
        self.CLAHE = None
        self.resize = None
        

        # TITLE
        self.label, self.entry = self.new_param(master, "MYELIN QUANTIFICATION")        
        self.label.grid(row=0, column=0, columnspan=2, sticky=W+E)


        # ROW 1:        
        self.label1, self.entry1 = self.new_param(master, "Scale (um/px, default 0.69): ")
        self.label1.grid(row=1, column=0, columnspan=1, sticky=W)
        self.entry1.grid(row=1, column=1, columnspan=1, sticky=W)
        
        # ROW 2:
        self.label2, self.entry2 = self.new_param(master, "minLength (um, default 12): ")
        self.label2.grid(row=2, column=0, columnspan=1, sticky=W)
        self.entry2.grid(row=2, column=1, columnspan=1, sticky=W)        

        # ROW 3:
        self.label3, self.entry3 = self.new_param(master, "Sensitivity (range 2 - 4, default 3): ")
        self.label3.grid(row=3, column=0, columnspan=1, sticky=W)
        self.entry3.grid(row=3, column=1, columnspan=1, sticky=W)        
        
        # ROW 4:
        self.label4, self.entry4 = self.new_param(master, "Rolling ball background sub size (px, default 0): ")
        self.label4.grid(row=4, column=0, columnspan=1, sticky=W)
        self.entry4.grid(row=4, column=1, columnspan=1, sticky=W)        
        
        # ROW 4:
        self.label5, self.entry5 = self.new_param(master, "CLAHE contrast enhancement (yes or no ==> 0 or 1, default 0)")
        self.label5.grid(row=5, column=0, columnspan=1, sticky=W)
        self.entry5.grid(row=5, column=1, columnspan=1, sticky=W) 
        
        # ROW 4:
        self.label6, self.entry6 = self.new_param(master, "Resize image to 0.69 um/px (yes or no ==> 0 or 1, default 0)")
        self.label6.grid(row=6, column=0, columnspan=1, sticky=W)
        self.entry6.grid(row=6, column=1, columnspan=1, sticky=W)       

                
        self.guess_button = Button(master, text="Ok", command=self.close)
        #self.reset_button = Button(master, text="Reset", command=self.reset, state=DISABLED)       
        
        self.guess_button.grid(row=10, column=0, columnspan=2)
        #self.reset_button.grid(row=10, column=1)

    def new_param(self, master, new_text):
        
        self.message = new_text
        self.label_text = StringVar()
        self.label_text.set(self.message)
        self.label = Label(master, textvariable=self.label_text)

        vcmd = master.register(self.validate) # we have to wrap the command
        self.entry = Entry(master, validate="key", validatecommand=(vcmd, '%P'))

        return self.label, self.entry
        
        
    def validate(self, new_text):
        if not new_text: # the field is being cleared
            self.guess = None
            return True
        return True
        
    def close(self):
        #if self.guess is None:
        #    self.message = "Please enter a value"
            
        #else:
        """ SAVE PARAMETERS??? Then do pop-out to give message"""       
        try:
                """ SEE IF TEXT ENTERED ARE INTEGERS"""
                if 0 <= float(self.entry1.get()) <= 100 and 0 <= float(self.entry2.get()) <= 100 and 0 <= float(self.entry3.get()) <= 100:
                    self.scale = self.entry1.get()
                    self.minLength = self.entry2.get()
                    self.sensitivity = self.entry3.get()
                    self.rolling_ball = self.entry4.get()
                    self.CLAHE = self.entry5.get()
                    self.resize = self.entry6.get()
                   
                else:
                    return False
        except ValueError:
                
                return False
        
        self.message = "Parameters saved: " + "Scale: " + self.scale

        self.label_text.set(self.message)

                
        self.master.destroy()
        
    
#root = Tk()
#my_gui = GUI(root)
#root.mainloop()
#root.destroy
#return my_gui.scale, my_gui.minLength, my_gui.sensitivity
