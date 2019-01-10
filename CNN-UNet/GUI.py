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

#        self.secret_number = random.randint(1, 100)
#        self.guess = 0
#        self.num_guesses = 0
    
        self.scale = None
        self.minLength = None
        self.sensitivity = None

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

#        try:
#            """ SEE IF TEXT ENTERED IS INT"""
#            guess = int(new_text)
#            if 1 <= guess <= 100:
#                self.guess = guess
#                return True
#            else:
#                return False
#        except ValueError:
#            return False

        return True


#    def reset(self):
#        self.entry.delete(0, END)
#        self.secret_number = random.randint(1, 100)
#        self.guess = 0
#        self.num_guesses = 0
#
#        self.message = "Guess a number from 1 to 100"
#        self.label_text.set(self.message)
#
#        self.guess_button.configure(state=NORMAL)
#        self.reset_button.configure(state=DISABLED)
        
        
    def close(self):
        self.num_guesses += 1

        if self.guess is None:
            self.message = "Please enter a value"

#        elif self.guess == self.secret_number:
#            suffix = '' if self.num_guesses == 1 else 'es'
#            self.message = "Congratulations! You guessed the number after %d guess%s." % (self.num_guesses, suffix)
#            self.guess_button.configure(state=DISABLED)
#            self.reset_button.configure(state=NORMAL)
#
#        elif self.guess < self.secret_number:
#            self.message = "Too low! Guess again!"
        else:
            """ SAVE PARAMETERS??? Then do pop-out to give message"""       
            try:
                """ SEE IF TEXT ENTERED ARE INTEGERS"""
                if 0 <= float(self.entry1.get()) <= 100 and 0 <= float(self.entry2.get()) <= 100 and 0 <= float(self.entry3.get()) <= 100:
                    self.scale = self.entry1.get()
                    self.minLength = self.entry2.get()
                    self.sensitivity = self.entry3.get()
                   
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
