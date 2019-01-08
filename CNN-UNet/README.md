# **CNN-UNet Myelin Quantification usage notes**


## Installation:
### **Windows installation**
  #### 1.	Install Python or update
  * Download here: [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)
  * Please select version 3.X
        
  #### 2.	Install packages
  Open command-terminal by typing “cmd” in windows search bar, then press "Enter"
  In command-terminal, type:
  
        pip install numpy pillow scipy matplotlib natsort scikit-image
   
   For tensorflow installation, type one of two options in command-line:

          pip install tensorflow-cpu  // if no GPU
          pip install tensorflow-gpu  // if computer has GPU

  #### 3.	Download files
  * Clone or download git repo
  * extract all files
         
   
### **Mac installation:**

  #### 1. Check to update python version
  * ensure version 3.X

  #### 2.	Install packages

      pip install numpy pillow scipy matplotlib natsort scikit-image
      
  For tensorflow installation, type one of two options in command-line:

          pip install tensorflow-cpu  // if no GPU
          pip install tensorflow-gpu  // if computer has GPU

  #### 3.	Download files
  * Clone or download git repo
  * extract all files

## Usage:
  ### 1.	Data format
   * please ensure all images are “.tiff” format
   *	channels are NOT separated
   *	all files to be analyzed are located in a SINGLE folder (see example below)

  ### 2.	Run main file
   In command console type:
           
           python mainUNet.py
  
   Then navigate the GUI
   * First thing that appears prompts you to enter some parameters for the analysis
   *	Then navigate to and select the directory containing the checkpoint file ("Checkpoint") directory
   *	Then navigate to and select the directory you wish to save the output
   *	Then navigate to and select the directory that contains the ".tiff" images to be analyzed

  ### 3. Understanding the output/results:
  Under the directory you selected to save all files, you should find:
    * ...
    * ...
    * ...

## Training:


## Demo run:
  ### 1. Run main file and follow directions in "Usage" above
  * when prompted with GUI, select the following folders:
      * "Checkpoint"
      * "Results"
      * "Demo-data"
     
  ### 2. Check the results
 

## Optional:
 * If want to use your own checkpoint from training, navigate to the folder “Checkpoints” and replace the files with your own checkpoint files.


## Troubleshooting:
1.	If program does not run or computer freezes:
    * Check the size of your images. If they are larger than 5000 x 5000 pixels, you may need to:
        * move to a computer with RAM > 8 GB
        * crop your image and analyze half of it at a time

