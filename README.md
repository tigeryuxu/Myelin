# **CNN-UNet Myelin Quantification usage notes**


## Installation:
### **Windows installation**
  #### 1.	Install Python or update
  * Download here: [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)
  * Please select version X.X or higher
        
  #### 2.	Install packages
  Open command-terminal by typing “cmd” in windows search bar, then press "Enter"
  In command-terminal, type:
  
        pip install numpy pillow scipy matplotlib natsort scikit-image
   
   For tensorflow installation, choose one of two options below depending on GPU:

          pip install tensorflow-cpu  // if no GPU

          pip install tensorflow-gpu  // if computer has GPU

  #### 3.	Download files
  * Clone or download git repo
  * extract all files
         
   
### **Mac installation:**

  #### 1. Check to update python version
  * ensure version X.X or higher

  #### 2.	Install packages

      pip install numpy pillow scipy matplotlib natsort scikit-image

  #### 3.	Download files

      pip install tensorflow-cpu  // if no GPU

      pip install tensorflow-gpu  // if computer has GPU

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
   *	Then navigate to and select the --- directory
   *	Then navigate to and select the --- directory
   *	Then navigate to and select the --- directory

## Training:


## Demo run:


## Optional:
 * If want to use your own checkpoint from training, navigate to the folder “Checkpoints” and replace the files with your own checkpoint files.


## Troubleshooting:
1.	If program does not run or computer freezes:
    * Size of images? Check RAM usage???

