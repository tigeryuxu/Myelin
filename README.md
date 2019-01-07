# **CNN-UNet Myelin Quantification usage notes**


## Installation:
### **Windows installation**
  #### 1.	Install Python (version X.X or higher) (skip if already installed)
  Download from: 
          python.com
        
  #### 2.	Install pip packages
  Open command-terminal by typing “cmd” in windows search bar
  In command-terminal, type:
  
        pip install numpy pillow scipy matplotlib natsort scikit-image
   
   For tensorflow installation, choose one of two options below depending on GPU:

          pip install tensorflow-cpu  // if have NO GPU

          pip install tensorflow-gpu  // if have GPU

  #### 3.	Download files
      i.	Clone or download git repo

### **Mac installation:**
  #### 1.	Install Python

  #### 2.	Pip packages

  #### 3.	Download files



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
-	If want to use your own checkpoint from training, navigate to the folder “Checkpoints” and replace the files with your own checkpoint files.


## Troubleshooting:
1.	If it does not run:
-	Size of images? Check RAM usage???

