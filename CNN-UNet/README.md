# **CNN-UNet Myelin Quantification usage notes**


## Installation:
### Windows installation (~15 - 30 mins)
  #### 1.	Install Anaconda
  * Download here: [https://www.anaconda.com/download/](https://www.anaconda.com/download/)
  * Select "Python 3.7 version" and the corresponding "64-bit" or "32-bit" processor version
  * Follow the instructions
     
  #### 2. Install Python
  * In Windows Start Menu, search for "Anaconda Prompt" and select it
  * In the black command window that pops up, type:
  
        conda install python=3.6
  
  * Since Tensorflow is only compatible with Python 3.6, please ensure that you enter the command above to install Python 3.6 alongside the default 3.7 in anaconda.
  * enter "y" if prompted
       
  #### 3.	Install packages
  In Anaconda command-terminal, type:
  
        pip install natsort opencv-python
        
  Then
  
        conda install -c https://conda.anaconda.org/conda-forge mahotas 
   
   For tensorflow installation:

        pip install tensorflow
       
  #### 4.	Download files
  * Navigate to home-page of this repository again
  * On the right-hand side, click the green button to "Clone or download ZIP file" of repo
  * Download ZIP and extract all files
  * Save anywhere on your computer
         
   
### **Mac installation:**

  #### 1. Check to update python version
  * ensure version 3.6
  * if not, find a downloadable version here: [https://www.python.org/downloads/release/python-368/](https://www.python.org/downloads/release/python-368/)

  #### 2.	Install packages
  Open a command-terminal and type:
  
      pip3 install numpy pillow scipy matplotlib natsort scikit-image opencv-python tensorflow
      
  To install the final package, mahotas, you will need to first install xcode:
  
      xcode-select --install
  
  A pop-up will jump out after the command above. Follow the instructions to install. Then type:
  
      pip3 install mahotas
      

  #### 3.	Download files
  * Navigate to home-page of this repository again
  * On the right-hand side, click the green button to "Clone or download ZIP file" of repo
  * Download ZIP and extract all files
  * Save anywhere on your computer
  

## Usage:
  ### 1.	Data format
   *  Please ensure all images are “.tiff” format
   *	Channels are NOT separated
   *  The stained sheaths (either MBP or O4) are in the **RED** channel.
   *  Cell nuclei are in the **BLUE** channel
   *	All files to be analyzed are located in a SINGLE folder (see "Demo-data" folder for example)

  ### 2.	Run main file
  1. (a) For Anaconda (Windows):
  
      * Search for "Spyder" in Windows search bar
      * Open the file "main_UNet.py" using Spyder
      * run by pressing the green run button
      
  1. (b) For Mac (command console):
  
      * In command console type:
           
           python3 main_UNet.py
  
  2. Then navigate the GUI
   *  First thing that appears prompts you to enter some parameters for the analysis:
       1. Scale = scale of image in um/px
       2. min Length = minimum length threshold for sheaths (default 12 um)
       3. Senstivity = length factor for cells that are single or doubly ensheathed (2 == lowest, 4 == highest sensitivity)
   *	Then navigate to and select the directory you wish to save the output
   *	Then navigate to and select the directory that contains the ".tiff" images to be analyzed

  ### 3. Understanding the output/results:
  Under the directory you selected to save all files, you should find:
  * all_fibers_image_name-of-file.pkl   --- contains sheaths identified in original matrix form
  * all_fibers_image_name-of-file.png   --- sheaths identified with cell labels as PNG
  * candidates0_name-of-file.tif        --- candidates selected for analysis
  * final_image_name-of-file.tif        --- sheaths overlaid ontop of original input image
  * masked_out_dil_name-of-file.csv     --- Output data corresponding to EACH INDIVIDUAL input image
    * Within this file, the rows, from top to bottom, correspond to:
       1. lengths of individual sheaths (in pixels, each excel array is a single sheath)
       2. number of ensheathed cells identified
       3. number of sheaths per cell (each excel array is a single cell)
       4. mean sheath length per cell (each excel array is a single cell)
       5. number of candidate cells analyzed
       6. number of total cell nuclei identified
  
  * A folder named "combined_CSVs" which contains: the raw data from EACH INDIVIDUAL input image is then combined in the files "Result_masked_out....csv". There should be 4 combined files, each corresponding to a specific parameter from the raw data (cells, lengths, mSLC, num_sheaths). Within each of these excel sheets, each row contains the raw data from the original INDIVIDUAL input image .csvs.
  
  For examples of these files, check under "Results/Demo-data-output/"
    
## Demo run:
  ### Run the "main_UNet.py" file by following the directions in "Usage" above
  * when prompted with GUI, select the following folders:
      *  create your own folder, then select it for output folder
      * "Demo-data" for input folder      


## Troubleshooting:
1.  Recommended computational specifications:
    * > 8GB RAM

2.	If program does not run or computer freezes:
    * Check the size of your images. If they are larger than 5000 x 5000 pixels, you may need to:
        * move to a computer with RAM > 8 GB
        * crop your image and analyze half of it at a time
        
3.  If you would like to use your own checkpoint from training:
    * navigate to the folder “Checkpoints” and replace the files with your own checkpoint files.
    
    
4. If the analysis is not picking up any sheaths, but they are evident to the human eye, consider enhancing the contrast in your images, either manually using Fiji, or an alternative algorithm like CLAHE.


