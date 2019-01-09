# **Heuristic Alg. Myelin Quantification usage notes**


## Installation:
### **Windows installation**
  #### 1.	Install MATLAB
  * Download here: [https://www.mathworks.com/downloads/](https://www.mathworks.com/downloads/)
  * Please select version 2017 or higher
        
  #### 2.	Download files
  * Clone or download git repo
  * extract all files
         
   
### **Mac installation:**
  #### 1.	Install MATLAB
  * Download here: [https://www.mathworks.com/downloads/](https://www.mathworks.com/downloads/)
  * Please select version 2017 or higher
        
  #### 2.	Download files
  * Clone or download git repo
  * extract all files


## Usage:
  ### 1.	Data format
   * please ensure all images are “.tiff” format
   *	individual channels are **separated**
   *	all files to be analyzed are located in a SINGLE folder (see "demo-data" folder for example)

  ### 2.	Run main file (mainDAPIfindStack72_run_whole_im_ADULT.m)
   1. Will first prompt you to select folder containing data to be analyzed
   2. Select order of channels to match order of files in folder
   3. Then navigate the GUI
   * First thing that appears prompts you to enter some parameters for the analysis
   ![Image of GUI](https://github.com/yxu233/Myelin/blob/master/Heuristic_algorithm/Images/GUI.PNG)
   
     * Name ==> leave empty
     * Batch names ==> leave empty
     * Scale ==> in um/px
     * Diameter_F ==> diameter of cell body (10 - 25)
     * Sigma ==> thickness of sheaths (5 - 10)
     * Sensitivy ==> determines sensitivity to identifying sheaths (high 0 - 2.0 low)
     * Min Length ==> minimum length of sheaths considered
     * DAPI size ==> Minimum area (px^2) still considered DAPI nucleus
     * Nano YN ==> colocalizes sheaths with nanofiber image (only select if have fiber)
     * Combine RG ==> combines O4 + MBP (if have both channels)
     * Verbose ==> for debugging, plots everything
     * Calibrate ==> helps determine Sigma and Sensitivity
     * Match Full Name ==> leave empty
  
   3. Option to "save parameters" that you entered so can load next time.

## Demo run:


## Troubleshooting:
1.	If program does not run or computer freezes:
    * Size of images? Check RAM usage???

