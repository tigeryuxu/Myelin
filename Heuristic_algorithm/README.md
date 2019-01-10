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
   *	individual channels are either separated, or kept together in single .tiff (see option "load five" in GUI below)
   *	all files to be analyzed are located in a SINGLE folder (see "demo-data" folder for example)

  ### 2.	Run main file (main_Heuristic-Myelin.m)
   1. Open the file in MATLAB and press the green "Run" button under the "Editor" tab
   2. Will first prompt you to select folder containing data to be analyzed
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
     * Load five ==> select this option if your raw data is split into individual channels. Will be prompted with new GUI to enter order of channels after submitting this form.
     * Enhance sheaths ==> applies CLAHE to sheaths (increases sensitivity)
     * Divide image ==> cuts image into smaller sections for analysis (saves computing space)
     * Nano YN ==> colocalizes sheaths with nanofiber image (only select if have fiber)
     * Combine RG ==> combines O4 + MBP (if have both channels)
     * Verbose ==> for debugging, plots everything
     * Calibrate ==> helps determine Sigma and Sensitivity
     * Match Full Name ==> leave empty
  
   4. Option to "save parameters" that you entered so can load next time.

## Demo run:
   1. Follow usage steps above, and select the "Demo-data" folder as the first input
   2. In the GUI, select "Load parameters", then navigate to the "Saved parameters" folder and select:
          
          save_params-20x.mat
          
   3. Once the parameters are loaded, press enter to start analysis
   4. To understand the results, navigate to the folder named "Demo-data_Result_X", where X is the largest number
   5. In the results folder, you will find:
      * X.mat file ==> contains all of the analysis saved as a MATLAB file for image number "X" (can ignore)
      * allAnalysis.txt ==> contains summary info (can ignore)
      * output_.csv ==> contains analysis info but saved as separate and categorized ".csv" files
      * Parameters used.mat ==> contains the parameters entered into the GUI
      * ResultX...png ==> output images of the analysis
      * summary.txt ==> includes some summary metrics (can ignore)
   
## Troubleshooting:
1.	If program does not run or computer freezes:
    * Size of images? Check RAM usage???

