run("Close All");
// read in files to "filesDir"
dir = getDirectory("Choose a Directory");
//dir = "J:\\DATA_2017-2018\\Optic_nerve\\EAE_miR_AAV2\\2018.08.07\\ON_11\\ROIs\\"
//dir = "C:\\Users\\Neuroimmunology Unit\\Documents\\GitHub\\Optic Nerve\\Etienne\\Control Images\\"
//dir = "C:\\Users\\Neuroimmunology Unit\\Documents\\GitHub\\Optic Nerve\\Training Data\\New folder\\"

//setBatchMode(true);
// ***ALSO MUST OPEN AN IMAGE OF THE CORRECT SIZE WHICH NAME MATCHES LINE #96

list = getFileList(dir);
list = Array.sort(list);
count = 0;
for (i=0; i<list.length; i++) {
     count++;
	 print(list[i]);
}
n = 0;
//processFiles(dir);
print(count + "files processed");

//add_color = 1;
//last_num_roi = 0;
//for (i = 0; i < list.length; i++) {

for (i = 0; i < list.length; i+=2) {
	// delete everything in ROImanager
	for (index = 0; index < roiManager("count"); index++) {
		roiManager("delete");
		print(index);
	}
	// (1) opens INPUT image
	path_input = dir + list[i];
	print(path_input);
	open(path_input);

	// (2) opens TRUTH image
	path_truth = dir + list[i + 1];
	print(path_truth);
	open(path_truth);
	run("8-bit");
	run("Close-");   //  imclosing just cleans up the image a bit

	// convert truth to ROI
	requires("1.33f");
	//run("Blobs (25K)");
	setThreshold(1, 255);
	run("Analyze Particles...", "minimum=50 maximum=9999999 bins=20 show=Nothing clear record");
	for (k=1; k<nResults; k++) {
	    x = getResult('XStart', k);
	    y = getResult('YStart', k);
	    doWand(x,y); // make into rectangle tool???
	    getSelectionCoordinates(x, y); 

		//show the ROI coordinates 
		for (j=0;j<x.length;j++){ 
		 print (""+x[j]+"  "+y[j]); 
		} 
		
		run("Select None"); 
		//wait(5); // no ROI now 
		
	    makeSelection("plygon", x, y); 

	    roiManager("add");
	}
	selectWindow(list[i]);
	roiManager("Show All");
	
	waitForUser("Wait");  // waits for user to add everything before continuing

	selectWindow(list[i]);
	// if user deletes everything in roi, then save a blank mask
	if (roiManager("count") == 0){
		print('blank');
		print(roiManager("count"));
		newImage("Untitled", "8-bit black", 640, 1024, 1);

		tmpStr = substring(list[i], 0, lengthOf(list[i]) - 9);
		sav_Name = tmpStr + "CORRECTED_neg_truth.tif";
		saveAs("Tiff", dir + sav_Name);

	}
	else {
		print('not blank');
		//run("select all")
		array1 = newArray("0");; 
		for (t=0;t<roiManager("count");t++){ 
		        array1 = Array.concat(array1,t); 
		        Array.print(array1); 
		} 
		roiManager("select", array1); 
		roiManager("Combine");
		run("Create Mask");
		//run("Invert");
		run("Invert LUT");
		run("RGB Color");
	
		tmpStr = substring(list[i], 0, lengthOf(list[i]) - 9);
		sav_Name = tmpStr + "CORRECTED_pos_truth.tif";
		saveAs("Tiff", dir + sav_Name);
	}

	run("Close All");
	call("java.lang.System.gc");    // clears memory leak
	call("java.lang.System.gc"); 
	call("java.lang.System.gc"); 
	call("java.lang.System.gc"); 
	call("java.lang.System.gc"); 
	call("java.lang.System.gc"); 
	call("java.lang.System.gc"); 
	call("java.lang.System.gc"); 
	call("java.lang.System.gc"); 
	call("java.lang.System.gc"); 

	// then delete all the ROIs so have clean slate
	//roiManager("delete");
}
