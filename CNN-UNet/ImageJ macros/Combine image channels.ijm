// read in files to "filesDir"
//dir = getDirectory("Choose a Directory");
//dir = "J:\\DATA_2017-2018\\Optic_nerve\\EAE_miR_AAV2\\2018.08.07\\ON_11\\ROIs\\"
//dir = "C:\\Users\\Neuroimmunology Unit\\Documents\\GitHub\\Optic Nerve\\Etienne\\Control Images\\"
//dir = "C:\\Users\\Neuroimmunology Unit\\Documents\\GitHub\\Optic Nerve\\Training Data\\New folder\\"
dir = getDirectory("Choose a Directory");
//setBatchMode(true);
// ***ALSO MUST OPEN AN IMAGE OF THE CORRECT SIZE WHICH NAME MATCHES LINE #96
count = 0;

list = getFileList(dir);
for (i=0; i<list.length; i++) {
     count++;
	 print(list[i]);
}
n = 0;
//processFiles(dir);
print(count + "files processed");

add_color = 1;
last_num_roi = 0;
for (i = 0; i < list.length; i+=5) {

	path = dir + list[i + 1];  // BRIGHTFIELD
	print(path);
	open(path);
	//selectWindow("path");
	path = dir + list[i + 2]; // Cy3
	print(path);
	open(path);

	path = dir + list[i + 3]; // DAPI
	print(path);
	open(path);
	
	//run("Split Channels");
	//selectWindow(list[i] + " (blue)");
	run("Merge Channels...", "c1=[" + list[i + 2] + "] c2=[" + list[i + 1] + "] c3=[" + list[i + 3] + "] create");
	run("RGB Color");

	// SAVE THE FILE
	//print(dir + "Mask" + path);
	tmpStr = substring(list[i], 0, lengthOf(list[i]) - 4);
	sav_Name = tmpStr + "_combined.tif";
	saveAs("Tiff", dir + sav_Name);	


	// MAKE MASK
	//makeRectangle(0, 1, 1024, 1024);   // SELECT SIZE OF OUTPUT MASK
	//run("Create Mask");

	//newImage("Labeling", "8-bit black", getWidth(), getHeight(), 1);
	
	//print(dir + "Mask" + path);
	//tmpStr = substring(list[i], 0, lengthOf(list[i]) - 4);
	//sav_Name = tmpStr + "_neg_truth.tif";
	//saveAs("Tiff", dir + sav_Name);	
	
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
}

