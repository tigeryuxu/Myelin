/*
 * Reads name of current image and saves ROI in same working directory as MACRO
 * 
 * First reads cwd and active image name
 * Splits by "." ***note that my names have TWO .tifs b/c I named them weirdly. change line 26 to "2" instead of "3" if only ONE ".tif" in name
 *
 * Then appends a numeric counter to the image_name to keep track of the cell number
 * thus, when saved, the ROI name will correlate with cell number
 * 
 * Finally, deletes your ROIs to return to clean slate
 */


// get current working directory
//cwd = getDirectory("current")        // will save images in the directory that your MACRO is located in

//cwd = "J:\\DATA_2017-2018\\Optic_nerve\\EAE_miR_AAV2\\2018.08.07\\ROIs\\"

//cwd = "E:\\new_Tiger\\ROIs\\"
//cwd = "C:\\Users\\Tiger\\Documents\\Tiger 2015\\Antel Lab\\Myelin Quantification\\Source\\DAPIfind\\20) ROIs - Media test\\ROIs\\"
cwd = "C:\\Users\\Tiger\\Documents\\Tiger 2015\\Antel Lab\\Myelin Quantification\\Source\\DAPIfind\\10) Super-resolution\\All generated training data masks\\ROIs\\180215W_uFNet-01_2b(23)_TigerTracings\\"

print(cwd);

// get name of current image
im_name = getTitle();
print(im_name);
splitted = split(im_name, ".");
first_name = splitted[0];

// new name of image to keep cell counter
print(splitted.length);
print(splitted[splitted.length - 1]);

new_name = "";
if (splitted.length == 2) {    // CHANGE TO 2 if only ONE ".tif" in image name
	x = 1;
	// create filename
	new_name = cwd + first_name + '_' + x + '.zip';
	
	new_rename = im_name + '.' + x;
	rename(new_rename);
	
}
else {
	num = splitted[splitted.length - 1];  // gets num idx of cell
	
	x = parseInt(num); // get number
	print(x);
	x = x + 1; // augment counter

	new_name = cwd + first_name + '_' + x + '.zip';

	chars_overhang = 1;  // accounts for how many final chars to take off, so doesn't explode exponentially
	if (x > 10) {
		chars_overhang = 2;
	}
	
	if (x > 100) {
		chars_overhang = 3;
	}

    im_name = substring(im_name, 0, lengthOf(im_name) - chars_overhang); // takes off final character(s)
	new_rename = im_name + x;
	rename(new_rename);
}

// save
roiManager("Save", new_name);
print(new_name);

// then delete all the ROIs so have clean slate
roiManager("delete");
