// provide a freehand ROI before running. 

getSelectionCoordinates(x, y); 

//show the ROI coordinates 
for (i=0;i<x.length;i++){ 
 print (""+x[i]+"  "+y[i]); 
} 

run("Select None"); 
wait(500); // no ROI now 

makeSelection("plygon", x, y); 

