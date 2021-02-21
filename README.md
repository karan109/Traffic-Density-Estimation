# Traffic Density Estimation
## Dependencies
* OpenCV

## Build executable
```bash
make
```
This builds the executable ./part1

## Run
```bash
./part1 [arg]
```
Here, [arg] is the name of the image. If not provided, then "empty.jpg" is used by default.
```bash
./part1
```
runs the code on Images/empty.jpg 

## Usage
* Build the executable
* Store the original image in the ./Images folder - eg. Images/traffic.jpg
* Execute the following command
```bash
./part1 traffic.jpg
```
* Select 4 points on the image using left mouse clicks starting from the top left, in a counter-clockwise manner.
* In case a wrong point is selected, press the 'u' key to UNDO or the 'r' key to RESET and start from the beginning
* Press the "Enter" key when done with the selection
* The transformed and cropped images will be visible
* Press "Enter" to save the images as "Crops/crop_traffic.jpg" and "Transforms/transform_traffic.jpg"
* Use the "Esc" key at any time to abort and quit the program

## Clean
To remove executables, use
```bash
make clean
```