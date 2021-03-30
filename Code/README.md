# Traffic Density Estimation
## Dependencies
* OpenCV

## Build executable
```bash
make
```
This builds the executable ./part2

## Run
```bash
./part2 [arg]
```
Here, [arg] is the name of the video. If not provided, then "trafficvideo.mp4" is used by default.
```bash
./part2
```
runs the code on Videos/trafficvideo.mp4 (Note that the video is not available as part of this repository and has to be downloaded by the user)

##Usage
* Build the executable
* Store the original video in the ./Videos folder - eg. Videos/trafficvideo.mp4
* Execute the following command
```bash
./part2 trafficvideo.mp4
```
* 2 windows will open up, showing the Queue Density and the Dynamic Density
* Note that this part uses a default homography determined manually by us and does not ask the user repeatedly to select 4 points
* After every 3 frames, the values of Queue Density and Dynamic Density are printed to the console
* After the video is completely processed, the density data is automatically stored at Outputs/user_out.txt. The Python plotting script can now be invoked to produce a graph saved to Outputs/user_graph.png
* Use the "Esc" key at any time to abort and quit the program

## Plot
```bash
make plot
```
will plot the file Outputs/user_out.txt and save the result to Outputs/user_graph.png

## Sample
Sample output file and graph are available at Outputs/ 

## Clean
To remove executables, use
```bash
make clean
```