CC = g++
CFLAGS = -pthread -std=c++11 
LDFLAGS = `pkg-config --cflags --libs opencv`
FILE = part2.cpp
FILEOUT = part2
PLOT = Plotting_script/plot.py
PY = python3
all: 
	$(CC) $(FILE) -o $(FILEOUT) $(CFLAGS) $(LDFLAGS) 
clean:
	rm $(FILEOUT)
	rm part1
	rm -f *.o
file_clean:
	rm Outputs/user_out.txt
	rm Outputs/user_graph.png
	rm Crops/*.jpg
	rm Transforms/*.jpg
	rm Videos/*.mp4
compress:
	tar cvzf ../2019CS10699_2019CS50446_ass1_$(FILEOUT).tar.gz .
run:
	./$(FILEOUT)
	$(PY) $(PLOT) 
plot:
	$(PY) $(PLOT)


