CC = g++
CFLAGS = -lpthread -pthread -std=c++11 
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
	rm Crops/*.jpg
	rm Transforms/*.jpg
	rm Videos/*.mp4
	rm Outputs/user_out.txt
	rm Outputs/user_graph.png
compress:
	tar cvzf ../2019CS10699_2019CS50446_ass1_$(FILEOUT).tar.gz .
run:
	./$(FILEOUT)
	$(PY) $(PLOT) 
plot:
	$(PY) $(PLOT)

baseline:
	$(CC) baseline.cpp -o baseline $(CFLAGS) $(LDFLAGS) 
	./baseline
method3:
	$(CC) method3.cpp -o method3 $(CFLAGS) $(LDFLAGS) 
	./method3


