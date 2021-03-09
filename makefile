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
	rm -f *.o
file_clean:
	rm Outputs/user_out.txt
	rm Outputs/user_graph.png
compress:
	tar cvzf ../2019CS10699_2019CS50446_ass1_$(FILEOUT).tar.gz .
clean_compress:
	rm ../*.tar.gz .
run:
	$(CC) $(FILE) -o $(FILEOUT) $(CFLAGS) $(LDFLAGS) && ./$(FILEOUT)
	./$(FILEOUT)
	$(PY) $(PLOT) 
plot:
	$(PY) $(PLOT)


