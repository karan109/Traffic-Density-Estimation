CC = g++
CFLAGS = -lpthread -pthread -std=c++11 
LDFLAGS = `pkg-config --cflags --libs opencv`
PLOT = Plotting_script/plot.py
PY = python3
all: 
	$(CC) part1.cpp -o part1 $(CFLAGS) $(LDFLAGS)
	$(CC) part2.cpp -o part2 $(CFLAGS) $(LDFLAGS)
	$(CC) method1.cpp -o method1 $(CFLAGS) $(LDFLAGS)
	$(CC) method2.cpp -o method2 $(CFLAGS) $(LDFLAGS)
	$(CC) method3.cpp -o method3 $(CFLAGS) $(LDFLAGS)
	$(CC) method4.cpp -o method4 $(CFLAGS) $(LDFLAGS)
	$(CC) accuracy.cpp -o accuracy $(CFLAGS) $(LDFLAGS)
	$(CC) baseline.cpp -o baseline $(CFLAGS) $(LDFLAGS)

baseline:
	$(CC) baseline.cpp -o baseline $(CFLAGS) $(LDFLAGS)
method1:
	$(CC) method1.cpp -o method1 $(CFLAGS) $(LDFLAGS)
method2:
	$(CC) method2.cpp -o method2 $(CFLAGS) $(LDFLAGS)

run_baseline:
	./baseline trafficvideo.mp4
run_method1:
	./method1 trafficvideo.mp4 5

run_method2:
	./method2 trafficvideo.mp4 1280 720 

clean:
	rm method1
	rm method2
	rm method3
	rm method4
	rm part1
	rm part2
	rm accuracy
	rm baseline
	rm -f *.o
file_clean:
	rm Crops/*.jpg
	rm Transforms/*.jpg
	rm Videos/*.mp4
compress:
	tar cvzf ../2019CS10699_2019CS50446_ass1_part3.tar.gz .
plot:
	$(PY) $(PLOT)

