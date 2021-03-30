CC = g++
CFLAGS = -lpthread -pthread -std=c++11 
LDFLAGS = `pkg-config --cflags --libs opencv`
PLOT = ../Analysis/Scripts/plot.py
COMPARE = ../Analysis/Scripts/compare.py
PY = python3
METRIC = abs
all:
	$(CC) baseline.cpp -o baseline $(CFLAGS) $(LDFLAGS)
	$(CC) method1.cpp -o method1 $(CFLAGS) $(LDFLAGS)
	$(CC) method2.cpp -o method2 $(CFLAGS) $(LDFLAGS)
	$(CC) method3.cpp -o method3 $(CFLAGS) $(LDFLAGS)
	$(CC) method4.cpp -o method4 $(CFLAGS) $(LDFLAGS)
	$(CC) method5.cpp -o method5 $(CFLAGS) $(LDFLAGS)
	$(CC) utility.cpp -o utility $(CFLAGS) $(LDFLAGS)

clean:
	rm baseline method1 method2 method3 method4 method5 utility
	rm -f *.o
vid_clean:
	rm ../Data/Videos/*.mp4
compress:
	tar cvzf ../../2019CS10699_2019CS50446_ass1_part3.tar.gz ../
plot:
	$(PY) $(PLOT)
	$(PY) $(PLOT) 1
	$(PY) $(PLOT) 2
	$(PY) $(PLOT) 3
	$(PY) $(PLOT) 4
	$(PY) $(PLOT) 5
compare:
	$(PY) $(COMPARE) 1 $(METRIC)
	$(PY) $(COMPARE) 2 $(METRIC)
	$(PY) $(COMPARE) 3 $(METRIC)
	$(PY) $(COMPARE) 4 $(METRIC)
	$(PY) $(COMPARE) 5 $(METRIC)
fig_clean:
	rm ../Analysis/Outputs/**/**/*.png
	rm ../Analysis/Outputs/**/*.png
	rm ../Analysis/Outputs/*.png