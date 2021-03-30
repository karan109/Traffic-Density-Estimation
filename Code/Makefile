CC = g++
CFLAGS = -lpthread -pthread -std=c++11 
LDFLAGS = `pkg-config --cflags --libs opencv`
PLOT = ../Analysis/Scripts/plot.py
PY = python3
all:
	$(CC) method1.cpp -o method1 $(CFLAGS) $(LDFLAGS)
	$(CC) method2.cpp -o method2 $(CFLAGS) $(LDFLAGS)
	$(CC) method3.cpp -o method3 $(CFLAGS) $(LDFLAGS)
	$(CC) method4.cpp -o method4 $(CFLAGS) $(LDFLAGS)
	$(CC) accuracy.cpp -o accuracy $(CFLAGS) $(LDFLAGS)
	$(CC) baseline.cpp -o baseline $(CFLAGS) $(LDFLAGS)
	$(CC) bonus.cpp -o bonus $(CFLAGS) $(LDFLAGS)

clean:
	rm method1 method2 method3 method4 baseline accuracy bonus
	rm -f *.o
file_clean:
	rm ../Data/Videos/*.mp4
compress:
	tar cvzf ../../2019CS10699_2019CS50446_ass1_part3.tar.gz ../
plot:
	$(PY) $(PLOT)