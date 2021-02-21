all:
	g++ part1.cpp -o part1 -pthread -std=c++11 `pkg-config --cflags --libs opencv` && ./part1
clean:
	rm part1