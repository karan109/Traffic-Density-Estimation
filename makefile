all:
	g++ part1.cpp -o part1 -pthread -std=c++11 `pkg-config --cflags --libs opencv`
clean:
	rm part1
image_clean:
	rm Crops/*.jpg
	rm Transforms/*.jpg
compress:
	tar cvzf ../2019CS10699_2019CS50446_ass1_part1.tar.gz .