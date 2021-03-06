all:
	g++ part1.cpp -o part1 -pthread -std=c++11 `pkg-config --cflags --libs opencv`
clean:
	rm part1
	rm part2
	rm helpers
	rm -f *.o
file_clean:
	rm Crops/*.jpg
	rm Transforms/*.jpg
	rm Videos/*.mp4
compress:
	tar cvzf ../2019CS10699_2019CS50446_ass1_part2.tar.gz .
run:
	g++ part1.cpp -o part1 -pthread -std=c++11 `pkg-config --cflags --libs opencv` && ./part1