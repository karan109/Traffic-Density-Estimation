all:
	g++ part2.cpp -o part2 -pthread -std=c++11 `pkg-config --cflags --libs opencv`
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
run part2:
	g++ part2.cpp -o part2 -pthread -std=c++11 `pkg-config --cflags --libs opencv`
	./part2
	python3 Plotting_script/plot.py
plot:
	python3 Plotting_script/plot.py


