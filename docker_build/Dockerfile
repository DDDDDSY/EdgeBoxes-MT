FROM jjanzic/docker-python3-opencv:contrib

RUN git clone https://github.com/sirebellum/RPi-Edge /RPi-Edge

#Image in
EXPOSE 6001
#Boxes out
Expose 6002

WORKDIR "/RPi-Edge"

#Automatically launch with 6 Gen threads and a Pred thread for every 2 Gen threads
CMD ["bash", "/RPi-Edge/fps_measure.sh", "4"]
