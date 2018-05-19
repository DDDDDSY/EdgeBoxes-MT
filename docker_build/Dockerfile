FROM jjanzic/docker-python3-opencv:contrib

RUN git clone https://github.com/sirebellum/RPi-Edge /RPi-Edge

#Image in
EXPOSE 9001
#Boxes out
Expose 9002

WORKDIR "/RPi-Edge"

CMD ["/bin/bash"]